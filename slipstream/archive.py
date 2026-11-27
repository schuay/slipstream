# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Keep every published build forever on dumb storage, written with rsync.

The archive is a content-addressed copy of this bus's objects plus the exact
bytes of every manifest:

    <target>/objects/<aa>/<name>
    <target>/manifests/<engine>/<key>/<sha256>.json

It never deletes, so it needs no lock of its own and tolerates more than one
writer. A run captures everything above the engine's cursor into private
staging under the bus lock (hardlinks keep the files alive once the lock is
released), uploads the objects, then the manifests, and only then advances
the cursor. A manifest's presence on the target is the commit point: rsync
renames each file into place, so with content-addressed names an existing
file is a complete one, and objects land first, so no archived manifest names
a missing object. A crashed run leaves the cursor where it was; the next one
rebuilds the batch and rsync skips what already landed.

The cursor in ``state/archive/<engine>.json`` is also a retention gate:
gated retention frees an entry on this bus only once it is archived.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from datetime import datetime, timedelta

from .bus import Bus, BusError, Entry
from .bus_protocol import (
    BusProtocol,
    digest,
    encoded,
    manifest_text,
    read,
    staging_fds,
)
from .config import ArchiveConfig, Config
from .durability import atomic_write
from .models import CommitKey
from .retention import Policy
from .service_logging import error, success, warning

STATE_VERSION = 1
RETRY_MIN_SECS = 60
RETRY_MAX_SECS = 3600
# A missing key or ssh agent must fail the run rather than hang it.
SSH = "ssh -o BatchMode=yes"


class ArchiveError(RuntimeError):
    """The archive cannot proceed with this engine (or at all)."""


def empty_state(target: str) -> dict:
    return {
        "version": STATE_VERSION,
        "target": target,
        "cursor": None,
        "skipped_dropped": None,
        "last_success_at": None,
        "last_error": None,
        "retry_after": 0,
        "failures": 0,
    }


def read_state(protocol: BusProtocol, engine: str) -> dict | None:
    """The engine's archive record, or None if it has never run.

    Raises ArchiveError for a file that exists but cannot be used.
    """
    import json

    path = protocol.state_path("archive", engine)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ArchiveError(f"unreadable archive state {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise ArchiveError(f"archive state {path} is not version {STATE_VERSION}")
    return data


def archive_cursor(protocol: BusProtocol, engine: str) -> CommitKey | None:
    """The highest key archived or skipped, or None if nothing yet."""
    state = read_state(protocol, engine)
    if state is None or not state.get("cursor"):
        return None
    try:
        return CommitKey.parse(state["cursor"])
    except (TypeError, ValueError) as exc:
        raise ArchiveError(f"archive state for {engine} has a bad cursor") from exc


def object_dest(name: str) -> str:
    return f"objects/{name[:2]}/{name}"


def manifest_dest(engine: str, key: CommitKey, text: str) -> str:
    return f"manifests/{engine}/{key}/{digest(text)}.json"


def next_run_at(cfg: ArchiveConfig, state: dict | None, now: float) -> float:
    """When this engine's next scheduled run is due (may be in the past).

    A failed run is retried at ``retry_after``, ahead of the schedule.
    Without ``at``: ``every_hours`` after the last successful run started.
    With it: at ``at`` today when nothing has succeeded yet, and otherwise
    at ``at`` on the day the interval since the last success ends.
    """
    state = state or {}
    last = state.get("last_success_at")
    retry = state.get("retry_after") or 0
    if state.get("last_error") and retry:
        return retry
    if cfg.at_minutes is None:
        return now if last is None else last + cfg.every_hours * 3600
    day = datetime.fromtimestamp(now if last is None else last)
    if last is not None:
        day += timedelta(hours=cfg.every_hours)
    hours, minutes = divmod(cfg.at_minutes, 60)
    return day.replace(hour=hours, minute=minutes, second=0, microsecond=0).timestamp()


def retry_delay(failures: int) -> float:
    return min(RETRY_MAX_SECS, RETRY_MIN_SECS * 2 ** max(0, failures - 1))


class Archiver:
    def __init__(self, cfg: Config, log=print, *, runner=subprocess.run):
        if cfg.bus is None or cfg.bus.archive is None:
            raise ArchiveError("archive requires a [bus.archive] section")
        self.cfg = cfg
        self.archive = cfg.bus.archive
        self.log = log
        self.runner = runner
        self.protocol = BusProtocol(cfg.bus.root)
        self.bus = Bus(cfg.bus.root)
        self.policy = Policy(cfg)
        ready = self.protocol.ready()
        self.protocol.expected_uuid = ready["uuid"]
        for engine in self.archive.engines:
            if ready["owners"].get(engine) != ready["uuid"]:
                raise ArchiveError(f"{engine} is not owned by this bus")

    # --- rsync ---

    def rsync(self, *args: str) -> None:
        command = [
            "rsync",
            "-r",
            "--ignore-existing",
            "--fsync",
            "--chmod=F0444,D0755",
            "-e",
            SSH,
        ]
        if self.archive.bwlimit:
            command.append(f"--bwlimit={self.archive.bwlimit}")
        command.extend(args)
        result = self.runner(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            pass_fds=staging_fds(),
        )
        if result.returncode != 0:
            raise OSError(
                f"rsync exited {result.returncode}: "
                f"{(result.stderr or '').strip() or 'no error output'}"
            )

    def probe(self) -> None:
        """rsync with --fsync on both ends, ssh and the target path.

        Probed rather than parsed from version output: --fsync needs rsync
        3.2.4 or later on the target as well as here.
        """
        with tempfile.TemporaryDirectory() as empty:
            try:
                self.rsync("--dry-run", f"{empty}/", f"{self.target_dir()}/")
            except OSError as exc:
                raise ArchiveError(f"archive target probe failed: {exc}") from exc

    def target_dir(self) -> str:
        return self.archive.target.rstrip("/") or "/"

    # --- state ---

    def check_targets(self) -> None:
        """Refuse a cursor that describes another store."""
        for engine in self.archive.engines:
            state = read_state(self.protocol, engine)
            if state is not None and state.get("target") != self.archive.target:
                raise ArchiveError(
                    f"{engine}: archive state is for {state.get('target')!r}, "
                    f"configured target is {self.archive.target!r}; move or remove "
                    f"{self.protocol.state_path('archive', engine)} deliberately"
                )

    def state(self, engine: str) -> dict:
        return read_state(self.protocol, engine) or empty_state(self.archive.target)

    def _save_locked(self, engine: str, state: dict) -> None:
        atomic_write(self.protocol.state_path("archive", engine), encoded(state))

    def save(self, engine: str, state: dict) -> None:
        with self.protocol.lock():
            self.protocol.ready()
            self._save_locked(engine, state)

    # --- one run ---

    def _capture_locked(self, engine: str, cursor, stage) -> tuple[list, dict]:
        """Link every entry above ``cursor`` into ``stage`` in archive layout.

        Returns the archived keys and the retention-gap skip, if any.
        """
        self.protocol._recover_locked(engine)
        bus = self.protocol.bus
        dropped = read(self.protocol.state_path("retention", engine)).get(
            "highest_dropped"
        )
        dropped = CommitKey.parse(dropped) if dropped else None
        keys = [k for k in bus.keys(engine) if cursor is None or k > cursor]
        archived, skipped = [], None
        if dropped is not None and (cursor is None or dropped > cursor):
            skipped = dropped
        for key in keys:
            path = bus.entry_path(engine, key)
            try:
                text = manifest_text(path)
            except FileNotFoundError:
                if dropped is not None and dropped >= key:
                    continue  # retention gap, recorded through ``skipped``
                raise ArchiveError(
                    f"{engine} {key} vanished without retention evidence"
                ) from None
            entry = Entry.from_json(text)
            if entry.engine != engine or entry.key != key:
                raise ArchiveError(f"{engine} {key}: manifest names another entry")
            if entry.version < 4:
                raise ArchiveError(
                    f"{engine} {key} is manifest version {entry.version}; the "
                    f"archive needs byte-named objects (version 4 or later)"
                )
            for obj in entry.objects():
                dest = stage.path / object_dest(obj.name)
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(bus.entry_object_path(entry, obj), dest)
            dest = stage.path / manifest_dest(engine, key, text)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(path, dest)
            archived.append(key)
        return archived, skipped

    def run_engine(self, engine: str) -> bool:
        """Archive everything above the cursor. Returns False on failure."""
        started = time.time()
        state = self.state(engine)
        cursor = CommitKey.parse(state["cursor"]) if state.get("cursor") else None
        try:
            # The stage is closed outside the bus lock: closing takes it.
            stage = self.protocol.stage()
            with stage:
                with self.protocol.lock():
                    archived, skipped = self._capture_locked(engine, cursor, stage)
                if archived:
                    self.rsync(
                        f"{stage.path}/objects/", f"{self.target_dir()}/objects/"
                    )
                    self.rsync(
                        f"{stage.path}/manifests/", f"{self.target_dir()}/manifests/"
                    )
                top = max([*archived, *([skipped] if skipped else [])], default=None)
                with self.protocol.lock():
                    current = self.state(engine)
                    if current.get("cursor") != state.get("cursor"):
                        raise ArchiveError(
                            f"{engine}: archive cursor moved during the run"
                        )
                    if top is not None and (cursor is None or top > cursor):
                        current["cursor"] = str(top)
                    if skipped is not None:
                        current["skipped_dropped"] = str(
                            max(
                                skipped,
                                CommitKey.parse(current["skipped_dropped"])
                                if current.get("skipped_dropped")
                                else skipped,
                            )
                        )
                    current.update(
                        last_success_at=started,
                        last_error=None,
                        retry_after=0,
                        failures=0,
                    )
                    self._save_locked(engine, current)
        except (OSError, BusError, ArchiveError, ValueError) as exc:
            state["failures"] = state.get("failures", 0) + 1
            state["last_error"] = str(exc)
            state["retry_after"] = time.time() + retry_delay(state["failures"])
            self.log(error(f"{engine}: archive run failed: {exc}"))
            try:
                self.save(engine, state)
            except OSError as save_exc:
                self.log(warning(f"{engine}: failure not recorded: {save_exc}"))
            return False
        if skipped is not None and (cursor is None or skipped > cursor):
            self.log(
                warning(
                    f"{engine}: retention dropped entries through {skipped} before "
                    f"they were archived"
                )
            )
        if archived:
            self.log(
                success(
                    f"{engine}: archived {len(archived)} entries through "
                    f"{archived[-1]} ({time.time() - started:.0f}s)"
                )
            )
        self.after_batch(engine, archived)
        return True

    def after_batch(self, engine: str, archived: list) -> None:
        """The archive gate advanced: release what it was holding."""
        if self.policy.gated:
            self.policy.release(self.bus, self.log, label=f"{engine}: ")

    # --- the loop ---

    def early(self, engine: str, state: dict, now: float) -> bool:
        """Run ahead of schedule: the builder is paused at the cap and this
        engine's oldest live entry is held by the archive gate. Nothing else
        moves a run forward, and a failed run still waits out its backoff."""
        from .retention import holding_gate

        if not self.policy.gated:
            return False
        if state.get("last_error") and (state.get("retry_after") or 0) > now:
            return False
        try:
            paused = any(
                self.bus.read_builder_state(name).publishing_paused_by_cap
                for name in self.bus.engines()
            )
            if not paused:
                return False
            keys = self.bus.keys(engine)
        except (OSError, BusError):
            return False
        held = holding_gate(self.cfg, self.bus.root, engine, keys[0] if keys else None)
        return held is not None and held.gate == "archive"

    def due(self, now: float, *, announce: bool = False) -> dict[str, float]:
        due = {}
        for engine in self.archive.engines:
            state = self.state(engine)
            at = next_run_at(self.archive, state, now)
            if at > now and self.early(engine, state, now):
                if announce:
                    self.log(
                        f"{engine}: builder paused at the cap with the archive "
                        f"holding; running early"
                    )
                at = now
            due[engine] = at
        return due

    def run(self, *, once: bool = False, should_stop=lambda: False) -> bool:
        """Run when due; with ``once``, run every engine now and return
        whether all of them succeeded."""
        self.check_targets()
        self.probe()
        if once:
            return all([self.run_engine(e) for e in self.archive.engines])
        while not should_stop():
            now = time.time()
            for engine, at in self.due(now, announce=True).items():
                if should_stop():
                    break
                if at <= now:
                    self.run_engine(engine)
            wake = min(self.due(time.time()).values(), default=time.time() + 60)
            self.wait(min(60.0, max(1.0, wake - time.time())), should_stop)
        return True

    def wait(self, seconds: float, should_stop) -> None:
        deadline = time.monotonic() + seconds
        while not should_stop() and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))
