# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The build bus: one box builds a commit, both boxes bench the artifact.

Layout, one root per machine::

    bus/topics/builds/<engine>/<key>.json    entries
    bus/blobs/builds/<engine>/<key>.tar.zst  payloads
    bus/state/builds/<engine>.json           builder state, published
    bus/state/bench/<engine>.json            local bencher state

Entries are keyed by commit key, not by a sequence number: the payload already
carries a monotone, machine-independent key. A cursor is therefore a key and a
consumer takes the next entry above it, so nothing needs contiguity and there
is no gap detection or seq allocation. The builder publishes strictly upward
per engine, because a key cursor cannot see anything published at or below
it; backfilling an older range needs an explicit cursor reset on every
consumer.

A key is ``str(CommitKey)``: the bare commit id for an engine that is its
own embedder, ``<embedder_id>-<commit_id>`` otherwise. Files written before
keys had two parts are therefore already named correctly.

Payloads are named by their entry's key rather than by content hash. Every
payload has exactly one referencing entry by construction, so content
addressing would buy only a refcount scan, an orphan sweep and a lock to make
that sweep safe. With entry-keyed names a crashed publish self-heals: the
builder retries the commit and the rebuild writes the same name. The sha256 is
in the entry either way, so transfer verification is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import MISSING, asdict, dataclass, field
from pathlib import Path

from .models import CommitKey

VERSION = 1

BLOB_SUFFIX = ".tar.zst"


def parse_key_stem(stem: str) -> CommitKey | None:
    """A filename stem as a key, or None for anything that is not one.

    Tmp files and strays share the directory; a stem that is not a key is
    simply not an entry, not an error.
    """
    try:
        return CommitKey.parse(stem)
    except ValueError:
        return None


class BusError(RuntimeError):
    """The bus root holds something this version cannot read."""


def _atomic_write(path: Path, data: str) -> None:
    """Write via a uniquely named tmp file and rename.

    Unique because two writers exist per machine (the daemon and an ad-hoc
    command), and rename rather than in-place because box1 reads these files
    over ssh, where a torn read would surface as a parse error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns():x}")
    tmp.write_text(data)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Entry:
    """One built commit, ready to bench.

    Key names match the store's columns, so a consumer can write the commits
    row straight from this. ``embedder_id`` defaults to 0 so an entry written
    before keys had two parts reads back as the scalar key it always was; the
    commit fields stay the engine's own commit, which is what names the entry.

    ``embedder`` is the outer commit a two-coordinate key was built under
    (``hash``, ``commit_id``, ``title`` of the chromium roll CL) and ``pins``
    what was pinned beneath it (``{"src/v8": hash}``); both are empty for an
    engine built from its own checkout, and absent from the entries such an
    engine wrote before the fields existed.
    """

    engine: str
    commit_id: int
    hash: str
    date: str
    timestamp: int
    title: str
    build_cfg_hash: str
    blob_sha256: str
    blob_bytes: int
    embedder_id: int = 0
    embedder: dict = field(default_factory=dict)
    pins: dict = field(default_factory=dict)
    builder: dict = field(default_factory=dict)
    built_at: int = 0
    build_secs: int = 0
    version: int = VERSION

    @property
    def key(self) -> CommitKey:
        return CommitKey(self.embedder_id, self.commit_id)

    @property
    def embedder_hash(self) -> str:
        return str(self.embedder.get("hash", "")) if self.embedder else ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str, where: str = "") -> Entry:
        try:
            data = json.loads(text)
        except ValueError as e:
            raise BusError(f"unreadable bus entry {where}: {e}") from e
        # Valid JSON that is not an object: null, a list, a bare number. Every
        # other malformed case here is a BusError, which a consumer survives,
        # and .get on one of these is an AttributeError, which it does not.
        if not isinstance(data, dict):
            raise BusError(f"bus entry {where} is not an object")
        version = data.get("version")
        if version != VERSION:
            raise BusError(
                f"bus entry {where} is version {version}, this slipstream reads "
                f"version {VERSION}; upgrade the consumer"
            )
        known = {f for f in cls.__dataclass_fields__}
        # Every field the constructor requires, not just the interesting ones:
        # a missing one would otherwise raise TypeError, which a consumer does
        # not treat as a bad entry and so does not survive.
        required = {
            name
            for name, f in cls.__dataclass_fields__.items()
            if f.default is MISSING and f.default_factory is MISSING
        }
        missing = required - data.keys()
        if missing:
            raise BusError(f"bus entry {where} is missing {sorted(missing)}")
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class BuilderState:
    """What the builder knows, published for consumers to read over ssh.

    ``updated_at`` is rewritten per entry rather than per cycle: a cycle
    drains, so it can span days, which would leave the file stalest exactly
    when the process is busiest. Without it a crashed builder leaves a
    plausible frontier and a null last_error forever.

    The key fields are written with ``CommitKey.to_json``: a bare int for a
    scalar key, so a v8 or jsc state file is byte-identical to one written
    before keys had two parts, and a pair otherwise.
    """

    KEY_FIELDS = ("frontier", "lowest_retained", "highest_dropped")

    frontier: CommitKey | None = None
    lowest_retained: CommitKey | None = None
    # The highest entry retention has ever deleted. A consumer whose cursor is
    # below it provably never benched that entry, which comparing against
    # lowest_retained does not establish: commit ids are not contiguous, so a
    # gap in them is not evidence of anything.
    highest_dropped: CommitKey | None = None
    # The most recent failures only, with the total beside them: consumers
    # read this file over ssh once per cycle, and an unbounded list grows for
    # the life of the engine.
    failed: list[dict] = field(default_factory=list)
    failed_total: int = 0
    publishing_paused_by_floor: bool = False
    last_error: str | None = None
    stalled_since: float | None = None
    # When a stalled engine may be attempted again. Persisted rather than held
    # in the process, because `build --once` from launchd is a supported
    # deployment and each invocation is a new process.
    stall_retry_after: float | None = None
    in_flight: dict | None = None
    updated_at: float = 0.0
    version: int = VERSION


@dataclass
class BenchState:
    """What a bencher knows. Separate from the builder's file because the two
    are separate processes and a shared file would clobber whichever wrote
    first.

    ``env`` records the inputs shared between the boxes -- slipstream version,
    OS, run configs, harness revisions -- with ``since``, the time those values
    took effect, so a divergence check reports when the two boxes parted rather
    than only that they currently agree.
    """

    KEY_FIELDS = ("cursor", "skipped_dropped")

    bot: str | None = None
    cursor: CommitKey | None = None
    lag: int = 0
    status_counts: dict = field(default_factory=dict)
    last_error: str | None = None
    stalled_since: float | None = None
    # When a stalled bencher may try again. Persisted for the same reason the
    # builder's is: `watch --once` from launchd is a new process each time.
    stall_retry_after: float | None = None
    # Highest entry this machine skipped because retention had already dropped
    # it. Its own record, because the builder's highest_dropped stops being
    # evidence the moment the cursor passes it.
    skipped_dropped: CommitKey | None = None
    benching_paused_by_floor: bool = False
    in_flight: dict | None = None
    env: dict = field(default_factory=dict)
    updated_at: float = 0.0
    version: int = VERSION


def state_to_json(state) -> str:
    data = asdict(state)
    for name in type(state).KEY_FIELDS:
        if data[name] is not None:
            data[name] = CommitKey.of(data[name]).to_json()
    return json.dumps(data, indent=1, sort_keys=True)


class Bus:
    """A bus root on this machine."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser()

    # --- paths ---

    def topic_dir(self, engine: str) -> Path:
        return self.root / "topics" / "builds" / engine

    def blob_dir(self, engine: str) -> Path:
        return self.root / "blobs" / "builds" / engine

    def entry_path(self, engine: str, key) -> Path:
        return self.topic_dir(engine) / f"{CommitKey.of(key)}.json"

    def blob_path(self, engine: str, key) -> Path:
        return self.blob_dir(engine) / f"{CommitKey.of(key)}{BLOB_SUFFIX}"

    def builder_state_path(self, engine: str) -> Path:
        return self.root / "state" / "builds" / f"{engine}.json"

    def bench_state_path(self, engine: str) -> Path:
        return self.root / "state" / "bench" / f"{engine}.json"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    # --- entries ---

    def keys(self, engine: str) -> list[CommitKey]:
        """Published keys, ascending. Ignores tmp files still being written."""
        try:
            names = os.listdir(self.topic_dir(engine))
        except FileNotFoundError:
            return []
        keys = []
        for name in names:
            stem, dot, ext = name.partition(".")
            key = parse_key_stem(stem) if dot and ext == "json" else None
            if key is not None:
                keys.append(key)
        return sorted(keys)

    def keys_above(self, engine: str, cursor) -> list[CommitKey]:
        keys = self.keys(engine)
        if cursor is None:
            return keys
        cursor = CommitKey.of(cursor)
        return [k for k in keys if k > cursor]

    def read_entry(self, engine: str, key) -> Entry | None:
        """The entry, or None if retention dropped it between listing and now."""
        path = self.entry_path(engine, key)
        try:
            text = path.read_text()
        except FileNotFoundError:
            return None
        return Entry.from_json(text, str(path))

    def publish(self, entry: Entry, blob: Path) -> None:
        """Move a packaged payload and its entry into the topic.

        Payload first, always: the reverse lets a consumer read an entry whose
        payload does not exist. A crash between the two leaves an unreferenced
        payload, which is the harmless direction -- ``gc`` reclaims it, and the
        builder's retry of that commit overwrites it under the same name.
        """
        dest = self.blob_path(entry.engine, entry.key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(blob, dest)
        _atomic_write(self.entry_path(entry.engine, entry.key), entry.to_json())

    def tmp_blob(self, engine: str, key) -> Path:
        """Where a packager writes before publish renames it into place."""
        self.blob_dir(engine).mkdir(parents=True, exist_ok=True)
        return self.blob_dir(engine) / (
            f"{CommitKey.of(key)}{BLOB_SUFFIX}.tmp-{os.getpid()}-{time.time_ns():x}"
        )

    # --- state files ---

    def read_builder_state(self, engine: str) -> BuilderState:
        return _read_state(self.builder_state_path(engine), BuilderState)

    def write_builder_state(self, engine: str, state: BuilderState) -> None:
        state.updated_at = time.time()
        _atomic_write(self.builder_state_path(engine), state_to_json(state))

    def read_bench_state(self, engine: str) -> BenchState:
        return _read_state(self.bench_state_path(engine), BenchState)

    def write_bench_state(self, engine: str, state: BenchState) -> None:
        state.updated_at = time.time()
        _atomic_write(self.bench_state_path(engine), state_to_json(state))

    # --- retention ---

    def prune(self, engine: str, retain_bytes: float, keep=None) -> list[CommitKey]:
        """Drop the oldest entries until the payloads fit the budget.

        A byte budget rather than a count: payload size per commit is what
        matters. Deletes the entry and then its payload, so a crash between the
        two leaves a payload with no entry, which gc reclaims; the reverse
        would leave an entry pointing at nothing, which a consumer must treat
        as a real error.

        What is retained is always a contiguous run ending at the newest entry.
        Keeping a smaller older entry below a dropped one would fit more in the
        budget, but it would leave a hole that nothing can detect: a consumer
        finds out about dropped entries by comparing its cursor against
        ``lowest_retained``, which a surviving older entry holds down, so it
        would step over the hole in silence.

        ``keep`` lowers the floor of that run rather than exempting one entry,
        for the commit just published: a ``build --retry`` republishes below
        the frontier, so its entry is the oldest and would otherwise be deleted
        by the very cycle that produced it. Exempting it alone would leave
        exactly the hole described above, so everything from it upward is
        retained and the budget is overshot for one cycle. The next publish
        prunes normally, dropping it along with its neighbours and reporting it.

        Retention does not coordinate with consumers. Detection does that
        instead: lowest_retained above a consumer's cursor means entries were
        dropped unread, which bus status reports.
        """
        keep = CommitKey.of(keep) if keep is not None else None
        dropped = []
        used = 0.0
        over_budget = False
        for key in reversed(self.keys(engine)):
            blob = self.blob_path(engine, key)
            try:
                size = blob.stat().st_size
            except FileNotFoundError:
                size = 0
            if not over_budget and used and used + size > retain_bytes:
                over_budget = True
            if not over_budget or (keep is not None and key >= keep):
                used += size
                continue
            self.entry_path(engine, key).unlink(missing_ok=True)
            blob.unlink(missing_ok=True)
            dropped.append(key)
        return sorted(dropped)

    def lowest_retained(self, engine: str) -> CommitKey | None:
        keys = self.keys(engine)
        return keys[0] if keys else None

    def blob_bytes(self, engine: str) -> int:
        total = 0
        for key in self.keys(engine):
            try:
                total += self.blob_path(engine, key).stat().st_size
            except FileNotFoundError:
                pass
        return total

    def gc(self, engines: list[str]) -> list[Path]:
        """Delete what no entry names: payloads left by a crash mid-publish or
        mid-prune, half-fetched payloads, and abandoned unpack directories.
        Returns what it removed. The caller holds the machine lock, so nothing
        here can be in use.
        """
        removed = []
        # _atomic_write leaves these anywhere it is used, including under
        # state/, which nothing else looked at.
        for state_dir in ("builds", "bench"):
            d = self.root / "state" / state_dir
            for partial in sorted(d.glob("*.tmp-*")) if d.exists() else []:
                partial.unlink(missing_ok=True)
                removed.append(partial)
        for engine in engines:
            # A kill mid-unpack leaves a full-size directory that no commit id
            # names. The consumer sweeps these too, but only on a cycle that
            # gets past its free-space floor -- which is the cycle this is
            # standing in for.
            roots = self.root / "roots" / engine
            for partial in sorted(roots.glob("*.unpacking")) if roots.exists() else []:
                shutil.rmtree(partial, ignore_errors=True)
                removed.append(partial)
        # An interrupted fetch leaves hundreds of MB here that no entry names.
        for partial in sorted(self.tmp_dir.glob("*")) if self.tmp_dir.exists() else []:
            if partial.is_file():
                partial.unlink(missing_ok=True)
                removed.append(partial)
        for engine in engines:
            # A crash between writing an entry's tmp file and renaming it leaks
            # one per crash into the topic, which keys() ignores and nothing
            # else looked at.
            for partial in self.topic_dir(engine).glob("*.tmp-*"):
                partial.unlink(missing_ok=True)
                removed.append(partial)
            published = set(self.keys(engine))
            try:
                names = os.listdir(self.blob_dir(engine))
            except FileNotFoundError:
                continue
            for name in names:
                path = self.blob_dir(engine) / name
                if name.endswith(BLOB_SUFFIX):
                    key = parse_key_stem(name[: -len(BLOB_SUFFIX)])
                    if key is not None and key in published:
                        continue
                elif ".tmp-" not in name:
                    continue
                path.unlink(missing_ok=True)
                removed.append(path)
        return removed


def state_from_json(text: str, cls, where: str = ""):
    """Parse a state file's contents. Also used for a copy read over ssh."""
    try:
        data = json.loads(text)
    except ValueError as e:
        raise BusError(f"unreadable bus state {where}: {e}") from e
    if not isinstance(data, dict):
        raise BusError(f"bus state {where} is not an object")
    if data.get("version") != VERSION:
        raise BusError(
            f"bus state {where} is version {data.get('version')}, this "
            f"slipstream reads version {VERSION}"
        )
    known = {f for f in cls.__dataclass_fields__}
    state = cls(**{k: v for k, v in data.items() if k in known})
    for name in cls.KEY_FIELDS:
        try:
            setattr(state, name, CommitKey.from_json(getattr(state, name)))
        except (TypeError, ValueError) as e:
            raise BusError(f"bus state {where} has a bad {name}: {e}") from e
    return state


def _read_state(path: Path, cls):
    try:
        text = path.read_text()
    except FileNotFoundError:
        return cls()
    return state_from_json(text, cls, str(path))


# --- consumer cursors ---


def cursor_path(out_dir: Path, source_name: str, engine: str) -> Path:
    return out_dir / "cursors" / source_name / "builds" / engine


def read_cursor(path: Path) -> CommitKey | None:
    """The last key benched from this source, or None if never.

    A torn cursor is reported rather than guessed at: the caller falls back to
    the highest done commit for the engine, which is derivable and cheap.
    """
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    try:
        return CommitKey.parse(text)
    except ValueError as e:
        raise BusError(f"unreadable cursor {path}: {text!r}") from e


def write_cursor(path: Path, key) -> None:
    _atomic_write(path, f"{CommitKey.of(key)}\n")
