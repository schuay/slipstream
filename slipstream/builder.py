# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Build every commit once and publish the artifact for both boxes to bench.

The builder walks an engine's history upward, one commit per cycle: check out,
build, package the run set, publish to the bus. It holds the machine lock
across all of that, because zstd -T0 and a compile saturate every core and a
build next to a measurement contaminates it.

Failures are recorded rather than skipped. Before this, a commit that failed to
build was skipped once and never revisited as soon as a later commit was marked
done, while a failure at the frontier was retried every cycle forever.
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import host
from .bus import Bus, BuilderState, BusError, Entry, sha256_file
from .collector import BenchCollector, BuildStepError, FetchError
from .config import Config, EngineConfig
from .lock import MachineLock

# A stalled engine still retries, or the guard against a permanent burn becomes
# a permanent stall. Each attempt is a full checkout, sync and compile holding
# the machine lock against the bencher, so it backs off to a multiple of the
# poll interval.
STALL_BACKOFF = 4
STALL_BACKOFF_CAP_SECS = 6 * 3600
DEFAULT_INTERVAL_SECS = 1800.0
# How many failures the published state file carries. Consumers cat it over
# ssh every cycle; the count travels separately so nothing is hidden.
MAX_REPORTED_FAILURES = 50

GB = 1_000_000_000
# A payload is tens of MB, so the published line reports its own unit rather
# than rounding every artifact to 0.0GB.
MB = 1_000_000


class BuildError(RuntimeError):
    """The builder cannot proceed with this engine."""


@dataclass
class BuildResult:
    commit_id: int | None = None
    published: bool = False
    failed_kind: str | None = None
    status: str | None = None


def build_cfg_hash(engine: EngineConfig) -> str:
    """Identifies the build inputs an artifact was produced with.

    Everything the user config can override, not just the compiler flags: the
    template offers build_cmd and sync_cmd as per-box settings, so hashing only
    gn_args would give two boxes building differently the same hash and leave
    run_env asserting inputs match when they do not. The run set is in for the
    same reason and is the one most likely to change -- adding a file to it
    changes what every later artifact contains, with nothing else to notice.
    """
    import hashlib

    parts = [
        "\n".join(BenchCollector._normalize_gn_args(engine.gn_args))
        if engine.gn_args
        else "",
        engine.build_cmd,
        engine.sync_cmd or "",
        "\n".join(engine.pre_build_patches),
        "\n".join(sorted(engine.run_set)),
    ]
    payload = "\n--\n".join(parts)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def package(
    src_dir: Path, run_set: list[str], dest: Path, *, caffeinate: bool = True
) -> None:
    """Archive the run set as tar plus zstd.

    COPYFILE_DISABLE stops bsdtar emitting AppleDouble ._ members, which would
    change what a codesign check proves. The explicit pipe to zstd is what
    saturates the cores the machine lock assumes; tar --zstd gives no way to
    pass -T0. Both halves run as argv lists rather than a shell string: every
    run_set entry and the source directory are interpolated paths.
    """
    missing = [entry for entry in run_set if not (src_dir / entry).exists()]
    if missing:
        raise BuildError(f"run_set entries missing from the build: {missing}")

    prefix = (
        ["caffeinate", "-im"] if caffeinate and platform.system() == "Darwin" else []
    )
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    tar = subprocess.Popen(
        [*prefix, "tar", "-cf", "-", "-C", str(src_dir), *run_set],
        stdout=subprocess.PIPE,
        env=env,
    )
    try:
        zstd = subprocess.Popen(
            [*prefix, "zstd", "-T0", "-3", "-q", "-o", str(dest)],
            stdin=tar.stdout,
            env=env,
        )
    except OSError:
        tar.kill()
        tar.wait()
        raise
    # Let tar see EPIPE if zstd dies.
    tar.stdout.close()
    zstd_rc = zstd.wait()
    tar_rc = tar.wait()
    if tar_rc != 0 or zstd_rc != 0:
        dest.unlink(missing_ok=True)
        raise BuildError(f"packaging failed (tar {tar_rc}, zstd {zstd_rc})")


class Builder:
    def __init__(
        self,
        cfg: Config,
        bus: Bus,
        *,
        verbose: bool = False,
        log: Callable[[str], None] | None = None,
        interval_secs: float = DEFAULT_INTERVAL_SECS,
    ):
        self.cfg = cfg
        self.bus = bus
        self.interval_secs = interval_secs
        self.collector = BenchCollector(cfg, verbose=verbose, role="build")
        self.store = self.collector.store
        self.lock = MachineLock("build")
        self.log = log or (lambda msg: None)
        self.identity = {"bot": cfg.bot_name, **host.identity()}

    # --- resolving what to build ---

    def frontier(self, engine_name: str) -> int | None:
        """The highest commit this builder has finished with.

        Published entries and terminal build failures both count, so a compile
        failure at the head is not rebuilt every cycle. [build] from is
        consulted only when there is neither, so a restart cannot rewind.
        """
        ids = self.bus.commit_ids(engine_name)
        published = ids[-1] if ids else None
        failed = self.store.max_terminal_build_id(engine_name)
        candidates = [c for c in (published, failed) if c is not None]
        if candidates:
            return max(candidates)
        return self.cfg.build.start_from.get(engine_name)

    def consecutive_burns(self, engine_name: str) -> int:
        """Infrastructure burns with no successful publish between them.

        A fact about the topic, not about build_state: a successful build
        leaves no row there, so counting rows alone would trip on three
        unrelated outages months apart and stall a healthy engine.
        """
        ids = self.bus.commit_ids(engine_name)
        floor = ids[-1] if ids else self.cfg.build.start_from.get(engine_name)
        if floor is None:
            # Nothing published and nowhere told to start: there is no history
            # to be consecutive with, and build_failures with no bound would
            # return every burn ever recorded -- the "three unrelated outages
            # months apart" this is written to avoid.
            return 0
        return sum(
            1
            for row in self.store.build_failures(engine_name, above=floor)
            if row["status"] == "infra_burned"
        )

    def next_commit(self, engine_name: str) -> dict | None:
        """The next commit to build, or None when up to date.

        Retries come first: entries above the failed commit exist by the time
        anyone retries, so the frontier rule would never pick it up again.
        """
        engine = self.cfg.engines[engine_name]
        for commit_id in self.store.build_retries_requested(engine_name):
            commit = self.collector.commit_metadata_for_id(engine, commit_id)
            if commit:
                return commit
        frontier = self.frontier(engine_name)
        if frontier is None:
            raise BuildError(
                f"{engine_name} has no build history and no [build] from entry; "
                f"set one to say where to start"
            )
        return self.collector.next_commit_after(engine, frontier)

    # --- one commit ---

    def free_gb(self) -> float:
        root = self.bus.root
        while not root.exists() and root != root.parent:
            root = root.parent
        return shutil.disk_usage(root).free / GB

    def build_one(self, engine_name: str) -> BuildResult:
        """Resolve, build, package and publish one commit."""
        engine = self.cfg.engines[engine_name]
        engine.require_run_set()
        state = self.bus.read_builder_state(engine_name)

        if self.free_gb() < self.cfg.build.min_free_gb:
            # Publishing stops rather than retention shrinking: a budget that
            # silently shrank under disk pressure would delete unread entries
            # exactly when nobody is watching.
            state.publishing_paused_by_floor = True
            state.last_error = f"only {self.free_gb():.0f}GB free"
            state.in_flight = None
            self._publish_state(engine_name, state)
            self.log(f"{engine_name}: paused, {self.free_gb():.0f}GB free")
            return BuildResult()
        state.publishing_paused_by_floor = False

        stalled = (
            self.consecutive_burns(engine_name) >= self.cfg.build.max_consecutive_burns
        )
        if stalled and not self._stall_window_open(state):
            # Still publish: this branch has already cleared the free-space
            # flag in memory, and an operator diagnosing the stall would
            # otherwise keep reading "paused below its floor" from a disk
            # problem that was fixed days ago.
            state.in_flight = None
            self._publish_state(engine_name, state)
            return BuildResult()

        commit = self.next_commit(engine_name)
        if commit is None:
            state.last_error = None
            # A builder killed between publishing and writing its state leaves
            # this set; refreshing it every cycle would report a package step
            # in progress forever.
            state.in_flight = None
            self._publish_state(engine_name, state)
            return BuildResult()

        commit_id = int(commit["commit_id"])
        state.in_flight = {
            "commit_id": commit_id,
            "phase": "build",
            "started_at": time.time(),
        }
        self._publish_state(engine_name, state)

        row = self.store.get_build_state(engine_name, commit_id)
        was_retry = row is not None and row["status"] == "retry_requested"
        log_path = self.cfg.logs_dir / f"build-{engine_name}-{commit_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        self.log(f"{engine_name}: building {commit_id} ({commit['hash'][:8]})")
        failure = self.collector.build_at(engine, commit["hash"], log_path)
        if failure is not None:
            return self._record_failure(
                engine_name,
                commit_id,
                failure,
                log_path,
                state,
                stalled=stalled,
                was_retry=was_retry,
            )

        state.in_flight = {
            "commit_id": commit_id,
            "phase": "package",
            "started_at": time.time(),
        }
        self._publish_state(engine_name, state)
        try:
            entry = self._package_and_publish(
                engine, commit, int(time.time() - started)
            )
        except (BuildError, OSError) as e:
            # Packaging is this machine's business, not the commit's.
            return self._record_failure(
                engine_name,
                commit_id,
                BuildStepError("package", 1),
                log_path,
                state,
                stalled=stalled,
                was_retry=was_retry,
                message=str(e),
            )

        self.store.clear_build_state(engine_name, commit_id)
        state.stall_retry_after = None
        dropped = self.bus.prune(
            engine_name, self.cfg.build.retain_gb * GB, keep=entry.commit_id
        )
        if dropped:
            self.log(f"{engine_name}: retention dropped {len(dropped)} entries")
            state.highest_dropped = max([state.highest_dropped or 0, *dropped])
        state.stalled_since = None
        state.last_error = None
        state.in_flight = None
        self._publish_state(engine_name, state)
        self.log(
            f"{engine_name}: published {commit_id} "
            f"({entry.blob_bytes / MB:.0f}MB, {entry.build_secs}s)"
        )
        return BuildResult(commit_id=commit_id, published=True)

    def _package_and_publish(
        self, engine: EngineConfig, commit: dict, build_secs: int
    ) -> Entry:
        commit_id = int(commit["commit_id"])
        blob = self.bus.tmp_blob(engine.name, commit_id)
        package(engine.require_src_dir(), engine.require_run_set(), blob)
        entry = Entry(
            engine=engine.name,
            commit_id=commit_id,
            hash=commit["hash"],
            date=commit.get("date", ""),
            timestamp=int(commit.get("timestamp", 0)),
            title=commit.get("title", ""),
            build_cfg_hash=build_cfg_hash(engine),
            blob_sha256=sha256_file(blob),
            blob_bytes=blob.stat().st_size,
            builder=dict(self.identity),
            built_at=int(time.time()),
            build_secs=build_secs,
        )
        self.bus.publish(entry, blob)
        return entry

    def _record_failure(
        self,
        engine_name: str,
        commit_id: int,
        failure,
        log_path: Path,
        state: BuilderState,
        *,
        stalled: bool,
        was_retry: bool = False,
        message: str | None = None,
    ) -> BuildResult:
        kind = failure.kind
        # Packaging is this machine's business too, so it retries like the rest.
        if kind == "compile":
            status = "compile_failed"
            attempts = self.store.record_build_failure(
                engine_name, commit_id, status, kind, str(log_path)
            )
        else:
            # A retry keeps its allowance while it still has attempts left.
            # infra_retry is only reachable through the frontier, which has
            # already moved past a commit anyone had to retry, so writing it
            # here would strand the commit: not retryable, and not terminal, so
            # absent from bus status too.
            pending = "retry_requested" if was_retry else "infra_retry"
            attempts = self.store.record_build_failure(
                engine_name, commit_id, pending, kind, str(log_path)
            )
            status = pending
            if stalled:
                # While stalled the builder keeps trying without burning
                # commits, or the guard against a permanent burn becomes a
                # permanent stall.
                self._arm_stall_backoff(state)
            elif attempts >= self.cfg.build.max_infra_attempts:
                # A sync failure caused by the commit itself is
                # indistinguishable by exit code from an outage, so an
                # unbounded "never advance" would wedge the engine.
                status = "infra_burned"
                # Reclassify the row this attempt just wrote rather than
                # recording a second one, which would double-count the attempt.
                self.store.set_build_status(engine_name, commit_id, status)
        state.last_error = (
            message or f"{kind} failed on {commit_id} (attempt {attempts})"
        )
        state.in_flight = None
        if self.consecutive_burns(engine_name) >= self.cfg.build.max_consecutive_burns:
            if state.stalled_since is None:
                state.stalled_since = time.time()
                self.log(
                    f"{engine_name}: stalled after "
                    f"{self.cfg.build.max_consecutive_burns} burns in a row; "
                    f"this is the machine, not the tree"
                )
            self._arm_stall_backoff(state)
        self._publish_state(engine_name, state)
        self.log(f"{engine_name}: {commit_id} {status} at {kind}, log {log_path}")
        return BuildResult(commit_id=commit_id, failed_kind=kind, status=status)

    # --- stall backoff ---

    def _stall_window_open(self, state: BuilderState) -> bool:
        return time.time() >= (state.stall_retry_after or 0.0)

    def _arm_stall_backoff(self, state: BuilderState) -> None:
        delay = min(self.interval_secs * STALL_BACKOFF, STALL_BACKOFF_CAP_SECS)
        state.stall_retry_after = time.time() + delay

    def _publish_state(self, engine_name: str, state: BuilderState) -> None:
        # Recomputed here rather than at the call sites that happen to change
        # them: the up-to-date, disk-floor and stall branches all publish too,
        # and a bus state directory that was wiped under a populated topic
        # would otherwise republish "frontier null, no failed builds" forever,
        # because no publish or failure ever comes along to refresh it.
        state.frontier = self.frontier(engine_name)
        state.lowest_retained = self.bus.lowest_retained(engine_name)
        failures = [dict(r) for r in self.store.build_failures(engine_name)]
        state.failed_total = len(failures)
        state.failed = failures[-MAX_REPORTED_FAILURES:]
        self.bus.write_builder_state(engine_name, state)

    # --- the loop ---

    def run_cycle(self, engine_names: list[str], should_stop) -> int:
        """One pass over the engines; returns how many commits were published."""
        published = 0
        for name in engine_names:
            if should_stop():
                break
            if not self.lock.acquire(should_stop, wait=True, log=self.log):
                break
            try:
                self.collector.head_commit_id(name)  # fetch, so origin/main is current
                result = self.build_one(name)
            except FetchError:
                self.log(f"{name}: skipping until fetch succeeds")
                continue
            except (BuildError, BusError, ValueError, OSError, sqlite3.Error) as e:
                # ValueError is how require_run_set and require_src_dir report
                # a misconfigured engine; BusError is a state file this version
                # cannot read; OSError is a full disk or a missing tar; and
                # sqlite3.Error, which is not an OSError, is every build_state
                # write. None of them may take the working engines down too.
                self.log(f"{name}: {e}")
                continue
            finally:
                self.lock.release()
            if result.published:
                published += 1
        return published
