# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Bench the artifacts another process built, instead of building them here.

A consumer holds a cursor per (source, engine): the commit id of the last entry
it finished with. Each cycle drains rather than handling one entry, since only
a real bench costs hours and the machine lock and the handoff delay apply per
bench, not per entry. Without that, a cursor reset two hundred entries back
would take two hundred cycles to walk back up through the already-done ones.

Provisioning is deliberately identical to the builder's: the run root is an
unpacked archive rather than a checkout, and nothing downstream knows the
difference. A "bench straight from the checkout" shortcut would let an archive
defect masquerade as a microarchitecture difference, which is the failure this
project is least able to see.
"""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
import tarfile
import time
from collections.abc import Callable
from typing import NamedTuple
from pathlib import Path

from . import __version__, host
from .bus import (
    BenchState,
    Bus,
    BusError,
    Entry,
    cursor_path,
    read_cursor,
    sha256_file,
    write_cursor,
)
from .collector import BenchCollector, BenchOutcome, outcome_status
from .config import BusSource, Config, EngineConfig

GB = 1_000_000_000

# A stalled consumer keeps trying, or the guard against burning the topic
# becomes a permanent stop. It attempts one commit per window, so a genuine
# machine problem costs one commit per backoff rather than the whole backlog.
STALL_BACKOFF = 4
STALL_BACKOFF_CAP_SECS = 6 * 3600
DEFAULT_INTERVAL_SECS = 1800.0


class DrainResult(NamedTuple):
    """What one cycle did, and why it stopped.

    A bare count conflates "nothing to do" with "the source was unreachable",
    which is the conflation the drain loop itself was fixed to avoid.
    """

    benched: int = 0
    error: str | None = None


class ConsumerError(RuntimeError):
    """The entry cannot be turned into a run root."""


# Everything a source or a bench can raise that is worth reporting and
# retrying next cycle rather than killing the daemon. BusError covers a
# malformed entry or state file; SubprocessError covers ssh and rsync.
# Neither tarfile.TarError nor sqlite3.Error is an OSError, and both are
# reachable from a bench cycle: a truncated archive from a zstd killed
# mid-stream, and "database or disk is full" from any of the store writes the
# cycle makes. Either would escape drain, kill the daemon and take the
# git-driven engines and the pusher with it.
TRANSPORT_ERRORS = (
    ConsumerError,
    BusError,
    OSError,
    subprocess.SubprocessError,
    tarfile.TarError,
    sqlite3.Error,
)


class ShaMismatch(ConsumerError):
    """The payload does not hash to what its entry says.

    Payload names are reusable, so a republish under a consumer that already
    read the old entry looks exactly like corruption; the entry is re-read once
    before this is reported.
    """


class LocalSource:
    """A bus root on this machine.

    It deletes nothing: the payload it would delete is the builder's own
    published one, whose entry is still present, which is the state a consumer
    is supposed to treat as a real error.
    """

    def __init__(self, source: BusSource):
        self.name = source.name
        self.bus = Bus(source.local_root)

    def ids_above(self, engine: str, cursor: int | None) -> list[int]:
        return self.bus.ids_above(engine, cursor)

    def read_entry(self, engine: str, commit_id: int) -> Entry | None:
        return self.bus.read_entry(engine, commit_id)

    def payload(self, engine: str, commit_id: int, dest: Path) -> Path:
        """Return a readable payload path. Local payloads are read in place."""
        del dest
        return self.bus.blob_path(engine, commit_id)

    def keep_payload(self) -> bool:
        return True

    def builder_state(self, engine: str):
        return self.bus.read_builder_state(engine)

    def bench_state(self, engine: str):
        return self.bus.read_bench_state(engine)


def open_source(source: BusSource) -> LocalSource:
    if source.is_local:
        return LocalSource(source)
    from .remote import SshSource

    return SshSource(source)


class BusConsumer:
    def __init__(
        self,
        cfg: Config,
        collector: BenchCollector,
        *,
        log: Callable[[str], None] | None = None,
        on_commit_done: Callable[[], None] | None = None,
        dry_run: bool = False,
        interval_secs: float = DEFAULT_INTERVAL_SECS,
    ):
        self.cfg = cfg
        self.collector = collector
        self.store = collector.store
        self.log = log or (lambda msg: None)
        self.on_commit_done = on_commit_done
        self.dry_run = dry_run
        self.interval_secs = interval_secs
        if cfg.bus is None:
            raise ConsumerError("no [bus] section in config")
        self.bus = Bus(cfg.bus.root)
        self.identity = host.identity()

    # --- cursors ---

    def cursor_file(self, source: BusSource, engine: str) -> Path:
        return cursor_path(self.cfg.out_dir, source.name, engine)

    def cursor(self, source: BusSource, engine: str) -> int | None:
        path = self.cursor_file(source, engine)
        try:
            return read_cursor(path)
        except BusError as e:
            # Start over rather than guess. max_done_commit_id looks like the
            # answer but is a maximum, not a watermark: an ad-hoc bench range
            # above the cursor would move it past entries that were never
            # measured, and ids_above is strict. Restarting costs nothing --
            # _skip_done walks past the done ones without fetching, which is
            # what a missing cursor file already does.
            self.log(f"{engine}: {e}; resuming from the start of the topic")
            return None

    def set_cursor(self, source: BusSource, engine: str, commit_id: int) -> None:
        write_cursor(self.cursor_file(source, engine), commit_id)

    # --- one entry ---

    def run_root(self, engine: str, commit_id: int) -> Path:
        return self.bus.root / "roots" / engine / str(commit_id)

    def provision(self, source, engine: EngineConfig, entry: Entry) -> Path:
        """Fetch, verify and unpack one entry into a run root."""
        root = self.run_root(engine.name, entry.commit_id)
        if root.exists():
            shutil.rmtree(root)
        tmp_payload = self.bus.tmp_dir / f"{engine.name}-{entry.commit_id}.tar.zst"
        tmp_payload.parent.mkdir(parents=True, exist_ok=True)
        payload = source.payload(engine.name, entry.commit_id, tmp_payload)
        if not payload.exists():
            raise ConsumerError(
                f"{engine.name} {entry.commit_id}: entry is published but its "
                f"payload is missing"
            )
        try:
            digest = sha256_file(payload)
            if digest != entry.blob_sha256:
                raise ShaMismatch(
                    f"{engine.name} {entry.commit_id}: payload sha256 {digest} "
                    f"does not match the entry's {entry.blob_sha256}"
                )
            tmp_root = root.with_name(root.name + ".unpacking")
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True)
            _unpack(payload, tmp_root)
            tmp_root.rename(root)
        except ShaMismatch:
            # Not an interrupted transfer, so there is nothing to resume from:
            # the bytes on disk are the wrong ones.
            if not source.keep_payload():
                tmp_payload.unlink(missing_ok=True)
            raise
        else:
            if not source.keep_payload():
                tmp_payload.unlink(missing_ok=True)
        self._trim_run_roots(engine.name, keep=entry.commit_id)
        return root

    def _trim_run_roots(self, engine: str, keep: int | None = None) -> None:
        """Keep the newest run_roots roots, and never the one about to be used.

        Roots are ranked by commit id, so a re-bench of an older commit unpacks
        the lowest-numbered root of the set: without ``keep`` the trim deletes
        it on the way out of provisioning, and every run of the commit the
        operator asked to repair then fails.
        """
        limit = self.cfg.bench.run_roots
        parent = self.bus.root / "roots" / engine
        try:
            roots = sorted(
                (p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()),
                key=lambda p: int(p.name),
            )
        except FileNotFoundError:
            return
        doomed = roots[:-limit] if limit > 0 else roots
        for old in doomed:
            if keep is not None and int(old.name) == keep:
                continue
            shutil.rmtree(old, ignore_errors=True)
        # A kill between unpacking and the rename leaves a full-size directory
        # that no commit id names, so nothing else would ever reclaim it.
        for partial in parent.glob("*.unpacking"):
            if keep is None or partial.name != f"{keep}.unpacking":
                shutil.rmtree(partial, ignore_errors=True)

    def free_gb(self) -> float:
        root = self.bus.root
        while not root.exists() and root != root.parent:
            root = root.parent
        return shutil.disk_usage(root).free / GB

    def bench_entry(
        self, source, engine: EngineConfig, entry: Entry, runs: int
    ) -> BenchOutcome:
        """Provision, bench and record one entry. The machine lock is held."""
        try:
            root = self.provision(source, engine, entry)
        except ShaMismatch:
            fresh = source.read_entry(engine.name, entry.commit_id)
            if fresh is None or fresh == entry:
                raise
            self.log(
                f"{engine.name} {entry.commit_id}: entry was republished, refetching"
            )
            entry = fresh
            root = self.provision(source, engine, entry)
        commit = {
            "hash": entry.hash,
            "commit_id": entry.commit_id,
            "date": entry.date,
            "timestamp": entry.timestamp,
            "title": entry.title,
        }
        provenance = self.collector.local_provenance(engine, runs)
        provenance.update(
            {
                "source": "bus",
                # The builder's, not this machine's: an Xcode or macOS update
                # there shifts every series on both bots on the same day.
                "toolchain": entry.builder.get("toolchain", ""),
                "build_cfg_hash": entry.build_cfg_hash,
            }
        )
        return self.collector.bench_at_root(
            engine,
            commit,
            root,
            runs,
            on_commit_done=self.on_commit_done,
            provenance=provenance,
        )

    # --- the cycle ---

    def drain(
        self,
        source: BusSource,
        engine_name: str,
        runs: int,
        should_stop: Callable[[], bool],
        take_lock: Callable[[], bool],
        release_lock: Callable[[], None],
    ) -> DrainResult:
        """Bench every entry above the cursor.

        Everything inside is wrapped: this writes the cursor and the state file
        on several paths, both under a filesystem that a stalled or paused
        cycle is very likely reacting to being full, and an escape here unwinds
        watch's per-engine loop and takes the daemon down with the git-driven
        engines and the pusher.
        """
        self._benched = 0
        try:
            return self._drain(
                source, engine_name, runs, should_stop, take_lock, release_lock
            )
        except TRANSPORT_ERRORS as e:
            self.log(f"{engine_name}: {e}")
            return DrainResult(self._benched, str(e))

    def _drain(
        self,
        source: BusSource,
        engine_name: str,
        runs: int,
        should_stop: Callable[[], bool],
        take_lock: Callable[[], bool],
        release_lock: Callable[[], None],
    ) -> DrainResult:
        engine = self.cfg.engines[engine_name]
        handle = open_source(source)
        if self.dry_run:
            return self._report_plan(source, handle, engine_name)
        try:
            state = self.bus.read_bench_state(engine_name)
        except BusError as e:
            # A state file this version cannot read (a rollback across a
            # version bump) must not kill the daemon at its first line.
            self.log(f"{engine_name}: {e}")
            return DrainResult(0, str(e))
        try:
            self.cfg.require_runs(engine_name)
        except ValueError as e:
            # Reported as a cycle error rather than raised: the cursor must not
            # advance past entries nothing measured, and one misconfigured
            # engine must not take the other down with it.
            state.last_error = str(e)
            self.log(f"{engine_name}: {state.last_error}")
            self._publish_state(
                engine_name, state, self.cursor(source, engine_name), handle, runs
            )
            return DrainResult(0, state.last_error)

        cursor = self.cursor(source, engine_name)
        # A killed bencher leaves this set, and every later cycle would
        # republish it with a fresh updated_at: bus status would report a bench
        # in progress forever and never show the file's age, which is the
        # crashed-versus-busy distinction it exists to make.
        state.in_flight = None

        # Before the flags are cleared: the reason the breaker tripped is what
        # tells an operator what to do, and it has to survive every cycle of
        # the backoff, not just the one that wrote it.
        stalled = state.stalled_since is not None
        if stalled and time.time() < (state.stall_retry_after or 0.0):
            # Say it every cycle: the backoff runs for hours, and silence here
            # is indistinguishable from a healthy idle engine.
            self.log(
                f"{engine_name}: stalled until "
                f"{time.strftime('%H:%M', time.localtime(state.stall_retry_after))}"
                f" -- {state.last_error}"
            )
            self._publish_state(engine_name, state, cursor, handle, runs)
            return DrainResult(0, state.last_error)

        # Cleared once per cycle rather than only on the path that benches
        # something: once the consumer is caught up, that path is never taken,
        # and a flag set by a disk that has since been freed would be
        # republished forever.
        state.benching_paused_by_floor = False
        state.last_error = None
        benched = 0
        # Before the floor check below, not only inside provision: a kill
        # mid-unpack leaves a multi-GB directory, and if that is what filled
        # the disk then the sweep that removes it would never be reached.
        self._trim_run_roots(engine_name)
        self._note_dropped_entries(handle, engine_name, state, cursor)
        # While stalled, one commit per window: a machine problem then costs a
        # commit per backoff instead of the whole backlog in one pass.
        limit = self.cfg.bench.max_consecutive_failures
        consecutive_failures = limit - 1 if stalled else 0
        while not should_stop():
            try:
                batch = handle.ids_above(engine_name, cursor)
            except TRANSPORT_ERRORS as e:
                # Listing failed, so there is nothing to skip past. Leave the
                # cursor and try again next cycle.
                state.last_error = f"listing {source.name}: {e}"
                self.log(f"{engine_name}: {state.last_error}")
                break
            if not batch:
                # Nothing above the cursor and nothing went wrong this cycle.
                # The per-entry success path is the only other place these are
                # cleared, and it is unreachable from here, so a stall fixed
                # while the topic was idle would be reported forever with no
                # error line beside it.
                if state.last_error is None:
                    state.stalled_since = None
                    state.stall_retry_after = None
                break
            commit_id, cursor = self._skip_done(
                source, engine_name, batch, cursor, should_stop
            )
            if commit_id is None:
                # The whole batch was already benched. Re-list rather than
                # stopping: entries published while we skipped are still ours.
                continue
            try:
                entry = handle.read_entry(engine_name, commit_id)
            except TRANSPORT_ERRORS as e:
                # Only a source that says "not there" advances the cursor; a
                # source that could not answer must not, or an outage would
                # silently skip unbenched commits.
                state.last_error = f"reading entry {commit_id}: {e}"
                self.log(f"{engine_name}: {state.last_error}")
                break
            if entry is None:
                # Retention dropped it, which is not an error but is a hole in
                # this machine's series. Persist a high-water mark: the
                # builder's highest_dropped signal self-erases as soon as the
                # cursor passes it, so nothing else would report this after the
                # next successful bench.
                self.log(
                    f"{engine_name}: entry {commit_id} was dropped by retention "
                    f"before this machine benched it"
                )
                state.skipped_dropped = max(state.skipped_dropped or 0, commit_id)
                cursor = commit_id
                self.set_cursor(source, engine_name, cursor)
                continue
            if self.free_gb() < self.cfg.bench.min_free_gb:
                state.benching_paused_by_floor = True
                state.last_error = f"only {self.free_gb():.0f}GB free"
                self.log(f"{engine_name}: paused, {self.free_gb():.0f}GB free")
                break
            if not take_lock():
                break
            # Everything from here to the finally, not just the bench: the
            # re-read and the cursor write below raise the same errors drain
            # is written to survive (ssh dropping over an hours-long lock
            # wait, a malformed entry, a full disk), and outside the try they
            # would leave the machine lock held for the life of the daemon --
            # blocking the builder and every ad-hoc command with it.
            try:
                # Re-read after the wait: take_lock can block for hours behind
                # a builder on a box that does both, and retention may have
                # dropped the entry meanwhile. Without this the payload is
                # simply missing, which the consumer is right to treat as a
                # real error -- but here it is ordinary retention.
                fresh = handle.read_entry(engine_name, commit_id)
                if fresh is None:
                    self.log(
                        f"{engine_name}: entry {commit_id} was dropped by "
                        f"retention while waiting for the machine"
                    )
                    state.skipped_dropped = max(state.skipped_dropped or 0, commit_id)
                    cursor = commit_id
                    self.set_cursor(source, engine_name, cursor)
                    continue
                entry = fresh
                state.in_flight = {
                    "commit_id": commit_id,
                    "phase": "bench",
                    "started_at": time.time(),
                }
                # Inside the try: it writes to the filesystem this path is
                # about running out of, and escaping here would leave the
                # machine lock held and kill the daemon.
                self._publish_state(engine_name, state, cursor, handle, runs)
                # The queue depth is the batch this cycle listed, so it does
                # not count entries published since; the hash is what matches
                # the run against a build without opening the state file.
                queued = (
                    len(batch)
                    if cursor is None
                    else sum(1 for i in batch if i > cursor)
                )
                self.log(
                    f"{engine_name}: benching {commit_id} ({entry.hash[:8]}), "
                    f"{queued} above the cursor"
                )
                outcome = self.bench_entry(handle, engine, entry, runs)
            except TRANSPORT_ERRORS as e:
                state.last_error = str(e)
                state.in_flight = None
                self.log(f"{engine_name}: {e}")
                break
            finally:
                release_lock()
            state.in_flight = None
            benched += 1
            self._benched = benched
            # After the commit is recorded, so a crash between the two
            # re-benches one commit, which is_done absorbs.
            cursor = commit_id
            self.set_cursor(source, engine_name, cursor)

            if outcome_status(outcome) == "failed":
                # Zero scores parsed, so the binary did not run at all. The
                # likeliest cause is this machine or the run set, not the
                # commit, and a cycle drains: without a breaker one pass marks
                # every entry in the topic failed, and is_done is status-blind,
                # so none of them is ever revisited.
                consecutive_failures += 1
                if consecutive_failures >= limit:
                    state.stalled_since = state.stalled_since or time.time()
                    state.stall_retry_after = time.time() + min(
                        self.interval_secs * STALL_BACKOFF, STALL_BACKOFF_CAP_SECS
                    )
                    state.last_error = (
                        f"{consecutive_failures} commits in a row produced no "
                        f"scores; this is the machine or the run set, not the "
                        f"tree. Check the run set, then "
                        f"'slipstream clear {engine_name} <first> <last>' and "
                        f"'watch --reset-cursor {engine_name}=<first-1>'."
                    )
                    self.log(f"{engine_name}: stalled -- {state.last_error}")
                    break
            else:
                consecutive_failures = 0
                state.stalled_since = None
                state.stall_retry_after = None
        try:
            self._publish_state(engine_name, state, cursor, handle, runs)
        except TRANSPORT_ERRORS as e:
            # Same exposure as the call inside the loop, which is guarded for
            # the same reason: it writes to the filesystem this path is about
            # running out of, and the caught-up path does no floor check at all.
            self.log(f"{engine_name}: writing bench state: {e}")
            return DrainResult(benched, str(e))
        return DrainResult(benched, state.last_error)

    def _report_plan(self, source, handle, engine_name: str) -> DrainResult:
        """Say what a real cycle would bench, and write nothing.

        A dry run has to be inert here, not merely quiet: the real path
        fetches, unpacks, clears the commit's partial scores and advances the
        cursor, and a cursor cannot go back on its own.
        """
        cursor = self.cursor(source, engine_name)
        try:
            ids = handle.ids_above(engine_name, cursor)
        except TRANSPORT_ERRORS as e:
            self.log(f"{engine_name}: listing {source.name}: {e}")
            return DrainResult(0, str(e))
        todo = [
            i for i in ids if not self.store.is_done(engine_name, self.cfg.platform, i)
        ]
        self.log(
            f"{engine_name}: would bench {len(todo)} of {len(ids)} entries above "
            f"{cursor} from {source.name}"
            + (f" (first {todo[0]}, last {todo[-1]})" if todo else "")
        )
        return DrainResult()

    def _note_dropped_entries(self, handle, engine_name, state, cursor) -> None:
        """Record entries retention deleted before this machine could bench them.

        The comparison has to be made and kept here, not left to bus status:
        the builder's highest_dropped stops being evidence the moment the
        cursor passes it, which the next successful bench does. An operator
        looking at a gap in the series a week later would otherwise find
        nothing that says one was ever there.
        """
        try:
            dropped = handle.builder_state(engine_name).highest_dropped
        except TRANSPORT_ERRORS:
            return  # the cycle will report the failure on its own
        if dropped is None or (cursor is not None and dropped <= cursor):
            return
        if (state.skipped_dropped or 0) >= dropped:
            return
        state.skipped_dropped = dropped
        self.log(
            f"{engine_name}: entries up to {dropped} were dropped by retention "
            f"before this machine benched them"
        )

    def _skip_done(
        self, source, engine_name, batch, cursor, should_stop
    ) -> tuple[int | None, int | None]:
        """Advance past entries this machine has already benched.

        One listing covers the whole batch rather than one per entry: after a
        cursor reset most of them are done, and re-listing for each would be an
        ssh round trip per skipped commit. Advancing without fetching is the
        point -- pulling hundreds of MB to discard it is what makes a reset
        expensive.
        """
        for commit_id in batch:
            if should_stop():
                return None, cursor
            if not self.store.is_done(engine_name, self.cfg.platform, commit_id):
                return commit_id, cursor
            cursor = commit_id
            self.set_cursor(source, engine_name, cursor)
        return None, cursor

    def _publish_state(
        self, engine_name, state: BenchState, cursor, handle, runs: int | None = None
    ):
        state.bot = self.cfg.bot_name
        state.cursor = cursor
        try:
            state.lag = len(handle.ids_above(engine_name, cursor))
        except TRANSPORT_ERRORS:
            # Writing the state file is the last thing a failing cycle does;
            # it must not raise for the same reason the cycle did.
            state.lag = -1
        state.status_counts = self.store.status_counts(engine_name, self.cfg.platform)
        engine = self.cfg.engines[engine_name]
        env = {
            "slipstream_version": __version__,
            "hw_model": self.identity["hw_model"],
            "os_version": self.identity["os_version"],
            "runs": runs if runs is not None else state.env.get("runs"),
            "run_configs": [
                f"{c.suite}/{c.variant}" for c in self.collector.run_configs(engine)
            ],
            "harness": self.collector.harness_revs(),
        }
        # `since` is when these values took effect, so a divergence report can
        # say when the two boxes parted rather than only that they differ now.
        if {k: v for k, v in state.env.items() if k != "since"} != env:
            env["since"] = time.time()
        else:
            env["since"] = state.env.get("since", time.time())
        state.env = env
        self.bus.write_bench_state(engine_name, state)


def _unpack(payload: Path, dest: Path) -> None:
    """Unpack a tar.zst payload, refusing members that escape the destination."""
    proc = subprocess.Popen(["zstd", "-dc", str(payload)], stdout=subprocess.PIPE)
    failed = None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            root = dest.resolve()
            for member in tf:
                target = (dest / member.name).resolve()
                # relative_to, not a string prefix: a sibling directory whose
                # name merely starts with the root's would pass that.
                if target != root and root not in target.parents:
                    raise ConsumerError(
                        f"payload member escapes the run root: {member.name}"
                    )
                tf.extract(member, dest, filter="tar")
    except BaseException as e:
        failed = e
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()
        # Closing the pipe kills zstd with EPIPE, so its exit code is only
        # meaningful when we got through the archive. Reporting it over the
        # rejection would tell the operator "decompression failed" for what is
        # actually a path-traversal refusal.
        if proc.wait() != 0 and failed is None:
            raise ConsumerError(f"zstd failed to decompress {payload}")
