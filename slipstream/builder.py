# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Build every commit once and publish the artifact for both boxes to bench.

The builder walks an engine's history upward: check out, build, package the
run set, publish to the bus, and on to the next until it is caught up. It
holds the machine lock across all of that, because zstd -T0 and a compile
saturate every core and a build next to a measurement contaminates it.

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

from . import delta, host
from .bus import (
    Blob,
    Bus,
    BuilderState,
    BusError,
    Delta,
    Entry,
    patch_name,
    sha256_file,
    tree_hash,
)
from .collector import BenchCollector, BuildStepError, FetchError
from .config import Config, DeltaConfig, EngineConfig
from .delta import DeltaError
from .lock import MachineLock
from .models import CommitKey
from .resolve import (
    BuildJob,
    DerivedResolver,
    EmbedderResolver,
    IdentityResolver,
    Resolver,
)

# A stalled engine still retries, or the guard against a permanent burn becomes
# a permanent stall. Each attempt is a full checkout, sync and compile holding
# the machine lock against the bencher, so it backs off to a multiple of the
# poll interval.
STALL_BACKOFF = 4
STALL_BACKOFF_CAP_SECS = 6 * 3600
DEFAULT_INTERVAL_SECS = 1800.0
# A cycle drains, but not forever: commits that keep landing must not keep it
# from re-reading its state, fetching again, or stopping when asked.
MAX_JOBS_PER_CYCLE = 50
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
    key: CommitKey | None = None
    published: bool = False
    failed_kind: str | None = None
    status: str | None = None


def build_cfg_hash(engine: EngineConfig, inherited: str | None = None) -> str:
    """Identifies the build inputs an artifact was produced with.

    Everything the user config can override, not just the compiler flags: the
    template offers build_cmd and sync_cmd as per-box settings, so hashing only
    gn_args would give two boxes building differently the same hash and leave
    run_env asserting inputs match when they do not. The run set is in for the
    same reason and is the one most likely to change -- adding a file to it
    changes what every later artifact contains, with nothing else to notice.

    A derived engine compiles nothing of its own; ``inherited`` is the inner
    entry's hash, so a change to how the inner engine is built shows on the
    derived series too, which is where the engine actually runs.
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
    if engine.embeds:
        # Where the inner engine goes is part of what was built. Appended
        # only when there is one, so an engine built from its own checkout
        # hashes exactly as it did before embedding existed.
        parts.append(f"{engine.embeds}@{engine.pin}")
    if engine.derives:
        parts.append(f"derives {engine.derives}@{inherited or ''}")
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


@dataclass(frozen=True)
class _Base:
    """The archive a new blob of a run set path is planned against."""

    id: str
    sha256: str
    bytes: int

    @classmethod
    def of(cls, blob: Blob) -> _Base:
        if blob.delta is None:
            return cls(blob.id, blob.sha256, blob.bytes)
        d = blob.delta
        return cls(d.base_id, d.base_sha256, d.base_bytes)


class _Stored:
    """What the bus already holds, read once per run set.

    A tree that is already a blob of some entry -- this engine's or another's,
    as a full archive or a delta -- is named again rather than re-stored. The
    newest blob at each of this engine's run set paths is where a new delta's
    base comes from: its own archive, or the archive it was itself planned
    against, so every delta since the last full store shares one base.
    """

    def __init__(self, bus: Bus, engine: str | None):
        self.bus = bus
        self.blobs: dict[str, Blob] = {}
        self.latest: dict[str, Blob] = {}
        for name in bus.engines():
            for entry in bus.entries(name):
                for blob in entry.blobs:
                    self.blobs[blob.id] = blob
                    if name == engine:
                        self.latest[blob.path] = blob

    def known(self, blob_id: str) -> Blob | None:
        blob = self.blobs.get(blob_id)
        if blob is None:
            return None
        if all(self.bus.has_object(o.name) for o in blob.objects()):
            return blob
        return None

    def base_for(self, path: str) -> _Base | None:
        blob = self.latest.get(path)
        if blob is None:
            return None
        base = _Base.of(blob)
        return base if self.bus.has_blob(base.id) else None


def store_run_set(
    bus: Bus,
    src_dir: Path,
    run_set: list[str],
    *,
    caffeinate: bool = True,
    engine: str | None = None,
    delta_cfg: DeltaConfig | None = None,
    log: Callable[[str], None] = lambda msg: None,
) -> list[Blob]:
    """Store each run set entry once per distinct tree, as a delta when it pays.

    The tree hash is computed first and the entry packaged only if no blob
    of that tree is stored: the ICU data file, the browser bundle, the
    runtime a commit did not touch are then hashed per publish but stored
    once. A tree new to the store is archived whole, then planned against
    the engine's newest base at that path; the plan replaces the archive
    when its patches fit the budget, and otherwise the archive stays and is
    the next base. Nothing on the delta path can fail a publish: any error
    in it is logged and the archive stored whole. The returned blobs are
    what the manifest names, in run set order.
    """
    missing = [entry for entry in run_set if not (src_dir / entry).exists()]
    if missing:
        raise BuildError(f"run_set entries missing from the build: {missing}")
    stored = _Stored(bus, engine)
    blobs = []
    for entry in run_set:
        blob_id = tree_hash(src_dir, entry)
        dest = bus.blob_path(blob_id)
        if dest.exists():
            blobs.append(Blob(entry, blob_id, sha256_file(dest), dest.stat().st_size))
            continue
        known = stored.known(blob_id)
        if known is not None:
            blobs.append(Blob(entry, blob_id, known.sha256, known.bytes, known.delta))
            continue
        tmp = bus.tmp_blob(blob_id)
        package(src_dir, [entry], tmp, caffeinate=caffeinate)
        planned = None
        if delta_cfg is not None and delta_cfg.enabled:
            planned = _try_delta(
                bus, tmp, stored.base_for(entry), delta_cfg, log, entry
            )
        if planned is not None:
            tmp.unlink()
            blobs.append(Blob(entry, blob_id, delta=planned))
        else:
            bus.store_blob(tmp, blob_id)
            blobs.append(Blob(entry, blob_id, sha256_file(dest), dest.stat().st_size))
    return blobs


def _try_delta(
    bus: Bus,
    archive: Path,
    base: _Base | None,
    cfg: DeltaConfig,
    log: Callable[[str], None],
    label: str,
) -> Delta | None:
    """Plan ``archive`` against ``base``, or say why it stays whole.

    Patches are stored as they are produced; the ones of a plan that is
    abandoned, over budget or by an error, are unreferenced and the sweep
    after the publish reclaims them.
    """
    size = archive.stat().st_size
    if base is None:
        log(f"{label}: no base, stored whole ({size / MB:.0f}MB)")
        return None
    if size < cfg.min_bytes:
        return None
    budget = int(size * cfg.max_ratio)

    def store_patch(sha: str, data: bytes) -> None:
        name = patch_name(sha)
        if bus.has_object(name):
            return
        tmp = bus.tmp_object(name)
        tmp.write_bytes(data)
        bus.store_object(tmp, name)

    t0 = time.monotonic()
    try:
        with (
            subprocess.Popen(
                ["zstd", "-dc", str(bus.blob_path(base.id))], stdout=subprocess.PIPE
            ) as base_proc,
            subprocess.Popen(
                ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE
            ) as target_proc,
        ):
            plan = delta.plan(
                base_proc.stdout,
                target_proc.stdout,
                store_patch,
                block_bytes=cfg.block_bytes,
                workers=cfg.workers or delta.default_workers(),
                max_patch_bytes=budget,
            )
            if plan is not None:
                # The plan ends with the target; a longer base is left
                # unread, and zstd must reach its own end for its exit code
                # to say whether the archive was sound.
                while base_proc.stdout.read(1 << 20):
                    pass
            # On an abandoned plan the reads stopped early; closing the pipes
            # ends zstd with EPIPE, which is the intended outcome.
            base_proc.stdout.close()
            target_proc.stdout.close()
        if plan is not None and (base_proc.returncode or target_proc.returncode):
            raise DeltaError("zstd failed while streaming the archives")
    except Exception as e:  # noqa: BLE001 -- the delta path may not fail a publish
        log(f"{label}: delta against {base.id[:12]} failed, stored whole: {e}")
        return None
    secs = time.monotonic() - t0
    if plan is None:
        log(
            f"{label}: delta against {base.id[:12]} exceeds {cfg.max_ratio:.0%} of "
            f"{size / MB:.0f}MB, stored whole as the new base ({secs:.0f}s)"
        )
        return None
    log(
        f"{label}: delta against {base.id[:12]}: {plan.patch_bytes / MB:.1f}MB of "
        f"patches for a {size / MB:.0f}MB archive ({secs:.0f}s)"
    )
    return Delta(base.id, base.sha256, base.bytes, plan)


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
        self._resolvers: dict[str, Resolver] = {}

    # --- resolving what to build ---

    def resolver(self, engine_name: str) -> Resolver:
        """The reader of this engine's series.

        One per engine, made on first use: the collector it wraps is shared,
        and the choice of resolver is a property of the engine's config, not
        of the cycle. An embedded engine needs the inner one's checkout on
        this box as well as its own; a missing one is this engine's problem,
        raised on the channel run_cycle already treats as misconfiguration.
        """
        if engine_name not in self._resolvers:
            engine = self.cfg.engines[engine_name]
            if engine.derives:
                inner = self.cfg.engines.get(engine.derives)
                if inner is None:
                    raise ValueError(
                        f"{engine_name} derives from {engine.derives}, which "
                        f"this machine has no [engines.{engine.derives}] for"
                    )
                self._seed_embedder_numbers(engine_name)
                self._resolvers[engine_name] = DerivedResolver(
                    self.bus,
                    engine,
                    inner,
                    number_for=lambda app, name=engine_name: self.store.embedder_number(
                        name, app.version, app.title
                    ),
                )
            elif engine.embeds:
                inner = self.cfg.engines.get(engine.embeds)
                if inner is None:
                    raise ValueError(
                        f"{engine_name} is built around {engine.embeds}, which "
                        f"this machine has no [engines.{engine.embeds}] for"
                    )
                inner.require_src_dir()
                self._resolvers[engine_name] = EmbedderResolver(
                    self.collector,
                    engine,
                    inner,
                    pin=engine.pin,
                    roll_file=engine.roll_file,
                    roll_regex=engine.roll_regex,
                )
            else:
                self._resolvers[engine_name] = IdentityResolver(self.collector, engine)
        return self._resolvers[engine_name]

    def _seed_embedder_numbers(self, engine_name: str) -> None:
        """Numbers the published manifests already use, into a store that
        lacks them. The topic is the durable record; the database is a
        cache of it that a rebuilt builder box starts without."""
        known: dict[str, int] = {}
        for entry in self.bus.entries(engine_name):
            if entry.embedder_hash:
                known[entry.embedder_hash] = entry.embedder_id
        if known:
            self.store.seed_embedder_numbers(engine_name, known)

    def frontier(self, engine_name: str) -> CommitKey | None:
        """The highest key this builder has finished with.

        Published entries and terminal build failures both count, so a compile
        failure at the head is not rebuilt every cycle. [build] from is
        consulted only when there is neither, so a restart cannot rewind.

        A derived engine with no history starts at the bottom: its candidates
        are the inner entries on this bus that carry the full run set, a
        bounded list with no past to reach into, where a git-driven engine
        with no [build] from would start at the first commit of the repo.
        """
        keys = self.bus.keys(engine_name)
        published = keys[-1] if keys else None
        failed = self.store.max_terminal_build_key(engine_name)
        candidates = [c for c in (published, failed) if c is not None]
        if candidates:
            return max(candidates)
        start = self.cfg.build.start_from.get(engine_name)
        if start is None and self.cfg.engines[engine_name].derives:
            return CommitKey(0, 0)
        return start

    def consecutive_burns(self, engine_name: str) -> int:
        """Infrastructure burns with no successful publish between them.

        A fact about the topic, not about build_state: a successful build
        leaves no row there, so counting rows alone would trip on three
        unrelated outages months apart and stall a healthy engine.
        """
        keys = self.bus.keys(engine_name)
        floor = keys[-1] if keys else self.cfg.build.start_from.get(engine_name)
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

    def next_job(self, engine_name: str) -> BuildJob | None:
        """The next job to build, or None when up to date.

        Retries come first: entries above the failed commit exist by the time
        anyone retries, so the frontier rule would never pick it up again.
        """
        resolver = self.resolver(engine_name)
        # Publication is durable before local cleanup. A failed cleanup must
        # never turn a fulfilled retry into another checkout and compile.
        published = set(self.bus.keys(engine_name))
        recorded = {
            CommitKey.from_commit(row) for row in self.store.build_failures(engine_name)
        }
        recorded.update(self.store.build_retries_requested(engine_name))
        for key in recorded:
            if key in published:
                self.store.clear_build_state(engine_name, key)
        for key in self.store.build_retries_requested(engine_name):
            job = resolver.for_key(key)
            if job:
                return job
        frontier = self.frontier(engine_name)
        if frontier is None:
            raise BuildError(
                f"{engine_name} has no build history and no [build] from entry; "
                f"set one to say where to start"
            )
        return resolver.next_after(frontier)

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

        job = self.next_job(engine_name)
        if job is None:
            state.last_error = None
            # A builder killed between publishing and writing its state leaves
            # this set; refreshing it every cycle would report a package step
            # in progress forever.
            state.in_flight = None
            self._publish_state(engine_name, state)
            return BuildResult()

        key = job.key
        state.in_flight = self._in_flight(job, "build")
        self._publish_state(engine_name, state)
        self.lock.set_job(f"{engine_name} {key}")
        # Said out loud: without -v nothing else reaches the console until the
        # build has published or failed, and a long one looks like a hang.
        self.log(f"{engine_name}: building {key}")

        row = self.store.get_build_state(engine_name, key)
        was_retry = row is not None and row["status"] == "retry_requested"
        log_path = self.cfg.logs_dir / f"build-{engine_name}-{key}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if job.compiles:
            self.log(f"{engine_name}: building {key} ({job.hash[:8]})")
            failure = self.collector.build_at(
                engine, job.checkout_hash, log_path, pins=job.pins
            )
            if failure is not None:
                return self._record_failure(
                    engine_name,
                    key,
                    failure,
                    log_path,
                    state,
                    stalled=stalled,
                    was_retry=was_retry,
                )
        else:
            self.log(
                f"{engine_name}: packaging {key} from {job.inherits.engine} "
                f"{job.inherits.key} ({job.hash[:8]})"
            )

        state.in_flight = self._in_flight(job, "package")
        self._publish_state(engine_name, state)
        try:
            entry = self._package_and_publish(engine, job, int(time.time() - started))
        except (BuildError, OSError) as e:
            # Packaging is this machine's business, not the commit's.
            return self._record_failure(
                engine_name,
                key,
                BuildStepError("package", 1),
                log_path,
                state,
                stalled=stalled,
                was_retry=was_retry,
                message=str(e),
            )

        cleanup_error = None
        try:
            self.store.clear_build_state(engine_name, key)
        except sqlite3.Error as e:
            self.store.conn.rollback()
            cleanup_error = f"published {key}; database cleanup deferred: {e}"
            self.log(f"{engine_name}: {cleanup_error}")
        state.stall_retry_after = None
        dropped = self.bus.prune(
            engine_name, self.cfg.build.retain_gb * GB, keep=entry.key
        )
        if dropped:
            self.log(f"{engine_name}: retention dropped {len(dropped)} entries")
            state.highest_dropped = max(
                k for k in (state.highest_dropped, *dropped) if k is not None
            )
        # After the prune of every publish, not only ones that dropped
        # something: a retry republishes under new blob ids and leaves the
        # old ones with no manifest.
        self.bus.sweep_blobs()
        state.stalled_since = None
        state.last_error = cleanup_error
        state.in_flight = None
        self._publish_state(engine_name, state)
        patches = sum(b.delta.plan.patch_bytes for b in entry.blobs if b.delta)
        self.log(
            f"{engine_name}: published {key} ({entry.blob_bytes / MB:.0f}MB"
            + (f", {patches / MB:.1f}MB of it patches" if patches else "")
            + f", {entry.build_secs}s)"
        )
        return BuildResult(key=key, published=True)

    @staticmethod
    def _in_flight(job: BuildJob, phase: str) -> dict:
        """What the state file says is being worked on.

        The outer commit's hash goes in only when there is one: bus status
        names the roll being built for an embedded engine, and an engine
        built from its own checkout keeps the record it always wrote.
        """
        record = {
            "commit_id": job.key.commit_id,
            "embedder_id": job.key.embedder_id,
            "phase": phase,
            "started_at": time.time(),
        }
        if job.embedder:
            record["embedder_hash"] = job.embedder.get("hash", "")
        return record

    def _package_and_publish(
        self, engine: EngineConfig, job: BuildJob, build_secs: int
    ) -> Entry:
        key = job.key
        commit = job.commit
        inherited = None
        if job.compiles:
            blobs = self._store(engine)
        else:
            # The inner entry's blobs by reference, then this engine's own.
            # The app can be replaced under us by its updater: its identity
            # is read again after archiving, and a change means what was
            # archived is not what the job names. Content addressing makes
            # the torn blob unique, not correctly labelled, so it is dropped
            # on the floor (the sweep reclaims it) and the cycle retries.
            resolver = self.resolver(engine.name)
            before, _ = resolver.installed()
            own = self._store(engine)
            after, _ = resolver.installed()
            if after != before:
                raise BuildError(
                    f"{before.title} became {after.title} while it was being "
                    f"packaged; retrying next cycle"
                )
            blobs = [*job.inherits.blobs, *own]
            inherited = job.inherits.build_cfg_hash
        entry = Entry(
            engine=engine.name,
            commit_id=key.commit_id,
            embedder_id=key.embedder_id,
            embedder=dict(job.embedder),
            pins=dict(job.pins),
            hash=commit["hash"],
            date=commit.get("date", ""),
            timestamp=int(commit.get("timestamp", 0)),
            title=commit.get("title", ""),
            build_cfg_hash=build_cfg_hash(engine, inherited),
            blobs=blobs,
            builder=dict(self.identity),
            built_at=int(time.time()),
            build_secs=build_secs,
        )
        self.bus.publish(entry)
        return entry

    def _store(self, engine: EngineConfig) -> list[Blob]:
        """This engine's own run set into the store, under its delta policy."""
        return store_run_set(
            self.bus,
            engine.require_src_dir(),
            engine.require_run_set(),
            engine=engine.name,
            delta_cfg=self.cfg.build.delta,
            log=lambda msg: self.log(f"{engine.name}: {msg}"),
        )

    def migrate(self, should_stop=lambda: False) -> None:
        """Bring the root's entries to the current format, once at startup.

        Under the machine lock like every other write to the root. The
        builder does it rather than the consumer because a bench-only box has
        no topic of its own and so nothing to migrate.
        """
        if not self.lock.acquire(should_stop, wait=True, log=self.log):
            return
        try:
            migrated = self.bus.migrate()
        finally:
            self.lock.release()
        if migrated:
            self.log(f"migrated {len(migrated)} entries to the blob store")

    def _record_failure(
        self,
        engine_name: str,
        key: CommitKey,
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
                engine_name, key, status, kind, str(log_path)
            )
        else:
            # A retry keeps its allowance while it still has attempts left.
            # infra_retry is only reachable through the frontier, which has
            # already moved past a commit anyone had to retry, so writing it
            # here would strand the commit: not retryable, and not terminal, so
            # absent from bus status too.
            pending = "retry_requested" if was_retry else "infra_retry"
            attempts = self.store.record_build_failure(
                engine_name, key, pending, kind, str(log_path)
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
                self.store.set_build_status(engine_name, key, status)
        state.last_error = message or f"{kind} failed on {key} (attempt {attempts})"
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
        self.log(f"{engine_name}: {key} {status} at {kind}, log {log_path}")
        return BuildResult(key=key, failed_kind=kind, status=status)

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
        """Drain what the engines have to build; returns how many published.

        Round-robin over the engines, one job each per turn, so a long roll
        on one cannot starve another. ``[build] batch`` says how many jobs
        one hold of the machine covers before the lock is handed on and
        taken again at the back of the queue; zero holds it until the known
        work is gone, which is what a warm build cache and a bench-only peer
        box waiting on the bus both want. The cycle itself is bounded, so
        work that keeps arriving cannot keep it from refreshing or stopping.

        An engine stays in the rotation only while it publishes. A failure
        drops it until the next cycle: an infrastructure failure would
        otherwise be retried back to back, and a stall or a floor has
        already said it wants to wait. Each engine fetches once per cycle,
        in its first turn; what lands after that is next cycle's.
        The last attempted engine is persisted, so bounded cycles and daemon
        restarts resume the rotation rather than favoring its first engine.
        """
        published = 0
        jobs = 0
        rotation = self.store.rotation_order("build", engine_names)
        fetched: set[str] = set()
        batch = self.cfg.build.batch
        while rotation and jobs < MAX_JOBS_PER_CYCLE and not should_stop():
            if not self.lock.acquire(should_stop, wait=True, log=self.log):
                break
            in_hold = 0
            try:
                while (
                    rotation
                    and jobs < MAX_JOBS_PER_CYCLE
                    and (batch == 0 or in_hold < batch)
                    and not should_stop()
                ):
                    name = rotation.pop(0)
                    try:
                        self.store.record_rotation_turn("build", name)
                        if name not in fetched:
                            self.resolver(name).fetch()  # so origin/main is current
                            fetched.add(name)
                        result = self.build_one(name)
                    except FetchError:
                        self.log(f"{name}: skipping until fetch succeeds")
                        continue
                    except (
                        BuildError,
                        BusError,
                        ValueError,
                        OSError,
                        sqlite3.Error,
                    ) as e:
                        # ValueError is how require_run_set and require_src_dir
                        # report a misconfigured engine; BusError is a state
                        # file this version cannot read; OSError is a full
                        # disk or a missing tar; and sqlite3.Error, which is
                        # not an OSError, is every build_state write. None of
                        # them may take the working engines down too.
                        self.store.conn.rollback()
                        self.log(f"{name}: {e}")
                        # The operation ended, even if its bookkeeping could
                        # not be committed. Clear stale in-flight state and
                        # expose the error.
                        try:
                            state = self.bus.read_builder_state(name)
                            state.in_flight = None
                            state.last_error = str(e)
                            self._publish_state(name, state)
                        except (BusError, OSError, sqlite3.Error):
                            pass
                        continue
                    if result.key is None:
                        continue  # nothing to build: up to date, floor, stall
                    jobs += 1
                    in_hold += 1
                    if result.published:
                        published += 1
                        rotation.append(name)
            finally:
                self.lock.release()
        return published
