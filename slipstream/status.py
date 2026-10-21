# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""What an operator needs to see when the bus is not moving.

Everything here is read-only. The point is to distinguish a long legitimate
hold from a hang: a crashed builder leaves a plausible frontier and a null
last_error forever, and a crashed bencher leaves a frozen cursor that looks
exactly like a three-hour bench in progress. So every reading comes with the
age of the file it was read from.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from datetime import UTC, datetime

from . import __version__, host
from . import bench_window
from .bus import Bus, BusError, cursor_path, read_cursor
from .config import BusSource, Config
from .consumer import TRANSPORT_ERRORS, open_source
from .lock import Holder, MachineLock, paused_until
from .models import CommitKey
from .store import CommitStore

GB = 1_000_000_000


@dataclass
class EngineStatus:
    engine: str
    source: str | None = None
    driven_by: str = "git"
    frontier: CommitKey | None = None
    cursor: CommitKey | None = None
    lag: int | None = None
    lowest_retained: CommitKey | None = None
    dropped_unread_up_to: CommitKey | None = None
    failed_builds: list[dict] = field(default_factory=list)
    failed_builds_total: int = 0
    stalled_since: float | None = None
    builder_error: str | None = None
    builder_age: float | None = None
    paused_by_floor: bool = False
    builder_in_flight: dict | None = None
    # This machine's own bencher, which is what actually stopped when nothing
    # is being measured. Distinct from the source's, above.
    bench_paused_by_floor: bool = False
    bench_error: str | None = None
    bench_in_flight: dict | None = None
    bench_age: float | None = None
    bench_stalled_since: float | None = None
    bench_skipped_dropped: CommitKey | None = None
    bench_skipped_upstream: CommitKey | None = None
    bench_skipped_permanent: CommitKey | None = None
    permanent_skip_reason: str | None = None
    status_counts: dict = field(default_factory=dict)
    orphan_scores: int = 0
    provenance: dict = field(default_factory=dict)
    env_divergence: list[str] = field(default_factory=list)
    env_since: float | None = None
    blob_gb: float = 0.0
    roots_gb: float = 0.0
    error: str | None = None
    transfer: dict = field(default_factory=dict)
    peer_age: float | None = None


@dataclass
class BusStatus:
    engines: list[EngineStatus] = field(default_factory=list)
    free_gb: float = 0.0
    lock_holder: str | None = None
    lock_waiters: list[str] = field(default_factory=list)
    paused_until: float | None = None
    window: WindowStatus | None = None


@dataclass
class WindowStatus:
    config: str
    decision: bench_window.Decision
    schedule: str
    override: bench_window.Override | str | None
    holder: Holder | None
    paused: float | None
    waiters: list[str]
    blockers: list[str] = field(default_factory=list)


def window_status(cfg, label):
    config = cfg.bench
    override = bench_window.read_override()

    decision = bench_window.evaluate(
        config.timezone, config.allowed_windows, override, datetime.now(UTC)
    )
    machine = MachineLock("status")
    try:
        holder = machine.probe()
        waiters = [str(w) for w in machine.waiters()]
    except OSError:
        holder = Holder(0, "unknown", 0)
        waiters = []
    schedule = (
        "unrestricted"
        if config.allowed_windows is None
        else (f"{config.allowed_windows or 'no scheduled opening'} ({config.timezone})")
    )
    blockers = []
    root = cfg.bus.root if cfg.bus else cfg.out_dir
    while not root.exists() and root != root.parent:
        root = root.parent
    try:
        free = shutil.disk_usage(root).free / GB
        if free < config.min_free_gb:
            blockers.append(
                f"disk floor: {free:.0f}GB free, {config.min_free_gb:g}GB required"
            )
        if cfg.bus:
            bus = Bus(cfg.bus.root)
            for name in cfg.engines:
                state = bus.read_bench_state(name)
                if state.last_error:
                    age = (
                        int(time.time() - state.updated_at)
                        if state.updated_at
                        else "unknown"
                    )
                    blockers.append(
                        f"{name}: last reported blocker ({age}s ago): {state.last_error}"
                    )
    except (OSError, BusError) as exc:
        blockers.append(f"cannot read other blockers: {exc}")
    return WindowStatus(
        label, decision, schedule, override, holder, paused_until(), waiters, blockers
    )


def render_window(report, echo):
    def instant(value):
        return value.astimezone().isoformat(timespec="seconds")

    decision, holder = report.decision, report.holder
    next_time = instant(decision.next_change) if decision.next_change else None
    if decision.allowed:
        summary = "Benchmarking permitted" + (
            f" until {next_time}" if next_time else ""
        )
    elif holder is not None and (
        holder.role not in ("bench", "watch", "build") or not holder.alive
    ):
        summary = "Admission closed; activity unknown"
    elif (
        holder is not None
        and holder.role in ("bench", "watch")
        and not holder.job.startswith("maintenance:")
    ):
        summary = "Finishing current commit"
    elif decision.error:
        summary = (
            "Admission blocked; repair override before relying on normal-use status"
        )
    else:
        summary = "Normal use permitted " + (
            f"until {next_time}"
            if next_time
            else "until reset or replaced"
            if decision.reason == "forced close"
            and isinstance(report.override, bench_window.Override)
            and report.override.expires_at is None
            else "(no scheduled opening)"
        )
    echo(f"{summary} — for {report.config}")
    echo(f"  window: {decision.reason}; schedule: {report.schedule}")
    if decision.error:
        echo(f"  error: {decision.error}")
    if next_time:
        echo(f"  next policy change: {next_time}")
    override = report.override
    if isinstance(override, bench_window.Override):
        expiry = (
            instant(override.expires_at)
            if override.expires_at
            else "until reset or replaced"
        )
        echo(f"  override: {override.mode}, expires {expiry}")
    if holder is not None:
        started = (
            datetime.fromtimestamp(holder.since)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        echo(f"  machine activity: {holder}; started {started}")
    else:
        echo("  machine activity: no active holder")
    if report.paused:
        echo(
            f"  PAUSED by operator until {instant(datetime.fromtimestamp(report.paused).astimezone())}; independent of window"
        )
    for blocker in report.blockers:
        echo(f"  {blocker}")
    for waiter in report.waiters:
        echo(f"  queued: {waiter}")
    echo(
        "  Configured-policy view: benchers must use this policy. Restart affected benchers after TOML changes; benchers retain startup configuration."
    )


def _dir_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def local_env(cfg: Config, collector, engine: str | None = None) -> dict:
    """This machine's copy of the inputs both boxes share.

    ``run_configs`` is per engine, because that is how the consumer records it:
    comparing a union over every configured engine against one engine's list
    reports a difference on every engine with fewer variants than the busiest.
    """
    identity = host.identity()
    env = {
        "slipstream_version": __version__,
        "hw_model": identity["hw_model"],
        "os_version": identity["os_version"],
        "harness": collector.harness_revs(),
    }
    if engine is not None:
        env["run_configs"] = collector.run_config_labels(cfg.engines[engine])
        env["runner_cfg_hash"] = collector._runner(cfg.engines[engine]).cfg_hash()
        env["suite_cfg_hash"] = collector.suite_cfg_hashes(cfg.engines[engine])
    return env


def compare_env(local: dict, remote: dict) -> list[str]:
    """Which shared inputs the two boxes disagree on.

    run_configs is compared as a set because the two boxes build it from their
    own engine lists; its entries carry the repeat count (``suite/variant@N``),
    so a series one box measures five times and the other three shows up
    here as an entry only each side has. suite_cfg_hash per suite, naming
    the suite, because one suite's protocol changing is one series' problem;
    the rest are compared as read.
    """
    out = []
    for key in ("slipstream_version", "os_version", "harness", "runner_cfg_hash"):
        if key in remote and remote[key] != local.get(key):
            out.append(f"{key}: here {local.get(key)!r}, there {remote[key]!r}")
    if "suite_cfg_hash" in remote:
        mine = local.get("suite_cfg_hash") or {}
        theirs = remote["suite_cfg_hash"] or {}
        for suite in sorted(set(mine) & set(theirs)):
            if mine[suite] != theirs[suite]:
                out.append(
                    f"suite_cfg_hash[{suite}]: here {mine[suite]!r}, "
                    f"there {theirs[suite]!r}"
                )
    if "run_configs" in remote and "run_configs" in local:
        missing = set(remote["run_configs"]) - set(local["run_configs"])
        extra = set(local["run_configs"]) - set(remote["run_configs"])
        if missing or extra:
            out.append(
                f"run_configs: only here {sorted(extra)}, only there {sorted(missing)}"
            )
    return out


def collect_status(
    cfg: Config,
    collector,
    engines: list[str] | None = None,
    config_label: str = "~/.config/slipstream/config.toml",
) -> BusStatus:
    store: CommitStore | None = collector.store if collector is not None else None
    bus = Bus(cfg.bus.root) if cfg.bus else None
    names = engines or list(cfg.engines)
    report = BusStatus()

    root = bus.root if bus else cfg.out_dir
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    report.free_gb = shutil.disk_usage(probe).free / GB
    report.window = window_status(cfg, config_label)
    report.paused_until = report.window.paused
    holder = report.window.holder
    report.lock_holder = str(holder) if holder else None
    report.lock_waiters = report.window.waiters

    for name in names:
        st = EngineStatus(engine=name)
        if bus and bus.protocol:
            from .bus_protocol import read

            progress = bus.protocol.state_path("transfer", name)
            peer = bus.protocol.state_path("peer-health", name)
            try:
                if progress.exists():
                    st.transfer = read(progress)
                    route = next((r for r in cfg.bus.routes if name in r.engines), None)
                    if route:
                        st.transfer["destination_label"] = (
                            f"{route.destination.ssh_host or 'local'}:{route.destination.root}"
                        )
                        if route.source.is_local:
                            st.transfer["backlog"] = len(
                                bus.keys_above(name, st.transfer["cursor"])
                            )
                if peer.exists():
                    st.peer_age = time.time() - read(peer)["fetched_at"]
            except (OSError, BusError) as exc:
                st.error = str(exc)
        if store is not None:
            st.status_counts = store.status_counts(name, cfg.platform)
            st.orphan_scores = len(
                store.commit_ids_missing_commit_row(name, cfg.platform)
            )
            st.provenance = store.run_env_source_counts(name)
        source: BusSource | None = cfg.bus.source_for(name) if cfg.bus else None
        if source is None:
            report.engines.append(st)
            continue
        st.driven_by = "bus"
        st.source = source.name
        try:
            _fill_bus_status(
                cfg,
                bus,
                source,
                name,
                st,
                local_env(cfg, collector, name) if collector is not None else None,
            )
        except TRANSPORT_ERRORS as e:
            # An unreachable source is the case this command exists for, so it
            # reports rather than raising. CalledProcessError, which is how
            # SshSource signals every transport failure, is not a RuntimeError.
            st.error = str(e)
        report.engines.append(st)
    return report


def _fill_bus_status(cfg, bus, source, name, st, mine):
    handle = open_source(source)
    cursor_file = cursor_path(cfg.out_dir, source.name, name)
    try:
        st.cursor = read_cursor(cursor_file)
    except BusError as e:
        st.error = str(e)
    builder = handle.builder_state(name)
    st.frontier = builder.frontier
    st.lowest_retained = builder.lowest_retained
    st.failed_builds = list(builder.failed)
    st.failed_builds_total = builder.failed_total or len(builder.failed)
    st.stalled_since = builder.stalled_since
    errors = [builder.last_error] if builder.last_error else []
    if builder.safari_acquisition_error:
        errors.append(f"Safari acquisition: {builder.safari_acquisition_error}")
    st.builder_error = "; ".join(errors) or None
    st.builder_age = time.time() - builder.updated_at if builder.updated_at else None
    st.paused_by_floor = builder.publishing_paused_by_floor
    st.builder_in_flight = builder.in_flight
    st.lag = len(handle.keys_above(name, st.cursor))

    mine_state = bus.read_bench_state(name)
    st.bench_paused_by_floor = mine_state.benching_paused_by_floor
    st.bench_error = mine_state.last_error
    st.bench_in_flight = mine_state.in_flight
    st.bench_stalled_since = mine_state.stalled_since
    st.bench_skipped_dropped = mine_state.skipped_dropped
    st.bench_skipped_upstream = mine_state.skipped_upstream
    st.bench_skipped_permanent = mine_state.skipped_permanent
    st.permanent_skip_reason = mine_state.permanent_skip_reason
    st.bench_age = (
        time.time() - mine_state.updated_at if mine_state.updated_at else None
    )
    if builder.highest_dropped is not None and (
        st.cursor is None or builder.highest_dropped > st.cursor
    ):
        # Retention deliberately does not coordinate with consumers, so this is
        # the only signal. Derived from what the builder actually deleted, not
        # from lowest_retained against the cursor: commit ids are not
        # contiguous, so a gap between those two is not evidence of a drop.
        st.dropped_unread_up_to = builder.highest_dropped

    # The source's own bencher block, so a divergence is reported against the
    # box the artifacts came from rather than against nothing.
    theirs = handle.bench_state(name).env
    if theirs and mine is not None:
        st.env_divergence = compare_env(mine, theirs)
        st.env_since = theirs.get("since")

    st.blob_gb = bus.footprint(name) / GB
    st.roots_gb = _dir_bytes(bus.root / "roots" / name) / GB


def _key_of(record: dict) -> str:
    """The key a state-file record names, in its CLI spelling.

    Records written before keys had two parts carry only commit_id, and one
    whose commit_id is missing is not a key at all.
    """
    if record.get("commit_id") is None:
        return "?"
    return str(CommitKey.from_commit(record))


def render(report: BusStatus, echo) -> None:
    def when(ts):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

    if report.window is not None:
        render_window(report.window, echo)
    echo(f"free space: {report.free_gb:.0f}GB")
    if report.paused_until:
        echo(f"PAUSED by operator until {when(report.paused_until)}")
    if report.lock_holder:
        echo(f"machine lock: held by {report.lock_holder}")
    else:
        echo("machine lock: free")
    for i, waiter in enumerate(report.lock_waiters, 1):
        echo(f"  {i}. {waiter}")

    for st in report.engines:
        echo("")
        echo(f"{st.engine} ({st.driven_by}{f' from {st.source}' if st.source else ''})")
        if st.error:
            echo(f"  error: {st.error}")
        if st.bench_skipped_upstream:
            echo(
                f"  source retention prevented transfer through {st.bench_skipped_upstream}"
            )
        if st.bench_skipped_permanent is not None:
            echo(
                f"  last permanent publication skip: {st.bench_skipped_permanent}: "
                f"{st.permanent_skip_reason}"
            )
        if st.transfer:
            progress = st.transfer
            acknowledged = (progress.get("acknowledged") or {}).get("key", "none")
            echo(
                f"  transfer to {progress.get('destination_label', progress['destination'])}: acknowledged {acknowledged}, backlog {progress.get('backlog', 'unknown')}"
            )
            echo(
                f"  transfer skipped through {progress.get('highest_skipped') or 'none'}, pending {progress.get('pending') or 'none'}"
            )
            if progress.get("last_error"):
                echo(f"  transfer error: {progress['last_error']}")
        if st.peer_age is not None:
            echo(
                f"  peer snapshot fetched {int(st.peer_age)}s ago (source timestamps preserved)"
            )
        if st.driven_by == "bus":
            echo(f"  frontier {st.frontier}, cursor {st.cursor}, lag {st.lag}")
            if st.builder_age is not None:
                echo(f"  builder state written {int(st.builder_age / 60)}m ago")
            else:
                echo("  builder state: never written")
            if st.dropped_unread_up_to is not None:
                echo(
                    f"  entries up to {st.dropped_unread_up_to} were dropped by "
                    f"retention before this machine read them (cursor "
                    f"{st.cursor if st.cursor is not None else 'unset'})"
                )
            if st.stalled_since:
                echo(f"  builder STALLED since {when(st.stalled_since)}")
            if st.paused_by_floor:
                echo("  builder paused: below its free-space floor")
            if st.builder_in_flight:
                roll = st.builder_in_flight.get("embedder_hash", "")
                echo(
                    f"  builder is on {_key_of(st.builder_in_flight)} "
                    f"({st.builder_in_flight.get('phase')}"
                    f"{f', under {roll[:12]}' if roll else ''})"
                )
            if st.builder_error:
                echo(f"  builder last error: {st.builder_error}")
            if st.bench_stalled_since:
                echo(
                    f"  this machine's bencher STALLED since "
                    f"{when(st.bench_stalled_since)}"
                )
            if st.bench_skipped_dropped is not None:
                echo(
                    f"  entries up to {st.bench_skipped_dropped} were skipped: "
                    f"retention dropped them before this machine benched them"
                )
            if st.bench_paused_by_floor:
                echo("  this machine's bencher paused: below its free-space floor")
            if st.bench_age is not None:
                # Unconditional, like the builder's: a killed bencher keeps
                # in_flight and a stale updated_at, and the age is the only
                # thing that tells it apart from a three-hour bench.
                echo(f"  bench state written {int(st.bench_age / 60)}m ago")
            if st.bench_in_flight:
                echo(
                    f"  last reported benching {_key_of(st.bench_in_flight)} since "
                    f"{when(st.bench_in_flight.get('started_at', 0))} (snapshot; machine lock shows current activity)"
                )
            if st.bench_error:
                echo(f"  bencher last error: {st.bench_error}")
            if st.failed_builds:
                shown = ", ".join(
                    f"{_key_of(f)} ({f.get('kind', '?')})"
                    for f in st.failed_builds[-5:]
                )
                echo(f"  {st.failed_builds_total} failed builds, latest: {shown}")
            echo(f"  disk: {st.blob_gb:.1f}GB payloads, {st.roots_gb:.1f}GB run roots")
            if st.env_divergence:
                echo(
                    f"  environment differs from the source"
                    f"{f' (since {when(st.env_since)})' if st.env_since else ''}:"
                )
                for line in st.env_divergence:
                    echo(f"    {line}")
        counts = ", ".join(f"{k} {v}" for k, v in sorted(st.status_counts.items()))
        echo(f"  benched: {counts or 'nothing yet'}")
        if st.provenance:
            split = ", ".join(f"{k} {v}" for k, v in sorted(st.provenance.items()))
            echo(f"  recent binaries: {split}")
        if st.orphan_scores:
            echo(
                f"  {st.orphan_scores} commits have scores but no commit row; "
                f"their scores cannot be exported"
            )
