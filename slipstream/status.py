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

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from datetime import UTC, datetime

from . import host
from . import bench_window
from .bus import Bus, BusError, cursor_path, read_cursor
from .config import BusSource, Config
from .consumer import LocalSource, TRANSPORT_ERRORS, open_source
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
    archive: dict = field(default_factory=dict)
    # Gated retention: what keeps the oldest live entry, and what is live.
    retention: dict = field(default_factory=dict)
    peer_age: float | None = None


@dataclass
class BusStatus:
    engines: list[EngineStatus] = field(default_factory=list)
    free_gb: float = 0.0
    lock_holder: str | None = None
    lock_waiters: list[str] = field(default_factory=list)
    paused_until: float | None = None
    window: WindowStatus | None = None
    # Gated retention: footprint against the cap, and builders paused at it.
    cap: dict = field(default_factory=dict)


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
    if isinstance(override, bench_window.Override) and (
        override.expires_at is None or override.expires_at > datetime.now(UTC)
    ):
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
    for i, waiter in enumerate(report.waiters, 1):
        echo(f"  queued {i}: {waiter}")


def _dir_bytes(path: Path) -> int:
    total = 0
    stack = [os.fspath(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def compare_env(local: dict, remote: dict) -> list[str]:
    """Where this machine differs from the source's builder.

    hw_model is published but not compared: the two boxes are different
    machines by design.
    """
    out = []
    for key in ("slipstream_version", "os_version", "toolchain"):
        if key in remote and remote[key] != local.get(key):
            out.append(f"{key}: here {local.get(key)!r}, there {remote[key]!r}")
    return out


def collect_status(
    cfg: Config,
    collector,
    engines: list[str] | None = None,
    config_label: str = "~/.config/slipstream/config.toml",
) -> BusStatus:
    store: CommitStore | None = (
        getattr(collector, "store", collector) if collector is not None else None
    )
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
    if bus and bus.protocol and cfg.bus.release_after:
        report.cap = _cap_status(cfg, bus)

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
                                bus.keys_above(
                                    name, st.transfer["cursor"], readonly=True
                                )
                            )
                if peer.exists():
                    st.peer_age = time.time() - read(peer)["fetched_at"]
            except (OSError, BusError) as exc:
                st.error = str(exc)
            archive = cfg.bus.archive
            if archive is not None and name in archive.engines:
                st.archive = _archive_status(bus, archive, name)
            if cfg.bus.release_after:
                st.retention = _retention_status(cfg, bus, name)
        if store is not None:
            st.status_counts = store.status_counts(name, cfg.platform)
            st.orphan_scores = len(store.keys_missing_commit_row(name, cfg.platform))
            st.provenance = store.run_env_source_counts(name)
        source: BusSource | None = cfg.bus.source_for(name) if cfg.bus else None
        if source is None:
            report.engines.append(st)
            continue
        st.driven_by = "bus"
        st.source = source.name
        try:
            _fill_bus_status(cfg, bus, source, name, st)
        except TRANSPORT_ERRORS as e:
            # An unreachable source is the case this command exists for, so it
            # reports rather than raising. CalledProcessError, which is how
            # SshSource signals every transport failure, is not a RuntimeError.
            st.error = str(e)
        report.engines.append(st)
    return report


def _archive_status(bus, archive, name) -> dict:
    """The archive's record for one engine, with its backlog and age."""
    from .archive import ArchiveError, read_state

    try:
        state = read_state(bus.protocol, name)
    except ArchiveError as exc:
        return {"error": str(exc)}
    if state is None:
        return {"never": True, "every_hours": archive.every_hours}
    report = dict(state, every_hours=archive.every_hours)
    try:
        report["backlog"] = len(bus.keys_above(name, state["cursor"], readonly=True))
    except (OSError, BusError, ValueError) as exc:
        report["error"] = str(exc)
    if state.get("last_success_at"):
        report["age"] = time.time() - state["last_success_at"]
    return report


def _retention_status(cfg, bus, name) -> dict:
    from .retention import holding_gate

    try:
        keys = bus.keys_above(name, None, readonly=True)
        report = {
            "gates": cfg.release_gates(name),
            "live_entries": len(keys),
            "live_bytes": bus.footprint(name, readonly=True),
        }
        held = holding_gate(cfg, bus.root, name, keys[0] if keys else None)
    except (OSError, BusError, ValueError) as exc:
        return {"error": str(exc)}
    if held is not None:
        report["holding"] = {
            "gate": held.gate,
            "cursor": str(held.cursor) if held.cursor else None,
            "error": held.error,
            "age": time.time() - held.updated_at if held.updated_at else None,
        }
    return report


def _cap_status(cfg, bus) -> dict:
    try:
        paused = [
            name
            for name in bus.engines()
            if bus.read_builder_state(name).publishing_paused_by_cap
        ]
        return {
            "used": bus.used_bytes(readonly=True),
            "cap": cfg.build.retain_gb * GB,
            "builders_paused": paused,
        }
    except (OSError, BusError) as exc:
        return {"error": str(exc)}


def _fill_bus_status(cfg, bus, source, name, st):
    handle = open_source(source)
    if isinstance(handle, LocalSource):
        handle.readonly = True
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

    if builder.env:
        st.env_divergence = compare_env(host.builder_env(), builder.env)
        st.env_since = builder.env.get("since")

    st.blob_gb = bus.footprint(name, readonly=True) / GB
    st.roots_gb = _dir_bytes(bus.root / "roots" / name) / GB


def _key_of(record: dict) -> str:
    """The key a state-file record names, in its CLI spelling.

    Records written before keys had two parts carry only commit_id, and one
    whose commit_id is missing is not a key at all.
    """
    if record.get("commit_id") is None:
        return "?"
    return str(CommitKey.from_commit(record))


def _render_retention(retention: dict, echo) -> None:
    if retention.get("error"):
        echo(f"  retention: unreadable: {retention['error']}")
        return
    gates = ", ".join(retention["gates"]) or "none (released at publish)"
    echo(
        f"  retention after {gates}: {retention['live_entries']} live entries, "
        f"{retention['live_bytes'] / GB:.1f}GB"
    )
    held = retention.get("holding")
    if held:
        age = f", written {held['age'] / 3600:.1f}h ago" if held.get("age") else ""
        why = f" ({held['error']})" if held.get("error") else ""
        echo(
            f"  oldest live entry held by {held['gate']}: cursor "
            f"{held['cursor'] or 'none'}{age}{why}"
        )


def _render_archive(archive: dict, echo) -> None:
    if archive.get("never"):
        echo("  archive: never run")
    elif "cursor" in archive:
        age = archive.get("age")
        late = age is not None and age > archive["every_hours"] * 3600 * 1.5
        last = (
            f"last success {age / 3600:.1f}h ago"
            if age is not None
            else "no successful run"
        )
        echo(
            f"  archive: cursor {archive['cursor'] or 'none'}, backlog "
            f"{archive.get('backlog', 'unknown')}, {last} (every "
            f"{archive['every_hours']:g}h){', OVERDUE' if late else ''}"
        )
        if archive.get("skipped_dropped"):
            echo(
                f"  archive skipped through {archive['skipped_dropped']}: "
                f"retention dropped them first"
            )
        if archive.get("last_error"):
            echo(f"  archive error: {archive['last_error']}")
    if archive.get("error"):
        echo(f"  archive error: {archive['error']}")


def render(report: BusStatus, echo) -> None:
    def when(ts):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))

    if report.window is not None:
        render_window(report.window, echo)
    echo(f"free space: {report.free_gb:.0f}GB")
    if report.cap:
        cap = report.cap
        if cap.get("error"):
            echo(f"artifact cap: unreadable: {cap['error']}")
        else:
            at_cap = cap["used"] >= cap["cap"]
            echo(
                f"artifacts: {cap['used'] / GB:.1f}GB of a {cap['cap'] / GB:g}GB cap"
                + (" (AT CAP: accept refuses, the builder pauses)" if at_cap else "")
            )
            for name in cap["builders_paused"]:
                echo(f"  {name}: builder paused at the cap")

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
        if st.archive:
            _render_archive(st.archive, echo)
        if st.retention:
            _render_retention(st.retention, echo)
        if st.peer_age is not None:
            echo(f"  peer snapshot fetched {int(st.peer_age)}s ago")
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
                    f"  environment differs from the source's builder"
                    f"{f' (since {when(st.env_since)})' if st.env_since else ''}:"
                )
                for line in st.env_divergence:
                    echo(f"    {line}")
        counts = ", ".join(f"{k} {v}" for k, v in sorted(st.status_counts.items()))
        echo(f"  benched: {counts or 'nothing yet'}")
        if len(st.provenance) > 1:
            split = ", ".join(f"{k} {v}" for k, v in sorted(st.provenance.items()))
            echo(f"  recent binaries: {split}")
        if st.orphan_scores:
            echo(
                f"  {st.orphan_scores} commits have scores but no commit row; "
                f"their scores cannot be exported"
            )
