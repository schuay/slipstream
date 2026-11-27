# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Gated retention: free a build when it is dead, not when a budget says so.

With ``[bus] release_after`` declared, an entry is dead once every gate that
applies to its engine on this bus has a cursor at or above its key (D018):

- ``bench``: this machine's consumer cursor for the engine's source;
- ``transfer``: the outbound transfer cursor, ``state/transfer/<engine>.json``;
- ``archive``: the archive cursor, ``state/archive/<engine>.json``.

Every gate reads a cursor local to the bus it releases. Each advances in key
order, so the dead set is a prefix of each topic. A cursor that cannot be
read counts as no progress. ``[build] retain_gb`` then stops being an
eviction budget and becomes an admission cap: the builder pauses and accept
refuses, and nothing live is evicted.

Without ``release_after`` everything here reduces to the byte budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .bus import RELEASE_ALL, Bus, BusError, cursor_path, read_cursor
from .models import CommitKey

GB = 1_000_000_000


class OverCap(OSError):
    """Installing would take the bus past its cap. Retryable: the cap frees
    as soon as a gate advances."""


@dataclass
class GateCursor:
    gate: str
    cursor: CommitKey | None
    # Why there is no cursor, when there is none; None means never written.
    error: str | None = None
    # When the cursor file was last written, if it exists.
    updated_at: float | None = None


def gate_cursor(cfg, root, gate: str, engine: str) -> GateCursor:
    """Read one gate's cursor for ``engine`` on the bus at ``root``."""
    from .bus_protocol import read

    if gate == "bench":
        path = cursor_path(cfg.out_dir, cfg.bus.source_for(engine).name, engine)
    else:
        kind = "transfer" if gate == "transfer" else "archive"
        path = root / "state" / kind / f"{engine}.json"
    try:
        updated = os.stat(path).st_mtime
    except FileNotFoundError:
        return GateCursor(gate, None)
    except OSError as exc:
        return GateCursor(gate, None, str(exc))
    try:
        if gate == "bench":
            cursor = read_cursor(path)
        else:
            value = read(path).get("cursor")
            cursor = CommitKey.parse(value) if value else None
    except (BusError, OSError, TypeError, ValueError) as exc:
        return GateCursor(gate, None, str(exc), updated)
    return GateCursor(gate, cursor, None, updated)


def gated(cfg) -> bool:
    bus = getattr(cfg, "bus", None)
    return bool(bus is not None and getattr(bus, "release_after", None))


def release_points(cfg, bus: Bus) -> dict:
    """Engine -> the highest key every applicable gate has passed.

    ``RELEASE_ALL`` for an engine no gate applies to, None for one whose
    gates have not all reached anything (including an unreadable cursor).
    """
    points = {}
    for engine in bus.engines():
        gates = cfg.release_gates(engine) if engine in cfg.engines else []
        if not gates:
            point = RELEASE_ALL
        else:
            cursors = [gate_cursor(cfg, bus.root, g, engine).cursor for g in gates]
            point = None if None in cursors else min(cursors)
        for limit in _derived_limits(cfg, bus, engine):
            point = _lower(point, limit)
        points[engine] = point
    return points


def _lower(point, limit):
    if point is None or limit is None:
        return None
    if point is RELEASE_ALL:
        return limit
    return min(point, limit)


def _derived_limits(cfg, bus: Bus, engine: str):
    """An engine derived from ``engine`` and built on this bus packages its
    entries from this bus's ``engine`` entries (Safari from jsc). An inner
    entry stays live until that builder's frontier is past it, or the
    derived series gets a hole no gate would have shown."""
    protocol = bus.protocol
    ready = protocol.ready() if protocol else None
    for name, other in cfg.engines.items():
        if other.derives != engine or name not in cfg.build.engines:
            continue
        if ready is not None and ready["owners"].get(name) != ready["uuid"]:
            continue
        try:
            frontier = bus.read_builder_state(name).frontier
        except BusError:
            frontier = None
        # Derived keys are (release, inner commit); inner keys are (0, commit).
        # The entry at the frontier stays too: a newly adopted release is
        # first published on that same inner commit.
        yield CommitKey(0, frontier.commit_id - 1) if frontier else None


def holding_gate(cfg, root, engine: str, oldest: CommitKey | None) -> GateCursor | None:
    """The applicable gate with the lowest cursor below ``oldest``, the
    engine's oldest live entry: what keeps it alive. None when no gate does
    (the newest-entry protection, or nothing live)."""
    if oldest is None or engine not in cfg.engines:
        return None
    readings = [gate_cursor(cfg, root, g, engine) for g in cfg.release_gates(engine)]
    held = [r for r in readings if r.cursor is None or r.cursor < oldest]
    if not held:
        return None
    # A gate with no cursor at all holds hardest.
    return min(held, key=lambda r: (r.cursor is not None, r.cursor or CommitKey(0, 0)))


class Policy:
    """How a bus at this machine frees artifacts: by gates, or by budget.

    ``cfg`` is this machine's config, or None for an endpoint started by a
    peer that predates local policy, which keeps the budget it was handed.
    """

    def __init__(self, cfg=None, retain_bytes: float | None = None):
        self.cfg = cfg
        self._retain_bytes = retain_bytes

    @property
    def retain_bytes(self) -> float:
        if self._retain_bytes is not None:
            return self._retain_bytes
        return self.cfg.build.retain_gb * GB

    @property
    def gated(self) -> bool:
        return self.cfg is not None and gated(self.cfg)

    @property
    def cap_bytes(self) -> float | None:
        """The admission cap, only when gates release artifacts."""
        return self.retain_bytes if self.gated else None

    def prune(self, bus, keep=None) -> dict:
        """Drop what this policy frees from ``bus`` (a Bus or BusProtocol).

        A migrated bus prunes through its protocol: under the bus lock and
        the pending-batch journal, never by unlinking manifests directly.
        """
        bus = _bus(bus)
        if not self.gated:
            return bus.prune(self.retain_bytes, keep)
        return bus.prune(None, keep, release=release_points(self.cfg, bus))

    def release(self, bus: Bus, log=None, *, label: str = "") -> dict:
        """Prune and sweep a migrated bus outside the machine lock, reporting
        rather than raising: freeing is never worth failing the caller for."""
        bus = _bus(bus)
        try:
            dropped = self.prune(bus)
            bus.sweep_blobs()
        except (OSError, BusError) as exc:
            if log is not None:
                from .service_logging import warning

                log(warning(f"{label}retention deferred: {exc}"))
            return {}
        if log is not None:
            for engine, keys in dropped.items():
                log(f"{engine}: released {len(keys)} entries through {keys[-1]}")
        return dropped

    def over_cap(self, bus: Bus) -> tuple[bool, int]:
        """Whether the bus is at or above its cap, and its footprint."""
        if self.cap_bytes is None:
            return False, 0
        used = _bus(bus).used_bytes()
        return used >= self.cap_bytes, used


def _bus(target) -> Bus:
    from .bus_protocol import BusProtocol

    if isinstance(target, BusProtocol):
        return Bus(target.root)
    if target._raw:
        raise ValueError("retention needs the bus, not a raw view of it")
    return target


CAP_REFUSAL = "destination over cap"


def cap_message(cap_bytes: float) -> str:
    # Stable text: the coordinator logs a retry only when the error changes.
    return f"{CAP_REFUSAL} ({cap_bytes / GB:.0f}GB of artifacts)"
