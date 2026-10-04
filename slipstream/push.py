# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Push slipstream scores to the configured push targets.

Each target receives the export CSV: a spool directory, or the perf Spanner
database. Pushing is incremental: only commits present in
``processing_state`` but not ``push_state`` are exported, and they are marked
pushed only after every target has accepted them. A failed target leaves
``push_state`` untouched, so the same commits are retried on the next push
(at-least-once; the receiver's INSERT OR UPDATE makes that idempotent).
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .config import PushConfig
from .store import CommitStore

from .delivery_batch import COLUMNS, to_csv

_COLUMNS = list(COLUMNS)
_ENGINE_IDX = _COLUMNS.index("engine")
_FLAGS_IDX = _COLUMNS.index("flags")


def _export_rows(
    store: CommitStore,
    engine: str,
    platform: str,
    valid_benchmarks_by_suite: dict[str, set[str]],
    keys: list | None = None,
) -> list[list]:
    rows = []
    for row in store.export_scores(engine, platform, valid_benchmarks_by_suite, keys):
        r = list(row)
        # Prefix flags with the engine so variants of different engines stay
        # distinct downstream, e.g. "default" -> "v8_default".
        r[_FLAGS_IDX] = f"{r[_ENGINE_IDX]}_{r[_FLAGS_IDX]}"
        rows.append(r)
    return rows


def _to_csv(rows: list[list]) -> str:
    return to_csv(rows)


def build_export_csv(
    store: CommitStore,
    engine: str,
    platform: str,
    valid_benchmarks_by_suite: dict[str, set[str]],
    keys: list | None = None,
) -> tuple[str, int]:
    """Export score rows for valid benchmarks, joined with commit metadata.

    Returns the CSV text and its row count; ("", 0) when nothing matches.
    """
    rows = _export_rows(store, engine, platform, valid_benchmarks_by_suite, keys)
    if not rows:
        return "", 0
    return _to_csv(rows), len(rows)


_SEQ_RE = re.compile(r"^(\d+)\.csv$")


def seq_name(seq: int) -> str:
    """File name of one entry in the spool log, zero padded so names sort."""
    return f"{seq:08d}.csv"


def parse_seq(name: str) -> int | None:
    """Sequence number of a spool file name, or None for anything else
    (a .tmp still being written, the lock file)."""
    m = _SEQ_RE.match(name)
    return int(m.group(1)) if m else None


def _init_fresh_spool(spool_dir: Path) -> None:
    """Give a spool nobody has written to its allocator, at zero.

    Done before the writer lock exists, because the lock file is created by
    whoever opens it first: a writer that merely lost the first race to a
    sibling would otherwise find a lock and no allocator -- exactly what an
    empty legacy spool looks like -- and refuse. ``link`` publishes the
    allocator only if none exists, so a late sibling cannot reset one that
    the winner has already advanced.
    """
    from .durability import fsync_dir

    allocation = spool_dir / ".sequence"
    if (spool_dir / ".lock").exists() or allocation.exists():
        return
    if any(parse_seq(p.name) for p in spool_dir.iterdir()):
        return  # a legacy log; the writer adopts its highest number
    fd, name = tempfile.mkstemp(prefix=".sequence-", dir=spool_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("0\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, allocation)
        except FileExistsError:
            return
        fsync_dir(spool_dir)
    finally:
        Path(name).unlink(missing_ok=True)


def spool_append(
    spool_dir: Path,
    csv_data: str,
    retain_days: int,
    *,
    timeout=30.0,
    should_stop=lambda: False,
    log=lambda msg: None,
    bot: str | None = None,
) -> int:
    """Append the CSV to the spool log; returns its sequence number.

    The log is contiguous from 1 and the consumer tracks it with a cursor, so
    allocation must never reuse a number. Delivery and manual pushes append
    under one writer lock. A crash after reservation can leave a gap; consumers
    stop visibly instead of silently reusing that identity.
    """
    from .durability import FileLock, atomic_write, durable_mkdir

    durable_mkdir(spool_dir)
    _init_fresh_spool(spool_dir)
    with FileLock(
        spool_dir / ".lock", timeout=timeout, should_stop=should_stop, log=log
    ):
        if bot is not None:
            binding = spool_dir / ".bot"
            try:
                existing = binding.read_text()
                if existing != bot:
                    raise ValueError(f"spool is bound to bot {existing!r}, not {bot!r}")
            except FileNotFoundError:
                atomic_write(binding, bot)
        seqs = [s for s in (parse_seq(p.name) for p in spool_dir.iterdir()) if s]
        allocation = spool_dir / ".sequence"
        try:
            last = int(allocation.read_text().strip())
            if last < max(seqs, default=0) or last < 0:
                raise ValueError("spool allocator is behind published entries")
        except FileNotFoundError:
            # A fresh spool got its allocator above; one without it was
            # written by a version that had none.
            if not seqs:
                raise ValueError(
                    "empty legacy spool has no allocator; recover its last sequence explicitly"
                )
            last = max(seqs)  # adopt legacy log without resetting it
        seq = last + 1
        # Reserve durably before publication: a crash can leave a visible gap,
        # but never reuses a number after retention or an interrupted append.
        atomic_write(allocation, f"{seq}\n")
        atomic_write(spool_dir / seq_name(seq), csv_data)
        _prune(spool_dir, retain_days)
    return seq


def _prune(spool_dir: Path, retain_days: int) -> None:
    """Drop log entries older than ``retain_days``.

    The producer does not know what the consumer has read, so this can prune
    past an undelivered entry; that surfaces there as a gap, never as silent
    loss.
    """
    from .durability import fsync_dir

    cutoff = time.time() - retain_days * 86400
    for p in spool_dir.iterdir():
        if parse_seq(p.name) and p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)
    fsync_dir(spool_dir)


def probe(
    push_cfg: PushConfig, log: Callable[[str], None], *, state_dir=None, settings=None
) -> None:
    """Header-only connectivity check through the shared coordinator."""
    from .delivery import Coordinator
    from .durability import target_lock_path

    c = Coordinator(
        [],
        push_cfg.targets,
        state_dir or target_lock_path("probe").parent / "probe",
        settings=settings,
        log=log,
    )
    with c.ownership():
        c.cycle(probe_bot=push_cfg.bot_name)
        if c.errors:
            raise c.errors[0]


def push(
    store,
    engines,
    push_cfg,
    valid_benchmarks_by_suite,
    platform,
    *,
    rebuild=False,
    log=None,
    settings=None,
    state_dir=None,
    should_stop=lambda: False,
):
    """One bounded local cycle, with the same recovery semantics as deliver."""
    from .delivery import Coordinator
    from .delivery_sources import LocalDbSource
    from .delivery_maintenance import rebuild_local

    sources = [
        LocalDbSource(store, e, platform, push_cfg.bot_name, valid_benchmarks_by_suite)
        for e in engines
    ]
    coordinator = Coordinator(
        sources,
        push_cfg.targets,
        state_dir or store.db_path.parent / "delivery",
        settings=settings,
        should_stop=should_stop,
        log=log or (lambda msg: None),
    )
    with coordinator.ownership():
        if rebuild:
            rebuild_local(coordinator, store, engines, platform, push_cfg.bot_name)
        else:
            from .delivery_maintenance import check_maintenance

            check_maintenance(coordinator)
        count = coordinator.cycle()
        if coordinator.errors:
            raise coordinator.errors[0]
        return count
