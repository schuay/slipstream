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

import contextlib
import csv
import fcntl
import io
import re
import time
from collections.abc import Callable
from pathlib import Path

from .config import PushConfig, PushTarget
from .store import CommitStore

_COLUMNS = [
    "engine",
    "platform",
    "commit_id",
    "suite",
    "flags",
    "benchmark",
    "metric",
    "run",
    "score",
    "timestamp",
    "git_hash",
    "commit_date",
    "commit_timestamp",
    "commit_title",
]
_ENGINE_IDX = _COLUMNS.index("engine")
_FLAGS_IDX = _COLUMNS.index("flags")


def _export_rows(
    store: CommitStore,
    engine: str,
    platform: str,
    valid_benchmarks_by_suite: dict[str, set[str]],
    commit_ids: list[int] | None = None,
) -> list[list]:
    rows = []
    for row in store.export_scores(
        engine, platform, valid_benchmarks_by_suite, commit_ids
    ):
        r = list(row)
        # Prefix flags with the engine so variants of different engines stay
        # distinct downstream, e.g. "default" -> "v8_default".
        r[_FLAGS_IDX] = f"{r[_ENGINE_IDX]}_{r[_FLAGS_IDX]}"
        rows.append(r)
    return rows


def _to_csv(rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_COLUMNS)
    w.writerows(rows)
    return buf.getvalue()


def build_export_csv(
    store: CommitStore,
    engine: str,
    platform: str,
    valid_benchmarks_by_suite: dict[str, set[str]],
    commit_ids: list[int] | None = None,
) -> tuple[str, int]:
    """Export score rows for valid benchmarks, joined with commit metadata.

    Returns the CSV text and its row count; ("", 0) when nothing matches.
    """
    rows = _export_rows(store, engine, platform, valid_benchmarks_by_suite, commit_ids)
    if not rows:
        return "", 0
    return _to_csv(rows), len(rows)


class TargetSession:
    """One target held open for the span of a push or relay cycle.

    ``deliver`` takes one export CSV and may be called any number of times;
    ``close`` finishes the cycle and returns a summary line if the target has
    one. Both raise the underlying error on failure: OSError for spool writes,
    the Spanner client's errors for spanner targets.
    """

    def deliver(self, csv_text: str) -> None:
        raise NotImplementedError

    def close(self) -> str | None:
        return None


class _SpoolSession(TargetSession):
    def __init__(self, target: PushTarget):
        self._dir = target.spool_dir
        self._retain_days = target.retain_days

    def deliver(self, csv_text: str) -> None:
        spool_append(self._dir, csv_text, self._retain_days)


class _SpannerSession(TargetSession):
    """Stages every delivered CSV over one connection and aggregates once.

    A relay cycle can hand over many files; the connection, the schema check,
    and the aggregation are per cycle, not per file.
    """

    def __init__(self, target: PushTarget, bot_name: str, *, rebuild: bool):
        from . import spanner

        self._spanner = spanner
        self._bot = bot_name
        self._refresh = target.refresh
        self._pending_wipe = rebuild
        self._wiped = False
        self._staged = 0
        self._failed = False
        try:
            self._db = spanner.connect(target.spanner)
            spanner.ensure_schema(self._db)
        except Exception:
            if hasattr(self, "_db"):
                self._db.close()
            raise

    def deliver(self, csv_text: str) -> None:
        try:
            # Mapping first: a CSV that fails to parse must not cost the bot
            # its staged rows on a rebuild.
            rows = self._spanner.rows_from_csv(
                csv_text, self._bot, self._spanner.current_timestamp(self._db)
            )
            if self._pending_wipe:
                self._spanner.wipe_bot(self._db, self._bot)
                self._pending_wipe = False
                self._wiped = True
            marker = self._spanner.begin_import(self._db, self._bot)
            self._staged += self._spanner.stage_rows(self._db, rows)
            self._spanner.finish_import(self._db, marker)
        except Exception:
            self._failed = True
            raise

    def close(self) -> str | None:
        try:
            if self._failed:
                # Staging commits in chunks, so a failure part way through a
                # CSV leaves the group short of runs. Aggregating that would
                # publish a mean over an incomplete run set; the retry
                # re-stages the whole file and the watermark picks it up then.
                return None
            skipped = self._spanner.refresh(self._db) if self._refresh else None
            return self._spanner.summary_line(
                self._bot,
                self._staged,
                rebuild=self._wiped,
                skipped=skipped,
                refreshed=self._refresh,
            )
        finally:
            self._db.close()


def open_target(
    target: PushTarget, bot_name: str, *, rebuild: bool = False
) -> TargetSession:
    """Open a session on one target. ``rebuild`` makes a spanner target wipe
    this bot's staged rows before the first delivery; other targets ignore it.
    """
    if target.spool_dir is not None:
        return _SpoolSession(target)
    return _SpannerSession(target, bot_name, rebuild=rebuild)


def deliver_once(
    target: PushTarget, bot_name: str, csv_data: str, *, rebuild: bool = False
) -> str | None:
    """Open a session, deliver one CSV, close it. Raises what the target raises."""
    session = open_target(target, bot_name, rebuild=rebuild)
    try:
        session.deliver(csv_data)
    except BaseException:
        # Still release the connection, but let the delivery error escape.
        with contextlib.suppress(Exception):
            session.close()
        raise
    return session.close()


_SEQ_RE = re.compile(r"^(\d+)\.csv$")


def seq_name(seq: int) -> str:
    """File name of one entry in the spool log, zero padded so names sort."""
    return f"{seq:08d}.csv"


def parse_seq(name: str) -> int | None:
    """Sequence number of a spool file name, or None for anything else
    (a .tmp still being written, the lock file)."""
    m = _SEQ_RE.match(name)
    return int(m.group(1)) if m else None


def spool_append(spool_dir: Path, csv_data: str, retain_days: int) -> int:
    """Append the CSV to the spool log; returns its sequence number.

    The log is contiguous from 1 and the consumer tracks it with a cursor, so
    allocation must not skip or reuse a number. Two writers exist on a
    collector (the background pusher and a manual ``slipstream push``), so the
    listing, the write, and the rename run under an exclusive lock. Writing to
    a .tmp first keeps a partial file out of the consumer's listing.
    """
    spool_dir.mkdir(parents=True, exist_ok=True)
    with open(spool_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        seqs = [s for s in (parse_seq(p.name) for p in spool_dir.iterdir()) if s]
        seq = max(seqs, default=0) + 1
        name = seq_name(seq)
        tmp = spool_dir / (name + ".tmp")
        tmp.write_text(csv_data)
        tmp.rename(spool_dir / name)
        _prune(spool_dir, retain_days)
    return seq


def _prune(spool_dir: Path, retain_days: int) -> None:
    """Drop log entries older than ``retain_days``.

    The producer does not know what the consumer has read, so this can prune
    past an undelivered entry; that surfaces there as a gap, never as silent
    loss.
    """
    cutoff = time.time() - retain_days * 86400
    for p in spool_dir.iterdir():
        if parse_seq(p.name) and p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


def probe(push_cfg: PushConfig, log: Callable[[str], None]) -> None:
    """Send a header-only CSV through every target, marking nothing.

    Exercises credentials, schema, and reachability the way a real push
    would; a target that fails raises as it would then.
    """
    for target in push_cfg.targets:
        summary = deliver_once(target, push_cfg.bot_name, _to_csv([]))
        log(f"{_describe(target)}: {summary or 'ok'}")


def _describe(target: PushTarget) -> str:
    if target.spool_dir is not None:
        return f"spool {target.spool_dir}"
    return f"spanner {target.spanner}"


def push(
    store: CommitStore,
    engines: list[str],
    push_cfg: PushConfig,
    valid_benchmarks_by_suite: dict[str, set[str]],
    platform: str,
    *,
    rebuild: bool = False,
    log: Callable[[str], None] | None = None,
) -> int:
    """Push all unpushed commits of ``engines`` to every target.

    Returns the number of score rows sent. Commits whose scores are entirely
    outside the benchmark whitelist have nothing to send; they are marked
    pushed anyway so they don't come back every cycle.

    ``rebuild`` first forgets every push for these engines, so the full
    history is re-sent, and tells targets to start from a clean slate.
    """
    if rebuild:
        for engine in engines:
            store.clear_push_state(engine, platform)
    rows: list[list] = []
    per_engine_commits: dict[str, list[int]] = {}
    for engine in engines:
        commit_ids = store.unpushed_commit_ids(engine, platform)
        if not commit_ids:
            continue
        rows.extend(
            _export_rows(store, engine, platform, valid_benchmarks_by_suite, commit_ids)
        )
        per_engine_commits[engine] = commit_ids

    # Release the read snapshot taken while exporting before delivering to the
    # (multi-second) targets. Holding it across them and then
    # upgrading to a write in mark_pushed would deadlock against the
    # collector's writer; a fresh short write below gets busy_timeout instead.
    store.conn.commit()

    if not per_engine_commits:
        return 0

    if rows:
        csv_data = _to_csv(rows)
        for target in push_cfg.targets:
            summary = deliver_once(target, push_cfg.bot_name, csv_data, rebuild=rebuild)
            if summary and log:
                log(summary)

    for engine, commit_ids in per_engine_commits.items():
        # A commit whose scores have no row in `commits` exports nothing, since
        # the export inner-joins them, but marking it pushed would retire those
        # scores for good. Hold it back and say so; every other commit with
        # nothing to export is marked as before.
        orphans = set(
            store.commit_ids_missing_commit_row(
                engine, platform, min(commit_ids), max(commit_ids)
            )
        )
        held = sorted(orphans.intersection(commit_ids))
        if held and log:
            log(
                f"{engine}: {len(held)} commits have scores but no commit row "
                f"(first {held[0]}); not marking them pushed"
            )
        store.mark_pushed(engine, platform, [c for c in commit_ids if c not in orphans])
    return len(rows)
