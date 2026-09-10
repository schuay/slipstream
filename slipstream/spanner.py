# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Push export CSVs into the perf Spanner database and aggregate them.

The database feeds other perf frontends, so the schema in
``data/spanner_schema.sql`` (including the stored trace_id on benchmarks)
is fixed. Rows land in the ``slipstream`` staging table with an
``imported_at`` stamp; ``refresh`` then re-aggregates only the
(bot, benchmark, commit) groups that gained rows since the watermark in
``meta`` into ``benchmarks``. Both steps are idempotent: staging rows are
upserted on their primary key and aggregates are recomputed over the full
run set of a group, so a replayed push converges to the same state.

All Spanner I/O goes through :class:`SpannerDb` so the mapping and SQL can
be tested against a recording fake.
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import io
import math
import os
from datetime import datetime, timezone
from importlib.resources import files as pkg_files
from pathlib import Path

from .config import parse_spanner_spec

# The pusher thread holds a gRPC channel open while the collector forks
# benchmarks, and where that is a real fork (macOS, which has no vfork path in
# subprocess) grpc's atfork handlers log stale poller fds over the progress
# output. Muting them is a workaround: the fix is to move BackgroundPusher into
# its own process. Must be set before google.cloud.spanner imports grpc.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")

IMPORT_TABLE = "slipstream"
AGG_TABLE = "benchmarks"
META_TABLE = "meta"
WATERMARK_KEY = "slipstream_last_imported_at"
INCOMPLETE_PREFIX = "slipstream_incomplete_import:"

IMPORT_COLUMNS = [
    "bot",
    "benchmark",
    "test",
    "metric",
    "variant",
    "platform",
    "commit_number",
    "run",
    "commit_time",
    "git_hash",
    "val",
    "imported_at",
]
AGG_COLUMNS = [
    "bot",
    "benchmark",
    "test",
    "submetric",
    "variant",
    "commit_number",
    "commit_time",
    "git_hash",
    "source",
    "mean",
    "min",
    "max",
    "stdev",
    "count",
]

# Suite names as the frontends know them.
_BENCHMARK_ALIASES = {
    "js2": "jetstream2.slipstream",
    "js3": "jetstream3.slipstream",
    "sp3": "speedometer3.1.slipstream",
}


class SpannerDb:
    """Thin wrapper over the Spanner DB-API connection and batch API."""

    def __init__(self, project: str, instance: str, database: str):
        import warnings

        # The ADC quota-project warning fires on every connection and is
        # irrelevant here.
        warnings.filterwarnings(
            "ignore",
            message="Your application has authenticated using end user credentials",
            category=UserWarning,
            module=r"google\.auth\._default",
        )
        from google.cloud import spanner
        from google.cloud.spanner_dbapi import connect

        # Built-in metrics fail on every push with a missing instance_id
        # label and log a Cloud Monitoring 400; disabling them on the client
        # is honoured, the environment variable is not.
        client = spanner.Client(project=project, disable_builtin_metrics=True)
        self._client = client
        self._con = connect(instance, database, client=client)
        self._con.autocommit = True
        self._database = self._con.database
        # Fail fast on bad credentials. The first DDL would otherwise sit in
        # the client's retry policy, which retries auth errors for an hour.
        try:
            self.query("SELECT 1")
        except Exception as e:
            self.close()
            raise ConnectionError(f"Spanner auth check failed: {e}") from e

    def query(self, sql: str, params: list | None = None) -> list[tuple]:
        with self._con.cursor() as cur:
            cur.execute(sql, params or None)
            return [tuple(r) for r in cur.fetchall()]

    def execute(self, sql: str, params: list | None = None) -> None:
        with self._con.cursor() as cur:
            cur.execute(sql, params or None)

    def partitioned_dml(self, sql: str, params: dict[str, str | int]) -> int:
        """Run DML over the whole table without the per-transaction limit."""
        from google.cloud.spanner_v1 import param_types

        types = {
            k: param_types.STRING if isinstance(v, str) else param_types.INT64
            for k, v in params.items()
        }
        return self._database.execute_partitioned_dml(
            sql, params=params, param_types=types
        )

    def upsert(self, table: str, columns: list[str], rows: list[tuple]) -> None:
        # Each commit is limited to 80k mutations. A row costs about one per
        # column for the table plus the same again per secondary index it
        # touches; benchmarks has two indexes and a stored generated column,
        # so budget three times the column count and aim for half the limit.
        chunk = max(500, 40_000 // (max(1, len(columns)) * 3))
        for i in range(0, len(rows), chunk):
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=table, columns=columns, values=rows[i : i + chunk]
                )

    def close(self) -> None:
        try:
            self._con.close()
        finally:
            try:
                self._release_channels()
            finally:
                lock = getattr(self, "_local_lock", None)
                if lock is not None:
                    lock.close()

    def _release_channels(self) -> None:
        """Close the gRPC channels this connection opened.

        The DB-API close only clears the session pool, and each of these
        clients holds a channel of its own, so a daemon that connects per push
        accumulates descriptors until grpc aborts the process trying to make
        another wakeup fd. Only already-created clients are touched: reading
        the public property would build a channel for the sake of closing it.
        """
        # Stops the multiplexed-session thread and deletes its session, so it
        # has to run while the channel it needs is still up.
        for step in (
            lambda: self._database.close(),
            lambda: self._api_transport(self._database, "_spanner_api"),
            lambda: self._api_transport(self._client, "_database_admin_api"),
            lambda: self._api_transport(self._client, "_instance_admin_api"),
        ):
            # A cleanup failure must not mask the error that led here, nor cost
            # the caller the rest of the teardown.
            with contextlib.suppress(Exception):
                step()

    @staticmethod
    def _api_transport(owner, attr: str) -> None:
        api = getattr(owner, attr, None)
        if api is not None:
            api.transport.close()


def connect(spec: str, *, exclusive: bool = True) -> SpannerDb:
    """Open the database. ``exclusive`` takes the local delivery lock.

    Reads do not need it: it exists to stop two deliveries interleaving, and
    taking it for a query would make a reporting command and a background push
    fail each other.
    """
    lock = _acquire_local_lock(spec) if exclusive else None
    try:
        db = SpannerDb(*parse_spanner_spec(spec))
    except Exception:
        if lock is not None:
            lock.close()
        raise
    db._local_lock = lock
    return db


def _acquire_local_lock(spec: str):
    """Exclude other processes on this machine from the same database."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    lock_dir = cache_dir / "slipstream" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    canonical_spec = "/".join(parse_spanner_spec(spec))
    digest = hashlib.sha256(canonical_spec.encode()).hexdigest()
    lock = open(lock_dir / f"spanner-{digest}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as e:
        lock.close()
        raise RuntimeError(
            f"another local delivery to Spanner target {spec} is active"
        ) from e
    return lock


# --- Schema ---


def schema_statements() -> list[str]:
    text = (pkg_files("slipstream.data") / "spanner_schema.sql").read_text()
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def ensure_schema(db) -> None:
    """Apply the DDL only when a table is missing; DDL is slow even as a no-op."""
    have = {
        r[0]
        for r in db.query(
            "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = ''"
        )
    }
    if {IMPORT_TABLE, AGG_TABLE, META_TABLE} <= have:
        return
    for stmt in schema_statements():
        db.execute(stmt)


# --- Staging rows ---


def variant_label(engine: str, flags: str) -> str:
    """Raw variant as stored in the staging table.

    Kept exactly as the previous ingest wrote it (the export has already
    prefixed flags with the engine, so the default variant of v8 reads
    "v8 (v8_default)"); refresh strips it back to the flags. Existing rows
    hold these strings, so this is part of the schema in practice.
    """
    if not engine:
        return flags
    if flags == "default":
        return engine
    return f"{engine} ({flags})"


def rows_from_csv(csv_text: str, bot: str, imported_at: datetime) -> list[tuple]:
    """Map export CSV rows to staging rows in IMPORT_COLUMNS order."""
    rows = []
    for r in csv.DictReader(io.StringIO(csv_text), skipinitialspace=True):
        suite = r["suite"].strip()
        ts = r.get("commit_timestamp", "").strip()
        commit_time = datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else None
        rows.append(
            (
                bot,
                _BENCHMARK_ALIASES.get(suite, suite),
                r["benchmark"].strip(),
                r["metric"].strip(),
                variant_label(r["engine"].strip(), r["flags"].strip()),
                r["platform"].strip(),
                int(r["commit_id"]),
                int(r["run"]),
                commit_time,
                r.get("git_hash", "").strip() or None,
                float(r["score"]),
                imported_at,
            )
        )
    return rows


def current_timestamp(db) -> datetime:
    """Return Spanner's clock for ordering imports across clients."""
    return _to_utc(db.query("SELECT CURRENT_TIMESTAMP()")[0][0])


def benchmark_name(suite: str) -> str:
    """The name a suite is staged under, e.g. js3 -> jetstream3.slipstream."""
    return _BENCHMARK_ALIASES.get(suite, suite)


def commit_numbers(db, bot: str, benchmark: str) -> list[int]:
    """Commit numbers this bot has staged for a benchmark, ascending.

    slipstream_refresh_idx is on (bot, benchmark, commit_number), so this reads
    the index rather than the table.
    """
    rows = db.query(
        f"SELECT DISTINCT commit_number FROM {IMPORT_TABLE}"
        " WHERE bot = %s AND benchmark = %s"
        " ORDER BY commit_number",
        [bot, benchmark],
    )
    return [r[0] for r in rows]


def wipe_bot(db, bot: str) -> int:
    return db.partitioned_dml(
        f"DELETE FROM {IMPORT_TABLE} WHERE bot = @bot", {"bot": bot}
    )


# --- Aggregation ---

# Normalises staging variants: "engine (flags)" -> "flags", bare "v8" ->
# "default", anything else unchanged.
_VARIANT_EXPR = """CASE
        WHEN s.variant LIKE '% (%' THEN REGEXP_EXTRACT(s.variant, r'\\((.+)\\)')
        WHEN s.variant IN ('v8') THEN 'default'
        ELSE s.variant
    END"""


def _to_utc(raw) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        dt = datetime.fromisoformat(raw)
    else:
        dt = raw
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _upsert_select(db, table: str, columns: list[str], select_sql: str, params):
    # INSERT OR UPDATE ... SELECT as one statement would exceed the mutation
    # limit on large groups; select first, then upsert in batches.
    rows = [
        tuple(None if isinstance(v, float) and math.isnan(v) else v for v in row)
        for row in db.query(select_sql, params)
    ]
    if rows:
        db.upsert(table, columns, rows)


def refresh(db) -> str | None:
    """Aggregate staging into benchmarks. Returns None when it ran, else why not.

    Steps, each an upsert into benchmarks:
      1. line items: every test except Overall, variants cleaned;
      2. JS2 Total as the geomean of its line items;
      3. Overall/Total-Score as test 'Total' (overrides the geomean for JS2,
         the only source of Total for JS3).
    Only groups with a row newer than the watermark are recomputed, always
    over the group's full run set. The watermark advances last, so a failed
    step is retried next time.
    """
    incomplete = db.query(
        f"SELECT key FROM {META_TABLE} WHERE STARTS_WITH(key, %s) LIMIT 1",
        [INCOMPLETE_PREFIX],
    )
    if incomplete:
        return "an import is incomplete"

    (raw_cutoff,) = db.query(f"SELECT MAX(imported_at) FROM {IMPORT_TABLE}")[0]
    if raw_cutoff is None:
        return "staging is empty"
    cutoff = _to_utc(raw_cutoff)

    found = db.query(f"SELECT value FROM {META_TABLE} WHERE key = %s", [WATERMARK_KEY])
    try:
        last = _to_utc(found[0][0]) if found else None
    except ValueError as e:
        # Silently treating this as "no watermark" would re-aggregate
        # everything on every push; make the operator fix the row.
        raise ValueError(f"unparseable {WATERMARK_KEY} in {META_TABLE}: {e}") from e
    if last is not None and last >= cutoff:
        return "no rows newer than the last refresh"

    if last is not None:
        # A derived table joined once. A correlated EXISTS made Spanner
        # rescan staging per outer row and time out on large tables.
        affected = (
            f"INNER JOIN (SELECT DISTINCT bot, benchmark, commit_number"
            f" FROM {IMPORT_TABLE} WHERE imported_at > %s) _ak"
            " ON {A}.bot = _ak.bot AND {A}.benchmark = _ak.benchmark"
            " AND {A}.commit_number = _ak.commit_number"
        )
        params = [last]
    else:
        affected = ""
        params = []

    _upsert_select(
        db,
        AGG_TABLE,
        AGG_COLUMNS,
        f"""
        SELECT
            s.bot, s.benchmark, s.test, '' AS submetric,
            {_VARIANT_EXPR} AS variant, s.commit_number,
            MIN(s.commit_time), MIN(s.git_hash), 'slipstream' AS source,
            AVG(s.val), MIN(s.val), MAX(s.val),
            COALESCE(STDDEV_SAMP(s.val), 0.0), COUNT(*)
        FROM {IMPORT_TABLE} s
        {affected.replace("{A}", "s")}
        WHERE s.metric IN ('Total-Score', 'Score', '')
          AND s.test != 'Overall'
        GROUP BY s.bot, s.benchmark, s.test, {_VARIANT_EXPR}, s.commit_number
        """,
        params,
    )
    _upsert_select(
        db,
        AGG_TABLE,
        AGG_COLUMNS,
        f"""
        SELECT
            a.bot, a.benchmark, 'Total' AS test, '' AS submetric,
            a.variant, a.commit_number,
            MIN(a.commit_time), MIN(a.git_hash), 'slipstream' AS source,
            EXP(AVG(LN(a.mean))),
            EXP(AVG(LN(GREATEST(a.min, 1e-9)))),
            EXP(AVG(LN(GREATEST(a.max, 1e-9)))),
            0.0, MIN(a.count)
        FROM {AGG_TABLE} a
        {affected.replace("{A}", "a")}
        WHERE a.benchmark = 'jetstream2.slipstream'
          AND a.test != 'Total' AND a.submetric = '' AND a.mean > 0
        GROUP BY a.bot, a.benchmark, a.variant, a.commit_number
        """,
        params,
    )
    _upsert_select(
        db,
        AGG_TABLE,
        AGG_COLUMNS,
        f"""
        SELECT
            s.bot, s.benchmark, 'Total' AS test, '' AS submetric,
            {_VARIANT_EXPR} AS variant, s.commit_number,
            MIN(s.commit_time), MIN(s.git_hash), 'slipstream' AS source,
            AVG(s.val), MIN(s.val), MAX(s.val),
            COALESCE(STDDEV_SAMP(s.val), 0.0), COUNT(*)
        FROM {IMPORT_TABLE} s
        {affected.replace("{A}", "s")}
        WHERE s.test = 'Overall' AND s.metric = 'Total-Score'
        GROUP BY s.bot, s.benchmark, {_VARIANT_EXPR}, s.commit_number
        """,
        params,
    )

    db.execute(
        f"INSERT OR UPDATE INTO {META_TABLE} (key, value) VALUES (%s, %s)",
        [WATERMARK_KEY, cutoff.isoformat()],
    )
    return None


# --- Entry point used by push targets ---


def stage_rows(db, rows: list[tuple]) -> int:
    """Upsert mapped rows into staging; returns the row count."""
    if rows:
        db.upsert(IMPORT_TABLE, IMPORT_COLUMNS, rows)
    return len(rows)


def begin_import(db, bot: str) -> str:
    """Persist a marker that prevents refresh after partial staging."""
    key = f"{INCOMPLETE_PREFIX}{bot}"
    db.execute(
        f"INSERT OR UPDATE INTO {META_TABLE} (key, value) VALUES (%s, %s)",
        [key, "staging"],
    )
    return key


def finish_import(db, key: str) -> None:
    """Clear the marker only after every staging chunk committed."""
    db.execute(f"DELETE FROM {META_TABLE} WHERE key = %s", [key])


def summary_line(
    bot: str,
    n_rows: int,
    *,
    rebuild: bool = False,
    skipped: str | None = None,
    refreshed: bool = True,
) -> str:
    """The line a push or relay cycle reports for a spanner target."""
    summary = f"{n_rows} rows staged for {bot}"
    if rebuild:
        summary += " (rebuild)"
    if refreshed:
        summary += f", aggregation skipped ({skipped})" if skipped else ", aggregated"
    return summary


def push_csv(
    db, bot: str, csv_text: str, *, rebuild: bool = False, refresh_agg: bool = True
) -> str:
    """Stage the CSV rows for ``bot`` and aggregate. Returns a summary line."""
    ensure_schema(db)
    # Mapping first: a CSV that fails to parse must not cost the bot its
    # staged rows on a rebuild.
    rows = rows_from_csv(csv_text, bot, current_timestamp(db))
    if rebuild:
        wipe_bot(db, bot)
    marker = begin_import(db, bot)
    n_rows = stage_rows(db, rows)
    finish_import(db, marker)
    return summary_line(
        bot,
        n_rows,
        rebuild=rebuild,
        skipped=refresh(db) if refresh_agg else None,
        refreshed=refresh_agg,
    )
