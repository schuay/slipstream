# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Push export CSVs into the perf Spanner database and aggregate them.

The database feeds other perf frontends, so the table shapes in
``data/spanner_schema.sql`` (including the stored trace_id on benchmarks)
are fixed; the indexes are slipstream's own. Rows land in the
``slipstream`` staging table with an ``imported_at`` stamp; ``refresh``
then re-aggregates only the (bot, benchmark, commit) groups that gained
rows since the watermark in ``meta`` into ``benchmarks``. Both steps are
idempotent: staging rows are upserted on their primary key and aggregates
are recomputed over the full run set of a group, so a replayed push
converges to the same state.

All Spanner I/O goes through :class:`SpannerDb` so the mapping and SQL can
be tested against a recording fake.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from importlib.resources import files as pkg_files
from pathlib import Path

from .config import parse_spanner_spec

IMPORT_TABLE = "slipstream"
AGG_TABLE = "benchmarks"
META_TABLE = "meta"
WATERMARK_KEY = "slipstream_last_imported_at"
INCOMPLETE_PREFIX = "slipstream_incomplete_import:"
# Indexes refresh() hints; declared in data/spanner_schema.sql.
IMPORTED_AT_INDEX = "slipstream_imported_at_idx"
GROUP_INDEX = "slipstream_group_idx"

# google-cloud-spanner switches for multiplexed sessions, per transaction type.
_MULTIPLEXED_SESSION_ENV = (
    "GOOGLE_CLOUD_SPANNER_MULTIPLEXED_SESSIONS",
    "GOOGLE_CLOUD_SPANNER_MULTIPLEXED_SESSIONS_PARTITIONED_OPS",
    "GOOGLE_CLOUD_SPANNER_MULTIPLEXED_SESSIONS_FOR_RW",
)

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

        # Workaround: Database.close() blocks up to 600s joining the
        # multiplexed-session maintenance thread (google-cloud-spanner 3.71).
        # TODO: Remove once fixed upstream.
        #
        # The first query on a multiplexed session starts a maintenance
        # thread that polls with an uninterruptible 10 minute sleep, and
        # Database.close() joins it: every push would sit in close() until
        # ten minutes after connecting. Pooled sessions close at once. The
        # client reads these per checkout, and each transaction type would
        # start the thread on its own.
        for var in _MULTIPLEXED_SESSION_ENV:
            os.environ[var] = "false"
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
        started = time.monotonic()
        self._log(f"query start sql={' '.join(sql.split())[:160]}")
        with self._con.cursor() as cur:
            cur.execute(sql, params or None)
            rows = [tuple(r) for r in cur.fetchall()]
        self._log(
            f"query rows={len(rows)} elapsed={time.monotonic() - started:.3f}s sql={' '.join(sql.split())[:160]}"
        )
        return rows

    def execute(self, sql: str, params: list | None = None) -> None:
        started = time.monotonic()
        with self._con.cursor() as cur:
            cur.execute(sql, params or None)
        self._log(
            f"execute elapsed={time.monotonic() - started:.3f}s sql={' '.join(sql.split())[:100]}"
        )

    def _log(self, msg):
        getattr(self, "log", lambda msg: None)(msg)

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
            started = time.monotonic()
            self._log(
                f"write start table={table} chunk={i // chunk + 1} rows={len(rows[i : i + chunk])}"
            )
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=table, columns=columns, values=rows[i : i + chunk]
                )
            self._log(
                f"write complete table={table} elapsed={time.monotonic() - started:.3f}s"
            )

    def close(self) -> None:
        errors = []
        for step in (self._con.close, self._release_channels):
            try:
                step()
            except Exception as exc:
                errors.append(exc)
                self._log(f"cleanup failed: {type(exc).__name__}: {exc}")
        lock = getattr(self, "_local_lock", None)
        if lock is not None:
            lock.close()
        if errors:
            raise errors[0]

    def _release_channels(self) -> None:
        """Attempt every cleanup step; report failure after releasing resources."""
        errors = []
        for step in (
            lambda: self._database.close(),
            lambda: self._api_transport(self._database, "_spanner_api"),
            lambda: self._api_transport(self._client, "_database_admin_api"),
            lambda: self._api_transport(self._client, "_instance_admin_api"),
        ):
            try:
                step()
            except Exception as exc:
                errors.append(exc)
                self._log(f"channel cleanup failed: {type(exc).__name__}: {exc}")
        if errors:
            raise errors[0]

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
    """Wait for other local deliveries to the same database to finish."""
    from .durability import FileLock, identity_digest

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    canonical_spec = "/".join(parse_spanner_spec(spec))
    path = (
        cache_dir
        / "slipstream"
        / "locks"
        / f"spanner-{identity_digest(canonical_spec)}.lock"
    )
    lock = FileLock(path)
    lock.__enter__()
    return lock


# --- Schema ---


def schema_statements() -> list[str]:
    text = (pkg_files("slipstream.data") / "spanner_schema.sql").read_text()
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("--"))
    return [s.strip() for s in body.split(";") if s.strip()]


def ensure_schema(db) -> None:
    """Apply the DDL only when a table is missing; DDL is slow even as a no-op.

    Indexes on an existing database are not touched: a backfill over the
    staging table takes far longer than a push should, so index changes are
    applied to the live database by hand (see :func:`index_drift`).
    """
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


_CREATE_INDEX = re.compile(r"CREATE INDEX IF NOT EXISTS (\w+)\s+ON (\w+)")


def declared_indexes() -> dict[str, str]:
    """Index name -> its CREATE statement, from the bundled DDL."""
    out = {}
    for stmt in schema_statements():
        m = _CREATE_INDEX.match(stmt)
        if m:
            out[m.group(1)] = stmt
    return out


def index_drift(db) -> tuple[list[str], list[str], list[str]]:
    """(missing, building, retired) index names relative to the bundled DDL.

    ``retired`` are indexes on slipstream's tables that the DDL no longer
    declares (earlier versions, or ones added by hand).
    """
    declared = declared_indexes()
    live = {
        r[0]: r[1]
        for r in db.query(
            "SELECT index_name, index_state FROM INFORMATION_SCHEMA.INDEXES"
            " WHERE table_schema = '' AND index_type = 'INDEX'"
            " AND table_name IN UNNEST(%s)",
            [[IMPORT_TABLE, AGG_TABLE, META_TABLE]],
        )
    }
    missing = [n for n in declared if n not in live]
    building = [n for n in declared if live.get(n, "READ_WRITE") != "READ_WRITE"]
    retired = sorted(n for n in live if n not in declared)
    return missing, building, retired


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
    reader = csv.DictReader(io.StringIO(csv_text), skipinitialspace=True)
    from .delivery_batch import COLUMNS

    return rows_from_records(
        [tuple(r.get(c, "") for c in COLUMNS) for r in reader], bot, imported_at
    )


def rows_from_records(records, bot: str, imported_at: datetime) -> list[tuple]:
    from .delivery_batch import COLUMNS

    rows = []
    for record in records:
        r = dict(zip(COLUMNS, record))
        suite = r["suite"].strip()
        ts = str(r.get("commit_timestamp", "")).strip()
        commit_time = datetime.fromtimestamp(int(ts), tz=timezone.utc) if ts else None
        rows.append(
            (
                bot,
                _BENCHMARK_ALIASES.get(suite, suite),
                r["benchmark"].strip(),
                r["metric"].strip(),
                variant_label(r["engine"].strip(), r["flags"].strip()),
                r["platform"].strip(),
                int(str(r["commit_id"])),
                int(str(r["run"])),
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

    slipstream_group_idx is on (bot, benchmark, commit_number), so this reads
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


def rebuild_bot(db, bot: str) -> int:
    """Explicit destructive repair, including obsolete aggregate score keys."""
    count = wipe_bot(db, bot)
    db.partitioned_dml(
        f"DELETE FROM {AGG_TABLE} WHERE bot = @bot AND source = 'slipstream'",
        {"bot": bot},
    )
    # After the explicit full wipe, old imports for this bot contain no partial
    # raw rows. Preserve other bots' markers and the global refresh watermark.
    db.partitioned_dml(
        f"DELETE FROM {META_TABLE} WHERE key = @legacy OR "
        "(STARTS_WITH(key, @prefix) AND STARTS_WITH(value, @bot))",
        {
            "legacy": INCOMPLETE_PREFIX + bot,
            "prefix": INCOMPLETE_PREFIX + "attempt:",
            "bot": f"{len(bot)}:{bot}:",
        },
    )
    return count


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


def _clean(row) -> tuple:
    return tuple(None if isinstance(v, float) and math.isnan(v) else v for v in row)


# One pass over staging yields both the line items and the suite total:
# Overall/Total-Score becomes test 'Total'. Staging has no line item named
# 'Total', so the two never share a group.
_AGG_SELECT = f"""
    SELECT
        s.bot, s.benchmark, IF(s.test = 'Overall', 'Total', s.test) AS test,
        '' AS submetric, {_VARIANT_EXPR} AS variant, s.commit_number,
        MIN(s.commit_time), MIN(s.git_hash), 'slipstream' AS source,
        AVG(s.val), MIN(s.val), MAX(s.val),
        COALESCE(STDDEV_SAMP(s.val), 0.0), COUNT(*)
    FROM {{source}}
    WHERE ((s.test != 'Overall' AND s.metric IN ('Total-Score', 'Score', ''))
           OR (s.test = 'Overall' AND s.metric = 'Total-Score'))
      {{where}}
    GROUP BY s.bot, s.benchmark, IF(s.test = 'Overall', 'Total', s.test),
        {_VARIANT_EXPR}, s.commit_number
"""

# JS2 groups without a harness total get the geomean of their line items. Only
# historical data lacks one: suite runs always print it.
_GEOMEAN_BENCHMARK = "jetstream2.slipstream"


def _geomean_totals(rows: list[tuple]) -> list[tuple]:
    """Total rows for JS2 groups that have line items but no Total."""
    groups: dict[tuple, list[tuple]] = defaultdict(list)
    have_total = set()
    for r in rows:
        bot, benchmark, test, _, variant, commit = r[:6]
        if benchmark != _GEOMEAN_BENCHMARK:
            continue
        key = (bot, benchmark, variant, commit)
        if test == "Total":
            have_total.add(key)
        elif r[9] is not None and r[9] > 0:
            groups[key].append(r)
    out = []
    for key, items in groups.items():
        if key in have_total:
            continue
        bot, benchmark, variant, commit = key

        def geo(col, floor=None):
            vals = [r[col] if floor is None else max(r[col], floor) for r in items]
            return math.exp(sum(math.log(v) for v in vals) / len(vals))

        times = [r[6] for r in items if r[6] is not None]
        hashes = [r[7] for r in items if r[7] is not None]
        out.append(
            (
                bot,
                benchmark,
                "Total",
                "",
                variant,
                commit,
                min(times) if times else None,
                min(hashes) if hashes else None,
                "slipstream",
                geo(9),
                geo(10, 1e-9),
                geo(11, 1e-9),
                0.0,
                min(r[13] for r in items),
            )
        )
    return out


def refresh(db) -> str | None:
    """Aggregate staging into benchmarks. Returns None when it ran, else why not.

    Every (bot, benchmark, commit) group with a row newer than the watermark
    in ``meta`` is recomputed over its full run set: line items (every test
    but Overall, variants cleaned) and test 'Total' from Overall/Total-Score,
    or for JS2 groups without one, the geomean of the line items. The
    watermark advances last, so a failed run is retried next time.

    Reads stay proportional to the new rows: the changed groups come off
    the imported_at index, and each (bot, benchmark) pair is then read
    through the covering group index at just those commits. Without a
    watermark (first run) everything is aggregated in one scan.
    """
    incomplete = db.query(
        f"SELECT key FROM {META_TABLE} WHERE STARTS_WITH(key, %s) LIMIT 1",
        [INCOMPLETE_PREFIX],
    )
    if incomplete:
        return f"an import is incomplete ({incomplete[0][0]}); shared-target refresh blocked"
    missing, building, _ = index_drift(db)
    if missing or building:
        return (
            f"indexes not ready ({', '.join(missing + building)});"
            " the database is behind data/spanner_schema.sql"
        )

    found = db.query(f"SELECT value FROM {META_TABLE} WHERE key = %s", [WATERMARK_KEY])
    try:
        last = _to_utc(found[0][0]) if found else None
    except ValueError as e:
        # Silently treating this as "no watermark" would re-aggregate
        # everything on every push; make the operator fix the row.
        raise ValueError(f"unparseable {WATERMARK_KEY} in {META_TABLE}: {e}") from e

    if last is None:
        (raw_cutoff,) = db.query(f"SELECT MAX(imported_at) FROM {IMPORT_TABLE}")[0]
        if raw_cutoff is None:
            return "staging is empty"
        rows = db.query(_AGG_SELECT.format(source=f"{IMPORT_TABLE} s", where=""), None)
    else:
        changed = db.query(
            f"SELECT bot, benchmark, commit_number, MAX(imported_at)"
            f" FROM {IMPORT_TABLE}@{{FORCE_INDEX={IMPORTED_AT_INDEX}}}"
            " WHERE imported_at > %s"
            " GROUP BY bot, benchmark, commit_number",
            [last],
        )
        if not changed:
            return "no rows newer than the last refresh"
        raw_cutoff = max(r[3] for r in changed)
        commits: dict[tuple[str, str], list[int]] = defaultdict(list)
        for bot, benchmark, commit, _ in changed:
            commits[(bot, benchmark)].append(commit)
        sql = _AGG_SELECT.format(
            source=f"{IMPORT_TABLE}@{{FORCE_INDEX={GROUP_INDEX}}} s",
            where="AND s.bot = %s AND s.benchmark = %s"
            " AND s.commit_number IN UNNEST(%s)",
        )
        rows = []
        for (bot, benchmark), nums in sorted(commits.items()):
            rows += db.query(sql, [bot, benchmark, sorted(nums)])

    rows = [_clean(r) for r in rows]
    rows += _geomean_totals(rows)
    if rows:
        db.upsert(AGG_TABLE, AGG_COLUMNS, rows)
    db.execute(
        f"INSERT OR UPDATE INTO {META_TABLE} (key, value) VALUES (%s, %s)",
        [WATERMARK_KEY, _to_utc(raw_cutoff).isoformat()],
    )
    return None


# --- Entry point used by push targets ---


def stage_rows(db, rows: list[tuple]) -> int:
    """Upsert mapped rows into staging; returns the row count."""
    if rows:
        db.upsert(IMPORT_TABLE, IMPORT_COLUMNS, rows)
    return len(rows)


def begin_import(db, bot: str, attempt: str, digest: str) -> str:
    """Only the exact interrupted attempt may clear its own import marker.

    Legacy per-bot markers deliberately remain until explicit reconciliation.
    """
    key = f"{INCOMPLETE_PREFIX}attempt:{attempt}"
    db.execute(
        f"INSERT OR UPDATE INTO {META_TABLE} (key, value) VALUES (%s, %s)",
        [key, f"{len(bot)}:{bot}:{digest}"],
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
    from .durability import identity_digest

    digest = identity_digest(csv_text)
    marker = begin_import(db, bot, identity_digest(bot + ":" + digest), digest)
    n_rows = stage_rows(db, rows)
    finish_import(db, marker)
    return summary_line(
        bot,
        n_rows,
        rebuild=rebuild,
        skipped=refresh(db) if refresh_agg else None,
        refreshed=refresh_agg,
    )
