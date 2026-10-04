# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Push export CSVs into the perf Spanner database and aggregate them.

The database feeds other perf frontends through ``benchmarks``, whose shape
in ``data/spanner_schema.sql`` (including the stored trace_id) is fixed. The
staging side is slipstream's own and is in transition between two designs,
both described in the DDL file:

* ``samples`` + ``commits`` + ``dirty_groups`` + ``imports``: a push writes
  its samples, the commit they belong to and the groups it touched in one
  commit; ``refresh`` drains ``dirty_groups``, re-aggregating each group
  over its full run set, and deletes only what it read.
* ``slipstream`` + ``meta``: the previous design, where a watermark over
  ``imported_at`` finds what changed.

A push target says which to write (``write``: legacy, both, samples) and
which to aggregate from (``aggregate_from``: legacy, samples). The frontend
labels, and with them trace_id, come out identical from both paths; the
functions that derive them are pinned by tests against the live values.
Both paths are idempotent: staging rows are upserted on their primary key
and aggregates are recomputed over the full run set of a group, so a
replayed push converges to the same state.

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
from typing import NamedTuple

from .config import (  # noqa: F401  (WRITE_MODES, AGGREGATE_SOURCES re-exported)
    AGGREGATE_SOURCES,
    WRITE_MODES,
    check_modes,
    parse_spanner_spec,
)

IMPORT_TABLE = "slipstream"
AGG_TABLE = "benchmarks"
META_TABLE = "meta"
SAMPLES_TABLE = "samples"
COMMITS_TABLE = "commits"
DIRTY_TABLE = "dirty_groups"
IMPORTS_TABLE = "imports"
STAGING_TABLES = (SAMPLES_TABLE, COMMITS_TABLE, DIRTY_TABLE, IMPORTS_TABLE)
WATERMARK_KEY = "slipstream_last_imported_at"
INCOMPLETE_PREFIX = "slipstream_incomplete_import:"
# Indexes refresh() hints; declared in data/spanner_schema.sql.
IMPORTED_AT_INDEX = "slipstream_imported_at_idx"
GROUP_INDEX = "slipstream_group_idx"

# What a push target may be told to write (WRITE_MODES) and to aggregate
# from (AGGREGATE_SOURCES) is defined with the target in config.py, so the
# config can validate the pair; see check_modes.

# The value the client library turns into the commit timestamp of the
# mutation, for columns declared with allow_commit_timestamp. Spelled out so
# the mapping needs no google import.
COMMIT_TIMESTAMP = "spanner.commit_timestamp()"

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
SAMPLE_COLUMNS = [
    "bot",
    "suite",
    "engine",
    "variant",
    "embedder_number",
    "commit_number",
    "test",
    "metric",
    "run",
    "value",
    "measured_at",
    "imported_at",
]
COMMIT_COLUMNS = [
    "engine",
    "embedder_number",
    "commit_number",
    "git_hash",
    "commit_time",
    "title",
    "embedder_hash",
    "embedder_title",
]
DIRTY_COLUMNS = [
    "bot",
    "suite",
    "engine",
    "variant",
    "embedder_number",
    "commit_number",
    "dirtied_at",
]

# Secondary indexes per table, for the mutation budget of a write: each row
# costs about one mutation per column for the table and the same again per
# index. benchmarks also has a stored generated column, counted as an index.
_INDEXES = {IMPORT_TABLE: 2, AGG_TABLE: 2}
# Each commit is limited to 80k mutations; aim for half.
_MUTATION_BUDGET = 40_000

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
        self.write([(table, columns, rows)])

    def write(self, groups: list[tuple[str, list[str], list[tuple]]]) -> None:
        """Upsert rows into several tables, chunked, each chunk one commit.

        Chunk k of every group goes into the same commit, so lists that are
        parallel (one old-design and one new-design row per record) stay
        together on disk: a push that dies between chunks leaves no record
        in one table without the other. Small groups (the commit, the dirty
        groups) are exhausted by the first chunk.
        """
        groups = [(t, c, r) for t, c, r in groups if r]
        if not groups:
            return
        cost = sum(len(c) * (1 + _INDEXES.get(t, 0)) for t, c, _ in groups)
        chunk = max(100, _MUTATION_BUDGET // cost)
        longest = max(len(r) for _, _, r in groups)
        for i in range(0, longest, chunk):
            started = time.monotonic()
            parts = [(t, c, r[i : i + chunk]) for t, c, r in groups if r[i : i + chunk]]
            self._log(
                f"write start chunk={i // chunk + 1} "
                + " ".join(f"{t}={len(r)}" for t, _, r in parts)
            )
            with self._database.batch() as batch:
                for table, columns, rows in parts:
                    batch.insert_or_update(table=table, columns=columns, values=rows)
            self._log(
                f"write complete chunk={i // chunk + 1} elapsed={time.monotonic() - started:.3f}s"
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
    if {IMPORT_TABLE, AGG_TABLE, META_TABLE, *STAGING_TABLES} <= have:
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
            [[IMPORT_TABLE, AGG_TABLE, META_TABLE, *STAGING_TABLES]],
        )
    }
    missing = [n for n in declared if n not in live]
    building = [n for n in declared if live.get(n, "READ_WRITE") != "READ_WRITE"]
    retired = sorted(n for n in live if n not in declared)
    return missing, building, retired


# --- Staging rows ---

# What the frontends see. benchmarks keys on (bot, benchmark, test, submetric,
# variant, commit_number) and stores a trace_id hashed from those labels, so
# the two functions below must keep producing today's labels for every
# existing series: the export CSV carries the suite short name and flags the
# exporter has prefixed with the engine ("v8_default"), the previous ingest
# stored "jetstream3.slipstream" and "v8 (v8_default)" and refresh reduced
# the variant back to "v8_default" in SQL. The new tables store "js3", "v8"
# and "default" and derive the same labels here. Pinned by tests against the
# live DISTINCT values.


def frontend_benchmark(suite: str) -> str:
    return _BENCHMARK_ALIASES.get(suite, suite)


def frontend_variant(engine: str, variant: str) -> str:
    return f"{engine}_{variant}"


def variant_of(engine: str, flags: str) -> str:
    """The run's own flags, with the engine prefix the exporter adds removed.

    Push's _export_rows writes flags as "<engine>_<flags>" so variants of
    different engines stay distinct in a table without an engine column.
    The new one has that column; the prefix would say it twice. Anything
    else is not a slipstream export and is refused rather than guessed.
    """
    prefix = f"{engine}_"
    if not engine or not flags.startswith(prefix):
        raise ValueError(f"flags {flags!r} do not start with the engine {engine!r}")
    return flags[len(prefix) :]


def variant_label(engine: str, flags: str) -> str:
    """Raw variant as stored in the previous staging table.

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


# The previous design's strings, and what they encode. Total over the live
# table on 2026-10-04; the backfill refuses anything outside it. Each entry
# round-trips: frontend_variant(*LEGACY_VARIANTS[v]) is what refresh's SQL
# made of v, and frontend_benchmark(LEGACY_BENCHMARKS[b]) == b.
LEGACY_VARIANTS = {
    "v8 (v8_default)": ("v8", "default"),
    "v8 (v8_per_line_item)": ("v8", "per_line_item"),
    "v8 (v8_turbolev_future)": ("v8", "turbolev_future"),
    "jsc (jsc_default)": ("jsc", "default"),
    "jsc (jsc_per_line_item)": ("jsc", "per_line_item"),
}
LEGACY_BENCHMARKS = {
    "jetstream2.slipstream": "js2",
    "jetstream3.slipstream": "js3",
}


def legacy_identity(variant: str, benchmark: str) -> tuple[str, str, str]:
    """(engine, variant, suite) of a previous-design row; total or an error."""
    try:
        engine, clean = LEGACY_VARIANTS[variant]
        return engine, clean, LEGACY_BENCHMARKS[benchmark]
    except KeyError as e:
        raise ValueError(f"no pinned mapping for legacy value {e.args[0]!r}") from e


def _commit_time(text) -> datetime | None:
    # Shared by both designs' rows so that commits.commit_time is exactly
    # what the previous design stored, and the aggregates agree.
    text = str(text).strip()
    return datetime.fromtimestamp(int(text), tz=timezone.utc) if text else None


def _measured_at(text) -> datetime | None:
    # The store keeps 0 for "unknown"; that is NULL, not 1970.
    text = str(text).strip()
    return (
        datetime.fromtimestamp(int(text), tz=timezone.utc)
        if text and int(text) > 0
        else None
    )


def rows_from_csv(csv_text: str, bot: str, imported_at: datetime) -> list[tuple]:
    """Map export CSV rows to previous-design rows in IMPORT_COLUMNS order."""
    return rows_from_records(records_from_csv(csv_text), bot, imported_at)


def records_from_csv(csv_text: str) -> list[tuple]:
    from .delivery_batch import COLUMNS

    reader = csv.DictReader(io.StringIO(csv_text), skipinitialspace=True)
    return [tuple(r.get(c, "") for c in COLUMNS) for r in reader]


def rows_from_records(records, bot: str, imported_at: datetime) -> list[tuple]:
    from .delivery_batch import COLUMNS

    rows = []
    for record in records:
        r = dict(zip(COLUMNS, record))
        suite = r["suite"].strip()
        rows.append(
            (
                bot,
                frontend_benchmark(suite),
                r["benchmark"].strip(),
                r["metric"].strip(),
                variant_label(r["engine"].strip(), r["flags"].strip()),
                r["platform"].strip(),
                int(str(r["commit_id"])),
                int(str(r["run"])),
                _commit_time(r.get("commit_timestamp", "")),
                r.get("git_hash", "").strip() or None,
                float(r["score"]),
                imported_at,
            )
        )
    return rows


class Staging(NamedTuple):
    """One push's rows for the current design, in column order."""

    samples: list[tuple]
    commits: list[tuple]
    dirty: list[tuple]


def staging_rows(records, bot: str) -> Staging:
    """Map export CSV records to samples, their commits and dirty groups.

    The export carries no embedder: everything delivery reads is embedder 0
    (see CommitStore.export_scores), and the perf database has no coordinate
    for anything else yet. Timestamps that are commit timestamps carry the
    sentinel; the commit's title comes along, the embedder columns stay
    NULL until the export has them.
    """
    from .delivery_batch import COLUMNS

    samples = []
    commits: dict[tuple, tuple] = {}
    dirty: dict[tuple, tuple] = {}
    for record in records:
        r = dict(zip(COLUMNS, record))
        engine = r["engine"].strip()
        variant = variant_of(engine, r["flags"].strip())
        suite = r["suite"].strip()
        commit = int(str(r["commit_id"]))
        group = (bot, suite, engine, variant, 0, commit)
        samples.append(
            (
                *group,
                r["benchmark"].strip(),
                r["metric"].strip(),
                int(str(r["run"])),
                float(r["score"]),
                _measured_at(r.get("timestamp", "")),
                COMMIT_TIMESTAMP,
            )
        )
        commits.setdefault(
            (engine, 0, commit),
            (
                engine,
                0,
                commit,
                r.get("git_hash", "").strip() or None,
                _commit_time(r.get("commit_timestamp", "")),
                r.get("commit_title", "").strip() or None,
                None,
                None,
            ),
        )
        dirty.setdefault(group, (*group, COMMIT_TIMESTAMP))
    return Staging(samples, list(commits.values()), list(dirty.values()))


def current_timestamp(db) -> datetime:
    """Return Spanner's clock for ordering imports across clients."""
    return _to_utc(db.query("SELECT CURRENT_TIMESTAMP()")[0][0])


def _check_write(write: str) -> None:
    if write not in WRITE_MODES:
        raise ValueError(f"write must be one of {WRITE_MODES}, not {write!r}")


def staging_groups(db, records, bot: str, *, write: str = "both") -> list[tuple]:
    """One push's rows per staging table, as :meth:`SpannerDb.write` takes them.

    Mapping happens here, before anything is written, so a record that
    does not map costs nothing. Note the timestamp of the previous-design
    rows is read from the database at mapping time.
    """
    _check_write(write)
    records = list(records)
    if not records:
        return []
    groups = []
    if write != "samples":
        groups.append(
            (
                IMPORT_TABLE,
                IMPORT_COLUMNS,
                rows_from_records(records, bot, current_timestamp(db)),
            )
        )
    if write != "legacy":
        staged = staging_rows(records, bot)
        groups += [
            (SAMPLES_TABLE, SAMPLE_COLUMNS, staged.samples),
            (COMMITS_TABLE, COMMIT_COLUMNS, staged.commits),
            (DIRTY_TABLE, DIRTY_COLUMNS, staged.dirty),
        ]
    return groups


def stage_records(db, records, bot: str, *, write: str = "both") -> int:
    """Upsert one push's records into the staging tables ``write`` names.

    With both designs written, each chunk carries its records' rows for
    both in one commit (see :meth:`SpannerDb.write`), so there is never a
    sample without its previous-design row or the other way round.
    """
    records = list(records)
    db.write(staging_groups(db, records, bot, write=write))
    return len(records)


def wipe_bot(db, bot: str, *, write: str = "both") -> int:
    """Delete a bot's staging rows; returns how many previous-design rows went.

    commits stays: it is per engine, and another bot may have measured the
    same commit.
    """
    _check_write(write)
    count = 0
    if write != "samples":
        count = db.partitioned_dml(
            f"DELETE FROM {IMPORT_TABLE} WHERE bot = @bot", {"bot": bot}
        )
    if write != "legacy":
        count = max(
            count,
            db.partitioned_dml(
                f"DELETE FROM {SAMPLES_TABLE} WHERE bot = @bot", {"bot": bot}
            ),
        )
        db.partitioned_dml(f"DELETE FROM {DIRTY_TABLE} WHERE bot = @bot", {"bot": bot})
    return count


def rebuild_bot(db, bot: str, *, write: str = "both") -> int:
    """Explicit destructive repair, including obsolete aggregate score keys."""
    count = wipe_bot(db, bot, write=write)
    db.partitioned_dml(
        f"DELETE FROM {AGG_TABLE} WHERE bot = @bot AND source = 'slipstream'",
        {"bot": bot},
    )
    # After the explicit full wipe, old imports for this bot contain no partial
    # raw rows. Preserve other bots' markers and the global refresh watermark.
    if write != "samples":
        db.partitioned_dml(
            f"DELETE FROM {META_TABLE} WHERE key = @legacy OR "
            "(STARTS_WITH(key, @prefix) AND STARTS_WITH(value, @bot))",
            {
                "legacy": INCOMPLETE_PREFIX + bot,
                "prefix": INCOMPLETE_PREFIX + "attempt:",
                "bot": f"{len(bot)}:{bot}:",
            },
        )
    if write != "legacy":
        # Finished imports stay as the ledger; an open one is what the wipe
        # just made moot.
        db.partitioned_dml(
            f"DELETE FROM {IMPORTS_TABLE} WHERE bot = @bot AND finished_at IS NULL",
            {"bot": bot},
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


# The watermark keeps Spanner's nanosecond precision. A plain datetime stops
# at microseconds, which leaves it just below the newest imported_at, so
# every refresh would re-aggregate the last push.
def _watermark_text(raw) -> str:
    from google.api_core.datetime_helpers import DatetimeWithNanoseconds

    if isinstance(raw, DatetimeWithNanoseconds):
        return raw.rfc3339()
    return _to_utc(raw).isoformat()


def _parse_watermark(text: str) -> datetime:
    from google.api_core.datetime_helpers import DatetimeWithNanoseconds

    try:
        return DatetimeWithNanoseconds.from_rfc3339(text)
    except ValueError:
        # Older rows were written with isoformat() ("+00:00", microseconds).
        return _to_utc(text)


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
            # fsum is exactly rounded, so the result does not depend on the
            # order the line items came out of SQL or on the Python version
            # (plain sum is naive before 3.12 and compensated after); the two
            # staging designs return items in different orders, and a Total
            # must not differ by an ulp between them.
            vals = [r[col] if floor is None else max(r[col], floor) for r in items]
            return math.exp(math.fsum(math.log(v) for v in vals) / len(vals))

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


def refresh(db, *, source: str = "legacy") -> str | None:
    """Aggregate staging into benchmarks. Returns None when it ran, else why not.

    ``source`` names the staging design to read: the previous one with its
    watermark (:func:`_refresh_legacy`) or the current one with its dirty
    groups (:func:`_refresh_samples`). Both recompute each affected group
    over its full run set, so a group aggregated by either path lands the
    same, and both leave their change-tracking state alone until the
    aggregates are written, so a failed run is retried next time.
    """
    if source not in AGGREGATE_SOURCES:
        raise ValueError(
            f"aggregate_from must be one of {AGGREGATE_SOURCES}, not {source!r}"
        )
    if source == "samples":
        return _refresh_samples(db)
    return _refresh_legacy(db)


def _refresh_legacy(db) -> str | None:
    """The watermark path over the previous staging table.

    Every (bot, benchmark, commit) group with a row newer than the watermark
    in ``meta`` is recomputed over its full run set: line items (every test
    but Overall, variants cleaned) and test 'Total' from Overall/Total-Score,
    or for JS2 groups without one, the geomean of the line items. The
    watermark advances last.

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
        last = _parse_watermark(found[0][0]) if found else None
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
        [WATERMARK_KEY, _watermark_text(raw_cutoff)],
    )
    return None


# The same aggregation over the current design. The group prefix (bot,
# suite, engine, variant, embedder) is a parameter, so what the SQL groups
# by is just the commit and the test; the frontend labels are derived in
# Python from the prefix. commit identity comes from commits; LEFT JOIN so a
# sample never disappears from its aggregate, it at worst lacks a time and
# hash as the previous design allowed. Filter and statistics are those of
# _AGG_SELECT, which is what makes the two paths agree.
_SAMPLES_AGG_SELECT = f"""
    SELECT
        s.commit_number, IF(s.test = 'Overall', 'Total', s.test) AS test,
        MIN(c.commit_time), MIN(c.git_hash),
        AVG(s.value), MIN(s.value), MAX(s.value),
        COALESCE(STDDEV_SAMP(s.value), 0.0), COUNT(*)
    FROM {SAMPLES_TABLE} s
    LEFT JOIN {COMMITS_TABLE} c
      ON c.engine = s.engine AND c.embedder_number = s.embedder_number
     AND c.commit_number = s.commit_number
    WHERE s.bot = %s AND s.suite = %s AND s.engine = %s AND s.variant = %s
      AND s.embedder_number = %s
      AND ((s.test != 'Overall' AND s.metric IN ('Total-Score', 'Score', ''))
           OR (s.test = 'Overall' AND s.metric = 'Total-Score'))
      {{where}}
    GROUP BY s.commit_number, IF(s.test = 'Overall', 'Total', s.test)
"""


def aggregate_samples(
    db,
    bot: str,
    suite: str,
    engine: str,
    variant: str,
    embedder: int,
    commits: list[int] | None = None,
) -> list[tuple]:
    """benchmarks rows (AGG_COLUMNS order) for one group prefix from samples.

    ``commits`` restricts to those commit numbers; None means every commit
    of the prefix, which is what a full re-aggregation or a validator wants.
    The JS2 geomean totals are not included: callers add
    :func:`_geomean_totals` over the complete row set, as refresh does.
    """
    params: list = [bot, suite, engine, variant, embedder]
    where = ""
    if commits is not None:
        where = "AND s.commit_number IN UNNEST(%s)"
        params.append(sorted(commits))
    benchmark = frontend_benchmark(suite)
    label = frontend_variant(engine, variant)
    rows = []
    for commit, test, commit_time, git_hash, *stats in db.query(
        _SAMPLES_AGG_SELECT.format(where=where), params
    ):
        rows.append(
            _clean(
                (
                    bot,
                    benchmark,
                    test,
                    "",
                    label,
                    commit,
                    commit_time,
                    git_hash,
                    "slipstream",
                    *stats,
                )
            )
        )
    return rows


def _refresh_samples(db) -> str | None:
    """The dirty-group path over the current staging tables.

    Every group in ``dirty_groups`` is recomputed over its full run set and
    the dirty rows are deleted last, by timestamp rather than by key: the
    read was a strong snapshot, so every row with dirtied_at no later than
    the newest one read was in it, and a group dirtied again by a push that
    committed after the snapshot carries a later timestamp and survives for
    the next refresh. A failed refresh leaves everything dirty.

    benchmarks has no column for an embedder, so a dirty group outside
    embedder 0 cannot be represented there; push never writes one, and
    rather than aggregate something half-described or skip it silently,
    refresh stops until an operator looks.
    """
    incomplete = db.query(
        f"SELECT attempt FROM {IMPORTS_TABLE} WHERE finished_at IS NULL LIMIT 1"
    )
    if incomplete:
        return (
            f"an import is incomplete (attempt {incomplete[0][0]});"
            " shared-target refresh blocked"
        )
    dirty = db.query(f"SELECT {', '.join(DIRTY_COLUMNS)} FROM {DIRTY_TABLE}")
    if not dirty:
        return "no dirty groups since the last refresh"
    outside = [r for r in dirty if r[4] != 0]
    if outside:
        return (
            f"{len(outside)} dirty groups are outside embedder 0"
            f" (first: {outside[0][:6]}); benchmarks has no column for that"
        )
    cutoff = max(r[6] for r in dirty)
    prefixes: dict[tuple, list[int]] = defaultdict(list)
    for *prefix, commit, _ in dirty:
        prefixes[tuple(prefix)].append(commit)
    rows = []
    for prefix, nums in sorted(prefixes.items()):
        rows += aggregate_samples(db, *prefix, commits=nums)
    rows += _geomean_totals(rows)
    if rows:
        db.upsert(AGG_TABLE, AGG_COLUMNS, rows)
    db.execute(f"DELETE FROM {DIRTY_TABLE} WHERE dirtied_at <= %s", [cutoff])
    return None


# --- Entry point used by push targets ---


class ImportHandle(NamedTuple):
    """What :func:`begin_import` recorded, for :func:`finish_import` to close.

    ``marker`` is the previous design's meta key, ``ledger`` says whether an
    imports row is open; either may be absent depending on the write mode.
    """

    attempt: str
    marker: str | None
    ledger: bool


def begin_import(
    db, bot: str, attempt: str, digest: str, *, write: str = "both"
) -> ImportHandle:
    """Record that ``attempt`` is staging, before its first chunk commits.

    Both designs block refresh while a record is open. The previous one
    keys a meta row by attempt, so only the exact interrupted attempt
    clears its own marker (legacy per-bot markers deliberately remain until
    explicit reconciliation). The current one opens an imports row; a
    retry of the attempt reopens it, finished_at and all, so an attempt
    that died between its last chunk and the receipt is open again until
    it finishes once more.
    """
    _check_write(write)
    marker = None
    if write != "samples":
        marker = f"{INCOMPLETE_PREFIX}attempt:{attempt}"
        db.execute(
            f"INSERT OR UPDATE INTO {META_TABLE} (key, value) VALUES (%s, %s)",
            [marker, f"{len(bot)}:{bot}:{digest}"],
        )
    ledger = write != "legacy"
    if ledger:
        db.execute(
            f"INSERT OR UPDATE INTO {IMPORTS_TABLE}"
            " (attempt, bot, digest, started_at, finished_at, row_count)"
            " VALUES (%s, %s, %s, PENDING_COMMIT_TIMESTAMP(), NULL, NULL)",
            [attempt, bot, digest],
        )
    return ImportHandle(attempt, marker, ledger)


def finish_import(db, handle: ImportHandle | str, rows: int | None = None) -> None:
    """Close the record only after every staging chunk committed.

    A bare string is a meta key, for clearing a previous-design marker on
    its own (see reconcile_legacy). ``rows`` goes into the ledger.
    """
    if isinstance(handle, str):
        handle = ImportHandle("", handle, False)
    if handle.marker is not None:
        db.execute(f"DELETE FROM {META_TABLE} WHERE key = %s", [handle.marker])
    if handle.ledger:
        db.execute(
            f"UPDATE {IMPORTS_TABLE}"
            " SET finished_at = PENDING_COMMIT_TIMESTAMP(), row_count = %s"
            " WHERE attempt = %s",
            [rows, handle.attempt],
        )


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
    db,
    bot: str,
    csv_text: str,
    *,
    rebuild: bool = False,
    refresh_agg: bool = True,
    write: str = "both",
    aggregate_from: str = "legacy",
) -> str:
    """Stage the CSV rows for ``bot`` and aggregate. Returns a summary line."""
    check_modes(write, aggregate_from)
    ensure_schema(db)
    # Mapping first: a CSV that fails to parse must not cost the bot its
    # staged rows on a rebuild.
    records = records_from_csv(csv_text)
    groups = staging_groups(db, records, bot, write=write)
    if rebuild:
        wipe_bot(db, bot, write=write)
    from .durability import identity_digest

    digest = identity_digest(csv_text)
    handle = begin_import(
        db, bot, identity_digest(bot + ":" + digest), digest, write=write
    )
    db.write(groups)
    finish_import(db, handle, len(records))
    return summary_line(
        bot,
        len(records),
        rebuild=rebuild,
        skipped=refresh(db, source=aggregate_from) if refresh_agg else None,
        refreshed=refresh_agg,
    )
