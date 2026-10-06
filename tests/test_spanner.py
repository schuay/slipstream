# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from slipstream import spanner
from slipstream.push import _COLUMNS

T0 = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


class FakeDb:
    """Records every call; answers queries from a scripted list."""

    def __init__(self, answers=None, indexes=None):
        self.answers = list(answers or [])
        self.calls: list[tuple] = []
        # index name -> state; every declared index ready unless overridden.
        self.indexes = (
            {n: "READ_WRITE" for n in spanner.declared_indexes()}
            if indexes is None
            else indexes
        )

    def query(self, sql, params=None):
        self.calls.append(("query", " ".join(sql.split()), params))
        if "STARTS_WITH(key" in sql or "finished_at IS NULL" in sql:
            return []
        if "CURRENT_TIMESTAMP" in sql:
            return [(T0,)]
        if "INFORMATION_SCHEMA.INDEXES" in sql:
            return list(self.indexes.items())
        return self.answers.pop(0) if self.answers else []

    def execute(self, sql, params=None):
        self.calls.append(("execute", " ".join(sql.split()), params))

    def upsert(self, table, columns, rows):
        self.calls.append(("upsert", table, columns, rows))

    def write(self, groups):
        # As in SpannerDb, one call writes several tables; recorded both as
        # the grouping and as one upsert per table.
        self.calls.append(("write", [t for t, _, _ in groups]))
        for table, columns, rows in groups:
            self.upsert(table, columns, rows)

    def partitioned_dml(self, sql, params):
        self.calls.append(("pdml", sql, params))
        return 0

    def close(self):
        self.calls.append(("close",))

    def of(self, kind, table=None):
        return [
            c for c in self.calls if c[0] == kind and (table is None or c[1] == table)
        ]


ALL_TABLES = [
    (t,)
    for t in (
        "slipstream",
        "benchmarks",
        "benchmarks_v2",
        "meta",
        *spanner.STAGING_TABLES,
    )
]


def _csv(*rows):
    lines = [",".join(_COLUMNS)]
    for r in rows:
        d = dict(
            engine="v8",
            platform="arm64",
            commit_id="100",
            suite="js3",
            flags="v8_default",
            benchmark="bench-a",
            metric="Total-Score",
            run="1",
            score="1.5",
            timestamp="0",
            git_hash="abc",
            commit_date="2026-01-01",
            commit_timestamp="1700000000",
            commit_title="t",
            embedder_id="0",
            embedder_hash="",
        )
        d.update(r)
        lines.append(",".join(d[c] for c in _COLUMNS))
    return "\n".join(lines) + "\n"


class TestRowMapping:
    def test_variant_strings_match_previous_ingest(self):
        # These are what existing rows hold; refresh strips to "v8_default".
        assert spanner.variant_label("v8", "v8_default") == "v8 (v8_default)"
        assert spanner.variant_label("v8", "v8_turbolev_future") == (
            "v8 (v8_turbolev_future)"
        )
        assert spanner.variant_label("jsc", "jsc_default") == "jsc (jsc_default)"
        assert spanner.variant_label("v8", "default") == "v8"
        assert spanner.variant_label("", "x") == "x"

    def test_both_designs_derive_the_same_frontend_labels(self):
        # benchmarks keys and trace_id hash on these labels, so for every
        # legacy value in the live table the new path must land on exactly
        # what the legacy SQL (REGEXP_EXTRACT of the parenthesised part)
        # made of it.
        import re

        for legacy, (engine, variant) in spanner.LEGACY_VARIANTS.items():
            expect = re.search(r"\((.+)\)", legacy).group(1)
            assert spanner.frontend_variant(engine, variant) == expect
            # And a fresh export of that variant maps to the same pair.
            flags = f"{engine}_{variant}"
            assert spanner.variant_label(engine, flags) == legacy
            assert spanner.variant_of(engine, flags) == variant
        for legacy, suite in spanner.LEGACY_BENCHMARKS.items():
            assert spanner.frontend_benchmark(suite) == legacy
        assert spanner.legacy_identity("v8 (v8_default)", "jetstream2.slipstream") == (
            "v8",
            "default",
            "js2",
        )
        with pytest.raises(ValueError, match="no pinned mapping"):
            spanner.legacy_identity("v8", "jetstream2.slipstream")
        with pytest.raises(ValueError, match="no pinned mapping"):
            spanner.legacy_identity("v8 (v8_default)", "speedometer3.1.slipstream")

    def test_variant_of_refuses_flags_without_the_engine_prefix(self):
        assert spanner.variant_of("v8", "v8_default") == "default"
        assert spanner.variant_of("jsc", "jsc_per_line_item") == "per_line_item"
        for engine, flags in (("v8", "default"), ("v8", "jsc_default"), ("", "x")):
            with pytest.raises(ValueError, match="do not start with the engine"):
                spanner.variant_of(engine, flags)

    def test_row_shape(self):
        (row,) = spanner.rows_from_csv(_csv({}), "bot1", T0)
        assert dict(zip(spanner.IMPORT_COLUMNS, row)) == {
            "bot": "bot1",
            "benchmark": "jetstream3.slipstream",
            "test": "bench-a",
            "metric": "Total-Score",
            "variant": "v8 (v8_default)",
            "platform": "arm64",
            "commit_number": 100,
            "run": 1,
            "commit_time": datetime.fromtimestamp(1700000000, tz=timezone.utc),
            "git_hash": "abc",
            "val": 1.5,
            "imported_at": T0,
        }

    def test_staging_rows_shape(self):
        records = spanner.records_from_csv(
            _csv(
                {"timestamp": "1700000100"},
                {"run": "2"},
                {"commit_id": "101", "git_hash": "def", "benchmark": "Overall"},
                {"suite": "js2"},
            )
        )
        staged = spanner.staging_rows(records, "bot1")
        assert len(staged.samples) == 4
        assert dict(zip(spanner.SAMPLE_COLUMNS, staged.samples[0])) == {
            "bot": "bot1",
            "suite": "js3",
            "engine": "v8",
            "variant": "default",
            "embedder_number": 0,
            "commit_number": 100,
            "test": "bench-a",
            "metric": "Total-Score",
            "run": 1,
            "value": 1.5,
            "measured_at": datetime.fromtimestamp(1700000100, tz=timezone.utc),
            "imported_at": spanner.COMMIT_TIMESTAMP,
        }
        # A zero bench timestamp is unknown, not 1970.
        assert staged.samples[1][10] is None
        # One commits row per (engine, embedder, commit), from the first
        # record that names it; embedder columns stay NULL.
        assert [dict(zip(spanner.COMMIT_COLUMNS, c)) for c in staged.commits] == [
            {
                "engine": "v8",
                "embedder_number": 0,
                "commit_number": n,
                "git_hash": h,
                "commit_time": datetime.fromtimestamp(1700000000, tz=timezone.utc),
                "title": "t",
                "embedder_hash": None,
                "embedder_title": None,
            }
            for n, h in ((100, "abc"), (101, "def"))
        ]
        # One dirty row per group, stamped at commit time.
        assert staged.dirty == [
            ("bot1", "js3", "v8", "default", 0, 100, spanner.COMMIT_TIMESTAMP),
            ("bot1", "js3", "v8", "default", 0, 101, spanner.COMMIT_TIMESTAMP),
            ("bot1", "js2", "v8", "default", 0, 100, spanner.COMMIT_TIMESTAMP),
        ]

    def test_commit_time_is_derived_identically_for_both_designs(self):
        # Including the previous design's quirk that a zero commit timestamp
        # is 1970, not NULL: commits.commit_time must equal what is stored.
        records = spanner.records_from_csv(
            _csv({}, {"commit_id": "101", "commit_timestamp": "0"})
        )
        legacy = spanner.rows_from_records(records, "b", T0)
        staged = spanner.staging_rows(records, "b")
        assert [r[8] for r in legacy] == [c[4] for c in staged.commits]
        assert staged.commits[1][4] == datetime(1970, 1, 1, tzinfo=timezone.utc)

    def test_unknown_suite_passes_through(self):
        (row,) = spanner.rows_from_csv(_csv({"suite": "sp3"}), "b", T0)
        assert row[1] == "speedometer3.1.slipstream"
        (row,) = spanner.rows_from_csv(_csv({"suite": "other"}), "b", T0)
        assert row[1] == "other"

    def test_parse_spec(self):
        from slipstream.config import parse_spanner_spec

        assert parse_spanner_spec("p/i/d") == ("p", "i", "d")
        with pytest.raises(ValueError):
            parse_spanner_spec("p/i")


class TestSchema:
    def test_ddl_only_when_a_table_is_missing(self):
        db = FakeDb([ALL_TABLES])
        spanner.ensure_schema(db)
        assert db.of("execute") == []

        # The previous design's tables alone are no longer enough.
        db = FakeDb([[("slipstream",), ("benchmarks",), ("meta",)]])
        spanner.ensure_schema(db)
        ddl = [c[1] for c in db.of("execute")]
        assert len(ddl) == len(spanner.schema_statements()) == 14
        assert any("trace_id INT64 NOT NULL AS (FARM_FINGERPRINT" in s for s in ddl)
        assert all(not s.startswith("--") for s in ddl)
        assert all("--" not in s for s in ddl), "inline comments break statements"
        created = [s.split()[5] for s in ddl if s.startswith("CREATE TABLE")]
        assert created == [
            "commits",
            "samples",
            "dirty_groups",
            "imports",
            "slipstream",
            "benchmarks",
            "benchmarks_v2",
            "meta",
        ]

    def test_benchmarks_v2_keys_on_the_embedder_and_keeps_trace_id(self):
        v1, v2 = [
            s
            for s in spanner.schema_statements()
            if "TABLE IF NOT EXISTS benchmarks" in s
        ]

        def key(stmt):
            return (
                stmt.split("PRIMARY KEY")[1].strip(" ();").replace(" ", "").split(",")
            )

        assert key(v1) == spanner.AGG_COLUMNS[:6]
        assert key(v2) == spanner.AGG_COLUMNS[:5] + ["embedder_number", "commit_number"]

        # The stored expression is byte-identical: a series keeps its id.
        def trace(stmt):
            return stmt.split("trace_id")[1].split("STORED")[0].replace(" ", "")

        assert trace(v1) == trace(v2)
        assert "'\\\\x1f'" in v2
        assert "embedder_hash   STRING(MAX)" in v2 and "embedder_hash" not in v1

    def test_samples_key_is_the_group_then_what_varies_in_it(self):
        (samples,) = [
            s for s in spanner.schema_statements() if "TABLE IF NOT EXISTS samples" in s
        ]
        key = samples.split("PRIMARY KEY")[1].strip(" ();").replace(" ", "").split(",")
        assert key == spanner.DIRTY_COLUMNS[:-1] + ["test", "metric", "run"]
        assert "allow_commit_timestamp = true" in samples

    def test_declared_indexes(self):
        idx = spanner.declared_indexes()
        assert set(idx) == {
            spanner.IMPORTED_AT_INDEX,
            spanner.GROUP_INDEX,
            "benchmarks_filter_idx",
            "benchmarks_trace_idx",
            "benchmarks_v2_filter_idx",
            "benchmarks_v2_trace_idx",
        }
        assert "STORING (commit_time, git_hash, val)" in idx[spanner.GROUP_INDEX]
        assert (
            "(trace_id, embedder_number, commit_number)"
            in idx["benchmarks_v2_trace_idx"]
        )


def _item(test, mean, *, bench="jetstream2.slipstream", commit=1, variant="v"):
    return (
        "b",
        bench,
        test,
        "",
        variant,
        commit,
        T0,
        "h",
        "slipstream",
        mean,
        mean / 2,
        mean * 2,
        0.5,
        3,
    )


class TestRefresh:
    def test_empty_staging(self):
        db = FakeDb([[], [(None,)]])
        assert spanner.refresh(db) == "staging is empty"
        assert db.of("upsert") == [] and db.of("execute") == []

    def test_watermark_up_to_date(self):
        db = FakeDb([[(T0.isoformat(),)], []])
        assert spanner.refresh(db) == "no rows newer than the last refresh"
        assert db.of("upsert") == [] and db.of("execute") == []

    def test_indexes_not_ready_skips_without_reading_staging(self):
        db = FakeDb(indexes={spanner.GROUP_INDEX: "WRITE_ONLY"})
        out = spanner.refresh(db)
        assert out.startswith("indexes not ready (")
        assert spanner.IMPORTED_AT_INDEX in out and spanner.GROUP_INDEX in out
        assert "spanner_schema.sql" in out
        assert not any("FROM slipstream" in c[1] for c in db.of("query"))
        assert db.of("upsert") == [] and db.of("execute") == []

    def test_first_run_is_full_and_sets_watermark_last(self):
        line = _item("t", 1.0, bench="jetstream3.slipstream")
        db = FakeDb([[], [(T0,)], [line]])
        assert spanner.refresh(db) is None
        (agg,) = [c for c in db.of("query") if "GROUP BY s.bot" in c[1]]
        assert "FORCE_INDEX" not in agg[1] and agg[2] is None
        # Both aggregate tables, in one write; the previous design's rows
        # are points at embedder 0 without an embedder hash.
        assert db.of("write") == [("write", ["benchmarks", "benchmarks_v2"])]
        assert db.of("upsert") == [
            ("upsert", "benchmarks", spanner.AGG_COLUMNS, [line]),
            ("upsert", "benchmarks_v2", spanner.AGG_V2_COLUMNS, [(*line, 0, None)]),
        ]
        assert db.calls[-1][0] == "execute"
        assert db.calls[-1][2] == [spanner.WATERMARK_KEY, T0.isoformat()]

    @pytest.mark.parametrize(
        "into, tables",
        [("benchmarks", ["benchmarks"]), ("benchmarks_v2", ["benchmarks_v2"])],
    )
    def test_aggregate_into_selects_the_tables(self, into, tables):
        line = _item("t", 1.0, bench="jetstream3.slipstream")
        db = FakeDb([[], [(T0,)], [line]])
        assert spanner.refresh(db, into=into) is None
        assert [u[1] for u in db.of("upsert")] == tables

    def test_unknown_aggregate_target_is_rejected_before_any_io(self):
        db = FakeDb()
        with pytest.raises(ValueError, match="aggregate_into must be one of"):
            spanner.refresh(db, into="all")
        assert db.calls == []

    def test_incremental_reads_changed_groups_through_indexes(self):
        last = datetime(2026, 9, 1, tzinfo=timezone.utc)
        earlier = datetime(2026, 9, 2, tzinfo=timezone.utc)
        changed = [
            ("b", "jetstream3.slipstream", 101, earlier),
            ("c", "x", 5, T0),
            ("b", "jetstream3.slipstream", 100, earlier),
        ]
        db = FakeDb([[(last.isoformat(),)], changed, [], []])
        assert spanner.refresh(db) is None
        queries = db.of("query")
        (keys,) = [c for c in queries if "MAX(imported_at)" in c[1]]
        assert f"FORCE_INDEX={spanner.IMPORTED_AT_INDEX}" in keys[1]
        assert keys[2] == [last]
        aggs = [c for c in queries if "GROUP BY s.bot" in c[1]]
        assert all(f"FORCE_INDEX={spanner.GROUP_INDEX}" in c[1] for c in aggs)
        assert all("IN UNNEST(%s)" in c[1] for c in aggs)
        assert [c[2] for c in aggs] == [
            ["b", "jetstream3.slipstream", [100, 101]],
            ["c", "x", [5]],
        ]
        # The cutoff is the newest row seen, not a scan of the whole table.
        assert db.calls[-1][2] == [spanner.WATERMARK_KEY, T0.isoformat()]

    def test_watermark_keeps_nanoseconds(self):
        # Spanner timestamps carry nanoseconds; a microsecond watermark would
        # sit just below the newest row and re-select it on every refresh.
        from google.api_core.datetime_helpers import DatetimeWithNanoseconds

        newest = DatetimeWithNanoseconds.from_rfc3339("2026-10-01T13:00:57.544872375Z")
        db = FakeDb(
            [[("2026-10-01T12:00:00.000000001Z",)], [("c", "x", 5, newest)], []]
        )
        assert spanner.refresh(db) is None
        (keys,) = [c for c in db.of("query") if "MAX(imported_at)" in c[1]]
        assert keys[2][0].nanosecond == 1
        assert db.calls[-1][2] == [
            spanner.WATERMARK_KEY,
            "2026-10-01T13:00:57.544872375Z",
        ]

    def test_legacy_isoformat_watermark_still_parses(self):
        db = FakeDb([[("2026-10-01T05:45:11.359679+00:00",)], []])
        assert spanner.refresh(db) == "no rows newer than the last refresh"
        (keys,) = [c for c in db.of("query") if "MAX(imported_at)" in c[1]]
        assert keys[2] == [
            datetime(2026, 10, 1, 5, 45, 11, 359679, tzinfo=timezone.utc)
        ]

    def test_sql_shape(self):
        db = FakeDb([[], [(T0,)], []])
        spanner.refresh(db)
        (agg,) = [c[1] for c in db.of("query") if "GROUP BY s.bot" in c[1]]
        assert "IF(s.test = 'Overall', 'Total', s.test) AS test" in agg
        assert "REGEXP_EXTRACT(s.variant" in agg
        assert (
            "s.test != 'Overall' AND s.metric IN "
            "('Total-Score', 'Total-Time', 'Score', '')" in agg
        )
        assert "IF(s.metric = 'Total-Time', 'Time', '') AS submetric" in agg
        assert "s.test = 'Overall' AND s.metric = 'Total-Score'" in agg
        assert "FROM benchmarks" not in agg

    def test_nan_becomes_null(self):
        row = _item("t", 1.0, bench="x")[:13] + (float("nan"),)
        db = FakeDb([[], [(T0,)], [row]])
        spanner.refresh(db)
        assert db.of("upsert")[0][3][0][-1] is None


class TestGeomeanTotals:
    def test_js2_group_without_total_gets_the_geomean(self):
        rows = [_item("a", 2.0), _item("b", 8.0)]
        (total,) = spanner._geomean_totals(rows)
        assert total[:6] == ("b", "jetstream2.slipstream", "Total", "", "v", 1)
        assert total[6:9] == (T0, "h", "slipstream")
        assert total[9] == pytest.approx(4.0)  # sqrt(2 * 8)
        assert total[10] == pytest.approx(2.0)  # sqrt(1 * 4)
        assert total[11] == pytest.approx(8.0)  # sqrt(4 * 16)
        assert total[12:] == (0.0, 3)

    def test_harness_total_wins(self):
        rows = [_item("a", 2.0), _item("b", 8.0), _item("Total", 5.0)]
        assert spanner._geomean_totals(rows) == []

    def test_each_embedder_is_its_own_group(self):
        # The same inner commit under two embedders (a roll boundary) gets a
        # Total per embedder, carrying that embedder's columns; a harness
        # Total under one of them does not cover the other.
        rows = [
            (*_item("a", 2.0), 0, None),
            (*_item("b", 8.0), 0, None),
            (*_item("a", 3.0), 7, "c0ffee"),
            (*_item("b", 27.0), 7, "c0ffee"),
            (*_item("Total", 1.0), 8, "f00d"),
            (*_item("a", 5.0), 8, "f00d"),
        ]
        totals = {t[14:]: t for t in spanner._geomean_totals(rows)}
        assert set(totals) == {(0, None), (7, "c0ffee")}
        assert totals[(0, None)][9] == pytest.approx(4.0)
        assert totals[(7, "c0ffee")][9] == pytest.approx(9.0)
        assert all(len(t) == len(spanner.AGG_V2_COLUMNS) for t in totals.values())

    def test_only_js2_and_positive_means(self):
        rows = [
            _item("a", 2.0, bench="jetstream3.slipstream"),
            _item("a", 0.0, commit=2),
            _item("a", 3.0, commit=3, variant="w"),
        ]
        (total,) = spanner._geomean_totals(rows)
        assert total[4:6] == ("w", 3) and total[9] == pytest.approx(3.0)

    def test_the_total_does_not_depend_on_line_item_order(self):
        # A left-to-right float sum of these logs lands on a different ulp
        # forward and reversed (checked on 3.11, whose sum() is naive); the
        # two staging designs return items in different orders, and both
        # may run on such a Python.
        means = [
            714.275, 1632.916, 1110.181, 1811.958, 1877.348, 197.054,
            39.997, 2512.489, 778.432, 703.376, 2986.937, 1411.055,
        ]  # fmt: skip
        rows = [_item(f"t{i}", m) for i, m in enumerate(means)]
        (forward,) = spanner._geomean_totals(rows)
        (backward,) = spanner._geomean_totals(rows[::-1])
        (shuffled,) = spanner._geomean_totals(rows[5:] + rows[:5])
        assert forward == backward == shuffled


class TestStagingGroups:
    def test_both_designs_in_one_write_call(self):
        db = FakeDb()
        n = spanner.stage_records(
            db, spanner.records_from_csv(_csv({}, {"run": "2"})), "b"
        )
        assert n == 2
        assert db.of("write") == [
            ("write", ["slipstream", "samples", "commits", "dirty_groups"])
        ]
        assert [len(c[3]) for c in db.of("upsert")] == [2, 2, 1, 1]
        assert db.of("upsert", "slipstream")[0][2] == spanner.IMPORT_COLUMNS
        assert db.of("upsert", "samples")[0][2] == spanner.SAMPLE_COLUMNS

    def test_embedded_record_keys_on_the_pair_and_skips_the_previous_design(self):
        chromium = "c" * 40
        records = spanner.records_from_csv(
            _csv(
                {},
                {
                    "engine": "chrome",
                    "flags": "chrome_default",
                    "commit_id": "109680",
                    "embedder_id": "1500123",
                    "embedder_hash": chromium,
                },
                {
                    "engine": "chrome",
                    "flags": "chrome_default",
                    "commit_id": "109680",
                    "embedder_id": "1500124",
                    "embedder_hash": "d" * 40,
                    "run": "2",
                },
            )
        )
        staged = spanner.staging_rows(records, "bot1")
        assert [s[:6] for s in staged.samples] == [
            ("bot1", "js3", "v8", "default", 0, 100),
            ("bot1", "js3", "chrome", "default", 1500123, 109680),
            ("bot1", "js3", "chrome", "default", 1500124, 109680),
        ]
        # Same inner commit under two chromium positions: two commit rows,
        # each with its own embedder hash; the inner hash is shared.
        assert [(c[0], c[1], c[2], c[3], c[6]) for c in staged.commits] == [
            ("v8", 0, 100, "abc", None),
            ("chrome", 1500123, 109680, "abc", chromium),
            ("chrome", 1500124, 109680, "abc", "d" * 40),
        ]
        assert [d[:6] for d in staged.dirty] == [s[:6] for s in staged.samples]
        # The previous-design table has no embedder coordinate (D1): it
        # receives the v8 row only, and both designs still go in one write.
        db = FakeDb()
        assert spanner.stage_records(db, records, "bot1") == 3
        assert [len(c[3]) for c in db.of("upsert")] == [1, 3, 3, 3]
        assert db.of("upsert", "slipstream")[0][3][0][6] == 100
        # An all-embedded push under write=legacy stages nothing at all.
        db = FakeDb()
        spanner.stage_records(db, records[1:], "bot1", write="legacy")
        assert db.of("upsert", "slipstream")[0][3] == []

    def test_write_modes_select_the_tables(self):
        records = spanner.records_from_csv(_csv({}))
        for write, tables in (
            ("legacy", ["slipstream"]),
            ("samples", ["samples", "commits", "dirty_groups"]),
        ):
            db = FakeDb()
            spanner.stage_records(db, records, "b", write=write)
            assert db.of("write") == [("write", tables)]
        # legacy rows need the database clock; samples carry the commit
        # timestamp sentinel and need no round trip.
        db = FakeDb()
        spanner.stage_records(db, records, "b", write="samples")
        assert db.of("query") == []
        with pytest.raises(ValueError, match="write must be one of"):
            spanner.stage_records(FakeDb(), records, "b", write="all")

    def test_nothing_is_written_for_no_records(self):
        db = FakeDb()
        assert spanner.stage_records(db, [], "b") == 0
        assert db.of("upsert") == [] and db.of("query") == []

    def test_a_record_that_does_not_map_writes_nothing(self):
        db = FakeDb()
        records = spanner.records_from_csv(_csv({}, {"flags": "default"}))
        with pytest.raises(ValueError, match="do not start with the engine"):
            spanner.stage_records(db, records, "b")
        assert db.of("write") == []


class TestWriteChunks:
    """SpannerDb.write without a connection: only the batching logic."""

    def _db(self):
        db = object.__new__(spanner.SpannerDb)
        commits = []

        class Batch:
            def __init__(self):
                self.parts = []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                commits.append(self.parts)

            def insert_or_update(self, table, columns, values):
                self.parts.append((table, len(values)))

        class Database:
            def batch(self):
                return Batch()

        db._database = Database()
        return db, commits

    def test_parallel_groups_share_commits_chunk_by_chunk(self, monkeypatch):
        monkeypatch.setattr(spanner, "_MUTATION_BUDGET", 100)
        db, commits = self._db()
        # slipstream costs 12*3, samples 12, commits 8, dirty 7: 63 per
        # record, under the floor of 100 records per chunk.
        rows = lambda n: [(i,) for i in range(n)]  # noqa: E731
        db.write(
            [
                ("slipstream", spanner.IMPORT_COLUMNS, rows(250)),
                ("samples", spanner.SAMPLE_COLUMNS, rows(250)),
                ("commits", spanner.COMMIT_COLUMNS, rows(3)),
                ("dirty_groups", spanner.DIRTY_COLUMNS, rows(3)),
            ]
        )
        assert commits == [
            [
                ("slipstream", 100),
                ("samples", 100),
                ("commits", 3),
                ("dirty_groups", 3),
            ],
            [("slipstream", 100), ("samples", 100)],
            [("slipstream", 50), ("samples", 50)],
        ]

    def test_chunk_size_follows_the_mutation_cost(self):
        db, commits = self._db()
        # benchmarks: 14 columns x (1 + 2 indexes) = 42 per row -> 952 rows.
        db.write([("benchmarks", spanner.AGG_COLUMNS, [(i,) for i in range(1000)])])
        assert [p[0][1] for p in commits] == [952, 48]
        # Empty groups are dropped; nothing at all means no commit.
        db.write([("x", ["a"], [])])
        assert len(commits) == 2


class TestImportLedger:
    def test_both_records_open_before_and_close_after_the_write(self):
        db = FakeDb()
        handle = spanner.begin_import(db, "bot", "a1", "d1")
        assert handle == spanner.ImportHandle(
            "a1", "slipstream_incomplete_import:attempt:a1", True
        )
        spanner.finish_import(db, handle, 7)
        ex = db.of("execute")
        assert [e[1].split()[0:3] for e in ex] == [
            ["INSERT", "OR", "UPDATE"],
            ["INSERT", "OR", "UPDATE"],
            ["DELETE", "FROM", "meta"],
            ["UPDATE", "imports", "SET"],
        ]
        assert ex[0][2] == [handle.marker, "3:bot:d1"]
        assert "PENDING_COMMIT_TIMESTAMP(), NULL, NULL" in ex[1][1]
        assert ex[1][2] == ["a1", "bot", "d1"]
        assert "finished_at = PENDING_COMMIT_TIMESTAMP()" in ex[3][1]
        assert ex[3][2] == [7, "a1"]

    def test_each_mode_keeps_only_its_own_record(self):
        db = FakeDb()
        handle = spanner.begin_import(db, "bot", "a1", "d1", write="legacy")
        spanner.finish_import(db, handle)
        assert handle.ledger is False and handle.marker is not None
        assert all("imports" not in e[1] for e in db.of("execute"))
        db = FakeDb()
        handle = spanner.begin_import(db, "bot", "a1", "d1", write="samples")
        spanner.finish_import(db, handle)
        assert handle.ledger is True and handle.marker is None
        assert all("meta" not in e[1] for e in db.of("execute"))

    def test_a_bare_key_clears_a_legacy_marker_only(self):
        db = FakeDb()
        spanner.finish_import(db, "slipstream_incomplete_import:bot")
        assert db.of("execute") == [
            (
                "execute",
                "DELETE FROM meta WHERE key = %s",
                ["slipstream_incomplete_import:bot"],
            )
        ]


class TestRebuild:
    def _tables(self, db):
        return [c[1].split()[2] for c in db.of("pdml")]

    def test_both_wipes_every_staging_table_but_commits(self):
        db = FakeDb()
        spanner.rebuild_bot(db, "bot")
        assert self._tables(db) == [
            "slipstream",
            "samples",
            "dirty_groups",
            "benchmarks",
            "benchmarks_v2",
            "meta",
            "imports",
        ]
        assert all(c[2]["bot"] == "bot" for c in db.of("pdml") if "meta" not in c[1])
        (imports,) = [c for c in db.of("pdml") if "imports" in c[1]]
        assert "finished_at IS NULL" in imports[1]

    def test_legacy_and_samples_modes(self):
        db = FakeDb()
        spanner.rebuild_bot(db, "bot", write="legacy")
        assert self._tables(db) == ["slipstream", "benchmarks", "benchmarks_v2", "meta"]
        db = FakeDb()
        spanner.rebuild_bot(db, "bot", write="samples")
        assert self._tables(db) == [
            "samples",
            "dirty_groups",
            "benchmarks",
            "benchmarks_v2",
            "imports",
        ]

    def test_aggregate_targets(self):
        db = FakeDb()
        spanner.rebuild_bot(db, "bot", write="samples", into="benchmarks_v2")
        assert self._tables(db) == [
            "samples",
            "dirty_groups",
            "benchmarks_v2",
            "imports",
        ]
        db = FakeDb()
        spanner.rebuild_bot(db, "bot", write="samples", into="benchmarks")
        assert self._tables(db) == ["samples", "dirty_groups", "benchmarks", "imports"]
        with pytest.raises(ValueError, match="aggregate_into must be one of"):
            spanner.rebuild_bot(FakeDb(), "bot", into="v3")


def _dirty(
    bot="b", suite="js3", engine="v8", variant="default", embedder=0, commit=1, at=T0
):
    return (bot, suite, engine, variant, embedder, commit, at)


class TestRefreshSamples:
    def test_unknown_source_is_rejected_before_any_io(self):
        db = FakeDb()
        with pytest.raises(ValueError, match="aggregate_from must be one of"):
            spanner.refresh(db, source="both")
        assert db.calls == []

    def test_nothing_dirty(self):
        db = FakeDb([[]])
        assert (
            spanner.refresh(db, source="samples")
            == "no dirty groups since the last refresh"
        )
        assert db.of("upsert") == [] and db.of("execute") == []
        # The dirty read follows the open-import check and nothing else:
        # no index check, no watermark.
        assert [q[1] for q in db.of("query")] == [
            "SELECT attempt FROM imports WHERE finished_at IS NULL LIMIT 1",
            f"SELECT {', '.join(spanner.DIRTY_COLUMNS)} FROM dirty_groups",
        ]

    def test_open_import_blocks(self):
        db = FakeDb()
        db.query = lambda sql, params=None: (
            [("a9",)] if "finished_at IS NULL" in sql else []
        )
        out = spanner.refresh(db, source="samples")
        assert "incomplete" in out and "a9" in out

    def test_groups_outside_embedder_zero_stop_a_benchmarks_only_refresh(self):
        db = FakeDb([[_dirty(), _dirty(embedder=7, commit=2)]])
        out = spanner.refresh(db, source="samples", into="benchmarks")
        assert (
            "outside embedder 0" in out and "('b', 'js3', 'v8', 'default', 7, 2)" in out
        )
        assert db.of("upsert") == [] and db.of("execute") == []

    def test_an_embedded_group_lands_in_benchmarks_v2_only(self):
        def agg(commit, test, mean, ehash):
            return (commit, test, "", T0, "h", ehash, mean, mean, mean, 0.0, 1)

        db = FakeDb(
            [
                [_dirty(commit=5), _dirty(embedder=7, commit=5)],
                [agg(5, "x", 1.0, None)],  # embedder 0
                [agg(5, "x", 2.0, "c0ffee")],  # embedder 7, same commit
            ]
        )
        assert spanner.refresh(db, source="samples") is None
        aggs = [q[2] for q in db.of("query") if "FROM samples s" in q[1]]
        assert aggs == [
            ["b", "js3", "v8", "default", 0, [5]],
            ["b", "js3", "v8", "default", 7, [5]],
        ]
        v1, v2 = db.of("upsert")
        assert v1[1:3] == ("benchmarks", spanner.AGG_COLUMNS)
        assert [r[5] for r in v1[3]] == [5] and all(len(r) == 14 for r in v1[3])
        assert v2[1:3] == ("benchmarks_v2", spanner.AGG_V2_COLUMNS)
        # Two points for one commit, one per embedder, with the outer hash.
        assert [(r[5], r[14], r[15], r[9]) for r in v2[3]] == [
            (5, 0, None, 1.0),
            (5, 7, "c0ffee", 2.0),
        ]
        assert db.calls[-1][1].startswith("DELETE FROM dirty_groups")

    def test_one_query_per_prefix_then_upsert_then_delete_up_to_the_cutoff(self):
        later = datetime(2026, 9, 5, 12, 0, 1, tzinfo=timezone.utc)
        dirty = [
            _dirty(commit=3),
            _dirty(suite="js2", commit=1, at=later),
            _dirty(commit=1),
            _dirty(bot="a", variant="per_line_item", commit=9),
        ]

        def agg(commit, test, mean):
            return (commit, test, "", T0, "h", None, mean, mean, mean, 0.0, 1)

        db = FakeDb(
            [
                dirty,
                [agg(9, "x", 1.0)],  # a / js3 / v8 / per_line_item
                [agg(1, "x", 2.0), agg(1, "y", 8.0)],  # b / js2 (no Total)
                [agg(1, "Total", 5.0), agg(3, "x", 7.0)],  # b / js3
            ]
        )
        assert spanner.refresh(db, source="samples") is None
        aggs = [q for q in db.of("query") if "FROM samples s" in q[1]]
        assert [q[2] for q in aggs] == [
            ["a", "js3", "v8", "per_line_item", 0, [9]],
            ["b", "js2", "v8", "default", 0, [1]],
            ["b", "js3", "v8", "default", 0, [1, 3]],
        ]
        assert all("IN UNNEST(%s)" in q[1] for q in aggs)
        up, v2 = db.of("upsert")
        assert up[1:3] == ("benchmarks", spanner.AGG_COLUMNS)
        assert v2[1:3] == ("benchmarks_v2", spanner.AGG_V2_COLUMNS)
        # Same rows in both, benchmarks_v2's with the embedder columns.
        assert [r[:14] for r in v2[3]] == up[3]
        assert {r[14:] for r in v2[3]} == {(0, None)}
        rows = {r[:6]: r for r in up[3]}
        # Labels are the frontend ones, and the JS2 group got its geomean.
        assert set(rows) == {
            ("a", "jetstream3.slipstream", "x", "", "v8_per_line_item", 9),
            ("b", "jetstream2.slipstream", "x", "", "v8_default", 1),
            ("b", "jetstream2.slipstream", "y", "", "v8_default", 1),
            ("b", "jetstream2.slipstream", "Total", "", "v8_default", 1),
            ("b", "jetstream3.slipstream", "Total", "", "v8_default", 1),
            ("b", "jetstream3.slipstream", "x", "", "v8_default", 3),
        }
        assert rows[("a", "jetstream3.slipstream", "x", "", "v8_per_line_item", 9)][6:] == (
            T0, "h", "slipstream", 1.0, 1.0, 1.0, 0.0, 1,
        )  # fmt: skip
        assert rows[("b", "jetstream2.slipstream", "Total", "", "v8_default", 1)][
            9
        ] == (pytest.approx(4.0))
        # The delete comes last and is bounded by the newest dirtied_at read,
        # not by the key set, so a group dirtied again meanwhile survives.
        assert db.calls[-1] == (
            "execute",
            "DELETE FROM dirty_groups WHERE dirtied_at <= %s",
            [later],
        )
        assert db.calls.index(v2) < len(db.calls) - 1

    def test_sql_shape(self):
        db = FakeDb([[_dirty()], []])
        spanner.refresh(db, source="samples")
        (agg,) = [q[1] for q in db.of("query") if "FROM samples s" in q[1]]
        # Same filter and statistics as the previous design's query.
        assert "IF(s.test = 'Overall', 'Total', s.test) AS test" in agg
        assert (
            "s.test != 'Overall' AND s.metric IN "
            "('Total-Score', 'Total-Time', 'Score', '')" in agg
        )
        assert "IF(s.metric = 'Total-Time', 'Time', '') AS submetric" in agg
        assert "s.test = 'Overall' AND s.metric = 'Total-Score'" in agg
        assert "COALESCE(STDDEV_SAMP(s.value), 0.0), COUNT(*)" in agg
        assert "LEFT JOIN commits c" in agg
        assert "MIN(c.commit_time), MIN(c.git_hash), MIN(c.embedder_hash)" in agg
        assert (
            "GROUP BY s.commit_number, IF(s.test = 'Overall', 'Total', s.test)" in agg
        )
        assert "FROM benchmarks" not in agg and "FROM slipstream" not in agg

    def test_nan_becomes_null(self):
        db = FakeDb(
            [[_dirty()], [(1, "x", "", T0, "h", None, 1.0, 1.0, 1.0, float("nan"), 1)]]
        )
        spanner.refresh(db, source="samples")
        assert db.of("upsert")[0][3][0][12] is None

    def test_failed_upsert_leaves_the_dirty_rows(self):
        db = FakeDb([[_dirty()], [(1, "x", "", T0, "h", None, 1.0, 1.0, 1.0, 0.0, 1)]])
        db.write = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        with pytest.raises(RuntimeError):
            spanner.refresh(db, source="samples")
        assert db.of("execute") == []

    def test_whole_prefix_when_no_commits_are_given(self):
        db = FakeDb([[]])
        assert spanner.aggregate_samples(db, "b", "js3", "v8", "default", 0) == []
        (q,) = db.of("query")
        assert "UNNEST" not in q[1] and q[2] == ["b", "js3", "v8", "default", 0]


class TestPushCsv:
    def _db(self, extra=()):
        return FakeDb([ALL_TABLES, *extra])

    def test_stages_both_designs_then_aggregates_from_samples(self):
        db = self._db([[_dirty(bot="bot1", commit=100)], []])
        out = spanner.push_csv(db, "bot1", _csv({}, {"run": "2"}))
        assert out == "2 rows staged for bot1, aggregated"
        assert db.of("write") == [
            ("write", ["slipstream", "samples", "commits", "dirty_groups"])
        ]
        (up,) = db.of("upsert", "slipstream")
        assert up[2] == spanner.IMPORT_COLUMNS and len(up[3]) == 2
        assert len(db.of("upsert", "samples")[0][3]) == 2
        assert db.of("pdml") == []
        # The dirty groups were drained: that was the path aggregated.
        assert any("FROM samples s" in q[1] for q in db.of("query"))
        assert not any("imported_at" in q[1] for q in db.of("query"))
        assert db.calls[-1][1].startswith("DELETE FROM dirty_groups")

    def test_aggregates_from_legacy_when_told(self):
        db = self._db([[], [(T0,)], []])
        out = spanner.push_csv(db, "bot1", _csv({}), aggregate_from="legacy")
        assert out == "1 rows staged for bot1, aggregated"
        assert not any("dirty_groups" in q[1] for q in db.of("query"))
        # The legacy watermark was written: that was the path aggregated.
        assert db.calls[-1][2][0] == spanner.WATERMARK_KEY

    def test_incompatible_modes_are_rejected_before_any_io(self):
        db = self._db()
        with pytest.raises(ValueError, match="does not write"):
            spanner.push_csv(
                db, "b", _csv({}), write="legacy", aggregate_from="samples"
            )
        with pytest.raises(ValueError, match="does not write"):
            spanner.push_csv(
                db, "b", _csv({}), write="samples", aggregate_from="legacy"
            )
        assert db.calls == []

    def test_no_refresh(self):
        db = self._db()
        assert spanner.push_csv(db, "b", _csv({}), refresh_agg=False) == (
            "1 rows staged for b"
        )
        assert db.of("query")[1:] == [("query", "SELECT CURRENT_TIMESTAMP()", None)]

    def test_rebuild_wipes_before_staging(self):
        db = self._db([[], [(None,)]])
        out = spanner.push_csv(db, "b", _csv({}), rebuild=True)
        assert "(rebuild)" in out and "no dirty groups" in out
        kinds = [c[0] for c in db.calls]
        assert kinds.index("pdml") < kinds.index("write")
        assert db.of("pdml")[0][2] == {"bot": "b"}

    def test_a_csv_that_does_not_map_costs_a_rebuild_nothing(self):
        db = self._db()
        with pytest.raises(ValueError):
            spanner.push_csv(db, "b", _csv({"flags": "nope"}), rebuild=True)
        assert db.of("pdml") == [] and db.of("write") == [] and db.of("execute") == []


class _Api:
    def __init__(self, closed):
        self.transport = self
        self._closed = closed

    def close(self):
        self._closed.append(self)


class TestSpannerDbTeardown:
    """close() must release everything the connection opened, in order."""

    def _db(self, order, *, database_api=True, admin_api=True, boom=False):
        db = object.__new__(spanner.SpannerDb)

        class Con:
            def close(self_):
                order.append("con")

        class Database:
            def close(self_):
                order.append("database")
                if boom:
                    raise RuntimeError("session delete failed")

        class Client:
            pass

        db._con, db._database, db._client = Con(), Database(), Client()
        closed: list = []
        if database_api:
            db._database._spanner_api = _Api(closed)
        if admin_api:
            db._client._database_admin_api = _Api(closed)
        return db, closed

    def test_close_releases_channels_and_the_lock(self):
        order: list[str] = []
        db, closed = self._db(order)

        class Lock:
            def close(self_):
                order.append("lock")

        db._local_lock = Lock()
        db.close()
        assert len(closed) == 2
        # The session manager needs its channel, so it goes before the
        # transports; the lock is released last whatever happens.
        assert order == ["con", "database", "lock"]

    def test_uncreated_apis_are_not_built_just_to_close_them(self):
        db, closed = self._db([], database_api=False, admin_api=False)
        db.close()
        assert closed == []
        assert not hasattr(db._database, "_spanner_api")

    def test_a_failing_step_does_not_strand_the_rest(self):
        order: list[str] = []
        db, closed = self._db(order, boom=True)
        with pytest.raises(RuntimeError, match="session delete failed"):
            db.close()
        assert len(closed) == 2


class TestSpannerDbSessions:
    def test_multiplexed_sessions_are_disabled_before_connecting(self, monkeypatch):
        # A multiplexed session's maintenance thread makes Database.close()
        # block for up to ten minutes; it must be off before the first query.
        import google.cloud.spanner as gspanner
        import google.cloud.spanner_dbapi as dbapi

        for var in spanner._MULTIPLEXED_SESSION_ENV:
            monkeypatch.setenv(var, "true")
        seen = {}

        class Cursor:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def execute(self_, sql, params=None):
                seen.update(
                    {v: os.environ[v] for v in spanner._MULTIPLEXED_SESSION_ENV}
                )

            def fetchall(self_):
                return [(1,)]

        class Con:
            database = None

            def cursor(self_):
                return Cursor()

        monkeypatch.setattr(gspanner, "Client", lambda **kw: object())
        monkeypatch.setattr(dbapi, "connect", lambda *a, **kw: Con())
        spanner.SpannerDb("p", "i", "d")
        assert seen == {v: "false" for v in spanner._MULTIPLEXED_SESSION_ENV}
