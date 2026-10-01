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
        if "STARTS_WITH(key" in sql:
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

    def partitioned_dml(self, sql, params):
        self.calls.append(("pdml", sql, params))
        return 0

    def close(self):
        self.calls.append(("close",))

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


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
        db = FakeDb([[("slipstream",), ("benchmarks",), ("meta",)]])
        spanner.ensure_schema(db)
        assert db.of("execute") == []

        db = FakeDb([[("slipstream",)]])
        spanner.ensure_schema(db)
        ddl = [c[1] for c in db.of("execute")]
        assert len(ddl) == len(spanner.schema_statements()) == 7
        assert any("trace_id INT64 NOT NULL AS (FARM_FINGERPRINT" in s for s in ddl)
        assert all(not s.startswith("--") for s in ddl)

    def test_declared_indexes(self):
        idx = spanner.declared_indexes()
        assert set(idx) == {
            spanner.IMPORTED_AT_INDEX,
            spanner.GROUP_INDEX,
            "benchmarks_filter_idx",
            "benchmarks_trace_idx",
        }
        assert "STORING (commit_time, git_hash, val)" in idx[spanner.GROUP_INDEX]


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
        assert db.of("upsert") == [
            ("upsert", "benchmarks", spanner.AGG_COLUMNS, [line])
        ]
        assert db.calls[-1][0] == "execute"
        assert db.calls[-1][2] == [spanner.WATERMARK_KEY, T0.isoformat()]

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
        assert "s.test != 'Overall' AND s.metric IN ('Total-Score', 'Score', '')" in agg
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

    def test_only_js2_and_positive_means(self):
        rows = [
            _item("a", 2.0, bench="jetstream3.slipstream"),
            _item("a", 0.0, commit=2),
            _item("a", 3.0, commit=3, variant="w"),
        ]
        (total,) = spanner._geomean_totals(rows)
        assert total[4:6] == ("w", 3) and total[9] == pytest.approx(3.0)


class TestPushCsv:
    def _db(self, extra=()):
        return FakeDb([[("slipstream",), ("benchmarks",), ("meta",)], *extra])

    def test_stages_then_aggregates(self):
        db = self._db([[], [(T0,)], []])
        out = spanner.push_csv(db, "bot1", _csv({}, {"run": "2"}))
        assert out == "2 rows staged for bot1, aggregated"
        (up,) = db.of("upsert")
        assert up[1] == "slipstream" and up[2] == spanner.IMPORT_COLUMNS
        assert len(up[3]) == 2 and db.of("pdml") == []

    def test_no_refresh(self):
        db = self._db()
        assert spanner.push_csv(db, "b", _csv({}), refresh_agg=False) == (
            "1 rows staged for b"
        )
        assert db.of("query")[1:] == [("query", "SELECT CURRENT_TIMESTAMP()", None)]

    def test_rebuild_wipes_before_staging(self):
        db = self._db([[], [(None,)]])
        out = spanner.push_csv(db, "b", _csv({}), rebuild=True)
        assert "(rebuild)" in out and "staging is empty" in out
        kinds = [c[0] for c in db.calls]
        assert kinds.index("pdml") < kinds.index("upsert")
        assert db.of("pdml")[0][2] == {"bot": "b"}


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
