# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from slipstream import spanner
from slipstream.config import PushTarget
from slipstream.push import _COLUMNS, deliver_once

T0 = datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc)


class FakeDb:
    """Records every call; answers queries from a scripted list."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.calls: list[tuple] = []

    def query(self, sql, params=None):
        self.calls.append(("query", " ".join(sql.split()), params))
        if "STARTS_WITH(key" in sql:
            return []
        if "CURRENT_TIMESTAMP" in sql:
            return [(T0,)]
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
        assert len(ddl) == len(spanner.schema_statements()) == 6
        assert any("trace_id INT64 NOT NULL AS (FARM_FINGERPRINT" in s for s in ddl)
        assert all(not s.startswith("--") for s in ddl)


class TestRefresh:
    def test_empty_staging(self):
        db = FakeDb([[(None,)]])
        assert spanner.refresh(db) == "staging is empty"
        assert db.of("upsert") == [] and db.of("execute") == []

    def test_watermark_up_to_date(self):
        db = FakeDb([[(T0,)], [(T0.isoformat(),)]])
        assert spanner.refresh(db) == "no rows newer than the last refresh"
        assert db.of("upsert") == []

    def test_first_run_is_full_and_sets_watermark_last(self):
        line = (
            "b",
            "jetstream3.slipstream",
            "t",
            "",
            "v8_default",
            1,
            T0,
            "h",
            "slipstream",
            1.0,
            1.0,
            1.0,
            0.0,
            3,
        )
        db = FakeDb([[(T0,)], [], [line], [], []])
        assert spanner.refresh(db) is None
        selects = [c for c in db.of("query")][3:]
        assert len(selects) == 3
        assert all("INNER JOIN" not in c[1] and c[2] == [] for c in selects)
        assert db.of("upsert") == [
            ("upsert", "benchmarks", spanner.AGG_COLUMNS, [line])
        ]
        assert db.calls[-1][0] == "execute"
        assert db.calls[-1][2] == [spanner.WATERMARK_KEY, T0.isoformat()]

    def test_incremental_joins_affected_groups(self):
        last = datetime(2026, 9, 1, tzinfo=timezone.utc)
        db = FakeDb([[(T0,)], [(last.isoformat(),)], [], [], []])
        assert spanner.refresh(db) is None
        selects = [c for c in db.of("query")][3:]
        assert all("INNER JOIN (SELECT DISTINCT" in c[1] for c in selects)
        assert all(c[2] == [last] for c in selects)
        assert "ON s.bot = _ak.bot" in selects[0][1]
        assert "ON a.bot = _ak.bot" in selects[1][1]

    def test_sql_shape(self):
        db = FakeDb([[(T0,)], [], [], [], []])
        spanner.refresh(db)
        s1, s2, s3 = [c[1] for c in db.of("query")][3:]
        assert "s.test != 'Overall'" in s1 and "REGEXP_EXTRACT(s.variant" in s1
        assert "a.benchmark = 'jetstream2.slipstream'" in s2 and "EXP(AVG(LN(" in s2
        assert "s.test = 'Overall' AND s.metric = 'Total-Score'" in s3

    def test_nan_becomes_null(self):
        db = FakeDb([[(T0,)], [], [("b",) * 13 + (float("nan"),)], [], []])
        spanner.refresh(db)
        assert db.of("upsert")[0][3][0][-1] is None

    def test_incomplete_import_blocks_refresh(self):
        class IncompleteDb(FakeDb):
            def query(self, sql, params=None):
                self.calls.append(("query", " ".join(sql.split()), params))
                return [("marker",)]

        db = IncompleteDb()
        assert spanner.refresh(db) == "an import is incomplete"
        assert not any("MAX(imported_at)" in c[1] for c in db.of("query"))


class TestPushCsv:
    def _db(self, extra=()):
        return FakeDb([[("slipstream",), ("benchmarks",), ("meta",)], *extra])

    def test_stages_then_aggregates(self):
        db = self._db([[(T0,)], [], [], [], []])
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
        db = self._db([[(None,)]])
        out = spanner.push_csv(db, "b", _csv({}), rebuild=True)
        assert "(rebuild)" in out and "staging is empty" in out
        kinds = [c[0] for c in db.calls]
        assert kinds.index("pdml") < kinds.index("upsert")
        assert db.of("pdml")[0][2] == {"bot": "b"}


class TestSpannerSession:
    def _connect(self, monkeypatch, db, seen):
        monkeypatch.setattr(
            spanner, "connect", lambda spec: (seen.__setitem__("spec", spec), db)[1]
        )

    def test_deliver_once_stages_and_closes(self, monkeypatch):
        seen = {}
        db = FakeDb([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, db, seen)
        t = PushTarget(spanner="p/i/d", refresh=False)
        out = deliver_once(t, "bot1", _csv({}, {"run": "2"}))
        assert seen["spec"] == "p/i/d"
        assert out == "2 rows staged for bot1"
        (up,) = db.of("upsert")
        assert up[1] == spanner.IMPORT_TABLE and len(up[3]) == 2
        assert db.calls[-1] == ("close",)

    def test_concurrent_local_session_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        first = spanner._acquire_local_lock("p/i/d")
        with pytest.raises(RuntimeError, match="another local delivery"):
            spanner._acquire_local_lock("/p/i/d/")
        first.close()

    def test_one_session_stages_many_and_aggregates_once(self, monkeypatch):
        from slipstream.push import open_target

        db = FakeDb(
            [[("slipstream",), ("benchmarks",), ("meta",)], [(T0,)], [], [], []]
        )
        self._connect(monkeypatch, db, {})
        session = open_target(PushTarget(spanner="p/i/d"), "bot1")
        session.deliver(_csv({}))
        session.deliver(_csv({"run": "2"}))
        assert session.close() == "2 rows staged for bot1, aggregated"
        assert len(db.of("upsert")) == 2
        # One schema check and one refresh for both deliveries.
        assert sum("INFORMATION_SCHEMA" in c[1] for c in db.of("query")) == 1
        assert sum("MAX(imported_at)" in c[1] for c in db.of("query")) == 1

    def test_rebuild_wipes_once_before_the_first_delivery(self, monkeypatch):
        from slipstream.push import open_target

        db = FakeDb([[("slipstream",), ("benchmarks",), ("meta",)], [(None,)]])
        self._connect(monkeypatch, db, {})
        session = open_target(PushTarget(spanner="p/i/d"), "b", rebuild=True)
        session.deliver(_csv({}))
        session.deliver(_csv({}))
        out = session.close()
        assert db.of("pdml") == [
            (
                "pdml",
                f"DELETE FROM {spanner.IMPORT_TABLE} WHERE bot = @bot",
                {"bot": "b"},
            )
        ]
        assert out.startswith("2 rows staged for b (rebuild)")

    def test_unparseable_csv_does_not_wipe(self, monkeypatch):
        """The wipe is only worth it if the replacement rows exist."""
        from slipstream.push import open_target

        db = FakeDb([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, db, {})
        session = open_target(PushTarget(spanner="p/i/d"), "b", rebuild=True)
        with pytest.raises(ValueError):
            session.deliver(_csv({"score": ""}))
        assert db.of("pdml") == [] and db.of("upsert") == []

    def test_failed_delivery_does_not_aggregate(self, monkeypatch):
        """Staging commits in chunks, so a failed CSV can be half in. Its
        groups must not be aggregated over an incomplete run set."""
        from slipstream.push import open_target

        class HalfStaged(FakeDb):
            def upsert(self, table, columns, rows):
                super().upsert(table, columns, rows)
                raise RuntimeError("DEADLINE_EXCEEDED")

        db = HalfStaged([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, db, {})
        session = open_target(PushTarget(spanner="p/i/d"), "b")
        with pytest.raises(RuntimeError):
            session.deliver(_csv({}))
        assert session.close() is None
        assert not any("MAX(imported_at)" in c[1] for c in db.of("query"))
        assert any(
            c[0] == "execute" and "INSERT OR UPDATE INTO meta" in c[1] for c in db.calls
        )
        assert not any(
            c[0] == "execute" and "DELETE FROM meta" in c[1] for c in db.calls
        )
        assert db.calls[-1] == ("close",)

    def test_later_session_cannot_refresh_a_partial_import(self, monkeypatch):
        from slipstream.push import open_target

        markers = set()

        class SharedDb(FakeDb):
            def query(self, sql, params=None):
                self.calls.append(("query", " ".join(sql.split()), params))
                if "STARTS_WITH(key" in sql:
                    return [(next(iter(markers)),)] if markers else []
                if "CURRENT_TIMESTAMP" in sql:
                    return [(T0,)]
                return self.answers.pop(0) if self.answers else []

            def execute(self, sql, params=None):
                super().execute(sql, params)
                if "INSERT OR UPDATE INTO meta" in sql and params[1] == "staging":
                    markers.add(params[0])
                elif "DELETE FROM meta" in sql:
                    markers.remove(params[0])

        class HalfStaged(SharedDb):
            def upsert(self, table, columns, rows):
                super().upsert(table, columns, rows)
                raise RuntimeError("DEADLINE_EXCEEDED")

        failed = HalfStaged([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, failed, {})
        first = open_target(PushTarget(spanner="p/i/d"), "a")
        with pytest.raises(RuntimeError):
            first.deliver(_csv({}))
        first.close()

        healthy = SharedDb([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, healthy, {})
        second = open_target(PushTarget(spanner="p/i/d"), "b")
        second.deliver(_csv({"run": "2"}))
        assert second.close() == (
            "1 rows staged for b, aggregation skipped (an import is incomplete)"
        )
        assert len(markers) == 1
        assert not any("MAX(imported_at)" in c[1] for c in healthy.of("query"))

        recovered = SharedDb([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, recovered, {})
        retry = open_target(PushTarget(spanner="p/i/d", refresh=False), "a")
        retry.deliver(_csv({"run": "3"}))
        retry.close()
        assert markers == set()

    def test_failed_delivery_still_closes_the_connection(self, monkeypatch):
        class Boom(FakeDb):
            def upsert(self, table, columns, rows):
                raise RuntimeError("DEADLINE_EXCEEDED")

        db = Boom([[("slipstream",), ("benchmarks",), ("meta",)]])
        self._connect(monkeypatch, db, {})
        with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
            deliver_once(PushTarget(spanner="p/i/d"), "b", _csv({}))
        assert db.calls[-1] == ("close",)


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
        db.close()
        assert len(closed) == 2
