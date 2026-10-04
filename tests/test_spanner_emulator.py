# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Integration test against the Cloud Spanner emulator.

Runs only when SPANNER_EMULATOR_HOST is set, e.g.
    gcloud emulators spanner start --host-port=localhost:9010
    SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/test_spanner_emulator.py
"""

from __future__ import annotations

import os
import uuid

import pytest

from slipstream import spanner
from tests.test_spanner import _csv

pytestmark = pytest.mark.skipif(
    not os.environ.get("SPANNER_EMULATOR_HOST"), reason="no Spanner emulator"
)

PROJECT = "test-project"


@pytest.fixture(scope="module")
def instance():
    from google.cloud import spanner as gspanner

    client = gspanner.Client(project=PROJECT)
    inst = client.instance("test-instance", configuration_name="emulator-config")
    if not inst.exists():
        inst.create().result(120)
    return inst


def _fresh_db(instance, ddl):
    name = "db" + uuid.uuid4().hex[:8]
    instance.database(name, ddl_statements=ddl).create().result(120)
    db = spanner.SpannerDb(PROJECT, instance.instance_id, name)
    return db


@pytest.fixture
def db(instance):
    db = _fresh_db(instance, spanner.schema_statements())
    yield db
    db.close()


def _agg(db):
    rows = db.query(
        "SELECT test, variant, mean, min, max, stdev, count, trace_id, source"
        f" FROM {spanner.AGG_TABLE} ORDER BY test, variant"
    )
    return {(r[0], r[1]): r[2:] for r in rows}


def test_ensure_schema_creates_tables_from_bundled_ddl(instance):
    db = _fresh_db(instance, [])
    try:
        spanner.ensure_schema(db)
        have = {
            r[0]
            for r in db.query(
                "SELECT table_name FROM INFORMATION_SCHEMA.TABLES WHERE table_schema = ''"
            )
        }
        assert have == {"slipstream", "benchmarks", "meta", *spanner.STAGING_TABLES}
        # Second call finds everything and runs no DDL.
        spanner.ensure_schema(db)
    finally:
        db.close()


def test_push_aggregate_and_rebuild(db):
    js3 = _csv(
        *[
            {"benchmark": "bench-a", "run": str(r), "score": s}
            for r, s in enumerate(("10", "12", "14"), 1)
        ],
        {"benchmark": "bench-b", "score": "5"},
        {"benchmark": "Overall", "score": "7"},
        {"benchmark": "bench-a", "flags": "v8_turbolev_future", "score": "20"},
    )
    out = spanner.push_csv(db, "bot1", js3)
    assert out == "6 rows staged for bot1, aggregated"

    agg = _agg(db)
    mean, mn, mx, stdev, count, trace_id, source = agg[("bench-a", "v8_default")]
    assert (mean, mn, mx, count, source) == (12.0, 10.0, 14.0, 3, "slipstream")
    assert stdev == pytest.approx(2.0)
    assert trace_id is not None
    assert agg[("bench-a", "v8_turbolev_future")][0] == 20.0
    # Overall becomes test 'Total' for JS3.
    assert agg[("Total", "v8_default")][0] == 7.0
    assert ("Overall", "v8_default") not in agg
    assert db.query(
        f"SELECT value FROM {spanner.META_TABLE} WHERE key = %s",
        [spanner.WATERMARK_KEY],
    )

    # Replaying the same CSV converges: same aggregates, no duplicates.
    spanner.push_csv(db, "bot1", js3)
    assert _agg(db)[("bench-a", "v8_default")][:5] == (
        12.0,
        10.0,
        14.0,
        pytest.approx(2.0),
        3,
    )
    (n,) = db.query(f"SELECT COUNT(*) FROM {spanner.IMPORT_TABLE}")[0]
    assert n == 6

    # A later push touching one group re-aggregates only that group's full run set.
    more = _csv({"benchmark": "bench-a", "run": "4", "score": "16"})
    assert spanner.push_csv(db, "bot1", more).endswith("aggregated")
    assert _agg(db)[("bench-a", "v8_default")][:5] == (
        13.0,
        10.0,
        16.0,
        pytest.approx(2.581988897),
        4,
    )

    # JS2 gets its Total as the geomean of line items.
    js2 = _csv(
        {"suite": "js2", "commit_id": "200", "benchmark": "Air", "score": "4"},
        {"suite": "js2", "commit_id": "200", "benchmark": "Basic", "score": "9"},
    )
    spanner.push_csv(db, "bot1", js2)
    rows = db.query(
        f"SELECT mean FROM {spanner.AGG_TABLE} WHERE benchmark = 'jetstream2.slipstream' AND test = 'Total'"
    )
    assert rows[0][0] == pytest.approx(6.0)

    # Nothing new: aggregation reports the skip instead of claiming work.
    assert (
        spanner.push_csv(db, "bot1", _csv())
        == "0 rows staged for bot1, aggregation skipped (no rows newer than the last refresh)"
    )

    # Rebuild wipes this bot's staging rows before re-staging.
    out = spanner.push_csv(db, "bot1", js3, rebuild=True)
    assert out.startswith("6 rows staged for bot1 (rebuild)")
    (n,) = db.query(f"SELECT COUNT(*) FROM {spanner.IMPORT_TABLE}")[0]
    assert n == 6


def test_bots_are_isolated(db):
    spanner.push_csv(db, "bot1", _csv({"score": "1"}))
    spanner.push_csv(db, "bot2", _csv({"score": "3"}))
    rows = dict(
        db.query(f"SELECT bot, mean FROM {spanner.AGG_TABLE} WHERE test = 'bench-a'")
    )
    assert rows == {"bot1": 1.0, "bot2": 3.0}
    spanner.wipe_bot(db, "bot1")
    assert db.query(f"SELECT DISTINCT bot FROM {spanner.IMPORT_TABLE}") == [("bot2",)]


def test_delivery_cycle_stages_files_and_multiple_bots_then_refreshes_once(
    db, tmp_path, monkeypatch
):
    from slipstream.config import PushTarget, RelaySource
    from slipstream.delivery import Coordinator
    from slipstream.delivery_sources import SshSpoolSource
    from slipstream.delivery_targets import SpannerSession
    from slipstream.relay import read_cursor
    from slipstream.config import DeliveryConfig
    from tests.test_delivery import MemorySpool

    class KeepOpen:
        def __getattr__(self, name):
            return getattr(db, name)

        def close(self):
            pass

    monkeypatch.setattr(spanner, "connect", lambda spec, **kw: KeepOpen())
    sources = [
        SshSpoolSource(
            RelaySource(bot, "/spool", bot, tmp_path / "relay"),
            settings=DeliveryConfig(),
            spool=MemorySpool(
                {
                    1: _csv({"run": "1", "score": "10"}),
                    2: _csv({"run": "2", "score": "20"}),
                }
            ),
        )
        for bot in ("box1", "box2")
    ]
    refreshes = []
    real_refresh = spanner.refresh
    monkeypatch.setattr(
        spanner,
        "refresh",
        lambda d, **kw: (refreshes.append(d), real_refresh(d, **kw))[1],
    )
    c = Coordinator(
        sources,
        [PushTarget(spanner="p/i/d")],
        tmp_path / "delivery",
        session_factory=lambda t, **kw: SpannerSession(t),
    )
    with c.ownership():
        assert c.cycle() == 4
    assert not c.errors
    assert len(refreshes) == 1
    assert all(read_cursor(src.path) == 2 for src in sources)
    rows = db.query(
        "SELECT bot, mean, count FROM benchmarks WHERE test='bench-a' ORDER BY bot"
    )
    assert rows == [("box1", 15.0, 2), ("box2", 15.0, 2)]


def test_exact_partial_import_blocks_other_payload_until_persisted_retry(
    db, tmp_path, monkeypatch
):
    from slipstream.config import PushTarget
    from slipstream.delivery import Coordinator
    from slipstream.delivery_targets import SpannerSession
    from tests.test_delivery import remote

    failed = [True]

    class FaultDb:
        def __getattr__(self, name):
            return getattr(db, name)

        def close(self):
            pass

        def write(self, groups):
            # As if the process died after the first of several chunks:
            # every table has the first record's rows, none has the rest.
            if failed[0]:
                db.write([(t, c, r[:1]) for t, c, r in groups])
                failed[0] = False
                raise RuntimeError("crash after first chunk")
            db.write(groups)

    monkeypatch.setattr(spanner, "connect", lambda spec, **kw: FaultDb())
    original = remote(
        tmp_path, "box1", {1: _csv({"score": "10"}, {"run": "2", "score": "20"})}
    )
    target = PushTarget(spanner="p/i/d")

    def make(sources):
        return Coordinator(
            sources,
            [target],
            tmp_path / "delivery",
            session_factory=lambda t, **kw: SpannerSession(t),
        )

    c = make([original])
    with c.ownership():
        assert c.cycle() == 0
    assert original.pending()
    assert not _agg(db)
    # A different unit/bot completes, but cannot clear the original marker.
    other = remote(tmp_path, "box2", {1: _csv({"score": "30"})})
    c = make([other])
    with c.ownership():
        assert c.cycle() == 1
    assert not _agg(db)
    assert db.query(
        "SELECT key FROM meta WHERE STARTS_WITH(key, %s)", [spanner.INCOMPLETE_PREFIX]
    )
    # New coordinator reconstructs the exact retained source and attempt.
    original = remote(
        tmp_path, "box1", {1: _csv({"score": "10"}, {"run": "2", "score": "20"})}
    )
    c = make([original])
    with c.ownership():
        assert c.cycle() == 2
    assert original.pending() is None
    assert db.query("SELECT bot, mean, count FROM benchmarks ORDER BY bot") == [
        ("box1", 15.0, 2),
        ("box2", 30.0, 1),
    ]


def test_failed_aggregate_write_retries_on_idle_without_reupload(
    db, tmp_path, monkeypatch
):
    from slipstream.config import PushTarget
    from slipstream.delivery import Coordinator
    from slipstream.delivery_targets import SpannerSession
    from tests.test_delivery import remote

    failed = [True]

    class FaultDb:
        def __getattr__(self, name):
            return getattr(db, name)

        def close(self):
            pass

        def upsert(self, table, columns, rows):
            db.upsert(table, columns, rows)
            if table == spanner.AGG_TABLE and failed[0]:
                failed[0] = False
                raise RuntimeError("aggregate failure before watermark")

    monkeypatch.setattr(spanner, "connect", lambda spec, **kw: FaultDb())
    source = remote(
        tmp_path, "box1", {1: _csv({"score": "10"}, {"run": "2", "score": "20"})}
    )

    def make():
        return Coordinator(
            [source],
            [PushTarget(spanner="p/i/d")],
            tmp_path / "delivery",
            session_factory=lambda t, **kw: SpannerSession(t),
        )

    c = make()
    with c.ownership():
        assert c.cycle() == 2
    assert c.errors and source.acknowledged(1)
    assert not db.query("SELECT value FROM meta WHERE key=%s", [spanner.WATERMARK_KEY])
    c = make()
    with c.ownership():
        assert c.cycle() == 0
    assert not c.errors and source.spool.fetched == [1]
    assert _agg(db)[("bench-a", "v8_default")][0] == 15.0
    assert db.query("SELECT value FROM meta WHERE key=%s", [spanner.WATERMARK_KEY])


# --- The current staging design (samples / commits / dirty_groups / imports) ---


def _benchmarks(db):
    """Every benchmarks row keyed by its primary key, all columns."""
    rows = db.query(
        f"SELECT {', '.join(spanner.AGG_COLUMNS)}, trace_id FROM {spanner.AGG_TABLE}"
    )
    out = {r[:6]: r[6:] for r in rows}
    assert len(out) == len(rows)
    return out


def _count(db, table, where="", params=None):
    return db.query(f"SELECT COUNT(*) FROM {table} {where}", params)[0][0]


# A spread of what the live data has: two bots, two engines and three
# variants, both suites, a JS3 harness total, a JS2 group without one, a
# zero bench timestamp, several runs, and metrics that are not aggregated.
def _varied_csvs():
    bot1 = _csv(
        *[
            {"benchmark": "bench-a", "run": str(r), "score": s}
            for r, s in enumerate(("10", "12", "14"), 1)
        ],
        {"benchmark": "bench-b", "score": "5", "timestamp": "1700000500"},
        {"benchmark": "bench-b", "metric": "Worst-Score", "score": "99"},
        {"benchmark": "Overall", "score": "7"},
        {"benchmark": "Overall", "metric": "Average-Score", "score": "8"},
        {"benchmark": "bench-a", "flags": "v8_turbolev_future", "score": "20"},
        {"suite": "js2", "commit_id": "200", "benchmark": "Air", "score": "4"},
        {"suite": "js2", "commit_id": "200", "benchmark": "Basic", "score": "9"},
        {"suite": "js2", "commit_id": "201", "benchmark": "Air", "score": "3"},
        {"suite": "js2", "commit_id": "201", "benchmark": "Overall", "score": "11"},
    )
    bot2 = _csv(
        {"engine": "jsc", "flags": "jsc_default", "commit_id": "100", "score": "30"},
        {"engine": "jsc", "flags": "jsc_default", "commit_id": "100", "run": "2", "score": "40"},
        {"engine": "jsc", "flags": "jsc_per_line_item", "commit_id": "100", "score": "31"},
        {"engine": "jsc", "flags": "jsc_default", "commit_id": "100", "benchmark": "Overall", "score": "35"},
        {"benchmark": "bench-a", "commit_id": "100", "score": "11"},
    )  # fmt: skip
    return bot1, bot2


def test_both_paths_aggregate_identically(db):
    """The guarantee of the migration: same benchmarks rows, trace_id included.

    Pushes write both designs and aggregate from the previous one; the
    dirty groups are left in place (the legacy refresh does not drain
    them), so emptying benchmarks and refreshing from samples recomputes
    exactly the same set of groups from the other tables.
    """
    bot1, bot2 = _varied_csvs()
    assert spanner.push_csv(db, "bot1", bot1).endswith("aggregated")
    assert spanner.push_csv(db, "bot2", bot2).endswith("aggregated")
    legacy = _benchmarks(db)
    # bot1: js3 bench-a, bench-b, Total (+ turbolev bench-a), js2 200 Air,
    # Basic, geomean Total, js2 201 Air, Total; bot2: jsc bench-a, Total,
    # per_line_item bench-a, v8 bench-a.
    assert len(legacy) == 13, sorted(legacy)
    # Sanity: both totals forms are there.
    assert (
        legacy[("bot1", "jetstream3.slipstream", "Total", "", "v8_default", 100)][3]
        == 7.0
    )
    geo = legacy[("bot1", "jetstream2.slipstream", "Total", "", "v8_default", 200)]
    assert geo[3] == pytest.approx(6.0)
    assert (
        legacy[("bot1", "jetstream2.slipstream", "Total", "", "v8_default", 201)][3]
        == 11.0
    )
    assert (
        "bot2",
        "jetstream3.slipstream",
        "bench-a",
        "",
        "jsc_default",
        100,
    ) in legacy

    # Both designs hold the same number of values.
    assert _count(db, spanner.SAMPLES_TABLE) == _count(db, spanner.IMPORT_TABLE) == 17
    assert _count(db, spanner.DIRTY_TABLE) == 7
    assert db.query(
        f"SELECT engine, commit_number, git_hash, title FROM {spanner.COMMITS_TABLE} ORDER BY 1, 2"
    ) == [
        ("jsc", 100, "abc", "t"),
        ("v8", 100, "abc", "t"),
        ("v8", 200, "abc", "t"),
        ("v8", 201, "abc", "t"),
    ]

    db.partitioned_dml(f"DELETE FROM {spanner.AGG_TABLE} WHERE TRUE", {})
    assert not _benchmarks(db)
    assert spanner.refresh(db, source="samples") is None
    fresh = _benchmarks(db)
    assert fresh == legacy
    assert _count(db, spanner.DIRTY_TABLE) == 0
    assert (
        spanner.refresh(db, source="samples")
        == "no dirty groups since the last refresh"
    )

    # And a later push re-aggregates the touched group over its full run
    # set on either path.
    more = _csv({"benchmark": "bench-a", "run": "4", "score": "16"})
    assert spanner.push_csv(db, "bot1", more, aggregate_from="samples").endswith(
        "aggregated"
    )
    key = ("bot1", "jetstream3.slipstream", "bench-a", "", "v8_default", 100)
    assert _benchmarks(db)[key][3:8] == (
        13.0,
        10.0,
        16.0,
        pytest.approx(2.581988897),
        4,
    )
    assert spanner.push_csv(db, "bot1", more) == ("1 rows staged for bot1, aggregated")
    assert _benchmarks(db)[key][3:8] == (
        13.0,
        10.0,
        16.0,
        pytest.approx(2.581988897),
        4,
    )


def test_samples_only_mode_writes_no_legacy_rows(db):
    bot1, _ = _varied_csvs()
    out = spanner.push_csv(db, "bot1", bot1, write="samples", aggregate_from="samples")
    assert out == "12 rows staged for bot1, aggregated"
    assert _count(db, spanner.IMPORT_TABLE) == 0
    assert _count(db, spanner.META_TABLE) == 0
    assert _count(db, spanner.SAMPLES_TABLE) == 12
    assert len(_benchmarks(db)) == 9
    # The ledger closed, with the row count.
    assert db.query(
        f"SELECT row_count, finished_at IS NOT NULL FROM {spanner.IMPORTS_TABLE}"
    ) == [(12, True)]
    # Timestamps: imported_at is a real commit timestamp, measured_at NULL
    # for a zero bench timestamp and set otherwise.
    (imported,) = db.query(f"SELECT MIN(imported_at) FROM {spanner.SAMPLES_TABLE}")[0]
    assert imported.year >= 2026
    assert db.query(
        f"SELECT COUNT(measured_at), COUNT(*) FROM {spanner.SAMPLES_TABLE}"
    ) == [(1, 12)]


def test_samples_refresh_deletes_only_the_dirty_rows_it_read(db, monkeypatch):
    """A group dirtied by a push that lands mid-refresh stays dirty."""
    spanner.push_csv(db, "bot1", _csv({"score": "10"}), aggregate_from="samples")
    late = _csv({"score": "30", "run": "2"}, {"commit_id": "101", "score": "50"})
    real_query = db.query
    slipped = []

    def query(sql, params=None):
        rows = real_query(sql, params)
        if f"FROM {spanner.DIRTY_TABLE}" in sql and not slipped:
            # Another push commits after the dirty read and before the
            # delete: it re-dirties (bot1, 100) and dirties (bot1, 101).
            slipped.append(
                spanner.stage_records(db, spanner.records_from_csv(late), "bot1")
            )
        return rows

    monkeypatch.setattr(db, "query", query)
    spanner.stage_records(
        db, spanner.records_from_csv(_csv({"score": "20", "run": "3"})), "bot1"
    )
    assert spanner.refresh(db, source="samples") is None
    assert slipped == [2]
    # Both groups the late push touched are still dirty. (Each statement is
    # its own strong read, so the aggregate query already saw the late
    # sample; what matters is that nothing marks it as aggregated.)
    assert sorted(
        db.query(f"SELECT commit_number FROM {spanner.DIRTY_TABLE} ORDER BY 1")
    ) == [(100,), (101,)]
    key = ("bot1", "jetstream3.slipstream", "bench-a", "", "v8_default", 100)
    assert key in _benchmarks(db) and key[:5] + (101,) not in _benchmarks(db)
    monkeypatch.setattr(db, "query", real_query)
    assert spanner.refresh(db, source="samples") is None
    assert _count(db, spanner.DIRTY_TABLE) == 0
    agg = _benchmarks(db)
    assert agg[key][3:8] == (20.0, 10.0, 30.0, 10.0, 3)
    assert agg[key[:5] + (101,)][3] == 50.0


def test_open_import_blocks_samples_refresh_until_the_attempt_finishes(
    db, tmp_path, monkeypatch
):
    from slipstream.config import PushTarget
    from slipstream.delivery import Coordinator
    from slipstream.delivery_targets import SpannerSession
    from tests.test_delivery import remote

    failed = [True]

    class FaultDb:
        def __getattr__(self, name):
            return getattr(db, name)

        def close(self):
            pass

        def write(self, groups):
            if failed[0]:
                db.write([(t, c, r[:1]) for t, c, r in groups])
                failed[0] = False
                raise RuntimeError("crash after first chunk")
            db.write(groups)

    monkeypatch.setattr(spanner, "connect", lambda spec, **kw: FaultDb())
    target = PushTarget(spanner="p/i/d", write="both", aggregate_from="samples")

    def make(sources):
        return Coordinator(
            sources,
            [target],
            tmp_path / "delivery",
            session_factory=lambda t, **kw: SpannerSession(t),
        )

    payload = _csv({"score": "10"}, {"run": "2", "score": "20"})
    original = remote(tmp_path, "box1", {1: payload})
    c = make([original])
    with c.ownership():
        assert c.cycle() == 0
    assert original.pending()
    # One imports row, open; one sample made it, and its group is dirty.
    assert db.query(
        f"SELECT bot, finished_at IS NULL FROM {spanner.IMPORTS_TABLE}"
    ) == [("box1", True)]
    assert _count(db, spanner.SAMPLES_TABLE) == 1
    assert _count(db, spanner.DIRTY_TABLE) == 1
    assert not _benchmarks(db)
    # Another bot stages fine, but refresh stays blocked by the open row.
    other = remote(tmp_path, "box2", {1: _csv({"score": "30"})})
    c = make([other])
    with c.ownership():
        assert c.cycle() == 1
    assert not _benchmarks(db)
    assert _count(db, spanner.DIRTY_TABLE) == 2
    # The retried attempt closes its row and refresh proceeds.
    original = remote(tmp_path, "box1", {1: payload})
    c = make([original])
    with c.ownership():
        assert c.cycle() == 2
    assert original.pending() is None
    assert db.query(
        f"SELECT bot, row_count, finished_at IS NULL FROM {spanner.IMPORTS_TABLE}"
        " ORDER BY bot"
    ) == [("box1", 2, False), ("box2", 1, False)]
    assert db.query("SELECT bot, mean, count FROM benchmarks ORDER BY bot") == [
        ("box1", 15.0, 2),
        ("box2", 30.0, 1),
    ]
    assert _count(db, spanner.DIRTY_TABLE) == 0
    # The previous design's marker was cleared too (write = both).
    assert not db.query(
        "SELECT key FROM meta WHERE STARTS_WITH(key, %s)", [spanner.INCOMPLETE_PREFIX]
    )


def test_rebuild_wipes_the_bots_samples_but_not_commits(db):
    bot1, bot2 = _varied_csvs()
    spanner.push_csv(db, "bot1", bot1)
    spanner.push_csv(db, "bot2", bot2)
    assert _count(db, spanner.COMMITS_TABLE) == 4
    # What the delivery path does on --rebuild: rebuild_bot, then re-stage.
    assert spanner.rebuild_bot(db, "bot1") == 12
    for table in (spanner.IMPORT_TABLE, spanner.SAMPLES_TABLE, spanner.DIRTY_TABLE):
        assert _count(db, table, "WHERE bot = 'bot1'") == 0, table
    assert not [k for k in _benchmarks(db) if k[0] == "bot1"]
    out = spanner.push_csv(db, "bot1", _csv({"score": "1"}))
    assert out == "1 rows staged for bot1, aggregated"
    assert _count(db, spanner.SAMPLES_TABLE, "WHERE bot = 'bot1'") == 1
    assert _count(db, spanner.IMPORT_TABLE, "WHERE bot = 'bot1'") == 1
    assert _count(db, spanner.SAMPLES_TABLE, "WHERE bot = 'bot2'") == 5
    assert _count(db, spanner.COMMITS_TABLE) == 4
    # Obsolete aggregate keys of the rebuilt bot are gone, the other bot's stay.
    assert {k[0] for k in _benchmarks(db)} == {"bot1", "bot2"}
    assert [k for k in _benchmarks(db) if k[0] == "bot1"] == [
        ("bot1", "jetstream3.slipstream", "bench-a", "", "v8_default", 100)
    ]
    # Finished imports remain as the ledger.
    assert _count(db, spanner.IMPORTS_TABLE, "WHERE bot = 'bot1'") == 2
