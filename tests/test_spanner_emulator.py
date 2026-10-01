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
        assert have == {"slipstream", "benchmarks", "meta"}
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
        spanner, "refresh", lambda d: (refreshes.append(d), real_refresh(d))[1]
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

        def upsert(self, table, columns, rows):
            if table == spanner.IMPORT_TABLE and failed[0]:
                db.upsert(table, columns, rows[:1])
                failed[0] = False
                raise RuntimeError("crash after first chunk")
            db.upsert(table, columns, rows)

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


def test_commit_numbers_per_bot(db):
    """compare-bots reads the pushed intersection from here."""
    spanner.push_csv(
        db,
        "box1-m1",
        _csv(
            {"benchmark": "bench-a", "commit_id": "100"},
            {"benchmark": "bench-a", "commit_id": "101"},
        ),
    )
    spanner.push_csv(
        db,
        "box2-m4",
        _csv(
            {"benchmark": "bench-a", "commit_id": "101"},
            {"benchmark": "bench-a", "commit_id": "102"},
        ),
    )
    suite = spanner._BENCHMARK_ALIASES["js3"]
    a = spanner.commit_numbers(db, "box1-m1", suite)
    b = spanner.commit_numbers(db, "box2-m4", suite)
    assert a == [100, 101]
    assert b == [101, 102]
    assert set(a) & set(b) == {101}
    assert spanner.commit_numbers(db, "box1-m1", "nosuch") == []
