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


def test_relay_cycle_stages_both_files_and_aggregates_once(db, tmp_path, monkeypatch):
    """A relay cycle over two spool entries: one connection, one aggregation,
    and the group's aggregate covers the runs from both files."""
    from slipstream import push as push_mod
    from slipstream.config import PushConfig, PushTarget, RelaySource
    from slipstream.relay import read_cursor, relay_source

    class KeepOpen:
        """The session closes its connection; this one belongs to the fixture."""

        def __getattr__(self, name):
            return getattr(db, name)

        def close(self):
            pass

    monkeypatch.setattr(spanner, "connect", lambda spec: KeepOpen())
    real_refresh = spanner.refresh
    refreshes = []
    monkeypatch.setattr(
        spanner, "refresh", lambda d: (refreshes.append(d), real_refresh(d))[1]
    )

    outbox = tmp_path / "outbox"
    push_mod.spool_append(outbox, _csv({"run": "1", "score": "10"}), 90)
    push_mod.spool_append(outbox, _csv({"run": "2", "score": "20"}), 90)

    class LocalSpool:
        def list(self):
            names = (push_mod.parse_seq(p.name) for p in outbox.iterdir())
            return sorted(s for s in names if s)

        def fetch(self, seq):
            return (outbox / push_mod.seq_name(seq)).read_text()

    class Cfg:
        push = PushConfig(bot_name="box1", targets=[PushTarget(spanner="p/i/d")])

    source = RelaySource(
        ssh_host="box2",
        spool_dir=str(outbox),
        bot_name="box2",
        cursor_dir=tmp_path / "relay",
    )
    logs = []
    assert relay_source(Cfg(), source, LocalSpool(), logs.append) == 2
    assert len(refreshes) == 1
    assert logs == ["box2: 2 rows staged for box2, aggregated"]
    assert read_cursor(tmp_path / "relay" / "box2.cursor") == 2
    assert _agg(db)[("bench-a", "v8_default")][:5] == (
        15.0,
        10.0,
        20.0,
        pytest.approx(7.0710678),
        2,
    )


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
