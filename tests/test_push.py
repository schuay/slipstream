# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import io
import os
import threading
import time
from pathlib import Path

import pytest

from slipstream.config import PushConfig, PushTarget
from slipstream.push import (
    _COLUMNS as _COLUMNS_FOR_TEST,
    build_export_csv,
    parse_seq,
    push,
    spool_append,
)

VALID = {"js3": {"bench-a", "bench-b"}, "js2": {"Air"}}


def _score(benchmark, suite="js3", flags="default", run=1, score=1.0):
    return {
        "suite": suite,
        "flags": flags,
        "benchmark": benchmark,
        "metric": "Total-Score",
        "run": run,
        "score": score,
    }


def _seed(store, commit_ids, platform="arm64", scores=None):
    store.insert_commits("v8", [{"hash": f"h{c}"} for c in commit_ids])
    for c in commit_ids:
        store.update_commit_metadata("v8", f"h{c}", c, "2026-01-01", c, f"title {c}")
        store.insert_scores("v8", platform, c, 0, scores or [_score("bench-a")])
        store.mark_done("v8", platform, c)


@pytest.fixture
def deliveries(monkeypatch):
    """Replace the spool write; records deliveries and can be told to fail."""
    calls = []
    state = {"fail": False}

    def append(spool_dir, csv_data, retain_days):
        calls.append((spool_dir, csv_data))
        if state["fail"] or state.get("fail_at") == len(calls):
            raise OSError(f"no space left on {spool_dir}")
        return len(calls)

    monkeypatch.setattr("slipstream.push.spool_append", append)
    return type("Deliveries", (), {"calls": calls, "state": state})


def _rows(csv_text):
    return list(csv.DictReader(io.StringIO(csv_text)))


class TestBuildExportCsv:
    def test_joins_commit_metadata_and_prefixes_flags(self, store):
        _seed(store, [100], scores=[_score("bench-a", flags="turbolev_future")])
        text, n = build_export_csv(store, "v8", "arm64", VALID)
        assert n == 1
        (row,) = _rows(text)
        assert row["git_hash"] == "h100"
        assert row["commit_title"] == "title 100"
        assert row["flags"] == "v8_turbolev_future"

    def test_filters_benchmarks_per_suite(self, store):
        # "Air" is valid for js2 only; under js3 it must be dropped.
        _seed(
            store,
            [100],
            scores=[_score("Air", suite="js3"), _score("Air", suite="js2")],
        )
        text, n = build_export_csv(store, "v8", "arm64", VALID)
        assert n == 1
        assert _rows(text)[0]["suite"] == "js2"

    def test_commit_filter(self, store):
        _seed(store, [100, 200, 300])
        text, n = build_export_csv(store, "v8", "arm64", VALID, commit_ids=[200])
        assert n == 1
        assert _rows(text)[0]["commit_id"] == "200"

    def test_empty_commit_filter_exports_nothing(self, store):
        _seed(store, [100])
        assert build_export_csv(store, "v8", "arm64", VALID, commit_ids=[]) == ("", 0)


SPOOL_A = PushTarget(spool_dir=Path("/spool/a"))
SPOOL_B = PushTarget(spool_dir=Path("/spool/b"))


class TestPush:
    cfg = PushConfig(bot_name="bot", targets=[SPOOL_A])

    def test_success_marks_pushed(self, store, deliveries):
        _seed(store, [100, 200])
        n = push(store, ["v8"], self.cfg, VALID, "arm64")
        assert n == 2
        assert len(deliveries.calls) == 1
        assert store.unpushed_commit_ids("v8", "arm64") == []

    def test_only_unpushed_commits_are_sent(self, store, deliveries):
        _seed(store, [100, 200])
        store.mark_pushed("v8", "arm64", [100])
        push(store, ["v8"], self.cfg, VALID, "arm64")
        ((_, csv_text),) = deliveries.calls
        assert [r["commit_id"] for r in _rows(csv_text)] == ["200"]

    def test_failure_leaves_push_state_untouched(self, store, deliveries):
        _seed(store, [100])
        deliveries.state["fail"] = True
        with pytest.raises(OSError):
            push(store, ["v8"], self.cfg, VALID, "arm64")
        assert store.unpushed_commit_ids("v8", "arm64") == [100]

    def test_all_engines_share_one_csv(self, store, deliveries):
        _seed(store, [100])
        store.insert_commits("jsc", [{"hash": "j1"}])
        store.update_commit_metadata("jsc", "j1", 500, "2026-01-01", 500, "t")
        store.insert_scores("jsc", "arm64", 500, 0, [_score("bench-a")])
        store.mark_done("jsc", "arm64", 500)
        assert push(store, ["v8", "jsc"], self.cfg, VALID, "arm64") == 2
        ((_, csv_text),) = deliveries.calls
        assert [r["engine"] for r in _rows(csv_text)] == ["v8", "jsc"]

    def test_every_target_must_succeed_before_marking(self, store, deliveries):
        _seed(store, [100])
        cfg = PushConfig(bot_name="bot", targets=[SPOOL_A, SPOOL_B])
        deliveries.state["fail_at"] = 2
        with pytest.raises(OSError):
            push(store, ["v8"], cfg, VALID, "arm64")
        assert len(deliveries.calls) == 2
        assert store.unpushed_commit_ids("v8", "arm64") == [100]

    def test_retry_after_failure_resends_to_all_targets(self, store, deliveries):
        _seed(store, [100])
        cfg = PushConfig(bot_name="bot", targets=[SPOOL_A, SPOOL_B])
        deliveries.state["fail_at"] = 2
        with pytest.raises(OSError):
            push(store, ["v8"], cfg, VALID, "arm64")
        deliveries.state.pop("fail_at")
        push(store, ["v8"], cfg, VALID, "arm64")
        assert len(deliveries.calls) == 4
        assert store.unpushed_commit_ids("v8", "arm64") == []

    def test_commits_without_exportable_rows_are_marked(self, store, deliveries):
        # Nothing to send, so no delivery, but the commit must not retry forever.
        _seed(store, [100], scores=[_score("not-in-whitelist")])
        assert push(store, ["v8"], self.cfg, VALID, "arm64") == 0
        assert deliveries.calls == []
        assert store.unpushed_commit_ids("v8", "arm64") == []

    def test_nothing_unpushed_delivers_nothing(self, store, deliveries):
        _seed(store, [100])
        store.mark_pushed("v8", "arm64", [100])
        assert push(store, ["v8"], self.cfg, VALID, "arm64") == 0
        assert deliveries.calls == []


def _spool_files(spool):
    """Log entries, oldest first; ignores the lock file and partial writes."""
    return sorted(p for p in spool.iterdir() if parse_seq(p.name))


class TestSpoolTarget:
    def test_writes_complete_file_and_marks(self, store, tmp_path):
        _seed(store, [100, 200])
        spool = tmp_path / "outbox"
        cfg = PushConfig(bot_name="bot", targets=[PushTarget(spool_dir=spool)])
        assert push(store, ["v8"], cfg, VALID, "arm64") == 2
        (f,) = _spool_files(spool)
        assert f.name == "00000001.csv"
        assert [r["commit_id"] for r in _rows(f.read_text())] == ["100", "200"]
        assert store.unpushed_commit_ids("v8", "arm64") == []

    def test_successive_pushes_take_the_next_seq(self, store, tmp_path):
        spool = tmp_path / "outbox"
        cfg = PushConfig(bot_name="bot", targets=[PushTarget(spool_dir=spool)])
        _seed(store, [100])
        push(store, ["v8"], cfg, VALID, "arm64")
        _seed(store, [200])
        push(store, ["v8"], cfg, VALID, "arm64")
        files = _spool_files(spool)
        assert [f.name for f in files] == ["00000001.csv", "00000002.csv"]
        assert _rows(files[0].read_text())[0]["commit_id"] == "100"
        assert _rows(files[1].read_text())[0]["commit_id"] == "200"

    def test_seq_continues_past_a_pruned_prefix(self, tmp_path):
        # Retention deleting old entries must not hand out a number twice.
        spool = tmp_path / "outbox"
        for _ in range(3):
            spool_append(spool, "x", 90)
        (spool / "00000001.csv").unlink()
        assert spool_append(spool, "x", 90) == 4

    def test_concurrent_writers_get_distinct_contiguous_seqs(self, tmp_path):
        spool = tmp_path / "outbox"
        spool.mkdir()
        seqs = []
        lock = threading.Lock()

        def write(i):
            barrier.wait()
            seq = spool_append(spool, f"row {i}\n", 90)
            with lock:
                seqs.append(seq)

        barrier = threading.Barrier(8)
        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(seqs) == list(range(1, 9))
        assert [f.name for f in _spool_files(spool)] == [
            f"{i:08d}.csv" for i in range(1, 9)
        ]
        # Every writer's payload survived; no file was overwritten.
        assert sorted(f.read_text() for f in _spool_files(spool)) == sorted(
            f"row {i}\n" for i in range(8)
        )

    def test_prune_drops_entries_past_retention(self, tmp_path):
        spool = tmp_path / "outbox"
        for _ in range(3):
            spool_append(spool, "x", 90)
        old = spool / "00000001.csv"
        stale = time.time() - 91 * 86400
        os.utime(old, (stale, stale))
        spool_append(spool, "x", 90)
        assert [f.name for f in _spool_files(spool)] == [
            "00000002.csv",
            "00000003.csv",
            "00000004.csv",
        ]

    def test_prune_keeps_entries_inside_retention(self, tmp_path):
        spool = tmp_path / "outbox"
        spool_append(spool, "x", 90)
        old = spool / "00000001.csv"
        recent = time.time() - 89 * 86400
        os.utime(old, (recent, recent))
        spool_append(spool, "x", 90)
        assert len(_spool_files(spool)) == 2


class TestRebuild:
    def test_rebuild_resends_history_and_flags_targets(self, store, monkeypatch):
        _seed(store, [100, 200])
        store.mark_pushed("v8", "arm64", [100, 200])
        seen = []
        monkeypatch.setattr(
            "slipstream.push.deliver_once",
            lambda t, b, d, *, rebuild: seen.append(
                (rebuild, [r["commit_id"] for r in _rows(d)])
            ),
        )
        assert push(store, ["v8"], self_cfg(), VALID, "arm64", rebuild=True) == 2
        assert seen == [(True, ["100", "200"])]
        assert store.unpushed_commit_ids("v8", "arm64") == []


def self_cfg():
    return PushConfig(bot_name="bot", targets=[SPOOL_A])


class TestProbe:
    def test_sends_header_only_to_every_target(self, tmp_path):
        from slipstream.push import probe

        spools = [tmp_path / "a", tmp_path / "b"]
        cfg = PushConfig(
            bot_name="bot", targets=[PushTarget(spool_dir=s) for s in spools]
        )
        logs = []
        probe(cfg, logs.append)
        for spool in spools:
            (f,) = _spool_files(spool)
            assert f.read_text().splitlines() == [",".join(_COLUMNS_FOR_TEST)]
        assert logs == [f"spool {s}: ok" for s in spools]

    def test_failure_propagates(self, deliveries):
        from slipstream.push import probe

        deliveries.state["fail"] = True
        with pytest.raises(OSError):
            probe(PushConfig(bot_name="bot", targets=[SPOOL_A]), lambda m: None)


class TestOrphanedScores:
    """Scores whose commit row is missing can never be exported, so marking
    them pushed would retire them silently and permanently."""

    def _cfg(self):
        return PushConfig(bot_name="box1", targets=[SPOOL_A])

    def test_held_back_and_reported(self, store, deliveries):
        _seed(store, [100])
        # 101 has scores but no commits row, as a bus consumer crashing between
        # writing scores and writing the entry's metadata would leave it.
        store.insert_scores("v8", "arm64", 101, 0, [_score("bench-a")])
        store.mark_done("v8", "arm64", 101)

        logged = []
        push(store, ["v8"], self._cfg(), VALID, "arm64", log=logged.append)

        assert store.unpushed_commit_ids("v8", "arm64") == [101]
        assert any("no commit row" in m for m in logged)

    def test_a_commit_with_nothing_exportable_is_still_marked(self, store, deliveries):
        """Deliberate: its scores are outside the whitelist, not missing."""
        _seed(store, [100], scores=[_score("not-whitelisted")])
        push(store, ["v8"], self._cfg(), VALID, "arm64")
        assert store.unpushed_commit_ids("v8", "arm64") == []


class TestOrphanScanIsBounded:
    """The background pusher runs this after every benched commit."""

    def test_only_the_candidate_range_is_scanned(self, store, deliveries):
        _seed(store, [100])
        # An orphan far below the candidates must not be dragged in.
        store.insert_scores("v8", "arm64", 1, 0, [_score("bench-a")])
        assert store.commit_ids_missing_commit_row("v8", "arm64") == [1]
        assert store.commit_ids_missing_commit_row("v8", "arm64", 100, 100) == []

    def test_push_bounds_the_scan_to_its_candidates(self, store, deliveries):
        """No behavioural difference -- the caller intersects anyway -- so the
        bound has to be asserted where it is applied."""
        _seed(store, [100, 200])
        seen = {}
        real = store.commit_ids_missing_commit_row

        def spy(engine, platform, lo=None, hi=None):
            seen["bounds"] = (lo, hi)
            return real(engine, platform, lo, hi)

        store.commit_ids_missing_commit_row = spy
        push(
            store,
            ["v8"],
            PushConfig(bot_name="box1", targets=[SPOOL_A]),
            VALID,
            "arm64",
        )
        assert seen["bounds"] == (100, 200)

    def test_an_orphan_inside_the_range_is_still_held_back(self, store, deliveries):
        _seed(store, [100])
        store.insert_scores("v8", "arm64", 101, 0, [_score("bench-a")])
        store.mark_done("v8", "arm64", 101)
        push(
            store,
            ["v8"],
            PushConfig(bot_name="box1", targets=[SPOOL_A]),
            VALID,
            "arm64",
        )
        assert store.unpushed_commit_ids("v8", "arm64") == [101]
