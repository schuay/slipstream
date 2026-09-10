# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import random


from slipstream.analyzer import PerfAnalyzer, _stats


class TestStats:
    def test_empty(self):
        assert _stats([]) == (0.0, 0.0)

    def test_single(self):
        mean, stdev = _stats([5.0])
        assert mean == 5.0
        assert stdev == 0.0

    def test_known_values(self):
        mean, stdev = _stats([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert abs(mean - 5.0) < 1e-10
        assert abs(stdev - 2.138) < 0.001  # sample stdev (ddof=1)


class TestPELTDetection:
    def test_single_changepoint(self, populated_store):
        analyzer = PerfAnalyzer()
        analyzer.load_from_db(populated_store, "v8")
        results = analyzer.analyze()

        assert len(results) == 1
        cp = results[0]
        assert cp.commit_id == 1015
        assert cp.direction == "improvement"
        assert cp.pct_change > 0.05
        assert cp.magnitude > 1.0
        assert cp.confidence == "high"

    def test_no_changepoint_flat_series(self, store):
        """Flat series should produce no change points."""
        rng = random.Random(42)
        for cid in range(1000, 1030):
            store.conn.execute(
                "INSERT INTO commits (engine,hash,commit_id,date,timestamp,title)"
                " VALUES (?,?,?,?,?,?)",
                ("v8", f"h{cid}", cid, "d", 0, "t"),
            )
        store.conn.commit()
        for cid in range(1000, 1030):
            scores = [
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "b",
                    "metric": "Total-Score",
                    "run": r,
                    "score": 100.0 + rng.gauss(0, 1),
                }
                for r in range(1, 4)
            ]
            store.insert_scores("v8", "x86_64", cid, 0, scores)

        analyzer = PerfAnalyzer()
        analyzer.load_from_db(store, "v8")
        assert analyzer.analyze() == []

    def test_too_few_points(self, store):
        """Series with < 4 points should return no change points."""
        for cid in [1, 2, 3]:
            store.conn.execute(
                "INSERT INTO commits (engine,hash,commit_id,date,timestamp,title)"
                " VALUES (?,?,?,?,?,?)",
                ("v8", f"h{cid}", cid, "d", 0, "t"),
            )
        store.conn.commit()
        for cid in [1, 2, 3]:
            store.insert_scores(
                "v8",
                "x86_64",
                cid,
                0,
                [
                    {
                        "suite": "js3",
                        "flags": "default",
                        "benchmark": "b",
                        "metric": "Total-Score",
                        "run": 1,
                        "score": 100.0,
                    },
                ],
            )

        analyzer = PerfAnalyzer()
        analyzer.load_from_db(store, "v8")
        assert analyzer.analyze() == []

    def test_regression_detected(self, store):
        """A drop in scores should be detected as a regression."""
        rng = random.Random(42)
        for cid in range(1000, 1030):
            store.conn.execute(
                "INSERT INTO commits (engine,hash,commit_id,date,timestamp,title)"
                " VALUES (?,?,?,?,?,?)",
                ("v8", f"h{cid}", cid, "d", 0, "t"),
            )
        store.conn.commit()
        for cid in range(1000, 1030):
            base = 110.0 if cid < 1015 else 100.0  # drop
            scores = [
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "b",
                    "metric": "Total-Score",
                    "run": r,
                    "score": base + rng.gauss(0, 1),
                }
                for r in range(1, 4)
            ]
            store.insert_scores("v8", "x86_64", cid, 0, scores)

        analyzer = PerfAnalyzer()
        analyzer.load_from_db(store, "v8")
        results = analyzer.analyze()
        assert len(results) == 1
        assert results[0].direction == "regression"
        assert results[0].pct_change < 0

    def test_multiple_changepoints(self, store):
        """Two shifts in a single series should produce two change points."""
        rng = random.Random(42)
        for cid in range(1000, 1060):
            store.conn.execute(
                "INSERT INTO commits (engine,hash,commit_id,date,timestamp,title)"
                " VALUES (?,?,?,?,?,?)",
                ("v8", f"h{cid}", cid, "d", 0, "t"),
            )
        store.conn.commit()
        for cid in range(1000, 1060):
            if cid < 1020:
                base = 100.0
            elif cid < 1040:
                base = 120.0
            else:
                base = 90.0
            scores = [
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "b",
                    "metric": "Total-Score",
                    "run": r,
                    "score": base + rng.gauss(0, 1),
                }
                for r in range(1, 4)
            ]
            store.insert_scores("v8", "x86_64", cid, 0, scores)

        analyzer = PerfAnalyzer()
        analyzer.load_from_db(store, "v8")
        results = analyzer.analyze()
        assert len(results) == 2
        cids = sorted(cp.commit_id for cp in results)
        assert cids == [1020, 1040]


class TestFilters:
    def test_include_bench(self, populated_store):
        analyzer = PerfAnalyzer()
        analyzer.load_from_db(populated_store, "v8")
        assert len(analyzer.analyze(include_bench=["test-bench"])) == 1
        assert len(analyzer.analyze(include_bench=["nonexistent"])) == 0

    def test_exclude_bench(self, populated_store):
        analyzer = PerfAnalyzer()
        analyzer.load_from_db(populated_store, "v8")
        assert len(analyzer.analyze(exclude_bench=["test-bench"])) == 0

    def test_include_score(self, populated_store):
        analyzer = PerfAnalyzer()
        analyzer.load_from_db(populated_store, "v8")
        assert len(analyzer.analyze(include_score=["Total-Score"])) == 1
        assert len(analyzer.analyze(include_score=["First"])) == 0

    def test_exclude_score(self, populated_store):
        analyzer = PerfAnalyzer()
        analyzer.load_from_db(populated_store, "v8")
        assert len(analyzer.analyze(exclude_score=["Total-Score"])) == 0


class TestCSVLoading:
    def test_load_new_format(self, csv_new_format):
        analyzer = PerfAnalyzer()
        assert analyzer.load_results(csv_new_format)
        assert "js3[default] test-bench" in analyzer.data

    def test_load_old_format(self, csv_old_format):
        analyzer = PerfAnalyzer()
        assert analyzer.load_results(csv_old_format)
        assert "test-bench" in analyzer.data

    def test_load_nonexistent(self, tmp_path):
        analyzer = PerfAnalyzer()
        assert not analyzer.load_results(tmp_path / "nope.csv")

    def test_load_commit_infos(self, commit_infos_csv):
        analyzer = PerfAnalyzer()
        analyzer.load_commit_infos(str(commit_infos_csv))
        assert 1000 in analyzer.commits
        assert analyzer.commits[1000].hash == "hash1000"

    def test_csv_analysis_matches_db(self, populated_store, tmp_path, commit_infos_csv):
        """CSV and DB loading should produce the same change point."""
        # Export scores to CSV
        csv_path = tmp_path / "raw_results-v8-x86_64.csv"
        with open(csv_path, "w") as f:
            f.write("b_type, flags, benchmark, score_type, commit_id, score\n")
            for row in populated_store.conn.execute(
                "SELECT suite, flags, benchmark, metric, commit_id, score FROM scores ORDER BY commit_id, run"
            ):
                f.write(", ".join(str(v) for v in row) + "\n")

        # Analyze via DB
        a_db = PerfAnalyzer()
        a_db.load_from_db(populated_store, "v8")
        r_db = a_db.analyze()

        # Analyze via CSV
        a_csv = PerfAnalyzer()
        a_csv.load_results(csv_path)
        a_csv.load_commit_infos(str(commit_infos_csv))
        r_csv = a_csv.analyze()

        assert len(r_db) == len(r_csv)
        assert r_db[0].commit_id == r_csv[0].commit_id
        assert abs(r_db[0].pct_change - r_csv[0].pct_change) < 1e-10
