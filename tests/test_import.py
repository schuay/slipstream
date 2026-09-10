# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path


class TestImportDedup:
    """Test that importing the same CSV twice doesn't create duplicate scores."""

    def test_duplicate_import(self, store, csv_new_format):
        """Importing the same file twice should not double the row count."""
        import csv as _csv

        def do_import(path: Path):
            run_counters: dict[tuple, int] = {}
            rows: list[tuple] = []
            with open(path) as f:
                for row in _csv.DictReader(f):
                    row = {k.strip(): v.strip() for k, v in row.items()}
                    cid = int(row["commit_id"])
                    suite = row["b_type"]
                    flags = row["flags"]
                    benchmark = row["benchmark"]
                    metric = row["score_type"]
                    score = float(row["score"])
                    key = (cid, suite, flags, benchmark, metric)
                    run_counters[key] = run_counters.get(key, 0) + 1
                    rows.append(
                        (
                            cid,
                            suite,
                            flags,
                            benchmark,
                            metric,
                            run_counters[key],
                            score,
                            0,
                        )
                    )
            store.bulk_insert_scores("v8", "arm64", rows)
            return len(rows)

        n1 = do_import(csv_new_format)
        count_after_first = store.conn.execute(
            "SELECT COUNT(*) FROM scores"
        ).fetchone()[0]

        n2 = do_import(csv_new_format)
        count_after_second = store.conn.execute(
            "SELECT COUNT(*) FROM scores"
        ).fetchone()[0]

        assert n1 == 6
        assert n2 == 6
        assert count_after_first == 6
        assert count_after_second == 6  # no duplicates

    def test_overlapping_files(self, store, tmp_path):
        """Two CSVs covering overlapping commit ranges should not create duplicates."""
        csv1 = tmp_path / "raw_results-v8-arm64-1000-1001.csv"
        csv1.write_text(
            "b_type, flags, benchmark, score_type, commit_id, score\n"
            "js3, default, bench, Total-Score, 1000, 100.0\n"
            "js3, default, bench, Total-Score, 1001, 101.0\n"
        )
        csv2 = tmp_path / "raw_results-v8-arm64-1001-1002.csv"
        csv2.write_text(
            "b_type, flags, benchmark, score_type, commit_id, score\n"
            "js3, default, bench, Total-Score, 1001, 101.0\n"
            "js3, default, bench, Total-Score, 1002, 102.0\n"
        )

        for path in [csv1, csv2]:
            import csv as _csv

            run_counters: dict[tuple, int] = {}
            rows: list[tuple] = []
            with open(path) as f:
                for raw_row in _csv.DictReader(f):
                    row = {k.strip(): v.strip() for k, v in raw_row.items()}
                    cid = int(row["commit_id"])
                    key = (
                        cid,
                        row["b_type"],
                        row["flags"],
                        row["benchmark"],
                        row["score_type"],
                    )
                    run_counters[key] = run_counters.get(key, 0) + 1
                    rows.append(
                        (
                            cid,
                            key[1],
                            key[2],
                            key[3],
                            key[4],
                            run_counters[key],
                            float(row["score"]),
                            0,
                        )
                    )
            store.bulk_insert_scores("v8", "arm64", rows)

        count = store.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        assert count == 3  # 1000, 1001, 1002 — no duplicate for 1001
