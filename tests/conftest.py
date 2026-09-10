# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import random

import pytest

from slipstream import lock as lock_mod
from slipstream.config import (
    BenchmarkConfig,
    Config,
    PushConfig,
    PushTarget,
    RunSpec,
)
from slipstream.store import CommitStore


@pytest.fixture(autouse=True)
def isolated_machine_lock(tmp_path_factory, monkeypatch):
    """Keep every test off the real ~/.cache/slipstream lock and pause file.

    The lock is per machine by design, so nothing points it at a temp path in
    production; the defaults are read at call time so a test can.
    """
    locks = tmp_path_factory.mktemp("locks")
    monkeypatch.setattr(lock_mod, "MACHINE_LOCK", locks / "machine.lock")
    monkeypatch.setattr(lock_mod, "PAUSE_FILE", locks / "pause")


@pytest.fixture
def config(tmp_path):
    """A real Config, so tests read the same attribute surface as the CLI does."""
    return Config(
        out_dir=tmp_path,
        results_dir="results",
        engines={},
        benchmarks={
            "js3": BenchmarkConfig(
                name="js3",
                dir=tmp_path / "js3",
                cli="cli.js",
                names=["test-bench"],
                score_regex="",
            )
        },
        runs=[RunSpec(engine="v8", suite="js3"), RunSpec(engine="jsc", suite="js3")],
        bot_name="test-bot",
        push=PushConfig(
            bot_name="test-bot", targets=[PushTarget(spool_dir=tmp_path / "outbox")]
        ),
        platform="arm64",
    )


@pytest.fixture
def store(tmp_path):
    """Fresh CommitStore backed by a temp DB."""
    s = CommitStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with 30 commits and scores: stable at 100 for 1000-1014, jump to 110 for 1015-1029."""
    rng = random.Random(42)
    for cid in range(1000, 1030):
        store.conn.execute(
            "INSERT INTO commits (engine, hash, commit_id, date, timestamp, title)"
            " VALUES (?,?,?,?,?,?)",
            ("v8", f"hash{cid}", cid, "2026-01-01", 1700000000 + cid, f"commit {cid}"),
        )
    store.conn.commit()

    for cid in range(1000, 1030):
        base = 100.0 if cid < 1015 else 110.0
        scores = [
            {
                "suite": "js3",
                "flags": "default",
                "benchmark": "test-bench",
                "metric": "Total-Score",
                "run": r,
                "score": base + rng.gauss(0, 1),
            }
            for r in range(1, 4)
        ]
        store.insert_scores("v8", "x86_64", cid, 1700000000, scores)
    return store


@pytest.fixture
def csv_new_format(tmp_path):
    """CSV file in 6-column format with header."""
    p = tmp_path / "raw_results-v8-arm64-1000-1002.csv"
    p.write_text(
        "b_type, flags, benchmark, score_type, commit_id, score\n"
        "js3, default, test-bench, Total-Score, 1000, 100.5\n"
        "js3, default, test-bench, Total-Score, 1000, 101.2\n"
        "js3, default, test-bench, Total-Score, 1001, 99.8\n"
        "js3, default, test-bench, Total-Score, 1001, 100.1\n"
        "js3, default, test-bench, Total-Score, 1002, 110.5\n"
        "js3, default, test-bench, Total-Score, 1002, 111.2\n"
    )
    return p


@pytest.fixture
def csv_old_format(tmp_path):
    """CSV file in 4-column format with header."""
    p = tmp_path / "old_results.csv"
    p.write_text(
        "benchmark, score_type, commit_id, score\n"
        "test-bench, Total-Score, 1000, 100.5\n"
        "test-bench, Total-Score, 1001, 99.8\n"
    )
    return p


@pytest.fixture
def commit_infos_csv(tmp_path):
    """Commit info CSV for the analyzer."""
    p = tmp_path / "commit-infos-v8.csv"
    lines = []
    for cid in range(1000, 1030):
        lines.append(
            f'{cid}, hash{cid}, 2026-01-01, "commit {cid}", {1700000000 + cid}'
        )
    p.write_text("\n".join(lines) + "\n")
    return p
