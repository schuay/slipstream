# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from slipstream.collector import BenchCollector
from slipstream.config import EngineConfig, RunSpec, load_config
from slipstream.models import Score
from slipstream.runners import (
    ChromiumRunner,
    RunRequest,
    RunResult,
    ShellRunner,
    geomean_overall,
    runner_for,
)


def _engine(name="v8", **kw):
    return EngineConfig(
        name=name,
        src_dir=None,
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
        **kw,
    )


class TestRuntime:
    def test_the_bundled_engines_declare_theirs(self, tmp_path):
        """A browser engine must not fall through to the shell runner, which
        would hand its binary a cli script and record whatever it printed."""
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f'out_dir = "{tmp_path}"\n[engines.v8]\n[engines.chrome]\n[engines.jsc]\n'
        )
        config = load_config(cfg)
        assert config.engines["v8"].runtime == "shell"
        assert config.engines["jsc"].runtime == "shell"
        assert config.engines["chrome"].runtime == "chromium"

    def test_an_unknown_runtime_is_rejected(self):
        with pytest.raises(ValueError, match="runtime must be one of"):
            _engine(runtime="wasmtime")

    def test_a_known_runtime_without_a_runner_is_refused_at_bench_time(self):
        """Known to the config, not yet driven: the config loads, the bench
        says so when asked to use it."""
        with pytest.raises(ValueError, match="no runner for runtime 'safari'"):
            runner_for("safari", log=lambda m: None, progress=lambda: None)

    def test_each_runtime_gets_its_runner(self):
        kw = dict(log=lambda m: None, progress=lambda: None)
        assert isinstance(runner_for("shell", **kw), ShellRunner)
        assert isinstance(runner_for("chromium", **kw), ChromiumRunner)


class TestRunRequest:
    def test_artifacts_are_named_by_run_and_config(self, config, tmp_path):
        req = RunRequest(
            engine=_engine(),
            run_root=tmp_path,
            bench=config.benchmarks["js3"],
            spec=RunSpec(engine="v8", suite="js3", variant="tlf"),
            run=2,
            res_dir=tmp_path / "res",
        )
        assert req.artifact("stdout", "txt") == tmp_path / "res/stdout.2.js3.tlf.txt"
        assert req.artifact("report", "json") == tmp_path / "res/report.2.js3.tlf.json"

    def test_the_run_overrides_the_suites_mode(self, config, tmp_path):
        bench = config.benchmarks["js3"]
        base = dict(
            engine=_engine(), run_root=tmp_path, bench=bench, run=1, res_dir=tmp_path
        )
        assert (
            RunRequest(spec=RunSpec(engine="v8", suite="js3"), **base).run_mode
            == "suite"
        )
        assert (
            RunRequest(
                spec=RunSpec(engine="v8", suite="js3", run_mode="per_benchmark"), **base
            ).run_mode
            == "per_benchmark"
        )


class TestGeomeanOverall:
    def test_geomean_of_total_scores_only(self):
        scores = [
            Score("js3", "default", "Air", "Total-Score", 1, 4.0),
            Score("js3", "default", "Air", "Startup-Score", 1, 100.0),
            Score("js3", "default", "Box2D", "Total-Score", 1, 9.0),
            Score("js3", "default", "Overall", "Total-Score", 1, 1.0),
        ]
        (overall,) = geomean_overall(scores, 1)
        assert overall == Score("js3", "default", "Overall", "Total-Score", 1, 6.0)

    def test_nothing_from_nothing(self):
        assert geomean_overall([], 1) == []
        assert (
            geomean_overall([Score("js3", "d", "Air", "Total-Score", 1, 0.0)], 1) == []
        )


class TestCollectorSeam:
    def _collector(self, config):
        config.engines["v8"] = _engine()
        config.benchmarks["js3"].dir.mkdir(parents=True, exist_ok=True)
        return BenchCollector(config)

    def test_scores_of_a_failed_run_are_not_kept(self, config, tmp_path, monkeypatch):
        c = self._collector(config)
        half = [Score("js3", "default", "Air", "Total-Score", 1, 4.0)]
        monkeypatch.setattr(c, "_runner", lambda engine: _Fixed(RunResult(False, half)))
        outcome = c._run_benchmarks(config.engines["v8"], "1", 1, tmp_path / "root")
        assert outcome == (0, 1, 0)
        assert c.store.conn.execute("SELECT count(*) FROM scores").fetchone()[0] == 0

    def test_the_runners_scores_reach_the_store(self, config, tmp_path, monkeypatch):
        c = self._collector(config)
        scores = [
            Score("js3", "default", "Air", "Total-Score", 1, 4.0),
            Score("js3", "default", "Overall", "Total-Score", 1, 4.0),
        ]
        monkeypatch.setattr(
            c, "_runner", lambda engine: _Fixed(RunResult(True, scores))
        )
        outcome = c._run_benchmarks(config.engines["v8"], "1", 2, tmp_path / "root")
        assert outcome == (2, 2, 4)
        rows = c.store.conn.execute(
            "SELECT benchmark, metric, run, score FROM scores ORDER BY run, benchmark"
        ).fetchall()
        assert [tuple(r) for r in rows] == [
            ("Air", "Total-Score", 1, 4.0),
            ("Overall", "Total-Score", 1, 4.0),
        ]

    def test_a_dry_run_runs_nothing(self, config, tmp_path, monkeypatch):
        c = BenchCollector(config, dry_run=True)
        config.engines["v8"] = _engine()
        monkeypatch.setattr(
            ShellRunner, "run", lambda self, req: pytest.fail("ran in dry-run")
        )
        outcome = c._run_benchmarks(config.engines["v8"], "1", 1, tmp_path / "root")
        assert outcome == (1, 1, 0)


class _Fixed:
    def __init__(self, result):
        self._result = result

    def run(self, req):
        return self._result
