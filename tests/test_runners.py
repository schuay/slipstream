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
    suite_cfg_hash,
)


def _engine(name="v8", binary_path="out/d8", **kw):
    return EngineConfig(
        name=name,
        src_dir=None,
        build_cmd="true",
        binary_path=binary_path,
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

    def test_an_unknown_runtime_is_refused_at_bench_time(self):
        with pytest.raises(ValueError, match="no runner for runtime 'servo'"):
            runner_for("servo", log=lambda m: None, progress=lambda: None)

    def test_each_runtime_gets_its_runner(self):
        from slipstream.runners import SafariRunner

        kw = dict(log=lambda m: None, progress=lambda: None)
        assert isinstance(runner_for("shell", **kw), ShellRunner)
        assert isinstance(runner_for("chromium", **kw), ChromiumRunner)
        assert isinstance(runner_for("safari", **kw), SafariRunner)


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
        c.cool_down = lambda log: pytest.fail("cooled down in dry-run")
        outcome = c._run_benchmarks(config.engines["v8"], "1", 1, tmp_path / "root")
        assert outcome == (1, 1, 0)

    def test_every_measurement_waits_for_the_machine_to_cool(
        self, config, tmp_path, monkeypatch
    ):
        """Before each (run, config), not once per commit: the previous suite
        is what heated the machine."""
        c = self._collector(config)
        order = []
        c.cool_down = lambda log: order.append("cool")

        class _Noting(_Fixed):
            def run(self, req):
                order.append("run")
                return super().run(req)

        monkeypatch.setattr(c, "_runner", lambda engine: _Noting(RunResult(True, [])))
        c._run_benchmarks(config.engines["v8"], "1", 3, tmp_path / "root")
        assert order == ["cool", "run"] * 3


class _Fixed:
    def __init__(self, result):
        self._result = result

    def run(self, req):
        return self._result


def _bundle(app, version, short="", name=None):
    import plistlib

    contents = app / "Contents"
    contents.mkdir(parents=True)
    info = {"CFBundleVersion": version}
    if short:
        info["CFBundleShortVersionString"] = short
    if name:
        info["CFBundleName"] = name
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    return app


class TestRunnerProvenance:
    """Every runner says how it drives an engine and what host application
    a run went through; the collector records both beside the build's."""

    kw = dict(log=lambda m: None, progress=lambda: None)

    def test_each_runtime_has_a_distinct_stable_cfg_hash(self):
        hashes = {
            rt: runner_for(rt, **self.kw).cfg_hash()
            for rt in ("shell", "chromium", "safari")
        }
        assert all(h.startswith("sha256:") for h in hashes.values())
        assert len(set(hashes.values())) == 3
        assert runner_for("safari", **self.kw).cfg_hash() == hashes["safari"]

    def test_the_fixed_flags_are_in_the_hash_and_the_variant_is_not(self, monkeypatch):
        from slipstream.runners import chromium

        before = ChromiumRunner(**self.kw).cfg_hash()
        monkeypatch.setattr(
            chromium, "CHROMIUM_FLAGS", (*chromium.CHROMIUM_FLAGS, "--x")
        )
        assert ChromiumRunner(**self.kw).cfg_hash() != before
        # The [[run]] flags are the variant, which the request carries; the
        # hash is a property of the runner alone and takes no request.
        assert (
            ChromiumRunner(**self.kw).cfg_hash() == ChromiumRunner(**self.kw).cfg_hash()
        )

    # The exact values, so that a change to how an engine is driven or a
    # suite is read shows up here first. Changing a value below is the
    # deliberate act of saying "runs from now on are a new series"; update
    # it together with the change that moved it, never on its own.
    RUNNER_HASHES = {
        "shell": "sha256:fdb9b01694d1f8052ba10fab1323d49e2d18efdecc040d325067d5c93513cd8b",
        "chromium": "sha256:b186c8202a74696dfbae27dbb7cdc830ba1ee4ecba88edd9a81f1984cf59a027",
        "safari": "sha256:e0fb2fae8195c645471ccc50cf91da9e164c0faf8744abadf4e2c53a73468b9a",
    }
    SUITE_HASHES = {
        "js2": "sha256:c8c206b3c459a43f8f7305b13be726b06b65cab2b7c7176a9d647acffcd14c14",
        "js3": "sha256:d8ad52ef09321611b173790a448042039dc1307feaef4275a4d57787736647a9",
        "sp3": "sha256:0140727a710fb747fd56259e58758c28ded078598674baf6e7e6b65c038ba4ac",
    }

    @pytest.mark.parametrize("runtime", sorted(RUNNER_HASHES))
    def test_the_runner_hashes_are_the_known_ones(self, runtime):
        assert runner_for(runtime, **self.kw).cfg_hash() == self.RUNNER_HASHES[runtime]

    @pytest.fixture
    def suites(self, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f'out_dir = "{tmp_path}"\n'
            + "".join(
                f'[benchmarks.{s}]\ndir = "{tmp_path}"\n' for s in self.SUITE_HASHES
            )
        )
        return load_config(cfg).benchmarks

    @pytest.mark.parametrize("suite", sorted(SUITE_HASHES))
    def test_the_suite_hashes_are_the_known_ones(self, suites, suite):
        assert suite_cfg_hash(suites[suite]) == self.SUITE_HASHES[suite]

    def test_the_injected_scripts_bytes_are_in_its_suites_hash(
        self, suites, monkeypatch
    ):
        """An edit to sp3-report.mjs changes what the page reports, and
        nothing else; the hash follows the bytes, and only sp3's."""
        from slipstream.config import BenchmarkConfig

        monkeypatch.setattr(
            BenchmarkConfig,
            "inject_bytes",
            lambda self: b"// edited\n" if self.inject else b"",
        )
        assert suite_cfg_hash(suites["sp3"]) != self.SUITE_HASHES["sp3"]
        assert suite_cfg_hash(suites["js3"]) == self.SUITE_HASHES["js3"]

    def test_the_parser_version_is_in_the_hash(self, suites, monkeypatch):
        from slipstream.runners import report

        monkeypatch.setitem(report.PARSER_VERSIONS, "speedometer", "2")
        assert suite_cfg_hash(suites["sp3"]) != self.SUITE_HASHES["sp3"]
        assert suite_cfg_hash(suites["js3"]) == self.SUITE_HASHES["js3"]

    def test_a_suites_dir_and_timeout_are_not_in_its_hash(self, suites):
        from dataclasses import replace

        sp3 = suites["sp3"]
        moved = replace(sp3, dir=sp3.dir / "elsewhere", timeout="1h")
        assert suite_cfg_hash(moved) == self.SUITE_HASHES["sp3"]

    def test_a_shell_binary_has_no_host_app(self, tmp_path):
        assert ShellRunner(**self.kw).host_env(_engine(), tmp_path) == {}

    def test_chromium_names_the_bundle_it_ran(self, tmp_path):
        engine = _engine(
            "chrome",
            runtime="chromium",
            binary_path="out/Chromium.app/Contents/MacOS/Chromium",
        )
        _bundle(
            tmp_path / "out" / "Chromium.app", "7300.0.1", "146.0.7300.1", "Chromium"
        )
        env = ChromiumRunner(**self.kw).host_env(engine, tmp_path)
        assert env == {"host_app": "Chromium 7300.0.1 (146.0.7300.1)"}

    def test_safari_names_the_packaged_stp_and_the_hosts_launcher(self, tmp_path):
        from slipstream.runners import SafariRunner

        launcher_app = _bundle(
            tmp_path / "sys" / "Safari.app", "20622.1.2", "26.1", "Safari"
        )
        engine = _engine(
            "safari",
            runtime="safari",
            binary_path=str(launcher_app / "Contents/MacOS/SafariForWebKitDevelopment"),
            run_set=["Safari Technology Preview.app"],
        )
        root = tmp_path / "root"
        _bundle(
            root / "Safari Technology Preview.app",
            "22626.1.8.19.2",
            "27.0",
            "Safari Technology Preview",
        )
        env = SafariRunner(**self.kw).host_env(engine, root)
        assert env == {
            "host_app": "Safari Technology Preview 22626.1.8.19.2 (27.0)",
            "launcher": "Safari 20622.1.2 (26.1)",
        }

    def test_a_missing_bundle_is_recorded_as_unknown_not_raised(self, tmp_path):
        from slipstream.runners import SafariRunner

        engine = _engine(
            "safari",
            runtime="safari",
            binary_path="/nowhere/Safari.app/Contents/MacOS/SafariForWebKitDevelopment",
            run_set=["Safari Technology Preview.app"],
        )
        env = SafariRunner(**self.kw).host_env(engine, tmp_path)
        assert env["host_app"].startswith("unknown (no application bundle")
        assert env["launcher"].startswith("unknown (no application bundle")
