# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import re
import subprocess
from types import SimpleNamespace

import pytest

from slipstream.collector import BenchCollector, FetchError
from slipstream.config import EngineConfig, RunSpec


class TestParseStdout:
    """Test _parse_stdout with real score regexes from benchmarks.toml."""

    JS3_REGEX = r"^([A-Za-z0-9-]+)\s+([A-Za-z0-9-]+-Score|Score)\s+([0-9.]+)\s+pts$"
    JS2_REGEX = r"^([A-Za-z0-9.-]+) (Stdlib-Score|Tests-Score|Total-Score|Startup-Score|Average-Score|Worst-Case-Score|Run-Time-Score): ([0-9.]+)$"
    JS2_SUITE_REGEX = r"^\s+([A-Za-z]+-Score):\s+([0-9.]+)$"

    def _make_collector(self, tmp_path):
        """Minimal collector for testing _parse_stdout (no real config needed)."""
        # We call _parse_stdout directly, which doesn't use self.cfg
        collector = object.__new__(BenchCollector)
        return collector

    def test_js3_output(self, tmp_path):
        stdout = tmp_path / "stdout.1.js3.default.txt"
        stdout.write_text(
            "regexp-octane          Total-Score      45.23 pts\n"
            "chai-wtb               Score            112.50 pts\n"
            "Overall                Score            78.00 pts\n"
        )
        collector = self._make_collector(tmp_path)
        patterns = {"js3": re.compile(self.JS3_REGEX)}
        results = collector._parse_stdout(stdout, "js3", "default", 1, patterns, {})

        assert len(results) == 3
        assert results[0]["benchmark"] == "regexp-octane"
        assert results[0]["metric"] == "Total-Score"
        assert results[0]["score"] == 45.23
        assert results[1]["benchmark"] == "chai-wtb"
        assert results[1]["metric"] == "Total-Score"  # "Score" → "Total-Score" for js3
        assert results[1]["score"] == 112.50
        assert results[2]["benchmark"] == "Overall"
        assert results[2]["metric"] == "Total-Score"
        assert results[2]["score"] == 78.00

    def test_js2_output(self, tmp_path):
        stdout = tmp_path / "stdout.1.js2.default.txt"
        stdout.write_text("Air Total-Score: 85.5\nAir Stdlib-Score: 90.2\n")
        collector = self._make_collector(tmp_path)
        patterns = {"js2": re.compile(self.JS2_REGEX)}
        results = collector._parse_stdout(stdout, "js2", "default", 1, patterns, {})

        assert len(results) == 2
        assert results[0]["benchmark"] == "Air"
        assert results[0]["metric"] == "Total-Score"
        assert results[1]["metric"] == "Stdlib-Score"

    def test_js2_suite_score(self, tmp_path):
        stdout = tmp_path / "stdout.1.js2.default.txt"
        stdout.write_text(
            "Air Total-Score: 85.5\n    Total-Score: 331.675\n    Mean-Score: 210.3\n"
        )
        collector = self._make_collector(tmp_path)
        patterns = {"js2": re.compile(self.JS2_REGEX)}
        suite_patterns = {"js2": re.compile(self.JS2_SUITE_REGEX)}
        results = collector._parse_stdout(
            stdout, "js2", "default", 1, patterns, suite_patterns
        )

        assert len(results) == 3
        overall = [r for r in results if r["benchmark"] == "Overall"]
        assert len(overall) == 2
        assert overall[0]["metric"] == "Total-Score"
        assert overall[0]["score"] == 331.675

    def test_empty_file(self, tmp_path):
        stdout = tmp_path / "stdout.1.js3.default.txt"
        stdout.write_text("")
        collector = self._make_collector(tmp_path)
        patterns = {"js3": re.compile(self.JS3_REGEX)}
        assert collector._parse_stdout(stdout, "js3", "default", 1, patterns, {}) == []

    def test_unknown_suite(self, tmp_path):
        stdout = tmp_path / "stdout.txt"
        stdout.write_text("data")
        collector = self._make_collector(tmp_path)
        assert collector._parse_stdout(stdout, "unknown", "default", 1, {}, {}) == []

    def test_run_and_flags_preserved(self, tmp_path):
        stdout = tmp_path / "stdout.txt"
        stdout.write_text("bench1          Total-Score      99.9 pts\n")
        collector = self._make_collector(tmp_path)
        patterns = {"js3": re.compile(self.JS3_REGEX)}
        results = collector._parse_stdout(
            stdout, "js3", "turbolev_future", 3, patterns, {}
        )

        assert len(results) == 1
        assert results[0]["flags"] == "turbolev_future"
        assert results[0]["run"] == 3


class TestRunEnv:
    """_run must never leave git free to block on an interactive prompt."""

    def _make_collector(self, tmp_path):
        collector = object.__new__(BenchCollector)
        collector.dry_run = False
        collector.verbose = False
        collector.log_file = tmp_path / "collector.log"
        return collector

    def test_git_terminal_prompt_disabled(self, tmp_path):
        collector = self._make_collector(tmp_path)
        res = collector._run(
            'echo "$GIT_TERMINAL_PROMPT"', capture=True, caffeinate=False
        )
        assert res.stdout.strip() == "0"

    def test_inherits_rest_of_environment(self, tmp_path):
        collector = self._make_collector(tmp_path)
        os.environ["SLIPSTREAM_TEST_VAR"] = "kept"
        try:
            res = collector._run(
                'echo "$SLIPSTREAM_TEST_VAR"', capture=True, caffeinate=False
            )
        finally:
            del os.environ["SLIPSTREAM_TEST_VAR"]
        assert res.stdout.strip() == "kept"

    def test_explicit_env_wins(self, tmp_path):
        collector = self._make_collector(tmp_path)
        res = collector._run(
            'echo "$GIT_TERMINAL_PROMPT"',
            env={"GIT_TERMINAL_PROMPT": "1", "PATH": os.environ["PATH"]},
            capture=True,
            caffeinate=False,
        )
        assert res.stdout.strip() == "1"


class TestFindFrontier:
    """A failed fetch must not fall through to a stale origin/main."""

    def _make_collector(self, tmp_path, run_results):
        collector = object.__new__(BenchCollector)
        collector.dry_run = False
        collector.verbose = False
        collector.log_file = tmp_path / "collector.log"
        collector.cfg = SimpleNamespace(
            engines={
                "jsc": EngineConfig(
                    name="jsc",
                    src_dir=tmp_path,
                    build_cmd="true",
                    binary_path="jsc",
                    id_regex=r"Canonical link:.*/([0-9]+)@",
                )
            },
            platform="arm64",
        )
        collector.store = SimpleNamespace(max_done_commit_id=lambda *a: 320132)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return run_results[len(calls) - 1]

        collector._run = fake_run
        return collector, calls

    def test_fetch_failure_raises(self, tmp_path):
        # stderr carries bracket text to confirm it survives rich markup escaping.
        failed = subprocess.CompletedProcess(
            "git fetch origin main",
            1,
            stdout="",
            stderr="fatal: could not read Username for 'https://github.com' [no tty]",
        )
        collector, calls = self._make_collector(tmp_path, [failed])

        with pytest.raises(FetchError):
            collector.find_frontier("jsc")
        # Only the fetch ran: falling through to git log would read a stale ref.
        assert calls == ["git fetch origin main"]

    def test_fetch_success_resolves_frontier(self, tmp_path):
        ok = subprocess.CompletedProcess("git fetch origin main", 0, "", "")
        head = subprocess.CompletedProcess("git log", 0, "deadbeef\n", "")
        collector, calls = self._make_collector(tmp_path, [ok, head])
        collector._commit_id_from_hash = lambda engine, h: 320200

        assert collector.find_frontier("jsc") == (320132, 320200)
        assert len(calls) == 2


class TestOutcomeStatus:
    def test_no_scores_is_failed(self):
        from slipstream.collector import BenchOutcome, outcome_status

        assert outcome_status(BenchOutcome(3, 3, 0)) == "failed"

    def test_a_failing_config_is_partial(self):
        from slipstream.collector import BenchOutcome, outcome_status

        assert outcome_status(BenchOutcome(2, 3, 40)) == "partial"

    def test_all_configs_with_scores_is_ok(self):
        from slipstream.collector import BenchOutcome, outcome_status

        assert outcome_status(BenchOutcome(3, 3, 40)) == "ok"


class TestMachineLockInCollect:
    """The build and every run of one commit happen under a single hold."""

    def _collector(self, config, tmp_path, monkeypatch, **kwargs):
        from slipstream.config import EngineConfig
        from slipstream.collector import BenchCollector

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=tmp_path / "src",
            build_cmd="true",
            binary_path="d8",
            id_regex=r"cr-commit-position: refs/heads/main@\{#([0-9]+)\}",
        )
        c = BenchCollector(config, **kwargs)
        c.lock.path = tmp_path / "machine.lock"
        c.store.upsert_commit("v8", "abc", 100, "2026-01-01", 17, "t")
        monkeypatch.setattr(
            c, "get_commit_list", lambda *a, **k: [{"hash": "abc", "title": "t"}]
        )
        monkeypatch.setattr(c, "populate_commit_metadata", lambda *a, **k: None)
        monkeypatch.setattr(c, "_provision", lambda *a, **k: tmp_path / "src")
        return c

    def test_held_across_build_and_runs_and_released_after(
        self, config, tmp_path, monkeypatch
    ):
        from slipstream.collector import BenchOutcome
        from slipstream.lock import MachineLock

        c = self._collector(config, tmp_path, monkeypatch)
        peer = MachineLock("peer", c.lock.path)
        during = {}

        def during_build(*a, **k):
            during["build"] = peer.try_acquire()
            return tmp_path / "src"

        def during_runs(*a, **k):
            during["runs"] = peer.try_acquire()
            return BenchOutcome(1, 1, 5)

        monkeypatch.setattr(c, "_provision", during_build)
        monkeypatch.setattr(c, "_run_benchmarks", during_runs)
        c.collect("v8", 99, 100)

        assert during == {"build": False, "runs": False}
        assert peer.try_acquire(), "the lock was not released after the commit"
        peer.release()
        assert c.store.get_status("v8", config.platform, 100) == "ok"

    def test_released_when_the_run_raises(self, config, tmp_path, monkeypatch):
        from slipstream.lock import MachineLock

        c = self._collector(config, tmp_path, monkeypatch)
        monkeypatch.setattr(
            c,
            "_run_benchmarks",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        c.collect("v8", 99, 100)
        assert MachineLock("peer", c.lock.path).try_acquire()
        assert c.store.get_status("v8", config.platform, 100) == "failed"

    def test_released_when_the_build_fails(self, config, tmp_path, monkeypatch):
        from slipstream.lock import MachineLock

        c = self._collector(config, tmp_path, monkeypatch)
        monkeypatch.setattr(c, "_provision", lambda *a, **k: None)
        c.collect("v8", 99, 100)
        assert MachineLock("peer", c.lock.path).try_acquire()
        assert not c.store.is_done("v8", config.platform, 100)

    def test_an_ad_hoc_run_refuses_to_queue_behind_a_daemon(
        self, config, tmp_path, monkeypatch
    ):
        import pytest as _pytest

        from slipstream.lock import LockBusy, MachineLock

        c = self._collector(config, tmp_path, monkeypatch)
        daemon = MachineLock("watch", c.lock.path)
        daemon.try_acquire()
        with _pytest.raises(LockBusy, match="watch"):
            c.collect("v8", 99, 100)
        daemon.release()

    def test_a_dry_run_takes_no_lock(self, config, tmp_path, monkeypatch):
        from slipstream.lock import MachineLock

        c = self._collector(config, tmp_path, monkeypatch, dry_run=True)
        holder = MachineLock("watch", c.lock.path)
        holder.try_acquire()
        c.collect("v8", 99, 100)  # would raise LockBusy if it tried
        holder.release()


class TestBenchAtRoot:
    """One commit measured from a provisioned run root, wherever it came from."""

    def _collector(self, config, tmp_path, monkeypatch):
        from slipstream.collector import BenchCollector
        from slipstream.config import EngineConfig

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=None,  # a bus consumer has no checkout
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
        )
        return BenchCollector(config)

    COMMIT = {
        "hash": "abc",
        "commit_id": 109680,
        "date": "2026-09-06",
        "timestamp": 17,
        "title": "a commit",
    }

    def test_writes_the_commit_row_with_the_state(self, config, tmp_path, monkeypatch):
        """A done commit with no commits row loses its scores permanently."""
        from slipstream.collector import BenchOutcome

        c = self._collector(config, tmp_path, monkeypatch)
        monkeypatch.setattr(c, "_run_benchmarks", lambda *a: BenchOutcome(2, 2, 40))
        c.bench_at_root(config.engines["v8"], self.COMMIT, tmp_path / "root", 3)

        assert c.store.is_done("v8", config.platform, 109680)
        assert c.store.get_status("v8", config.platform, 109680) == "ok"
        (row,) = c.store.get_commits_with_metadata("v8", ["abc"])
        assert row["commit_id"] == 109680 and row["title"] == "a commit"
        assert c.store.commit_ids_missing_commit_row("v8", config.platform) == []

    def test_a_failed_run_is_recorded_not_hidden(self, config, tmp_path, monkeypatch):
        from slipstream.collector import BenchOutcome

        c = self._collector(config, tmp_path, monkeypatch)
        monkeypatch.setattr(c, "_run_benchmarks", lambda *a: BenchOutcome(0, 2, 0))
        c.bench_at_root(config.engines["v8"], self.COMMIT, tmp_path / "root", 3)
        assert c.store.get_status("v8", config.platform, 109680) == "failed"

    def test_a_colliding_commit_id_does_not_wedge_the_consumer(
        self, config, tmp_path, monkeypatch
    ):
        from slipstream.collector import BenchOutcome

        c = self._collector(config, tmp_path, monkeypatch)
        c.store.upsert_commit("v8", "incumbent", 109680, "d", 0, "held")
        monkeypatch.setattr(c, "_run_benchmarks", lambda *a: BenchOutcome(2, 2, 40))
        c.bench_at_root(config.engines["v8"], self.COMMIT, tmp_path / "root", 3)

        assert c.store.get_status("v8", config.platform, 109680) == "failed"
        (row,) = c.store.get_commits_with_metadata("v8", ["incumbent"])
        assert row["title"] == "held"

    def test_notifies_the_pusher_only_after_the_commit_is_recorded(
        self, config, tmp_path, monkeypatch
    ):
        from slipstream.collector import BenchOutcome

        c = self._collector(config, tmp_path, monkeypatch)
        monkeypatch.setattr(c, "_run_benchmarks", lambda *a: BenchOutcome(2, 2, 40))
        seen = []
        c.bench_at_root(
            config.engines["v8"],
            self.COMMIT,
            tmp_path / "root",
            3,
            on_commit_done=lambda: seen.append(
                c.store.is_done("v8", config.platform, 109680)
            ),
        )
        assert seen == [True]

    def test_the_run_root_decides_where_the_binary_comes_from(
        self, config, tmp_path, monkeypatch
    ):
        c = self._collector(config, tmp_path, monkeypatch)
        engine = config.engines["v8"]
        engine.dyld_lib_path = "lib"
        seen = {}

        def fake_cmd(argv, cwd, env, out_f, err_f, stderr_file, label):
            seen["argv"] = argv
            seen["dyld"] = env.get("DYLD_FRAMEWORK_PATH")
            return True

        monkeypatch.setattr(c, "_run_bench_cmd", fake_cmd)
        root = tmp_path / "roots" / "v8" / "109680"
        config.benchmarks["js3"].dir.mkdir(parents=True, exist_ok=True)
        c._run_benchmarks(engine, "109680", 1, root)

        assert str(root / "out/d8") in seen["argv"]
        assert seen["dyld"] == str(root / "lib")

    def test_the_command_is_a_list_so_a_generated_root_cannot_split(
        self, config, tmp_path, monkeypatch
    ):
        c = self._collector(config, tmp_path, monkeypatch)
        seen = {}

        def fake_cmd(argv, *a, **k):
            seen["argv"] = argv
            return True

        monkeypatch.setattr(c, "_run_bench_cmd", fake_cmd)
        root = tmp_path / "a root with spaces"
        config.benchmarks["js3"].dir.mkdir(parents=True, exist_ok=True)
        c._run_benchmarks(config.engines["v8"], "109680", 1, root)
        assert str(root / "out/d8") in seen["argv"]


class TestConfiguredMatrix:
    """The [[run]] list is what runs, and it is per engine."""

    def _collector(self, config, tmp_path, monkeypatch):
        from slipstream.collector import BenchCollector
        from slipstream.config import EngineConfig

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=None,
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
        )
        config.benchmarks["js3"].dir.mkdir(parents=True, exist_ok=True)
        return BenchCollector(config)

    def test_a_session_refuses_an_engine_with_no_entries(
        self, config, tmp_path, monkeypatch
    ):
        """Without this it measures nothing and marks every commit failed,
        which reads as a broken engine rather than a missing config line."""
        c = self._collector(config, tmp_path, monkeypatch)
        config.runs[:] = [RunSpec(engine="jsc", suite="js3")]
        with pytest.raises(ValueError, match="would measure nothing"):
            c.collect("v8", 1, 2)

    def test_only_this_engines_entries_run(self, config, tmp_path, monkeypatch):
        c = self._collector(config, tmp_path, monkeypatch)
        config.runs.append(RunSpec(engine="jsc", suite="js3", variant="jsc_only"))
        assert [r.variant for r in c.run_configs(config.engines["v8"])] == ["default"]

    def test_the_flags_reach_the_command(self, config, tmp_path, monkeypatch):
        c = self._collector(config, tmp_path, monkeypatch)
        config.runs[:] = [
            RunSpec(
                engine="v8",
                suite="js3",
                flags=("--turbolev-future", "--no-lazy-feedback-allocation"),
                variant="tlf",
            )
        ]
        seen = []
        monkeypatch.setattr(
            c, "_run_bench_cmd", lambda argv, *a, **k: seen.append(argv) or True
        )
        c._run_benchmarks(config.engines["v8"], "109680", 1, tmp_path / "root")
        (argv,) = seen
        assert (
            argv.index("--turbolev-future")
            == argv.index("--no-lazy-feedback-allocation") - 1
        )
        assert argv.index("--turbolev-future") > argv.index(
            str(tmp_path / "root/out/d8")
        )

    def test_a_per_benchmark_run_gets_a_synthesized_overall(
        self, config, tmp_path, monkeypatch
    ):
        """The harness prints no overall when invoked per line item, whatever
        the variant is called."""
        c = self._collector(config, tmp_path, monkeypatch)
        config.runs[:] = [
            RunSpec(
                engine="v8", suite="js3", variant="by_item", run_mode="per_benchmark"
            )
        ]
        config.benchmarks[
            "js3"
        ].score_regex = r"^([A-Za-z0-9-]+)\s+(Total-Score)\s+([0-9.]+)\s+pts$"
        res_dir = config.commit_results_dir("v8", "109680")
        res_dir.mkdir(parents=True, exist_ok=True)

        def fake_cmd(argv, *a, **k):
            (res_dir / "stdout.1.js3.by_item.txt").write_text(
                "Air Total-Score 4.0 pts\nBox2D Total-Score 9.0 pts\n"
            )
            return True

        monkeypatch.setattr(c, "_run_bench_cmd", fake_cmd)
        c._run_benchmarks(config.engines["v8"], "109680", 1, tmp_path / "root")
        rows = c.store.conn.execute(
            "SELECT benchmark, score FROM scores WHERE benchmark = 'Overall'"
        ).fetchall()
        assert [tuple(r) for r in rows] == [("Overall", 6.0)]


class TestResultsLayout:
    def test_two_engines_at_one_commit_id_keep_their_own_logs(
        self, config, tmp_path, monkeypatch
    ):
        """The id spaces are independent, so a shared directory has whichever
        engine benches second overwrite the first's logs."""
        from slipstream.collector import BenchCollector
        from slipstream.config import EngineConfig

        for name in ("v8", "jsc"):
            config.engines[name] = EngineConfig(
                name=name,
                src_dir=None,
                build_cmd="true",
                binary_path="out/d8",
                id_regex=r"#([0-9]+)",
            )
        config.benchmarks["js3"].dir.mkdir(parents=True, exist_ok=True)
        c = BenchCollector(config)
        monkeypatch.setattr(c, "_run_bench_cmd", lambda *a, **k: True)
        for name in ("v8", "jsc"):
            c._run_benchmarks(config.engines[name], "500", 1, tmp_path / "root")

        written = sorted(
            p.relative_to(config.results_path).as_posix()
            for p in config.results_path.rglob("stdout*")
        )
        assert written == [
            "jsc/500/stdout.1.js3.default.txt",
            "v8/500/stdout.1.js3.default.txt",
        ]


class TestLocalProvenance:
    def test_a_locally_built_bench_is_recorded_as_local(
        self, config, tmp_path, monkeypatch
    ):
        """Otherwise the bus/local split in bus status can only ever say bus,
        and a repair with `bench` hides among archive-built neighbours."""
        from slipstream.collector import BenchCollector, BenchOutcome
        from slipstream.config import EngineConfig

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=tmp_path / "src",
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
            run_set=["out"],
        )
        c = BenchCollector(config)
        monkeypatch.setattr(c, "harness_revs", lambda: {"js3": "abc1234"})
        monkeypatch.setattr(c, "_run_benchmarks", lambda *a, **k: BenchOutcome(1, 1, 5))
        c.bench_at_root(
            config.engines["v8"],
            {"hash": "h", "commit_id": 100, "date": "d", "timestamp": 0, "title": "t"},
            tmp_path / "src",
            3,
        )
        row = c.store.get_run_env("v8", 100)
        assert row["source"] == "local" and row["runs"] == 3
        assert row["build_cfg_hash"].startswith("sha256:")
        assert c.store.run_env_source_counts("v8") == {"local": 1}

    def test_a_missing_benchmark_dir_does_not_kill_the_bench(self, config, tmp_path):
        """Provenance runs after scores are already written."""
        from slipstream.collector import BenchCollector

        config.benchmarks["js3"].dir = tmp_path / "gone"
        assert BenchCollector(config).harness_revs() == {"js3": ""}


class TestGnGenRecovery:
    def test_a_failed_gn_gen_is_retried(self, config, tmp_path, monkeypatch):
        """args.gn is written before gn gen runs, so matching args alone would
        make the next attempt skip it and blame the commit for the compile."""
        from slipstream.collector import BenchCollector
        from slipstream.config import EngineConfig

        src = tmp_path / "src"
        (src / "out").mkdir(parents=True)
        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=src,
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
            gn_args="is_debug = false",
        )
        c = BenchCollector(config)
        runs = []

        def failing_gn(cmd, **kwargs):
            runs.append(cmd)
            import subprocess as sp

            return sp.CompletedProcess(cmd, 1)

        monkeypatch.setattr(c, "_run", failing_gn)
        assert c._ensure_gn_args(config.engines["v8"]) == 1
        assert c._ensure_gn_args(config.engines["v8"]) == 1
        assert len(runs) == 2, "the second attempt skipped gn gen"

        # With a build dir behind the args, it is a no-op.
        (src / "out" / "build.ninja").write_text("")
        assert c._ensure_gn_args(config.engines["v8"]) == 0
        assert len(runs) == 2


class TestInterruptedCommitIsRemeasured:
    def test_partial_scores_are_cleared_on_the_git_path_too(
        self, config, tmp_path, monkeypatch
    ):
        """^C during run 2 leaves rows that INSERT OR IGNORE would preserve,
        half measured on each side of the interrupt."""
        from slipstream.collector import BenchCollector, BenchOutcome
        from slipstream.config import EngineConfig

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=tmp_path / "src",
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
        )
        c = BenchCollector(config)
        monkeypatch.setattr(c, "harness_revs", lambda: {})
        stale = {
            "suite": "js3",
            "flags": "default",
            "benchmark": "b",
            "metric": "Total-Score",
            "run": 1,
            "score": 1.0,
        }
        c.store.insert_scores("v8", config.platform, 100, 0, [stale])

        def rerun(engine, commit_id, runs, run_root):
            c.store.insert_scores(
                "v8", config.platform, 100, 0, [{**stale, "score": 99.0}]
            )
            return BenchOutcome(1, 1, 1)

        monkeypatch.setattr(c, "_run_benchmarks", rerun)
        c.bench_at_root(
            config.engines["v8"],
            {"hash": "h", "commit_id": 100, "date": "d", "timestamp": 0, "title": "t"},
            tmp_path / "src",
            1,
        )
        scores = c.store.conn.execute(
            "SELECT score FROM scores WHERE commit_id=100"
        ).fetchall()
        assert [r[0] for r in scores] == [99.0]


class TestDryRunIsInert:
    def _engine(self, tmp_path):
        from slipstream.config import EngineConfig

        src = tmp_path / "src"
        (src / "cfgs").mkdir(parents=True)
        (src / "cfgs" / "Base.xcconfig").write_text(
            "GCC_TREAT_WARNINGS_AS_ERRORS = YES\n"
        )
        return EngineConfig(
            name="v8",
            src_dir=src,
            build_cmd="ninja",
            binary_path="out/rel/d8",
            id_regex=r"#([0-9]+)",
            gn_args="is_debug = false",
            pre_build_patches=["cfgs/Base.xcconfig"],
        )

    def test_build_at_writes_nothing_into_the_checkout(self, config, tmp_path):
        from slipstream.collector import BenchCollector

        engine = self._engine(tmp_path)
        config.engines["v8"] = engine
        BenchCollector(config, dry_run=True).build_at(engine, "deadbeef")
        assert not (engine.src_dir / "out" / "rel" / "args.gn").exists()
        assert "YES" in (engine.src_dir / "cfgs" / "Base.xcconfig").read_text(), (
            "the xcconfig was patched during a dry run"
        )

    def test_a_real_build_still_writes_them(self, config, tmp_path, monkeypatch):
        from slipstream.collector import BenchCollector

        engine = self._engine(tmp_path)
        config.engines["v8"] = engine
        c = BenchCollector(config)
        monkeypatch.setattr(c, "_run", lambda *a, **k: _ok())
        c.build_at(engine, "deadbeef")
        assert (engine.src_dir / "out" / "rel" / "args.gn").exists()
        assert "NO" in (engine.src_dir / "cfgs" / "Base.xcconfig").read_text()

    def test_a_dry_run_with_clear_shows_the_work(self, config, tmp_path, monkeypatch):
        """Nothing is cleared, so is_done would skip every commit and the plan
        would be empty."""
        from slipstream.collector import BenchCollector

        engine = self._engine(tmp_path)
        config.engines["v8"] = engine
        c = BenchCollector(config, dry_run=True)
        c.store.upsert_commit("v8", "abc", 100, "2026-01-01", 17, "t")
        c.store.mark_done("v8", config.platform, 100)
        monkeypatch.setattr(
            c, "get_commit_list", lambda *a, **k: [{"hash": "abc", "title": "t"}]
        )
        monkeypatch.setattr(c, "populate_commit_metadata", lambda *a, **k: None)
        monkeypatch.setattr(c, "_provision", lambda *a, **k: tmp_path / "src")
        benched = []
        monkeypatch.setattr(
            c, "_run_benchmarks", lambda *a: benched.append(a[1]) or _outcome()
        )
        c.collect("v8", 99, 100, clear=True)
        assert benched == ["100"], "the dry run skipped the commit it would clear"
        assert c.store.is_done("v8", config.platform, 100), "state was really cleared"


def _ok():
    import subprocess as sp

    return sp.CompletedProcess("x", 0, "", "")


def _outcome():
    from slipstream.collector import BenchOutcome

    return BenchOutcome(1, 1, 5)


class TestCollisionDropsTheScores:
    def test_scores_do_not_ship_under_the_other_commits_hash(
        self, config, tmp_path, monkeypatch
    ):
        """export_scores joins on (engine, commit_id), so leaving them would
        send this commit's numbers under the incumbent's git hash, and the
        missing-commit-row guard cannot see it because a row does exist."""
        from slipstream.collector import BenchCollector, BenchOutcome
        from slipstream.config import EngineConfig

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=None,
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
        )
        c = BenchCollector(config)
        monkeypatch.setattr(c, "harness_revs", lambda: {})
        c.store.upsert_commit("v8", "hashA", 100, "2026-01-01", 1, "A")

        def measure(engine, commit_id, runs, run_root):
            c.store.insert_scores(
                "v8",
                config.platform,
                100,
                0,
                [
                    {
                        "suite": "js3",
                        "flags": "default",
                        "benchmark": "test-bench",
                        "metric": "Total-Score",
                        "run": 1,
                        "score": 42.0,
                    }
                ],
            )
            return BenchOutcome(1, 1, 1)

        monkeypatch.setattr(c, "_run_benchmarks", measure)
        c.bench_at_root(
            config.engines["v8"],
            {
                "hash": "hashB",
                "commit_id": 100,
                "date": "d",
                "timestamp": 0,
                "title": "B",
            },
            tmp_path / "root",
            1,
        )
        assert c.store.get_status("v8", config.platform, 100) == "failed"
        assert (
            c.store.export_scores("v8", config.platform, {"js3": {"test-bench"}}) == []
        )
        # The incumbent is untouched.
        (row,) = c.store.get_commits_with_metadata("v8", ["hashA"])
        assert row["title"] == "A"


class TestAMidRunFailureExportsNothing:
    def test_partial_scores_go_with_the_failure(self, config, tmp_path, monkeypatch):
        """ "failed" means no scores everywhere else; leaving the runs that
        completed would export and push half a commit as if it were measured."""
        from slipstream.collector import BenchCollector
        from slipstream.config import EngineConfig

        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=None,
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
        )
        c = BenchCollector(config)
        monkeypatch.setattr(c, "harness_revs", lambda: {})
        c.store.upsert_commit("v8", "abc", 100, "2026-01-01", 17, "t")

        def die_after_one_run(engine, commit_id, runs, run_root):
            c.store.insert_scores(
                "v8",
                config.platform,
                100,
                0,
                [
                    {
                        "suite": "js3",
                        "flags": "default",
                        "benchmark": "test-bench",
                        "metric": "Total-Score",
                        "run": 1,
                        "score": 42.0,
                    }
                ],
            )
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(c, "_run_benchmarks", die_after_one_run)
        c.bench_at_root(
            config.engines["v8"],
            {
                "hash": "abc",
                "commit_id": 100,
                "date": "d",
                "timestamp": 0,
                "title": "t",
            },
            tmp_path / "root",
            3,
        )
        assert c.store.get_status("v8", config.platform, 100) == "failed"
        assert (
            c.store.export_scores("v8", config.platform, {"js3": {"test-bench"}}) == []
        )
