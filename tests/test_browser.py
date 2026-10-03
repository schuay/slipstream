# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import signal
import sys
import urllib.request
from pathlib import Path

import pytest

from slipstream.config import BenchmarkConfig, RunSpec, load_config, parse_duration
from slipstream.models import Score
from slipstream.runners import BenchServer, BrowserRunner, Command, RunRequest
from slipstream.runners.report import ReportError, parse_report
from test_runners import _engine

FAKE_BROWSER = Path(__file__).with_name("fake_browser.py")


def _metric(v):
    return {"current": [v]}


def _js2_report(**totals):
    tests = {
        name: {
            "metrics": {"Score": _metric(total), "Time": ["Geometric"]},
            "tests": {
                "First": {
                    "metrics": {"Score": _metric(total / 2), "Time": _metric(10)}
                },
                "Average": {
                    "metrics": {"Score": _metric(total * 2), "Time": _metric(5)}
                },
            },
        }
        for name, total in totals.items()
    }
    return {"JetStream2.0": {"metrics": {"Score": ["Geometric"]}, "tests": tests}}


class TestParseReport:
    def test_js2_subscores_take_the_shells_names(self):
        body = json.dumps(_js2_report(Air=4.0)).encode()
        scores = parse_report(body, "js2", "default", 1, {"First": "Startup-Score"})
        assert scores == [
            Score("js2", "default", "Air", "Total-Score", 1, 4.0),
            Score("js2", "default", "Air", "Startup-Score", 1, 2.0),
            Score("js2", "default", "Air", "Average-Score", 1, 8.0),
        ]

    def test_js3_reports_the_subscore_under_time(self):
        """run-benchmark wants a Time there, so that is where JetStream 3
        puts the score; there is no separate time to confuse it with."""
        doc = {
            "JetStream3.0": {
                "tests": {
                    "Air": {
                        "metrics": {"Score": _metric(4.0)},
                        "tests": {"Worst": {"metrics": {"Time": _metric(3.0)}}},
                    }
                }
            }
        }
        scores = parse_report(json.dumps(doc).encode(), "js3", "default", 2, {})
        assert scores == [
            Score("js3", "default", "Air", "Total-Score", 2, 4.0),
            Score("js3", "default", "Air", "Worst-Score", 2, 3.0),
        ]

    def test_a_failed_benchmark_has_no_score(self):
        doc = _js2_report(Air=4.0)
        doc["JetStream2.0"]["tests"]["Air"]["metrics"]["Score"] = _metric(None)
        scores = parse_report(json.dumps(doc).encode(), "js2", "default", 1, {})
        assert [s.metric for s in scores] == ["First-Score", "Average-Score"]

    @pytest.mark.parametrize(
        "body", [b"not json", b"[]", b"{}", b'{"a": 1, "b": 2}', b'{"JS": {"x": 1}}']
    )
    def test_anything_else_is_an_error(self, body):
        with pytest.raises(ReportError):
            parse_report(body, "js2", "default", 1, {})


class TestBenchServer:
    def test_serves_the_suite_with_the_types_a_browser_needs(self, tmp_path):
        (tmp_path / "index.html").write_text("<html>")
        (tmp_path / "m.wasm").write_bytes(b"\0asm")
        with BenchServer(tmp_path) as server:
            with urllib.request.urlopen(server.url("index.html", report="true")) as r:
                assert r.read() == b"<html>"
                assert r.headers["Cache-Control"] == "no-store"
            with urllib.request.urlopen(server.url("m.wasm")) as r:
                assert r.headers["Content-Type"] == "application/wasm"
        assert any("GET /index.html?report=true" in line for line in server.requests)

    def test_takes_the_report(self, tmp_path):
        with BenchServer(tmp_path) as server:
            assert server.wait_for_report(0.05) is None
            req = urllib.request.Request(
                server.url("report"), data=b'{"ok": 1}', method="POST"
            )
            with urllib.request.urlopen(req) as r:
                assert r.status == 200
            assert server.wait_for_report(5) == b'{"ok": 1}'

    def test_other_posts_are_not_reports(self, tmp_path):
        with BenchServer(tmp_path) as server:
            req = urllib.request.Request(server.url("x"), data=b"1", method="POST")
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(req)
            assert exc.value.code == 404
            assert server.report is None


class _FakeBrowser(BrowserRunner):
    def __init__(self, mode="report", report=None, **kw):
        super().__init__(log=lambda m: self.logged.append(m), progress=lambda: None)
        self.logged: list[str] = []
        self.cleaned = 0
        self.mode = mode
        self.report = report
        self.pid = None

    def command(self, req, url):
        env = dict(os.environ, FAKE_BROWSER=self.mode)
        if self.report is not None:
            env["FAKE_REPORT"] = json.dumps(self.report)
        return Command([sys.executable, str(FAKE_BROWSER), url], env=env)

    def cleanup(self, req):
        self.cleaned += 1

    def terminate(self, proc):
        self.pid = proc.pid
        super().terminate(proc)


@pytest.fixture
def req(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "index.html").write_text("<html>")
    bench = BenchmarkConfig(
        name="js2",
        dir=suite,
        cli="cli.js",
        names=["Air", "Box2D", "Overall"],
        score_regex="",
        timeout="2s",
        report_metrics={"First": "Startup-Score"},
    )
    res = tmp_path / "res"
    res.mkdir()
    return RunRequest(
        engine=_engine(runtime="chromium"),
        run_root=tmp_path,
        bench=bench,
        spec=RunSpec(engine="v8", suite="js2", variant="default"),
        run=1,
        res_dir=res,
    )


def _gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # Reaped by us but still a zombie until waited; waitpid confirms.
    try:
        return os.waitpid(pid, os.WNOHANG)[0] == pid
    except ChildProcessError:
        return True


class TestBrowserRunner:
    def test_a_report_becomes_scores(self, req):
        runner = _FakeBrowser(report=_js2_report(Air=4.0, Box2D=9.0))
        result = runner.run(req)
        assert result.ok, runner.logged
        by_key = {(s.benchmark, s.metric): s.score for s in result.scores}
        assert by_key[("Air", "Total-Score")] == 4.0
        assert by_key[("Air", "Startup-Score")] == 2.0
        assert by_key[("Box2D", "Average-Score")] == 18.0
        assert by_key[("Overall", "Total-Score")] == 6.0
        assert json.loads(req.artifact("report", "json").read_text()) == _js2_report(
            Air=4.0, Box2D=9.0
        )
        stderr = req.artifact("stderr", "txt").read_text()
        assert stderr.startswith(f"$ {sys.executable} ")
        assert "--- requests ---" in stderr
        assert "GET /index.html?report=true" in stderr
        assert "POST /report" in stderr
        assert runner.cleaned == 1
        assert _gone(runner.pid)

    def test_a_browser_that_dies_fails_the_config(self, req):
        runner = _FakeBrowser(mode="exit")
        result = runner.run(req)
        assert not result.ok and result.scores == []
        assert any("exited (3) before reporting" in m for m in runner.logged)
        assert not req.artifact("report", "json").exists()
        assert runner.cleaned == 1

    def test_a_browser_that_never_reports_is_killed_at_the_timeout(self, req):
        runner = _FakeBrowser(mode="hang")
        result = runner.run(req)
        assert not result.ok
        assert any("no report within 2s" in m for m in runner.logged)
        assert _gone(runner.pid)

    def test_a_report_missing_a_benchmark_is_not_a_clean_run(self, req):
        runner = _FakeBrowser(report=_js2_report(Air=4.0))
        result = runner.run(req)
        assert not result.ok
        assert any("missing ['Box2D']" in m for m in runner.logged)
        assert all(s.benchmark != "Overall" for s in result.scores)

    def test_an_unreadable_report_fails(self, req):
        runner = _FakeBrowser(report={"JetStream2.0": {}})
        result = runner.run(req)
        assert not result.ok
        assert any("no tests" in m for m in runner.logged)

    def test_a_browser_that_cannot_start_fails(self, req):
        class Missing(_FakeBrowser):
            def command(self, req, url):
                return Command(["/nonexistent/browser", url])

        runner = Missing()
        result = runner.run(req)
        assert not result.ok
        assert any("cannot start browser" in m for m in runner.logged)
        assert runner.cleaned == 1

    def test_terminate_escalates_to_sigkill(self, monkeypatch):
        """A browser that ignores SIGTERM must still be gone before the
        next run starts."""
        import subprocess

        import slipstream.runners.browser as browser

        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,sys,time;signal.signal(15,signal.SIG_IGN);"
                "print('ready',flush=True);time.sleep(60)",
            ],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        assert proc.stdout.readline() == b"ready\n"
        monkeypatch.setattr(browser, "GRACE_SECONDS", 0.2)
        _FakeBrowser().terminate(proc)
        assert proc.returncode == -signal.SIGKILL


class TestConfig:
    def test_js2_renames_and_js3_does_not(self, tmp_path):
        for s in ("js2", "js3"):
            (tmp_path / s).mkdir()
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f'out_dir = "{tmp_path}"\n'
            f'[benchmarks.js2]\ndir = "{tmp_path}/js2"\n'
            f'[benchmarks.js3]\ndir = "{tmp_path}/js3"\n'
        )
        config = load_config(cfg)
        assert config.benchmarks["js2"].report_metrics == {
            "First": "Startup-Score",
            "Worst": "Worst-Case-Score",
            "MainRun": "Tests-Score",
            "Runtime": "Run-Time-Score",
        }
        assert config.benchmarks["js3"].report_metrics == {}
        assert config.benchmarks["js3"].timeout_seconds == 900

    @pytest.mark.parametrize(
        "text,seconds", [("15m", 900), ("90", 90), ("2s", 2), ("1.5h", 5400)]
    )
    def test_durations(self, text, seconds):
        assert parse_duration(text) == seconds

    def test_a_bad_duration_is_rejected(self):
        with pytest.raises(ValueError, match="not a duration"):
            parse_duration("soon")
