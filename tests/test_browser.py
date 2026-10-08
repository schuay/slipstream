# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import urllib.request
from http.client import IncompleteRead
from pathlib import Path

import pytest

from slipstream.config import BenchmarkConfig, RunSpec, load_config, parse_duration
from slipstream.models import Score
from slipstream.runners import BenchServer, BrowserRunner, Command, RunRequest
from slipstream.runners.report import (
    ReportError,
    parse_for,
    parse_report,
    parse_speedometer,
)
from slipstream.runners.server import _Handler, inject_tag, injected
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


def _sp3_metric(mean, unit="ms"):
    return {"unit": unit, "mean": mean, "delta": 0, "sum": mean, "values": [mean]}


def _sp3_report(score=20.0, **suites):
    """What sp3-report.mjs posts: benchmarkClient.metrics, flattened the
    way Speedometer flattens it, with the nested and per-iteration
    entries a real run also has."""
    metrics = {}
    for name, mean in suites.items():
        metrics[name] = _sp3_metric(mean)
        metrics[f"{name}/Adding100Items"] = _sp3_metric(mean / 2)
        metrics[f"{name}/Adding100Items/sync"] = _sp3_metric(mean / 4)
    metrics["Iteration-0-Total"] = _sp3_metric(sum(suites.values()))
    metrics["Geomean"] = _sp3_metric(50.0)
    metrics["Score"] = _sp3_metric(score, unit="score")
    return {"metrics": metrics}


class TestParseSpeedometer:
    def test_a_suites_mean_is_its_time_and_score_is_the_overall(self):
        body = json.dumps(_sp3_report(20.0, Charts=100.0, Editors=200.0)).encode()
        scores = parse_speedometer(body, "sp3", "default", 1, {})
        assert scores == [
            Score("sp3", "default", "Charts", "Total-Time", 1, 100.0),
            Score("sp3", "default", "Editors", "Total-Time", 1, 200.0),
            Score("sp3", "default", "Overall", "Total-Score", 1, 20.0),
        ]

    def test_only_what_crossbench_keeps_is_kept(self):
        """Steps, Iteration-N-Total and Geomean are left out; the perf
        database wants the same traces crossbench's runs produce."""
        body = json.dumps(_sp3_report(20.0, Charts=100.0)).encode()
        kept = {
            (s.benchmark, s.metric) for s in parse_speedometer(body, "sp3", "", 1, {})
        }
        assert kept == {("Charts", "Total-Time"), ("Overall", "Total-Score")}

    def test_a_suite_that_never_ran_has_no_time(self):
        doc = _sp3_report(20.0, Charts=100.0)
        doc["metrics"]["Editors"] = _sp3_metric(float("nan"))
        doc["metrics"]["Mail"] = _sp3_metric(None)
        scores = parse_speedometer(json.dumps(doc).encode(), "sp3", "", 1, {})
        assert [s.benchmark for s in scores] == ["Charts", "Overall"]

    def test_the_pages_error_is_the_message(self):
        body = json.dumps({"error": {"message": "boom", "stack": "at x"}}).encode()
        with pytest.raises(ReportError, match="the page reported an error: boom"):
            parse_speedometer(body, "sp3", "", 1, {})

    def test_a_report_without_the_overall_is_not_a_report(self):
        doc = _sp3_report(20.0, Charts=100.0)
        del doc["metrics"]["Score"]
        with pytest.raises(ReportError, match="no Score"):
            parse_speedometer(json.dumps(doc).encode(), "sp3", "", 1, {})

    @pytest.mark.parametrize(
        "body", [b"not json", b"[]", b"{}", b'{"metrics": {}}', b'{"metrics": 1}']
    )
    def test_anything_else_is_an_error(self, body):
        with pytest.raises(ReportError):
            parse_speedometer(body, "sp3", "", 1, {})

    def test_the_format_picks_the_parser(self):
        assert parse_for("jetstream") is parse_report
        assert parse_for("speedometer") is parse_speedometer
        with pytest.raises(ValueError, match="no parser"):
            parse_for("octane")


class TestBenchServer:
    @pytest.mark.parametrize("error", [ConnectionResetError, BrokenPipeError])
    def test_client_disconnect_is_logged_without_a_traceback(
        self, tmp_path, monkeypatch, capsys, error
    ):
        (tmp_path / "index.html").write_text("response body")

        def disconnect(*args):
            raise error("client canceled the response")

        monkeypatch.setattr(_Handler, "copyfile", disconnect)
        with BenchServer(tmp_path) as server:
            with urllib.request.urlopen(server.url("index.html")) as response:
                with pytest.raises(IncompleteRead):
                    response.read()
            assert any(
                f"client disconnected: {error.__name__}" in line
                for line in server.requests
            )
            req = urllib.request.Request(server.url("report"), data=b"{}")
            with urllib.request.urlopen(req) as response:
                assert response.status == 200
            assert server.wait_for_report(1) == b"{}"
        assert capsys.readouterr().err == ""

    def test_unexpected_server_errors_still_print_a_traceback(self, tmp_path, capsys):
        server = BenchServer(tmp_path)
        try:
            try:
                raise RuntimeError("unexpected server error")
            except RuntimeError:
                server.handle_error(None, ("127.0.0.1", 1234))
        finally:
            server.server_close()
        assert "RuntimeError: unexpected server error" in capsys.readouterr().err

    def test_queues_a_burst_of_browser_connections(self, tmp_path):
        # Pause accepting to reproduce a browser's burst of parallel assets.
        # The old five-connection backlog drops connections on macOS.
        server = BenchServer(tmp_path)
        clients = []
        try:
            for _ in range(16):
                clients.append(
                    socket.create_connection(("127.0.0.1", server.port), timeout=1)
                )
        finally:
            for client in clients:
                client.close()
            server.server_close()

    def test_serves_the_suite_with_the_types_a_browser_needs(self, tmp_path):
        (tmp_path / "index.html").write_text("<html>")
        (tmp_path / "m.wasm").write_bytes(b"\0asm")
        with BenchServer(tmp_path) as server:
            with urllib.request.urlopen(server.url("index.html", report="true")) as r:
                assert r.read() == b"<html>"
                assert r.headers["Cache-Control"] == "no-store"
                assert r.headers["Cross-Origin-Opener-Policy"] == "same-origin"
                assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
            with urllib.request.urlopen(server.url("m.wasm")) as r:
                assert r.headers["Content-Type"] == "application/wasm"
                assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
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

    def test_page_url_spells_the_query_as_the_suite_does(self, tmp_path):
        with BenchServer(tmp_path) as server:
            assert server.page_url("index.html", "startAutomatically=true").endswith(
                "/index.html?startAutomatically=true"
            )
            assert server.page_url("index.html", "").endswith("/index.html")

    def test_the_inject_script_is_appended_to_the_page_only(self, tmp_path):
        """The suite's checkout is served as is, except that the one page
        the browser is sent to gets the module tag before </body>; the
        script itself comes from the package, not the checkout."""
        from importlib.resources import files

        page = b"<html><body><h1>hi</h1>\n</body></html>\n"
        (tmp_path / "index.html").write_bytes(page)
        (tmp_path / "other.html").write_bytes(page)
        with BenchServer(tmp_path, inject="sp3-report.mjs") as server:
            with urllib.request.urlopen(server.url("index.html")) as r:
                html = r.read()
                assert r.headers["Content-Type"] == "text/html"
                assert r.headers["Cross-Origin-Opener-Policy"] == "same-origin"
                assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
            with urllib.request.urlopen(server.url("other.html")) as r:
                assert r.read() == page
                assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
            with urllib.request.urlopen(server.url("__slipstream/sp3-report.mjs")) as r:
                assert r.headers["Content-Type"] == "text/javascript"
                assert r.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
                script = r.read()
        tag = inject_tag("sp3-report.mjs")
        assert html.count(tag) == 1
        assert html.index(tag) < html.index(b"</body>")
        assert html.replace(tag + b"\n", b"") == page
        assert (
            script == files("slipstream.data").joinpath("sp3-report.mjs").read_bytes()
        )
        assert b"didFinishLastIteration" in script and b"/report" in script

    def test_a_page_without_a_body_tag_still_gets_the_script(self):
        assert injected(b"<p>x</p>", "s.mjs").endswith(inject_tag("s.mjs") + b"\n")

    def test_without_an_inject_nothing_is_rewritten(self, tmp_path):
        (tmp_path / "index.html").write_bytes(b"<body></body>")
        with BenchServer(tmp_path) as server:
            with urllib.request.urlopen(server.url("index.html")) as r:
                assert r.read() == b"<body></body>"
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(server.url("__slipstream/sp3-report.mjs"))
            assert exc.value.code == 404


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
def req(tmp_path, monkeypatch):
    """A run request against the fake browser. The poll interval is the
    granularity of every wait in the runner, so it is shortened to keep
    these tests fast rather than half a second each."""
    import slipstream.runners.browser as browser

    monkeypatch.setattr(browser, "POLL_SECONDS", 0.02)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "index.html").write_text("<html>")
    bench = BenchmarkConfig(
        name="js2",
        dir=suite,
        cli="cli.js",
        names=["Air", "Box2D", "Overall"],
        score_regex="",
        timeout="0.3s",
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


@pytest.fixture
def sp3_req(req):
    """The same request for a Speedometer run: no shell, a page the server
    appends the reporter to, and the overall coming from the page."""
    bench = BenchmarkConfig(
        name="sp3",
        dir=req.bench.dir,
        names=["Charts", "Editors", "Overall"],
        timeout="0.3s",
        query="startAutomatically=true",
        inject="sp3-report.mjs",
        report_format="speedometer",
    )
    spec = RunSpec(engine="chrome", suite="sp3", variant="default")
    return RunRequest(**{**req.__dict__, "bench": bench, "spec": spec})


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
        assert any("no report within 0.3s" in m for m in runner.logged)
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


class TestSpeedometerRun:
    def test_the_page_is_fetched_with_the_reporter_and_its_overall_kept(self, sp3_req):
        runner = _FakeBrowser(report=_sp3_report(20.0, Charts=100.0, Editors=200.0))
        result = runner.run(sp3_req)
        assert result.ok, runner.logged
        by_key = {(s.benchmark, s.metric): s.score for s in result.scores}
        assert by_key == {
            ("Charts", "Total-Time"): 100.0,
            ("Editors", "Total-Time"): 200.0,
            # Speedometer's own Score, not a geomean of the times above.
            ("Overall", "Total-Score"): 20.0,
        }
        stderr = sp3_req.artifact("stderr", "txt").read_text()
        assert "GET /index.html?startAutomatically=true" in stderr
        assert "GET /__slipstream/sp3-report.mjs" in stderr
        assert "POST /report" in stderr

    def test_the_pages_error_fails_the_config(self, sp3_req):
        runner = _FakeBrowser(report={"error": {"message": "boom", "stack": ""}})
        result = runner.run(sp3_req)
        assert not result.ok and result.scores == []
        assert any("the page reported an error: boom" in m for m in runner.logged)

    def test_a_suite_the_page_did_not_time_is_missing(self, sp3_req):
        runner = _FakeBrowser(report=_sp3_report(20.0, Charts=100.0))
        result = runner.run(sp3_req)
        assert not result.ok
        assert any("missing ['Editors']" in m for m in runner.logged)


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

    def test_js3_names_include_the_worker_only_benchmarks(self, tmp_path):
        # The shell never runs them, but a browser report must have them and
        # the delivery filter must let them through, and both read this list.
        (tmp_path / "js3").mkdir()
        cfg = tmp_path / "config.toml"
        cfg.write_text(
            f'out_dir = "{tmp_path}"\n[benchmarks.js3]\ndir = "{tmp_path}/js3"\n'
        )
        names = set(load_config(cfg).benchmarks["js3"].names)
        assert {"bomb-workers", "segmentation"} <= names

    @pytest.mark.parametrize(
        "text,seconds", [("15m", 900), ("90", 90), ("2s", 2), ("1.5h", 5400)]
    )
    def test_durations(self, text, seconds):
        assert parse_duration(text) == seconds

    def test_a_bad_duration_is_rejected(self):
        with pytest.raises(ValueError, match="not a duration"):
            parse_duration("soon")


class TestChromiumRunner:
    def _runner(self):
        from slipstream.runners import ChromiumRunner

        return ChromiumRunner(log=lambda m: None, progress=lambda: None)

    def test_the_command_is_the_built_bundle_with_a_fresh_profile(self, req):
        from slipstream.runners.chromium import CHROMIUM_FLAGS

        spec = RunSpec(
            engine="chrome", suite="js2", flags=("--js-flags=--maglev",), variant="mg"
        )
        req = RunRequest(**{**req.__dict__, "spec": spec})
        runner = self._runner()
        cmd = runner.command(req, "http://127.0.0.1:1/index.html?report=true")

        assert cmd.argv[0] == str(req.run_root / "out/d8")
        profile = cmd.argv[1].removeprefix("--user-data-dir=")
        assert Path(profile).is_dir() and not os.listdir(profile)
        assert cmd.argv[2 : 2 + len(CHROMIUM_FLAGS)] == list(CHROMIUM_FLAGS)
        assert "--headless=new" in cmd.argv
        assert cmd.argv[-2:] == [
            "--js-flags=--maglev",
            "http://127.0.0.1:1/index.html?report=true",
        ]

        runner.cleanup(req)
        assert not Path(profile).exists()

    def test_field_trials_are_off_and_benchmarking_mode_on(self):
        """A non-branded build applies fieldtrial_testing_config.json unless
        told not to, and that file moves with the tree."""
        from slipstream.runners.chromium import CHROMIUM_FLAGS

        assert "--disable-field-trial-config" in CHROMIUM_FLAGS
        assert "--enable-benchmarking" in CHROMIUM_FLAGS

    def test_headless_has_the_viewport_crossbench_gives_it(self):
        """Speedometer does layout; a headless window of some default
        size would measure something else than crossbench's 1500x1000."""
        from slipstream.runners.chromium import CHROMIUM_FLAGS

        assert "--headless=new" in CHROMIUM_FLAGS
        assert "--window-size=1500,1000" in CHROMIUM_FLAGS

    def test_cleanup_without_a_command_is_fine(self, req):
        self._runner().cleanup(req)

    def test_a_real_run_through_the_fake_browser(self, req, monkeypatch):
        """The whole path with ChromiumRunner's command: the fake stands in
        for the binary at binary_path."""
        from slipstream.runners import ChromiumRunner

        out = req.run_root / "out"
        out.mkdir()
        binary = out / "d8"
        binary.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE_BROWSER} "$@"\n')
        binary.chmod(0o755)
        monkeypatch.setenv("FAKE_REPORT", json.dumps(_js2_report(Air=4.0, Box2D=9.0)))

        runner = ChromiumRunner(log=lambda m: None, progress=lambda: None)
        result = runner.run(req)
        assert result.ok
        assert ("Overall", "Total-Score", 6.0) in {
            (s.benchmark, s.metric, s.score) for s in result.scores
        }
        stderr = req.artifact("stderr", "txt").read_text()
        assert "--disable-field-trial-config" in stderr
        assert runner._profile is None
