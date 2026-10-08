# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The Safari runner: launch through the host's development launcher, and
accept a run only on evidence that the processes ran the run root's engine
inside the run root's STP."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from slipstream.config import BenchmarkConfig, EngineConfig, RunSpec
from slipstream.runners import Command, RunRequest
from slipstream.runners import safari as safari_mod
from slipstream.runners.safari import SafariRunner
from test_browser import FAKE_BROWSER, _js2_report

LAUNCHER = "/Applications/Safari.app/Contents/MacOS/SafariForWebKitDevelopment"


def _safari_engine():
    return EngineConfig(
        name="safari",
        src_dir=None,
        build_cmd="",
        binary_path=LAUNCHER,
        id_regex="",
        dyld_lib_path=[
            "WebKitBuild/Release",
            "Safari Technology Preview.app/Contents/Frameworks",
        ],
        run_set=["Safari Technology Preview.app"],
        runtime="safari",
        derives="jsc",
    )


@pytest.fixture
def req(tmp_path, monkeypatch):
    import slipstream.runners.browser as browser

    monkeypatch.setattr(browser, "POLL_SECONDS", 0.02)
    monkeypatch.setattr(safari_mod, "QUIET_SECONDS", 0.01)
    monkeypatch.setattr(safari_mod, "POLL_SECONDS", 0.002)
    monkeypatch.setattr(safari_mod, "GRACE_SECONDS", 0.05)
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "index.html").write_text("<html>")
    bench = BenchmarkConfig(
        name="js2",
        dir=suite,
        cli="cli.js",
        names=["Air", "Box2D", "Overall"],
        score_regex="",
        timeout="0.5s",
    )
    res = tmp_path / "res"
    res.mkdir()
    return RunRequest(
        engine=_safari_engine(),
        run_root=tmp_path / "root",
        bench=bench,
        spec=RunSpec(engine="safari", suite="js2", variant="default"),
        run=1,
        res_dir=res,
    )


class _Host:
    """What the machine looks like: a process table and each process's
    mapped images. ``root`` is the run root the images should come from."""

    def __init__(self, root: Path, launcher_pid: int, live: bool = True):
        self.root = root
        # A host that is not live yet shows an empty process table, as the
        # machine must before a launch; ``_Runner.command`` makes it live.
        self.live = live
        self.procs: list[tuple[int, str]] = [(launcher_pid, LAUNCHER)]
        self.images: dict[int, list[str]] = {
            launcher_pid: [
                f"{root}/Safari Technology Preview.app/Contents/Frameworks/"
                "Safari.framework/Versions/A/Safari"
            ]
        }

    def content(self, pid: int, jsc_dir: str | None = None) -> None:
        self.procs.append(
            (
                pid,
                f"{jsc_dir or self.root}/WebKitBuild/Release/"
                "com.apple.WebKit.WebContent.xpc/Contents/MacOS/com.apple.WebKit.WebContent",
            )
        )
        base = jsc_dir or f"{self.root}/WebKitBuild/Release"
        self.images[pid] = [
            f"{base}/JavaScriptCore.framework/Versions/A/JavaScriptCore"
        ]


class _Runner(SafariRunner):
    """The real runner over a scripted host; the browser is the fake."""

    def __init__(self, host: _Host | None = None, report=None, mode="report"):
        super().__init__(log=lambda m: self.logged.append(m), progress=lambda: None)
        self.logged: list[str] = []
        self.host = host
        self.report = report
        self.mode = mode
        self.real_command = False

    def command(self, req, url):
        if self.host:
            self.host.live = True
        if self.real_command:
            return super().command(req, url)
        env = dict(os.environ, FAKE_BROWSER=self.mode)
        if self.report is not None:
            env["FAKE_REPORT"] = json.dumps(self.report)
        return Command([sys.executable, str(FAKE_BROWSER), url], env=env)

    def processes(self):
        return list(self.host.procs) if self.host and self.host.live else []

    def _signal_process(self, pid, name, sig):
        if self.host:
            self.host.procs = [p for p in self.host.procs if p != (pid, name)]

    def identity(self, pid):
        return (
            (pid, 0)
            if self.host and any(p == pid for p, _ in self.host.procs)
            else None
        )

    def domain_helpers(self, pid):
        return {p for p, n in self.host.procs if "com.apple.WebKit." in n}

    def loaded_images(self, pid):
        return list(self.host.images.get(pid, [])) if self.host else []


class _Proc:
    def __init__(self, pid):
        self.pid = pid


class TestCommand:
    def test_on_macos_the_env_rides_on_arch(self, req, monkeypatch):
        monkeypatch.setattr(safari_mod, "_is_macos", lambda: True)
        runner = _Runner()
        runner.real_command = True
        cmd = runner.command(req, "http://127.0.0.1:1/index.html?report=true")
        root = req.run_root
        search = f"{root}/WebKitBuild/Release:{root}/Safari Technology Preview.app/Contents/Frameworks"
        assert cmd.argv[:2] == ["/usr/bin/arch", "-arm64e"]
        pairs = cmd.argv[2:10]
        assert pairs[0::2] == ["-e"] * 4
        assert pairs[1::2] == [
            f"DYLD_FRAMEWORK_PATH={search}",
            f"DYLD_LIBRARY_PATH={search}",
            f"__XPC_DYLD_FRAMEWORK_PATH={search}",
            f"__XPC_DYLD_LIBRARY_PATH={search}",
        ]
        assert cmd.argv[10:13] == [
            LAUNCHER,
            "-HomePage",
            "http://127.0.0.1:1/index.html?report=true",
        ]
        assert cmd.argv[13:] == list(safari_mod.LAUNCH_ARGS)
        assert "DYLD_FRAMEWORK_PATH" not in cmd.env  # SIP would drop it anyway

    def test_elsewhere_the_env_is_the_env(self, req, monkeypatch):
        monkeypatch.setattr(safari_mod, "_is_macos", lambda: False)
        runner = _Runner()
        runner.real_command = True
        cmd = runner.command(req, "u")
        assert cmd.argv[0] == LAUNCHER
        assert cmd.env["__XPC_DYLD_LIBRARY_PATH"].startswith(str(req.run_root))


class TestPrecondition:
    @pytest.mark.parametrize(
        "name",
        [
            "/System/Cryptexes/App/System/Applications/Safari.app/Contents/MacOS/Safari",
            LAUNCHER,
            "/Applications/Safari Technology Preview.app/Contents/MacOS/Safari Technology Preview",
        ],
    )
    def test_stale_safari_processes_are_drained_before_launch(self, req, name):
        host = _Host(req.run_root, 1)
        host.procs = [(4242, name)]
        runner = _Runner(host)
        assert runner.precondition(req) is None
        assert host.procs == []
        assert any("SIGTERM" in line and "(4242)" in line for line in runner.logged)

    def test_a_quiet_machine_passes(self, req):
        host = _Host(req.run_root, 1)
        host.procs = [(1, "/sbin/launchd"), (77, "/usr/bin/ssh-agent")]
        assert _Runner(host).precondition(req) is None


class TestVerify:
    def test_passes_when_both_layers_come_from_the_root(self, req):
        host = _Host(req.run_root, 100)
        host.content(200)
        host.content(201)
        verdict = _Runner(host).verify(req, _Proc(100))
        assert verdict.ok
        assert verdict.notes[0].startswith("launcher 100: Safari.framework from")
        assert [n.split(":")[0] for n in verdict.notes[1:]] == [
            "WebContent 200",
            "WebContent 201",
        ]

    def test_the_launcher_on_another_safari_framework_fails(self, req):
        host = _Host(req.run_root, 100)
        host.images[100] = [
            "/Applications/Safari Technology Preview.app/Contents/Frameworks/"
            "Safari.framework/Versions/A/Safari"
        ]
        host.content(200)
        verdict = _Runner(host).verify(req, _Proc(100))
        assert not verdict.ok and "not the run root's" in verdict.problem

    def test_no_content_process_with_an_engine_fails(self, req):
        host = _Host(req.run_root, 100)
        verdict = _Runner(host).verify(req, _Proc(100))
        assert not verdict.ok and "no WebContent process" in verdict.problem

    def test_one_content_process_on_the_system_engine_fails_all(self, req):
        host = _Host(req.run_root, 100)
        host.content(200)
        host.content(201, jsc_dir="/System/Library/Frameworks")
        verdict = _Runner(host).verify(req, _Proc(100))
        assert not verdict.ok
        assert "201: /System/Library/Frameworks" in verdict.problem

    def test_tmp_and_private_tmp_are_the_same_root(self, req, tmp_path):
        """Image inspection reports realpaths; the run root may be spelled through a
        symlink (macOS /tmp -> /private/tmp)."""
        link = tmp_path / "link"
        link.symlink_to(req.run_root)
        host = _Host(req.run_root, 100)  # images under the real path
        host.content(200)
        from dataclasses import replace

        verdict = _Runner(host).verify(replace(req, run_root=link), _Proc(100))
        assert verdict.ok


class TestRun:
    def test_a_verified_run_is_scores_with_the_evidence_on_file(self, req):
        host = _Host(req.run_root, 0, live=False)
        runner = _Runner(host, report=_js2_report(Air=4.0, Box2D=9.0))
        # The launcher pid is only known once launched; the host learns it.
        real_verify = runner.verify

        def verify(req_, proc):
            host.procs[0] = (proc.pid, LAUNCHER)
            host.images[proc.pid] = host.images.pop(0, host.images.get(proc.pid, []))
            if not any(p for p, _ in host.procs if p == 555):
                host.content(555)
            return real_verify(req_, proc)

        runner.verify = verify
        result = runner.run(req)
        assert result.ok, runner.logged
        stderr = req.artifact("stderr", "txt").read_text()
        assert "provenance: launcher" in stderr
        assert "provenance: WebContent 555: JavaScriptCore from" in stderr

    def test_a_failed_check_at_report_time_fails_the_config(self, req):
        host = _Host(
            req.run_root, 0, live=False
        )  # launcher images under pid 0: never match
        runner = _Runner(host, report=_js2_report(Air=4.0, Box2D=9.0))
        result = runner.run(req)
        assert not result.ok and result.scores == []
        assert any(
            "provenance: the launcher runs no Safari.framework" in m
            for m in runner.logged
        )
        assert not req.artifact("report", "json").exists()

    def test_the_early_check_fails_fast(self, req, monkeypatch):
        import slipstream.runners.browser as browser

        monkeypatch.setattr(browser, "EARLY_CHECK_SECONDS", 0.0)
        from dataclasses import replace

        long_req = replace(req, bench=replace(req.bench, timeout="10s"))
        runner = _Runner(_Host(req.run_root, 0, live=False), mode="hang")
        t0 = time.monotonic()
        result = runner.run(long_req)
        assert not result.ok
        assert time.monotonic() - t0 < 5
        assert any("provenance:" in m for m in runner.logged)

    def test_terminate_takes_the_identified_content_processes_too(
        self, req, monkeypatch
    ):
        import slipstream.runners.browser as browser

        monkeypatch.setattr(browser, "GRACE_SECONDS", 0.5)
        monkeypatch.setattr(safari_mod, "GRACE_SECONDS", 0.5)
        monkeypatch.setattr(safari_mod, "_is_macos", lambda: False)
        sleeper = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        launcher = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        host = _Host(req.run_root, launcher.pid)
        host.content(sleeper.pid)
        runner = _Runner(host)
        assert runner.verify(req, launcher).ok

        def signal_process(pid, name, sig):
            assert pid in (launcher.pid, sleeper.pid)
            safari_mod._kill(pid, sig)
            runner.host.procs = [p for p in runner.host.procs if p != (pid, name)]

        runner._signal_process = signal_process
        runner.terminate(launcher)
        assert launcher.poll() is not None
        try:
            sleeper.wait(2)
        except subprocess.TimeoutExpired:
            sleeper.kill()
            pytest.fail("the content process outlived terminate()")
        assert runner._content_pids == []


class TestRecovery:
    def test_a_process_ignoring_term_is_killed(self, req):
        host = _Host(req.run_root, 4242)
        runner = _Runner(host)
        signals = []

        def signal_process(pid, name, sig):
            signals.append(sig)
            if sig == signal.SIGKILL:
                host.procs.clear()

        runner._signal_process = signal_process
        assert runner.precondition(req) is None
        assert signals == [signal.SIGTERM, signal.SIGKILL]

    def test_a_late_helper_resets_the_quiet_wait(self, req):
        host = _Host(req.run_root, 4242)
        runner = _Runner(host)
        scans = 0

        def processes():
            nonlocal scans
            scans += 1
            if scans == 3:
                host.content(4243)
            return list(host.procs)

        runner.processes = processes
        assert runner.precondition(req) is None
        assert host.procs == []
        assert any("(4243)" in line for line in runner.logged)

    def test_cleanup_also_runs_when_provenance_failed(self, req):
        host = _Host(req.run_root, 4242)
        host.content(4243, jsc_dir="/System/Library/Frameworks")
        runner = _Runner(host)
        host.images[4242] = []
        assert not runner.verify(req, _Proc(4242)).ok
        assert runner._content_pids == []
        # The fake launcher already exited; neither it nor its group can
        # account for launchd's WebContent process.
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        runner.terminate(proc)
        assert host.procs == []
        assert any("(4243)" in line for line in runner.logged)

    def test_unkillable_process_blocks_launch_with_bounded_recovery(self, req):
        host = _Host(req.run_root, 4242)
        runner = _Runner(host)
        runner._signal_process = lambda *args: None
        start = time.monotonic()
        result = runner.run(req)
        assert not result.ok
        assert time.monotonic() - start < 1
        assert "quiet state" in req.artifact("stderr", "txt").read_text()

    def test_failed_process_inspection_blocks_launch(self, req):
        runner = _Runner()

        def processes():
            raise RuntimeError("ps unavailable")

        runner.processes = processes
        assert "ps unavailable" in runner.precondition(req)

    def test_process_scan_is_limited_to_current_user(self, monkeypatch):
        def run(argv, **kwargs):
            assert argv == ["ps", "-U", str(os.getuid()), "-o", "pid=,comm="]
            assert kwargs["timeout"] == safari_mod.GRACE_SECONDS
            return subprocess.CompletedProcess(argv, 0, "42 /sbin/launchd\n", "")

        monkeypatch.setattr(subprocess, "run", run)
        assert _Runner().processes() == []  # scripted host
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        assert runner.processes() == [(42, "/sbin/launchd")]

    def test_pid_reused_for_another_executable_is_not_signaled(self, monkeypatch):
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        runner.processes = lambda: [(4242, "/usr/bin/ssh")]
        killed = []
        monkeypatch.setattr(safari_mod, "_kill", lambda *args: killed.append(args))
        runner._signal_process(4242, LAUNCHER, signal.SIGTERM)
        assert killed == []


class TestRetry:
    def test_transient_provenance_failure_gets_a_clean_measurement(self, req):
        from slipstream.runners.browser import Verdict

        host = _Host(req.run_root, 0, live=False)
        runner = _Runner(host, report=_js2_report(Air=4.0, Box2D=9.0))
        attempts = 0
        original_command = runner.command
        original_verify = runner.verify

        def command(req_, url):
            host.procs = [(0, LAUNCHER)]
            return original_command(req_, url)

        def verify(req_, proc):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                host.content(555, jsc_dir="/System/Library/Frameworks")
                return Verdict(
                    False, [], "a stale content process loaded the system engine"
                )
            host.procs[0] = (proc.pid, LAUNCHER)
            host.images[proc.pid] = host.images[0]
            host.content(556)
            return original_verify(req_, proc)

        runner.command = command
        runner.verify = verify
        result = runner.run(req)
        assert result.ok, runner.logged
        assert attempts == 2
        assert any("retrying" in message for message in runner.logged)
        recovery = req.artifact("stderr-recovery", "txt").read_text()
        assert "stale content process" in recovery
        assert "WebContent 556" in req.artifact("stderr", "txt").read_text()

    @pytest.mark.parametrize(
        "problem,retried",
        [
            ("provenance: foreign engine", True),
            ("browser exited (1) before reporting", True),
            ("cannot start browser: launcher unavailable", True),
            ("the page reported an error: workload is broken", False),
            ("no report within 900s", False),
        ],
    )
    def test_only_environment_failures_retry_and_only_once(
        self, req, monkeypatch, problem, retried
    ):
        from slipstream.runners.base import RunResult
        from slipstream.runners.browser import BrowserRunner

        calls = []

        def run(self, request):
            calls.append(request)
            self._fail("test", problem)
            return RunResult(False, [])

        monkeypatch.setattr(BrowserRunner, "run", run)
        result = _Runner().run(req)
        assert not result.ok
        assert len(calls) == (2 if retried else 1)


class TestOwnership:
    def test_foreign_system_worker_is_neither_verified_nor_killed(self, req):
        host = _Host(req.run_root, 100)
        host.content(200)
        host.content(300, jsc_dir="/System/Library/Frameworks")
        runner = _Runner(host)
        runner.domain_helpers = lambda owner: {200}
        assert runner.verify(req, _Proc(100)).ok
        assert runner._content_pids == [200]
        assert runner.precondition(req) is None
        assert [pid for pid, _ in host.procs] == [300]
        assert not any("(300)" in line for line in runner.logged)

    def test_unattributed_orphan_is_not_killed(self, req):
        host = _Host(req.run_root, 100)
        host.content(300)
        host.procs = host.procs[1:]
        runner = _Runner(host)
        assert runner.precondition(req) is None
        assert [pid for pid, _ in host.procs] == [300]

    def test_missing_owned_engine_fails_even_with_a_good_worker(self, req):
        host = _Host(req.run_root, 100)
        host.content(200)
        host.content(201)
        host.images[201] = []
        verdict = _Runner(host).verify(req, _Proc(100))
        assert not verdict.ok
        assert "201: no JavaScriptCore" in verdict.problem

    def test_root_prefix_is_not_root_membership(self, req):
        host = _Host(req.run_root, 100)
        host.content(200, jsc_dir=f"{req.run_root}-other")
        assert not _Runner(host).verify(req, _Proc(100)).ok

    def test_same_executable_with_reused_pid_is_not_signaled(self, monkeypatch):
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        runner._owned[4242] = (LAUNCHER, (1, 2))
        runner.processes = lambda: [(4242, LAUNCHER)]
        runner.identity = lambda pid: (1, 3)
        killed = []
        monkeypatch.setattr(safari_mod, "_kill", lambda *args: killed.append(args))
        runner._signal_process(4242, LAUNCHER, signal.SIGKILL)
        assert killed == []

    def test_ownership_failure_blocks_cleanup_without_signals(self, req):
        host = _Host(req.run_root, 100)
        host.content(200)
        runner = _Runner(host)

        def domain(owner):
            raise RuntimeError("launchd unavailable")

        runner.domain_helpers = domain
        assert "launchd unavailable" in runner.precondition(req)
        assert len(host.procs) == 2

    def test_identity_changes_during_attribution_are_not_adopted(self, req):
        host = _Host(req.run_root, 100)
        host.content(200)
        runner = _Runner(host)
        identities = {100: (100, 0), 200: (200, 0)}
        runner.identity = identities.get

        def domain(owner):
            identities[200] = (200, 1)
            return {200}

        runner.domain_helpers = domain
        assert runner._discover(100) == set()
        assert 200 not in runner._owned

    def test_launchd_parser_uses_only_live_webkit_services(self, monkeypatch):
        output = """pid/100 = {
\tservices = {
\t       0      - com.apple.WebKit.WebContent
\t     201      - com.apple.WebKit.WebContent.Development.ABCD
\t     202      - com.apple.WebKit.GPU.ABCD
\t     300      - com.apple.unrelated
\t}
\tunmanaged processes = {
\t     400      - com.apple.WebKit.WebContent
\t}
}
"""

        def run(argv, **kwargs):
            assert argv == ["/bin/launchctl", "print", "pid/100"]
            return subprocess.CompletedProcess(argv, 0, output, "")

        monkeypatch.setattr(subprocess, "run", run)
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        assert runner.domain_helpers(100) == {201, 202}

    def test_image_reader_includes_shared_cache_paths_with_spaces(self, monkeypatch):
        path = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/JavaScriptCore"
        safari = "/Users/bench/bus/roots/safari/run/Safari Technology Preview.app/Contents/Frameworks/Safari.framework/Versions/A/Safari"

        def run(argv, **kwargs):
            assert argv == ["/usr/bin/vmmap", "-w", "100"]
            assert kwargs["timeout"] == 30.0
            output = (
                "REGION TYPE                    START - END         [ VSIZE] PRT/MAX SHRMOD  REGION DETAIL\n"
                f"__TEXT                      100-200 [ 16K 16K 0K 0K] r-x/r-x SM=COW  {path}\n"
                f"__TEXT                      300-400 [ 16K 16K 0K 0K] r-x/r-x SM=COW  {safari}\n"
                f"__LINKEDIT                  500-600 [ 16K 16K 0K 0K] r--/r-- SM=COW  {safari}\n"
                "==== Summary for process 100\n__TEXT 32K 2\n"
            )
            return subprocess.CompletedProcess(argv, 0, output, "")

        monkeypatch.setattr(subprocess, "run", run)
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        assert runner.loaded_images(100) == [path, safari]

    def test_image_inspection_failure_is_a_failed_verdict(self, req):
        host = _Host(req.run_root, 100)
        runner = _Runner(host)

        def images(pid):
            raise RuntimeError("vmmap failed")

        runner.loaded_images = images
        verdict = runner.verify(req, _Proc(100))
        assert not verdict.ok
        assert "vmmap failed" in verdict.problem

    @pytest.mark.parametrize(
        "output", ["", "pid/100 = {\n}", "pid/200 = {\n\tservices = {\n\t}\n}"]
    )
    def test_unrecognized_domain_output_fails_closed(self, monkeypatch, output):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, output, ""),
        )
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        with pytest.raises(RuntimeError, match="unrecognized launchd"):
            runner.domain_helpers(100)

    @pytest.mark.parametrize(
        "returncode, output, problem",
        [
            (0, "Summary only\n", "missing summary marker"),
            (0, "==== Summary for process 100\n", "no image paths"),
            (1, "", "vmmap failed"),
        ],
    )
    def test_unusable_vmmap_is_an_inspection_error(
        self, monkeypatch, returncode, output, problem
    ):
        def run(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv, returncode, output, "inspection error"
            )

        monkeypatch.setattr(subprocess, "run", run)
        runner = SafariRunner(log=lambda _: None, progress=lambda: None)
        with pytest.raises(RuntimeError, match=problem):
            runner.loaded_images(100)

    @pytest.mark.parametrize(
        "bad_row",
        [
            "__TEXT changed output format",
            "__TEXT 300-400 [ 16K] r-x/r-x SM=COW",
            "__TEXT 300-400 [ 16K] r-x/r-x SM=COW /Users/*/Safari",
        ],
    )
    def test_partial_image_parse_fails_verification(self, req, monkeypatch, bad_row):
        output = (
            "__TEXT 100-200 [ 16K] r-x/r-x SM=COW /valid/image\n"
            f"{bad_row}\n==== Summary for process 100\n"
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda argv, **kw: subprocess.CompletedProcess(argv, 0, output, ""),
        )
        runner = _Runner(_Host(req.run_root, 100))
        runner.loaded_images = SafariRunner.loaded_images.__get__(runner)
        verdict = runner.verify(req, _Proc(100))
        assert not verdict.ok
        assert "vmmap parse failure for 100" in verdict.problem
        assert bad_row in verdict.problem
