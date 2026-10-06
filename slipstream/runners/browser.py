# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The browser runtimes: a served page, a launched browser, a report back.

What differs between browsers is how one is started -- argv, environment,
what to clean up afterwards -- and a subclass says that in ``command`` and
``cleanup``. What differs between suites is the page, its query string,
whether a reporter script is appended to it and which parser reads the
report, and the suite's config says that. The run itself is here: serve
the suite, start the browser at the suite's page, wait for the page's
``POST /report`` or for the browser to die or the suite's timeout to pass,
tear the browser down, and turn the report into scores. Any of timeout,
early exit, an unreadable report or a report missing benchmarks fails the
config, never the commit.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.markup import escape

from ..config import EngineConfig
from .base import Log, Progress, RunRequest, RunResult, cfg_digest, geomean_overall
from .report import TOTAL_METRICS, ReportError, parse_for
from .server import BenchServer

POLL_SECONDS = 0.5
GRACE_SECONDS = 5.0
# How long after the page's first request the early provenance check runs:
# long enough for the browser to have started its content process and
# loaded the engine, early enough that a wrong-engine run costs seconds.
EARLY_CHECK_SECONDS = 10.0


@dataclass(frozen=True)
class Command:
    """How to start the browser at a URL."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    cwd: Path | None = None


@dataclass(frozen=True)
class Verdict:
    """What a provenance check found. ``notes`` go to the stderr file
    either way, so a passed check leaves the evidence too."""

    ok: bool
    notes: list[str] = field(default_factory=list)
    problem: str = ""


class BrowserRunner:
    runtime = "browser"

    def __init__(self, *, log: Log, progress: Progress):
        self._log = log
        self._progress = progress

    # --- what a browser defines ---

    def command(self, req: RunRequest, url: str) -> Command:
        raise NotImplementedError

    def conventions(self) -> tuple[str, ...]:
        """What every run of this browser gets, for ``cfg_hash``: the fixed
        flags, the launch mechanism. Not the ``[[run]]`` flags, which are
        the variant, and not the page, which is the suite's
        (``suite_cfg_hash``)."""
        raise NotImplementedError

    def cfg_hash(self) -> str:
        return cfg_digest(self.runtime, "POST /report", *self.conventions())

    def host_env(self, engine: EngineConfig, run_root: Path) -> dict[str, str]:
        return {}

    def precondition(self, req: RunRequest) -> str | None:
        """Why the browser must not be launched right now, or None."""
        return None

    def verify(self, req: RunRequest, proc: subprocess.Popen) -> Verdict:
        """That what is running is what the run root holds. Called once
        shortly after the page's first request and once when the report is
        in; either failure fails the config. A browser whose binary is the
        artifact has nothing to check."""
        return Verdict(True)

    def cleanup(self, req: RunRequest) -> None:
        """After the browser is gone, whatever ``command`` set up."""

    def terminate(self, proc: subprocess.Popen) -> None:
        """SIGTERM to the browser's process group, SIGKILL after a grace
        period. The group, because a browser is a tree of helpers and the
        launcher dying does not take them all with it."""
        if proc.poll() is not None:
            return
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            proc.wait()

    # --- the run ---

    def run(self, req: RunRequest) -> RunResult:
        label = f"{req.spec.suite} ({req.spec.variant})"
        bench = req.bench
        stderr_file = req.artifact("stderr", "txt")
        problem = self.precondition(req)
        if problem:
            self._fail(label, problem)
            stderr_file.write_text(f"not launched: {problem}\n")
            return RunResult(False, [])
        with (
            BenchServer(bench.dir, page=bench.page, inject=bench.inject) as server,
            open(stderr_file, "w") as err_f,
        ):
            url = server.page_url(bench.page, bench.query)
            cmd = self.command(req, url)
            err_f.write(f"$ {shlex.join(cmd.argv)}\n")
            err_f.flush()
            self._progress()
            try:
                proc = subprocess.Popen(
                    cmd.argv,
                    env=cmd.env,
                    cwd=cmd.cwd,
                    stdout=err_f,
                    stderr=err_f,
                    start_new_session=True,
                )
            except OSError as exc:
                self._fail(label, f"cannot start browser: {exc}")
                self.cleanup(req)
                return RunResult(False, [])
            try:
                body = self._await_report(
                    server, proc, bench.timeout_seconds, label, req, err_f
                )
                if body is not None and not self._verified(req, proc, label, err_f):
                    body = None
            finally:
                self.terminate(proc)
                self.cleanup(req)
                err_f.write("\n--- requests ---\n")
                err_f.write("".join(f"{line}\n" for line in server.requests))
        if body is None:
            return RunResult(False, [])
        self._progress()
        req.artifact("report", "json").write_bytes(body)

        try:
            scores = parse_for(bench.report_format)(
                body,
                req.spec.suite,
                req.spec.variant,
                req.run,
                bench.report_metrics,
            )
        except ReportError as exc:
            self._fail(label, str(exc))
            return RunResult(False, [])

        expected = {n for n in bench.names if n != "Overall"}
        reported = {
            s.benchmark
            for s in scores
            if s.metric in TOTAL_METRICS and s.benchmark != "Overall"
        }
        if reported != expected:
            missing = sorted(expected - reported)
            extra = sorted(reported - expected)
            self._fail(
                label,
                "report does not match the suite's benchmark list"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else ""),
            )
            return RunResult(False, scores)
        # A report that carries the suite's own overall (Speedometer's
        # Score) keeps it; JetStream's carries only the per-test numbers.
        if not any(s.benchmark == "Overall" for s in scores):
            scores.extend(geomean_overall(scores, req.run))
        return RunResult(True, scores)

    def _await_report(
        self,
        server: BenchServer,
        proc: subprocess.Popen,
        timeout: float,
        label: str,
        req: RunRequest,
        err_f,
    ) -> bytes | None:
        deadline = time.monotonic() + timeout
        first_request_at: float | None = None
        checked_early = False
        while True:
            body = server.wait_for_report(POLL_SECONDS)
            if body is not None:
                return body
            if proc.poll() is not None:
                self._fail(
                    label, f"browser exited ({proc.returncode}) before reporting"
                )
                return None
            now = time.monotonic()
            if now > deadline:
                self._fail(label, f"no report within {timeout:g}s")
                return None
            if first_request_at is None and server.requests:
                first_request_at = now
            if (
                not checked_early
                and first_request_at is not None
                and now - first_request_at >= EARLY_CHECK_SECONDS
            ):
                checked_early = True
                if not self._verified(req, proc, label, err_f):
                    return None

    def _verified(self, req: RunRequest, proc, label: str, err_f) -> bool:
        verdict = self.verify(req, proc)
        for note in verdict.notes:
            err_f.write(f"provenance: {note}\n")
        err_f.flush()
        if not verdict.ok:
            self._fail(label, f"provenance: {verdict.problem}")
        return verdict.ok

    def _fail(self, label: str, what: str) -> None:
        self._log(f"\n    [red]Error [{label}]: {escape(what)}[/red]")


def _signal_group(proc: subprocess.Popen, sig: signal.Signals) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
