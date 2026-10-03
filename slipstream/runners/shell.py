# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The shell runtime: d8 and jsc, driven by argv, scores read off stdout."""

from __future__ import annotations

import os
import platform
import re
import shlex
import signal
import subprocess
from pathlib import Path

from rich.markup import escape

from ..models import Score
from .base import Log, Progress, RunRequest, RunResult, geomean_overall


def parse_stdout(
    stdout_file: Path,
    suite: str,
    flags: str,
    run: int,
    score_regex: str,
    suite_score_regex: str | None,
) -> list[Score]:
    """Parse a harness's stdout into scores.

    ``score_regex`` has three groups, benchmark, metric and value;
    ``suite_score_regex`` two, metric and value, for the suite-wide lines the
    harness prints indented under the per-benchmark ones.
    """
    pattern = re.compile(score_regex)
    suite_pattern = re.compile(suite_score_regex) if suite_score_regex else None
    results = []
    for line in stdout_file.read_text().splitlines():
        m = pattern.match(line.strip())
        if m:
            bench, metric, score = m.groups()
            if suite == "js3" and metric == "Score":
                metric = "Total-Score"
            results.append(Score(suite, flags, bench, metric, run, float(score)))
        elif suite_pattern:
            m2 = suite_pattern.match(line)
            if m2:
                metric, score = m2.groups()
                results.append(
                    Score(suite, flags, "Overall", metric, run, float(score))
                )
    return results


class ShellRunner:
    def __init__(self, *, log: Log, progress: Progress):
        self._log = log
        self._progress = progress

    def run(self, req: RunRequest) -> RunResult:
        binary = req.run_root / req.engine.binary_path
        env = os.environ.copy()
        env_prefix: list[str] = []
        if req.engine.dyld_lib_path:
            dyld_path = str(req.run_root / req.engine.dyld_lib_path)
            env["DYLD_LIBRARY_PATH"] = dyld_path
            env["DYLD_FRAMEWORK_PATH"] = dyld_path
            # macOS SIP strips DYLD_ vars from protected processes (caffeinate/gtimeout),
            # so we pass them explicitly via `env`.
            env_prefix = [
                "env",
                f"DYLD_FRAMEWORK_PATH={dyld_path}",
                f"DYLD_LIBRARY_PATH={dyld_path}",
            ]

        is_macos = platform.system() == "Darwin"
        prefix = ["caffeinate", "-im"] if is_macos else []
        prefix += ["gtimeout" if is_macos else "timeout"]

        bench = req.bench
        suite = req.spec.suite
        stdout_file = req.artifact("stdout", "txt")
        stderr_file = req.artifact("stderr", "txt")
        ok = True
        with open(stdout_file, "w") as out_f, open(stderr_file, "w") as err_f:
            base = [
                *prefix,
                bench.timeout,
                *env_prefix,
                str(binary),
                *req.spec.flags,
                f"{bench.dir}/{bench.cli}",
            ]
            if req.run_mode == "suite":
                self._progress()
                if not self.exec(
                    base, bench.dir, env, out_f, err_f, stderr_file, suite
                ):
                    ok = False
            else:
                for i, name in enumerate(b for b in bench.names if b != "Overall"):
                    if i % 5 == 0:
                        self._progress()
                    if not self.exec(
                        [*base, "--", name],
                        bench.dir,
                        env,
                        out_f,
                        err_f,
                        stderr_file,
                        f"{suite}/{name}",
                    ):
                        ok = False
        if not ok:
            return RunResult(False, [])

        scores = parse_stdout(
            stdout_file,
            suite,
            req.spec.variant,
            req.run,
            bench.score_regex,
            bench.suite_score_regex,
        )
        if req.run_mode == "per_benchmark":
            scores.extend(geomean_overall(scores, req.run))
        return RunResult(True, scores)

    def exec(
        self,
        argv: list[str],
        cwd: Path,
        env: dict,
        out_f,
        err_f,
        stderr_file: Path,
        label: str,
    ) -> bool:
        """Run a single benchmark subprocess. Returns True on success.

        argv, not a shell string: the run root is a generated path on a bus
        consumer, and it is interpolated into every one of these.
        """
        cmd = shlex.join(argv)
        try:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=out_f, stderr=err_f)
        except Exception as exc:
            self._log(f"\n    [red]Exception [{label}]: {escape(str(exc))}[/red]")
            self._log(f"    [dim]$ cd {cwd} && {escape(cmd)}[/dim]")
            err_f.write(f"\n$ cd {cwd} && {cmd}\n")
            return False
        if result.returncode in (-signal.SIGINT, -signal.SIGTERM, 130, 143):
            raise KeyboardInterrupt
        if result.returncode != 0:
            err_f.flush()
            err_f.write(f"\n$ cd {cwd} && {cmd}\n")
            err_f.flush()
            stderr_tail = stderr_file.read_text()[-300:].strip()
            self._log(
                f"\n    [red]Error (exit {result.returncode}) [{label}]: "
                f"{escape(stderr_tail or '(no stderr)')}[/red]"
            )
            self._log(f"    [dim]$ cd {cwd} && {escape(cmd)}[/dim]")
            return False
        return True
