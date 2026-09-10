# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import platform
import signal
import subprocess
import time
from collections import namedtuple
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from .config import Config, EngineConfig, RunSpec
from .lock import MachineLock
from .store import CommitIdCollision, CommitStore

console = Console()


class FetchError(RuntimeError):
    """A git fetch failed, so the remote frontier cannot be trusted.

    Raised rather than returned so callers can't confuse it with the
    legitimate "no commits yet" case.
    """


# What one commit's benchmark session produced. ``scores`` is the total number
# of score rows parsed across every run and config; zero means the commit was
# not measured at all, which no count of configs can express.
BenchOutcome = namedtuple("BenchOutcome", ["configs_ok", "configs_total", "scores"])


# Which build step failed. "compile" is the commit's fault; every other kind
# is infrastructure, to be retried without advancing the frontier.
BuildStepError = namedtuple("BuildStepError", ["kind", "returncode"])


def outcome_status(outcome: BenchOutcome) -> str:
    if outcome.scores == 0:
        return "failed"
    return "ok" if outcome.configs_ok == outcome.configs_total else "partial"


class BenchCollector:
    def __init__(
        self,
        config: Config,
        dry_run: bool = False,
        verbose: bool = False,
        *,
        role: str = "bench",
        wait_for_lock: bool = False,
        backup: bool = True,
    ):
        """``role`` names this process in the machine lock file.

        ``wait_for_lock`` distinguishes a daemon, which waits out its peer,
        from an ad-hoc command, which reports the holder and gives up rather
        than blocking for hours behind a bench. ``backup`` is off for reporting
        commands, which should not snapshot the db and rotate its backups on
        every run; they still open it read/write, because they read columns a
        db that predates them does not have.
        """
        self.cfg = config
        self.dry_run = dry_run
        self.verbose = verbose
        self.lock = MachineLock(role)
        self.wait_for_lock = wait_for_lock
        self.cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = (
            self.cfg.logs_dir / f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        self.store = CommitStore(
            self.cfg.metadata_dir / "slipstream.db",
            bot=self.cfg.bot_name,
            backup=backup,
        )

    def _log(self, message: str, **kwargs):
        timestamp = datetime.now().strftime("%H:%M:%S")
        console.print(f"[{timestamp}] {message}", **kwargs)
        if not self.dry_run:
            from rich.text import Text

            with open(self.log_file, "a") as f:
                f.write(f"[{timestamp}] {Text.from_markup(message).plain}\n")

    def _run(
        self,
        cmd: str,
        cwd: Path | None = None,
        env: dict | None = None,
        capture: bool = False,
        caffeinate: bool = True,
    ) -> subprocess.CompletedProcess:
        if caffeinate and not self.dry_run and platform.system() == "Darwin":
            cmd = f"caffeinate -im {cmd}"
        if self.dry_run and not capture:
            self._log(f"[yellow]Would run:[/yellow] {escape(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if self.verbose:
            self._log(f"[dim]$ {escape(cmd)}[/dim]")
        # An unattended collector has nobody to answer a git credential prompt,
        # and a blocked git blocks the whole watch loop. Fail fast instead.
        child_env = dict(os.environ if env is None else env)
        child_env.setdefault("GIT_TERMINAL_PROMPT", "0")
        return subprocess.run(
            cmd, shell=True, cwd=cwd, env=child_env, capture_output=capture, text=True
        )

    # --- Commit resolution ---

    def _commit_hash_from_id(self, engine: EngineConfig, commit_id: int) -> str:
        src = engine.require_src_dir()
        # Replace the capture group in the regex with the literal ID for grepping
        grep_pattern = re.sub(r"\(\[0-9\][+?]\)", str(commit_id), engine.id_regex)
        grep_pattern = re.sub(r"\(\[0-9\]\{6\}\)", str(commit_id), grep_pattern)
        res = self._run(
            f'git log origin/main --pretty=format:%H --grep="{grep_pattern}" -n 1',
            cwd=src,
            capture=True,
            caffeinate=False,
        )
        return res.stdout.strip()

    def _commit_id_from_hash(
        self, engine: EngineConfig, commit_hash: str
    ) -> str | None:
        res = self._run(
            f"git show -s {commit_hash}",
            cwd=engine.require_src_dir(),
            capture=True,
            caffeinate=False,
        )
        # Use the last match: reverts/relands copy the original commit message, so
        # the commit's own ID always appears last.
        matches = re.findall(engine.id_regex, res.stdout, re.MULTILINE)
        return matches[-1] if matches else None

    _METADATA_FORMAT = "%H|%cs|%ct|%s|%b%n--END-COMMIT--"

    def _parse_commit_metadata(self, engine: EngineConfig, raw: str) -> dict | None:
        """One --END-COMMIT-- record into a commit dict, or None if it has no id."""
        parts = raw.strip().split("|", 4)
        if len(parts) < 5:
            return None
        commit_hash, date_str, ts, subject, body = parts
        # Use the last match: reverts and relands copy the original commit
        # message, so the commit's own id always appears last.
        matches = re.findall(engine.id_regex, subject + "\n" + body, re.MULTILINE)
        if not matches:
            return None
        return {
            "hash": commit_hash,
            "commit_id": int(matches[-1]),
            "date": date_str,
            "timestamp": int(ts),
            "title": subject.replace('"', ""),
        }

    def next_commit_after(self, engine: EngineConfig, commit_id: int) -> dict | None:
        """The oldest commit above ``commit_id`` that touches this engine.

        For the builder, which walks history itself rather than reading what
        this machine has benched. Honours path_filter, so jsc does not build a
        commit that changed nothing it measures.
        """
        src = engine.require_src_dir()
        start_hash = self._commit_hash_from_id(engine, commit_id)
        if not start_hash:
            raise ValueError(f"cannot resolve {engine.name} commit {commit_id}")
        path_filter = engine.path_filter or ""
        res = self._run(
            f'git log --reverse --pretty=format:"{self._METADATA_FORMAT}"'
            f" {start_hash}..origin/main -- {path_filter}",
            cwd=src,
            capture=True,
            caffeinate=False,
        )
        for raw in res.stdout.split("--END-COMMIT--"):
            if not raw.strip():
                continue
            commit = self._parse_commit_metadata(engine, raw)
            if commit and commit["commit_id"] > commit_id:
                return commit
        return None

    def commit_metadata_for_id(
        self, engine: EngineConfig, commit_id: int
    ) -> dict | None:
        """Full metadata for one commit id, resolved from the checkout."""
        commit_hash = self._commit_hash_from_id(engine, commit_id)
        if not commit_hash:
            return None
        res = self._run(
            f'git log -1 --pretty=format:"{self._METADATA_FORMAT}" {commit_hash}',
            cwd=engine.require_src_dir(),
            capture=True,
            caffeinate=False,
        )
        return self._parse_commit_metadata(engine, res.stdout)

    # --- Commit list & metadata ---

    def get_commit_list(
        self,
        engine: EngineConfig,
        start_id: int,
        end_id: int,
        include_start: bool = True,
    ) -> list[dict]:
        src = engine.require_src_dir()
        start_hash = self._commit_hash_from_id(engine, start_id)
        end_hash = self._commit_hash_from_id(engine, end_id)
        if not start_hash or not end_hash:
            raise ValueError(f"Could not resolve hashes for range {start_id}..{end_id}")

        path_filter = engine.path_filter or ""
        res = self._run(
            f'git log --reverse --pretty=format:"%H %s" {start_hash}..{end_hash} {path_filter}',
            cwd=src,
            capture=True,
            caffeinate=False,
        )
        commits = []
        for line in res.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split(" ", 1)
            commits.append(
                {"hash": parts[0], "title": parts[1] if len(parts) > 1 else ""}
            )

        if include_start:
            # Prepend the start commit itself
            res_start = self._run(
                f'git log --pretty=format:"%H %s" -1 {start_hash}',
                cwd=src,
                capture=True,
                caffeinate=False,
            )
            if res_start.stdout.strip():
                parts = res_start.stdout.strip().split(" ", 1)
                commits.insert(
                    0, {"hash": parts[0], "title": parts[1] if len(parts) > 1 else ""}
                )

        self.store.insert_commits(engine.name, commits)
        return commits

    def populate_commit_metadata(self, engine: EngineConfig, commits: list[dict]):
        """Extract and cache commit metadata (IDs, dates, titles) in the DB."""
        hashes = [c["hash"] for c in commits]
        if not hashes:
            return
        src = engine.require_src_dir()

        missing = self.store.count_missing_metadata(engine.name, hashes)
        if missing:
            self._log(
                f"Extracting metadata for {missing}/{len(hashes)} commits "
                f"(rest already cached)..."
            )
            start_hash, end_hash = hashes[0], hashes[-1]
            res = self._run(
                f"git log {start_hash}^..{end_hash}"
                f' --pretty=format:"{self._METADATA_FORMAT}"',
                cwd=src,
                capture=True,
                caffeinate=False,
            )

            for raw in res.stdout.strip().split("--END-COMMIT--"):
                if not raw.strip():
                    continue
                commit = self._parse_commit_metadata(engine, raw)
                if commit is None:
                    self._log(
                        f"[yellow]Warning: no commit ID found in "
                        f"{raw.strip()[:8]}, skipping.[/yellow]"
                    )
                    continue
                self.store.update_commit_metadata(
                    engine.name,
                    commit["hash"],
                    commit["commit_id"],
                    commit["date"],
                    commit["timestamp"],
                    commit["title"],
                )
            self.store.conn.commit()
        else:
            self._log(f"Using cached metadata for all {len(hashes)} commits")

    # --- Build ---

    def _apply_pre_build_patches(self, engine: EngineConfig):
        """Replace GCC_TREAT_WARNINGS_AS_ERRORS=YES with NO in xcconfig files."""
        src = engine.require_src_dir()
        if self.dry_run:
            self._log(f"[yellow]Would patch:[/yellow] {engine.pre_build_patches}")
            return
        for rel_path in engine.pre_build_patches:
            path = src / rel_path
            if path.exists():
                text = path.read_text()
                path.write_text(
                    text.replace(
                        "GCC_TREAT_WARNINGS_AS_ERRORS = YES",
                        "GCC_TREAT_WARNINGS_AS_ERRORS = NO",
                    )
                )

    @staticmethod
    def _normalize_gn_args(text: str) -> list[str]:
        """Normalize GN args to sorted non-blank, non-comment, whitespace-stripped lines."""
        return sorted(
            line
            for raw in text.splitlines()
            if (line := raw.split("#")[0].replace(" ", "").strip())
        )

    def _ensure_gn_args(self, engine: EngineConfig, log: Path | None = None) -> int:
        """Set up the GN build directory if gn_args are configured.

        Runs `gn gen` if the build dir doesn't exist or if args.gn differs.
        """
        if not engine.gn_args:
            return 0
        src = engine.require_src_dir()
        # Extract build dir from binary_path (e.g. "out/release-lto/d8" -> "out/release-lto")
        build_dir = str(Path(engine.binary_path).parent)
        args_path = src / build_dir / "args.gn"
        desired = engine.gn_args.strip() + "\n"
        desired_norm = self._normalize_gn_args(desired)

        # build.ninja as well as the args: args.gn is written before gn gen
        # runs, so a failed gn gen leaves the args looking current with no
        # build dir behind them. The next attempt would skip gn gen, the
        # compile would fail, and the commit would be blamed for it.
        if (
            args_path.exists()
            and (args_path.parent / "build.ninja").exists()
            and self._normalize_gn_args(args_path.read_text()) == desired_norm
        ):
            return 0

        console.print(f"  [dim]gn gen {build_dir}[/dim]")
        if self.dry_run:
            self._log(f"[yellow]Would write:[/yellow] {args_path}")
            return self._run(
                f"gn gen {build_dir}{self._quiet(log)}", cwd=src
            ).returncode
        args_path.parent.mkdir(parents=True, exist_ok=True)
        args_path.write_text(desired)
        return self._run(f"gn gen {build_dir}{self._quiet(log)}", cwd=src).returncode

    def _quiet(self, log: Path | None) -> str:
        """Redirection suffix for a build command: to a log, or to nowhere."""
        if log is not None:
            return f" >>{shlex.quote(str(log))} 2>&1"
        return "" if self.verbose else " >/dev/null 2>&1"

    def _provision(self, engine: EngineConfig, commit_hash: str) -> Path | None:
        """Put a runnable build in place and return the root it lives under.

        On a machine with a checkout that is the checkout, built at the given
        commit. A bus consumer unpacks an archive instead and returns that
        directory; ``binary_path`` and ``dyld_lib_path`` resolve against
        whichever it is, so the run path does not know the difference.
        Returns None if the build failed.
        """
        if self.build_at(engine, commit_hash) is not None:
            return None
        return engine.require_src_dir()

    def build_at(
        self, engine: EngineConfig, commit_hash: str, log: Path | None = None
    ) -> BuildStepError | None:
        """Check out and build one commit. Returns the first step that failed.

        The steps are checked separately because only one of them is the
        commit's fault. A wedged tree, a full disk or a DEPS fetch outage all
        fail here, and collapsing them into one bool would let the builder
        blame the commit and burn it permanently. sync runs before the compile
        rather than instead of it, for the same reason.
        """
        src = engine.require_src_dir()
        steps: list[tuple[str, str, bool]] = [
            ("reset", "git reset --hard", False),
            ("clean", "git clean -fd", False),
            ("checkout", f"git checkout {commit_hash}", False),
        ]
        for kind, cmd, caffeinate in steps:
            res = self._run(cmd + self._quiet(log), cwd=src, caffeinate=caffeinate)
            if res.returncode != 0:
                return BuildStepError(kind, res.returncode)

        try:
            self._apply_pre_build_patches(engine)
        except OSError as exc:
            self._log(f"  [red]patch step failed: {escape(str(exc))}[/red]")
            return BuildStepError("patch", 1)

        rc = self._ensure_gn_args(engine, log)
        if rc != 0:
            return BuildStepError("gn", rc)

        if engine.sync_cmd:
            rc = self._run(engine.sync_cmd + self._quiet(log), cwd=src).returncode
            if rc != 0:
                return BuildStepError("sync", rc)

        rc = self._run(engine.build_cmd + self._quiet(log), cwd=src).returncode
        if rc != 0:
            return BuildStepError("compile", rc)
        return None

    # --- Score parsing ---

    def _parse_stdout(
        self,
        stdout_file: Path,
        suite: str,
        flags: str,
        run: int,
        patterns: dict[str, re.Pattern],
        suite_patterns: dict[str, re.Pattern],
    ) -> list[dict]:
        """Parse a stdout file into score dicts for DB insertion."""
        pattern = patterns.get(suite)
        if not pattern:
            return []
        suite_pattern = suite_patterns.get(suite)
        results = []
        for line in stdout_file.read_text().splitlines():
            m = pattern.match(line.strip())
            if m:
                bench, metric, score = m.groups()
                if suite == "js3" and metric == "Score":
                    metric = "Total-Score"
                results.append(
                    {
                        "suite": suite,
                        "flags": flags,
                        "benchmark": bench,
                        "metric": metric,
                        "run": run,
                        "score": float(score),
                    }
                )
            elif suite_pattern:
                m2 = suite_pattern.match(line)
                if m2:
                    metric, score = m2.groups()
                    results.append(
                        {
                            "suite": suite,
                            "flags": flags,
                            "benchmark": "Overall",
                            "metric": metric,
                            "run": run,
                            "score": float(score),
                        }
                    )
        return results

    @staticmethod
    def _compute_geomean_overall(scores: list[dict], run: int) -> list[dict]:
        """Compute geometric mean of Total-Score across all benchmarks.

        Per-line-item runs don't produce Overall scores from the harness,
        so we synthesize one by geomeaning the Total-Score of each benchmark.
        """
        total_scores: list[float] = []
        suite = None
        flags = None
        for s in scores:
            if s["benchmark"] == "Overall":
                continue
            if s["metric"] != "Total-Score":
                continue
            suite = s["suite"]
            flags = s["flags"]
            total_scores.append(s["score"])

        positive = [v for v in total_scores if v > 0]
        if not positive:
            return []
        geomean = math.exp(sum(math.log(v) for v in positive) / len(positive))
        return [
            {
                "suite": suite,
                "flags": flags,
                "benchmark": "Overall",
                "metric": "Total-Score",
                "run": run,
                "score": geomean,
            }
        ]

    # --- Benchmark runs ---

    def _run_bench_cmd(
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

    def _run_benchmarks(
        self, engine: EngineConfig, commit_id: str, runs: int, run_root: Path
    ) -> BenchOutcome:
        """Run every benchmark configuration and record what came back.

        Counts are over (run, config) pairs, so a suite that fails on one run
        of three shows as partial rather than as a clean pass.
        """
        res_dir = self.cfg.commit_results_dir(engine.name, commit_id)
        res_dir.mkdir(parents=True, exist_ok=True)

        patterns = {
            b_type: re.compile(cfg.score_regex)
            for b_type, cfg in self.cfg.benchmarks.items()
        }
        suite_patterns = {
            b_type: re.compile(cfg.suite_score_regex)
            for b_type, cfg in self.cfg.benchmarks.items()
            if cfg.suite_score_regex
        }

        binary = run_root / engine.binary_path
        env = os.environ.copy()
        env_prefix: list[str] = []
        if engine.dyld_lib_path:
            dyld_path = str(run_root / engine.dyld_lib_path)
            env["DYLD_LIBRARY_PATH"] = dyld_path
            env["DYLD_FRAMEWORK_PATH"] = dyld_path
            # macOS SIP strips DYLD_ vars from protected processes (caffeinate/gtimeout),
            # so we pass them explicitly via `env`.
            env_prefix = [
                "env",
                f"DYLD_FRAMEWORK_PATH={dyld_path}",
                f"DYLD_LIBRARY_PATH={dyld_path}",
            ]

        run_configs = self.run_configs(engine)

        is_macos = platform.system() == "Darwin"
        prefix = ["caffeinate", "-im"] if is_macos else []
        prefix += ["gtimeout" if is_macos else "timeout"]

        configs_ok = 0
        configs_total = 0
        score_total = 0
        for run in range(1, runs + 1):
            self._log(f"  Run [bold]{run}/{runs}[/bold]:")
            for rc in run_configs:
                bench_cfg = self.cfg.benchmarks[rc.suite]
                run_mode = rc.run_mode or bench_cfg.run_mode
                stdout_file = res_dir / f"stdout.{run}.{rc.suite}.{rc.variant}.txt"
                stderr_file = res_dir / f"stderr.{run}.{rc.suite}.{rc.variant}.txt"

                console.print(f"    {rc.suite} ({rc.variant}): ", end="")
                t0 = time.time()
                configs_total += 1
                run_ok = True
                if not self.dry_run:
                    with (
                        open(stdout_file, "w") as out_f,
                        open(stderr_file, "w") as err_f,
                    ):
                        base = [
                            *prefix,
                            bench_cfg.timeout,
                            *env_prefix,
                            str(binary),
                            *rc.flags,
                            f"{bench_cfg.dir}/{bench_cfg.cli}",
                        ]
                        if run_mode == "suite":
                            console.print(".", end="", highlight=False)
                            if not self._run_bench_cmd(
                                base,
                                bench_cfg.dir,
                                env,
                                out_f,
                                err_f,
                                stderr_file,
                                rc.suite,
                            ):
                                run_ok = False
                        else:
                            for i, bench in enumerate(
                                b for b in bench_cfg.names if b != "Overall"
                            ):
                                if i % 5 == 0:
                                    console.print(".", end="", highlight=False)
                                if not self._run_bench_cmd(
                                    [*base, "--", bench],
                                    bench_cfg.dir,
                                    env,
                                    out_f,
                                    err_f,
                                    stderr_file,
                                    f"{rc.suite}/{bench}",
                                ):
                                    run_ok = False
                else:
                    console.print("." * 5, end="")
                elapsed = int(time.time() - t0)
                if run_ok and not self.dry_run:
                    scores = self._parse_stdout(
                        stdout_file, rc.suite, rc.variant, run, patterns, suite_patterns
                    )
                    if run_mode == "per_benchmark":
                        scores.extend(self._compute_geomean_overall(scores, run))
                    if scores:
                        self.store.insert_scores(
                            engine.name,
                            self.cfg.platform,
                            int(commit_id),
                            int(time.time()),
                            scores,
                        )
                    score_total += len(scores)
                if run_ok:
                    configs_ok += 1
                    console.print(f" [green]OK ({elapsed}s)[/green]")
                else:
                    console.print(f" [yellow]ERRORS ({elapsed}s)[/yellow]")

        return BenchOutcome(configs_ok, configs_total, score_total)

    def _take_machine_lock(self, should_stop: Callable[[], bool]) -> bool:
        """Acquire for one commit. False means a shutdown was requested."""
        if self.dry_run:
            return True
        return self.lock.acquire(
            should_stop, wait=self.wait_for_lock, log=lambda m: self._log(f"  {m}")
        )

    def bench_at_root(
        self,
        engine: EngineConfig,
        commit: dict,
        run_root: Path,
        runs: int,
        on_commit_done: Callable[[], None] | None = None,
        provenance: dict | None = None,
    ) -> BenchOutcome:
        """Measure one commit from a provisioned run root, and record it.

        The caller holds the machine lock and decides where the run root came
        from: a checkout on a git-driven engine, an unpacked archive on a bus
        consumer. Everything after that is identical, which is what stops an
        archive defect from masquerading as a microarchitecture difference.

        ``provenance`` describes where the binary came from; it defaults to
        "built here", which is what a git-driven engine and an ad-hoc bench
        range both are.
        """
        commit_id = str(commit["commit_id"])
        # Scores of a commit interrupted part way through must go before it is
        # measured again: insert_scores is INSERT OR IGNORE with run in the
        # primary key, so the surviving rows would win for the configs they
        # cover and leave a run number half measured on each side of the
        # interrupt. Both the git path and the bus path arrive here.
        if not self.dry_run:
            self.store.clear_scores(
                engine.name, self.cfg.platform, [int(commit["commit_id"])]
            )
        outcome = BenchOutcome(0, 0, 0)
        try:
            outcome = self._run_benchmarks(engine, commit_id, runs, run_root)
        except KeyboardInterrupt:
            self._log(
                f"  [yellow]Interrupted on {commit_id} — will retry on next run[/yellow]"
            )
            raise
        except Exception as exc:
            # Still marked done, as before, but now recorded as failed: this
            # path used to be indistinguishable from a clean run. The scores
            # written before it died go too -- outcome is (0, 0, 0) here, so
            # the commit is labelled failed while whatever runs completed would
            # otherwise still be exported and pushed as if the commit were
            # measured. "failed" means no scores everywhere else.
            self._log(f"  [red]Fatal error during benchmarks: {escape(str(exc))}[/red]")
            if not self.dry_run:
                self.store.clear_scores(
                    engine.name, self.cfg.platform, [int(commit["commit_id"])]
                )

        if self.dry_run:
            return outcome

        status = outcome_status(outcome)
        try:
            self.store.record_done(
                engine.name,
                self.cfg.platform,
                commit,
                status=status,
                configs_ok=outcome.configs_ok,
                configs_total=outcome.configs_total,
            )
        except CommitIdCollision as e:
            # An anomaly, not routine, but it must not wedge a consumer: record
            # the commit as failed and let the caller move on. The scores go
            # with it -- the id belongs to another hash, so export_scores would
            # join them to that commit's metadata and ship this commit's
            # numbers under the other one's git hash. The missing-commit-row
            # guard cannot see that, because a row for the id does exist.
            self._log(f"  [red]{escape(str(e))}[/red]")
            self.store.clear_scores(
                engine.name, self.cfg.platform, [int(commit["commit_id"])]
            )
            self.store.mark_done(
                engine.name,
                self.cfg.platform,
                int(commit["commit_id"]),
                status="failed",
            )
            # Zero scores, because they were just deleted. The caller's circuit
            # breaker reads this: returning the original outcome would look
            # like a clean run, and systematic collisions (a restored db, a
            # changed id_regex) would march the whole topic into failed with
            # the breaker never firing.
            return BenchOutcome(0, outcome.configs_total, 0)
        self.store.record_run_env(
            engine.name,
            int(commit["commit_id"]),
            provenance or self.local_provenance(engine, runs),
        )
        if on_commit_done is not None:
            on_commit_done()
        return outcome

    def local_provenance(self, engine: EngineConfig, runs: int) -> dict:
        """What produced the numbers when the binary was built on this machine.

        Recorded so the mixture is visible: a repair with `bench` puts a
        locally built commit among archive-built neighbours, which is exactly
        the case the re-bench rules refuse for one command and permit for
        another.
        """
        from . import __version__, host
        from .builder import build_cfg_hash

        identity = host.identity()
        return {
            "source": "local",
            "runs": runs,
            "run_configs": json.dumps(
                sorted(f"{c.suite}/{c.variant}" for c in self.run_configs(engine))
            ),
            "harness_revs": json.dumps(self.harness_revs(), sort_keys=True),
            "hw_model": identity["hw_model"],
            "os_version": identity["os_version"],
            "toolchain": identity["toolchain"],
            "build_cfg_hash": build_cfg_hash(engine),
            "slipstream_version": __version__,
        }

    def run_configs(self, engine: EngineConfig) -> list[RunSpec]:
        """This engine's share of the configured [[run]] matrix.

        The one place the matrix is read: the bench loop, both provenance
        records, and the cross-box divergence check all come through here.
        """
        return [r for r in self.cfg.runs if r.engine == engine.name]

    def harness_revs(self) -> dict[str, str]:
        """Each benchmark suite's checked-out revision.

        A harness update is a shared input to both boxes' numbers, so a change
        landing on one box first shifts one series while everything else about
        the two still matches.
        """
        revs = {}
        for name, bench_cfg in self.cfg.benchmarks.items():
            try:
                res = self._run(
                    "git rev-parse --short HEAD",
                    cwd=bench_cfg.dir,
                    capture=True,
                    caffeinate=False,
                )
            except OSError:
                # A missing or unreadable benchmark dir is worth an empty
                # revision, not an exception: this is provenance, and it runs
                # after a bench that has already produced its scores.
                revs[name] = ""
                continue
            revs[name] = res.stdout.strip() if res.returncode == 0 else ""
        return revs

    # --- Frontier detection ---

    def find_frontier(
        self, engine_name: str, fetch: bool = True
    ) -> tuple[int | None, int | None]:
        """Return (last_done_id, head_id) for an engine."""
        last_done = self.store.max_done_commit_id(engine_name, self.cfg.platform)
        return last_done, self.head_commit_id(engine_name, fetch=fetch)

    def head_commit_id(self, engine_name: str, fetch: bool = True) -> int | None:
        """The newest commit id on origin/main for this engine.

        Split out of find_frontier because the builder's frontier comes from
        what it has published, not from what this machine has benched.
        """
        engine = self.cfg.engines[engine_name]
        src = engine.require_src_dir()

        if fetch:
            res = self._run(
                "git fetch origin main",
                cwd=src,
                capture=True,
                caffeinate=False,
            )
            # Without this the stale origin/main below reads as "up to date"
            # and the engine silently stops advancing.
            if res.returncode != 0:
                self._log(f"[red]{engine_name}: git fetch failed[/red]")
                if res.stderr:
                    self._log(f"[dim]{escape(res.stderr.strip())}[/dim]")
                raise FetchError(engine_name)

        path_filter = engine.path_filter or ""
        res = self._run(
            f"git log -1 --pretty=format:%H origin/main -- {path_filter}",
            cwd=src,
            capture=True,
            caffeinate=False,
        )
        head_hash = res.stdout.strip()
        if not head_hash:
            return None
        head_id = self._commit_id_from_hash(engine, head_hash)
        return int(head_id) if head_id else None

    # --- Main entry point ---

    def collect(
        self,
        engine_name: str,
        start_id: int,
        end_id: int,
        step: int = 1,
        runs: int = 3,
        clear: bool = False,
        include_start: bool = True,
        on_commit_done: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        """Benchmark commits in (start_id, end_id].

        ``on_commit_done`` is invoked after each commit reaches the done state
        (scores written and marked). It must be cheap and non-blocking: the
        background pusher uses it to drain new scores without delaying the next
        build.

        ``should_stop`` lets a shutdown request reach the per-commit path,
        where the wait for the machine lock can otherwise last hours. It
        defaults to never stopping, so an ad-hoc run needs no handler.
        """
        should_stop = should_stop or (lambda: False)
        engine = self.cfg.engines[engine_name]
        self.cfg.require_runs(engine_name)

        commits = self.get_commit_list(
            engine, start_id, end_id, include_start=include_start
        )
        self.populate_commit_metadata(engine, commits)

        sampled = list(
            self.store.get_commits_with_metadata(
                engine.name, [c["hash"] for c in commits]
            )
        )[::step]

        # Commits a dry run would have cleared: nothing was, so is_done below
        # would skip every one of them and the plan would show no work at all.
        cleared: set[int] = set()
        if clear:
            clear_ids = [r["commit_id"] for r in sampled]
            self._log(
                f"[yellow]Clearing results and state for {len(clear_ids)} commits...[/yellow]"
            )
            if not self.dry_run:
                self.store.clear_range(engine_name, self.cfg.platform, clear_ids)
                for cid in clear_ids:
                    p = self.cfg.commit_results_dir(engine_name, cid)
                    if p.exists():
                        shutil.rmtree(p)
            else:
                cleared = set(clear_ids)

        t_start = time.time()
        total = len(sampled)
        for idx, row in enumerate(sampled):
            commit_id_int = row["commit_id"]
            commit_id = str(commit_id_int)
            commit_hash = row["hash"]

            eta = ""
            if idx > 0:
                remaining = (time.time() - t_start) / idx * (total - idx)
                eta = str(timedelta(seconds=int(remaining)))

            self._log(
                f"\n[bold green]=== [{idx + 1}/{total}] ID: {commit_id} "
                f"({commit_hash[:8]}) | ETA: {eta or 'N/A'}[/bold green]"
            )

            if commit_id_int not in cleared and self.store.is_done(
                engine_name, self.cfg.platform, commit_id_int
            ):
                self._log("  [yellow]Skipping: already done[/yellow]")
                continue

            # The build and every run of this commit happen under one hold, so
            # a peer's build cannot land between two of its runs.
            if not self._take_machine_lock(should_stop):
                break
            try:
                t0 = time.time()
                console.print("  Building... ", end="")
                run_root = self._provision(engine, commit_hash)
                if run_root is None:
                    console.print(f"[red]FAILED ({int(time.time() - t0)}s)[/red]")
                    continue
                console.print(f"[green]OK ({int(time.time() - t0)}s)[/green]")

                self.bench_at_root(
                    engine, dict(row), run_root, runs, on_commit_done=on_commit_done
                )
            finally:
                self.lock.release()

            if should_stop():
                break
