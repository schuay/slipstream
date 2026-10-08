# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import platform
import subprocess
import time
from collections import namedtuple
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from . import host
from .commit_filter import V8CommitFilter
from .config import Config, EngineConfig, RunSpec
from .lock import MachineLock
from .models import CommitKey
from .runners import RunRequest, RunResult, Runner, runner_for, suite_cfg_hash
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
        # Before every measurement; a seam so tests do not read pmset.
        self.cool_down: Callable[[Callable[[str], None]], float] = host.cool_down
        self._commit_relevance: dict[tuple, bool] = {}

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
        # The capture group becomes the literal id. git's --grep is a substring
        # match, so #5003 alone would also find #50031 and, newest first,
        # return it: when nothing follows the group, close the id with a
        # non-digit or end of line. When the pattern goes on (jsc's `@`), that
        # tail already delimits it, and a boundary in front of it would eat
        # the character the tail then fails to find. ERE for the alternation;
        # the bundled patterns read the same either way.
        group = re.search(r"\(\[0-9\](?:[+?]|\{6\})\)", engine.id_regex)
        if group is None:
            grep_pattern = engine.id_regex
        else:
            tail = engine.id_regex[group.end() :]
            literal = str(commit_id) if tail else f"{commit_id}([^0-9]|$)"
            grep_pattern = engine.id_regex[: group.start()] + literal + tail
        res = self._run(
            f"git log origin/main --pretty=format:%H --extended-regexp"
            f' --grep="{grep_pattern}" -n 5',
            cwd=src,
            capture=True,
            caffeinate=False,
        )
        # Bounded so git stops at the first hits rather than walking all of
        # history, and verified so the pattern's precision is not the contract.
        for candidate in res.stdout.split():
            if self._commit_id_from_hash(engine, candidate) == str(commit_id):
                return candidate
        return ""

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

    def commit_is_relevant(
        self, engine: EngineConfig, commit_hash: str, *, build_engine=None
    ) -> bool:
        """Whether a V8 commit changes inputs this benchmark build measures.

        Unknown changes and failed reads are measured. The outer engine's GN
        args govern V8 when it is built inside Chromium.
        """
        if engine.name != "v8":
            return True
        build_engine = build_engine or engine
        key = (engine.src_dir, commit_hash, build_engine.gn_args, self.cfg.platform)
        if key in self._commit_relevance:
            return self._commit_relevance[key]
        sha = shlex.quote(commit_hash)
        changes = self._run(
            f"git diff-tree --root --no-commit-id --name-only --no-renames -r -z {sha}",
            cwd=engine.require_src_dir(),
            capture=True,
            caffeinate=False,
        )
        if changes.returncode:
            return True
        paths = [p for p in changes.stdout.split("\0") if p]
        policy = V8CommitFilter(build_engine.gn_args or "", self.cfg.platform)
        relevant = not paths
        for path in paths:
            if policy.ignores_path(path):
                continue
            if path == "DEPS":
                versions = [
                    self._run(
                        f"git show {sha}{suffix}:DEPS",
                        cwd=engine.require_src_dir(),
                        capture=True,
                        caffeinate=False,
                    )
                    for suffix in ("^", "")
                ]
                if all(
                    v.returncode == 0 for v in versions
                ) and policy.ignores_deps_change(
                    versions[0].stdout, versions[1].stdout
                ):
                    continue
            relevant = True
            break
        self._commit_relevance[key] = relevant
        return relevant

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
            if (
                commit
                and commit["commit_id"] > commit_id
                and self.commit_is_relevant(engine, commit["hash"])
            ):
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
            if not self.commit_is_relevant(engine, parts[0]):
                continue
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
                patched = text.replace(
                    "GCC_TREAT_WARNINGS_AS_ERRORS = YES",
                    "GCC_TREAT_WARNINGS_AS_ERRORS = NO",
                )
                if patched != text:
                    path.write_text(patched)

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
        build_dir = engine.build_dir
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
        self,
        engine: EngineConfig,
        commit_hash: str,
        log: Path | None = None,
        *,
        pins: dict[str, str] | None = None,
    ) -> BuildStepError | None:
        """Check out and build one commit. Returns the first step that failed.

        The steps are checked separately because only one of them is the
        commit's fault. A wedged tree, a full disk or a DEPS fetch outage all
        fail here, and collapsing them into one bool would let the builder
        blame the commit and burn it permanently. sync runs before the compile
        rather than instead of it, for the same reason.

        ``pins`` maps gclient dep paths to the revisions this build wants
        under the checkout, written into the deps file before sync so that
        sync is what moves them; the checkout at the start of the next build
        takes the edit back out.
        """
        src = engine.require_src_dir()
        steps: list[tuple[str, str, bool]] = [
            # Move directly to the target revision, discarding tracked edits
            # from patches/pins without an intermediate reset. Keep untracked
            # build and package-manager state; clean -fd deletes Xcode's
            # xcshareddata/swiftpm even though WebKitBuild itself is ignored.
            ("checkout", f"git checkout --force {commit_hash}", False),
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

        for dep, rev in (pins or {}).items():
            cmd = f"gclient setdep --deps-file={engine.roll_file} -r {dep}@{rev}"
            rc = self._run(cmd + self._quiet(log), cwd=src, caffeinate=False).returncode
            if rc != 0:
                return BuildStepError("pin", rc)

        if engine.sync_cmd:
            rc = self._run(engine.sync_cmd + self._quiet(log), cwd=src).returncode
            if rc != 0:
                return BuildStepError("sync", rc)

        # After sync: gn reads build/config and the toolchain, and sync is
        # what puts those at the commit's revision.
        rc = self._ensure_gn_args(engine, log)
        if rc != 0:
            return BuildStepError("gn", rc)

        rc = self._run(engine.build_cmd + self._quiet(log), cwd=src).returncode
        if rc != 0:
            return BuildStepError("compile", rc)
        return None

    # --- Benchmark runs ---

    def _runner(self, engine: EngineConfig) -> Runner:
        return runner_for(
            engine.runtime,
            log=self._log,
            progress=lambda: console.print(".", end="", highlight=False),
        )

    def _run_benchmarks(
        self, engine: EngineConfig, key: CommitKey, run_root: Path
    ) -> BenchOutcome:
        """Run every benchmark configuration and record what came back.

        Counts are over (run, config) pairs, so a suite that fails on one run
        of three shows as partial rather than as a clean pass.

        Each [[run]] entry has its own count. The rounds stay interleaved
        across configs rather than finishing one config before the next, so a
        drift in the machine over the hour lands on every series alike; a
        config past its count simply sits out the remaining rounds.
        """
        res_dir = self.cfg.commit_results_dir(engine.name, key)
        res_dir.mkdir(parents=True, exist_ok=True)

        runner = self._runner(engine)
        run_configs = self.run_configs(engine)
        rounds = self.max_runs(engine)

        configs_ok = 0
        configs_total = 0
        score_total = 0
        for run in range(1, rounds + 1):
            self._log(f"  Run [bold]{run}/{rounds}[/bold]:")
            for rc in run_configs:
                if run > rc.runs:
                    continue
                # Before the config's progress line, so the wait's log lines
                # stand on their own and the elapsed time is the measurement's.
                if not self.dry_run:
                    self.cool_down(self._log)
                console.print(f"    {rc.suite} ({rc.variant}): ", end="")
                t0 = time.time()
                configs_total += 1
                if self.dry_run:
                    console.print("." * 5, end="")
                    result = RunResult(True, [])
                else:
                    result = runner.run(
                        RunRequest(
                            engine=engine,
                            run_root=run_root,
                            bench=self.cfg.benchmarks[rc.suite],
                            spec=rc,
                            run=run,
                            res_dir=res_dir,
                        )
                    )
                elapsed = int(time.time() - t0)
                scores = result.scores if result.ok else []
                if scores:
                    self.store.insert_scores(
                        engine.name,
                        self.cfg.platform,
                        key,
                        int(time.time()),
                        [s._asdict() for s in scores],
                    )
                score_total += len(scores)
                if result.ok:
                    configs_ok += 1
                    console.print(f" [green]OK ({elapsed}s)[/green]")
                else:
                    console.print(f" [yellow]ERRORS ({elapsed}s)[/yellow]")

        return BenchOutcome(configs_ok, configs_total, score_total)

    def _take_machine_lock(
        self, should_stop: Callable[[], bool], job: str = ""
    ) -> bool:
        """Acquire for one commit. False means a shutdown was requested."""
        if self.dry_run:
            return True
        return self.lock.acquire(
            should_stop,
            wait=self.wait_for_lock,
            log=lambda m: self._log(f"  {m}"),
            job=job,
        )

    def bench_at_root(self, engine, commit, run_root, provenance=None):
        if self.dry_run:
            return self._bench_at_root(engine, commit, run_root, provenance)
        key = CommitKey.from_commit(commit)
        with self.store.result_locks(engine.name, self.cfg.platform, [key]):
            self.store.check_pending(engine.name, self.cfg.platform, [key])
            return self._bench_at_root(engine, commit, run_root, provenance)

    def _bench_at_root(
        self,
        engine: EngineConfig,
        commit: dict,
        run_root: Path,
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
        key = CommitKey.from_commit(commit)
        # Scores of a commit interrupted part way through must go before it is
        # measured again: insert_scores is INSERT OR IGNORE with run in the
        # primary key, so the surviving rows would win for the configs they
        # cover and leave a run number half measured on each side of the
        # interrupt. Both the git path and the bus path arrive here.
        if not self.dry_run:
            self.store.clear_scores(engine.name, self.cfg.platform, [key])
        outcome = BenchOutcome(0, 0, 0)
        try:
            outcome = self._run_benchmarks(engine, key, run_root)
        except KeyboardInterrupt:
            self._log(
                f"  [yellow]Interrupted on {key} — will retry on next run[/yellow]"
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
                self.store.clear_scores(engine.name, self.cfg.platform, [key])

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
            self.store.clear_scores(engine.name, self.cfg.platform, [key])
            self.store.mark_done(engine.name, self.cfg.platform, key, status="failed")
            # Zero scores, because they were just deleted. The caller's circuit
            # breaker reads this: returning the original outcome would look
            # like a clean run, and systematic collisions (a restored db, a
            # changed id_regex) would march the whole topic into failed with
            # the breaker never firing.
            return BenchOutcome(0, outcome.configs_total, 0)
        self.store.record_run_env(
            engine.name,
            key,
            provenance or self.local_provenance(engine, run_root),
        )
        return outcome

    def local_provenance(
        self, engine: EngineConfig, run_root: Path | None = None
    ) -> dict:
        """What produced the numbers when the binary was built on this machine.

        Recorded so the mixture is visible: a repair with `bench` puts a
        locally built commit among archive-built neighbours, which is exactly
        the case the re-bench rules refuse for one command and permit for
        another.

        Two of the fields are the runner's. ``runner_cfg_hash`` is how the
        engine was driven, beside ``build_cfg_hash`` for how it was built;
        ``host_env`` is what the run went through on this machine that is
        neither -- for a browser, the application around the engine and the
        launcher that started it -- and needs the run root to read it.
        ``suite_cfg_hash`` is the suites' side of the same question, one
        hash per suite this engine ran.

        ``runs`` is the number of rounds, which is the largest count among
        the engine's entries; a single column cannot hold one count per
        entry, and the rounds are what the score rows' run numbers go up to.
        """
        from . import __version__
        from .builder import build_cfg_hash

        identity = host.identity()
        runner = self._runner(engine)
        host_env = runner.host_env(engine, run_root) if run_root is not None else {}
        return {
            "source": "local",
            "runs": self.max_runs(engine),
            "run_configs": json.dumps(self.run_config_labels(engine)),
            "harness_revs": json.dumps(self.harness_revs(), sort_keys=True),
            "hw_model": identity["hw_model"],
            "os_version": identity["os_version"],
            "toolchain": identity["toolchain"],
            "build_cfg_hash": build_cfg_hash(engine),
            "runner_cfg_hash": runner.cfg_hash(),
            "suite_cfg_hash": json.dumps(self.suite_cfg_hashes(engine), sort_keys=True),
            "host_env": json.dumps(host_env, sort_keys=True),
            "slipstream_version": __version__,
        }

    def run_configs(self, engine: EngineConfig) -> list[RunSpec]:
        """This engine's share of the configured [[run]] matrix.

        The one place the matrix is read: the bench loop, both provenance
        records, and the cross-box divergence check all come through here.
        """
        return [r for r in self.cfg.runs if r.engine == engine.name]

    def run_config_labels(self, engine: EngineConfig) -> list[str]:
        """The matrix as provenance and `bus status` spell it, sorted:
        ``suite/variant@runs`` per entry."""
        return sorted(c.label for c in self.run_configs(engine))

    def max_runs(self, engine: EngineConfig) -> int:
        """How many rounds a commit of this engine takes: the largest count
        among its [[run]] entries, 0 for an engine with none."""
        return max((r.runs for r in self.run_configs(engine)), default=0)

    def suite_cfg_hashes(self, engine: EngineConfig) -> dict[str, str]:
        """``suite -> suite_cfg_hash`` for the suites this engine runs: how
        each is driven and read, which is per suite where ``runner_cfg_hash``
        is per engine. A change to one suite's protocol shows on that suite
        and no other."""
        return {
            suite: suite_cfg_hash(self.cfg.benchmarks[suite])
            for suite in sorted({c.suite for c in self.run_configs(engine)})
        }

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
        """Return (last_done_id, head_id) for an engine.

        Scalar ids: this is the git-driven path, which builds the engine from
        its own checkout and so is embedder 0 by construction.
        """
        last_done = self.store.max_done_key(
            engine_name, self.cfg.platform, embedder_id=0
        )
        return (
            last_done.commit_id if last_done else None,
            self.head_commit_id(engine_name, fetch=fetch),
        )

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
        clear: bool = False,
        include_start: bool = True,
        should_stop: Callable[[], bool] | None = None,
        max_commits: int | None = None,
    ):
        """Benchmark commits in (start_id, end_id].

        Returns the number measured. ``max_commits`` bounds attempts per turn
        so watch can hand the next lock acquisition to another engine.

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
        cleared: set[CommitKey] = set()
        if clear:
            clear_keys = [CommitKey.from_commit(r) for r in sampled]
            self._log(
                f"[yellow]Clearing results and state for {len(clear_keys)} commits...[/yellow]"
            )
            if not self.dry_run:
                with self.store.result_locks(
                    engine_name, self.cfg.platform, clear_keys
                ):
                    self.store.clear_range(engine_name, self.cfg.platform, clear_keys)
                    for key in clear_keys:
                        p = self.cfg.commit_results_dir(engine_name, key)
                        if p.exists():
                            shutil.rmtree(p)
            else:
                cleared = set(clear_keys)

        t_start = time.time()
        total = len(sampled)
        attempted = 0
        completed = 0
        for idx, row in enumerate(sampled):
            if max_commits is not None and attempted >= max_commits:
                break
            key = CommitKey.from_commit(row)
            commit_id = str(key)
            commit_hash = row["hash"]

            eta = ""
            if idx > 0:
                remaining = (time.time() - t_start) / idx * (total - idx)
                eta = str(timedelta(seconds=int(remaining)))

            self._log(
                f"\n[bold green]=== [{idx + 1}/{total}] ID: {commit_id} "
                f"({commit_hash[:8]}) | ETA: {eta or 'N/A'}[/bold green]"
            )

            if key not in cleared and self.store.is_done(
                engine_name, self.cfg.platform, key
            ):
                self._log("  [yellow]Skipping: already done[/yellow]")
                continue

            # The build and every run of this commit happen under one hold, so
            # a peer's build cannot land between two of its runs.
            if not self._take_machine_lock(should_stop, f"{engine_name} {key}"):
                break
            attempted += 1
            try:
                t0 = time.time()
                console.print("  Building... ", end="")
                run_root = self._provision(engine, commit_hash)
                if run_root is None:
                    console.print(f"[red]FAILED ({int(time.time() - t0)}s)[/red]")
                    continue
                console.print(f"[green]OK ({int(time.time() - t0)}s)[/green]")

                self.bench_at_root(engine, dict(row), run_root)
                completed += 1
            finally:
                self.lock.release()

            if should_stop():
                break
        return completed
