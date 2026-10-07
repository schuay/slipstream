# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Safari: system Safari's development launcher around the run root.

The artifact is a WebKit build plus a copy of Safari Technology Preview;
the process is the host's ``SafariForWebKitDevelopment``, the one Safari
executable entitled to take ``DYLD_*`` overrides. The four search-path
variables put the run root's ``WebKitBuild/Release`` first and the packaged
STP's ``Contents/Frameworks`` second, so the engine is ours and the browser
around it is the STP the entry names. They are passed through ``arch -e``:
SIP strips ``DYLD_*`` from a protected binary's environment, but ``arch``
hands its ``-e`` pairs to the program it execs.

Nothing about that launch is self-evidencing. STP's own executable ignores
the overrides and runs the system engine with no visible difference, and
``open -a`` starts stock Safari; both were measured once and discarded. So
a run is accepted only on evidence from the processes themselves: the
launcher has loaded ``Safari.framework`` from the run root, and every
WebContent process with a ``JavaScriptCore`` mapped has it from the run
root, at least one of them existing. WebContent processes are launchd's
children, not ours, which is why the check is over all of them and why
the benchmark account is drained of Safari and WebContent before a run.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import time
from pathlib import Path

from ..config import EngineConfig
from ..hostapp import bundle_of, describe
from .base import RunRequest, RunResult
from .browser import GRACE_SECONDS, BrowserRunner, Command, Verdict, _signal_group

ARCH = "/usr/bin/arch"
DYLD_VARS = (
    "DYLD_FRAMEWORK_PATH",
    "DYLD_LIBRARY_PATH",
    "__XPC_DYLD_FRAMEWORK_PATH",
    "__XPC_DYLD_LIBRARY_PATH",
)
# Verified by hand before any of this existed: a window at the URL, no
# session restore, no state restoration prompt. The URL goes in as the
# home page because handing it to an already-running Safari is what `open`
# does, and that is the stock one.
LAUNCH_ARGS = (
    "-NewWindowBehavior", "0",
    "-NewTabBehavior", "0",
    "-AlwaysRestoreSessionAtLaunch", "0",
    "-ApplePersistenceIgnoreStateQuietly", "1",
)  # fmt: skip
WEBCONTENT = "com.apple.WebKit.WebContent"
SAFARI_PROCESSES = ("Safari", "SafariForWebKitDevelopment", "Safari Technology Preview")
JSC_IMAGE = "JavaScriptCore.framework/Versions/A/JavaScriptCore"
SAFARI_IMAGE = "Safari.framework/Versions/A/Safari"
QUIET_SECONDS = 0.3
POLL_SECONDS = 0.1


def _is_macos() -> bool:
    return platform.system() == "Darwin"


class SafariRunner(BrowserRunner):
    runtime = "safari"

    def __init__(self, **kw):
        super().__init__(**kw)
        self._content_pids: list[int] = []
        self._retryable = False
        self._recovery_problem = ""

    def conventions(self) -> tuple[str, ...]:
        return (
            f"{ARCH} -arm64e -e <var>=<dyld_search_path> for " + ", ".join(DYLD_VARS),
            "<launcher> -HomePage <url>",
            *LAUNCH_ARGS,
            "drain current-user Safari/WebContent before and after each run",
            "retry launch/provenance failure once after cleanup",
        )

    def run(self, req: RunRequest) -> RunResult:
        self._retryable = False
        self._recovery_problem = ""
        result = super().run(req)
        if not result.ok and self._retryable:
            stderr = req.artifact("stderr", "txt")
            if stderr.exists():
                with stderr.open("a") as output:
                    output.write(f"\nrecovery reason: {self._recovery_problem}\n")
                stderr.replace(req.artifact("stderr-recovery", "txt"))
            self._log("Safari recovery: retrying the measurement after cleanup")
            self._retryable = False
            result = super().run(req)
        return result

    def _fail(self, label: str, what: str) -> None:
        if what.startswith(("provenance:", "browser exited", "cannot start browser:")):
            self._retryable = True
            self._recovery_problem = what
        super()._fail(label, what)

    def host_env(self, engine: EngineConfig, run_root: Path) -> dict[str, str]:
        # host_app is the STP the entry packaged, read from the run root;
        # launcher is the macOS Safari whose SafariForWebKitDevelopment ran it,
        # which is host state and the one thing here the archive does not pin.
        apps = [Path(run_root) / e for e in engine.run_set if e.endswith(".app")]
        return {
            "host_app": describe(apps[0]) if apps else "",
            "launcher": describe(bundle_of(engine.resolve_binary(run_root))),
        }

    # --- launch ---

    def command(self, req: RunRequest, url: str) -> Command:
        search = req.engine.dyld_search_path(req.run_root) or ""
        launcher = str(req.engine.resolve_binary(req.run_root))
        args = [launcher, "-HomePage", url, *LAUNCH_ARGS]
        env = dict(os.environ)
        if _is_macos():
            # arch execs the launcher under the same pid, so the process the
            # collector holds is Safari's, and the group it leads is ours.
            prefix = [ARCH, "-arm64e"]
            for var in DYLD_VARS:
                prefix += ["-e", f"{var}={search}"]
            return Command([*prefix, *args], env=env)
        for var in DYLD_VARS:
            env[var] = search
        return Command(args, env=env)

    def precondition(self, req: RunRequest) -> str | None:
        return self._quiesce()

    def _quiesce(self) -> str | None:
        """Bounded recovery for this dedicated benchmark account.

        Rescan throughout cleanup: launchd owns WebContent, and helpers can
        appear after verification or while their launcher is shutting down.
        """
        try:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                deadline = time.monotonic() + GRACE_SECONDS
                signaled: set[tuple[int, str]] = set()
                quiet_since = None
                while time.monotonic() < deadline:
                    running = [(p, n) for p, n in self.processes() if _is_safari(n)]
                    if not running:
                        if quiet_since is None:
                            quiet_since = time.monotonic()
                        if time.monotonic() - quiet_since >= QUIET_SECONDS:
                            return None
                    else:
                        quiet_since = None
                        for pid, name in running:
                            if (pid, name) not in signaled:
                                self._log(f"Safari cleanup: {sig.name} {name} ({pid})")
                                self._signal_process(pid, name, sig)
                                signaled.add((pid, name))
                    time.sleep(POLL_SECONDS)
            running = [(p, n) for p, n in self.processes() if _is_safari(n)]
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return f"cannot clean up Safari processes: {exc}"
        return "Safari cleanup did not reach a quiet state: " + ", ".join(
            f"{name} ({pid})" for pid, name in running[:4]
        )

    def _signal_process(self, pid: int, name: str, sig: signal.Signals) -> None:
        # Recheck user ownership and executable identity before signaling a
        # PID from an earlier snapshot; the original process may have exited.
        if (pid, name) in self.processes():
            _kill(pid, sig)

    # --- evidence ---

    def verify(self, req: RunRequest, proc: subprocess.Popen) -> Verdict:
        root = str(req.run_root)
        notes: list[str] = []

        def under_root(path: str) -> bool:
            return path.startswith(root) or path.startswith(os.path.realpath(root))

        safari = _image(self.loaded_images(proc.pid), SAFARI_IMAGE)
        notes.append(f"launcher {proc.pid}: Safari.framework from {safari or '(none)'}")
        if not safari or not under_root(safari):
            return Verdict(
                False,
                notes,
                f"the launcher runs {safari or 'no'} Safari.framework, "
                f"not the run root's",
            )

        content: list[tuple[int, str]] = []
        try:
            processes = self.processes()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return Verdict(False, notes, f"cannot inspect WebContent processes: {exc}")
        for pid, name in processes:
            if WEBCONTENT not in name:
                continue
            jsc = _image(self.loaded_images(pid), JSC_IMAGE)
            if jsc:
                content.append((pid, jsc))
                notes.append(f"WebContent {pid}: JavaScriptCore from {jsc}")
        if not content:
            return Verdict(False, notes, "no WebContent process has loaded an engine")
        foreign = [f"{pid}: {jsc}" for pid, jsc in content if not under_root(jsc)]
        if foreign:
            return Verdict(
                False,
                notes,
                "a WebContent process runs an engine outside the run "
                "root: " + "; ".join(foreign),
            )
        self._content_pids = [pid for pid, _ in content]
        return Verdict(True, notes)

    # --- teardown ---

    def terminate(self, proc: subprocess.Popen) -> None:
        try:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if proc.poll() is not None:
                    break
                _signal_group(proc, sig)
                try:
                    proc.wait(GRACE_SECONDS)
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                self._log(f"Safari cleanup: launcher {proc.pid} did not exit")
        finally:
            self._content_pids = []
            problem = self._quiesce()
            if problem:
                self._log(f"Safari cleanup: {problem}; next run will retry cleanup")

    # --- the host, behind two seams for the tests ---

    def processes(self) -> list[tuple[int, str]]:
        """``(pid, command)`` for this user's processes; the command is the
        executable path for an app, which is how a WebContent process names
        itself."""
        result = subprocess.run(
            ["ps", "-U", str(os.getuid()), "-o", "pid=,comm="],
            capture_output=True,
            text=True,
            timeout=GRACE_SECONDS,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"ps failed ({result.returncode}): {result.stderr.strip()}"
            )
        out = result.stdout
        procs = []
        for line in out.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                procs.append((int(parts[0]), parts[1]))
        return procs

    def loaded_images(self, pid: int) -> list[str]:
        """Paths mapped into ``pid``, from ``lsof``; empty if it is gone."""
        try:
            out = subprocess.run(
                ["lsof", "-F", "n", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=GRACE_SECONDS,
                check=False,
            ).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        return [line[1:] for line in out.splitlines() if line.startswith("n/")]


def _is_safari(command: str) -> bool:
    return Path(command).name in SAFARI_PROCESSES or WEBCONTENT in command


def _image(images: list[str], suffix: str) -> str | None:
    for path in images:
        if path.endswith(suffix):
            return path
    return None


def _kill(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
