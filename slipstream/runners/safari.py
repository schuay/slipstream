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
nothing of Safari's may be running when a run starts.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import time
from pathlib import Path

from .base import RunRequest
from .browser import GRACE_SECONDS, BrowserRunner, Command, Verdict

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


def _is_macos() -> bool:
    return platform.system() == "Darwin"


class SafariRunner(BrowserRunner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._content_pids: list[int] = []

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
        running = [
            f"{name} ({pid})" for pid, name in self.processes() if _is_safari(name)
        ]
        if running:
            return (
                "Safari is already running: "
                + ", ".join(running[:4])
                + ("..." if len(running) > 4 else "")
                + "; quit it, nothing of Safari's may run beside a measurement"
            )
        return None

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
        for pid, name in self.processes():
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
        super().terminate(proc)
        # The content processes we identified, and only those: they belong
        # to launchd, so the launcher's group did not reach them.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            alive = [pid for pid in self._content_pids if _alive(pid)]
            if not alive:
                break
            for pid in alive:
                _kill(pid, sig)
            deadline = time.monotonic() + GRACE_SECONDS
            while time.monotonic() < deadline and any(_alive(p) for p in alive):
                time.sleep(0.1)
        self._content_pids = []
        if _is_macos() and any(_is_safari(n) for _, n in self.processes()):
            # Last resort, for whatever of Safari's is still standing.
            subprocess.run(
                ["osascript", "-e", 'quit app "Safari"'],
                capture_output=True,
                timeout=GRACE_SECONDS,
                check=False,
            )

    # --- the host, behind two seams for the tests ---

    def processes(self) -> list[tuple[int, str]]:
        """``(pid, command)`` for every process; the command is the
        executable path for an app, which is how a WebContent process names
        itself."""
        try:
            out = subprocess.run(
                ["ps", "-axo", "pid=,comm="],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        except OSError:
            return []
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
                check=False,
            ).stdout
        except OSError:
            return []
        return [line[1:] for line in out.splitlines() if line.startswith("n/")]


def _is_safari(command: str) -> bool:
    return Path(command).name in SAFARI_PROCESSES or WEBCONTENT in command


def _image(images: list[str], suffix: str) -> str | None:
    for path in images:
        if path.endswith(suffix):
            return path
    return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
