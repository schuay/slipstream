# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from .base import Log, Progress, RunRequest, RunResult, Runner, geomean_overall
from .shell import ShellRunner

__all__ = [
    "Log",
    "Progress",
    "RunRequest",
    "RunResult",
    "Runner",
    "ShellRunner",
    "geomean_overall",
    "runner_for",
]


def runner_for(runtime: str, *, log: Log, progress: Progress) -> Runner:
    """The runner for an engine's ``runtime``.

    The runtimes the config accepts and the ones this can drive are two
    lists on purpose: an engine whose runtime is known but not yet driven is
    a configuration this box cannot bench, which is this error, not a
    rejected config file.
    """
    if runtime == "shell":
        return ShellRunner(log=log, progress=progress)
    raise ValueError(f"no runner for runtime {runtime!r}")
