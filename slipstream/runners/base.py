# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""What one (run, config) pair does, behind a runtime-specific Runner.

The collector owns the loop over runs and configs, the score store and the
outcome counts; a runner owns one measurement: start the engine on the
suite, wait for it, turn what came back into scores. Which runner an engine
gets is the engine's ``runtime`` -- a shell binary is driven by argv, a
browser by a served page -- so the collector never asks what kind of thing
it is benching.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import BenchmarkConfig, EngineConfig, RunSpec
from ..models import Score

Log = Callable[[str], None]
Progress = Callable[[], None]


@dataclass(frozen=True)
class RunRequest:
    """One measurement: this engine, unpacked at this root, on this config.

    ``res_dir`` is where the runner leaves whatever it captured; the files
    are named through ``artifact`` so every runtime's leftovers sort next to
    each other by run and config.
    """

    engine: EngineConfig
    run_root: Path
    bench: BenchmarkConfig
    spec: RunSpec
    run: int
    res_dir: Path

    @property
    def run_mode(self) -> str:
        return self.spec.run_mode or self.bench.run_mode

    def artifact(self, kind: str, ext: str) -> Path:
        return (
            self.res_dir
            / f"{kind}.{self.run}.{self.spec.suite}.{self.spec.variant}.{ext}"
        )


@dataclass(frozen=True)
class RunResult:
    """``ok`` is the runner's verdict on the measurement as a whole. The
    collector discards the scores of a run that is not ok: a suite that died
    halfway has numbers for the half that ran, and those would read as a
    clean run of a shorter suite."""

    ok: bool
    scores: list[Score]


class Runner(Protocol):
    def run(self, req: RunRequest) -> RunResult: ...

    def cfg_hash(self) -> str:
        """Identifies how this runner drives an engine: the flags every run
        gets, the launch mechanism, the protocol the numbers come back by.
        Recorded beside ``build_cfg_hash`` so a change to the harness side
        shows in provenance the way a change to the build does. The
        ``[[run]]`` flags are not in it: they are the variant."""
        ...

    def host_env(self, engine: EngineConfig, run_root: Path) -> dict[str, str]:
        """Host facts a run depends on that are neither the build nor this
        runner: the application a browser ran inside, the launcher that
        started it. Empty for a shell binary."""
        ...


def cfg_digest(kind: str, *parts: str) -> str:
    """``sha256:`` over ``kind`` and its conventions, in the form of
    ``build_cfg_hash`` so the two columns read alike."""
    import hashlib

    payload = "\n--\n".join((kind, *parts))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def geomean_overall(scores: list[Score], run: int) -> list[Score]:
    """Synthesise ``Overall Total-Score`` as the geomean of every benchmark's.

    For the runs where the harness prints no overall of its own: a shell run
    per line item, or a browser report, which carries only the per-test
    numbers.
    """
    total_scores: list[float] = []
    suite = None
    flags = None
    for s in scores:
        if s.benchmark == "Overall":
            continue
        if s.metric != "Total-Score":
            continue
        suite = s.suite
        flags = s.flags
        total_scores.append(s.score)

    positive = [v for v in total_scores if v > 0]
    if not positive:
        return []
    geomean = math.exp(sum(math.log(v) for v in positive) / len(positive))
    return [Score(suite, flags, "Overall", "Total-Score", run, geomean)]
