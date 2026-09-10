# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field


@dataclass
class CommitInfo:
    id: int
    hash: str
    date: str
    timestamp: int
    title: str


@dataclass
class ChangePoint:
    benchmark: str  # "js3[default] regex-dna-SP"
    score_type: str
    commit_id: int  # where the change is detected
    prev_commit_id: int  # last benchmarked commit before the change
    direction: str  # "improvement" / "regression"
    magnitude: float  # Cohen's d
    pct_change: float  # (after - before) / before
    confidence: str  # "high" / "medium" / "low"
    seg_before_mean: float
    seg_after_mean: float
    # (commit_id, probability) pairs for alternative breakpoint locations
    candidates: list[tuple[int, float]] = field(default_factory=list)
