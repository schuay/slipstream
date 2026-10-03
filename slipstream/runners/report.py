# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""JetStream's run-benchmark report, as the page POSTs it to ``/report``.

Both suites emit the same shape, one suite key at the top::

    {"JetStream3.0": {"metrics": {...}, "tests": {
        "<benchmark>": {"metrics": {"Score": {"current": [s]}, ...},
                        "tests": {"<sub>": {"metrics": {...: {"current": [v]}}}}}}}}

A benchmark's ``Score`` is the shell's ``Total-Score``; a sub-test's score
is the shell's ``<sub>-Score`` unless the suite's ``report_metrics`` renames
it. JetStream 2 puts the sub-score under ``Score`` and the sub-time under
``Time``; JetStream 3 reports only the sub-score, under ``Time``, which is
what run-benchmark expects there. Times are not kept either way.
"""

from __future__ import annotations

import json
from typing import Any

from ..models import Score


class ReportError(ValueError):
    """The body is not a JetStream report."""


def parse_report(
    body: bytes, suite: str, flags: str, run: int, report_metrics: dict[str, str]
) -> list[Score]:
    try:
        doc = json.loads(body)
    except ValueError as exc:
        raise ReportError(f"report is not JSON: {exc}") from None
    if not isinstance(doc, dict) or len(doc) != 1:
        raise ReportError("report has no single suite key")
    (root,) = doc.values()
    tests = root.get("tests") if isinstance(root, dict) else None
    if not isinstance(tests, dict):
        raise ReportError("report has no tests")

    scores: list[Score] = []
    for bench, entry in tests.items():
        value = _current(entry.get("metrics", {}).get("Score"))
        if value is not None:
            scores.append(Score(suite, flags, bench, "Total-Score", run, value))
        for sub, sub_entry in entry.get("tests", {}).items():
            metrics = sub_entry.get("metrics", {})
            value = _current(metrics.get("Score", metrics.get("Time")))
            if value is None:
                continue
            metric = report_metrics.get(sub, f"{sub}-Score")
            scores.append(Score(suite, flags, bench, metric, run, value))
    return scores


def _current(metric: Any) -> float | None:
    """The one value under ``{"current": [v]}``; None for anything else,
    which includes a failed benchmark's null score."""
    if not isinstance(metric, dict):
        return None
    current = metric.get("current")
    if not isinstance(current, list) or len(current) != 1:
        return None
    v = current[0]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)
