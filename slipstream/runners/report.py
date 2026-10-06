# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""What a browser POSTs to ``/report``, turned into scores.

Two formats, named by the suite's ``report_format``.

``jetstream`` is run-benchmark's, which JetStream's driver emits under
``?report=true``. Both JetStream suites emit the same shape, one suite key
at the top::

    {"JetStream3.0": {"metrics": {...}, "tests": {
        "<benchmark>": {"metrics": {"Score": {"current": [s]}, ...},
                        "tests": {"<sub>": {"metrics": {...: {"current": [v]}}}}}}}}

A benchmark's ``Score`` is the shell's ``Total-Score``; a sub-test's score
is the shell's ``<sub>-Score`` unless the suite's ``report_metrics`` renames
it. JetStream 2 puts the sub-score under ``Score`` and the sub-time under
``Time``; JetStream 3 reports only the sub-score, under ``Time``, which is
what run-benchmark expects there. Times are not kept either way.

``speedometer`` is what slipstream's injected ``sp3-report.mjs`` posts:
``{"metrics": benchmarkClient.metrics}`` or ``{"error": {...}}``. The
metrics are a flat map of ``name -> Metric`` with ``/`` separating a suite
from its steps::

    {"TodoMVC-React-Complex-DOM": {"unit": "ms", "mean": m, "values": [...], ...},
     "TodoMVC-React-Complex-DOM/Adding100Items": {...},
     "TodoMVC-React-Complex-DOM/Adding100Items/sync": {...},
     "Iteration-0-Total": {...}, "Geomean": {...},
     "Score": {"unit": "score", "mean": s, ...}}

What is kept is what crossbench keeps without ``--detailed-metrics``: the
top-level names that are neither ``Iteration-*`` nor ``Geomean``, by their
``mean`` over the page's iterations. A suite's mean is milliseconds, lower
is better, and goes in as ``Total-Time``; ``Score`` is Speedometer's own
overall and goes in as ``Overall/Total-Score``, so the runner has nothing
to synthesise. The metric name is the unit all the way to the perf
database, which keys its traces on it.
"""

from __future__ import annotations

import json
import math
from typing import Any

from ..models import Score

# Bumped when a parser's output for the same body changes; part of the
# suite's cfg_hash, so the change shows in provenance.
PARSER_VERSIONS = {"jetstream": "1", "speedometer": "1"}

# The metrics that name a benchmark's headline number, in either direction.
TOTAL_METRICS = ("Total-Score", "Total-Time")


class ReportError(ValueError):
    """The body is not a report of the expected format."""


def parse_for(report_format: str):
    """The parser for a suite's ``report_format``; the same signature for
    each, ``(body, suite, flags, run, report_metrics)``."""
    try:
        return _PARSERS[report_format]
    except KeyError:
        raise ValueError(f"no parser for report format {report_format!r}") from None


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


def parse_speedometer(
    body: bytes, suite: str, flags: str, run: int, report_metrics: dict[str, str]
) -> list[Score]:
    try:
        doc = json.loads(body)
    except ValueError as exc:
        raise ReportError(f"report is not JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise ReportError("report is not an object")
    error = doc.get("error")
    if error is not None:
        message = error.get("message") if isinstance(error, dict) else error
        raise ReportError(f"the page reported an error: {message}")
    metrics = doc.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ReportError("report has no metrics")

    scores: list[Score] = []
    overall = None
    for name, metric in metrics.items():
        if "/" in name or name.startswith("Iteration-") or name == "Geomean":
            continue
        value = _mean(metric)
        if value is None:
            continue
        if name == "Score":
            overall = Score(suite, flags, "Overall", "Total-Score", run, value)
        else:
            scores.append(Score(suite, flags, name, "Total-Time", run, value))
    if overall is None:
        raise ReportError("report has no Score")
    scores.append(overall)
    return scores


_PARSERS = {"jetstream": parse_report, "speedometer": parse_speedometer}


def _current(metric: Any) -> float | None:
    """The one value under ``{"current": [v]}``; None for anything else,
    which includes a failed benchmark's null score."""
    if not isinstance(metric, dict):
        return None
    current = metric.get("current")
    if not isinstance(current, list) or len(current) != 1:
        return None
    v = current[0]
    return _number(v)


def _mean(metric: Any) -> float | None:
    """A Speedometer Metric's ``mean``; None when it is not a finite
    number, which is what a suite that never ran leaves behind."""
    if not isinstance(metric, dict):
        return None
    return _number(metric.get("mean"))


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if not math.isfinite(v):
        return None
    return float(v)
