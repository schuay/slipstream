# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import math
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import ruptures
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.table import Table
from rich import box

from . import compat
from .models import ChangePoint, CommitInfo
from .store import CommitStore

console = Console()


def _stats(samples: list[float]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    mean = sum(samples) / len(samples)
    stdev = (
        math.sqrt(sum((x - mean) ** 2 for x in samples) / (len(samples) - 1))
        if len(samples) > 1
        else 0.0
    )
    return mean, stdev


class PerfAnalyzer:
    def __init__(
        self,
        min_change: float = 0.01,
        penalty: float = 3.0,
        min_effect_size: float = 0.5,
    ):
        self.min_change = min_change
        self.penalty = penalty
        self.min_effect_size = min_effect_size
        # data[bench_key][metric][commit_id] = [scores]
        self.data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.commits: dict[int, CommitInfo] = {}
        self._store: CommitStore | None = None
        self._engine: str | None = None

    # --- Data loading ---

    def load_from_db(self, store: CommitStore, engine: str):
        """Load all score data from the SQLite database."""
        self._store = store
        self._engine = engine

        for row in store.get_distinct_series_keys(engine):
            suite, flags, benchmark, metric = row
            bench_key = f"{suite}[{flags}] {benchmark}"
            for sr in store.get_series(engine, suite, flags, benchmark, metric):
                self.data[bench_key][metric][sr["commit_id"]].append(sr["score"])

        for row in store.get_all_commits(engine):
            self.commits[row["commit_id"]] = CommitInfo(
                id=row["commit_id"],
                hash=row["hash"],
                date=row["date"],
                timestamp=row["timestamp"],
                title=row["title"],
            )

    def load_results(self, csv_path: Path) -> bool:
        """Load from the interchange CSV (see slipstream.compat).

        A file carrying more than one bot is refused: the two machines'
        series would be pooled into one and change points reported across the
        seam. Older files carry no bot and are trusted.
        """
        if not csv_path.exists():
            console.print(f"[red]Error: {csv_path} not found.[/red]")
            return False

        with open(csv_path) as f:
            first = next(csv.reader(f), [])
            f.seek(0)
            try:
                fieldnames, has_bot = compat.sniff(first)
            except ValueError as e:
                console.print(f"[red]Error: {e} in {csv_path}[/red]")
                return False

            bots: set[str] = set()
            for raw_row in csv.DictReader(f, fieldnames=fieldnames):
                try:
                    # A short row leaves None here and an over-long one
                    # collects the rest under a None key; a merged file, which
                    # is what the multi-bot check exists for, is the one most
                    # likely to be ragged.
                    row = {
                        k.strip(): (v or "").strip()
                        for k, v in raw_row.items()
                        if k is not None
                    }
                    if has_bot:
                        bots.add(row["bot"])
                        if len(bots) > 1:
                            console.print(
                                f"[red]Error: {csv_path} holds more than one bot "
                                f"({', '.join(sorted(bots))}); analyzing them "
                                f"together would report change points across "
                                f"the seam.[/red]"
                            )
                            return False
                    if "b_type" in row:
                        key = f"{row['b_type']}[{row['flags']}] {row['benchmark']}"
                    else:
                        key = row["benchmark"]
                    self.data[key][row["score_type"]][int(row["commit_id"])].append(
                        float(row["score"])
                    )
                except (ValueError, KeyError):
                    continue
        return True

    def load_commit_infos(self, pattern: str):
        """Load commit metadata from CSV glob (backward-compatible fallback)."""
        for path in glob(pattern):
            with open(path) as f:
                for row in csv.reader(f):
                    if not row or len(row) < 5:
                        continue
                    try:
                        cid = int(row[0].strip())
                        self.commits[cid] = CommitInfo(
                            id=cid,
                            hash=row[1].strip(),
                            date=row[2].strip(),
                            title=row[3].strip(),
                            timestamp=int(row[4].strip()),
                        )
                    except (ValueError, IndexError):
                        continue

    def filter_by_date(self, since: str | None = None, until: str | None = None):
        """Remove commits (and their scores) outside the given date range."""
        to_remove = set()
        for cid, info in self.commits.items():
            if not info.date:
                continue
            if since and info.date < since:
                to_remove.add(cid)
            if until and info.date > until:
                to_remove.add(cid)
        for cid in to_remove:
            del self.commits[cid]
        for bench in self.data:
            for metric in self.data[bench]:
                for cid in to_remove:
                    self.data[bench][metric].pop(cid, None)

    # --- PELT analysis ---

    def _analyze_series(
        self,
        bench_key: str,
        metric: str,
        commit_scores: dict[int, list[float]],
    ) -> list[ChangePoint]:
        """Run PELT on a single benchmark time series."""
        cids = sorted(commit_scores.keys())
        if len(cids) < 4:
            return []

        means = []
        stdevs = []
        for cid in cids:
            m, s = _stats(commit_scores[cid])
            means.append(m)
            stdevs.append(s)

        means_arr = np.array(means)
        stdevs_arr = np.array(stdevs)

        # Reliability check: median CV
        valid = means_arr > 0
        if not valid.any():
            return []
        cvs = np.where(valid, stdevs_arr / means_arr, 0.0)
        median_cv = float(np.median(cvs[valid]))
        confidence = (
            "high" if median_cv < 0.05 else "medium" if median_cv < 0.15 else "low"
        )

        # PELT on mean time series
        signal = means_arr.reshape(-1, 1)
        algo = ruptures.Pelt(model="rbf", min_size=2)
        try:
            bkps = algo.fit_predict(signal, pen=self.penalty)
        except Exception:
            return []

        # Refine PELT breakpoints by minimising within-segment SSR in a ±3
        # window.  PELT finds the right number of changes but its exact
        # positions can be off by ±1-2 samples; a local MLE scan fixes this.
        # We keep per-candidate SSRs to derive location probabilities later.
        n_pts = len(cids)
        refined_bkps = []
        # candidate_ssrs[i] = {array_index: ssr} for the i-th breakpoint
        candidate_ssrs: list[dict[int, float]] = []
        prev_edge = 0
        for i, bk in enumerate(bkps):
            if bk >= n_pts:
                break
            next_edge = bkps[i + 1] if i + 1 < len(bkps) else n_pts
            best_bk = bk
            best_ssr = float("inf")
            ssrs: dict[int, float] = {}
            lo = max(prev_edge + 1, bk - 3)
            hi = min(next_edge, bk + 4)  # exclusive
            for candidate in range(lo, hi):
                sb = means_arr[prev_edge:candidate]
                sa = means_arr[candidate:next_edge]
                if len(sb) < 1 or len(sa) < 1:
                    continue
                ssr = float(
                    np.sum((sb - sb.mean()) ** 2) + np.sum((sa - sa.mean()) ** 2)
                )
                ssrs[candidate] = ssr
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_bk = candidate
            refined_bkps.append(best_bk)
            candidate_ssrs.append(ssrs)
            prev_edge = best_bk

        # Build segments from refined breakpoints
        results = []
        prev_bk = 0
        all_bkps = refined_bkps + [len(cids)]
        for i, bk in enumerate(all_bkps):
            if bk >= len(cids):
                break
            # Segment before: [prev_bk, bk), segment after: [bk, next_bk)
            next_bk = all_bkps[i + 1] if i + 1 < len(all_bkps) else len(cids)
            seg_before = means_arr[prev_bk:bk]
            seg_after = means_arr[bk:next_bk]

            if len(seg_before) < 1 or len(seg_after) < 1:
                prev_bk = bk
                continue

            m_before = float(np.mean(seg_before))
            m_after = float(np.mean(seg_after))

            if m_before == 0:
                prev_bk = bk
                continue

            pct_change = (m_after - m_before) / m_before

            # Cohen's d from raw samples
            samples_before = []
            for cid in cids[prev_bk:bk]:
                samples_before.extend(commit_scores[cid])
            samples_after = []
            for cid in cids[bk:next_bk]:
                samples_after.extend(commit_scores[cid])

            sb_mean, sb_std = _stats(samples_before)
            sa_mean, sa_std = _stats(samples_after)
            n_b, n_a = len(samples_before), len(samples_after)
            denom = max(n_b + n_a - 2, 1)
            pooled_std = math.sqrt(
                ((n_b - 1) * sb_std**2 + (n_a - 1) * sa_std**2) / denom
            )
            cohens_d = (sa_mean - sb_mean) / pooled_std if pooled_std > 0 else 0.0

            if (
                abs(pct_change) < self.min_change
                and abs(cohens_d) < self.min_effect_size
            ):
                prev_bk = bk
                continue

            direction = "improvement" if pct_change > 0 else "regression"

            # Compute candidate probabilities from profile likelihood:
            # P(bk=k) ∝ SSR(k)^(-n/2)  (Gaussian piecewise-constant model)
            candidates: list[tuple[int, float]] = []
            ssrs = candidate_ssrs[i]
            if ssrs:
                positions = sorted(ssrs.keys())
                log_liks = [
                    -(n_pts / 2) * math.log(max(ssrs[c], 1e-20)) for c in positions
                ]
                max_ll = max(log_liks)
                probs = [math.exp(ll - max_ll) for ll in log_liks]
                total_p = sum(probs)
                candidates = [
                    (cids[c], probs[j] / total_p)
                    for j, c in enumerate(positions)
                    if probs[j] / total_p >= 0.01
                ]

            results.append(
                ChangePoint(
                    benchmark=bench_key,
                    score_type=metric,
                    commit_id=cids[bk],
                    prev_commit_id=cids[bk - 1],
                    direction=direction,
                    magnitude=abs(cohens_d),
                    pct_change=pct_change,
                    confidence=confidence,
                    seg_before_mean=m_before,
                    seg_after_mean=m_after,
                    candidates=candidates,
                )
            )
            prev_bk = bk

        return results

    def analyze(
        self,
        include_bench: list[str] | None = None,
        exclude_bench: list[str] | None = None,
        include_score: list[str] | None = None,
        exclude_score: list[str] | None = None,
    ) -> list[ChangePoint]:
        results = []
        for bench in sorted(self.data):
            if include_bench and not any(p in bench for p in include_bench):
                continue
            if exclude_bench and any(p in bench for p in exclude_bench):
                continue

            for metric in sorted(self.data[bench]):
                if include_score and not any(p in metric for p in include_score):
                    continue
                if exclude_score and any(p in metric for p in exclude_score):
                    continue

                commit_scores = {
                    cid: scores
                    for cid, scores in self.data[bench][metric].items()
                    if cid in self.commits
                }
                results.extend(self._analyze_series(bench, metric, commit_scores))
        return results

    def _get_commit_range(self, cp: ChangePoint) -> list[CommitInfo]:
        """Get all commits between prev_commit_id and commit_id (exclusive/inclusive)."""
        if self._store and self._engine:
            rows = self._store.get_commits_in_range(
                self._engine, cp.prev_commit_id, cp.commit_id
            )
            return [
                CommitInfo(
                    id=r["commit_id"],
                    hash=r["hash"],
                    date=r["date"],
                    timestamp=r["timestamp"],
                    title=r["title"],
                )
                for r in rows
            ]
        # CSV fallback: return only the known commits in range
        return [
            self.commits[cid]
            for cid in sorted(self.commits)
            if cp.prev_commit_id < cid <= cp.commit_id
        ]

    # --- Reporting ---

    def _format_candidates(self, cp: ChangePoint) -> str | None:
        """Format alternative breakpoint candidates, or None if unambiguous."""
        if not cp.candidates:
            return None
        top_prob = max(p for _, p in cp.candidates)
        if top_prob >= 0.90:
            return None
        parts = []
        for cid, prob in cp.candidates:
            if prob >= 0.05:
                parts.append(f"{cid}[dim]({prob:.0%})[/dim]")
        return " | ".join(parts) if parts else None

    def print_report(
        self,
        results: list[ChangePoint],
        group_by_commit: bool = False,
    ):
        if not results:
            console.print("No change points detected.")
            return

        if group_by_commit:
            groups: dict[int, list[ChangePoint]] = defaultdict(list)
            for cp in results:
                groups[cp.commit_id].append(cp)

            for cid in sorted(groups):
                info = self.commits.get(cid)
                range_commits = self._get_commit_range(groups[cid][0])
                if len(range_commits) <= 1 and info:
                    h = info.hash[:10] if info.hash else ""
                    title = rich_escape(info.title[:70]) if info.title else ""
                    header = f"Commit {cid} {h} {title}".strip()
                else:
                    prev_cid = groups[cid][0].prev_commit_id
                    header = f"Commit range {prev_cid + 1}..{cid} ({len(range_commits)} commits)"
                console.print(f"\n[bold]{header}[/bold]")
                alt = self._format_candidates(groups[cid][0])
                if alt:
                    console.print(f"  candidates: {alt}")
                    for c_cid, c_prob in groups[cid][0].candidates:
                        if c_prob < 0.05:
                            continue
                        c_info = self.commits.get(c_cid)
                        if c_info:
                            t = rich_escape(c_info.title) if c_info.title else ""
                            console.print(f"  [dim]{c_cid} {t}[/dim]")
                else:
                    for c in range_commits:
                        t = rich_escape(c.title) if c.title else ""
                        console.print(f"  [dim]{c.id} {t}[/dim]")

                table = Table(
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold",
                    padding=(0, 1),
                )
                table.add_column("BENCHMARK")
                table.add_column("METRIC")
                table.add_column("CHANGE", justify="right")
                table.add_column("EFFECT", justify="right")
                table.add_column("CONF")

                for cp in sorted(
                    groups[cid], key=lambda x: abs(x.pct_change), reverse=True
                ):
                    pct = cp.pct_change * 100
                    color = "green" if cp.direction == "improvement" else "red"
                    table.add_row(
                        rich_escape(cp.benchmark),
                        cp.score_type,
                        f"[{color}]{pct:+.2f}%[/{color}]",
                        f"{cp.magnitude:.2f}d",
                        cp.confidence,
                    )
                console.print(table)

        else:
            results = sorted(results, key=lambda x: abs(x.pct_change), reverse=True)
            table = Table(
                box=box.SIMPLE, show_header=True, header_style="bold", padding=(0, 1)
            )
            table.add_column("BENCHMARK")
            table.add_column("METRIC")
            table.add_column("CHANGE", justify="right")
            table.add_column("EFFECT", justify="right")
            table.add_column("CONF")
            table.add_column("COMMIT RANGE", no_wrap=False)

            for cp in results:
                pct = cp.pct_change * 100
                color = "green" if cp.direction == "improvement" else "red"
                range_commits = self._get_commit_range(cp)
                n = len(range_commits)
                info = self.commits.get(cp.commit_id)
                if n <= 1 and info:
                    h = info.hash[:10] if info.hash else ""
                    title = rich_escape(info.title[:40]) if info.title else ""
                    commit_desc = f"{cp.commit_id} {h} {title}".strip()
                else:
                    commit_desc = (
                        f"{cp.prev_commit_id + 1}..{cp.commit_id} ({n} commits)"
                    )
                alt = self._format_candidates(cp)
                if alt:
                    commit_desc += f"\n  also: {alt}"
                table.add_row(
                    rich_escape(cp.benchmark),
                    cp.score_type,
                    f"[{color}]{pct:+.2f}%[/{color}]",
                    f"{cp.magnitude:.2f}d",
                    cp.confidence,
                    commit_desc,
                )
            console.print(table)
