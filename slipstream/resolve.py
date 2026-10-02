# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""What to build next, and what building it means.

The builder asks one question -- "what comes after this key?" -- and gets a
``BuildJob`` back: a key for the series, the engine commit that key measures,
and the recipe for putting the tree in that state. A ``Resolver`` answers it.
For an engine built from its own checkout (v8, jsc) the answer is the next
commit itself and the recipe is ``git checkout``; for an engine built inside
another's tree the key has two coordinates and the recipe pins the inner
engine under the outer one. The builder does not know which it has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .models import CommitKey


@dataclass(frozen=True)
class BuildJob:
    """One point on an engine's series and how to produce it.

    ``commit`` is the engine's own commit -- hash, commit_id, date, timestamp,
    title -- the one the key's ``commit_id`` names and the one every row and
    bus entry describes. ``checkout_hash`` is what ``git checkout`` gets in
    the engine's ``src_dir``; for an engine that is its own embedder it is
    ``commit["hash"]``, otherwise it is the embedder's commit and ``pins``
    says where the engine goes under it. ``embedder`` is that outer commit's
    identity, empty when there is none.
    """

    key: CommitKey
    commit: dict
    checkout_hash: str
    pins: dict[str, str] = field(default_factory=dict)
    embedder: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return self.commit["hash"]


class Resolver(Protocol):
    """The series of one engine, read from wherever its history lives."""

    def fetch(self) -> None:
        """Bring the history up to date. Raises FetchError if it cannot."""

    def next_after(self, frontier: CommitKey) -> BuildJob | None:
        """The lowest key above ``frontier`` with something to build, or None."""

    def for_key(self, key: CommitKey) -> BuildJob | None:
        """The job for exactly ``key``, or None if the key names nothing."""


class IdentityResolver:
    """An engine whose series is its own commit log: key = (0, commit_id).

    Wraps the collector's git walk rather than owning one, so the same
    checkout, path_filter and id_regex serve the builder and the local bench.
    """

    def __init__(self, collector, engine):
        self.collector = collector
        self.engine = engine

    def fetch(self) -> None:
        self.collector.head_commit_id(self.engine.name)

    def next_after(self, frontier: CommitKey) -> BuildJob | None:
        self._own(frontier)
        commit = self.collector.next_commit_after(self.engine, frontier.commit_id)
        return self._job(commit)

    def for_key(self, key: CommitKey) -> BuildJob | None:
        self._own(key)
        commit = self.collector.commit_metadata_for_id(self.engine, key.commit_id)
        return self._job(commit)

    def _own(self, key: CommitKey) -> None:
        # A key with an embedder coordinate names a point this engine's
        # series does not have; it can only come from a [build] from entry
        # or a --retry written for a different kind of engine.
        if key.embedder_id != 0:
            raise ValueError(
                f"{self.engine.name} is built from its own checkout; "
                f"{key} names an embedder it does not have"
            )

    def _job(self, commit: dict | None) -> BuildJob | None:
        if commit is None:
            return None
        return BuildJob(
            key=CommitKey(0, int(commit["commit_id"])),
            commit=commit,
            checkout_hash=commit["hash"],
        )


class EmbedderResolver:
    """An engine built inside another's tree: key = (outer position, inner id).

    chrome under V8 rolls. The outer checkout's history is read for commits
    that move the inner pin in ``roll_file`` (chromium's DEPS); each such
    roll CL ``R`` taking the pin from ``old`` to ``new`` expands into the
    inner commits ``old, old+1, ..., new``, every one keyed ``(pos(R), id)``
    and built by checking out ``R`` and pinning the inner engine beneath it.
    ``(R, old)`` is ``R``'s parent's code under ``R``'s tree, so the step
    onto it isolates the chromium-side delta; the steps after it isolate one
    V8 commit each.

    The scan is bounded by the frontier: its outer coordinate names the roll
    being walked, so "what is next" is either further inside that roll or
    the first forward roll after it. Reverts are rolls whose ``new`` is not
    a descendant of ``old`` and are skipped; the re-roll that follows covers
    the same inner range, and resumes at the frontier's inner commit under
    the new outer one -- which is itself a chromium-only step, correctly
    keyed. A roll whose ``new`` is at or below the frontier's inner commit
    adds nothing and is skipped too.

    Inner history is read from the inner engine's own checkout, not from the
    copy under the outer tree, which sits at whatever was last synced.
    """

    def __init__(
        self,
        collector,
        outer,
        inner,
        *,
        pin: str,
        roll_file: str,
        roll_regex: str,
    ):
        self.collector = collector
        self.outer = outer
        self.inner = inner
        self.pin = pin
        self.roll_file = roll_file
        self.roll_regex = re.compile(roll_regex)

    # --- Resolver ---

    def fetch(self) -> None:
        self.collector.head_commit_id(self.outer.name)
        self.collector.head_commit_id(self.inner.name)

    def next_after(self, frontier: CommitKey) -> BuildJob | None:
        outer_pos, inner_id = frontier
        start = None
        if outer_pos:
            start = self.collector._commit_hash_from_id(self.outer, outer_pos)
            if not start:
                raise ValueError(
                    f"{self.outer.name} has no commit at position {outer_pos}"
                )
            bounds = self._bounds(
                self._git(self.outer, f"show --format= -p {start} -- {self.roll_file}")
            )
            if bounds and self._forward(*bounds):
                # Still inside the frontier's roll: the next inner commit
                # above the one just built, up to the roll's new pin.
                job = self._first_within(start, *bounds, min_id=inner_id + 1)
                if job:
                    return job
        rng = f"{start}..origin/main" if start else "origin/main"
        for roll, (old, new) in self._rolls(rng):
            if not self._forward(old, new):
                continue
            new_id = self._inner_id(new)
            if new_id is None or new_id <= inner_id:
                continue
            # A fresh roll starts at its own old pin -- the chromium-only
            # step -- unless the frontier is already past it (a re-roll after
            # a revert), in which case it resumes there.
            job = self._first_within(roll, old, new, min_id=inner_id)
            if job:
                return job
        return None

    def for_key(self, key: CommitKey) -> BuildJob | None:
        job = self.next_after(key.before())
        return job if job and job.key == key else None

    # --- the outer side ---

    def _git(self, engine, args: str) -> str:
        res = self.collector._run(
            f"git {args}",
            cwd=engine.require_src_dir(),
            capture=True,
            caffeinate=False,
        )
        return res.stdout if res.returncode == 0 else ""

    def _bounds(self, patch: str) -> tuple[str, str] | None:
        """The pin before and after, from the diff of ``roll_file``.

        Read off the removed and added lines rather than from two ``git
        show``s of the file: one call per candidate, and no parent to name
        for a root commit.
        """
        old = new = None
        for line in patch.splitlines():
            if line.startswith(("---", "+++")) or len(line) < 2:
                continue
            m = self.roll_regex.search(line[1:])
            if not m:
                continue
            if line[0] == "-":
                old = m.group(1)
            elif line[0] == "+":
                new = m.group(1)
        return (old, new) if old and new and old != new else None

    def _rolls(self, rng: str):
        """Commits in ``rng`` that move the pin, oldest first, with bounds."""
        out = self._git(
            self.outer,
            f"log --reverse --format=%x00%H -p {rng} -- {self.roll_file}",
        )
        for chunk in out.split("\0"):
            if not chunk.strip():
                continue
            sha, _, patch = chunk.partition("\n")
            bounds = self._bounds(patch)
            if bounds:
                yield sha.strip(), bounds

    def _outer_commit(self, sha: str) -> dict | None:
        raw = self._git(
            self.outer,
            f'log -1 --pretty=format:"{self.collector._METADATA_FORMAT}" {sha}',
        )
        return self.collector._parse_commit_metadata(self.outer, raw)

    # --- the inner side ---

    def _inner_id(self, sha: str) -> int | None:
        found = self.collector._commit_id_from_hash(self.inner, sha)
        return int(found) if found else None

    def _forward(self, old: str, new: str) -> bool:
        """A roll, not a revert of one: ``new`` descends from ``old``."""
        res = self.collector._run(
            f"git merge-base --is-ancestor {old} {new}",
            cwd=self.inner.require_src_dir(),
            capture=True,
            caffeinate=False,
        )
        return res.returncode == 0

    def _first_within(
        self, roll: str, old: str, new: str, *, min_id: int
    ) -> BuildJob | None:
        """The lowest inner commit in ``old..=new`` with id >= ``min_id``.

        ``old`` itself is always a candidate -- it is the roll's base point,
        whatever the inner engine's path_filter says -- and the commits
        above it are the ones the inner engine's own cadence would build.
        Both bounds are known to the checkout: ``_forward`` said so.
        """
        fmt = self.collector._METADATA_FORMAT
        path_filter = self.inner.path_filter or ""
        base = self._git(self.inner, f'log -1 --pretty=format:"{fmt}" {old}')
        rest = self._git(
            self.inner,
            f'log --reverse --pretty=format:"{fmt}" {old}..{new} -- {path_filter}',
        )
        for raw in (base + rest).split("--END-COMMIT--"):
            if not raw.strip():
                continue
            commit = self.collector._parse_commit_metadata(self.inner, raw)
            if commit and commit["commit_id"] >= min_id:
                return self._job(roll, commit)
        return None

    def _job(self, roll: str, commit: dict) -> BuildJob:
        outer = self._outer_commit(roll)
        if outer is None:
            raise ValueError(
                f"{self.outer.name} commit {roll[:12]} has no position; "
                f"it cannot key a build"
            )
        return BuildJob(
            key=CommitKey(int(outer["commit_id"]), int(commit["commit_id"])),
            commit=commit,
            checkout_hash=roll,
            pins={self.pin: commit["hash"]},
            embedder={
                "hash": outer["hash"],
                "commit_id": int(outer["commit_id"]),
                "title": outer.get("title", ""),
            },
        )
