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
from pathlib import Path
from typing import Callable, NamedTuple, Protocol

from .bus import Bus, Entry
from .collector import FetchError
from .hostapp import HostApp, HostAppError, read_app
from .models import CommitKey


class _Point(NamedTuple):
    """A commit on the inner engine's main line, with the id the series uses."""

    sha: str
    id: int


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

    ``inherits`` is set for a derived engine: the inner engine's published
    entry whose blobs this job's entry will name. There is then nothing to
    check out or compile, and ``checkout_hash`` is empty.
    """

    key: CommitKey
    commit: dict
    checkout_hash: str
    pins: dict[str, str] = field(default_factory=dict)
    embedder: dict = field(default_factory=dict)
    inherits: Entry | None = None

    @property
    def hash(self) -> str:
        return self.commit["hash"]

    @property
    def compiles(self) -> bool:
        return self.inherits is None


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

    What chromium pins is not a main commit. V8 cuts a release branch for
    every roll: ``main@{#N}`` plus one "Version X.Y.Z" commit that bumps
    ``include/v8-version.h``, positioned ``refs/heads/X.Y.Z@{#1}`` and
    carrying ``Cr-Branched-From: <sha>-refs/heads/main@{#N}``. Two such
    endpoints share no branch, so neither is the other's ancestor, and the
    position each carries is ``1``. Each endpoint is therefore read as the
    main commit it was cut from -- the series is main's, and the ids on it
    are what the inner engine's own series uses -- and the roll is compared,
    expanded and keyed in those terms. Every key ``(R, N)`` pins ``main@{#N}``
    then, the endpoints included; what the shipped Chrome had there differs
    from that by the version string alone.

    The scan is bounded by the frontier: its outer coordinate names the roll
    being walked, so "what is next" is either further inside that roll or
    the first forward roll after it. Reverts are rolls whose ``new`` is below
    ``old`` on main and are skipped; the re-roll that follows covers the same
    inner range, and resumes at the frontier's inner commit under the new
    outer one -- which is itself a chromium-only step, correctly keyed. A
    roll whose ``new`` is below the frontier's inner commit adds nothing and
    is skipped too; one ending exactly on it is that chromium-only step.

    Inner history is read from the inner engine's own checkout, not from the
    copy under the outer tree, which sits at whatever was last synced. An
    endpoint that checkout has not fetched -- a fresh roll, usually -- is
    fetched by hash on the spot; one that cannot be is a FetchError, so the
    builder retries next cycle rather than scanning past the roll and
    building a later one, which would have moved the frontier over commits
    nobody built.
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
        # What a sha stands for never changes, and consecutive steps ask
        # about the same roll and the same two endpoints: remember them.
        self._points: dict[str, _Point] = {}
        self._outer_commits: dict[str, dict] = {}

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
            ends = self._endpoints(
                self._bounds(
                    self._git(
                        self.outer, f"show --format= -p {start} -- {self.roll_file}"
                    )
                )
            )
            if ends and self._forward(*ends):
                # Still inside the frontier's roll: the next inner commit
                # above the one just built, up to the roll's new pin.
                job = self._first_within(start, *ends, min_id=inner_id + 1)
                if job:
                    return job
        rng = f"{start}..origin/main" if start else "origin/main"
        for roll, bounds in self._rolls(rng):
            ends = self._endpoints(bounds)
            if ends is None or not self._forward(*ends):
                continue
            if ends[1].id < inner_id:
                continue
            # A fresh roll starts at its own old pin -- the chromium-only
            # step -- unless the frontier is already past it (a re-roll after
            # a revert), in which case it resumes there.
            job = self._first_within(roll, *ends, min_id=inner_id, preserve_first=True)
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
        if sha in self._outer_commits:
            return self._outer_commits[sha]
        raw = self._git(
            self.outer,
            f'log -1 --pretty=format:"{self.collector._METADATA_FORMAT}" {sha}',
        )
        commit = self.collector._parse_commit_metadata(self.outer, raw)
        if commit:
            self._outer_commits[sha] = commit
        return commit

    # --- the inner side ---

    def _endpoints(
        self, bounds: tuple[str, str] | None
    ) -> tuple[_Point, _Point] | None:
        """Both pins of a roll as main-line points, or None if either is
        nothing the inner series has a place for."""
        if bounds is None:
            return None
        old, new = (self._main_point(sha) for sha in bounds)
        return (old, new) if old and new else None

    def _main_point(self, sha: str) -> _Point | None:
        """The main commit a pin stands for, with the id the series uses.

        The pin itself when it is on main; otherwise the commit named by its
        ``Cr-Branched-From`` trailer, which is how a release-branch head says
        where it was cut. The id is read off that main commit with the inner
        engine's own id_regex, so the resolver keeps no second notion of it.
        """
        if sha in self._points:
            return self._points[sha]
        self._ensure_fetched(sha)
        if self._on_main(sha):
            base = sha
        else:
            m = re.search(
                r"^ *Cr-Branched-From: ([0-9a-f]{40})",
                self._git(self.inner, f"show -s {sha}"),
                re.MULTILINE,
            )
            if not m:
                return None
            base = m.group(1)
        found = self.collector._commit_id_from_hash(self.inner, base)
        if not found:
            return None
        point = self._points[sha] = _Point(base, int(found))
        return point

    def _on_main(self, sha: str) -> bool:
        res = self.collector._run(
            f"git merge-base --is-ancestor {sha} origin/main",
            cwd=self.inner.require_src_dir(),
            capture=True,
            caffeinate=False,
        )
        return res.returncode == 0

    def _ensure_fetched(self, sha: str) -> None:
        """Have ``sha`` in the inner checkout, fetching it by hash if not.

        The routine fetch brings main; a branch head is not on it, and the
        remote serves any commit by hash. Failing that is the fetch's
        failure, retried next cycle, not a roll to walk past.
        """
        src = self.inner.require_src_dir()
        have = self.collector._run(
            f"git cat-file -e {sha}^{{commit}}", cwd=src, capture=True, caffeinate=False
        )
        if have.returncode == 0:
            return
        got = self.collector._run(
            f"git fetch origin {sha}", cwd=src, capture=True, caffeinate=False
        )
        if got.returncode != 0:
            raise FetchError(
                f"{self.inner.name}: {sha[:12]} is not in the checkout and "
                f"could not be fetched"
            )

    def _forward(self, old: _Point, new: _Point) -> bool:
        """A roll, not a revert of one: ``new`` is not behind ``old`` on main.

        Equal is a roll too -- two branch heads cut from the same main
        commit -- and expands to its base point alone.
        """
        return old.id <= new.id

    def _first_within(
        self,
        roll: str,
        old: _Point,
        new: _Point,
        *,
        min_id: int,
        preserve_first: bool = False,
    ) -> BuildJob | None:
        """The lowest inner commit in ``old..=new`` with id >= ``min_id``.

        ``old`` itself is always a candidate -- it is the roll's base point,
        whatever the inner engine's path_filter says -- and the commits
        above it are the ones the inner engine's own cadence would build.
        """
        fmt = self.collector._METADATA_FORMAT
        path_filter = self.inner.path_filter or ""
        base = self._git(self.inner, f'log -1 --pretty=format:"{fmt}" {old.sha}')
        rest = self._git(
            self.inner,
            f'log --reverse --pretty=format:"{fmt}" {old.sha}..{new.sha}'
            f" -- {path_filter}",
        )
        for raw in (base + rest).split("--END-COMMIT--"):
            if not raw.strip():
                continue
            commit = self.collector._parse_commit_metadata(self.inner, raw)
            if (
                commit
                and commit["commit_id"] >= min_id
                and (
                    preserve_first
                    or commit["hash"] == old.sha
                    or self.collector.commit_is_relevant(
                        self.inner, commit["hash"], build_engine=self.outer
                    )
                )
            ):
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


class DerivedResolver:
    """An engine whose entries wrap another's: key = (B, inner key).

    Safari Technology Preview around jsc's WebKit build. There is no source
    and nothing to compile: each inner entry ``w`` that has what this engine
    needs becomes ``(B, w)`` -- the inner entry's blobs plus this engine's own
    run set, which is the installed app. ``B`` names the app. Its
    ``Info.plist`` has a build string and no position, so the builder
    assigns one per distinct string through ``number_for``, which the store
    keeps monotone; the string itself is the embedder hash.

    An inner entry qualifies when its blobs cover the inner engine's current
    run set: a jsc entry archived before the WebKit runtime joined the run
    set has a shell and nothing a browser can load, and would crash STP at
    startup. Migrated version 1 entries have no paths and never qualify.

    The series moves with two causes. A new inner entry above the frontier
    is the ordinary step. A new app (auto-update landed) is first published
    as ``(B', w)`` on the frontier's own ``w``, so the app change is a step
    of its own rather than folded into the next WebKit commit; the inner
    entries above ``w`` then follow under ``B'``. An app older than the
    frontier's (a reinstall) is an error until a newer one is installed:
    the series does not go backwards.
    """

    def __init__(
        self,
        bus: Bus,
        engine,
        inner,
        *,
        number_for: Callable[[HostApp], int],
        installed: Callable[[], HostApp] | None = None,
    ):
        self.bus = bus
        self.engine = engine
        self.inner = inner
        self.number_for = number_for
        self._installed = installed or self._read_installed

    # --- the app ---

    def app_path(self) -> Path:
        run_set = self.engine.require_run_set()
        if self.engine.src_dir is None:
            raise ValueError(
                f"engine {self.engine.name} has no src_dir on this machine; set it "
                f"to the directory holding {run_set[0]!r}"
            )
        return self.engine.src_dir / run_set[0]

    def _read_installed(self) -> HostApp:
        try:
            return read_app(self.app_path())
        except HostAppError as e:
            raise ValueError(f"{self.engine.name}: {e}") from None

    def installed(self) -> tuple[HostApp, int]:
        app = self._installed()
        return app, self.number_for(app)

    # --- Resolver ---

    def fetch(self) -> None:
        """The inner topic is on this bus and the app is on this disk."""

    def eligible(self, entry: Entry) -> bool:
        have = {b.path for b in entry.blobs}
        return all(p in have for p in self.inner.run_set)

    def _candidates(self) -> list[Entry]:
        return [e for e in self.bus.entries(self.inner.name) if self.eligible(e)]

    def next_after(self, frontier: CommitKey) -> BuildJob | None:
        app, number = self.installed()
        prev_number, w = frontier
        if prev_number and number < prev_number:
            raise ValueError(
                f"{self.engine.name}: the installed {app.title} is number {number}, "
                f"below the series' frontier {frontier}; the series does not go "
                f"backwards, so nothing is published until a newer one is installed"
            )
        if prev_number and number > prev_number:
            # The app-only step: same inner commit, new app.
            on_frontier = self.bus.read_entry(self.inner.name, CommitKey(0, w))
            if on_frontier is not None and self.eligible(on_frontier):
                return self._job(number, app, on_frontier)
            # Retention took it; the step lands on the next entry instead.
        for entry in self._candidates():
            if entry.key.commit_id > w:
                return self._job(number, app, entry)
        return None

    def for_key(self, key: CommitKey) -> BuildJob | None:
        app, number = self.installed()
        if key.embedder_id != number:
            # Another app's entry cannot be rebuilt: the bytes are gone from
            # the disk, and this engine has no history to reach back into.
            return None
        entry = self.bus.read_entry(self.inner.name, CommitKey(0, key.commit_id))
        if entry is None or not self.eligible(entry):
            return None
        return self._job(number, app, entry)

    def _job(self, number: int, app: HostApp, inner: Entry) -> BuildJob:
        return BuildJob(
            key=CommitKey(number, inner.commit_id),
            commit={
                "hash": inner.hash,
                "commit_id": inner.commit_id,
                "date": inner.date,
                "timestamp": inner.timestamp,
                "title": inner.title,
            },
            checkout_hash="",
            embedder={"hash": app.version, "commit_id": number, "title": app.title},
            inherits=inner,
        )
