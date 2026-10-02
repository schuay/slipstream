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
