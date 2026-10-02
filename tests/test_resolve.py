# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slipstream.models import CommitKey
from slipstream.resolve import BuildJob, IdentityResolver


def _commit(cid):
    return {
        "hash": f"hash{cid}",
        "commit_id": cid,
        "date": "2026-09-06",
        "timestamp": 1757116800 + cid,
        "title": f"commit {cid}",
    }


class FakeCollector:
    """The three collector calls the identity resolver is built on."""

    def __init__(self, history):
        self.history = history
        self.fetched = []

    def head_commit_id(self, name, fetch=True):
        self.fetched.append(name)
        return max(self.history)

    def next_commit_after(self, engine, commit_id):
        later = [c for c in self.history if c > commit_id]
        return _commit(later[0]) if later else None

    def commit_metadata_for_id(self, engine, commit_id):
        return _commit(commit_id) if commit_id in self.history else None


@pytest.fixture
def resolver():
    engine = SimpleNamespace(name="v8")
    return IdentityResolver(FakeCollector([100, 101, 103]), engine)


class TestIdentityResolver:
    def test_the_next_job_is_the_next_commit_under_a_scalar_key(self, resolver):
        job = resolver.next_after(CommitKey(0, 100))
        assert job == BuildJob(
            key=CommitKey(0, 101), commit=_commit(101), checkout_hash="hash101"
        )
        assert job.pins == {} and job.embedder == {}
        assert job.hash == "hash101"

    def test_a_gap_in_history_is_skipped_not_invented(self, resolver):
        assert resolver.next_after(CommitKey(0, 101)).key == CommitKey(0, 103)

    def test_up_to_date_is_none(self, resolver):
        assert resolver.next_after(CommitKey(0, 103)) is None

    def test_for_key_names_exactly_that_commit_or_nothing(self, resolver):
        assert resolver.for_key(CommitKey(0, 101)).checkout_hash == "hash101"
        assert resolver.for_key(CommitKey(0, 102)) is None

    def test_fetch_goes_to_the_engines_checkout(self, resolver):
        resolver.fetch()
        assert resolver.collector.fetched == ["v8"]

    @pytest.mark.parametrize("method", ["next_after", "for_key"])
    def test_a_key_with_an_embedder_is_a_misconfiguration(self, resolver, method):
        """[build] from or --retry wrote a two-coordinate key for an engine
        whose series has one; say so rather than silently dropping the
        embedder and building the wrong thing."""
        with pytest.raises(ValueError, match="own checkout"):
            getattr(resolver, method)(CommitKey(7, 100))


class TestBuildJob:
    def test_is_immutable(self):
        job = BuildJob(key=CommitKey(0, 1), commit=_commit(1), checkout_hash="h")
        with pytest.raises(AttributeError):
            job.key = CommitKey(0, 2)  # type: ignore[misc]
