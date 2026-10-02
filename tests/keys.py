# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Spell scalar ids as the CommitKeys the code now returns.

Tests are written against v8/jsc, whose keys are all embedder 0; ``K1(n)`` is
that key and ``K(a, b, ...)`` a list of them, so an expectation still reads as
the ids it is about.
"""

from __future__ import annotations

from slipstream.models import CommitKey


def K1(commit_id: int, embedder_id: int = 0) -> CommitKey:
    return CommitKey(embedder_id, commit_id)


def K(*commit_ids: int) -> list[CommitKey]:
    return [CommitKey(0, c) for c in commit_ids]
