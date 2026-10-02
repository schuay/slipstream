# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import NamedTuple

_KEY_RE = re.compile(r"^(?:(\d+)-)?(\d+)$")


class CommitKey(NamedTuple):
    """Where a build sits in an engine's series. Ordered lexicographically.

    ``embedder_id`` is 0 for an engine that is its own embedder (v8, jsc): the
    key is then the commit id alone, and every scalar id ever written -- in
    the db, in a bus filename, in a cursor or state file -- is already a valid
    key with nothing rewritten. For an engine built inside another's tree it
    is the embedder's position: chrome is keyed (chromium position of the roll
    CL, V8 position pinned under it), so the same V8 commit can be measured
    under two chromium revisions and the series still has one point per key.

    The order is lexicographic: along the series each step changes one
    coordinate, so a step is either a range of the engine's commits or a
    range of the embedder's, never a mixture.

    The scalar spelling is kept wherever ``embedder_id`` is 0 -- ``str``,
    ``to_json`` -- so files and state written before this type existed read
    back unchanged, and v8 and jsc keep writing byte-identical ones.
    """

    embedder_id: int
    commit_id: int

    @classmethod
    def of(cls, value) -> CommitKey:
        """Normalise the spellings a key arrives in.

        An int is a scalar key; a two-tuple or list is the pair; a str is the
        CLI form. A CommitKey passes through. Anything else is a bug at the
        call site, not a value to coerce.
        """
        if isinstance(value, CommitKey):
            return value
        if isinstance(value, bool):
            raise TypeError(f"not a commit key: {value!r}")
        if isinstance(value, int):
            return cls(0, value)
        if isinstance(value, str):
            return cls.parse(value)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return cls(int(value[0]), int(value[1]))
        raise TypeError(f"not a commit key: {value!r}")

    @classmethod
    def parse(cls, text: str) -> CommitKey:
        """The CLI and filename form: ``109680`` or ``1534000-109680``."""
        m = _KEY_RE.match(text.strip())
        if not m:
            raise ValueError(f"not a commit key: {text!r}")
        return cls(int(m.group(1) or 0), int(m.group(2)))

    @classmethod
    def from_commit(cls, commit) -> CommitKey:
        """From a commit dict or row: ``commit_id`` plus an optional ``embedder_id``."""
        try:
            embedder = commit["embedder_id"]
        except (KeyError, IndexError):
            embedder = 0
        return cls(int(embedder or 0), int(commit["commit_id"]))

    @classmethod
    def from_json(cls, value) -> CommitKey | None:
        """Inverse of ``to_json``; None stays None, for optional state fields."""
        if value is None:
            return None
        return cls.of(value)

    def to_json(self):
        """An int for a scalar key, else ``[embedder_id, commit_id]``.

        Not ``str(self)``: the state files are read by ``bus status`` over
        ssh and compared numerically, and an int that became a string would
        change their meaning for every reader that predates this type.
        """
        return self.commit_id if self.embedder_id == 0 else list(self)

    def before(self) -> CommitKey:
        """The key just below this one, for a cursor that must re-read it."""
        return CommitKey(self.embedder_id, self.commit_id - 1)

    def __str__(self) -> str:
        if self.embedder_id == 0:
            return str(self.commit_id)
        return f"{self.embedder_id}-{self.commit_id}"

    def __repr__(self) -> str:
        return f"CommitKey({self.embedder_id}, {self.commit_id})"
