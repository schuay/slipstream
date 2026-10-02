# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from slipstream.models import CommitKey


class TestSpellings:
    def test_scalar_round_trips_as_the_bare_id(self):
        """Everything v8 and jsc ever wrote is a scalar; it must read back as
        the same key and write out as the same text."""
        k = CommitKey.of(109680)
        assert k == CommitKey(0, 109680)
        assert str(k) == "109680"
        assert k.to_json() == 109680
        assert CommitKey.parse("109680") == k
        assert CommitKey.from_json(109680) == k

    def test_composite_round_trips_through_text_and_json(self):
        k = CommitKey(1534000, 109680)
        assert str(k) == "1534000-109680"
        assert CommitKey.parse(str(k)) == k
        assert k.to_json() == [1534000, 109680]
        assert CommitKey.from_json(k.to_json()) == k
        assert CommitKey.of((1534000, 109680)) == k

    def test_from_commit_reads_the_optional_embedder(self):
        assert CommitKey.from_commit({"commit_id": 5}) == CommitKey(0, 5)
        assert CommitKey.from_commit({"commit_id": 5, "embedder_id": None}) == (0, 5)
        assert CommitKey.from_commit({"commit_id": 5, "embedder_id": 9}) == (9, 5)

    @pytest.mark.parametrize("bad", ["", "abc", "1-2-3", "-5", "1-"])
    def test_malformed_text_is_refused(self, bad):
        with pytest.raises(ValueError):
            CommitKey.parse(bad)

    @pytest.mark.parametrize("bad", [True, 1.5, None, (1,), {"a": 1}])
    def test_other_types_are_a_bug_not_a_coercion(self, bad):
        with pytest.raises(TypeError):
            CommitKey.of(bad)

    def test_from_json_keeps_none(self):
        assert CommitKey.from_json(None) is None


class TestOrder:
    def test_lexicographic(self):
        """Embedder first: a later roll sorts above every key of an earlier
        one, whatever the engine ids, so the series is one monotone line."""
        assert CommitKey(0, 999) < CommitKey(1, 0)
        assert CommitKey(1, 5) < CommitKey(1, 6)
        assert max(CommitKey(2, 1), CommitKey(1, 100)) == CommitKey(2, 1)

    def test_before_steps_the_engine_coordinate(self):
        assert CommitKey(3, 10).before() == CommitKey(3, 9)
        assert CommitKey(0, 10).before() == CommitKey(0, 9)

    def test_an_int_is_not_a_key(self):
        """No implicit equality with ints: a test or a caller that mixes the
        two should fail loudly, not pass by accident on embedder 0."""
        assert CommitKey(0, 5) != 5
        assert CommitKey(0, 5) == (0, 5)
