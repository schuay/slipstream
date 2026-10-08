# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The delta layer on its own: streams in, plan and patches out, and the
exact bytes back. No bus, no files; block sizes are tiny so every shape of
stream fits in a test."""

from __future__ import annotations

import hashlib
import io
import random
import tarfile

import pytest

from slipstream.delta import (
    Block,
    BlockReader,
    DeltaError,
    DeltaMismatch,
    DeltaPlan,
    plan,
    reconstruct,
)

BLOCK = 1024


def _data(n: int, seed: int = 1) -> bytes:
    """Compressible-but-not-constant bytes, like a binary: random words
    repeated, so bsdiff has something to find."""
    rng = random.Random(seed)
    words = [bytes(rng.randrange(256) for _ in range(16)) for _ in range(64)]
    out = bytearray()
    while len(out) < n:
        out += rng.choice(words)
    return bytes(out[:n])


class _Store:
    def __init__(self):
        self.patches: dict[str, bytes] = {}
        self.stores = 0

    def store(self, sha: str, data: bytes) -> None:
        self.stores += 1
        self.patches[sha] = data

    def load(self, sha: str) -> bytes:
        return self.patches[sha]


def _plan(base: bytes, target: bytes, store: _Store | None = None, **kw):
    store = store or _Store()
    p = plan(
        io.BytesIO(base),
        io.BytesIO(target),
        store.store,
        block_bytes=kw.pop("block_bytes", BLOCK),
        **kw,
    )
    return p, store


def _rebuild(base: bytes, p: DeltaPlan, store: _Store) -> bytes:
    return b"".join(reconstruct(io.BytesIO(base), p, store.load))


class TestRoundTrip:
    def test_identical_streams_need_no_patches(self):
        base = _data(10 * BLOCK)
        p, store = _plan(base, base)
        assert all(b.patch is None for b in p.blocks)
        assert store.stores == 0
        assert p.tar_bytes == len(base)
        assert p.tar_sha256 == hashlib.sha256(base).hexdigest()
        assert _rebuild(base, p, store) == base

    def test_only_the_changed_block_gets_a_patch(self):
        base = _data(10 * BLOCK)
        target = bytearray(base)
        target[3 * BLOCK + 100 : 3 * BLOCK + 110] = b"XXXXXXXXXX"
        target = bytes(target)
        p, store = _plan(base, target)
        assert [b.patch is not None for b in p.blocks] == [i == 3 for i in range(10)]
        assert store.stores == 1
        assert p.patch_bytes < BLOCK, "a ten-byte edit is not a whole block"
        assert _rebuild(base, p, store) == target

    def test_a_longer_target_diffs_its_tail_against_nothing(self):
        base = _data(4 * BLOCK)
        target = base + _data(3 * BLOCK + 17, seed=2)
        p, store = _plan(base, target)
        assert len(p.blocks) == 8
        assert p.blocks[-1].bytes == 17
        assert all(b.patch is not None for b in p.blocks[4:])
        assert _rebuild(base, p, store) == target

    def test_a_shorter_target_ends_where_it_ends(self):
        base = _data(6 * BLOCK)
        target = base[: 2 * BLOCK + 5]
        p, store = _plan(base, target)
        assert [b.bytes for b in p.blocks] == [BLOCK, BLOCK, 5]
        assert _rebuild(base, p, store) == target

    def test_an_insertion_shifts_everything_after_it(self):
        """The case tar produces when a file grows: every later block
        differs from the base's, yet bsdiff finds the shifted content."""
        base = _data(8 * BLOCK)
        target = base[: BLOCK // 2] + b"inserted" + base[BLOCK // 2 :]
        p, store = _plan(base, target)
        assert all(b.patch is not None for b in p.blocks)
        assert p.patch_bytes < len(target) // 4, "shifted blocks are cheap"
        assert _rebuild(base, p, store) == target

    def test_an_empty_target_is_an_empty_plan(self):
        p, store = _plan(_data(BLOCK), b"")
        assert p.blocks == () and p.tar_bytes == 0
        assert _rebuild(_data(BLOCK), p, store) == b""

    def test_unrelated_streams_still_round_trip(self):
        base = _data(3 * BLOCK, seed=1)
        target = _data(3 * BLOCK, seed=99)
        p, store = _plan(base, target)
        assert _rebuild(base, p, store) == target


class TestPatchSharing:
    def test_identical_edits_to_identical_blocks_share_one_patch(self):
        block = _data(BLOCK)
        base = block * 4
        edited = bytearray(block)
        edited[10:14] = b"abcd"
        target = bytes(edited) * 2 + block * 2
        p, store = _plan(base, target)
        assert p.blocks[0].patch == p.blocks[1].patch
        assert store.stores == 1
        assert len(p.patches()) == 1
        assert _rebuild(base, p, store) == target

    def test_store_is_called_once_per_distinct_patch_before_return(self):
        base = _data(5 * BLOCK)
        target = _data(5 * BLOCK, seed=7)
        p, store = _plan(base, target)
        assert set(store.patches) == set(p.patches())
        assert all(len(store.patches[s]) == n for s, n in p.patches().items())


class TestBudget:
    def test_over_budget_returns_none(self):
        base = _data(4 * BLOCK)
        target = _data(4 * BLOCK, seed=5)
        p, store = _plan(base, target, max_patch_bytes=10)
        assert p is None
        # What was stored before the abort is the caller's to reclaim.
        assert store.stores >= 1

    def test_within_budget_returns_the_plan(self):
        base = _data(4 * BLOCK)
        target = bytearray(base)
        target[5] ^= 0xFF
        p, _ = _plan(base, bytes(target), max_patch_bytes=BLOCK)
        assert p is not None


class TestWorkers:
    def test_a_pool_produces_the_same_plan(self):
        base = _data(12 * BLOCK)
        target = bytearray(base)
        for i in (1, 4, 5, 9, 11):
            target[i * BLOCK + 7 : i * BLOCK + 9] = b"zz"
        target = bytes(target)
        serial, s1 = _plan(base, target, workers=1)
        pooled, s2 = _plan(base, target, workers=3)
        assert serial == pooled
        assert s1.patches == s2.patches
        assert _rebuild(base, pooled, s2) == target

    def test_a_pool_honours_the_budget(self):
        base = _data(12 * BLOCK)
        target = _data(12 * BLOCK, seed=3)
        p, _ = _plan(base, target, workers=3, max_patch_bytes=10)
        assert p is None


class TestReconstructVerifies:
    def _changed(self):
        base = _data(6 * BLOCK)
        target = bytearray(base)
        target[2 * BLOCK + 1 : 2 * BLOCK + 4] = b"abc"
        target = bytes(target)
        p, store = _plan(base, target)
        return base, target, p, store

    def test_a_tampered_patch_is_refused(self):
        base, _, p, store = self._changed()
        (sha,) = store.patches
        data = bytearray(store.patches[sha])
        data[-1] ^= 0xFF
        store.patches[sha] = bytes(data)
        with pytest.raises(DeltaMismatch, match="corrupt"):
            _rebuild(base, p, store)

    def test_a_wrong_base_is_refused_before_any_byte_is_yielded(self):
        base, _, p, store = self._changed()
        wrong = _data(6 * BLOCK, seed=42)
        gen = reconstruct(io.BytesIO(wrong), p, store.load)
        with pytest.raises(DeltaMismatch, match="block 0"):
            next(gen)

    def test_a_truncated_base_is_refused(self):
        base, _, p, store = self._changed()
        with pytest.raises(DeltaMismatch, match="block 5"):
            _rebuild(base[: 5 * BLOCK], p, store)

    def test_a_plan_whose_total_lies_is_refused_at_the_end(self):
        base, target, p, store = self._changed()
        lying = DeltaPlan(p.block_bytes, "0" * 64, p.tar_bytes, p.blocks)
        out = []
        with pytest.raises(DeltaMismatch, match="stream"):
            for block in reconstruct(io.BytesIO(base), lying, store.load):
                out.append(block)
        # Every block passed its own check; only the stream total failed.
        assert b"".join(out) == target


class TestJson:
    def test_round_trips(self):
        base = _data(3 * BLOCK)
        target = _data(3 * BLOCK, seed=2)
        p, _ = _plan(base, target)
        again = DeltaPlan.from_json(p.to_json())
        assert again == p
        assert "patch" not in Block("a", 1).to_json()

    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda d: d.pop("blocks"), "missing"),
            (lambda d: d.update(block_bytes=0), "malformed"),
            (lambda d: d.update(tar_bytes=1), "do not add up"),
            (lambda d: d["blocks"].append({"sha256": "x"}), "missing"),
            (lambda d: d["blocks"][0].update(extra=1), "unknown keys"),
            (lambda d: d["blocks"][0].update(patch=5), "malformed"),
        ],
    )
    def test_malformed_is_refused(self, mutate, match):
        base = _data(2 * BLOCK)
        p, _ = _plan(base, base)
        d = p.to_json()
        mutate(d)
        with pytest.raises(DeltaError, match=match):
            DeltaPlan.from_json(d)


class TestBlockReaderFeedsTar:
    def _tar(self, files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def test_a_reconstructed_tar_extracts_the_target_files(self, tmp_path):
        base_files = {"app/bin": _data(3 * BLOCK), "app/res": _data(BLOCK, seed=2)}
        target_files = dict(base_files)
        edited = bytearray(base_files["app/bin"])
        edited[BLOCK + 3 : BLOCK + 6] = b"new"
        target_files["app/bin"] = bytes(edited)
        base, target = self._tar(base_files), self._tar(target_files)
        p, store = _plan(base, target)
        reader = BlockReader(reconstruct(io.BytesIO(base), p, store.load))
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            tf.extractall(tmp_path, filter="tar")
        assert (tmp_path / "app/bin").read_bytes() == target_files["app/bin"]
        assert (tmp_path / "app/res").read_bytes() == target_files["app/res"]

    def test_short_reads_are_served_from_the_current_block(self):
        blocks = iter([b"abc", b"", b"defg"])
        r = BlockReader(blocks)
        assert r.read(2) == b"ab"
        assert r.read(2) == b"c"
        assert r.read(10) == b"defg"
        assert r.read(1) == b""
