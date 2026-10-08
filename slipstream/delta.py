# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Block-wise binary deltas between two tar streams.

A bus blob is a tar stream of one run set entry. Between two adjacent commits
most of that stream is identical and the rest differs by scattered small
edits inside one large executable, which whole-file content addressing
cannot see. This module cuts both streams into fixed blocks and, per block,
either notes that the target equals the base or produces one bsdiff patch
from the base block to the target block. The result is a :class:`DeltaPlan`:
what the manifest records, and what :func:`reconstruct` turns back into the
exact target stream from the base and the patches.

Nothing here knows about the bus. The planner takes two readable streams and
a callback to store each distinct patch; the reconstructor takes the base
stream, the plan and a callback to load a patch, and yields verified blocks.
Both run in bounded memory: a block or two per worker, never a whole tar.

Patches are BSDIFF40 as bsdiff4 produces them (bzip2 inside), named by the
sha256 of the patch bytes, so identical patches -- the same block changed
the same way in two targets of one base -- are stored once.
"""

from __future__ import annotations

import hashlib
import io
import multiprocessing
import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import BinaryIO

import bsdiff4

# What the builder cuts a stream into. Recorded per plan, so it can change
# without breaking a reader; 8 MiB is what the storage study measured and
# bounds bsdiff's working set to ~100 MB per worker.
BLOCK_BYTES = 8 << 20


class DeltaError(RuntimeError):
    """The delta layer cannot do what was asked: a block failed its own
    verification while planning, or a plan is malformed."""


class DeltaMismatch(DeltaError):
    """Reconstruction produced bytes the plan does not describe: a tampered
    or wrong patch, a wrong base, or a truncated stream."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Block:
    """One block of the target stream. ``patch`` is None when the block is
    the base's block at the same index, byte for byte."""

    sha256: str
    bytes: int
    patch: str | None = None
    patch_bytes: int = 0

    def to_json(self) -> dict:
        d = {"sha256": self.sha256, "bytes": self.bytes}
        if self.patch is not None:
            d["patch"] = self.patch
            d["patch_bytes"] = self.patch_bytes
        return d

    @classmethod
    def from_json(cls, data: dict, where: str) -> Block:
        if not isinstance(data, dict):
            raise DeltaError(f"{where}: block is not an object: {data!r}")
        try:
            sha, n = data["sha256"], data["bytes"]
        except KeyError as e:
            raise DeltaError(f"{where}: block is missing {e}") from None
        if not isinstance(sha, str) or not isinstance(n, int) or isinstance(n, bool):
            raise DeltaError(f"{where}: malformed block {data!r}")
        patch = data.get("patch")
        patch_bytes = data.get("patch_bytes", 0)
        if patch is not None and (
            not isinstance(patch, str)
            or not isinstance(patch_bytes, int)
            or isinstance(patch_bytes, bool)
        ):
            raise DeltaError(f"{where}: malformed block {data!r}")
        unknown = set(data) - {"sha256", "bytes", "patch", "patch_bytes"}
        if unknown:
            raise DeltaError(f"{where}: block has unknown keys {sorted(unknown)}")
        return cls(sha, n, patch, patch_bytes if patch is not None else 0)


@dataclass(frozen=True)
class DeltaPlan:
    """How to rebuild one tar stream from a base stream and patches.

    ``tar_sha256`` and ``tar_bytes`` describe the whole target stream and are
    the last thing reconstruction checks; every block is also checked on its
    own as it is produced, so a bad patch is caught before its bytes go
    anywhere.
    """

    block_bytes: int
    tar_sha256: str
    tar_bytes: int
    blocks: tuple[Block, ...]

    def patches(self) -> dict[str, int]:
        """Distinct patch sha256 -> patch bytes, in first-use order."""
        out: dict[str, int] = {}
        for b in self.blocks:
            if b.patch is not None:
                out.setdefault(b.patch, b.patch_bytes)
        return out

    @property
    def patch_bytes(self) -> int:
        return sum(self.patches().values())

    def to_json(self) -> dict:
        return {
            "block_bytes": self.block_bytes,
            "tar_sha256": self.tar_sha256,
            "tar_bytes": self.tar_bytes,
            "blocks": [b.to_json() for b in self.blocks],
        }

    @classmethod
    def from_json(cls, data: dict, where: str = "delta") -> DeltaPlan:
        if not isinstance(data, dict):
            raise DeltaError(f"{where}: not an object")
        missing = {"block_bytes", "tar_sha256", "tar_bytes", "blocks"} - set(data)
        if missing:
            raise DeltaError(f"{where}: missing {sorted(missing)}")
        block_bytes, sha, n, blocks = (
            data["block_bytes"],
            data["tar_sha256"],
            data["tar_bytes"],
            data["blocks"],
        )
        if (
            not isinstance(block_bytes, int)
            or isinstance(block_bytes, bool)
            or block_bytes < 1
            or not isinstance(sha, str)
            or not isinstance(n, int)
            or isinstance(n, bool)
            or not isinstance(blocks, list)
        ):
            raise DeltaError(f"{where}: malformed")
        parsed = tuple(
            Block.from_json(b, f"{where} block {i}") for i, b in enumerate(blocks)
        )
        if sum(b.bytes for b in parsed) != n:
            raise DeltaError(f"{where}: block sizes do not add up to tar_bytes")
        return cls(block_bytes, sha, n, parsed)


def _read_block(stream: BinaryIO, n: int) -> bytes:
    """Exactly ``n`` bytes unless the stream ends first. A pipe can return
    short reads long before EOF, and a short block in the middle would put
    every later block out of step with the base."""
    parts = []
    want = n
    while want:
        chunk = stream.read(want)
        if not chunk:
            break
        parts.append(chunk)
        want -= len(chunk)
    return b"".join(parts)


def _diff_block(base: bytes, target: bytes) -> tuple[bytes, str]:
    """One worker's job: the patch, checked by applying it.

    bsdiff4 is C code fed a block of a tar; applying the patch straight
    back is cheap next to producing it, and it is the one check that
    catches a bad patch where it is produced rather than on a bencher an
    hour later.
    """
    patch = bsdiff4.diff(base, target)
    if bsdiff4.patch(base, patch) != target:
        raise DeltaError("patch does not reproduce its block")
    return patch, _sha(patch)


def plan(
    base: BinaryIO,
    target: BinaryIO,
    store_patch: Callable[[str, bytes], None],
    *,
    block_bytes: int = BLOCK_BYTES,
    workers: int = 1,
    max_patch_bytes: int | None = None,
) -> DeltaPlan | None:
    """Plan ``target`` against ``base``, storing each distinct patch.

    Both streams are read once, sequentially, in lockstep. A target block
    equal to the base's block at that index is recorded by hash alone;
    every other block is diffed, on ``workers`` processes when more than
    one. Blocks past the end of the base are diffed against nothing, which
    bsdiff handles as plain compression; the plan ends with the target.

    ``store_patch(sha256, data)`` is called once per distinct patch, before
    the plan is returned, so a caller that renames patches into a store has
    every one of them by the time it writes the manifest. Returns None when
    the distinct patch bytes exceed ``max_patch_bytes``: the caller stores
    the target whole instead, and this target is the next base. Patches
    already stored by then are the caller's to reclaim; on the bus the
    reference sweep does it.
    """
    hasher = hashlib.sha256()
    total = 0
    blocks: list[Block | None] = []
    stored: dict[str, int] = {}
    patch_total = 0

    def record_patch(index: int, target_sha: str, size: int, patch: bytes, sha: str):
        nonlocal patch_total
        if sha not in stored:
            store_patch(sha, patch)
            stored[sha] = len(patch)
            patch_total += len(patch)
        blocks[index] = Block(target_sha, size, sha, len(patch))

    def over_budget() -> bool:
        return max_patch_bytes is not None and patch_total > max_patch_bytes

    pool = (
        ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        if workers > 1
        else None
    )
    # (index, target sha, size, future). Bounded so a fast reader cannot
    # queue a whole tar's worth of blocks in memory ahead of the workers.
    pending: deque[tuple[int, str, int, Future]] = deque()
    try:
        while True:
            if pool is not None:
                while len(pending) >= 2 * workers:
                    i, tsha, n, fut = pending.popleft()
                    record_patch(i, tsha, n, *fut.result())
                    if over_budget():
                        return None
            target_block = _read_block(target, block_bytes)
            if not target_block:
                break
            base_block = _read_block(base, block_bytes)
            index = len(blocks)
            blocks.append(None)
            hasher.update(target_block)
            total += len(target_block)
            target_sha = _sha(target_block)
            if base_block == target_block:
                blocks[index] = Block(target_sha, len(target_block))
                continue
            if pool is None:
                record_patch(
                    index,
                    target_sha,
                    len(target_block),
                    *_diff_block(base_block, target_block),
                )
                if over_budget():
                    return None
            else:
                fut = pool.submit(_diff_block, base_block, target_block)
                pending.append((index, target_sha, len(target_block), fut))
        while pending:
            i, tsha, n, fut = pending.popleft()
            record_patch(i, tsha, n, *fut.result())
            if over_budget():
                return None
    finally:
        if pool is not None:
            # Nothing left to wait for on the normal path; on an early return
            # the unfinished diffs are wasted work, not something to keep.
            pool.shutdown(wait=True, cancel_futures=True)

    assert all(b is not None for b in blocks)
    return DeltaPlan(
        block_bytes=block_bytes,
        tar_sha256=hasher.hexdigest(),
        tar_bytes=total,
        blocks=tuple(blocks),  # type: ignore[arg-type]
    )


def reconstruct(
    base: BinaryIO,
    delta: DeltaPlan,
    load_patch: Callable[[str], bytes],
) -> Iterator[bytes]:
    """Yield the target stream, block by block, from the base and patches.

    Every block is checked against the plan before it is yielded, and the
    whole stream against ``tar_sha256``/``tar_bytes`` at the end, so a
    consumer feeding this into tar never writes a byte the plan did not
    describe. Raises :class:`DeltaMismatch` on the first discrepancy.
    """
    hasher = hashlib.sha256()
    total = 0
    for i, block in enumerate(delta.blocks):
        base_block = _read_block(base, delta.block_bytes)
        if block.patch is None:
            out = base_block
        else:
            patch = load_patch(block.patch)
            if _sha(patch) != block.patch:
                raise DeltaMismatch(f"block {i}: patch {block.patch[:12]} is corrupt")
            out = bsdiff4.patch(base_block, patch)
        if len(out) != block.bytes or _sha(out) != block.sha256:
            raise DeltaMismatch(
                f"block {i}: reconstructed {len(out)} bytes do not match the plan"
                + ("" if block.patch is None else "; wrong base or patch")
            )
        hasher.update(out)
        total += len(out)
        yield out
    if total != delta.tar_bytes or hasher.hexdigest() != delta.tar_sha256:
        raise DeltaMismatch("reconstructed stream does not match the plan")


class BlockReader(io.RawIOBase):
    """A readable file over an iterator of byte blocks, for ``tarfile``.

    tarfile's streaming mode wants ``read(n)``; the reconstructor yields
    whole blocks. This hands out slices of the current block and pulls the
    next when it runs dry, so nothing larger than one block is held.
    """

    def __init__(self, blocks: Iterator[bytes]):
        self._blocks = blocks
        self._buf = memoryview(b"")
        self._pos = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while self._pos >= len(self._buf):
            try:
                nxt = next(self._blocks)
            except StopIteration:
                return 0
            self._buf = memoryview(nxt)
            self._pos = 0
        n = min(len(b), len(self._buf) - self._pos)
        b[:n] = self._buf[self._pos : self._pos + n]
        self._pos += n
        return n


def default_workers() -> int:
    return os.cpu_count() or 1
