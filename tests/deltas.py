# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Build real delta blobs for tests, the way the builder will.

The consumer's reconstruction has to be exercised against patches bsdiff
actually produced over tar streams tar actually wrote; a hand-made plan
would test the plumbing and nothing else.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from slipstream import delta
from slipstream.builder import package
from slipstream.bus import Blob, Bus, Delta, patch_name, tree_hash

# Small enough that a test archive spans several blocks: tar writes in
# 10 KiB records, so even one tiny file is three of these.
BLOCK_BYTES = 4096


def delta_blob(
    bus: Bus, base: Blob, src_dir: Path, entry: str, *, block_bytes=BLOCK_BYTES
) -> Blob:
    """Package ``src_dir/entry`` as a delta against ``base``'s archive."""
    assert base.delta is None, "the base of a delta is a full archive"
    blob_id = tree_hash(src_dir, entry)
    tmp = bus.tmp_blob(blob_id)
    package(src_dir, [entry], tmp, caffeinate=False)

    def store_patch(sha: str, data: bytes) -> None:
        p = bus.tmp_object(patch_name(sha))
        p.write_bytes(data)
        bus.store_object(p, patch_name(sha))

    with (
        subprocess.Popen(
            ["zstd", "-dc", str(bus.blob_path(base.id))], stdout=subprocess.PIPE
        ) as b,
        subprocess.Popen(["zstd", "-dc", str(tmp)], stdout=subprocess.PIPE) as t,
    ):
        plan = delta.plan(b.stdout, t.stdout, store_patch, block_bytes=block_bytes)
    tmp.unlink()
    assert plan is not None
    return Blob(entry, blob_id, delta=Delta(base.id, base.sha256, base.bytes, plan))
