# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Local exclusion and crash-durable publication for delivery state."""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
import time
from pathlib import Path


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_mkdir(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        fsync_dir(directory.parent)


def atomic_write(path: Path, data: str) -> None:
    durable_mkdir(path.parent)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def durable_unlink(path: Path) -> None:
    path.unlink(missing_ok=True)
    fsync_dir(path.parent)


def identity_digest(identity: str) -> str:
    return hashlib.sha256(identity.encode()).hexdigest()


def target_lock_path(identity: str) -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    name = (
        f"spanner-{identity_digest(identity.removeprefix('spanner:'))}"
        if identity.startswith("spanner:")
        else identity_digest(identity)
    )
    return cache / "slipstream" / "locks" / f"{name}.lock"


class FileLock:
    """A bounded, cancellable flock; close releases even after a failed wait."""

    def __init__(
        self,
        path: Path,
        *,
        timeout=30.0,
        should_stop=lambda: False,
        log=lambda msg: None,
    ):
        self.path = path
        self.timeout = timeout
        self.should_stop = should_stop
        self.log = log
        self._file = None

    def __enter__(self):
        durable_mkdir(self.path.parent)
        self._file = self.path.open("a")
        start = time.monotonic()
        reported = False
        try:
            while True:
                if self.should_stop():
                    raise InterruptedError(f"cancelled waiting for {self.path}")
                try:
                    fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.log(
                        f"lock acquired {self.path.name} elapsed={time.monotonic() - start:.3f}s"
                    )
                    return self
                except BlockingIOError:
                    if not reported:
                        self.log(f"lock wait {self.path.name}")
                        reported = True
                    if time.monotonic() - start >= self.timeout:
                        raise TimeoutError(
                            f"lock wait exceeded {self.timeout}s: {self.path}"
                        )
                    time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __exit__(self, *args):
        self.close()
