# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Local exclusion and crash-durable publication for delivery state."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
import sys
from pathlib import Path


def flush_file(fd: int) -> None:
    """Flush file data to stable storage, including the macOS device cache."""
    os.fsync(fd)
    if sys.platform == "darwin":
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)


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
            flush_file(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
        # fsync of a directory persists the rename to the filesystem. Flush
        # the device cache after that metadata operation on macOS as well.
        if sys.platform == "darwin":
            with path.open("rb") as persisted:
                flush_file(persisted.fileno())
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


# Holds longer than this are reported on release, before any waiter times out.
LONG_HOLD_SECONDS = 5.0

_holder_role = Path(sys.argv[0]).name
_long_hold_log = None


def report_lock_holds(role: str, log) -> None:
    """Name this process in lock holder records and log its long holds."""
    global _holder_role, _long_hold_log
    _holder_role, _long_hold_log = role, log


def _call_site(depth=3) -> str:
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename == __file__:
        frame = frame.f_back
    sites = []
    while frame is not None and len(sites) < depth:
        code = frame.f_code
        sites.append(f"{Path(code.co_filename).name}:{frame.f_lineno} {code.co_name}")
        frame = frame.f_back
    return " < ".join(sites)


class FileLock:
    """A bounded, cancellable flock; close releases even after a failed wait.

    The holder writes who it is into the lock file, so a waiter that times
    out can name it. Nothing else reads the contents.
    """

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
        self._site = None
        self._acquired = None

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
                    self._record_holder()
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
                            + self._describe_holder()
                        )
                    time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def _record_holder(self):
        self._site = _call_site()
        self._acquired = time.monotonic()
        record = {
            "pid": os.getpid(),
            "role": _holder_role,
            "site": self._site,
            "since": time.time(),
        }
        # Append mode writes at the end, which is offset 0 once truncated.
        self._file.truncate(0)
        self._file.write(json.dumps(record))
        self._file.flush()

    def _describe_holder(self) -> str:
        try:
            record = json.loads(self.path.read_text())
            held = time.time() - float(record["since"])
            return (
                f"; held {held:.1f}s by {record['role']} pid {record['pid']}"
                f" at {record['site']}"
            )
        except (OSError, ValueError, KeyError, TypeError):
            # No record yet, or a read racing the holder's write.
            return ""

    def close(self):
        if self._file is None:
            return
        held = None
        try:
            if self._acquired is not None:
                held = time.monotonic() - self._acquired
                self._acquired = None
                # Truncate before unlocking so the next waiter never reads a
                # released holder's record.
                self._file.truncate(0)
        finally:
            self._file.close()
            self._file = None
        if held is not None and held >= LONG_HOLD_SECONDS and _long_hold_log:
            _long_hold_log(f"held {self.path.name} for {held:.1f}s at {self._site}")

    def __exit__(self, *args):
        self.close()
