# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""One lock per machine, protecting its CPU and thermal state.

A build next to a measurement contaminates it, so the builder and the bencher
take this lock around their expensive phases: the builder across checkout,
build and packaging, the bencher across all runs of one commit rather than per
run, since a build landing between run 1 and run 2 would contaminate exactly
the within-commit variance the analyzer reads as noise.

It is per machine, not per out_dir, and it is deliberately not used to
serialize schema changes: it is held for hours, and every store-opening command
would then have to wait out a bench before it could open the db.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

LOCKS_DIR = Path.home() / ".cache" / "slipstream" / "locks"
MACHINE_LOCK = LOCKS_DIR / "machine.lock"
PAUSE_FILE = LOCKS_DIR / "pause"

POLL_SECS = 2.0
# After releasing, a holder waits before it may acquire again, or it wins its
# own re-acquisition race and starves a peer that is polling. Not derived from
# watch's sleep, which is `interval - work` and therefore zero exactly when a
# cycle overran its interval -- the contended case.
#
# Two lengths, because the two callers pay differently. A daemon releases once
# per commit and has all day, so it yields generously. An ad-hoc command
# releases once per commit too, and a full minute each over a 200-commit range
# is hours; it yields for a couple of poll periods instead, which is all a
# waiting daemon needs to win the lock.
HANDOFF_SECS = 60.0
ADHOC_HANDOFF_SECS = 4.0
DEFAULT_PAUSE_TTL = 4 * 3600

# EX_TEMPFAIL: the work is fine, the machine is busy.
EXIT_BUSY = 75


class LockBusy(RuntimeError):
    """Another process holds the machine lock."""


class Holder:
    """Who is holding the lock, as recorded in the lock file."""

    def __init__(self, pid: int, role: str, since: float):
        self.pid = pid
        self.role = role
        self.since = since

    @property
    def alive(self) -> bool:
        # The file outlives a killed holder, so the pid is checked before it is
        # reported as the reason for a refusal.
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def __str__(self) -> str:
        held = int(time.time() - self.since)
        alive = "" if self.alive else " (no longer running)"
        return f"{self.role} pid {self.pid}, holding for {held // 60}m{alive}"


def _read_holder(path: Path) -> Holder | None:
    try:
        data = json.loads(path.read_text() or "{}")
        return Holder(int(data["pid"]), str(data["role"]), float(data["since"]))
    except (OSError, ValueError, KeyError, TypeError):
        # Also covers a read racing the holder's own write; the only cost is a
        # vaguer report line.
        return None


def paused_until(path: Path | None = None) -> float | None:
    """When the operator's pause expires, or None if not paused.

    The pause carries a TTL so a forgotten one cannot silently stop all
    measurement.
    """
    path = path or PAUSE_FILE
    try:
        until = float(json.loads(path.read_text())["until"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if until <= time.time():
        return None
    return until


def pause(ttl: float = DEFAULT_PAUSE_TTL, path: Path | None = None) -> float:
    """Ask the daemons not to take the lock again. Returns the expiry time."""
    path = path or PAUSE_FILE
    until = time.time() + ttl
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps({"until": until, "at": time.time()}))
    tmp.rename(path)
    return until


def resume(path: Path | None = None) -> None:
    (path or PAUSE_FILE).unlink(missing_ok=True)


class MachineLock:
    """Exclusive, advisory, released by the kernel when the process dies.

    flock is not inherited by subprocesses, so a build's children do not hold
    it open past the parent.
    """

    def __init__(self, role: str, path: Path | None = None):
        self.role = role
        self.path = path or MACHINE_LOCK
        self._fd: int | None = None
        self._released_at: float | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def holder(self) -> Holder | None:
        """Who holds it, for reporting a refusal. May be stale if nobody does."""
        return _read_holder(self.path)

    def probe(self) -> Holder | None:
        """Who holds the lock right now, without taking it.

        try_acquire would answer this too, but it takes the exclusive lock and
        writes its own holder record, so a read-only report could fail an
        ad-hoc command that ran beside it and leave the file naming a holder
        that has already gone. A shared lock only tests whether an exclusive
        one is held, and writes nothing.
        """
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            return None
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _read_holder(self.path)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
        finally:
            os.close(fd)

    def try_acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Not open(path, "w"): that truncates at open, so a waiter would erase
        # the identity of the holder it is about to report.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except OSError:
            os.close(fd)
            raise
        # Before the holder record is written: from here the lock is held, and
        # if this is not set, release() is a no-op and the fd is never closed.
        # flock does not recurse across open file descriptions, so the same
        # process would then be refused by its own lock for the rest of its
        # life, with no holder in the file to say why.
        self._fd = fd
        try:
            os.ftruncate(fd, 0)
            os.write(
                fd,
                json.dumps(
                    {"pid": os.getpid(), "role": self.role, "since": time.time()}
                ).encode(),
            )
            os.fsync(fd)
        except OSError:
            # The record is diagnostic; the lock is the thing. A full disk must
            # not cost the caller the lock it just took.
            pass
        return True

    def acquire(
        self,
        should_stop: Callable[[], bool] | None = None,
        *,
        wait: bool,
        log: Callable[[str], None] | None = None,
    ) -> bool:
        """Take the lock. Returns False if we gave up.

        Daemons wait; ad-hoc commands try once and raise LockBusy so the
        operator is told who is holding it rather than blocking for hours
        behind a bench. Waiting polls rather than blocking in flock: a
        blocking flock is retried by CPython after a signal handler that does
        not raise, so a shutdown request would not be seen until the holder
        released.
        """
        should_stop = should_stop or (lambda: False)
        self._wait_for_handoff(
            should_stop, HANDOFF_SECS if wait else ADHOC_HANDOFF_SECS
        )
        if not wait:
            if self.try_acquire():
                return True
            holder = self.holder()
            raise LockBusy(
                f"another slipstream process holds the machine lock"
                f"{f': {holder}' if holder else ''}"
            )
        announced = False
        announced_pause = False
        while not should_stop():
            if self._paused(log if not announced_pause else None):
                announced_pause = True
                time.sleep(POLL_SECS)
                continue
            announced_pause = False
            if self.try_acquire():
                return True
            if not announced and log:
                holder = self.holder()
                log(f"waiting for the machine lock{f' ({holder})' if holder else ''}")
                announced = True
            time.sleep(POLL_SECS)
        return False

    def _paused(self, log: Callable[[str], None] | None) -> bool:
        until = paused_until()
        if until is None:
            return False
        if log:
            log(
                f"paused by operator until {time.strftime('%H:%M', time.localtime(until))}"
            )
        return True

    def _wait_for_handoff(
        self, should_stop: Callable[[], bool], seconds: float
    ) -> None:
        if self._released_at is None:
            return
        # monotonic, not wall clock: an NTP step during the handoff would
        # otherwise skip it or stretch it arbitrarily.
        deadline = self._released_at + seconds
        while not should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            # The clock can cross the deadline between the check and the
            # subtraction, and sleep raises on a negative.
            time.sleep(max(0.0, min(POLL_SECS, remaining)))

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None
        self._released_at = time.monotonic()

    def __enter__(self) -> MachineLock:
        return self

    def __exit__(self, *exc) -> None:
        self.release()
