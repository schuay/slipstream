# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""One lock per machine, protecting its CPU and thermal state.

A build next to a measurement contaminates it, so the builder and the bencher
take this lock around their expensive phases: the builder across checkout,
build and packaging, the bencher across all runs of one commit rather than per
run, since a build landing between run 1 and run 2 would contaminate exactly
the within-commit variance downstream reads as noise.

It is per machine, not per out_dir, and it is deliberately not used to
serialize schema changes: it is held for hours, and every store-opening command
would then have to wait out a bench before it could open the db.

Admission is first come, first served, and nothing arbitrates it. A single
flock has no queue: when the holder lets go, whoever asks next wins, and a
process that releases and immediately asks again is always next. The old
answer was a mandatory sleep after every release. The answer here is a chain
of flocks, one per waiter:

    locks/machine.lock          the lock itself, as before
    locks/queue/.seq            a counter
    locks/queue/<seq>-<pid>     one ticket per waiting or holding process

A ticket is a position. Its owner holds LOCK_EX on it from before anyone can
see it until its work is done. Everyone else only ever takes LOCK_SH on
tickets: non-blocking to ask whether the owner is still there, blocking to
wait for it. A waiter blocks on the youngest live ticket older than its own,
so a release wakes exactly its successor, in the kernel, with no polling. The
machine lock is then free by construction, and only a process that knows
nothing of tickets can contend for it. A releaser's next ticket is numbered
after every current waiter's, which is all the fairness there is to want. A
crash drops both locks at once; the ticket left behind is swept by the next
process that scans past it.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

LOCKS_DIR = Path.home() / ".cache" / "slipstream" / "locks"
MACHINE_LOCK = LOCKS_DIR / "machine.lock"
PAUSE_FILE = LOCKS_DIR / "pause"

# The only polling left: a paused waiter looking for the pause to lift, and a
# blocked one looking at should_stop and the pause file. Neither is on the
# handoff path.
PAUSE_POLL_SECS = 2.0
STOP_POLL_SECS = 1.0
# How long work left behind by a dead holder gets to end on SIGTERM.
REAP_GRACE_SECS = 10.0
DEFAULT_PAUSE_TTL = 4 * 3600

# EX_TEMPFAIL: the work is fine, the machine is busy.
EXIT_BUSY = 75


class LockBusy(RuntimeError):
    """Another process holds the machine lock."""


class Holder:
    """Who is holding the lock, as recorded in the lock file."""

    def __init__(
        self,
        pid: int,
        role: str,
        since: float,
        job: str = "",
        pgid: int | None = None,
        ticket: int | None = None,
    ):
        self.pid = pid
        self.role = role
        self.since = since
        self.job = job
        self.pgid = pgid
        # The holder's own ticket number, if it queued for the lock, so a
        # queue listing can leave out that one ticket and no other.
        self.ticket = ticket

    @property
    def alive(self) -> bool:
        # The file outlives a killed holder, so the pid is checked before it is
        # reported as the reason for a refusal.
        return _pid_alive(self.pid)

    def __str__(self) -> str:
        held = int(time.time() - self.since)
        alive = "" if self.alive else " (no longer running)"
        job = f" on {self.job}" if self.job else ""
        return f"{self.role} pid {self.pid}{job}, holding for {held // 60}m{alive}"


class Waiter(NamedTuple):
    """A ticket in the queue: who is waiting, for what, since when."""

    seq: int
    pid: int
    role: str
    job: str
    since: float
    alive: bool

    def __str__(self) -> str:
        waited = int(time.time() - self.since)
        job = f" on {self.job}" if self.job else ""
        gone = "" if self.alive else " (no longer running)"
        return f"{self.role} pid {self.pid}{job}, waiting for {waited // 60}m{gone}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_record(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        # Also covers a read racing the writer; the only cost is a vaguer
        # report line.
        return None
    return data if isinstance(data, dict) else None


def _read_holder(path: Path) -> Holder | None:
    data = _read_record(path)
    if not data:
        return None
    try:
        pgid = data.get("pgid")
        ticket = data.get("ticket")
        return Holder(
            int(data["pid"]),
            str(data["role"]),
            float(data["since"]),
            str(data.get("job", "")),
            int(pgid) if pgid is not None else None,
            int(ticket) if ticket is not None else None,
        )
    except (ValueError, KeyError, TypeError):
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


def own_process_group() -> None:
    """Lead a process group of our own, so that work left behind by a crash
    can be found and ended by the next holder.

    A foreground job and a launchd service already lead their own group; a
    process started by a non-interactive shell sits in the shell's, which no
    one may kill wholesale. Children inherit the group, so a build's compiler
    and a bench's engine are then exactly the members of ours. The holder
    record names the group only when it is led by the holder, which is the
    condition under which killing it reaches nothing else.

    A member of the terminal's foreground group stays put: ``build | tee``
    puts us in the pipeline's group, and leaving it would take us out of
    reach of Ctrl-C. An operator is at that terminal; the reaper is for the
    unattended case.
    """
    try:
        if os.getpgid(0) == os.getpid():
            return
        for fd in (0, 1, 2):
            try:
                if os.tcgetpgrp(fd) == os.getpgid(0):
                    return
            except OSError:
                continue  # not a terminal
        os.setpgid(0, 0)
    except OSError:
        pass


def _block(fd: int, op: int, should_stop: Callable[[], bool]) -> bool:
    """flock(fd, op), blocking, while the caller keeps watching should_stop.

    CPython retries a blocking flock after a signal handler that does not
    raise, so a shutdown request would not be seen until the lock arrived;
    the blocking call goes in a thread instead. True: acquired, the caller
    owns fd. False: abandoned; the thread owns fd and gives the lock straight
    back when the kernel finally hands it over, so an abandoned wait never
    holds anything. Whichever side moves first claims the fd under a lock,
    so the two cannot both think the other has it.
    """
    claim = threading.Lock()
    owner: list[str | None] = [None]
    failure: list[BaseException | None] = [None]
    done = threading.Event()

    def run() -> None:
        try:
            fcntl.flock(fd, op)
        except OSError as e:
            failure[0] = e
            with claim:
                if owner[0] is None:
                    owner[0] = "caller"
                else:
                    os.close(fd)
            done.set()
            return
        with claim:
            if owner[0] is None:
                owner[0] = "caller"
                done.set()
                return
        # Abandoned: nobody wants it any more.
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        done.set()

    threading.Thread(target=run, daemon=True).start()
    while not done.wait(STOP_POLL_SECS):
        if should_stop():
            with claim:
                if owner[0] is None:
                    owner[0] = "thread"
                    return False
            # The thread handed it over in the meantime; it is ours.
            done.wait()
            break
    if failure[0] is not None:
        os.close(fd)
        raise failure[0]
    return True


class _Ticket:
    def __init__(self, fd: int, seq: int, path: Path):
        self.fd: int | None = fd
        self.seq = seq
        self.path = path

    def drop(self) -> None:
        if self.fd is not None:
            os.close(self.fd)  # releases LOCK_EX: the successor wakes
            self.fd = None
        self.path.unlink(missing_ok=True)


class MachineLock:
    """Exclusive, advisory, released by the kernel when the process dies.

    flock is not inherited by subprocesses, so a build's children do not hold
    it open past the parent. Which is why the holder record names a process
    group: that is how the next holder finds what a dead one left running.
    """

    def __init__(self, role: str, path: Path | None = None):
        self.role = role
        self.path = path or MACHINE_LOCK
        self._fd: int | None = None
        self._ticket: _Ticket | None = None
        self._job = ""

    @property
    def queue_dir(self) -> Path:
        # Derived at use rather than in __init__: tests point path somewhere
        # after construction, and the queue has to move with it.
        return self.path.parent / "queue"

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

    def waiters(self) -> list[Waiter]:
        """The queue in order, without the holder's own ticket. Read-only."""
        holder = self.probe()
        out = []
        for path in self._tickets():
            seq, pid = _parse_ticket(path.name)
            if holder is not None and seq == holder.ticket:
                continue
            data = _read_record(path) or {}
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                alive = True
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                alive = False
            finally:
                os.close(fd)
            out.append(
                Waiter(
                    seq,
                    pid,
                    str(data.get("role", "?")),
                    str(data.get("job", "")),
                    float(data.get("since", 0.0)),
                    alive,
                )
            )
        return out

    # --- taking it ---

    def try_acquire(self, job: str = "") -> bool:
        """Take the lock if it is free right now; no ticket, no pause.

        This is the ad-hoc path. It steps around a paused queue, which is
        what pause is for, and it can never jump a live one: a chain head
        takes the machine within a millisecond of it being free.
        """
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
        self._took(fd, job, None)
        return True

    def acquire(
        self,
        should_stop: Callable[[], bool] | None = None,
        *,
        wait: bool,
        log: Callable[[str], None] | None = None,
        job: str = "",
    ) -> bool:
        """Take the lock. Returns False if we gave up.

        Daemons queue; ad-hoc commands try once and raise LockBusy so the
        operator is told who is holding it rather than blocking for hours
        behind a bench.
        """
        if self._fd is not None:
            return True
        if not wait:
            if self.try_acquire(job):
                return True
            holder = self.holder()
            raise LockBusy(
                f"another slipstream process holds the machine lock"
                f"{f': {holder}' if holder else ''}"
            )
        should_stop = should_stop or (lambda: False)
        ticket = self._publish(job)
        acquired = False
        try:
            announced = False
            announced_pause = False
            while not should_stop():
                pred = self._predecessor(ticket.seq)
                if pred is not None:
                    pfd, ppath = pred
                    if log and not announced:
                        ahead = _read_record(ppath) or {}
                        log(
                            "waiting for the machine lock behind "
                            f"{ahead.get('role', '?')} pid {ahead.get('pid', '?')}"
                        )
                        announced = True
                    if not _block(pfd, fcntl.LOCK_SH, should_stop):
                        return False
                    os.close(pfd)  # never hold a predecessor's lock
                    ppath.unlink(missing_ok=True)  # it is finished or dead
                    continue  # woke: rescan, never assume
                if self._paused(log if not announced_pause else None):
                    announced_pause = True
                    if not _sleep(PAUSE_POLL_SECS, should_stop):
                        return False
                    continue
                announced_pause = False
                # Head of the chain, nothing in front: the machine is free
                # unless a process that knows nothing of tickets has it, and
                # then this sleeps until it lets go. A pause set meanwhile
                # has to be seen from inside that sleep too, or the operator's
                # command would lose to us the moment the ad-hoc holder lets
                # go; the ticket keeps our place while we step back.
                self.path.parent.mkdir(parents=True, exist_ok=True)
                mfd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
                paused = False

                def stop_or_pause() -> bool:
                    nonlocal paused
                    paused = paused_until() is not None
                    return paused or should_stop()

                if not _block(mfd, fcntl.LOCK_EX, stop_or_pause):
                    if paused and not should_stop():
                        continue
                    return False
                self._took(mfd, job, ticket)
                acquired = True
                return True
            return False
        finally:
            if not acquired:
                ticket.drop()

    def set_job(self, job: str) -> None:
        """Name what the hold is for, once known, so status can say."""
        self._job = job
        if self._fd is not None:
            self._write_holder(self._fd, job)
        if self._ticket is not None and self._ticket.fd is not None:
            _rewrite(self._ticket.fd, self._ticket_record(job))

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)  # the machine first,
        os.close(self._fd)
        self._fd = None
        if self._ticket is not None:
            self._ticket.drop()  # then wake the successor
            self._ticket = None

    def __enter__(self) -> MachineLock:
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    # --- internals ---

    def _took(self, fd: int, job: str, ticket: _Ticket | None) -> None:
        # Before the holder record is written: from here the lock is held, and
        # if this is not set, release() is a no-op and the fd is never closed.
        # flock does not recurse across open file descriptions, so the same
        # process would then be refused by its own lock for the rest of its
        # life, with no holder in the file to say why.
        previous = _read_holder(self.path)
        self._fd = fd
        self._ticket = ticket
        self._job = job
        self._reap(previous)
        self._write_holder(fd, job)

    def _write_holder(self, fd: int, job: str) -> None:
        pid = os.getpid()
        pgid = os.getpgid(0)
        record = {"pid": pid, "role": self.role, "since": time.time(), "job": job}
        if pgid == pid:
            # Only a group we lead: killing it then reaches our descendants
            # and nothing else.
            record["pgid"] = pgid
        if self._ticket is not None:
            record["ticket"] = self._ticket.seq
        try:
            _rewrite(fd, record)
        except OSError:
            # The record is diagnostic; the lock is the thing. A full disk must
            # not cost the caller the lock it just took.
            pass

    def _reap(self, previous: Holder | None) -> None:
        """End what a dead holder left running before starting our own work.

        The lock is not inherited, so a compiler or an engine that outlived a
        crashed holder is invisible to it; the process group is how it is
        found. Only a group led by the dead holder itself is touched.
        """
        if (
            previous is None
            or previous.pid == os.getpid()
            or previous.pgid is None
            or previous.pgid != previous.pid
            or previous.alive
        ):
            return
        try:
            os.killpg(previous.pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + REAP_GRACE_SECS
        while time.monotonic() < deadline:
            time.sleep(0.1)
            try:
                os.killpg(previous.pgid, 0)
            except ProcessLookupError:
                return
        try:
            os.killpg(previous.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def _paused(self, log: Callable[[str], None] | None) -> bool:
        until = paused_until()
        if until is None:
            return False
        if log:
            log(
                f"paused by operator until {time.strftime('%H:%M', time.localtime(until))}"
            )
        return True

    def _tickets(self) -> list[Path]:
        try:
            names = os.listdir(self.queue_dir)
        except FileNotFoundError:
            return []
        found = []
        for name in names:
            seq, pid = _parse_ticket(name)
            if seq is not None:
                found.append((seq, self.queue_dir / name))
        return [path for _, path in sorted(found)]

    def _ticket_record(self, job: str) -> dict:
        return {"pid": os.getpid(), "role": self.role, "job": job, "since": time.time()}

    def _publish(self, job: str) -> _Ticket:
        """A new ticket at the back of the queue, locked before it is seen.

        Locked first, then renamed into place: a scanner that met it between
        creation and its lock would take it for dead and sweep it. The lock is
        on the inode, so the rename keeps it. The number is taken under the
        counter's own lock, in the same step as the rename, so no ticket with
        a smaller number can appear after anyone's scan.
        """
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=self.queue_dir)
        tmp = Path(tmp_name)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _rewrite(fd, self._ticket_record(job))
            seq_fd = os.open(self.queue_dir / ".seq", os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(seq_fd, fcntl.LOCK_EX)
                raw = os.read(seq_fd, 32)
                try:
                    counter = int(raw)
                except ValueError:
                    counter = 0
                # Also above every ticket on disk: a counter file torn by a
                # crash must not number a newcomer ahead of the queue.
                highest = max(
                    (_parse_ticket(p.name)[0] for p in self._tickets()), default=0
                )
                seq = max(counter, highest) + 1
                os.lseek(seq_fd, 0, os.SEEK_SET)
                os.ftruncate(seq_fd, 0)
                os.write(seq_fd, str(seq).encode())
                path = self.queue_dir / f"{seq:012d}-{os.getpid()}"
                os.rename(tmp, path)
            finally:
                os.close(seq_fd)
        except BaseException:
            os.close(fd)
            tmp.unlink(missing_ok=True)
            raise
        return _Ticket(fd, seq, path)

    def _predecessor(self, seq: int) -> tuple[int, Path] | None:
        """(fd, path) of the youngest live ticket older than ``seq``.

        Youngest, not oldest: each waiter then waits on exactly the one in
        front of it, and a release wakes one process rather than all of them.
        Dead tickets met on the way are swept.
        """
        for path in reversed(self._tickets()):
            other, _ = _parse_ticket(path.name)
            if other >= seq:
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
            except FileNotFoundError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return fd, path  # live
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            path.unlink(missing_ok=True)
        return None


def _parse_ticket(name: str) -> tuple[int, int] | tuple[None, None]:
    seq, sep, pid = name.partition("-")
    if not sep or not seq.isdigit() or not pid.isdigit():
        return None, None
    return int(seq), int(pid)


def _rewrite(fd: int, record: dict) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, json.dumps(record).encode())


def _sleep(seconds: float, should_stop: Callable[[], bool]) -> bool:
    """Sleep in small steps; False if a stop was requested meanwhile."""
    deadline = time.monotonic() + seconds
    while True:
        if should_stop():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(0.1, remaining))
