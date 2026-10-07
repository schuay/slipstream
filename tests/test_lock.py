# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from slipstream import lock as lock_mod
from slipstream.lock import LockBusy, MachineLock


@pytest.fixture
def lock_path(tmp_path):
    return tmp_path / "machine.lock"


@pytest.fixture
def pause_path(tmp_path, monkeypatch):
    p = tmp_path / "pause"
    monkeypatch.setattr(lock_mod, "PAUSE_FILE", p)
    return p


@pytest.fixture
def quick(monkeypatch):
    """Short polls where the code polls at all: a paused waiter, a stop check."""
    monkeypatch.setattr(lock_mod, "PAUSE_POLL_SECS", 0.02)
    monkeypatch.setattr(lock_mod, "STOP_POLL_SECS", 0.02)


class TestExclusion:
    def test_two_locks_in_one_process_are_the_same_file(self, lock_path):
        """flock is per open file description, so a second open must be refused."""
        a = MachineLock("builder", lock_path)
        b = MachineLock("bencher", lock_path)
        assert a.try_acquire()
        assert not b.try_acquire()
        a.release()
        assert b.try_acquire()
        b.release()

    def test_acquire_is_reentrant_for_the_holder(self, lock_path):
        a = MachineLock("builder", lock_path)
        assert a.try_acquire()
        assert a.try_acquire()
        assert a.acquire(wait=True)
        a.release()
        assert not a.held

    def test_release_of_an_unheld_lock_is_a_noop(self, lock_path):
        MachineLock("builder", lock_path).release()


class TestHolderFile:
    def test_records_who_holds_it(self, lock_path):
        a = MachineLock("builder", lock_path)
        a.try_acquire("v8 101")
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid() and data["role"] == "builder"
        assert data["job"] == "v8 101"
        assert "on v8 101" in str(a.holder())
        a.release()

    def test_the_job_can_be_named_once_known(self, lock_path):
        a = MachineLock("builder", lock_path)
        a.acquire(wait=True)
        a.set_job("chrome 1534000-109680")
        assert a.holder().job == "chrome 1534000-109680"
        a.release()

    def test_a_waiter_does_not_erase_the_identity_it_reports(self, lock_path):
        """open(path, "w") truncates at open; the waiter must not use it."""
        a = MachineLock("builder", lock_path)
        a.try_acquire()
        b = MachineLock("bencher", lock_path)
        assert not b.try_acquire()
        holder = b.holder()
        assert holder is not None
        assert holder.role == "builder" and holder.pid == os.getpid()
        assert "builder" in str(holder)
        a.release()

    def test_a_dead_holder_is_reported_as_such(self, lock_path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        # A pid that cannot exist: the file outlives a killed holder.
        lock_path.write_text(
            json.dumps({"pid": 2**30, "role": "builder", "since": time.time()})
        )
        holder = MachineLock("bencher", lock_path).holder()
        assert not holder.alive
        assert "no longer running" in str(holder)

    def test_a_torn_file_is_not_fatal(self, lock_path):
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text('{"pid": 1, "rol')
        assert MachineLock("bencher", lock_path).holder() is None


class TestAcquirePolicy:
    def test_ad_hoc_reports_the_holder_instead_of_waiting(self, lock_path):
        a = MachineLock("watch", lock_path)
        a.try_acquire()
        b = MachineLock("bench", lock_path)
        with pytest.raises(LockBusy, match="watch"):
            b.acquire(wait=False)
        a.release()

    def test_ad_hoc_leaves_no_ticket_behind(self, lock_path):
        a = MachineLock("watch", lock_path)
        a.try_acquire()
        b = MachineLock("bench", lock_path)
        with pytest.raises(LockBusy):
            b.acquire(wait=False)
        assert b.waiters() == []
        a.release()
        b.acquire(wait=False)
        assert b.waiters() == []
        b.release()

    def test_a_daemon_gives_up_when_asked_to_stop(self, lock_path, quick):
        a = MachineLock("builder", lock_path)
        a.try_acquire()
        b = MachineLock("watch", lock_path)
        calls = [0]

        def should_stop():
            calls[0] += 1
            return calls[0] > 3

        assert b.acquire(should_stop, wait=True) is False
        assert b.waiters() == [], "a cancelled wait left its ticket"
        a.release()

    def test_a_daemon_takes_it_once_free(self, lock_path):
        b = MachineLock("watch", lock_path)
        assert b.acquire(wait=True) is True
        b.release()

    def test_a_stop_request_before_waiting_takes_nothing(self, lock_path):
        a = MachineLock("watch", lock_path)
        assert a.acquire(lambda: True, wait=True) is False
        assert not a.held and a.waiters() == []


def _hold_in_thread(lock, ready=None, *, hold_for=0.0, should_stop=None, log=None):
    """Acquire on a thread; returns (thread, result list)."""
    result = []

    def run():
        result.append(lock.acquire(should_stop, wait=True, log=log))
        if ready is not None:
            ready.set()
        if result[-1]:
            time.sleep(hold_for)
            lock.release()

    t = threading.Thread(target=run)
    t.start()
    return t, result


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


class TestQueue:
    """Admission is first come, first served, handed over by the kernel."""

    def test_waiters_are_served_in_arrival_order(self, lock_path):
        holder = MachineLock("build", lock_path)
        holder.try_acquire()
        order = []
        threads = []
        for i in range(3):
            peer = MachineLock(f"w{i}", lock_path)
            t = threading.Thread(
                target=lambda p=peer: (
                    p.acquire(wait=True),
                    order.append(p.role),
                    p.release(),
                )
            )
            t.start()
            threads.append(t)
            # Each is queued before the next starts, so arrival order is known.
            assert _wait_until(lambda n=len(threads): len(holder.waiters()) == n)
        assert [w.role for w in holder.waiters()] == ["w0", "w1", "w2"]
        holder.release()
        for t in threads:
            t.join(timeout=5)
        assert order == ["w0", "w1", "w2"]
        assert holder.waiters() == []

    def test_a_releaser_queues_behind_everyone_waiting(self, lock_path):
        """The reason the old code slept a minute after releasing: a
        releaser that asks again at once must not win its own race."""
        a = MachineLock("build", lock_path)
        a.try_acquire()
        b = MachineLock("watch", lock_path)
        t, got = _hold_in_thread(b, hold_for=0.2)
        assert _wait_until(lambda: len(a.waiters()) == 1)
        a.release()
        # b is woken by the kernel; a asks again immediately and must wait.
        t0 = time.monotonic()
        assert a.acquire(wait=True)
        assert time.monotonic() - t0 >= 0.15
        t.join(timeout=5)
        assert got == [True]
        a.release()

    def test_handoff_is_immediate(self, lock_path):
        a = MachineLock("build", lock_path)
        a.try_acquire()
        b = MachineLock("watch", lock_path)
        entered = threading.Event()
        t, _ = _hold_in_thread(b, entered)
        assert _wait_until(lambda: len(a.waiters()) == 1)
        t0 = time.monotonic()
        a.release()
        assert entered.wait(timeout=5)
        assert time.monotonic() - t0 < 0.5
        t.join(timeout=5)

    def test_a_cancelled_waiter_in_the_middle_is_skipped(self, lock_path, quick):
        a = MachineLock("build", lock_path)
        a.try_acquire()
        stop_b = threading.Event()
        b = MachineLock("w-b", lock_path)
        tb, got_b = _hold_in_thread(b, should_stop=stop_b.is_set)
        assert _wait_until(lambda: len(a.waiters()) == 1)
        c = MachineLock("w-c", lock_path)
        entered_c = threading.Event()
        tc, got_c = _hold_in_thread(c, entered_c)
        assert _wait_until(lambda: len(a.waiters()) == 2)
        stop_b.set()
        tb.join(timeout=5)
        assert got_b == [False]
        assert [w.role for w in a.waiters()] == ["w-c"]
        assert not entered_c.is_set(), "c entered while a still held"
        a.release()
        assert entered_c.wait(timeout=5)
        tc.join(timeout=5)
        assert got_c == [True]

    def test_a_stop_while_blocked_returns_promptly(self, lock_path, quick):
        a = MachineLock("build", lock_path)
        a.try_acquire()
        stop = threading.Event()
        b = MachineLock("watch", lock_path)
        t, got = _hold_in_thread(b, should_stop=stop.is_set)
        assert _wait_until(lambda: len(a.waiters()) == 1)
        t0 = time.monotonic()
        stop.set()
        t.join(timeout=5)
        assert got == [False] and time.monotonic() - t0 < 1.0
        # The abandoned wait holds nothing: a peer gets it the moment a lets go.
        a.release()
        c = MachineLock("bench", lock_path)
        assert c.acquire(wait=True)
        c.release()

    def test_a_dead_ticket_is_swept_by_the_next_scan(self, lock_path):
        a = MachineLock("build", lock_path)
        a.try_acquire()
        queue = a.queue_dir
        queue.mkdir(parents=True, exist_ok=True)
        # A ticket whose owner died without closing it cleanly: no lock on it.
        (queue / f"{1:012d}-{2**30}").write_text(
            json.dumps({"pid": 2**30, "role": "ghost", "job": "", "since": 0})
        )
        assert [w.role for w in a.waiters()] == ["ghost"]
        assert a.waiters()[0].alive is False
        b = MachineLock("watch", lock_path)
        t, _ = _hold_in_thread(b)
        assert _wait_until(lambda: [w.role for w in a.waiters()] == ["watch"])
        a.release()
        t.join(timeout=5)
        assert not list(queue.glob("*-*"))

    def test_a_torn_counter_cannot_number_a_newcomer_ahead(self, lock_path):
        a = MachineLock("build", lock_path)
        a.try_acquire()
        b = MachineLock("w1", lock_path)
        t, _ = _hold_in_thread(b)
        assert _wait_until(lambda: len(a.waiters()) == 1)
        (a.queue_dir / ".seq").write_text("")  # as a crash mid-write leaves it
        c = MachineLock("w2", lock_path)
        order = []
        tc = threading.Thread(
            target=lambda: (c.acquire(wait=True), order.append("w2"), c.release())
        )
        tc.start()
        assert _wait_until(lambda: len(a.waiters()) == 2)
        assert [w.role for w in a.waiters()] == ["w1", "w2"]
        a.release()
        t.join(timeout=5)
        tc.join(timeout=5)


class TestPause:
    def test_pause_blocks_a_daemon_and_resume_releases_it(
        self, lock_path, pause_path, quick
    ):
        lock_mod.pause(3600, pause_path)
        a = MachineLock("watch", lock_path)
        polls = [0]

        def should_stop():
            polls[0] += 1
            if polls[0] == 5:
                lock_mod.resume(pause_path)
            return polls[0] > 200

        assert a.acquire(should_stop, wait=True) is True
        a.release()

    def test_a_paused_waiter_keeps_its_place_and_an_ad_hoc_steps_around(
        self, lock_path, pause_path, quick
    ):
        """Pause exists so an operator command can run; the daemon in front
        of it must neither take the machine nor lose its position. The hard
        case: the daemon is already asleep in the kernel on machine.lock
        behind a holder without a ticket when the pause lands."""
        a = MachineLock("build", lock_path)
        a.try_acquire()
        b = MachineLock("watch", lock_path)
        entered = threading.Event()
        logs = []
        t, _ = _hold_in_thread(b, entered, log=logs.append)
        assert _wait_until(lambda: len(a.waiters()) == 1)
        lock_mod.pause(3600, pause_path)
        assert _wait_until(lambda: any("paused" in m for m in logs))
        a.release()
        time.sleep(0.1)
        assert not entered.is_set()
        assert [w.role for w in b.waiters()] == ["watch"]
        bench = MachineLock("bench", lock_path)
        assert bench.acquire(wait=False)
        time.sleep(0.1)
        assert not entered.is_set()
        bench.release()
        lock_mod.resume(pause_path)
        assert entered.wait(timeout=5)
        t.join(timeout=5)

    def test_an_expired_pause_does_not_block(self, lock_path, pause_path):
        lock_mod.pause(-1, pause_path)
        assert lock_mod.paused_until(pause_path) is None
        a = MachineLock("watch", lock_path)
        assert a.acquire(wait=True) is True
        a.release()

    def test_resume_without_a_pause_is_a_noop(self, pause_path):
        lock_mod.resume(pause_path)


_WORKER = """
import os, sys, time
sys.path.insert(0, %(cwd)r)
from pathlib import Path
from slipstream.lock import MachineLock
lock = MachineLock(%(role)r, Path(%(path)r))
out = open(%(log)r, "a")
def emit(ev, seq=0):
    out.write(f"{time.monotonic_ns()} {os.getpid()} {lock.role} {ev} {seq}\\n")
    out.flush()
for i in range(%(rounds)d):
    assert lock.acquire(wait=True)
    emit("enter", lock._ticket.seq)
    time.sleep(%(hold)f)
    emit("exit", lock._ticket.seq)
    lock.release()
"""


class TestAcrossProcesses:
    def _spawn(self, lock_path, log, role, rounds, hold):
        code = _WORKER % {
            "cwd": str(Path.cwd()),
            "role": role,
            "path": str(lock_path),
            "log": str(log),
            "rounds": rounds,
            "hold": hold,
        }
        return subprocess.Popen([sys.executable, "-c", code])

    def test_the_kernel_releases_it_when_the_holder_dies(self, lock_path):
        code = (
            "import sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from slipstream.lock import MachineLock\n"
            "from pathlib import Path\n"
            "l = MachineLock('builder', Path(%r))\n"
            "assert l.try_acquire()\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n" % (str(Path.cwd()), str(lock_path))
        )
        p = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
        )
        try:
            assert p.stdout.readline().strip() == "held"
            assert not MachineLock("watch", lock_path).try_acquire()
            holder = MachineLock("watch", lock_path).holder()
            assert holder.pid == p.pid and holder.alive
        finally:
            p.kill()
            p.wait()
        assert MachineLock("watch", lock_path).try_acquire()

    def test_three_processes_exclude_each_other_in_fifo_order(
        self, lock_path, tmp_path
    ):
        """Each re-queues the instant it releases, so with a handoff sleep
        gone the only thing keeping them fair is the queue itself."""
        log = tmp_path / "events"
        log.touch()
        procs = []
        for i, role in enumerate("ABC"):
            procs.append(self._spawn(lock_path, log, role, rounds=4, hold=0.03))
            time.sleep(0.05)
        for p in procs:
            assert p.wait(timeout=30) == 0
        holding = None
        last_seq = 0
        order = []
        for line in log.read_text().splitlines():
            _, pid, role, event, seq = line.split()
            if event == "enter":
                assert holding is None, f"{role} entered while {holding} held"
                assert int(seq) > last_seq, "not in ticket order"
                holding, last_seq = role, int(seq)
                order.append(role)
            else:
                holding = None
        assert len(order) == 12
        # Once all three are queued, no role holds twice within three holds.
        last = min(max(i for i, r in enumerate(order) if r == x) for x in "ABC")
        for i in range(3, last - 1):
            assert len(set(order[i : i + 3])) == 3, order

    def test_a_waiter_takes_over_when_the_holder_is_killed(self, lock_path):
        a = MachineLock("build", lock_path)
        a.try_acquire()
        code = (
            "import os, sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from slipstream.lock import MachineLock\n"
            "l = MachineLock('watch', Path(%r))\n"
            "l.acquire(wait=True)\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n" % (str(Path.cwd()), str(lock_path))
        )
        waiter = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
        )
        try:
            assert _wait_until(lambda: len(a.waiters()) == 1)
            # "Killed" from the waiter's point of view: the holder vanishes
            # without releasing. Closing the fd is what the kernel does.
            os.close(a._fd)
            a._fd = None
            assert waiter.stdout.readline().strip() == "held"
            assert MachineLock("status", lock_path).probe().role == "watch"
        finally:
            waiter.kill()
            waiter.wait()


class TestReaper:
    def test_work_left_by_a_dead_holder_is_ended_by_the_next(self, lock_path):
        """A compiler outliving a crashed builder would run beside the next
        bench. The lock is not inherited, so the process group is how it is
        found; the holder leads its own, so only its descendants are in it."""
        code = (
            "import os, subprocess, sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from pathlib import Path\n"
            "from slipstream.lock import MachineLock, own_process_group\n"
            "own_process_group()\n"
            "l = MachineLock('build', Path(%r))\n"
            "assert l.try_acquire()\n"
            "child = subprocess.Popen(['sleep', '60'])\n"
            "print(child.pid, flush=True)\n"
            "time.sleep(60)\n" % (str(Path.cwd()), str(lock_path))
        )
        # Unattended, as the reaper is meant for: a child whose stdin or
        # stderr were the developer's terminal would stay in its foreground
        # group, by design, and record no pgid.
        p = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        child_pid = int(p.stdout.readline())
        assert json.loads(lock_path.read_text())["pgid"] == p.pid
        p.kill()
        p.wait()  # dead, not a zombie: that is what the next holder sees
        os.kill(child_pid, 0)  # the orphan is still running
        nxt = MachineLock("watch", lock_path)
        assert nxt.try_acquire()
        try:
            assert _wait_until(lambda: not _alive(child_pid), timeout=5)
        finally:
            nxt.release()
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_a_group_not_led_by_the_holder_is_left_alone(self, lock_path):
        """Our own group when we do not lead it is the shell's, or pytest's."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(
                {"pid": 2**30, "role": "build", "since": 0, "pgid": os.getpgid(0)}
            )
        )
        a = MachineLock("watch", lock_path)
        assert a.try_acquire()  # would have killed this very process
        a.release()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TestAcquireIsAtomic:
    def test_a_failed_holder_write_does_not_wedge_the_lock(
        self, lock_path, monkeypatch
    ):
        """flock succeeded, so the lock is held; if _fd were not set, release
        would be a no-op and the process would be refused by its own lock for
        the rest of its life, with no holder record to say why."""
        import os as _os

        def enospc(fd, n):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(_os, "ftruncate", enospc)
        a = MachineLock("watch", lock_path)
        assert a.try_acquire()
        assert a.held
        a.release()
        assert not a.held
        # A peer can take it now, which is the proof it was really released.
        b = MachineLock("build", lock_path)
        assert b.try_acquire()
        b.release()

    def test_a_non_contention_flock_error_does_not_leak_the_fd(
        self, lock_path, monkeypatch
    ):
        import fcntl as _fcntl

        def enolck(fd, op):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(_fcntl, "flock", enolck)
        a = MachineLock("watch", lock_path)
        with pytest.raises(OSError):
            a.try_acquire()
        assert not a.held


class TestProbeDoesNotTake:
    def test_a_probe_does_not_block_an_ad_hoc_command(self, lock_path):
        """bus status is read-only; it must not be able to fail a bench."""
        reporter = MachineLock("status", lock_path)
        assert reporter.probe() is None
        assert reporter.waiters() == []
        # An ad-hoc command still gets the lock after the probe.
        bench = MachineLock("bench", lock_path)
        bench.acquire(wait=False)
        bench.release()

    def test_a_probe_names_a_real_holder(self, lock_path):
        holder = MachineLock("watch", lock_path)
        holder.try_acquire()
        seen = MachineLock("status", lock_path).probe()
        assert seen is not None and seen.role == "watch"
        holder.release()
        assert MachineLock("status", lock_path).probe() is None

    def test_a_probe_writes_nothing(self, lock_path):
        holder = MachineLock("watch", lock_path)
        holder.try_acquire()
        before = lock_path.read_bytes()
        MachineLock("status", lock_path).probe()
        assert lock_path.read_bytes() == before
        holder.release()

    def test_the_holders_own_ticket_is_not_a_waiter(self, lock_path):
        holder = MachineLock("watch", lock_path)
        holder.acquire(wait=True)
        assert MachineLock("status", lock_path).waiters() == []
        holder.release()

    def test_a_missing_lock_file_probes_as_free(self, tmp_path):
        assert MachineLock("status", tmp_path / "nope.lock").probe() is None
