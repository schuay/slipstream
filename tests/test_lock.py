# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

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
        a.release()
        assert not a.held

    def test_release_of_an_unheld_lock_is_a_noop(self, lock_path):
        MachineLock("builder", lock_path).release()


class TestHolderFile:
    def test_records_who_holds_it(self, lock_path):
        a = MachineLock("builder", lock_path)
        a.try_acquire()
        data = json.loads(lock_path.read_text())
        assert data["pid"] == os.getpid() and data["role"] == "builder"
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

    def test_a_daemon_gives_up_when_asked_to_stop(self, lock_path, monkeypatch):
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        a = MachineLock("builder", lock_path)
        a.try_acquire()
        b = MachineLock("watch", lock_path)
        calls = [0]

        def should_stop():
            calls[0] += 1
            return calls[0] > 3

        assert b.acquire(should_stop, wait=True) is False
        a.release()

    def test_a_daemon_takes_it_once_free(self, lock_path, monkeypatch):
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        b = MachineLock("watch", lock_path)
        assert b.acquire(wait=True) is True
        b.release()


class TestHandoff:
    def test_the_releaser_waits_before_re_acquiring(self, lock_path, monkeypatch):
        """Otherwise it wins its own re-acquisition race and starves its peer."""
        monkeypatch.setattr(lock_mod, "HANDOFF_SECS", 0.2)
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        a = MachineLock("watch", lock_path)
        a.acquire(wait=True)
        a.release()
        t0 = time.monotonic()
        a.acquire(wait=True)
        assert time.monotonic() - t0 >= 0.15
        a.release()

    def test_the_first_acquire_does_not_wait(self, lock_path, monkeypatch):
        monkeypatch.setattr(lock_mod, "HANDOFF_SECS", 5.0)
        a = MachineLock("watch", lock_path)
        t0 = time.monotonic()
        a.acquire(wait=True)
        assert time.monotonic() - t0 < 1.0
        a.release()

    def test_a_stop_request_cuts_the_handoff_short(self, lock_path, monkeypatch):
        monkeypatch.setattr(lock_mod, "HANDOFF_SECS", 30.0)
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        a = MachineLock("watch", lock_path)
        a.acquire(wait=True)
        a.release()
        t0 = time.monotonic()
        assert a.acquire(lambda: True, wait=True) is False
        assert time.monotonic() - t0 < 1.0


class TestPause:
    def test_pause_blocks_a_daemon_and_resume_releases_it(
        self, lock_path, pause_path, monkeypatch
    ):
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        lock_mod.pause(3600, pause_path)
        a = MachineLock("watch", lock_path)
        polls = [0]

        def should_stop():
            polls[0] += 1
            if polls[0] == 5:
                lock_mod.resume(pause_path)
            return polls[0] > 20

        assert a.acquire(should_stop, wait=True) is True
        a.release()

    def test_an_expired_pause_does_not_block(self, lock_path, pause_path):
        lock_mod.pause(-1, pause_path)
        assert lock_mod.paused_until(pause_path) is None
        a = MachineLock("watch", lock_path)
        assert a.acquire(wait=True) is True
        a.release()

    def test_resume_without_a_pause_is_a_noop(self, pause_path):
        lock_mod.resume(pause_path)


class TestAcrossProcesses:
    def test_the_kernel_releases_it_when_the_holder_dies(self, lock_path):
        code = (
            "import sys, time\n"
            "sys.path.insert(0, %r)\n"
            "from slipstream.lock import MachineLock\n"
            "from pathlib import Path\n"
            "l = MachineLock('builder', Path(%r))\n"
            "assert l.try_acquire()\n"
            "print('held', flush=True)\n"
            "time.sleep(30)\n" % (str(__import__("pathlib").Path.cwd()), str(lock_path))
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


class TestHandoffScope:
    def test_an_ad_hoc_command_yields_briefly_not_for_a_minute(
        self, lock_path, monkeypatch
    ):
        """bench releases once per commit; a full minute each over a long range
        is hours. It still yields long enough for a polling daemon to win."""
        monkeypatch.setattr(lock_mod, "HANDOFF_SECS", 30.0)
        monkeypatch.setattr(lock_mod, "ADHOC_HANDOFF_SECS", 0.2)
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        a = MachineLock("bench", lock_path)
        a.acquire(wait=False)
        a.release()
        t0 = time.monotonic()
        a.acquire(wait=False)
        elapsed = time.monotonic() - t0
        assert 0.15 <= elapsed < 5.0
        a.release()

    def test_a_polling_daemon_wins_the_ad_hoc_handoff(self, lock_path, monkeypatch):
        """The point of yielding at all: the releaser must not win its own
        re-acquisition race against a peer that is polling."""
        import threading

        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        monkeypatch.setattr(lock_mod, "ADHOC_HANDOFF_SECS", 0.5)
        adhoc = MachineLock("bench", lock_path)
        daemon = MachineLock("watch", lock_path)
        assert adhoc.try_acquire()

        polling = threading.Event()
        give_up = threading.Event()

        def wait_for_it():
            polling.set()
            # Bounded, so a regression fails the test instead of hanging it.
            daemon.acquire(give_up.is_set, wait=True)

        t = threading.Thread(target=wait_for_it)
        t.start()
        try:
            assert polling.wait(timeout=5)
            time.sleep(0.05)  # the daemon is in its poll loop
            adhoc.release()
            with pytest.raises(LockBusy, match="watch"):
                adhoc.acquire(wait=False)
        finally:
            give_up.set()
            t.join(timeout=5)
            adhoc.release()
            daemon.release()

    def test_a_daemon_still_waits(self, lock_path, monkeypatch):
        monkeypatch.setattr(lock_mod, "HANDOFF_SECS", 0.2)
        monkeypatch.setattr(lock_mod, "POLL_SECS", 0.01)
        a = MachineLock("watch", lock_path)
        a.acquire(wait=True)
        a.release()
        t0 = time.monotonic()
        a.acquire(wait=True)
        assert time.monotonic() - t0 >= 0.15
        a.release()


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
            raise OSError(37, "No locks available")

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

    def test_a_missing_lock_file_probes_as_free(self, tmp_path):
        assert MachineLock("status", tmp_path / "nope.lock").probe() is None
