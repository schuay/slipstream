# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import threading
import time

import pytest

from slipstream.admission import Admission, AdmissionStopped, StopReason
from slipstream.bench_window import write_override
from slipstream.config import BenchProcessConfig
from slipstream.lock import MachineLock


def test_closed_takes_no_ticket_or_machine_lock(tmp_path):
    machine = MachineLock("watch", tmp_path / "machine.lock")
    write_override("close")
    with pytest.raises(AdmissionStopped) as caught:
        with Admission(BenchProcessConfig(), machine, wait=True).commit("v8 123"):
            pytest.fail("closed work admitted")
    assert caught.value.reason == StopReason.DEFERRED
    assert not machine.path.exists()
    assert not machine.queue_dir.exists()


def test_final_check_releases_lock_before_side_effects(tmp_path, monkeypatch):
    machine = MachineLock("bench", tmp_path / "machine.lock")
    acquire = machine.acquire

    def close_after_acquire(*args, **kwargs):
        result = acquire(*args, **kwargs)
        write_override("close")
        return result

    monkeypatch.setattr(machine, "acquire", close_after_acquire)
    with pytest.raises(AdmissionStopped):
        with Admission(BenchProcessConfig(), machine).commit("jsc 123"):
            pytest.fail("racing close admitted")
    assert not machine.held
    assert MachineLock("status", machine.path).probe() is None


def test_admitted_commit_finishes_and_retains_identity(tmp_path):
    machine = MachineLock("watch", tmp_path / "machine.lock")
    admission = Admission(BenchProcessConfig(), machine)
    with admission.commit("v8 123"):
        started = machine.probe().since
        write_override("close")
        for _ in range(3):
            assert machine.probe().job == "v8 123"
            assert machine.held
        machine.set_job("v8 123 teardown")
        assert machine.probe().since == started
    with pytest.raises(AdmissionStopped):
        with admission.commit("v8 124"):
            pytest.fail("next commit admitted")


def test_queued_close_withdraws_ticket_and_does_not_block_builder(
    tmp_path, monkeypatch
):
    import slipstream.lock as mod

    monkeypatch.setattr(mod, "STOP_POLL_SECS", 0.01)
    path = tmp_path / "machine.lock"
    holder, bencher, builder = (
        MachineLock(role, path) for role in ("build", "watch", "build")
    )
    holder.acquire(wait=True)
    reasons, errors = [], []

    def wait_bench():
        try:
            with Admission(BenchProcessConfig(), bencher, wait=True).commit("v8 123"):
                errors.append("admitted")
        except AdmissionStopped as exc:
            reasons.append(exc.reason)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=wait_bench)
    thread.start()
    try:
        deadline = time.monotonic() + 2
        while len(holder.waiters()) < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(holder.waiters()) == 1
        write_override("close")
        thread.join(2)
        assert not thread.is_alive()
        assert reasons == [StopReason.DEFERRED] and not errors
        assert holder.waiters() == []
    finally:
        holder.release()
        thread.join(2)
    # The abandoned kernel wait must not retain the bencher's queue ticket.
    assert builder.acquire(wait=True)
    builder.release()


def test_predicate_exception_releases_ticket_and_abandoned_fd(tmp_path, monkeypatch):
    import slipstream.lock as mod

    monkeypatch.setattr(mod, "STOP_POLL_SECS", 0.01)
    holder = MachineLock("build", tmp_path / "machine.lock")
    holder.try_acquire()
    waiter = MachineLock("watch", holder.path)
    checks = 0

    def broken():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RuntimeError("policy read failed")
        return False

    try:
        with pytest.raises(RuntimeError, match="policy read failed"):
            waiter.acquire(broken, wait=True)
        assert holder.waiters() == []
    finally:
        holder.release()
    peer = MachineLock("build", holder.path)
    assert peer.acquire(wait=True)
    peer.release()


def test_exception_inside_commit_releases(tmp_path):
    machine = MachineLock("bench", tmp_path / "machine.lock")
    with pytest.raises(RuntimeError):
        with Admission(BenchProcessConfig(), machine).commit("v8 123"):
            raise RuntimeError("provisioning")
    assert not machine.held
