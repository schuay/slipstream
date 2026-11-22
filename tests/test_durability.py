# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import sys

import pytest

from slipstream import durability
from slipstream.durability import FileLock


def test_lock_timeout_names_holder_in_another_process(tmp_path):
    path = tmp_path / ".bus.lock"
    script = f"""
import sys, time
from pathlib import Path
from slipstream.durability import FileLock, report_lock_holds
report_lock_holds("build", print)
def package():
    with FileLock(Path({str(path)!r})):
        print("held", flush=True)
        sys.stdin.read()
package()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline() == "held\n"
        with pytest.raises(TimeoutError) as exc:
            with FileLock(path, timeout=0.2):
                pass
        message = str(exc.value)
        assert f"by build pid {child.pid} at <string>:" in message
        assert "package" in message
        # The waiter's failed wait leaves the live holder's record alone.
        assert f'"pid": {child.pid}' in path.read_text()
    finally:
        child.stdin.close()
        child.wait(timeout=5)
    # A released lock carries no stale holder for the next waiter to report.
    assert path.read_text() == ""


def test_long_hold_is_reported_on_release(tmp_path, monkeypatch):
    reports = []
    monkeypatch.setattr(durability, "LONG_HOLD_SECONDS", 0)
    monkeypatch.setattr(durability, "_long_hold_log", reports.append)
    with FileLock(tmp_path / "lock", report_holds=True):
        pass
    with FileLock(tmp_path / "other"):
        pass
    assert len(reports) == 1
    assert reports[0].startswith("held lock for ")
    assert "test_long_hold_is_reported_on_release" in reports[0]


def test_short_hold_and_failed_wait_are_not_reported(tmp_path, monkeypatch):
    reports = []
    monkeypatch.setattr(durability, "_long_hold_log", reports.append)
    with FileLock(tmp_path / "lock", report_holds=True) as held:
        monkeypatch.setattr(durability, "LONG_HOLD_SECONDS", 0)
        with pytest.raises(TimeoutError):
            with FileLock(held.path, timeout=0, report_holds=True):
                pass
        assert reports == []
        monkeypatch.setattr(durability, "LONG_HOLD_SECONDS", 3600)
    assert reports == []


def test_unwritable_holder_record_does_not_fail_the_lock(tmp_path, monkeypatch):
    def full(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(durability.os, "ftruncate", full)
    with FileLock(tmp_path / "lock") as held:
        with pytest.raises(TimeoutError):
            with FileLock(held.path, timeout=0):
                pass
    with FileLock(tmp_path / "lock", timeout=0):
        pass
