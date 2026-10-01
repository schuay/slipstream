# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""SSH wire bounds and historical cursor compatibility."""

import subprocess
import sys
import time

import pytest

from slipstream.relay import RemoteSpool, read_cursor, write_cursor


def test_cursor_missing_is_zero_but_corruption_is_not(tmp_path):
    path = tmp_path / "old.cursor"
    assert read_cursor(path) == 0
    for bad in ("", "oops", "-1", "1.2"):
        path.write_text(bad)
        with pytest.raises(ValueError, match="unreadable cursor"):
            read_cursor(path)
    write_cursor(path, 41)
    assert read_cursor(path) == 41
    assert not path.with_suffix(".tmp").exists()


def test_remote_quotes_paths_and_bounds_discovery(monkeypatch):
    commands = []

    def ssh(self, command):
        commands.append(command)
        return "00000002.csv\n00000004.csv\n4\n"

    monkeypatch.setattr(RemoteSpool, "_ssh", ssh)
    spool = RemoteSpool("box2", "~/my outbox")
    assert spool.list(cursor=1, limit=2) == [2, 4]
    assert "~/'my outbox'" in commands[0]
    assert "head -n 2" in commands[0]
    assert "($0 + 0) > 1" in commands[0]
    spool.fetch(4)
    assert commands[1] == "cat ~/'my outbox'/00000004.csv"


@pytest.mark.parametrize("reason", ["timeout", "cancel", "oversize", "exit"])
def test_real_ssh_transport_is_bounded_and_reaped(monkeypatch, reason):
    real_popen = subprocess.Popen
    children = []

    def popen(argv, **kwargs):
        script = (
            "print('x'*20000, flush=True); import time; time.sleep(30)"
            if reason == "oversize"
            else "raise SystemExit(255)"
            if reason == "exit"
            else "import time; time.sleep(30)"
        )
        proc = real_popen([sys.executable, "-c", script], **kwargs)
        children.append(proc)
        return proc

    monkeypatch.setattr("slipstream.relay.subprocess.Popen", popen)
    start = time.monotonic()
    spool = RemoteSpool(
        "box",
        "/spool",
        timeout=0.15,
        max_bytes=1024,
        should_stop=lambda: reason == "cancel",
    )
    expected = {
        "timeout": TimeoutError,
        "cancel": InterruptedError,
        "oversize": ValueError,
        "exit": subprocess.CalledProcessError,
    }[reason]
    with pytest.raises(expected):
        spool.fetch(1)
    assert time.monotonic() - start < 2
    assert children[0].poll() is not None
