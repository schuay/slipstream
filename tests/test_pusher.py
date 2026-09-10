# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
import time

import pytest

from slipstream.pusher import BackgroundPusher
from slipstream.store import CommitStore


@pytest.fixture
def cfg(config):
    # The primary connection creates + migrates the schema, as the collector
    # would before the pusher starts.
    primary = CommitStore(config.metadata_dir / "slipstream.db", bot=config.bot_name)
    primary.close()
    return config


def _make_pusher(cfg, monkeypatch, recorder):
    """BackgroundPusher whose push is replaced by `recorder`."""
    monkeypatch.setattr("slipstream.pusher.do_push", recorder)
    return BackgroundPusher(cfg, ["v8"])


def test_notify_triggers_drain(cfg, monkeypatch):
    calls = threading.Event()

    def fake_push(*args, **kwargs):
        calls.set()
        return 5

    pusher = _make_pusher(cfg, monkeypatch, fake_push)
    pusher.start()
    try:
        pusher.notify()
        assert calls.wait(timeout=5), "drain was not triggered by notify()"
    finally:
        pusher.close()


def test_close_drains_final_pending_work(cfg, monkeypatch):
    count = [0]

    def fake_push(*args, **kwargs):
        count[0] += 1
        return 0

    pusher = _make_pusher(cfg, monkeypatch, fake_push)
    pusher.start()
    pusher.notify()
    pusher.close()
    # At least one drain must have run, and the worker thread must be joined.
    assert count[0] >= 1
    assert not pusher._thread.is_alive()


def test_push_failure_does_not_kill_worker(cfg, monkeypatch):
    outcomes = []

    def flaky_push(*args, **kwargs):
        outcomes.append(len(outcomes))
        if len(outcomes) == 1:
            raise RuntimeError("target unreachable")
        return 1

    logs = []
    monkeypatch.setattr("slipstream.pusher.do_push", flaky_push)
    pusher = BackgroundPusher(cfg, ["v8"], log=logs.append)
    pusher.start()
    try:
        pusher.notify()
        # Give the first (failing) drain time to run, then a second must still
        # be serviced by the same live worker.
        deadline = time.monotonic() + 5
        while len(outcomes) < 2 and time.monotonic() < deadline:
            pusher.notify()
            time.sleep(0.02)
        assert len(outcomes) >= 2, "worker died after a push failure"
        assert any("failed" in m for m in logs)
    finally:
        pusher.close()


def test_secondary_connection_coexists_with_primary(cfg):
    """The pusher's secondary connection must not disturb the primary db."""
    db = cfg.metadata_dir / "slipstream.db"
    primary = CommitStore(db)
    secondary = CommitStore(db, backup=False, init_schema=False)
    try:
        primary.mark_done("v8", "arm64", 100)
        # Secondary sees the committed write and can write push_state without
        # tripping "created in a different thread" (same thread here) or locks.
        assert secondary.unpushed_commit_ids("v8", "arm64") == [100]
        secondary.mark_pushed("v8", "arm64", [100])
        assert primary.unpushed_commit_ids("v8", "arm64") == []
    finally:
        primary.close()
        secondary.close()


def test_no_backup_files_from_secondary(cfg):
    """Secondary connections must not snapshot the db (backup=False)."""
    db = cfg.metadata_dir / "slipstream.db"
    before = list(db.parent.glob("*.bak"))
    s = CommitStore(db, backup=False, init_schema=False)
    s.close()
    after = list(db.parent.glob("*.bak"))
    assert before == after
