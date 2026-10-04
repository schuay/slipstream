# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import signal
import subprocess
import sys
import threading
import time

import pytest

from slipstream.config import DeliveryConfig, PushTarget, RelaySource
from slipstream.delivery import Coordinator
from slipstream.delivery_batch import Batch, Receipt
from slipstream.delivery_sources import LocalDbSource, SshSpoolSource
from slipstream.delivery_targets import SpannerSession, SpoolSession, target_identity
from slipstream.durability import FileLock, target_lock_path
from slipstream.push import parse_seq, spool_append
from slipstream.relay import read_cursor
from slipstream.store import StoreError
from tests.test_push import VALID, _seed
from tests.test_spanner import ALL_TABLES, FakeDb, T0, _csv


class MemorySpool:
    def __init__(self, entries):
        self.entries = entries
        self.fetched = []

    def list(self, *, cursor=0, limit=17):
        seqs = sorted(self.entries)
        return sorted(set([s for s in seqs if s > cursor][:limit] + seqs[-1:]))

    def fetch(self, seq):
        self.fetched.append(seq)
        return self.entries[seq]


def local(store, engine="v8", bot="bot"):
    return LocalDbSource(store, engine, "arm64", bot, VALID)


def remote(tmp_path, bot, entries):
    cfg = RelaySource(bot, "/spool", bot, tmp_path / "relay")
    return SshSpoolSource(cfg, settings=DeliveryConfig(), spool=MemorySpool(entries))


def coordinator(tmp_path, sources, targets=None, **kwargs):
    if targets is None and len({s.bot for s in sources}) > 1:
        target = PushTarget(spanner="test/instance/db")

        class MultiBotTarget:
            def stage(self, batch, attempt):
                spool_append(
                    tmp_path / "output" / batch.bot, batch.csv(), 90, bot=batch.bot
                )
                return Receipt(target_identity(target), attempt, batch.digest)

            def refresh(self):
                pass

            def close(self):
                pass

        kwargs.setdefault("session_factory", lambda t, **kw: MultiBotTarget())
        targets = [target]
    return Coordinator(
        sources,
        targets or [PushTarget(spool_dir=tmp_path / "output")],
        tmp_path / "delivery",
        **kwargs,
    )


def cycle(c):
    with c.ownership():
        return c.cycle()


def files(path):
    return sorted(p for p in path.iterdir() if parse_seq(p.name))


def test_bounded_snapshot_rotates_between_sources_and_engines(tmp_path, store):
    _seed(store, list(range(1, 10)))
    a = local(store)
    b = remote(
        tmp_path, "remote", {i: _csv({"commit_id": str(i)}) for i in range(1, 10)}
    )
    c = coordinator(tmp_path, [a, b], settings=DeliveryConfig(max_units=3))
    assert cycle(c) == 3
    assert store.unpushed_commit_ids("v8", "arm64")[0] == 4
    assert read_cursor(b.path) == 0
    assert b.spool.fetched == []
    assert cycle(c) == 3
    assert read_cursor(b.path) == 3
    assert store.unpushed_commit_ids("v8", "arm64")[0] == 4


def test_new_work_waits_until_next_snapshot(tmp_path, store):
    _seed(store, [1])
    source = local(store)

    class Arrivals(SpoolSession):
        def stage(self, batch, attempt):
            receipt = super().stage(batch, attempt)
            _seed(store, [2])
            return receipt

    c = coordinator(tmp_path, [source], session_factory=lambda t, **kw: Arrivals(t))
    assert cycle(c) == 1
    assert store.unpushed_commit_ids("v8", "arm64") == [2]


@pytest.mark.parametrize("kind", ["local", "remote"])
def test_single_source_uses_cycle_unit_budget(tmp_path, store, kind):
    if kind == "local":
        _seed(store, list(range(1, 10)))
        source = local(store)
    else:
        source = remote(
            tmp_path, "remote", {i: _csv({"commit_id": str(i)}) for i in range(1, 10)}
        )
    c = coordinator(
        tmp_path,
        [source],
        settings=DeliveryConfig(max_units=8),
    )
    assert cycle(c) == 8
    assert all(source.acknowledged(unit) for unit in range(1, 9))
    assert not source.acknowledged(9)


@pytest.mark.parametrize("kind", ["local", "remote", "filtered"])
def test_service_drains_backlog_before_idle_poll(tmp_path, store, monkeypatch, kind):
    if kind == "remote":
        source = remote(
            tmp_path, "remote", {i: _csv({"commit_id": str(i)}) for i in range(1, 20)}
        )
    else:
        _seed(store, list(range(1, 20)))
        source = local(store)
        if kind == "filtered":
            source.valid = {"js3": {"other"}}
    stop = False
    events = []

    def log(message):
        nonlocal stop
        events.append(message)
        if message.startswith("next poll"):
            assert all(source.acknowledged(unit) for unit in range(1, 20))
            stop = True

    def sleep(seconds):
        pytest.fail("service slept before draining backlog")

    monkeypatch.setattr("slipstream.delivery.time.sleep", sleep)
    c = coordinator(
        tmp_path,
        [source],
        settings=DeliveryConfig(max_units=8, poll_seconds=600),
        log=log,
        should_stop=lambda: stop,
    )
    assert c.run()
    assert sum(m.startswith("cycle ") for m in events) == 4  # 8, 8, 3, idle
    assert not c.errors


def test_remote_soft_budget_does_not_skip_to_smaller_later_unit(tmp_path):
    source = remote(
        tmp_path,
        "remote",
        {1: _csv({}), 2: _csv({}, {"run": "2"}, {"run": "3"}), 3: _csv({})},
    )
    c = coordinator(tmp_path, [source], settings=DeliveryConfig(max_rows=3))
    assert cycle(c) == 1
    assert read_cursor(source.path) == 1
    assert not source.attempt_path.exists()
    assert not c.errors
    assert cycle(c) == 3
    assert read_cursor(source.path) == 2
    assert cycle(c) == 1
    assert read_cursor(source.path) == 3


def test_idle_service_uses_source_retry_deadline(tmp_path):
    source = remote(tmp_path, "remote", {1: "invalid"})
    events = []
    stop = False

    def log(message):
        nonlocal stop
        events.append(message)
        if message.startswith("next poll"):
            delay = float(message.removeprefix("next poll in ").removesuffix("s"))
            assert 0 < delay <= 5
            stop = True

    c = coordinator(
        tmp_path,
        [source],
        settings=DeliveryConfig(poll_seconds=600, retry_seconds=5),
        log=log,
        should_stop=lambda: stop,
    )
    assert c.run()
    assert c.errors
    assert any("next poll" in m for m in events)


def test_remote_failure_does_not_stop_healthy_local_source(tmp_path, store):
    _seed(store, [1])
    broken = remote(tmp_path, "remote", {1: "invalid"})
    c = coordinator(tmp_path, [broken, local(store)])
    assert cycle(c) == 1
    assert c.errors
    assert read_cursor(broken.path) == 0
    assert store.unpushed_commit_ids("v8", "arm64") == []
    assert cycle(c) == 0
    assert broken.spool.fetched == [1]  # source backoff, independent idle refresh


def test_ack_each_unit_after_every_target_and_release_before_refresh(tmp_path, store):
    _seed(store, [1, 2])
    source = local(store)
    sessions = []

    class Session(SpoolSession):
        def stage(self, batch, attempt):
            assert not store.conn.in_transaction
            assert not source.acknowledged(batch.unit)
            assert store._result_locks
            return super().stage(batch, attempt)

        def refresh(self):
            assert not store._result_locks
            assert source.acknowledged(1) and source.acknowledged(2)
            sessions.append(self.identity)

    targets = [
        PushTarget(spool_dir=tmp_path / "b"),
        PushTarget(spool_dir=tmp_path / "a"),
    ]
    c = coordinator(
        tmp_path, [source], targets, session_factory=lambda t, **kw: Session(t)
    )
    assert cycle(c) == 2
    assert sessions == sorted(target_identity(t) for t in targets)


@pytest.mark.parametrize(
    "boundary", ["partial", "receipt", "ack", "retire", "refresh", "cleanup"]
)
def test_failure_boundaries_preserve_durable_outcomes(
    tmp_path, store, monkeypatch, boundary
):
    _seed(store, [1])
    source = local(store)

    class Failure(SpoolSession):
        def stage(self, batch, attempt):
            receipt = super().stage(batch, attempt)
            if boundary == "partial":
                raise OSError("partial")
            if boundary == "receipt":
                return Receipt(self.identity, "wrong", batch.digest)
            return receipt

        def refresh(self):
            if boundary == "refresh":
                raise RuntimeError("refresh")

        def close(self):
            if boundary == "cleanup":
                raise RuntimeError("cleanup")

    if boundary in ("ack", "retire"):
        monkeypatch.setattr(
            source,
            "acknowledge" if boundary == "ack" else "retire",
            lambda *a: (_ for _ in ()).throw(OSError(boundary)),
        )
    c = coordinator(tmp_path, [source], session_factory=lambda t, **kw: Failure(t))
    cycle(c)
    assert c.errors
    assert not store._result_locks
    assert source.acknowledged(1) == (boundary in ("retire", "refresh", "cleanup"))
    assert bool(source.pending()) == (boundary not in ("refresh", "cleanup"))
    # Target lock is released even if cleanup fails.
    with FileLock(target_lock_path(c.identities[0]), timeout=0.1):
        pass


@pytest.mark.parametrize("crash", ["stage", "ack"])
def test_real_process_crash_and_persisted_restart(tmp_path, store, crash):
    _seed(store, [1])
    path = store.db_path
    target = tmp_path / "output"
    script = f"""
import os
from pathlib import Path
from slipstream.store import CommitStore
from slipstream.delivery import Coordinator
from slipstream.config import PushTarget
from slipstream.delivery_sources import LocalDbSource
from slipstream.delivery_targets import SpoolSession
store = CommitStore(Path({str(path)!r}), backup=False)
source = LocalDbSource(store, "v8", "arm64", "bot", {VALID!r})
class Crash(SpoolSession):
    def stage(self, batch, attempt):
        result = super().stage(batch, attempt)
        if {crash!r} == "stage": os._exit(19)
        return result
if {crash!r} == "ack": source.retire = lambda: os._exit(19)
c = Coordinator([source], [PushTarget(spool_dir=Path({str(target)!r}))], Path({str(tmp_path / "delivery")!r}), session_factory=lambda t, **kw: Crash(t))
with c.ownership(): c.cycle()
"""
    result = subprocess.run([sys.executable, "-c", script], timeout=10)
    assert result.returncode == 19
    source = local(store)
    record = source.pending()
    assert record
    if crash == "stage":
        with pytest.raises(StoreError, match="unresolved"):
            store.clear_range("v8", "arm64", [1])
        # Another commit remains collectible while this unit is protected.
        _seed(store, [2])
        store.clear_range("v8", "arm64", [2])
    c = coordinator(tmp_path, [source])
    assert cycle(c) == (1 if crash == "stage" else 0)
    assert source.pending() is None
    assert source.acknowledged(1)
    assert len(files(target)) == (2 if crash == "stage" else 1)


def test_retry_preserves_exact_attempt_and_blocks_changed_payload_or_targets(
    tmp_path, store
):
    _seed(store, [1])
    source = local(store)
    attempts = []

    class Broken(SpoolSession):
        def stage(self, batch, attempt):
            attempts.append(attempt)
            raise OSError("failed")

    c = coordinator(tmp_path, [source], session_factory=lambda t, **kw: Broken(t))
    cycle(c)
    original = source.pending()
    different = coordinator(
        tmp_path, [source], [PushTarget(spool_dir=tmp_path / "new")]
    )
    assert cycle(different) == 0
    assert "target set changed" in str(different.errors[0])
    store.conn.execute(
        "UPDATE scores SET score=4"
    )  # emulate externally corrupted source
    store.conn.commit()
    resumed = coordinator(tmp_path, [source])
    assert cycle(resumed) == 0
    assert "digest mismatch" in str(resumed.errors[0])
    assert source.pending() == original
    store.conn.execute("UPDATE scores SET score=1")
    store.conn.commit()
    assert cycle(coordinator(tmp_path, [source])) == 1
    assert source.pending() is None


def test_remote_restart_validates_payload_and_missing_file(tmp_path):
    source = remote(tmp_path, "remote", {1: _csv({})})

    class Broken(SpoolSession):
        def stage(self, *args):
            raise RuntimeError("partial target")

    cycle(coordinator(tmp_path, [source], session_factory=lambda t, **kw: Broken(t)))
    original = json.loads(source.attempt_path.read_text())
    source = remote(tmp_path, "remote", {})
    c = coordinator(tmp_path, [source])
    assert cycle(c) == 0
    assert c.errors and read_cursor(source.path) == 0
    source.spool.entries[1] = _csv({"score": "8"})
    c = coordinator(tmp_path, [source])
    assert cycle(c) == 0
    assert "digest mismatch" in str(c.errors[0])
    source.spool.entries[1] = _csv({})
    assert cycle(coordinator(tmp_path, [source])) == 1
    assert read_cursor(source.path) == 1
    assert not source.attempt_path.exists()
    assert original["unit"] == 1


def test_remote_gap_delivers_prefix_then_blocks(tmp_path):
    source = remote(tmp_path, "remote", {1: _csv({}), 3: _csv({})})
    assert cycle(coordinator(tmp_path, [source])) == 1
    c = coordinator(tmp_path, [source])
    assert cycle(c) == 0
    assert "gap: expected 2, found 3" in str(c.errors[0])
    assert read_cursor(source.path) == 1


def test_empty_filtered_units_are_explicitly_acknowledged(tmp_path, store):
    _seed(store, [1])
    source = LocalDbSource(store, "v8", "arm64", "bot", {"js3": {"other"}})
    logs = []
    assert cycle(coordinator(tmp_path, [source], log=logs.append)) == 0
    assert source.acknowledged(1)
    assert any("ack " in m and "rows=0" in m for m in logs)
    assert not (tmp_path / "output").exists()


def test_oversized_valid_unit_is_not_starved_and_hard_bound_is_visible(tmp_path):
    source = remote(tmp_path, "remote", {1: _csv({}, {"run": "2"}), 2: _csv({})})
    c = coordinator(
        tmp_path, [source], settings=DeliveryConfig(max_rows=1, max_bytes=1)
    )
    assert cycle(c) == 2
    assert read_cursor(source.path) == 1
    source = remote(tmp_path, "remote", {1: _csv({}, {"run": "2"}), 2: _csv({})})
    c = coordinator(
        tmp_path, [source], settings=DeliveryConfig(max_bytes=1, max_payload_bytes=10)
    )
    assert cycle(c) == 0
    assert "max_payload_bytes" in str(c.errors[0])
    assert read_cursor(source.path) == 1


def test_spool_sequence_survives_empty_retention_and_reservation_crash(
    tmp_path, monkeypatch
):
    spool = tmp_path / "spool"
    assert spool_append(spool, "a", 90) == 1
    files(spool)[0].unlink()
    assert spool_append(spool, "b", 90) == 2
    from slipstream import durability

    real_write = durability.atomic_write

    def fail_publication(path, data):
        if path.suffix == ".csv":
            raise OSError("crash before publication")
        real_write(path, data)

    monkeypatch.setattr(durability, "atomic_write", fail_publication)
    with pytest.raises(OSError):
        spool_append(spool, "c", 90)
    monkeypatch.setattr(durability, "atomic_write", real_write)
    assert spool_append(spool, "d", 90) == 4
    assert [p.name for p in files(spool)] == ["00000002.csv", "00000004.csv"]


def test_second_process_cannot_take_owner_or_commit_lock(tmp_path, store):
    source = local(store)
    c = coordinator(tmp_path, [source])
    script = f"""
from pathlib import Path
from slipstream.store import CommitStore
from slipstream.delivery import Coordinator
from slipstream.delivery_sources import LocalDbSource
from slipstream.config import PushTarget, DeliveryConfig
s = CommitStore(Path({str(store.db_path)!r}), backup=False)
source = LocalDbSource(s, "v8", "arm64", "bot", {VALID!r})
try:
    c = Coordinator([source], [PushTarget(spool_dir=Path({str(tmp_path / "output")!r}))], Path({str(tmp_path / "delivery")!r}), settings=DeliveryConfig(lock_seconds=0.1))
    with c.ownership(): pass
except TimeoutError: pass
else: raise RuntimeError("duplicate owner")
try:
    with s.result_locks("v8", "arm64", [1], timeout=0.1): pass
except TimeoutError: pass
else: raise RuntimeError("same commit raced")
with s.result_locks("v8", "arm64", [2], timeout=0.1): pass
"""
    with c.ownership(), store.result_locks("v8", "arm64", [1]):
        assert subprocess.run([sys.executable, "-c", script], timeout=5).returncode == 0


def test_lock_wait_and_idle_poll_cancel_promptly(tmp_path):
    stop = threading.Event()
    held = FileLock(tmp_path / "lock")
    with held:
        threading.Timer(0.1, stop.set).start()
        start = time.monotonic()
        with pytest.raises(InterruptedError):
            with FileLock(held.path, timeout=30, should_stop=stop.is_set):
                pass
        assert time.monotonic() - start < 1
    stop.clear()
    c = coordinator(
        tmp_path, [], should_stop=stop.is_set, settings=DeliveryConfig(poll_seconds=30)
    )
    threading.Timer(0.1, stop.set).start()
    start = time.monotonic()
    assert c.run()
    assert time.monotonic() - start < 1


def test_sigterm_during_actual_service_poll(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f'out_dir = "{tmp_path}"\n[push]\nbot_name = "bot"\n'
        f'[[push.targets]]\nspool_dir = "{tmp_path}/output"\n'
        "[delivery]\nlocal = false\nremote = false\npoll_seconds = 30\n"
    )
    output = tmp_path / "log"
    with output.open("w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "slipstream", "deliver", "--config", str(cfg)],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 5
            while "next poll" not in output.read_text() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert "next poll" in output.read_text(), output.read_text()
            start = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            assert proc.wait(timeout=2) == 0
            assert time.monotonic() - start < 1
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class MarkerDb(FakeDb):
    def __init__(self):
        super().__init__()
        self.markers = {}
        self.partial = False
        self.watermark = False
        self.rows = []

    def query(self, sql, params=None):
        if "INFORMATION_SCHEMA.INDEXES" in sql:
            from slipstream import spanner

            return [(name, "READ_WRITE") for name in spanner.declared_indexes()]
        if "INFORMATION_SCHEMA" in sql:
            return ALL_TABLES
        if "STARTS_WITH(key" in sql:
            return [(k,) for k in self.markers][:1]
        if "MAX(imported_at)" in sql:
            return [(T0,)]
        if "CURRENT_TIMESTAMP" in sql:
            return [(T0,)]
        if "SELECT value FROM meta" in sql:
            if params and params[0] in self.markers:
                return [(self.markers[params[0]],)]
            return [(T0.isoformat(),)] if self.watermark else []
        return []

    def execute(self, sql, params=None):
        super().execute(sql, params)
        if "INSERT OR UPDATE INTO meta" in sql:
            if params[0].startswith("slipstream_incomplete_import:"):
                self.markers[params[0]] = params[1]
            else:
                self.watermark = True
        elif "DELETE FROM meta" in sql:
            self.markers.pop(params[0], None)

    def upsert(self, table, columns, rows):
        super().upsert(table, columns, rows)
        if self.partial:
            raise RuntimeError("partial write")


def test_same_bot_different_payload_cannot_clear_exact_partial_marker(monkeypatch):
    db = MarkerDb()
    monkeypatch.setattr("slipstream.spanner.connect", lambda spec, **kw: db)
    session = SpannerSession(PushTarget(spanner="p/i/d"))
    original = Batch.remote("remote", "bot", 1, _csv({}, {"run": "2"}))
    db.partial = True
    with pytest.raises(RuntimeError):
        session.stage(original, "original")
    db.partial = False
    session.stage(Batch.remote("remote", "bot", 2, _csv({"run": "3"})), "unrelated")
    assert set(db.markers) == {"slipstream_incomplete_import:attempt:original"}
    assert "incomplete" in session.refresh()
    assert not db.watermark
    session.stage(original, "original")
    assert not db.markers
    assert session.refresh() is None
    assert db.watermark
    calls = len(db.calls)
    session.close()
    assert db.calls[calls:] == [("close",)]


def test_legacy_markers_require_explicit_reconciliation(monkeypatch):
    db = MarkerDb()
    db.markers["slipstream_incomplete_import:bot"] = "staging"
    monkeypatch.setattr("slipstream.spanner.connect", lambda spec, **kw: db)
    session = SpannerSession(PushTarget(spanner="p/i/d"))
    session.stage(Batch.remote("remote", "bot", 1, _csv({})), "new")
    assert "incomplete" in session.refresh()
    session.reconcile_legacy("bot")
    assert session.refresh() is None


def test_refresh_failed_after_ack_is_retried_without_upload_on_idle_cycle(
    tmp_path, store
):
    _seed(store, [1])
    source = local(store)
    stages, refreshes = [], []

    class Refresh(SpoolSession):
        def stage(self, batch, attempt):
            stages.append(batch.unit)
            return super().stage(batch, attempt)

        def refresh(self):
            refreshes.append(1)
            if len(refreshes) == 1:
                raise RuntimeError("aggregation failed")

    c = coordinator(tmp_path, [source], session_factory=lambda t, **kw: Refresh(t))
    assert cycle(c) == 1
    assert c.errors and source.acknowledged(1)
    assert cycle(c) == 0
    assert not c.errors
    assert stages == [1] and len(refreshes) == 2


def test_one_session_accepts_multiple_bot_identities_and_refreshes_once(
    tmp_path, store, monkeypatch
):
    _seed(store, [1])
    db = MarkerDb()
    monkeypatch.setattr("slipstream.spanner.connect", lambda spec, **kw: db)
    opened = []

    def factory(target, **kwargs):
        opened.append(target)
        return SpannerSession(target)

    c = coordinator(
        tmp_path,
        [local(store), remote(tmp_path, "other", {1: _csv({})})],
        [PushTarget(spanner="p/i/d")],
        session_factory=factory,
    )
    assert cycle(c) == 2
    assert len(opened) == 1
    assert {call[3][0][0] for call in db.of("upsert", "slipstream")} == {"bot", "other"}
    assert {call[3][0][0] for call in db.of("upsert", "samples")} == {"bot", "other"}
    assert db.calls[-1] == ("close",)


def stalled_worker(pipe, target):
    """Real spawned transport that hangs after accepting a request."""
    pipe.send(("ok", None))
    pipe.recv()
    time.sleep(30)


@pytest.mark.parametrize("stop", [False, True])
def test_spanner_worker_deadline_and_shutdown_reap_real_process(
    tmp_path, monkeypatch, stop
):
    from slipstream import delivery_targets

    monkeypatch.setattr(delivery_targets, "_spanner_worker", stalled_worker)
    flag = threading.Event()
    settings = DeliveryConfig(io_seconds=2, shutdown_seconds=0.1)
    session = delivery_targets.BoundedSpannerSession(
        PushTarget(spanner="p/i/d"), settings=settings, should_stop=flag.is_set
    )
    session.settings.io_seconds = 0.2 if not stop else 5
    if stop:
        threading.Timer(0.1, flag.set).start()
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        session.stage(Batch.remote("remote", "bot", 1, _csv({})), "attempt")
    assert time.monotonic() - start < 1
    assert not session.process.is_alive()
    session.close()


def test_cli_has_delivery_and_removed_relay():
    from typer.testing import CliRunner
    from slipstream.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["deliver", "--help"]).exit_code == 0
    assert runner.invoke(app, ["relay", "--help"]).exit_code != 0


def test_interrupted_rebuild_blocks_another_state_directory_and_resumes(
    tmp_path, store
):
    from slipstream.delivery_maintenance import rebuild_local, target_guard

    _seed(store, [1])
    store.mark_pushed("v8", "arm64", [1])
    target = PushTarget(spanner="p/i/d")
    fail = [True]
    wipes = []

    class Maintenance:
        def wipe(self, bot):
            wipes.append(bot)
            if fail[0]:
                raise RuntimeError("partial wipe")
            return 1

        def close(self):
            pass

    c = coordinator(
        tmp_path,
        [local(store)],
        [target],
        session_factory=lambda *a, **kw: Maintenance(),
    )
    with c.ownership():
        with pytest.raises(RuntimeError, match="partial wipe"):
            rebuild_local(c, store, ["v8"], "arm64", "bot")
    assert (tmp_path / "delivery" / "maintenance.json").exists()
    assert target_guard(c.identities[0]).exists()
    other = coordinator(
        tmp_path / "other",
        [local(store)],
        [target],
        session_factory=lambda *a, **kw: pytest.fail("opened guarded target"),
    )
    assert cycle(other) == 0
    assert "interrupted target maintenance" in str(other.errors[0])
    fail[0] = False
    with c.ownership():
        rebuild_local(c, store, ["v8"], "arm64", "bot")
    assert not target_guard(c.identities[0]).exists()
    assert not (tmp_path / "delivery" / "maintenance.json").exists()
    assert wipes == ["bot", "bot"]
    assert store.unpushed_commit_ids("v8", "arm64") == [1]


def test_new_targets_do_not_implicitly_replay_acknowledged_history(tmp_path, store):
    _seed(store, [1])
    source = local(store)
    assert cycle(coordinator(tmp_path, [source])) == 1
    new = tmp_path / "new"
    assert cycle(coordinator(tmp_path, [source], [PushTarget(spool_dir=new)])) == 0
    assert not new.exists()


def test_local_payload_is_rechecked_after_discovery_under_commit_lock(tmp_path, store):
    _seed(store, [1])
    source = local(store)
    discover = source.discover

    def clearing(limit):
        units = discover(limit)
        store.clear_range("v8", "arm64", [1])
        return units

    source.discover = clearing
    assert cycle(coordinator(tmp_path, [source])) == 0
    assert not source.acknowledged(1)
    assert source.pending() is None
    assert not (tmp_path / "output").exists()


def test_malformed_identity_is_rejected_even_for_spool_targets(tmp_path):
    for field, value in (
        ("git_hash", ""),
        ("commit_timestamp", ""),
        ("score", "nan"),
        ("engine", ""),
        ("run", "0"),
    ):
        source = remote(tmp_path, "remote", {1: _csv({field: value})})
        c = coordinator(tmp_path, [source])
        assert cycle(c) == 0
        assert c.errors and not source.attempt_path.exists()
        assert read_cursor(source.path) == 0


def test_rebuild_deletes_obsolete_aggregate_keys_explicitly():
    from slipstream import spanner

    db = FakeDb()
    spanner.rebuild_bot(db, "bot")
    assert any("DELETE FROM benchmarks" in call[1] for call in db.of("pdml"))
    assert any("source = 'slipstream'" in call[1] for call in db.of("pdml"))


def test_empty_legacy_spool_does_not_silently_reset_sequence(tmp_path):
    path = tmp_path / "spool"
    path.mkdir()
    (path / ".lock").touch()
    with pytest.raises(ValueError, match="empty legacy spool"):
        spool_append(path, "x", 90)
    assert not (path / "00000001.csv").exists()


def test_local_bot_legacy_reconciliation_without_remote_source(tmp_path, monkeypatch):
    from slipstream.config import Config, PushConfig
    from slipstream.delivery_maintenance import maintain_remote

    db = MarkerDb()
    db.markers["slipstream_incomplete_import:bot"] = "staging"
    monkeypatch.setattr("slipstream.spanner.connect", lambda spec, **kw: db)
    target = PushTarget(spanner="p/i/d")
    cfg = Config(tmp_path, "results", {}, {}, push=PushConfig("bot", [target]))
    c = coordinator(
        tmp_path, [], [target], session_factory=lambda t, **kw: SpannerSession(t)
    )
    with c.ownership():
        maintain_remote(c, cfg, legacy="bot")
    assert not db.markers


def test_corrupt_attempt_is_not_discarded_even_if_unit_was_acknowledged(
    tmp_path, store
):
    _seed(store, [1])
    source = local(store)
    source.save(
        dict(
            version=1,
            source=source.identity,
            bot="bot",
            unit=1,
            targets=[target_identity(PushTarget(spool_dir=tmp_path / "output"))],
            attempt="a" * 32,
            digest="corrupted",
        )
    )
    store.mark_pushed("v8", "arm64", [1])
    c = coordinator(tmp_path, [source])
    assert cycle(c) == 0
    assert c.errors and "corrupt" in str(c.errors[0])
    assert source.pending()


def test_shared_logging_has_one_plain_sink_with_bounded_retention(tmp_path, capsys):
    from slipstream.service_logging import EventLog

    path = tmp_path / "logs" / "deliver.log"
    events = EventLog("deliver", path)
    events("task start source=bot unit=1")
    events("target refresh failed; retry")
    events.error("task failed")
    assert events.handler.maxBytes == 5 * 1024 * 1024
    assert events.handler.backupCount == 3
    events.close()
    text = path.read_text()
    assert "INFO slipstream.deliver" in text
    assert "WARNING slipstream.deliver" in text
    assert "ERROR slipstream.deliver" in text
    assert "\x1b" not in text and not capsys.readouterr().err


def test_spool_cannot_mix_bot_identities(tmp_path, store):
    target = PushTarget(spool_dir=tmp_path / "spool")
    with pytest.raises(ValueError, match="one bot identity"):
        coordinator(tmp_path, [local(store), remote(tmp_path, "other", {})], [target])
    session = SpoolSession(target)
    session.stage(Batch.remote("a", "bot-a", 1, _csv({})), "first")
    with pytest.raises(ValueError, match="bound to bot"):
        session.stage(Batch.remote("b", "bot-b", 1, _csv({})), "second")
    assert len(files(target.spool_dir)) == 1
    assert (target.spool_dir / ".bot").read_text() == "bot-a"


def test_empty_remote_reset_is_not_silently_treated_as_caught_up(tmp_path):
    source = remote(tmp_path, "remote", {})
    from slipstream.relay import write_cursor

    write_cursor(source.path, 12)
    c = coordinator(tmp_path, [source])
    assert cycle(c) == 0
    assert "spool reset or missing allocator" in str(c.errors[0])
    assert read_cursor(source.path) == 12


def test_watch_has_no_delivery_side_effects(config, store, monkeypatch):
    from types import SimpleNamespace
    from typer.testing import CliRunner
    from slipstream.cli import app
    from slipstream.config import EngineConfig

    _seed(store, [1])
    config.engines["v8"] = EngineConfig("v8", None, "", "", "")
    monkeypatch.setattr("slipstream.cli._load_config", lambda path: config)
    monkeypatch.setattr("slipstream.cli._host_preflight", lambda cfg: None)
    collector = SimpleNamespace(store=store, find_frontier=lambda name: (1, 1))
    monkeypatch.setattr("slipstream.cli._open_collector", lambda *a, **kw: collector)
    monkeypatch.setattr(
        "slipstream.delivery_targets.open_session",
        lambda *a, **kw: pytest.fail("watch opened a delivery target"),
    )
    result = CliRunner().invoke(app, ["watch", "v8", "--once"])
    assert result.exit_code == 0, result.output
    # Watch closes its collector store; durable progress remains unacknowledged.
    from slipstream.store import CommitStore

    reopened = CommitStore(store.db_path, backup=False)
    try:
        assert reopened.unpushed_commit_ids("v8", "arm64") == [1]
    finally:
        reopened.close()


def test_separate_processes_with_opposite_target_order_do_not_deadlock(tmp_path):
    scripts = []
    for index in range(2):
        targets = [str(tmp_path / "a"), str(tmp_path / "b")][:: 1 if index == 0 else -1]
        scripts.append(f"""
import time
from pathlib import Path
from slipstream.store import CommitStore
from slipstream.delivery import Coordinator
from slipstream.delivery_sources import LocalDbSource
from slipstream.delivery_targets import SpoolSession
from slipstream.config import PushTarget
from tests.test_push import _seed, VALID
store = CommitStore(Path({str(tmp_path / str(index) / "db")!r}), backup=False)
_seed(store, [1])
class Slow(SpoolSession):
    def stage(self, *args):
        time.sleep(0.05)
        return super().stage(*args)
c = Coordinator([LocalDbSource(store, "v8", "arm64", "bot", VALID)], [PushTarget(spool_dir=Path(p)) for p in {targets!r}], Path({str(tmp_path / str(index) / "delivery")!r}), session_factory=lambda t, **kw: Slow(t))
assert c.run(once=True)
assert store.unpushed_commit_ids("v8", "arm64") == []
store.close()
""")
    children = [subprocess.Popen([sys.executable, "-c", s]) for s in scripts]
    try:
        assert [p.wait(timeout=10) for p in children] == [0, 0]
    finally:
        for p in children:
            if p.poll() is None:
                p.kill()
                p.wait()
    assert len(files(tmp_path / "a")) == len(files(tmp_path / "b")) == 2
