# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from slipstream import bench_window, cli
from slipstream.admission import CollectionResult, StopReason
from slipstream.collector import BenchCollector
from slipstream.config import BusConfig, BusSource, EngineConfig


@pytest.fixture
def window_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        f'out_dir = "{tmp_path}/out"\n[bench]\ntimezone = "Europe/Berlin"\nallowed_windows = []\n'
    )
    return path


def invoke(path, *args):
    return CliRunner().invoke(cli.app, [*args, "--config", str(path)])


def test_controls_and_status_without_daemon_bus_or_db(window_config, tmp_path):
    result = invoke(window_config, "window", "status")
    assert result.exit_code == 0, result.output
    assert "Normal use permitted" in result.output
    assert f"for {window_config}" in result.output
    assert "Restart affected benchers" in result.output
    assert bench_window.read_override() is None
    assert list(bench_window.override_path().parent.iterdir()) == []
    result = invoke(window_config, "window", "open", "--for", "6h")
    assert result.exit_code == 0 and "Benchmarking permitted" in result.output
    assert "expires 20" in result.output
    result = invoke(window_config, "window", "close")
    assert result.exit_code == 0 and "until reset or replaced" in result.output
    result = invoke(window_config, "window", "close", "--for", "2h")
    assert result.exit_code == 0 and bench_window.read_override().expires_at
    result = invoke(window_config, "window", "reset")
    assert result.exit_code == 0 and bench_window.read_override() is None
    result = invoke(window_config, "bus", "status")
    assert result.exit_code == 0 and "Normal use permitted" in result.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("open",),
        ("open", "--for", "0h"),
        ("close", "--for", "0s"),
        ("open", "--for", "-1h"),
    ],
)
def test_invalid_duration(window_config, args):
    assert invoke(window_config, "window", *args).exit_code != 0
    assert bench_window.read_override() is None


def test_corrupt_override_is_visible_and_reset_repairs(window_config):
    path = bench_window.override_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{")
    result = invoke(window_config, "window", "status")
    assert result.exit_code == 0 and "invalid override" in result.output
    assert "error:" in result.output
    assert invoke(window_config, "window", "reset").exit_code == 0
    assert bench_window.read_override() is None


@pytest.fixture
def watch_env(config, tmp_path, monkeypatch):
    config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=tmp_path,
        build_cmd="true",
        binary_path="d8",
        id_regex=r"#([0-9]+)",
    )
    config.bus = BusConfig(
        root=tmp_path / "bus",
        sources=[BusSource("local", str(tmp_path / "bus"), engines=["v8"])],
    )
    collector = BenchCollector(config, wait_for_lock=True)
    monkeypatch.setattr(cli, "_load_config", lambda path: config)
    monkeypatch.setattr(cli, "_open_collector", lambda *a, **kw: collector)
    return config, collector


@pytest.mark.parametrize("dry_run", [False, True])
def test_bounded_commands_defer_without_destructive_work(
    watch_env, monkeypatch, dry_run
):
    config, collector = watch_env
    config.bench.timezone = "UTC"
    config.bench.allowed_windows = []
    monkeypatch.setattr(cli, "_shutdown_flag", lambda: lambda: False)
    collector.store.upsert_commit("v8", "abc", 100, "2026-01-01", 1, "test")
    collector.store.mark_done("v8", config.platform, 100)
    monkeypatch.setattr(
        collector, "get_commit_list", lambda *a, **kw: [{"hash": "abc"}]
    )
    monkeypatch.setattr(collector, "populate_commit_metadata", lambda *a: None)
    monkeypatch.setattr(
        collector, "_provision", lambda *a: pytest.fail("provisioned while closed")
    )
    args = ["bench", "--once"]
    if dry_run:
        args += ["--dry-run"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (0 if dry_run else 75), result.output
    assert "closed" in result.output
    assert "up to date" not in result.output
    assert not collector.lock.path.exists()


@pytest.mark.parametrize("notification_while_closed", [False, True])
@pytest.mark.parametrize("defer_during_drain", [False, True])
def test_watch_reopens_on_expiry_without_new_publication(
    watch_env, monkeypatch, tmp_path, notification_while_closed, defer_during_drain
):
    _, collector = watch_env
    clock = [datetime(2026, 10, 19, 0, 0, tzinfo=UTC)]
    current = bench_window.current
    monkeypatch.setattr(
        bench_window, "current", lambda config: current(config, now=clock[0])
    )
    stopped = [False]
    monkeypatch.setattr(cli, "_shutdown_flag", lambda: lambda: stopped[0])
    schedules, dispatch = [], []
    events = tmp_path / "events"
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for name in ("pre-window", "post-window"):
        hook = hooks / name
        hook.write_text(f"#!/bin/sh\necho {name} >> events\n")
        hook.chmod(0o755)

    def subscribe(sources, schedule, log):
        schedules.append(schedule)
        schedule.changed(["v8"])
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr("slipstream.notifications.Subscriptions", subscribe)

    def drain(self, source, name, should_stop, **kwargs):
        with events.open("a") as f:
            f.write("job\n")
        dispatch.append(clock[0])
        if len(dispatch) == 1:
            bench_window.write_override("close", 1, now=clock[0])
            return CollectionResult(
                reason=StopReason.DEFERRED
                if defer_during_drain
                else StopReason.COMPLETE
            )
        stopped[0] = True
        return CollectionResult(1)

    monkeypatch.setattr("slipstream.consumer.BusConsumer.drain", drain)

    def sleep(seconds):
        assert len(dispatch) == 1, "dispatched while closed"
        if defer_during_drain:
            assert "v8" in schedules[0].pending
            assert "v8" not in schedules[0].deadlines
        if notification_while_closed:
            schedules[0].changed(["v8"])
        clock[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(cli.time_mod, "sleep", sleep)
    result = invoke(tmp_path / "config.toml", "bench")
    assert result.exit_code == 0, result.output
    assert dispatch == [
        datetime(2026, 10, 19, tzinfo=UTC),
        datetime(2026, 10, 19, 0, 0, 1, tzinfo=UTC),
    ]
    assert result.output.count("Benchmark window: forced close") == 1
    assert events.read_text().splitlines() == [
        "pre-window",
        "job",
        "post-window",
        "pre-window",
        "job",
    ]


def test_open_does_not_resume_bus_pause(window_config):
    from slipstream.lock import pause, paused_until

    pause(60)
    result = invoke(window_config, "window", "open", "--for", "1h")
    assert result.exit_code == 0
    assert (
        "Benchmarking permitted" in result.output
        and "PAUSED by operator" in result.output
    )
    assert paused_until() is not None


def test_partial_bounded_deferral_retains_completed_work(watch_env, monkeypatch):
    config, collector = watch_env
    monkeypatch.setattr(cli, "_shutdown_flag", lambda: lambda: False)
    counts = []

    def drain(self, source, name, should_stop, **kwargs):
        if counts:
            return CollectionResult(reason=StopReason.DEFERRED)
        counts.append(100)
        collector.store.mark_done("v8", config.platform, 100)
        bench_window.write_override("close")
        return CollectionResult(1)

    monkeypatch.setattr("slipstream.consumer.BusConsumer.drain", drain)
    args = ["bench", "--once"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 75, result.output
    assert counts == [100]
    # The command closed its connection. Read the persisted completion anew.
    from slipstream.store import CommitStore

    with_store = CommitStore(config.metadata_dir / "slipstream.db", backup=False)
    try:
        assert with_store.is_done("v8", config.platform, 100)
        assert not with_store.is_done("v8", config.platform, 101)
    finally:
        with_store.close()


def test_concurrent_cli_mutations_are_serialized(window_config, tmp_path):
    import subprocess
    import sys

    from slipstream.bench_window import Override

    locks = tmp_path / "concurrent-locks"
    script = """
import sys
from pathlib import Path
from typer.testing import CliRunner
from slipstream import cli, lock
lock.LOCKS_DIR = Path(sys.argv[1])
lock.MACHINE_LOCK = lock.LOCKS_DIR / 'machine.lock'
lock.PAUSE_FILE = lock.LOCKS_DIR / 'pause'
result = CliRunner().invoke(cli.app, ['window', *sys.argv[3:], '--config', sys.argv[2]])
assert result.exit_code == 0, (result.output, result.exception)
"""
    commands = [["open", "--for", "1h"], ["close", "--for", "2h"], ["reset"], ["close"]]
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(locks), str(window_config), *args]
        )
        for args in commands
    ]
    try:
        assert [process.wait(timeout=10) for process in processes] == [0] * 4
        assert isinstance(
            bench_window.read_override(locks / "bench-window.json"),
            (Override, type(None)),
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_timed_close_on_empty_schedule_does_not_claim_reset_will_open(window_config):
    result = invoke(window_config, "window", "close", "--for", "2h")
    assert result.exit_code == 0, result.output
    assert "Normal use permitted (no scheduled opening)" in result.output
    assert "expires 20" in result.output


@pytest.mark.parametrize("close_after_job", [False, True])
def test_watch_hooks_once_per_transition_after_last_job(
    watch_env, tmp_path, monkeypatch, close_after_job
):
    _, collector = watch_env
    events = []
    from slipstream.window_hooks import WindowHooks

    original_update = WindowHooks.update

    def update(self, allowed):
        if allowed != self.open:
            events.append("open" if allowed else "close")
        original_update(self, allowed)

    monkeypatch.setattr(WindowHooks, "update", update)
    monkeypatch.setattr(cli, "_shutdown_flag", lambda: lambda: False)

    def drain(self, *args, **kwargs):
        events.append("job")
        if events.count("job") == 2:
            if close_after_job:
                bench_window.write_override("close")
            return CollectionResult()
        return CollectionResult(1)

    monkeypatch.setattr("slipstream.consumer.BusConsumer.drain", drain)
    result = invoke(tmp_path / "config.toml", "bench", "--once")
    assert result.exit_code == 0, result.output
    assert events == ["open", "job", "job"] + (["close"] if close_after_job else [])


@pytest.mark.parametrize("dry_run", [False, True])
def test_pre_window_failure_prevents_jobs(watch_env, tmp_path, monkeypatch, dry_run):
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-window"
    hook.write_text("#!/bin/sh\nexit 42\n")
    hook.chmod(0o755)
    monkeypatch.setattr(cli, "_shutdown_flag", lambda: lambda: False)
    calls = []

    def drain(self, *args, **kwargs):
        calls.append("job")
        return CollectionResult()

    monkeypatch.setattr("slipstream.consumer.BusConsumer.drain", drain)
    args = ["bench", "--once"] + (["--dry-run"] if dry_run else [])
    result = invoke(tmp_path / "config.toml", *args)
    assert result.exit_code == (0 if dry_run else 1), result.output
    assert calls == (["job"] if dry_run else [])
    if not dry_run:
        assert "pre-window" in result.output and "42" in result.output
