# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import subprocess

import pytest

from slipstream.hooks import HookError, Hooks
from slipstream import power_hooks


@pytest.mark.parametrize("failure", [None, "pre", "job", "post"])
def test_paired_hooks_restore_on_failures(tmp_path, failure):
    directory = tmp_path / "hooks"
    directory.mkdir()
    events = tmp_path / "events"
    for phase in ("pre", "post"):
        hook = directory / f"{phase}-build"
        hook.write_text(
            f'#!/bin/sh\necho "{phase} $SLIPSTREAM_HOOK_STATE_DIR" >> events\n'
            + ("exit 42\n" if failure == phase else "")
        )
        hook.chmod(0o755)
    hooks = Hooks(tmp_path / "config.toml")

    def job():
        with hooks.job("build", tmp_path / "state", engine="v8", key=123):
            with events.open("a") as f:
                f.write("job\n")
            # Internal compilation must not nest another pair inside the
            # builder's pair that also covers packaging and publication.
            with hooks.job("build", tmp_path / "state", engine="v8", key=123):
                if failure == "job":
                    raise ValueError("job failed")

    if failure:
        with pytest.raises(ValueError if failure == "job" else HookError):
            job()
    else:
        job()
    lines = events.read_text().splitlines()
    assert [line.split()[0] for line in lines] == (
        ["pre", "post"] if failure == "pre" else ["pre", "job", "post"]
    )
    assert lines[0].split()[1] == lines[-1].split()[1]
    assert bool(list((tmp_path / "state").iterdir())) == (failure == "post")


@pytest.mark.parametrize(
    "mode,key,initial,target",
    [
        ("low", "lowpowermode", "0", "1"),
        ("low", "powermode", "2", "1"),
        ("performance", "lowpowermode", "1", "0"),
        ("performance", "powermode", "1", "2"),
    ],
)
def test_power_modes_and_exact_restoration(
    tmp_path, monkeypatch, mode, key, initial, target
):
    state = {key: initial}
    monkeypatch.setattr(power_hooks, "is_macos", lambda: True)
    monkeypatch.setattr(power_hooks, "read_power", lambda: dict(state))
    monkeypatch.setattr(power_hooks, "write_power", lambda k, v: state.update({k: v}))
    power_hooks.apply(mode, tmp_path)
    assert state == {key: target}
    assert (tmp_path / "power.json").stat().st_mode & 0o777 == 0o600
    power_hooks.apply("restore", tmp_path)
    assert state == {key: initial}
    assert not (tmp_path / "power.json").exists()


@pytest.mark.parametrize("reject", [False, True])
def test_unavailable_high_power_falls_back_to_automatic(tmp_path, monkeypatch, reject):
    state = {"powermode": "1"}
    monkeypatch.setattr(power_hooks, "is_macos", lambda: True)
    monkeypatch.setattr(power_hooks, "read_power", lambda: dict(state))

    def write(key, value):
        if value == "2":
            if reject:
                raise subprocess.CalledProcessError(1, "pmset")
            return
        state[key] = value

    monkeypatch.setattr(power_hooks, "write_power", write)
    power_hooks.apply("performance", tmp_path)
    assert state == {"powermode": "0"}
    power_hooks.apply("restore", tmp_path)
    assert state == {"powermode": "1"}


def test_failed_power_write_retains_snapshot_for_post_hook(tmp_path, monkeypatch):
    state = {"lowpowermode": "0"}
    monkeypatch.setattr(power_hooks, "is_macos", lambda: True)
    monkeypatch.setattr(power_hooks, "read_power", lambda: dict(state))

    def fail(k, v):
        state[k] = v
        raise subprocess.CalledProcessError(1, "pmset")

    monkeypatch.setattr(power_hooks, "write_power", fail)
    with pytest.raises(subprocess.CalledProcessError):
        power_hooks.apply("low", tmp_path)
    assert (tmp_path / "power.json").exists()
    monkeypatch.setattr(power_hooks, "write_power", lambda k, v: state.update({k: v}))
    power_hooks.apply("restore", tmp_path)
    assert state == {"lowpowermode": "0"}


def test_power_write_ignores_cached_sudo_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(
        power_hooks.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw))
    )
    power_hooks.write_power("powermode", "2")
    assert calls == [
        (
            ["/usr/bin/sudo", "-n", "-k", "/usr/bin/pmset", "-c", "powermode", "2"],
            {"check": True},
        )
    ]


def test_installer_limits_sudo_to_exact_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(power_hooks, "is_macos", lambda: True)
    monkeypatch.setattr(power_hooks, "read_power", lambda: {"lowpowermode": "0"})
    checks = []
    monkeypatch.setattr(
        power_hooks.subprocess, "run", lambda cmd, **kw: checks.append(cmd)
    )
    power_hooks.install(tmp_path / "config.toml")
    directory = tmp_path / "hooks"
    rule = (directory / "pmset.sudoers").read_text()
    assert rule.split("NOPASSWD: ")[1].strip() == (
        "/usr/bin/pmset -c lowpowermode 0, /usr/bin/pmset -c lowpowermode 1"
    )
    assert checks == [
        ["/usr/sbin/visudo", "-c", "-f", str(directory / "pmset.sudoers")]
    ]
    for name, mode in [
        ("pre-build", "low"),
        ("post-build", "restore"),
        ("pre-bench", "performance"),
        ("post-bench", "restore"),
    ]:
        path = directory / name
        assert path.stat().st_mode & 0o111
        assert f"slipstream.power_hooks {mode}" in path.read_text()
    (directory / "pre-build").write_text("custom\n")
    with pytest.raises(RuntimeError, match="existing hook differs"):
        power_hooks.install(tmp_path / "config.toml")
    assert (directory / "pre-build").read_text() == "custom\n"


def test_pre_bench_precedes_preflight_and_post_runs_on_abort(
    config, tmp_path, monkeypatch
):
    from slipstream import host
    from slipstream.collector import BenchCollector
    from slipstream.config import EngineConfig
    from keys import K1

    config.config_path = tmp_path / "config.toml"
    config.host.preflight = "abort"
    directory = tmp_path / "hooks"
    directory.mkdir()
    for name in ("pre-bench", "post-bench"):
        hook = directory / name
        hook.write_text(f"#!/bin/sh\necho {name} >> events\n")
        hook.chmod(0o755)
    events = tmp_path / "events"
    collector = BenchCollector(config)
    engine = EngineConfig("v8", tmp_path, "true", "d8", "")

    def preflight(mode, log):
        assert collector.lock.held
        assert events.read_text().splitlines() == ["pre-bench"]
        with events.open("a") as f:
            f.write("preflight\n")
        return False

    monkeypatch.setattr(host, "preflight", preflight)
    monkeypatch.setattr(
        collector,
        "_bench_with_result_lock",
        lambda *a: pytest.fail("measured after failed preflight"),
    )
    try:
        with pytest.raises(HookError, match="not benchmark-ready"):
            with collector.admission().commit("test"):
                collector.bench_at_root(
                    engine, {"commit_id": 123, "embedder_id": 0}, tmp_path
                )
        assert not collector.lock.held
        assert not collector.store.is_done("v8", config.platform, K1(123))
        assert events.read_text().splitlines() == [
            "pre-bench",
            "preflight",
            "post-bench",
        ]
    finally:
        collector.store.close()
