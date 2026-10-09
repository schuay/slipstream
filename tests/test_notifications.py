# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import json
import select
import subprocess
import sys
import threading
import time

import pytest

from slipstream import notifications as n
from slipstream.bus import _atomic_write
from slipstream.config import BusSource


@pytest.fixture
def subscription(tmp_path):
    if not hasattr(select, "kqueue"):
        pytest.skip("requires kqueue")
    sub = n.DirectorySubscription(tmp_path / "bus", ["v8", "jsc"])
    try:
        yield sub
    finally:
        sub.close()


def changed(sub, engine="v8"):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if engine in sub.changes(0.1):
            return
    pytest.fail("publication did not wake subscription")


def test_first_publish_and_atomic_replace(subscription):
    topic = subscription.topics["v8"]
    _atomic_write(topic / "1.json", "{}")
    changed(subscription)
    _atomic_write(topic / "2.json", "{}")
    changed(subscription)
    assert subscription.changes(0) == set()


def test_directory_replacement(subscription):
    topic = subscription.topics["v8"]
    _atomic_write(topic / "1.json", "{}")
    changed(subscription)
    topic.rename(topic.with_name("old"))
    _atomic_write(topic / "2.json", "{}")
    changed(subscription)
    _atomic_write(topic / "3.json", "{}")
    changed(subscription)


def test_ancestor_replacement(subscription):
    topic = subscription.topics["v8"]
    _atomic_write(topic / "1.json", "{}")
    changed(subscription)
    root = topic.parents[2]
    root.rename(root.with_name("old-bus"))
    _atomic_write(topic / "2.json", "{}")
    changed(subscription)
    _atomic_write(topic / "3.json", "{}")
    changed(subscription)


def test_unrelated_parent_writes_do_not_schedule_scan(subscription):
    root = subscription.topics["v8"].parents[2]
    (root.parent / "unrelated").write_text("hi")
    assert subscription.changes(0.1) == set()
    assert not root.exists()  # subscribing does not create the bus


def test_publish_then_process_exit(subscription):
    topic = subscription.topics["v8"]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os,sys; from pathlib import Path; "
            "from slipstream.bus import _atomic_write; "
            "_atomic_write(Path(sys.argv[1]), '{}'); os._exit(0)",
            str(topic / "1.json"),
        ],
        check=True,
    )
    changed(subscription)


def test_generation_race_and_backlog():
    schedule = n.WorkSchedule(["v8"], [], 1800)
    assert not schedule.ready()  # subscription must become ready first
    schedule.changed(["v8"])
    generation = schedule.snapshot("v8")
    schedule.changed(["v8"])
    schedule.finished("v8", generation)
    assert schedule.ready() == {"v8"}
    # A bounded drain cannot clear pending work, even without new events.
    for _ in range(100):
        schedule.finished("v8", schedule.snapshot("v8"), more=True)
        assert schedule.ready() == {"v8"}
    schedule.finished("v8", schedule.snapshot("v8"))
    assert not schedule.ready()


def test_retry_and_git_deadlines(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(n.time, "monotonic", lambda: now[0])
    schedule = n.WorkSchedule(["v8"], ["jsc"], 1800)
    schedule.git_finished("jsc")
    schedule.changed(["v8"])
    schedule.finished("v8", schedule.snapshot("v8"), retry=60)
    now[0] += 30
    schedule.changed(["v8"])
    assert not schedule.ready()
    now[0] += 30
    assert schedule.ready() == {"v8"}
    schedule.finished("v8", schedule.snapshot("v8"))
    assert not schedule.ready()
    now[0] = 1900
    assert schedule.ready() == {"jsc"}


def test_event_interrupts_idle_wait():
    schedule = n.WorkSchedule(["v8"], [], 1800)
    done = threading.Event()
    thread = threading.Thread(target=lambda: (schedule.wait(lambda: False), done.set()))
    thread.start()
    schedule.changed(["v8"])
    assert done.wait(0.5)
    thread.join()


def reader():
    # Instantiate without starting threads to test the stream boundary directly.
    obj = n.Subscriptions.__new__(n.Subscriptions)
    obj.stop = threading.Event()
    return obj


def stream_process(monkeypatch, script):
    real_popen = subprocess.Popen
    processes = []
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        process = real_popen([sys.executable, "-u", "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(n.subprocess, "Popen", popen)
    return processes, commands


def test_remote_ready_changed_and_shutdown(monkeypatch):
    messages = [
        {"version": 1, "type": "ready", "engines": ["v8"]},
        {"version": 1, "type": "changed", "engines": ["v8"]},
    ]
    script = (
        "import time\n"
        + "\n".join(
            f"print({json.dumps(message)!r}, flush=True)" for message in messages
        )
        + "\ntime.sleep(30)"
    )
    processes, commands = stream_process(monkeypatch, script)
    obj = reader()
    received = []

    def receive(names):
        received.append(names)
        if len(received) == 2:
            obj.stop.set()

    obj._remote(BusSource("m6", "~/bus space", ssh_host="m6"), ["v8"], receive)
    assert received == [["v8"], ["v8"]]
    assert commands[0][0] == "ssh"
    assert commands[0][-2] == "m6"
    assert "~/" in commands[0][-1]
    assert "--engine v8" in commands[0][-1]
    assert processes[0].poll() is not None


@pytest.mark.parametrize("on_path", [False, True])
def test_remote_command_finds_tool_without_shell_startup_files(
    tmp_path, monkeypatch, on_path
):
    fallback = tmp_path / ".local/bin"
    fallback.mkdir(parents=True)
    tools = tmp_path / "custom bin"
    tools.mkdir()
    engines = ["v8; echo unexpected"]
    expected_args = [
        "bus",
        "subscribe",
        str(tmp_path / "bus space"),
        "--engine",
        *engines,
    ]
    message = {"version": 1, "type": "ready", "engines": engines}
    script = (
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"assert sys.argv[1:] == {expected_args!r}\n"
        f"print(json.dumps({message!r}), flush=True)\n"
        "sys.stdin.read()\n"
    )
    executable = (tools if on_path else fallback) / "slipstream"
    executable.write_text(script)
    executable.chmod(0o755)
    if on_path:
        # An existing PATH installation must take precedence over the fallback.
        (fallback / "slipstream").write_text("#!/bin/sh\nexit 1\n")
        (fallback / "slipstream").chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", f"{tools}:/usr/bin:/bin" if on_path else "/usr/bin:/bin")
    real_popen = subprocess.Popen

    def popen(command, **kwargs):
        return real_popen(["/bin/sh", "-c", command[-1]], **kwargs)

    monkeypatch.setattr(n.subprocess, "Popen", popen)
    obj = reader()
    received = []

    def receive(names):
        received.append(names)
        obj.stop.set()

    obj._remote(BusSource("remote", "~/bus space", ssh_host="remote"), engines, receive)
    assert received == [engines]


@pytest.mark.parametrize(
    "message",
    [
        [],
        {"version": 2, "type": "ready", "engines": ["v8"]},
        {"version": 1, "type": "changed", "engines": ["v8"]},
        {"version": 1, "type": "ready", "engines": []},
        {"version": 1, "type": "ready", "engines": ["unknown"]},
    ],
)
def test_remote_rejects_invalid_protocol(monkeypatch, message):
    processes, _ = stream_process(monkeypatch, f"print({json.dumps(message)!r})")
    with pytest.raises(ValueError, match="invalid subscription"):
        reader()._remote(BusSource("m6", "/bus", ssh_host="m6"), ["v8"], lambda _: None)
    assert processes[0].poll() is not None


def test_reconnect_marks_pending_again(monkeypatch):
    obj = reader()
    obj.schedule = n.WorkSchedule(["v8"], [], 1800)
    obj.log = lambda _: None
    attempts = []

    def remote(source, engines, receive):
        receive(engines)
        attempts.append(obj.schedule.snapshot("v8"))
        if len(attempts) == 2:
            obj.stop.set()
        raise ValueError("EOF")

    monkeypatch.setattr(obj, "_remote", remote)
    obj._run(BusSource("m6", "/bus", ssh_host="m6"), ["v8"])
    assert attempts == [1, 2]
    assert obj.schedule.ready() == {"v8"}


def test_local_reader_survives_busy_consumer(tmp_path):
    if not hasattr(select, "kqueue"):
        pytest.skip("requires kqueue")
    source = BusSource("local", str(tmp_path / "bus"))
    schedule = n.WorkSchedule(["v8"], [], 1800)
    subscriptions = n.Subscriptions([(source, ["v8"])], schedule, lambda _: None)
    try:
        schedule.wait(lambda: False)
        generation = schedule.snapshot("v8")
        schedule.finished("v8", generation)
        _atomic_write(source.local_root / "topics/builds/v8/1.json", "{}")
        deadline = time.monotonic() + 3
        schedule.wait(lambda: time.monotonic() > deadline)
        assert schedule.ready() == {"v8"}
        # Finishing an older scan must preserve the incoming notification.
        schedule.finished("v8", generation)
        assert schedule.ready() == {"v8"}
    finally:
        subscriptions.close()
    assert all(not thread.is_alive() for thread in subscriptions.threads)


def test_subscription_command_stream_and_client_exit(tmp_path):
    if not hasattr(select, "kqueue"):
        pytest.skip("requires kqueue")
    root = tmp_path / "bus"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "slipstream",
            "bus",
            "subscribe",
            str(root),
            "--engine",
            "v8",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        ready = json.loads(process.stdout.readline())
        assert ready == {"version": 1, "type": "ready", "engines": ["v8"]}
        _atomic_write(root / "topics/builds/v8/1.json", "{}")
        assert select.select([process.stdout], [], [], 3)[0]
        message = json.loads(process.stdout.readline())
        assert message == {"version": 1, "type": "changed", "engines": ["v8"]}
        # A disconnected SSH client must not leave a sleeping remote process.
        process.stdin.close()
        assert process.wait(timeout=3) == 0
        assert process.stderr.read() == b""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
        process.stdin.close()
