# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import hashlib
from dataclasses import replace

import pytest

from slipstream.bus import Blob, BusError, Entry
from slipstream.bus_protocol import BusProtocol, identity, read
from slipstream.transfer import LocalEndpoint, Coordinator
from slipstream.config import BusConfig, BusSource, TransferRoute


def publication(protocol, key, data=b"archive", tree="tree", engine="v8"):
    import uuid

    sha = hashlib.sha256(data).hexdigest()
    entry = Entry(
        engine,
        key,
        f"hash{key}",
        "",
        key,
        "",
        "cfg",
        [Blob("out", tree, sha, len(data))],
        version=4,
        owner_uuid=protocol.ready()["uuid"],
        publication_uuid=str(uuid.uuid4()),
    )
    with protocol.stage() as stage:
        path = stage.path / entry.objects()[0].name
        path.write_bytes(data)
        stage.paths[path.name] = path
        stage.entry, stage.text = entry, entry.to_json()
        protocol.publish_staged(stage)
    return entry


@pytest.fixture
def pair(tmp_path):
    src, dst = BusProtocol(tmp_path / "src"), BusProtocol(tmp_path / "dst")
    owner = src.bus_uuid()
    src.migrate({"v8": owner})
    dst.migrate({"v8": owner})
    return src, dst


def deliver(src, dst, key):
    with src.capture("v8", key) as stage:
        return LocalEndpoint(dst.root, 1e9).accept(stage)


def test_exact_manifest_and_replay_after_retention(pair):
    src, dst = pair
    first = publication(src, 1)
    receipt = deliver(src, dst, 1)
    assert receipt["identity"] == identity(first.to_json())
    assert (
        dst.bus.entry_path("v8", 1).read_bytes()
        == src.bus.entry_path("v8", 1).read_bytes()
    )
    # Completed receipt remains authoritative after local retention.
    dst.bus.entry_path("v8", 1).unlink()
    assert deliver(src, dst, 1) == receipt
    assert not dst.bus.entry_path("v8", 1).exists()


def test_prepared_recovery_after_crash(pair, monkeypatch):
    src, dst = pair
    publication(src, 1)
    original = dst._manifest_locked
    monkeypatch.setattr(
        dst, "_manifest_locked", lambda *a: (_ for _ in ()).throw(OSError("crash"))
    )
    with src.capture("v8", 1) as stage:
        with pytest.raises(OSError, match="crash"):
            dst.accept(stage)
    assert read(dst.state_path("acceptance", "v8"))["state"] == "prepared"
    monkeypatch.setattr(dst, "_manifest_locked", original)
    dst.sweep_blobs()
    assert dst.receipt("v8")["state"] == "complete"
    assert dst.read_entry("v8", 1)


def test_skip_fences_delayed_endpoint(pair):
    src, dst = pair
    publication(src, 1)
    with src.capture("v8", 1) as delayed:
        dst.skip("v8", 1, src.ready()["uuid"])
        with pytest.raises(BusError, match="watermark"):
            dst.accept(delayed)
    assert read(dst.state_path("upstream-loss", "v8"))["highest_dropped"] == "1"


def test_capture_survives_retention_gc_and_active_stage(pair):
    src, dst = pair
    first = publication(src, 1, b"one")
    publication(src, 2, b"two")
    with src.capture("v8", 1) as capture:
        src.prune(0)
        src.gc()
        assert not src.bus.entry_path("v8", 1).exists()
        assert capture.paths[first.objects()[0].name].read_bytes() == b"one"
        assert capture.path.exists()
        dst.accept(capture)
    assert not capture.path.exists()
    assert src.snapshot("v8")["retention"]["highest_dropped"] == "1"


def test_equal_tree_distinct_bytes(pair):
    src, _ = pair
    first = publication(src, 1, b"one", tree="same")
    second = publication(src, 2, b"two", tree="same")
    assert first.objects()[0].name != second.objects()[0].name
    assert all(
        src.bus.object_path(e.objects()[0].name).exists() for e in (first, second)
    )


def test_owner_conflict_and_changed_publication(pair):
    src, dst = pair
    publication(src, 1)
    deliver(src, dst, 1)
    with src.capture("v8", 1) as stage:
        stage.text = replace(stage.entry, title="different").to_json()
        stage.entry = Entry.from_json(stage.text)
        with pytest.raises(BusError, match="watermark"):
            dst.accept(stage)
        stage.entry.owner_uuid = dst.ready()["uuid"]
        stage.text = stage.entry.to_json()
        with pytest.raises(BusError, match="owner"):
            dst.accept(stage)


def test_missing_state_blocks_sweep(pair):
    src, _ = pair
    publication(src, 1)
    src.state_path("acceptance", "v8").unlink()
    with pytest.raises(BusError, match="operational state"):
        src.sweep_blobs()


def test_abandoned_staging_reclaimed(pair):
    src, _ = pair
    stage = src.stage()
    stage.owner.close()  # equivalent to process exit for the OS-held lease
    assert stage.path in src.gc()
    stage.owner = None


def test_migration_same_legacy_identity_and_resume(tmp_path):
    from slipstream.bus import Bus
    import uuid

    owner = str(uuid.uuid4())
    entries = []
    for name in ("a", "b"):
        bus = Bus(tmp_path / name)
        data = b"archive"
        sha = hashlib.sha256(data).hexdigest()
        tmp = bus.tmp_blob("tree")
        tmp.write_bytes(data)
        bus.store_blob(tmp, "tree")
        entry = Entry(
            "v8", 1, "hash", "", 1, "", "cfg", [Blob("out", "tree", sha, len(data))]
        )
        bus.publish(entry)
        protocol = BusProtocol(bus.root)
        protocol.migrate({"v8": owner})
        entries.append(protocol.bus.entry_path("v8", 1).read_bytes())
        assert protocol.migrate({"v8": owner}) == []
        assert protocol.read_entry("v8", 1).version == 4
    assert entries[0] == entries[1]


def test_prune_intent_recovers_after_unlink(pair, monkeypatch):
    src, _ = pair
    publication(src, 1, b"one")
    publication(src, 2, b"two")
    original = src._write

    def fail(kind, engine, data):
        if kind == "retention" and data["highest_dropped"]:
            raise OSError("crash after unlink")
        original(kind, engine, data)

    monkeypatch.setattr(src, "_write", fail)
    with pytest.raises(OSError):
        src.prune(0)
    monkeypatch.setattr(src, "_write", original)
    assert src.snapshot("v8")["retention"]["highest_dropped"] == "1"
    assert src.keys("v8")[0].commit_id == 2


def test_canonical_lock_alias(pair, tmp_path):
    from slipstream.durability import FileLock

    src, _ = pair
    alias = tmp_path / "alias"
    alias.symlink_to(src.root, target_is_directory=True)
    with src.lock():
        with pytest.raises(TimeoutError):
            with FileLock(BusProtocol(alias).root / ".bus.lock", timeout=0):
                pass


def test_coordinator_pending_lost_ack_then_retention(pair, config):
    src, dst = pair
    first = publication(src, 1, b"one")
    publication(src, 2, b"two")
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    config.bus = BusConfig(src.root, routes=[route])
    coordinator = Coordinator(config)
    record = {
        "source": route.source_uuid,
        "destination": route.destination_uuid,
        "cursor": None,
        "acknowledged": None,
        "pending": {"op": "deliver", "key": "1", "identity": identity(first.to_json())},
        "retry_after": 0,
    }
    deliver(src, dst, 1)
    src.prune(0)
    src.sweep_blobs()
    coordinator.step("v8", record)
    assert record["cursor"] == "1" and record["pending"] is None
    assert record["acknowledged"] == identity(first.to_json())


def remote_endpoint(protocol, monkeypatch):
    import shlex
    import subprocess
    import sys
    from slipstream.transfer import RemoteEndpoint

    real_popen = subprocess.Popen
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        assert command[0] == "ssh"
        cli = shlex.split(command[-1].split("exec slipstream ", 1)[1])
        return real_popen([sys.executable, "-m", "slipstream", *cli], **kwargs)

    monkeypatch.setattr("slipstream.transfer.subprocess.Popen", launch)
    endpoint = RemoteEndpoint(
        BusSource("peer", str(protocol.root), ssh_host="peer"),
        {"handshake": 5, "inactivity": 5, "total": 10},
        1e9,
    )
    return endpoint, commands


def test_real_endpoint_push_and_pull(pair, monkeypatch):
    src, dst = pair
    entry = publication(src, 1, b"bytes" * 20000)
    remote, commands = remote_endpoint(dst, monkeypatch)
    assert remote.info()["uuid"] == dst.ready()["uuid"]
    with src.capture("v8", 1) as capture:
        receipt = remote.accept(capture)
    assert receipt["identity"] == identity(entry.to_json())
    # The same remote capture protocol supplies a pull. It retains its lease
    # until the client leaves the context and acknowledges completion.
    with remote.capture("v8", 1, src) as capture:
        assert capture.text == entry.to_json()
        assert capture.paths[entry.objects()[0].name].read_bytes() == b"bytes" * 20000
        dst.gc()
        assert list((dst.root / "staging").glob("*"))
    assert remote.receipt("v8")["state"] == "complete"
    assert commands


def test_coordinator_once_freezes_upper_bound(pair, config, monkeypatch):
    src, dst = pair
    publication(src, 1)
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    config.bus = BusConfig(src.root, routes=[route])
    events = []
    coordinator = Coordinator(config, log=events.append)
    record = {
        "source": route.source_uuid,
        "destination": route.destination_uuid,
        "cursor": None,
        "acknowledged": None,
        "pending": None,
        "highest_skipped": None,
        "retry_after": 0,
    }
    coordinator.save("v8", record)
    destination = coordinator.routes["v8"][2]
    accept = destination.accept

    def accept_and_publish(capture):
        result = accept(capture)
        publication(src, 2, b"later")
        return result

    monkeypatch.setattr(destination, "accept", accept_and_publish)
    assert coordinator.run(once=True)
    assert dst.read_entry("v8", 1) and dst.read_entry("v8", 2) is None

    assert "queued=1, through=1" in events[0]
    assert any("accepted 1, remaining=0, elapsed=" in event for event in events)
    assert not any("accepted 2" in event for event in events)


def test_coordinator_singleton(pair, config):
    from slipstream.durability import FileLock

    src, _ = pair
    config.bus = BusConfig(src.root)
    with FileLock(src.root / ".coordinator.lock", timeout=0):
        with pytest.raises(TimeoutError):
            Coordinator(config).run(once=True)


def test_hung_endpoint_is_terminated_and_reaped(pair, monkeypatch):
    import subprocess
    import sys
    from slipstream.transfer import RemoteEndpoint

    src, _ = pair
    real_popen = subprocess.Popen
    processes = []

    def launch(command, **kwargs):
        proc = real_popen(
            [sys.executable, "-c", "import time; time.sleep(100)"], **kwargs
        )
        processes.append(proc)
        return proc

    monkeypatch.setattr("slipstream.transfer.subprocess.Popen", launch)
    endpoint = RemoteEndpoint(
        BusSource("peer", str(src.root), ssh_host="peer"),
        {"handshake": 0.05, "inactivity": 0.05, "total": 0.1},
        1e9,
    )
    with pytest.raises(TimeoutError):
        endpoint.info()
    assert all(proc.poll() is not None for proc in processes)


def test_child_retains_staging_lease_after_owner_exit(pair):
    import subprocess
    import sys

    src, _ = pair
    script = """
import os, subprocess, sys
from slipstream.bus_protocol import BusProtocol, staging_fds
with BusProtocol(sys.argv[1]).stage() as stage:
    print(stage.path, flush=True)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], pass_fds=staging_fds())
    os._exit(0)
"""
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(src.root)], stdout=subprocess.PIPE, text=True
    )
    from pathlib import Path

    path = Path(proc.stdout.readline().strip())
    proc.wait(timeout=5)
    src.gc()
    assert path.exists()
    # The orphan inherits stdout as well; EOF proves that child exited.
    proc.stdout.read()
    proc.stdout.close()
    src.gc()
    assert not path.exists()


@pytest.mark.parametrize("boundary", ["install", "prepared", "manifest", "complete"])
def test_acceptance_crash_boundaries_recover_or_replay(pair, monkeypatch, boundary):
    src, dst = pair
    entry = publication(src, 1)
    method = {
        "install": "_install_locked",
        "prepared": "_write",
        "manifest": "_manifest_locked",
        "complete": "_write",
    }[boundary]
    original = getattr(dst, method)

    def crash(*args):
        result = original(*args)
        if method != "_write" or (
            args[0] == "acceptance" and args[2]["state"] == boundary
        ):
            raise OSError("injected durable boundary failure")
        return result

    monkeypatch.setattr(dst, method, crash)
    with src.capture("v8", 1) as capture:
        with pytest.raises(OSError, match="injected"):
            dst.accept(capture)
    monkeypatch.setattr(dst, method, original)
    dst.sweep_blobs()
    receipt = deliver(src, dst, 1)
    assert receipt["identity"] == identity(entry.to_json())
    assert dst.bus.entry_path("v8", 1).read_text() == entry.to_json()


def test_no_machine_lock_and_no_network_under_bus_lock(pair, config, monkeypatch):
    from slipstream.durability import FileLock

    src, dst = pair
    publication(src, 1)
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    config.bus = BusConfig(src.root, routes=[route])
    coordinator = Coordinator(config)
    coordinator.save(
        "v8",
        {
            "source": route.source_uuid,
            "destination": route.destination_uuid,
            "cursor": None,
            "pending": None,
            "retry_after": 0,
        },
    )
    monkeypatch.setattr(
        "slipstream.lock.MachineLock.acquire",
        lambda *a, **k: pytest.fail("transfer acquired machine lock"),
    )
    for endpoint in coordinator.routes["v8"][1:]:
        for name in ("info", "snapshot", "receipt", "health", "mirror", "accept"):
            original = getattr(endpoint, name)

            def checked(*args, _original=original):
                for protocol in pair:
                    with FileLock(protocol.root / ".bus.lock", timeout=0):
                        pass
                return _original(*args)

            monkeypatch.setattr(endpoint, name, checked)
    assert coordinator.run(once=True)


def test_endpoint_control_eof_cancels_verification(pair):
    import subprocess
    import sys
    import time
    from slipstream.transfer import Wire

    src, _ = pair
    entry = publication(src, 1)
    script = """
import sys, time
from slipstream.bus_protocol import Staging
from slipstream.transfer import serve
Staging.verify = lambda self: time.sleep(60)
serve(sys.argv[1], sys.stdin.fileno(), sys.stdout.fileno(), 1000000)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(src.root)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        wire = Wire(process.stdout.fileno(), process.stdin.fileno(), handshake=3)
        wire.json({"op": "accept", "manifest": entry.to_json(), "transfer_version": 2})
        assert wire.message()["missing"] == []
        # Verification is deliberately stalled in the child.
        time.sleep(0.1)
        started = time.monotonic()
        process.stdin.close()
        assert process.wait(timeout=3) == 0
        assert time.monotonic() - started < 3
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdin.close()
        process.stdout.close()


@pytest.mark.parametrize("owners_on_m6", [set(), {"v8", "jsc"}, {"jsc"}])
def test_real_endpoints_all_on_m3_all_on_m6_and_mixed(
    tmp_path, config, monkeypatch, owners_on_m6
):
    m3, m6 = BusProtocol(tmp_path / "m3"), BusProtocol(tmp_path / "m6")
    ids = [m3.bus_uuid(), m6.bus_uuid()]
    owners = {engine: ids[engine in owners_on_m6] for engine in ("v8", "jsc")}
    routes = []
    for engine in owners:
        pull = engine in owners_on_m6
        source = BusSource(
            "m6" if pull else "m3",
            str(m6.root if pull else m3.root),
            ssh_host="m6" if pull else None,
        )
        destination = BusSource(
            "m3" if pull else "m6",
            str(m3.root if pull else m6.root),
            ssh_host=None if pull else "m6",
        )
        routes.append(
            TransferRoute(
                engine, [engine], source, destination, ids[pull], ids[not pull]
            )
        )
    m3.migrate(owners, routes)
    m6.migrate(owners)
    for engine in owners:
        publication(
            m6 if engine in owners_on_m6 else m3, 1, engine.encode(), engine=engine
        )
    config.bus = BusConfig(m3.root, owners=owners, routes=routes)
    remote_endpoint(m6, monkeypatch)
    assert Coordinator(config).run(once=True)
    for engine in owners:
        a, b = m3.read_entry(engine, 1), m6.read_entry(engine, 1)
        assert a.to_json() == b.to_json()
        assert (
            m3.bus.entry_path(engine, 1).read_bytes()
            == m6.bus.entry_path(engine, 1).read_bytes()
        )
        for obj in a.objects():
            assert (
                m3.bus.object_path(obj.name).read_bytes()
                == m6.bus.object_path(obj.name).read_bytes()
            )


def test_missing_progress_is_not_reset_by_migration(pair):
    src, dst = pair
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    with pytest.raises(BusError, match="operational state"):
        src.migrate(src.ready()["owners"], [route])


def test_endpoint_uuid_change_requires_reconciliation(pair, monkeypatch):
    import uuid

    src, dst = pair
    remote, _ = remote_endpoint(dst, monkeypatch)
    remote.expected_uuid = str(uuid.uuid4())
    with pytest.raises(BusError, match="UUID changed"):
        remote.info()


def test_migration_resumes_after_manifest_rewrite(tmp_path, monkeypatch):
    from slipstream.bus import Bus

    bus = Bus(tmp_path / "legacy")
    sha = hashlib.sha256(b"archive").hexdigest()
    tmp = bus.tmp_blob("tree")
    tmp.write_bytes(b"archive")
    bus.store_blob(tmp, "tree")
    bus.publish(Entry("v8", 1, "hash", "", 1, "", "cfg", [Blob("out", "tree", sha, 7)]))
    protocol = BusProtocol(bus.root)
    owner = protocol.bus_uuid()
    original = protocol._write
    monkeypatch.setattr(
        protocol,
        "_write",
        lambda *a: (_ for _ in ()).throw(OSError("migration interrupted")),
    )
    with pytest.raises(OSError):
        protocol.migrate({"v8": owner})
    canonical = bus.entry_path("v8", 1).read_bytes()
    assert Entry.from_json(canonical).version == 4
    monkeypatch.setattr(protocol, "_write", original)
    protocol.migrate({"v8": owner})
    assert bus.entry_path("v8", 1).read_bytes() == canonical


def test_skip_recovery_does_not_rewrite_loss_or_notify_forever(pair):
    src, dst = pair
    dst.skip("v8", 1, src.ready()["uuid"])
    path = dst.state_path("upstream-loss", "v8")
    inode = path.stat().st_ino
    dst.snapshot("v8")
    dst.snapshot("v8")
    assert path.stat().st_ino == inode


def test_health_failure_does_not_backoff_artifact(pair, config, monkeypatch):
    src, dst = pair
    publication(src, 1)
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    config.bus = BusConfig(src.root, routes=[route])
    coordinator = Coordinator(config)
    coordinator.save(
        "v8",
        {
            "source": route.source_uuid,
            "destination": route.destination_uuid,
            "cursor": None,
            "pending": None,
            "retry_after": 0,
        },
    )
    monkeypatch.setattr(
        coordinator.routes["v8"][1],
        "health",
        lambda engine: (_ for _ in ()).throw(OSError("health unavailable")),
    )
    assert coordinator.run(once=True)
    progress = read(coordinator.path("v8"))
    assert progress["cursor"] == "1" and progress["retry_after"] == 0
    assert progress["health_error"] == "health unavailable"


def test_missing_readiness_is_not_reset_by_identity_or_migrate(pair):
    src, _ = pair
    (src.root / "ready.json").unlink()
    with pytest.raises(BusError, match="readiness"):
        src.bus_uuid()
    with pytest.raises(BusError, match="readiness"):
        src.migrate({"v8": "00000000-0000-0000-0000-000000000001"})


def test_restart_lost_ack_after_source_retention(pair, config, monkeypatch):
    src, dst = pair
    publication(src, 1, b"one")
    publication(src, 2, b"two")
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    config.bus = BusConfig(src.root, routes=[route])
    first = Coordinator(config, log=lambda _: None)
    first.save(
        "v8",
        {
            "source": route.source_uuid,
            "destination": route.destination_uuid,
            "cursor": None,
            "pending": None,
            "retry_after": 0,
        },
    )
    destination = first.routes["v8"][2]
    accept = destination.accept

    def lost_ack(capture):
        accept(capture)
        raise OSError("acknowledgement lost")

    monkeypatch.setattr(destination, "accept", lost_ack)
    assert not first.run(once=True)
    src.prune(0)
    src.sweep_blobs()
    restarted = Coordinator(config)
    progress = read(restarted.path("v8"))
    assert progress["pending"]["key"] == "1"
    progress["retry_after"] = 0  # persisted deadline has elapsed on restart
    restarted.save("v8", progress)
    assert restarted.run(once=True)
    progress = read(restarted.path("v8"))
    assert progress["acknowledged"]["key"] == "2"
    assert progress.get("highest_skipped") is None


def test_offline_rebuild_rewinds_without_replacing_online(pair, config):
    src, dst = pair
    old = publication(src, 1)
    publication(src, 2, b"later")
    route = TransferRoute(
        "test",
        ["v8"],
        BusSource("src", str(src.root)),
        BusSource("dst", str(dst.root)),
        src.ready()["uuid"],
        dst.ready()["uuid"],
    )
    config.bus = BusConfig(src.root, routes=[route])
    coordinator = Coordinator(config)
    initial = {
        "source": route.source_uuid,
        "destination": route.destination_uuid,
        "cursor": None,
        "pending": None,
        "retry_after": 0,
    }
    coordinator.save("v8", initial)
    assert coordinator.run(once=True)
    # Quiescent operator maintenance: remove exact old manifests and replace
    # the destination watermark and coordinator progress in locked writes.
    for protocol in pair:
        with protocol.lock():
            protocol.bus.entry_path("v8", 1).unlink()
    with dst.lock():
        dst._write("acceptance", "v8", {"state": "empty"})
    new = publication(src, 1, b"rebuilt")
    assert new.publication_uuid != old.publication_uuid
    coordinator.save("v8", initial)
    assert Coordinator(config).run(once=True)
    assert dst.read_entry("v8", 1) == new
    assert dst.read_entry("v8", 2) == src.read_entry("v8", 2)


def test_gc_contends_with_install_and_cannot_see_unpublished_objects(pair, monkeypatch):
    import threading
    from slipstream.durability import FileLock

    src, dst = pair
    entry = publication(src, 1)
    install = dst._install_locked
    workers = []

    def installing(stage):
        install(stage)
        # Installation through journal/manifest publication must be one lock.
        with pytest.raises(TimeoutError):
            with FileLock(dst.root / ".bus.lock", timeout=0):
                pass
        worker = threading.Thread(target=BusProtocol(dst.root).sweep_blobs)
        worker.start()
        workers.append(worker)

    monkeypatch.setattr(dst, "_install_locked", installing)
    with src.capture("v8", 1) as capture:
        dst.accept(capture)
    for worker in workers:
        worker.join(timeout=5)
        assert not worker.is_alive()
    assert dst.bus.object_path(entry.objects()[0].name).exists()


def test_data_inactivity_timeout_includes_a_blocked_writer():
    import os
    import time
    from slipstream.transfer import Wire, attempt_deadline

    incoming, control = os.pipe()
    sink, outgoing = os.pipe()
    try:
        wire = Wire(incoming, outgoing, inactivity=0.05, total=1)
        wire.started = True
        started = time.monotonic()
        with attempt_deadline(1):
            with pytest.raises(TimeoutError, match="inactivity"):
                wire.send(b"x" * 1_000_000)
        assert time.monotonic() - started < 0.5
    finally:
        for fd in (incoming, control, sink, outgoing):
            os.close(fd)


def test_handshake_deadline_is_not_extended_by_trickling_bytes():
    import os
    import threading
    from slipstream.transfer import Wire

    incoming, control = os.pipe()
    sink, outgoing = os.pipe()
    stopped = threading.Event()

    def trickle():
        for byte in b"{" + b" " * 30 + b"}\n":
            if stopped.wait(0.01):
                return
            os.write(control, bytes([byte]))

    worker = threading.Thread(target=trickle)
    try:
        wire = Wire(incoming, outgoing, handshake=0.05, inactivity=0.5, total=1)
        worker.start()
        with pytest.raises(TimeoutError):
            wire.message()
    finally:
        stopped.set()
        worker.join()
        for fd in (incoming, control, sink, outgoing):
            os.close(fd)


def test_transfer_preserves_exact_noncanonical_manifest_bytes(pair):
    src, dst = pair
    entry = publication(src, 1)
    raw = (entry.to_json().replace("\n", "\r\n") + "\r\n").encode()
    with src.lock():
        src.bus.entry_path("v8", 1).write_bytes(raw)
    receipt = deliver(src, dst, 1)
    assert dst.bus.entry_path("v8", 1).read_bytes() == raw
    assert receipt["identity"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert deliver(src, dst, 1) == receipt


@pytest.mark.parametrize("direction", ["push", "pull"])
def test_shared_delta_base_only_crosses_wire_once(pair, monkeypatch, direction):
    import uuid
    from slipstream.bus import Delta
    from slipstream.delta import Block, DeltaPlan
    from slipstream.transfer import Wire

    src, dst = pair
    base = publication(src, 1, b"large base" * 10000)
    patch = b"small patch"
    sha = hashlib.sha256(patch).hexdigest()
    plan = DeltaPlan(1, "target", 1, (Block("block", 1, sha, len(patch)),))
    delta = replace(
        base,
        commit_id=2,
        publication_uuid=str(uuid.uuid4()),
        blobs=[
            Blob(
                "out",
                "new-tree",
                delta=Delta(
                    base.blobs[0].id, base.blobs[0].sha256, base.blobs[0].bytes, plan
                ),
            )
        ],
    )
    with src.stage() as stage:
        stage.entry, stage.text = delta, delta.to_json()
        for obj in delta.objects():
            path = stage.path / obj.name
            path.write_bytes(patch if obj.sha256 == sha else b"large base" * 10000)
            stage.paths[obj.name] = path
        src.publish_staged(stage)

    # Count payload bytes across real endpoint processes, not control messages.
    transferred = []
    if direction == "push":
        original = Wire.send_objects

        def send(self, stage, names):
            transferred.append(
                sum(o.bytes for o in stage.entry.objects() if o.name in names)
            )
            return original(self, stage, names)

        monkeypatch.setattr(Wire, "send_objects", send)
        remote, _ = remote_endpoint(dst, monkeypatch)

        def transfer(key):
            with LocalEndpoint(src.root, 1e9).capture("v8", key, dst) as stage:
                remote.accept(stage)
    else:
        original = Wire.receive_objects

        def receive(self, stage):
            transferred.append(
                sum(o.bytes for o in stage.entry.objects() if o.name not in stage.paths)
            )
            return original(self, stage)

        monkeypatch.setattr(Wire, "receive_objects", receive)
        remote, _ = remote_endpoint(src, monkeypatch)

        def transfer(key):
            with remote.capture("v8", key, dst) as stage:
                LocalEndpoint(dst.root, 1e9).accept(stage)

    transfer(1)
    transfer(2)
    assert transferred == [base.blobs[0].bytes, len(patch)]
    assert dst.read_entry("v8", 2).to_json() == delta.to_json()


def test_reused_objects_are_pinned_and_not_rehashed(pair, monkeypatch):
    src, dst = pair
    first = publication(src, 1)
    deliver(src, dst, 1)
    second = publication(src, 2)
    with dst.stage() as stage:
        assert stage.reuse(second.to_json()) == []
        dst.prune(0)
        dst.sweep_blobs()
        assert stage.paths[first.objects()[0].name].read_bytes() == b"archive"

        def unexpected(*args):
            pytest.fail("reused object was hashed or flushed")

        monkeypatch.setattr("slipstream.bus_protocol.sha256_file", unexpected)
        import os
        from slipstream.bus_protocol import flush_file

        inode = stage.paths[first.objects()[0].name].stat().st_ino

        def flush(fd):
            if os.fstat(fd).st_ino == inode:
                unexpected()
            flush_file(fd)

        monkeypatch.setattr("slipstream.bus_protocol.flush_file", flush)
        dst.accept(stage)
    assert dst.read_entry("v8", 2)


def test_new_transfer_bytes_are_verified(pair):
    src, dst = pair
    entry = publication(src, 1)
    with dst.stage() as stage:
        assert stage.reuse(entry.to_json()) == [entry.objects()[0].name]
        path = stage.path / "received"
        path.write_bytes(b"corrupt")
        stage.paths[entry.objects()[0].name] = path
        with pytest.raises(BusError, match="checksum"):
            dst.accept(stage)
    assert dst.snapshot("v8")["keys"] == []


def test_replay_returns_receipt_without_reading_source_objects(pair, monkeypatch):
    src, dst = pair
    publication(src, 1)
    receipt = deliver(src, dst, 1)
    remote, _ = remote_endpoint(dst, monkeypatch)
    with src.capture("v8", 1) as stage:
        stage.paths.clear()
        assert remote.accept(stage) == receipt
        assert LocalEndpoint(dst.root, 1e9).accept(stage) == receipt


@pytest.mark.parametrize(
    "names", [["unknown"], ["duplicate", "duplicate"], [None], "all"]
)
def test_invalid_missing_object_request(pair, names):
    from slipstream.transfer import Wire

    src, _ = pair
    publication(src, 1)
    with src.capture("v8", 1) as stage:
        with pytest.raises(BusError, match="missing-object"):
            Wire.send_objects(None, stage, names)


@pytest.mark.parametrize("op", ["capture", "accept"])
def test_old_transfer_protocol_fails_before_payload(pair, monkeypatch, op):
    src, _ = pair
    entry = publication(src, 1)
    remote, _ = remote_endpoint(src, monkeypatch)
    with pytest.raises(BusError, match="update both endpoints"):
        remote.call(op, engine="v8", key="1", manifest=entry.to_json())


def test_reuse_rejects_wrong_size(pair):
    src, dst = pair
    entry = publication(src, 1)
    deliver(src, dst, 1)
    # A known filename with a wrong size is never trusted or requested anew.
    dst.bus.object_path(entry.objects()[0].name).write_bytes(b"truncated")
    with dst.stage() as stage:
        with pytest.raises(BusError, match="existing object size mismatch"):
            stage.reuse(entry.to_json())
