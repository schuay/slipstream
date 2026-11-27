# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Gated retention (D018): an entry is freed once every gate passed it."""

import time

import pytest

from test_archive import FakeRsync, publish

from slipstream.archive import Archiver, empty_state
from slipstream.bus import Bus, BuilderState, cursor_path, write_cursor
from slipstream.bus_protocol import BusProtocol, encoded, read
from slipstream.config import (
    ArchiveConfig,
    BusConfig,
    BusSource,
    EngineConfig,
    TransferRoute,
    validate_release_after,
)
from slipstream.durability import atomic_write
from slipstream.models import CommitKey
from slipstream.retention import OverCap, Policy, holding_gate, release_points
from slipstream.transfer import Coordinator, LocalEndpoint

PEER = "00000000-0000-0000-0000-00000000beef"


def _engine(name):
    return EngineConfig(
        name=name,
        src_dir=None,
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
    )


@pytest.fixture
def protocol(tmp_path):
    protocol = BusProtocol(tmp_path / "bus")
    owner = protocol.bus_uuid()
    protocol.migrate({"v8": owner, "jsc": owner, "safari": owner})
    return protocol


@pytest.fixture
def builder_box(config, protocol, tmp_path):
    """A builder bus: released after transfer to a peer and the archive."""
    config.engines = {"v8": _engine("v8"), "jsc": _engine("jsc")}
    owner = protocol.ready()["uuid"]
    route = TransferRoute(
        "out",
        ["v8"],
        BusSource("out-source", str(protocol.root)),
        BusSource("out-destination", "/peer", ssh_host="peer"),
        owner,
        PEER,
    )
    config.bus = BusConfig(
        protocol.root,
        owners={"v8": owner, "jsc": owner},
        routes=[route],
        archive=ArchiveConfig(str(tmp_path / "nas"), ["v8"]),
        release_after=["transfer", "archive"],
    )
    return config


def set_gate(protocol, kind, engine, key):
    path = protocol.state_path(kind, engine)
    data = read(path) if path.exists() else {}
    if kind == "archive" and not data:
        data = empty_state("x")
    data["cursor"] = str(key) if key is not None else None
    atomic_write(path, encoded(data))


def keys(protocol, engine="v8"):
    return [k.commit_id for k in protocol.keys(engine)]


class TestDeadPrefix:
    def test_the_dead_prefix_is_the_minimum_over_applicable_gates(
        self, builder_box, protocol
    ):
        for key in range(1, 6):
            publish(protocol, key, {"out": f"build {key}".encode()})
        set_gate(protocol, "transfer", "v8", 4)
        set_gate(protocol, "archive", "v8", 2)
        policy = Policy(builder_box)
        assert policy.prune(protocol) == {"v8": [CommitKey.of(1), CommitKey.of(2)]}
        assert keys(protocol) == [3, 4, 5]
        set_gate(protocol, "archive", "v8", 5)
        policy.prune(protocol)
        # The newest entry survives even when every gate has passed it.
        assert keys(protocol) == [5]
        assert read(protocol.state_path("retention", "v8"))["highest_dropped"] == "4"

    def test_the_keep_suffix_is_protected(self, builder_box, protocol):
        for key in range(1, 4):
            publish(protocol, key, {"out": f"build {key}".encode()})
        set_gate(protocol, "transfer", "v8", 3)
        set_gate(protocol, "archive", "v8", 3)
        bus = Bus(protocol.root)
        bus.prune(None, ("v8", 2), release=release_points(builder_box, bus))
        assert keys(protocol) == [2, 3]

    def test_live_entries_are_never_evicted_for_the_budget(self, builder_box, protocol):
        builder_box.build.retain_gb = 1e-9
        for key in range(1, 4):
            publish(protocol, key, {"out": f"build {key}".encode()})
        set_gate(protocol, "transfer", "v8", 3)
        assert Policy(builder_box).prune(protocol) == {}
        assert keys(protocol) == [1, 2, 3]

    @pytest.mark.parametrize("cursor", ["missing", "torn"])
    def test_an_unreadable_cursor_frees_nothing(self, builder_box, protocol, cursor):
        for key in range(1, 4):
            publish(protocol, key, {"out": f"build {key}".encode()})
        set_gate(protocol, "transfer", "v8", 3)
        if cursor == "torn":
            path = protocol.state_path("archive", "v8")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{")
        assert Policy(builder_box).prune(protocol) == {}
        held = holding_gate(builder_box, protocol.root, "v8", CommitKey.of(1))
        assert held.gate == "archive" and held.cursor is None
        assert (held.error is not None) == (cursor == "torn")

    def test_an_engine_no_gate_applies_to_keeps_only_its_newest(
        self, builder_box, protocol
    ):
        for key in range(1, 4):
            publish(protocol, key, {"out": f"jsc {key}".encode()}, engine="jsc")
        Policy(builder_box).prune(protocol)
        assert keys(protocol, "jsc") == [3]

    def test_inner_entries_wait_for_the_derived_builder(self, builder_box, protocol):
        """Safari is packaged from jsc entries on this bus: freeing one first
        would leave a hole in the Safari series."""
        from dataclasses import replace

        builder_box.engines["safari"] = replace(_engine("safari"), derives="jsc")
        builder_box.build.engines = ["jsc", "safari"]
        for key in range(1, 5):
            publish(protocol, key, {"out": f"jsc {key}".encode()}, engine="jsc")
        assert Policy(builder_box).prune(protocol) == {}, "no safari frontier yet"
        protocol.write_builder_state("safari", BuilderState(frontier=CommitKey(7, 2)))
        Policy(builder_box).prune(protocol)
        # The entry at the frontier stays: a new release is first published
        # on that same inner commit.
        assert keys(protocol, "jsc") == [2, 3, 4]

    def test_freed_objects_are_swept(self, builder_box, protocol):
        first = publish(protocol, 1, {"out": b"one"})
        publish(protocol, 2, {"out": b"two"})
        set_gate(protocol, "transfer", "v8", 2)
        set_gate(protocol, "archive", "v8", 2)
        Policy(builder_box).release(protocol)
        assert not protocol.bus.object_path(first.objects()[0].name).exists()

    def test_without_release_after_the_budget_still_evicts(self, builder_box, protocol):
        builder_box.bus.release_after = []
        builder_box.build.retain_gb = 1e-9
        for key in range(1, 4):
            publish(protocol, key, {"out": f"build {key}".encode()})
        Policy(builder_box).prune(protocol)
        assert keys(protocol) == [3]


class TestConfig:
    def test_a_gate_that_applies_to_no_engine_is_refused(self, builder_box):
        builder_box.bus.release_after = ["transfer", "archive", "bench"]
        with pytest.raises(ValueError, match="'bench' applies to no engine"):
            validate_release_after(builder_box)
        builder_box.bus.release_after = ["archive"]
        builder_box.bus.archive = None
        with pytest.raises(ValueError, match="no \\[bus.archive\\]"):
            validate_release_after(builder_box)

    def test_declared_gates_apply_per_engine(self, builder_box):
        assert builder_box.release_gates("v8") == ["transfer", "archive"]
        assert builder_box.release_gates("jsc") == []

    def test_bench_applies_with_a_run_and_a_source(self, config, protocol):
        config.engines = {"v8": _engine("v8"), "jsc": _engine("jsc")}
        config.bus = BusConfig(
            protocol.root,
            sources=[BusSource("local", str(protocol.root), engines=["v8"])],
            release_after=["bench"],
        )
        validate_release_after(config)
        # jsc has a [[run]] entry but no source claims it.
        assert config.release_gates("v8") == ["bench"]
        assert config.release_gates("jsc") == []

    @pytest.mark.parametrize(
        "value, match",
        [(["bench", "deliver"], "unknown gates"), (["bench", "bench"], "twice")],
    )
    def test_release_after_values(self, tmp_path, value, match):
        from slipstream.config import _parse_bus

        with pytest.raises(ValueError, match=match):
            _parse_bus({"root": str(tmp_path), "release_after": value}, {})


@pytest.fixture
def bencher_box(config, protocol):
    """A bencher bus released after this machine's bench cursor."""
    config.engines = {"v8": _engine("v8")}
    config.bus = BusConfig(
        protocol.root,
        sources=[BusSource("local", str(protocol.root), engines=["v8"])],
        release_after=["bench"],
    )
    return config


def bench_cursor(cfg, key):
    write_cursor(cursor_path(cfg.out_dir, "local", "v8"), key)


class TestCap:
    def test_accept_refuses_over_the_cap(self, bencher_box, protocol, tmp_path):
        src = BusProtocol(tmp_path / "src")
        owner = src.bus_uuid()
        src.migrate({"v8": owner})
        dst = BusProtocol(tmp_path / "dst")
        dst.bus_uuid()
        dst.migrate({"v8": owner})
        bencher_box.bus.root = dst.root
        bencher_box.bus.sources = [BusSource("local", str(dst.root), engines=["v8"])]
        publish(src, 1, {"out": b"x" * 600})
        publish(src, 2, {"out": b"y" * 600})
        bencher_box.build.retain_gb = 1000 / 1e9
        endpoint = LocalEndpoint(dst.root, 1e12, policy=Policy(bencher_box))
        with src.capture("v8", 1) as stage:
            endpoint.accept(stage)
        with src.capture("v8", 2) as stage:
            with pytest.raises(OverCap, match="destination over cap"):
                endpoint.accept(stage)
        assert [k.commit_id for k in dst.keys("v8")] == [1]
        # Benching 1 releases nothing (it is the newest), but once 2 fits...
        bench_cursor(bencher_box, 1)
        bencher_box.build.retain_gb = 2000 / 1e9
        with src.capture("v8", 2) as stage:
            endpoint.accept(stage)
        # ...accept's own release frees the benched entry.
        assert [k.commit_id for k in dst.keys("v8")] == [2]

    def test_the_coordinator_retries_without_advancing_and_logs_once(
        self, config, tmp_path
    ):
        src = BusProtocol(tmp_path / "src")
        owner = src.bus_uuid()
        dst = BusProtocol(tmp_path / "dst")
        dst_id = dst.bus_uuid()
        route = TransferRoute(
            "t",
            ["v8"],
            BusSource("s", str(src.root)),
            BusSource("d", str(dst.root)),
            owner,
            dst_id,
        )
        src.migrate({"v8": owner}, [route])
        dst.migrate({"v8": owner})
        publish(src, 1, {"out": b"z" * 600})
        config.engines = {"v8": _engine("v8")}
        config.bus = BusConfig(src.root, owners={"v8": owner}, routes=[route])
        events = []
        coordinator = Coordinator(config, log=events.append)
        # A destination at its cap, whatever its gates.
        dest = coordinator.routes["v8"][2]
        dest.policy = type(
            "Capped",
            (),
            {
                "cap_bytes": 100,
                "prune": lambda self, bus: {},
                "release": lambda self, bus, *a, **k: {},
            },
        )()
        assert not coordinator.run(once=True)
        record = read(src.state_path("transfer", "v8"))
        assert (
            record["cursor"] is None and "destination over cap" in record["last_error"]
        )
        assert record["pending"]["op"] == "deliver"
        assert not coordinator.failed  # retryable, not a conflict
        errors = [e for e in events if "destination over cap" in str(e)]
        assert len(errors) == 1
        # The next attempt fails the same way and says nothing new.
        record["retry_after"] = 0
        atomic_write(src.state_path("transfer", "v8"), encoded(record))
        again = Coordinator(config, log=events.append)
        again.routes["v8"][2].policy = dest.policy
        again.run(once=True)
        assert len([e for e in events if "destination over cap" in str(e)]) == 1


class TestReleaseTriggers:
    def test_transfer_releases_after_deliver(self, config, tmp_path):
        src = BusProtocol(tmp_path / "src")
        owner = src.bus_uuid()
        dst = BusProtocol(tmp_path / "dst")
        dst_id = dst.bus_uuid()
        route = TransferRoute(
            "t",
            ["v8"],
            BusSource("s", str(src.root)),
            BusSource("d", str(dst.root)),
            owner,
            dst_id,
        )
        src.migrate({"v8": owner}, [route])
        dst.migrate({"v8": owner})
        for key in (1, 2, 3):
            publish(src, key, {"out": f"b{key}".encode()})
        config.engines = {"v8": _engine("v8")}
        config.bus = BusConfig(
            src.root, owners={"v8": owner}, routes=[route], release_after=["transfer"]
        )
        assert Coordinator(config, log=lambda m: None).run(once=True)
        assert keys(src) == [3]
        assert keys(dst) == [1, 2, 3]

    def test_the_consumer_releases_after_its_cursor_advances(
        self, bencher_box, protocol
    ):
        from slipstream.consumer import BusConsumer

        for key in (1, 2, 3):
            publish(protocol, key, {"out": f"b{key}".encode()})
        consumer = BusConsumer.__new__(BusConsumer)
        consumer.cfg, consumer.bus = bencher_box, Bus(protocol.root)
        consumer.policy, consumer.log = Policy(bencher_box), lambda m: None
        consumer.set_cursor(bencher_box.bus.sources[0], "v8", 2)
        assert keys(protocol) == [3]

    def test_the_archive_releases_after_a_batch(self, builder_box, protocol, tmp_path):
        builder_box.bus.release_after = ["archive"]
        (tmp_path / "nas").mkdir()
        for key in (1, 2, 3):
            publish(protocol, key, {"out": f"b{key}".encode()})
        archiver = Archiver(builder_box, log=lambda m: None, runner=FakeRsync())
        assert archiver.run(once=True)
        assert keys(protocol) == [3]


class TestEarlyArchive:
    @pytest.fixture
    def archiver(self, builder_box, protocol):
        for key in (1, 2):
            publish(protocol, key, {"out": f"b{key}".encode()})
        set_gate(protocol, "transfer", "v8", 2)
        archiver = Archiver(builder_box, log=lambda m: None, runner=FakeRsync())
        state = archiver.state("v8")
        state["last_success_at"] = time.time()  # not due for a day
        archiver.save("v8", state)
        return archiver

    def _pause(self, protocol, paused):
        protocol.write_builder_state(
            "v8", BuilderState(publishing_paused_by_cap=paused)
        )

    def test_fires_on_a_cap_pause_with_the_archive_holding(self, archiver, protocol):
        self._pause(protocol, True)
        now = time.time()
        assert archiver.due(now)["v8"] == now

    def test_not_without_a_cap_pause(self, archiver, protocol):
        self._pause(protocol, False)
        now = time.time()
        assert archiver.due(now)["v8"] > now + 3600

    def test_not_when_another_gate_holds(self, archiver, protocol):
        self._pause(protocol, True)
        set_gate(protocol, "transfer", "v8", None)
        set_gate(protocol, "archive", "v8", 2)
        state = archiver.state("v8")
        state["last_success_at"] = time.time()
        archiver.save("v8", state)
        now = time.time()
        assert archiver.due(now)["v8"] > now + 3600

    def test_not_during_a_failure_backoff(self, archiver, protocol):
        self._pause(protocol, True)
        state = archiver.state("v8")
        state.update(last_error="boom", retry_after=time.time() + 600)
        archiver.save("v8", state)
        now = time.time()
        assert archiver.due(now)["v8"] > now + 500


def test_a_bus_at_its_cap_releases_dead_entries_before_refusing(bencher_box, tmp_path):
    """Without this a caught-up bencher at its cap refuses forever: its
    consumer has nothing to bench, so nothing else would release."""
    src = BusProtocol(tmp_path / "src")
    owner = src.bus_uuid()
    src.migrate({"v8": owner})
    dst = BusProtocol(tmp_path / "dst")
    dst.bus_uuid()
    dst.migrate({"v8": owner})
    bencher_box.bus.root = dst.root
    bencher_box.bus.sources = [BusSource("local", str(dst.root), engines=["v8"])]
    for key in (1, 2, 3):
        publish(src, key, {"out": bytes([key]) * 600})
    # Delivered under the old budget, then benched to the newest.
    budget = LocalEndpoint(dst.root, 1e12)
    for key in (1, 2):
        with src.capture("v8", key) as stage:
            budget.accept(stage)
    bench_cursor(bencher_box, 2)
    bencher_box.build.retain_gb = 1300 / 1e9  # 1200 held + 600 incoming > cap
    gated = LocalEndpoint(dst.root, 1e12, policy=Policy(bencher_box))
    with src.capture("v8", 3) as stage:
        gated.accept(stage)  # refused if 1 were still held
    # 1 went to make room, then 2, benched and no longer newest, after.
    assert [k.commit_id for k in dst.keys("v8")] == [3]


def _home_config(tmp_path, dst, gated):
    home = tmp_path / "home"
    (home / ".config" / "slipstream").mkdir(parents=True)
    (home / ".config" / "slipstream" / "config.toml").write_text(
        f'out_dir = "{tmp_path / "out"}"\n'
        f'[engines.v8]\nsrc_dir = "{tmp_path}"\n'
        f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
        '[[run]]\nengine = "v8"\nsuite = "js3"\n'
        f'[bus]\nroot = "{dst.root}"\n'
        + ('release_after = ["bench"]\n' if gated else "")
        + f'[[bus.sources]]\nname = "local"\nroot = "{dst.root}"\n'
        "[build]\nretain_gb = 0.000001\n"  # 1000 bytes
    )
    return home


@pytest.mark.parametrize("gated", [True, False])
def test_the_endpoint_applies_local_gates_or_else_the_peers_budget(
    tmp_path, monkeypatch, gated
):
    from test_transfer import remote_endpoint

    from slipstream import transfer

    src = BusProtocol(tmp_path / "src")
    owner = src.bus_uuid()
    src.migrate({"v8": owner})
    dst = BusProtocol(tmp_path / "dst")
    dst.bus_uuid()
    dst.migrate({"v8": owner})
    monkeypatch.setenv("HOME", str(_home_config(tmp_path, dst, gated)))
    publish(src, 1, {"out": b"q" * 2000})
    remote, _ = remote_endpoint(dst, monkeypatch)  # passes --retain-bytes 1e9
    sent = []
    real_send = transfer.Wire.send_objects
    monkeypatch.setattr(
        transfer.Wire,
        "send_objects",
        lambda self, stage, names: sent.append(names) or real_send(self, stage, names),
    )
    with src.capture("v8", 1) as stage:
        if gated:
            with pytest.raises(OSError, match="destination over cap"):
                remote.accept(stage)
        else:
            remote.accept(stage)
    if gated:
        assert sent == [], "refused before any object was sent"
        assert dst.keys("v8") == []
    else:
        assert [k.commit_id for k in dst.keys("v8")] == [1]
