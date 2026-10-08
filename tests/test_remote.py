# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from slipstream.bus import Blob, Bus, BuilderState, Entry
from slipstream.config import BusSource
from slipstream.consumer import ShaMismatch
from slipstream.remote import SshSource, quote_remote
from keys import K, K1


def _entry(commit_id, blobs=None):
    return Entry(
        engine="v8",
        commit_id=commit_id,
        hash=f"hash{commit_id}",
        date="2026-09-06",
        timestamp=1757116800,
        title=f"commit {commit_id}",
        build_cfg_hash="sha256:cfg",
        blobs=blobs or [Blob("out", f"id{commit_id}", "sha", 3)],
    )


def _store(bus, payload, path="out"):
    blob_id = hashlib.sha256(payload).hexdigest()[:16]
    tmp = bus.tmp_blob(blob_id)
    tmp.write_bytes(payload)
    bus.store_blob(tmp, blob_id)
    return Blob(path, blob_id, hashlib.sha256(payload).hexdigest(), len(payload))


@pytest.fixture
def remote(tmp_path, monkeypatch):
    """An SshSource whose ssh and rsync act on a local directory.

    The far side is a real bus root, so the path shapes and the listing
    parsing are exercised rather than mocked away.
    """
    far = Bus(tmp_path / "far")
    source = SshSource(
        BusSource(name="box2", root=str(far.root), ssh_host="box2", bwlimit="20000")
    )
    calls = []

    def fake_run(cmd, timeout=None):
        calls.append(cmd)
        if cmd[0] == "ssh":
            # Run the remote command locally; the paths are real. It is the
            # last argument, after the host and whatever options were passed.
            return subprocess.run(
                ["bash", "-c", cmd[-1]], capture_output=True, text=True
            )
        if cmd[0] == "rsync":
            local = [
                c.replace("box2:", "")
                for c in cmd
                if not c.startswith("-e") and not c.startswith("ssh -o")
            ]
            return subprocess.run(local, capture_output=True, text=True)
        raise AssertionError(cmd)

    monkeypatch.setattr("slipstream.remote._run", fake_run)
    return type("R", (), {"far": far, "source": source, "calls": calls})


def _publish(far, commit_id, payload=None, blobs=None):
    if blobs is None:
        if payload is None:
            payload = f"payload {commit_id}".encode()
        blobs = [_store(far, payload)]
    entry = _entry(commit_id, blobs)
    far.publish(entry)
    return entry


class TestListing:
    def test_lists_ids_above_the_cursor(self, remote):
        for cid in (100, 101, 102):
            _publish(remote.far, cid)
        assert remote.source.keys_above("v8", 100) == K(101, 102)
        assert remote.source.keys_above("v8", None) == K(100, 101, 102)

    def test_an_empty_topic_is_not_an_error(self, remote):
        assert remote.source.keys_above("v8", None) == K()

    def test_tmp_files_are_ignored(self, remote):
        _publish(remote.far, 100)
        (remote.far.topic_dir("v8") / "101.json.tmp-1-a").write_text("{}")
        assert remote.source.keys_above("v8", None) == K(100)

    def test_an_ssh_failure_is_raised(self, remote, monkeypatch):
        def broken(cmd, timeout=None):
            return subprocess.CompletedProcess(cmd, 255, "", "ssh: connect failed")

        monkeypatch.setattr("slipstream.remote._run", broken)
        with pytest.raises(subprocess.CalledProcessError):
            remote.source.keys_above("v8", None)


class TestEntries:
    def test_reads_an_entry(self, remote):
        published = _publish(remote.far, 100)
        assert remote.source.read_entry("v8", 100) == published

    def test_a_missing_entry_is_none(self, remote):
        assert remote.source.read_entry("v8", 100) is None


class TestFetch:
    @pytest.fixture
    def local(self, tmp_path):
        return Bus(tmp_path / "local")

    def test_streams_to_the_store_with_the_rate_limit(self, remote, local):
        entry = _publish(remote.far, 100, b"payload bytes")
        paths = remote.source.fetch(entry, local)
        assert paths == [local.blob_path(entry.blobs[0].id)]
        assert paths[0].read_bytes() == b"payload bytes"
        (rsync,) = [c for c in remote.calls if c[0] == "rsync"]
        assert "--bwlimit=20000" in rsync
        # Not ssh cat: that decodes the blob as text and buffers it whole.
        assert not any(c[0] == "ssh" and "cat" in c[2] for c in remote.calls)
        # Nothing is left in tmp once the blob is stored.
        assert not list(local.tmp_dir.glob("*"))

    def test_only_blobs_the_store_lacks_cross_the_link(self, remote, local):
        """The ICU file is the same for every commit; it is fetched once."""
        icu = _store(remote.far, b"icu" * 10, path="icudtl.dat")
        first = _publish(remote.far, 100, blobs=[_store(remote.far, b"d8 100"), icu])
        second = _publish(remote.far, 101, blobs=[_store(remote.far, b"d8 101"), icu])
        remote.source.fetch(first, local)
        remote.calls.clear()
        paths = remote.source.fetch(second, local)
        assert [p.name for p in paths] == [f"{b.id}.tar.zst" for b in second.blobs]
        rsyncs = [c for c in remote.calls if c[0] == "rsync"]
        assert len(rsyncs) == 1 and second.blobs[0].id in rsyncs[0][-2]

    def test_no_rate_limit_when_unset(self, local, monkeypatch):
        source = SshSource(BusSource(name="box2", root="/bus", ssh_host="box2"))
        calls = []

        def fake(cmd, timeout=None):
            calls.append(cmd)
            # Produce the file rsync would have, with the right content.
            local.tmp_dir.mkdir(parents=True, exist_ok=True)
            (local.tmp_dir / "abc.tar.zst").write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("slipstream.remote._run", fake)
        blob = Blob("out", "abc", hashlib.sha256(b"x").hexdigest(), 1)
        source.fetch(_entry(100, [blob]), local)
        assert not any(c.startswith("--bwlimit") for c in calls[0])

    def test_a_failed_fetch_keeps_the_partial_for_the_next_attempt(self, remote, local):
        """--partial --inplace are there because a blob is hundreds of MB
        over a link slow enough to want a bwlimit. The partial has a stable
        name, so the next attempt finds it."""
        entry = _entry(999, [Blob("out", "missing", "sha", 3)])
        local.tmp_dir.mkdir(parents=True)
        partial = local.tmp_dir / "missing.tar.zst"
        partial.write_bytes(b"half a blob")
        with pytest.raises(subprocess.CalledProcessError):
            remote.source.fetch(entry, local)
        assert partial.read_bytes() == b"half a blob"

    def test_a_blob_that_does_not_match_is_deleted_not_resumed(self, remote, local):
        """Nothing to resume from: the bytes on disk are the wrong ones."""
        entry = _publish(remote.far, 100, b"good")
        remote.far.blob_path(entry.blobs[0].id).write_bytes(b"bad!")
        with pytest.raises(ShaMismatch):
            remote.source.fetch(entry, local)
        assert not list(local.tmp_dir.glob("*"))
        assert not local.has_blob(entry.blobs[0].id)

    def test_the_remote_path_is_quoted_like_every_other_one(self, local, monkeypatch):
        """Listing and entry reads would work and only the fetch would fail,
        with an rsync error pointing at nothing."""
        calls = []

        def fake(cmd, timeout=None):
            calls.append(cmd)
            local.tmp_dir.mkdir(parents=True, exist_ok=True)
            (local.tmp_dir / "abc.tar.zst").write_bytes(b"x")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("slipstream.remote._run", fake)
        source = SshSource(BusSource(name="box2", root="~/a dir/bus", ssh_host="box2"))
        blob = Blob("out", "abc", hashlib.sha256(b"x").hexdigest(), 1)
        source.fetch(_entry(100, [blob]), local)
        (remote_arg,) = [c for c in calls[0] if c.startswith("box2:")]
        assert remote_arg == "box2:~/'a dir/bus/blobs/abc.tar.zst'"

    def test_a_version_one_entry_is_fetched_from_its_old_path(self, remote, local):
        """A builder not yet restarted since the upgrade still serves these."""
        payload = b"old payload"
        legacy = remote.far.legacy_blob_path("v8", 100)
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(payload)
        data = json.loads(_entry(100).to_json())
        del data["blobs"]
        data.update(
            version=1,
            blob_sha256=hashlib.sha256(payload).hexdigest(),
            blob_bytes=len(payload),
        )
        remote.far.entry_path("v8", 100).parent.mkdir(parents=True)
        remote.far.entry_path("v8", 100).write_text(json.dumps(data))

        entry = remote.source.read_entry("v8", 100)
        assert entry.version == 1
        (path,) = remote.source.fetch(entry, local)
        assert path.read_bytes() == payload
        (rsync,) = [c for c in remote.calls if c[0] == "rsync"]
        assert rsync[-2].endswith("/blobs/builds/v8/100.tar.zst")


class TestBuilderState:
    def test_reads_the_far_builders_state(self, remote):
        remote.far.write_builder_state(
            "v8", BuilderState(frontier=109680, lowest_retained=109120)
        )
        state = remote.source.builder_state("v8")
        assert state.frontier == K1(109680) and state.lowest_retained == K1(109120)

    def test_a_missing_state_file_reads_as_empty(self, remote):
        assert remote.source.builder_state("v8").frontier is None


class TestQuoting:
    def test_a_home_relative_path_still_expands(self):
        assert quote_remote("~/slipstream/bus") == "~/slipstream/bus"

    def test_a_path_with_a_space_is_quoted(self):
        assert quote_remote("/a dir/bus") == "'/a dir/bus'"

    def test_a_home_relative_path_with_a_space_is_quoted_after_the_tilde(self):
        assert quote_remote("~/a dir/bus") == "~/'a dir/bus'"


def test_a_remote_source_drives_the_consumer(config, tmp_path, monkeypatch, remote):
    """The consumer does not know which kind of source it has."""
    from slipstream.builder import store_run_set
    from slipstream.collector import BenchCollector, BenchOutcome
    from slipstream.config import BusConfig, EngineConfig
    from slipstream.consumer import BusConsumer

    src = tmp_path / "build"
    (src / "out").mkdir(parents=True)
    (src / "out" / "d8").write_bytes(b"binary")
    blobs = store_run_set(remote.far, src, ["out"], caffeinate=False)
    entry = _entry(100, blobs)
    remote.far.publish(entry)

    bus_source = BusSource(
        name="box2", root=str(remote.far.root), ssh_host="box2", engines=["v8"]
    )
    config.bus = BusConfig(root=tmp_path / "local-bus", sources=[bus_source])
    config.bench.min_free_gb = 0.001
    config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=None,
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
    )
    collector = BenchCollector(config, role="watch")
    collector.lock.path = tmp_path / "machine.lock"
    monkeypatch.setattr(collector, "harness_revs", lambda: {})
    roots = []
    monkeypatch.setattr(
        collector,
        "_run_benchmarks",
        lambda e, cid, root: (roots.append(root), BenchOutcome(1, 1, 5))[1],
    )
    consumer = BusConsumer(config, collector)

    result = consumer.drain(
        bus_source,
        "v8",
        lambda: False,
        lambda: collector.lock.try_acquire(),
        collector.lock.release,
    )
    assert result.benched == 1 and result.error is None
    assert (roots[0] / "out" / "d8").read_bytes() == b"binary"
    assert collector.store.is_done("v8", config.platform, 100)
    # The fetched blob is in this machine's store, named by the bench state
    # and so kept for the next entry to share; tmp is clean.
    local = Bus(config.bus.root)
    assert local.has_blob(entry.blobs[0].id)
    assert local.read_bench_state("v8").blobs == [entry.blobs[0].id]
    assert not list(local.tmp_dir.glob("*"))


class TestUnreachableIsNotMissing:
    """A source that cannot answer must never look like one saying "not there":
    the consumer steps over a missing entry and advances its cursor."""

    @pytest.fixture
    def unreachable(self, monkeypatch):
        source = SshSource(BusSource(name="box2", root="/bus", ssh_host="box2"))
        monkeypatch.setattr(
            "slipstream.remote._run",
            lambda cmd, timeout=None: subprocess.CompletedProcess(
                cmd, 255, "", "ssh: connect to host box2 port 22: Connection refused"
            ),
        )
        return source

    def test_read_entry_raises(self, unreachable):
        with pytest.raises(subprocess.CalledProcessError):
            unreachable.read_entry("v8", 100)

    def test_builder_state_raises(self, unreachable):
        with pytest.raises(subprocess.CalledProcessError):
            unreachable.builder_state("v8")

    def test_bench_state_raises(self, unreachable):
        with pytest.raises(subprocess.CalledProcessError):
            unreachable.bench_state("v8")

    def test_a_missing_entry_is_still_none(self, remote):
        _publish(remote.far, 100)
        assert remote.source.read_entry("v8", 999) is None


class TestTransportHardening:
    """A consumer fetches while holding the machine lock, so nothing here may
    wait forever or stop to ask for a password."""

    def test_ssh_gets_batch_mode_keepalives_and_a_timeout(self, remote):
        _publish(remote.far, 100)
        remote.source.read_entry("v8", 100)
        (ssh,) = [c for c in remote.calls if c[0] == "ssh"]
        assert "BatchMode=yes" in ssh
        assert "ConnectTimeout=15" in ssh
        assert "ServerAliveInterval=15" in ssh

    def test_rsync_carries_an_io_timeout_and_the_same_ssh_options(
        self, remote, tmp_path
    ):
        entry = _publish(remote.far, 100, b"payload")
        remote.source.fetch(entry, Bus(tmp_path / "local"))
        (rsync,) = [c for c in remote.calls if c[0] == "rsync"]
        assert any(c.startswith("--timeout=") for c in rsync)
        assert any("BatchMode=yes" in c for c in rsync)

    def test_a_hung_ssh_becomes_an_error_not_a_wait(self, monkeypatch):
        """A hang would hold the machine lock until someone noticed."""
        source = SshSource(BusSource(name="box2", root="/bus", ssh_host="box2"))
        monkeypatch.setattr(
            "slipstream.remote.subprocess.run",
            lambda *a, **k: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(a[0], k.get("timeout") or 0)
            ),
        )
        with pytest.raises(subprocess.CalledProcessError) as exc:
            source.read_entry("v8", 100)
        # A transport failure, so the consumer leaves its cursor alone rather
        # than reading it as a dropped entry.
        assert exc.value.returncode == 124 and "timed out" in exc.value.stderr

    def test_an_ssh_error_mentioning_a_missing_file_is_not_a_missing_entry(
        self, monkeypatch
    ):
        """A broken key says "No such file or directory" too, and reading that
        as retention makes a box report itself up to date forever."""
        source = SshSource(BusSource(name="box2", root="/bus", ssh_host="box2"))
        monkeypatch.setattr(
            "slipstream.remote._run",
            lambda cmd, timeout=None: subprocess.CompletedProcess(
                cmd,
                255,
                "",
                "Warning: Identity file ~/.ssh/id_box2 not accessible: "
                "No such file or directory.\nPermission denied (publickey).",
            ),
        )
        with pytest.raises(subprocess.CalledProcessError):
            source.keys_above("v8", None)
        with pytest.raises(subprocess.CalledProcessError):
            source.read_entry("v8", 100)

    def test_missing_is_decided_by_an_exit_code_not_a_message(self, monkeypatch):
        """stderr is localised, and ssh's own errors read like a missing file."""
        from slipstream.remote import MISSING_EXIT

        source = SshSource(BusSource(name="box2", root="/bus", ssh_host="box2"))
        monkeypatch.setattr(
            "slipstream.remote._run",
            lambda cmd, timeout=None: subprocess.CompletedProcess(
                cmd, MISSING_EXIT, "", ""
            ),
        )
        assert source.read_entry("v8", 100) is None
        assert source.keys_above("v8", None) == K()
        assert source.builder_state("v8").frontier is None

    def test_a_localised_error_message_is_still_an_error(self, monkeypatch):
        """A German remote would otherwise stall the cursor forever."""
        source = SshSource(BusSource(name="box2", root="/bus", ssh_host="box2"))
        monkeypatch.setattr(
            "slipstream.remote._run",
            lambda cmd, timeout=None: subprocess.CompletedProcess(
                cmd, 1, "", "cat: /bus/x.json: Datei oder Verzeichnis nicht gefunden"
            ),
        )
        with pytest.raises(subprocess.CalledProcessError):
            source.read_entry("v8", 100)
