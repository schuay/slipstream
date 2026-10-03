# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json

import pytest

from slipstream.bus import (
    BenchState,
    Blob,
    Bus,
    BuilderState,
    BusError,
    Entry,
    cursor_path,
    read_cursor,
    tree_hash,
    write_cursor,
)
from keys import K, K1


def _entry(commit_id, engine="v8", blobs=None, **kw):
    return Entry(
        engine=engine,
        commit_id=commit_id,
        hash=f"hash{commit_id}",
        date="2026-09-06",
        timestamp=1757116800,
        title=f"commit {commit_id}",
        build_cfg_hash="sha256:cfg",
        blobs=blobs if blobs is not None else [Blob("out", f"id{commit_id}", "sha", 0)],
        builder={"bot": "box2-m4"},
        built_at=1757116999,
        build_secs=1183,
        **kw,
    )


def _store(bus, payload, path="out"):
    """Store ``payload`` as a blob whose id is its content hash."""
    blob_id = hashlib.sha256(payload).hexdigest()[:16]
    tmp = bus.tmp_blob(blob_id)
    tmp.write_bytes(payload)
    bus.store_blob(tmp, blob_id)
    return Blob(path, blob_id, hashlib.sha256(payload).hexdigest(), len(payload))


def _bytes(commit_id, n):
    """``n`` bytes that differ per commit, so two entries do not share a blob."""
    return (f"{commit_id}-".encode() * n)[:n]


def _publish(bus, commit_id, payload=None, engine="v8", blobs=None, **kw):
    if blobs is None:
        if payload is None:
            payload = f"payload {engine} {commit_id}".encode()
        blobs = [_store(bus, payload)]
    entry = _entry(commit_id, engine=engine, blobs=blobs, **kw)
    bus.publish(entry)
    return entry


@pytest.fixture
def bus(tmp_path):
    return Bus(tmp_path / "bus")


class TestPublish:
    def test_round_trip(self, bus):
        published = _publish(bus, 109680, b"payload")
        assert bus.keys("v8") == K(109680)
        read = bus.read_entry("v8", 109680)
        assert read == published
        assert bus.blob_path(read.blobs[0].id).read_bytes() == b"payload"

    def test_blobs_are_staged_in_tmp_and_stored_flat(self, bus):
        """tmp/ is on the same filesystem as blobs/, so store_blob is a rename;
        a crash before it leaves the tmp file for gc, not a half blob."""
        tmp = bus.tmp_blob("abc")
        assert tmp.parent == bus.tmp_dir
        tmp.write_bytes(b"x")
        assert not bus.has_blob("abc")
        assert bus.store_blob(tmp, "abc") == bus.blobs_dir / "abc.tar.zst"
        assert bus.has_blob("abc") and not tmp.exists()

    def test_a_republish_replaces_the_manifest(self, bus):
        """A crashed publish self-heals: the retry writes the same entry name.
        The old content's blob is unreferenced from then on."""
        first = _publish(bus, 100, b"first")
        _publish(bus, 100, b"second")
        assert bus.keys("v8") == K(100)
        read = bus.read_entry("v8", 100)
        assert bus.blob_path(read.blobs[0].id).read_bytes() == b"second"
        assert bus.sweep_blobs() == [bus.blob_path(first.blobs[0].id)]

    def test_tmp_files_are_not_listed_as_entries(self, bus):
        _publish(bus, 100)
        (bus.topic_dir("v8") / "101.json.tmp-9-ab").write_text("{}")
        assert bus.keys("v8") == K(100)

    def test_engines_are_separate(self, bus):
        _publish(bus, 100, engine="v8")
        _publish(bus, 500, engine="jsc")
        assert bus.keys("v8") == K(100)
        assert bus.keys("jsc") == K(500)
        assert bus.engines() == ["jsc", "v8"]

    def test_an_empty_root_lists_nothing(self, bus):
        assert bus.keys("v8") == K()
        assert bus.read_entry("v8", 1) is None
        assert bus.engines() == []


class TestTreeHash:
    def test_identical_trees_hash_alike_regardless_of_mtime(self, tmp_path):
        import os

        for name in ("a", "b"):
            d = tmp_path / name / "out"
            d.mkdir(parents=True)
            (d / "d8").write_bytes(b"binary")
            (d / "icudtl.dat").write_bytes(b"icu")
        os.utime(tmp_path / "a" / "out" / "d8", (1, 1))
        assert tree_hash(tmp_path / "a", "out") == tree_hash(tmp_path / "b", "out")

    def test_content_path_and_mode_all_count(self, tmp_path):
        import os

        base = tmp_path / "src" / "out"
        base.mkdir(parents=True)
        (base / "d8").write_bytes(b"binary")
        before = tree_hash(tmp_path / "src", "out")

        (base / "d8").write_bytes(b"binary2")
        changed_content = tree_hash(tmp_path / "src", "out")
        (base / "d8").write_bytes(b"binary")
        assert changed_content != before

        os.chmod(base / "d8", 0o755)
        assert tree_hash(tmp_path / "src", "out") != before
        os.chmod(base / "d8", 0o644)
        assert tree_hash(tmp_path / "src", "out") == before

        # The same bytes under another run set path unpack elsewhere.
        other = tmp_path / "src2" / "out2"
        other.mkdir(parents=True)
        (other / "d8").write_bytes(b"binary")
        assert tree_hash(tmp_path / "src2", "out2") != before

    def test_a_symlink_is_hashed_as_a_link(self, tmp_path):
        base = tmp_path / "src" / "app"
        base.mkdir(parents=True)
        (base / "real").write_bytes(b"x")
        (base / "link").symlink_to("real")
        with_link = tree_hash(tmp_path / "src", "app")
        (base / "link").unlink()
        (base / "link").write_bytes(b"x")
        assert tree_hash(tmp_path / "src", "app") != with_link

    def test_a_single_file_entry(self, tmp_path):
        (tmp_path / "icudtl.dat").write_bytes(b"icu")
        assert tree_hash(tmp_path, "icudtl.dat") == tree_hash(tmp_path, "icudtl.dat")


class TestEntryFormat:
    def test_an_unknown_version_is_refused(self, bus):
        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = json.loads(path.read_text())
        data["version"] = 99
        path.write_text(json.dumps(data))
        with pytest.raises(BusError, match="version 99"):
            bus.read_entry("v8", 100)

    def test_a_torn_entry_is_refused(self, bus):
        _publish(bus, 100)
        bus.entry_path("v8", 100).write_text('{"version": 1, "engi')
        with pytest.raises(BusError, match="unreadable"):
            bus.read_entry("v8", 100)

    def test_unknown_fields_are_ignored(self, bus):
        """A newer builder may add fields at the same version."""
        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = json.loads(path.read_text())
        data["future_field"] = "x"
        path.write_text(json.dumps(data))
        assert bus.read_entry("v8", 100).commit_id == 100

    def test_a_malformed_blob_list_is_a_bus_error(self, bus):
        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = json.loads(path.read_text())
        data["blobs"] = [{"id": "x"}]
        path.write_text(json.dumps(data))
        with pytest.raises(BusError, match="malformed blob"):
            bus.read_entry("v8", 100)
        data["blobs"] = []
        path.write_text(json.dumps(data))
        with pytest.raises(BusError, match="names no blobs"):
            bus.read_entry("v8", 100)


def _publish_v1(bus, commit_id, payload=b"old payload", engine="v8"):
    """An entry exactly as a version 1 builder wrote it, payload and all."""
    legacy = bus.legacy_blob_path(engine, commit_id)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(payload)
    data = json.loads(_entry(commit_id, engine=engine).to_json())
    del data["blobs"]
    data["version"] = 1
    data["blob_sha256"] = hashlib.sha256(payload).hexdigest()
    data["blob_bytes"] = len(payload)
    bus.entry_path(engine, commit_id).parent.mkdir(parents=True, exist_ok=True)
    bus.entry_path(engine, commit_id).write_text(json.dumps(data))
    return data


class TestVersionOneEntries:
    """What a builder wrote before manifests reads back as a one-blob
    manifest, resolves to the old path until migrated, and migrates in place."""

    def test_reads_as_a_single_blob_manifest(self, bus):
        data = _publish_v1(bus, 100)
        entry = bus.read_entry("v8", 100)
        assert entry.version == 1
        assert entry.blobs == [
            Blob("", data["blob_sha256"], data["blob_sha256"], len(b"old payload"))
        ]
        assert entry.blob_bytes == len(b"old payload")
        assert bus.entry_blob_path(entry, entry.blobs[0]) == bus.legacy_blob_path(
            "v8", 100
        )

    def test_a_version_one_entry_without_its_payload_fields_is_refused(self, bus):
        _publish_v1(bus, 100)
        path = bus.entry_path("v8", 100)
        data = json.loads(path.read_text())
        del data["blob_sha256"]
        path.write_text(json.dumps(data))
        with pytest.raises(BusError, match="missing.*blobs"):
            bus.read_entry("v8", 100)

    def test_migrate_moves_the_payload_into_the_store(self, bus):
        data = _publish_v1(bus, 100)
        _publish(bus, 101)  # already current, untouched
        assert bus.migrate() == K(100)
        entry = bus.read_entry("v8", 100)
        assert entry.version == 2
        assert bus.blob_path(data["blob_sha256"]).read_bytes() == b"old payload"
        assert not bus.legacy_blob_path("v8", 100).exists()
        assert not (bus.root / "blobs" / "builds").exists()
        assert bus.migrate() == []

    def test_migrate_is_safe_to_interrupt(self, bus):
        """After the hardlink but before the rewrite, the entry is still
        version 1 and resolves to whichever copy exists."""
        data = _publish_v1(bus, 100)
        import os

        os.link(bus.legacy_blob_path("v8", 100), bus.blob_path(data["blob_sha256"]))
        entry = bus.read_entry("v8", 100)
        # Both exist: the store wins, since it is where the entry is going.
        assert bus.entry_blob_path(entry, entry.blobs[0]) == bus.blob_path(
            data["blob_sha256"]
        )
        assert bus.migrate() == K(100)
        assert bus.read_entry("v8", 100).version == 2

    def test_a_missing_payload_is_left_for_the_consumer_to_report(self, bus):
        _publish_v1(bus, 100)
        bus.legacy_blob_path("v8", 100).unlink()
        assert bus.migrate() == []
        assert bus.read_entry("v8", 100).version == 1

    def test_gc_reclaims_a_version_one_payload_with_no_entry(self, bus):
        _publish_v1(bus, 100)
        _publish_v1(bus, 101)
        bus.entry_path("v8", 100).unlink()
        removed = bus.gc()
        assert removed == [bus.legacy_blob_path("v8", 100)]
        assert bus.legacy_blob_path("v8", 101).exists()


class TestCursorSemantics:
    def test_ids_above_the_cursor(self, bus):
        for cid in (100, 200, 300):
            _publish(bus, cid)
        assert bus.keys_above("v8", 100) == K(200, 300)
        assert bus.keys_above("v8", None) == K(100, 200, 300)
        assert bus.keys_above("v8", 300) == K()

    def test_cursor_round_trip(self, tmp_path):
        p = cursor_path(tmp_path, "box2", "v8")
        assert read_cursor(p) is None
        write_cursor(p, 109680)
        assert read_cursor(p) == K1(109680)
        write_cursor(p, 109681)
        assert read_cursor(p) == K1(109681)
        assert [q.name for q in p.parent.iterdir()] == ["v8"]

    def test_a_torn_cursor_is_reported(self, tmp_path):
        p = cursor_path(tmp_path, "box2", "v8")
        p.parent.mkdir(parents=True)
        p.write_text("half-writ")
        with pytest.raises(BusError, match="unreadable cursor"):
            read_cursor(p)

    def test_sources_have_separate_cursors(self, tmp_path):
        write_cursor(cursor_path(tmp_path, "local", "v8"), 1)
        write_cursor(cursor_path(tmp_path, "box2", "v8"), 2)
        assert read_cursor(cursor_path(tmp_path, "local", "v8")) == K1(1)
        assert read_cursor(cursor_path(tmp_path, "box2", "v8")) == K1(2)


class TestState:
    def test_builder_state_round_trip(self, bus):
        state = BuilderState(frontier=109680, lowest_retained=109120)
        state.failed = [{"commit_id": 109655, "kind": "compile", "at": 1}]
        bus.write_builder_state("v8", state)
        read = bus.read_builder_state("v8")
        assert read.frontier == K1(109680) and read.failed[0]["kind"] == "compile"
        assert read.updated_at > 0

    def test_bench_state_round_trip(self, bus):
        bus.write_bench_state("v8", BenchState(bot="box2-m4", cursor=109679, lag=1))
        read = bus.read_bench_state("v8")
        assert (read.bot, read.cursor, read.lag) == ("box2-m4", K1(109679), 1)

    def test_the_two_state_files_do_not_share_a_path(self, bus):
        """Separate processes write them; one file would clobber whichever wrote first."""
        assert bus.builder_state_path("v8") != bus.bench_state_path("v8")

    def test_missing_state_reads_as_empty(self, bus):
        assert bus.read_builder_state("v8").frontier is None
        assert bus.read_bench_state("v8").cursor is None

    def test_a_bad_version_is_refused(self, bus):
        bus.write_builder_state("v8", BuilderState(frontier=1))
        p = bus.builder_state_path("v8")
        p.write_text(json.dumps({"version": 7, "frontier": 1}))
        with pytest.raises(BusError, match="version 7"):
            bus.read_builder_state("v8")


class TestRetention:
    def test_drops_oldest_first_to_fit_the_budget(self, bus):
        entries = {
            cid: _publish(bus, cid, _bytes(cid, 100)) for cid in (100, 200, 300, 400)
        }
        dropped = bus.prune("v8", 250)
        assert dropped == K(100, 200)
        assert bus.keys("v8") == K(300, 400)
        assert bus.lowest_retained("v8") == K1(300)
        # The manifest goes first; the blob follows at the sweep.
        assert bus.has_blob(entries[100].blobs[0].id)
        bus.sweep_blobs()
        assert not bus.has_blob(entries[100].blobs[0].id)
        assert bus.has_blob(entries[300].blobs[0].id)

    def test_the_newest_entry_is_never_dropped(self, bus):
        """A budget smaller than one artifact must not empty the topic."""
        _publish(bus, 100, b"x" * 1000)
        assert bus.prune("v8", 10) == K()
        assert bus.keys("v8") == K(100)

    def test_nothing_to_do_under_budget(self, bus):
        for cid in (100, 200):
            _publish(bus, cid, _bytes(cid, 10))
        assert bus.prune("v8", 10_000) == K()
        assert bus.footprint("v8") == 20

    def test_gc_reclaims_a_blob_with_no_entry(self, bus):
        orphan = _publish(bus, 100)
        kept = _publish(bus, 200)
        bus.entry_path("v8", 100).unlink()  # as a crash mid-prune leaves it
        removed = bus.gc()
        assert removed == [bus.blob_path(orphan.blobs[0].id)]
        assert bus.has_blob(kept.blobs[0].id)

    def test_gc_reclaims_an_abandoned_tmp_blob(self, bus):
        tmp = bus.tmp_blob("abc")
        tmp.write_bytes(b"partial")
        assert bus.gc() == [tmp]
        assert not tmp.exists()

    def test_gc_keeps_everything_referenced(self, bus):
        entry = _publish(bus, 100)
        assert bus.gc() == []
        assert bus.has_blob(entry.blobs[0].id)


class TestSharedBlobs:
    """Dedupe on publish, the fetch cache and retention are one mechanism:
    reference counting over manifests."""

    def test_a_shared_blob_is_counted_once(self, bus):
        icu = _store(bus, b"i" * 100, path="icudtl.dat")
        _publish(bus, 100, blobs=[_store(bus, _bytes(100, 50)), icu])
        _publish(bus, 200, blobs=[_store(bus, _bytes(200, 50)), icu])
        assert bus.footprint("v8") == 200

    def test_dropping_an_entry_frees_only_what_it_alone_named(self, bus):
        icu = _store(bus, b"i" * 100, path="icudtl.dat")
        old = _publish(bus, 100, blobs=[_store(bus, _bytes(100, 50)), icu])
        for cid in (200, 300):
            _publish(bus, cid, blobs=[_store(bus, _bytes(cid, 50)), icu])
        # Newest first: 300 costs 150, 200 another 50, 100 another 50.
        assert bus.prune("v8", 220) == K(100)
        assert bus.sweep_blobs() == [bus.blob_path(old.blobs[0].id)]
        assert bus.has_blob(icu.id)

    def test_an_entry_sharing_everything_costs_nothing(self, bus):
        same = [_store(bus, b"x" * 100)]
        for cid in (100, 200, 300):
            _publish(bus, cid, blobs=same)
        assert bus.prune("v8", 100) == K()
        assert bus.footprint("v8") == 100

    def test_blobs_are_shared_across_engines(self, bus):
        """A safari entry names the jsc runtime blobs plus the app bundle;
        dropping it leaves the runtime to jsc and frees only the bundle."""
        runtime = _store(bus, b"r" * 10, path="WebKitBuild/Release")
        app = _store(bus, b"a" * 100, path="Safari Technology Preview.app")
        _publish(bus, 100, engine="jsc", blobs=[runtime])
        _publish(bus, 100, engine="safari", blobs=[runtime, app])
        assert bus.sweep_blobs() == []
        bus.entry_path("safari", 100).unlink()
        assert bus.sweep_blobs() == [bus.blob_path(app.id)]
        assert bus.has_blob(runtime.id)

    def test_a_benchers_held_blobs_survive_the_sweep(self, bus):
        """On a bench-only box no manifest names a fetched blob; the state
        file does, so the next entry can reuse what this one shares."""
        held = _store(bus, b"held")
        stale = _store(bus, b"stale")
        bus.write_bench_state("v8", BenchState(blobs=[held.id]))
        assert bus.sweep_blobs() == [bus.blob_path(stale.id)]
        assert bus.has_blob(held.id)
        bus.write_bench_state("v8", BenchState(blobs=[]))
        assert bus.sweep_blobs() == [bus.blob_path(held.id)]


class TestRetentionLeavesNoHole:
    def test_what_survives_is_contiguous(self, bus):
        """A smaller older entry kept below a dropped one would fit more in
        the budget, but it holds lowest_retained down and hides the hole."""
        for cid, size in ((100, 40), (200, 100), (300, 200)):
            _publish(bus, cid, _bytes(cid, size))
        assert bus.prune("v8", 250) == K(100, 200)
        assert bus.keys("v8") == K(300)
        assert bus.lowest_retained("v8") == K1(300)

    def test_the_entry_just_published_is_kept(self, bus):
        """build --retry republishes below the frontier, so its entry is the
        oldest and the same cycle would otherwise delete it."""
        for cid in (100, 200, 300):
            _publish(bus, cid, _bytes(cid, 100))
        bus.prune("v8", 250)
        _publish(bus, 50, _bytes(50, 100))
        assert bus.prune("v8", 250, keep=50) == K()
        assert K1(50) in bus.keys("v8")

    def test_keep_does_not_protect_an_unrelated_entry(self, bus):
        for cid in (100, 200, 300):
            _publish(bus, cid, _bytes(cid, 100))
        assert bus.prune("v8", 250, keep=300) == K(100)

    def test_keep_lowers_the_floor_rather_than_punching_a_hole(self, bus):
        """Exempting one entry would leave a gap below lowest_retained, which
        is the signal a consumer uses to notice entries went missing."""
        for cid in (100, 200, 201, 202):
            _publish(bus, cid, _bytes(cid, 100))
        assert bus.prune("v8", 250, keep=100) == K()
        assert bus.keys("v8") == K(100, 200, 201, 202)
        assert bus.lowest_retained("v8") == K1(100)

    def test_the_overshoot_lasts_one_cycle(self, bus):
        for cid in (100, 200, 201, 202):
            _publish(bus, cid, _bytes(cid, 100))
        bus.prune("v8", 250, keep=100)
        # The next publish prunes normally and drops it with its neighbours.
        _publish(bus, 203, _bytes(203, 100))
        assert bus.prune("v8", 250, keep=203) == K(100, 200, 201)
        assert bus.keys("v8") == K(202, 203)


class TestEntryRequiredFields:
    def test_a_missing_field_is_a_bus_error_not_a_type_error(self, bus):
        """A consumer survives BusError and dies on TypeError."""
        import json as _json

        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = _json.loads(path.read_text())
        del data["title"]
        path.write_text(_json.dumps(data))
        with pytest.raises(BusError, match="missing.*title"):
            bus.read_entry("v8", 100)

    def test_optional_fields_may_be_absent(self, bus):
        import json as _json

        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = _json.loads(path.read_text())
        del data["build_secs"]
        path.write_text(_json.dumps(data))
        assert bus.read_entry("v8", 100).build_secs == 0

    def test_gc_reclaims_a_half_fetched_payload(self, bus):
        """An interrupted rsync leaves hundreds of MB no entry names."""
        bus.tmp_dir.mkdir(parents=True)
        partial = bus.tmp_dir / "v8-100.tar.zst"
        partial.write_bytes(b"half a payload")
        assert bus.gc() == [partial]
        assert not partial.exists()

    def test_gc_reclaims_a_leaked_entry_tmp_file(self, bus):
        """A crash between writing the entry tmp and renaming it leaks one per
        crash; commit_ids ignores them and nothing else looked."""
        _publish(bus, 100)
        leaked = bus.topic_dir("v8") / "101.json.tmp-9-abc"
        leaked.write_text("{}")
        assert leaked in bus.gc()
        assert not leaked.exists()
        assert bus.keys("v8") == K(100)

    @pytest.mark.parametrize("text", ["null", "[1, 2]", "42", '"a string"'])
    def test_json_that_is_not_an_object_is_a_bus_error(self, bus, text):
        """Every other malformed case is a BusError, which a consumer
        survives; .get on one of these is an AttributeError, which it does
        not -- it kills watch and every engine with it."""
        _publish(bus, 100)
        bus.entry_path("v8", 100).write_text(text)
        with pytest.raises(BusError, match="not an object"):
            bus.read_entry("v8", 100)

    @pytest.mark.parametrize("text", ["null", "[1, 2]", "42"])
    def test_the_same_holds_for_a_state_file(self, bus, text):
        bus.write_builder_state("v8", BuilderState(frontier=1))
        bus.builder_state_path("v8").write_text(text)
        with pytest.raises(BusError, match="not an object"):
            bus.read_builder_state("v8")

    def test_a_consumer_survives_all_of_them(self):
        from slipstream.consumer import TRANSPORT_ERRORS

        assert issubclass(BusError, TRANSPORT_ERRORS)

    def test_gc_reclaims_a_leaked_state_tmp_file(self, bus):
        bus.write_builder_state("v8", BuilderState(frontier=1))
        leaked = bus.builder_state_path("v8").with_name("v8.json.tmp-9-abc")
        leaked.write_text("{}")
        assert leaked in bus.gc()
        assert bus.read_builder_state("v8").frontier == K1(1)


class TestKeySpellingOnDisk:
    """v8 and jsc keep writing exactly what they wrote before the key grew an
    embedder coordinate; a composite key gets the ``e-c`` spelling in names
    and the ``[e, c]`` pair in state, and both sort as one series."""

    def test_a_scalar_key_leaves_every_file_as_it_was(self, bus, tmp_path):
        _publish(bus, 109680)
        assert bus.entry_path("v8", 109680).name == "109680.json"
        assert (
            json.loads(bus.entry_path("v8", 109680).read_text())["commit_id"] == 109680
        )

        p = cursor_path(tmp_path, "box2", "v8")
        write_cursor(p, 109680)
        assert p.read_text() == "109680\n"

        bus.write_builder_state("v8", BuilderState(frontier=109680))
        raw = json.loads(bus.builder_state_path("v8").read_text())
        assert raw["frontier"] == 109680 and raw["lowest_retained"] is None

    def test_an_entry_written_before_the_field_existed_reads_as_embedder_zero(
        self, bus
    ):
        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = json.loads(path.read_text())
        del data["embedder_id"]
        path.write_text(json.dumps(data))
        assert bus.read_entry("v8", 100).key == K1(100)

    def test_a_composite_key_round_trips_through_names_and_state(self, bus, tmp_path):
        from slipstream.models import CommitKey

        key = CommitKey(1534000, 109680)
        _publish(bus, key.commit_id, engine="chrome", embedder_id=key.embedder_id)

        assert bus.entry_path("chrome", key).name == "1534000-109680.json"
        assert bus.keys("chrome") == [key]
        assert bus.read_entry("chrome", key).key == key

        p = cursor_path(tmp_path, "box2", "chrome")
        write_cursor(p, key)
        assert p.read_text() == "1534000-109680\n" and read_cursor(p) == key

        bus.write_bench_state("chrome", BenchState(bot="b", cursor=key, lag=0))
        raw = json.loads(bus.bench_state_path("chrome").read_text())
        assert raw["cursor"] == [1534000, 109680]
        assert bus.read_bench_state("chrome").cursor == key

    def test_keys_sort_by_embedder_then_commit(self, bus):
        from slipstream.models import CommitKey

        for key in (CommitKey(2, 5), CommitKey(1, 900), CommitKey(1, 10)):
            _publish(
                bus,
                key.commit_id,
                _bytes(key.commit_id, 1),
                engine="chrome",
                embedder_id=key.embedder_id,
            )
        assert bus.keys("chrome") == [
            CommitKey(1, 10),
            CommitKey(1, 900),
            CommitKey(2, 5),
        ]
        assert bus.keys_above("chrome", CommitKey(1, 10)) == [
            CommitKey(1, 900),
            CommitKey(2, 5),
        ]
        # Retention walks the same order: the lowest key goes first.
        assert bus.prune("chrome", 1) == [CommitKey(1, 10), CommitKey(1, 900)]

    def test_a_state_file_with_a_bad_key_is_a_bus_error(self, bus):
        bus.write_builder_state("v8", BuilderState(frontier=1))
        path = bus.builder_state_path("v8")
        data = json.loads(path.read_text())
        data["frontier"] = "not-a-key"
        path.write_text(json.dumps(data))
        with pytest.raises(BusError):
            bus.read_builder_state("v8")


class TestEmbedderFields:
    """``embedder`` and ``pins`` describe how an embedded engine's entry was
    produced; an engine built from its own checkout leaves them empty, and
    entries from before the fields existed read back the same way."""

    def test_round_trip(self, bus):
        from slipstream.models import CommitKey

        key = CommitKey(1534000, 109680)
        _publish(
            bus,
            key.commit_id,
            engine="chrome",
            embedder_id=key.embedder_id,
            embedder={"hash": "cr" * 20, "commit_id": 1534000, "title": "Roll V8"},
            pins={"src/v8": "hash109680"},
        )
        read = bus.read_entry("chrome", key)
        assert read.embedder["hash"] == "cr" * 20
        assert read.embedder_hash == "cr" * 20
        assert read.pins == {"src/v8": "hash109680"}

    def test_absent_fields_read_as_empty(self, bus):
        _publish(bus, 100)
        path = bus.entry_path("v8", 100)
        data = json.loads(path.read_text())
        del data["embedder"], data["pins"]
        path.write_text(json.dumps(data))
        read = bus.read_entry("v8", 100)
        assert read.embedder == {} and read.pins == {} and read.embedder_hash == ""
