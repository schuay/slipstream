# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from slipstream.bus import (
    BenchState,
    Bus,
    BuilderState,
    BusError,
    Entry,
    cursor_path,
    read_cursor,
    sha256_file,
    write_cursor,
)


def _entry(commit_id, engine="v8", **kw):
    return Entry(
        engine=engine,
        commit_id=commit_id,
        hash=f"hash{commit_id}",
        date="2026-09-06",
        timestamp=1757116800,
        title=f"commit {commit_id}",
        build_cfg_hash="sha256:cfg",
        blob_sha256="sha",
        blob_bytes=0,
        builder={"bot": "box2-m4"},
        built_at=1757116999,
        build_secs=1183,
        **kw,
    )


def _publish(bus, commit_id, payload=b"x", engine="v8"):
    tmp = bus.tmp_blob(engine, commit_id)
    tmp.write_bytes(payload)
    entry = _entry(commit_id, engine=engine)
    entry.blob_sha256 = sha256_file(tmp)
    entry.blob_bytes = len(payload)
    bus.publish(entry, tmp)
    return entry


@pytest.fixture
def bus(tmp_path):
    return Bus(tmp_path / "bus")


class TestPublish:
    def test_round_trip(self, bus):
        published = _publish(bus, 109680, b"payload")
        assert bus.commit_ids("v8") == [109680]
        read = bus.read_entry("v8", 109680)
        assert read == published
        assert bus.blob_path("v8", 109680).read_bytes() == b"payload"

    def test_the_payload_lands_before_the_entry(self, bus, monkeypatch):
        """The reverse lets a consumer read an entry whose payload is absent."""
        seen = []
        real = bus.entry_path

        def spy(engine, commit_id):
            seen.append(bus.blob_path(engine, commit_id).exists())
            return real(engine, commit_id)

        monkeypatch.setattr(bus, "entry_path", spy)
        _publish(bus, 100)
        assert seen and all(seen)

    def test_a_republish_reuses_the_name(self, bus):
        """A crashed publish self-heals: the retry writes the same payload name."""
        _publish(bus, 100, b"first")
        _publish(bus, 100, b"second")
        assert bus.blob_path("v8", 100).read_bytes() == b"second"
        assert bus.commit_ids("v8") == [100]

    def test_tmp_files_are_not_listed_as_entries(self, bus):
        _publish(bus, 100)
        (bus.topic_dir("v8") / "101.json.tmp-9-ab").write_text("{}")
        assert bus.commit_ids("v8") == [100]

    def test_engines_are_separate(self, bus):
        _publish(bus, 100, engine="v8")
        _publish(bus, 500, engine="jsc")
        assert bus.commit_ids("v8") == [100]
        assert bus.commit_ids("jsc") == [500]

    def test_an_empty_root_lists_nothing(self, bus):
        assert bus.commit_ids("v8") == []
        assert bus.read_entry("v8", 1) is None


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


class TestCursorSemantics:
    def test_ids_above_the_cursor(self, bus):
        for cid in (100, 200, 300):
            _publish(bus, cid)
        assert bus.ids_above("v8", 100) == [200, 300]
        assert bus.ids_above("v8", None) == [100, 200, 300]
        assert bus.ids_above("v8", 300) == []

    def test_cursor_round_trip(self, tmp_path):
        p = cursor_path(tmp_path, "box2", "v8")
        assert read_cursor(p) is None
        write_cursor(p, 109680)
        assert read_cursor(p) == 109680
        write_cursor(p, 109681)
        assert read_cursor(p) == 109681
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
        assert read_cursor(cursor_path(tmp_path, "local", "v8")) == 1
        assert read_cursor(cursor_path(tmp_path, "box2", "v8")) == 2


class TestState:
    def test_builder_state_round_trip(self, bus):
        state = BuilderState(frontier=109680, lowest_retained=109120)
        state.failed = [{"commit_id": 109655, "kind": "compile", "at": 1}]
        bus.write_builder_state("v8", state)
        read = bus.read_builder_state("v8")
        assert read.frontier == 109680 and read.failed[0]["kind"] == "compile"
        assert read.updated_at > 0

    def test_bench_state_round_trip(self, bus):
        bus.write_bench_state("v8", BenchState(bot="box2-m4", cursor=109679, lag=1))
        read = bus.read_bench_state("v8")
        assert (read.bot, read.cursor, read.lag) == ("box2-m4", 109679, 1)

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
        for cid in (100, 200, 300, 400):
            _publish(bus, cid, b"x" * 100)
        dropped = bus.prune("v8", 250)
        assert dropped == [100, 200]
        assert bus.commit_ids("v8") == [300, 400]
        assert not bus.blob_path("v8", 100).exists()
        assert bus.lowest_retained("v8") == 300

    def test_the_newest_entry_is_never_dropped(self, bus):
        """A budget smaller than one payload must not empty the topic."""
        _publish(bus, 100, b"x" * 1000)
        assert bus.prune("v8", 10) == []
        assert bus.commit_ids("v8") == [100]

    def test_nothing_to_do_under_budget(self, bus):
        for cid in (100, 200):
            _publish(bus, cid, b"x" * 10)
        assert bus.prune("v8", 10_000) == []
        assert bus.blob_bytes("v8") == 20

    def test_gc_reclaims_a_payload_with_no_entry(self, bus):
        _publish(bus, 100)
        _publish(bus, 200)
        bus.entry_path("v8", 100).unlink()  # as a crash mid-prune leaves it
        removed = bus.gc(["v8"])
        assert [p.name for p in removed] == ["100.tar.zst"]
        assert bus.blob_path("v8", 200).exists()

    def test_gc_reclaims_an_abandoned_tmp_payload(self, bus):
        tmp = bus.tmp_blob("v8", 100)
        tmp.write_bytes(b"partial")
        assert bus.gc(["v8"]) == [tmp]
        assert not tmp.exists()

    def test_gc_keeps_everything_referenced(self, bus):
        _publish(bus, 100)
        assert bus.gc(["v8"]) == []
        assert bus.blob_path("v8", 100).exists()


class TestRetentionLeavesNoHole:
    def test_what_survives_is_contiguous(self, bus):
        """A smaller older entry kept below a dropped one would fit more in
        the budget, but it holds lowest_retained down and hides the hole."""
        for cid, size in ((100, 40), (200, 100), (300, 200)):
            _publish(bus, cid, b"x" * size)
        assert bus.prune("v8", 250) == [100, 200]
        assert bus.commit_ids("v8") == [300]
        assert bus.lowest_retained("v8") == 300

    def test_the_entry_just_published_is_kept(self, bus):
        """build --retry republishes below the frontier, so its entry is the
        oldest and the same cycle would otherwise delete it."""
        for cid in (100, 200, 300):
            _publish(bus, cid, b"x" * 100)
        bus.prune("v8", 250)
        _publish(bus, 50, b"x" * 100)
        assert bus.prune("v8", 250, keep=50) == []
        assert 50 in bus.commit_ids("v8")

    def test_keep_does_not_protect_an_unrelated_entry(self, bus):
        for cid in (100, 200, 300):
            _publish(bus, cid, b"x" * 100)
        assert bus.prune("v8", 250, keep=300) == [100]

    def test_keep_lowers_the_floor_rather_than_punching_a_hole(self, bus):
        """Exempting one entry would leave a gap below lowest_retained, which
        is the signal a consumer uses to notice entries went missing."""
        for cid in (100, 200, 201, 202):
            _publish(bus, cid, b"x" * 100)
        assert bus.prune("v8", 250, keep=100) == []
        assert bus.commit_ids("v8") == [100, 200, 201, 202]
        assert bus.lowest_retained("v8") == 100

    def test_the_overshoot_lasts_one_cycle(self, bus):
        for cid in (100, 200, 201, 202):
            _publish(bus, cid, b"x" * 100)
        bus.prune("v8", 250, keep=100)
        # The next publish prunes normally and drops it with its neighbours.
        _publish(bus, 203, b"x" * 100)
        assert bus.prune("v8", 250, keep=203) == [100, 200, 201]
        assert bus.commit_ids("v8") == [202, 203]


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
        assert bus.gc(["v8"]) == [partial]
        assert not partial.exists()

    def test_gc_reclaims_a_leaked_entry_tmp_file(self, bus):
        """A crash between writing the entry tmp and renaming it leaks one per
        crash; commit_ids ignores them and nothing else looked."""
        _publish(bus, 100)
        leaked = bus.topic_dir("v8") / "101.json.tmp-9-abc"
        leaked.write_text("{}")
        assert leaked in bus.gc(["v8"])
        assert not leaked.exists()
        assert bus.commit_ids("v8") == [100]

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
        assert leaked in bus.gc(["v8"])
        assert bus.read_builder_state("v8").frontier == 1
