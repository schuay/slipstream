# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import copy
import hashlib
import json
import plistlib
import uuid

import pytest

from slipstream import safari_packages as stp
from slipstream.builder import Builder, store_run_set
from slipstream.bus import Bus, BusError, Entry, PermanentEntryError
from slipstream.bus_protocol import BusProtocol
from slipstream.collector import BenchCollector
from slipstream.config import BusConfig, BusSource, EngineConfig
from slipstream.consumer import BusConsumer, LocalSource, ShaMismatch
from slipstream.hostapp import read_app
from slipstream.models import CommitKey
from slipstream.transfer import LocalEndpoint


def metadata(release=252, majors=(26, 27)):
    return {
        "release": release,
        "homebrew_revision": "a" * 40,
        "packages": {
            str(n): {
                "os_major": n,
                "source_url": f"https://secure-appldnld.apple.com/{release}/{n}.dmg",
                "dmg_sha256": hashlib.sha256(f"{release}/{n}".encode()).hexdigest(),
            }
            for n in majors
        },
    }


@pytest.fixture
def acquisition(tmp_path, monkeypatch, store):
    state = {"metadata": metadata(), "downloads": [], "fail": None}

    def resolve(majors, revision=None):
        if state["fail"] == "upstream":
            raise OSError("upstream outage")
        result = copy.deepcopy(state["metadata"])
        result["packages"] = {str(n): result["packages"][str(n)] for n in majors}
        return result

    def download(package, path):
        major = package["os_major"]
        state["downloads"].append(major)
        if state["fail"] == major:
            raise OSError("interrupted download")
        path.write_text(package["dmg_sha256"])

    def extract(dmg, dest, major):
        root = dest / "app"
        contents = root / stp.APP / "Contents"
        contents.mkdir(parents=True)
        (contents / "Info.plist").write_bytes(
            plistlib.dumps(
                {
                    "CFBundleVersion": f"{major}-{state['metadata']['release']}",
                }
            )
        )
        return root

    monkeypatch.setattr(stp, "resolve_packages", resolve)
    monkeypatch.setattr(stp, "download_package", download)
    monkeypatch.setattr(stp, "extract_package", extract)
    bus = Bus(tmp_path / "bus")
    logs = []

    def manager(majors=(26, 27), target=bus):
        return stp.SafariReleases(
            target,
            "safari",
            majors,
            lambda app: store.embedder_number("safari", app.version, app.title),
            logs.append,
        )

    state.update(bus=bus, manager=manager, logs=logs, store=store)
    return state


def entry_for(record, bus, commit=10):
    private = Bus(bus.root, _raw=True)
    private.format_version = 4
    src = bus.root / "test-build"
    release = src / "WebKitBuild/Release"
    release.mkdir(parents=True, exist_ok=True)
    (release / "jsc").write_bytes(b"shared webkit")
    blobs = store_run_set(private, src, ["WebKitBuild/Release"], caffeinate=False)
    return Entry(
        "safari",
        commit,
        f"hash{commit}",
        "",
        1,
        "",
        "cfg",
        blobs,
        embedder_id=record["embedder_id"],
        version=5,
        safari_release=copy.deepcopy(record),
    )


def test_complete_adoption_restart_outage_and_no_publications(acquisition):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    original = copy.deepcopy(m.record)
    assert original["os_majors"] == [26, 27]
    assert a["downloads"] == [26, 27]
    assert len(a["store"].embedder_numbers("safari")) == 1
    assert a["bus"].sweep_blobs() == []
    assert len(a["bus"].adopted_objects()) == 2
    a["fail"] = "upstream"
    restarted = a["manager"]()
    restarted.refresh()
    assert restarted.record == original
    assert restarted.installed()[1] == original["embedder_id"]
    assert "upstream outage" in restarted.last_error


@pytest.mark.parametrize("failure", [26, 27])
def test_interrupted_acquisition_does_not_allocate_or_replace(acquisition, failure):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    original = copy.deepcopy(m.record)
    a["metadata"] = metadata(253)
    a["fail"] = failure
    m.refresh()
    assert m.record == original == stp.read_release(m.path)
    assert len(a["store"].embedder_numbers("safari")) == 1
    a["fail"] = None
    m.refresh()
    assert m.record["embedder_id"] == original["embedder_id"] + 1


def test_interrupted_adoption_reuses_allocated_number(acquisition, monkeypatch):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    original = copy.deepcopy(m.record)
    a["metadata"] = metadata(253)
    write = stp.atomic_write
    monkeypatch.setattr(
        stp, "atomic_write", lambda *args: (_ for _ in ()).throw(OSError("power loss"))
    )
    m.refresh()
    assert m.record == original
    allocated = a["store"].embedder_numbers("safari")["stp-release:253"]
    a["bus"].gc()
    monkeypatch.setattr(stp, "atomic_write", write)
    restarted = a["manager"]()
    restarted.refresh()
    assert restarted.record["embedder_id"] == allocated
    assert len(a["store"].embedder_numbers("safari")) == 2


def test_unseen_lower_release_is_rejected_before_allocation(acquisition):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    a["metadata"] = metadata(251)
    m.refresh()
    assert "rollback" in m.last_error
    assert m.record["release"] == 252
    assert a["downloads"] == [26, 27]
    assert len(a["store"].embedder_numbers("safari")) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_url", "https://secure-appldnld.apple.com/respin.dmg"),
        ("dmg_sha256", "f" * 64),
        ("os_major", 28),
    ],
)
def test_same_release_conflict_keeps_pinned_bytes(acquisition, field, value):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    original = copy.deepcopy(m.record)
    hashes = {
        o.name: a["bus"].object_path(o.name).read_bytes()
        for o in a["bus"].adopted_objects()
    }
    a["metadata"]["packages"]["27"][field] = value
    m.refresh()
    assert "conflict" in m.last_error
    assert m.record == original
    assert a["downloads"] == [26, 27]
    assert all(
        a["bus"].object_path(name).read_bytes() == data for name, data in hashes.items()
    )


def test_active_os_list_changes_only_on_next_release(acquisition):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    old_entry = entry_for(m.record, a["bus"])
    a["bus"].publish(old_entry)
    upgraded = a["manager"]((27,))
    upgraded.refresh()
    assert upgraded.record == m.record
    a["metadata"] = metadata(253, (27,))
    upgraded.refresh()
    assert upgraded.record["os_majors"] == [27]
    assert a["bus"].sweep_blobs() == []  # old publication keeps its 26 package
    a["bus"].entry_path("safari", old_entry.key).unlink()
    removed = a["bus"].sweep_blobs()
    assert len(removed) == 3
    assert len(a["bus"].adopted_objects()) == 1


def test_checksum_failure_blocks_first_adoption(tmp_path, monkeypatch, store):
    # Exercise the real downloader's verification seam, without real network.
    import io

    monkeypatch.setattr(
        stp.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong bytes")
    )
    with pytest.raises(BusError, match="checksum mismatch"):
        stp.download_package(metadata()["packages"]["26"], tmp_path / "download.dmg")


def test_manifest_roundtrip_and_reject_legacy(acquisition):
    a = acquisition
    m = acquisition["manager"]()
    m.refresh()
    entry = entry_for(m.record, a["bus"])
    assert Entry.from_json(entry.to_json()) == entry
    assert len(entry.objects()) == 3
    legacy = copy.deepcopy(entry)
    legacy.version = 4
    with pytest.raises(BusError, match="version 5"):
        Entry.from_json(legacy.to_json())
    legacy.safari_release = {}
    with pytest.raises(PermanentEntryError, match="legacy"):
        stp.select_package(legacy, 27)
    with pytest.raises(BusError, match="no package"):
        stp.select_package(entry, 28)


def test_provision_selects_os_package_and_records_bytes(
    acquisition, config, monkeypatch
):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    entry = entry_for(m.record, a["bus"])
    a["bus"].publish(entry)
    engine = EngineConfig(
        "safari",
        None,
        "",
        "/Applications/Safari.app/Contents/MacOS/SafariForWebKitDevelopment",
        "",
        runtime="safari",
        run_set=[stp.APP],
        derives="jsc",
    )
    config.engines["safari"] = engine
    source = BusSource(name="local", root=str(a["bus"].root), engines=["safari"])
    config.bus = BusConfig(root=a["bus"].root, sources=[source])
    collector = BenchCollector(config, role="watch")
    consumer = BusConsumer(config, collector)
    consumer.set_cursor(source, "safari", CommitKey(1, 9))
    identities = []
    for major in (26, 27):
        monkeypatch.setattr(stp, "host_os_major", lambda: major)
        root = consumer.provision(LocalSource(source), engine, entry)
        assert read_app(root / stp.APP).version == f"{major}-252"
        selected = json.loads((root / "safari-package.json").read_text())
        assert selected == m.record["packages"][str(major)]
        monkeypatch.setattr(collector, "harness_revs", lambda: {})
        provenance = collector.local_provenance(engine, root)
        identities.append(provenance["runner_cfg_hash"])
        assert (
            json.loads(provenance["host_env"])["package_sha256"]
            == selected["blob"]["archive"]["sha256"]
        )
    assert identities[0] != identities[1]
    assert consumer.cursor_file(source, "safari").read_text().strip() == "1-9"
    monkeypatch.setattr(stp, "host_os_major", lambda: 28)
    with pytest.raises(BusError, match="no package"):
        consumer.provision(LocalSource(source), engine, entry)
    assert consumer.cursor_file(source, "safari").read_text().strip() == "1-9"
    monkeypatch.setattr(stp, "host_os_major", lambda: 27)
    a["bus"].object_path(entry.objects()[1].name).write_bytes(b"corrupt")
    with pytest.raises(ShaMismatch):
        consumer.provision(LocalSource(source), engine, entry)
    assert consumer.cursor_file(source, "safari").read_text().strip() == "1-9"


def test_version5_transfer_and_gc_protect_all_packages(acquisition, tmp_path):
    a = acquisition
    src = BusProtocol(tmp_path / "source")
    dst = BusProtocol(tmp_path / "destination")
    owner = src.bus_uuid()
    src.migrate({"safari": owner})
    dst.migrate({"safari": owner})
    bus = Bus(src.root)
    m = a["manager"](target=bus)
    m.refresh()
    entry = entry_for(m.record, bus)
    entry.owner_uuid = owner
    entry.publication_uuid = str(uuid.uuid4())
    with src.stage() as stage:
        stage.entry, stage.text = entry, entry.to_json()
        stage.paths = {o.name: bus.object_path(o.name) for o in entry.objects()}
        src.publish_staged(stage)
    with src.capture("safari", entry.key) as stage:
        LocalEndpoint(dst.root, 1e9).accept(stage)
    assert dst.read_entry("safari", entry.key) == entry
    assert dst.sweep_blobs() == []
    bus.entry_path("safari", entry.key).unlink()
    assert len(src.sweep_blobs()) == 1  # only WebKit is unreferenced
    assert len(bus.adopted_objects()) == 2


def test_migration_allocates_above_retained_and_dropped_history(acquisition, config):
    a = acquisition
    from slipstream.bus import BuilderState

    a["bus"].write_builder_state(
        "safari",
        BuilderState(frontier=CommitKey(42, 10), highest_dropped=CommitKey(41, 9)),
    )
    config.engines["jsc"] = EngineConfig(
        "jsc", None, "", "jsc", "", run_set=["WebKitBuild/Release"]
    )
    config.engines["safari"] = EngineConfig(
        "safari",
        None,
        "",
        "/Safari",
        "",
        runtime="safari",
        derives="jsc",
        run_set=[stp.APP],
    )
    b = Builder(config, a["bus"])
    r = b.resolver("safari")
    r.fetch()
    assert r.installed()[1] == 43
    # A lost database reconstructs the adopted number, without publications.
    a["bus"].write_builder_state(
        "safari",
        BuilderState(frontier=CommitKey(43, 10), highest_dropped=CommitKey(43, 10)),
    )
    b.store.conn.execute("DELETE FROM embedders")
    b.store.conn.commit()
    restarted = Builder(config, a["bus"])
    assert restarted.resolver("safari").installed()[1] == 43
    assert restarted.store.embedder_numbers("safari") == {"stp-release:252": 43}


@pytest.mark.parametrize("release,error", [(253, "rollback"), (254, "conflict")])
def test_stale_resolver_preserves_newer_adopted_record(acquisition, release, error):
    a = acquisition
    stale = a["manager"]()
    stale.refresh()
    other = a["manager"]()
    a["metadata"] = metadata(254)
    other.refresh()
    adopted = copy.deepcopy(other.record)
    downloads = list(a["downloads"])
    numbers = a["store"].embedder_numbers("safari")

    a["metadata"] = metadata(release)
    a["metadata"]["packages"]["27"]["dmg_sha256"] = "f" * 64
    stale.refresh()

    assert error in stale.last_error
    assert stale.record == stp.read_release(stale.path) == adopted
    assert a["downloads"] == downloads
    assert a["store"].embedder_numbers("safari") == numbers


def test_cask_parser_pins_both_packages_and_rejects_mismatched_release():
    def block(scope, release, filename):
        return f'''  on_{scope} :or_newer do
    version "{release},download-id"
    sha256 "{"a" * 64}"
    url "https://secure-appldnld.apple.com/STP/#{{version.csv.second}}/{filename}"
  end\n'''

    text = block("tahoe", 252, "SafariTechnologyPreview.dmg") + block(
        "golden_gate", 252, "SafariTechPreview#{version.csv.first}.dmg"
    )
    result = stp.parse_cask(text, [26, 27], "a" * 40)
    assert result["release"] == 252
    assert result["packages"]["27"]["source_url"].endswith(
        "/download-id/SafariTechPreview252.dmg"
    )
    with pytest.raises(BusError, match="different public releases"):
        stp.parse_cask(
            text.replace('version "252', 'version "253', 1), [26, 27], "a" * 40
        )
    with pytest.raises(BusError, match="no recognized"):
        stp.parse_cask(text.replace("on_tahoe", "on_unknown"), [26, 27], "a" * 40)


@pytest.mark.parametrize("migrated", [False, True])
def test_builder_publishes_shared_webkit_and_both_packages(
    acquisition, config, migrated
):
    a = acquisition
    bus = a["bus"]
    source = bus.root / "source"
    (source / "WebKitBuild/Release").mkdir(parents=True)
    (source / "WebKitBuild/Release/jsc").write_bytes(b"webkit")
    blobs = store_run_set(bus, source, ["WebKitBuild/Release"], caffeinate=False)
    parent = Entry("jsc", 10, "webkit-hash", "", 1, "WebKit", "parent-cfg", blobs)
    bus.publish(parent)
    if migrated:
        protocol = BusProtocol(bus.root)
        owner = protocol.bus_uuid()
        protocol.migrate({"jsc": owner, "safari": owner})
        bus = Bus(bus.root)
    config.engines["jsc"] = EngineConfig(
        "jsc", None, "", "jsc", "", run_set=["WebKitBuild/Release"]
    )
    config.engines["safari"] = EngineConfig(
        "safari",
        None,
        "",
        "/Safari",
        "",
        runtime="safari",
        derives="jsc",
        run_set=[stp.APP],
    )
    b = Builder(config, bus)
    resolver = b.resolver("safari")
    resolver.fetch()
    job = resolver.next_after(CommitKey(0, 0))
    entry = b._package_and_publish(config.engines["safari"], job, 0)
    assert entry.version == 5
    assert entry.embedder_hash == "stp-release:252"
    assert entry.blobs == parent.blobs
    assert len(entry.objects()) == 3
    assert bus.read_entry("safari", entry.key) == entry
    for obj in entry.objects():
        assert stp.sha256_file(bus.object_path(obj.name)) == obj.sha256
    assert b.resolver("safari").next_after(entry.key) is None


@pytest.mark.parametrize("failure", ["missing", "checksum", "legacy"])
def test_drain_classifies_publication_errors(acquisition, config, monkeypatch, failure):
    from slipstream.config import RunSpec

    a = acquisition
    m = a["manager"]()
    m.refresh()
    entry = entry_for(m.record, a["bus"])
    if failure == "legacy":
        entry.safari_release = {}
        entry.version = 4
    a["bus"].publish(entry)
    engine = EngineConfig(
        "safari",
        None,
        "",
        "/Safari",
        "",
        runtime="safari",
        derives="jsc",
        run_set=[stp.APP],
    )
    config.engines["safari"] = engine
    config.runs = [RunSpec(engine="safari", suite="js3")]
    config.bench.min_free_gb = 0.001
    source = BusSource(name="local", root=str(a["bus"].root), engines=["safari"])
    config.bus = BusConfig(root=a["bus"].root, sources=[source])
    collector = BenchCollector(config, role="watch")
    consumer = BusConsumer(config, collector)
    monkeypatch.setattr(
        stp, "host_os_major", lambda: 28 if failure == "missing" else 27
    )
    monkeypatch.setattr(collector, "harness_revs", lambda: {})
    if failure == "checksum":
        a["bus"].object_path(entry.objects()[-1].name).write_bytes(b"corrupt")
    consumer.set_cursor(source, "safari", CommitKey(1, 9))
    result = consumer.drain(source, "safari", lambda: False)
    assert result.benched == 0
    state = a["bus"].read_bench_state("safari")
    if failure == "legacy":
        assert consumer.cursor(source, "safari") == entry.key
        assert state.cursor == entry.key
        assert state.skipped_permanent == entry.key
        assert "legacy" in state.permanent_skip_reason
        assert state.last_error is None
        assert result.error is None
        assert state.consecutive_failures == 0
        return
    assert consumer.cursor_file(source, "safari").read_text().strip() == "1-9"
    assert state.skipped_permanent is None
    assert state.last_error
    assert state.cursor == CommitKey(1, 9)


def test_initial_acquisition_failure_does_not_allocate(acquisition):
    a = acquisition
    a["fail"] = 27
    m = a["manager"]()
    with pytest.raises(BusError, match="no adopted"):
        m.refresh()
    assert not m.path.exists()
    assert a["store"].embedder_numbers("safari") == {}


@pytest.mark.parametrize("value", [[], [26, 26], [True], [28], "26"])
def test_active_os_config_rejects_invalid_sets(value):
    from slipstream.config import _parse_build

    with pytest.raises(ValueError, match="safari_os_majors"):
        _parse_build({"safari_os_majors": value}, {})


def test_restart_after_record_commit_reuses_identity(acquisition, monkeypatch):
    a = acquisition
    m = a["manager"]()
    m.refresh()
    a["metadata"] = metadata(253)
    write = stp.atomic_write

    def committed_then_interrupted(*args):
        write(*args)
        raise OSError("interrupted after rename")

    monkeypatch.setattr(stp, "atomic_write", committed_then_interrupted)
    m.refresh()
    # Even an uncertain write outcome recovers the committed record.
    assert m.record["release"] == 253
    a["fail"] = "upstream"
    restarted = a["manager"]()
    restarted.refresh()
    assert restarted.record == m.record
    assert len(a["store"].embedder_numbers("safari")) == 2


def test_historical_results_raise_initial_allocation_floor(acquisition, config):
    a = acquisition
    config.engines["jsc"] = EngineConfig(
        "jsc", None, "", "jsc", "", run_set=["WebKitBuild/Release"]
    )
    config.engines["safari"] = EngineConfig(
        "safari",
        None,
        "",
        "/Safari",
        "",
        runtime="safari",
        derives="jsc",
        run_set=[stp.APP],
    )
    b = Builder(config, a["bus"])
    b.store.conn.execute(
        "INSERT INTO commits (engine, embedder_id, hash, commit_id) VALUES ('safari', 70, 'old-webkit', 10)"
    )
    b.store.conn.commit()
    resolver = b.resolver("safari")
    resolver.fetch()
    assert resolver.installed()[1] == 71
