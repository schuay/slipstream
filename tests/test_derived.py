# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""A derived engine's series: the inner topic times the installed app."""

from __future__ import annotations

import plistlib

import pytest

from slipstream.bus import Blob, Bus, Entry
from slipstream.config import EngineConfig
from slipstream.hostapp import HostApp, HostAppError, read_app
from slipstream.models import CommitKey
from slipstream.resolve import DerivedResolver

RUNTIME = ["WebKitBuild/Release/jsc", "WebKitBuild/Release/WebKit.framework"]


def _jsc_entry(commit_id, paths=RUNTIME, cfg="sha256:jsc"):
    return Entry(
        engine="jsc",
        commit_id=commit_id,
        hash=f"hash{commit_id}",
        date="2026-10-03",
        timestamp=1790000000 + commit_id,
        title=f"webkit {commit_id}",
        build_cfg_hash=cfg,
        blobs=[Blob(p, f"id-{commit_id}-{i}", "sha", 1) for i, p in enumerate(paths)],
    )


def _app(version, short="27.0"):
    return HostApp("Safari Technology Preview", version, short)


class Numbers:
    """What the store does: first distinct hash is 1, then 2, ..."""

    def __init__(self):
        self.seen: dict[str, int] = {}

    def __call__(self, app):
        return self.seen.setdefault(app.version, len(self.seen) + 1)


@pytest.fixture
def world(tmp_path):
    bus = Bus(tmp_path / "bus")
    jsc = EngineConfig(
        "jsc", None, "make", "WebKitBuild/Release/jsc", "", run_set=RUNTIME
    )
    safari = EngineConfig(
        "safari",
        tmp_path / "Applications",
        "",
        "/Applications/Safari.app/Contents/MacOS/SafariForWebKitDevelopment",
        "",
        run_set=["Safari Technology Preview.app"],
        derives="jsc",
    )
    for c in (10, 11, 12):
        bus.publish(_jsc_entry(c))
    state = {"app": _app("22626.1.8.19.2")}
    resolver = DerivedResolver(
        bus, safari, jsc, number_for=Numbers(), installed=lambda: state["app"]
    )
    resolver.installed()  # the app the series so far was published under: 1
    return bus, resolver, state


class TestDerivedResolver:
    def test_the_first_job_is_the_first_inner_entry_above_from(self, world):
        bus, r, _ = world
        job = r.next_after(CommitKey(0, 10))
        assert job.key == CommitKey(1, 11)
        assert not job.compiles and job.checkout_hash == ""
        assert job.inherits.key == CommitKey(0, 11)
        assert job.commit["hash"] == "hash11" and job.commit["title"] == "webkit 11"
        assert job.embedder == {
            "hash": "22626.1.8.19.2",
            "commit_id": 1,
            "title": "Safari Technology Preview 22626.1.8.19.2 (27.0)",
        }

    def test_walks_the_inner_topic_under_one_app(self, world):
        _, r, _ = world
        assert r.next_after(CommitKey(1, 11)).key == CommitKey(1, 12)
        assert r.next_after(CommitKey(1, 12)) is None

    def test_a_new_app_is_first_a_step_on_the_frontiers_own_commit(self, world):
        _, r, state = world
        state["app"] = _app("22627.1.1.1.1")
        job = r.next_after(CommitKey(1, 11))
        assert job.key == CommitKey(2, 11)
        assert job.embedder["hash"] == "22627.1.1.1.1"
        # and then the topic continues under the new number
        assert r.next_after(CommitKey(2, 11)).key == CommitKey(2, 12)

    def test_the_app_step_lands_on_the_next_entry_if_retention_took_the_commit(
        self, world
    ):
        bus, r, state = world
        bus.entry_path("jsc", CommitKey(0, 11)).unlink()
        state["app"] = _app("22627.1.1.1.1")
        assert r.next_after(CommitKey(1, 11)).key == CommitKey(2, 12)

    def test_an_older_app_than_the_frontier_is_an_error_not_a_step_back(self, world):
        _, r, state = world
        r.number_for(_app("new"))  # number 2 exists; the installed one is 1
        state["app"] = _app("22626.1.8.19.2")
        with pytest.raises(ValueError, match="does not go backwards"):
            r.next_after(CommitKey(2, 12))

    def test_only_inner_entries_with_the_whole_run_set_qualify(self, world):
        bus, r, _ = world
        bus.publish(_jsc_entry(13, paths=["WebKitBuild/Release/jsc"]))  # shell only
        bus.publish(_jsc_entry(14))
        assert r.next_after(CommitKey(1, 12)).key == CommitKey(1, 14)

    def test_a_migrated_version_one_entry_never_qualifies(self, world):
        bus, r, _ = world
        bus.publish(_jsc_entry(13, paths=[""]))
        assert r.next_after(CommitKey(1, 12)) is None

    def test_for_key_only_under_the_installed_app(self, world):
        _, r, state = world
        assert r.for_key(CommitKey(1, 12)).key == CommitKey(1, 12)
        assert r.for_key(CommitKey(1, 99)) is None
        state["app"] = _app("22627.1.1.1.1")
        assert r.for_key(CommitKey(1, 12)) is None  # app 1 is gone from the disk
        assert r.for_key(CommitKey(2, 12)).key == CommitKey(2, 12)

    def test_the_app_is_read_from_src_dir_and_the_run_set(self, tmp_path):
        apps = tmp_path / "Applications"
        bundle = apps / "Safari Technology Preview.app" / "Contents"
        bundle.mkdir(parents=True)
        with open(bundle / "Info.plist", "wb") as f:
            plistlib.dump(
                {
                    "CFBundleName": "Safari Technology Preview",
                    "CFBundleVersion": "22626.1.8.19.2",
                    "CFBundleShortVersionString": "27.0",
                },
                f,
            )
        safari = EngineConfig(
            "safari",
            apps,
            "",
            "/x",
            "",
            run_set=["Safari Technology Preview.app"],
            derives="jsc",
        )
        jsc = EngineConfig("jsc", None, "make", "jsc", "", run_set=RUNTIME)
        r = DerivedResolver(Bus(tmp_path / "bus"), safari, jsc, number_for=Numbers())
        app, number = r.installed()
        assert (app.version, app.short, number) == ("22626.1.8.19.2", "27.0", 1)

    def test_no_src_dir_says_what_to_set(self, tmp_path):
        safari = EngineConfig(
            "safari",
            None,
            "",
            "/x",
            "",
            run_set=["Safari Technology Preview.app"],
            derives="jsc",
        )
        jsc = EngineConfig("jsc", None, "make", "jsc", "", run_set=RUNTIME)
        r = DerivedResolver(Bus(tmp_path / "bus"), safari, jsc, number_for=Numbers())
        with pytest.raises(ValueError, match="set it to the directory holding"):
            r.installed()


class TestReadApp:
    def test_a_missing_bundle_and_a_missing_version_are_errors(self, tmp_path):
        with pytest.raises(HostAppError, match="no application bundle"):
            read_app(tmp_path / "Nope.app")
        contents = tmp_path / "Thing.app" / "Contents"
        contents.mkdir(parents=True)
        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump({"CFBundleShortVersionString": "1.0"}, f)
        with pytest.raises(HostAppError, match="no CFBundleVersion"):
            read_app(tmp_path / "Thing.app")

    def test_the_name_falls_back_to_the_bundle_stem(self, tmp_path):
        contents = tmp_path / "Thing.app" / "Contents"
        contents.mkdir(parents=True)
        with open(contents / "Info.plist", "wb") as f:
            plistlib.dump({"CFBundleVersion": "7"}, f)
        app = read_app(tmp_path / "Thing.app")
        assert app == HostApp("Thing", "7", "")
        assert app.title == "Thing 7"
