# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from slipstream import archive as archive_mod
from slipstream.archive import (
    ArchiveError,
    Archiver,
    archive_cursor,
    next_run_at,
    read_state,
    retry_delay,
)
from slipstream.bus import Blob, Delta, Entry
from slipstream.bus_protocol import BusProtocol, digest
from slipstream.config import ArchiveConfig
from slipstream.delta import Block, DeltaPlan
from slipstream.models import CommitKey


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def publish(protocol, key, files, *, engine="v8", release=None):
    """Publish an entry whose objects are ``files`` (object name -> bytes)."""
    blobs = []
    payloads = {}
    for path, spec in files.items():
        if isinstance(spec, bytes):
            blobs.append(Blob(path, f"tree-{_sha(spec)}", _sha(spec), len(spec)))
            payloads[f"{_sha(spec)}.tar.zst"] = spec
        else:  # (base bytes, [patch bytes, ...])
            base, patches = spec
            plan = DeltaPlan(
                1 << 20,
                _sha(b"tar" + b"".join(patches)),
                len(patches),
                tuple(Block(_sha(p + b"t"), 1, _sha(p), len(p)) for p in patches),
            )
            blobs.append(
                Blob(
                    path,
                    f"tree-delta-{key}",
                    delta=Delta(f"tree-{_sha(base)}", _sha(base), len(base), plan),
                )
            )
            payloads[f"{_sha(base)}.tar.zst"] = base
            payloads.update({f"{_sha(p)}.bsdiff": p for p in patches})
    if release is not None:
        payloads[f"{_sha(release)}.tar.zst"] = release
    entry = Entry(
        engine,
        key,
        f"hash{key}",
        "",
        key,
        "",
        "cfg",
        blobs,
        built_at=int(time.time()) + key,
        version=5 if release is not None else 4,
        owner_uuid=protocol.ready()["uuid"],
        publication_uuid=str(uuid.uuid4()),
        safari_release=_release_record(release) if release is not None else {},
        embedder_id=7 if release is not None else 0,
    )
    with protocol.stage() as stage:
        for obj in entry.objects():
            path = stage.path / obj.name
            path.write_bytes(payloads[obj.name])
            stage.paths[obj.name] = path
        stage.entry, stage.text = entry, entry.to_json()
        protocol.publish_staged(stage)
    return entry


def _release_record(package):
    from slipstream.safari_packages import APP

    return {
        "schema": 1,
        "release": 252,
        "embedder_id": 7,
        "homebrew_revision": "a" * 40,
        "os_majors": [26],
        "packages": {
            "26": {
                "os_major": 26,
                "app_build": "26-252",
                "source_url": "https://secure-appldnld.apple.com/252/26.dmg",
                "dmg_sha256": "b" * 64,
                "blob": Blob(APP, "c" * 64, _sha(package), len(package)).to_json(),
            }
        },
    }


@pytest.fixture
def protocol(tmp_path):
    protocol = BusProtocol(tmp_path / "bus")
    owner = protocol.bus_uuid()
    protocol.migrate({"v8": owner, "safari": owner})
    return protocol


class FakeRsync:
    """Records rsync calls and copies like ``-r --ignore-existing`` would."""

    def __init__(self):
        self.calls = []
        self.fail = {}  # destination suffix -> exception, once

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        src, dest = command[-2], command[-1]
        if "--dry-run" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        for suffix, message in list(self.fail.items()):
            if dest.endswith(suffix):
                del self.fail[suffix]
                return subprocess.CompletedProcess(command, 23, "", message)
        src, dest = Path(src), Path(dest)
        for path in sorted(src.rglob("*")):
            target = dest / path.relative_to(src)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        return subprocess.CompletedProcess(command, 0, "", "")

    def uploads(self):
        return [c[-1].rsplit("/", 2)[-2] for c in self.calls if "--dry-run" not in c]


def make(protocol, target, *, engines=("v8",), runner=None, **kw):
    cfg = SimpleNamespace(
        bus=SimpleNamespace(
            root=protocol.root,
            archive=ArchiveConfig(str(target), list(engines), **kw),
        )
    )
    logs = []
    archiver = Archiver(cfg, log=logs.append, runner=runner or FakeRsync())
    archiver.logs = logs
    return archiver


@pytest.fixture
def target(tmp_path):
    path = tmp_path / "nas"
    path.mkdir()
    return path


def test_layout_shards_objects_and_names_manifests_by_their_bytes(protocol, target):
    first = publish(protocol, 1, {"out": b"base"})
    second = publish(protocol, 2, {"out": (b"base", [b"p1", b"p2"])})
    stp = publish(
        protocol, 3, {"WebKitBuild": b"webkit"}, engine="safari", release=b"app"
    )
    archiver = make(protocol, target, engines=("v8", "safari"))
    assert archiver.run(once=True)
    for entry in (first, second, stp):
        for obj in entry.objects():
            path = target / "objects" / obj.name[:2] / obj.name
            assert path.read_bytes() and _sha(path.read_bytes()) == obj.sha256
        text = protocol.bus.entry_path(entry.engine, entry.key).read_bytes()
        manifest = (
            target / "manifests" / entry.engine / str(entry.key) / f"{_sha(text)}.json"
        )
        assert manifest.read_bytes() == text
    # The release package the Safari entry embeds is archived with it.
    assert (target / "objects" / _sha(b"app")[:2] / f"{_sha(b'app')}.tar.zst").exists()
    assert archive_cursor(protocol, "v8") == CommitKey.of(2)
    assert archive_cursor(protocol, "safari") == CommitKey(7, 3)
    assert not list((protocol.root / "staging").glob("*"))


def test_objects_are_uploaded_before_manifests(protocol, target):
    publish(protocol, 1, {"out": b"one"})
    runner = FakeRsync()
    assert make(protocol, target, runner=runner).run(once=True)
    assert runner.uploads() == ["objects", "manifests"]
    command = runner.calls[-1]
    assert "--ignore-existing" in command and "--fsync" in command
    assert "--inplace" not in command and "--partial" not in command
    assert command[command.index("-e") + 1] == "ssh -o BatchMode=yes"


def test_bwlimit_is_passed_through(protocol, target):
    publish(protocol, 1, {"out": b"one"})
    runner = FakeRsync()
    make(protocol, target, runner=runner, bwlimit="50m").run(once=True)
    assert all("--bwlimit=50m" in c for c in runner.calls)


def test_remote_rsync_path_is_passed_through(protocol, target):
    publish(protocol, 1, {"out": b"one"})
    runner = FakeRsync()
    archiver = make(protocol, target, runner=runner, remote_rsync_path="/opt/bin/rsync")
    archiver.probe()
    assert archiver.run(once=True)
    assert len(runner.calls) >= 3  # probes, objects, manifests
    assert all("--rsync-path=/opt/bin/rsync" in c for c in runner.calls)


def test_no_rsync_path_by_default(protocol, target):
    publish(protocol, 1, {"out": b"one"})
    runner = FakeRsync()
    make(protocol, target, runner=runner).run(once=True)
    assert not any(a.startswith("--rsync-path") for c in runner.calls for a in c)


def test_a_crash_after_the_objects_repeats_cleanly_and_advances_once(protocol, target):
    publish(protocol, 1, {"out": b"one"})
    publish(protocol, 2, {"out": b"two"})
    runner = FakeRsync()
    runner.fail["/manifests/"] = "connection reset"
    archiver = make(protocol, target, runner=runner)
    assert not archiver.run(once=True)
    state = read_state(protocol, "v8")
    assert state["cursor"] is None and "connection reset" in state["last_error"]
    assert state["failures"] == 1 and state["retry_after"] > time.time()
    assert list((target / "objects").rglob("*.tar.zst"))
    assert not (target / "manifests").exists()

    assert archiver.run(once=True)
    state = read_state(protocol, "v8")
    assert state["cursor"] == "2" and state["last_error"] is None
    assert state["failures"] == 0 and state["retry_after"] == 0
    assert len(list((target / "manifests").rglob("*.json"))) == 2
    # Nothing new: the cursor stays and nothing is uploaded.
    calls = len(runner.calls)
    assert archiver.run(once=True)
    assert read_state(protocol, "v8")["cursor"] == "2"
    assert [c for c in runner.calls[calls:] if "--dry-run" not in c] == []


def test_later_runs_archive_only_what_is_new(protocol, target):
    publish(protocol, 1, {"out": b"one"})
    archiver = make(protocol, target)
    assert archiver.run(once=True)
    publish(protocol, 2, {"out": b"two"})
    assert archiver.run_engine("v8")
    assert archive_cursor(protocol, "v8") == CommitKey.of(2)
    assert any("archived 1 entries through 2" in m for m in archiver.logs)


def test_a_retention_gap_with_evidence_is_skipped_and_recorded(protocol, target):
    for key in (1, 2, 3):
        publish(protocol, key, {"out": f"build {key}".encode()})
    protocol.prune(0)  # drops all but the newest
    protocol.sweep_blobs()
    archiver = make(protocol, target)
    assert archiver.run(once=True)
    state = read_state(protocol, "v8")
    assert state["cursor"] == "3" and state["skipped_dropped"] == "2"
    assert any("dropped entries through 2" in m for m in archiver.logs)
    assert [p.parent.name for p in (target / "manifests" / "v8").rglob("*.json")] == [
        "3"
    ]


def test_a_vanished_entry_without_evidence_stops_the_engine(
    protocol, target, monkeypatch
):
    publish(protocol, 1, {"out": b"one"})
    publish(protocol, 2, {"out": b"two"})
    real = archive_mod.manifest_text

    def vanishing(path):
        if path.name.startswith("1"):
            raise FileNotFoundError(path)
        return real(path)

    monkeypatch.setattr(archive_mod, "manifest_text", vanishing)
    archiver = make(protocol, target)
    assert not archiver.run(once=True)
    state = read_state(protocol, "v8")
    assert state["cursor"] is None
    assert "without retention evidence" in state["last_error"]


def test_a_changed_target_refuses_to_start(protocol, target, tmp_path):
    publish(protocol, 1, {"out": b"one"})
    assert make(protocol, target).run(once=True)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ArchiveError, match="configured target"):
        make(protocol, other).run(once=True)


def test_a_failed_probe_refuses_to_start(protocol, target):
    def broken(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "unknown option --fsync")

    with pytest.raises(ArchiveError, match="unknown option --fsync"):
        make(protocol, target, runner=broken).run(once=True)


def test_an_engine_owned_elsewhere_is_refused(tmp_path, target):
    protocol = BusProtocol(tmp_path / "bus")
    protocol.bus_uuid()
    protocol.migrate({"v8": str(uuid.uuid4())})
    with pytest.raises(ArchiveError, match="not owned"):
        make(protocol, target)


def test_an_unreadable_state_file_is_an_error(protocol, target):
    path = protocol.state_path("archive", "v8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{")
    with pytest.raises(ArchiveError, match="unreadable"):
        archive_cursor(protocol, "v8")


class TestSchedule:
    def _cfg(self, **kw):
        return ArchiveConfig("/t", ["v8"], **kw)

    def _at(self, text):
        return datetime.fromisoformat(text).timestamp()

    def test_without_at_the_first_run_is_now(self):
        now = self._at("2026-11-27 15:00")
        assert next_run_at(self._cfg(), None, now) == now

    def test_without_at_it_runs_every_hours_after_the_last_start(self):
        last = self._at("2026-11-27 15:00")
        state = {"last_success_at": last}
        assert (
            next_run_at(self._cfg(every_hours=6), state, last + 60) == last + 6 * 3600
        )

    def test_with_at_the_first_run_is_at_today(self):
        now = self._at("2026-11-27 01:00")
        assert next_run_at(self._cfg(at="03:00"), None, now) == self._at(
            "2026-11-27 03:00"
        )

    def test_with_at_a_late_run_does_not_push_the_next_one_back(self):
        state = {"last_success_at": self._at("2026-11-27 05:30")}
        assert next_run_at(
            self._cfg(at="03:00"), state, self._at("2026-11-27 06:00")
        ) == self._at("2026-11-28 03:00")

    def test_weekly(self):
        state = {"last_success_at": self._at("2026-11-27 03:00")}
        assert next_run_at(
            self._cfg(at="03:00", every_hours=168), state, self._at("2026-11-28 03:00")
        ) == self._at("2026-12-04 03:00")

    def test_a_weekly_run_that_succeeds_after_midnight_does_not_drift(self):
        # The 03:00 slot failed and retried until 00:30 the next day.
        state = {"last_success_at": self._at("2026-11-28 00:30")}
        assert next_run_at(
            self._cfg(at="03:00", every_hours=168), state, self._at("2026-11-28 01:00")
        ) == self._at("2026-12-04 03:00")

    def test_a_failure_retries_with_backoff(self):
        state = {
            "last_success_at": self._at("2026-11-27 03:00"),
            "last_error": "boom",
            "retry_after": self._at("2026-11-27 03:05"),
        }
        assert next_run_at(
            self._cfg(at="03:00"), state, self._at("2026-11-27 03:01")
        ) == self._at("2026-11-27 03:05")

    def test_backoff_doubles_from_a_minute_to_an_hour(self):
        assert [retry_delay(n) for n in (1, 2, 3, 7, 8, 20)] == [
            60,
            120,
            240,
            3600,
            3600,
            3600,
        ]

    def test_the_daemon_runs_what_is_due(self, protocol, target, monkeypatch):
        publish(protocol, 1, {"out": b"one"})
        archiver = make(protocol, target, every_hours=24)
        ran = []
        stops = iter([False, False, True])
        monkeypatch.setattr(archiver, "run_engine", lambda e: ran.append(e) or True)
        monkeypatch.setattr(archiver, "wait", lambda secs, stop: None)
        archiver.run(should_stop=lambda: next(stops))
        assert ran == ["v8"]


def _rsync_supports_fsync():
    if shutil.which("rsync") is None:
        return False
    with_fsync = subprocess.run(
        ["rsync", "--fsync", "--version"], capture_output=True, text=True
    )
    return with_fsync.returncode == 0


@pytest.mark.skipif(not _rsync_supports_fsync(), reason="needs rsync with --fsync")
def test_a_real_rsync_round_trip_into_a_local_directory(protocol, target):
    publish(protocol, 1, {"out": b"base"})
    entry = publish(protocol, 2, {"out": (b"base", [b"p1"])})
    archiver = make(protocol, target, runner=subprocess.run)
    assert archiver.run(once=True)
    assert archiver.run(once=True)  # idempotent
    text = protocol.bus.entry_path("v8", 2).read_bytes()
    manifest = target / "manifests" / "v8" / "2" / f"{digest(text.decode())}.json"
    assert manifest.read_bytes() == text
    assert json.loads(manifest.read_text())["commit_id"] == 2
    for obj in entry.objects():
        path = target / "objects" / obj.name[:2] / obj.name
        assert _sha(path.read_bytes()) == obj.sha256
        assert path.stat().st_mode & 0o777 == 0o444


def test_a_failure_never_moves_the_cursor_back(protocol, target, monkeypatch):
    publish(protocol, 1, {"out": b"one"})
    archiver = make(protocol, target)
    assert archiver.run(once=True)
    publish(protocol, 2, {"out": b"two"})
    other = make(protocol, target)
    real = archiver.rsync

    def overlapping(*args):
        real(*args)
        if args[-1].endswith("/manifests/"):
            assert other.run_engine("v8")  # another writer finishes first

    monkeypatch.setattr(archiver, "rsync", overlapping)
    assert not archiver.run_engine("v8")
    state = read_state(protocol, "v8")
    assert state["cursor"] == "2" and state["last_success_at"]
    assert "cursor moved" in state["last_error"]


def test_the_daemon_survives_an_unreadable_state_file(protocol, target, monkeypatch):
    publish(protocol, 1, {"out": b"one"})
    archiver = make(protocol, target)
    path = protocol.state_path("archive", "v8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{")
    assert not archiver.run_engine("v8")
    calls = []
    monkeypatch.setattr(archiver, "wait", lambda secs, stop: None)
    monkeypatch.setattr(archiver, "check_targets", lambda: None)
    archiver.run(should_stop=lambda: calls.append(1) or len(calls) > 3)
    assert any("unreadable archive state" in str(m) for m in archiver.logs)


def test_the_backoff_holds_when_the_failure_cannot_be_recorded(
    protocol, target, monkeypatch
):
    publish(protocol, 1, {"out": b"one"})
    runner = FakeRsync()
    runner.fail["/objects/"] = "connection reset"
    archiver = make(protocol, target, runner=runner)

    def unwritable(engine, state):
        raise OSError("disk full")

    monkeypatch.setattr(archiver, "_save_locked", unwritable)
    assert not archiver.run_engine("v8")
    now = time.time()
    assert archiver.due(now)["v8"] >= now + 50
