# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Pinned, OS-independent STP acquisition. Never executes an installer.

The adopted record is a GC root, independent of publication retention. Objects
are durable before allocation; the complete record is durable before a resolver
can expose its identity. The machine lock serializes acquisition and allocation;
the bus lock serializes the short installation/record commit against GC.
"""

from __future__ import annotations

import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile
import urllib.request
from contextlib import nullcontext
from pathlib import Path

from .bus import Blob, Bus, BusError, sha256_file
from .durability import atomic_write, durable_mkdir, flush_file, fsync_dir
from .hostapp import HostApp, read_app

APP = "Safari Technology Preview.app"
CASK = "Casks/s/safari-technology-preview.rb"
REPO = "Homebrew/homebrew-cask"
SCOPES = {26: "tahoe", 27: "golden_gate"}


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def validate_release(record):
    try:
        assert record["schema"] == 1
        assert type(record["release"]) is int and record["release"] > 0
        assert type(record["embedder_id"]) is int and record["embedder_id"] > 0
        assert re.fullmatch(r"[0-9a-f]{40}", record["homebrew_revision"])
        majors = record["os_majors"]
        assert isinstance(majors, list) and majors
        assert all(type(n) is int and n > 0 for n in majors)
        assert len(set(majors)) == len(majors)
        assert set(record["packages"]) == {str(n) for n in majors}
        for major, package in record["packages"].items():
            assert package["os_major"] == int(major)
            assert package["app_build"] and isinstance(package["app_build"], str)
            assert package["source_url"].startswith(
                "https://secure-appldnld.apple.com/"
            )
            assert _sha(package["dmg_sha256"])
            blob = Blob.from_json(package["blob"], "Safari release")
            assert blob.path == APP and blob.delta is None
            assert _sha(blob.id) and _sha(blob.sha256)
            assert type(blob.bytes) is int and blob.bytes > 0
    except (AssertionError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise BusError("malformed adopted Safari release") from exc


def read_release(path):
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise BusError(f"unreadable Safari release {path}: {exc}") from exc
    validate_release(record)
    return record


def release_blobs(record):
    if not record:
        return []
    validate_release(record)
    return [
        Blob.from_json(p["blob"], "Safari release") for p in record["packages"].values()
    ]


def host_os_major():
    version = platform.mac_ver()[0]
    if not version:
        raise BusError("Safari packages require a macOS host")
    return int(version.split(".")[0])


def select_package(entry, major):
    if entry.version != 5 or not entry.safari_release:
        raise BusError(
            "incompatible legacy Safari publication; requires OS packages (version 5)"
        )
    validate_release(entry.safari_release)
    try:
        return entry.safari_release["packages"][str(major)]
    except KeyError:
        raise BusError(
            f"Safari release {entry.safari_release['release']} has no package for macOS {major}; commit remains pending"
        ) from None


def _read_url(url):
    request = urllib.request.Request(url, headers={"User-Agent": "slipstream"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_cask(text, majors, revision):
    """Parse only the known declarative cask shape; never evaluate Ruby."""
    packages = {}
    releases = set()
    for major in majors:
        scope = SCOPES[major]
        match = re.search(
            rf"^  on_{scope}[^\n]* do\s*\n"
            r'\s*version "(\d+),([^"\n]+)"\s*\n'
            r'\s*sha256 "([0-9a-f]{64})"\s*\n'
            r'\s*url "([^"\n]+)"',
            text,
            re.M,
        )
        if not match:
            raise BusError(f"Homebrew metadata has no recognized macOS {major} package")
        release, download_id, sha, url = match.groups()
        url = url.replace("#{version.csv.second}", download_id).replace(
            "#{version.csv.first}", release
        )
        if "#{" in url or not url.startswith("https://secure-appldnld.apple.com/"):
            raise BusError("unrecognized Safari download URL")
        releases.add(int(release))
        packages[str(major)] = {"os_major": major, "source_url": url, "dmg_sha256": sha}
    if len(releases) != 1:
        raise BusError("Homebrew Safari packages name different public releases")
    return {
        "release": releases.pop(),
        "homebrew_revision": revision,
        "packages": packages,
    }


def resolve_packages(majors, revision=None):
    if revision is None:
        commits = json.loads(
            _read_url(
                f"https://api.github.com/repos/{REPO}/commits?path={CASK}&per_page=1"
            )
        )
        revision = commits[0]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise BusError("invalid Homebrew revision")
    text = _read_url(
        f"https://raw.githubusercontent.com/{REPO}/{revision}/{CASK}"
    ).decode()
    return parse_cask(text, majors, revision)


def download_package(package, path):
    request = urllib.request.Request(
        package["source_url"], headers={"User-Agent": "slipstream"}
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        path.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    if sha256_file(path) != package["dmg_sha256"]:
        raise BusError(f"Safari DMG checksum mismatch for macOS {package['os_major']}")


def extract_package(dmg, dest, major):
    """Mount read-only, expand the flat pkg and copy its payload, without scripts."""
    mount = dest / "mount"
    mount.mkdir(parents=True)
    subprocess.run(
        [
            "hdiutil",
            "attach",
            "-readonly",
            "-nobrowse",
            "-mountpoint",
            str(mount),
            str(dmg),
        ],
        check=True,
        capture_output=True,
    )
    try:
        pkgs = list(mount.glob("*.pkg"))
        if len(pkgs) != 1:
            raise BusError("Safari DMG must contain exactly one package")
        expanded = dest / "expanded"
        subprocess.run(
            ["pkgutil", "--expand-full", str(pkgs[0]), str(expanded)],
            check=True,
            capture_output=True,
        )
    finally:
        subprocess.run(
            ["hdiutil", "detach", str(mount)], check=True, capture_output=True
        )
    apps = list(expanded.rglob(APP))
    if len(apps) != 1:
        raise BusError("Safari package must contain exactly one STP app")
    source = apps[0]
    with (source / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if info.get("CFBundleIdentifier") != "com.apple.SafariTechnologyPreview":
        raise BusError("Safari package contains an unexpected bundle identity")
    minimum = info.get("LSMinimumSystemVersion", "")
    if not minimum or int(minimum.split(".")[0]) > major:
        raise BusError(f"Safari app cannot support macOS {major}: minimum {minimum!r}")
    if not (
        source / "Contents/Frameworks/Safari.framework/Versions/A/Safari"
    ).is_file():
        raise BusError("Safari package is missing Safari.framework")
    read_app(source)
    root = dest / "app"
    root.mkdir()
    shutil.copytree(source, root / APP, symlinks=True)
    return root


class SafariReleases:
    def __init__(self, bus, engine, majors, number_for, log=lambda message: None):
        self.bus, self.engine = bus, engine
        self.majors, self.number_for, self.log = sorted(majors), number_for, log
        self.path = bus.root / "state" / "safari" / f"{engine}.json"
        self.record = read_release(self.path)
        self.last_error = None

    def installed(self):
        if not self.record:
            self.refresh()
        record = self.record
        return HostApp(
            "Safari Technology Preview",
            f"stp-release:{record['release']}",
            str(record["release"]),
        ), record["embedder_id"]

    def refresh(self):
        try:
            self._refresh()
            self.last_error = None
        except (
            OSError,
            ValueError,
            BusError,
            subprocess.SubprocessError,
            KeyError,
            IndexError,
        ) as exc:
            self.last_error = str(exc)
            self.log(f"{self.engine}: Safari acquisition failed: {exc}")
            self.record = read_release(self.path)
            if not self.record:
                raise BusError(f"no adopted Safari release: {exc}") from exc
            # A committed record is usable even if upstream has disappeared.
            self._verify_objects()

    def _verify_objects(self):
        for blob in release_blobs(self.record):
            obj = blob.objects(5)[0]
            path = self.bus.object_path(obj.name)
            if path.stat().st_size != obj.bytes or sha256_file(path) != obj.sha256:
                raise BusError(f"adopted Safari object is corrupt: {obj.name}")

    def _refresh(self):
        from .builder import store_run_set

        # An OS-list change does not mutate an already adopted release. Resolve
        # its old map too, so an upstream removal/change is still a conflict.
        candidate = resolve_packages(self.majors)
        if self.record:
            if candidate["release"] < self.record["release"]:
                raise BusError("Safari public release rollback rejected")
            if candidate["release"] == self.record["release"]:
                if self.majors != self.record["os_majors"]:
                    candidate = resolve_packages(
                        self.record["os_majors"], candidate["homebrew_revision"]
                    )
                pinned = self.record["packages"]
                for major in self.record["os_majors"]:
                    key = str(major)
                    if any(
                        candidate["packages"][key][k] != pinned[key][k]
                        for k in ("source_url", "dmg_sha256", "os_major")
                    ):
                        raise BusError("acquisition conflict for frozen Safari release")
                self._verify_objects()
                return
        with tempfile.TemporaryDirectory(prefix="slipstream-stp-") as directory:
            work = Path(directory)
            private = Bus(work / "bus", _raw=True)
            private.format_version = 4
            packages = {}
            for major in self.majors:
                package = dict(candidate["packages"][str(major)])
                dmg = work / f"{major}.dmg"
                download_package(package, dmg)
                root = extract_package(dmg, work / str(major), major)
                package["app_build"] = read_app(root / APP).version
                package["blob"] = store_run_set(private, root, [APP], caffeinate=False)[
                    0
                ].to_json()
                packages[str(major)] = package
            record = {
                "schema": 1,
                "release": candidate["release"],
                "homebrew_revision": candidate["homebrew_revision"],
                "os_majors": self.majors,
                "packages": packages,
            }
            protocol = self.bus.protocol
            with protocol.lock() if protocol else nullcontext():
                durable_mkdir(self.bus.blobs_dir)
                for package in packages.values():
                    obj = Blob.from_json(package["blob"], "Safari").objects(5)[0]
                    source = private.object_path(obj.name)
                    dest = self.bus.object_path(obj.name)
                    if dest.exists():
                        if sha256_file(dest) != obj.sha256:
                            raise BusError(
                                f"existing Safari object is corrupt: {obj.name}"
                            )
                    else:
                        tmp = self.bus.tmp_object(obj.name)
                        try:
                            shutil.copyfile(source, tmp)
                            with tmp.open("rb") as stream:
                                flush_file(stream.fileno())
                            os.replace(tmp, dest)
                        finally:
                            tmp.unlink(missing_ok=True)
                fsync_dir(self.bus.blobs_dir)
                app = HostApp(
                    "Safari Technology Preview",
                    f"stp-release:{record['release']}",
                    str(record["release"]),
                )
                record["embedder_id"] = self.number_for(app)
                validate_release(record)
                atomic_write(self.path, json.dumps(record, sort_keys=True, indent=2))
                self.record = record
