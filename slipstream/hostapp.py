# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""The identity of an application bundle, read off its ``Info.plist``.

Two users: the builder, for which the installed Safari Technology Preview
is an embedder coordinate that has to be named per publish; and the
provenance record, which notes the host application a browser run went
through (the STP build, the Chromium version, the macOS Safari whose
launcher hosted the run).
"""

from __future__ import annotations

import plistlib
from pathlib import Path
from typing import NamedTuple


class HostApp(NamedTuple):
    """``version`` is ``CFBundleVersion``, the build string, and is the
    identity; ``short`` is ``CFBundleShortVersionString``, the marketing
    version, for humans; ``name`` is ``CFBundleName`` or the bundle's
    stem."""

    name: str
    version: str
    short: str

    @property
    def title(self) -> str:
        return f"{self.name} {self.version} ({self.short})" if self.short else (
            f"{self.name} {self.version}"
        )


class HostAppError(RuntimeError):
    """The bundle is not there or has no readable identity."""


def read_app(app: Path) -> HostApp:
    """The identity of the bundle at ``app`` (``…/Foo.app``)."""
    plist = Path(app) / "Contents" / "Info.plist"
    try:
        with open(plist, "rb") as f:
            info = plistlib.load(f)
    except FileNotFoundError:
        raise HostAppError(f"no application bundle at {app}") from None
    except (OSError, plistlib.InvalidFileException, ValueError) as e:
        raise HostAppError(f"unreadable {plist}: {e}") from None
    version = str(info.get("CFBundleVersion", "")).strip()
    if not version:
        raise HostAppError(f"{plist} has no CFBundleVersion")
    return HostApp(
        name=str(info.get("CFBundleName") or Path(app).stem),
        version=version,
        short=str(info.get("CFBundleShortVersionString", "")).strip(),
    )
