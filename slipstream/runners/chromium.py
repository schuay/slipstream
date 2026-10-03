# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Chromium: the built bundle from the run root, a fresh profile per run."""

from __future__ import annotations

import shutil
import tempfile

from .base import RunRequest
from .browser import BrowserRunner, Command

# What every Chromium run gets, before the [[run]] flags. A change here is
# a change to every chrome series, and shows up in provenance as such.
#
# The field-trial pair is the one that matters for the numbers. A
# non-branded build applies fieldtrial_testing_config.json by default,
# and that file moves with the tree, so a series would otherwise record
# experiment churn as engine changes; --disable-field-trial-config turns
# it off (--enable-benchmarking used to imply that, and since M139 does
# not). --enable-benchmarking is still benchmarking mode: no updater,
# metrics, Chrome Labs, model downloads. The background pair keeps the
# page at full speed if the window is ever not in front.
CHROMIUM_FLAGS = (
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-search-engine-choice-screen",
    "--disable-field-trial-config",
    "--enable-benchmarking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-background-networking",
    "--disable-extensions",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
)


class ChromiumRunner(BrowserRunner):
    def __init__(self, **kw):
        super().__init__(**kw)
        self._profile: str | None = None

    def command(self, req: RunRequest, url: str) -> Command:
        # A fresh profile per run: no cache, no state from the previous
        # commit, and nothing a crashed run leaves behind.
        self._profile = tempfile.mkdtemp(prefix="slipstream-chromium-")
        argv = [
            str(req.run_root / req.engine.binary_path),
            f"--user-data-dir={self._profile}",
            *CHROMIUM_FLAGS,
            *req.spec.flags,
            url,
        ]
        return Command(argv)

    def cleanup(self, req: RunRequest) -> None:
        if self._profile is not None:
            shutil.rmtree(self._profile, ignore_errors=True)
            self._profile = None
