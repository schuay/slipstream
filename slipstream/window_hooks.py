# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Executable hooks for window transitions observed by one watch process."""

import os
from pathlib import Path
import subprocess

from . import config as config_mod


class HookError(Exception):
    pass


class WindowHooks:
    def __init__(self, config_path=None, *, dry_run=False, log=lambda msg: None):
        self.directory = (
            Path(config_path or config_mod._DEFAULT_CONFIG_PATH).absolute().parent
            / "hooks"
        )
        self.dry_run = dry_run
        self.log = log
        self.open = False

    def update(self, allowed):
        if allowed == self.open:
            return
        # Record the transition before invoking the hook so an error cannot
        # cause the same hook to run again during cleanup.
        self.open = allowed
        name = "pre-window" if allowed else "post-window"
        path = self.directory / name
        if self.dry_run or not path.is_file():
            return
        if not os.access(path, os.X_OK):
            self.log(f"Ignoring non-executable window hook: {path}")
            return
        self.log(f"Running window hook: {path}")
        try:
            subprocess.run([str(path)], cwd=self.directory.parent, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HookError(f"window hook {path}: {exc}") from exc
