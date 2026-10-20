# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Executable hooks for window transitions observed by one watch process."""

from .hooks import HookError, Hooks  # noqa: F401 -- compatibility export


class WindowHooks(Hooks):
    def __init__(self, config_path=None, *, dry_run=False, log=lambda msg: None):
        super().__init__(config_path, dry_run=dry_run, log=log)
        self.open = False

    def update(self, allowed):
        if allowed == self.open:
            return
        # Record the transition before invoking the hook so an error cannot
        # cause the same hook to run again during cleanup.
        self.open = allowed
        name = "pre-window" if allowed else "post-window"
        self.run(name)
