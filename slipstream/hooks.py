# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Git-style executable hooks and paired job lifecycle hooks."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from . import config as config_mod


class HookError(RuntimeError):
    pass


def hooks_directory(config_path=None):
    return (
        Path(config_path or config_mod._DEFAULT_CONFIG_PATH).absolute().parent / "hooks"
    )


class Hooks:
    def __init__(self, config_path=None, *, dry_run=False, log=lambda msg: None):
        self.directory = hooks_directory(config_path)
        self.dry_run = dry_run
        self.log = log
        self._active = set()

    def run(self, name, env=None):
        path = self.directory / name
        if self.dry_run or not path.is_file():
            return
        if not os.access(path, os.X_OK):
            self.log(f"Ignoring non-executable hook: {path}")
            return
        self.log(f"Running hook: {path}")
        try:
            subprocess.run(
                [str(path)],
                cwd=self.directory.parent,
                env=dict(os.environ, SLIPSTREAM_PYTHON=sys.executable, **(env or {})),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise HookError(f"hook {path}: {exc}") from exc

    @contextmanager
    def job(self, kind, state_root, *, engine, key):
        names = [f"pre-{kind}", f"post-{kind}"]
        if (
            kind in self._active
            or self.dry_run
            or not any((self.directory / name).is_file() for name in names)
        ):
            yield
            return
        state_root.mkdir(parents=True, exist_ok=True)
        state = Path(tempfile.mkdtemp(prefix=f"{kind}-", dir=state_root))
        env = {
            "SLIPSTREAM_HOOK_STATE_DIR": str(state.absolute()),
            "SLIPSTREAM_ENGINE": engine,
            "SLIPSTREAM_COMMIT": str(key),
        }
        restored = False
        self._active.add(kind)
        try:
            try:
                self.run(names[0], env)
                yield
            finally:
                # Also undo a pre-hook that failed after partially changing
                # the host. Keep its snapshot if restoration itself fails.
                self.run(names[1], env)
                restored = True
        finally:
            self._active.remove(kind)
            if restored:
                shutil.rmtree(state)
            else:
                self.log(f"Hook state retained at {state}")
