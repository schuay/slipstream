# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import pytest

from slipstream import config
from slipstream.window_hooks import HookError, WindowHooks


def test_default_directory_and_missing_hooks():
    hooks = WindowHooks()
    assert hooks.directory == config._DEFAULT_CONFIG_PATH.parent / "hooks"
    hooks.update(False)
    hooks.update(True)
    hooks.update(True)
    hooks.update(False)


def test_non_executable_hook_is_ignored(tmp_path):
    directory = tmp_path / "hooks"
    directory.mkdir()
    (directory / "pre-window").write_text("exit 42\n")
    messages = []
    hooks = WindowHooks(tmp_path / "config.toml", log=messages.append)
    hooks.update(True)
    hooks.update(True)
    assert len(messages) == 1
    assert "non-executable" in messages[0]


def test_failed_post_hook_is_not_repeated(tmp_path):
    directory = tmp_path / "hooks"
    directory.mkdir()
    post = directory / "post-window"
    post.write_text("#!/bin/sh\nexit 42\n")
    post.chmod(0o755)
    hooks = WindowHooks(tmp_path / "config.toml")
    hooks.update(True)
    with pytest.raises(HookError, match="post-window.*42"):
        hooks.update(False)
    hooks.update(False)
