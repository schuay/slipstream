#!/usr/bin/env bash
# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

# Install the default macOS power hooks for the account running slipstream.
# Pass --config PATH for a config outside ~/.config/slipstream/config.toml.
set -euo pipefail

if ((EUID == 0)); then
  echo "Run this script without sudo, as the account running slipstream." >&2
  echo "The installer prints the separate sudo command to enable power changes." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to run the hook installer from this checkout." >&2
  exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

# Use this checkout's environment so the helper and installed hooks agree.
exec uv run --project "$repo_root" python -m slipstream.power_hooks install "$@"
