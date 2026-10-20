# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""macOS power-mode helper for paired build/bench hooks.

Run as ``python -m slipstream.power_hooks low|performance|restore`` from a hook.
Only the AC power profile is changed. Each pair gets its own durable snapshot.
"""

import argparse
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import subprocess
import sys

from .host import is_macos, parse_pmset_sections
from .hooks import hooks_directory

PMSET = "/usr/bin/pmset"
KEYS = ("lowpowermode", "powermode")


def install(config_path=None):
    """Install unprivileged hooks and write a reviewable sudoers fragment."""
    if not is_macos():
        raise RuntimeError("power hooks require macOS")
    directory = hooks_directory(config_path)
    user = pwd.getpwuid(os.getuid()).pw_name
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", user):
        raise ValueError("username cannot be safely written as a sudoers user")
    keys = read_power()
    if not keys:
        raise RuntimeError("pmset reports no supported power-mode setting")
    commands = [
        f"{PMSET} -c {key} {value}"
        for key in keys
        for value in (("0", "1", "2") if key == "powermode" else ("0", "1"))
    ]
    contents = {}
    for name, mode in (
        ("pre-build", "low"),
        ("post-build", "restore"),
        ("pre-bench", "performance"),
        ("post-bench", "restore"),
    ):
        contents[name] = (
            "#!/bin/sh\n# Installed by slipstream.power_hooks.\n"
            f"exec {shlex.quote(sys.executable)} -m slipstream.power_hooks {mode}\n"
        )
    # Refuse to replace any custom hook before writing any of the four files.
    for name, content in contents.items():
        path = directory / name
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f"existing hook differs; merge it manually: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = directory / name
        path.write_text(content)
        path.chmod(0o755)
    rule = directory / "pmset.sudoers"
    rule.write_text(
        "# Only AC power-mode changes; no shells or general pmset access.\n"
        f"{user} ALL=(root) NOPASSWD: " + ", ".join(commands) + "\n"
    )
    subprocess.run(["/usr/sbin/visudo", "-c", "-f", str(rule)], check=True)
    print(f"Installed power hooks in {directory}")
    print("Enable their exact-command sudo rule with:")
    print(
        f"sudo /usr/bin/install -o root -g wheel -m 0440 {shlex.quote(str(rule))} /etc/sudoers.d/slipstream-power"
    )


def read_power():
    result = subprocess.run(
        [PMSET, "-g", "custom"], check=True, capture_output=True, text=True
    )
    ac = parse_pmset_sections(result.stdout).get("AC Power", {})
    return {key: ac[key] for key in KEYS if key in ac}


def write_power(key, value):
    # Never prompt an unattended daemon. The host must grant these particular
    # pmset commands via sudoers. Ignore cached credentials so an interactive
    # sudo session cannot mask missing rules until a long job's restoration.
    subprocess.run(["/usr/bin/sudo", "-n", "-k", PMSET, "-c", key, str(value)], check=True)


def apply(mode, state_dir):
    if not is_macos():
        raise RuntimeError("power hooks require macOS")
    snapshot = Path(state_dir) / "power.json"
    if mode == "restore":
        if not snapshot.exists():
            return  # The pre-hook may have failed before it changed anything.
        saved = json.loads(snapshot.read_text())
        for key in KEYS:
            if key in saved:
                write_power(key, saved[key])
        if read_power() != saved:
            raise RuntimeError(f"power-mode restoration failed; snapshot: {snapshot}")
        snapshot.unlink()
        return

    if snapshot.exists():
        raise RuntimeError(f"restore the existing power snapshot first: {snapshot}")
    saved = read_power()
    if not saved:
        raise RuntimeError("pmset reports no supported power-mode setting")
    key = "powermode" if "powermode" in saved else "lowpowermode"
    target = "1" if mode == "low" else ("2" if key == "powermode" else "0")
    # Save before the first write, so the post-hook can undo a partial failure.
    with snapshot.open("x") as f:
        os.chmod(snapshot, 0o600)
        json.dump(saved, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        write_power(key, target)
    except subprocess.CalledProcessError:
        if not (mode == "performance" and key == "powermode"):
            raise
        # Unsupported High Power Mode can be rejected as well as ignored.
        # The read-back below selects Automatic in either case.
    if read_power().get(key) != target:
        if mode == "performance" and key == "powermode":
            # Some Macs expose powermode yet cannot enable High Power Mode.
            # Automatic still disables Low Power Mode, the blocking preflight
            # requirement; High Power Mode itself is advisory.
            write_power(key, "0")
            if read_power().get(key) == "0":
                print("High Power Mode unavailable; using Automatic (Low Power off).")
                return
        raise RuntimeError(f"pmset did not apply {key}={target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("low", "performance", "restore", "install"))
    parser.add_argument("--config", type=Path, help="Config whose hooks to install")
    parser.add_argument(
        "--state-dir", default=os.environ.get("SLIPSTREAM_HOOK_STATE_DIR")
    )
    args = parser.parse_args()
    if args.mode != "install" and not args.state_dir:
        parser.error("run inside a job hook, or supply --state-dir for recovery")
    try:
        if args.mode == "install":
            install(args.config)
        else:
            apply(args.mode, args.state_dir)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Power hook failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
