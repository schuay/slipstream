# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Benchmark host power state on macOS.

A collector that runs for months must never suspend, never blank into a lock
screen, and never sit in a degraded power mode. On macOS those are `pmset`
settings, which persist across reboots, so the host is configured once with
``slipstream host --apply`` and only read back afterwards. Nothing here needs
root except ``--apply`` and ``--restore``.

Every write is followed by a re-read: `pmset` accepts `powermode` on hardware
that has no High Power Mode and silently does nothing, so the only honest
report is the one built from state read after the fact.
"""

from __future__ import annotations

import functools
import json
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

BLOCKING = "blocking"
ADVISORY = "advisory"

PREFLIGHT_MODES = ("warn", "abort", "off")

# `pmset -g custom` and `pmset -g` both print "Header:" followed by indented
# settings; these are the headers we read from.
_AC = "AC Power"
_BATTERY = "Battery Power"
_SYSTEM_WIDE = "System-wide power settings"

_READ_COMMANDS = {
    "sw_vers": ["sw_vers", "-productVersion"],
    "hw_model": ["sysctl", "-n", "hw.model"],
    "pmset_custom": ["pmset", "-g", "custom"],
    "pmset_live": ["pmset", "-g"],
    "pmset_batt": ["pmset", "-g", "batt"],
    "screensaver": [
        "defaults",
        "-currentHost",
        "read",
        "com.apple.screensaver",
        "idleTime",
    ],
}


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _read(cmd: list[str]) -> str | None:
    """Run a read-only command. Returns stdout, or None if it failed."""
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout if res.returncode == 0 else None


def parse_pmset_sections(text: str) -> dict[str, dict[str, str]]:
    """Parse `pmset -g [custom]` into {header: {setting: value}}.

    The value is the *last* whitespace-separated field rather than the second:
    several settings have spaces in the name ("Sleep On Power Button 1").
    """
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            current = sections.setdefault(stripped[:-1].strip(), {})
            continue
        if current is None:
            continue
        stripped = re.sub(r"\s*\(.*\)$", "", stripped)
        parts = stripped.rsplit(None, 1)
        if len(parts) == 2:
            current[parts[0].strip()] = parts[1]
    return sections


def parse_power_source(text: str) -> bool | None:
    """True when `pmset -g batt` reports the machine is drawing from AC."""
    m = re.search(r"Now drawing from '([^']+)'", text)
    if not m:
        return None
    return "AC" in m.group(1)


@dataclass(frozen=True)
class HostState:
    os_version: str | None = None
    hw_model: str | None = None
    on_ac: bool | None = None
    sleep_disabled: bool | None = None
    ac: dict[str, str] = field(default_factory=dict)
    battery: dict[str, str] = field(default_factory=dict)
    screensaver_idle: int | None = None
    raw: dict[str, str] = field(default_factory=dict)


def read_state(read: Callable[[list[str]], str | None] | None = None) -> HostState:
    # Resolved here rather than as a default so that tests (and anything else
    # that patches the module) can substitute the reader.
    read = read or _read
    raw = {}
    for name, cmd in _READ_COMMANDS.items():
        out = read(cmd)
        if out is not None:
            raw[name] = out

    custom = parse_pmset_sections(raw.get("pmset_custom", ""))
    live = parse_pmset_sections(raw.get("pmset_live", ""))

    sleep_disabled = None
    disabled = live.get(_SYSTEM_WIDE, {}).get("SleepDisabled")
    if disabled is not None:
        sleep_disabled = disabled == "1"

    screensaver_idle = None
    if "screensaver" in raw:
        try:
            screensaver_idle = int(raw["screensaver"].strip())
        except ValueError:
            pass

    return HostState(
        os_version=(raw.get("sw_vers") or "").strip() or None,
        hw_model=(raw.get("hw_model") or "").strip() or None,
        on_ac=parse_power_source(raw.get("pmset_batt", "")),
        sleep_disabled=sleep_disabled,
        ac=custom.get(_AC, {}),
        battery=custom.get(_BATTERY, {}),
        screensaver_idle=screensaver_idle,
        raw=raw,
    )


@dataclass(frozen=True)
class Check:
    name: str
    want: str
    got: str
    ok: bool
    severity: str = BLOCKING
    fix: str | None = None
    # False when pmset does not report the setting on this host: nothing to
    # compare, and nothing a write could make comparable.
    available: bool = True


def _setting(state: HostState, key: str, want: str, name: str, severity: str) -> Check:
    """One AC-profile setting. Absent keys are unsupported, not unset.

    pmset prints the whole profile it supports, so a key that is missing
    cannot be read back after a write either; reporting it as wrong would be
    a failure no --apply could clear.
    """
    got = state.ac.get(key)
    if got is None:
        return Check(
            name=name,
            want=want,
            got=f"unsupported on {state.hw_model or 'this model'}",
            ok=True,
            severity=severity,
            available=False,
        )
    return Check(
        name=name,
        want=want,
        got=got,
        ok=got == want,
        severity=severity,
        fix=f"sudo pmset -c {key} {want}",
    )


def checks(state: HostState) -> list[Check]:
    out = [
        Check(
            name="on AC power",
            want="yes",
            got={True: "yes", False: "no", None: "unknown"}[state.on_ac],
            ok=state.on_ac is True,
            severity=BLOCKING,
            fix="plug the machine in",
        ),
        _setting(state, "sleep", "0", "system sleep (AC)", BLOCKING),
        _setting(state, "displaysleep", "0", "display sleep (AC)", BLOCKING),
        _setting(state, "lowpowermode", "0", "low power mode (AC)", BLOCKING),
        _setting(state, "disksleep", "0", "disk sleep (AC)", ADVISORY),
        _setting(state, "autorestart", "1", "restart after power loss", ADVISORY),
        Check(
            name="sleep disabled (kernel flag)",
            want="1",
            # Unlike the profile settings, pmset does not print this one
            # until it is set, so an absent flag is unset rather than
            # unsupported: still worth writing, never a reason to abort.
            got=(
                "not set"
                if state.sleep_disabled is None
                else ("1" if state.sleep_disabled else "0")
            ),
            ok=state.sleep_disabled is True,
            severity=ADVISORY,
            fix="sudo pmset -a disablesleep 1",
        ),
    ]

    # No supported way to read back whether the screen locks: askForPassword has
    # not been honoured since 10.13.4. A screensaver that never starts on a
    # display that never sleeps cannot lock, so that is what we check instead.
    if state.screensaver_idle is None:
        out.append(
            Check(
                name="screensaver",
                want="0 (off)",
                got="unreadable (key not set)",
                ok=False,
                severity=ADVISORY,
                fix="defaults -currentHost write com.apple.screensaver idleTime -int 0",
            )
        )
    else:
        out.append(
            Check(
                name="screensaver",
                want="0 (off)",
                got=str(state.screensaver_idle),
                ok=state.screensaver_idle == 0,
                severity=BLOCKING,
                fix="defaults -currentHost write com.apple.screensaver idleTime -int 0",
            )
        )

    out.append(_setting(state, "powermode", "2", "high power mode", ADVISORY))
    return out


def failures(results: list[Check]) -> list[Check]:
    return [c for c in results if not c.ok and c.severity == BLOCKING]


# The AC-profile settings --apply writes, in report order.
AC_SETTINGS = [
    ("sleep", "0"),
    ("displaysleep", "0"),
    ("disksleep", "0"),
    ("autorestart", "1"),
    ("womp", "1"),
    ("lowpowermode", "0"),
    ("powermode", "2"),
]


def apply_commands(state: HostState) -> list[list[str]]:
    """The privileged commands that bring a host to the wanted state.

    One setting per invocation: pmset takes a list left to right, so a key
    this hardware does not have would take the rest of the line down with it.
    Only keys the host reports are written; a key pmset never prints cannot be
    read back afterwards, which is the only evidence a write happened.
    """
    # These are all the AC profile, so set -c and leave battery behaviour
    # alone; disablesleep is the exception, a system-wide flag that pmset does
    # not print until it is set, so it is written unconditionally.
    cmds = [
        ["sudo", "pmset", "-c", key, want]
        for key, want in AC_SETTINGS
        if key in state.ac
    ]
    cmds.append(["sudo", "pmset", "-a", "disablesleep", "1"])
    cmds.append(
        [
            "defaults",
            "-currentHost",
            "write",
            "com.apple.screensaver",
            "idleTime",
            "-int",
            "0",
        ]
    )
    return cmds


def restore_commands(saved: dict) -> list[list[str]]:
    """Undo an --apply from the values it saved beforehand."""
    ac = saved.get("ac", {})
    cmds = []
    for key in ("sleep", "displaysleep", "disksleep", "autorestart", "womp"):
        if key in ac:
            cmds.append(["sudo", "pmset", "-c", key, ac[key]])
    for key in ("lowpowermode", "powermode"):
        if key in ac:
            cmds.append(["sudo", "pmset", "-c", key, ac[key]])
    # These two are written unconditionally by --apply, so they are undone
    # unconditionally. A snapshot value of None means the key was absent, which
    # is the common case on a machine that has never been applied to, and is
    # what the reports call "not set" and "unreadable (key not set)" -- so the
    # undo is to put it back to absent (or, for disablesleep, which pmset does
    # not print until set, to its 0 default).
    disabled = saved.get("sleep_disabled")
    cmds.append(["sudo", "pmset", "-a", "disablesleep", "1" if disabled else "0"])
    idle = saved.get("screensaver_idle")
    if idle is None:
        cmds.append(
            ["defaults", "-currentHost", "delete", "com.apple.screensaver", "idleTime"]
        )
    else:
        cmds.append(
            [
                "defaults",
                "-currentHost",
                "write",
                "com.apple.screensaver",
                "idleTime",
                "-int",
                str(idle),
            ]
        )
    return cmds


def save_snapshot(path: Path, state: HostState) -> None:
    """Record the settings --apply is about to overwrite, so --restore can undo it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ac": state.ac,
                "battery": state.battery,
                "sleep_disabled": state.sleep_disabled,
                "screensaver_idle": state.screensaver_idle,
            },
            indent=2,
        )
    )


def load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text())


def report(state: HostState, results: list[Check], log: Callable[[str], None]) -> None:
    where = " ".join(x for x in (state.hw_model, state.os_version) if x)
    log(f"host: {where or 'unknown'}")
    width = max(len(c.name) for c in results)
    for c in results:
        if not c.available:
            status = "n/a "
        elif c.ok:
            status = "ok  "
        else:
            status = "FAIL" if c.severity == BLOCKING else "warn"
        line = f"  [{status}] {c.name.ljust(width)}  {c.got}"
        if not c.ok:
            line += f"  (want {c.want})"
        log(line)
    for c in results:
        if not c.ok and c.fix:
            log(f"  fix: {c.fix}")


def preflight(mode: str, log: Callable[[str], None]) -> bool:
    """Report host state before a long session. False on a blocking failure.

    Takes the mode rather than a Config: config imports this module.
    """
    if mode == "off" or not is_macos():
        return True
    state = read_state()
    results = checks(state)
    report(state, results, log)
    return not failures(results)


# --- Machine identity, recorded with every artifact and every bench ---
#
# A toolchain or OS bump on the builder shifts every series on *both* bots on
# the same day and would otherwise be attributed to a commit; the bencher's own
# environment is what distinguishes the two series in the first place. Nothing
# acts on a change automatically -- an automatic rebuild on a toolchain bump
# would re-measure the whole history -- but it is recorded and reported.


@functools.cache
def hw_model() -> str:
    if is_macos():
        return (_read(["sysctl", "-n", "hw.model"]) or "").strip()
    return platform.machine()


@functools.cache
def os_version() -> str:
    if is_macos():
        return (_read(["sw_vers", "-productVersion"]) or "").strip()
    return platform.release()


@functools.cache
def toolchain() -> str:
    """First line of the compiler's version banner.

    Cached: identity() is called once per benched commit and once per engine in
    bus status, and none of these change while the process runs.
    """
    out = _read(["clang", "--version"])
    return out.splitlines()[0].strip() if out else ""


def identity() -> dict:
    return {
        "hw_model": hw_model(),
        "os_version": os_version(),
        "toolchain": toolchain(),
    }
