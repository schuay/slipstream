# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import shlex

import pytest

from slipstream import host


# pmset aligns values with spaces, uses a tab for SleepDisabled, and has at
# least one setting whose name contains spaces ("Sleep On Power Button").
PMSET_CUSTOM_NO_POWERMODE = """Battery Power:
 lowpowermode         0
 standby              1
 Sleep On Power Button 1
 hibernatefile        /var/vm/sleepimage
 powernap             0
 networkoversleep     0
 disksleep            10
 sleep                1
 hibernatemode        3
 ttyskeepawake        1
 displaysleep         2
 tcpkeepalive         1
AC Power:
 lowpowermode         0
 standby              1
 Sleep On Power Button 1
 hibernatefile        /var/vm/sleepimage
 womp                 1
 powernap             1
 networkoversleep     0
 disksleep            10
 sleep                1
 hibernatemode        3
 ttyskeepawake        1
 displaysleep         10
 tcpkeepalive         1
 autorestart          0
"""

PMSET_CUSTOM_WITH_POWERMODE = PMSET_CUSTOM_NO_POWERMODE.replace(
    "AC Power:\n lowpowermode         0\n",
    "AC Power:\n lowpowermode         0\n powermode            0\n",
)

PMSET_LIVE = """System-wide power settings:
 SleepDisabled\t\t0
Currently in use:
 standby              1
 Sleep On Power Button 1
 hibernatefile        /var/vm/sleepimage
 displaysleep         10
 sleep                1
"""

# Captured from the collector (Mac15,9, macOS 26.6.2). An Apple silicon
# laptop prints neither lowpowermode nor autorestart, and leaves the
# System-wide section empty until disablesleep is set.
PMSET_CUSTOM_APPLE_SILICON = """Battery Power:
 Sleep On Power Button 1
 powermode            0
 standby              1
 hibernatemode        3
 powernap             1
 hibernatefile        /var/vm/sleepimage
 displaysleep         2
 womp                 0
 sleep                1
 lessbright           1
 disksleep            10
AC Power:
 Sleep On Power Button 1
 powermode            0
 standby              1
 hibernatemode        3
 powernap             1
 hibernatefile        /var/vm/sleepimage
 displaysleep         10
 womp                 1
 sleep                1
 disksleep            10
"""

# No SleepDisabled line, and a value annotated by whatever holds a power
# assertion.
PMSET_LIVE_NO_SLEEPDISABLED = """System-wide power settings:
Currently in use:
 standby              1
 disksleep            10
 sleep                1 (sleep prevented by powerd, Amphetamine)
 displaysleep         2
 powermode            0
 womp                 0
"""

PMSET_BATT_AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=12345678)\t100%; charged; 0:00 remaining present: true
"""

PMSET_BATT_BATTERY = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=12345678)\t74%; discharging; 3:41 remaining present: true
"""


def fake_reader(*, custom, live=PMSET_LIVE, batt=PMSET_BATT_AC, screensaver="0\n"):
    out = {
        ("sw_vers", "-productVersion"): "26.1\n",
        ("sysctl", "-n", "hw.model"): "Mac15,6\n",
        ("pmset", "-g", "custom"): custom,
        ("pmset", "-g"): live,
        ("pmset", "-g", "batt"): batt,
    }
    if screensaver is not None:
        out[
            (
                "defaults",
                "-currentHost",
                "read",
                "com.apple.screensaver",
                "idleTime",
            )
        ] = screensaver
    return lambda cmd: out.get(tuple(cmd))


class TestParsePmset:
    def test_splits_sections(self):
        got = host.parse_pmset_sections(PMSET_CUSTOM_NO_POWERMODE)
        assert set(got) == {"Battery Power", "AC Power"}
        assert got["AC Power"]["sleep"] == "1"
        assert got["Battery Power"]["displaysleep"] == "2"

    def test_setting_name_may_contain_spaces(self):
        # Taking the second field instead of the last would read this as "On".
        got = host.parse_pmset_sections(PMSET_CUSTOM_NO_POWERMODE)
        assert got["AC Power"]["Sleep On Power Button"] == "1"

    def test_tab_separated_value(self):
        got = host.parse_pmset_sections(PMSET_LIVE)
        assert got["System-wide power settings"]["SleepDisabled"] == "0"

    def test_annotated_live_value(self):
        # "sleep 1 (sleep prevented by powerd, Amphetamine)": the last field
        # would otherwise be read as the value.
        got = host.parse_pmset_sections(PMSET_LIVE_NO_SLEEPDISABLED)
        assert got["Currently in use"]["sleep"] == "1"

    def test_section_without_settings(self):
        got = host.parse_pmset_sections(PMSET_LIVE_NO_SLEEPDISABLED)
        assert got["System-wide power settings"] == {}

    def test_lowpowermode_is_not_powermode(self):
        # `pmset -g custom | grep powermode` matches lowpowermode too; the key
        # lookup is what tells High Power Mode support apart.
        got = host.parse_pmset_sections(PMSET_CUSTOM_NO_POWERMODE)
        assert "lowpowermode" in got["AC Power"]
        assert "powermode" not in got["AC Power"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            (PMSET_BATT_AC, True),
            (PMSET_BATT_BATTERY, False),
            ("", None),
        ],
    )
    def test_power_source(self, text, expected):
        assert host.parse_power_source(text) is expected


class TestReadState:
    def test_reads_every_field(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE))
        assert state.hw_model == "Mac15,6"
        assert state.os_version == "26.1"
        assert state.on_ac is True
        assert state.sleep_disabled is False
        assert state.ac["displaysleep"] == "10"
        assert state.screensaver_idle == 0
        assert "powermode" not in state.ac

    def test_powermode_key_present(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_WITH_POWERMODE))
        assert "powermode" in state.ac

    def test_absent_sleepdisabled_is_not_false(self):
        # The flag is not printed until it is set; reading that as 0 would
        # claim knowledge the output does not carry.
        state = host.read_state(
            fake_reader(
                custom=PMSET_CUSTOM_APPLE_SILICON, live=PMSET_LIVE_NO_SLEEPDISABLED
            )
        )
        assert state.sleep_disabled is None

    def test_failed_commands_leave_fields_unknown(self):
        state = host.read_state(lambda cmd: None)
        assert state.hw_model is None
        assert state.on_ac is None
        assert state.sleep_disabled is None
        assert state.ac == {}

    def test_unset_screensaver_key(self):
        state = host.read_state(
            fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE, screensaver=None)
        )
        assert state.screensaver_idle is None


def _settings_set(cmds):
    """The (scope, key, value) triples a list of pmset/defaults commands sets."""
    out = set()
    for cmd in cmds:
        if cmd[:2] == ["sudo", "pmset"]:
            scope, rest = cmd[2], cmd[3:]
            for i in range(0, len(rest) - 1, 2):
                out.add((scope, rest[i], rest[i + 1]))
        elif cmd[0] == "defaults":
            out.add(("defaults", cmd[-3], cmd[-1]))
    return out


def _named(results):
    return {c.name: c for c in results}


class TestChecks:
    def test_defaults_are_not_benchmark_ready(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE))
        by_name = _named(host.checks(state))
        assert by_name["system sleep (AC)"].ok is False
        assert by_name["display sleep (AC)"].ok is False
        assert by_name["sleep disabled (kernel flag)"].ok is False
        assert by_name["on AC power"].ok is True
        assert by_name["low power mode (AC)"].ok is True

    def test_configured_host_passes(self):
        custom = PMSET_CUSTOM_NO_POWERMODE.replace(
            " womp                 1\n",
            " womp                 1\n autorestart          1\n",
        )
        # Rewrite only the AC block: it is the second half of the output.
        battery, _, ac = custom.partition("AC Power:")
        ac = (
            ac.replace(" sleep                1", " sleep                0")
            .replace(" displaysleep         10", " displaysleep         0")
            .replace(" disksleep            10", " disksleep            0")
            .replace(" autorestart          0\n", "")
        )
        state = host.read_state(
            fake_reader(
                custom=battery + "AC Power:" + ac,
                live=PMSET_LIVE.replace("SleepDisabled\t\t0", "SleepDisabled\t\t1"),
            )
        )
        assert host.failures(host.checks(state)) == []

    def test_on_battery_is_blocking(self):
        state = host.read_state(
            fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE, batt=PMSET_BATT_BATTERY)
        )
        assert "on AC power" in {c.name for c in host.failures(host.checks(state))}

    def test_unsupported_high_power_mode_is_a_fact_not_a_failure(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE))
        check = _named(host.checks(state))["high power mode"]
        assert check.ok is True
        assert check.severity == host.ADVISORY
        assert "unsupported" in check.got
        assert check not in host.failures(host.checks(state))

    def test_supported_high_power_mode_is_checked(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_WITH_POWERMODE))
        check = _named(host.checks(state))["high power mode"]
        assert check.ok is False
        assert check.got == "0"
        # Still advisory: a bot with the fans on the default curve is usable.
        assert check.severity == host.ADVISORY

    def test_absent_setting_is_not_a_failure(self):
        # pmset on this laptop has no lowpowermode at all. Reporting it as
        # wrong would block the collector on a setting no write can create.
        state = host.read_state(
            fake_reader(
                custom=PMSET_CUSTOM_APPLE_SILICON, live=PMSET_LIVE_NO_SLEEPDISABLED
            )
        )
        check = _named(host.checks(state))["low power mode (AC)"]
        assert check.ok is True
        assert check.available is False
        assert check.fix is None
        assert "unsupported" in check.got

    def test_laptop_without_those_keys_can_pass(self):
        battery, _, ac = PMSET_CUSTOM_APPLE_SILICON.partition("AC Power:")
        ac = (
            ac.replace(" sleep                1", " sleep                0")
            .replace(" displaysleep         10", " displaysleep         0")
            .replace(" disksleep            10", " disksleep            0")
        )
        state = host.read_state(
            fake_reader(
                custom=battery + "AC Power:" + ac,
                live=PMSET_LIVE_NO_SLEEPDISABLED,
            )
        )
        assert host.failures(host.checks(state)) == []

    def test_unset_sleep_disabled_warns_and_keeps_its_fix(self):
        # Not readable, but the write is still worth making, so this stays a
        # warning with a fix rather than an unsupported setting.
        state = host.read_state(
            fake_reader(
                custom=PMSET_CUSTOM_APPLE_SILICON, live=PMSET_LIVE_NO_SLEEPDISABLED
            )
        )
        check = _named(host.checks(state))["sleep disabled (kernel flag)"]
        assert check.ok is False
        assert check.available is True
        assert check.severity == host.ADVISORY
        assert check.fix == "sudo pmset -a disablesleep 1"

    def test_unreadable_screensaver_is_advisory(self):
        state = host.read_state(
            fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE, screensaver=None)
        )
        check = _named(host.checks(state))["screensaver"]
        assert check.ok is False
        assert check.severity == host.ADVISORY

    def test_enabled_screensaver_is_blocking(self):
        state = host.read_state(
            fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE, screensaver="1200\n")
        )
        assert "screensaver" in {c.name for c in host.failures(host.checks(state))}


class TestCommands:
    def test_apply_skips_powermode_when_unsupported(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE))
        flat = [" ".join(c) for c in host.apply_commands(state)]
        assert not any("powermode 2" in c for c in flat)
        assert "sudo pmset -a disablesleep 1" in flat

    def test_apply_sets_powermode_when_supported(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_WITH_POWERMODE))
        flat = [" ".join(c) for c in host.apply_commands(state)]
        assert "sudo pmset -c powermode 2" in flat

    def test_apply_skips_settings_the_host_does_not_report(self):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_APPLE_SILICON))
        flat = [" ".join(c) for c in host.apply_commands(state)]
        assert not any("lowpowermode" in c for c in flat)
        assert not any("autorestart" in c for c in flat)
        assert "sudo pmset -c powermode 2" in flat
        assert "sudo pmset -a disablesleep 1" in flat

    def test_apply_writes_one_setting_per_command(self):
        # pmset takes a list left to right, so grouping a rejected key with a
        # good one loses both.
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE))
        for cmd in host.apply_commands(state):
            if cmd[:2] == ["sudo", "pmset"]:
                assert len(cmd) == 5, cmd

    def test_restore_returns_saved_values(self, tmp_path):
        state = host.read_state(fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE))
        path = tmp_path / "host_settings.bak.json"
        host.save_snapshot(path, state)
        flat = [" ".join(c) for c in host.restore_commands(host.load_snapshot(path))]
        assert "sudo pmset -c sleep 1" in flat
        assert "sudo pmset -c displaysleep 10" in flat
        # SleepDisabled was 0 before the apply, so restore turns it back off.
        assert "sudo pmset -a disablesleep 0" in flat

    @pytest.mark.parametrize(
        "custom",
        [PMSET_CUSTOM_NO_POWERMODE, PMSET_CUSTOM_WITH_POWERMODE],
        ids=["no-powermode", "powermode"],
    )
    def test_every_suggested_fix_is_what_apply_runs(self, custom):
        # The report prints `fix: <cmd>` for copy-pasting. A fix that differs
        # from --apply (say -a where apply uses -c) would leave the host in a
        # different state depending on which route the operator took.
        state = host.read_state(fake_reader(custom=custom))
        applied = _settings_set(host.apply_commands(state))
        for check in host.checks(state):
            if check.fix and check.fix.split()[0] in ("sudo", "defaults"):
                assert _settings_set([shlex.split(check.fix)]) <= applied, check.fix


class TestReport:
    def test_unsupported_setting_reads_as_na_without_a_fix(self):
        state = host.read_state(
            fake_reader(
                custom=PMSET_CUSTOM_APPLE_SILICON, live=PMSET_LIVE_NO_SLEEPDISABLED
            )
        )
        logged = []
        host.report(state, host.checks(state), logged.append)
        line = next(x for x in logged if "low power mode" in x)
        assert "[n/a ]" in line
        assert not any("lowpowermode" in x for x in logged)


class TestPreflight:
    def test_off_mode_reads_nothing(self):
        logged = []
        assert host.preflight("off", logged.append) is True
        assert logged == []

    def test_non_macos_reads_nothing(self, monkeypatch):
        monkeypatch.setattr(host, "is_macos", lambda: False)
        logged = []
        assert host.preflight("warn", logged.append) is True
        assert logged == []

    def test_reports_and_fails_on_macos(self, monkeypatch):
        monkeypatch.setattr(host, "is_macos", lambda: True)
        monkeypatch.setattr(
            host, "_read", fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE)
        )
        logged = []
        assert host.preflight("warn", logged.append) is False
        assert any("system sleep (AC)" in line for line in logged)
        assert any("sudo pmset -c sleep 0" in line for line in logged)


class TestRestoreUndoesEverythingApplyWrites:
    """apply writes disablesleep and idleTime unconditionally, and on a machine
    that has never been applied to both keys are absent beforehand."""

    def _snapshot(self, **overrides):
        saved = {"ac": {}, "sleep_disabled": None, "screensaver_idle": None}
        saved.update(overrides)
        return saved

    def test_an_unset_disablesleep_is_returned_to_its_default(self):
        cmds = host.restore_commands(self._snapshot())
        assert ["sudo", "pmset", "-a", "disablesleep", "0"] in cmds

    def test_an_unset_screensaver_key_is_deleted_not_written(self):
        cmds = host.restore_commands(self._snapshot())
        assert [
            "defaults",
            "-currentHost",
            "delete",
            "com.apple.screensaver",
            "idleTime",
        ] in cmds

    def test_recorded_values_are_put_back(self):
        cmds = host.restore_commands(
            self._snapshot(sleep_disabled=True, screensaver_idle=1200)
        )
        assert ["sudo", "pmset", "-a", "disablesleep", "1"] in cmds
        assert [
            "defaults",
            "-currentHost",
            "write",
            "com.apple.screensaver",
            "idleTime",
            "-int",
            "1200",
        ] in cmds

    def test_every_key_apply_writes_has_an_undo(self, tmp_path):
        """Through the real save/restore path: --apply snapshots this host
        first, so the snapshot carries whatever pmset reported."""
        state = host.HostState(ac={k: "x" for k, _ in host.AC_SETTINGS}, raw={})
        backup = tmp_path / "host.bak.json"
        host.save_snapshot(backup, state)
        applied = {tuple(c[:-1]) for c in host.apply_commands(state)}
        restored = {
            tuple(c[:-1]) for c in host.restore_commands(host.load_snapshot(backup))
        }
        # The screensaver undo is a delete, not a write, so match on the key.
        missing = {
            c for c in applied if c not in restored and "com.apple.screensaver" not in c
        }
        assert not missing, f"apply writes with no undo: {sorted(missing)}"
