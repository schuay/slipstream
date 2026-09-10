# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from slipstream.config import PushTarget, load_config


def _write(tmp_path, push_section):
    p = tmp_path / "config.toml"
    p.write_text(f'out_dir = "{tmp_path}"\n{push_section}')
    return p


class TestPushConfig:
    def test_absent(self, tmp_path):
        assert load_config(_write(tmp_path, "")).push is None

    def test_targets_required(self, tmp_path):
        with pytest.raises(ValueError, match="push.targets"):
            load_config(_write(tmp_path, '[push]\nbot_name = "m2"\n'))

    def test_explicit_targets(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[push]\nbot_name = "m2"\n'
                '[[push.targets]]\nspool_dir = "~/outbox"\n'
                '[[push.targets]]\nspanner = "p/i/d"\n',
            )
        )
        assert cfg.push.targets == [
            PushTarget(spool_dir=Path.home() / "outbox"),
            PushTarget(spanner="p/i/d"),
        ]

    @pytest.mark.parametrize(
        "target", ['command = "skiz-push --bot {bot}"', 'ssh_host = "m1"']
    )
    def test_command_targets_are_rejected(self, tmp_path, target):
        """The old path out was a shell command piping the CSV to skiz."""
        with pytest.raises(ValueError, match="command targets are gone"):
            load_config(
                _write(
                    tmp_path, f'[push]\nbot_name = "m2"\n[[push.targets]]\n{target}\n'
                )
            )

    def test_spool_target(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[push]\nbot_name = "m2"\n[[push.targets]]\nspool_dir = "~/outbox"\n',
            )
        )
        (t,) = cfg.push.targets
        assert t.spanner is None
        assert t.spool_dir == Path.home() / "outbox"

    def test_spool_retention_default_and_override(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[push]\nbot_name = "m2"\n'
                '[[push.targets]]\nspool_dir = "~/a"\n'
                '[[push.targets]]\nspool_dir = "~/b"\nretain_days = 30\n',
            )
        )
        assert [t.retain_days for t in cfg.push.targets] == [90, 30]

    @pytest.mark.parametrize(
        "target",
        [
            'spool_dir = "y"\nspanner = "p/i/d"',
            "",
            'spanner = "p/i/d"\nretain_days = 30',
            'spool_dir = "y"\nretain_days = 0',
            'spool_dir = "y"\nretain_days = "week"',
            'spool_dir = "y"\nretain_days = true',
        ],
    )
    def test_invalid_target(self, tmp_path, target):
        with pytest.raises(ValueError, match="push.targets"):
            load_config(
                _write(
                    tmp_path, f'[push]\nbot_name = "m2"\n[[push.targets]]\n{target}\n'
                )
            )


class TestRelayConfig:
    def test_none(self, tmp_path):
        assert load_config(_write(tmp_path, "")).relays == []

    def test_source_with_default_cursor_dir(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[[relay]]\nssh_host = "box2"\nspool_dir = "~/outbox"\nbot_name = "b2"\n',
            )
        )
        (r,) = cfg.relays
        assert (r.ssh_host, r.spool_dir, r.bot_name) == ("box2", "~/outbox", "b2")
        assert r.cursor_dir == tmp_path / "relay"

    def test_explicit_cursor_dir(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[[relay]]\nssh_host = "box2"\nspool_dir = "~/o"\nbot_name = "b2"\n'
                'cursor_dir = "~/cursors"\n',
            )
        )
        assert cfg.relays[0].cursor_dir == Path.home() / "cursors"

    def test_duplicate_cursor_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must not share"):
            load_config(
                _write(
                    tmp_path,
                    '[[relay]]\nssh_host = "one"\nspool_dir = "~/a"\nbot_name = "b"\n'
                    '[[relay]]\nssh_host = "two"\nspool_dir = "~/b"\nbot_name = "b"\n',
                )
            )

    def test_equivalent_cursor_paths_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="must not share"):
            load_config(
                _write(
                    tmp_path,
                    '[[relay]]\nssh_host = "one"\nspool_dir = "~/a"\nbot_name = "b"\n'
                    'cursor_dir = "one/../shared"\n'
                    '[[relay]]\nssh_host = "two"\nspool_dir = "~/b"\nbot_name = "b"\n'
                    'cursor_dir = "shared"\n',
                )
            )

    def test_spanner_target(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[push]\nbot_name = "m"\n[[push.targets]]\nspanner = "p/i/d"\nrefresh = false\n',
            )
        )
        (t,) = cfg.push.targets
        assert (t.spanner, t.refresh, t.spool_dir) == ("p/i/d", False, None)

    def test_bad_spanner_spec_rejected_at_load(self, tmp_path):
        with pytest.raises(ValueError, match="project/instance/database"):
            load_config(
                _write(
                    tmp_path,
                    '[push]\nbot_name = "m"\n[[push.targets]]\nspanner = "p/i"\n',
                )
            )

    def test_refresh_only_for_spanner(self, tmp_path):
        with pytest.raises(ValueError, match="refresh"):
            load_config(
                _write(
                    tmp_path,
                    '[push]\nbot_name = "m"\n[[push.targets]]\n'
                    'spool_dir = "y"\nrefresh = false\n',
                )
            )


class TestHostConfig:
    def test_defaults_to_warn(self, tmp_path):
        assert load_config(_write(tmp_path, "")).host.preflight == "warn"

    @pytest.mark.parametrize("mode", ["warn", "abort", "off"])
    def test_accepted_modes(self, tmp_path, mode):
        cfg = load_config(_write(tmp_path, f'[host]\npreflight = "{mode}"\n'))
        assert cfg.host.preflight == mode

    def test_unknown_mode_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="host"):
            load_config(_write(tmp_path, '[host]\npreflight = "yes"\n'))


class TestBotName:
    def test_absent_is_none(self, tmp_path):
        assert load_config(_write(tmp_path, "")).bot_name is None

    def test_top_level(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(f'out_dir = "{tmp_path}"\nbot_name = "box2-m4"\n')
        assert load_config(p).bot_name == "box2-m4"

    def test_inherited_from_push(self, tmp_path):
        """Configs written before the bus name the machine only in [push]."""
        cfg = load_config(
            _write(
                tmp_path,
                '[push]\nbot_name = "m2"\n[[push.targets]]\nspool_dir = "~/o"\n',
            )
        )
        assert cfg.bot_name == "m2"

    def test_push_name_must_match(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box1"\n'
            '[push]\nbot_name = "box2"\n[[push.targets]]\nspool_dir = "~/o"\n'
        )
        with pytest.raises(ValueError, match="must match"):
            load_config(p)

    def test_empty_rejected(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(f'out_dir = "{tmp_path}"\nbot_name = "  "\n')
        with pytest.raises(ValueError, match="non-empty"):
            load_config(p)

    def test_collides_with_a_relay_source(self, tmp_path):
        """Two boxes under one bot are averaged together in Spanner, silently."""
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box2"\n'
            '[[relay]]\nssh_host = "h"\nspool_dir = "/s"\nbot_name = "box2"\n'
        )
        with pytest.raises(ValueError, match="relay"):
            load_config(p)

    def test_require_bot_name(self, tmp_path):
        cfg = load_config(_write(tmp_path, ""))
        with pytest.raises(ValueError, match="bot_name"):
            cfg.require_bot_name()


class TestEngineWithoutCheckout:
    def test_a_table_alone_configures_the_engine(self, tmp_path):
        """A box that only benches bus artifacts has no checkout."""
        cfg = load_config(_write(tmp_path, "[engines.v8]\n"))
        assert cfg.engines["v8"].src_dir is None

    def test_no_table_means_not_configured(self, tmp_path):
        assert load_config(_write(tmp_path, "")).engines == {}

    def test_require_src_dir_names_the_engine(self, tmp_path):
        cfg = load_config(_write(tmp_path, "[engines.v8]\n"))
        with pytest.raises(ValueError, match="v8 has no checkout"):
            cfg.engines["v8"].require_src_dir()

    def test_build_inputs_are_overridable(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[engines.v8]\nsrc_dir = "/v8"\nsync_cmd = "mine"\n'
                'gn_args = "is_debug = true"\n',
            )
        )
        assert cfg.engines["v8"].sync_cmd == "mine"
        assert cfg.engines["v8"].gn_args == "is_debug = true"


class TestRunSet:
    def test_defaults_to_the_bundled_set(self, tmp_path):
        cfg = load_config(_write(tmp_path, '[engines.v8]\nsrc_dir = "/v8"\n'))
        assert cfg.engines["v8"].run_set == [
            "out/release-lto/d8",
            "out/release-lto/icudtl.dat",
            "out/release-lto/snapshot_blob.bin",
        ]

    def test_an_explicitly_empty_set_raises_when_used(self, tmp_path):
        cfg = load_config(
            _write(tmp_path, '[engines.v8]\nsrc_dir = "/v8"\nrun_set = []\n')
        )
        with pytest.raises(ValueError, match="no run_set"):
            cfg.engines["v8"].require_run_set()

    def test_configured(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[engines.v8]\nsrc_dir = "/v8"\n'
                'run_set = ["out/release-lto/d8", "out/release-lto/icudtl.dat"]\n',
            )
        )
        assert cfg.engines["v8"].require_run_set() == [
            "out/release-lto/d8",
            "out/release-lto/icudtl.dat",
        ]

    @pytest.mark.parametrize("entry", ["/abs/path", "../escape", "out/../../x", " "])
    def test_entries_may_not_escape_src_dir(self, tmp_path, entry):
        with pytest.raises(ValueError, match="run_set"):
            load_config(
                _write(
                    tmp_path, f'[engines.v8]\nsrc_dir = "/v8"\nrun_set = ["{entry}"]\n'
                )
            )

    def test_must_be_a_list_of_strings(self, tmp_path):
        with pytest.raises(ValueError, match="run_set"):
            load_config(
                _write(tmp_path, '[engines.v8]\nsrc_dir = "/v8"\nrun_set = 3\n')
            )


class TestBusConfig:
    def test_absent(self, tmp_path):
        assert load_config(_write(tmp_path, "")).bus is None

    def test_root_required(self, tmp_path):
        with pytest.raises(ValueError, match="bus.*root"):
            load_config(_write(tmp_path, "[bus]\n"))

    def test_local_and_remote_sources(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                "[engines.v8]\n[engines.jsc]\n"
                '[bus]\nroot = "~/bus"\n'
                '[[bus.sources]]\nname = "local"\nroot = "~/bus"\nengines = ["v8"]\n'
                '[[bus.sources]]\nname = "box2"\nssh_host = "box2"\n'
                'root = "~/slipstream/bus"\nengines = ["jsc"]\nbwlimit = "20000"\n',
            )
        )
        local, remote = cfg.bus.sources
        assert local.is_local and local.local_root == Path.home() / "bus"
        assert not remote.is_local and remote.bwlimit == "20000"
        assert cfg.bus.source_for("jsc") is remote
        with pytest.raises(ValueError, match="box2"):
            remote.local_root

    def test_an_engine_with_no_source_is_git_driven(self, tmp_path):
        cfg = load_config(_write(tmp_path, '[engines.v8]\n[bus]\nroot = "~/bus"\n'))
        assert cfg.bus.source_for("v8") is None

    def test_a_source_without_engines_takes_everything(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                '[engines.v8]\n[bus]\nroot = "~/bus"\n'
                '[[bus.sources]]\nname = "local"\nroot = "~/bus"\n',
            )
        )
        assert cfg.bus.source_for("v8").name == "local"

    def test_two_sources_may_not_claim_one_engine(self, tmp_path):
        with pytest.raises(ValueError, match="claimed by both"):
            load_config(
                _write(
                    tmp_path,
                    '[engines.v8]\n[bus]\nroot = "~/bus"\n'
                    '[[bus.sources]]\nname = "a"\nroot = "~/a"\nengines = ["v8"]\n'
                    '[[bus.sources]]\nname = "b"\nroot = "~/b"\nengines = ["v8"]\n',
                )
            )

    def test_a_catch_all_may_not_share_with_another_source(self, tmp_path):
        with pytest.raises(ValueError, match="only one source may omit engines"):
            load_config(
                _write(
                    tmp_path,
                    '[engines.v8]\n[engines.jsc]\n[bus]\nroot = "~/bus"\n'
                    '[[bus.sources]]\nname = "a"\nroot = "~/a"\nengines = ["v8"]\n'
                    '[[bus.sources]]\nname = "b"\nroot = "~/b"\n',
                )
            )

    def test_duplicate_source_names_are_rejected(self, tmp_path):
        """The name is the cursor path component, so two would share a cursor."""
        with pytest.raises(ValueError, match="used twice"):
            load_config(
                _write(
                    tmp_path,
                    '[engines.v8]\n[engines.jsc]\n[bus]\nroot = "~/bus"\n'
                    '[[bus.sources]]\nname = "a"\nroot = "~/a"\nengines = ["v8"]\n'
                    '[[bus.sources]]\nname = "a"\nroot = "~/b"\nengines = ["jsc"]\n',
                )
            )

    def test_an_unknown_engine_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no \\[engines.v8\\] table"):
            load_config(
                _write(
                    tmp_path,
                    '[bus]\nroot = "~/bus"\n'
                    '[[bus.sources]]\nname = "a"\nroot = "~/a"\nengines = ["v8"]\n',
                )
            )


class TestBuildAndBenchConfig:
    def test_defaults(self, tmp_path):
        cfg = load_config(_write(tmp_path, ""))
        assert cfg.build.engines == [] and cfg.build.retain_gb == 400.0
        assert cfg.build.max_infra_attempts == 5
        assert cfg.bench.run_roots == 2

    def test_values(self, tmp_path):
        cfg = load_config(
            _write(
                tmp_path,
                "[engines.v8]\n"
                '[build]\nengines = ["v8"]\nretain_gb = 250\nmin_free_gb = 80\n'
                "max_infra_attempts = 2\nmax_consecutive_burns = 4\n"
                "from = { v8 = 109680 }\n"
                "[bench]\nmin_free_gb = 50\nrun_roots = 3\n",
            )
        )
        assert cfg.build.engines == ["v8"] and cfg.build.retain_gb == 250
        assert cfg.build.start_from == {"v8": 109680}
        assert cfg.build.max_consecutive_burns == 4
        assert (cfg.bench.min_free_gb, cfg.bench.run_roots) == (50, 3)

    def test_unknown_engine_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="no \\[engines.v8\\] table"):
            load_config(_write(tmp_path, '[build]\nengines = ["v8"]\n'))

    @pytest.mark.parametrize(
        "section,line",
        [
            ("[build]", "retain_gb = 0"),
            ("[build]", 'min_free_gb = "lots"'),
            ("[build]", "max_infra_attempts = true"),
            ("[bench]", "run_roots = -1"),
        ],
    )
    def test_non_positive_values_rejected(self, tmp_path, section, line):
        with pytest.raises(ValueError, match="positive"):
            load_config(_write(tmp_path, f"{section}\n{line}\n"))


def _bench_box(tmp_path, run_section):
    """A config that measures: engines and suites both configured."""
    p = tmp_path / "config.toml"
    p.write_text(
        f'out_dir = "{tmp_path}"\n'
        "[engines.v8]\n[engines.jsc]\n"
        f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
        f"{run_section}"
    )
    return p


class TestRunMatrix:
    def test_entries_become_run_specs(self, tmp_path):
        cfg = load_config(
            _bench_box(
                tmp_path,
                '[[run]]\nengine = "v8"\nsuite = "js3"\n'
                '[[run]]\nengine = "v8"\nsuite = "js3"\n'
                'variant = "per_line_item"\nrun_mode = "per_benchmark"\n',
            )
        )
        assert [(r.engine, r.suite, r.variant, r.run_mode) for r in cfg.runs] == [
            ("v8", "js3", "default", ""),
            ("v8", "js3", "per_line_item", "per_benchmark"),
        ]

    def test_a_single_flag_names_the_variant(self, tmp_path):
        cfg = load_config(
            _bench_box(
                tmp_path,
                '[[run]]\nengine = "v8"\nsuite = "js3"\nflags = ["--turbolev-future"]\n',
            )
        )
        (run,) = cfg.runs
        assert run.variant == "turbolev_future"
        assert run.flags == ("--turbolev-future",)

    def test_several_flags_need_a_name(self, tmp_path):
        with pytest.raises(ValueError, match="name the variant"):
            load_config(
                _bench_box(
                    tmp_path,
                    '[[run]]\nengine = "v8"\nsuite = "js3"\nflags = ["--a", "--b"]\n',
                )
            )

    def test_a_variant_is_restricted_to_what_a_filename_takes(self, tmp_path):
        """It also keys the store and the perf database's variant column."""
        with pytest.raises(ValueError, match="variant"):
            load_config(
                _bench_box(
                    tmp_path,
                    '[[run]]\nengine = "v8"\nsuite = "js3"\nvariant = "../etc"\n',
                )
            )

    def test_an_unconfigured_engine_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not configured"):
            load_config(_bench_box(tmp_path, '[[run]]\nengine = "v9"\nsuite = "js3"\n'))

    def test_an_unconfigured_suite_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="not configured"):
            load_config(_bench_box(tmp_path, '[[run]]\nengine = "v8"\nsuite = "js2"\n'))

    def test_a_duplicate_would_overwrite_its_own_scores(self, tmp_path):
        with pytest.raises(ValueError, match="already defined"):
            load_config(
                _bench_box(
                    tmp_path,
                    '[[run]]\nengine = "v8"\nsuite = "js3"\n'
                    '[[run]]\nengine = "v8"\nsuite = "js3"\n',
                )
            )

    def test_an_unknown_key_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown keys"):
            load_config(
                _bench_box(
                    tmp_path, '[[run]]\nengine = "v8"\nsuite = "js3"\nflag = "-x"\n'
                )
            )

    def test_a_bad_run_mode_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="run_mode"):
            load_config(
                _bench_box(
                    tmp_path,
                    '[[run]]\nengine = "v8"\nsuite = "js3"\nrun_mode = "each"\n',
                )
            )

    def test_a_box_set_up_to_measure_must_say_what(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to measure"):
            load_config(_bench_box(tmp_path, ""))

    def test_a_box_that_only_relays_needs_no_matrix(self, tmp_path):
        assert load_config(_write(tmp_path, "")).runs == []

    def test_an_engine_with_no_entries_is_refused_at_the_bench(self, tmp_path):
        """Otherwise the session measures nothing and marks the commit failed."""
        cfg = load_config(
            _bench_box(tmp_path, '[[run]]\nengine = "v8"\nsuite = "js3"\n')
        )
        assert [r.engine for r in cfg.require_runs("v8")] == ["v8"]
        with pytest.raises(ValueError, match="would measure nothing"):
            cfg.require_runs("jsc")


class TestStrictKeys:
    """A key slipstream does not read is a config error, not a comment."""

    def test_a_top_level_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['outdir'\]"):
            load_config(_write(tmp_path, 'outdir = "~/x"\n'))

    def test_an_engine_slipstream_does_not_know(self, tmp_path):
        with pytest.raises(ValueError, match="not an engine"):
            load_config(_write(tmp_path, '[engines.spidermonkey]\nsrc_dir = "~/sm"\n'))

    def test_a_suite_slipstream_does_not_know(self, tmp_path):
        with pytest.raises(ValueError, match="not a suite"):
            load_config(_write(tmp_path, '[benchmarks.sp3]\ndir = "~/sp3"\n'))

    def test_an_engine_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['srcdir'\]"):
            load_config(_write(tmp_path, '[engines.v8]\nsrcdir = "~/v8"\n'))

    def test_a_bundled_engine_key_is_not_the_users_to_set(self, tmp_path):
        """It would be read from the bundled file anyway, so setting it here
        silently does nothing."""
        with pytest.raises(ValueError, match="bundled config"):
            load_config(_write(tmp_path, "[engines.v8]\nid_regex = '#([0-9]+)'\n"))

    def test_a_benchmark_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['directory'\]"):
            load_config(_write(tmp_path, '[benchmarks.js3]\ndirectory = "~/js3"\n'))

    def test_a_push_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['botname'\]"):
            load_config(_write(tmp_path, '[push]\nbotname = "m2"\n'))

    def test_a_push_target_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['spanner_db'\]"):
            load_config(
                _write(
                    tmp_path,
                    '[push]\nbot_name = "m2"\n[[push.targets]]\nspanner_db = "p/i/d"\n',
                )
            )

    def test_a_retired_target_key_keeps_its_own_message(self, tmp_path):
        """The generic rejection would bury the upgrade instruction."""
        with pytest.raises(ValueError, match="command targets are gone"):
            load_config(
                _write(
                    tmp_path,
                    '[push]\nbot_name = "m2"\n[[push.targets]]\ncommand = "scp"\n',
                )
            )

    def test_a_relay_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['host'\]"):
            load_config(
                _write(
                    tmp_path,
                    '[[relay]]\nssh_host = "h"\nspool_dir = "/s"\n'
                    'bot_name = "b"\nhost = "x"\n',
                )
            )

    def test_a_bus_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['rooot'\]"):
            load_config(_write(tmp_path, '[bus]\nroot = "/b"\nrooot = "/b"\n'))

    def test_a_bus_source_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['bw_limit'\]"):
            load_config(
                _write(
                    tmp_path,
                    f'[bus]\nroot = "{tmp_path}"\n'
                    '[[bus.sources]]\nname = "s"\nroot = "/b"\nbw_limit = "20"\n',
                )
            )

    def test_a_build_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['retain'\]"):
            load_config(_write(tmp_path, "[build]\nretain = 400\n"))

    def test_a_bench_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['run_root'\]"):
            load_config(_write(tmp_path, "[bench]\nrun_root = 2\n"))

    def test_a_host_key_typo(self, tmp_path):
        with pytest.raises(ValueError, match=r"unknown keys \['pre_flight'\]"):
            load_config(_write(tmp_path, '[host]\npre_flight = "off"\n'))

    def test_the_shipped_example_still_loads(self, tmp_path):
        """Every key it demonstrates has to survive the strict pass."""
        from importlib.resources import files as pkg_files

        p = tmp_path / "config.toml"
        p.write_text((pkg_files("slipstream.data") / "config.toml.example").read_text())
        assert len(load_config(p).runs) == 5
