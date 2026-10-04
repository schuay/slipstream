# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess

import pytest

from slipstream.cli import _parse_interval
from keys import K


class TestParseInterval:
    def test_minutes(self):
        assert _parse_interval("30m") == 1800
        assert _parse_interval("1m") == 60

    def test_hours(self):
        assert _parse_interval("2h") == 7200

    def test_seconds(self):
        assert _parse_interval("90s") == 90

    def test_bare_number_defaults_to_minutes(self):
        assert _parse_interval("5") == 300

    def test_whitespace(self):
        assert _parse_interval("  30m  ") == 1800

    def test_invalid(self):
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_interval("abc")

    def test_invalid_unit(self):
        import typer

        with pytest.raises(typer.BadParameter):
            _parse_interval("30d")


class TestParseCursorArg:
    def test_bot_only_replays_everything(self):
        from slipstream.relay import parse_cursor_arg

        assert parse_cursor_arg("box2") == ("box2", 0)

    def test_explicit_seq(self):
        from slipstream.relay import parse_cursor_arg

        assert parse_cursor_arg("box2=41") == ("box2", 41)

    @pytest.mark.parametrize("value", ["=4", "box2=x", "box2=-1", "box2=4.5"])
    def test_invalid(self, value):
        from slipstream.relay import parse_cursor_arg

        with pytest.raises(ValueError):
            parse_cursor_arg(value)


class TestDeliveryAdmin:
    """--rebuild and --reset-cursor act on the cursor before the first cycle."""

    def _cfg(self, tmp_path, target):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\n'
            '[push]\nbot_name = "box1"\n'
            f"[[push.targets]]\n{target}\n"
            '[[relay]]\nssh_host = "box2"\nspool_dir = "~/o"\nbot_name = "b2"\n'
        )
        return p

    def _run(self, cfg_path, args, monkeypatch, entries=(1, 2, 3)):
        from typer.testing import CliRunner

        from slipstream.cli import app

        class FakeSpool:
            def __init__(self, host, spool_dir, **kwargs):
                pass

            def list(self, **kwargs):
                return list(entries)

            def fetch(self, seq):
                from tests.test_spanner import _csv

                return _csv({})

        monkeypatch.setattr("slipstream.delivery_sources.RemoteSpool", FakeSpool)
        monkeypatch.setattr(
            "slipstream.delivery.Coordinator.run", lambda self, **kw: True
        )
        from slipstream.delivery_targets import SpannerSession

        monkeypatch.setattr(
            "slipstream.delivery_targets.BoundedSpannerSession",
            lambda target, **kw: SpannerSession(target),
        )
        return CliRunner().invoke(
            app, ["deliver", "--once", "--config", str(cfg_path), *args]
        )

    def test_reset_cursor_writes_the_file(self, tmp_path, monkeypatch):
        from slipstream.relay import read_cursor

        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(cfg, ["--reset-cursor", "b2=2"], monkeypatch)
        assert res.exit_code == 0, res.output
        assert read_cursor(tmp_path / "relay" / "b2.cursor") == 2

    def test_reset_cursor_past_the_log_is_rejected(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(cfg, ["--reset-cursor", "b2=9"], monkeypatch)
        assert res.exit_code == 1
        assert "holds entries up to 3" in res.output
        assert not (tmp_path / "relay" / "b2.cursor").exists()

    def test_admin_ssh_failure_has_no_traceback(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')

        class BrokenSpool:
            def __init__(self, host, spool_dir, **kwargs):
                pass

            def list(self, **kwargs):
                raise subprocess.CalledProcessError(255, "ssh")

        monkeypatch.setattr("slipstream.delivery_sources.RemoteSpool", BrokenSpool)
        from typer.testing import CliRunner

        from slipstream.cli import app

        res = CliRunner().invoke(
            app,
            ["deliver", "--once", "--config", str(cfg), "--reset-cursor", "b2"],
        )
        assert res.exit_code == 1
        assert "Error:" in res.output
        assert "Traceback" not in res.output

    def test_admin_lock_conflict_has_no_traceback(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, 'spanner = "p/i/d"')
        monkeypatch.setattr(
            "slipstream.spanner.connect",
            lambda spec, **kw: (_ for _ in ()).throw(RuntimeError("delivery active")),
        )
        res = self._run(cfg, ["--rebuild", "b2"], monkeypatch)
        assert res.exit_code == 1
        assert "Error: delivery active" in res.output
        assert "Traceback" not in res.output

    def test_rebuild_refuses_when_the_log_head_is_gone(self, tmp_path, monkeypatch):
        from slipstream import spanner
        from slipstream.relay import read_cursor

        monkeypatch.setattr(
            spanner, "connect", lambda spec, **kw: pytest.fail("wiped before checking")
        )
        cfg = self._cfg(tmp_path, 'spanner = "p/i/d"')
        res = self._run(cfg, ["--rebuild", "b2"], monkeypatch, entries=(4, 5))
        assert res.exit_code == 1
        assert "retains its log only from 4" in res.output
        assert "entries 1..3" in res.output
        assert read_cursor(tmp_path / "relay" / "b2.cursor") == 0

    def test_rebuild_refuses_when_the_log_is_empty(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(cfg, ["--rebuild", "b2"], monkeypatch, entries=())
        assert res.exit_code == 1
        assert "no spool entries to replay" in res.output

    def test_forced_rebuild_resumes_at_the_oldest_entry(self, tmp_path, monkeypatch):
        from slipstream import spanner
        from slipstream.relay import read_cursor

        monkeypatch.setattr(
            spanner, "connect", lambda spec, **kw: pytest.fail("no spanner")
        )
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(
            cfg, ["--rebuild", "b2", "--force"], monkeypatch, entries=(4, 5)
        )
        assert res.exit_code == 0, res.output
        assert read_cursor(tmp_path / "relay" / "b2.cursor") == 3
        assert "replaying from 4" in res.output

    def test_force_without_rebuild_is_rejected(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(cfg, ["--force"], monkeypatch)
        assert res.exit_code == 1
        assert "--force applies to --rebuild" in res.output

    def test_rebuild_wipes_spanner_targets_then_zeroes_the_cursor(
        self, tmp_path, monkeypatch
    ):
        from slipstream import spanner
        from slipstream.relay import read_cursor, write_cursor

        write_cursor(tmp_path / "relay" / "b2.cursor", 12)
        wiped = []

        class Db:
            def query(self, *args):
                from tests.test_spanner import ALL_TABLES

                return ALL_TABLES

            def close(self):
                wiped.append("closed")

        monkeypatch.setattr(spanner, "connect", lambda spec, **kw: Db())
        monkeypatch.setattr(
            spanner, "rebuild_bot", lambda db, bot, **kw: wiped.append(bot) or 3
        )
        cfg = self._cfg(tmp_path, 'spanner = "p/i/d"')
        res = self._run(cfg, ["--rebuild", "b2"], monkeypatch)
        assert res.exit_code == 0, res.output
        assert wiped == ["b2", "closed"]
        assert "wiped 3 staged rows" in res.output
        assert read_cursor(tmp_path / "relay" / "b2.cursor") == 0

    def test_unknown_bot_is_rejected(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(cfg, ["--reset-cursor", "nope"], monkeypatch)
        assert res.exit_code == 1
        assert "no [[relay]] source with bot_name 'nope'" in res.output

    def test_rebuild_and_reset_cursor_are_exclusive(self, tmp_path, monkeypatch):
        cfg = self._cfg(tmp_path, f'spool_dir = "{tmp_path}/outbox"')
        res = self._run(cfg, ["--rebuild", "b2", "--reset-cursor", "b2"], monkeypatch)
        assert res.exit_code == 1


class TestHostCommand:
    def _config(self, tmp_path):
        cfg_path = tmp_path / "config.toml"
        cfg_path.write_text(f'out_dir = "{tmp_path}"\n')
        return cfg_path

    def test_reports_not_applicable_off_macos(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from slipstream.cli import app

        monkeypatch.setattr("slipstream.host.is_macos", lambda: False)
        res = CliRunner().invoke(app, ["host", "--config", str(self._config(tmp_path))])
        assert res.exit_code == 0
        assert "macOS-only" in res.output
        assert "Traceback" not in res.output

    def test_exits_nonzero_when_a_blocking_check_fails(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from test_host import PMSET_CUSTOM_NO_POWERMODE, fake_reader

        from slipstream import host
        from slipstream.cli import app

        monkeypatch.setattr(host, "is_macos", lambda: True)
        monkeypatch.setattr(
            host, "_read", fake_reader(custom=PMSET_CUSTOM_NO_POWERMODE)
        )
        res = CliRunner().invoke(app, ["host", "--config", str(self._config(tmp_path))])
        assert res.exit_code == 1
        assert "system sleep (AC)" in res.output
        assert "Traceback" not in res.output


class TestClearScope:
    def test_it_leaves_the_other_engines_logs_alone(self, tmp_path):
        """The two engines' commit id spaces are independent, so a clear used
        to delete logs for a commit it was never asked about."""
        from typer.testing import CliRunner

        from slipstream.cli import app
        from slipstream.config import load_config
        from slipstream.store import CommitStore

        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box1"\n'
            "[engines.v8]\n[engines.jsc]\n"
            f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
            '[[run]]\nengine = "v8"\nsuite = "js3"\n'
            '[[run]]\nengine = "jsc"\nsuite = "js3"\n'
        )
        cfg = load_config(p)
        store = CommitStore(cfg.metadata_dir / "slipstream.db")
        for engine in ("v8", "jsc"):
            store.upsert_commit(engine, f"h{engine}", 500, "d", 0, "t")
            store.mark_done(engine, cfg.platform, 500)
            d = cfg.commit_results_dir(engine, 500)
            d.mkdir(parents=True)
            (d / "stdout.1.js3.default.txt").write_text("x")
        store.close()

        res = CliRunner().invoke(
            app, ["clear", "v8", "500", "--yes", "--config", str(p)]
        )
        assert res.exit_code == 0, res.output
        assert not cfg.commit_results_dir("v8", 500).exists()
        assert cfg.commit_results_dir("jsc", 500).exists()


class TestBusCommands:
    def _cfg(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box1"\n'
            "[engines.v8]\n"
            f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
            '[[run]]\nengine = "v8"\nsuite = "js3"\n'
            f'[bus]\nroot = "{tmp_path}/bus"\n'
            '[[bus.sources]]\nname = "local"\n'
            f'root = "{tmp_path}/bus"\nengines = ["v8"]\n'
        )
        return p

    def _invoke(self, args):
        from typer.testing import CliRunner

        from slipstream.cli import app

        return CliRunner().invoke(app, args)

    def test_status_runs_on_an_empty_bus(self, tmp_path):
        res = self._invoke(["bus", "status", "--config", str(self._cfg(tmp_path))])
        assert res.exit_code == 0, res.output
        assert "v8 (bus from local)" in res.output
        assert "builder state: never written" in res.output

    def test_gc_reports_what_it_removed(self, tmp_path):
        from slipstream.bus import Bus

        bus = Bus(tmp_path / "bus")
        orphan = bus.tmp_blob("abc")
        orphan.write_bytes(b"partial")
        unreferenced = bus.blob_path("def")
        unreferenced.parent.mkdir(parents=True)
        unreferenced.write_bytes(b"no manifest names me")
        res = self._invoke(["bus", "gc", "--config", str(self._cfg(tmp_path))])
        assert res.exit_code == 0, res.output
        assert "2 unreferenced files removed" in res.output
        assert not orphan.exists() and not unreferenced.exists()

    def test_pause_and_resume(self, tmp_path):
        from slipstream import lock as lock_mod

        res = self._invoke(["bus", "pause", "--ttl", "1h"])
        assert res.exit_code == 0 and "Paused until" in res.output
        assert lock_mod.paused_until() is not None
        res = self._invoke(["bus", "resume"])
        assert res.exit_code == 0 and "Resumed" in res.output
        assert lock_mod.paused_until() is None

    def test_gc_refuses_while_a_build_holds_the_machine(self, tmp_path):
        """An unreferenced payload is also what a build in progress looks like
        from outside."""
        from slipstream.lock import MachineLock

        holder = MachineLock("build")
        assert holder.try_acquire()
        try:
            res = self._invoke(["bus", "gc", "--config", str(self._cfg(tmp_path))])
            assert res.exit_code == 75
            assert "build" in res.output
        finally:
            holder.release()

    def test_status_does_not_snapshot_the_db(self, tmp_path):
        from slipstream.store import CommitStore

        cfg = self._cfg(tmp_path)
        db = tmp_path / "metadata" / "slipstream.db"
        CommitStore(db, bot="box1").close()
        before = list(db.parent.glob("*.bak"))
        assert self._invoke(["bus", "status", "--config", str(cfg)]).exit_code == 0
        assert list(db.parent.glob("*.bak")) == before

    def test_reset_cursor_for_an_unwatched_engine(self, tmp_path):
        """The engine named by --reset-cursor need not be one being watched."""
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box1"\n'
            "[engines.v8]\n[engines.jsc]\n"
            f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
            '[[run]]\nengine = "v8"\nsuite = "js3"\n'
            f'[bus]\nroot = "{tmp_path}/bus"\n'
            '[[bus.sources]]\nname = "local"\n'
            f'root = "{tmp_path}/bus"\nengines = ["v8"]\n'
        )
        res = self._invoke(
            ["watch", "jsc", "--reset-cursor", "v8=1000", "--config", str(p)]
        )
        assert res.exit_code == 0, res.output
        assert "cursor reset to 1000" in res.output

    def test_reset_cursor_for_a_git_driven_engine_is_an_error(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box1"\n'
            "[engines.v8]\n[engines.jsc]\n"
            f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
            '[[run]]\nengine = "v8"\nsuite = "js3"\n'
            f'[bus]\nroot = "{tmp_path}/bus"\n'
            '[[bus.sources]]\nname = "local"\n'
            f'root = "{tmp_path}/bus"\nengines = ["v8"]\n'
        )
        res = self._invoke(
            ["watch", "v8", "--reset-cursor", "jsc=1000", "--config", str(p)]
        )
        assert res.exit_code == 1
        assert "no bus source" in res.output


class TestBuildProbe:
    def _invoke(self, args):
        from typer.testing import CliRunner

        from slipstream.cli import app

        return CliRunner().invoke(app, args)

    def test_reports_a_missing_run_set(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box2"\n'
            f'[engines.v8]\nsrc_dir = "{tmp_path}/src"\nrun_set = []\n'
            f'[bus]\nroot = "{tmp_path}/bus"\n'
            '[build]\nengines = ["v8"]\nmin_free_gb = 0.001\n'
        )
        res = self._invoke(["build", "--probe", "--config", str(p)])
        assert res.exit_code == 1
        assert "no run_set" in res.output
        assert "tar:" in res.output and "zstd:" in res.output

    def test_passes_with_a_run_set_and_a_frontier(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box2"\n'
            f'[engines.v8]\nsrc_dir = "{tmp_path}/src"\nrun_set = ["out"]\n'
            f'[bus]\nroot = "{tmp_path}/bus"\n'
            '[build]\nengines = ["v8"]\nmin_free_gb = 0.001\n'
            "from = { v8 = 109680 }\n"
        )
        res = self._invoke(["build", "--probe", "--config", str(p)])
        assert res.exit_code == 0, res.output
        assert "v8: frontier 109680" in res.output
        assert "writable" in res.output

    def test_publishes_nothing(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}"\nbot_name = "box2"\n'
            f'[engines.v8]\nsrc_dir = "{tmp_path}/src"\nrun_set = ["out"]\n'
            f'[bus]\nroot = "{tmp_path}/bus"\n'
            '[build]\nengines = ["v8"]\nmin_free_gb = 0.001\n'
            "from = { v8 = 1 }\n"
        )
        self._invoke(["build", "--probe", "--config", str(p)])
        from slipstream.bus import Bus

        assert Bus(tmp_path / "bus").keys("v8") == K()


class TestCheckoutlessEngine:
    """config.toml.example promises every command that needs git says so by name."""

    def _cfg(self, tmp_path):
        p = tmp_path / "config.toml"
        p.write_text(
            f'out_dir = "{tmp_path}/out"\nbot_name = "box1"\n'
            "[engines.v8]\n"
            f'[benchmarks.js3]\ndir = "{tmp_path}"\n'
            '[[run]]\nengine = "v8"\nsuite = "js3"\n'
        )
        return p

    def _invoke(self, args):
        from typer.testing import CliRunner

        from slipstream.cli import app

        return CliRunner().invoke(app, args)

    def test_bench(self, tmp_path):
        res = self._invoke(
            ["bench", "v8", "1", "2", "--config", str(self._cfg(tmp_path))]
        )
        assert res.exit_code == 1
        assert "no checkout on this machine" in res.output
        assert "Traceback" not in res.output
