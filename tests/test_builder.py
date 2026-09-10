# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
import tarfile
import time
from pathlib import Path

import pytest

from slipstream.builder import BuildError, Builder, build_cfg_hash, package
from slipstream.bus import Bus
from slipstream.collector import BuildStepError
from slipstream.config import EngineConfig


def _engine(src_dir, run_set=("out/d8",), **kw):
    return EngineConfig(
        name="v8",
        src_dir=src_dir,
        build_cmd="autoninja -C out d8",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
        run_set=list(run_set),
        **kw,
    )


def _commit(commit_id):
    return {
        "hash": f"hash{commit_id}",
        "commit_id": commit_id,
        "date": "2026-09-06",
        "timestamp": 1757116800 + commit_id,
        "title": f"commit {commit_id}",
    }


@pytest.fixture
def builder(config, tmp_path, monkeypatch):
    """A Builder whose git and compile steps are replaced by scripted results."""
    src = tmp_path / "src"
    (src / "out").mkdir(parents=True)
    (src / "out" / "d8").write_bytes(b"binary")
    config.engines["v8"] = _engine(src)
    config.build.engines = ["v8"]
    config.build.start_from = {"v8": 100}
    config.build.min_free_gb = 0.001  # the floor has its own tests

    b = Builder(config, Bus(tmp_path / "bus"))
    b.lock.path = tmp_path / "machine.lock"
    b.logs = []
    b.log = b.logs.append
    # History: 100 is the starting frontier, the rest are buildable.
    b.history = [100, 101, 102, 103, 104]
    b.failures: list = []  # scripted build_at results, one per call

    def next_commit_after(engine, commit_id):
        later = [c for c in b.history if c > commit_id]
        return _commit(later[0]) if later else None

    def build_at(engine, commit_hash, log=None):
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("build log\n")
        return b.failures.pop(0) if b.failures else None

    monkeypatch.setattr(b.collector, "build_at", build_at)
    monkeypatch.setattr(b.collector, "next_commit_after", next_commit_after)
    monkeypatch.setattr(
        b.collector, "commit_metadata_for_id", lambda e, cid: _commit(cid)
    )
    monkeypatch.setattr(b.collector, "head_commit_id", lambda name, fetch=True: 999)
    monkeypatch.setattr("slipstream.builder.package", _fake_package)
    return b


def _fake_package(src_dir, run_set, dest, *, caffeinate=True):
    dest.write_bytes(b"archive of " + b",".join(e.encode() for e in run_set))


class TestPublishing:
    def test_the_published_line_sizes_the_payload_in_its_own_unit(self, builder):
        """A payload is tens of MB, so in GB every artifact reads 0.0."""
        builder.build_one("v8")
        (line,) = [m for m in builder.logs if "published" in m]
        assert "MB," in line and "GB" not in line, line

    def test_publishes_the_next_commit_above_the_frontier(self, builder):
        result = builder.build_one("v8")
        assert result.published and result.commit_id == 101
        assert builder.bus.commit_ids("v8") == [101]
        entry = builder.bus.read_entry("v8", 101)
        assert entry.hash == "hash101" and entry.title == "commit 101"
        assert entry.blob_sha256 and entry.blob_bytes > 0
        assert entry.build_cfg_hash.startswith("sha256:")

    def test_walks_upward_one_commit_per_cycle(self, builder):
        for _ in range(3):
            builder.build_one("v8")
        assert builder.bus.commit_ids("v8") == [101, 102, 103]

    def test_up_to_date_publishes_nothing(self, builder):
        builder.history = [100]
        assert builder.build_one("v8").published is False
        assert builder.bus.commit_ids("v8") == []

    def test_the_state_file_reports_the_frontier(self, builder):
        builder.build_one("v8")
        state = builder.bus.read_builder_state("v8")
        assert state.frontier == 101 and state.lowest_retained == 101
        assert state.in_flight is None and state.updated_at > 0

    def test_an_engine_with_no_run_set_is_refused(self, builder, config):
        config.engines["v8"].run_set = []
        with pytest.raises(ValueError, match="no run_set"):
            builder.build_one("v8")


class TestFrontier:
    def test_from_is_only_consulted_when_there_is_no_state(self, builder):
        assert builder.frontier("v8") == 100
        builder.build_one("v8")
        assert builder.frontier("v8") == 101
        # A restart must not rewind past what was published.
        builder.cfg.build.start_from = {"v8": 50}
        assert builder.frontier("v8") == 101

    def test_a_terminal_failure_advances_it(self, builder):
        """Otherwise a compile failure at the head is rebuilt every cycle."""
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        assert builder.frontier("v8") == 101
        assert builder.build_one("v8").commit_id == 102

    def test_a_non_terminal_failure_does_not(self, builder):
        builder.failures = [BuildStepError("sync", 1)]
        builder.build_one("v8")
        assert builder.frontier("v8") == 100
        assert builder.build_one("v8").commit_id == 101

    def test_no_history_and_no_from_is_an_error(self, builder):
        builder.cfg.build.start_from = {}
        with pytest.raises(BuildError, match="from"):
            builder.next_commit("v8")


class TestFailureClassification:
    def test_a_compile_failure_is_the_commits_fault(self, builder):
        builder.failures = [BuildStepError("compile", 1)]
        result = builder.build_one("v8")
        assert result.status == "compile_failed"
        row = builder.store.get_build_state("v8", 101)
        assert row["kind"] == "compile" and row["log_path"]

    @pytest.mark.parametrize(
        "kind", ["reset", "clean", "checkout", "patch", "gn", "sync"]
    )
    def test_every_other_step_is_infrastructure(self, builder, kind):
        builder.failures = [BuildStepError(kind, 1)]
        result = builder.build_one("v8")
        assert result.status == "infra_retry"
        assert builder.frontier("v8") == 100

    def test_a_packaging_failure_is_infrastructure(self, builder, monkeypatch):
        def boom(*a, **k):
            raise BuildError("run_set entries missing from the build: ['out/d8']")

        monkeypatch.setattr("slipstream.builder.package", boom)
        result = builder.build_one("v8")
        assert result.status == "infra_retry"
        assert builder.bus.commit_ids("v8") == []

    def test_infra_retries_are_bounded(self, builder):
        """A bad DEPS pin is indistinguishable by exit code from an outage, so
        "never advance" cannot be unbounded."""
        builder.cfg.build.max_infra_attempts = 3
        for attempt in range(1, 4):
            builder.failures = [BuildStepError("sync", 1)]
            result = builder.build_one("v8")
            assert result.commit_id == 101
            assert result.status == ("infra_burned" if attempt == 3 else "infra_retry")
        assert builder.frontier("v8") == 101

    def test_a_success_clears_the_failure_row(self, builder):
        builder.failures = [BuildStepError("sync", 1)]
        builder.build_one("v8")
        assert builder.store.get_build_state("v8", 101) is not None
        builder.build_one("v8")
        assert builder.store.get_build_state("v8", 101) is None


def _open_stall_window(builder, engine="v8"):
    """Bring the persisted backoff deadline forward, as waiting it out would."""
    state = builder.bus.read_builder_state(engine)
    state.stall_retry_after = None
    builder.bus.write_builder_state(engine, state)


class TestCircuitBreaker:
    def _burn(self, builder, count):
        builder.cfg.build.max_infra_attempts = 1
        for _ in range(count):
            builder.failures = [BuildStepError("sync", 1)]
            builder.build_one("v8")

    def test_stalls_after_consecutive_burns(self, builder):
        """Three commits failing sync in a row is the machine, not the tree."""
        builder.cfg.build.max_consecutive_burns = 3
        self._burn(builder, 3)
        assert builder.consecutive_burns("v8") == 3
        assert builder.bus.read_builder_state("v8").stalled_since is not None

    def test_a_publish_resets_the_run(self, builder):
        """Consecutive means with no successful publish between them, so three
        unrelated outages months apart must not stall a healthy engine."""
        builder.cfg.build.max_consecutive_burns = 3
        self._burn(builder, 2)
        builder.cfg.build.max_infra_attempts = 5
        builder.build_one("v8")  # publishes 103
        assert builder.consecutive_burns("v8") == 0
        assert builder.bus.read_builder_state("v8").stalled_since is None

    def test_the_backoff_survives_a_fresh_process(self, builder):
        """`build --once` from launchd is a new Builder every invocation, so an
        in-memory deadline would let a stalled engine rebuild every run."""
        builder.cfg.build.max_consecutive_burns = 2
        self._burn(builder, 2)
        state = builder.bus.read_builder_state("v8")
        assert state.stall_retry_after and state.stall_retry_after > time.time()

        from slipstream.builder import Builder

        fresh = Builder(builder.cfg, builder.bus)
        assert fresh._stall_window_open(fresh.bus.read_builder_state("v8")) is False

    def test_a_stalled_builder_backs_off_but_keeps_trying(self, builder):
        builder.cfg.build.max_consecutive_burns = 2
        self._burn(builder, 2)
        assert builder.bus.read_builder_state("v8").stalled_since is not None
        # Backed off: this cycle does nothing at all.
        published_before = builder.bus.commit_ids("v8")
        assert builder.build_one("v8").commit_id is None
        assert builder.bus.commit_ids("v8") == published_before
        # Once the window opens it tries again, and a success un-stalls it.
        _open_stall_window(builder)
        builder.cfg.build.max_infra_attempts = 5
        assert builder.build_one("v8").published
        assert builder.bus.read_builder_state("v8").stalled_since is None

    def test_a_failed_attempt_while_stalled_burns_nothing(self, builder):
        builder.cfg.build.max_consecutive_burns = 2
        self._burn(builder, 2)
        burns = builder.consecutive_burns("v8")
        _open_stall_window(builder)
        builder.failures = [BuildStepError("sync", 1)]
        builder.build_one("v8")
        assert builder.consecutive_burns("v8") == burns


class TestRetry:
    def test_a_retry_is_picked_up_ahead_of_new_work(self, builder):
        """Entries above the failed commit exist by then, so the frontier rule
        would never reach it again."""
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")  # 101 fails
        builder.build_one("v8")  # 102 publishes
        assert builder.store.request_build_retry("v8", 101)
        assert builder.next_commit("v8")["commit_id"] == 101

    def test_a_successful_retry_clears_the_allowance(self, builder):
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        builder.store.request_build_retry("v8", 101)
        assert builder.build_one("v8").commit_id == 101
        assert builder.store.get_build_state("v8", 101) is None
        assert builder.next_commit("v8")["commit_id"] == 102

    def test_a_failed_retry_is_not_sticky(self, builder):
        """Otherwise one operator action head-of-line blocks the engine."""
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        builder.store.request_build_retry("v8", 101)
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        assert builder.store.get_build_state("v8", 101)["status"] == "compile_failed"
        assert builder.next_commit("v8")["commit_id"] == 102

    def test_a_burned_commit_reburns_rather_than_retrying_forever(self, builder):
        builder.cfg.build.max_infra_attempts = 1
        builder.failures = [BuildStepError("sync", 1)]
        builder.build_one("v8")
        assert builder.store.get_build_state("v8", 101)["status"] == "infra_burned"
        builder.store.request_build_retry("v8", 101)
        builder.failures = [BuildStepError("sync", 1)]
        assert builder.build_one("v8").status == "infra_burned"


class TestDiskFloor:
    def test_publishing_stops_below_the_floor(self, builder, monkeypatch):
        monkeypatch.setattr(builder, "free_gb", lambda: 5.0)
        builder.cfg.build.min_free_gb = 100
        assert builder.build_one("v8").published is False
        assert builder.bus.commit_ids("v8") == []
        state = builder.bus.read_builder_state("v8")
        assert state.publishing_paused_by_floor
        assert "free" in state.last_error

    def test_retention_does_not_shrink_under_pressure(self, builder, monkeypatch):
        """A budget that silently shrank would delete unread entries exactly
        when nobody is watching."""
        builder.build_one("v8")
        monkeypatch.setattr(builder, "free_gb", lambda: 5.0)
        builder.cfg.build.min_free_gb = 100
        builder.build_one("v8")
        assert builder.bus.commit_ids("v8") == [101]


class TestRetention:
    def test_prunes_after_publishing(self, builder):
        builder.cfg.build.retain_gb = 30 / 1_000_000_000  # 30 bytes
        for _ in range(3):
            builder.build_one("v8")
        ids = builder.bus.commit_ids("v8")
        assert ids and ids[-1] == 103
        assert builder.bus.read_builder_state("v8").lowest_retained == ids[0]


class TestPackaging:
    def test_round_trip_through_tar_and_zstd(self, tmp_path):
        src = tmp_path / "src"
        (src / "out" / "lib").mkdir(parents=True)
        (src / "out" / "d8").write_bytes(b"\x7fELF fake")
        (src / "out" / "lib" / "icu.dat").write_bytes(b"data")
        (src / "out" / "link").symlink_to("d8")
        (src / "ignored.o").write_bytes(b"junk")

        dest = tmp_path / "a.tar.zst"
        package(src, ["out/d8", "out/lib", "out/link"], dest, caffeinate=False)

        out = tmp_path / "unpacked"
        out.mkdir()
        subprocess.run(
            ["tar", "--use-compress-program=zstd", "-xf", str(dest), "-C", str(out)],
            check=True,
        )
        assert (out / "out" / "d8").read_bytes() == b"\x7fELF fake"
        assert (out / "out" / "lib" / "icu.dat").read_bytes() == b"data"
        # Symlinks are preserved, not followed: dereferencing a framework
        # bundle would silently change what the archive contains.
        assert (out / "out" / "link").is_symlink()
        assert not (out / "ignored.o").exists()

    def test_a_missing_run_set_entry_is_an_error(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        with pytest.raises(BuildError, match="missing"):
            package(src, ["out/d8"], tmp_path / "a.tar.zst", caffeinate=False)

    def test_no_partial_archive_is_left_behind(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        src.mkdir()
        (src / "f").write_bytes(b"x")
        dest = tmp_path / "a.tar.zst"
        monkeypatch.setenv("PATH", str(tmp_path / "nothing"))
        with pytest.raises((BuildError, OSError, FileNotFoundError)):
            package(src, ["f"], dest, caffeinate=False)
        assert not dest.exists()

    def test_the_archive_holds_no_appledouble_members(self, tmp_path):
        """COPYFILE_DISABLE: a ._ member would change what a codesign proves."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "f").write_bytes(b"x")
        dest = tmp_path / "a.tar.zst"
        package(src, ["f"], dest, caffeinate=False)
        raw = subprocess.run(
            ["zstd", "-dc", str(dest)], capture_output=True, check=True
        ).stdout
        (tmp_path / "a.tar").write_bytes(raw)
        with tarfile.open(tmp_path / "a.tar") as tf:
            assert not any(Path(n).name.startswith("._") for n in tf.getnames())


class TestBuildCfgHash:
    def test_changes_with_the_run_set(self, tmp_path):
        a = _engine(tmp_path, run_set=["out/d8"])
        b = _engine(tmp_path, run_set=["out/d8", "out/icudtl.dat"])
        assert build_cfg_hash(a) != build_cfg_hash(b)

    def test_is_order_independent(self, tmp_path):
        a = _engine(tmp_path, run_set=["out/d8", "out/icu"])
        b = _engine(tmp_path, run_set=["out/icu", "out/d8"])
        assert build_cfg_hash(a) == build_cfg_hash(b)

    def test_changes_with_the_build_args(self, tmp_path):
        a = _engine(tmp_path, gn_args="is_debug = false")
        b = _engine(tmp_path, gn_args="is_debug = true")
        assert build_cfg_hash(a) != build_cfg_hash(b)

    def test_ignores_gn_arg_formatting(self, tmp_path):
        a = _engine(tmp_path, gn_args="is_debug = false\nx = 1")
        b = _engine(tmp_path, gn_args="# a comment\nx=1\nis_debug=false\n")
        assert build_cfg_hash(a) == build_cfg_hash(b)


class TestCycle:
    def test_holds_the_machine_lock_across_the_build(self, builder, monkeypatch):
        from slipstream.lock import MachineLock

        peer = MachineLock("bencher", builder.lock.path)
        held = []
        real = builder.build_one
        monkeypatch.setattr(
            builder,
            "build_one",
            lambda name: (held.append(peer.try_acquire()), real(name))[1],
        )
        assert builder.run_cycle(["v8"], lambda: False) == 1
        assert held == [False]
        assert peer.try_acquire()

    def test_a_fetch_failure_skips_the_engine(self, builder, monkeypatch):
        from slipstream.collector import FetchError

        def boom(name, fetch=True):
            raise FetchError(name)

        monkeypatch.setattr(builder.collector, "head_commit_id", boom)
        assert builder.run_cycle(["v8"], lambda: False) == 0
        assert any("fetch" in m for m in builder.logs)

    def test_a_stop_request_ends_the_cycle(self, builder):
        assert builder.run_cycle(["v8"], lambda: True) == 0
        assert builder.bus.commit_ids("v8") == []


class TestReportedFrontier:
    def test_a_terminal_failure_moves_the_reported_frontier(self, builder):
        """Otherwise a consumer sees a cursor at the frontier and no reason
        for the gap between it and the newest entry."""
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        state = builder.bus.read_builder_state("v8")
        assert state.frontier == 101
        assert state.failed[0]["commit_id"] == 101

    def test_a_non_terminal_failure_does_not_move_it(self, builder):
        builder.build_one("v8")  # publish 101
        builder.failures = [BuildStepError("sync", 1)]
        builder.build_one("v8")
        assert builder.bus.read_builder_state("v8").frontier == 101


class TestStatePublishing:
    def test_a_stalled_cycle_still_refreshes_the_state_file(self, builder):
        """An operator diagnosing a stall must not keep reading a floor error
        from a disk problem that was fixed days ago."""
        builder.cfg.build.max_consecutive_burns = 2
        builder.cfg.build.max_infra_attempts = 1
        for _ in range(2):
            builder.failures = [BuildStepError("sync", 1)]
            builder.build_one("v8")
        state = builder.bus.read_builder_state("v8")
        state.publishing_paused_by_floor = True
        state.last_error = "only 40GB free"
        state.in_flight = {"commit_id": 1, "phase": "build", "started_at": 0}
        builder.bus.write_builder_state("v8", state)

        builder.build_one("v8")  # stalled, outside the retry window
        after = builder.bus.read_builder_state("v8")
        assert after.publishing_paused_by_floor is False
        assert after.in_flight is None
        assert after.stalled_since is not None

    def test_retention_records_what_it_dropped(self, builder):
        """The provable dropped-unread signal: commit ids are not contiguous,
        so lowest_retained against a cursor proves nothing."""
        builder.cfg.build.retain_gb = 30 / 1_000_000_000
        for _ in range(3):
            builder.build_one("v8")
        state = builder.bus.read_builder_state("v8")
        assert state.highest_dropped is not None
        assert state.highest_dropped < state.frontier


class TestDaemonSurvival:
    """A build cycle reports and moves on; it does not exit with a traceback."""

    def test_a_missing_packaging_tool_is_recorded(self, builder, monkeypatch):
        """launchd's PATH is minimal, so tar or zstd going missing is plausible."""

        def no_tar(*a, **k):
            raise FileNotFoundError(2, "No such file or directory", "tar")

        monkeypatch.setattr("slipstream.builder.package", no_tar)
        assert builder.run_cycle(["v8"], lambda: False) == 0
        assert builder.store.get_build_state("v8", 101)["status"] == "infra_retry"

    def test_a_full_disk_during_publish_is_recorded(self, builder, monkeypatch):
        def enospc(*a, **k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(builder.bus, "publish", enospc)
        assert builder.run_cycle(["v8"], lambda: False) == 0
        assert builder.store.get_build_state("v8", 101)["status"] == "infra_retry"

    def test_a_full_or_locked_db_is_recorded(self, builder, monkeypatch):
        """Every build_state write goes through sqlite3, and sqlite3.Error is
        not an OSError."""
        import sqlite3

        def full(*a, **k):
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(type(builder.store), "get_build_state", full)
        assert builder.run_cycle(["v8"], lambda: False) == 0
        assert any("disk is full" in m for m in builder.logs)

    def test_an_unreadable_state_file_does_not_kill_the_cycle(self, builder):
        import json

        path = builder.bus.builder_state_path("v8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 99}))
        assert builder.run_cycle(["v8"], lambda: False) == 0
        assert any("version 99" in m for m in builder.logs)


class TestAttemptCounting:
    def test_a_burn_counts_one_attempt_not_two(self, builder):
        builder.cfg.build.max_infra_attempts = 2
        for _ in range(2):
            builder.failures = [BuildStepError("sync", 1)]
            builder.build_one("v8")
        row = builder.store.get_build_state("v8", 101)
        assert row["status"] == "infra_burned" and row["attempts"] == 2


class TestRetryFailureIsNotStranded:
    """A retried commit is below the frontier, so it is only reachable through
    its own allowance; losing that loses the commit with nothing reporting it."""

    def test_an_infra_failure_during_a_retry_keeps_the_allowance(self, builder):
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")  # 101 fails
        builder.build_one("v8")  # 102 publishes, frontier moves past 101
        builder.store.request_build_retry("v8", 101)

        builder.failures = [BuildStepError("sync", 1)]
        assert builder.build_one("v8").commit_id == 101
        assert builder.store.build_retries_requested("v8") == [101]
        assert builder.next_commit("v8")["commit_id"] == 101

    def test_it_still_burns_once_the_attempts_run_out(self, builder):
        builder.cfg.build.max_infra_attempts = 2
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        builder.store.request_build_retry("v8", 101)
        for _ in range(2):
            builder.failures = [BuildStepError("sync", 1)]
            builder.build_one("v8")
        row = builder.store.get_build_state("v8", 101)
        assert row["status"] == "infra_burned"
        assert builder.store.build_retries_requested("v8") == []
        # Terminal, so it is visible rather than silently gone.
        assert [r["commit_id"] for r in builder.store.build_failures("v8")] == [101]

    def test_a_compile_failure_during_a_retry_is_still_terminal(self, builder):
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        builder.store.request_build_retry("v8", 101)
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")
        assert builder.store.get_build_state("v8", 101)["status"] == "compile_failed"
        assert builder.next_commit("v8")["commit_id"] == 102


class TestNoStaleInFlight:
    def test_no_stale_in_flight_on_the_up_to_date_branch(self, builder):
        """A builder killed after publishing but before writing its state."""
        from slipstream.bus import BuilderState

        builder.history = [100]  # nothing new upstream
        builder.bus.write_builder_state(
            "v8",
            BuilderState(
                frontier=100,
                in_flight={"commit_id": 100, "phase": "package", "started_at": 1},
            ),
        )
        builder.build_one("v8")
        assert builder.bus.read_builder_state("v8").in_flight is None

    def test_no_stale_in_flight_on_the_disk_floor_branch(self, builder, monkeypatch):
        from slipstream.bus import BuilderState

        builder.bus.write_builder_state(
            "v8",
            BuilderState(
                in_flight={"commit_id": 100, "phase": "build", "started_at": 1}
            ),
        )
        monkeypatch.setattr(builder, "free_gb", lambda: 5.0)
        builder.cfg.build.min_free_gb = 100
        builder.build_one("v8")
        state = builder.bus.read_builder_state("v8")
        assert state.in_flight is None and state.publishing_paused_by_floor


class TestBuildCfgHashCoversOverrides:
    """The config template offers build_cmd and sync_cmd as per-box settings,
    so two boxes building differently must not produce the same hash."""

    def _e(self, tmp_path, **kw):
        return _engine(tmp_path, gn_args="is_debug = false", **kw)

    def test_build_cmd_is_covered_even_with_gn_args(self, tmp_path):
        a = self._e(tmp_path)
        b = self._e(tmp_path)
        b.build_cmd = "autoninja -C out/other d8"
        assert build_cfg_hash(a) != build_cfg_hash(b)

    def test_sync_cmd_is_covered(self, tmp_path):
        a = self._e(tmp_path)
        b = self._e(tmp_path)
        b.sync_cmd = "gclient sync --nohooks"
        assert build_cfg_hash(a) != build_cfg_hash(b)

    def test_pre_build_patches_are_covered(self, tmp_path):
        a = self._e(tmp_path)
        b = self._e(tmp_path)
        b.pre_build_patches = ["Source/Base.xcconfig"]
        assert build_cfg_hash(a) != build_cfg_hash(b)

    def test_identical_config_still_matches(self, tmp_path):
        assert build_cfg_hash(self._e(tmp_path)) == build_cfg_hash(self._e(tmp_path))


class TestStateIsRefreshedOnEveryPublish:
    def test_a_wiped_state_dir_is_rebuilt_without_a_publish(self, builder):
        """The up-to-date branch publishes too; it used to send whatever the
        fields happened to hold, so a wiped bus/state stayed empty forever."""
        builder.build_one("v8")  # publishes 101
        builder.failures = [BuildStepError("compile", 1)]
        builder.build_one("v8")  # 102 fails terminally

        builder.bus.builder_state_path("v8").unlink()
        builder.history = [100, 101, 102]  # nothing new upstream
        builder.build_one("v8")

        state = builder.bus.read_builder_state("v8")
        assert state.frontier == 102
        assert state.lowest_retained == 101
        assert [f["commit_id"] for f in state.failed] == [102]

    def test_the_floor_branch_reports_the_same_facts(self, builder, monkeypatch):
        builder.build_one("v8")
        builder.bus.builder_state_path("v8").unlink()
        monkeypatch.setattr(builder, "free_gb", lambda: 5.0)
        builder.cfg.build.min_free_gb = 100
        builder.build_one("v8")
        state = builder.bus.read_builder_state("v8")
        assert state.frontier == 101 and state.publishing_paused_by_floor


class TestBurnsAreCountedAgainstTheTopic:
    """ "Consecutive" means with no successful publish between them, or three
    unrelated outages months apart stall a healthy engine."""

    def test_an_empty_topic_does_not_resurrect_old_burns(self, builder):
        for cid in (10, 20, 30):
            builder.store.record_build_failure("v8", cid, "infra_burned", "sync")
        # The bus root was recreated; start_from says where we are now.
        assert builder.bus.commit_ids("v8") == []
        assert builder.cfg.build.start_from == {"v8": 100}
        assert builder.consecutive_burns("v8") == 0

    def test_burns_above_the_floor_still_count(self, builder):
        for cid in (101, 102, 103):
            builder.store.record_build_failure("v8", cid, "infra_burned", "sync")
        assert builder.consecutive_burns("v8") == 3

    def test_a_publish_becomes_the_floor(self, builder):
        for cid in (101, 102):
            builder.store.record_build_failure("v8", cid, "infra_burned", "sync")
        builder.history = [100, 101, 102, 103]
        builder.build_one("v8")  # publishes 103
        assert builder.consecutive_burns("v8") == 0

    def test_no_topic_and_no_start_point_counts_nothing(self, builder):
        builder.cfg.build.start_from = {}
        for cid in (10, 20, 30):
            builder.store.record_build_failure("v8", cid, "infra_burned", "sync")
        assert builder.consecutive_burns("v8") == 0
