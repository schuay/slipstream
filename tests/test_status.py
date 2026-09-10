# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import time

import pytest

from slipstream.bus import (
    BenchState,
    Bus,
    BuilderState,
    Entry,
    write_cursor,
    cursor_path,
)
from slipstream.collector import BenchCollector
from slipstream.lock import MachineLock
from slipstream.config import BusConfig, BusSource, EngineConfig, RunSpec
from slipstream.status import collect_status, compare_env, render


def _entry(commit_id):
    return Entry(
        engine="v8",
        commit_id=commit_id,
        hash=f"hash{commit_id}",
        date="2026-09-06",
        timestamp=1757116800,
        title=f"commit {commit_id}",
        build_cfg_hash="sha256:cfg",
        blob_sha256="sha",
        blob_bytes=4,
    )


@pytest.fixture
def env(config, tmp_path, monkeypatch):
    bus_root = tmp_path / "bus"
    config.bus = BusConfig(
        root=bus_root,
        sources=[BusSource(name="local", root=str(bus_root), engines=["v8"])],
    )
    config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=None,
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
    )
    collector = BenchCollector(config, role="status")
    collector.lock.path = tmp_path / "machine.lock"
    monkeypatch.setattr(collector, "harness_revs", lambda: {"js3": "abc1234"})
    monkeypatch.setattr(
        "slipstream.status.MachineLock",
        lambda role: MachineLock(role, tmp_path / "machine.lock"),
    )
    monkeypatch.setattr("slipstream.status.paused_until", lambda: None)
    bus = Bus(bus_root)

    def publish(commit_id, payload=b"blob"):
        tmp = bus.tmp_blob("v8", commit_id)
        tmp.write_bytes(payload)
        bus.publish(_entry(commit_id), tmp)

    return type(
        "Env",
        (),
        {
            "cfg": config,
            "bus": bus,
            "collector": collector,
            "publish": staticmethod(publish),
            "tmp_path": tmp_path,
        },
    )


def _status(env, name="v8"):
    report = collect_status(env.cfg, env.collector, [name])
    (st,) = report.engines
    return report, st


class TestBusDrivenEngine:
    def test_frontier_cursor_and_lag(self, env):
        for cid in (100, 101, 102):
            env.publish(cid)
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=102, lowest_retained=100)
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 100)

        _, st = _status(env)
        assert st.driven_by == "bus" and st.source == "local"
        assert (st.frontier, st.cursor, st.lag) == (102, 100, 2)
        assert st.builder_age is not None

    def test_entries_dropped_before_this_machine_read_them(self, env):
        """Retention deliberately does not coordinate; this is the only signal."""
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=200, lowest_retained=150, highest_dropped=149)
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 120)
        _, st = _status(env)
        assert st.dropped_unread_up_to == 149

    def test_no_drop_when_retention_is_behind_the_cursor(self, env):
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=200, lowest_retained=150, highest_dropped=149)
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 150)
        _, st = _status(env)
        assert st.dropped_unread_up_to is None

    def test_sparse_commit_ids_are_not_read_as_a_drop(self, env):
        """jsc's path_filter leaves gaps in the published ids, so a gap between
        the cursor and lowest_retained is not evidence of anything."""
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=110, lowest_retained=105, highest_dropped=100)
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 100)
        _, st = _status(env)
        assert st.dropped_unread_up_to is None

    def test_a_stalled_or_paused_builder_is_surfaced(self, env):
        env.bus.write_builder_state(
            "v8",
            BuilderState(
                frontier=100,
                stalled_since=time.time() - 3600,
                publishing_paused_by_floor=True,
                last_error="sync failed on 101 (attempt 5)",
                failed=[{"commit_id": 101, "kind": "sync", "at": 1}],
            ),
        )
        _, st = _status(env)
        assert st.stalled_since and st.paused_by_floor
        assert st.builder_error and len(st.failed_builds) == 1

    def test_a_builder_that_never_wrote_state_is_visible(self, env):
        """A crashed builder otherwise leaves a plausible frontier forever."""
        _, st = _status(env)
        assert st.builder_age is None

    def test_disk_use_is_reported_per_engine(self, env):
        env.publish(100, payload=b"x" * 1000)
        _, st = _status(env)
        assert st.blob_gb > 0

    def test_a_torn_cursor_is_reported_not_raised(self, env):
        path = cursor_path(env.cfg.out_dir, "local", "v8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("half-writ")
        _, st = _status(env)
        assert "unreadable cursor" in st.error


class TestGitDrivenEngine:
    def test_reports_only_local_facts(self, env):
        env.cfg.bus.sources = []
        env.collector.store.mark_done("v8", env.cfg.platform, 100, status="ok")
        _, st = _status(env)
        assert st.driven_by == "git" and st.source is None
        assert st.status_counts == {"ok": 1}
        assert st.frontier is None


class TestStatusDoesNotTakeTheLock:
    def test_it_leaves_the_lock_file_alone(self, env, tmp_path):
        """try_acquire creates the file and writes a holder record; a
        read-only report must not be able to fail an ad-hoc command."""
        lock_path = tmp_path / "machine.lock"
        assert not lock_path.exists()
        collect_status(env.cfg, env.collector, ["v8"])
        assert not lock_path.exists(), "bus status created the lock file"

    def test_it_does_not_rewrite_an_existing_holder_record(self, env, tmp_path):
        from slipstream.lock import MachineLock

        holder = MachineLock("watch", tmp_path / "machine.lock")
        holder.try_acquire()
        before = (tmp_path / "machine.lock").read_bytes()
        report = collect_status(env.cfg, env.collector, ["v8"])
        assert (tmp_path / "machine.lock").read_bytes() == before
        assert report.lock_holder and "watch" in report.lock_holder
        holder.release()


class TestLocalFacts:
    def test_status_counts_and_orphans(self, env):
        store = env.collector.store
        store.mark_done("v8", env.cfg.platform, 100, status="ok")
        store.mark_done("v8", env.cfg.platform, 101, status="failed")
        store.insert_scores(
            "v8",
            env.cfg.platform,
            102,
            0,
            [
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "b",
                    "metric": "Total-Score",
                    "run": 1,
                    "score": 1.0,
                }
            ],
        )
        _, st = _status(env)
        assert st.status_counts == {"ok": 1, "failed": 1}
        assert st.orphan_scores == 1

    def test_provenance_split(self, env):
        store = env.collector.store
        store.record_run_env("v8", 100, {"source": "bus"})
        store.record_run_env("v8", 101, {"source": "local"})
        _, st = _status(env)
        assert st.provenance == {"bus": 1, "local": 1}

    def test_the_machine_lock_holder_is_named(self, env):
        env.collector.lock.role = "watch"
        env.collector.lock.try_acquire()
        report, _ = _status(env)
        assert report.lock_holder and "watch" in report.lock_holder
        env.collector.lock.release()

    def test_a_free_lock_says_so(self, env):
        report, _ = _status(env)
        assert report.lock_holder is None


class TestEnvironmentDivergence:
    def test_reports_what_differs_and_since_when(self, env):
        since = time.time() - 3 * 7 * 86400
        env.bus.write_bench_state(
            "v8",
            BenchState(
                env={
                    "slipstream_version": "0.0.1",
                    "os_version": "26.0",
                    "harness": {"js3": "def5678"},
                    "run_configs": ["js3/default"],
                    "since": since,
                }
            ),
        )
        _, st = _status(env)
        assert st.env_since == since
        assert any("slipstream_version" in line for line in st.env_divergence)
        assert any("harness" in line for line in st.env_divergence)

    def test_no_divergence_when_they_match(self, env):
        from slipstream.status import local_env

        mine = local_env(env.cfg, env.collector)
        env.bus.write_bench_state("v8", BenchState(env={**mine, "since": 1.0}))
        _, st = _status(env)
        assert st.env_divergence == []


class TestCompareEnv:
    def test_run_configs_compared_as_sets(self):
        local = {"run_configs": ["a", "b"], "harness": {}}
        assert compare_env(local, {"run_configs": ["b", "a"]}) == []
        (line,) = compare_env(local, {"run_configs": ["a"]})
        assert "only here ['b']" in line

    def test_keys_absent_from_the_source_are_not_reported(self):
        assert compare_env({"harness": {}}, {}) == []


def test_render_produces_lines(env):
    env.publish(100)
    env.bus.write_builder_state("v8", BuilderState(frontier=100, lowest_retained=100))
    report, _ = _status(env)
    lines = []
    render(report, lines.append)
    text = "\n".join(lines)
    assert "free space" in text and "machine lock" in text and "v8 (bus" in text


class TestPerEngineRunConfigs:
    def test_an_engine_with_fewer_variants_does_not_look_divergent(
        self, config, tmp_path, monkeypatch
    ):
        """local_env used to union every engine's configs and compare that
        against one engine's, so jsc always looked different from itself."""
        from slipstream.status import local_env

        config.runs.append(
            RunSpec(
                engine="v8",
                suite="js3",
                flags=("--turbolev-future",),
                variant="turbolev_future",
            )
        )
        config.engines["v8"] = EngineConfig(
            name="v8",
            src_dir=None,
            build_cmd="true",
            binary_path="out/d8",
            id_regex=r"#([0-9]+)",
        )
        config.engines["jsc"] = EngineConfig(
            name="jsc",
            src_dir=None,
            build_cmd="true",
            binary_path="jsc",
            id_regex=r"/([0-9]+)@",
        )
        collector = BenchCollector(config, role="status")
        monkeypatch.setattr(collector, "harness_revs", lambda: {})
        jsc = local_env(config, collector, "jsc")["run_configs"]
        v8 = local_env(config, collector, "v8")["run_configs"]
        assert "js3/turbolev_future" in v8
        assert "js3/turbolev_future" not in jsc
        assert (
            compare_env(local_env(config, collector, "jsc"), {"run_configs": jsc}) == []
        )


class TestUnreachableSource:
    def test_status_reports_rather_than_raising(self, env, monkeypatch):
        """An unreachable source is what this command exists for. ssh signals
        every failure with CalledProcessError, which is not a RuntimeError."""
        import subprocess

        class Dead:
            def __getattr__(self, name):
                def boom(*a, **k):
                    raise subprocess.CalledProcessError(255, "ssh")

                return boom

        monkeypatch.setattr("slipstream.status.open_source", lambda src: Dead())
        _, st = _status(env)
        assert st.error and "255" in st.error
        # The local facts still come through.
        assert st.status_counts == {}


class TestLocalBencherState:
    """The source's builder is not what stopped when nothing is being benched."""

    def test_a_paused_local_bencher_is_reported(self, env):
        from slipstream.bus import BenchState

        env.bus.write_builder_state("v8", BuilderState(frontier=200))
        env.bus.write_bench_state(
            "v8",
            BenchState(
                cursor=100,
                benching_paused_by_floor=True,
                last_error="only 40GB free",
            ),
        )
        _, st = _status(env)
        assert st.bench_paused_by_floor and st.bench_error == "only 40GB free"
        assert st.bench_age is not None

        lines = []
        render(collect_status(env.cfg, env.collector, ["v8"]), lines.append)
        text = "\n".join(lines)
        assert "bencher paused" in text and "only 40GB free" in text

    def test_in_flight_work_is_distinguishable_from_a_hang(self, env):
        from slipstream.bus import BenchState

        env.bus.write_builder_state(
            "v8",
            BuilderState(
                frontier=200,
                in_flight={"commit_id": 201, "phase": "package", "started_at": 1},
            ),
        )
        env.bus.write_bench_state(
            "v8",
            BenchState(
                cursor=100,
                in_flight={"commit_id": 101, "phase": "bench", "started_at": 1},
            ),
        )
        lines = []
        render(collect_status(env.cfg, env.collector, ["v8"]), lines.append)
        text = "\n".join(lines)
        assert "builder is on 201 (package)" in text
        assert "benching 101 since" in text


class TestBenchAgeIsAlwaysShown:
    def test_a_killed_bencher_shows_its_age_beside_the_in_flight_line(self, env):
        """The age is the only thing that tells a dead daemon from a busy one,
        and in_flight is exactly what a killed one leaves behind."""
        from slipstream.bus import BenchState

        env.bus.write_builder_state("v8", BuilderState(frontier=200))
        env.bus.write_bench_state(
            "v8",
            BenchState(
                cursor=100,
                in_flight={"commit_id": 101, "phase": "bench", "started_at": 1},
            ),
        )
        lines = []
        render(collect_status(env.cfg, env.collector, ["v8"]), lines.append)
        text = "\n".join(lines)
        assert "bench state written" in text
        assert "benching 101 since" in text


class TestDropReportWithNoCursor:
    def test_a_box_that_has_benched_nothing_still_sees_the_gap(self, env):
        """A new box, or one whose cursor file was lost, pointed at a builder
        that has already pruned: that is exactly when the field is wanted."""
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=200, lowest_retained=150, highest_dropped=149)
        )
        _, st = _status(env)
        assert st.cursor is None
        assert st.dropped_unread_up_to == 149

        lines = []
        render(collect_status(env.cfg, env.collector, ["v8"]), lines.append)
        assert "cursor unset" in "\n".join(lines)
