# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import time

import pytest

from slipstream.bus import (
    BenchState,
    Blob,
    Bus,
    BuilderState,
    Entry,
    write_cursor,
    cursor_path,
)
from slipstream.collector import BenchCollector
from slipstream.lock import MachineLock
from slipstream.config import BusConfig, BusSource, EngineConfig
from slipstream import host
from slipstream.status import collect_status, render
from keys import K1


def _entry(commit_id, blobs):
    return Entry(
        engine="v8",
        commit_id=commit_id,
        hash=f"hash{commit_id}",
        date="2026-09-06",
        timestamp=1757116800,
        title=f"commit {commit_id}",
        build_cfg_hash="sha256:cfg",
        blobs=blobs,
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
        blob_id = f"id{commit_id}"
        tmp = bus.tmp_blob(blob_id)
        tmp.write_bytes(payload)
        bus.store_blob(tmp, blob_id)
        bus.publish(_entry(commit_id, [Blob("out", blob_id, "sha", len(payload))]))

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
    def test_permanent_skip_remains_visible_after_cursor_advances(self, env):
        env.bus.write_bench_state(
            "v8",
            BenchState(
                skipped_permanent=K1(100),
                permanent_skip_reason="immutable publication defect",
            ),
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 101)
        report, st = _status(env)
        assert st.bench_skipped_permanent == K1(100)
        lines = []
        render(report, lines.append)
        assert any(
            "last permanent publication skip: 100: immutable publication defect" in line
            for line in lines
        )

    def test_frontier_cursor_and_lag(self, env):
        for cid in (100, 101, 102):
            env.publish(cid)
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=102, lowest_retained=100)
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 100)

        _, st = _status(env)
        assert st.driven_by == "bus" and st.source == "local"
        assert (st.frontier, st.cursor, st.lag) == (K1(102), K1(100), 2)
        assert st.builder_age is not None

    def test_entries_dropped_before_this_machine_read_them(self, env):
        """Retention deliberately does not coordinate; this is the only signal."""
        env.bus.write_builder_state(
            "v8", BuilderState(frontier=200, lowest_retained=150, highest_dropped=149)
        )
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 120)
        _, st = _status(env)
        assert st.dropped_unread_up_to == K1(149)

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

    def test_v4_status_neither_takes_bus_lock_nor_flushes(self, env, monkeypatch):
        """A migrated bus must be inspectable while transfer or build holds
        .bus.lock, and status must never issue device-cache flushes."""
        import hashlib
        import json
        import uuid

        from slipstream.bus_protocol import BusProtocol
        from slipstream.durability import FileLock

        protocol = BusProtocol(env.bus.root)
        owner = protocol.bus_uuid()
        protocol.migrate({"v8": owner})
        for cid in (100, 101):
            payload = f"payload-{cid}".encode()
            sha = hashlib.sha256(payload).hexdigest()
            entry = Entry(
                "v8",
                cid,
                f"hash{cid}",
                "2026-09-06",
                1757116800,
                f"commit {cid}",
                "sha256:cfg",
                [Blob("out", f"tree{cid}", sha, len(payload))],
                version=4,
                owner_uuid=owner,
                publication_uuid=str(uuid.uuid4()),
            )
            with protocol.stage() as stage:
                p = stage.path / entry.objects()[0].name
                p.write_bytes(payload)
                stage.paths[p.name] = p
                stage.entry, stage.text = entry, entry.to_json()
                protocol.publish_staged(stage)
        protocol.write_builder_state("v8", BuilderState(frontier=101))
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 100)

        root_dir = env.bus.root / "roots" / "v8" / "100" / "sub"
        root_dir.mkdir(parents=True)
        (root_dir / "d8").write_bytes(b"r" * 500)
        (root_dir / "link").symlink_to(root_dir / "d8")

        monkeypatch.setattr(
            "slipstream.durability.flush_file",
            lambda fd: pytest.fail("bus status flushed a file"),
        )
        monkeypatch.setattr(
            BusProtocol,
            "lock",
            lambda self, timeout=30.0: pytest.fail("bus status took .bus.lock"),
        )
        with FileLock(env.bus.root / ".bus.lock", timeout=0):
            _, st = _status(env)
        assert (st.frontier, st.cursor, st.lag) == (K1(101), K1(100), 1)
        assert st.blob_gb == pytest.approx(22 / 1_000_000_000)
        assert st.roots_gb == pytest.approx(500 / 1_000_000_000)

        # Unrecovered pending prunes are reflected in memory without mutating disk.
        write_cursor(cursor_path(env.cfg.out_dir, "local", "v8"), 99)
        ret_path = protocol.state_path("retention", "v8")
        ret = json.loads(ret_path.read_text())
        ret["pending"] = [{"key": "100", "sha256": "deadbeef"}]
        ret_bytes = json.dumps(ret).encode()
        ret_path.write_bytes(ret_bytes)
        _, st = _status(env)
        assert st.lowest_retained == K1(101)
        assert st.dropped_unread_up_to == K1(100)
        assert st.blob_gb == pytest.approx(11 / 1_000_000_000)
        assert ret_path.read_bytes() == ret_bytes


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

    def test_orphan_scores_include_embedded_engines(self, env):
        from slipstream.models import CommitKey

        store = env.collector.store
        for emb in (1534000, 1534001):
            store.insert_scores(
                "v8",
                env.cfg.platform,
                CommitKey(emb, 100),
                0,
                [
                    {
                        "suite": "js3",
                        "flags": "default",
                        "benchmark": "b",
                        "metric": "Total-Score",
                        "run": run,
                        "score": 1.0,
                    }
                    for run in (1, 2)
                ],
            )
        store.upsert_commit(
            "v8", "h", CommitKey(1534000, 100), "d", 0, "t", embedder_hash="cr"
        )
        _, st = _status(env)
        assert st.orphan_scores == 1

    def test_provenance_split(self, env):
        store = env.collector.store
        store.record_run_env("v8", 100, {"source": "bus"})
        store.record_run_env("v8", 101, {"source": "local"})
        _, st = _status(env)
        assert st.provenance == {"bus": 1, "local": 1}
        lines = []
        render(_status(env)[0], lines.append)
        assert "  recent binaries: bus 1, local 1" in lines

    def test_a_single_provenance_is_not_rendered(self, env):
        env.collector.store.record_run_env("v8", 100, {"source": "bus"})
        report, st = _status(env)
        assert st.provenance == {"bus": 1}
        lines = []
        render(report, lines.append)
        assert not any("recent binaries" in line for line in lines)

    def test_the_machine_lock_holder_is_named(self, env):
        env.collector.lock.role = "watch"
        env.collector.lock.try_acquire()
        report, _ = _status(env)
        assert report.lock_holder and "watch" in report.lock_holder
        env.collector.lock.release()

    def test_a_free_lock_says_so(self, env):
        report, _ = _status(env)
        assert report.lock_holder is None
        assert report.lock_waiters == []

    def test_the_queue_is_listed_behind_the_holder(self, env):
        import threading

        env.collector.lock.role = "watch"
        env.collector.lock.try_acquire()
        waiter = MachineLock("build", env.tmp_path / "machine.lock")
        t = threading.Thread(
            target=lambda: (waiter.acquire(wait=True, job="v8 101"), waiter.release())
        )
        t.start()
        deadline = time.monotonic() + 5
        while not env.collector.lock.waiters() and time.monotonic() < deadline:
            time.sleep(0.005)
        report, _ = _status(env)
        env.collector.lock.release()
        t.join(timeout=5)
        assert len(report.lock_waiters) == 1
        assert "build" in report.lock_waiters[0] and "v8 101" in report.lock_waiters[0]
        lines = []
        render(report, lines.append)
        assert any(line.startswith("  queued 1: build") for line in lines)


class TestEnvironmentDivergence:
    def test_reports_what_differs_from_the_builder_and_since_when(self, env):
        since = time.time() - 3 * 7 * 86400
        theirs = {
            **host.builder_env(),
            "slipstream_version": "0.0.1",
            "os_version": "26.0",
            "since": since,
        }
        env.bus.write_builder_state("v8", BuilderState(env=theirs))
        report, st = _status(env)
        assert st.env_since == since
        assert [line.split(":")[0] for line in st.env_divergence] == [
            "slipstream_version",
            "os_version",
        ]
        lines = []
        render(report, lines.append)
        assert any("differs from the source's builder" in line for line in lines)

    def test_hw_model_is_not_compared(self, env):
        theirs = {**host.builder_env(), "hw_model": "Mac14,6", "since": 1.0}
        env.bus.write_builder_state("v8", BuilderState(env=theirs))
        _, st = _status(env)
        assert st.env_divergence == []

    def test_a_builder_that_publishes_no_env_shows_no_divergence(self, env):
        env.bus.write_builder_state("v8", BuilderState(frontier=100))
        _, st = _status(env)
        assert st.env_divergence == [] and st.env_since is None


def test_render_produces_lines(env):
    env.publish(100)
    env.bus.write_builder_state("v8", BuilderState(frontier=100, lowest_retained=100))
    report, _ = _status(env)
    lines = []
    render(report, lines.append)
    text = "\n".join(lines)
    assert "free space" in text and "machine activity" in text and "v8 (bus" in text
    assert "machine lock" not in text and "recent binaries" not in text


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
        assert st.dropped_unread_up_to == K1(149)

        lines = []
        render(collect_status(env.cfg, env.collector, ["v8"]), lines.append)
        assert "cursor unset" in "\n".join(lines)


class TestWindowStatus:
    def render_window(self, env):
        from slipstream.status import window_status, render_window

        lines = []
        render_window(window_status(env.cfg, "/selected/config.toml"), lines.append)
        return "\n".join(lines)

    def test_closed_ignores_stale_bus_in_flight(self, env):
        from slipstream.bench_window import write_override

        write_override("close")
        env.bus.write_bench_state(
            "v8", BenchState(in_flight={"commit_id": 123, "started_at": 1})
        )
        _, report = _status(env)
        assert report.bench_in_flight
        lines = self.render_window(env)
        assert "Normal use permitted" in lines
        assert "Finishing current commit" not in lines
        assert "for /selected/config.toml" in lines

    @pytest.mark.parametrize("role", ["bench", "watch"])
    def test_actual_git_manual_or_bus_holder_finishes(self, env, role):
        from slipstream.bench_window import write_override

        machine = MachineLock(role, env.collector.lock.path)
        machine.try_acquire("v8 123")
        try:
            write_override("close")
            lines = self.render_window(env)
            assert "Finishing current commit" in lines
            assert "v8 123" in lines and "started 20" in lines
            assert "Normal use permitted" not in lines
        finally:
            machine.release()
        assert "Normal use permitted" in self.render_window(env)

    def test_builder_is_separate_activity(self, env):
        from slipstream.bench_window import write_override

        write_override("close")
        machine = MachineLock("build", env.collector.lock.path)
        machine.try_acquire("jsc 123")
        try:
            lines = self.render_window(env)
            assert "Normal use permitted" in lines
            assert "build pid" in lines and "jsc 123" in lines
        finally:
            machine.release()

    def test_unidentified_holder_never_claims_normal_use(self, env):
        from slipstream.bench_window import write_override

        write_override("close")
        machine = MachineLock("watch", env.collector.lock.path)
        machine.try_acquire("v8 123")
        try:
            machine.path.write_text("corrupt metadata")
            lines = self.render_window(env)
            assert "activity unknown" in lines
            assert "Normal use permitted" not in lines
        finally:
            machine.release()

    def test_selected_config_changes_are_labeled_and_not_loaded_watch_policy(self, env):
        env.cfg.bench.timezone = "UTC"
        env.cfg.bench.allowed_windows = []
        lines = self.render_window(env)
        assert "Normal use permitted" in lines
        assert "for /selected/config.toml" in lines
        env.cfg.bench.allowed_windows = None
        assert "Benchmarking permitted" in self.render_window(env)

    def test_other_blockers_do_not_reserve_normal_use_when_window_open(self, env):
        env.bus.write_bench_state(
            "v8", BenchState(last_error="stalled", updated_at=time.time())
        )
        lines = self.render_window(env)
        assert "Benchmarking permitted" in lines
        assert "last reported blocker" in lines and "stalled" in lines
        assert "Normal use permitted" not in lines


def test_safari_acquisition_error_remains_visible_while_publishing(env):
    env.bus.write_builder_state(
        "v8", BuilderState(safari_acquisition_error="frozen release conflict")
    )
    _, status = _status(env)
    assert status.builder_error == "Safari acquisition: frozen release conflict"
