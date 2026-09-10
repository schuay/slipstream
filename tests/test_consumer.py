# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from slipstream.builder import package
from slipstream.bus import Bus, Entry, sha256_file
from slipstream.collector import BenchCollector, BenchOutcome
from slipstream.config import BusConfig, BusSource, EngineConfig
from slipstream.consumer import BusConsumer, ConsumerError, ShaMismatch


def _commit(commit_id):
    return {
        "hash": f"hash{commit_id}",
        "commit_id": commit_id,
        "date": "2026-09-06",
        "timestamp": 1757116800 + commit_id,
        "title": f"commit {commit_id}",
    }


@pytest.fixture
def setup(config, tmp_path, monkeypatch):
    """A local bus with a builder-shaped publisher and a consumer reading it."""
    bus_root = tmp_path / "bus"
    config.bus = BusConfig(
        root=bus_root,
        sources=[BusSource(name="local", root=str(bus_root), engines=["v8"])],
    )
    config.bench.min_free_gb = 0.001
    config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=None,  # a consumer needs no checkout
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
        run_set=["out"],
    )
    collector = BenchCollector(config, role="watch")
    collector.lock.path = tmp_path / "machine.lock"
    consumer = BusConsumer(config, collector)
    consumer.logs = []
    consumer.log = consumer.logs.append
    monkeypatch.setattr(collector, "harness_revs", lambda: {"js3": "abc1234"})

    bus = Bus(bus_root)

    def publish(commit_id, binary=b"#!/bin/sh\nexit 0\n"):
        src = tmp_path / "build" / str(commit_id)
        (src / "out").mkdir(parents=True, exist_ok=True)
        (src / "out" / "d8").write_bytes(binary)
        blob = bus.tmp_blob("v8", commit_id)
        package(src, ["out"], blob, caffeinate=False)
        commit = _commit(commit_id)
        entry = Entry(
            engine="v8",
            commit_id=commit_id,
            hash=commit["hash"],
            date=commit["date"],
            timestamp=commit["timestamp"],
            title=commit["title"],
            build_cfg_hash="sha256:cfg",
            blob_sha256=sha256_file(blob),
            blob_bytes=blob.stat().st_size,
            builder={"bot": "box2-m4", "toolchain": "clang-21"},
            built_at=1757116999,
            build_secs=1183,
        )
        bus.publish(entry, blob)
        return entry

    return type(
        "Setup",
        (),
        {
            "cfg": config,
            "bus": bus,
            "consumer": consumer,
            "collector": collector,
            "source": config.bus.sources[0],
            "publish": staticmethod(publish),
            "tmp_path": tmp_path,
        },
    )


def _drain(setup, runs=1, should_stop=None):
    """The bench count; tests that care about the reason use drain directly."""
    return _drain_result(setup, runs, should_stop).benched


def _drain_result(setup, runs=1, should_stop=None):
    return setup.consumer.drain(
        setup.source,
        "v8",
        runs,
        should_stop or (lambda: False),
        lambda: setup.collector.lock.try_acquire(),
        setup.collector.lock.release,
    )


@pytest.fixture
def scored(setup, monkeypatch):
    """Record which run roots were benched, without running a benchmark."""
    seen = []

    def fake(engine, commit_id, runs, run_root):
        seen.append((int(commit_id), Path(run_root)))
        return BenchOutcome(1, 1, 10)

    monkeypatch.setattr(setup.collector, "_run_benchmarks", fake)
    return seen


class TestBenchingLine:
    def test_it_names_the_commit_and_the_queue_behind_it(self, setup, scored):
        """`watch` otherwise shows benchmark progress with no way to tell which
        commit it is on, or how much is left."""
        entries = [setup.publish(cid) for cid in (100, 101, 102)]
        assert _drain(setup) == 3
        lines = [m for m in setup.consumer.logs if "benching" in m]
        assert entries[0].hash[:8] in lines[0]
        # The queue shrinks as the batch drains, and the entry being benched
        # is itself still above the cursor.
        assert [m.rsplit(", ", 1)[1] for m in lines] == [
            "3 above the cursor",
            "2 above the cursor",
            "1 above the cursor",
        ]


class TestEngineWithNoRunEntries:
    def test_the_cycle_reports_it_and_leaves_the_cursor(self, setup):
        """Advancing past entries nothing measured is irreversible, so this is
        a cycle error rather than a raise that would kill the daemon."""
        before = setup.consumer.cursor(setup.source, "v8")
        setup.cfg.runs[:] = []
        res = _drain_result(setup)
        assert res.benched == 0
        assert "would measure nothing" in res.error
        assert setup.consumer.cursor(setup.source, "v8") == before


class TestDraining:
    def test_benches_every_entry_in_one_cycle(self, setup, scored):
        """A cursor reset must not take one cycle per entry to walk back up."""
        for cid in (100, 101, 102):
            setup.publish(cid)
        assert _drain(setup) == 3
        assert [cid for cid, _ in scored] == [100, 101, 102]
        assert setup.consumer.cursor(setup.source, "v8") == 102

    def test_a_second_cycle_does_nothing(self, setup, scored):
        setup.publish(100)
        _drain(setup)
        assert _drain(setup) == 0

    def test_new_entries_are_picked_up(self, setup, scored):
        setup.publish(100)
        _drain(setup)
        setup.publish(101)
        assert _drain(setup) == 1
        assert setup.consumer.cursor(setup.source, "v8") == 101

    def test_a_done_commit_advances_without_fetching(self, setup, scored, monkeypatch):
        setup.publish(100)
        setup.publish(101)
        setup.collector.store.mark_done("v8", setup.cfg.platform, 100)
        fetched = []
        real = setup.consumer.provision
        monkeypatch.setattr(
            setup.consumer,
            "provision",
            lambda src, eng, entry: (
                fetched.append(entry.commit_id),
                real(src, eng, entry),
            )[1],
        )
        _drain(setup)
        assert fetched == [101]
        assert setup.consumer.cursor(setup.source, "v8") == 101

    def test_a_stop_request_ends_the_drain(self, setup, scored):
        for cid in (100, 101):
            setup.publish(cid)
        assert _drain(setup, should_stop=lambda: True) == 0

    def test_the_cursor_advances_only_after_the_commit_is_recorded(
        self, setup, monkeypatch
    ):
        setup.publish(100)
        seen = {}

        def fake(engine, commit_id, runs, run_root):
            seen["cursor_during"] = setup.consumer.cursor(setup.source, "v8")
            return BenchOutcome(1, 1, 10)

        monkeypatch.setattr(setup.collector, "_run_benchmarks", fake)
        _drain(setup)
        assert seen["cursor_during"] is None
        assert setup.consumer.cursor(setup.source, "v8") == 100

    def test_a_dropped_entry_is_skipped_not_reported(self, setup, scored):
        """Retention between the listing and the read is not an error."""
        setup.publish(100)
        setup.publish(101)
        setup.bus.entry_path("v8", 100).unlink()
        assert _drain(setup) == 1
        assert [cid for cid, _ in scored] == [101]


class TestProvisioning:
    def test_the_run_root_holds_the_unpacked_archive(self, setup, scored):
        setup.publish(100, binary=b"the real d8")
        _drain(setup)
        (_, root) = scored[0]
        assert (root / "out" / "d8").read_bytes() == b"the real d8"

    def test_a_corrupt_payload_is_refused(self, setup, scored):
        setup.publish(100)
        setup.bus.blob_path("v8", 100).write_bytes(b"not the payload")
        assert _drain(setup) == 0
        assert scored == []
        assert "sha256" in setup.bus.read_bench_state("v8").last_error

    def test_a_republished_entry_is_refetched_before_corruption_is_claimed(
        self, setup, scored
    ):
        """Payload names are reusable, so a republish looks exactly like
        corruption to a consumer holding the old entry."""
        setup.publish(100)
        stale = setup.bus.read_entry("v8", 100)
        setup.publish(100, binary=b"rebuilt")
        # Bench the stale entry directly: that is what a consumer that listed
        # before the republish is holding.
        from slipstream.consumer import LocalSource

        source = LocalSource(setup.source)
        setup.consumer.bench_entry(source, setup.cfg.engines["v8"], stale, 1)
        assert scored and (scored[0][1] / "out" / "d8").read_bytes() == b"rebuilt"

    def test_a_genuinely_corrupt_payload_still_raises(self, setup):
        from slipstream.consumer import LocalSource

        entry = setup.publish(100)
        setup.bus.blob_path("v8", 100).write_bytes(b"corrupt")
        with pytest.raises(ShaMismatch):
            setup.consumer.bench_entry(
                LocalSource(setup.source), setup.cfg.engines["v8"], entry, 1
            )

    def test_a_missing_payload_is_a_real_error(self, setup, scored):
        setup.publish(100)
        setup.bus.blob_path("v8", 100).unlink()
        assert _drain(setup) == 0
        assert "payload is missing" in setup.bus.read_bench_state("v8").last_error

    def test_a_local_source_keeps_the_builders_payload(self, setup, scored):
        setup.publish(100)
        _drain(setup)
        assert setup.bus.blob_path("v8", 100).exists()

    def test_only_run_roots_is_kept(self, setup, scored):
        setup.cfg.bench.run_roots = 2
        for cid in (100, 101, 102):
            setup.publish(cid)
        _drain(setup)
        roots = sorted(p.name for p in (setup.bus.root / "roots" / "v8").iterdir())
        assert roots == ["101", "102"]


class TestResumeAfterInterrupt:
    def test_partial_scores_are_dropped_before_the_re_bench(self, setup, monkeypatch):
        """scores is INSERT OR IGNORE with run in the key, so a retry that kept
        the interrupted run's rows would half-measure that run number."""
        store = setup.collector.store
        setup.publish(100)
        stale = {
            "suite": "js3",
            "flags": "default",
            "benchmark": "test-bench",
            "metric": "Total-Score",
            "run": 1,
            "score": 1.0,
        }
        store.insert_scores("v8", setup.cfg.platform, 100, 0, [stale])

        def fake(engine, commit_id, runs, run_root):
            store.insert_scores(
                "v8", setup.cfg.platform, 100, 0, [{**stale, "score": 99.0}]
            )
            return BenchOutcome(1, 1, 1)

        monkeypatch.setattr(setup.collector, "_run_benchmarks", fake)
        _drain(setup)
        scores = store.conn.execute(
            "SELECT score FROM scores WHERE commit_id=100"
        ).fetchall()
        assert [r[0] for r in scores] == [99.0]


class TestFreeSpace:
    def test_benching_stops_below_the_floor(self, setup, scored, monkeypatch):
        setup.publish(100)
        setup.cfg.bench.min_free_gb = 100
        monkeypatch.setattr(setup.consumer, "free_gb", lambda: 5.0)
        assert _drain(setup) == 0
        state = setup.bus.read_bench_state("v8")
        assert state.benching_paused_by_floor and "free" in state.last_error


class TestProvenance:
    def test_records_both_environments(self, setup, scored):
        setup.publish(100)
        _drain(setup, runs=3)
        row = setup.collector.store.get_run_env("v8", 100)
        assert row["source"] == "bus" and row["runs"] == 3
        assert row["build_cfg_hash"] == "sha256:cfg"
        # The builder's toolchain, because an update on it shifts both series.
        assert row["toolchain"] == "clang-21"
        assert json.loads(row["harness_revs"]) == {"js3": "abc1234"}
        assert "js3/default" in json.loads(row["run_configs"])
        assert row["slipstream_version"]

    def test_clearing_a_commit_drops_its_provenance(self, setup, scored):
        setup.publish(100)
        _drain(setup)
        setup.collector.store.clear_range("v8", setup.cfg.platform, [100])
        assert setup.collector.store.get_run_env("v8", 100) is None


class TestBenchState:
    def test_reports_cursor_lag_and_environment(self, setup, scored):
        for cid in (100, 101):
            setup.publish(cid)
        _drain(setup)
        state = setup.bus.read_bench_state("v8")
        assert state.cursor == 101 and state.lag == 0
        assert state.bot == setup.cfg.bot_name
        assert state.status_counts == {"ok": 2}
        assert state.env["harness"] == {"js3": "abc1234"}
        assert state.env["since"] > 0

    def test_lag_counts_unbenched_entries(self, setup, monkeypatch):
        for cid in (100, 101, 102):
            setup.publish(cid)
        done = []

        def one_then_stop(engine, commit_id, runs, run_root):
            done.append(int(commit_id))
            return BenchOutcome(1, 1, 10)

        monkeypatch.setattr(setup.collector, "_run_benchmarks", one_then_stop)
        _drain(setup, should_stop=lambda: bool(done))
        state = setup.bus.read_bench_state("v8")
        assert done == [100]
        assert state.cursor == 100 and state.lag == 2

    def test_since_is_kept_while_the_environment_is_unchanged(self, setup, scored):
        setup.publish(100)
        _drain(setup)
        first = setup.bus.read_bench_state("v8").env["since"]
        setup.publish(101)
        _drain(setup)
        assert setup.bus.read_bench_state("v8").env["since"] == first

    def test_since_moves_when_the_environment_changes(self, setup, scored, monkeypatch):
        setup.publish(100)
        _drain(setup)
        first = setup.bus.read_bench_state("v8").env["since"]
        monkeypatch.setattr(setup.collector, "harness_revs", lambda: {"js3": "def5678"})
        setup.publish(101)
        _drain(setup)
        assert setup.bus.read_bench_state("v8").env["since"] > first


class TestCursorRecovery:
    def test_a_torn_cursor_restarts_rather_than_guessing(self, setup, scored):
        """max_done_commit_id looks like the answer but is a maximum, not a
        watermark: an ad-hoc bench above the cursor would move it past entries
        that were never measured."""
        setup.publish(100)
        _drain(setup)
        setup.collector.store.mark_done("v8", setup.cfg.platform, 110500)
        path = setup.consumer.cursor_file(setup.source, "v8")
        path.write_text("half-writ")
        assert setup.consumer.cursor(setup.source, "v8") is None
        assert any("unreadable cursor" in m for m in setup.consumer.logs)

    def test_restarting_re_benches_only_what_is_not_done(self, setup, scored):
        for cid in (100, 101):
            setup.publish(cid)
        _drain(setup)
        assert [cid for cid, _ in scored] == [100, 101]
        setup.consumer.cursor_file(setup.source, "v8").write_text("half-writ")
        scored.clear()
        _drain(setup)
        assert scored == [], "the walk back up re-benched done commits"
        assert setup.consumer.cursor(setup.source, "v8") == 101


class TestUnpackSafety:
    def test_a_member_escaping_the_run_root_is_refused(self, setup, tmp_path):
        import tarfile

        from slipstream.consumer import _unpack

        evil = tmp_path / "evil.tar"
        with tarfile.open(evil, "w") as tf:
            victim = tmp_path / "victim"
            victim.write_text("x")
            tf.add(victim, arcname="../escaped")
        payload = tmp_path / "evil.tar.zst"
        import subprocess

        subprocess.run(["zstd", "-q", "-o", str(payload), str(evil)], check=True)
        dest = tmp_path / "root"
        dest.mkdir()
        with pytest.raises(ConsumerError, match="escapes"):
            _unpack(payload, dest)


def test_one_box_builds_and_benches_through_the_bus(config, tmp_path, monkeypatch):
    """Step 7a's shape: build and a bus-driven watch on one machine, no ssh.

    Exercises publish, list, verify, unpack, provenance and the machine lock
    against a real archive, with a d8 stand-in that prints a parseable score.
    """
    from slipstream.builder import Builder
    from slipstream.config import BenchmarkConfig

    src = tmp_path / "src"
    (src / "out").mkdir(parents=True)
    d8 = src / "out" / "d8"
    d8.write_text("#!/bin/sh\necho 'test-bench Total-Score 42.5 pts'\n")
    d8.chmod(0o755)

    bench_dir = tmp_path / "js3"
    bench_dir.mkdir()
    (bench_dir / "cli.js").write_text("// harness\n")
    config.benchmarks = {
        "js3": BenchmarkConfig(
            name="js3",
            dir=bench_dir,
            cli="cli.js",
            names=["test-bench"],
            score_regex=(
                r"^([A-Za-z0-9-]+)\s+([A-Za-z0-9-]+-Score|Score)\s+([0-9.]+)\s+pts$"
            ),
            run_mode="suite",
        )
    }
    bus_root = tmp_path / "bus"
    config.bus = BusConfig(
        root=bus_root,
        sources=[BusSource(name="local", root=str(bus_root), engines=["v8"])],
    )
    config.build.engines = ["v8"]
    config.build.start_from = {"v8": 100}
    config.build.min_free_gb = 0.001
    config.bench.min_free_gb = 0.001
    config.engines["v8"] = EngineConfig(
        name="v8",
        src_dir=src,
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
        run_set=["out"],
    )

    builder = Builder(config, Bus(bus_root))
    builder.lock.path = tmp_path / "machine.lock"
    monkeypatch.setattr(
        builder.collector,
        "next_commit_after",
        lambda engine, cid: _commit(101) if cid < 101 else None,
    )
    monkeypatch.setattr(builder.collector, "build_at", lambda *a, **k: None)
    assert builder.build_one("v8").published

    # Poison the checkout: if the bench ran the source tree's binary rather
    # than the unpacked archive's, the score below would be 0.1.
    d8.write_text("#!/bin/sh\necho 'test-bench Total-Score 0.1 pts'\n")

    collector = BenchCollector(config, role="watch")
    collector.lock.path = tmp_path / "machine.lock"
    monkeypatch.setattr(collector, "harness_revs", lambda: {"js3": "abc1234"})
    consumer = BusConsumer(config, collector)
    result = consumer.drain(
        config.bus.sources[0],
        "v8",
        1,
        lambda: False,
        lambda: collector.lock.try_acquire(),
        collector.lock.release,
    )

    assert result.benched == 1 and result.error is None
    store = collector.store
    rows = store.get_series("v8", "js3", "default", "test-bench", "Total-Score")
    assert [(r["commit_id"], r["score"]) for r in rows] == [(101, 42.5)]
    # The commit row lands with the state, from the entry rather than from git.
    (commit_row,) = store.get_commits_with_metadata("v8", ["hash101"])
    assert commit_row["commit_id"] == 101 and commit_row["title"] == "commit 101"
    assert store.get_status("v8", config.platform, 101) == "ok"
    assert store.get_run_env("v8", 101)["source"] == "bus"
    assert consumer.cursor(config.bus.sources[0], "v8") == 101


class TestTransportFailures:
    """A source that cannot answer stops the cycle; it never advances past a
    commit this machine has not benched."""

    class Flaky:
        def __init__(self, real, fail_on):
            self.real = real
            self.fail_on = fail_on

        def __getattr__(self, name):
            if name == self.fail_on:

                def boom(*a, **k):
                    raise subprocess.CalledProcessError(255, "ssh")

                return boom
            return getattr(self.real, name)

    def _drain_with(self, setup, monkeypatch, fail_on):
        from slipstream.consumer import LocalSource

        real = LocalSource(setup.source)
        monkeypatch.setattr(
            "slipstream.consumer.open_source",
            lambda src: self.Flaky(real, fail_on),
        )
        return _drain(setup)

    def test_a_failed_listing_leaves_the_cursor_alone(self, setup, scored, monkeypatch):
        setup.publish(100)
        assert self._drain_with(setup, monkeypatch, "ids_above") == 0
        assert setup.consumer.cursor(setup.source, "v8") is None
        assert "listing" in setup.bus.read_bench_state("v8").last_error

    def test_a_failed_entry_read_leaves_the_cursor_alone(
        self, setup, scored, monkeypatch
    ):
        setup.publish(100)
        setup.publish(101)
        assert self._drain_with(setup, monkeypatch, "read_entry") == 0
        assert setup.consumer.cursor(setup.source, "v8") is None
        assert "reading entry" in setup.bus.read_bench_state("v8").last_error

    def test_a_dropped_entry_still_advances(self, setup, scored):
        """The distinction the two tests above rest on."""
        setup.publish(100)
        setup.publish(101)
        setup.bus.entry_path("v8", 100).unlink()
        _drain(setup)
        assert setup.consumer.cursor(setup.source, "v8") == 101


class TestRunsInTheEnvBlock:
    def test_runs_is_published(self, setup, scored):
        """It is one of the shared inputs that can silently differ per box."""
        setup.publish(100)
        _drain(setup, runs=5)
        assert setup.bus.read_bench_state("v8").env["runs"] == 5

    def test_a_change_moves_since(self, setup, scored):
        setup.publish(100)
        _drain(setup, runs=3)
        first = setup.bus.read_bench_state("v8").env["since"]
        setup.publish(101)
        _drain(setup, runs=5)
        state = setup.bus.read_bench_state("v8")
        assert state.env["runs"] == 5 and state.env["since"] > first


class TestRunRootTrim:
    def test_the_root_just_unpacked_survives(self, setup, scored):
        """A re-bench of an older commit unpacks the lowest-numbered root, so
        the trim would delete the very root about to be measured."""
        setup.cfg.bench.run_roots = 2
        for cid in (500, 501):
            (setup.bus.root / "roots" / "v8" / str(cid)).mkdir(parents=True)
        setup.publish(100)
        _drain(setup)
        assert [cid for cid, _ in scored] == [100]
        root = setup.bus.root / "roots" / "v8" / "100"
        assert root.exists() and (root / "out" / "d8").exists()
        assert setup.collector.store.get_status("v8", setup.cfg.platform, 100) == "ok"


class TestBatchedListing:
    def test_one_listing_covers_a_run_of_done_commits(self, setup, scored):
        """Re-listing per entry is an ssh round trip per skipped commit."""
        from slipstream.consumer import LocalSource

        for cid in (100, 101, 102, 103):
            setup.publish(cid)
        for cid in (100, 101, 102):
            setup.collector.store.mark_done("v8", setup.cfg.platform, cid)

        real = LocalSource(setup.source)
        listings = []

        class Counting:
            def __getattr__(self, name):
                if name == "ids_above":

                    def listed(*a, **k):
                        listings.append(a)
                        return real.ids_above(*a, **k)

                    return listed
                return getattr(real, name)

        import slipstream.consumer as mod

        original = mod.open_source
        mod.open_source = lambda src: Counting()
        try:
            assert _drain(setup) == 1
        finally:
            mod.open_source = original
        assert [cid for cid, _ in scored] == [103]
        # One listing for the batch, one to confirm nothing is left, plus the
        # state file's own count. Not one per skipped commit.
        assert len(listings) <= 4


class TestDryRun:
    """A dry run on a bus engine has to be inert, not merely quiet: the real
    path clears the commit's scores and advances a cursor that cannot go back."""

    def _dry(self, setup):
        from slipstream.consumer import BusConsumer

        c = BusConsumer(setup.cfg, setup.collector, dry_run=True)
        c.logs = []
        c.log = c.logs.append
        return c.drain(
            setup.source,
            "v8",
            1,
            lambda: False,
            lambda: setup.collector.lock.try_acquire(),
            setup.collector.lock.release,
        ).benched, c

    def test_nothing_is_fetched_cleared_or_advanced(self, setup, scored):
        setup.publish(100)
        store = setup.collector.store
        store.insert_scores(
            "v8",
            setup.cfg.platform,
            100,
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
        n, c = self._dry(setup)
        assert n == 0 and scored == []
        assert store.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
        assert c.cursor(setup.source, "v8") is None
        assert not (setup.bus.root / "roots").exists()
        assert any("would bench 1" in m for m in c.logs)

    def test_it_reports_what_is_pending(self, setup, scored):
        for cid in (100, 101):
            setup.publish(cid)
        setup.collector.store.mark_done("v8", setup.cfg.platform, 100)
        _, c = self._dry(setup)
        assert any("would bench 1 of 2" in m for m in c.logs)


class TestUnreadableState:
    def test_a_state_file_from_a_newer_version_does_not_kill_the_daemon(
        self, setup, scored
    ):
        import json

        setup.publish(100)
        path = setup.bus.bench_state_path("v8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 99}))
        assert _drain(setup) == 0
        assert any("version 99" in m for m in setup.consumer.logs)


class TestUnpackEscape:
    def test_a_sibling_sharing_the_prefix_is_refused(self, tmp_path):
        """The check was a string prefix, so /roots/100-evil passed for /roots/100."""
        import subprocess
        import tarfile

        from slipstream.consumer import _unpack

        payload_src = tmp_path / "x.tar"
        with tarfile.open(payload_src, "w") as tf:
            f = tmp_path / "f"
            f.write_text("x")
            tf.add(f, arcname="../100-evil/f")
        payload = tmp_path / "x.tar.zst"
        subprocess.run(["zstd", "-q", "-o", str(payload), str(payload_src)], check=True)
        dest = tmp_path / "100"
        dest.mkdir()
        with pytest.raises(ConsumerError, match="escapes"):
            _unpack(payload, dest)


class TestAbandonedUnpackDirs:
    def test_a_kill_mid_unpack_is_reclaimed(self, setup, scored):
        """No commit id names it, so nothing else would ever remove it."""
        roots = setup.bus.root / "roots" / "v8"
        roots.mkdir(parents=True)
        abandoned = roots / "99.unpacking"
        abandoned.mkdir()
        (abandoned / "big").write_bytes(b"x" * 100)
        setup.publish(100)
        _drain(setup)
        assert not abandoned.exists()
        assert (roots / "100").exists()


class TestStaleFlagsAreCleared:
    def test_a_freed_disk_clears_the_pause_on_a_quiet_cycle(self, setup, scored):
        """Once caught up, the bench path is never taken, so a flag cleared
        only there would be republished forever."""
        setup.publish(100)
        setup.cfg.bench.min_free_gb = 10**9
        _drain(setup)
        assert setup.bus.read_bench_state("v8").benching_paused_by_floor

        setup.cfg.bench.min_free_gb = 0.001
        _drain(setup)  # benches 100
        setup.consumer.log = lambda m: None
        _drain(setup)  # nothing new
        state = setup.bus.read_bench_state("v8")
        assert not state.benching_paused_by_floor and state.last_error is None


class TestDrainReportsWhyItStopped:
    def test_a_transport_failure_is_not_up_to_date(self, setup, scored, monkeypatch):
        from slipstream.consumer import LocalSource

        setup.publish(100)
        real = LocalSource(setup.source)

        class Broken:
            def __getattr__(self, name):
                if name == "ids_above":

                    def boom(*a, **k):
                        raise subprocess.CalledProcessError(255, "ssh")

                    return boom
                return getattr(real, name)

        monkeypatch.setattr("slipstream.consumer.open_source", lambda src: Broken())
        result = _drain_result(setup)
        assert result.benched == 0 and result.error is not None

    def test_nothing_to_do_reports_no_error(self, setup, scored):
        setup.publish(100)
        _drain(setup)
        result = _drain_result(setup)
        assert result.benched == 0 and result.error is None


class TestStaleInFlight:
    def test_a_killed_bencher_does_not_report_work_forever(self, setup, scored):
        """Every later cycle would republish it with a fresh updated_at, so the
        state file's age -- the crashed-versus-busy signal -- is never shown."""
        from slipstream.bus import BenchState

        setup.bus.write_bench_state(
            "v8",
            BenchState(
                cursor=99,
                in_flight={"commit_id": 999, "phase": "bench", "started_at": 1},
            ),
        )
        _drain(setup)  # nothing published, so nothing to do
        assert setup.bus.read_bench_state("v8").in_flight is None


class TestUnpackErrorIsNotMasked:
    def test_the_rejection_is_reported_not_a_decompression_failure(self, tmp_path):
        """Closing the pipe kills zstd, so its exit code would win."""
        import subprocess as sp
        import tarfile

        from slipstream.consumer import _unpack

        src = tmp_path / "x.tar"
        with tarfile.open(src, "w") as tf:
            f = tmp_path / "f"
            # The escaping member first, then enough incompressible padding
            # that zstd is still streaming when the rejection closes the pipe.
            f.write_bytes(b"x")
            tf.add(f, arcname="../escaped")
            import os as _os

            f.write_bytes(_os.urandom(4 << 20))
            for i in range(8):
                tf.add(f, arcname=f"pad{i}")
        payload = tmp_path / "x.tar.zst"
        sp.run(["zstd", "-q", "-o", str(payload), str(src)], check=True)
        dest = tmp_path / "root"
        dest.mkdir()
        with pytest.raises(ConsumerError) as exc:
            _unpack(payload, dest)
        assert "escapes the run root" in str(exc.value)
        assert "failed to decompress" not in str(exc.value)


class TestConsumerCircuitBreaker:
    """A cycle drains, so a machine-side failure would otherwise mark every
    entry in the topic failed in one pass -- and is_done is status-blind, so
    none of them would ever be revisited."""

    def _all_fail(self, setup, monkeypatch):
        from slipstream.collector import BenchOutcome

        seen = []

        def no_scores(engine, commit_id, runs, run_root):
            seen.append(int(commit_id))
            return BenchOutcome(0, 3, 0)  # the binary never ran

        monkeypatch.setattr(setup.collector, "_run_benchmarks", no_scores)
        return seen

    def test_it_stops_after_consecutive_failures(self, setup, monkeypatch):
        setup.cfg.bench.max_consecutive_failures = 3
        for cid in range(100, 120):
            setup.publish(cid)
        seen = self._all_fail(setup, monkeypatch)
        _drain(setup)
        assert len(seen) == 3, f"marched through {len(seen)} of 20 commits"
        state = setup.bus.read_bench_state("v8")
        assert state.stalled_since and "not the tree" in state.last_error
        assert state.stall_retry_after > time.time()

    def test_the_rest_of_the_backlog_is_untouched(self, setup, monkeypatch):
        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 110):
            setup.publish(cid)
        self._all_fail(setup, monkeypatch)
        _drain(setup)
        counts = setup.collector.store.status_counts("v8", setup.cfg.platform)
        assert counts == {"failed": 2}

    def test_a_stalled_consumer_waits_out_its_backoff(self, setup, monkeypatch):
        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 110):
            setup.publish(cid)
        seen = self._all_fail(setup, monkeypatch)
        _drain(setup)
        assert len(seen) == 2
        _drain(setup)  # inside the window
        assert len(seen) == 2, "the backoff was not honoured"

    def test_it_tries_one_commit_per_window(self, setup, monkeypatch):
        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 110):
            setup.publish(cid)
        seen = self._all_fail(setup, monkeypatch)
        _drain(setup)
        state = setup.bus.read_bench_state("v8")
        state.stall_retry_after = 0.0
        setup.bus.write_bench_state("v8", state)
        _drain(setup)
        assert len(seen) == 3, "a stalled retry should cost one commit, not a run"

    def test_a_success_clears_the_stall(self, setup, monkeypatch, scored):
        from slipstream.collector import BenchOutcome

        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 110):
            setup.publish(cid)
        self._all_fail(setup, monkeypatch)
        _drain(setup)
        assert setup.bus.read_bench_state("v8").stalled_since

        state = setup.bus.read_bench_state("v8")
        state.stall_retry_after = 0.0
        setup.bus.write_bench_state("v8", state)
        monkeypatch.setattr(
            setup.collector, "_run_benchmarks", lambda *a: BenchOutcome(3, 3, 40)
        )
        _drain(setup)
        after = setup.bus.read_bench_state("v8")
        assert after.stalled_since is None and after.stall_retry_after is None

    def test_a_partial_run_is_not_a_failure(self, setup, monkeypatch):
        """Only zero scores means the binary did not run; a suite that dropped
        one benchmark is partial and must not trip the breaker."""
        from slipstream.collector import BenchOutcome

        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 105):
            setup.publish(cid)
        monkeypatch.setattr(
            setup.collector, "_run_benchmarks", lambda *a: BenchOutcome(2, 3, 30)
        )
        _drain(setup)
        assert setup.bus.read_bench_state("v8").stalled_since is None
        counts = setup.collector.store.status_counts("v8", setup.cfg.platform)
        assert counts == {"partial": 5}

    def test_the_reason_survives_the_whole_backoff(self, setup, monkeypatch):
        """The next cycle used to wipe last_error while leaving stalled_since,
        so watch printed "up to date" and bus status showed nothing."""
        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 110):
            setup.publish(cid)
        self._all_fail(setup, monkeypatch)
        _drain(setup)
        reason = setup.bus.read_bench_state("v8").last_error
        assert reason and "not the tree" in reason

        result = _drain_result(setup)  # inside the backoff window
        assert result.error == reason, "the stall became invisible"
        assert setup.bus.read_bench_state("v8").last_error == reason
        assert setup.bus.read_bench_state("v8").stalled_since

    def test_a_systematic_collision_trips_the_breaker(self, setup, monkeypatch):
        """A restored db or a changed id_regex collides on every commit; the
        collision path used to report the run as clean."""
        from slipstream.collector import BenchOutcome

        setup.cfg.bench.max_consecutive_failures = 2
        store = setup.collector.store
        for cid in range(100, 110):
            setup.publish(cid)
            store.upsert_commit("v8", f"other{cid}", cid, "d", 0, "incumbent")

        def measured(engine, commit_id, runs, run_root):
            store.insert_scores(
                "v8",
                setup.cfg.platform,
                int(commit_id),
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
            return BenchOutcome(3, 3, 10)

        monkeypatch.setattr(setup.collector, "_run_benchmarks", measured)
        _drain(setup)
        assert setup.bus.read_bench_state("v8").stalled_since
        assert store.status_counts("v8", setup.cfg.platform) == {"failed": 2}


class TestCorruptArchive:
    def test_tar_errors_are_transport_errors(self):
        """TarError is not an OSError, so it used to escape drain entirely,
        killing watch and taking the other engines with it."""
        import tarfile

        from slipstream.consumer import TRANSPORT_ERRORS

        assert issubclass(tarfile.ReadError, TRANSPORT_ERRORS)

    def test_an_unreadable_archive_is_reported_not_fatal(
        self, setup, scored, monkeypatch
    ):
        import tarfile

        setup.publish(100)

        def unreadable(payload, dest):
            raise tarfile.ReadError("unexpected end of data")

        monkeypatch.setattr("slipstream.consumer._unpack", unreadable)
        result = _drain_result(setup)
        assert result.benched == 0 and "unexpected end of data" in result.error
        # The cursor stays put, so the commit is not skipped.
        assert setup.consumer.cursor(setup.source, "v8") is None


class TestDroppedEntriesAreRecorded:
    def test_a_gap_survives_the_cursor_moving_past_it(self, setup, scored):
        """box1 offline a week, box2's retention drops what it never listed.
        The builder's highest_dropped stops being evidence as soon as the
        cursor passes it, which the next bench does."""
        from slipstream.bus import BuilderState

        setup.bus.write_builder_state(
            "v8", BuilderState(frontier=151, lowest_retained=151, highest_dropped=150)
        )
        setup.publish(151)
        _drain(setup)
        state = setup.bus.read_bench_state("v8")
        assert state.skipped_dropped == 150
        assert any("dropped by retention" in m for m in setup.consumer.logs)

        # The cursor is now past it, so highest_dropped alone proves nothing.
        setup.publish(152)
        _drain(setup)
        after = setup.bus.read_bench_state("v8")
        assert after.cursor == 152 and after.skipped_dropped == 150

    def test_nothing_is_claimed_when_retention_stayed_behind(self, setup, scored):
        from slipstream.bus import BuilderState

        setup.publish(100)
        _drain(setup)
        setup.bus.write_builder_state(
            "v8", BuilderState(frontier=101, highest_dropped=100)
        )
        setup.publish(101)
        _drain(setup)
        assert setup.bus.read_bench_state("v8").skipped_dropped is None


class TestReclaimIsReachableUnderTheFloor:
    def test_an_abandoned_unpack_dir_is_swept_before_the_floor_check(
        self, setup, scored, monkeypatch
    ):
        """If that directory is what filled the disk, a sweep that only runs
        inside provision can never run."""
        roots = setup.bus.root / "roots" / "v8"
        roots.mkdir(parents=True)
        abandoned = roots / "99.unpacking"
        abandoned.mkdir()
        (abandoned / "big").write_bytes(b"x" * 1000)

        setup.publish(100)
        setup.cfg.bench.min_free_gb = 10**9  # the floor stops the cycle
        monkeypatch.setattr(setup.consumer, "free_gb", lambda: 1.0)
        _drain(setup)

        assert not abandoned.exists()
        assert setup.bus.read_bench_state("v8").benching_paused_by_floor

    def test_bus_gc_reclaims_them_too(self, setup):
        roots = setup.bus.root / "roots" / "v8"
        roots.mkdir(parents=True)
        abandoned = roots / "99.unpacking"
        abandoned.mkdir()
        (abandoned / "big").write_bytes(b"x")
        keep = roots / "100"
        keep.mkdir()

        removed = setup.bus.gc(["v8"])
        assert abandoned in removed and not abandoned.exists()
        assert keep.exists(), "a retained run root was reclaimed"


class TestNoWriteCanKillTheDaemon:
    """drain writes the cursor and the state file on several paths, all under
    a filesystem a stalled or paused cycle is probably reacting to being full."""

    def _enospc_on(self, monkeypatch, target):
        def boom(*a, **k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(target[0], target[1], boom)

    def test_a_failed_state_write_in_the_stall_branch(self, setup, monkeypatch):
        from slipstream.bus import BenchState

        setup.publish(100)
        setup.bus.write_bench_state(
            "v8",
            BenchState(
                stalled_since=1.0,
                stall_retry_after=time.time() + 3600,
                last_error="stalled",
            ),
        )
        self._enospc_on(monkeypatch, (type(setup.bus), "write_bench_state"))
        result = _drain_result(setup)
        assert result.error and "No space" in result.error

    def test_a_failed_cursor_write(self, setup, monkeypatch):
        setup.publish(100)
        setup.collector.store.mark_done("v8", setup.cfg.platform, 100)
        self._enospc_on(monkeypatch, (type(setup.consumer), "set_cursor"))
        result = _drain_result(setup)
        assert result.error and "No space" in result.error

    def test_a_failed_end_of_cycle_write(self, setup, monkeypatch):
        self._enospc_on(monkeypatch, (type(setup.bus), "write_bench_state"))
        result = _drain_result(setup)
        assert result.error and "No space" in result.error

    def test_work_already_done_is_still_reported(self, setup, scored, monkeypatch):
        """The count is what was measured and recorded, not what was written
        afterwards: losing it would hide a cycle's whole output."""
        for cid in (100, 101):
            setup.publish(cid)
        real = type(setup.consumer).set_cursor
        calls = [0]

        def fail_on_the_second(self, source, engine, commit_id):
            calls[0] += 1
            if calls[0] > 1:
                raise OSError(28, "No space left on device")
            return real(self, source, engine, commit_id)

        monkeypatch.setattr(type(setup.consumer), "set_cursor", fail_on_the_second)
        result = _drain_result(setup)
        assert result.benched == 2 and result.error
        assert [cid for cid, _ in scored] == [100, 101]


class TestStallIsAudible:
    def test_the_backoff_says_so_every_cycle(self, setup, monkeypatch):
        setup.cfg.bench.max_consecutive_failures = 2
        for cid in range(100, 110):
            setup.publish(cid)
        from slipstream.collector import BenchOutcome

        monkeypatch.setattr(
            setup.collector, "_run_benchmarks", lambda *a: BenchOutcome(0, 3, 0)
        )
        _drain(setup)
        setup.consumer.logs.clear()
        _drain(setup)  # inside the window
        assert any("stalled until" in m for m in setup.consumer.logs)


class TestStallClearsWhenThereIsNothingLeft:
    def test_an_idle_topic_ends_a_stall(self, setup, monkeypatch):
        """The per-entry success path is the only other place these clear, and
        it is unreachable with an empty batch: a stall fixed while the topic
        was idle would be reported forever, with no error line beside it."""
        from slipstream.bus import BenchState

        setup.bus.write_bench_state(
            "v8",
            BenchState(
                cursor=100,
                stalled_since=1.0,
                stall_retry_after=0.0,  # the window is open
                last_error="stalled",
            ),
        )
        _drain(setup)  # nothing published above the cursor
        state = setup.bus.read_bench_state("v8")
        assert state.stalled_since is None and state.stall_retry_after is None

    def test_a_cycle_that_failed_keeps_the_stall(self, setup, monkeypatch):
        from slipstream.bus import BenchState
        from slipstream.consumer import LocalSource

        setup.bus.write_bench_state(
            "v8",
            BenchState(cursor=100, stalled_since=1.0, stall_retry_after=0.0),
        )
        real = LocalSource(setup.source)

        class Broken:
            def __getattr__(self, name):
                if name == "ids_above":

                    def boom(*a, **k):
                        raise subprocess.CalledProcessError(255, "ssh")

                    return boom
                return getattr(real, name)

        monkeypatch.setattr("slipstream.consumer.open_source", lambda src: Broken())
        _drain(setup)
        assert setup.bus.read_bench_state("v8").stalled_since == 1.0


class TestStoreErrorsAreSurvivable:
    """Every store write in a cycle goes through sqlite3, and sqlite3.Error is
    not an OSError -- so the "no write may kill the daemon" net had a hole for
    exactly the full-disk case it was written for."""

    def test_sqlite_errors_are_transport_errors(self):
        import sqlite3

        from slipstream.consumer import TRANSPORT_ERRORS

        assert not issubclass(sqlite3.OperationalError, OSError)
        assert issubclass(sqlite3.Error, TRANSPORT_ERRORS)

    def test_a_full_db_is_reported_not_fatal(self, setup, monkeypatch):
        import sqlite3

        setup.publish(100)

        def full(*a, **k):
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(type(setup.collector.store), "clear_scores", full)
        result = _drain_result(setup)
        assert result.benched == 0 and "disk is full" in result.error

    def test_a_locked_db_in_skip_done_is_reported(self, setup, monkeypatch):
        import sqlite3

        setup.publish(100)

        def locked(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(type(setup.collector.store), "is_done", locked)
        result = _drain_result(setup)
        assert result.error and "locked" in result.error


class TestEntryIsRevalidatedUnderTheLock:
    def test_retention_during_the_wait_is_not_a_missing_payload(
        self, setup, scored, monkeypatch
    ):
        """take_lock can block for hours behind a builder on a box that does
        both; a drop in that window is ordinary retention, not an error."""
        setup.publish(100)
        setup.publish(101)

        def drop_100_while_waiting():
            setup.bus.entry_path("v8", 100).unlink(missing_ok=True)
            setup.bus.blob_path("v8", 100).unlink(missing_ok=True)
            return setup.collector.lock.try_acquire()

        result = setup.consumer.drain(
            setup.source,
            "v8",
            1,
            lambda: False,
            drop_100_while_waiting,
            setup.collector.lock.release,
        )
        assert result.error is None, result.error
        assert [cid for cid, _ in scored] == [101]
        assert setup.bus.read_bench_state("v8").skipped_dropped == 100


class TestTheLockIsNeverLeaked:
    """The lock is process-wide: leaking it blocks the builder and every
    ad-hoc command for the life of the daemon, and an idle topic never
    reaches another take_lock to release it."""

    def _drain_with_broken_reread(self, setup, boom):
        from slipstream.consumer import LocalSource

        real = LocalSource(setup.source)
        calls = [0]

        class Flaky:
            def __getattr__(self, name):
                if name == "read_entry":

                    def read(*a, **k):
                        calls[0] += 1
                        if calls[0] > 1:  # the re-read under the lock
                            raise boom
                        return real.read_entry(*a, **k)

                    return read
                return getattr(real, name)

        import slipstream.consumer as mod

        original = mod.open_source
        mod.open_source = lambda src: Flaky()
        try:
            return _drain_result(setup)
        finally:
            mod.open_source = original

    def test_an_ssh_drop_during_the_re_read_releases_it(self, setup, scored):
        setup.publish(100)
        result = self._drain_with_broken_reread(
            setup, subprocess.CalledProcessError(255, "ssh")
        )
        assert result.error
        assert not setup.collector.lock.held, "the machine lock was leaked"

    def test_a_malformed_entry_during_the_re_read_releases_it(self, setup, scored):
        from slipstream.bus import BusError

        setup.publish(100)
        result = self._drain_with_broken_reread(setup, BusError("not an object"))
        assert result.error
        assert not setup.collector.lock.held

    def test_a_failed_cursor_write_on_the_dropped_path_releases_it(
        self, setup, scored, monkeypatch
    ):
        setup.publish(100)

        def gone(*a, **k):
            return None

        monkeypatch.setattr(
            type(setup.consumer),
            "set_cursor",
            lambda *a: (_ for _ in ()).throw(OSError(28, "No space left on device")),
        )
        from slipstream.consumer import LocalSource

        real = LocalSource(setup.source)
        calls = [0]

        class Vanishing:
            def __getattr__(self, name):
                if name == "read_entry":

                    def read(*a, **k):
                        calls[0] += 1
                        return real.read_entry(*a, **k) if calls[0] == 1 else None

                    return read
                return getattr(real, name)

        import slipstream.consumer as mod

        original = mod.open_source
        mod.open_source = lambda src: Vanishing()
        try:
            result = _drain_result(setup)
        finally:
            mod.open_source = original
        assert result.error
        assert not setup.collector.lock.held

    def test_a_peer_can_take_it_afterwards(self, setup, scored):
        from slipstream.lock import MachineLock

        setup.publish(100)
        self._drain_with_broken_reread(setup, subprocess.CalledProcessError(255, "ssh"))
        peer = MachineLock("build", setup.collector.lock.path)
        assert peer.try_acquire(), "the builder would block forever"
        peer.release()
