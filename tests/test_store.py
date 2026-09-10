# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations


class TestCommitOperations:
    def test_insert_and_query_commits(self, store):
        commits = [
            {"hash": "abc123", "title": "fix bug"},
            {"hash": "def456", "title": "add feature"},
        ]
        store.insert_commits("v8", commits)

        # Duplicate insert is no-op
        store.insert_commits("v8", commits)
        row = store.conn.execute(
            "SELECT COUNT(*) FROM commits WHERE engine='v8'"
        ).fetchone()
        assert row[0] == 2

    def test_update_and_query_metadata(self, store):
        store.insert_commits("v8", [{"hash": "abc123", "title": "fix"}])
        store.update_commit_metadata(
            "v8", "abc123", 99707, "2026-01-01", 1700000000, "fix bug"
        )

        rows = store.get_commits_with_metadata("v8", ["abc123"])
        assert len(rows) == 1
        assert rows[0]["commit_id"] == 99707
        assert rows[0]["date"] == "2026-01-01"

    def test_update_metadata_idempotent(self, store):
        """Second update with different data doesn't overwrite (commit_id IS NULL guard)."""
        store.insert_commits("v8", [{"hash": "abc123", "title": "fix"}])
        store.update_commit_metadata(
            "v8", "abc123", 99707, "2026-01-01", 1700000000, "fix bug"
        )
        store.update_commit_metadata(
            "v8", "abc123", 99999, "2026-02-01", 1700099999, "wrong"
        )

        rows = store.get_commits_with_metadata("v8", ["abc123"])
        assert rows[0]["commit_id"] == 99707

    def test_count_missing_metadata(self, store):
        store.insert_commits("v8", [{"hash": "a"}, {"hash": "b"}])
        assert store.count_missing_metadata("v8", ["a", "b"]) == 2

        store.update_commit_metadata("v8", "a", 1, "d", 0, "t")
        assert store.count_missing_metadata("v8", ["a", "b"]) == 1


class TestProcessingState:
    def test_done_lifecycle(self, store):
        store.insert_commits("v8", [{"hash": "abc"}])
        store.update_commit_metadata("v8", "abc", 100, "d", 0, "t")

        assert not store.is_done("v8", "arm64", 100)
        store.mark_done("v8", "arm64", 100)
        assert store.is_done("v8", "arm64", 100)
        # Different platform is not done
        assert not store.is_done("v8", "x86_64", 100)

    def test_clear_range(self, store):
        store.insert_commits("v8", [{"hash": "a"}, {"hash": "b"}])
        store.update_commit_metadata("v8", "a", 100, "d", 0, "t")
        store.update_commit_metadata("v8", "b", 200, "d", 0, "t")
        store.mark_done("v8", "arm64", 100)
        store.mark_done("v8", "arm64", 200)
        store.insert_scores(
            "v8",
            "arm64",
            100,
            0,
            [
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "b",
                    "metric": "m",
                    "run": 1,
                    "score": 1.0,
                },
            ],
        )

        store.clear_range("v8", "arm64", [100])
        assert not store.is_done("v8", "arm64", 100)
        assert store.is_done("v8", "arm64", 200)
        assert store.get_series("v8", "js3", "default", "b", "m") == []

    def test_max_done_commit_id(self, store):
        assert store.max_done_commit_id("v8", "arm64") is None

        store.insert_commits("v8", [{"hash": "a"}, {"hash": "b"}])
        store.update_commit_metadata("v8", "a", 100, "d", 0, "t")
        store.update_commit_metadata("v8", "b", 200, "d", 0, "t")
        store.mark_done("v8", "arm64", 100)
        store.mark_done("v8", "arm64", 200)
        assert store.max_done_commit_id("v8", "arm64") == 200
        assert store.max_done_commit_id("v8", "x86_64") is None


class TestScores:
    def test_insert_and_query_scores(self, store):
        scores = [
            {
                "suite": "js3",
                "flags": "default",
                "benchmark": "regex",
                "metric": "Total-Score",
                "run": 1,
                "score": 100.0,
            },
            {
                "suite": "js3",
                "flags": "default",
                "benchmark": "regex",
                "metric": "Total-Score",
                "run": 2,
                "score": 101.0,
            },
        ]
        store.insert_scores("v8", "arm64", 1000, 1700000000, scores)

        rows = store.get_series("v8", "js3", "default", "regex", "Total-Score")
        assert len(rows) == 2
        assert rows[0]["score"] == 100.0
        assert rows[1]["score"] == 101.0

    def test_duplicate_scores_ignored(self, store):
        scores = [
            {
                "suite": "js3",
                "flags": "default",
                "benchmark": "regex",
                "metric": "Total-Score",
                "run": 1,
                "score": 100.0,
            }
        ]
        store.insert_scores("v8", "arm64", 1000, 1700000000, scores)
        store.insert_scores("v8", "arm64", 1000, 1700000000, scores)

        rows = store.get_series("v8", "js3", "default", "regex", "Total-Score")
        assert len(rows) == 1

    def test_bulk_insert_and_dedup(self, store):
        rows = [
            (1000, "js3", "default", "regex", "Total-Score", 1, 100.0, 1700000000),
            (1000, "js3", "default", "regex", "Total-Score", 2, 101.0, 1700000000),
        ]
        store.bulk_insert_scores("v8", "arm64", rows)
        store.bulk_insert_scores("v8", "arm64", rows)  # duplicate

        result = store.get_series("v8", "js3", "default", "regex", "Total-Score")
        assert len(result) == 2

    def test_distinct_series_keys(self, store):
        store.insert_scores(
            "v8",
            "arm64",
            1000,
            0,
            [
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "a",
                    "metric": "Total-Score",
                    "run": 1,
                    "score": 1.0,
                },
                {
                    "suite": "js3",
                    "flags": "default",
                    "benchmark": "b",
                    "metric": "First",
                    "run": 1,
                    "score": 2.0,
                },
                {
                    "suite": "js2",
                    "flags": "default",
                    "benchmark": "c",
                    "metric": "Total-Score",
                    "run": 1,
                    "score": 3.0,
                },
            ],
        )
        keys = store.get_distinct_series_keys("v8")
        assert len(keys) == 3

    def test_engine_isolation(self, store):
        scores = [
            {
                "suite": "js3",
                "flags": "default",
                "benchmark": "a",
                "metric": "Total-Score",
                "run": 1,
                "score": 1.0,
            }
        ]
        store.insert_scores("v8", "arm64", 1000, 0, scores)
        store.insert_scores("jsc", "arm64", 1000, 0, scores)

        assert len(store.get_series("v8", "js3", "default", "a", "Total-Score")) == 1
        assert len(store.get_series("jsc", "js3", "default", "a", "Total-Score")) == 1


class TestPushState:
    def _seed_done(self, store, commit_ids):
        store.insert_commits("v8", [{"hash": f"h{c}"} for c in commit_ids])
        for c in commit_ids:
            store.update_commit_metadata("v8", f"h{c}", c, "d", 0, "t")
            store.mark_done("v8", "arm64", c)

    def test_unpushed_initially_all_done(self, store):
        self._seed_done(store, [100, 200, 300])
        assert store.unpushed_commit_ids("v8", "arm64") == [100, 200, 300]

    def test_mark_pushed_filters_next_call(self, store):
        self._seed_done(store, [100, 200, 300])
        store.mark_pushed("v8", "arm64", [100, 200])
        assert store.unpushed_commit_ids("v8", "arm64") == [300]

    def test_mark_pushed_idempotent(self, store):
        self._seed_done(store, [100])
        store.mark_pushed("v8", "arm64", [100])
        store.mark_pushed("v8", "arm64", [100])  # no error
        assert store.unpushed_commit_ids("v8", "arm64") == []

    def test_clear_range_reinvalidates_push(self, store):
        self._seed_done(store, [100, 200])
        store.mark_pushed("v8", "arm64", [100, 200])
        assert store.unpushed_commit_ids("v8", "arm64") == []

        # Re-bench 100: its scores, processing_state, AND push_state all clear.
        store.clear_range("v8", "arm64", [100])
        # 100 is no longer "done" → not in unpushed. Re-run mark_done:
        store.mark_done("v8", "arm64", 100)
        assert store.unpushed_commit_ids("v8", "arm64") == [100]

    def test_platform_isolation(self, store):
        self._seed_done(store, [100])
        store.mark_pushed("v8", "arm64", [100])
        # x86_64 has no "done" commit here, so empty is expected.
        assert store.unpushed_commit_ids("v8", "x86_64") == []

    def test_engine_isolation(self, store):
        self._seed_done(store, [100])
        store.insert_commits("jsc", [{"hash": "jh"}])
        store.update_commit_metadata("jsc", "jh", 100, "d", 0, "t")
        store.mark_done("jsc", "arm64", 100)

        store.mark_pushed("v8", "arm64", [100])
        assert store.unpushed_commit_ids("v8", "arm64") == []
        assert store.unpushed_commit_ids("jsc", "arm64") == [100]


class TestStatus:
    def test_default_status_is_ok(self, store):
        store.mark_done("v8", "arm64", 100)
        assert store.get_status("v8", "arm64", 100) == "ok"

    def test_status_upserts_both_ways(self, store):
        store.mark_done(
            "v8", "arm64", 100, status="failed", configs_ok=0, configs_total=3
        )
        assert store.get_status("v8", "arm64", 100) == "failed"
        store.mark_done("v8", "arm64", 100, status="ok", configs_ok=3, configs_total=3)
        assert store.get_status("v8", "arm64", 100) == "ok"
        row = store.conn.execute(
            "SELECT configs_ok, configs_total FROM processing_state"
            " WHERE engine='v8' AND commit_id=100"
        ).fetchone()
        assert (row[0], row[1]) == (3, 3)

    def test_mark_done_without_status_preserves_it(self, store):
        """import calls mark_done with no status; it must not overwrite a run's verdict."""
        store.mark_done("v8", "arm64", 100, status="failed")
        store.mark_done("v8", "arm64", 100)
        assert store.get_status("v8", "arm64", 100) == "failed"

    def test_is_done_is_status_blind(self, store):
        store.mark_done("v8", "arm64", 100, status="failed")
        assert store.is_done("v8", "arm64", 100)

    def test_status_counts(self, store):
        store.mark_done("v8", "arm64", 100, status="ok")
        store.mark_done("v8", "arm64", 101, status="ok")
        store.mark_done("v8", "arm64", 102, status="partial")
        assert store.status_counts("v8", "arm64") == {"ok": 2, "partial": 1}


class TestUpsertCommit:
    def test_upsert_writes_and_updates(self, store):
        store.upsert_commit("v8", "abc", 100, "2026-01-01", 17, "title")
        store.upsert_commit("v8", "abc", 100, "2026-01-02", 18, "retitled")
        rows = store.get_commits_with_metadata("v8", ["abc"])
        assert len(rows) == 1
        assert rows[0]["title"] == "retitled"
        assert rows[0]["date"] == "2026-01-02"

    def test_upsert_fills_in_a_hash_only_row(self, store):
        store.insert_commits("v8", [{"hash": "abc", "title": "t"}])
        store.upsert_commit("v8", "abc", 100, "2026-01-01", 17, "t")
        assert store.get_commits_with_metadata("v8", ["abc"])[0]["commit_id"] == 100

    def test_colliding_commit_id_raises_and_keeps_incumbent(self, store):
        from slipstream.store import CommitIdCollision

        store.upsert_commit("v8", "abc", 100, "d", 0, "first")
        try:
            store.upsert_commit("v8", "def", 100, "d", 0, "second")
            raise AssertionError("expected CommitIdCollision")
        except CommitIdCollision as e:
            assert "abc" in str(e)
        # The incumbent survives: a replace would have deleted it and orphaned
        # its scores from the export join.
        rows = store.get_commits_with_metadata("v8", ["abc"])
        assert len(rows) == 1 and rows[0]["title"] == "first"


class TestScoresWithoutCommitRow:
    def test_reports_only_orphans(self, store):
        store.upsert_commit("v8", "abc", 100, "d", 0, "t")
        for cid in (100, 101):
            store.insert_scores(
                "v8",
                "arm64",
                cid,
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
        assert store.commit_ids_missing_commit_row("v8", "arm64") == [101]


class TestClearScores:
    def test_clears_scores_only(self, store):
        store.upsert_commit("v8", "abc", 100, "d", 0, "t")
        store.insert_scores(
            "v8",
            "arm64",
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
        store.mark_done("v8", "arm64", 100)
        store.mark_pushed("v8", "arm64", [100])

        store.clear_scores("v8", "arm64", [100])
        n = store.conn.execute(
            "SELECT COUNT(*) FROM scores WHERE commit_id=100"
        ).fetchone()[0]
        assert n == 0
        assert store.is_done("v8", "arm64", 100)
        assert store.unpushed_commit_ids("v8", "arm64") == []


class TestBotIdentity:
    def test_records_and_stamps_rows(self, tmp_path):
        from slipstream.store import CommitStore

        s = CommitStore(tmp_path / "t.db", bot="box2-m4")
        s.upsert_commit("v8", "abc", 100, "d", 0, "t")
        s.insert_scores(
            "v8",
            "arm64",
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
        s.mark_done("v8", "arm64", 100)
        s.mark_pushed("v8", "arm64", [100])
        assert s.get_meta("local_bot") == "box2-m4"
        assert s.bot == "box2-m4"
        for table in ("scores", "processing_state", "push_state"):
            got = s.conn.execute(f"SELECT DISTINCT bot FROM {table}").fetchall()
            assert [r[0] for r in got] == ["box2-m4"]
        s.close()

    def test_backfills_rows_written_before_a_name_was_set(self, tmp_path):
        from slipstream.store import CommitStore

        s = CommitStore(tmp_path / "t.db")
        s.mark_done("v8", "arm64", 100)
        assert s.conn.execute("SELECT bot FROM processing_state").fetchone()[0] is None
        s.close()

        s = CommitStore(tmp_path / "t.db", bot="box1-m1")
        assert (
            s.conn.execute("SELECT bot FROM processing_state").fetchone()[0]
            == "box1-m1"
        )
        assert s.get_meta("local_bot") == "box1-m1"
        s.close()

    def test_a_foreign_db_is_refused(self, tmp_path):
        from slipstream.store import BotMismatch, CommitStore

        CommitStore(tmp_path / "t.db", bot="box1-m1").close()
        try:
            CommitStore(tmp_path / "t.db", bot="box2-m4")
            raise AssertionError("expected BotMismatch")
        except BotMismatch as e:
            assert "box1-m1" in str(e) and "box2-m4" in str(e)

    def test_readonly_open_neither_migrates_nor_writes(self, tmp_path):
        from slipstream.store import CommitStore

        s = CommitStore(tmp_path / "t.db", bot="box1-m1")
        s.close()
        r = CommitStore(tmp_path / "t.db", readonly=True, bot="box1-m1")
        assert r.get_meta("schema_version") is not None
        r.close()

    def test_readonly_refuses_a_foreign_db(self, tmp_path):
        from slipstream.store import BotMismatch, CommitStore

        CommitStore(tmp_path / "t.db", bot="box1-m1").close()
        try:
            CommitStore(tmp_path / "t.db", readonly=True, bot="box2-m4")
            raise AssertionError("expected BotMismatch")
        except BotMismatch:
            pass


class TestMigration:
    """The pre-bot schema must gain its columns on the next open."""

    def _legacy_db(self, path):
        import sqlite3

        conn = sqlite3.connect(str(path))
        conn.executescript("""
            CREATE TABLE commits (
                engine TEXT NOT NULL, hash TEXT NOT NULL, commit_id INTEGER,
                date TEXT NOT NULL DEFAULT '', timestamp INTEGER NOT NULL DEFAULT 0,
                title TEXT NOT NULL DEFAULT '', PRIMARY KEY (engine, hash));
            CREATE UNIQUE INDEX idx_commit_id ON commits (engine, commit_id)
                WHERE commit_id IS NOT NULL;
            CREATE TABLE scores (
                engine TEXT NOT NULL, platform TEXT NOT NULL, commit_id INTEGER NOT NULL,
                suite TEXT NOT NULL, flags TEXT NOT NULL DEFAULT 'default',
                benchmark TEXT NOT NULL, metric TEXT NOT NULL, run INTEGER NOT NULL,
                score REAL NOT NULL, timestamp INTEGER NOT NULL,
                PRIMARY KEY (engine, platform, commit_id, suite, flags, benchmark, metric, run));
            CREATE TABLE processing_state (
                engine TEXT NOT NULL, platform TEXT NOT NULL, commit_id INTEGER NOT NULL,
                PRIMARY KEY (engine, platform, commit_id));
            CREATE TABLE push_state (
                engine TEXT NOT NULL, platform TEXT NOT NULL, commit_id INTEGER NOT NULL,
                pushed_at INTEGER NOT NULL, PRIMARY KEY (engine, platform, commit_id));
            INSERT INTO commits VALUES ('v8','abc',100,'2026-01-01',17,'t');
            INSERT INTO scores VALUES ('v8','arm64',100,'js3','default','b','Total-Score',1,1.0,0);
            INSERT INTO processing_state VALUES ('v8','arm64',100);
            INSERT INTO push_state VALUES ('v8','arm64',100,17);
        """)
        conn.commit()
        conn.close()

    def test_columns_added_and_backfilled(self, tmp_path):
        from slipstream.store import CommitStore

        db = tmp_path / "legacy.db"
        self._legacy_db(db)
        s = CommitStore(db, bot="box1-m1")
        assert s.get_status("v8", "arm64", 100) == "ok"
        for table in ("scores", "processing_state", "push_state"):
            assert s.conn.execute(f"SELECT bot FROM {table}").fetchone()[0] == "box1-m1"
        # Existing rows survive.
        assert s.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
        s.close()

    def test_migration_is_idempotent(self, tmp_path):
        from slipstream.store import CommitStore

        db = tmp_path / "legacy.db"
        self._legacy_db(db)
        CommitStore(db, bot="box1-m1").close()
        s = CommitStore(db, bot="box1-m1")
        cols = [r[1] for r in s.conn.execute("PRAGMA table_info(processing_state)")]
        assert cols.count("status") == 1 and cols.count("bot") == 1
        s.close()

    def test_a_concurrent_opener_does_not_break_the_other(self, tmp_path):
        """Two daemons open the same file; one ALTER wins, the other no-ops."""
        from slipstream.store import CommitStore

        db = tmp_path / "legacy.db"
        self._legacy_db(db)
        a = CommitStore(db, bot="box1-m1", backup=False)
        b = CommitStore(db, bot="box1-m1", backup=False)
        for s in (a, b):
            assert s.get_status("v8", "arm64", 100) == "ok"
            s.close()


class TestBotGuardSurvivesALockedDb:
    def test_a_foreign_db_is_refused_even_when_the_migration_defers(
        self, tmp_path, monkeypatch
    ):
        """The migration returns quietly on a locked db; the identity guard
        must not be skipped with it."""
        import sqlite3

        from slipstream.store import BotMismatch, CommitStore

        db = tmp_path / "t.db"
        CommitStore(db, bot="box1-m1").close()

        real = CommitStore._add_column

        def locked(self, table, column, decl):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(CommitStore, "_add_column", locked)
        try:
            CommitStore(db, bot="box2-m4")
            raise AssertionError("expected BotMismatch")
        except BotMismatch:
            pass
        monkeypatch.setattr(CommitStore, "_add_column", real)


class TestMigrationIsNotSilentlyDeferred:
    def test_a_permanently_locked_db_is_an_error_not_a_shrug(
        self, tmp_path, monkeypatch
    ):
        """A daemon holds one connection for days, so there is no next open:
        deferring means running the whole lifetime without the columns."""
        import sqlite3

        from slipstream import store as store_mod
        from slipstream.store import CommitStore, StoreError

        monkeypatch.setattr(store_mod, "_MIGRATE_RETRY_SECS", 0.0)

        def locked(self, table, column, decl):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(CommitStore, "_add_column", locked)
        with __import__("pytest").raises(StoreError, match="locked by another writer"):
            CommitStore(tmp_path / "t.db", bot="box1")

    def test_a_transient_lock_is_retried(self, tmp_path, monkeypatch):
        import sqlite3

        from slipstream import store as store_mod
        from slipstream.store import CommitStore

        monkeypatch.setattr(store_mod, "_MIGRATE_RETRY_SECS", 0.0)
        real = CommitStore._add_column
        attempts = [0]

        def flaky(self, table, column, decl):
            attempts[0] += 1
            if attempts[0] <= 2:
                raise sqlite3.OperationalError("database is locked")
            return real(self, table, column, decl)

        monkeypatch.setattr(CommitStore, "_add_column", flaky)
        s = CommitStore(tmp_path / "t.db", bot="box1")
        s.mark_done("v8", "arm64", 100)
        assert s.get_status("v8", "arm64", 100) == "ok"
        s.close()
