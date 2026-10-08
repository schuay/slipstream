# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from slipstream.models import CommitKey


class TestRotation:
    def test_rotation_survives_reopening_and_keeps_roles_separate(self, tmp_path):
        from slipstream.store import CommitStore

        path = tmp_path / "rotation.db"
        with_store = CommitStore(path)
        with_store.record_rotation_turn("build", "v8")
        with_store.close()
        reopened = CommitStore(path)
        try:
            assert reopened.rotation_order("build", ["v8", "jsc"]) == ["jsc", "v8"]
            assert reopened.rotation_order("watch", ["v8", "jsc"]) == ["v8", "jsc"]
            assert reopened.rotation_order("build", ["jsc"]) == ["jsc"]
            assert reopened.rotation_order("build", []) == []
        finally:
            reopened.close()


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

    def test_max_done_key(self, store):
        assert store.max_done_key("v8", "arm64") is None

        store.insert_commits("v8", [{"hash": "a"}, {"hash": "b"}])
        store.update_commit_metadata("v8", "a", 100, "d", 0, "t")
        store.update_commit_metadata("v8", "b", 200, "d", 0, "t")
        store.mark_done("v8", "arm64", 100)
        store.mark_done("v8", "arm64", 200)
        assert store.max_done_key("v8", "arm64") == CommitKey(0, 200)
        assert store.max_done_key("v8", "arm64", embedder_id=0) == CommitKey(0, 200)
        assert store.max_done_key("v8", "arm64", embedder_id=7) is None
        assert store.max_done_key("v8", "x86_64") is None


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
        assert store.unpushed_keys("v8", "arm64") == [
            CommitKey(0, 100),
            CommitKey(0, 200),
            CommitKey(0, 300),
        ]

    def test_mark_pushed_filters_next_call(self, store):
        self._seed_done(store, [100, 200, 300])
        store.mark_pushed("v8", "arm64", [100, 200])
        assert store.unpushed_keys("v8", "arm64") == [CommitKey(0, 300)]

    def test_mark_pushed_idempotent(self, store):
        self._seed_done(store, [100])
        store.mark_pushed("v8", "arm64", [100])
        store.mark_pushed("v8", "arm64", [100])  # no error
        assert store.unpushed_keys("v8", "arm64") == []

    def test_clear_range_reinvalidates_push(self, store):
        self._seed_done(store, [100, 200])
        store.mark_pushed("v8", "arm64", [100, 200])
        assert store.unpushed_keys("v8", "arm64") == []

        # Re-bench 100: its scores, processing_state, AND push_state all clear.
        store.clear_range("v8", "arm64", [100])
        # 100 is no longer "done" → not in unpushed. Re-run mark_done:
        store.mark_done("v8", "arm64", 100)
        assert store.unpushed_keys("v8", "arm64") == [CommitKey(0, 100)]

    def test_platform_isolation(self, store):
        self._seed_done(store, [100])
        store.mark_pushed("v8", "arm64", [100])
        # x86_64 has no "done" commit here, so empty is expected.
        assert store.unpushed_keys("v8", "x86_64") == []

    def test_engine_isolation(self, store):
        self._seed_done(store, [100])
        store.insert_commits("jsc", [{"hash": "jh"}])
        store.update_commit_metadata("jsc", "jh", 100, "d", 0, "t")
        store.mark_done("jsc", "arm64", 100)

        store.mark_pushed("v8", "arm64", [100])
        assert store.unpushed_keys("v8", "arm64") == []
        assert store.unpushed_keys("jsc", "arm64") == [CommitKey(0, 100)]


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
        assert store.unpushed_keys("v8", "arm64") == []


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


class TestSchemaTooNew:
    """A db migrated by a newer slipstream is refused at open, not at the
    first write that happens to hit a reshaped key."""

    def _future_db(self, tmp_path):
        from slipstream.store import CommitStore

        db = tmp_path / "t.db"
        s = CommitStore(db, bot="box1-m1")
        s.conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        s.conn.commit()
        s.close()
        return db

    def test_a_writer_is_refused_before_backup_or_migration(self, tmp_path):
        import pytest

        from slipstream.store import CommitStore, SchemaTooNew

        db = self._future_db(tmp_path)
        before = sorted(tmp_path.glob("t.*.bak"))
        with pytest.raises(SchemaTooNew, match="schema version 99.*\\.bak"):
            CommitStore(db, bot="box1-m1")
        # Nothing was written: no new snapshot, version untouched.
        assert sorted(tmp_path.glob("t.*.bak")) == before
        import sqlite3

        c = sqlite3.connect(db)
        assert (
            c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
            == "99"
        )
        c.close()

    def test_a_readonly_open_is_refused_too(self, tmp_path):
        import pytest

        from slipstream.store import CommitStore, SchemaTooNew

        db = self._future_db(tmp_path)
        with pytest.raises(SchemaTooNew):
            CommitStore(db, readonly=True)

    def test_an_older_or_unnumbered_db_is_not_newer(self, tmp_path):
        """No meta table, no version row and a non-numeric value are all the
        migration's business, not the guard's."""
        import sqlite3

        from slipstream.store import SCHEMA_VERSION, CommitStore

        for name, setup in (
            ("nometa.db", "CREATE TABLE commits (engine TEXT, hash TEXT)"),
            (
                "norow.db",
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            ),
            (
                "junk.db",
                "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                "INSERT INTO meta VALUES ('schema_version', 'v2-ish')",
            ),
        ):
            c = sqlite3.connect(tmp_path / name)
            c.executescript(setup)
            c.commit()
            c.close()
            s = CommitStore(tmp_path / name)
            assert s.get_meta("schema_version") == SCHEMA_VERSION, name
            s.close()


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


class TestEmbedderKeyMigration:
    """A v2 db keys rows by commit_id alone; v3 puts embedder_id ahead of it.

    SQLite cannot ALTER a primary key, so every keyed table is rebuilt. What
    matters: no row is lost, every old row is embedder 0, the old index names
    are gone, and the same commit_id under a second embedder is a new row.
    """

    def _v2_db(self, path):
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
                score REAL NOT NULL, timestamp INTEGER NOT NULL, bot TEXT,
                PRIMARY KEY (engine, platform, commit_id, suite, flags, benchmark, metric, run));
            CREATE INDEX idx_scores_lookup
                ON scores (engine, suite, flags, benchmark, metric, commit_id);
            CREATE TABLE processing_state (
                engine TEXT NOT NULL, platform TEXT NOT NULL, commit_id INTEGER NOT NULL,
                bot TEXT, status TEXT NOT NULL DEFAULT 'ok',
                configs_ok INTEGER, configs_total INTEGER,
                PRIMARY KEY (engine, platform, commit_id));
            CREATE TABLE push_state (
                engine TEXT NOT NULL, platform TEXT NOT NULL, commit_id INTEGER NOT NULL,
                pushed_at INTEGER NOT NULL, bot TEXT,
                PRIMARY KEY (engine, platform, commit_id));
            CREATE TABLE run_env (
                engine TEXT NOT NULL, bot TEXT NOT NULL DEFAULT '',
                commit_id INTEGER NOT NULL, source TEXT NOT NULL DEFAULT 'local',
                runs INTEGER, run_configs TEXT NOT NULL DEFAULT '[]',
                harness_revs TEXT NOT NULL DEFAULT '{}', hw_model TEXT NOT NULL DEFAULT '',
                os_version TEXT NOT NULL DEFAULT '', toolchain TEXT NOT NULL DEFAULT '',
                build_cfg_hash TEXT NOT NULL DEFAULT '',
                slipstream_version TEXT NOT NULL DEFAULT '',
                recorded_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (engine, bot, commit_id));
            CREATE TABLE build_state (
                engine TEXT NOT NULL, commit_id INTEGER NOT NULL, status TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT '', log_path TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0, last_attempt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (engine, commit_id));
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta VALUES ('schema_version', '2');
            INSERT INTO commits VALUES ('v8','abc',100,'2026-01-01',17,'t100');
            INSERT INTO commits VALUES ('v8','def',101,'2026-01-02',18,'t101');
            INSERT INTO scores VALUES ('v8','arm64',100,'js3','default','b','Total-Score',1,1.0,0,'box1');
            INSERT INTO scores VALUES ('v8','arm64',101,'js3','default','b','Total-Score',1,2.0,0,'box1');
            INSERT INTO processing_state VALUES ('v8','arm64',100,'box1','ok',1,1);
            INSERT INTO processing_state VALUES ('v8','arm64',101,'box1','failed',0,1);
            INSERT INTO push_state VALUES ('v8','arm64',100,17,'box1');
            INSERT INTO run_env (engine, bot, commit_id) VALUES ('v8','box1',100);
            INSERT INTO build_state VALUES ('v8',101,'compile_failed','compile','/log',2,5);
        """)
        conn.commit()
        conn.close()

    def _open(self, tmp_path):
        from slipstream.store import CommitStore

        db = tmp_path / "v2.db"
        self._v2_db(db)
        return CommitStore(db, bot="box1")

    def test_every_keyed_table_gains_embedder_id_at_zero(self, tmp_path):
        from slipstream.store import _KEYED_TABLES

        s = self._open(tmp_path)
        for table in _KEYED_TABLES:
            info = {r[1]: r for r in s.conn.execute(f"PRAGMA table_info({table})")}
            assert "embedder_id" in info, table
            assert info["embedder_id"][5] > 0, f"{table}: embedder_id not in the key"
            rows = s.conn.execute(
                f"SELECT DISTINCT embedder_id FROM {table}"
            ).fetchall()
            assert [r[0] for r in rows] in ([0], []), table
        assert s.get_meta("schema_version") == "3"
        s.close()

    def test_rows_survive_and_read_back_through_the_api(self, tmp_path):
        s = self._open(tmp_path)
        assert s.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2
        assert s.is_done("v8", "arm64", 100) and s.is_done("v8", "arm64", 101)
        assert s.get_status("v8", "arm64", 101) == "failed"
        assert s.max_done_key("v8", "arm64") == CommitKey(0, 101)
        assert s.unpushed_keys("v8", "arm64") == [CommitKey(0, 101)]
        series = s.get_series("v8", "js3", "default", "b", "Total-Score")
        assert [(r["embedder_id"], r["commit_id"], r["score"]) for r in series] == [
            (0, 100, 1.0),
            (0, 101, 2.0),
        ]
        assert [r["commit_id"] for r in s.build_failures("v8")] == [101]
        s.close()

    def test_old_index_names_are_gone_and_new_ones_exist(self, tmp_path):
        s = self._open(tmp_path)
        names = {
            r[0]
            for r in s.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert not names & {"idx_commit_id", "idx_scores_lookup"}
        assert {"idx_commit_key", "idx_scores_series"} <= names
        s.close()

    def test_the_same_commit_id_under_another_embedder_is_a_new_row(self, tmp_path):
        s = self._open(tmp_path)
        s.mark_done("v8", "arm64", CommitKey(7, 100))
        assert s.is_done("v8", "arm64", CommitKey(7, 100))
        assert s.max_done_key("v8", "arm64") == CommitKey(7, 100)
        assert s.max_done_key("v8", "arm64", embedder_id=0) == CommitKey(0, 101)
        s.clear_range("v8", "arm64", [CommitKey(0, 100), CommitKey(0, 101)])
        assert s.is_done("v8", "arm64", CommitKey(7, 100))
        assert not s.is_done("v8", "arm64", 100)
        s.close()

    def test_reopening_is_a_no_op(self, tmp_path):
        s = self._open(tmp_path)
        s.close()
        from slipstream.store import CommitStore

        s = CommitStore(tmp_path / "v2.db", bot="box1")
        cols = [r[1] for r in s.conn.execute("PRAGMA table_info(scores)")]
        assert cols.count("embedder_id") == 1
        assert s.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 2
        s.close()

    def test_commits_gains_embedder_hash_through_the_rebuild(self, tmp_path):
        s = self._open(tmp_path)
        rows = s.conn.execute(
            "SELECT commit_id, embedder_hash FROM commits ORDER BY commit_id"
        ).fetchall()
        assert [tuple(r) for r in rows] == [(100, ""), (101, "")]
        s.close()

    def test_run_env_gains_the_runners_columns(self, tmp_path):
        s = self._open(tmp_path)
        row = s.get_run_env("v8", 100)
        assert row["runner_cfg_hash"] == "" and row["host_env"] == "{}"
        s.record_run_env(
            "v8",
            101,
            {"runner_cfg_hash": "sha256:r", "host_env": '{"host_app": "STP 1"}'},
        )
        row = s.get_run_env("v8", 101)
        assert row["runner_cfg_hash"] == "sha256:r"
        assert row["host_env"] == '{"host_app": "STP 1"}'
        s.close()


class TestEmbedderHash:
    def test_upsert_writes_and_updates_it(self, store):
        store.upsert_commit(
            "chrome", "v8h", CommitKey(1534000, 100), "d", 0, "t", "crA"
        )
        store.upsert_commit(
            "chrome", "v8h", CommitKey(1534000, 100), "d", 0, "t", "crB"
        )
        row = store.get_commits_in_range(
            "chrome", CommitKey(1534000, 99), CommitKey(1534000, 100)
        )[0]
        assert row["embedder_hash"] == "crB" and row["embedder_id"] == 1534000

    def test_record_done_takes_it_from_the_commit_dict(self, store):
        store.record_done(
            "chrome",
            "arm64",
            {"hash": "v8h", "commit_id": 100, "embedder_id": 7, "embedder_hash": "cr"},
        )
        assert store.get_all_commits("chrome")[0]["embedder_hash"] == "cr"
        store.record_done("v8", "arm64", {"hash": "h", "commit_id": 1})
        assert store.get_all_commits("v8")[0]["embedder_hash"] == ""

    def test_a_v3_db_from_before_the_column_gains_it(self, tmp_path):
        """The column arrived after the key rebuild; a db rebuilt without it
        is already version 3, so it comes by ALTER, not by another rebuild."""
        from slipstream.store import CommitStore

        db = tmp_path / "t.db"
        s = CommitStore(db)
        s.upsert_commit("v8", "h", 1, "d", 0, "t")
        s.conn.execute("ALTER TABLE commits DROP COLUMN embedder_hash")
        s.conn.commit()
        s.close()
        s = CommitStore(db)
        assert s.get_all_commits("v8")[0]["embedder_hash"] == ""
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


class TestWriteRollback:
    def test_failed_record_done_releases_writer_and_discards_partial_metadata(
        self, store, monkeypatch
    ):
        import sqlite3

        def schema_error(*args, **kwargs):
            # Metadata has already been written when processing-state fails.
            store.conn.execute(
                "INSERT INTO processing_state(no_such_column) VALUES (1)"
            )

        monkeypatch.setattr(store, "_mark_done", schema_error)
        with pytest.raises(sqlite3.OperationalError):
            store.record_done("v8", "arm64", {"hash": "abc", "commit_id": 101})
        assert not store.conn.in_transaction
        assert store.get_all_commits("v8") == []
        # A different daemon can write immediately after the failed operation.
        other = sqlite3.connect(store.db_path, timeout=0)
        try:
            other.execute("BEGIN IMMEDIATE")
        finally:
            other.rollback()
            other.close()
