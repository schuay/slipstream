# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path


class StoreError(RuntimeError):
    """A store invariant was violated."""


class BotMismatch(StoreError):
    """The db was written by a different bot than this machine is configured as.

    Almost always a db file copied between machines: continuing would write two
    machines' rows into one series, which nothing downstream can separate again.
    """


class CommitIdCollision(StoreError):
    """Two hashes claim the same commit_id for one engine.

    The partial unique index on (engine, commit_id) rejects the second one.
    """


# Guarded ALTERs applied to dbs created before these columns existed. New dbs
# get them from _init_schema and the ALTER is a no-op.
_ADDED_COLUMNS = [
    ("scores", "bot", "TEXT"),
    ("processing_state", "bot", "TEXT"),
    ("processing_state", "status", "TEXT NOT NULL DEFAULT 'ok'"),
    ("processing_state", "configs_ok", "INTEGER"),
    ("processing_state", "configs_total", "INTEGER"),
    ("push_state", "bot", "TEXT"),
]

SCHEMA_VERSION = "2"

# A long writer can hold the db past busy_timeout. Retried rather than
# deferred, because a daemon has no "next open".
_MIGRATE_ATTEMPTS = 3
_MIGRATE_RETRY_SECS = 5.0


class CommitStore:
    """SQLite-backed store for commit metadata and processing state.

    One db holds one bot: ``bot`` is an informational column on the row tables,
    never a key and never filtered on, and the invariant is kept at the
    ingress boundary (``import``) instead. See DECISIONS.md D014.
    """

    def __init__(
        self,
        db_path: Path,
        readonly: bool = False,
        *,
        backup: bool | None = None,
        init_schema: bool = True,
        bot: str | None = None,
    ):
        """Open a connection to the store.

        ``backup`` defaults to ``not readonly``. Set it False for secondary
        connections (e.g. the background pusher) that must not snapshot the
        db again. ``init_schema`` may be disabled when the caller knows the
        schema already exists, to avoid re-running migrations concurrently
        from a second connection; ``readonly`` disables it too, so reporting
        commands read what is there rather than writing to it.

        ``bot`` names this machine. It is recorded in ``meta.local_bot`` on the
        first open that has one, and a later open under a different name is
        refused.
        """
        self._db_path = db_path
        self._bot = bot
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if backup is None:
            backup = not readonly
        # Whether the db existed before we opened it: connect() creates an empty
        # file, so this must be captured before connect() to avoid backing up a
        # freshly created (empty) db.
        pre_existing = db_path.exists()
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        # busy_timeout first, so the WAL conversion below (and every later op)
        # waits on a concurrent writer rather than raising immediately.
        self.conn.execute("PRAGMA busy_timeout=30000")
        # WAL lets the background pusher's reader coexist with the collector's
        # writer instead of deadlocking on lock promotion. It is a persistent
        # property of the db file, so setting it on any connection converts it.
        self.conn.execute("PRAGMA journal_mode=WAL")
        if backup and pre_existing:
            # Snapshot before _init_schema so the backup predates any migration.
            self._backup(self.conn, db_path)
        # A reporting command reads what is there rather than migrating a live
        # db on every run. A db that does not exist yet has nothing to read, so
        # it is created either way: otherwise the first export or analyze on a
        # fresh machine fails on a missing table instead of reporting nothing.
        if init_schema and (not readonly or not pre_existing):
            self._init_schema()
        elif bot is not None:
            self._check_bot()

    @staticmethod
    def _backup(conn: sqlite3.Connection, db_path: Path, keep: int = 10):
        # Use SQLite's online backup API rather than a file copy: in WAL mode
        # recent commits live in the -wal sidecar, so a bare copy of the .db
        # would silently lose them. conn.backup() produces a consistent,
        # fully checkpointed snapshot regardless of journal mode.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = db_path.with_suffix(f".{ts}.bak")
        dest = sqlite3.connect(str(backup))
        try:
            conn.backup(dest)
        finally:
            dest.close()
        backups = sorted(db_path.parent.glob(f"{db_path.stem}.*.bak"), reverse=True)
        for old in backups[keep:]:
            old.unlink()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS commits (
                engine      TEXT    NOT NULL,
                hash        TEXT    NOT NULL,
                commit_id   INTEGER,
                date        TEXT    NOT NULL DEFAULT '',
                timestamp   INTEGER NOT NULL DEFAULT 0,
                title       TEXT    NOT NULL DEFAULT '',
                PRIMARY KEY (engine, hash)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_commit_id
                ON commits (engine, commit_id)
                WHERE commit_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS scores (
                engine     TEXT    NOT NULL,
                platform   TEXT    NOT NULL,
                commit_id  INTEGER NOT NULL,
                suite      TEXT    NOT NULL,
                flags      TEXT    NOT NULL DEFAULT 'default',
                benchmark  TEXT    NOT NULL,
                metric     TEXT    NOT NULL,
                run        INTEGER NOT NULL,
                score      REAL    NOT NULL,
                timestamp  INTEGER NOT NULL,
                bot        TEXT,
                PRIMARY KEY (engine, platform, commit_id, suite, flags, benchmark, metric, run)
            );
            CREATE INDEX IF NOT EXISTS idx_scores_lookup
                ON scores (engine, suite, flags, benchmark, metric, commit_id);

            CREATE TABLE IF NOT EXISTS processing_state (
                engine        TEXT    NOT NULL,
                platform      TEXT    NOT NULL,
                commit_id     INTEGER NOT NULL,
                bot           TEXT,
                status        TEXT    NOT NULL DEFAULT 'ok',
                configs_ok    INTEGER,
                configs_total INTEGER,
                PRIMARY KEY (engine, platform, commit_id)
            );

            CREATE TABLE IF NOT EXISTS push_state (
                engine     TEXT    NOT NULL,
                platform   TEXT    NOT NULL,
                commit_id  INTEGER NOT NULL,
                pushed_at  INTEGER NOT NULL,
                bot        TEXT,
                PRIMARY KEY (engine, platform, commit_id)
            );

            CREATE TABLE IF NOT EXISTS run_env (
                engine             TEXT    NOT NULL,
                bot                TEXT    NOT NULL DEFAULT '',
                commit_id          INTEGER NOT NULL,
                source             TEXT    NOT NULL DEFAULT 'local',
                runs               INTEGER,
                run_configs        TEXT    NOT NULL DEFAULT '[]',
                harness_revs       TEXT    NOT NULL DEFAULT '{}',
                hw_model           TEXT    NOT NULL DEFAULT '',
                os_version         TEXT    NOT NULL DEFAULT '',
                toolchain          TEXT    NOT NULL DEFAULT '',
                build_cfg_hash     TEXT    NOT NULL DEFAULT '',
                slipstream_version TEXT    NOT NULL DEFAULT '',
                recorded_at        INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (engine, bot, commit_id)
            );

            CREATE TABLE IF NOT EXISTS build_state (
                engine       TEXT    NOT NULL,
                commit_id    INTEGER NOT NULL,
                status       TEXT    NOT NULL,
                kind         TEXT    NOT NULL DEFAULT '',
                log_path     TEXT    NOT NULL DEFAULT '',
                attempts     INTEGER NOT NULL DEFAULT 0,
                last_attempt INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (engine, commit_id)
            );

            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        # Migrate scores: UNIQUE constraint -> PRIMARY KEY (idempotent)
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='scores'"
        ).fetchone()
        if row and "PRIMARY KEY" not in row[0]:
            self.conn.executescript("""
                ALTER TABLE scores RENAME TO _scores_old;
                CREATE TABLE scores (
                    engine     TEXT    NOT NULL,
                    platform   TEXT    NOT NULL,
                    commit_id  INTEGER NOT NULL,
                    suite      TEXT    NOT NULL,
                    flags      TEXT    NOT NULL DEFAULT 'default',
                    benchmark  TEXT    NOT NULL,
                    metric     TEXT    NOT NULL,
                    run        INTEGER NOT NULL,
                    score      REAL    NOT NULL,
                    timestamp  INTEGER NOT NULL,
                    bot        TEXT,
                    PRIMARY KEY (engine, platform, commit_id, suite, flags, benchmark, metric, run)
                );
                INSERT INTO scores
                    (engine, platform, commit_id, suite, flags, benchmark, metric,
                     run, score, timestamp)
                    SELECT engine, platform, commit_id, suite, flags, benchmark,
                           metric, run, score, timestamp FROM _scores_old;
                DROP TABLE _scores_old;
                CREATE INDEX IF NOT EXISTS idx_scores_lookup
                    ON scores (engine, suite, flags, benchmark, metric, commit_id);
            """)
        self.conn.commit()
        self._migrate()

    def _migrate(self):
        """Add the columns older dbs lack and record this machine's identity.

        Every command opens the store and two daemons open the same file, so
        this races itself: the ALTERs attempt-and-catch rather than
        check-then-act, and the backfill is idempotent on ``bot IS NULL``.
        A writer holding a transaction makes both raise once busy_timeout
        expires. It is retried, and then it is fatal: a daemon opens the store
        once and holds the connection for days, so "defer to the next open"
        would mean running its whole lifetime against a db missing the columns
        and dying at the first mark_done with "no such column: status" instead
        of saying so here.
        """
        # Before the try: this does not depend on the migration, and the
        # handler below returns quietly on a locked db. Swallowed, it would
        # accept a db copied from the other machine for this whole process.
        self._check_bot()
        for attempt in range(_MIGRATE_ATTEMPTS):
            try:
                self._run_migration()
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() and "busy" not in str(e).lower():
                    raise
                self.conn.rollback()
                if attempt == _MIGRATE_ATTEMPTS - 1:
                    raise StoreError(
                        f"{self._db_path} is locked by another writer, so it "
                        f"could not be migrated. Stop whatever is holding it "
                        f"and retry."
                    ) from e
                time.sleep(_MIGRATE_RETRY_SECS)

    def _run_migration(self):
        for table, column, decl in _ADDED_COLUMNS:
            self._add_column(table, column, decl)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        if self._bot is not None and self.get_meta("bot_backfilled") is None:
            # Once, not on every open: the UPDATE is a full scan of scores
            # under a write lock, and every command opens the store.
            self.conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('local_bot', ?)",
                (self._bot,),
            )
            for table, column, _ in _ADDED_COLUMNS:
                if column == "bot":
                    self.conn.execute(
                        f"UPDATE {table} SET bot=? WHERE bot IS NULL", (self._bot,)
                    )
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('bot_backfilled', ?)",
                (self._bot,),
            )
        self.conn.commit()

    def _add_column(self, table: str, column: str, decl: str) -> None:
        try:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise

    def _check_bot(self) -> None:
        """Refuse a db that belongs to another machine."""
        if self._bot is None:
            return
        recorded = self.get_meta("local_bot")
        if recorded is not None and recorded != self._bot:
            raise BotMismatch(
                f"{self._db_path} was written by bot {recorded!r}, "
                f"but this machine is configured as {self._bot!r}"
            )

    def get_meta(self, key: str) -> str | None:
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None  # db predates the meta table and was opened readonly
        return row[0] if row else None

    @property
    def bot(self) -> str | None:
        """The bot this db belongs to: the configured name, else the recorded one."""
        return self._bot or self.get_meta("local_bot")

    # --- Commit insert ---

    def insert_commits(self, engine: str, commits: list[dict]):
        """Insert (hash, title) rows; no-op if already present."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO commits (engine, hash, title) VALUES (?,?,?)",
            [(engine, c["hash"], c.get("title", "")) for c in commits],
        )
        self.conn.commit()

    # --- Commit metadata ---

    def count_missing_metadata(self, engine: str, hashes: list[str]) -> int:
        placeholders = ",".join("?" * len(hashes))
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM commits"
            f" WHERE engine=? AND hash IN ({placeholders}) AND commit_id IS NULL",
            [engine, *hashes],
        ).fetchone()
        return row[0] if row else 0

    def update_commit_metadata(
        self,
        engine: str,
        hash: str,
        commit_id: int,
        date: str,
        timestamp: int,
        title: str,
    ):
        """Set metadata only if not already populated."""
        self.conn.execute(
            """UPDATE commits SET commit_id=?, date=?, timestamp=?, title=?
               WHERE engine=? AND hash=? AND commit_id IS NULL""",
            (commit_id, date, timestamp, title, engine, hash),
        )

    def upsert_commit(
        self,
        engine: str,
        hash: str,
        commit_id: int,
        date: str,
        timestamp: int,
        title: str,
    ):
        """Write commit metadata that did not come from a local git log.

        The bus carries it in the entry, so the row must be written even for a
        commit this machine has never resolved: ``export_scores`` inner-joins
        ``commits``, and ``push()`` marks a candidate pushed whether or not it
        exported rows, so a done commit with no row here loses its scores for
        good.

        Not INSERT OR REPLACE: ``commits`` carries a partial unique index on
        (engine, commit_id), and replace through it deletes the incumbent row
        when another hash holds that id, orphaning that commit's scores from
        the export join.
        """
        self._upsert_commit(engine, hash, commit_id, date, timestamp, title)
        self.conn.commit()

    def _upsert_commit(
        self,
        engine: str,
        hash: str,
        commit_id: int,
        date: str,
        timestamp: int,
        title: str,
    ):
        try:
            self.conn.execute(
                "INSERT INTO commits (engine, hash, commit_id, date, timestamp, title)"
                " VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(engine, hash) DO UPDATE SET"
                "   commit_id=excluded.commit_id, date=excluded.date,"
                "   timestamp=excluded.timestamp, title=excluded.title",
                (engine, hash, commit_id, date, timestamp, title),
            )
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            other = self.conn.execute(
                "SELECT hash FROM commits WHERE engine=? AND commit_id=?",
                (engine, commit_id),
            ).fetchone()
            raise CommitIdCollision(
                f"{engine} {commit_id} is already held by "
                f"{other[0] if other else '?'}, not {hash}"
            ) from e

    def commit_ids_missing_commit_row(
        self,
        engine: str,
        platform: str,
        lo: int | None = None,
        hi: int | None = None,
    ) -> list[int]:
        """Commit ids with scores but no row in ``commits``.

        Their scores can never be exported (the export inner-joins ``commits``)
        so they must not be marked pushed, and the count is worth surfacing:
        it means a bench wrote scores without the metadata that goes with them.

        ``lo``/``hi`` bound the scan to a commit range. The background pusher
        runs this after every benched commit and is meant to be cheap, and
        unbounded it is a DISTINCT LEFT JOIN over the engine's whole history.
        A range rather than an id list, so the caller's set can be any size.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT s.commit_id FROM scores s"
            " LEFT JOIN commits c ON s.engine=c.engine AND s.commit_id=c.commit_id"
            " WHERE s.engine=? AND s.platform=? AND c.hash IS NULL"
            "   AND (? IS NULL OR s.commit_id >= ?)"
            "   AND (? IS NULL OR s.commit_id <= ?)"
            " ORDER BY s.commit_id",
            (engine, platform, lo, lo, hi, hi),
        ).fetchall()
        return [r[0] for r in rows]

    def get_commits_with_metadata(
        self, engine: str, hashes: list[str]
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(hashes))
        return self.conn.execute(
            f"SELECT hash, commit_id, date, timestamp, title FROM commits"
            f" WHERE engine=? AND hash IN ({placeholders}) AND commit_id IS NOT NULL"
            f" ORDER BY commit_id",
            [engine, *hashes],
        ).fetchall()

    # --- Processing state ---

    def is_done(self, engine: str, platform: str, commit_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processing_state"
            " WHERE engine=? AND platform=? AND commit_id=?",
            (engine, platform, commit_id),
        ).fetchone()
        return row is not None

    def mark_done(
        self,
        engine: str,
        platform: str,
        commit_id: int,
        status: str | None = None,
        configs_ok: int | None = None,
        configs_total: int | None = None,
    ):
        """Record a commit as processed. The latest write wins.

        A re-bench must be able to move a commit from ``failed`` to ``ok`` and
        back, so this is an upsert rather than an insert. ``status=None``
        preserves whatever is already recorded (``import`` has no run to
        report), and defaults to ``ok`` for a row that does not exist yet.
        """
        self._mark_done(engine, platform, commit_id, status, configs_ok, configs_total)
        self.conn.commit()

    def _mark_done(
        self,
        engine: str,
        platform: str,
        commit_id: int,
        status: str | None,
        configs_ok: int | None,
        configs_total: int | None,
    ):
        self.conn.execute(
            "INSERT INTO processing_state"
            " (engine, platform, commit_id, bot, status, configs_ok, configs_total)"
            " VALUES (:engine, :platform, :commit_id, :bot,"
            "         COALESCE(:status, 'ok'), :configs_ok, :configs_total)"
            " ON CONFLICT(engine, platform, commit_id) DO UPDATE SET"
            "   status        = COALESCE(:status, processing_state.status),"
            "   configs_ok    = COALESCE(:configs_ok, processing_state.configs_ok),"
            "   configs_total = COALESCE(:configs_total, processing_state.configs_total),"
            "   bot           = COALESCE(processing_state.bot, :bot)",
            {
                "engine": engine,
                "platform": platform,
                "commit_id": commit_id,
                "bot": self._bot,
                "status": status,
                "configs_ok": configs_ok,
                "configs_total": configs_total,
            },
        )

    def record_done(
        self,
        engine: str,
        platform: str,
        commit: dict,
        *,
        status: str | None = None,
        configs_ok: int | None = None,
        configs_total: int | None = None,
    ):
        """Write the commit's metadata and its processing state together.

        One transaction, because the two are one fact: ``export_scores``
        inner-joins ``commits`` while ``push()`` marks every candidate commit
        pushed, so a commit marked done with no metadata row has its scores
        marked pushed and never sent. On a bus consumer the metadata comes from
        the entry rather than from git, so this is the normal path.
        """
        self._upsert_commit(
            engine,
            commit["hash"],
            int(commit["commit_id"]),
            commit.get("date", ""),
            int(commit.get("timestamp", 0)),
            commit.get("title", ""),
        )
        self._mark_done(
            engine,
            platform,
            int(commit["commit_id"]),
            status,
            configs_ok,
            configs_total,
        )
        self.conn.commit()

    def get_status(self, engine: str, platform: str, commit_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM processing_state"
            " WHERE engine=? AND platform=? AND commit_id=?",
            (engine, platform, commit_id),
        ).fetchone()
        return row[0] if row else None

    def status_counts(self, engine: str, platform: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM processing_state"
            " WHERE engine=? AND platform=? GROUP BY status",
            (engine, platform),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def clear_scores(self, engine: str, platform: str, commit_ids: list[int]):
        """Delete only the scores of commits, leaving their state alone.

        For resuming an interrupted commit: ``scores`` is INSERT OR IGNORE with
        ``run`` in the primary key, so a retry that kept the partial rows would
        leave one run number half measured on each side of the interrupt.
        """
        if not commit_ids:
            return
        ph = ",".join("?" * len(commit_ids))
        self.conn.execute(
            f"DELETE FROM scores WHERE engine=? AND platform=? AND commit_id IN ({ph})",
            [engine, platform, *commit_ids],
        )
        self.conn.commit()

    def clear_range(self, engine: str, platform: str, commit_ids: list[int]):
        """Delete scores, processing state, and push state for specific commits.

        Clearing push_state too means any re-benchmarked commit will be
        re-pushed on the next push cycle.
        """
        if not commit_ids:
            return
        ph = ",".join("?" * len(commit_ids))
        self.conn.execute(
            f"DELETE FROM scores WHERE engine=? AND platform=? AND commit_id IN ({ph})",
            [engine, platform, *commit_ids],
        )
        self.conn.execute(
            f"DELETE FROM processing_state WHERE engine=? AND platform=? AND commit_id IN ({ph})",
            [engine, platform, *commit_ids],
        )
        self.conn.execute(
            f"DELETE FROM push_state WHERE engine=? AND platform=? AND commit_id IN ({ph})",
            [engine, platform, *commit_ids],
        )
        self.conn.execute(
            f"DELETE FROM run_env WHERE engine=? AND commit_id IN ({ph})",
            [engine, *commit_ids],
        )
        self.conn.commit()

    def get_all_commits(self, engine: str) -> list[sqlite3.Row]:
        """All commits with metadata, ordered by commit_id."""
        return self.conn.execute(
            "SELECT hash, commit_id, date, timestamp, title FROM commits"
            " WHERE engine=? AND commit_id IS NOT NULL"
            " ORDER BY commit_id",
            (engine,),
        ).fetchall()

    def get_commits_in_range(
        self, engine: str, after_id: int, up_to_id: int
    ) -> list[sqlite3.Row]:
        """All commits with after_id < commit_id <= up_to_id, ordered by commit_id."""
        return self.conn.execute(
            "SELECT hash, commit_id, date, timestamp, title FROM commits"
            " WHERE engine=? AND commit_id IS NOT NULL"
            " AND commit_id > ? AND commit_id <= ?"
            " ORDER BY commit_id",
            (engine, after_id, up_to_id),
        ).fetchall()

    # --- Scores ---

    def insert_scores(
        self,
        engine: str,
        platform: str,
        commit_id: int,
        timestamp: int,
        scores: list[dict],
    ):
        """Insert raw score rows. Each dict: {suite, flags, benchmark, metric, run, score}."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO scores (engine, platform, commit_id, suite, flags,"
            " benchmark, metric, run, score, timestamp, bot)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    engine,
                    platform,
                    commit_id,
                    s["suite"],
                    s["flags"],
                    s["benchmark"],
                    s["metric"],
                    s["run"],
                    s["score"],
                    timestamp,
                    self._bot,
                )
                for s in scores
            ],
        )
        self.conn.commit()

    def bulk_insert_scores(self, engine: str, platform: str, rows: list[tuple]):
        """Insert raw score tuples: (commit_id, suite, flags, benchmark, metric, run, score, timestamp)."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO scores (engine, platform, commit_id, suite, flags,"
            " benchmark, metric, run, score, timestamp, bot)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [(engine, platform, *r, self._bot) for r in rows],
        )
        self.conn.commit()

    def get_series(
        self,
        engine: str,
        suite: str,
        flags: str,
        benchmark: str,
        metric: str,
    ) -> list[sqlite3.Row]:
        """Return (commit_id, score) rows ordered by commit_id for one benchmark series."""
        return self.conn.execute(
            "SELECT commit_id, score FROM scores"
            " WHERE engine=? AND suite=? AND flags=? AND benchmark=? AND metric=?"
            " ORDER BY commit_id",
            (engine, suite, flags, benchmark, metric),
        ).fetchall()

    def export_scores(
        self,
        engine: str,
        platform: str,
        valid_benchmarks_by_suite: dict[str, set[str]],
        commit_ids: list[int] | None = None,
    ) -> list[sqlite3.Row]:
        """Score rows joined with commit metadata, for pushing.

        Benchmarks are filtered per suite so that e.g. js2-only names don't
        leak into js3. ``commit_ids`` restricts the export; it is materialised
        in a temp table to sidestep SQLite's bound-parameter limit.
        """
        clauses = []
        params: list = [engine, platform]
        for suite, names in valid_benchmarks_by_suite.items():
            if not names:
                continue
            placeholders = ",".join("?" * len(names))
            clauses.append(f"(s.suite = ? AND s.benchmark IN ({placeholders}))")
            params.append(suite)
            params.extend(names)
        if not clauses:
            return []
        where_suite = " OR ".join(clauses)

        commit_filter = ""
        if commit_ids is not None:
            if not commit_ids:
                return []
            self.conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _export_commits"
                " (commit_id INTEGER PRIMARY KEY)"
            )
            self.conn.execute("DELETE FROM _export_commits")
            self.conn.executemany(
                "INSERT INTO _export_commits (commit_id) VALUES (?)",
                [(c,) for c in commit_ids],
            )
            commit_filter = (
                " AND s.commit_id IN (SELECT commit_id FROM _export_commits)"
            )

        try:
            return self.conn.execute(
                "SELECT s.engine, s.platform, s.commit_id, s.suite, s.flags,"
                "       s.benchmark, s.metric, s.run, s.score, s.timestamp,"
                "       c.hash, c.date, c.timestamp, c.title"
                " FROM scores s"
                " JOIN commits c ON s.engine = c.engine AND s.commit_id = c.commit_id"
                f" WHERE s.engine = ? AND s.platform = ? AND ({where_suite})"
                f"{commit_filter}"
                " ORDER BY s.commit_id, s.suite, s.flags, s.benchmark, s.metric, s.run",
                params,
            ).fetchall()
        finally:
            if commit_filter:
                self.conn.execute("DELETE FROM _export_commits")

    def export_compat_rows(self, engine: str, valid_names: set[str]) -> list[tuple]:
        """Score rows in the interchange CSV's column order, minus the bot.

        Filtered to the configured benchmark names, so an engine's ad-hoc or
        renamed benchmarks stay out of a file meant to be read elsewhere.
        """
        if not valid_names:
            return []
        placeholders = ",".join("?" * len(valid_names))
        rows = self.conn.execute(
            "SELECT suite, flags, benchmark, metric, commit_id, score"
            f" FROM scores WHERE engine=? AND benchmark IN ({placeholders})"
            " ORDER BY commit_id, suite, benchmark, metric, run",
            (engine, *sorted(valid_names)),
        ).fetchall()
        return [tuple(r) for r in rows]

    def get_distinct_series_keys(self, engine: str) -> list[sqlite3.Row]:
        """Return distinct (suite, flags, benchmark, metric) tuples."""
        return self.conn.execute(
            "SELECT DISTINCT suite, flags, benchmark, metric FROM scores"
            " WHERE engine=? ORDER BY suite, flags, benchmark, metric",
            (engine,),
        ).fetchall()

    # --- Provenance ---

    def record_run_env(self, engine: str, commit_id: int, env: dict) -> None:
        """What produced one commit's numbers, for the bot that measured them.

        The builder's toolchain matters because an Xcode or macOS update on it
        shifts every series on both bots on the same day, and would otherwise
        be attributed to a commit. This machine's own environment matters
        because it is what distinguishes the two series. ``source`` records
        whether the binary came off the bus or was built here: git-driven
        engines and ad-hoc bench ranges both produce locally built numbers
        beside archive-built neighbours, and that mixture should be visible
        rather than forbidden in one command and silently allowed in another.
        """
        # runs is the only nullable column: the rest are NOT NULL with a text
        # default, so a caller that knows less than all of it still writes a row.
        defaults = {
            "source": "local",
            "runs": None,
            "run_configs": "[]",
            "harness_revs": "{}",
        }
        columns = [
            "source",
            "runs",
            "run_configs",
            "harness_revs",
            "hw_model",
            "os_version",
            "toolchain",
            "build_cfg_hash",
            "slipstream_version",
        ]
        values = [env.get(c, defaults.get(c, "")) for c in columns]
        values = [
            defaults.get(c, "") if v is None and c != "runs" else v
            for c, v in zip(columns, values)
        ]
        assignments = ", ".join(f"{c}=excluded.{c}" for c in columns)
        self.conn.execute(
            f"INSERT INTO run_env (engine, bot, commit_id, {', '.join(columns)},"
            f" recorded_at) VALUES (?,?,?,{','.join('?' * len(columns))},?)"
            f" ON CONFLICT(engine, bot, commit_id) DO UPDATE SET"
            f" {assignments}, recorded_at=excluded.recorded_at",
            (engine, self._bot or "", commit_id, *values, int(time.time())),
        )
        self.conn.commit()

    def get_run_env(self, engine: str, commit_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM run_env WHERE engine=? AND bot=? AND commit_id=?",
            (engine, self._bot or "", commit_id),
        ).fetchone()

    def run_env_source_counts(self, engine: str, last: int = 50) -> dict[str, int]:
        """How the most recent commits' binaries were produced."""
        rows = self.conn.execute(
            "SELECT source, COUNT(*) FROM ("
            "  SELECT source FROM run_env WHERE engine=?"
            "  ORDER BY commit_id DESC LIMIT ?"
            ") GROUP BY source",
            (engine, last),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # --- Build state ---
    #
    # Load-bearing local state on the builder, not a report: a recorded
    # terminal failure is what stops a commit that cannot be built from being
    # rebuilt every cycle forever, and what lets the frontier advance past it.

    def record_build_failure(
        self, engine: str, commit_id: int, status: str, kind: str, log_path: str = ""
    ) -> int:
        """Record a failed build attempt; returns the new attempt count."""
        row = self.conn.execute(
            "SELECT attempts FROM build_state WHERE engine=? AND commit_id=?",
            (engine, commit_id),
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        self.conn.execute(
            "INSERT INTO build_state"
            " (engine, commit_id, status, kind, log_path, attempts, last_attempt)"
            " VALUES (?,?,?,?,?,?,?)"
            " ON CONFLICT(engine, commit_id) DO UPDATE SET"
            "   status=excluded.status, kind=excluded.kind,"
            "   log_path=excluded.log_path, attempts=excluded.attempts,"
            "   last_attempt=excluded.last_attempt",
            (engine, commit_id, status, kind, log_path, attempts, int(time.time())),
        )
        self.conn.commit()
        return attempts

    def set_build_status(self, engine: str, commit_id: int, status: str) -> None:
        """Reclassify an existing failure without counting another attempt."""
        self.conn.execute(
            "UPDATE build_state SET status=? WHERE engine=? AND commit_id=?",
            (status, engine, commit_id),
        )
        self.conn.commit()

    def request_build_retry(self, engine: str, commit_id: int) -> bool:
        """Allow one more attempt at a commit the frontier has moved past.

        Persisted rather than an effect of the command: entries above the
        failed commit exist by the time anyone retries, so deleting the row
        would leave the builder resolving "next commit above the frontier" and
        never picking this one up.
        """
        cur = self.conn.execute(
            "UPDATE build_state SET status='retry_requested'"
            " WHERE engine=? AND commit_id=?",
            (engine, commit_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def clear_build_state(self, engine: str, commit_id: int) -> None:
        self.conn.execute(
            "DELETE FROM build_state WHERE engine=? AND commit_id=?",
            (engine, commit_id),
        )
        self.conn.commit()

    def get_build_state(self, engine: str, commit_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM build_state WHERE engine=? AND commit_id=?",
            (engine, commit_id),
        ).fetchone()

    def build_retries_requested(self, engine: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT commit_id FROM build_state"
            " WHERE engine=? AND status='retry_requested' ORDER BY commit_id",
            (engine,),
        ).fetchall()
        return [r[0] for r in rows]

    def max_terminal_build_id(self, engine: str) -> int | None:
        """Highest commit with a terminal build failure.

        Non-terminal rows are excluded by construction: including them would
        advance the frontier past exactly the commit the retry rule says must
        be attempted again, and the retry would never happen. It also keeps a
        crash between the payload rename and the entry rename self-healing,
        since that leaves a non-terminal row.
        """
        row = self.conn.execute(
            "SELECT MAX(commit_id) FROM build_state"
            " WHERE engine=? AND status IN ('compile_failed', 'infra_burned')",
            (engine,),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def build_failures(
        self, engine: str, above: int | None = None
    ) -> list[sqlite3.Row]:
        """Terminal failures, for the state file and the circuit breaker."""
        return self.conn.execute(
            "SELECT commit_id, status, kind, log_path, attempts, last_attempt"
            " FROM build_state"
            " WHERE engine=? AND status IN ('compile_failed', 'infra_burned')"
            "   AND (? IS NULL OR commit_id > ?)"
            " ORDER BY commit_id",
            (engine, above, above),
        ).fetchall()

    # --- Push state ---

    def unpushed_commit_ids(self, engine: str, platform: str) -> list[int]:
        """Return commit_ids that are done but not yet pushed, ordered ascending."""
        rows = self.conn.execute(
            "SELECT ps.commit_id FROM processing_state ps"
            " LEFT JOIN push_state pu"
            "   ON pu.engine=ps.engine AND pu.platform=ps.platform"
            "  AND pu.commit_id=ps.commit_id"
            " WHERE ps.engine=? AND ps.platform=? AND pu.commit_id IS NULL"
            " ORDER BY ps.commit_id",
            (engine, platform),
        ).fetchall()
        return [r[0] for r in rows]

    def mark_pushed(self, engine: str, platform: str, commit_ids: list[int]) -> None:
        """Mark commit_ids as successfully pushed to Spanner."""
        if not commit_ids:
            return
        now = int(time.time())
        self.conn.executemany(
            "INSERT OR REPLACE INTO push_state"
            " (engine, platform, commit_id, pushed_at, bot) VALUES (?,?,?,?,?)",
            [(engine, platform, cid, now, self._bot) for cid in commit_ids],
        )
        self.conn.commit()

    def clear_push_state(self, engine: str, platform: str) -> None:
        """Forget every push, so the next push re-sends the full history."""
        self.conn.execute(
            "DELETE FROM push_state WHERE engine=? AND platform=?", (engine, platform)
        )
        self.conn.commit()

    def max_done_commit_id(self, engine: str, platform: str) -> int | None:
        """Return the highest commit_id processed on this platform, or None."""
        row = self.conn.execute(
            "SELECT MAX(commit_id) FROM processing_state WHERE engine=? AND platform=?",
            (engine, platform),
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def close(self):
        self.conn.close()
