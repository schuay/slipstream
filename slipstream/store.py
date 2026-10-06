# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

from .models import CommitKey


class StoreError(RuntimeError):
    """A store invariant was violated."""


class BotMismatch(StoreError):
    """The db was written by a different bot than this machine is configured as.

    Almost always a db file copied between machines: continuing would write two
    machines' rows into one series, which nothing downstream can separate again.
    """


class SchemaTooNew(StoreError):
    """The db was migrated by a newer slipstream than this one.

    Migrations are one-way: a key rebuild leaves upserts of the older code
    with an ON CONFLICT target that no longer names a key, and SQLite reports
    that at the first write, not at open. Refusing here names the real
    problem -- a downgrade -- and the way back, which is the ``.bak`` taken
    before the migration, not the code revert alone.
    """


class CommitIdCollision(StoreError):
    """Two hashes claim the same key for one engine.

    The partial unique index on (engine, embedder_id, commit_id) rejects the
    second one.
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
    ("commits", "embedder_hash", "TEXT NOT NULL DEFAULT ''"),
    ("run_env", "runner_cfg_hash", "TEXT NOT NULL DEFAULT ''"),
    ("run_env", "host_env", "TEXT NOT NULL DEFAULT '{}'"),
    ("run_env", "suite_cfg_hash", "TEXT NOT NULL DEFAULT '{}'"),
]

# Version 3 put embedder_id into every key. It is a primary key column, which
# SQLite cannot ALTER in, so a db that lacks it is rebuilt table by table.
SCHEMA_VERSION = "3"

# A long writer can hold the db past busy_timeout. Retried rather than
# deferred, because a daemon has no "next open".
_MIGRATE_ATTEMPTS = 3
_MIGRATE_RETRY_SECS = 5.0


def _write(method):
    """Roll back failed writes so an idle daemon cannot retain the writer lock."""

    @wraps(method)
    def write(self, *args, **kwargs):
        with self.conn:
            return method(self, *args, **kwargs)

    return write


# One definition per table, shared by the fresh-db path and the rebuild: the
# migration recreates a table from this text, so there is no second copy to
# drift. Every keyed table carries embedder_id ahead of commit_id, so the
# natural row order is CommitKey order.
_KEYED_TABLES = {
    # embedder_hash is the outer commit a two-coordinate key was built under
    # (the chromium roll CL for chrome); '' for an engine that is its own
    # embedder. The id is in the key; the hash is here because a position is
    # not something git can check out.
    "commits": """
        CREATE TABLE IF NOT EXISTS commits (
            engine        TEXT    NOT NULL,
            embedder_id   INTEGER NOT NULL DEFAULT 0,
            hash          TEXT    NOT NULL,
            commit_id     INTEGER,
            date          TEXT    NOT NULL DEFAULT '',
            timestamp     INTEGER NOT NULL DEFAULT 0,
            title         TEXT    NOT NULL DEFAULT '',
            embedder_hash TEXT    NOT NULL DEFAULT '',
            PRIMARY KEY (engine, embedder_id, hash)
        )""",
    "scores": """
        CREATE TABLE IF NOT EXISTS scores (
            engine      TEXT    NOT NULL,
            platform    TEXT    NOT NULL,
            embedder_id INTEGER NOT NULL DEFAULT 0,
            commit_id   INTEGER NOT NULL,
            suite       TEXT    NOT NULL,
            flags       TEXT    NOT NULL DEFAULT 'default',
            benchmark   TEXT    NOT NULL,
            metric      TEXT    NOT NULL,
            run         INTEGER NOT NULL,
            score       REAL    NOT NULL,
            timestamp   INTEGER NOT NULL,
            bot         TEXT,
            PRIMARY KEY (engine, platform, embedder_id, commit_id, suite, flags,
                         benchmark, metric, run)
        )""",
    "processing_state": """
        CREATE TABLE IF NOT EXISTS processing_state (
            engine        TEXT    NOT NULL,
            platform      TEXT    NOT NULL,
            embedder_id   INTEGER NOT NULL DEFAULT 0,
            commit_id     INTEGER NOT NULL,
            bot           TEXT,
            status        TEXT    NOT NULL DEFAULT 'ok',
            configs_ok    INTEGER,
            configs_total INTEGER,
            PRIMARY KEY (engine, platform, embedder_id, commit_id)
        )""",
    "push_state": """
        CREATE TABLE IF NOT EXISTS push_state (
            engine      TEXT    NOT NULL,
            platform    TEXT    NOT NULL,
            embedder_id INTEGER NOT NULL DEFAULT 0,
            commit_id   INTEGER NOT NULL,
            pushed_at   INTEGER NOT NULL,
            bot         TEXT,
            PRIMARY KEY (engine, platform, embedder_id, commit_id)
        )""",
    "run_env": """
        CREATE TABLE IF NOT EXISTS run_env (
            engine             TEXT    NOT NULL,
            bot                TEXT    NOT NULL DEFAULT '',
            embedder_id        INTEGER NOT NULL DEFAULT 0,
            commit_id          INTEGER NOT NULL,
            source             TEXT    NOT NULL DEFAULT 'local',
            runs               INTEGER,
            run_configs        TEXT    NOT NULL DEFAULT '[]',
            harness_revs       TEXT    NOT NULL DEFAULT '{}',
            hw_model           TEXT    NOT NULL DEFAULT '',
            os_version         TEXT    NOT NULL DEFAULT '',
            toolchain          TEXT    NOT NULL DEFAULT '',
            build_cfg_hash     TEXT    NOT NULL DEFAULT '',
            runner_cfg_hash    TEXT    NOT NULL DEFAULT '',
            host_env           TEXT    NOT NULL DEFAULT '{}',
            suite_cfg_hash     TEXT    NOT NULL DEFAULT '{}',
            slipstream_version TEXT    NOT NULL DEFAULT '',
            recorded_at        INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (engine, bot, embedder_id, commit_id)
        )""",
    "build_state": """
        CREATE TABLE IF NOT EXISTS build_state (
            engine       TEXT    NOT NULL,
            embedder_id  INTEGER NOT NULL DEFAULT 0,
            commit_id    INTEGER NOT NULL,
            status       TEXT    NOT NULL,
            kind         TEXT    NOT NULL DEFAULT '',
            log_path     TEXT    NOT NULL DEFAULT '',
            attempts     INTEGER NOT NULL DEFAULT 0,
            last_attempt INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (engine, embedder_id, commit_id)
        )""",
}

# Indexes by table, so the rebuild can recreate exactly the ones that belong
# to it. Names changed with the key: an index of the old name survives on the
# renamed table until it is dropped, and CREATE IF NOT EXISTS would see it.
_INDEXES = {
    "commits": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_commit_key
            ON commits (engine, embedder_id, commit_id)
            WHERE commit_id IS NOT NULL""",
    "scores": """
        CREATE INDEX IF NOT EXISTS idx_scores_series
            ON scores (engine, suite, flags, benchmark, metric,
                       embedder_id, commit_id)""",
}
_OLD_INDEXES = ("idx_commit_id", "idx_scores_lookup")


def _key_filter(keys: list[CommitKey]) -> tuple[str, list]:
    """A WHERE fragment matching any of ``keys``, with its parameters.

    One (embedder_id, commit_id) pair per key rather than a row-value IN,
    which SQLite only accepts against a subquery.
    """
    sql = " OR ".join("(embedder_id=? AND commit_id=?)" for _ in keys)
    params: list = []
    for k in keys:
        params.extend(k)
    return f"({sql})", params


class CommitStore:
    """SQLite-backed store for commit metadata and processing state.

    One db holds one bot: ``bot`` is an informational column on the row tables,
    never a key and never filtered on; the only writer is this machine's own
    bench session.

    Rows are keyed by ``CommitKey``. Every method that takes a key accepts
    the scalar spelling too (an int is embedder 0), because that is what the
    CLI and the perf database speak; what comes back is always a
    ``CommitKey``.
    """

    def __init__(
        self,
        db_path: Path,
        readonly: bool = False,
        *,
        backup: bool | None = None,
        init_schema: bool = True,
        bot: str | None = None,
        busy_timeout_ms: int = 30000,
        should_stop=lambda: False,
    ):
        """Open a connection to the store.

        ``backup`` defaults to ``not readonly``. Set it False for secondary
        connections (e.g. delivery) that must not snapshot the
        db again. ``init_schema`` may be disabled when the caller knows the
        schema already exists, to avoid re-running migrations concurrently
        from a second connection; ``readonly`` disables it too, so reporting
        commands read what is there rather than writing to it.

        ``bot`` names this machine. It is recorded in ``meta.local_bot`` on the
        first open that has one, and a later open under a different name is
        refused.
        """
        self._db_path = db_path.resolve()
        self._result_locks = {}
        self._bot = bot
        self._should_stop = should_stop
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
        self.conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        self.conn.execute("PRAGMA synchronous=FULL")
        # WAL lets delivery's reader coexist with the collector's
        # writer instead of deadlocking on lock promotion. It is a persistent
        # property of the db file, so setting it on any connection converts it.
        self.conn.execute("PRAGMA journal_mode=WAL")
        try:
            # Before the backup and before any schema work: a db from a newer
            # slipstream is refused whole, readonly or not, since even its
            # reads may name columns this version does not know.
            if pre_existing:
                self._check_schema_version()
            if backup and pre_existing:
                # Snapshot before _init_schema so the backup predates any
                # migration.
                self._backup(self.conn, db_path)
            # A reporting command reads what is there rather than migrating a
            # live db on every run. A db that does not exist yet has nothing
            # to read, so it is created either way: otherwise the first
            # readonly command on a fresh machine fails on a missing table
            # instead of reporting nothing.
            if init_schema and (not readonly or not pre_existing):
                self._init_schema()
            elif bot is not None:
                self._check_bot()
        except BaseException:
            self.conn.close()
            raise

    def _check_schema_version(self) -> None:
        """Refuse a db migrated past this version's schema.

        A db without a meta table or a version row predates the number and
        is older, not newer; the migration handles it. A value that is not a
        number is left to the migration too, which overwrites it.
        """
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
        except sqlite3.OperationalError:
            return  # no meta table: older than any version number
        if row is None:
            return
        try:
            found = int(row[0])
        except (TypeError, ValueError):
            return
        if found > int(SCHEMA_VERSION):
            raise SchemaTooNew(
                f"{self._db_path} has schema version {found}, but this "
                f"slipstream reads version {SCHEMA_VERSION}. Migrations are "
                f"one-way: upgrade slipstream, or restore the .bak snapshot "
                f"taken before the migration alongside the db."
            )

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
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS delivery_attempts (
                source TEXT PRIMARY KEY, record TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedders (
                engine     TEXT    NOT NULL,
                hash       TEXT    NOT NULL,
                number     INTEGER NOT NULL,
                title      TEXT    NOT NULL DEFAULT '',
                first_seen INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (engine, hash),
                UNIQUE (engine, number)
            );
            """
            + ";\n".join(_KEYED_TABLES.values())
            + ";\n"
        )
        # Migrate scores: UNIQUE constraint -> PRIMARY KEY (idempotent). Older
        # than the key rebuild below, which takes this table's shape from it.
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
            if self._should_stop():
                raise InterruptedError("store migration cancelled")
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
                until = time.monotonic() + _MIGRATE_RETRY_SECS
                while time.monotonic() < until:
                    if self._should_stop():
                        raise InterruptedError("store migration retry cancelled")
                    time.sleep(min(0.05, max(0, until - time.monotonic())))

    def _run_migration(self):
        for table, column, decl in _ADDED_COLUMNS:
            self._add_column(table, column, decl)
        # After the ALTERs: the rebuild copies every column the old table has,
        # so the ones added above must be there first or the new table would
        # be created without them and the ALTER never run again.
        for table in _KEYED_TABLES:
            if "embedder_id" not in self._columns(table):
                self._rebuild_with_embedder(table)
        for name in _OLD_INDEXES:
            self.conn.execute(f"DROP INDEX IF EXISTS {name}")
        for ddl in _INDEXES.values():
            self.conn.execute(ddl)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
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

    def _columns(self, table: str) -> list[str]:
        return [r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")]

    def _rebuild_with_embedder(self, table: str) -> None:
        """Recreate ``table`` with embedder_id in its key, keeping every row.

        Every existing row is embedder 0 by definition -- nothing wrote any
        other kind before the column existed -- so the copy names only the
        old columns and the new one takes its default. The old table's
        indexes go with it when it is dropped.
        """
        old = f"_{table}_old"
        cols = ", ".join(self._columns(table))
        self.conn.executescript(
            f"""
            ALTER TABLE {table} RENAME TO {old};
            {_KEYED_TABLES[table]};
            INSERT INTO {table} ({cols}) SELECT {cols} FROM {old};
            DROP TABLE {old};
            """
        )

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
    #
    # insert_commits, count_missing_metadata, update_commit_metadata and
    # get_commits_with_metadata serve the git-driven path, which resolves
    # hashes in the engine's own checkout: embedder 0 by construction. An
    # embedded engine's rows arrive through record_done with a full key.

    @_write
    def insert_commits(self, engine: str, commits: list[dict]):
        """Insert (hash, title) rows; no-op if already present."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO commits (engine, embedder_id, hash, title)"
            " VALUES (?,0,?,?)",
            [(engine, c["hash"], c.get("title", "")) for c in commits],
        )
        self.conn.commit()

    # --- Commit metadata ---

    def count_missing_metadata(self, engine: str, hashes: list[str]) -> int:
        placeholders = ",".join("?" * len(hashes))
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM commits"
            f" WHERE engine=? AND embedder_id=0 AND hash IN ({placeholders})"
            f" AND commit_id IS NULL",
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
               WHERE engine=? AND embedder_id=0 AND hash=? AND commit_id IS NULL""",
            (commit_id, date, timestamp, title, engine, hash),
        )

    @_write
    def upsert_commit(
        self,
        engine: str,
        hash: str,
        key,
        date: str,
        timestamp: int,
        title: str,
        embedder_hash: str = "",
    ):
        """Write commit metadata that did not come from a local git log.

        The bus carries it in the entry, so the row must be written even for a
        commit this machine has never resolved: ``export_scores`` inner-joins
        ``commits``, and ``push()`` marks a candidate pushed whether or not it
        exported rows, so a done commit with no row here loses its scores for
        good.

        Not INSERT OR REPLACE: ``commits`` carries a partial unique index on
        (engine, embedder_id, commit_id), and replace through it deletes the
        incumbent row when another hash holds that key, orphaning that
        commit's scores from the export join.
        """
        self._upsert_commit(
            engine, hash, CommitKey.of(key), date, timestamp, title, embedder_hash
        )
        self.conn.commit()

    def _upsert_commit(
        self,
        engine: str,
        hash: str,
        key: CommitKey,
        date: str,
        timestamp: int,
        title: str,
        embedder_hash: str = "",
    ):
        try:
            self.conn.execute(
                "INSERT INTO commits"
                " (engine, embedder_id, hash, commit_id, date, timestamp, title,"
                "  embedder_hash)"
                " VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(engine, embedder_id, hash) DO UPDATE SET"
                "   commit_id=excluded.commit_id, date=excluded.date,"
                "   timestamp=excluded.timestamp, title=excluded.title,"
                "   embedder_hash=excluded.embedder_hash",
                (
                    engine,
                    key.embedder_id,
                    hash,
                    key.commit_id,
                    date,
                    timestamp,
                    title,
                    embedder_hash or "",
                ),
            )
        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            other = self.conn.execute(
                "SELECT hash FROM commits"
                " WHERE engine=? AND embedder_id=? AND commit_id=?",
                (engine, *key),
            ).fetchone()
            raise CommitIdCollision(
                f"{engine} {key} is already held by "
                f"{other[0] if other else '?'}, not {hash}"
            ) from e

    def commit_ids_missing_commit_row(
        self,
        engine: str,
        platform: str,
        lo: int | None = None,
        hi: int | None = None,
        *,
        embedder_id: int = 0,
    ) -> list[int]:
        """Commit ids with scores but no row in ``commits``.

        Their scores can never be exported (the export inner-joins ``commits``)
        so they must not be marked pushed, and the count is worth surfacing:
        it means a bench wrote scores without the metadata that goes with them.

        ``lo``/``hi`` bound the scan to a commit range. The background pusher
        runs this after every benched commit and is meant to be cheap, and
        unbounded it is a DISTINCT LEFT JOIN over the engine's whole history.
        A range rather than an id list, so the caller's set can be any size.

        One embedder at a time: the git-driven frontier asks about its own
        series (embedder 0); delivery asks about the key it is shipping.
        """
        rows = self.conn.execute(
            "SELECT DISTINCT s.commit_id FROM scores s"
            " LEFT JOIN commits c ON s.engine=c.engine"
            "   AND s.embedder_id=c.embedder_id AND s.commit_id=c.commit_id"
            " WHERE s.engine=? AND s.platform=? AND s.embedder_id=? AND c.hash IS NULL"
            "   AND (? IS NULL OR s.commit_id >= ?)"
            "   AND (? IS NULL OR s.commit_id <= ?)"
            " ORDER BY s.commit_id",
            (engine, platform, embedder_id, lo, lo, hi, hi),
        ).fetchall()
        return [r[0] for r in rows]

    def key_missing_commit_row(self, engine: str, platform: str, key) -> bool:
        """Whether one key has scores but no ``commits`` row (see above)."""
        key = CommitKey.of(key)
        return bool(
            self.commit_ids_missing_commit_row(
                engine,
                platform,
                key.commit_id,
                key.commit_id,
                embedder_id=key.embedder_id,
            )
        )

    def get_commits_with_metadata(
        self, engine: str, hashes: list[str]
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(hashes))
        return self.conn.execute(
            f"SELECT hash, embedder_id, commit_id, date, timestamp, title, embedder_hash FROM commits"
            f" WHERE engine=? AND embedder_id=0 AND hash IN ({placeholders})"
            f" AND commit_id IS NOT NULL"
            f" ORDER BY commit_id",
            [engine, *hashes],
        ).fetchall()

    # --- Processing state ---

    def is_done(self, engine: str, platform: str, key) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM processing_state"
            " WHERE engine=? AND platform=? AND embedder_id=? AND commit_id=?",
            (engine, platform, *CommitKey.of(key)),
        ).fetchone()
        return row is not None

    @_write
    def mark_done(
        self,
        engine: str,
        platform: str,
        key,
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
        self._mark_done(
            engine, platform, CommitKey.of(key), status, configs_ok, configs_total
        )
        self.conn.commit()

    def _mark_done(
        self,
        engine: str,
        platform: str,
        key: CommitKey,
        status: str | None,
        configs_ok: int | None,
        configs_total: int | None,
    ):
        self.conn.execute(
            "INSERT INTO processing_state"
            " (engine, platform, embedder_id, commit_id, bot, status,"
            "  configs_ok, configs_total)"
            " VALUES (:engine, :platform, :embedder_id, :commit_id, :bot,"
            "         COALESCE(:status, 'ok'), :configs_ok, :configs_total)"
            " ON CONFLICT(engine, platform, embedder_id, commit_id) DO UPDATE SET"
            "   status        = COALESCE(:status, processing_state.status),"
            "   configs_ok    = COALESCE(:configs_ok, processing_state.configs_ok),"
            "   configs_total = COALESCE(:configs_total, processing_state.configs_total),"
            "   bot           = COALESCE(processing_state.bot, :bot)",
            {
                "engine": engine,
                "platform": platform,
                "embedder_id": key.embedder_id,
                "commit_id": key.commit_id,
                "bot": self._bot,
                "status": status,
                "configs_ok": configs_ok,
                "configs_total": configs_total,
            },
        )

    @_write
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
        key = CommitKey.from_commit(commit)
        self._upsert_commit(
            engine,
            commit["hash"],
            key,
            commit.get("date", ""),
            int(commit.get("timestamp", 0)),
            commit.get("title", ""),
            commit.get("embedder_hash", ""),
        )
        self._mark_done(
            engine,
            platform,
            key,
            status,
            configs_ok,
            configs_total,
        )
        self.conn.commit()

    def get_status(self, engine: str, platform: str, key) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM processing_state"
            " WHERE engine=? AND platform=? AND embedder_id=? AND commit_id=?",
            (engine, platform, *CommitKey.of(key)),
        ).fetchone()
        return row[0] if row else None

    def status_counts(self, engine: str, platform: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) FROM processing_state"
            " WHERE engine=? AND platform=? GROUP BY status",
            (engine, platform),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    @property
    def db_path(self) -> Path:
        return self._db_path

    @contextlib.contextmanager
    def result_locks(self, engine, platform, keys, **kwargs):
        from .durability import FileLock, identity_digest

        with contextlib.ExitStack() as stack:
            for key in sorted({CommitKey.of(k) for k in keys}):
                ident = (engine, platform, key)
                if ident in self._result_locks:
                    continue
                path = self._db_path.parent / ("." + self._db_path.name + ".locks")
                # to_json, so a scalar key digests to the same lock file it
                # always did: delivery takes these too, by commit number.
                stack.enter_context(
                    FileLock(
                        path
                        / (
                            identity_digest(
                                json.dumps((engine, platform, key.to_json()))
                            )
                            + ".lock"
                        ),
                        **kwargs,
                    )
                )
                self._result_locks[ident] = True
                stack.callback(self._result_locks.pop, ident)
            yield

    def pending_attempt(self, source):
        row = self.conn.execute(
            "SELECT record FROM delivery_attempts WHERE source=?", (source,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    @_write
    def save_attempt(self, source, record):
        self.conn.execute(
            "INSERT INTO delivery_attempts VALUES (?, ?)",
            (source, json.dumps(record, sort_keys=True)),
        )
        self.conn.commit()

    @_write
    def retire_attempt(self, source):
        self.conn.execute("DELETE FROM delivery_attempts WHERE source=?", (source,))
        self.conn.commit()

    def check_pending(self, engine, platform, keys):
        wanted = {CommitKey.of(k) for k in keys}
        for row in self.conn.execute("SELECT record FROM delivery_attempts"):
            record = json.loads(row[0])
            if (
                record["engine"] == engine
                and record["platform"] == platform
                and CommitKey.of(record["unit"]) in wanted
            ):
                raise StoreError(
                    f"unresolved delivery attempt {record['attempt']} protects "
                    f"{engine}/{platform}/{record['unit']}; retry delivery first"
                )

    def clear_scores(self, engine: str, platform: str, keys: list):
        with self.result_locks(engine, platform, keys):
            self.check_pending(engine, platform, keys)
            self._clear_scores(engine, platform, keys)

    @_write
    def _clear_scores(self, engine: str, platform: str, keys: list):
        """Delete only the scores of commits, leaving their state alone.

        For resuming an interrupted commit: ``scores`` is INSERT OR IGNORE with
        ``run`` in the primary key, so a retry that kept the partial rows would
        leave one run number half measured on each side of the interrupt.
        """
        if not keys:
            return
        where, params = _key_filter([CommitKey.of(k) for k in keys])
        self.conn.execute(
            f"DELETE FROM scores WHERE engine=? AND platform=? AND {where}",
            [engine, platform, *params],
        )
        self.conn.commit()

    def clear_range(self, engine: str, platform: str, keys: list):
        with self.result_locks(engine, platform, keys):
            self.check_pending(engine, platform, keys)
            self._clear_range(engine, platform, keys)

    @_write
    def _clear_range(self, engine: str, platform: str, keys: list):
        """Delete scores, processing state, and push state for specific commits.

        Clearing push_state too means any re-benchmarked commit will be
        re-pushed on the next push cycle.
        """
        if not keys:
            return
        where, params = _key_filter([CommitKey.of(k) for k in keys])
        for table in ("scores", "processing_state", "push_state"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE engine=? AND platform=? AND {where}",
                [engine, platform, *params],
            )
        self.conn.execute(
            f"DELETE FROM run_env WHERE engine=? AND {where}",
            [engine, *params],
        )
        self.conn.commit()

    def get_all_commits(self, engine: str) -> list[sqlite3.Row]:
        """All commits with metadata, in key order."""
        return self.conn.execute(
            "SELECT hash, embedder_id, commit_id, date, timestamp, title, embedder_hash FROM commits"
            " WHERE engine=? AND commit_id IS NOT NULL"
            " ORDER BY embedder_id, commit_id",
            (engine,),
        ).fetchall()

    def get_commits_in_range(self, engine: str, after, up_to) -> list[sqlite3.Row]:
        """All commits with after < key <= up_to, in key order."""
        after = CommitKey.of(after)
        up_to = CommitKey.of(up_to)
        return self.conn.execute(
            "SELECT hash, embedder_id, commit_id, date, timestamp, title, embedder_hash FROM commits"
            " WHERE engine=? AND commit_id IS NOT NULL"
            " AND (embedder_id, commit_id) > (?, ?)"
            " AND (embedder_id, commit_id) <= (?, ?)"
            " ORDER BY embedder_id, commit_id",
            (engine, *after, *up_to),
        ).fetchall()

    # --- Scores ---

    @_write
    def insert_scores(
        self,
        engine: str,
        platform: str,
        key,
        timestamp: int,
        scores: list[dict],
    ):
        """Insert raw score rows. Each dict: {suite, flags, benchmark, metric, run, score}."""
        key = CommitKey.of(key)
        self.conn.executemany(
            "INSERT OR IGNORE INTO scores (engine, platform, embedder_id, commit_id,"
            " suite, flags, benchmark, metric, run, score, timestamp, bot)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    engine,
                    platform,
                    key.embedder_id,
                    key.commit_id,
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

    def get_series(
        self,
        engine: str,
        suite: str,
        flags: str,
        benchmark: str,
        metric: str,
    ) -> list[sqlite3.Row]:
        """Return (embedder_id, commit_id, score) rows in key order for one series."""
        return self.conn.execute(
            "SELECT embedder_id, commit_id, score FROM scores"
            " WHERE engine=? AND suite=? AND flags=? AND benchmark=? AND metric=?"
            " ORDER BY embedder_id, commit_id",
            (engine, suite, flags, benchmark, metric),
        ).fetchall()

    def export_scores(
        self,
        engine: str,
        platform: str,
        valid_benchmarks_by_suite: dict[str, set[str]],
        keys: list | None = None,
    ) -> list[sqlite3.Row]:
        """Score rows joined with commit metadata, for pushing.

        Benchmarks are filtered per suite so that e.g. js2-only names don't
        leak into js3. ``keys`` restricts the export to those commit keys; it
        is materialised in a temp table to sidestep SQLite's bound-parameter
        limit.

        Rows come out in spool column order (``delivery_batch.COLUMNS``):
        the engine's commit as ``commit_id`` and the embedder pair last, so
        an embedded series travels with both of its coordinates.
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
        if keys is not None:
            if not keys:
                return []
            self.conn.execute("DROP TABLE IF EXISTS _export_commits")
            self.conn.execute(
                "CREATE TEMP TABLE _export_commits"
                " (embedder_id INTEGER NOT NULL, commit_id INTEGER NOT NULL,"
                "  PRIMARY KEY (embedder_id, commit_id))"
            )
            self.conn.executemany(
                "INSERT INTO _export_commits (embedder_id, commit_id) VALUES (?,?)",
                [tuple(CommitKey.of(k)) for k in keys],
            )
            commit_filter = (
                " AND EXISTS (SELECT 1 FROM _export_commits x"
                "  WHERE x.embedder_id = s.embedder_id AND x.commit_id = s.commit_id)"
            )

        try:
            return self.conn.execute(
                "SELECT s.engine, s.platform, s.commit_id, s.suite, s.flags,"
                "       s.benchmark, s.metric, s.run, s.score, s.timestamp,"
                "       c.hash, c.date, c.timestamp, c.title,"
                "       s.embedder_id, c.embedder_hash"
                " FROM scores s"
                " JOIN commits c ON s.engine = c.engine"
                "   AND s.embedder_id = c.embedder_id AND s.commit_id = c.commit_id"
                f" WHERE s.engine = ? AND s.platform = ?"
                f"   AND ({where_suite})"
                f"{commit_filter}"
                " ORDER BY s.embedder_id, s.commit_id, s.suite, s.flags,"
                "          s.benchmark, s.metric, s.run",
                params,
            ).fetchall()
        finally:
            if commit_filter:
                self.conn.execute("DROP TABLE IF EXISTS _export_commits")

    # --- Provenance ---

    @_write
    def record_run_env(self, engine: str, key, env: dict) -> None:
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
        key = CommitKey.of(key)
        # runs is the only nullable column: the rest are NOT NULL with a text
        # default, so a caller that knows less than all of it still writes a row.
        defaults = {
            "source": "local",
            "runs": None,
            "run_configs": "[]",
            "harness_revs": "{}",
            "host_env": "{}",
            "suite_cfg_hash": "{}",
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
            "runner_cfg_hash",
            "host_env",
            "suite_cfg_hash",
            "slipstream_version",
        ]
        values = [env.get(c, defaults.get(c, "")) for c in columns]
        values = [
            defaults.get(c, "") if v is None and c != "runs" else v
            for c, v in zip(columns, values)
        ]
        assignments = ", ".join(f"{c}=excluded.{c}" for c in columns)
        self.conn.execute(
            f"INSERT INTO run_env (engine, bot, embedder_id, commit_id,"
            f" {', '.join(columns)}, recorded_at)"
            f" VALUES (?,?,?,?,{','.join('?' * len(columns))},?)"
            f" ON CONFLICT(engine, bot, embedder_id, commit_id) DO UPDATE SET"
            f" {assignments}, recorded_at=excluded.recorded_at",
            (engine, self._bot or "", *key, *values, int(time.time())),
        )
        self.conn.commit()

    def get_run_env(self, engine: str, key) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM run_env"
            " WHERE engine=? AND bot=? AND embedder_id=? AND commit_id=?",
            (engine, self._bot or "", *CommitKey.of(key)),
        ).fetchone()

    def run_env_source_counts(self, engine: str, last: int = 50) -> dict[str, int]:
        """How the most recent commits' binaries were produced."""
        rows = self.conn.execute(
            "SELECT source, COUNT(*) FROM ("
            "  SELECT source FROM run_env WHERE engine=?"
            "  ORDER BY embedder_id DESC, commit_id DESC LIMIT ?"
            ") GROUP BY source",
            (engine, last),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # --- Build state ---
    #
    # Load-bearing local state on the builder, not a report: a recorded
    # terminal failure is what stops a commit that cannot be built from being
    # rebuilt every cycle forever, and what lets the frontier advance past it.

    @_write
    def record_build_failure(
        self, engine: str, key, status: str, kind: str, log_path: str = ""
    ) -> int:
        """Record a failed build attempt; returns the new attempt count."""
        key = CommitKey.of(key)
        row = self.conn.execute(
            "SELECT attempts FROM build_state"
            " WHERE engine=? AND embedder_id=? AND commit_id=?",
            (engine, *key),
        ).fetchone()
        attempts = (row[0] if row else 0) + 1
        self.conn.execute(
            "INSERT INTO build_state"
            " (engine, embedder_id, commit_id, status, kind, log_path, attempts,"
            "  last_attempt)"
            " VALUES (?,?,?,?,?,?,?,?)"
            " ON CONFLICT(engine, embedder_id, commit_id) DO UPDATE SET"
            "   status=excluded.status, kind=excluded.kind,"
            "   log_path=excluded.log_path, attempts=excluded.attempts,"
            "   last_attempt=excluded.last_attempt",
            (engine, *key, status, kind, log_path, attempts, int(time.time())),
        )
        self.conn.commit()
        return attempts

    @_write
    def set_build_status(self, engine: str, key, status: str) -> None:
        """Reclassify an existing failure without counting another attempt."""
        self.conn.execute(
            "UPDATE build_state SET status=?"
            " WHERE engine=? AND embedder_id=? AND commit_id=?",
            (status, engine, *CommitKey.of(key)),
        )
        self.conn.commit()

    @_write
    def request_build_retry(self, engine: str, key) -> bool:
        """Allow one more attempt at a commit the frontier has moved past.

        Persisted rather than an effect of the command: entries above the
        failed commit exist by the time anyone retries, so deleting the row
        would leave the builder resolving "next commit above the frontier" and
        never picking this one up.
        """
        cur = self.conn.execute(
            "UPDATE build_state SET status='retry_requested'"
            " WHERE engine=? AND embedder_id=? AND commit_id=?",
            (engine, *CommitKey.of(key)),
        )
        self.conn.commit()
        return cur.rowcount > 0

    @_write
    def clear_build_state(self, engine: str, key) -> None:
        self.conn.execute(
            "DELETE FROM build_state WHERE engine=? AND embedder_id=? AND commit_id=?",
            (engine, *CommitKey.of(key)),
        )
        self.conn.commit()

    def get_build_state(self, engine: str, key) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM build_state WHERE engine=? AND embedder_id=? AND commit_id=?",
            (engine, *CommitKey.of(key)),
        ).fetchone()

    def build_retries_requested(self, engine: str) -> list[CommitKey]:
        rows = self.conn.execute(
            "SELECT embedder_id, commit_id FROM build_state"
            " WHERE engine=? AND status='retry_requested'"
            " ORDER BY embedder_id, commit_id",
            (engine,),
        ).fetchall()
        return [CommitKey.from_commit(r) for r in rows]

    # --- builder-assigned embedder numbers ---
    #
    # An embedder with no position of its own (Safari Technology Preview: its
    # Info.plist has a build string but no release number) gets one here:
    # the first distinct hash is 1, the next 2, in the order the builder met
    # them. Never 0, which is what "no embedder" is everywhere else.

    def embedder_number(self, engine: str, hash_: str, title: str = "") -> int:
        """The number for ``hash_``, assigning the next one if it is new."""
        row = self.conn.execute(
            "SELECT number FROM embedders WHERE engine=? AND hash=?", (engine, hash_)
        ).fetchone()
        if row:
            return int(row[0])
        with self.conn:
            top = self.conn.execute(
                "SELECT COALESCE(MAX(number), 0) FROM embedders WHERE engine=?",
                (engine,),
            ).fetchone()[0]
            number = int(top) + 1
            self.conn.execute(
                "INSERT INTO embedders (engine, hash, number, title, first_seen)"
                " VALUES (?, ?, ?, ?, ?)",
                (engine, hash_, number, title, int(time.time())),
            )
        return number

    def embedder_numbers(self, engine: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT hash, number FROM embedders WHERE engine=?", (engine,)
        ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def seed_embedder_numbers(self, engine: str, known: dict[str, int]) -> None:
        """Record pairs this store has no row for, e.g. read back from the
        published manifests after the builder's database was lost. A pair
        that contradicts an existing row is an error, not a silent renumber."""
        with self.conn:
            for hash_, number in known.items():
                row = self.conn.execute(
                    "SELECT number FROM embedders WHERE engine=? AND hash=?",
                    (engine, hash_),
                ).fetchone()
                if row is None:
                    self.conn.execute(
                        "INSERT INTO embedders (engine, hash, number, first_seen)"
                        " VALUES (?, ?, ?, ?)",
                        (engine, hash_, int(number), int(time.time())),
                    )
                elif int(row[0]) != int(number):
                    raise StoreError(
                        f"{engine}: embedder {hash_!r} is number {row[0]} here "
                        f"but {number} on the bus"
                    )

    def max_terminal_build_key(self, engine: str) -> CommitKey | None:
        """Highest commit with a terminal build failure.

        Non-terminal rows are excluded by construction: including them would
        advance the frontier past exactly the commit the retry rule says must
        be attempted again, and the retry would never happen. It also keeps a
        crash between the payload rename and the entry rename self-healing,
        since that leaves a non-terminal row.
        """
        row = self.conn.execute(
            "SELECT embedder_id, commit_id FROM build_state"
            " WHERE engine=? AND status IN ('compile_failed', 'infra_burned')"
            " ORDER BY embedder_id DESC, commit_id DESC LIMIT 1",
            (engine,),
        ).fetchone()
        return CommitKey.from_commit(row) if row else None

    def build_failures(self, engine: str, above=None) -> list[sqlite3.Row]:
        """Terminal failures, for the state file and the circuit breaker."""
        above = CommitKey.of(above) if above is not None else None
        return self.conn.execute(
            "SELECT embedder_id, commit_id, status, kind, log_path, attempts,"
            "       last_attempt"
            " FROM build_state"
            " WHERE engine=? AND status IN ('compile_failed', 'infra_burned')"
            "   AND (? IS NULL OR (embedder_id, commit_id) > (?, ?))"
            " ORDER BY embedder_id, commit_id",
            (
                engine,
                above.commit_id if above else None,
                above.embedder_id if above else None,
                above.commit_id if above else None,
            ),
        ).fetchall()

    # --- Push state ---

    def unpushed_keys(
        self, engine: str, platform: str, limit: int | None = None
    ) -> list[CommitKey]:
        """Keys that are done but not yet pushed, in CommitKey order.

        Every embedder: an embedded series is delivered under its own pair,
        and the perf database keys on the same pair.
        """
        rows = self.conn.execute(
            "SELECT ps.embedder_id, ps.commit_id FROM processing_state ps"
            " LEFT JOIN push_state pu"
            "   ON pu.engine=ps.engine AND pu.platform=ps.platform"
            "  AND pu.embedder_id=ps.embedder_id AND pu.commit_id=ps.commit_id"
            " WHERE ps.engine=? AND ps.platform=?"
            "   AND pu.commit_id IS NULL"
            " ORDER BY ps.embedder_id, ps.commit_id LIMIT ?",
            (engine, platform, limit if limit is not None else -1),
        ).fetchall()
        return [CommitKey(r[0], r[1]) for r in rows]

    def is_pushed(self, engine: str, platform: str, key) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM push_state WHERE engine=? AND platform=?"
            " AND embedder_id=? AND commit_id=?",
            (engine, platform, *CommitKey.of(key)),
        ).fetchone()
        return row is not None

    @_write
    def mark_pushed(self, engine: str, platform: str, keys: list) -> None:
        """Mark keys as successfully pushed to Spanner."""
        if not keys:
            return
        now = int(time.time())
        self.conn.executemany(
            "INSERT OR REPLACE INTO push_state"
            " (engine, platform, embedder_id, commit_id, pushed_at, bot)"
            " VALUES (?,?,?,?,?,?)",
            [(engine, platform, *CommitKey.of(k), now, self._bot) for k in keys],
        )
        self.conn.commit()

    @_write
    def clear_push_state(self, engine: str, platform: str) -> None:
        """Forget every push, so the next push re-sends the full history."""
        self.conn.execute(
            "DELETE FROM push_state WHERE engine=? AND platform=?", (engine, platform)
        )
        self.conn.commit()

    def max_done_key(
        self, engine: str, platform: str, *, embedder_id: int | None = None
    ) -> CommitKey | None:
        """The highest key processed on this platform, or None.

        ``embedder_id`` restricts to one embedder's rows: the git-driven
        frontier is a scalar id and must not read a key from another series
        as its own.
        """
        row = self.conn.execute(
            "SELECT embedder_id, commit_id FROM processing_state"
            " WHERE engine=? AND platform=? AND (? IS NULL OR embedder_id=?)"
            " ORDER BY embedder_id DESC, commit_id DESC LIMIT 1",
            (engine, platform, embedder_id, embedder_id),
        ).fetchone()
        return CommitKey.from_commit(row) if row else None

    def close(self):
        self.conn.close()
