# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Source progress stays separate from destination acceptance."""

from __future__ import annotations

import json
from pathlib import Path

from .delivery_batch import Batch
from .durability import atomic_write, durable_unlink, identity_digest
from .relay import RemoteSpool, cursor_path, read_cursor, write_cursor


class LocalDbSource:
    def __init__(self, store, engine, platform, bot, valid):
        self.store, self.engine, self.platform = store, engine, platform
        if store.bot and store.bot != bot:
            raise ValueError("delivery bot differs from local DB bot")
        self.bot, self.valid = bot, valid
        self.identity = f"local:{store.db_path}:{engine}:{platform}:{bot}"

    def pending(self):
        record = self.store.pending_attempt(self.identity)
        if record:
            return record
        for row in self.store.conn.execute("SELECT record FROM delivery_attempts"):
            other = json.loads(row[0])
            if other["engine"] == self.engine and other["platform"] == self.platform:
                return other  # validation blocks changed source/bot identities
        return None

    def acknowledged(self, unit):
        row = self.store.conn.execute(
            "SELECT 1 FROM push_state WHERE engine=? AND platform=? "
            "AND embedder_id=0 AND commit_id=?",
            (self.engine, self.platform, unit),
        ).fetchone()
        return row is not None

    def discover(self, limit):
        return self.store.unpushed_commit_ids(self.engine, self.platform, limit)

    def prepare(self, unit, settings):
        self.store.conn.execute(
            f"PRAGMA busy_timeout={max(1, int(min(1, settings.shutdown_seconds) * 1000))}"
        )
        from .push import _export_rows

        if not self.store.is_done(
            self.engine, self.platform, unit
        ) or self.acknowledged(unit):
            return None
        if self.store.commit_ids_missing_commit_row(
            self.engine, self.platform, unit, unit
        ):
            raise ValueError(f"{self.identity}/{unit}: scores have no commit row")
        # Hard bound even before materializing score rows. SQLite length counts
        # characters; budget four bytes per character plus CSV escaping overhead.
        size = self.store.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(length(suite)+length(flags)+length(benchmark)"
            "+length(metric)+length(engine)+length(platform)+160), 0) FROM scores "
            "WHERE engine=? AND platform=? AND embedder_id=0 AND commit_id=?",
            (self.engine, self.platform, unit),
        ).fetchone()
        meta = self.store.conn.execute(
            "SELECT length(hash)+length(date)+length(title)+32 FROM commits "
            "WHERE engine=? AND embedder_id=0 AND commit_id=?",
            (self.engine, unit),
        ).fetchone()
        if (
            size[1] + size[0] * (meta[0] if meta else 0)
        ) * 8 > settings.max_payload_bytes:
            raise ValueError(
                f"{self.identity}/{unit}: estimated payload exceeds max_payload_bytes"
            )
        rows = _export_rows(self.store, self.engine, self.platform, self.valid, [unit])
        self.store.conn.commit()  # no SQLite transaction crosses target I/O
        return Batch.local(self.identity, self.bot, unit, rows)

    def save(self, record):
        record = dict(record, engine=self.engine, platform=self.platform)
        self.store.save_attempt(self.identity, record)

    def retire(self):
        self.store.retire_attempt(self.identity)

    def acknowledge(self, batch):
        self.store.mark_pushed(self.engine, self.platform, [batch.unit])


class SshSpoolSource:
    def __init__(self, source, *, settings, should_stop=lambda: False, spool=None):
        self.source = source
        self.bot = source.bot_name
        self.path = cursor_path(source)
        self.identity = f"ssh:{source.ssh_host}:{source.spool_dir}:{source.bot_name}:{self.path.resolve()}"
        self.attempt_path = self.path.with_suffix(".attempt.json")
        self.spool = spool or RemoteSpool(
            source.ssh_host,
            source.spool_dir,
            timeout=settings.io_seconds,
            max_bytes=settings.max_payload_bytes,
            should_stop=should_stop,
        )

    def pending(self):
        try:
            record = json.loads(self.attempt_path.read_text())
        except FileNotFoundError:
            return None
        if not isinstance(record, dict):
            raise ValueError(f"corrupt attempt metadata {self.attempt_path}")
        return record

    def acknowledged(self, unit):
        return read_cursor(self.path) >= unit

    def discover(self, limit):
        cursor = read_cursor(self.path)
        seqs = self.spool.list(cursor=cursor, limit=limit + 1)
        if cursor > max(seqs, default=0):
            raise ValueError(
                f"cursor {cursor} is past newest entry {max(seqs, default=0)}; spool reset or missing allocator"
            )
        pending = [s for s in seqs if s > cursor]
        selected = []
        for seq in pending[:limit]:
            want = cursor + 1 + len(selected)
            if seq != want:
                if selected:
                    break  # next cycle reports gap at the now-contiguous cursor
                raise ValueError(
                    f"gap: expected {want}, found {seq}; use deliver --reset-cursor explicitly"
                )
            selected.append(seq)
        return selected

    def prepare(self, unit, settings):
        return Batch.remote(self.identity, self.bot, unit, self.spool.fetch(unit))

    def save(self, record):
        atomic_write(self.attempt_path, json.dumps(record, sort_keys=True))

    def retire(self):
        durable_unlink(self.attempt_path)

    def acknowledge(self, batch):
        cursor = read_cursor(self.path)
        if batch.unit != cursor + 1:
            raise ValueError(f"non-contiguous acknowledgement {cursor} -> {batch.unit}")
        write_cursor(self.path, batch.unit)


def validate_record(record, source, targets):
    import re

    if (
        not isinstance(record, dict)
        or type(record.get("version")) is not int
        or record["version"] != 1
        or record.get("source") != source.identity
        or record.get("targets") != targets
        or record.get("bot") != source.bot
        or type(record.get("unit")) is not int
        or record["unit"] < 0
        or not isinstance(record.get("targets"), list)
        or not record["targets"]
        or any(not isinstance(t, str) for t in record["targets"])
        or not isinstance(record.get("attempt"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", record["attempt"])
        or not isinstance(record.get("digest"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", record["digest"])
    ):
        raise ValueError(
            f"{source.identity}: pending attempt identity/target set changed or corrupt; explicit recovery required"
        )


def owner_path(state_dir: Path):
    return state_dir / "owner.lock"


def cursor_lock_path(source):
    # Shared across state directories using this same historical cursor.
    return source.path.parent / (
        identity_digest(str(source.path.resolve())) + ".owner.lock"
    )
