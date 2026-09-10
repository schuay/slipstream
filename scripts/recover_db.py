#!/usr/bin/env python3
# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Recover a slipstream.db after a `database is locked` crash and convert it to WAL.

Background: `watch` ran two writer connections (the collector and the background
pusher) against a rollback-journal db. The pusher held a read snapshot across the
multi-second push subprocess, deadlocking the collector's commit; SQLite returned
"database is locked" immediately, crashing watch. The code fix moves the db to WAL
and shortens the pusher's transaction. This script brings an already-crashed
machine's db to a clean, converted state before the fixed `watch` is restarted.

It is safe to run repeatedly. It does NOT touch scores or push_state beyond what
SQLite's own recovery does: a partial transaction from the crash is rolled back on
open, un-pushed commits stay unpushed (and re-push idempotently), and un-benchmarked
commits stay un-benchmarked (and re-run under INSERT OR IGNORE).

Usage:
    python scripts/recover_db.py [--db PATH] [--yes]

--db defaults to the standard location under the user config's metadata_dir if
slipstream is importable, else ~/.local/share/slipstream/metadata/slipstream.db.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path

# User config location (mirrors slipstream.config._DEFAULT_CONFIG_PATH). The db
# lives at <out_dir>/metadata/slipstream.db, where out_dir comes from this file.
CONFIG_PATH = Path.home() / ".config" / "slipstream" / "config.toml"


def out_dir_from_config() -> Path | None:
    """Read out_dir straight from the user config TOML.

    Done with stdlib tomllib rather than importing slipstream, so the script
    resolves the db even when run with a Python that can't import the installed
    package (e.g. slipstream installed as a uv tool in its own venv).
    """
    try:
        with open(CONFIG_PATH, "rb") as f:
            user = tomllib.load(f)
        return Path(user["out_dir"]).expanduser()
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def default_db_path() -> Path | None:
    out = out_dir_from_config()
    return out / "metadata" / "slipstream.db" if out else None


def _search_for_dbs() -> list[Path]:
    """Best-effort search of common locations for a stray slipstream.db."""
    roots = [
        Path.home() / "Library" / "Application Support" / "slipstream",  # macOS
        Path.home() / ".local" / "share" / "slipstream",
        Path.home() / ".config" / "slipstream",
        Path.home() / "slipstream",
        Path.cwd(),
    ]
    seen: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("slipstream.db"):
            if p not in seen:
                seen.append(p)
    return seen


def find_holders(db: Path) -> list[str]:
    """Return lines describing processes with the db file (or its sidecars) open.

    Uses lsof if available. An empty list means either nothing holds it or lsof
    is unavailable; the caller warns rather than trusting emptiness blindly.
    """
    targets = [str(db), str(db) + "-wal", str(db) + "-shm", str(db) + "-journal"]
    try:
        out = subprocess.run(
            ["lsof", "--", *targets],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    # Drop the header row lsof prints when there are matches.
    return [ln for ln in lines if not ln.startswith("COMMAND")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=None, help="path to slipstream.db")
    ap.add_argument(
        "--yes", action="store_true", help="proceed without the confirmation prompt"
    )
    args = ap.parse_args()

    db = args.db or default_db_path()
    if db is None or not db.exists():
        where = f" at {db}" if db is not None else ""
        print(f"error: slipstream.db not found{where}.", file=sys.stderr)
        if args.db is None:
            print(f"  (read out_dir from {CONFIG_PATH})", file=sys.stderr)
            found = _search_for_dbs()
            if found:
                print("  found these candidates:", file=sys.stderr)
                for p in found:
                    print(f"    {p}", file=sys.stderr)
        print("  pass the path explicitly with --db PATH.", file=sys.stderr)
        return 1
    print(f"db: {db}")

    # 1. Refuse to run while a process still holds the db: a live watch would
    #    keep re-creating the very contention we are recovering from, and a
    #    concurrent writer during recovery is unsafe.
    holders = find_holders(db)
    if holders:
        print("\nrefusing to run: the db is still open by:", file=sys.stderr)
        for ln in holders:
            print("  " + ln, file=sys.stderr)
        print(
            "\nStop the running watch/pusher first (e.g. kill the process), "
            "then re-run.",
            file=sys.stderr,
        )
        return 2
    print("no live holders detected (lsof)")

    if not args.yes:
        resp = input("\nProceed with backup + WAL conversion? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("aborted.")
            return 1

    # 2. Back up via the online backup API (WAL-safe, consistent snapshot).
    #    Opening the connection here also triggers SQLite's automatic recovery
    #    of any hot rollback-journal left by the crash.
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA busy_timeout=30000")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = db.with_suffix(f".recover_{ts}.bak")
    dest = sqlite3.connect(str(bak))
    try:
        conn.backup(dest)
    finally:
        dest.close()
    print(f"backup written: {bak}")

    # Read a single-row PRAGMA and fully finalize its cursor: an unclosed read
    # cursor keeps a lock on the connection that would make a later checkpoint
    # fail with SQLITE_LOCKED.
    def pragma1(sql: str):
        cur = conn.execute(sql)
        try:
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            cur.close()

    # 3. Integrity check before mutating anything further.
    ok = pragma1("PRAGMA integrity_check")
    print(f"integrity_check: {ok}")
    if ok != "ok":
        print(
            "error: integrity check failed; not converting. Inspect the backup.",
            file=sys.stderr,
        )
        conn.close()
        return 3

    # 4. Convert to WAL. Then close and reopen: converting from a rollback
    #    journal leaves per-connection lock state that makes an immediate
    #    same-connection checkpoint fail with SQLITE_LOCKED, so the checkpoint
    #    (and the reporting reads) run on a fresh connection.
    mode = pragma1("PRAGMA journal_mode=WAL")
    print(f"journal_mode: {mode}")
    conn.close()

    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA busy_timeout=30000")
    cur = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    busy, log, ckpt = cur.fetchone()
    cur.close()
    if busy:
        # Non-fatal: WAL conversion (the essential outcome) already succeeded.
        # The fixed store code checkpoints on its own during normal operation.
        print(
            f"wal_checkpoint(TRUNCATE): busy=1 log={log} checkpointed={ckpt} "
            "(could not fully truncate; harmless, will checkpoint on next run)"
        )
    else:
        print(f"wal_checkpoint(TRUNCATE): busy=0 log={log} checkpointed={ckpt}")

    # 5. Report what the restarted watch will re-push (informational only).
    try:
        rows = conn.execute(
            "SELECT ps.engine, ps.platform, COUNT(*)"
            " FROM processing_state ps"
            " LEFT JOIN push_state pu"
            "   ON pu.engine=ps.engine AND pu.platform=ps.platform"
            "  AND pu.commit_id=ps.commit_id"
            " WHERE pu.commit_id IS NULL"
            " GROUP BY ps.engine, ps.platform"
        ).fetchall()
        if rows:
            print("\nunpushed commits (will re-push idempotently on restart):")
            for engine, plat, n in rows:
                print(f"  {engine}/{plat}: {n}")
        else:
            print("\nno unpushed commits pending.")
    except sqlite3.OperationalError:
        # Older db without push_state: nothing to report.
        pass

    conn.close()
    print("\nrecovery complete. Restart with the fixed build:  slipstream watch ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
