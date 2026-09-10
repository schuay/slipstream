# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Drain remote spool logs into this machine's push targets.

A machine without access to the push target appends its export CSVs to a
spool directory (a ``spool_dir`` push target) as a log numbered from 1. This
machine, which can ssh to it and reach the real target, reads the entries
past its cursor and delivers them to its own targets under the source's bot
name. The source keeps its log until retention drops it; this machine keeps
no copy and deletes nothing remotely.

The cursor is one integer per source, advanced after an entry reached every
target, so a crash re-delivers at most one entry -- harmless, the receiver
upserts. Delivering across a hole in the log would drop scores silently, so
a gap stops the source instead.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from .config import Config, RelaySource
from .push import open_target, parse_seq, seq_name


class RemoteSpool:
    """The spool log on a source machine, reached over ssh."""

    def __init__(self, ssh_host: str, spool_dir: str):
        self._host = ssh_host
        # Quote the path for the remote shell, but leave a leading ~/ bare so
        # it still expands to the remote home.
        if spool_dir.startswith("~/"):
            self._dir = "~/" + shlex.quote(spool_dir[2:])
        else:
            self._dir = shlex.quote(spool_dir)

    def _ssh(self, command: str) -> str:
        res = subprocess.run(
            ["ssh", self._host, command], capture_output=True, text=True, check=True
        )
        return res.stdout

    def list(self) -> list[int]:
        """Sequence numbers of complete entries, ascending."""
        # Skips the lock file and the .tmp files still being written.
        out = self._ssh(f"ls -1 {self._dir}")
        return sorted(s for s in (parse_seq(n) for n in out.split()) if s)

    def fetch(self, seq: int) -> str:
        return self._ssh(f"cat {self._dir}/{seq_name(seq)}")


def cursor_path(source: RelaySource) -> Path:
    return source.cursor_dir / f"{source.bot_name}.cursor"


def read_cursor(path: Path) -> int:
    """Last sequence number delivered to every target; 0 when never relayed."""
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return 0
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(f"unreadable cursor {path}: {text!r}") from e


def write_cursor(path: Path, seq: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(f"{seq}\n")
    tmp.replace(path)


def relay_source(
    cfg: Config,
    source: RelaySource,
    spool: RemoteSpool,
    log: Callable[[str], None],
) -> int:
    """Deliver one source's entries past the cursor. Returns how many landed.

    Targets are opened once for the whole cycle, so a backlog costs one
    connection and one aggregation rather than one per entry.
    """
    path = cursor_path(source)
    cursor = read_cursor(path)
    seqs = spool.list()
    if seqs and cursor > seqs[-1]:
        log(
            f"{source.bot_name}: cursor {cursor} is past the newest entry"
            f" {seqs[-1]} on {source.ssh_host}; the spool may have been reset"
            f" (--reset-cursor {source.bot_name}=0 to replay it)"
        )
        return 0
    pending = [s for s in seqs if s > cursor]
    if not pending:
        return 0

    # Everything up to the first hole is deliverable; the hole itself is the
    # operator's call. Delivering the prefix first leaves the gap at the head,
    # where skipping it costs only the entries that are really gone.
    gap = None
    for i, seq in enumerate(pending):
        want = cursor + 1 + i
        if seq != want:
            gap = (want, seq)
            pending = pending[:i]
            break

    delivered = _deliver(cfg, source, spool, path, pending, log) if pending else 0
    if gap and delivered == len(pending):
        want, seq = gap
        # Retention on the source took entries this machine never read.
        # Skipping them would drop those scores without a trace, so the
        # operator decides, with --reset-cursor.
        log(
            f"{source.bot_name}: gap: expected {want}, found {seq}"
            f" (entries {want}..{seq - 1} are gone from {source.ssh_host};"
            f" --reset-cursor {source.bot_name}={seq - 1} to resume at {seq})"
        )
    return delivered


def _deliver(
    cfg: Config,
    source: RelaySource,
    spool: RemoteSpool,
    path: Path,
    pending: list[int],
    log: Callable[[str], None],
) -> int:
    """Deliver a contiguous run of entries, advancing the cursor after each."""
    delivered = 0
    sessions = []
    try:
        for target in cfg.push.targets:
            sessions.append(open_target(target, source.bot_name))
        for seq in pending:
            name = seq_name(seq)
            try:
                csv_text = spool.fetch(seq)
            except Exception as e:
                log(f"{source.bot_name}: fetching {name} from {source.ssh_host}: {e}")
                break
            try:
                for session in sessions:
                    session.deliver(csv_text)
            except Exception as e:
                # Whatever the target raised (a failed command, a Spanner
                # error): the cursor still points at the last entry that got
                # through and the rest wait for the next cycle.
                log(f"{source.bot_name}: delivery of {name} failed: {e}")
                break
            try:
                write_cursor(path, seq)
            except OSError as e:
                # Going on would put the cursor further behind what has been
                # delivered, and every one of those entries would be replayed.
                log(f"{source.bot_name}: recording the cursor at {seq} failed: {e}")
                break
            delivered += 1
    finally:
        for session in sessions:
            try:
                summary = session.close()
            except Exception as e:
                # The delivered rows are staged and the cursor is left
                # advanced; any later refresh against the database aggregates
                # them, including `slipstream push --probe` on this machine.
                log(f"{source.bot_name}: closing target failed: {e}")
            else:
                if summary:
                    log(f"{source.bot_name}: {summary}")
    return delivered


def find_source(cfg: Config, bot_name: str) -> RelaySource:
    for source in cfg.relays:
        if source.bot_name == bot_name:
            return source
    raise ValueError(f"no [[relay]] source with bot_name {bot_name!r}")


def parse_cursor_arg(value: str) -> tuple[str, int]:
    """Split ``BOT`` or ``BOT=SEQ`` into the bot name and the sequence to
    resume after (0, the whole log, when no sequence is given)."""
    bot, _, raw = value.partition("=")
    if not bot:
        raise ValueError(f"expected BOT or BOT=SEQ, got {value!r}")
    if not raw:
        return bot, 0
    if not raw.isdigit():
        raise ValueError(f"cursor must be a non-negative integer, got {raw!r}")
    return bot, int(raw)


def _open_spool(source: RelaySource, spool: RemoteSpool | None) -> RemoteSpool:
    return (
        spool if spool is not None else RemoteSpool(source.ssh_host, source.spool_dir)
    )


def reset_cursor(
    cfg: Config,
    bot_name: str,
    seq: int,
    log: Callable[[str], None],
    *,
    spool: RemoteSpool | None = None,
) -> None:
    """Replay the source's log from ``seq`` + 1 on the next cycle."""
    source = find_source(cfg, bot_name)
    seqs = _open_spool(source, spool).list()
    newest = max(seqs, default=0)
    if seq > newest:
        # Past the end nothing is ever pending, so the relay would report
        # neither a delivery nor a gap: it would just go quiet.
        raise ValueError(
            f"{source.ssh_host} holds entries up to {newest};"
            f" a cursor of {seq} would deliver nothing"
        )
    oldest = min(seqs, default=1)
    if seq < oldest - 1:
        # Below the retained head every cycle reports the same gap.
        raise ValueError(
            f"{source.ssh_host} retains its log from {oldest};"
            f" a cursor of {seq} would stall on the gap. Use {oldest - 1} to"
            " deliver everything it still has."
        )
    write_cursor(cursor_path(source), seq)
    log(f"{bot_name}: cursor reset to {seq}")


def rebuild_source(
    cfg: Config,
    bot_name: str,
    log: Callable[[str], None],
    *,
    force: bool = False,
    spool: RemoteSpool | None = None,
) -> None:
    """Wipe this bot's staging rows and replay the source's log.

    Existing aggregate rows are not deleted. Replaying replaces aggregates
    represented by the retained log, but cannot remove obsolete aggregate
    keys that the replay no longer produces.

    The wipe is only safe if the source can still supply what it deletes, so
    the log is listed first: a head that no longer starts at 1 means retention
    has taken entries the replay cannot bring back, and the rebuild is refused
    unless ``force`` says that loss is acceptable.
    """
    source = find_source(cfg, bot_name)
    seqs = _open_spool(source, spool).list()
    if not seqs:
        raise ValueError(
            f"{source.ssh_host} has no spool entries to replay;"
            f" refusing to wipe {bot_name}"
        )
    start = min(seqs)
    if start > 1 and not force:
        raise ValueError(
            f"{source.ssh_host} retains its log only from {start}, so a rebuild"
            f" would drop the scores in entries 1..{start - 1} for good."
            " Re-run with --force to rebuild from what is left."
        )
    # The cursor moves first: a wipe that fails half way then replays, where
    # the other order would leave the rows deleted and nothing pending.
    write_cursor(cursor_path(source), start - 1)
    log(f"{bot_name}: cursor reset to {start - 1}; replaying from {start}")
    for target in cfg.push.targets:
        if target.spanner is None:
            continue
        from . import spanner

        db = spanner.connect(target.spanner)
        try:
            n = spanner.wipe_bot(db, bot_name)
        finally:
            db.close()
        log(f"{bot_name}: wiped {n} staged rows from {target.spanner}")


def relay_all(cfg: Config, log: Callable[[str], None]) -> int:
    total = 0
    for source in cfg.relays:
        try:
            n = relay_source(
                cfg, source, RemoteSpool(source.ssh_host, source.spool_dir), log
            )
        except Exception as e:
            # Source unreachable or cursor unwritable: retry next cycle.
            log(f"{source.bot_name}: relay from {source.ssh_host} failed: {e}")
            continue
        if n:
            log(f"{source.bot_name}: delivered {n} file(s)")
        total += n
    return total
