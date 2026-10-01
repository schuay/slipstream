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
import time
from collections.abc import Callable
from pathlib import Path

from .config import Config, RelaySource
from .push import parse_seq, seq_name
from .durability import atomic_write


class RemoteSpool:
    """The spool log on a source machine, reached over ssh."""

    def __init__(
        self,
        ssh_host: str,
        spool_dir: str,
        *,
        timeout=30.0,
        max_bytes=64 * 1024 * 1024,
        should_stop=lambda: False,
    ):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.should_stop = should_stop
        self._host = ssh_host
        # Quote the path for the remote shell, but leave a leading ~/ bare so
        # it still expands to the remote home.
        if spool_dir.startswith("~/"):
            self._dir = "~/" + shlex.quote(spool_dir[2:])
        else:
            self._dir = shlex.quote(spool_dir)

    def _ssh(self, command: str) -> str:
        import tempfile

        # File output bounds memory; abort transfer on deadline, cancellation or
        # size excess. Killing ssh also closes the remote command's transport.
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            with subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ServerAliveInterval=5",
                    "-o",
                    "ServerAliveCountMax=2",
                    self._host,
                    command,
                ],
                stdout=output,
                stderr=errors,
            ) as proc:
                deadline = time.monotonic() + self.timeout
                try:
                    while proc.poll() is None:
                        if self.should_stop():
                            raise InterruptedError("SSH cancelled")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("SSH operation deadline exceeded")
                        if output.tell() > self.max_bytes or errors.tell() > 65536:
                            raise ValueError("SSH payload exceeds configured bound")
                        time.sleep(0.02)
                    if proc.returncode:
                        raise subprocess.CalledProcessError(proc.returncode, "ssh")
                    output.seek(0)
                    data = output.read(self.max_bytes + 1)
                    if len(data) > self.max_bytes:
                        raise ValueError("SSH payload exceeds max_payload_bytes")
                    return data.decode("utf-8")
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()

    def list(self, *, cursor=0, limit=17) -> list[int]:
        """Bounded window plus newest sequence, for reset/gap detection."""
        # Published files only. A durable allocator also reveals a crash between
        # reserving a sequence and publishing it, even if retention emptied files.
        listing = f"LC_ALL=C ls -1 {self._dir} | grep -E '^[0-9]+\\.csv$' | sort -n"
        command = (
            f"test -d {self._dir} || exit 2; "
            f"({listing} | awk '($0 + 0) > {cursor}' | head -n {limit}; "
            f"{listing} | tail -n 1; "
            f"if test -f {self._dir}/.sequence; then cat {self._dir}/.sequence; fi)"
        )
        out = self._ssh(command)
        # .sequence is emitted as an integer, not a filename.
        seqs = []
        for name in out.split():
            seq = parse_seq(name) if name.endswith(".csv") else int(name)
            if seq:
                seqs.append(seq)
        return sorted(set(seqs))

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
        value = int(text)
        if value < 0:
            raise ValueError("negative cursor")
        return value
    except ValueError as e:
        raise ValueError(f"unreadable cursor {path}: {text!r}") from e


def write_cursor(path: Path, seq: int) -> None:
    if seq < 0:
        raise ValueError("negative cursor")
    atomic_write(path, f"{seq}\n")


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
    seqs = _open_spool(source, spool).list(limit=1)
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
