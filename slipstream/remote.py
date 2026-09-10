# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""A bus root on another machine, reached over ssh.

Listing and reading entries mirror the relay's RemoteSpool: small text over
`ssh host cat`. Fetching a payload does not. RemoteSpool runs ssh with
capture_output and text=True, which would decode a .tar.zst as UTF-8 and buffer
hundreds of MB in the memory of a machine that is benchmarking. Payloads go
through rsync instead, which streams to a file and takes a rate limit.

Connectivity is one-directional by design: box1 reaches box2, box2 initiates
nothing and is never written to from here.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .bus import BLOB_SUFFIX, BenchState, BuilderState, Entry, state_from_json
from .config import BusSource


# A consumer fetches while holding the machine lock, so a half-open connection
# to the source would hang the whole box until someone noticed. Nothing here
# may wait forever, and nothing may stop to ask for a password: BatchMode makes
# ssh fail rather than prompt, the keepalives detect a peer that stopped
# answering, and the small text commands get a wall-clock bound as well.
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
]
SSH_TIMEOUT_SECS = 120
# The remote command exits with this when the path is simply not there. Chosen
# to collide with neither ssh's own 255 nor the 1 and 2 that ls and cat use for
# their other failures.
MISSING_EXIT = 3
# rsync's own I/O inactivity timeout, not a wall clock: a payload is hundreds
# of MB and may legitimately take a long time over a slow link.
RSYNC_IO_TIMEOUT_SECS = 300


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Every process this module starts, in one place so tests can stand in."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise subprocess.CalledProcessError(
            124, cmd, "", f"timed out after {timeout}s"
        ) from e


def quote_remote(path: str) -> str:
    """Quote a path for the remote shell, leaving a leading ~/ to expand."""
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


class SshSource:
    def __init__(self, source: BusSource):
        self.name = source.name
        self.host = source.ssh_host
        self.root = source.root.rstrip("/")
        self.bwlimit = source.bwlimit

    # --- paths on the far side ---

    def _topic_dir(self, engine: str) -> str:
        return f"{self.root}/topics/builds/{engine}"

    def _entry_path(self, engine: str, commit_id: int) -> str:
        return f"{self._topic_dir(engine)}/{commit_id}.json"

    def _blob_path(self, engine: str, commit_id: int) -> str:
        return f"{self.root}/blobs/builds/{engine}/{commit_id}{BLOB_SUFFIX}"

    def _ssh(self, command: str) -> subprocess.CompletedProcess:
        return _run(["ssh", *SSH_OPTIONS, self.host, command], timeout=SSH_TIMEOUT_SECS)

    # --- the source interface ---

    def _missing(self, res: subprocess.CompletedProcess) -> bool:
        """Whether the far side said "not there" rather than failing to answer.

        The distinction is load-bearing in both directions. Read as missing, an
        unreachable host makes the consumer step over commits it never
        benched; read as unreachable, a legitimately absent entry stalls the
        cursor forever.

        So the remote command reports it with an exit code of its own rather
        than a message: stderr is localised, and ssh's own "Identity file ...
        No such file or directory" reads exactly like a missing entry. ssh
        reserves 255 for itself and passes everything else through.
        """
        return res.returncode == MISSING_EXIT

    def ids_above(self, engine: str, cursor: int | None) -> list[int]:
        topic = quote_remote(self._topic_dir(engine))
        res = self._ssh(f"[ -d {topic} ] || exit {MISSING_EXIT}; ls -1 {topic}")
        if res.returncode != 0:
            # A topic directory that does not exist yet is an empty topic, not
            # an error: the builder creates it with its first publish.
            if self._missing(res):
                return []
            raise subprocess.CalledProcessError(
                res.returncode, f"ssh {self.host}", res.stdout, res.stderr
            )
        ids = []
        for name in res.stdout.split():
            stem, dot, ext = name.partition(".")
            if dot and ext == "json" and stem.isdigit():
                ids.append(int(stem))
        ids.sort()
        return ids if cursor is None else [i for i in ids if i > cursor]

    def read_entry(self, engine: str, commit_id: int) -> Entry | None:
        path = quote_remote(self._entry_path(engine, commit_id))
        res = self._ssh(f"[ -f {path} ] || exit {MISSING_EXIT}; cat {path}")
        if res.returncode != 0:
            if self._missing(res):
                return None  # retention dropped it, or it was never there
            # Anything else is the transport. Returning None here would make an
            # unreachable host look like a dropped entry, and the consumer
            # would advance its cursor past a commit it never benched.
            raise subprocess.CalledProcessError(
                res.returncode, f"ssh {self.host}", res.stdout, res.stderr
            )
        return Entry.from_json(res.stdout, f"{self.host}:{path}")

    def payload(self, engine: str, commit_id: int, dest: Path) -> Path:
        """Stream the payload to ``dest``, resuming a previous attempt.

        rsync rather than scp: the rate limit units differ between them
        (rsync KB/s, scp Kbit/s), so the tool is pinned to keep the config
        value meaningful.
        """
        cmd = [
            "rsync",
            "--partial",
            "--inplace",
            f"--timeout={RSYNC_IO_TIMEOUT_SECS}",
            "-e",
            " ".join(["ssh", *SSH_OPTIONS]),
        ]
        if self.bwlimit:
            cmd.append(f"--bwlimit={self.bwlimit}")
        remote = quote_remote(self._blob_path(engine, commit_id))
        cmd += [f"{self.host}:{remote}", str(dest)]
        res = _run(cmd)
        if res.returncode != 0:
            # The partial file stays, which is what --partial --inplace are
            # for: a payload is hundreds of MB over a link slow enough to want
            # a bwlimit, and deleting it here would restart from zero every
            # time the connection dropped. bus gc reclaims one that never
            # completes.
            raise subprocess.CalledProcessError(
                res.returncode, cmd, res.stdout, res.stderr
            )
        return dest

    def keep_payload(self) -> bool:
        # A remote payload is this machine's own copy; it goes as soon as the
        # unpack is verified.
        return False

    def builder_state(self, engine: str) -> BuilderState:
        """What the far builder knows: frontier, retention, failures, stalls.

        Without it a dead builder beside a live topic looks entirely healthy
        from here.
        """
        path = quote_remote(f"{self.root}/state/builds/{engine}.json")
        res = self._ssh(f"[ -f {path} ] || exit {MISSING_EXIT}; cat {path}")
        if res.returncode != 0:
            if self._missing(res):
                return BuilderState()
            raise subprocess.CalledProcessError(
                res.returncode, f"ssh {self.host}", res.stdout, res.stderr
            )
        return state_from_json(res.stdout, BuilderState, f"{self.host}:{path}")

    def bench_state(self, engine: str) -> BenchState:
        """What the far bencher knows, including the environment block.

        A dead bencher beside a live builder looks entirely healthy from here
        without it.
        """
        path = quote_remote(f"{self.root}/state/bench/{engine}.json")
        res = self._ssh(f"[ -f {path} ] || exit {MISSING_EXIT}; cat {path}")
        if res.returncode != 0:
            if self._missing(res):
                return BenchState()
            raise subprocess.CalledProcessError(
                res.returncode, f"ssh {self.host}", res.stdout, res.stderr
            )
        return state_from_json(res.stdout, BenchState, f"{self.host}:{path}")
