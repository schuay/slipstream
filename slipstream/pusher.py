# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Background pusher: drains newly benchmarked scores to the push targets
without blocking the collection loop.

A single worker thread owns its own SQLite connection (connections are
thread-affine) and runs :func:`slipstream.push.push` whenever the collector
signals new work. Wake signals coalesce: any number of ``notify()`` calls
collapse into at most one pending drain, so a slow or failing push target can
never grow a backlog or delay measurements. Because ``push()`` is incremental
and idempotent (a failed push leaves ``push_state`` untouched and retries next
wake), the worker needs no queue of its own beyond a single "work pending" bit.
"""

from __future__ import annotations

import contextlib
import threading
import traceback
from collections.abc import Callable

from .config import Config
from .push import push as do_push
from .store import CommitStore


class BackgroundPusher:
    """Single-threaded, coalescing background push worker.

    Lifecycle: ``start()`` once, ``notify()`` per newly finished commit (cheap,
    non-blocking), then ``close()`` for a final synchronous drain on shutdown.
    """

    def __init__(
        self,
        cfg: Config,
        engines: list[str],
        *,
        log: Callable[[str], None] | None = None,
    ):
        self._cfg = cfg
        self._engines = list(engines)
        self._log = log or (lambda _msg: None)

        self._wake = threading.Event()
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="slipstream-pusher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def notify(self) -> None:
        """Signal that new scores may be available. Non-blocking and coalescing.

        Safe to call from the collection thread after every commit; the worker
        drains whatever is unpushed, so bursts collapse into one push.
        """
        self._wake.set()

    def close(self, timeout: float = 120.0) -> None:
        """Request a final drain and wait for the worker to finish.

        Any commits marked done before this call are guaranteed to be attempted
        once more before the worker exits.
        """
        self._stopping = True
        self._wake.set()
        self._thread.join(timeout)

    def _run(self) -> None:
        # Create the connection in this thread. No backup (the collector's
        # store already snapshotted on open) and no schema init (the primary
        # connection owns migrations; running them here would race).
        store = CommitStore(
            self._cfg.metadata_dir / "slipstream.db",
            backup=False,
            init_schema=False,
            bot=self._cfg.bot_name,
        )
        try:
            while True:
                self._wake.wait()
                # Clear before draining so a notify() arriving mid-drain is not
                # lost: it re-sets the bit and we loop once more.
                self._wake.clear()
                self._drain(store)
                if self._stopping:
                    break
        finally:
            store.close()

    def _drain(self, store: CommitStore) -> None:
        try:
            n = do_push(
                store,
                self._engines,
                self._cfg.push,
                self._cfg.valid_names_by_suite,
                self._cfg.platform,
                log=self._log,
            )
            if n:
                self._log(f"pushed {n} scores")
        except Exception:
            # A push failure must never kill the worker or the collector.
            # push_state is untouched on failure, so the same commits retry on
            # the next wake (at-least-once, idempotent on the receiver).
            # Roll back so a failed drain leaves no open transaction holding a
            # lock into the next wake. Guarded: a rollback that itself raises
            # must not escape and kill the worker thread.
            with contextlib.suppress(Exception):
                store.conn.rollback()
            self._log("background push failed:\n" + traceback.format_exc())
