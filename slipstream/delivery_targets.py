# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Durable staging, explicit refresh, and bounded Spanner process ownership."""

from __future__ import annotations

import multiprocessing
import time

from .config import DeliveryConfig, parse_spanner_spec
from .delivery_batch import Receipt


def target_identity(target):
    if (target.spool_dir is None) == (target.spanner is None):
        raise ValueError("target needs exactly one spool_dir or spanner")
    if target.spool_dir is not None:
        return f"spool:{target.spool_dir.resolve()}"
    return "spanner:" + "/".join(parse_spanner_spec(target.spanner))


class SpoolSession:
    def __init__(
        self,
        target,
        *,
        settings=None,
        should_stop=lambda: False,
        log=lambda msg: None,
        **kwargs,
    ):
        self.settings = settings or DeliveryConfig()
        self.should_stop = should_stop
        self.log = log
        self.target = target
        self.identity = target_identity(target)

    def stage(self, batch, attempt):
        from .push import spool_append

        spool_append(
            self.target.spool_dir,
            batch.csv(),
            self.target.retain_days,
            timeout=self.settings.lock_seconds,
            should_stop=self.should_stop,
            log=self.log,
            bot=batch.bot,
        )
        return Receipt(self.identity, attempt, batch.digest)

    def refresh(self):
        return None

    def close(self):
        pass


class SpannerSession:
    """Multi-bot session; callers own all target locks before opening."""

    def __init__(self, target, *, log=lambda msg: None):
        from . import spanner

        self.target = target
        self.identity = target_identity(target)
        self.log = log
        self.db = spanner.connect(target.spanner, exclusive=False)
        self.db.log = log
        try:
            spanner.ensure_schema(self.db)
        except BaseException:
            self.db.close()
            raise

    def stage(self, batch, attempt):
        from . import spanner

        write = self.target.write
        # Map before opening the import record: a batch that does not map
        # leaves nothing behind to block refresh.
        groups = spanner.staging_groups(self.db, batch.rows, batch.bot, write=write)
        handle = spanner.begin_import(
            self.db, batch.bot, attempt, batch.digest, write=write
        )
        self.db.write(groups)
        count = len(batch.rows)
        spanner.finish_import(self.db, handle, count)
        self.log(f"staged bot={batch.bot} attempt={attempt} rows={count}")
        return Receipt(self.identity, attempt, batch.digest)

    def refresh(self):
        from . import spanner

        if not self.target.refresh:
            return "refresh=false; staging only"
        return spanner.refresh(self.db, source=self.target.aggregate_from)

    def wipe(self, bot):
        from . import spanner

        return spanner.rebuild_bot(self.db, bot, write=self.target.write)

    def reconcile_legacy(self, bot):
        from . import spanner

        # Explicit operator assertion: complete retained history was replayed.
        key = spanner.INCOMPLETE_PREFIX + bot
        found = self.db.query(
            f"SELECT value FROM {spanner.META_TABLE} WHERE key = %s", [key]
        )
        if found and found[0][0] != "staging":
            raise ValueError(
                "marker is not a legacy per-bot import; refusing reconciliation"
            )
        spanner.finish_import(self.db, key)

    def close(self):
        self.db.close()


def _spanner_worker(pipe, target):
    session = None
    try:
        session = SpannerSession(target, log=lambda msg: pipe.send(("log", msg)))
        pipe.send(("ok", None))
        while True:
            method, args = pipe.recv()
            try:
                result = getattr(session, method)(*args)
                pipe.send(("ok", result))
            except Exception as exc:
                pipe.send(("error", f"{type(exc).__name__}: {exc}"))
            if method == "close":
                session = None
                break
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        pipe.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if session is not None:
            session.close()
        pipe.close()


class BoundedSpannerSession:
    """Kill a stalled SDK operation without advancing source progress.

    A spawned child owns the gRPC channels; no channel is forked and no retry
    survives the parent's deadline. Destination mutations remain at-least-once.
    """

    def __init__(
        self,
        target,
        *,
        settings,
        should_stop=lambda: False,
        log=lambda msg: None,
        shutdown_deadline=lambda: None,
    ):
        self.shutdown_deadline = shutdown_deadline
        self.identity = target_identity(target)
        self.settings = settings
        self.should_stop = should_stop
        self.log = log
        self.stopping_at = None
        ctx = multiprocessing.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(
            target=_spanner_worker, args=(child, target), daemon=True
        )
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self._terminate()
            raise

    def _receive(self):
        deadline = time.monotonic() + self.settings.io_seconds
        while True:
            now = time.monotonic()
            if self.should_stop() and self.stopping_at is None:
                self.stopping_at = now
            shared_deadline = self.shutdown_deadline()
            if shared_deadline is not None:
                deadline = min(deadline, shared_deadline)
            if self.stopping_at is not None:
                deadline = min(
                    deadline, self.stopping_at + self.settings.shutdown_seconds
                )
            if now >= deadline:
                self._terminate()
                raise TimeoutError(f"{self.identity} operation deadline exceeded")
            if self.pipe.poll(min(0.05, deadline - now)):
                kind, value = self.pipe.recv()
                if kind == "log":
                    self.log(value)
                elif kind == "error":
                    raise RuntimeError(value)
                else:
                    return value
            elif not self.process.is_alive():
                raise RuntimeError(f"{self.identity} worker exited without a receipt")

    def _call(self, method, *args):
        if not self.process.is_alive():
            raise RuntimeError(f"{self.identity} worker unavailable")
        self.pipe.send((method, args))
        return self._receive()

    def stage(self, batch, attempt):
        return self._call("stage", batch, attempt)

    def refresh(self):
        return self._call("refresh")

    def wipe(self, bot):
        return self._call("wipe", bot)

    def reconcile_legacy(self, bot):
        return self._call("reconcile_legacy", bot)

    def _terminate(self):
        if self.process.is_alive():
            self.process.kill()
        self.process.join(timeout=1)
        self.pipe.close()

    def close(self):
        try:
            if self.process.is_alive():
                self._call("close")
        finally:
            self._terminate()


def open_session(target, **kwargs):
    if target.spool_dir is not None:
        return SpoolSession(target, **kwargs)
    return BoundedSpannerSession(target, **kwargs)
