# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""One owner, bounded fair snapshots, durable staging and independent refresh."""

from __future__ import annotations

import contextlib
import time
import uuid
from pathlib import Path

from .config import DeliveryConfig, validate_delivery_identities
from .delivery_sources import (
    LocalDbSource,
    SshSpoolSource,
    cursor_lock_path,
    owner_path,
    validate_record,
)
from .delivery_targets import open_session, target_identity
from .durability import FileLock, target_lock_path


class Coordinator:
    def __init__(
        self,
        sources,
        targets,
        state_dir,
        *,
        settings=None,
        should_stop=lambda: False,
        log=lambda msg: None,
        session_factory=open_session,
    ):
        self.sources = list(sources)
        self.targets = sorted(targets, key=target_identity)
        self.identities = [target_identity(t) for t in self.targets]
        if not self.targets or len(set(self.identities)) != len(self.identities):
            raise ValueError("delivery requires distinct targets")
        if len({s.identity for s in self.sources}) != len(self.sources):
            raise ValueError("duplicate delivery sources")
        if (
            any(t.spool_dir for t in self.targets)
            and len({s.bot for s in self.sources}) > 1
        ):
            raise ValueError(
                "spool targets require one bot identity; use Spanner for multi-bot delivery"
            )
        self.state_dir = Path(state_dir)
        self.settings = settings or DeliveryConfig()
        self.stopping_at = None

        def stopping():
            if should_stop() and self.stopping_at is None:
                self.stopping_at = time.monotonic()
            return self.stopping_at is not None

        self.should_stop = stopping
        self.log = log
        self.session_factory = session_factory
        self.retry = {}
        self.rotation = 0
        self.errors = []

    def shutdown_deadline(self):
        return (
            self.stopping_at + self.settings.shutdown_seconds
            if self.stopping_at is not None
            else None
        )

    def lock(self, path):
        return FileLock(
            path,
            timeout=self.settings.lock_seconds,
            should_stop=self.should_stop,
            log=self.log,
        )

    @contextlib.contextmanager
    def ownership(self):
        paths = {owner_path(self.state_dir)}
        for source in self.sources:
            if isinstance(source, LocalDbSource):
                paths.add(source.store.db_path.with_suffix(".delivery.owner.lock"))
            elif isinstance(source, SshSpoolSource):
                paths.add(cursor_lock_path(source))
        with contextlib.ExitStack() as locks:
            for path in sorted(paths):
                locks.enter_context(self.lock(path))
            yield

    def fail(self, source, exc):
        self.errors.append(exc)
        failures, _ = self.retry.get(source.identity, (0, 0))
        failures += 1
        delay = min(
            self.settings.max_retry_seconds,
            self.settings.retry_seconds * 2 ** min(failures - 1, 20),
        )
        self.retry[source.identity] = failures, time.monotonic() + delay
        self.log(
            f"source={source.identity} failed: {exc}; retry_in={delay:.1f}s retry_at_unix={time.time() + delay:.3f}"
        )

    def cycle(self, *, probe_bot=None):
        """Caller holds ownership. All preparation precedes target acquisition."""
        from .delivery_maintenance import check_maintenance, check_target_maintenance

        check_maintenance(self)
        started = time.monotonic()
        deadline = started + self.settings.cycle_seconds
        self.errors = []
        delivered_rows = 0
        selected = []
        rows = size = 0
        ordered = self.sources[self.rotation :] + self.sources[: self.rotation]
        if self.sources:
            self.rotation = (self.rotation + 1) % len(self.sources)
        # Selection itself is bounded. Do not refill the snapshot after ack.
        candidates = []
        for source in ordered:
            if self.should_stop() or time.monotonic() >= deadline:
                break
            if self.retry.get(source.identity, (0, 0))[1] > time.monotonic():
                continue
            try:
                record = source.pending()
                if record:
                    validate_record(record, source, record.get("targets"))
                if record and source.acknowledged(record["unit"]):
                    source.retire()
                    record = None
                    self.log(
                        f"source={source.identity} reconciled acknowledged attempt"
                    )
                if record:
                    validate_record(record, source, self.identities)
                    units = [record["unit"]]
                else:
                    units = source.discover(
                        min(
                            self.settings.quantum,
                            self.settings.max_units - len(candidates),
                        )
                    )
                candidates.extend((source, unit, record) for unit in units)
            except Exception as exc:
                self.fail(source, exc)
            if len(candidates) >= self.settings.max_units:
                break

        # Sorted commit locks, then sorted target locks. Each local lock gets
        # its own stack so ack can release it before refresh and peer staging.
        holds = {}
        sessions = {}
        failed_sources = set()
        failed_units = set()
        try:
            for source, unit, _ in sorted(
                candidates, key=lambda c: (c[0].identity, c[1])
            ):
                if self.should_stop() or time.monotonic() >= deadline:
                    break
                if isinstance(source, LocalDbSource):
                    hold = contextlib.ExitStack()
                    holds[(source.identity, unit)] = hold
                    try:
                        hold.enter_context(
                            source.store.result_locks(
                                source.engine,
                                source.platform,
                                [unit],
                                timeout=min(
                                    self.settings.lock_seconds,
                                    max(0.01, deadline - time.monotonic()),
                                ),
                                should_stop=lambda: (
                                    self.should_stop() or time.monotonic() >= deadline
                                ),
                                log=self.log,
                            )
                        )
                    except Exception as exc:
                        self.fail(source, exc)
                        failed_units.add((source.identity, unit))
            for source, unit, record in candidates:
                if (
                    source.identity in failed_sources
                    or (source.identity, unit) in failed_units
                    or self.should_stop()
                    or time.monotonic() >= deadline
                ):
                    continue
                try:
                    before = time.monotonic()
                    self.log(f"prepare start source={source.identity} unit={unit}")
                    batch = source.prepare(unit, self.settings)
                    if batch is None:
                        continue
                    if batch.size > self.settings.max_payload_bytes:
                        raise ValueError(f"unit {unit} exceeds max_payload_bytes")
                    # An oversized valid unit gets one attempt, alone for that
                    # source. A hard maximum prevents unbounded memory use.
                    if selected and (
                        rows + len(batch.rows) > self.settings.max_rows
                        or size + batch.size > self.settings.max_bytes
                    ):
                        continue
                    if record and record["digest"] != batch.digest:
                        raise ValueError(
                            f"unit {unit}: pending payload digest mismatch"
                        )
                    self.log(
                        f"prepared source={source.identity} unit={unit} rows={len(batch.rows)} bytes={batch.size} elapsed={time.monotonic() - before:.3f}s"
                    )
                    selected.append((source, batch, record))
                    rows += len(batch.rows)
                    size += batch.size
                except Exception as exc:
                    self.fail(source, exc)
                    failed_sources.add(source.identity)
            preparation_failed = failed_sources.copy()
            failed_sources.clear()
            # No network/source reads or further commit lock acquisition below.
            with contextlib.ExitStack() as target_locks:
                locked = []
                for target in self.targets:
                    identity = target_identity(target)
                    try:
                        target_locks.enter_context(
                            FileLock(
                                target_lock_path(identity),
                                timeout=min(
                                    self.settings.lock_seconds,
                                    max(0.01, deadline - time.monotonic()),
                                ),
                                should_stop=self.should_stop,
                                log=self.log,
                            )
                        )
                        check_target_maintenance(identity)
                        locked.append(target)
                    except Exception as exc:
                        self.errors.append(exc)
                        self.log(f"target={identity} lock failed: {exc}")
                        break
                target_locks.callback(self._close, sessions)
                if len(locked) == len(self.targets):
                    for target in locked:
                        identity = target_identity(target)
                        if self.should_stop():
                            break
                        try:
                            self.log(f"target={identity} open start")
                            sessions[identity] = self.session_factory(
                                target,
                                settings=self.settings,
                                should_stop=self.should_stop,
                                shutdown_deadline=self.shutdown_deadline,
                                log=lambda msg, i=identity: self.log(
                                    f"target={i} {msg}"
                                ),
                            )
                        except Exception as exc:
                            self.errors.append(exc)
                            self.log(f"target={identity} open failed: {exc}")
                if probe_bot is not None and len(sessions) == len(self.targets):
                    from .delivery_batch import Batch
                    from .durability import identity_digest

                    batch = Batch.local("probe", probe_bot, 0, ())
                    attempt = identity_digest("probe:" + probe_bot + batch.digest)
                    for identity, session in sessions.items():
                        try:
                            session.stage(batch, attempt)
                            self.log(f"{identity}: probe accepted")
                        except Exception as exc:
                            self.errors.append(exc)
                            self.log(f"{identity}: probe failed: {exc}")
                for source, batch, record in selected:
                    if (
                        source.identity in failed_sources
                        or self.should_stop()
                        or time.monotonic() >= deadline
                    ):
                        continue
                    try:
                        if len(sessions) != len(self.targets):
                            raise RuntimeError("not all targets available")
                        if record is None:
                            record = dict(
                                version=1,
                                source=source.identity,
                                bot=batch.bot,
                                unit=batch.unit,
                                targets=self.identities,
                                attempt=uuid.uuid4().hex,
                                digest=batch.digest,
                            )
                            source.save(record)  # durable before first external write
                        before = time.monotonic()
                        for identity, session in sessions.items() if batch.rows else []:
                            if self.should_stop():
                                raise InterruptedError("shutdown before target staging")
                            stage_start = time.monotonic()
                            self.log(
                                f"staging source={source.identity} target={identity} unit={batch.unit} rows={len(batch.rows)} bytes={batch.size} attempt={record['attempt']}"
                            )
                            receipt = session.stage(batch, record["attempt"])
                            self.log(
                                f"staged target={identity} unit={batch.unit} elapsed={time.monotonic() - stage_start:.3f}s"
                            )
                            if (receipt.target, receipt.attempt, receipt.digest) != (
                                identity,
                                record["attempt"],
                                batch.digest,
                            ):
                                raise ValueError("target returned a mismatched receipt")
                        if self.should_stop():
                            raise InterruptedError(
                                "shutdown before acknowledgement; retry retained unit"
                            )
                        source.acknowledge(batch)
                        delivered_rows += len(batch.rows)
                        progress = (
                            f"cursor={batch.unit}"
                            if isinstance(source, SshSpoolSource)
                            else f"commit={batch.unit}"
                        )
                        self.log(
                            f"ack source={source.identity} unit={batch.unit} {progress} rows={len(batch.rows)} elapsed={time.monotonic() - before:.3f}s"
                        )
                        source.retire()  # ack-before-delete is reconciled after restart
                        if source.identity not in preparation_failed:
                            self.retry.pop(source.identity, None)
                    except Exception as exc:
                        self.fail(source, exc)
                        failed_sources.add(source.identity)
                    finally:
                        hold = holds.pop((source.identity, batch.unit), None)
                        if hold:
                            hold.close()
                # Also release unattempted commit locks before refresh.
                for hold in holds.values():
                    hold.close()
                holds.clear()
                for identity, session in sessions.items():
                    if self.should_stop():
                        break
                    before = time.monotonic()
                    try:
                        result = session.refresh()
                        self.log(
                            f"target={identity} refresh={result or 'complete'} elapsed={time.monotonic() - before:.3f}s"
                        )
                    except Exception as exc:
                        self.errors.append(exc)
                        self.log(
                            f"target={identity} refresh failed: {exc}; retry on idle cycle"
                        )
                # Cleanup occurs while target locks are held, and never refreshes.
                self._close(sessions)
                sessions.clear()
        finally:
            for hold in holds.values():
                hold.close()
        self.log(
            f"cycle rows={delivered_rows} units={len(selected)} elapsed={time.monotonic() - started:.3f}s errors={len(self.errors)}"
        )
        return delivered_rows

    def _close(self, sessions):
        for identity, session in sessions.items():
            before = time.monotonic()
            self.log(f"target={identity} cleanup start")
            try:
                session.close()
            except Exception as exc:
                self.errors.append(exc)
                self.log(f"target={identity} cleanup failed: {exc}")
            finally:
                self.log(
                    f"target={identity} cleanup elapsed={time.monotonic() - before:.3f}s"
                )

    def run(self, *, once=False):
        with self.ownership():
            while not self.should_stop():
                self.cycle()
                if once:
                    return not self.errors
                self.log(f"next poll in {self.settings.poll_seconds}s")
                until = time.monotonic() + self.settings.poll_seconds
                while not self.should_stop() and time.monotonic() < until:
                    time.sleep(min(0.1, max(0, until - time.monotonic())))
        return True


def configured_coordinator(cfg, *, log, should_stop=lambda: False, local_only=False):
    from .store import CommitStore

    if cfg.push is None:
        raise ValueError("[push] is required for delivery")
    validate_delivery_identities(cfg.push, cfg.relays)
    sources, store = [], None
    if cfg.delivery.local and cfg.engines:
        path = cfg.metadata_dir / "slipstream.db"
        if not path.exists():
            raise ValueError(
                f"local delivery DB missing: {path}; set delivery.local=false for remote-only service"
            )
        import sqlite3

        try:
            store = CommitStore(
                path,
                bot=cfg.require_bot_name(),
                backup=False,
                busy_timeout_ms=max(
                    1, int(min(1, cfg.delivery.shutdown_seconds) * 1000)
                ),
                should_stop=should_stop,
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"local delivery DB unavailable: {exc}") from exc
        sources.extend(
            LocalDbSource(
                store, e, cfg.platform, cfg.push.bot_name, cfg.valid_names_by_suite
            )
            for e in cfg.engines
        )
    if cfg.delivery.remote and not local_only:
        sources.extend(
            SshSpoolSource(src, settings=cfg.delivery, should_stop=should_stop)
            for src in cfg.relays
        )
    return Coordinator(
        sources,
        cfg.push.targets,
        cfg.out_dir / "delivery",
        settings=cfg.delivery,
        should_stop=should_stop,
        log=log,
    ), store
