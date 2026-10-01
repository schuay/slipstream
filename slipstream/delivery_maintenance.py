# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Explicit replay/rebuild under the coordinator's ownership and target locks."""

from __future__ import annotations

import contextlib
import json

from .delivery_sources import SshSpoolSource, validate_record
from .delivery_targets import target_identity
from .durability import atomic_write, durable_unlink, target_lock_path
from .relay import (
    cursor_path,
    find_source,
    parse_cursor_arg,
    reset_cursor,
    write_cursor,
)


def check_maintenance(coordinator):
    path = coordinator.state_dir / "maintenance.json"
    if path.exists():
        raise RuntimeError(
            f"interrupted maintenance at {path}; rerun the exact maintenance command"
        )


def _intent(coordinator, record):
    path = coordinator.state_dir / "maintenance.json"
    record = dict(record, targets=coordinator.identities)
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError(
                "interrupted maintenance parameters changed; restore original config/command"
            )
    else:
        atomic_write(path, json.dumps(record, sort_keys=True))
    return path


def _no_pending(coordinator):
    for source in coordinator.sources:
        record = source.pending()
        if record:
            validate_record(record, source, record.get("targets"))
            if source.acknowledged(record["unit"]):
                source.retire()
            else:
                raise ValueError(
                    f"unresolved delivery attempt for {source.identity}; retry it before maintenance"
                )


def target_guard(identity):
    return target_lock_path(identity).with_suffix(".maintenance.json")


def check_target_maintenance(identity):
    path = target_guard(identity)
    if path.exists():
        raise RuntimeError(
            f"interrupted target maintenance at {path}; resume original rebuild"
        )


@contextlib.contextmanager
def _targets(coordinator, *, intent=None):
    sessions = {}
    guards = []
    error_count = len(coordinator.errors)
    with contextlib.ExitStack() as locks:
        try:
            for target in coordinator.targets:
                locks.enter_context(
                    coordinator.lock(target_lock_path(target_identity(target)))
                )
            for identity in coordinator.identities:
                guard = target_guard(identity)
                if guard.exists():
                    if intent is None or json.loads(guard.read_text()) != intent:
                        raise ValueError(
                            "target has unresolved maintenance from another command/config"
                        )
                elif intent is not None:
                    atomic_write(guard, json.dumps(intent, sort_keys=True))
                if intent is not None:
                    guards.append(guard)
            for target in coordinator.targets:
                if target.spanner:
                    sessions[target_identity(target)] = coordinator.session_factory(
                        target,
                        settings=coordinator.settings,
                        should_stop=coordinator.should_stop,
                        shutdown_deadline=coordinator.shutdown_deadline,
                        log=coordinator.log,
                    )
            yield sessions
        finally:
            coordinator._close(sessions)
        if len(coordinator.errors) > error_count:
            raise coordinator.errors[error_count]
        for guard in guards:
            durable_unlink(guard)


def _wipe(coordinator, sessions, bot):
    for identity, session in sessions.items():
        count = session.wipe(bot)
        coordinator.log(
            f"{bot}: wiped {count} staged rows and slipstream aggregates from {identity}"
        )


def rebuild_local(coordinator, store, engines, platform, bot):
    _no_pending(coordinator)
    # A bot wipe deletes every engine/platform. Never erase a history which the
    # requested replay omits, including one no longer listed in config.
    histories = store.conn.execute(
        "SELECT DISTINCT engine, platform FROM processing_state"
    ).fetchall()
    if any(r[0] not in engines or r[1] != platform for r in histories):
        raise ValueError(
            "local rebuild must replay every engine/platform stored for this bot"
        )
    if store.conn.execute("SELECT 1 FROM delivery_attempts LIMIT 1").fetchone():
        raise ValueError("unresolved local attempts block bot rebuild")
    path = _intent(
        coordinator,
        dict(
            kind="local-rebuild",
            bot=bot,
            db=str(store.db_path),
            engines=sorted(engines),
            platform=platform,
        ),
    )
    # Reset first; if wipe fails, the maintenance intent blocks normal delivery.
    with _targets(coordinator, intent=json.loads(path.read_text())) as sessions:
        for engine in engines:
            store.clear_push_state(engine, platform)
        _wipe(coordinator, sessions, bot)
    durable_unlink(path)


def maintain_remote(
    coordinator, cfg, *, rebuild=None, reset=None, legacy=None, force=False
):
    _no_pending(coordinator)
    bot = rebuild or legacy or parse_cursor_arg(reset)[0]
    if legacy:
        if bot != cfg.push.bot_name:
            find_source(cfg, bot)
        check_maintenance(coordinator)
        with _targets(coordinator) as sessions:
            for identity, session in sessions.items():
                session.reconcile_legacy(bot)
                coordinator.log(f"{bot}: reconciled legacy marker on {identity}")
        return
    source_cfg = find_source(cfg, bot)
    source = next(
        (
            s
            for s in coordinator.sources
            if isinstance(s, SshSpoolSource) and s.bot == bot
        ),
        None,
    )
    if source is None:
        raise ValueError(
            "remote source disabled; enable delivery.remote for maintenance"
        )
    if reset:
        check_maintenance(coordinator)
        reset_cursor(cfg, *parse_cursor_arg(reset), coordinator.log, spool=source.spool)
        return
    seqs = source.spool.list(limit=1)
    if not seqs:
        raise ValueError("no spool entries to replay; refusing to wipe")
    start = min(seqs)
    if start > 1 and not force:
        raise ValueError(
            f"source retains its log only from {start}; entries 1..{start - 1} cannot be replayed; use --force explicitly"
        )
    # Check the retained head is actually readable before destructive writes.
    source.prepare(start, coordinator.settings)
    path = _intent(
        coordinator,
        dict(
            kind="remote-rebuild",
            bot=bot,
            source=source.identity,
            start=start,
            force=force,
        ),
    )
    with _targets(coordinator, intent=json.loads(path.read_text())) as sessions:
        write_cursor(cursor_path(source_cfg), start - 1)
        coordinator.log(f"{bot}: cursor reset to {start - 1}; replaying from {start}")
        _wipe(coordinator, sessions, bot)
    durable_unlink(path)
