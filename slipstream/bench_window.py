# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

"""Daily wall-clock admission policy and a single durable local override.

Evaluation is pure: callers supply the clock and the complete state snapshot.
Persistence never takes the machine lock.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import lock
from .durability import FileLock, atomic_write, durable_unlink


def validate(timezone: str | None, windows: list[str] | None) -> None:
    if timezone is not None:
        try:
            ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            raise ValueError("[bench] timezone must name an IANA timezone") from None
    if windows is None:
        return
    if timezone is None:
        raise ValueError("[bench] allowed_windows requires an explicit timezone")
    if not isinstance(windows, list):
        raise ValueError("[bench] allowed_windows must be a list of HH:MM-HH:MM")
    for window in windows:
        _interval(window)


def _interval(window: str) -> tuple[int, int]:
    if not isinstance(window, str) or not re.fullmatch(
        r"(?:[01]\d|2[0-3]):[0-5]\d-(?:[01]\d|2[0-3]):[0-5]\d", window
    ):
        raise ValueError("[bench] window must use HH:MM-HH:MM (00:00 to 23:59)")
    start, end = (int(t[:2]) * 60 + int(t[3:]) for t in window.split("-"))
    if start == end:
        raise ValueError("[bench] window start and end must differ")
    return start, end


@dataclass(frozen=True)
class Override:
    mode: str
    written_at: datetime
    expires_at: datetime | None = None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    next_change: datetime | None = None
    error: str | None = None


def _scheduled(now, zone, intervals):
    local = now.astimezone(zone)
    minute = local.hour * 60 + local.minute
    return any(
        start <= minute < end if start < end else minute >= start or minute < end
        for start, end in intervals
    )


def _next_change(now, zone, intervals):
    """Order valid local endpoints and offset discontinuities in UTC.

    Endpoints in a gap do not exist; both folds do. A discontinuity can
    change permission even when no endpoint exists at that instant.
    """
    candidates = set()
    today = now.astimezone(zone).date()
    for day in range(5):
        date = today + timedelta(days=day)
        for minute in {m for pair in intervals for m in pair}:
            local = datetime(date.year, date.month, date.day) + timedelta(
                minutes=minute
            )
            for fold in (0, 1):
                utc = local.replace(tzinfo=zone, fold=fold).astimezone(UTC)
                if utc.astimezone(zone).replace(tzinfo=None) == local:
                    candidates.add(utc)
    left = now.replace(microsecond=0)
    for _ in range(96):
        right = left + timedelta(hours=1)
        if left.astimezone(zone).utcoffset() != right.astimezone(zone).utcoffset():
            lo, hi = left, right
            offset = lo.astimezone(zone).utcoffset()
            while (hi - lo).total_seconds() > 1:
                mid = lo + timedelta(seconds=int((hi - lo).total_seconds() // 2))
                if mid.astimezone(zone).utcoffset() == offset:
                    lo = mid
                else:
                    hi = mid
            candidates.add(hi)
        left = right
    for candidate in sorted(candidates):
        if candidate > now and _scheduled(candidate, zone, intervals) != _scheduled(
            candidate - timedelta(microseconds=1), zone, intervals
        ):
            return candidate
    return None


def evaluate(timezone, windows, override, now: datetime) -> Decision:
    if now.tzinfo is None:
        raise ValueError("evaluation requires an aware clock")
    now = now.astimezone(UTC)
    if isinstance(override, str):
        return Decision(False, "invalid override", error=override)
    if override is not None and (
        override.expires_at is None or now < override.expires_at
    ):
        return Decision(
            override.mode == "open", f"forced {override.mode}", override.expires_at
        )
    if windows is None:
        return Decision(True, "unrestricted")
    intervals = [_interval(w) for w in windows]
    zone = ZoneInfo(timezone)
    allowed = _scheduled(now, zone, intervals)
    return Decision(
        allowed,
        "scheduled open" if allowed else "scheduled closed",
        _next_change(now, zone, intervals) if intervals else None,
    )


def override_path() -> Path:
    return lock.LOCKS_DIR / "bench-window.json"


def _instant(value):
    if not isinstance(value, str):
        raise ValueError("invalid UTC timestamp")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC")
    return result.astimezone(UTC)


def read_override(path: Path | None = None) -> Override | str | None:
    path = path or override_path()
    try:
        data = json.loads(path.read_text())
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != 1
        ):
            raise ValueError("unsupported override schema")
        if data.get("mode") not in ("open", "close"):
            raise ValueError("invalid override mode")
        written = _instant(data["written_at"])
        expiry = (
            _instant(data["expires_at"]) if data.get("expires_at") is not None else None
        )
        if (data["mode"] == "open" and expiry is None) or (
            expiry is not None and expiry <= written
        ):
            raise ValueError("invalid override expiry")
        return Override(data["mode"], written, expiry)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{path}: {exc}; use slipstream window reset to repair"


def write_override(
    mode: str | None, seconds: float | None = None, *, path=None, now=None
):
    if mode not in (None, "open", "close") or (mode == "open" and seconds is None):
        raise ValueError("open requires a positive duration")
    if seconds is not None and (
        seconds <= 0 or seconds == float("inf") or seconds != seconds
    ):
        raise ValueError("override duration must be positive and finite")
    path = path or override_path()
    now = (now or datetime.now(UTC)).astimezone(UTC)
    with FileLock(path.with_suffix(".lock")):
        if mode is None:
            durable_unlink(path)
            return None
        expiry = now + timedelta(seconds=seconds) if seconds is not None else None
        atomic_write(
            path,
            json.dumps(
                {
                    "version": 1,
                    "mode": mode,
                    "written_at": now.isoformat(),
                    "expires_at": expiry.isoformat() if expiry else None,
                }
            ),
        )
    return Override(mode, now, expiry)


def current(config, *, now=None, path=None) -> Decision:
    return evaluate(
        config.timezone,
        config.allowed_windows,
        read_override(path),
        now or datetime.now(UTC),
    )
