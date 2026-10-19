# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from slipstream.bench_window import (
    Override,
    evaluate,
    read_override,
    validate,
    write_override,
)
from slipstream.config import _parse_bench


def at(text):
    return datetime.fromisoformat(text).astimezone(UTC)


def decision(windows, text, zone="Europe/Berlin", override=None):
    return evaluate(zone, windows, override, at(text))


@pytest.mark.parametrize(
    "windows,text,allowed,next_time",
    [
        (None, "2026-10-19T12:00Z", True, None),
        ([], "2026-10-19T12:00Z", False, None),
        (["22:00-06:00"], "2026-10-19T20:00Z", True, "2026-10-20T04:00Z"),
        (["22:00-06:00"], "2026-10-19T04:00Z", False, "2026-10-19T20:00Z"),
        (["22:00-06:00"], "2026-10-19T22:00Z", True, "2026-10-20T04:00Z"),
        (
            ["09:00-11:00", "10:00-12:00", "12:00-13:00"],
            "2026-10-19T08:00Z",
            True,
            "2026-10-19T11:00Z",
        ),
        (["00:00-12:00", "12:00-00:00"], "2026-10-19T08:00Z", True, None),
        (["23:00-00:00"], "2026-10-19T22:00Z", False, "2026-10-20T21:00Z"),
    ],
)
def test_daily_union(windows, text, allowed, next_time):
    result = decision(windows, text)
    assert result.allowed is allowed
    assert result.next_change == (at(next_time) if next_time else None)


@pytest.mark.parametrize(
    "windows,text,allowed,next_time",
    [
        # The nonexistent 02:30 endpoint cannot postpone closing to 03:30.
        (["01:00-02:30"], "2026-03-29T00:00Z", True, "2026-03-29T01:00Z"),
        (["02:10-02:40"], "2026-03-29T00:00Z", False, "2026-03-30T00:10Z"),
        # Enter and leave the same local interval twice in the autumn fold.
        (["02:10-02:40"], "2026-10-25T00:10Z", True, "2026-10-25T00:40Z"),
        (["02:10-02:40"], "2026-10-25T00:40Z", False, "2026-10-25T01:10Z"),
        (["02:10-02:40"], "2026-10-25T01:10Z", True, "2026-10-25T01:40Z"),
        (["01:00-02:30"], "2026-10-25T00:40Z", False, "2026-10-25T01:00Z"),
        # A half-hour DST jump also changes permission at the discontinuity.
        (["01:00-02:15"], "2026-10-03T15:00Z", True, "2026-10-03T15:30Z"),
    ],
)
def test_dst(windows, text, allowed, next_time):
    zone = "Australia/Lord_Howe" if text.startswith("2026-10-03") else "Europe/Berlin"
    result = decision(windows, text, zone)
    assert result.allowed is allowed
    assert result.next_change == at(next_time)


@pytest.mark.parametrize(
    "data",
    [
        {"allowed_windows": []},
        {"timezone": "unknown/zone"},
        {"timezone": 12},
        {"timezone": "UTC", "allowed_windows": "22:00-06:00"},
        *(
            {"timezone": "UTC", "allowed_windows": [s]}
            for s in [
                "24:00-06:00",
                "22:60-06:00",
                "2:00-06:00",
                "00:00-00:00",
                None,
            ]
        ),
    ],
)
def test_invalid_configuration(data):
    with pytest.raises(ValueError):
        _parse_bench(data)


def test_config_defaults_and_validation():
    assert _parse_bench({}).allowed_windows is None
    assert (
        _parse_bench({"timezone": "UTC", "allowed_windows": []}).allowed_windows == []
    )
    validate("Europe/Berlin", ["22:00-06:00"])


def test_override_expiry_replacement_restart_and_clock_jump(tmp_path):
    path = tmp_path / "bench-window.json"
    now = at("2026-10-25T00:30Z")
    write_override("close", path=path, now=now)
    assert not evaluate(None, None, read_override(path), now).allowed
    opened = write_override("open", 3600, path=path, now=now)
    assert opened == read_override(path)
    assert evaluate("UTC", [], read_override(path), now + timedelta(minutes=59)).allowed
    assert not evaluate(
        "UTC", [], read_override(path), now + timedelta(hours=1)
    ).allowed
    assert evaluate("UTC", [], read_override(path), now - timedelta(hours=1)).allowed
    # Older closure does not return on expiry.
    assert evaluate(None, None, read_override(path), now + timedelta(hours=2)).allowed
    write_override(None, path=path)
    assert read_override(path) is None


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "[]",
        "{}",
        '{"version":2}',
        '{"version":1,"mode":"open","written_at":"2026-01-01T00:00Z"}',
        '{"version":1,"mode":"close","written_at":"2026-01-01T00:00"}',
    ],
)
def test_corrupt_state_blocks_until_reset(tmp_path, raw):
    path = tmp_path / "bench-window.json"
    path.write_text(raw)
    result = evaluate(None, None, read_override(path), datetime.now(UTC))
    assert not result.allowed and result.error
    write_override(None, path=path)
    assert read_override(path) is None


def test_concurrent_atomic_reads_and_mutations(tmp_path):
    path = tmp_path / "bench-window.json"

    def mutate(i):
        write_override("open" if i % 2 else "close", 60, path=path)
        assert not isinstance(read_override(path), str)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(mutate, range(30)))
    assert isinstance(read_override(path), Override)


@pytest.mark.parametrize(
    "mode,seconds", [("open", None), ("close", 0), ("open", -1), ("open", float("nan"))]
)
def test_invalid_override(mode, seconds, tmp_path):
    with pytest.raises(ValueError):
        write_override(mode, seconds, path=tmp_path / "state")


def test_next_change_is_permission_change_even_when_override_expiry_is_not():
    now = at("2026-10-19T20:00Z")  # 22:00 local
    override = Override("open", now, now + timedelta(hours=1))
    result = evaluate("Europe/Berlin", ["22:00-06:00"], override, now)
    assert result.next_change == at("2026-10-20T04:00Z")
    assert evaluate(None, None, override, now).next_change is None
    override = Override("close", now, now + timedelta(hours=12))
    result = evaluate("Europe/Berlin", ["22:00-06:00"], override, now)
    assert result.next_change == at("2026-10-20T20:00Z")


def test_unreadable_state_blocks_admission(tmp_path):
    path = tmp_path / "bench-window.json"
    path.mkdir()  # A directory is not a readable override document.
    result = evaluate(None, None, read_override(path), datetime.now(UTC))
    assert not result.allowed and result.error


@pytest.mark.parametrize("mode", ["open", "close"])
@pytest.mark.parametrize("injected", [False, True])
def test_override_duration_starts_after_mutation_lock(
    mode, injected, tmp_path, monkeypatch
):
    import slipstream.bench_window as mod

    before = at("2026-10-19T12:00Z")
    after = before + timedelta(seconds=30)
    clock = [before]

    class Clock:
        @staticmethod
        def now(zone):
            return clock[0]

    class ContendedLock:
        def __init__(self, path):
            pass

        def __enter__(self):
            clock[0] = after

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(mod, "datetime", Clock)
    monkeypatch.setattr(mod, "FileLock", ContendedLock)
    result = write_override(
        mode, 1, path=tmp_path / "state.json", now=before if injected else None
    )
    expected = before if injected else after
    assert result.written_at == expected
    assert result.expires_at == expected + timedelta(seconds=1)
