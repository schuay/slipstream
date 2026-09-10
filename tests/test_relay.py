# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from slipstream.config import PushConfig, PushTarget, RelaySource
from slipstream.relay import read_cursor, relay_source, write_cursor


class FakeSpool:
    """In-memory stand-in for RemoteSpool, keyed by sequence number."""

    def __init__(self, entries: dict[int, str]):
        self.entries = dict(entries)
        self.fetched: list[int] = []

    def list(self):
        return sorted(self.entries)

    def fetch(self, seq):
        self.fetched.append(seq)
        return self.entries[seq]


class FakeSession:
    """Records deliveries and returns a summary once, like a spanner target."""

    def __init__(self, target, bot_name, calls, state):
        self.target = target
        self.bot = bot_name
        self._calls = calls
        self._state = state
        self.staged = 0

    def deliver(self, csv_text):
        self._calls.append((self.target, self.bot, csv_text))
        if self._state["fail_on"] and self._state["fail_on"] in csv_text:
            raise subprocess.CalledProcessError(1, "x")
        self.staged += 1

    def close(self):
        self._state["closes"] += 1
        if self._state["close_raises"]:
            raise RuntimeError("refresh timed out")
        return f"{self.staged} rows staged"


class FakeCfg:
    def __init__(self, targets):
        self.push = PushConfig(bot_name="box1", targets=targets)
        self.relays = []


@pytest.fixture
def delivered(monkeypatch):
    """Replaces open_target so no real target is touched."""
    calls = []
    state = {"fail_on": None, "closes": 0, "close_raises": False, "sessions": []}

    def open_target(target, bot_name, *, rebuild=False):
        s = FakeSession(target, bot_name, calls, state)
        state["sessions"].append(s)
        return s

    monkeypatch.setattr("slipstream.relay.open_target", open_target)
    return type("D", (), {"calls": calls, "state": state})


def _source(tmp_path):
    return RelaySource(
        ssh_host="box2", spool_dir="/o", bot_name="box2", cursor_dir=tmp_path / "relay"
    )


def _cursor(tmp_path):
    return read_cursor(tmp_path / "relay" / "box2.cursor")


class TestCursor:
    def test_missing_file_is_zero(self, tmp_path):
        assert read_cursor(tmp_path / "nope.cursor") == 0

    def test_round_trip_is_atomic(self, tmp_path):
        p = tmp_path / "sub" / "box2.cursor"
        write_cursor(p, 7)
        assert read_cursor(p) == 7
        write_cursor(p, 8)
        assert read_cursor(p) == 8
        assert [q.name for q in p.parent.iterdir()] == ["box2.cursor"]

    def test_corrupt_file_is_reported(self, tmp_path):
        p = tmp_path / "box2.cursor"
        p.write_text("half-written")
        with pytest.raises(ValueError, match="unreadable cursor"):
            read_cursor(p)


class TestRelaySource:
    def test_delivers_new_entries_and_advances(self, tmp_path, delivered):
        spool = FakeSpool({1: "a", 2: "b"})
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            spool,
            lambda m: None,
        )
        assert n == 2
        assert [(b, d) for _, b, d in delivered.calls] == [("box2", "a"), ("box2", "b")]
        assert _cursor(tmp_path) == 2
        # Nothing is deleted on the source.
        assert spool.list() == [1, 2]

    def test_already_delivered_entries_are_skipped(self, tmp_path, delivered):
        write_cursor(tmp_path / "relay" / "box2.cursor", 2)
        spool = FakeSpool({1: "a", 2: "b", 3: "c"})
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            spool,
            lambda m: None,
        )
        assert n == 1
        assert spool.fetched == [3]
        assert _cursor(tmp_path) == 3

    def test_nothing_pending_opens_no_target(self, tmp_path, delivered):
        write_cursor(tmp_path / "relay" / "box2.cursor", 2)
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a", 2: "b"}),
            lambda m: None,
        )
        assert n == 0
        assert delivered.state["sessions"] == []

    def test_cursor_past_recreated_spool_is_reported(self, tmp_path, delivered):
        write_cursor(tmp_path / "relay" / "box2.cursor", 42)
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a", 2: "b", 3: "c"}),
            logs.append,
        )
        assert n == 0
        assert delivered.state["sessions"] == []
        assert logs == [
            "box2: cursor 42 is past the newest entry 3 on box2; the spool may"
            " have been reset (--reset-cursor box2=0 to replay it)"
        ]

    def test_all_targets_receive_each_entry(self, tmp_path, delivered):
        targets_dirs = [Path("/spool/a"), Path("/spool/b")]
        targets = [PushTarget(spool_dir=d) for d in targets_dirs]
        relay_source(
            FakeCfg(targets), _source(tmp_path), FakeSpool({1: "a"}), lambda m: None
        )
        assert [t.spool_dir for t, _, _ in delivered.calls] == targets_dirs

    def test_one_session_per_cycle_whatever_the_backlog(self, tmp_path, delivered):
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a", 2: "b", 3: "c"}),
            logs.append,
        )
        assert n == 3
        assert len(delivered.state["sessions"]) == 1
        assert delivered.state["closes"] == 1
        assert logs == ["box2: 3 rows staged"]

    def test_failure_stops_the_source_at_the_cursor(self, tmp_path, delivered):
        delivered.state["fail_on"] = "b"
        logs = []
        cfg = FakeCfg([PushTarget(spool_dir=Path("/spool/x"))])
        n = relay_source(
            cfg, _source(tmp_path), FakeSpool({1: "a", 2: "b", 3: "c"}), logs.append
        )
        assert n == 1
        # The first entry is delivered and kept; the third is not attempted.
        assert [d for _, _, d in delivered.calls] == ["a", "b"]
        assert _cursor(tmp_path) == 1
        assert any("delivery of 00000002.csv failed" in m for m in logs)
        # The session is closed even so, staging what did get through.
        assert delivered.state["closes"] == 1

        # Next cycle, with the target healthy, resumes at the failed entry.
        delivered.state["fail_on"] = None
        n = relay_source(
            cfg, _source(tmp_path), FakeSpool({1: "a", 2: "b", 3: "c"}), logs.append
        )
        assert n == 2
        assert [d for _, _, d in delivered.calls[2:]] == ["b", "c"]
        assert _cursor(tmp_path) == 3

    def test_close_failure_leaves_the_cursor_advanced(self, tmp_path, delivered):
        delivered.state["close_raises"] = True
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a"}),
            logs.append,
        )
        # The rows are staged; only the aggregation failed, and the next
        # refresh picks them up.
        assert n == 1
        assert _cursor(tmp_path) == 1
        assert any("closing target failed: refresh timed out" in m for m in logs)

    def test_every_session_is_closed_when_one_close_fails(self, tmp_path, delivered):
        delivered.state["close_raises"] = True
        targets_dirs = [Path("/spool/a"), Path("/spool/b")]
        targets = [PushTarget(spool_dir=d) for d in targets_dirs]
        relay_source(
            FakeCfg(targets), _source(tmp_path), FakeSpool({1: "a"}), lambda m: None
        )
        assert delivered.state["closes"] == 2

    def test_fetch_failure_names_the_source(self, tmp_path, delivered):
        class BrokenSpool(FakeSpool):
            def fetch(self, seq):
                raise subprocess.CalledProcessError(255, "ssh")

        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            BrokenSpool({1: "a"}),
            logs.append,
        )
        assert n == 0
        assert delivered.calls == []
        assert any("fetching 00000001.csv from box2" in m for m in logs)

    def test_cursor_write_failure_is_reported_as_such(
        self, tmp_path, delivered, monkeypatch
    ):
        def full_disk(path, seq):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("slipstream.relay.write_cursor", full_disk)
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a", 2: "b"}),
            logs.append,
        )
        assert n == 0
        # The entry was delivered; only the cursor did not stick, so the next
        # cycle replays it rather than skipping ahead.
        assert [d for _, _, d in delivered.calls] == ["a"]
        assert any("recording the cursor at 1 failed" in m for m in logs)

    def test_any_delivery_error_stops_the_source(self, tmp_path, monkeypatch):
        """A Spanner target raises the client's own exception types."""

        class Boom:
            def deliver(self, csv_text):
                raise RuntimeError("DEADLINE_EXCEEDED")

            def close(self):
                return None

        monkeypatch.setattr("slipstream.relay.open_target", lambda t, b: Boom())
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a"}),
            logs.append,
        )
        assert n == 0
        assert _cursor(tmp_path) == 0
        assert any("DEADLINE_EXCEEDED" in m for m in logs)


class TestGapDetection:
    def test_gap_at_the_head_stops_without_delivering(self, tmp_path, delivered):
        # Retention on the source dropped entries 1 and 2.
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({3: "c", 4: "d"}),
            logs.append,
        )
        assert n == 0
        assert delivered.calls == []
        assert _cursor(tmp_path) == 0
        assert logs == [
            "box2: gap: expected 1, found 3 (entries 1..2 are gone from box2;"
            " --reset-cursor box2=2 to resume at 3)"
        ]

    def test_gap_inside_the_list_delivers_the_prefix_first(self, tmp_path, delivered):
        # 1 and 2 are still on the source, so skipping them would throw away
        # scores that are there for the taking; only 3 and 4 are gone.
        logs = []
        cfg = FakeCfg([PushTarget(spool_dir=Path("/spool/x"))])
        n = relay_source(
            cfg, _source(tmp_path), FakeSpool({1: "a", 2: "b", 5: "e"}), logs.append
        )
        assert n == 2
        assert [d for _, _, d in delivered.calls] == ["a", "b"]
        assert _cursor(tmp_path) == 2
        assert logs[-1] == (
            "box2: gap: expected 3, found 5 (entries 3..4 are gone from box2;"
            " --reset-cursor box2=4 to resume at 5)"
        )

        # Taking the advice loses 3 and 4 and nothing else.
        write_cursor(tmp_path / "relay" / "box2.cursor", 4)
        logs.clear()
        n = relay_source(
            cfg, _source(tmp_path), FakeSpool({1: "a", 2: "b", 5: "e"}), logs.append
        )
        assert n == 1
        assert delivered.calls[-1][2] == "e"
        assert not any("gap" in m for m in logs)

    def test_a_delivery_failure_in_the_prefix_hides_the_gap(self, tmp_path, delivered):
        # The failure is the actionable line; reporting the hole behind it
        # would only invite skipping entries that were never tried.
        delivered.state["fail_on"] = "b"
        logs = []
        n = relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a", 2: "b", 5: "e"}),
            logs.append,
        )
        assert n == 1
        assert any("delivery of 00000002.csv failed" in m for m in logs)
        assert not any("gap" in m for m in logs)

    def test_gap_is_measured_from_the_cursor(self, tmp_path, delivered):
        write_cursor(tmp_path / "relay" / "box2.cursor", 2)
        logs = []
        relay_source(
            FakeCfg([PushTarget(spool_dir=Path("/spool/x"))]),
            _source(tmp_path),
            FakeSpool({1: "a", 2: "b", 4: "d"}),
            logs.append,
        )
        assert logs[0].startswith("box2: gap: expected 3, found 4")
        assert "--reset-cursor box2=3 to resume at 4" in logs[0]


class TestAdmin:
    def _cfg_with_source(self, tmp_path, targets):
        cfg = FakeCfg(targets)
        cfg.relays = [_source(tmp_path)]
        return cfg

    def test_reset_cursor_within_the_log(self, tmp_path):
        from slipstream.relay import reset_cursor

        cfg = self._cfg_with_source(tmp_path, [PushTarget(spool_dir=Path("/spool/x"))])
        reset_cursor(
            cfg, "box2", 2, lambda m: None, spool=FakeSpool({1: "", 2: "", 3: ""})
        )
        assert _cursor(tmp_path) == 2

    def test_reset_cursor_past_the_log_is_refused(self, tmp_path):
        from slipstream.relay import reset_cursor

        cfg = self._cfg_with_source(tmp_path, [PushTarget(spool_dir=Path("/spool/x"))])
        with pytest.raises(ValueError, match="holds entries up to 3"):
            reset_cursor(
                cfg, "box2", 4, lambda m: None, spool=FakeSpool({1: "", 3: ""})
            )
        assert _cursor(tmp_path) == 0

    def test_rebuild_rewinds_the_cursor_before_wiping(self, tmp_path, monkeypatch):
        """A wipe that fails must leave a cursor that replays, not one that
        points past rows nothing will restore."""
        from slipstream import spanner
        from slipstream.relay import rebuild_source

        write_cursor(tmp_path / "relay" / "box2.cursor", 3)

        def broken(spec):
            raise RuntimeError("DEADLINE_EXCEEDED")

        monkeypatch.setattr(spanner, "connect", broken)
        cfg = self._cfg_with_source(tmp_path, [PushTarget(spanner="p/i/d")])
        with pytest.raises(RuntimeError):
            rebuild_source(
                cfg, "box2", lambda m: None, spool=FakeSpool({1: "", 2: "", 3: ""})
            )
        assert _cursor(tmp_path) == 0

    def test_reset_cursor_below_the_retained_head_is_refused(self, tmp_path):
        from slipstream.relay import reset_cursor

        cfg = self._cfg_with_source(tmp_path, [PushTarget(spool_dir=Path("/spool/x"))])
        with pytest.raises(ValueError, match="retains its log from 40"):
            reset_cursor(
                cfg, "box2", 2, lambda m: None, spool=FakeSpool({40: "", 41: ""})
            )
        assert _cursor(tmp_path) == 0
        # The head itself is the useful value and stays allowed.
        reset_cursor(cfg, "box2", 39, lambda m: None, spool=FakeSpool({40: "", 41: ""}))
        assert _cursor(tmp_path) == 39

    def test_rebuild_refuses_before_wiping_what_cannot_be_replayed(self, tmp_path):
        from slipstream.relay import rebuild_source

        cfg = self._cfg_with_source(tmp_path, [PushTarget(spanner="p/i/d")])
        with pytest.raises(ValueError, match=r"entries 1\.\.3"):
            rebuild_source(cfg, "box2", lambda m: None, spool=FakeSpool({4: "", 5: ""}))

    def test_unknown_bot(self, tmp_path):
        from slipstream.relay import rebuild_source

        cfg = self._cfg_with_source(tmp_path, [PushTarget(spool_dir=Path("/spool/x"))])
        with pytest.raises(ValueError, match=r"no \[\[relay\]\] source"):
            rebuild_source(cfg, "nope", lambda m: None, spool=FakeSpool({1: ""}))


class TestRelayAll:
    def test_survives_a_broken_source(self, tmp_path, monkeypatch):
        from slipstream.relay import relay_all

        class Broken:
            def __init__(self, host, d):
                pass

            def list(self):
                raise RuntimeError("no route to host")

        monkeypatch.setattr("slipstream.relay.RemoteSpool", Broken)
        cfg = FakeCfg([PushTarget(spool_dir=Path("/spool/x"))])
        cfg.relays = [_source(tmp_path)]
        logs = []
        assert relay_all(cfg, logs.append) == 0
        assert any("no route to host" in m for m in logs)


class TestRemoteSpool:
    def test_list_parses_seqs_and_quotes_dir(self, monkeypatch):
        from slipstream.relay import RemoteSpool

        cmds = []

        def run(argv, **kwargs):
            cmds.append(argv)
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="00000002.csv\n00000001.csv\n00000003.csv.tmp\n.lock\n",
                stderr="",
            )

        monkeypatch.setattr("slipstream.relay.subprocess.run", run)
        spool = RemoteSpool("box2", "~/my outbox")
        assert spool.list() == [1, 2]
        assert cmds[0][:2] == ["ssh", "box2"]
        # ~ must stay bare for the remote shell to expand it; the rest is quoted.
        assert cmds[0][2] == "ls -1 ~/'my outbox'"
        spool.fetch(1)
        assert cmds[1][2] == "cat ~/'my outbox'/00000001.csv"

    def test_absolute_dir_is_quoted(self, monkeypatch):
        from slipstream.relay import RemoteSpool

        cmds = []
        monkeypatch.setattr(
            "slipstream.relay.subprocess.run",
            lambda argv, **k: (
                cmds.append(argv)
                or subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
            ),
        )
        RemoteSpool("box2", "/Users/x/out box").fetch(9)
        assert cmds[0][2] == "cat '/Users/x/out box'/00000009.csv"

    def test_missing_dir_is_an_error(self, monkeypatch):
        from slipstream.relay import RemoteSpool

        def run(argv, **kwargs):
            raise subprocess.CalledProcessError(2, argv, stderr="not found")

        monkeypatch.setattr("slipstream.relay.subprocess.run", run)
        with pytest.raises(subprocess.CalledProcessError):
            RemoteSpool("box2", "~/missing").list()


def test_relay_stages_every_entry_over_one_spanner_session(tmp_path, monkeypatch):
    """The real spanner session, against a recording fake: one schema check
    and one aggregation however many entries the cycle carries."""
    from slipstream import spanner
    from tests.test_spanner import FakeDb, T0, _csv

    db = FakeDb([[("slipstream",), ("benchmarks",), ("meta",)], [(T0,)], [], [], []])
    monkeypatch.setattr(spanner, "connect", lambda spec: db)
    logs = []
    n = relay_source(
        FakeCfg([PushTarget(spanner="p/i/d")]),
        _source(tmp_path),
        FakeSpool({1: _csv({}), 2: _csv({"run": "2"})}),
        logs.append,
    )
    assert n == 2
    assert [c[1] for c in db.of("upsert")] == ["slipstream", "slipstream"]
    assert sum("INFORMATION_SCHEMA" in c[1] for c in db.of("query")) == 1
    assert sum("MAX(imported_at)" in c[1] for c in db.of("query")) == 1
    assert logs == ["box2: 2 rows staged for box2, aggregated"]
    assert db.calls[-1] == ("close",)


def test_spooled_push_is_relayed_intact(tmp_path, store):
    """End to end: two spool_dir pushes on the source, relayed into a real
    spool target on this machine."""
    from slipstream.push import push
    from tests.test_push import VALID, _rows, _seed

    outbox = tmp_path / "outbox"
    cfg = PushConfig(bot_name="box2", targets=[PushTarget(spool_dir=outbox)])
    _seed(store, [100, 200])
    push(store, ["v8"], cfg, VALID, "arm64")
    _seed(store, [300])
    push(store, ["v8"], cfg, VALID, "arm64")

    class LocalSpool:
        def list(self):
            from slipstream.push import parse_seq

            return sorted(s for s in (parse_seq(p.name) for p in outbox.iterdir()) if s)

        def fetch(self, seq):
            from slipstream.push import seq_name

            return (outbox / seq_name(seq)).read_text()

    received = tmp_path / "received"
    target = PushTarget(spool_dir=received)
    logs = []
    n = relay_source(FakeCfg([target]), _source(tmp_path), LocalSpool(), logs.append)
    assert n == 2
    assert _cursor(tmp_path) == 2
    from slipstream.push import parse_seq

    relayed = sorted(p for p in received.iterdir() if parse_seq(p.name))
    rows = [r for f in relayed for r in _rows(f.read_text())]
    assert [r["commit_id"] for r in rows] == ["100", "200", "300"]

    # A second cycle with nothing new delivers nothing.
    assert (
        relay_source(FakeCfg([target]), _source(tmp_path), LocalSpool(), logs.append)
        == 0
    )
