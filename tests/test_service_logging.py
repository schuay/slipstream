# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

import io
import logging
import re
import subprocess
import sys

import pytest
from rich.console import Console

from slipstream import service_logging as logs


@pytest.fixture
def terminal(monkeypatch):
    terminal = logs._Terminal()
    monkeypatch.setattr(logs, "_terminal", terminal)
    monkeypatch.delenv("NO_COLOR", raising=False)
    return terminal


@pytest.mark.parametrize("role", ["build", "bench", "transfer", "deliver", "push"])
def test_plain_records_are_literal_and_explicit(role, capsys, terminal):
    log = logs.EventLog(role)
    log("failed is a filename, [red] is literal\nsecond line")
    log(logs.warning("retry"))
    log(logs.error("failure\nmore detail"))
    out = capsys.readouterr()
    assert not out.out
    lines = out.err.splitlines()
    assert len(lines) == 5
    assert all(re.match(r"\d\d:\d\d:\d\d " + role + " ", line) for line in lines)
    assert lines[0].endswith("failed is a filename, [red] is literal")
    assert "WARNING" not in lines[0]
    assert lines[2].endswith("WARNING: retry")
    assert all("ERROR:" in line for line in lines[3:])
    assert "\x1b" not in out.err


@pytest.mark.parametrize("tty", [True, False])
@pytest.mark.parametrize("no_color", [True, False])
def test_progress_and_file_are_independent(
    terminal, tmp_path, monkeypatch, tty, no_color
):
    stream = io.StringIO()
    terminal.console = Console(file=stream, force_terminal=tty, color_system="standard")
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")
    path = tmp_path / "bench.log"
    log = logs.EventLog("bench", mirror_path=path)
    log(logs.message(("v8 123", "bold"), (" (abc)", "dim")))
    with log.progress("    js3 (default):"):
        log.dot()
        log.dot()
        log.finish("OK (2s)")
    output = stream.getvalue()
    assert ("\x1b" in output) == (tty and not no_color)
    assert (".." in output) == tty
    assert output.count("OK (2s)") == 1
    persistent = path.read_text()
    assert "\x1b" not in persistent and ".." not in persistent
    assert persistent.splitlines()[1].endswith("bench     js3 (default): OK (2s)")


def test_diagnostic_from_another_logger_closes_partial_line(terminal, tmp_path):
    stream = io.StringIO()
    terminal.console = Console(file=stream, force_terminal=True)
    log = logs.EventLog("bench", mirror_path=tmp_path / "bench.log")
    with log.progress("    js3:"):
        log.dot()
        logs.EventLog("transfer").error("connection lost")
        log.dot()
        log.finish("ERRORS (1s)", failed=True)
    output = logs._clean(stream.getvalue())
    lines = output.splitlines()
    assert lines[0] == "    js3: ."
    assert "transfer ERROR: connection lost" in lines[1]
    assert "bench     js3: ERRORS (1s)" in lines[2]
    assert output.count("ERRORS") == 1


@pytest.mark.parametrize(
    "exception, outcome", [(KeyboardInterrupt, "INTERRUPTED"), (RuntimeError, "FAILED")]
)
@pytest.mark.parametrize("tty", [True, False])
def test_exception_finishes_progress(terminal, tmp_path, exception, outcome, tty):
    stream = io.StringIO()
    terminal.console = Console(file=stream, force_terminal=tty)
    path = tmp_path / "bench.log"
    log = logs.EventLog("bench", mirror_path=path)
    with pytest.raises(exception):
        with log.progress("    js3:"):
            log.dot()
            raise exception()
    assert stream.getvalue().count(outcome) == 1
    assert path.read_text().count(outcome) == 1
    assert terminal.partial is None


def test_file_rotation_and_no_extra_stream(terminal, tmp_path, capsys):
    path = tmp_path / "service.log"
    log = logs.EventLog("deliver", path)
    assert log.handler.maxBytes == 5 * 1024 * 1024
    assert log.handler.backupCount == 3
    log.handler.maxBytes = 100
    for i in range(12):
        log(logs.success(f"delivered batch {i} " + "x" * 60))
    log.close()
    assert len(list(tmp_path.iterdir())) == 4
    assert not capsys.readouterr().err
    assert "delivered batch 11" in path.read_text()
    assert all("\x1b" not in p.read_text() for p in tmp_path.iterdir())


def test_controls_are_removed_but_brackets_remain(terminal, capsys):
    logs.EventLog("build")("\x1b[31mfailed\x1b[0m\r [red] /x[y]\x1b]0;title\x07")
    assert capsys.readouterr().err.endswith("build failed [red] /x[y]\n")


def test_prefix_adapter_preserves_severity_and_spans():
    original = logs.message(("bad", "bold"), level=logging.ERROR)
    prefixed = logs.prefixed("engine: ", original)
    assert prefixed == "engine: bad"
    assert prefixed.level == logging.ERROR
    assert prefixed.text.spans[0].start == len("engine: ")


def test_redirected_service_flushes_before_exit(tmp_path):
    # launchd's StandardErrorPath is a regular file, with stdout independent.
    path = tmp_path / "stderr.log"
    script = """
import os
from slipstream.service_logging import EventLog
log = EventLog('transfer')
log('ready')
with log.progress('copy:'):
    log.dot()
    log.finish('OK')
os._exit(0)
"""
    with path.open("w") as stderr:
        result = subprocess.run(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=stderr
        )
    assert result.returncode == 0 and not result.stdout
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("transfer ready")
    assert lines[1].endswith("transfer copy: OK")


def test_collector_suites_reach_both_plain_sinks(
    config, tmp_path, monkeypatch, capsys, terminal
):
    from types import SimpleNamespace
    from slipstream.collector import BenchCollector
    from slipstream.config import EngineConfig
    from slipstream.models import CommitKey
    from slipstream.runners import RunResult

    engine = EngineConfig(
        name="v8",
        src_dir=tmp_path,
        build_cmd="true",
        binary_path="out/d8",
        id_regex=r"#([0-9]+)",
    )
    config.engines["v8"] = engine
    collector = BenchCollector(config)
    collector.cool_down = lambda log: 0

    def run(request):
        collector.output.dot()
        collector._log(logs.error("literal [red] diagnostic"))
        return RunResult(False, [])

    monkeypatch.setattr(collector, "_runner", lambda engine: SimpleNamespace(run=run))
    try:
        outcome = collector._run_benchmarks(engine, CommitKey(0, 100), tmp_path)
        assert outcome.configs_ok == 0 and outcome.configs_total > 0
        output = capsys.readouterr()
        assert not output.out and "..." not in output.err
        persisted = collector.log_file.read_text()
        assert persisted == output.err
        assert persisted.count("ERRORS (") == outcome.configs_total
        assert "literal [red] diagnostic" in persisted
    finally:
        collector.store.close()


def test_prefix_dimming_does_not_dim_message(terminal):
    terminal.console = Console(file=io.StringIO(), force_terminal=True, record=True)
    logs.EventLog("build")("ordinary prose")
    # Inspect Rich's rendered segments, before ANSI serialization.
    prose = [s for s in terminal.console._record_buffer if "ordinary prose" in s.text]
    assert prose and all(not s.style or not s.style.dim for s in prose)
