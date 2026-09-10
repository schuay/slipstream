# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
import subprocess
import time as time_mod
import tomllib
from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import load_config
from .collector import BenchCollector, FetchError
from .analyzer import PerfAnalyzer
from . import host as host_mod

app = typer.Typer(
    name="slipstream",
    help="JS engine benchmark collector and analyzer.",
    no_args_is_help=True,
)


def _load_config(path: Optional[Path]):
    """load_config with a one-line error instead of a traceback."""
    try:
        return load_config(path)
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as e:
        typer.echo(f"Config error: {e}", err=True)
        raise typer.Exit(1)


def _open_collector(cfg, **kwargs):
    """BenchCollector, reporting an identity mismatch in one line.

    Its constructor opens the store, so the D014 "db file copied between
    machines" guard reaches bench, watch, build and bus status through here
    rather than as a traceback.
    """
    from .store import StoreError

    try:
        return BenchCollector(cfg, **kwargs)
    except StoreError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _open_store(cfg, **kwargs):
    """Open this machine's store, reporting an identity mismatch in one line."""
    from .store import CommitStore, StoreError

    try:
        return CommitStore(
            cfg.metadata_dir / "slipstream.db", bot=cfg.bot_name, **kwargs
        )
    except StoreError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def _shutdown_flag(label: str = "current commit"):
    """Install SIGINT/SIGTERM handlers; returns a callable reading the flag.

    A callable rather than a variable so the flag reaches the collector's
    per-commit path, where a wait for the machine lock can last hours.
    """
    import signal

    state = {"stop": False}

    def handle(signum, frame):
        if state["stop"]:
            raise SystemExit(1)
        state["stop"] = True
        typer.echo(f"\nShutdown requested — finishing {label} (^C again to force)...")

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)
    return lambda: state["stop"]


def _host_preflight(cfg):
    """Report macOS host power state; stop the session if configured to."""
    ok = host_mod.preflight(cfg.host.preflight, lambda msg: typer.echo(msg))
    if not ok and cfg.host.preflight == "abort":
        typer.echo(
            "Error: host is not benchmark-ready. Run 'slipstream host --apply', "
            'or set [host] preflight = "warn" to run anyway.',
            err=True,
        )
        raise typer.Exit(1)


@app.command()
def bench(
    engine: Annotated[str, typer.Argument(help="Engine to benchmark: v8 or jsc")],
    start: Annotated[int, typer.Argument(help="Start commit ID")],
    end: Annotated[int, typer.Argument(help="End commit ID")],
    step: int = typer.Option(1, help="Sample every N-th commit"),
    runs: int = typer.Option(3, help="Benchmark iterations per commit"),
    config: Optional[Path] = typer.Option(
        None, help="User config path (default: ~/.config/slipstream/config.toml)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would run without executing"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show build output"),
    clear: bool = typer.Option(
        False, "--clear", help="Clear results directory before starting"
    ),
):
    """Collect benchmark results across a range of commits."""
    cfg = _load_config(config)
    if engine not in cfg.engines:
        typer.echo(
            f"Error: engine '{engine}' not configured. Available: {list(cfg.engines)}",
            err=True,
        )
        raise typer.Exit(1)

    _host_preflight(cfg)

    from .lock import EXIT_BUSY, LockBusy

    collector = _open_collector(cfg, dry_run=dry_run, verbose=verbose, role="bench")
    try:
        collector.collect(
            engine,
            start,
            end,
            step=step,
            runs=runs,
            clear=clear,
            should_stop=_shutdown_flag(),
        )
    except ValueError as e:
        # An engine configured without src_dir, on a box that only benches
        # artifacts. Every command that needs git says so by name.
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except LockBusy as e:
        # An operator command does not queue behind a daemon's bench.
        typer.echo(f"Error: {e}", err=True)
        typer.echo(
            "Run 'slipstream bus pause' to have the daemons stand down.", err=True
        )
        raise typer.Exit(EXIT_BUSY)


@app.command()
def analyze(
    csv_path: Annotated[
        Optional[Path], typer.Argument(help="Path to raw_results CSV (omit to use DB)")
    ] = None,
    engine: Optional[str] = typer.Option(
        None, help="Engine (required when using DB, ignored with CSV)"
    ),
    commits: Optional[str] = typer.Option(
        None, help="Glob for commit info CSVs (CSV mode only)"
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
    min_change: float = typer.Option(0.01, help="Minimum change threshold (0.01 = 1%)"),
    penalty: float = typer.Option(
        3.0, help="PELT penalty (higher = fewer change points)"
    ),
    min_effect: float = typer.Option(0.5, help="Minimum Cohen's d effect size"),
    group: bool = typer.Option(False, "--group", help="Group results by commit"),
    include_bench: Optional[list[str]] = typer.Option(
        None, "--include-bench", help="Include only matching benchmarks"
    ),
    exclude_bench: Optional[list[str]] = typer.Option(
        None, "--exclude-bench", help="Exclude matching benchmarks"
    ),
    include_score: Optional[list[str]] = typer.Option(
        None, "--include-score", help="Include only matching score types"
    ),
    exclude_score: Optional[list[str]] = typer.Option(
        None, "--exclude-score", help="Exclude matching score types"
    ),
    since: Optional[str] = typer.Option(
        None,
        help="Only include commits on or after this date (YYYY-MM-DD or '2 weeks ago')",
    ),
    until: Optional[str] = typer.Option(
        None,
        help="Only include commits on or before this date (YYYY-MM-DD or 'yesterday')",
    ),
):
    """Detect change points in benchmark results (DB or CSV)."""

    cfg = _load_config(config)
    analyzer = PerfAnalyzer(
        min_change=min_change, penalty=penalty, min_effect_size=min_effect
    )

    if csv_path:
        if not analyzer.load_results(csv_path):
            raise typer.Exit(1)
        commit_pattern = commits or str(cfg.metadata_dir / "commit-infos-*.csv")
        analyzer.load_commit_infos(commit_pattern)
    else:
        if not engine:
            typer.echo("Error: --engine required when analyzing from DB", err=True)
            raise typer.Exit(1)
        store = _open_store(cfg, readonly=True)
        analyzer.load_from_db(store, engine)

    since_date = _parse_date(since) if since else None
    until_date = _parse_date(until) if until else None
    if since_date or until_date:
        analyzer.filter_by_date(since=since_date, until=until_date)

    results = analyzer.analyze(
        include_bench=include_bench,
        exclude_bench=exclude_bench,
        include_score=include_score,
        exclude_score=exclude_score,
    )
    analyzer.print_report(results, group_by_commit=group)


@app.command(name="next-range")
def next_range(
    engine: Annotated[str, typer.Argument(help="Engine name: v8 or jsc")],
    config: Optional[Path] = typer.Option(None, help="User config path"),
    no_fetch: bool = typer.Option(False, "--no-fetch", help="Skip git fetch"),
):
    """Print the commit range from latest processed to newest in git."""
    cfg = _load_config(config)
    if engine not in cfg.engines:
        typer.echo(f"Error: engine '{engine}' not configured.", err=True)
        raise typer.Exit(1)

    if not no_fetch:
        typer.echo("Fetching origin main...")

    collector = _open_collector(cfg)
    try:
        start_id, end_id = collector.find_frontier(engine, fetch=not no_fetch)
    except FetchError:
        # find_frontier already reported the underlying git error.
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if start_id is None:
        typer.echo("No processed commits found in store for this engine.")
        raise typer.Exit(1)
    if end_id is None:
        typer.echo("Could not extract commit ID from origin/main HEAD.")
        raise typer.Exit(1)

    typer.echo(f"Latest done : {start_id}")
    typer.echo(f"Newest in git: {end_id}")
    typer.echo(f"\n  slipstream bench {engine} {start_id} {end_id}")


def _parse_date(value: str) -> str:
    """Parse a date string like '2026-01-15' or '2 weeks ago' into YYYY-MM-DD."""
    import dateparser

    dt = dateparser.parse(value)
    if dt is None:
        raise typer.BadParameter(f"Cannot parse date: {value!r}")
    return dt.strftime("%Y-%m-%d")


def _parse_interval(value: str) -> int:
    """Parse an interval string like '30m', '2h', '90s' into seconds."""
    m = re.match(r"^(\d+)\s*([smh]?)$", value.strip())
    if not m:
        raise typer.BadParameter(f"Invalid interval: {value!r} (use e.g. 30m, 2h, 90s)")
    amount, unit = int(m.group(1)), m.group(2) or "m"
    return amount * {"s": 1, "m": 60, "h": 3600}[unit]


@app.command()
def watch(
    engines: Annotated[
        Optional[list[str]],
        typer.Argument(help="Engines to watch (default: all configured)"),
    ] = None,
    interval: str = typer.Option("30m", help="Poll interval, e.g. 30m, 2h, 90s"),
    step: int = typer.Option(1, help="Sample every N-th commit"),
    runs: int = typer.Option(3, help="Benchmark iterations per commit"),
    once: bool = typer.Option(False, "--once", help="Single poll cycle then exit"),
    no_push: bool = typer.Option(False, "--no-push", help="Skip all pushing"),
    reset_cursor: Optional[str] = typer.Option(
        None,
        "--reset-cursor",
        metavar="ENGINE[=ID]",
        help="Re-bench a bus-driven engine from above ID (default: from the start)",
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show build output"),
):
    """Watch for new commits and benchmark them incrementally.

    An engine claimed by a bus source is bus-driven: it benches published
    artifacts in commit order from its cursor. An engine without one is
    git-driven and builds what it measures, which is how a single box runs.
    """
    import time

    cfg = _load_config(config)
    interval_secs = _parse_interval(interval)
    engine_names = engines or list(cfg.engines.keys())

    for name in engine_names:
        if name not in cfg.engines:
            typer.echo(f"Error: engine '{name}' not configured.", err=True)
            raise typer.Exit(1)

    def source_for(name):
        return cfg.bus.source_for(name) if cfg.bus else None

    if step != 1 and any(source_for(name) for name in engine_names):
        # In git mode --step samples the commit list. A cursor equivalent would
        # advance past unbenched entries irreversibly, destroying the commit
        # overlap between the two boxes, which is the whole deliverable.
        typer.echo(
            "Error: --step does not apply to bus-driven engines; "
            "use --reset-cursor to re-bench a range.",
            err=True,
        )
        raise typer.Exit(1)

    collector = _open_collector(
        cfg, dry_run=dry_run, verbose=verbose, role="watch", wait_for_lock=True
    )

    consumer = None
    if cfg.bus is not None and cfg.bus.sources:
        # Built whenever the machine has any bus source, not only when a
        # watched engine has one: --reset-cursor names an engine of its own.
        from .consumer import BusConsumer, ConsumerError

        try:
            consumer = BusConsumer(
                cfg,
                collector,
                log=lambda m: typer.echo(f"  {m}"),
                dry_run=dry_run,
                interval_secs=interval_secs,
            )
        except ConsumerError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    if reset_cursor is not None:
        name, _, raw = reset_cursor.partition("=")
        source = source_for(name) if name in cfg.engines else None
        if source is None or consumer is None:
            typer.echo(
                f"Error: '{name}' has no bus source, so it has no cursor.", err=True
            )
            raise typer.Exit(1)
        if raw and not raw.isdigit():
            raise typer.BadParameter("--reset-cursor takes ENGINE or ENGINE=ID")
        if dry_run:
            typer.echo(
                f"{name}: would reset the cursor to {raw or 0} (source {source.name})"
            )
            return
        consumer.set_cursor(source, name, int(raw) if raw else 0)
        typer.echo(f"{name}: cursor reset to {raw or 0} (source {source.name})")
        typer.echo(
            "Entries at or below it are still skipped unless their scores are "
            f"cleared: slipstream clear {name} <first> <last>"
        )
        return

    # Verify cold-start: each git-driven engine must have at least one done
    # commit. A bus-driven engine starts from a cursor instead.
    for name in engine_names:
        if source_for(name) is None and (
            collector.store.max_done_commit_id(name, cfg.platform) is None
        ):
            typer.echo(
                f"Error: no done commits for '{name}'. "
                f"Run 'slipstream bench {name} <start> <end>' first.",
                err=True,
            )
            raise typer.Exit(1)

    _host_preflight(cfg)

    # Background pusher: drains new scores after each commit without delaying
    # the next build. A single worker with coalescing wakes keeps pushes
    # regular even when a target is slow. Disabled for dry runs and when
    # pushing is turned off or unconfigured.
    push_enabled = cfg.push and not no_push and not dry_run
    pusher = None
    on_commit_done = None
    if push_enabled:
        from .pusher import BackgroundPusher

        pusher = BackgroundPusher(
            cfg, engine_names, log=lambda msg: typer.echo(f"  {msg}")
        )
        pusher.start()
        on_commit_done = pusher.notify
    if consumer is not None:
        consumer.on_commit_done = on_commit_done

    should_stop = _shutdown_flag()

    typer.echo(
        f"Watching {', '.join(engine_names)} "
        f"(interval={interval}, step={step}, runs={runs})"
    )

    try:
        while not should_stop():
            t0 = time.monotonic()
            for name in engine_names:
                if should_stop():
                    break
                source = source_for(name)
                if source is not None:
                    result = consumer.drain(
                        source,
                        name,
                        runs,
                        should_stop,
                        lambda: collector.lock.acquire(
                            should_stop,
                            wait=True,
                            log=lambda m: typer.echo(f"  {m}"),
                        ),
                        collector.lock.release,
                    )
                    if dry_run:
                        # It has already said what it would do.
                        continue
                    if result.benched:
                        typer.echo(
                            f"  {name}: {result.benched} benched from {source.name}"
                        )
                    elif result.error:
                        # Reported already; do not follow it with "up to date".
                        pass
                    else:
                        typer.echo(f"  {name}: up to date with {source.name}")
                    continue
                try:
                    last_done, head_id = collector.find_frontier(name)
                except FetchError:
                    typer.echo(f"  {name}: skipping until fetch succeeds")
                    continue
                except ValueError as e:
                    typer.echo(f"  {name}: {e}")
                    continue
                if last_done is None or head_id is None:
                    typer.echo(f"  {name}: could not determine frontier, skipping")
                    continue
                if head_id <= last_done:
                    typer.echo(f"  {name}: up to date at {last_done}")
                    continue

                typer.echo(f"  {name}: new commits {last_done + 1}..{head_id}")
                collector.collect(
                    name,
                    last_done,
                    head_id,
                    step=step,
                    runs=runs,
                    include_start=False,
                    on_commit_done=on_commit_done,
                    should_stop=should_stop,
                )

            if once:
                break

            # Interruptible sleep, accounting for time spent working
            work_time = time.monotonic() - t0
            sleep_secs = int(interval_secs - min(interval_secs, work_time))
            for _ in range(sleep_secs):
                if should_stop():
                    break
                time.sleep(1)
    finally:
        if pusher is not None:
            typer.echo("Draining background pushes...")
            pusher.close()

    typer.echo("Watch stopped.")


@app.command()
def clear(
    engine: Annotated[str, typer.Argument(help="Engine name (v8, jsc)")],
    start: Annotated[int, typer.Argument(help="First commit ID to clear")],
    end: Annotated[
        Optional[int], typer.Argument(help="Last commit ID (default: just start)")
    ] = None,
    config: Optional[Path] = typer.Option(None, help="User config path"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask"),
):
    """Forget a range of commits so they can be measured again.

    Drops their scores, processing state, push state, provenance and stdout
    directories. Pair it with 'watch --reset-cursor' on a bus-driven engine:
    the reset alone does nothing, because a consumer skips a commit that is
    already done without fetching it.

    Not 'bench --clear' for a commit that came off the bus: that rebuilds
    locally, so the repaired commit's numbers would come from a different
    binary than its neighbours, and an archive defect would then look like a
    microarchitecture difference.
    """
    import shutil as _shutil

    cfg = _load_config(config)
    if engine not in cfg.engines:
        typer.echo(f"Error: engine '{engine}' not configured.", err=True)
        raise typer.Exit(1)
    end = start if end is None else end
    if end < start:
        typer.echo("Error: end must not be below start.", err=True)
        raise typer.Exit(1)

    store = _open_store(cfg)
    ids = [
        r["commit_id"]
        for r in store.get_commits_in_range(engine, start - 1, end)
        if store.is_done(engine, cfg.platform, r["commit_id"])
    ]
    if not ids:
        typer.echo(f"Nothing done for {engine} in {start}..{end}.")
        store.close()
        return
    if not yes:
        typer.confirm(
            f"Clear {len(ids)} commits of {engine} ({ids[0]}..{ids[-1]})?", abort=True
        )

    store.clear_range(engine, cfg.platform, ids)
    for cid in ids:
        path = cfg.commit_results_dir(engine, cid)
        if path.exists():
            _shutil.rmtree(path)
    typer.echo(f"Cleared {len(ids)} commits. They will be measured again.")
    store.close()


@app.command(name="import")
def import_csv(
    engine: Annotated[str, typer.Argument(help="Engine name (v8, jsc)")],
    files: Annotated[list[Path], typer.Argument(help="CSV files to import")],
    bot: str = typer.Option(
        ..., "--bot", help="Bot the scores belong to; must be this machine's"
    ),
    allow_legacy: bool = typer.Option(
        False,
        "--allow-legacy",
        help="Accept a CSV with no bot column, trusting --bot",
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
    platform_name: Optional[str] = typer.Option(
        None, "--platform", help="Platform override (default: inferred from filename)"
    ),
):
    """Import existing CSV result files into the SQLite scores table.

    One db holds one bot, so this refuses anything that is not this machine's
    own data. Foreign scores would land under the local bot name and could not
    be found or separated again afterwards; compare two machines in the perf
    frontend, which keys on bot, or with 'slipstream analyze <csv>'.
    """
    import csv as _csv

    from . import compat

    cfg = _load_config(config)
    if engine not in cfg.engines:
        typer.echo(f"Error: engine '{engine}' not configured.", err=True)
        raise typer.Exit(1)

    store = _open_store(cfg)
    local_bot = store.bot
    if local_bot is None:
        typer.echo(
            "Error: this db has no bot name. Set a top-level bot_name in the "
            "config and re-run.",
            err=True,
        )
        raise typer.Exit(1)
    if bot != local_bot:
        typer.echo(
            f"Error: --bot {bot!r} is not this machine ({local_bot!r}). "
            "One db holds one bot.",
            err=True,
        )
        raise typer.Exit(1)

    total = 0
    for path in files:
        if not path.exists():
            typer.echo(f"  {path}: not found, skipping")
            continue

        # Infer platform from filename: raw_results-v8-arm64-... -> arm64
        plat = platform_name
        if not plat:
            m = re.search(r"raw_results-\w+-([\w]+)", path.stem)
            plat = m.group(1) if m else "unknown"

        run_counters: dict[tuple, int] = {}
        rows: list[tuple] = []

        with open(path) as f:
            first = next(_csv.reader(f), [])
            f.seek(0)
            try:
                fieldnames, has_bot = compat.sniff(first)
            except ValueError as e:
                typer.echo(f"  {path}: {e}, skipping")
                continue
            if not has_bot and not allow_legacy:
                typer.echo(
                    f"Error: {path} has no bot column, so --bot cannot be "
                    "checked against it. Pass --allow-legacy if it really is "
                    f"{bot}'s data.",
                    err=True,
                )
                raise typer.Exit(1)

            for raw_row in _csv.DictReader(f, fieldnames=fieldnames):
                try:
                    row = {
                        k.strip(): (v or "").strip()
                        for k, v in raw_row.items()
                        if k is not None
                    }
                    cid = int(row["commit_id"])
                    suite = row.get("b_type", "js2")
                    flags = row.get("flags", "default")
                    benchmark = row["benchmark"]
                    metric = row["score_type"]
                    score = float(row["score"])
                except (ValueError, KeyError):
                    continue
                row_bot = row.get("bot")
                # `is not None`, not truthiness: a blank cell in a 7-column
                # file (hand-concatenated, or partially populated) would
                # otherwise bypass the check and land under the local bot.
                # analyzer.load_results already refuses the same file.
                if row_bot is not None and row_bot != bot:
                    typer.echo(
                        f"Error: {path} holds {row_bot or '(blank)'}'s scores, "
                        f"not {bot}'s.",
                        err=True,
                    )
                    raise typer.Exit(1)

                key = (cid, suite, flags, benchmark, metric)
                run_counters[key] = run_counters.get(key, 0) + 1
                rows.append(
                    (cid, suite, flags, benchmark, metric, run_counters[key], score)
                )

        if rows:
            # The commit timestamps are read after the bot checks: a refused
            # file must not have cost a query.
            ts_lookup = {
                r["commit_id"]: r["timestamp"] for r in store.get_all_commits(engine)
            }
            store.bulk_insert_scores(
                engine, plat, [(*r, ts_lookup.get(r[0], 0)) for r in rows]
            )
            for cid in {r[0] for r in rows}:
                # No status: these scores come with no run of our own to report,
                # and must not overwrite the verdict of one.
                store.mark_done(engine, plat, cid)
        typer.echo(f"  {path}: {len(rows)} scores ({plat})")
        total += len(rows)

    typer.echo(f"Total: {total} scores imported")
    store.close()


@app.command(name="export")
def export_csv(
    engine: Annotated[str, typer.Argument(help="Engine name (v8, jsc)")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output CSV path (default: stdout)"),
    ] = None,
    commit_infos: Annotated[
        Optional[Path],
        typer.Option("--commit-infos", help="Also write commit-infos CSV to this path"),
    ] = None,
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Export scores and commit metadata to CSV (skiz-compatible format)."""
    import csv as _csv
    import sys

    from . import compat

    cfg = _load_config(config)
    store = _open_store(cfg, readonly=True)

    local_bot = store.bot
    if local_bot is None:
        # An empty column would fail this file's own re-import.
        typer.echo(
            "Error: this db has no bot name. Set a top-level bot_name in the "
            "config and re-run.",
            err=True,
        )
        raise typer.Exit(1)

    # Only export benchmarks in the known name lists
    valid_names: set[str] = set()
    for bench_cfg in cfg.benchmarks.values():
        valid_names.update(bench_cfg.names)
    rows = store.export_compat_rows(engine, valid_names)

    dest = open(output, "w", newline="") if output else sys.stdout
    try:
        w = _csv.writer(dest)
        w.writerow(compat.FIELDS_7COL)
        for r in rows:
            w.writerow([*r, local_bot])
    finally:
        if output:
            dest.close()

    count = len(rows)
    if output:
        typer.echo(f"{count} scores written to {output}")
    else:
        typer.echo(f"{count} scores", err=True)

    if commit_infos:
        commits = store.get_all_commits(engine)
        with open(commit_infos, "w", newline="") as f:
            w = _csv.writer(f)
            for c in commits:
                w.writerow(
                    [c["commit_id"], c["hash"], c["date"], c["title"], c["timestamp"]]
                )
        typer.echo(f"{len(commits)} commits written to {commit_infos}")

    store.close()


@app.command()
def relay(
    interval: str = typer.Option("10m", help="Poll interval, e.g. 10m, 1h"),
    once: bool = typer.Option(False, "--once", help="Single cycle then exit"),
    rebuild: Optional[str] = typer.Option(
        None,
        "--rebuild",
        metavar="BOT",
        help="Wipe BOT's Spanner staging rows and replay its whole log;"
        " obsolete aggregate keys are not removed",
    ),
    reset_cursor: Optional[str] = typer.Option(
        None,
        "--reset-cursor",
        metavar="BOT[=SEQ]",
        help="Replay BOT's log from SEQ + 1 (default 0: all of it), without a wipe",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="With --rebuild: wipe even though the source can no longer replay"
        " its whole log",
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Drain remote spool logs into this machine's push targets.

    Sources are the config's relay entries.
    """
    import time

    cfg = _load_config(config)
    if not cfg.relays:
        typer.echo("Error: no [[relay]] sources in config.", err=True)
        raise typer.Exit(1)
    if not cfg.push:
        typer.echo("Error: [push] is required to relay.", err=True)
        raise typer.Exit(1)
    if rebuild and reset_cursor:
        typer.echo("Error: --rebuild already resets the cursor.", err=True)
        raise typer.Exit(1)
    if force and not rebuild:
        typer.echo("Error: --force applies to --rebuild.", err=True)
        raise typer.Exit(1)

    from .relay import (
        parse_cursor_arg,
        rebuild_source,
        relay_all,
        reset_cursor as do_reset_cursor,
    )

    def echo(msg: str) -> None:
        typer.echo(f"  {msg}")

    try:
        if rebuild:
            rebuild_source(cfg, rebuild, echo, force=force)
        elif reset_cursor:
            do_reset_cursor(cfg, *parse_cursor_arg(reset_cursor), echo)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    interval_secs = _parse_interval(interval)
    try:
        while True:
            relay_all(cfg, echo)
            if once:
                break
            time.sleep(interval_secs)
    except KeyboardInterrupt:
        typer.echo("Relay stopped.")


def _parse_engine_id(value: str, flag: str) -> tuple[str, int]:
    """Parse an ENGINE=ID argument."""
    engine, _, raw = value.partition("=")
    if not engine or not raw.isdigit():
        raise typer.BadParameter(f"{flag} takes ENGINE=ID, e.g. v8=109680")
    return engine, int(raw)


@app.command()
def build(
    engines: Annotated[
        Optional[list[str]],
        typer.Argument(help="Engines to build (default: the configured build engines)"),
    ] = None,
    interval: str = typer.Option("30m", help="Poll interval, e.g. 30m, 2h"),
    once: bool = typer.Option(False, "--once", help="Single cycle then exit"),
    retry: Optional[str] = typer.Option(
        None,
        "--retry",
        metavar="ENGINE=ID",
        help="Allow one more attempt at a commit whose build failed",
    ),
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Check this machine can build and publish; builds nothing",
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Echo each build command; its output goes to the per-commit log",
    ),
):
    """Build commits and publish the artifacts to the bus."""
    import time

    from .builder import Builder
    from .bus import Bus

    cfg = _load_config(config)
    if cfg.bus is None:
        typer.echo("Error: no [bus] section in config.", err=True)
        raise typer.Exit(1)
    engine_names = engines or cfg.build.engines
    if not engine_names:
        typer.echo("Error: no engines to build; set [build] engines.", err=True)
        raise typer.Exit(1)
    for name in engine_names:
        if name not in cfg.engines:
            typer.echo(f"Error: engine '{name}' not configured.", err=True)
            raise typer.Exit(1)

    from .store import StoreError

    interval_secs = _parse_interval(interval)
    try:
        builder = Builder(
            cfg,
            Bus(cfg.bus.root),
            verbose=verbose,
            log=lambda msg: typer.echo(f"  {msg}"),
            interval_secs=interval_secs,
        )
    except StoreError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if retry:
        engine, commit_id = _parse_engine_id(retry, "--retry")
        if not builder.store.request_build_retry(engine, commit_id):
            typer.echo(
                f"Error: {engine} {commit_id} has no recorded build failure.", err=True
            )
            raise typer.Exit(1)
        typer.echo(f"{engine} {commit_id} will be attempted again.")
        # A commit-id cursor cannot see anything at or below it, so a
        # republished entry sits unread unless every consumer is reset.
        typer.echo(
            f"Each consumer needs: slipstream watch --reset-cursor "
            f"{engine}={commit_id - 1}"
        )
        return

    if probe:
        _build_probe(builder, engine_names)
        return

    _host_preflight(cfg)
    should_stop = _shutdown_flag("current build")
    typer.echo(f"Building {', '.join(engine_names)} (interval={interval})")

    while not should_stop():
        t0 = time.monotonic()
        builder.run_cycle(engine_names, should_stop)
        if once:
            break
        sleep_secs = int(interval_secs - min(interval_secs, time.monotonic() - t0))
        for _ in range(sleep_secs):
            if should_stop():
                break
            time.sleep(1)
    typer.echo("Builder stopped.")


def _build_probe(builder, engine_names: list[str]) -> None:
    """Everything the builder needs, checked without building anything."""
    import shutil as _shutil

    ok = True
    root = builder.bus.root
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe_file = root / ".probe"
        probe_file.write_text("")
        probe_file.unlink()
        typer.echo(f"bus root {root}: writable")
    except OSError as e:
        typer.echo(f"bus root {root}: {e}", err=True)
        ok = False
    for tool in ("tar", "zstd"):
        found = _shutil.which(tool)
        typer.echo(f"{tool}: {found or 'NOT FOUND'}")
        ok = ok and bool(found)
    free = builder.free_gb()
    floor = builder.cfg.build.min_free_gb
    typer.echo(f"free space: {free:.0f}GB (floor {floor:.0f}GB)")
    ok = ok and free >= floor
    for name in engine_names:
        engine = builder.cfg.engines[name]
        try:
            engine.require_src_dir()
            engine.require_run_set()
            typer.echo(f"{name}: frontier {builder.frontier(name)}")
        except ValueError as e:
            typer.echo(f"{name}: {e}", err=True)
            ok = False
    if not ok:
        raise typer.Exit(1)


@app.command(name="compare-bots")
def compare_bots(
    bots: Annotated[list[str], typer.Argument(help="Two bot names to compare")],
    suite: str = typer.Option(
        "js3", help="Suite to compare commit coverage for (js2, js3, sp3)"
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Report which commits both bots have pushed, and which only one has.

    Queries the perf database, since that is where the two series meet: locally
    one db holds one bot. What it compares is the *pushed* intersection, and a
    relayed bot's path is longer (bench, spool, relay, stage, aggregate), so a
    small persistent deficit on that side is structural rather than a coverage
    problem.
    """
    cfg = _load_config(config)
    if len(bots) != 2:
        typer.echo("Error: compare-bots takes exactly two bot names.", err=True)
        raise typer.Exit(1)
    target = next(
        (t.spanner for t in (cfg.push.targets if cfg.push else []) if t.spanner), None
    )
    if not target:
        typer.echo("Error: no spanner push target in config.", err=True)
        raise typer.Exit(1)

    from . import spanner

    benchmark = spanner.benchmark_name(suite)
    db = spanner.connect(target, exclusive=False)
    try:
        per_bot = {bot: spanner.commit_numbers(db, bot, benchmark) for bot in bots}
    finally:
        db.close()

    a, b = (set(per_bot[bot]) for bot in bots)
    both = a & b
    typer.echo(f"benchmark {benchmark}")
    for bot in bots:
        typer.echo(f"  {bot}: {len(per_bot[bot])} commits")
    typer.echo(f"  both: {len(both)}")
    for bot, mine, theirs in ((bots[0], a, b), (bots[1], b, a)):
        only = sorted(mine - theirs)
        if only:
            typer.echo(
                f"  only {bot}: {len(only)} ({only[0]}..{only[-1]}, latest {only[-5:]})"
            )


bus_app = typer.Typer(
    name="bus", help="Inspect and gate the build bus.", no_args_is_help=True
)
app.add_typer(bus_app)


@bus_app.command("status")
def bus_status(
    engines: Annotated[
        Optional[list[str]], typer.Argument(help="Engines (default: all configured)")
    ] = None,
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Report what the bus and this machine's bench state look like."""
    from .status import collect_status, render

    cfg = _load_config(config)
    for name in engines or []:
        if name not in cfg.engines:
            typer.echo(f"Error: engine '{name}' not configured.", err=True)
            raise typer.Exit(1)
    # No db snapshot for a reporting command; it still opens read/write,
    # because it reads columns an unmigrated db does not have.
    collector = _open_collector(cfg, role="status", backup=False)
    render(collect_status(cfg, collector, engines), typer.echo)


@bus_app.command("gc")
def bus_gc(
    engines: Annotated[
        Optional[list[str]], typer.Argument(help="Engines (default: all configured)")
    ] = None,
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Delete payloads with no entry, left by a crash mid-publish or mid-prune.

    Takes the machine lock: an unreferenced payload is also what a build in
    progress looks like from outside, both while zstd is writing its tmp file
    and in the window between the payload landing and its entry being written.
    """
    from .bus import Bus
    from .lock import EXIT_BUSY, LockBusy, MachineLock

    cfg = _load_config(config)
    if cfg.bus is None:
        typer.echo("Error: no [bus] section in config.", err=True)
        raise typer.Exit(1)
    lock = MachineLock("bus gc")
    try:
        lock.acquire(wait=False)
    except LockBusy as e:
        typer.echo(f"Error: {e}", err=True)
        typer.echo(
            "Run 'slipstream bus pause' to have the daemons stand down.", err=True
        )
        raise typer.Exit(EXIT_BUSY)
    try:
        removed = Bus(cfg.bus.root).gc(engines or list(cfg.engines))
    finally:
        lock.release()
    for path in removed:
        typer.echo(f"  removed {path}")
    typer.echo(f"{len(removed)} unreferenced payloads removed.")


@bus_app.command("pause")
def bus_pause(
    ttl: str = typer.Option(
        "4h", help="How long the pause lasts before it expires on its own"
    ),
):
    """Ask the daemons to stop taking the machine lock, so an operator can.

    It does not preempt: whoever holds the lock keeps it until the commit it
    is working on is finished, which can be hours.
    """
    from .lock import pause as do_pause

    until = do_pause(_parse_interval(ttl))
    typer.echo(
        f"Paused until {time_mod.strftime('%H:%M', time_mod.localtime(until))}. "
        "Run 'slipstream bus resume' when done."
    )


@bus_app.command("resume")
def bus_resume():
    """Let the daemons take the machine lock again."""
    from .lock import paused_until, resume as do_resume

    was = paused_until()
    do_resume()
    typer.echo("Resumed." if was else "Was not paused.")


@app.command()
def config():
    """Print the config template."""
    from importlib.resources import files as pkg_files

    typer.echo(
        (pkg_files("slipstream.data") / "config.toml.example").read_text(),
        nl=False,
    )


@app.command()
def push(
    engines: Annotated[
        Optional[list[str]],
        typer.Argument(help="Engines to push (default: all configured)"),
    ] = None,
    config: Optional[Path] = typer.Option(None, help="User config path"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print CSV to stdout"),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Re-send the full history; spanner targets wipe this bot first",
    ),
    probe: bool = typer.Option(
        False,
        "--probe",
        help="Send an empty push through every target to verify them; marks nothing",
    ),
):
    """Push unpushed scores to the configured push targets."""
    import subprocess

    from .push import build_export_csv, probe as do_probe, push as do_push

    cfg = _load_config(config)
    if not cfg.push:
        typer.echo("Error: no [push] section in config.", err=True)
        raise typer.Exit(1)

    if probe:
        try:
            do_probe(cfg.push, lambda msg: typer.echo(f"  {msg}"))
        except Exception as e:
            typer.echo(f"Probe failed: {e}", err=True)
            raise typer.Exit(1)
        return

    engine_names = engines or list(cfg.engines.keys())
    # Push now writes to push_state on success; open the store read/write so
    # the backup hook runs and mark_pushed() persists.
    store = _open_store(cfg)

    if dry_run:
        for name in engine_names:
            csv_data, count = build_export_csv(
                store, name, cfg.platform, cfg.valid_names_by_suite
            )
            if csv_data:
                typer.echo(csv_data, nl=False)
            typer.echo(f"# {name}: {count} rows", err=True)
        store.close()
        return

    try:
        n = do_push(
            store,
            engine_names,
            cfg.push,
            cfg.valid_names_by_suite,
            cfg.platform,
            rebuild=rebuild,
            log=lambda msg: typer.echo(f"  {msg}"),
        )
        typer.echo(f"Pushed {n} scores")
    except (subprocess.CalledProcessError, OSError) as e:
        typer.echo(f"Push failed: {e}", err=True)
        raise typer.Exit(1)
    finally:
        store.close()


@app.command(name="host")
def host_cmd(
    apply: bool = typer.Option(
        False, "--apply", help="Apply the settings; prompts for sudo"
    ),
    restore: bool = typer.Option(
        False, "--restore", help="Restore the settings saved by the last --apply"
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Dump the command output the report is parsed from"
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Check (or configure) this machine's power state for unattended benchmarking."""
    import subprocess

    cfg = _load_config(config)
    if not host_mod.is_macos():
        typer.echo("Nothing to check: host power settings are macOS-only.")
        return

    backup = cfg.metadata_dir / "host_settings.bak.json"
    state = host_mod.read_state()

    if raw:
        for name, text in state.raw.items():
            typer.echo(f"--- {name} ---")
            typer.echo(text, nl=False)
        return

    if apply and restore:
        typer.echo("Error: --apply and --restore are mutually exclusive.", err=True)
        raise typer.Exit(1)

    if apply or restore:
        if restore:
            if not backup.exists():
                typer.echo(f"Error: no saved settings at {backup}.", err=True)
                raise typer.Exit(1)
            cmds = host_mod.restore_commands(host_mod.load_snapshot(backup))
        else:
            host_mod.save_snapshot(backup, state)
            typer.echo(f"Saved current settings to {backup}")
            cmds = host_mod.apply_commands(state)

        # A key this hardware rejects must not cost the settings after it;
        # what actually took effect is decided by the re-read below.
        failed = []
        for cmd in cmds:
            typer.echo(f"  $ {' '.join(cmd)}")
            try:
                returncode = subprocess.run(cmd).returncode
            except OSError as e:
                typer.echo(f"    {e}", err=True)
                returncode = -1
            # `defaults delete` exits non-zero when the key is already absent,
            # which for a restore is the wanted end state, not a failure. The
            # common case is a machine that never had the key set.
            if returncode != 0 and "delete" not in cmd:
                failed.append(" ".join(cmd))
        # pmset accepts settings the hardware does not implement and silently
        # does nothing, so the report below is built from state read afterwards.
        state = host_mod.read_state()
        typer.echo("")
        for cmd in failed:
            typer.echo(f"Warning: failed: {cmd}", err=True)

    results = host_mod.checks(state)
    host_mod.report(state, results, lambda msg: typer.echo(msg))
    if host_mod.failures(results):
        raise typer.Exit(1)
