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
from .models import CommitKey
from .collector import BenchCollector, FetchError
from . import host as host_mod

app = typer.Typer(
    name="slipstream",
    help="JS engine benchmark builder, bencher and publisher.",
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


def _shutdown_flag(label: str = "current commit", log=None):
    """Install SIGINT/SIGTERM handlers; returns a callable reading the flag.

    A callable rather than a variable so the flag reaches the collector's
    per-commit path, where a wait for the machine lock can last hours.

    Also where the process takes a process group of its own: every command
    that installs this may hold the machine lock, and the next holder finds
    what a crashed one left running by that group.
    """
    import signal

    from .lock import own_process_group

    own_process_group()
    state = {"stop": False}

    def handle(signum, frame):
        if state["stop"]:
            raise SystemExit(1)
        state["stop"] = True
        (log or typer.echo)(
            f"Shutdown requested; finishing {label} (^C again to force)"
        )

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
    once: bool = typer.Option(False, "--once", help="Single poll cycle then exit"),
    no_push: bool = typer.Option(
        False,
        "--no-push",
        help="Accepted for old service definitions; watch only collects",
    ),
    reset_cursor: Optional[str] = typer.Option(
        None,
        "--reset-cursor",
        metavar="ENGINE[=KEY]",
        help="Re-bench a bus-driven engine from above KEY (default: from the start)",
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
        try:
            # (0, 0) is below every key, including an embedded engine's, whose
            # embedder id is a position and so never 0.
            cursor = CommitKey.parse(raw) if raw else CommitKey(0, 0)
        except ValueError:
            raise typer.BadParameter("--reset-cursor takes ENGINE or ENGINE=KEY")
        if dry_run:
            typer.echo(
                f"{name}: would reset the cursor to {cursor} (source {source.name})"
            )
            return
        consumer.set_cursor(source, name, cursor)
        typer.echo(f"{name}: cursor reset to {cursor} (source {source.name})")
        typer.echo(
            "Entries at or below it are still skipped unless their scores are "
            f"cleared: slipstream clear {name} <first> <last>"
        )
        return

    # Verify cold-start: each git-driven engine must have at least one done
    # commit. A bus-driven engine starts from a cursor instead.
    for name in engine_names:
        if source_for(name) is None and (
            collector.store.max_done_key(name, cfg.platform, embedder_id=0) is None
        ):
            typer.echo(
                f"Error: no done commits for '{name}'. "
                f"Run 'slipstream bench {name} <start> <end>' first.",
                err=True,
            )
            raise typer.Exit(1)

    _host_preflight(cfg)

    if no_push:
        typer.echo("watch only collects; --no-push has no effect")

    should_stop = _shutdown_flag()

    typer.echo(f"Watching {', '.join(engine_names)} (interval={interval}, step={step})")

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
                    include_start=False,
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
        collector.store.close()

    typer.echo("Watch stopped.")


@app.command()
def clear(
    engine: Annotated[str, typer.Argument(help="Engine name (v8, jsc)")],
    start: Annotated[
        str, typer.Argument(help="First key to clear, e.g. 109680 or 1534000-109680")
    ],
    end: Annotated[
        Optional[str], typer.Argument(help="Last key (default: just start)")
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
    try:
        first = CommitKey.parse(start)
        last = CommitKey.parse(end) if end is not None else first
    except ValueError as e:
        raise typer.BadParameter(str(e))
    if last < first:
        typer.echo("Error: end must not be below start.", err=True)
        raise typer.Exit(1)

    store = _open_store(cfg)
    keys = [
        key
        for r in store.get_commits_in_range(engine, first.before(), last)
        if store.is_done(engine, cfg.platform, key := CommitKey.from_commit(r))
    ]
    if not keys:
        typer.echo(f"Nothing done for {engine} in {first}..{last}.")
        store.close()
        return
    if not yes:
        typer.confirm(
            f"Clear {len(keys)} commits of {engine} ({keys[0]}..{keys[-1]})?",
            abort=True,
        )

    try:
        with store.result_locks(engine, cfg.platform, keys):
            store.clear_range(engine, cfg.platform, keys)
            for key in keys:
                path = cfg.commit_results_dir(engine, key)
                if path.exists():
                    _shutil.rmtree(path)
        typer.echo(f"Cleared {len(keys)} commits. They will be measured again.")
    except (RuntimeError, OSError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        store.close()


@app.command()
def deliver(
    once: bool = typer.Option(
        False, "--once", help="One bounded cycle, including idle refresh"
    ),
    rebuild: Optional[str] = typer.Option(
        None,
        "--rebuild",
        metavar="BOT",
        help="Wipe remote BOT staging and aggregates, then replay retained spool",
    ),
    reset_cursor: Optional[str] = typer.Option(
        None, "--reset-cursor", metavar="BOT[=SEQ]"
    ),
    reconcile_legacy: Optional[str] = typer.Option(
        None,
        "--reconcile-legacy",
        metavar="BOT",
        help="After verified full replay: explicitly clear only BOT's old per-bot marker",
    ),
    force: bool = typer.Option(
        False, "--force", help="Allow rebuild from a retained suffix"
    ),
    config: Optional[Path] = typer.Option(None, help="User config path"),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Rotating plain event log (default: stderr)"
    ),
):
    """Deliver local results and remote SSH spools under one machine owner."""
    from .delivery import configured_coordinator
    from .delivery_maintenance import maintain_remote

    if sum(bool(v) for v in (rebuild, reset_cursor, reconcile_legacy)) > 1:
        typer.echo(
            "Error: --rebuild, --reset-cursor and --reconcile-legacy are exclusive",
            err=True,
        )
        raise typer.Exit(1)
    if force and not rebuild:
        typer.echo("Error: --force applies to --rebuild", err=True)
        raise typer.Exit(1)
    cfg = _load_config(config)
    from .service_logging import EventLog

    echo = EventLog("deliver", log_file)
    should_stop = _shutdown_flag("delivery unit", log=echo)

    store = None
    try:
        echo(
            f"Deliver starting local={cfg.delivery.local} remote={cfg.delivery.remote} poll_seconds={cfg.delivery.poll_seconds}"
        )
        coordinator, store = configured_coordinator(
            cfg, log=echo, should_stop=should_stop
        )
        if rebuild or reset_cursor or reconcile_legacy:
            with coordinator.ownership():
                maintain_remote(
                    coordinator,
                    cfg,
                    rebuild=rebuild,
                    reset=reset_cursor,
                    legacy=reconcile_legacy,
                    force=force,
                )
            return
        echo(
            f"Deliver started bot={cfg.push.bot_name} sources={len(coordinator.sources)} targets={coordinator.identities} poll_seconds={cfg.delivery.poll_seconds}"
        )
        if not coordinator.run(once=once):
            raise typer.Exit(1)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        echo.error(f"Error: {exc}")
        raise typer.Exit(1)
    finally:
        if store is not None:
            store.close()
        echo("Deliver stopped")
        echo.close()


def _parse_engine_key(value: str, flag: str) -> tuple[str, CommitKey]:
    """Parse an ENGINE=KEY argument."""
    engine, _, raw = value.partition("=")
    try:
        if not engine:
            raise ValueError(value)
        return engine, CommitKey.parse(raw)
    except ValueError:
        raise typer.BadParameter(
            f"{flag} takes ENGINE=KEY, e.g. v8=109680 or chrome=1534000-109680"
        )


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
        metavar="ENGINE=KEY",
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
        engine, key = _parse_engine_key(retry, "--retry")
        if not builder.store.request_build_retry(engine, key):
            typer.echo(
                f"Error: {engine} {key} has no recorded build failure.", err=True
            )
            raise typer.Exit(1)
        typer.echo(f"{engine} {key} will be attempted again.")
        # A key cursor cannot see anything at or below it, so a republished
        # entry sits unread unless every consumer is reset.
        typer.echo(
            f"Each consumer needs: slipstream watch --reset-cursor "
            f"{engine}={key.before()}"
        )
        return

    if probe:
        _build_probe(builder, engine_names)
        return

    _host_preflight(cfg)
    should_stop = _shutdown_flag("current build")
    typer.echo(f"Building {', '.join(engine_names)} (interval={interval})")
    builder.migrate(should_stop)

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
            # Constructing the resolver is what checks the inner checkout.
            builder.resolver(name)
            typer.echo(f"{name}: frontier {builder.frontier(name)}")
        except ValueError as e:
            typer.echo(f"{name}: {e}", err=True)
            ok = False
        if engine.embeds:
            found = _shutil.which("gclient")
            typer.echo(
                f"{name}: pins {engine.embeds} at {engine.pin} via gclient: {found or 'NOT FOUND'}"
            )
            ok = ok and bool(found)
    if not ok:
        raise typer.Exit(1)


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
    config: Optional[Path] = typer.Option(None, help="User config path"),
):
    """Delete blobs nothing names, left by a crash mid-publish or mid-prune.

    Takes the machine lock: an unreferenced blob is also what a build in
    progress looks like from outside, both while zstd is writing its tmp file
    and in the window between the blobs landing and their entry being written.
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
        removed = Bus(cfg.bus.root).gc()
    finally:
        lock.release()
    for path in removed:
        typer.echo(f"  removed {path}")
    typer.echo(f"{len(removed)} unreferenced files removed.")


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
    log_file: Optional[Path] = typer.Option(
        None, "--log-file", help="Rotating plain event log (default: stderr)"
    ),
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

    from .service_logging import EventLog

    if probe:
        events = EventLog("push", log_file)
        try:
            do_probe(
                cfg.push,
                events,
                state_dir=cfg.out_dir / "delivery",
                settings=cfg.delivery,
            )
        except Exception as e:
            events.error(f"Probe failed: {e}")
            raise typer.Exit(1)
        finally:
            events.close()
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

    events = EventLog("push", log_file)
    try:
        n = do_push(
            store,
            engine_names,
            cfg.push,
            cfg.valid_names_by_suite,
            cfg.platform,
            rebuild=rebuild,
            log=events,
            settings=cfg.delivery,
            state_dir=cfg.out_dir / "delivery",
            should_stop=_shutdown_flag("delivery unit", log=events),
        )
        typer.echo(f"Pushed {n} scores")
    except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as e:
        events.error(f"Push failed: {e}")
        raise typer.Exit(1)
    finally:
        store.close()
        events.close()


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
