# Copyright 2026 The slipstream developers
# SPDX-License-Identifier: MIT

from __future__ import annotations

import platform as _platform
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from importlib.resources import files as pkg_files

from .host import PREFLIGHT_MODES


def parse_spanner_spec(spec: str) -> tuple[str, str, str]:
    """Split "project/instance/database"."""
    parts = spec.strip("/").split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"spanner target must be project/instance/database: {spec!r}")
    return parts[0], parts[1], parts[2]


@dataclass
class PushTarget:
    """Where an export CSV goes. Exactly one of spool_dir, spanner.

    spool_dir: the CSV is appended to a sequenced log there, for a machine
    with access to the real target to collect with ``slipstream relay``.
    ``retain_days`` bounds how long entries are kept.
    spanner: "project/instance/database" of the perf database. ``refresh``
    aggregates after each push.
    """

    spool_dir: Path | None = None
    retain_days: int = 90
    spanner: str | None = None
    refresh: bool = True


@dataclass
class PushConfig:
    bot_name: str  # identifies this machine in the perf database
    targets: list[PushTarget]


@dataclass
class HostConfig:
    """How `bench` and `watch` react to the macOS host power state.

    ``preflight``: "warn" reports and continues, "abort" refuses to start when a
    blocking check fails, "off" skips the check. Inert off macOS.
    """

    preflight: str = "warn"


@dataclass
class RelaySource:
    """A remote spool log this machine drains into its own push targets."""

    ssh_host: str
    spool_dir: str  # path on ssh_host, as the remote shell expands it
    bot_name: str  # bot the spooled scores belong to
    cursor_dir: Path  # holds <bot_name>.cursor, the last seq delivered


@dataclass
class EngineConfig:
    """An engine this machine knows about.

    ``src_dir`` is optional: a machine that only benches artifacts from the bus
    has no checkout, and an engine is configured by the presence of its table.
    Everything that needs a checkout goes through ``require_src_dir``, because
    subprocess treats cwd=None as the launch directory, so a missing one would
    otherwise run git in whatever repo the daemon was started in.

    ``run_set`` lists what has to be packaged for the binary to run elsewhere,
    relative to src_dir. It is a shared input to both boxes' numbers, so it is
    hashed into the artifact's build_cfg_hash along with the build args.
    """

    name: str
    src_dir: Path | None
    build_cmd: str
    binary_path: str  # relative to src_dir
    id_regex: str
    path_filter: str = ""
    sync_cmd: str | None = None
    dyld_lib_path: str | None = None  # relative to src_dir (JSC/macOS)
    pre_build_patches: list[str] = field(default_factory=list)
    gn_args: str | None = None  # GN args for build dir setup
    run_set: list[str] = field(default_factory=list)

    def require_src_dir(self) -> Path:
        if self.src_dir is None:
            raise ValueError(f"engine {self.name} has no checkout on this machine")
        return self.src_dir

    def require_run_set(self) -> list[str]:
        if not self.run_set:
            raise ValueError(
                f"engine {self.name} has no run_set, so its build cannot be "
                f"packaged; list what the binary needs at runtime under "
                f"[engines.{self.name}] run_set"
            )
        return self.run_set


@dataclass
class BusSource:
    """A bus root this machine consumes build entries from.

    ``name`` is the cursor directory component, so the local source is named
    too. ``root`` is a path as the holding machine's shell expands it.
    ``engines`` restricts the source to a subset; empty means all of them.
    """

    name: str
    root: str
    ssh_host: str | None = None
    engines: list[str] = field(default_factory=list)
    bwlimit: str | None = None  # rsync --bwlimit, KB/s

    @property
    def is_local(self) -> bool:
        return self.ssh_host is None

    @property
    def local_root(self) -> Path:
        if not self.is_local:
            raise ValueError(f"bus source {self.name} is on {self.ssh_host}")
        return Path(self.root).expanduser()


@dataclass
class BusConfig:
    root: Path
    sources: list[BusSource] = field(default_factory=list)

    def source_for(self, engine: str) -> BusSource | None:
        """The source an engine is benched from, or None if it is git-driven."""
        for src in self.sources:
            if not src.engines or engine in src.engines:
                return src
        return None


@dataclass
class BuildConfig:
    """What ``slipstream build`` publishes, and how much of it is kept."""

    engines: list[str] = field(default_factory=list)
    retain_gb: float = 400.0  # per engine, oldest entries dropped first
    min_free_gb: float = 100.0
    max_infra_attempts: int = 5
    max_consecutive_burns: int = 3
    # Only consulted when an engine has neither published entries nor terminal
    # build failures, so a restart cannot rewind the frontier.
    start_from: dict[str, int] = field(default_factory=dict)


@dataclass
class BenchProcessConfig:
    """Limits on the bencher, per role rather than per source."""

    min_free_gb: float = 100.0
    run_roots: int = 2  # unpacked roots kept per engine
    # Consecutive commits that produced no scores at all before the consumer
    # stops. A run set missing a runtime dependency fails every commit, and a
    # cycle drains, so without this one pass marks the whole topic failed.
    max_consecutive_failures: int = 3


@dataclass
class BenchmarkConfig:
    name: str
    dir: Path
    cli: str
    names: list[str]
    score_regex: str
    timeout: str = "10m"
    run_mode: str = "suite"
    suite_score_regex: str | None = None


# A variant name keys the flags column in the store, the variant column in the
# perf database, and the stdout file of every run, so it is restricted to what
# is safe in all three.
_VARIANT_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Every section is closed: a key slipstream does not read is a config error,
# not a comment. The engine and benchmark lists are what the user config may
# set, which is narrower than what the bundled defaults hold.
_TOP_LEVEL_KEYS = (
    "out_dir",
    "results_dir",
    "bot_name",
    "engines",
    "benchmarks",
    "run",
    "push",
    "host",
    "relay",
    "bus",
    "build",
    "bench",
)
_ENGINE_KEYS = ("src_dir", "build_cmd", "sync_cmd", "gn_args", "run_set")
_BENCHMARK_KEYS = ("dir",)
RUN_MODES = ("suite", "per_benchmark")


@dataclass(frozen=True)
class RunSpec:
    """One measurement this machine takes of every commit.

    The matrix is listed rather than derived: an engine's interesting flags are
    a choice about what to watch, not a property of the engine, and a flag
    worth running on one suite is often not worth its hour on another.

    ``run_mode`` overrides the suite default; "" means take it. "per_benchmark"
    invokes the harness once per line item, which prints no overall score, so
    the collector synthesizes one.
    """

    engine: str
    suite: str
    flags: tuple[str, ...] = ()
    variant: str = "default"
    run_mode: str = ""


@dataclass
class Config:
    out_dir: Path
    results_dir: str
    engines: dict[str, EngineConfig]
    benchmarks: dict[str, BenchmarkConfig]
    # Empty only in tests that never bench; load_config refuses a config
    # without a matrix rather than quietly measuring nothing.
    runs: list[RunSpec] = field(default_factory=list)
    # Names this machine wherever its data is compared with another's. Optional
    # at load, because a box benching before it can push has no [push] section
    # to take it from, and an error at use.
    bot_name: str | None = None
    push: PushConfig | None = None
    relays: list[RelaySource] = field(default_factory=list)
    host: HostConfig = field(default_factory=HostConfig)
    bus: BusConfig | None = None
    build: BuildConfig = field(default_factory=BuildConfig)
    bench: BenchProcessConfig = field(default_factory=BenchProcessConfig)
    # CPU architecture of this host, e.g. "arm64". Keys scores and processing
    # state in the store; it is a host property, not a user setting.
    platform: str = field(default_factory=_platform.machine)

    def require_runs(self, engine: str) -> list[RunSpec]:
        """This engine's matrix, refusing an engine that has none.

        Checked where a bench session starts rather than at load, because an
        engine can legitimately be configured only to build it. Without this
        the session measures nothing and marks every commit failed, which
        reads as a broken engine rather than a missing config line.
        """
        runs = [r for r in self.runs if r.engine == engine]
        if not runs:
            raise ValueError(
                f"no [[run]] entry names engine {engine!r}, so benching it "
                f"would measure nothing"
            )
        return runs

    def require_bot_name(self) -> str:
        if not self.bot_name:
            raise ValueError(
                "this machine has no bot_name; add a top-level "
                'bot_name = "..." to the config'
            )
        return self.bot_name

    @property
    def valid_names_by_suite(self) -> dict[str, set[str]]:
        return {name: set(bench.names) for name, bench in self.benchmarks.items()}

    @property
    def metadata_dir(self) -> Path:
        return self.out_dir / "metadata"

    @property
    def results_path(self) -> Path:
        return self.out_dir / self.results_dir

    def commit_results_dir(self, engine: str, commit_id) -> Path:
        """Where one commit's stdout and stderr logs live.

        Under the engine, because the two engines' commit id spaces are
        independent: a shared directory has them overwrite each other's logs
        wherever the numbers happen to collide, and makes `clear` delete the
        other engine's along with its own.
        """
        return self.results_path / engine / str(commit_id)

    @property
    def logs_dir(self) -> Path:
        return self.out_dir / "logs"


_DEFAULT_CONFIG_PATH = Path.home() / ".config" / "slipstream" / "config.toml"
_DATA = pkg_files("slipstream.data")


def load_config(user_config_path: Path | None = None) -> Config:
    path = user_config_path or _DEFAULT_CONFIG_PATH
    with open(path, "rb") as f:
        user = tomllib.load(f)

    with (_DATA / "engines.toml").open("rb") as f:
        engine_defaults: dict = tomllib.load(f)

    with (_DATA / "benchmarks.toml").open("rb") as f:
        bench_defaults: dict = tomllib.load(f)

    _reject_unknown(user, _TOP_LEVEL_KEYS, str(path))
    for name in user.get("engines", {}):
        if name not in engine_defaults:
            raise ValueError(
                f"[engines.{name}] is not an engine slipstream knows about "
                f"(known: {sorted(engine_defaults)})"
            )
    for name in user.get("benchmarks", {}):
        if name not in bench_defaults:
            raise ValueError(
                f"[benchmarks.{name}] is not a suite slipstream knows about "
                f"(known: {sorted(bench_defaults)})"
            )

    out_dir = Path(user["out_dir"]).expanduser()
    results_dir = user.get("results_dir", "results")

    bot_name = user.get("bot_name")
    if bot_name is not None and (not isinstance(bot_name, str) or not bot_name.strip()):
        raise ValueError("bot_name must be a non-empty string")

    engines: dict[str, EngineConfig] = {}
    for name, defaults in engine_defaults.items():
        user_engine = user.get("engines", {}).get(name)
        if user_engine is None:
            continue
        _reject_unknown(
            user_engine, _ENGINE_KEYS, f"[engines.{name}]", settable=defaults
        )
        src_dir = user_engine.get("src_dir")
        # The build inputs are overridable: the two boxes do not have to build
        # the same way, and run_set is discovered per platform.
        engines[name] = EngineConfig(
            name=name,
            src_dir=Path(src_dir).expanduser() if src_dir else None,
            build_cmd=user_engine.get("build_cmd", defaults["build_cmd"]),
            binary_path=defaults["binary_path"],
            id_regex=defaults["id_regex"],
            path_filter=defaults.get("path_filter", ""),
            sync_cmd=user_engine.get("sync_cmd", defaults.get("sync_cmd")),
            dyld_lib_path=defaults.get("dyld_lib_path"),
            pre_build_patches=defaults.get("pre_build_patches", []),
            gn_args=user_engine.get("gn_args", defaults.get("gn_args")),
            run_set=_parse_run_set(
                name, user_engine.get("run_set", defaults.get("run_set", []))
            ),
        )

    benchmarks: dict[str, BenchmarkConfig] = {}
    for name, defaults in bench_defaults.items():
        user_bench = user.get("benchmarks", {}).get(name, {})
        _reject_unknown(
            user_bench, _BENCHMARK_KEYS, f"[benchmarks.{name}]", settable=defaults
        )
        if "dir" not in user_bench:
            continue
        names = (_DATA / defaults["names_file"]).read_text().strip().splitlines()
        benchmarks[name] = BenchmarkConfig(
            name=name,
            dir=Path(user_bench["dir"]).expanduser(),
            cli=defaults["cli"],
            names=names,
            score_regex=defaults["score_regex"],
            timeout=defaults.get("timeout", "10m"),
            run_mode=defaults.get("run_mode", "suite"),
            suite_score_regex=defaults.get("suite_score_regex"),
        )

    runs = _parse_runs(user.get("run", []), engines, benchmarks, path)

    push_data = user.get("push")
    push_cfg = None
    if push_data is not None:
        _reject_unknown(push_data, ("bot_name", "targets"), "[push]")
    if push_data and "bot_name" in push_data:
        targets = [_parse_target(t) for t in push_data.get("targets", [])]
        if not targets:
            raise ValueError("[push] needs at least one [[push.targets]] entry")
        push_cfg = PushConfig(bot_name=push_data["bot_name"], targets=targets)

    host_data = user.get("host", {})
    _reject_unknown(host_data, ("preflight",), "[host]")
    host_preflight = host_data.get("preflight", "warn")
    if host_preflight not in PREFLIGHT_MODES:
        raise ValueError(
            f"[host] preflight must be one of {list(PREFLIGHT_MODES)}: "
            f"{host_preflight!r}"
        )
    host_cfg = HostConfig(preflight=host_preflight)

    for r in user.get("relay", []):
        _reject_unknown(
            r, ("ssh_host", "spool_dir", "bot_name", "cursor_dir"), "[[relay]]"
        )
    relays = [
        RelaySource(
            ssh_host=r["ssh_host"],
            spool_dir=r["spool_dir"],
            bot_name=r["bot_name"],
            cursor_dir=Path(r["cursor_dir"]).expanduser()
            if "cursor_dir" in r
            else out_dir / "relay",
        )
        for r in user.get("relay", [])
    ]
    cursor_paths = [(r.cursor_dir / f"{r.bot_name}.cursor").resolve() for r in relays]
    if len(cursor_paths) != len(set(cursor_paths)):
        raise ValueError("[[relay]] entries must not share a bot_name and cursor_dir")

    bus_cfg = _parse_bus(user.get("bus"), engines)
    build_cfg = _parse_build(user.get("build", {}), engines)
    bench_cfg = _parse_bench(user.get("bench", {}))

    if push_cfg is not None:
        if bot_name is None:
            # Configs written before the bus name the machine only in [push].
            bot_name = push_cfg.bot_name
        elif push_cfg.bot_name != bot_name:
            raise ValueError(
                f"[push] bot_name {push_cfg.bot_name!r} must match the top-level "
                f"bot_name {bot_name!r}"
            )
    if bot_name is not None and any(r.bot_name == bot_name for r in relays):
        # Two boxes under one bot would be averaged together in the perf
        # database, whose aggregate key has no platform, with no error anywhere.
        raise ValueError(f"bot_name {bot_name!r} is also a [[relay]] source")

    return Config(
        out_dir=out_dir,
        results_dir=results_dir,
        bot_name=bot_name,
        engines=engines,
        benchmarks=benchmarks,
        runs=runs,
        push=push_cfg,
        relays=relays,
        host=host_cfg,
        bus=bus_cfg,
        build=build_cfg,
        bench=bench_cfg,
    )


def _reject_unknown(data: dict, allowed, section: str, settable=None) -> None:
    """Refuse keys this section does not read.

    A key that is quietly ignored is the worst kind of config error: the file
    says one thing and the machine does another, and nothing reports it until
    the numbers are already wrong. ``settable`` names keys that exist in the
    bundled defaults but are not the user's to override, so those get told
    apart from a typo.
    """
    unknown = sorted(set(data) - set(allowed))
    if not unknown:
        return
    if settable is not None:
        fixed = [k for k in unknown if k in settable]
        if fixed:
            raise ValueError(
                f"{section} {fixed} come from the bundled config and cannot be set here"
            )
    raise ValueError(f"{section} unknown keys {unknown}")


def _parse_runs(entries, engines, benchmarks, path: Path) -> list[RunSpec]:
    """Read the [[run]] matrix, refusing anything that would measure nothing.

    Every name is checked against what this machine has configured: an engine
    or suite that is only a typo would otherwise drop its rows silently, and a
    gap in a series is far harder to notice than a refusal to start.
    """
    if not entries:
        # Only a machine set up to measure is refused: engines and suites both
        # configured and nothing said about what to run with them is never
        # what was meant. A box that only relays or only builds configures at
        # most one of the two and has no matrix to give.
        if engines and benchmarks:
            raise ValueError(
                f"{path} configures engines and benchmarks but no [[run]] "
                "entries, so there is nothing to measure. Add e.g.:\n\n"
                '  [[run]]\n  engine = "v8"\n  suite = "js3"\n\n'
                "`slipstream config` prints a full example."
            )
        return []
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError("[[run]] must be a list of tables")

    runs: list[RunSpec] = []
    seen: dict[tuple[str, str, str], int] = {}
    for i, entry in enumerate(entries):
        where = f"[[run]] #{i + 1}"
        unknown = set(entry) - {"engine", "suite", "flags", "variant", "run_mode"}
        if unknown:
            raise ValueError(f"{where}: unknown keys {sorted(unknown)}")

        engine = entry.get("engine")
        if not isinstance(engine, str) or engine not in engines:
            raise ValueError(
                f"{where}: engine {engine!r} is not configured; add "
                f"[engines.{engine}] or fix the name "
                f"(configured: {sorted(engines) or 'none'})"
            )
        suite = entry.get("suite")
        if not isinstance(suite, str) or suite not in benchmarks:
            raise ValueError(
                f"{where}: suite {suite!r} is not configured; add "
                f"[benchmarks.{suite}] with a dir "
                f"(configured: {sorted(benchmarks) or 'none'})"
            )

        flags = entry.get("flags", [])
        if not isinstance(flags, list) or not all(
            isinstance(f, str) and f.strip() for f in flags
        ):
            raise ValueError(f"{where}: flags must be a list of non-empty strings")

        variant = entry.get("variant")
        if variant is None:
            if not flags:
                variant = "default"
            elif len(flags) == 1:
                variant = flags[0].lstrip("-").replace("-", "_")
            else:
                raise ValueError(
                    f"{where}: name the variant; it cannot be derived from "
                    f"{len(flags)} flags"
                )
        if not isinstance(variant, str) or not _VARIANT_RE.match(variant):
            raise ValueError(
                f"{where}: variant {variant!r} must match {_VARIANT_RE.pattern}"
            )

        run_mode = entry.get("run_mode", "")
        if run_mode not in ("", *RUN_MODES):
            raise ValueError(
                f"{where}: run_mode must be one of {list(RUN_MODES)}: {run_mode!r}"
            )

        key = (engine, suite, variant)
        if key in seen:
            # The three are the store's key for a score row, so a duplicate
            # would run twice and have the second insert overwrite the first.
            raise ValueError(
                f"{where}: {engine}/{suite}/{variant} is already defined by "
                f"[[run]] #{seen[key] + 1}"
            )
        seen[key] = i
        runs.append(
            RunSpec(
                engine=engine,
                suite=suite,
                flags=tuple(flags),
                variant=variant,
                run_mode=run_mode,
            )
        )
    return runs


def _parse_run_set(engine: str, entries) -> list[str]:
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        raise ValueError(f"[engines.{engine}] run_set must be a list of paths")
    for entry in entries:
        path = PurePosixPath(entry)
        if path.is_absolute() or ".." in path.parts or not entry.strip():
            raise ValueError(
                f"[engines.{engine}] run_set entries are relative to src_dir "
                f"and may not escape it: {entry!r}"
            )
    return list(entries)


def _parse_bus(data, engines: dict[str, EngineConfig]) -> BusConfig | None:
    if data is None:
        return None
    _reject_unknown(data, ("root", "sources"), "[bus]")
    if "root" not in data:
        raise ValueError("[bus] needs a root")
    sources: list[BusSource] = []
    claimed: dict[str, str] = {}
    for entry in data.get("sources", []):
        _reject_unknown(
            entry,
            ("name", "ssh_host", "root", "engines", "bwlimit"),
            "[[bus.sources]]",
        )
        for key in ("name", "root"):
            if key not in entry:
                raise ValueError(f"[[bus.sources]] needs a {key}")
        source = BusSource(
            name=entry["name"],
            root=entry["root"],
            ssh_host=entry.get("ssh_host"),
            engines=list(entry.get("engines", [])),
            bwlimit=entry.get("bwlimit"),
        )
        if any(s.name == source.name for s in sources):
            # The name is the cursor path component, so two sources sharing one
            # would share a cursor and skip each other's entries.
            raise ValueError(f"[[bus.sources]] name {source.name!r} is used twice")
        for engine in source.engines:
            if engine not in engines:
                raise ValueError(
                    f"[[bus.sources]] {source.name} claims engine {engine!r}, "
                    f"which has no [engines.{engine}] table"
                )
            if engine in claimed:
                raise ValueError(
                    f"[[bus.sources]] {engine} is claimed by both "
                    f"{claimed[engine]} and {source.name}"
                )
            claimed[engine] = source.name
        sources.append(source)
    # A source with no engine list takes everything, so at most one may exist,
    # and then it must be the only source.
    catch_all = [s for s in sources if not s.engines]
    if len(catch_all) > 1 or (catch_all and len(sources) > 1):
        raise ValueError(
            "[[bus.sources]] only one source may omit engines, and then it must "
            "be the only one"
        )
    return BusConfig(root=Path(data["root"]).expanduser(), sources=sources)


def _parse_build(data: dict, engines: dict[str, EngineConfig]) -> BuildConfig:
    _reject_unknown(
        data,
        (
            "engines",
            "retain_gb",
            "min_free_gb",
            "max_infra_attempts",
            "max_consecutive_burns",
            "from",
        ),
        "[build]",
    )
    names = list(data.get("engines", []))
    for name in names:
        if name not in engines:
            raise ValueError(
                f"[build] engines names {name!r}, which has no [engines.{name}] table"
            )
    start_from = {}
    for name, value in data.get("from", {}).items():
        if name not in engines:
            raise ValueError(f"[build] from names unknown engine {name!r}")
        start_from[name] = int(value)
    return BuildConfig(
        engines=names,
        retain_gb=_positive(data, "retain_gb", 400.0, "[build]"),
        min_free_gb=_positive(data, "min_free_gb", 100.0, "[build]"),
        max_infra_attempts=int(_positive(data, "max_infra_attempts", 5, "[build]")),
        max_consecutive_burns=int(
            _positive(data, "max_consecutive_burns", 3, "[build]")
        ),
        start_from=start_from,
    )


def _parse_bench(data: dict) -> BenchProcessConfig:
    _reject_unknown(
        data,
        ("min_free_gb", "run_roots", "max_consecutive_failures"),
        "[bench]",
    )
    return BenchProcessConfig(
        min_free_gb=_positive(data, "min_free_gb", 100.0, "[bench]"),
        run_roots=int(_positive(data, "run_roots", 2, "[bench]")),
        max_consecutive_failures=int(
            _positive(data, "max_consecutive_failures", 3, "[bench]")
        ),
    )


def _positive(data: dict, key: str, default, section: str):
    value = data.get(key, default)
    # bool is an int subclass, and a true here would silently mean one.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{section} {key} must be a positive number")
    return value


def _parse_target(t: dict) -> PushTarget:
    if "command" in t or "ssh_host" in t:
        raise ValueError(
            "[[push.targets]] command targets are gone; push to spanner "
            "directly, or to a spool_dir for a machine that can"
        )
    _reject_unknown(
        t, ("spool_dir", "retain_days", "spanner", "refresh"), "[[push.targets]]"
    )
    kinds = [k for k in ("spool_dir", "spanner") if k in t]
    if len(kinds) != 1:
        raise ValueError("[[push.targets]] needs exactly one of spool_dir, spanner")
    kind = kinds[0]
    if kind != "spanner" and "refresh" in t:
        raise ValueError("[[push.targets]] refresh applies to spanner targets only")
    if kind != "spool_dir" and "retain_days" in t:
        raise ValueError("[[push.targets]] retain_days applies to spool targets only")
    if kind == "spool_dir":
        retain_days = t.get("retain_days", 90)
        # bool is an int subclass, and retain_days = true would silently
        # mean one day.
        if isinstance(retain_days, bool) or not isinstance(retain_days, int):
            raise ValueError("[[push.targets]] retain_days must be a positive integer")
        if retain_days < 1:
            raise ValueError("[[push.targets]] retain_days must be a positive integer")
        return PushTarget(
            spool_dir=Path(t["spool_dir"]).expanduser(), retain_days=retain_days
        )
    parse_spanner_spec(t["spanner"])
    return PushTarget(spanner=t["spanner"], refresh=t.get("refresh", True))
