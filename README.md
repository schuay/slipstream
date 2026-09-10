# slipstream

Slipstream measures JavaScript engine performance across a range of commits and
tells you which commits changed it. It builds V8 or JavaScriptCore at each
commit, runs JetStream2 or JetStream3, stores every raw score in SQLite, and
runs PELT change-point detection over the resulting time series to find the
commits where a score shifted.

It is built for unattended operation: a `watch` daemon benchmarks new commits as
they land, and two machines can split the work, one building and both measuring
the same artifacts.

## Requirements

Python 3.11.4 or newer (the bus consumer needs PEP 706 tar extraction filters),
plus a local checkout of whatever engine you want to measure and of the
benchmark suites. Building V8 needs `depot_tools`; building JSC needs the WebKit
build scripts. The defaults are tuned for arm64 macOS, which is what the bots
run; the bundled V8 `gn_args` set `use_remoteexec = true`, so override them if
you have no RBE backend.

## Install

```sh
uv tool install .
slipstream config > ~/.config/slipstream/config.toml
```

Then edit the config: it needs `out_dir`, a `bot_name`, an `[engines.*]` table
per engine with its `src_dir`, a `[benchmarks.*]` table per suite with its
`dir`, and the `[[run]]` entries naming what this machine measures. The config
is closed -- an unknown key is a startup error, not a comment -- so the template
is the reference for what can be set.

## Measuring

```sh
slipstream bench v8 109000 109100
slipstream analyze --engine v8
slipstream watch
```

`bench` walks a commit range, building and measuring each commit. `analyze`
reports the change points it finds, with a Cohen's d effect size per segment.
`watch` is the same thing as a daemon: it picks up where the last run left off,
measures each new commit, and pushes the scores onward.

`clear` forgets a range so it can be measured again, which is also how you
backfill history for a `[[run]]` entry added after the fact.

## Two machines

`build` and `watch` can be split across a pair of boxes over a shared directory
called the bus. One box runs `slipstream build`, which checks out each commit,
builds it, packages the engine's `run_set`, and publishes it. Both boxes run
`slipstream watch`, which unpacks published artifacts and measures them. This
keeps the two sets of numbers comparable, since they come from the same binary,
and it keeps a slow build off the critical path of the faster box.

`bus status` reports what each side has done and flags a `[[run]]` matrix that
the two boxes do not agree on. `bus pause`, `bus resume`, and `bus gc` handle
the rest.

## Sending scores somewhere

`push` delivers scores to every entry in `[[push.targets]]`, tracked per commit
so a cycle only sends what is new. A target is either a Spanner database
(`spanner = "project/instance/database"`, with the DDL in
`slipstream/data/spanner_schema.sql`) or a `spool_dir`, which appends to a
sequenced log for a machine that has no route to the database. `slipstream
relay` runs on a machine that does, draining a remote spool over ssh into its
own targets.

Scores also move as CSV: `export` writes it and `import` reads it. One database
holds one bot, and `import` refuses anything that is not its own machine's data.

## Development

```sh
uv sync
uv run pytest tests/
uv run ruff check slipstream tests && uv run ruff format slipstream tests
scripts/install-hooks
```

`install-hooks` points `core.hooksPath` at `.githooks`, which runs ruff, checks
that every source file carries its SPDX header, and scans staged content
against the pattern files in `.git/private-hooks/` (untracked, one regex per
line, empty when you have no such patterns).

The Spanner tests run against the emulator and skip unless
`SPANNER_EMULATOR_HOST` is set.

## License

MIT. See `LICENSE`.
