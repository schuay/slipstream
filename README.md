# slipstream

Slipstream measures JavaScript engine performance commit by commit. It builds
V8 or JavaScriptCore at each commit, runs JetStream2, JetStream3 or
Speedometer 3, stores every raw score in SQLite, and delivers the series to
the perf database, where the analysis lives.

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

For the untrusted RBE setup, each V8/Chromium checkout's `.gclient` solution
needs these `custom_vars` before running `gclient runhooks`:

```python
"custom_vars": {
    "download_remoteexec_cfg": True,
    "rbe_instance": "projects/rbe-chromium-untrusted/instances/default_instance",
},
```

Without the download flag, Chromium's hooks skip fetching the rewrapper
configuration. `gn gen` then fails with a missing `rewrapper_mac.cfg`, even
after a successful `gclient sync`. RBE credentials and service access are
also required; successful GN generation alone does not verify remote execution.

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

For upgrades, stop all running `build`, `watch`, and `deliver` processes before
`uv tool install --force .`, then restart all three. A running process keeps
its loaded code after reinstalling; leaving an old daemon running across a
schema migration can cause incompatible writes and retained SQLite locks.

On a dedicated macOS benchmark account, disable the media and photo analysis
agents; they can take a core for minutes at a time next to a running engine.

JSC archives include matching WebKit frameworks, XPC helpers and MiniBrowser,
selecting commits that touch `Source/JavaScriptCore`. Builds target `arm64e` to
use the pointer-authentication ABI and JIT paths used by Safari on Apple Silicon.
Results keep the host architecture label `arm64` and the existing series. The
build command and archived set are in `slipstream/data/engines.toml` and can be
overridden per engine in `config.toml`.

## Measuring

```sh
slipstream bench v8 109000 109100
slipstream watch
```

`bench` walks a commit range, building and measuring each commit. `watch` is
the same thing as a daemon: it picks up where the last run left off,
measures each new commit, and persists the scores. Run `slipstream deliver`
separately to send them onward.

Bus-driven `watch` uses macOS/BSD filesystem notifications for new builds.
Local sources are watched directly; SSH sources use a persistent
`slipstream bus subscribe` connection, then fetch artifacts as usual. Install
this version on the source machine first. The SSH command adds `~/.local/bin`
(the default `uv tool install` location) to PATH; custom installation directories
must be on the noninteractive SSH PATH. Startup and reconnect scan from the
cursor; there is no build-polling fallback. `--interval` controls git fetches.
Consumer errors and disk-floor
recovery retry after one minute; repeated benchmark failures use a two-hour
stall deadline. `--once` performs one bounded scan/drain without subscribing.
Notifications wake discovery immediately; local benchmarking still waits for
the builder's configured batch and the existing machine lock.

V8 selection skips commits whose changes are entirely in `tools/`, `test/`,
`agents/`, `docs/`, ownership metadata, or CPU backends unused by the host and
target build. DEPS changes are skipped when limited to test dependencies,
those excluded directories, or Android packages for a non-Android build;
variables used exclusively by those dependencies are included in the check.
Changes to runtime dependencies, hooks, build settings, and unknown inputs
remain candidates. This applies to V8 inside Chromium rolls too, preserving
each roll's first measurement. An explicitly requested range start or V8 build
retry is still measured even if it would otherwise be skipped.

`clear` forgets a range so it can be measured again, which is also how you
backfill history for a `[[run]]` entry added after the fact.

Chrome runs use `--headless=new` (Chromium 112 or newer), without opening a
browser window or requiring a display, with a 1500x1000 window, the viewport
crossbench gives it. Headless mode changes the browser's rendering
environment; compare scores against a headed baseline before joining the two
into one performance series.

The local benchmark server uses Crossbench's cross-origin isolation headers
(`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`), enabling high-resolution timers
where the browser supports them. It also sends `Cache-Control: no-store`.
Chrome explicitly suppresses model downloads, crashpad metrics and translation
triggers alongside its existing benchmarking/background flags. HTTP policy and
Chrome flags are included in the runner configuration hash; adopting them
changes the measurement baseline for existing browser results.

Safari runs require a dedicated benchmark account. The runner stops that
account's Safari launchers and WebKit helpers attributed to their launchd PID
domains before and after each measurement. It retains attributed process start
times through teardown, rescans for late helpers, and rechecks identity before
signaling. Unrelated WebKit clients (including Software Update's documentation
renderer) and unattributed orphan helpers are left alone. Cleanup uses bounded
TERM/KILL waits and waits for Safari's processes to become quiet. A launch or
engine provenance failure gets one fresh attempt after cleanup; the first
attempt's stderr is kept as `stderr-recovery`. Workload failures are not retried.

Safari provenance checks inspect owned WebContent workers and the launcher with
`vmmap -w`, which includes system dyld shared-cache libraries that `lsof` can
omit and preserves full paths under home directories that `sample` redacts.
Missing images and inspection errors fail verification. The early inspection
can overlap measurement. This policy changes the configuration
hash, so measurements are distinguishable from the old inspection policy.
Safari startup can still trigger Software Update activity; scoped cleanup does
not establish a background-free measurement environment.

Speedometer 3 (`sp3`) runs in a browser only; a `[[run]]` pairing it with a
shell engine is refused. A run is one page load with Speedometer's own ten
iterations. Each suite's mean time is stored as `Total-Time` (milliseconds,
lower is better) and Speedometer's `Score` as `Overall`'s `Total-Score`, the
same numbers crossbench reports. The page cannot report on its own, so the
local server appends a small module to `index.html` that POSTs the metrics
back; the checkout is never modified.

How many times each commit is measured is a property of the `[[run]]` entry,
`runs = N` (default 3), so `chrome`/`sp3` can take five rounds while
`chrome`/`js3` takes three. Rounds are interleaved across an engine's entries.

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

Consecutive builds of an engine differ little, so the builder stores a changed
`run_set` entry as bsdiff patches against the newest full archive at that
path, and a watcher that already holds the archive moves only the patches.
Patches that would come to more than `delta_max_ratio` of the full archive
(default half) make it a new full archive instead, which is what the next
builds are patched against; `delta_max_ratio = 0` turns this off. Any trouble
on the delta path stores the archive whole, so a publish never fails because
of it. Entries carrying patches are version 3: when upgrading, restart the
watchers before the builder, since an old watcher refuses an entry it cannot
read and the builder will not go back to rewrite it.

## Sending scores somewhere

`deliver` continuously drains bounded cycles of local results and remote SSH
spools. `push` performs one bounded local cycle through the same pipeline.
Both deliver scores to every entry in `[[push.targets]]`, tracked per commit
so a cycle only sends what is new. A target is either a Spanner database
(`spanner = "project/instance/database"`, with the DDL in
`slipstream/data/spanner_schema.sql`) or a `spool_dir`, which appends to a
sequenced log for a machine that has no route to the database. The receiving
machine configures `[[relay]]` sources and runs one `slipstream deliver` owner
for both its local DB and remote spools. The `relay` command is removed.
`slipstream deliver --help` lists the maintenance switches (explicit replay,
rebuild, legacy reconciliation); the `launchd/` example keeps it running.

## Development

```sh
uv sync
uv run pytest            # parallel by default; -n0 for a single process
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
