# Hook templates

Slipstream runs executable hooks from `hooks/` beside the selected config file
(default: `~/.config/slipstream/hooks/`). Hooks are optional and receive no
arguments; their working directory is the config directory. Dry runs skip them.

| Hooks | When they run | Template behavior |
| --- | --- | --- |
| `pre-window` / `post-window` | Once per observed opening/closing; closing waits for the active job | Empty customization points |
| `pre-build` / `post-build` | Around each build job, including packaging, under the machine lock | Select Low Power, then restore the previous AC power settings |
| `pre-bench` / `post-bench` | Around each measurement job, under the machine lock; pre-hook precedes preflight | Disable Low Power, request High Power where supported (otherwise Automatic), then restore the previous AC power settings |

Starting `bench` in an open window invokes `pre-window`; stopping it while open
does not invoke `post-window`. Window hooks do not run between jobs.
The four power templates require macOS and an installed slipstream version with
`power_hooks`. The runner supplies `SLIPSTREAM_PYTHON`, so no local Python path
needs to be embedded in the templates.

## Install hooks

For the default macOS power hooks, run from the repository root, without sudo:

```sh
./scripts/install-default-hooks.sh
# Or select a different config:
./scripts/install-default-hooks.sh --config /path/to/config.toml
```

This uses `uv` to prepare the checkout's Python environment, installs the four
power hooks, and generates and validates a sudoers fragment for the current
account and supported power settings. Run the printed `sudo install` command
to enable power changes. Existing custom hooks are not overwritten; window hooks
are left alone. Keep the checkout and its Python environment in place, since
the generated hooks use that interpreter.

For manual installation or customized window hooks, use the templates below.
From the repository root, copy templates without the `.example` suffix and make
them executable. This preserves existing hooks; merge those manually if needed.
For a custom config, change `hooks_dir` to its sibling `hooks/` directory.

```sh
hooks_dir="$HOME/.config/slipstream/hooks"
mkdir -p "$hooks_dir"
for template in packaging/hooks/*.example; do
    target="$hooks_dir/$(basename "$template" .example)"
    test -e "$target" || install -m 755 "$template" "$target"
done
```

## Enable power changes

`pmset` needs root privileges to change power settings. The sudoers template
grants the service account passwordless access to **only the listed, exact AC
power-mode commands**, keeping slipstream and its hooks unprivileged. Hooks use
`sudo -n -k` to ignore cached authentication and fail when a command lacks
passwordless permission. No sudoers rule is
needed for the empty window templates.

Copy the rule, replace `your_user` with the account running slipstream, and keep
only the setting(s) reported by `pmset -g custom` on the target Mac:

```sh
cp packaging/sudoers/slipstream-power.example /tmp/slipstream-power.sudoers
vi /tmp/slipstream-power.sudoers
visudo -c -f /tmp/slipstream-power.sudoers
sudo install -o root -g wheel -m 0440 \
    /tmp/slipstream-power.sudoers /etc/sudoers.d/slipstream-power
```

Install only after validation succeeds. Repeat on each bot; supported power
settings can differ. Battery settings are untouched.

Post-hooks also run after failed pre-hooks or jobs. Each pair shares a private
`SLIPSTREAM_HOOK_STATE_DIR`; failed restoration retains its snapshot under
`<out_dir>/metadata/hook-state/`. A killed process cannot restore automatically.
For recovery, use slipstream's Python environment to run
`python -m slipstream.power_hooks restore --state-dir /path/to/retained/state`.

If builds initially succeed and a later post-hook reports `sudo: a password is
required`, older hooks may have relied on cached sudo authentication that expired
during the job. Install the generated `hooks/pmset.sudoers` rule as described
above, including every restoration value, then restore the retained snapshot
before restarting the builder. A successful build does not prove passwordless
authorization is configured.
