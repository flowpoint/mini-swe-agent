# Runtime environment configuration

Most behavior belongs in [YAML configuration](yaml_configuration.md). This page lists the small set of process environment variables still used by the local fork. There is no setup wizard or credential `.env` loader.

## Supported variables

| Variable | Purpose |
| --- | --- |
| `MSWEA_MODEL_NAME` | Fallback llama.cpp alias when no CLI or YAML `model_name` is set. The packaged config already sets `local`. |
| `MSWEA_MINI_CONFIG_PATH` | Replace the implicit `mini.yaml` path used when no `-c` option is supplied. |
| `MSWEA_CONFIG_DIR` | Add a directory searched for named YAML configuration files. |
| `MSWEA_GLOBAL_CONFIG_DIR` | Change the Linux directory used for the default trajectory and inspector state. |
| `XDG_CONFIG_HOME` | Set the base configuration directory; defaults to `~/.config`. |
| `MSWEA_GLOBAL_CALL_LIMIT` | Bound model calls across agent instances in the current Python process; `0` disables it. |
| `MSWEA_SILENT_STARTUP` | Suppress the package startup banner when set to a non-empty value. |
| `MSWEA_INSPECTOR_STYLE_PATH` | Select an alternate trajectory-inspector stylesheet. |

Set variables for the process that launches `mini`:

```bash
MSWEA_MINI_CONFIG_PATH=/path/to/local.yaml \
MSWEA_GLOBAL_CALL_LIMIT=100 \
mini --task "Run the focused tests"
```

For repeatable runs, prefer a checked-in, secret-free YAML file over shell startup files.

## Precedence

For the model alias, an explicit CLI `--model` value wins over YAML, and YAML wins over `MSWEA_MODEL_NAME`. Command-line `-c` specifications are recursively merged in order. If any `-c` is present, include `mini.yaml` first when later entries are only overrides:

```bash
mini \
  -c mini.yaml \
  -c model.model_name=local \
  -c model.base_url=http://127.0.0.1:8080/v1 \
  -t "Inspect this project"
```

`MSWEA_MINI_CONFIG_PATH` changes only the implicit default. It is not added when explicit `-c` options are supplied.

## Limits

Use per-agent `step_limit` and `wall_time_limit_seconds` for normal runs. `MSWEA_GLOBAL_CALL_LIMIT` is useful when several agents share one Python process.

`MSWEA_GLOBAL_COST_LIMIT` still exists for trajectory compatibility, but local requests always add zero cost. Do not rely on it as a safety limit.

## Paths

On Linux, the default configuration directory is `$XDG_CONFIG_HOME/mini-swe-agent`, or:

```text
~/.config/mini-swe-agent
```

The default `mini` trajectory is `last_mini_run.traj.json` inside that directory. Override it with `mini -o /path/to/run.traj.json` or `agent.output_path` in YAML.

The package may expose a `.env` path for compatibility, but it is not loaded by the local runtime. Do not put credentials there.

{% include-markdown "../_footer.md" %}
