# YAML configuration

mini-SWE-agent merges YAML files and command-line `key=value` specifications into four top-level sections: `model`, `agent`, `environment`, and `run`. The packaged default is `src/minisweagent/config/mini.yaml`.

## Complete local example

A standalone configuration needs the prompt templates as well as the model settings:

~~~~yaml
model:
  model_class: llamacpp
  model_name: local
  base_url: http://127.0.0.1:8080/v1
  request_timeout: 600
  connect_timeout: 10
  model_kwargs:
    temperature: 0
    max_tokens: 4096
  action_regex: '```bash\s*\n(.*?)\n```'
  observation_template: |
    <returncode>{{output.returncode}}</returncode>
    <output>
    {{output.output}}
    </output>
  format_error_template: |
    Format error: {{error}}
    Respond with exactly one fenced `bash` block containing the next command.

agent:
  system_template: |
    You are a software engineering assistant running on Linux.
    To act, respond with exactly one fenced `bash` block.
  instance_template: |
    Please solve this issue: {{task}}
  mode: confirm
  step_limit: 100
  wall_time_limit_seconds: 3600
  max_consecutive_format_errors: 3

environment:
  environment_class: local
  cwd: ""
  timeout: 30
  env:
    PAGER: cat
~~~~

The packaged prompt is more detailed and is the recommended base. Jinja variables such as `{{task}}`, Linux system information, environment variables, model settings, and counters are available to templates. Undefined variables are errors.

## Model settings

| Key | Description |
| --- | --- |
| `model_class` | Use `llamacpp`. There is no provider fallback. |
| `model_name` | Must match the `llama-server --alias` value. |
| `base_url` | Absolute HTTP(S) API root, normally `http://127.0.0.1:8080/v1`. Do not include `/chat/completions`. |
| `request_timeout` | Positive finite read timeout in seconds. |
| `connect_timeout` | Positive finite connect timeout; if null, `request_timeout` is used. |
| `model_kwargs` | JSON generation fields forwarded to llama.cpp. `temperature: 0` provides deterministic sampling. |
| `action_regex` | Extracts exactly one command from assistant text. |
| `observation_template` | Formats a shell result as the next user message. |
| `format_error_template` | Feedback after zero, multiple, or incomplete actions. |

The adapter always supplies `model` and `messages` and forces `stream: false`. Configuration cannot enable streaming, multimodal content, or native tool calls.

## Agent settings

- `system_template` and `instance_template` are required. The system prompt must tell the model to emit exactly one fenced `bash` block whenever it acts.
- `mode` is `confirm`, `yolo`, or `human`. Keep `confirm` for normal interactive work; `--yolo` overrides it for trusted unattended runs.
- `step_limit` and `wall_time_limit_seconds` bound local inference. Zero disables the corresponding limit.
- `max_consecutive_format_errors` stops repeated invalid action formatting; zero disables this check.
- `output_path` selects the JSON trajectory destination. The CLI `-o` option overrides it.
- `cost_limit` should remain `0`. Local responses report `cost: 0.0`, so step and wall-time limits are the effective safeguards.

A compatible action and parser pair looks like:

~~~~text
Assistant response:

```bash
python -m pytest -q
```
~~~~

The regular expression and prompt must stay in sync. The default pattern accepts optional whitespace after `bash` and captures the command between the opening and closing fences.

## Local environment settings

`environment_class: local` executes commands through the Linux shell.

- `cwd` sets the command working directory; an empty value uses the directory where `mini` was started.
- `timeout` limits each command in seconds and terminates its process group on expiry.
- `env` adds or overrides environment variables for commands.

Each action starts a new shell process. Directory changes and temporary shell variables do not persist between actions unless they are included again. Do not store secrets in YAML; the trajectory records configuration and messages.

## Merge configurations on the command line

If any `-c` option is supplied, the implicit default file is not added. Include `mini.yaml` explicitly when applying only overrides:

```bash
mini \
  -c mini.yaml \
  -c model.model_kwargs.max_tokens=2048 \
  -c agent.step_limit=50 \
  -c environment.timeout=120 \
  -t "Run the relevant tests"
```

Later specifications recursively override earlier ones. Values are parsed as JSON when possible, so `0`, `false`, `null`, arrays, and objects retain their types; other values remain strings.

Use a full custom file directly when it defines every required field:

```bash
mini -c /path/to/local.yaml -t "Fix the reported issue"
```

Configuration filenames without `.yaml` are resolved from the current directory, `MSWEA_CONFIG_DIR`, and the packaged configuration directories. `MSWEA_MINI_CONFIG_PATH` can select a different implicit default.

## Run settings

A YAML file may provide the task:

~~~~yaml
run:
  task: "Add a regression test for the parser"
~~~~

`--task` overrides it. If neither source supplies a task, `mini` prompts interactively.

--8<-- "docs/_footer.md"
