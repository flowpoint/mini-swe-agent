# mini-SWE-agent: local llama.cpp fork

mini-SWE-agent is a small software-engineering agent for Linux. This fork primarily targets a local [llama.cpp](https://github.com/ggml-org/llama.cpp) server through its OpenAI-compatible Chat Completions endpoint. Compatible hosted endpoints can use bearer-token authentication. It does not use LiteLLM, the OpenAI Python SDK, cloud-provider routing, or telemetry.

## Supported runtime

- Linux and Python 3.11 or newer
- Synchronous, text-only chat completions collected from a stream
- A single configured endpoint, normally a local `llama-server`
- Commands emitted as exactly one fenced `bash` block
- The existing local shell environment
- Deterministic generation when `temperature: 0`

## Install

From this checkout:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e '.[dev]'
pytest
```

The default installation uses `requests` for HTTP and a private, trimmed PyYAML safe loader for configuration. Rich/Pygments provide terminal rendering, and prompt-toolkit handles editing. It does not install `litellm`, `openai`, or `datasets`.

## Start llama.cpp

Use a GGUF model with a suitable chat template and give the server the alias expected by mini-SWE-agent:

```bash
llama-server \
  --model /path/to/model.gguf \
  --alias local \
  --host 127.0.0.1 \
  --port 8080
```

The default client URL is `http://127.0.0.1:8080/v1`, so requests are sent to `/v1/chat/completions`. The model name must match `--alias`.

## Run the agent

Create a tested config with `mini setup` (default: localhost:8080, alphabetically first advertised model). Setup fails without writing if discovery or the chat test fails. For a hosted endpoint, use `mini setup --base-url URL --api-key-env OPENAI_API_KEY` with the token in that environment variable. From the repository you want to modify:

```bash
mini --task "Fix the failing tests and explain the change"
```

The audit timeout controls each command: `0` runs immediately (the default), a positive number waits that many seconds for objections, and `inf` blocks until you explicitly continue. Ctrl-C interrupts; concurrent auditors can send SIGINT. Use `--audit-timeout`, `agent.audit_timeout` in config, or `/audit SECONDS` at a prompt:

```bash
mini --audit-timeout 3 --task "Add regression coverage for the parser"
```

The agent executes real shell commands with your current user's permissions. Trajectories are saved atomically to unique files under `$XDG_DATA_HOME/mini-swe-agent/trajectories` (default `~/.local/share`); `latest.traj.json` points to the latest run. Full command logs live in a directory beside each trajectory, with bounded head/tail excerpts passed to the model. Exit codes are 0 for submission, 1 for errors, 2 for invalid CLI arguments, 3 for limits, 4 for repeated malformed actions, and 130 for an unhandled interruption.

## Configuration

Missing configs and unknown settings fail before launch. The default user config is `$XDG_CONFIG_HOME/mini-swe-agent/mini.yaml` (default `~/.config`). The packaged `mini.yaml` selects `model_class: llamacpp` and uses deterministic defaults:

~~~~yaml
model:
  model_class: llamacpp
  model_name: local
  base_url: http://127.0.0.1:8080/v1
  request_timeout: 600  # Inactivity between response bytes; null disables.
  connect_timeout: 10
  model_kwargs:
    temperature: 0
    max_tokens: 4096
  action_regex: '```bash\s*\n(.*?)\n```'

agent:
  step_limit: 100
  max_context_chars: 65536

environment:
  timeout: 300  # Inactivity: output or process-group CPU/I/O; null disables.
  max_output_bytes: 16384
~~~~

Startup discovery uses a separate 3-second timeout, configurable with `--check-timeout`. Generation has no total deadline; setup's `--request-timeout` sets the generation inactivity timeout. Command liveness counts Linux process-group CPU ticks, I/O counters, and log growth. The bundled config has no elapsed-time limit. Context compaction keeps the task, user corrections, and recent complete turns without changing saved history; override `DefaultAgent.compact(messages)` to customize it. `max_context_chars` bounds text, not tokens: tune it to the model's context window. Cost settings and global cost/call counters have been removed; trajectories retain per-run call counts, elapsed seconds, and reported token usage.

Assistant responses render as Markdown with highlighted code on terminals; piped output retains the original text. Set `agent.code_theme` to `ansi_light` for light terminals or another Pygments theme. `NO_COLOR` disables colors. Load a full custom configuration with `mini -c /path/to/config.yaml -t "..."`, or merge overrides onto the packaged base:

```bash
mini -c mini.yaml -c model.base_url=http://127.0.0.1:8080/v1 -t "Inspect this project"
```

## Optional SWE-bench support

SWE-bench runners, Docker support, and benchmark configs live in `benchmarks/` and are excluded from the installed package. Run them from the repository root using the installed core:

```bash
python -m pip install . -r benchmarks/requirements.txt
python -m benchmarks.swebench --help
python -m benchmarks.swebench_single --help
```

Docker must be available for benchmark execution. To override benchmark settings, include `-c benchmarks/swebench.yaml` before additional `-c key=value` arguments. `mini-extra` provides only `inspect`; the old benchmark subcommands and `[swebench]` install extra have been removed.

## Limitations

This fork supports Chat Completions only, without multimodal input, native function calling, provider fallback, or monetary cost calculation. Responses are displayed after the stream completes. Remote liveness is observable only through response data; silent prefill needs a sufficiently generous `request_timeout`. Server lifecycle, model loading, context sizing, and chat-template selection remain server responsibilities. The CLI targets Linux.

## License and upstream

mini-SWE-agent is distributed under the [MIT License](LICENSE.md). See the [upstream project](https://github.com/SWE-agent/mini-swe-agent) for its original authors and broader provider-oriented releases.
