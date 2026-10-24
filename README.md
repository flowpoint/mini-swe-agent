# mini-SWE-agent: local llama.cpp fork

mini-SWE-agent is a small software-engineering agent for Linux. This fork talks directly to one local [llama.cpp](https://github.com/ggml-org/llama.cpp) server through its OpenAI-compatible Chat Completions endpoint. It does not use LiteLLM, the OpenAI Python SDK, cloud-provider routing, or telemetry.

## Supported runtime

- Linux and Python 3.11 or newer
- Synchronous, non-streaming, text-only chat completions
- A single local `llama-server`
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

The default installation uses `requests` for HTTP. It does not install `litellm`, `openai`, or `datasets`.

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

From the repository you want to modify:

```bash
mini --task "Fix the failing tests and explain the change"
```

The default confirmation mode shows each shell command before it runs. For a trusted checkout, confirmation can be disabled:

```bash
mini --yolo --task "Add regression coverage for the parser"
```

The agent executes real shell commands with your current user's permissions. Review commands carefully and use an isolated working tree for untrusted tasks.

## Configuration

The packaged `mini.yaml` selects `model_class: llamacpp` and uses deterministic defaults:

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

agent:
  step_limit: 100
  wall_time_limit_seconds: 3600
~~~~

Load a full custom configuration with `mini -c /path/to/config.yaml -t "..."`, or merge overrides onto the packaged base:

```bash
mini -c mini.yaml -c model.base_url=http://127.0.0.1:8080/v1 -t "Inspect this project"
```

## Optional SWE-bench support

SWE-bench dataset loading is isolated from the normal runtime. Install it only when needed:

```bash
python -m pip install 'mini-swe-agent[swebench]'
mini-extra swebench --help
```

Without the extra, SWE-bench commands report the required install command; normal imports and `mini` startup continue to work.

## Limitations

This fork has no streaming, multimodal input, native function calling, provider fallback, automatic model discovery, or monetary cost calculation. It supports Chat Completions only, reports cost as `0.0`, and does not target Windows or macOS. Server lifecycle, model loading, context sizing, and chat-template selection remain llama.cpp responsibilities.

## License and upstream

mini-SWE-agent is distributed under the [MIT License](LICENSE.md). See the [upstream project](https://github.com/SWE-agent/mini-swe-agent) for its original authors and broader provider-oriented releases.
