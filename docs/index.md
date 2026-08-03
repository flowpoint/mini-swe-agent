# mini-SWE-agent for local llama.cpp

mini-SWE-agent is a minimal Linux coding agent that sends text messages directly to one local `llama-server`. The runtime uses synchronous `requests` calls to the OpenAI-compatible Chat Completions endpoint and executes model actions in a local bash environment.

!!! warning "Commands run on your machine"

    The agent executes real shell commands with your user's permissions. Use confirmation mode and an isolated checkout for untrusted tasks. `--yolo` is intended only for controlled environments.

## Start locally

Install the package from this checkout:

```bash
python -m pip install .
```

Start llama.cpp in a separate terminal:

```bash
llama-server \
  --model /path/to/model.gguf \
  --alias local \
  --host 127.0.0.1 \
  --port 8080
```

Then run the agent from the repository it should modify:

```bash
mini --task "Find the failing test, fix the cause, and verify the change"
```

The packaged configuration uses `model_class: llamacpp`, model alias `local`, and `http://127.0.0.1:8080/v1`. Requests are posted to `/v1/chat/completions` with `stream: false` and `temperature: 0`.

## How the loop works

1. The system prompt and task are sent as text messages.
2. llama.cpp returns assistant text using the GGUF model's configured chat template.
3. The adapter extracts exactly one fenced `bash` block.
4. `LocalEnvironment` executes that command in a new Linux shell process.
5. The result is appended as a user observation and the loop repeats.

The trajectory is the same linear message history, plus the raw server response, token usage, llama.cpp timings, timestamps, parsed actions, and zero monetary cost.

## Continue reading

- [Quickstart](quickstart.md) — installation, server startup, and the first run
- [Model setup](models/quickstart.md) — adapter fields and HTTP contract
- [Local llama.cpp server](models/local_models.md) — server configuration and retries
- [YAML configuration](advanced/yaml_configuration.md) — prompts, limits, and overrides
- [Troubleshooting](models/troubleshooting.md) — common server and response errors
- [Output files](usage/output_files.md) — trajectory structure

## Optional SWE-bench support

Benchmark dataset loading is not part of the default install:

```bash
python -m pip install 'mini-swe-agent[swebench]'
mini-extra swebench --help
```

Normal imports and `mini --help` work without `datasets`.

## Deliberate scope

This fork supports Linux, Python 3.11+, one local llama.cpp server, non-streaming text Chat Completions, and fenced bash actions. It does not implement native tool calls, multimodal messages, provider routing, fallback endpoints, model discovery, or monetary cost accounting.

{% include-markdown "_footer.md" %}
