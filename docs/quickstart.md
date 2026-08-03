# Quickstart

This fork runs mini-SWE-agent against one local llama.cpp server. No cloud account, API key, setup wizard, or provider configuration is required.

## 1. Install the client

Requirements are Linux, Python 3.11 or newer, and a separately installed `llama-server` binary.

From this checkout:

```bash
python -m pip install .
```

For an editable development installation:

```bash
python -m pip install -e '.[dev]'
```

## 2. Start llama.cpp

Choose a GGUF model with a chat template and start the server in another terminal:

```bash
llama-server \
  --model /path/to/model.gguf \
  --alias local \
  --host 127.0.0.1 \
  --port 8080
```

`--alias local` must match the default `model_name: local`. The client uses `http://127.0.0.1:8080/v1/chat/completions` so llama.cpp can apply the model's chat template.

## 3. Run mini-SWE-agent

Change into the repository the agent should modify, then run:

```bash
mini --task "Find the cause of the failing tests and fix it"
```

The default `confirm` mode asks before every shell action. Use `--yolo` only in a trusted, isolated checkout:

```bash
mini --yolo --task "Add tests for the configuration loader"
```

The model may include a short explanation, but every action must contain exactly one fenced `bash` block. Commands run in the current local shell environment with your user's permissions.

## Change local model settings

The packaged `mini.yaml` already selects the llama.cpp adapter with deterministic generation. To retain that base and add command-line overrides:

```bash
mini \
  -c mini.yaml \
  -c model.base_url=http://127.0.0.1:8080/v1 \
  -c model.model_name=local \
  -c model.model_kwargs.max_tokens=2048 \
  -t "Inspect the parser"
```

When `-c` is supplied, list `mini.yaml` first if subsequent `key=value` items are only overrides. A complete custom YAML file can instead be loaded with:

```bash
mini -c /path/to/local.yaml -t "Run the focused tests"
```

See [Model setup](models/quickstart.md) for every adapter field and [YAML configuration](advanced/yaml_configuration.md) for merge behavior and agent limits.

## Optional SWE-bench commands

Normal agent use does not install or import `datasets`. Add it only for SWE-bench:

```bash
python -m pip install 'mini-swe-agent[swebench]'
mini-extra swebench --help
```

SWE-bench execution also needs the benchmark's container/runtime prerequisites. Without the extra, the command exits with a concise installation hint.

## Important limits

Only Linux, synchronous non-streaming Chat Completions, text messages, and fenced `bash` actions are supported. There is no provider fallback, native tool calling, multimodal input, automatic model discovery, or monetary cost accounting. The trajectory still records the response, token usage, llama.cpp timings, timestamp, and `cost: 0.0`.

--8<-- "docs/_footer.md"
