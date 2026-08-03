# Migrating to the local llama.cpp fork

This fork narrows the upstream v2 runtime to Linux and one local llama.cpp server. Existing provider-oriented configurations need a small, explicit migration.

## Requirements

- Use Python 3.11 or newer on Linux.
- Run `llama-server` separately with a GGUF model and chat template.
- Install the project in a clean environment so removed SDKs are not accidentally imported from an old environment.

The default dependency set no longer includes `litellm`, `openai`, or `datasets`. SWE-bench dataset support is available only through `mini-swe-agent[swebench]`.

## Replace the model block

Use:

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
~~~~

Start the matching server alias:

```bash
llama-server \
  --model /path/to/model.gguf \
  --alias local \
  --host 127.0.0.1 \
  --port 8080
```

`base_url` is the `/v1` API root. The adapter appends `/chat/completions` and never falls back to another endpoint.

## Replace action formatting

Native tool-call and alternate action formats are not supported. Prompts must request exactly one fenced bash action:

~~~~text
```bash
git status
```
~~~~

Keep the packaged `system_template`, `action_regex`, and `format_error_template` together. Observations remain user-role text messages.

## Update Python imports

Replace provider model imports with:

```python
from minisweagent.models.llamacpp_model import LlamaCppModel

model = LlamaCppModel(
    model_name="local",
    base_url="http://127.0.0.1:8080/v1",
    model_kwargs={"temperature": 0, "max_tokens": 4096},
)
```

Or use `get_model(config={"model_class": "llamacpp", ...})`. Model selection no longer infers a provider from the model name.

## Replace limits

Local trajectory cost is always `0.0`. Remove monetary `cost_limit` assumptions and set:

```yaml
agent:
  step_limit: 100
  wall_time_limit_seconds: 3600
```

Keep finite `connect_timeout`, `request_timeout`, and `environment.timeout` values.

## CLI behavior

The main command remains:

```bash
mini --task "Fix the failing test"
```

The packaged `mini.yaml` is the default. When supplying overrides, explicitly retain it:

```bash
mini -c mini.yaml -c model.model_name=local -t "Inspect the project"
```

Normal CLI startup does not import benchmark dependencies.

## Trajectories

The `mini-swe-agent-1.1` trajectory envelope remains. Assistant metadata now records the llama.cpp JSON response, `usage`, `timings`, `timestamp`, and zero cost. Messages are text-only and actions appear under `extra.actions`.

## Removed behavior

Streaming, alternate completion endpoints, multimodal content, native function calls, cloud routing, model discovery, fallback providers, credential loading, and monetary cost calculation are deliberately absent.

--8<-- "docs/_footer.md"
