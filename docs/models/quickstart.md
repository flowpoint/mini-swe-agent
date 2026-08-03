# Model setup

`LlamaCppModel` is the only production model adapter in this fork. It uses a persistent `requests.Session` and the documented OpenAI-compatible Chat Completions API exposed by `llama-server`.

## Start the server

```bash
llama-server \
  --model /path/to/model.gguf \
  --alias local \
  --host 127.0.0.1 \
  --port 8080
```

Use a GGUF model with an appropriate chat template. The client intentionally sends Chat Completions messages so llama.cpp, rather than mini-SWE-agent, applies that template.

## Default model configuration

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

| Field | Meaning |
| --- | --- |
| `model_class` | Must be `llamacpp`. |
| `model_name` | Model alias supplied to `llama-server --alias`. |
| `base_url` | Absolute HTTP(S) API root. Use the documented `/v1` root; a missing suffix is normalized. Embedded credentials are rejected. |
| `request_timeout` | Finite response/read timeout in seconds. |
| `connect_timeout` | Optional finite connect timeout. If omitted, `request_timeout` is used for both phases. |
| `model_kwargs` | Generation fields forwarded in the JSON body, such as `temperature` and `max_tokens`. |
| `action_regex` | Pattern that extracts exactly one fenced `bash` action. |
| `observation_template` | Converts shell output into the next user message. |
| `format_error_template` | Tells the model how to correct missing or multiple actions. |

Set `temperature: 0` for deterministic sampling. `model`, `messages`, and `stream` are controlled by the adapter; `stream` is always `false`.

## HTTP contract

For the defaults above, the adapter posts to `http://127.0.0.1:8080/v1/chat/completions` with a body like:

```json
{
  "model": "local",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "stream": false,
  "temperature": 0,
  "max_tokens": 4096
}
```

Only `system`, `user`, and `assistant` roles with string content are accepted. Assistant text is read from `choices[0].message.content`. A successful response with invalid JSON, missing choices, or missing text content raises a protocol error and is not retried.

## Actions and trajectory metadata

When the model wants to act, it must return exactly one block:

~~~~text
```bash
git status
```
~~~~

The existing text-action parser rejects zero or multiple matching blocks. The trajectory retains the original response plus:

```python
{
    "usage": response_json.get("usage", {}),
    "timings": response_json.get("timings", {}),
    "cost": 0.0,
    "timestamp": ...,
}
```

## Retry behavior

The adapter makes at most three attempts, with short exponential delays, for connection failures, timeouts, and HTTP `429`, `502`, `503`, or `504`. Other HTTP errors—including `400`, `401`, `403`, and `404`—fail immediately with a bounded diagnostic excerpt. Invalid model names, context-window errors, malformed payloads, and malformed successful responses are not retried.

The session ignores proxy and credential settings inherited from the environment. There is no API-key lookup, provider routing, fallback endpoint, streaming, or model discovery.

--8<-- "docs/_footer.md"
