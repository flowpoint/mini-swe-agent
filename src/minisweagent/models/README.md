# Model interface

This fork has one runtime model adapter: `llamacpp_model.py`. `LlamaCppModel` sends synchronous, non-streaming HTTP requests to a local llama.cpp server through a reused `requests.Session`.

## Configuration

The model mapping accepts:

- `model_class` — must be `llamacpp`.
- `model_name` — the alias passed to `llama-server --alias`.
- `base_url` — the OpenAI-compatible API root, normally `http://127.0.0.1:8080/v1`.
- `request_timeout` — finite read timeout in seconds.
- `connect_timeout` — optional finite connection timeout.
- `model_kwargs` — generation values such as `temperature` and `max_tokens`.
- `action_regex` — pattern used by the agent to extract exactly one fenced `bash` block.

The adapter posts only to `{base_url}/chat/completions` and forces `stream: false`. It does not read API keys, route to providers, discover models, or fall back to an external service.

## Response contract

Assistant text must be present at `choices[0].message.content`. The adapter preserves the full response, usage, llama.cpp timings, a zero cost, and a timestamp in trajectory metadata. Malformed successful responses raise a protocol error instead of being ignored.

`get_model` selects this adapter for `model_class: llamacpp`. `DeterministicModel` remains available for tests.
