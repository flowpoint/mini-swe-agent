# Model interface

This fork has one runtime model adapter: `llamacpp_model.py`. `LlamaCppModel` collects streamed Chat Completions from llama.cpp or a compatible endpoint through a reused `requests.Session`.

## Configuration

The model mapping accepts:

- `model_class` — must be `llamacpp`.
- `model_name` — the alias passed to `llama-server --alias`.
- `base_url` — the OpenAI-compatible API root, normally `http://127.0.0.1:8080/v1`.
- `request_timeout` — inactivity between response bytes in seconds (default 600); `null` disables it.
- `connect_timeout` — optional finite connection timeout.
- `verify_ssl` — TLS certificate and hostname verification (default `true`); set `false` to accept self-signed or otherwise invalid certificates. `mini setup --no-verify-ssl` tests and saves this setting.
- `api_key_env` — optional environment-variable name holding a bearer token; the token itself is never saved.
- `model_kwargs` — generation values such as `temperature` and `max_tokens`.
- `action_regex` — optional extraction override; by default the execution target selects a `python` or `bash` fence. The target comes from `environment.language`, independently of the API provider.

The adapter posts to `{base_url}/chat/completions` with `stream: true` and checks availability through `/models`. `mini setup` discovers and tests settings before saving them. Credentials are read only from the explicitly configured variable; redirects and ambient proxy/netrc settings are disabled. There is no provider routing or fallback. Unknown config keys fail; `model_kwargs` remains open for sampling extensions.

## Response contract

SSE deltas are assembled into `choices[0].message.content`; JSON responses from servers that ignore streaming are also accepted. Streams must finish before their actions can run. The adapter preserves the assembled response, reported usage, llama.cpp timings, and a timestamp in trajectory metadata. Broken or malformed streams fail without replaying the generation request. Active streams may run indefinitely; read timeouts measure silence.

`get_model` selects this adapter for `model_class: llamacpp`. `DeterministicModel` remains available for tests.
