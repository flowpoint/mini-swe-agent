# Local llama.cpp server

This fork supports one local model runtime: `llama-server` on Linux. The client uses the documented OpenAI-compatible Chat Completions endpoint and has no dependency on llama.cpp internals.

## Prerequisites

Install llama.cpp separately and choose a GGUF model whose metadata includes a suitable chat template. Python model code, tokenization, context allocation, and server lifecycle all remain llama.cpp responsibilities.

Start the server with an explicit alias:

```bash
llama-server \
  --model /path/to/model.gguf \
  --alias local \
  --host 127.0.0.1 \
  --port 8080
```

The default configuration expects:

- API root: `http://127.0.0.1:8080/v1`
- Chat endpoint: `http://127.0.0.1:8080/v1/chat/completions`
- Model alias: `local`

The adapter does not probe the server or discover loaded models. Keep `model_name` equal to the server's `--alias` value.

## Configure mini-SWE-agent

The packaged `mini.yaml` already contains a working local configuration:

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

Use `temperature: 0` when reproducible generation is important. `request_timeout` is the finite read timeout; `connect_timeout` limits connection setup. Long local generations may need a larger read timeout, but neither timeout should be infinite.

To change a value while retaining the packaged prompt and environment:

```bash
mini \
  -c mini.yaml \
  -c model.model_name=my-alias \
  -c model.base_url=http://127.0.0.1:9090/v1 \
  -t "Inspect this repository"
```

## Action format

The system prompt requires one fenced `bash` action whenever the model wants to run a command:

~~~~text
```bash
git status --short
```
~~~~

A response with no matching block or more than one matching block is returned to the model as a format error. Native tool/function calls are not accepted.

## Requests and retries

Each query is a synchronous JSON `POST` with `stream: false`. The adapter reuses one `requests.Session` and records the response, token usage, llama.cpp timings, timestamp, and zero monetary cost in the trajectory.

Connection errors, timeouts, and HTTP `429`, `502`, `503`, or `504` are retried up to three total attempts. Client errors such as `400`, `401`, `403`, and `404` fail immediately. Invalid JSON or a malformed successful response also fails immediately.

## Troubleshooting

- **Connection refused or timed out:** verify that `llama-server` is running on the configured host and port. Increase only `request_timeout` if generation itself is slow.
- **HTTP 404:** use the API root ending in `/v1`, not the full `/chat/completions` URL, and verify the running llama.cpp build exposes Chat Completions.
- **Unknown model:** make `model_name` match `--alias` exactly.
- **Context-window error:** increase the server's configured context when starting llama.cpp or reduce the prompt/history and `max_tokens`.
- **Format errors:** choose a model/chat template that follows instructions and reinforce the requirement for exactly one fenced `bash` block.

No API key or provider name is needed. The adapter ignores inherited proxy and credential settings and will never fall back to another endpoint.

## Deliberate limitations

Only Linux, Python 3.11+, text messages, non-streaming Chat Completions, and the current local shell are supported. There is no multimodal input, native function calling, cloud routing, fallback provider, automatic model discovery, or monetary cost calculation.

--8<-- "docs/_footer.md"
