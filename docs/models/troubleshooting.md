# Model troubleshooting

The runtime has one network path: `LlamaCppModel` sends `POST {base_url}/chat/completions` to the configured local `llama-server`. Start diagnosis with the client error and the server log from the same request.

## Cannot connect

For `ConnectionError` or a connection timeout:

- Confirm `llama-server` is running on the configured host and port.
- Keep `base_url` at the API root, normally `http://127.0.0.1:8080/v1`.
- Check that the server was bound to `127.0.0.1` or another intentionally reachable Linux address.
- Increase `connect_timeout` only for slow connection setup.

Connection failures and timeouts are retried up to three total attempts. The adapter ignores inherited proxy settings, so proxy environment variables cannot redirect local prompts.

## Read timeout

Local generation may exceed the finite `request_timeout`. Increase it in YAML if the server continues generating normally:

```yaml
model:
  request_timeout: 900
  connect_timeout: 10
```

A read timeout is retried. Avoid an infinite timeout; `agent.wall_time_limit_seconds` should still bound the overall run.

## HTTP errors

- **400:** inspect the server's diagnostic text. Common causes are an unsupported generation field, a context-window overflow, or an invalid request. This is not retried.
- **404:** `base_url` should end at `/v1`, not `/v1/chat/completions`. Check that the llama.cpp build exposes Chat Completions.
- **429, 502, 503, or 504:** these statuses are treated as transient and retried up to three attempts.
- **Other 4xx errors:** fail immediately. Verify `model_name` matches the exact `llama-server --alias` value.

Diagnostics include a bounded excerpt of the response and do not print URL credentials; URLs containing credentials are rejected.

## Invalid JSON or malformed response

A successful HTTP response must be JSON with string content at:

```text
choices[0].message.content
```

Missing `choices`, a missing message, non-text content, or invalid JSON raises `LlamaCppProtocolError` without retrying. These errors usually indicate an incompatible endpoint or server-side response transformation.

## The model does not produce an action

Every action must contain exactly one matching block:

~~~~text
```bash
git status
```
~~~~

Zero or multiple fenced bash blocks produce a format error. If this repeats:

- use a model with a reliable instruction-following chat template;
- ensure llama.cpp loaded the intended template;
- keep the packaged system prompt and `action_regex` together;
- reduce `max_tokens` only if responses are excessively long, not if the action is being truncated.

Native tool/function calls are not parsed.

## Unexpected model output

Set `temperature: 0` for deterministic generation settings. Reproducibility can still depend on the model file, llama.cpp version, context, and server parameters. Confirm that no additional server-side sampling flags override the request.

## SWE-bench dependency error

Install dataset support explicitly:

```bash
python -m pip install 'mini-swe-agent[swebench]'
```

The normal `mini` command does not need that extra.

--8<-- "docs/_footer.md"
