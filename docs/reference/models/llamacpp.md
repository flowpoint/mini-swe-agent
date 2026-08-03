# llama.cpp Model

`LlamaCppModel` is the default model adapter. It uses a reusable `requests.Session` to
call the OpenAI-compatible `POST /v1/chat/completions` endpoint exposed by a local
`llama-server`. Requests are text-only, synchronous, and always non-streaming.

```yaml
model:
  model_class: llamacpp
  model_name: local
  base_url: http://127.0.0.1:8080/v1
  connect_timeout: 10
  request_timeout: 600
  model_kwargs:
    temperature: 0
    max_tokens: 4096
```

The configured server alias must match `model_name`. Assistant replies must contain
exactly one fenced `bash` block; malformed successful responses raise a protocol error.

::: minisweagent.models.llamacpp_model

{% include-markdown "../../_footer.md" %}
