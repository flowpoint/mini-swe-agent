# Models Overview

The supported runtime adapter is [`LlamaCppModel`](llamacpp.md). It sends synchronous,
non-streaming Chat Completions requests to one local `llama-server` and extracts exactly
one fenced `bash` action from each assistant response.

Select it explicitly in YAML:

```yaml
model:
  model_class: llamacpp
  model_name: local
  base_url: http://127.0.0.1:8080/v1
```

[`DeterministicModel`](test_models.md) remains available for tests. Generic selection
helpers are documented under [Model Utilities](utils.md).

{% include-markdown "../../_footer.md" %}
