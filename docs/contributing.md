# Contributing

Contributions should preserve the fork's narrow runtime: Linux, Python 3.11+, one local llama.cpp server, synchronous text Chat Completions, and fenced bash actions.

## Good contributions

- Small correctness, diagnostics, and reliability fixes
- Mocked HTTP and parser regression tests
- Linux local-shell safety and timeout improvements
- llama.cpp interoperability that uses the documented Chat Completions API
- Clear local setup, configuration, and troubleshooting documentation
- Optional SWE-bench improvements that do not affect the default install

Provider SDKs, routing layers, streaming, multimodal messages, native tool calls, telemetry, automatic fallback endpoints, and platform-specific Windows/macOS support are outside this fork's scope.

## Development setup

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

Run focused tests while iterating, then the full suite:

```bash
python -m pytest -q tests/models/test_llamacpp_model.py
python -m pytest -q
ruff check .
```

Build documentation changes strictly:

```bash
mkdocs build --strict
```

Install `mini-swe-agent[swebench]` only when changing benchmark functionality.

## Testing expectations

HTTP tests must use mocked sessions or responses; the test suite must not require a real `llama-server`. Cover both successful protocol parsing and failure classification. In particular, verify finite timeout tuples, non-retryable 4xx responses, retryable transient failures, malformed successful responses, metadata preservation, and `stream: false`.

Tests for default imports and CLI help should run in an environment without `litellm`, `openai`, or `datasets`. Benchmark tests must provide a useful missing-extra message.

## Code style

Keep the adapter and control flow explicit. Use ordinary typed Python, four-space indentation, focused functions, and descriptive `snake_case` names. Ruff and pre-commit define formatting and lint policy. Avoid undeclared dependencies and compatibility layers.

## Pull requests

Keep each change focused. Explain the behavior and motivation, list verification commands and results, and call out dependency or configuration changes. Add a regression test for bug fixes. Never include credentials, local model files, trajectories containing sensitive output, caches, or generated build artifacts.

{% include-markdown "_footer.md" %}
