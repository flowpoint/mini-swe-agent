# Cookbook

The local fork has three runtime components: a llama.cpp model adapter, an agent loop, and a Linux command environment. Keep extensions small and preserve the text-only, non-streaming boundary.

## Build an agent from the packaged config

Using the packaged YAML avoids duplicating the system and observation prompts:

```python
from minisweagent.agents import get_agent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model

config = get_config_from_spec("mini.yaml")
model = get_model(config=config["model"])
environment = get_environment(config["environment"], default_type="local")
agent = get_agent(model, environment, config["agent"], default_type="interactive")

try:
    agent.run("Add a regression test for the parser")
finally:
    model.close()
```

`get_model` resolves `model_class: llamacpp` to `LlamaCppModel`. There is no provider fallback.

## Construct the adapter directly

Use direct construction when embedding the client without the CLI:

```python
from minisweagent.models.llamacpp_model import LlamaCppModel

model = LlamaCppModel(
    model_name="local",
    base_url="http://127.0.0.1:8080/v1",
    request_timeout=600,
    connect_timeout=10,
    model_kwargs={"temperature": 0, "max_tokens": 4096},
    action_regex=r"```bash\s*\n(.*?)\n```",
)
```

A model query expects a list of text messages with `system`, `user`, or `assistant` roles. It returns an assistant message whose `extra` mapping contains parsed actions and response metadata. Call `model.close()` when the owning process is finished.

## Add a local execution policy

Confirmation mode is the primary safety boundary. For an additional project-specific check, subclass `LocalEnvironment` and reject commands before they reach the shell:

```python
import re

from minisweagent.environments.local import LocalEnvironment


class RestrictedEnvironment(LocalEnvironment):
    forbidden = (r"\bsudo\b", r"\brm\s+-rf\s+/", r"\bmkfs\b")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        if any(re.search(pattern, command) for pattern in self.forbidden):
            return {
                "output": "Blocked by local execution policy",
                "returncode": 126,
                "exception_info": "",
            }
        return super().execute(action, cwd, timeout=timeout)
```

Keep policy failures visible in the observation rather than silently changing a command.

## Customize prompts and actions together

The system prompt and `action_regex` form one protocol. If the fence changes, update both and add parser tests. The default contract is:

~~~~text
```bash
one command or one multi-line shell script
```
~~~~

The parser must find exactly one block. Do not add a second compatibility parser or native tool-call fallback.

## Testing an extension

Mock `requests.Session.post`; unit tests must not require a running llama.cpp server. Cover success, malformed responses, timeout/status retry classification, `stream: false`, and fenced-action extraction.

```bash
python -m pytest -q
ruff check .
```

Use the optional SWE-bench dependencies only for benchmark-specific work.

## Design constraints

Extensions should remain Linux-only, synchronous, text-only, and explicit. Do not add provider SDKs, routing layers, streaming, telemetry, credential discovery, or automatic fallback endpoints. If a feature needs a different transport or protocol, keep it outside the default runtime.

{% include-markdown "../_footer.md" %}
