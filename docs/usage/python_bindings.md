# Python bindings

!!! abstract "Overview"

    This page shows the most basic example of how to use mini-SWE-agent as a Python library.
    For more advanced usage, subclassing, and mix & match of components, see [subclassing and more](../advanced/cookbook.md).

## Hello world

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
    agent.run("Write a hello world program")
finally:
    model.close()
```

This uses the packaged `llamacpp` configuration, model alias `local`, and the local Linux shell. Start the matching `llama-server` before running the example.

{% include-markdown "../_footer.md" %}
