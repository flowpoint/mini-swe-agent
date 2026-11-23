"""Environment implementations for mini-SWE-agent."""

import copy
from importlib import import_module

from minisweagent import Environment

_ENVIRONMENT_MAPPING = {
    "local": "minisweagent.environments.local.LocalEnvironment",
    "podman": "minisweagent.environments.podman.PodmanEnvironment",
}


def get_environment_class(spec: str) -> type[Environment]:
    try:
        module, name = _ENVIRONMENT_MAPPING[spec].rsplit(".", 1)
    except KeyError:
        available = ", ".join(sorted(_ENVIRONMENT_MAPPING))
        raise ValueError(f"Unknown environment type: {spec}. Available: {available}") from None
    return getattr(import_module(module), name)


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)
