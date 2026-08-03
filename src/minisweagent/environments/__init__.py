"""Environment implementations for mini-SWE-agent."""

import copy

from minisweagent import Environment
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.environments.local import LocalEnvironment

_ENVIRONMENT_MAPPING = {
    "docker": DockerEnvironment,
    "local": LocalEnvironment,
}


def get_environment_class(spec: str) -> type[Environment]:
    try:
        return _ENVIRONMENT_MAPPING[spec]
    except KeyError:
        available = ", ".join(sorted(_ENVIRONMENT_MAPPING))
        raise ValueError(f"Unknown environment type: {spec}. Available: {available}") from None


def get_environment(config: dict, *, default_type: str = "") -> Environment:
    config = copy.deepcopy(config)
    environment_class = config.pop("environment_class", default_type)
    return get_environment_class(environment_class)(**config)
