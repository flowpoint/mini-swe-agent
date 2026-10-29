"""This file provides convenience functions for selecting models.
You can ignore this file completely if you explicitly set your model in your run script.
"""

import copy
import importlib
import os

from minisweagent import Model


def get_model(input_model_name: str | None = None, config: dict | None = None) -> Model:
    """Get an initialized model object from any kind of user input or settings."""
    resolved_model_name = get_model_name(input_model_name, config)
    if config is None:
        config = {}
    config = copy.deepcopy(config)
    config["model_name"] = resolved_model_name

    model_class = get_model_class(resolved_model_name, config.pop("model_class", ""))

    return model_class(**config)


def get_model_name(input_model_name: str | None = None, config: dict | None = None) -> str:
    """Get a model name from any kind of user input or settings."""
    if config is None:
        config = {}
    if input_model_name:
        return input_model_name
    if from_config := config.get("model_name"):
        return from_config
    if from_env := os.getenv("MSWEA_MODEL_NAME"):
        return from_env
    raise ValueError("No model name set. Specify model_name or use the built-in local configuration.")


_MODEL_CLASS_MAPPING = {
    "llamacpp": "minisweagent.models.llamacpp_model.LlamaCppModel",
    "deterministic": "minisweagent.models.test_models.DeterministicModel",
}


def get_model_class(model_name: str, model_class: str = "") -> type:
    """Select one of the explicitly supported model classes.

    ``model_name`` is the llama.cpp server alias, not a provider-routing hint.
    ``DeterministicModel`` is retained only for tests.
    """
    selected = model_class or "llamacpp"
    try:
        full_path = _MODEL_CLASS_MAPPING[selected]
    except KeyError:
        available = ", ".join(sorted(_MODEL_CLASS_MAPPING))
        raise ValueError(f"Unknown model class: {selected}. Available: {available}") from None

    module_name, class_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
