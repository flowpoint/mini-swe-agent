import os
from unittest.mock import patch

import pytest

from minisweagent.models import get_model, get_model_class, get_model_name
from minisweagent.models.test_models import DeterministicModel, make_output


class TestGetModelName:
    # Common config used across tests - model_name should be direct, not nested under "model"
    CONFIG_WITH_MODEL_NAME = {"model_name": "config-model"}

    def test_input_model_name_takes_precedence(self):
        """Test that explicit input_model_name overrides all other sources."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name("input-model", self.CONFIG_WITH_MODEL_NAME) == "input-model"

    def test_config_takes_precedence_over_env(self):
        """Test that config takes precedence over environment variable."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_env_var_fallback(self):
        """Test that environment variable is used when no config provided."""
        with patch.dict(os.environ, {"MSWEA_MODEL_NAME": "env-model"}):
            assert get_model_name(None, {}) == "env-model"

    def test_config_fallback(self):
        """Test that config model name is used when input and env are missing."""
        with patch.dict(os.environ, {}, clear=True):
            assert get_model_name(None, self.CONFIG_WITH_MODEL_NAME) == "config-model"

    def test_raises_error_when_no_model_configured(self):
        """Test that ValueError is raised when no model is configured anywhere."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No model name set"):
                get_model_name(None, {})

            with pytest.raises(ValueError, match="No model name set"):
                get_model_name(None, None)


class TestGetModelClass:
    def test_llamacpp_model_selection(self):
        from minisweagent.models.llamacpp_model import LlamaCppModel

        assert get_model_class("local", "llamacpp") is LlamaCppModel

    def test_default_model_selection(self):
        """Omitting model_class never falls back to a cloud-provider adapter."""
        from minisweagent.models.llamacpp_model import LlamaCppModel

        assert get_model_class("local") is LlamaCppModel

    def test_unknown_model_class_raises(self):
        with pytest.raises(ValueError, match="Unknown model class: cloud-provider"):
            get_model_class("local", "cloud-provider")


class TestGetModel:
    def test_config_deep_copy(self):
        """Test that get_model preserves original config via deep copy."""
        original_config = {"model_kwargs": {"temperature": 0}, "outputs": [make_output("test", [])]}

        with patch("minisweagent.models.get_model_class") as mock_get_class:
            mock_get_class.return_value = lambda **kwargs: DeterministicModel(
                outputs=[make_output("test", [])], model_name="test"
            )
            get_model("test-model", original_config)
            assert original_config["model_kwargs"]["temperature"] == 0
            assert "model_name" not in original_config

    def test_integration_with_compatible_model(self):
        """Test get_model works end-to-end with a model that handles extra kwargs."""
        with patch("minisweagent.models.get_model_class") as mock_get_class:
            hello_output = make_output("hello", [])

            def compatible_model(**kwargs):
                # Filter to only what DeterministicModel accepts, provide defaults
                config_args = {k: v for k, v in kwargs.items() if k in ["outputs", "model_name"]}
                if "outputs" not in config_args:
                    config_args["outputs"] = [make_output("default", [])]
                return DeterministicModel(**config_args)

            mock_get_class.return_value = compatible_model
            model = get_model("test-model", {"outputs": [hello_output]})
            assert isinstance(model, DeterministicModel)
            assert model.config.outputs == [hello_output]
            assert model.config.model_name == "test-model"

    def test_get_llamacpp_model(self):
        from minisweagent.models.llamacpp_model import LlamaCppModel

        model = get_model(
            "local",
            {
                "model_class": "llamacpp",
                "base_url": "http://127.0.0.1:8080/v1",
                "request_timeout": 90,
                "connect_timeout": 3,
                "model_kwargs": {"temperature": 0, "max_tokens": 2048},
            },
        )

        assert isinstance(model, LlamaCppModel)
        assert model.config.model_name == "local"
        assert model.config.base_url == "http://127.0.0.1:8080/v1"
        assert model.config.request_timeout == 90
        assert model.config.connect_timeout == 3
        assert model.config.model_kwargs == {"temperature": 0, "max_tokens": 2048}

    def test_get_deterministic_model(self):
        """Test that get_model can instantiate DeterministicModel via model_class parameter."""
        outputs = [make_output("hello", []), make_output("world", [])]
        config = {"outputs": outputs}
        model = get_model("test-model", config | {"model_class": "deterministic"})

        assert isinstance(model, DeterministicModel)
        assert model.config.outputs == outputs

        assert model.config.model_name == "test-model"
