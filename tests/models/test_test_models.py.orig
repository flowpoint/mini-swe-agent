import logging
import time

import pytest

import minisweagent.models
from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, DeterministicModelConfig, make_output


def test_basic_functionality_and_cost_tracking(reset_global_stats):
    """Test basic model functionality, cost tracking, and sequential output."""
    model = DeterministicModel(
        outputs=[
            make_output("```bash\necho hello", [{"command": "echo hello"}]),
            make_output("```bash\necho world", [{"command": "echo world"}]),
        ]
    )

    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```bash\necho hello"
    assert result["extra"]["actions"] == [{"command": "echo hello"}]
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 1
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 1.0

    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```bash\necho world"
    assert result["extra"]["actions"] == [{"command": "echo world"}]
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 2
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 2.0


def test_custom_cost_and_multiple_models(reset_global_stats):
    model1 = DeterministicModel(outputs=[make_output("one", [])], cost_per_call=2.5)
    model2 = DeterministicModel(outputs=[make_output("two", [])], cost_per_call=3.0)

    assert model1.query([{"role": "user", "content": "test"}])["content"] == "one"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 2.5

    assert model2.query([{"role": "user", "content": "test"}])["content"] == "two"
    assert minisweagent.models.GLOBAL_MODEL_STATS.cost == 5.5
    assert minisweagent.models.GLOBAL_MODEL_STATS.n_calls == 2


def test_config_model():
    config = DeterministicModelConfig(
        outputs=[make_output("Test", [{"command": "test"}])], model_name="custom", cost_per_call=5.0
    )

    assert config.cost_per_call == 5.0
    assert config.model_name == "custom"
    assert DeterministicModel(**config.model_dump()).config.cost_per_call == 5.0


def test_plain_text_message_formatting():
    model = DeterministicModel(outputs=[])
    message = {"role": "user", "content": "hello"}
    assert model.format_message(**message) == message


def test_sleep_and_warning_commands(caplog):
    model = DeterministicModel(
        outputs=[
            make_output("", [{"command": "/sleep 0.1"}]),
            make_output("after sleep", [{"command": "echo after_sleep"}]),
        ]
    )
    start_time = time.time()
    assert model.query([{"role": "user", "content": "test"}])["content"] == "after sleep"
    assert time.time() - start_time >= 0.1

    model = DeterministicModel(
        outputs=[
            make_output("", [{"command": "/warning Test message"}]),
            make_output("after warning", [{"command": "echo after_warning"}]),
        ]
    )
    with caplog.at_level(logging.WARNING):
        assert model.query([{"role": "user", "content": "test"}])["content"] == "after warning"
    assert "Test message" in caplog.text


def test_raise_exception():
    model = DeterministicModel(outputs=[make_output("", [{"raise": FormatError()}])])
    with pytest.raises(FormatError):
        model.query([{"role": "user", "content": "test"}])
