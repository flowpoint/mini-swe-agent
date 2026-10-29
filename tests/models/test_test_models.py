import logging
import time

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.test_models import DeterministicModel, DeterministicModelConfig, make_output


def test_basic_functionality_and_sequential_output():
    """Test basic model functionality and sequential output."""
    model = DeterministicModel(
        outputs=[
            make_output("```bash\necho hello", [{"command": "echo hello"}]),
            make_output("```bash\necho world", [{"command": "echo world"}]),
        ]
    )

    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```bash\necho hello"
    assert result["extra"]["actions"] == [{"command": "echo hello"}]

    result = model.query([{"role": "user", "content": "test"}])
    assert result["content"] == "```bash\necho world"
    assert result["extra"]["actions"] == [{"command": "echo world"}]


def test_independent_models():
    model1 = DeterministicModel(outputs=[make_output("one", [])])
    model2 = DeterministicModel(outputs=[make_output("two", [])])

    assert model1.query([{"role": "user", "content": "test"}])["content"] == "one"

    assert model2.query([{"role": "user", "content": "test"}])["content"] == "two"


def test_config_model():
    config = DeterministicModelConfig(outputs=[make_output("Test", [{"command": "test"}])], model_name="custom")

    assert config.model_name == "custom"
    assert DeterministicModel(**config.model_dump()).config.outputs == config.outputs


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
