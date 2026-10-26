import re
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.run.mini import DEFAULT_CONFIG_FILE, app, main


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def test_main_starts_without_setup_wizard():
    """Test that mini starts directly from its loaded configuration."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            audit_timeout=None,
            output=None,
            model_class=None,
            agent_class=None,
        )

        mock_get_model.assert_called_once()
        mock_agent.run.assert_called_once_with("Test task")


def test_mini_command_calls_run_interactive():
    """Test that mini command creates agent via get_agent."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test", "audit_timeout": float("inf")}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with task provided (so prompt is not called)
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            audit_timeout=None,
            output=None,
            model_class=None,
            agent_class=None,
        )

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        assert args[0] == mock_model  # model
        assert isinstance(args[1], LocalEnvironment)
        # Verify agent.run was called with the task
        mock_agent.run.assert_called_once_with("Test task")


def test_mini_calls_prompt_when_no_task_provided():
    """Test that mini calls prompt when no task is provided."""
    with (
        patch("minisweagent.run.mini._multiline_prompt") as mock_prompt,
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_prompt.return_value = "User provided task"
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test", "audit_timeout": float("inf")}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function without task
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task=None,  # No task provided
            audit_timeout=None,
            output=None,
            model_class=None,
            agent_class=None,
        )

        # Verify prompt was called
        mock_prompt.assert_called_once()

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        # Verify agent.run was called with the task from prompt
        mock_agent.run.assert_called_once_with("User provided task")


def test_mini_with_explicit_model():
    """Test that mini works with explicitly provided model."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {
            "agent": {"system_template": "test", "audit_timeout": 0},
            "env": {},
            "model": {"default_config": "test"},
        }

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with explicit model
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="gpt-4",
            task="Test task with explicit model",
            audit_timeout=0,
            output=None,
            model_class=None,
            agent_class=None,
        )

        # Verify get_model was called (model name is merged into config)
        mock_get_model.assert_called_once()

        # Verify get_agent was called
        mock_get_agent.assert_called_once()
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test task with explicit model")


def test_zero_audit_timeout_overrides_config():
    """An explicit zero must override a configured audit window."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test", "audit_timeout": 5}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with audit_timeout=0
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test yolo task",
            audit_timeout=0,
            output=None,
            model_class=None,
            agent_class=None,
        )

        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        assert args[2].get("audit_timeout") == 0
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test yolo task")


def test_missing_audit_override_preserves_config():
    """Omitting the CLI override preserves the configured audit window."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test", "audit_timeout": 1.25}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with audit_timeout=None (default)
        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test confirm task",
            audit_timeout=None,
            output=None,
            model_class=None,
            agent_class=None,
        )

        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        assert args[2].get("audit_timeout") == 1.25
        # Verify agent.run was called
        mock_agent.run.assert_called_once_with("Test confirm task")


def test_cost_limit_zero_is_preserved():
    """Test that an explicit cost_limit=0 is not dropped during config merge."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        mock_get_model.return_value = Mock()
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}
        mock_agent = Mock()
        mock_get_agent.return_value = mock_agent

        main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test cost limit task",
            audit_timeout=None,
            cost_limit=0,
            output=None,
            model_class=None,
            agent_class=None,
        )

        assert mock_get_agent.call_args.args[2]["cost_limit"] == 0


def test_mini_help():
    """Test that mini --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run mini-SWE-agent in your local environment." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--audit-timeout" in clean_output
    assert "--output" in clean_output
    assert "--environment-class" not in clean_output


def test_mini_help_with_argparse(capsys):
    """Test help functionality through the standard-library parser."""
    with pytest.raises(SystemExit) as exc_info:
        app(["--help"])

    assert exc_info.value.code == 0
    # Strip ANSI color codes for reliable text matching
    clean_output = strip_ansi_codes(capsys.readouterr().out)
    assert "Run mini-SWE-agent in your local environment." in clean_output
    assert "--help" in clean_output
    assert "--config" in clean_output
    assert "--model" in clean_output
    assert "--task" in clean_output
    assert "--audit-timeout" in clean_output
    assert "--output" in clean_output


def test_python_m_minisweagent_help():
    """Test that python -m minisweagent --help works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "mini-SWE-agent" in result.stdout


def test_mini_script_help():
    """Test that the mini script entry point help works."""
    result = subprocess.run(
        ["mini", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "mini-SWE-agent" in result.stdout


def test_installed_commands_have_no_aliases():
    assert {
        entry.name for entry in distribution("mini-swe-agent").entry_points if entry.group == "console_scripts"
    } == {"mini", "mini-extra"}


def test_mini_extra_help():
    """Test that mini-extra --help works correctly."""
    result = subprocess.run(
        ["mini-extra", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    clean_output = strip_ansi_codes(result.stdout)
    assert "Run extra mini-SWE-agent commands." in clean_output
    assert "inspect" in clean_output
    assert "swebench" not in clean_output


@pytest.mark.parametrize(
    ("subcommand", "args"),
    [("i", []), ("inspector", ["--help"]), ("missing", []), ("swebench", []), ("swebench-single", ["--help"])],
)
def test_mini_extra_rejects_aliases_and_unknown_commands(subcommand: str, args: list[str]):
    result = subprocess.run(
        ["mini-extra", subcommand, *args],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert subcommand in result.stderr


def test_mini_extra_inspect_help():
    """Test that the remaining installed utility works."""
    result = subprocess.run(
        ["mini-extra", "inspect", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "usage: mini-extra inspect" in result.stdout
    assert "--help" in result.stdout
    assert "--reasoning" in result.stdout


def test_mini_rejects_environment_selection(capsys):
    with pytest.raises(SystemExit) as error:
        app(["--environment-class", "docker"])
    assert error.value.code == 2
    assert "unrecognized arguments: --environment-class docker" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("environment_class", "overrides"),
    [
        ("docker", []),
        ("minisweagent.environments.docker.DockerEnvironment", []),
        ("local", ["environment.environment_class=docker"]),
    ],
)
def test_mini_rejects_nonlocal_config_before_loading_model(
    tmp_path: Path, environment_class: str, overrides: list[str]
):
    config = tmp_path / "agent.yaml"
    config.write_text(
        f"environment:\n  environment_class: {environment_class}\n"
        "model:\n  model_name: unused\n  model_class: unavailable\n"
    )
    with pytest.raises(SystemExit, match="run mini itself inside your container"):
        main(config_spec=[str(config), *overrides], output=tmp_path / "trajectory.json")
    assert not (tmp_path / "trajectory.json").exists()


def test_exit_immediately_flag_sets_confirm_exit_false():
    """Test that --exit-immediately flag sets confirm_exit to False in agent config."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = False
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function with --exit-immediately flag
        agent = main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            audit_timeout=None,
            output=None,
            exit_immediately=True,  # This should set confirm_exit=False
            model_class=None,
            agent_class=None,
        )

        # Verify the agent's config has confirm_exit set to False
        assert agent.config.confirm_exit is False


def test_no_exit_immediately_flag_sets_confirm_exit_true():
    """Test that when --exit-immediately flag is not used, confirm_exit defaults to True."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Create mock agent with config
        mock_agent = Mock()
        mock_agent.config.confirm_exit = True
        mock_agent.run.return_value = {"exit_status": "Success", "submission": "Result"}
        mock_get_agent.return_value = mock_agent

        # Call main function without --exit-immediately flag (defaults to False)
        agent = main(
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            model_name="test-model",
            task="Test task",
            audit_timeout=None,
            output=None,
            model_class=None,
            agent_class=None,
        )

        # Verify the agent's config has confirm_exit set to True
        assert agent.config.confirm_exit is True


def test_exit_immediately_flag_with_cli_adapter():
    """Test --exit-immediately through the standard-library CLI adapter."""
    with (
        patch("minisweagent.run.mini.get_agent") as mock_get_agent,
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.run.mini.get_config_from_spec") as mock_get_config,
    ):
        # Setup mocks
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        mock_get_config.return_value = {"agent": {"system_template": "test"}, "env": {}, "model": {}}

        # Setup mock agent instance
        mock_agent = Mock()
        mock_agent.run.return_value = {"exit_status": "Success", "result": "Result"}
        mock_agent.messages = []
        mock_get_agent.return_value = mock_agent

        assert app(
            ["--task", "Test task", "--exit-immediately", "--model", "test-model", "--config", str(DEFAULT_CONFIG_FILE)]
        ) == 0

        mock_get_agent.assert_called_once()
        args, kwargs = mock_get_agent.call_args
        # The config (third positional arg) should contain confirm_exit
        assert args[2].get("confirm_exit") is False


def test_output_file_is_created(tmp_path):
    """Test that output trajectory file is created when --output is specified."""
    output_file = tmp_path / "test_traj.json"

    # Create a temporary config file
    config_file = tmp_path / "test_config.yaml"
    default_config_path = Path("src/minisweagent/config/default.yaml")
    config_file.write_text(default_config_path.read_text())

    with (
        patch("minisweagent.run.mini.get_model") as mock_get_model,
        patch("minisweagent.agents.prompt_user._multiline_prompt_session.prompt", return_value=""),
    ):
        # Setup mocks
        mock_model = Mock()
        mock_model.config = Mock()
        mock_model.config.model_dump.return_value = {}
        mock_model.serialize.return_value = {
            "info": {
                "config": {"model": {}, "model_type": "MockModel"},
            }
        }
        mock_model.get_template_vars.return_value = {}
        mock_model.format_message.side_effect = lambda **kwargs: dict(**kwargs)
        # query now returns dict with extra["actions"]
        mock_model.query.side_effect = [
            {
                "role": "assistant",
                "content": "```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done\n```",
                "extra": {"actions": [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\necho done"}]},
            },
        ]
        # format_observation_messages returns observation messages
        mock_model.format_observation_messages.return_value = []
        mock_get_model.return_value = mock_model

        assert app(
            [
                "--task",
                "Test task",
                "--model",
                "test-model",
                "--output",
                str(output_file),
                "--config",
                str(config_file),
            ],
        ) == 0
        assert output_file.exists(), f"Output file {output_file} was not created"
