import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from minisweagent.models.test_models import DeterministicModel, make_output
from minisweagent.run.mini import DEFAULT_CONFIG_FILE, main
from tests.conftest import assert_observations_match


def test_mini_executes_locally_without_loading_docker(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = tmp_path / "agent.yaml"
    config.write_text(
        json.dumps(
            {
                "agent": {"agent_class": "default", "system_template": "Test agent", "instance_template": "{{task}}"},
                "model": {
                    "model_class": "deterministic",
                    "model_name": "deterministic",
                    "outputs": [
                        make_output(
                            "Write", [{"command": 'printf "%s\\n" "$MINI_TEST_VALUE" > result.txt; pwd >> result.txt'}]
                        ),
                        make_output("Done", [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}]),
                    ],
                },
                "environment": {
                    "environment_class": "local",
                    "cwd": str(workspace),
                    "env": {"MINI_TEST_VALUE": "local"},
                },
            }
        )
    )
    trajectory = tmp_path / "trajectory.json"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\nfrom minisweagent.run.mini import app\napp(sys.argv[1:])\n"
            "assert 'minisweagent.environments.docker' not in sys.modules",
            "-c",
            str(config),
            "-t",
            "Write in the working directory",
            "-o",
            str(trajectory),
        ],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (workspace / "result.txt").read_text() == f"local\n{workspace}\n"
    info = json.loads(trajectory.read_text())["info"]
    assert info["exit_status"] == "Submitted"
    assert info["config"]["environment_type"] == "minisweagent.environments.local.LocalEnvironment"


def _make_model_from_fixture(text_outputs: list[str], cost_per_call: float = 1.0, **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from trajectory fixture data (raw text outputs)."""

    def parse_command(text: str) -> list[dict]:
        match = re.search(r"```mswea_bash_command\s*\n(.*?)\n```", text, re.DOTALL)
        return [{"command": match.group(1)}] if match else []

    return DeterministicModel(
        outputs=[make_output(text, parse_command(text), cost=cost_per_call) for text in text_outputs],
        cost_per_call=cost_per_call,
        **kwargs,
    )


def test_local_end_to_end(local_test_data):
    """Test the complete flow from CLI to final result using real environment but deterministic model"""

    model_responses = local_test_data["model_responses"]
    expected_observations = local_test_data["expected_observations"]

    with (
        patch("minisweagent.run.mini.get_model") as mock_model_class,
        patch("minisweagent.agents.utils.prompt_user.prompt_session.prompt", side_effect=lambda *a, **kw: ""),
        patch(
            "minisweagent.agents.utils.prompt_user._multiline_prompt_session.prompt", side_effect=lambda *a, **kw: ""
        ),
        patch("builtins.input", return_value=""),  # For LimitsExceeded handling
    ):
        mock_model_class.return_value = _make_model_from_fixture(model_responses)
        agent = main(
            model_name="tardis",
            config_spec=[str(DEFAULT_CONFIG_FILE)],
            audit_timeout=0,
            task="Blah blah blah",
            output=None,
            cost_limit=10,
            model_class=None,
            agent_class=None,
        )  # type: ignore

    assert agent is not None
    messages = agent.messages

    # Verify we have the right number of messages
    # Should be: system + user (initial) + (assistant + user) * number_of_steps
    expected_total_messages = 2 + (len(model_responses) * 2)
    assert len(messages) == expected_total_messages, f"Expected {expected_total_messages} messages, got {len(messages)}"

    assert_observations_match(expected_observations, messages)

    assert agent.n_calls == len(model_responses), f"Expected {len(model_responses)} steps, got {agent.n_calls}"
