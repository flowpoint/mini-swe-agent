import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.swebench import DEFAULT_CONFIG_FILE, SWEBENCH_DEPENDENCY_ERROR
from benchmarks.swebench_single import DEFAULT_CONFIG_FILE as SINGLE_CONFIG_FILE
from minisweagent.config import get_config_from_spec
from tests.test_minimal_dependencies import _run_with_removed_dependencies


@pytest.mark.parametrize(
    ("module", "option"),
    [("swebench", "--workers"), ("swebench_single", "--instance")],
)
def test_checkout_runner_help(module, option):
    result = subprocess.run(
        [sys.executable, "-m", f"benchmarks.{module}", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert option in result.stdout
    assert "-c benchmarks/swebench.yaml" in result.stdout


def test_checkout_default_config():
    assert DEFAULT_CONFIG_FILE == SINGLE_CONFIG_FILE == Path(__file__).resolve().parents[2] / "benchmarks/swebench.yaml"
    config = get_config_from_spec(DEFAULT_CONFIG_FILE)
    assert config["environment"]["environment_class"] == "docker"
    assert config["model"]["model_class"] == "llamacpp"


def test_swebench_command_reports_dependency_without_side_effects(tmp_path):
    output_path = tmp_path / "swebench-output"
    result = _run_with_removed_dependencies(
        f"""
from benchmarks.swebench import app
app(["--output", {str(output_path)!r}])
""",
        tmp_path,
    )
    assert result.returncode == 1
    assert SWEBENCH_DEPENDENCY_ERROR in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()
