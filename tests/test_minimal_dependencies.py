import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from minisweagent.config import builtin_config_dir, get_config_from_spec

_REPO_ROOT = Path(__file__).parent.parent
_SWEBENCH_DEPENDENCY_ERROR = (
    "SWE-bench support requires the optional dependency group:\npip install 'mini-swe-agent[swebench]'"
)


def _run_with_removed_dependencies(code: str, tmp_path: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "MSWEA_SILENT_STARTUP": "1",
        "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"),
    }
    pythonpath = [str(_REPO_ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    bootstrap = """
import sys
sys.modules.update({"datasets": None, "litellm": None, "openai": None, "platformdirs": None, "typer": None})
"""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(bootstrap + code)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_package_import_does_not_require_removed_dependencies(tmp_path):
    result = _run_with_removed_dependencies(
        """
import minisweagent
import minisweagent.models
print("import-ok")
""",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-ok"


def test_main_cli_help_does_not_require_datasets(tmp_path):
    result = _run_with_removed_dependencies(
        """
import io
from contextlib import redirect_stdout

from minisweagent.run.mini import app

stdout = io.StringIO()
try:
    with redirect_stdout(stdout):
        app(["--help"])
except SystemExit as e:
    assert e.code == 0
else:
    raise AssertionError("Expected --help to exit")
assert "Run mini-SWE-agent in your local environment." in stdout.getvalue()
""",
        tmp_path,
    )

    assert result.returncode == 0, result.stderr


def test_swebench_command_reports_optional_dependency_without_side_effects(tmp_path):
    output_path = tmp_path / "swebench-output"
    result = _run_with_removed_dependencies(
        f"""
import sys
from minisweagent.run.utilities.mini_extra import main
sys.argv = ["mini-extra", "swebench", "--output", {str(output_path)!r}]
main()
""",
        tmp_path,
    )

    assert result.returncode != 0
    assert _SWEBENCH_DEPENDENCY_ERROR in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()


@pytest.mark.parametrize("config_name", ["default.yaml", "mini.yaml"])
def test_default_configs_select_deterministic_llamacpp_text_actions(config_name):
    config = get_config_from_spec(builtin_config_dir / config_name)
    model = config["model"]
    agent = config["agent"]

    assert model["model_class"] == "llamacpp"
    assert model["model_name"] == "local"
    assert model["base_url"] == "http://127.0.0.1:8080/v1"
    assert model["request_timeout"] == 600
    assert model["connect_timeout"] == 10
    assert model["model_kwargs"] == {"temperature": 0, "max_tokens": 4096}
    assert model["action_regex"] == r"```bash\s*\n(.*?)\n```"
    assert agent["step_limit"] == 100
    assert agent["wall_time_limit_seconds"] == 3600
    assert agent.get("cost_limit", 0) == 0
    prompt = f"{agent['system_template']}\n{agent['instance_template']}".lower()
    assert "exactly one fenced" in prompt
    assert "```bash" in prompt
