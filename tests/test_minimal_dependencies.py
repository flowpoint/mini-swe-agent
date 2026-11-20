import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.llamacpp_model import LlamaCppModel

_REPO_ROOT = Path(__file__).parent.parent


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


@pytest.mark.parametrize(
    ("config_name", "language"),
    [(name, language) for name in ("default.yaml", "mini.yaml") for language in ("python", "bash")],
)
def test_default_configs_select_deterministic_llamacpp_text_actions(config_name, language):
    config = get_config_from_spec(builtin_config_dir / config_name)
    model = config["model"]
    agent = config["agent"]

    assert model["model_class"] == "llamacpp"
    assert model["model_name"] == "local"
    assert model["base_url"] == "http://127.0.0.1:8080/v1"
    assert model["request_timeout"] == 600
    assert model["connect_timeout"] == 10
    assert model["model_kwargs"] == {"temperature": 0, "max_tokens": 4096}
    assert config["environment"]["language"] == "python"
    assert model.get("action_regex") is None
    assert agent["step_limit"] == 0
    assert agent.get("wall_time_limit_seconds", 0) == 0
    assert agent.get("cost_limit", 0) == 0
    adapter = LlamaCppModel(**{key: value for key, value in model.items() if key != "model_class"})
    with adapter.session:
        runtime = DefaultAgent(adapter, LocalEnvironment(**(config["environment"] | {"language": language})), **agent)
        prompt = runtime._render_template(agent["system_template"])
        assert "exactly one fenced" in prompt and f"```{language}" in prompt
        assert adapter.parse_actions(prompt, target=runtime.env.target) == [{"command": runtime.env.target.example}]
