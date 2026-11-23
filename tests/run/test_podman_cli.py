import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.environments.test_podman import podman_image as podman_image
from tests.run.test_setup import chat_response
from tests.run.test_setup import model_api as model_api


def _cli(tmp_path: Path, url: str, *options: str, environment: dict | None = None, path: str | None = None):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "model": {"model_name": "test:model", "base_url": url + "/v1"},
                "environment": {"cwd": str(workspace), **(environment or {})},
            }
        )
    )
    return subprocess.run(
        [
            sys.executable,
            "-P",
            "-m",
            "minisweagent",
            "-c",
            "mini.yaml",
            "-c",
            str(config),
            "--task",
            "Finish",
            "--exit-immediately",
            "--step-limit",
            "1",
            *options,
        ],
        cwd=workspace,
        env=os.environ
        | {
            "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "PATH": os.environ.get("PATH", "") if path is None else path,
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("options", "settings"),
    [(["--sandbox"], {}), (["--ask-network"], {}), ([], {"environment_class": "podman"})],
)
def test_sandbox_requires_explicit_image_before_model_access(tmp_path: Path, model_api, options: list, settings: dict):
    url, _, calls = model_api
    result = _cli(tmp_path, url, *options, environment=settings)
    assert result.returncode != 0
    assert "sandbox requires --sandbox-image IMAGE" in result.stderr
    assert calls == []
    assert not (tmp_path / "data/mini-swe-agent/trajectories").exists()


@pytest.mark.parametrize(
    ("options", "settings"),
    [
        (["--sandbox", "--sandbox-image", "localhost/test"], {}),
        (["--ask-network", "--sandbox-image", "localhost/test"], {}),
        (["--sandbox-image", "localhost/test"], {}),
        ([], {"environment_class": "podman", "image": "localhost/test"}),
    ],
)
def test_missing_podman_never_falls_back_to_local(tmp_path: Path, model_api, options: list, settings: dict):
    url, _, calls = model_api
    result = _cli(tmp_path, url, *options, environment=settings, path="")
    assert result.returncode != 0
    assert "Podman is required" in result.stderr
    assert calls == []
    assert not (tmp_path / "data/mini-swe-agent/trajectories").exists()


def test_conflicting_sandbox_flags_are_rejected(tmp_path: Path, model_api):
    url, _, calls = model_api
    result = _cli(tmp_path, url, "--sandbox", "--ask-network")
    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr
    assert calls == []


def test_cli_sandbox_keeps_model_access_and_saves_decisions(tmp_path: Path, model_api, podman_image: str):
    url, routes, calls = model_api
    routes["/llama/v1/chat/completions"] = (
        200,
        chat_response(
            "```python\nfrom pathlib import Path\nPath('executed').touch()\n"
            "print('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')\n```"
        ),
    )
    result = _cli(
        tmp_path,
        url,
        "--sandbox",
        "--sandbox-image",
        podman_image,
        "--language",
        "python",
        environment={"environment_class": "podman", "network": "ask", "language": "bash"},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "workspace/executed").exists()
    assert [call[0] for call in calls] == ["GET", "POST"]
    trajectory = json.loads((tmp_path / "data/mini-swe-agent/trajectories/latest.traj.json").read_text())
    assert trajectory["info"]["config"]["environment"]["network"] == "none"
    assert trajectory["info"]["config"]["environment"]["language"] == "python"
    assert trajectory["messages"][-1]["extra"]["network_access"]["allowed"] is False
    assert trajectory["messages"][-1]["extra"]["sandbox_image"]


def test_unavailable_image_fails_before_model_access(tmp_path: Path, model_api, podman_image: str):
    url, _, calls = model_api
    result = _cli(tmp_path, url, "--sandbox", "--sandbox-image", "localhost/mini-test-absent:does-not-exist")
    assert result.returncode != 0
    assert "Podman image inspect failed" in result.stderr
    assert calls == []
    assert list((tmp_path / "workspace").iterdir()) == []
