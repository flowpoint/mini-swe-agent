import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

from minisweagent.models.test_models import make_output


@pytest.fixture
def history_config(tmp_path: Path) -> Path:
    path = tmp_path / "agent.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "agent": {"agent_class": "default", "system_template": "Test agent", "instance_template": "{{task}}"},
                "model": {
                    "model_class": "deterministic",
                    "model_name": "deterministic",
                    "outputs": [make_output("Done", [{"command": "print('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')"}])],
                },
            }
        )
    )
    return path


def run_script(tmp_path: Path, code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=tmp_path,
        env={
            **os.environ,
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"),
            "MSWEA_SILENT_STARTUP": "1",
        },
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_runs_keep_history_and_latest_including_failures(tmp_path: Path, history_config: Path):
    result = run_script(
        tmp_path,
        """
import sys
from minisweagent.run.mini import main

for task in ("first", "second"):
    main(task=task, config_spec=[sys.argv[1]])
main(task="failed", config_spec=[sys.argv[1], "model.outputs=[]"])
""",
        str(history_config),
    )
    assert result.returncode != 0
    assert "IndexError" in result.stderr
    directory = tmp_path / "data/mini-swe-agent/trajectories"
    histories = {
        json.loads(path.read_text())["messages"][1]["content"]: path
        for path in directory.glob("*.traj.json")
        if not path.is_symlink()
    }
    assert set(histories) == {"first", "second", "failed"}
    for task, path in histories.items():
        assert json.loads(path.read_text())["info"]["exit_status"] == (
            "IndexError" if task == "failed" else "Submitted"
        )
    assert (directory / "latest.traj.json").readlink() == Path(histories["failed"].name)
    assert (directory / "latest.traj.json").resolve() == histories["failed"]
    assert not list((tmp_path / "config").glob("*.traj.json"))
    assert len(list(directory.glob("*.traj.json"))) == 4
    assert len(list(directory.glob("*.outputs/command-*.log"))) == 2


def test_concurrent_runs_have_distinct_files(tmp_path: Path, history_config: Path):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda task: run_script(
                    tmp_path,
                    """
import sys
from minisweagent.run.mini import app
app(["-c", sys.argv[1], "-t", sys.argv[2]])
""",
                    str(history_config),
                    task,
                ),
                ["one", "two", "three", "four"],
            )
        )
    for result in results:
        assert result.returncode == 0, result.stderr
    directory = tmp_path / "data/mini-swe-agent/trajectories"
    paths = [path for path in directory.glob("*.traj.json") if not path.is_symlink()]
    assert {json.loads(path.read_text())["messages"][1]["content"] for path in paths} == {
        "one",
        "two",
        "three",
        "four",
    }
    assert (directory / "latest.traj.json").resolve() in paths
    assert len(list(directory.glob("*.traj.json"))) == 5
    assert len(list(directory.glob("*.outputs/command-*.log"))) == 4


def test_explicit_output_and_disabled_saving(tmp_path: Path, history_config: Path):
    result = run_script(
        tmp_path,
        """
import sys
from minisweagent.run.mini import app, main

app(["-c", sys.argv[1], "-t", "custom", "-o", "custom.traj.json"])
main(task="unsaved", config_spec=[sys.argv[1]], output=None)
""",
        str(history_config),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / "custom.traj.json").read_text())["messages"][1]["content"] == "custom"
    assert not (tmp_path / "data/mini-swe-agent/trajectories").exists()
    assert len(list((tmp_path / "data/mini-swe-agent/outputs").glob("command-*.log"))) == 1
    assert len(list((tmp_path / "custom.traj.outputs").glob("command-*.log"))) == 1


@pytest.mark.parametrize(
    ("data_home", "expected_root"),
    [(None, ".local/share"), ("", ".local/share"), ("relative/path", ".local/share"), ("/custom/data", "/custom/data")],
)
def test_data_directory_respects_xdg(tmp_path: Path, data_home: str | None, expected_root: str):
    result = run_script(
        tmp_path,
        """
import os
import sys

os.environ.pop("XDG_DATA_HOME", None)
if len(sys.argv) > 1:
    os.environ["XDG_DATA_HOME"] = sys.argv[1]
from minisweagent.run.mini import DEFAULT_OUTPUT_FILE
print(DEFAULT_OUTPUT_FILE)
""",
        *([] if data_home is None else [data_home]),
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == Path.home() / expected_root / "mini-swe-agent/trajectories/latest.traj.json"
    assert not (tmp_path / "data").exists()
