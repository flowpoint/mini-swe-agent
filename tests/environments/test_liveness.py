import shlex
import sys
import time
from pathlib import Path

import pytest

from minisweagent.environments.local import LocalEnvironment


@pytest.mark.parametrize(
    ("activity", "body"),
    [
        ("cpu", "sum(range(10000))"),
        ("io", "Path(__file__).read_bytes(); time.sleep(0.05)"),
        ("output", "print('alive', flush=True); time.sleep(0.05)"),
    ],
)
def test_child_activity_keeps_commands_alive(tmp_path: Path, activity: str, body: str):
    script = tmp_path / f"{activity}.py"
    script.write_text(
        "import time\nfrom pathlib import Path\n"
        "end = time.monotonic() + 1.2\n"
        f"while time.monotonic() < end:\n    {body}\n"
        "print('finished')\n"
    )
    parent = f"import subprocess; subprocess.run([{sys.executable!r}, {str(script)!r}], check=True)"
    started = time.monotonic()
    result = LocalEnvironment(timeout=0.3, output_dir=tmp_path).execute(
        {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(parent)}"}
    )
    assert result["returncode"] == 0, result
    assert time.monotonic() - started > 1.2
    assert result["output"].endswith("finished\n")
    assert Path(result["extra"]["output_path"]).read_text() == result["output"]


def test_inactivity_stops_stalls_and_can_be_disabled(tmp_path: Path):
    env = LocalEnvironment(timeout=0.2, output_dir=tmp_path)
    started = time.monotonic()
    result = env.execute({"command": "printf 'before stall'; sleep 10"})
    assert time.monotonic() - started < 3
    assert result["returncode"] == -1
    assert result["extra"]["exception_type"] == "TimeoutExpired"
    assert result["output"] == Path(result["extra"]["output_path"]).read_text() == "before stall"
    env.config.timeout = None
    assert env.execute({"command": "sleep 0.4; echo done"})["output"] == "done\n"


def test_large_output_is_bounded_and_kept_in_unique_files(tmp_path: Path):
    env = LocalEnvironment(max_output_bytes=1024, output_dir=tmp_path)
    command = f"{shlex.quote(sys.executable)} -c \"print('HEAD' + 'x' * 3000000 + 'TAIL')\""
    result = env.execute({"command": command})
    path = Path(result["extra"]["output_path"])
    assert result["returncode"] == 0
    assert result["output"].startswith("HEAD") and result["output"].endswith("TAIL\n")
    assert str(path) in result["output"] and "truncated" in result["output"]
    assert len(result["output"]) < 1024 + len(str(path)) + 100
    assert path.read_text() == "HEAD" + "x" * 3000000 + "TAIL\n"
    assert Path(env.execute({"command": "echo next"})["extra"]["output_path"]) != path
    assert path.stat().st_size == 3000009
