import json
import os
import sys
from pathlib import Path

import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted


def test_python_actions_start_fresh_and_preserve_only_files(tmp_path: Path):
    env = LocalEnvironment(cwd=str(tmp_path), env={"MINI_FRESH": "configured"}, output_dir=tmp_path / "logs")
    first = env.execute(
        {
            "command": """import os, sys, json, types, builtins
from pathlib import Path
private_variable = 42
sys.modules['mini_private_module'] = types.ModuleType('mini_private_module')
builtins.mini_private_attribute = 42
Path('kept.json').write_text(json.dumps([os.getpid(), sys.executable]))
Path('child').mkdir()
os.chdir('child')
os.environ['MINI_FRESH'] = 'changed'
print('first')
"""
        }
    )
    assert first["returncode"] == 0, first
    second = env.execute(
        {
            "command": """import os, sys, json, builtins
from pathlib import Path
assert 'private_variable' not in globals()
assert 'mini_private_module' not in sys.modules
assert not hasattr(builtins, 'mini_private_attribute')
assert os.environ['MINI_FRESH'] == 'configured'
previous_pid, executable = json.loads(Path('kept.json').read_text())
assert previous_pid != os.getpid()
assert executable == sys.executable
print(json.dumps([os.getpid(), str(Path.cwd())]))
"""
        }
    )
    assert second["returncode"] == 0, second
    pid, cwd = json.loads(second["output"])
    assert pid != os.getpid() and cwd == str(tmp_path)
    assert Path(first["extra"]["output_path"]).read_text() == "first\n"
    assert first["extra"]["output_path"] != second["extra"]["output_path"]
    assert env.serialize()["info"]["config"]["environment"]["interpreter"] == [sys.executable, "-u", "-c"]
    assert env.serialize()["info"]["config"]["environment"]["language"] == "python"


def test_python_errors_are_observations_and_completion_still_works(tmp_path: Path):
    env = LocalEnvironment(cwd=str(tmp_path), output_dir=tmp_path)
    for source, error in [
        ("if True print('bad')", "SyntaxError"),
        ("    print('bad')", "IndentationError"),
        ("1 / 0", "ZeroDivisionError"),
    ]:
        result = env.execute({"command": source})
        assert result["returncode"] == 1 and error in result["output"]
        assert Path(result["extra"]["output_path"]).read_text() == result["output"]
    assert env.execute({"command": "print('$HOME; $(touch forbidden)')"})["output"] == "$HOME; $(touch forbidden)\n"
    assert not (tmp_path / "forbidden").exists()
    assert (
        env.execute({"command": "import sys; print('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'); sys.exit(2)"})[
            "returncode"
        ]
        == 2
    )
    with pytest.raises(Submitted) as submitted:
        env.execute({"command": env.target.submit + "\nprint('done')"})
    assert submitted.value.messages[0]["extra"]["submission"] == "done\n"
    assert Path(submitted.value.messages[0]["extra"]["output_path"]).read_text().endswith("\ndone\n")


def test_python_interpreter_override_is_executed_and_recorded(tmp_path: Path):
    env = LocalEnvironment(interpreter=[sys.executable, "-u", "-X", "dev", "-c"], output_dir=tmp_path)
    assert env.execute({"command": "import sys; print(sys.flags.dev_mode)"})["output"] == "True\n"
    assert env.serialize()["info"]["config"]["environment"]["interpreter"] == env.config.interpreter
