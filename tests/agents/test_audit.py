import json
import os
import selectors
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import pytest
from pydantic import ValidationError

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output


def test_each_command_waits_then_runs_without_input():
    for timeout in (0, 0.05):
        agent = DefaultAgent(
            DeterministicModel(outputs=[]),
            LocalEnvironment(),
            system_template="Test",
            instance_template="{{task}}",
            audit_timeout=timeout,
        )
        started = time.monotonic()
        messages = agent.execute_actions(
            make_output("Run", [{"command": "print('first')"}, {"command": "print('second')"}])
        )
        assert time.monotonic() - started >= 2 * timeout
        assert "first" in messages[0]["content"]
        assert "second" in messages[1]["content"]


@contextmanager
def running_agent(tmp_path: Path, timeout: str, commands: list[str]):
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import json
import sys
from pathlib import Path
from prompt_toolkit.application import create_app_session
from prompt_toolkit.output import DummyOutput
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel, make_output

agent = DefaultAgent(
    DeterministicModel(outputs=[]), LocalEnvironment(),
    system_template='Test', instance_template='{{task}}', audit_timeout=float(sys.argv[1]),
)
with create_app_session(output=DummyOutput()):
    try:
        agent.execute_actions(make_output('Run', [{'command': command} for command in json.loads(sys.argv[2])]))
    finally:
        agent.save(Path('trajectory.json'))
""",
            timeout,
            json.dumps(commands),
        ],
        cwd=tmp_path,
        env={**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config")},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            yield process
        finally:
            if (pid_file := tmp_path / "running.pid").exists():
                with suppress(ProcessLookupError):
                    os.killpg(int(pid_file.read_text()), signal.SIGKILL)
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)


def wait_for_audit(process: subprocess.Popen):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=5), "Audit window did not start"
    assert "Audit (" in process.stdout.readline()


@pytest.mark.parametrize(
    ("response", "error"), [("reject this command\x07", "UserInterruption"), (None, "KeyboardInterrupt")]
)
def test_infinite_audit_requires_continuation_for_each_command(tmp_path: Path, response: str | None, error: str):
    with running_agent(
        tmp_path,
        "inf",
        [f"from pathlib import Path; Path('{name}.txt').write_text('{name}\\n')" for name in ("first", "second")],
    ) as process:
        wait_for_audit(process)
        with pytest.raises(subprocess.TimeoutExpired):
            process.wait(timeout=0.05)
        assert not (tmp_path / "first.txt").exists()
        process.stdin.write("\x07")
        process.stdin.flush()
        wait_for_audit(process)
        assert (tmp_path / "first.txt").read_text() == "first\n"
        assert not (tmp_path / "second.txt").exists()
        if response is None:
            process.send_signal(signal.SIGINT)
        else:
            process.stdin.write(response)
            process.stdin.flush()
        _, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert error in stderr
        assert not (tmp_path / "second.txt").exists()
        trajectory = json.loads((tmp_path / "trajectory.json").read_text())
        assert len(trajectory["messages"]) == 1
        assert trajectory["info"]["config"]["agent"]["audit_timeout"] == "inf"
        json.dumps(trajectory, allow_nan=False)


def test_external_interrupt_vetoes_finite_audit(tmp_path: Path):
    with running_agent(tmp_path, "5", ["from pathlib import Path; Path('forbidden').touch()"]) as process:
        wait_for_audit(process)
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert "KeyboardInterrupt" in stderr
        assert not (tmp_path / "forbidden").exists()


def test_external_interrupt_stops_running_local_command_at_zero_timeout(tmp_path: Path):
    with running_agent(
        tmp_path,
        "0",
        [
            "import os, time; from pathlib import Path; Path('process.tmp').write_text(str(os.getpid())); Path('process.tmp').rename('running.pid'); time.sleep(30)"
        ],
    ) as process:
        deadline = time.monotonic() + 5
        while not (tmp_path / "running.pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        pid = int((tmp_path / "running.pid").read_text())
        process.send_signal(signal.SIGINT)
        _, stderr = process.communicate(timeout=5)
        assert process.returncode != 0
        assert "KeyboardInterrupt" in stderr
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize(("timeout", "argument"), [(-1, "-1"), (float("nan"), "nan"), (float("-inf"), "-inf")])
def test_invalid_audit_timeout_fails_in_config_and_cli(tmp_path: Path, timeout: float, argument: str):
    with pytest.raises(ValidationError, match="audit_timeout"):
        AgentConfig(system_template="Test", instance_template="{{task}}", audit_timeout=timeout)
    result = subprocess.run(
        [sys.executable, "-m", "minisweagent", f"--audit-timeout={argument}"],
        env={**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config")},
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 2
    assert "must be non-negative" in result.stderr
