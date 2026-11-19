import json
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from minisweagent.models.llamacpp_model import LlamaCppModel, LlamaCppProtocolError
from minisweagent.models.test_models import make_output
from tests.run.test_setup import run_cli


@pytest.fixture
def streaming_api():
    state = {
        "delay": 0,
        "mode": "complete",
        "token": None,
        "content": "```python\nprint('COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT')\n```",
    }
    calls, release = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply(None)

        def do_POST(self):
            self.reply(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def reply(self, payload):
            auth = self.headers.get("Authorization")
            calls.append((self.command, self.path, auth, payload))
            if state["token"] and auth != f"Bearer {state['token']}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if payload else "application/json")
            self.end_headers()
            if payload is None:
                self.wfile.write(json.dumps({"data": [{"id": "local"}]} if self.path == "/v1/models" else {}).encode())
                return
            assert payload["stream"] is True
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                if state["mode"] == "stall":
                    release.wait(5)
                    return
                for fragment in [state["content"][i : i + 10] for i in range(0, len(state["content"]), 10)]:
                    time.sleep(state["delay"])
                    self.wfile.write(
                        ("data: " + json.dumps({"choices": [{"delta": {"content": fragment}}]}) + "\n\n").encode()
                    )
                    self.wfile.flush()
                if state["mode"] == "disconnect":
                    return
                self.wfile.write(b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n')
                self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 8}}\n\n')
                self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}/v1", state, calls
        release.set()
        server.shutdown()
        thread.join()


def test_setup_and_run_use_auth_and_keep_slow_active_streams_alive(tmp_path: Path, streaming_api):
    url, state, calls = streaming_api
    state.update(token="secret-test-token", delay=0.08)
    env = {"MINI_TEST_TOKEN": state["token"]}
    started = time.monotonic()
    result = run_cli(
        tmp_path,
        "setup",
        "--base-url",
        url,
        "--api-key-env",
        "MINI_TEST_TOKEN",
        "--check-timeout",
        "0.2",
        "--request-timeout",
        "0.25",
        env_overrides=env,
    )
    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started > 0.25
    assert "auth: $MINI_TEST_TOKEN" in result.stdout
    config = (tmp_path / "config/mini-swe-agent/mini.yaml").read_text()
    assert 'api_key_env: "MINI_TEST_TOKEN"' in config
    assert "request_timeout: 0.25" in config
    result = run_cli(tmp_path, "-t", "finish", "--exit-immediately", env_overrides=env)
    assert result.returncode == 0, result.stderr
    trajectory = tmp_path / "data/mini-swe-agent/trajectories/latest.traj.json"
    saved = json.loads(trajectory.read_text())
    assert saved["info"]["exit_status"] == "Submitted"
    assert saved["messages"][2]["extra"]["usage"] == {"prompt_tokens": 12, "completion_tokens": 8}
    assert saved["info"]["model_stats"]["usage"] == {"prompt_tokens": 12, "completion_tokens": 8}
    assert all(call[2] == f"Bearer {state['token']}" for call in calls)
    assert state["token"] not in config + trajectory.read_text() + result.stdout + result.stderr
    count = len(calls)
    result = run_cli(tmp_path, "-t", "finish", "--exit-immediately", env_overrides={"MINI_TEST_TOKEN": ""})
    assert result.returncode != 0 and "MINI_TEST_TOKEN" in result.stderr
    assert len(calls) == count


@pytest.mark.parametrize(
    ("mode", "error"), [("stall", requests.RequestException), ("disconnect", LlamaCppProtocolError)]
)
def test_silent_or_incomplete_streams_fail_without_replaying_the_request(streaming_api, mode, error):
    url, state, calls = streaming_api
    state["mode"] = mode
    model = LlamaCppModel(base_url=url, model_name="local", request_timeout=0.15)
    try:
        with pytest.raises(error):
            model.query([{"role": "user", "content": "test"}])
    finally:
        model.close()
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("mode", "content", "code", "status"),
    [
        (
            "disconnect",
            "```python\nfrom pathlib import Path; Path('should-not-run').touch()\n```",
            1,
            "LlamaCppProtocolError",
        ),
        ("complete", "no python action", 4, "RepeatedFormatError"),
    ],
)
def test_protocol_failures_have_distinct_exit_codes_and_saved_history(
    tmp_path: Path, streaming_api, mode, content, code, status
):
    url, state, _ = streaming_api
    state.update(mode=mode, content=content)
    config = tmp_path / "config.yaml"
    config.write_text(json.dumps({"model": {"base_url": url}}))
    path = tmp_path / "trajectory.json"
    result = run_cli(
        tmp_path, "-c", "mini.yaml", "-c", str(config), "-t", "test", "-o", str(path), "--exit-immediately"
    )
    assert result.returncode == code, result.stderr
    assert json.loads(path.read_text())["info"]["exit_status"] == status
    if status == "RepeatedFormatError":
        assert json.loads(path.read_text())["info"]["model_stats"]["usage"] == {
            "prompt_tokens": 36,
            "completion_tokens": 24,
        }
    assert not (tmp_path / "should-not-run").exists()


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("agent.audit_timout=3", "audit_timout"),
        ("model.request_timout=1", "request_timout"),
        ("environment.timout=1", "timout"),
        ("environment.language=typo", "language"),
        ("environment.interpreter=[]", "interpreter"),
        ("run.taks=test", "taks"),
        ("agnt.step_limit=1", "agnt"),
    ],
)
def test_config_typos_fail_before_network_or_trajectory_creation(tmp_path: Path, streaming_api, override, error):
    url, _, calls = streaming_api
    result = run_cli(tmp_path, "-c", "mini.yaml", "-c", f"model.base_url={url}", "-c", override, "-t", "test")
    assert result.returncode != 0 and error in result.stderr
    assert calls == []
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("limit", "task", "status"), [(1, "test", "LimitsExceeded"), (100, "x" * 300, "ContextLimitExceeded")]
)
def test_batch_limits_exit_nonzero_with_complete_trajectories(tmp_path: Path, limit: int, task: str, status: str):
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "agent": {
                    "agent_class": "default",
                    "system_template": "test",
                    "instance_template": "{{task}}",
                    "step_limit": limit,
                    "max_context_chars": 250,
                },
                "model": {
                    "model_class": "deterministic",
                    "model_name": "deterministic",
                    "outputs": [make_output("test", [{"command": "pass"}])],
                },
            }
        )
    )
    result = run_cli(tmp_path, "-c", str(config), "-t", task, "--exit-immediately")
    assert result.returncode == 3, result.stderr
    latest = tmp_path / "data/mini-swe-agent/trajectories/latest.traj.json"
    assert latest.is_symlink()
    assert json.loads(latest.read_text())["info"]["exit_status"] == status


def test_batch_interrupt_saves_history_and_stops_command_group(tmp_path: Path):
    config, trajectory = tmp_path / "config.yaml", tmp_path / "trajectory.json"
    config.write_text(
        json.dumps(
            {
                "agent": {"agent_class": "default", "system_template": "test", "instance_template": "{{task}}"},
                "model": {
                    "model_class": "deterministic",
                    "model_name": "deterministic",
                    "outputs": [
                        make_output(
                            "wait",
                            [
                                {
                                    "command": "import os, time; from pathlib import Path; print('partial', end=''); Path('command.pid').write_text(str(os.getpid())); time.sleep(60)"
                                }
                            ],
                        )
                    ],
                },
            }
        )
    )
    with subprocess.Popen(
        [sys.executable, "-m", "minisweagent", "-c", str(config), "-t", "test", "-o", str(trajectory)],
        cwd=tmp_path,
        env=os.environ | {"XDG_CONFIG_HOME": str(tmp_path / "config")},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            deadline = time.monotonic() + 5
            pid_file = tmp_path / "command.pid"
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert pid_file.exists(), "Command did not start"
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=5)
            assert process.returncode == 130, stdout + stderr
            assert json.loads(trajectory.read_text())["info"]["exit_status"] == "Interrupted"
            with pytest.raises(ProcessLookupError):
                os.killpg(int(pid_file.read_text()), 0)
            logs = list((tmp_path / "trajectory.outputs").glob("command-*.log"))
            assert len(logs) == 1 and logs[0].read_text() == "partial"
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
