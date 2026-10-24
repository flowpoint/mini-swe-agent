import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml


def chat_response(content: str, finish_reason: str = "stop") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}]}


@pytest.fixture
def model_api(request):
    prefix, port = getattr(request, "param", ("/llama", 0))
    if port:
        with socket.socket() as probe:
            probe.settimeout(0.1)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                pytest.skip(f"Local port {port} is already in use")
    routes = {
        "/llama/v1/models": (200, {"data": [{"id": "test:model"}]}),
        "/llama/props": (
            200,
            {
                "default_generation_settings": {
                    "n_ctx": 2048,
                    "params": {"temperature": 0.6, "max_tokens": -1, "top_p": 0.9, "top_k": 30},
                },
            },
        ),
        "/llama/v1/chat/completions": (
            200,
            chat_response("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"),
        ),
    }
    routes = {path.replace("/llama", prefix, 1): response for path, response in routes.items()}
    calls = []
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(("GET", self.path, None))
            self.reply()

        def do_POST(self):
            calls.append(("POST", self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.reply()

        def reply(self):
            status, body = routes.get(urlsplit(self.path).path, (404, {}))
            if status == 0:
                release.wait(10)
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}{prefix}", routes, calls
        release.set()
        server.shutdown()
        thread.join()


def run_cli(tmp_path: Path, *args: str, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "MSWEA_SILENT_STARTUP": "",
    }
    for key in ("MSWEA_GLOBAL_CONFIG_DIR", "MSWEA_MINI_CONFIG_PATH"):
        env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-m", "minisweagent", *args],
        cwd=tmp_path,
        env=env | (env_overrides or {}),
        input="",
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_setup_detects_settings_and_default_run_uses_them(tmp_path: Path, model_api):
    url, _, calls = model_api
    result = run_cli(tmp_path, "setup", "--base-url", f"{url}/v1/")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    path = tmp_path / "config/mini-swe-agent/mini.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["model"]["base_url"] == f"{url}/v1"
    assert config["model"]["model_name"] == "test:model"
    assert config["model"]["model_kwargs"] == {"temperature": 0.6, "max_tokens": 1024}
    assert "# top_p: 0.9" in path.read_text()
    assert "#   step_limit: 100" in path.read_text()
    assert "#   audit_timeout: 0" in path.read_text()
    assert result.stdout.splitlines() == [
        f"api: {url}/v1",
        'model: "test:model"',
        "context: 2048",
        "temperature: 0.6 (detected)",
        "max_tokens: 1024 (client reply limit)",
        "tested: chat completion and Bash action format",
        f"wrote: {path}",
    ]
    assert [call[0] for call in calls] == ["GET", "GET", "POST"]
    assert parse_qs(urlsplit(calls[1][1]).query) == {"model": ["test:model"]}
    assert calls[2][0:2] == ("POST", "/llama/v1/chat/completions")
    assert calls[2][2]["stream"] is False
    assert calls[2][2]["messages"][0]["role"] == "system"
    assert "exactly one fenced" in calls[2][2]["messages"][0]["content"]
    assert not (tmp_path / "data").exists()

    result = run_cli(tmp_path, "-t", "Finish immediately", "--exit-immediately")
    assert result.returncode == 0, result.stderr
    assert [call[0] for call in calls] == ["GET", "GET", "POST", "GET", "POST"]
    assert calls[3][1] == "/llama/v1/models"
    assert calls[-1][0:2] == ("POST", "/llama/v1/chat/completions")
    assert calls[-1][2]["model"] == "test:model"
    assert calls[-1][2]["temperature"] == 0.6
    assert calls[-1][2]["max_tokens"] == 1024
    assert {key: value for key, value in calls[2][2].items() if key != "messages"} == {
        key: value for key, value in calls[-1][2].items() if key != "messages"
    }
    trajectory = tmp_path / "data/mini-swe-agent/trajectories/latest.traj.json"
    assert json.loads(trajectory.read_text())["info"]["exit_status"] == "Submitted"
    assert json.loads(trajectory.read_text())["info"]["config"]["agent"]["audit_timeout"] == 0


def test_setup_generic_api_and_explicit_model(tmp_path: Path, model_api):
    url, routes, calls = model_api
    model = 'model: "quoted"\nvalue'
    routes["/llama/v1/models"] = (200, {"data": [{"id": "z-last"}, {"id": model}, {"id": "a-first"}]})
    routes["/llama/props"] = (404, {})
    result = run_cli(tmp_path, "setup", "--base-url", url)
    assert result.returncode == 0, result.stderr
    path = tmp_path / "config/mini-swe-agent/mini.yaml"
    assert yaml.safe_load(path.read_text())["model"]["model_name"] == "a-first"
    assert calls[-1][2]["model"] == "a-first"
    original = path.read_bytes()
    result = run_cli(tmp_path, "setup", "--base-url", url, "--model", "missing", "--force")
    assert result.returncode != 0
    assert "not found" in result.stderr
    assert path.read_bytes() == original
    result = run_cli(tmp_path, "setup", "--base-url", url, "--model", model, "--force")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    config = yaml.safe_load((tmp_path / "config/mini-swe-agent/mini.yaml").read_text())
    assert config["model"]["model_name"] == model
    assert config["model"]["model_kwargs"] == {"temperature": 0, "max_tokens": 4096}
    assert "context: not advertised" in result.stdout
    assert "temperature: 0 (client default)" in result.stdout
    assert calls[-1][2]["model"] == model


@pytest.mark.parametrize(
    ("names", "expected"),
    [(["zeta", "alpha", "middle"], "alpha"), (["Zeta", "beta", "Alpha"], "Alpha"), (["alpha", "Alpha"], "Alpha")],
)
def test_setup_selects_alphabetical_api_name(tmp_path: Path, model_api, names: list[str], expected: str):
    url, routes, calls = model_api
    routes["/llama/v1/models"] = (200, {"data": [{"id": name, "name": "other-display-name"} for name in names]})
    result = run_cli(tmp_path, "setup", "--base-url", url)
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load((tmp_path / "config/mini-swe-agent/mini.yaml").read_text())["model"]["model_name"] == expected
    assert parse_qs(urlsplit(calls[1][1]).query) == {"model": [expected]}
    assert calls[-1][2]["model"] == expected


@pytest.mark.parametrize(
    ("model_api", "expected_url"),
    [(("", 8080), "http://127.0.0.1:8080/v1")],
    indirect=["model_api"],
)
def test_setup_without_options_uses_local_server(tmp_path: Path, model_api, expected_url: str):
    _, routes, calls = model_api
    routes["/v1/models"] = (200, {"data": [{"id": "z-model"}, {"id": "a-model"}]})
    result = run_cli(tmp_path, "setup")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    config = yaml.safe_load((tmp_path / "config/mini-swe-agent/mini.yaml").read_text())
    assert config["model"]["base_url"] == expected_url
    assert config["model"]["model_name"] == "a-model"
    assert calls[0][1] == "/v1/models"
    assert calls[-1][0:2] == ("POST", "/v1/chat/completions")
    assert calls[-1][2]["model"] == "a-model"


def test_setup_preserves_existing_config_and_honors_config_override(tmp_path: Path, model_api):
    url, _, calls = model_api
    path = tmp_path / "custom.yaml"
    path.write_text("# keep my settings\nmodel:\n  model_name: previous\n")
    original = path.read_bytes()
    env = {"MSWEA_MINI_CONFIG_PATH": str(path)}
    result = run_cli(tmp_path, "setup", "--base-url", url, env_overrides=env)
    assert result.returncode != 0
    assert "--force" in result.stderr
    assert path.read_bytes() == original
    assert calls == []
    result = run_cli(tmp_path, "setup", "--base-url", url, "--force", env_overrides=env)
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(path.read_text())["model"]["model_name"] == "test:model"
    assert not (tmp_path / "config/mini-swe-agent/mini.yaml").exists()
    result = run_cli(tmp_path, "-t", "Finish", "--exit-immediately", env_overrides=env)
    assert result.returncode == 0, result.stderr
    assert calls[-1][2]["model"] == "test:model"


def test_setup_accepts_legacy_generation_settings(tmp_path: Path, model_api):
    url, routes, _ = model_api
    routes["/llama/props"] = (
        200,
        {
            "default_generation_settings": {
                "n_ctx": 8192,
                "temperature": 0.1,
                "max_tokens": -1,
                "n_predict": 256,
            }
        },
    )
    result = run_cli(tmp_path, "setup", "--base-url", url)
    assert result.returncode == 0, result.stderr
    config = yaml.safe_load((tmp_path / "config/mini-swe-agent/mini.yaml").read_text())
    assert config["model"]["model_kwargs"] == {"temperature": 0.1, "max_tokens": 256}
    assert "context: 8192" in result.stdout


@pytest.mark.parametrize(
    ("route", "status", "body", "error"),
    [
        ("models", 503, {}, "HTTP 503"),
        ("models", 200, [], "expected a JSON object"),
        ("models", 200, {"data": []}, "no valid models"),
        ("models", 200, {"data": [{"id": None}]}, "no valid models"),
        ("props", 500, {}, "HTTP 500"),
        ("props", 200, {"default_generation_settings": None}, "invalid server generation settings"),
        ("props", 200, {"default_generation_settings": {"n_ctx": 0}}, "invalid server context size"),
    ],
)
def test_failed_detection_does_not_replace_config(tmp_path: Path, model_api, route, status, body, error):
    url, routes, _ = model_api
    routes["/llama/v1/models" if route == "models" else "/llama/props"] = (status, body)
    path = tmp_path / "config/mini-swe-agent/mini.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("# existing config\n")
    result = run_cli(tmp_path, "setup", "--base-url", url, "--force")
    assert result.returncode != 0
    assert error in result.stderr
    assert result.stdout == ""
    assert path.read_text() == "# existing config\n"


@pytest.mark.parametrize(
    ("status", "body", "existing", "error"),
    [
        (400, {"error": "unsupported model settings"}, False, "HTTP 400"),
        (503, {}, True, "HTTP 503"),
        (200, b"not json", True, "JSONDecodeError"),
        (200, {"choices": []}, True, "choices"),
        (200, chat_response(""), False, "FormatError"),
        (200, chat_response("Just plain text"), True, "FormatError"),
        (200, chat_response("```bash\n\n```"), True, "empty Bash action"),
        (200, chat_response("```bash\necho one\n```\n```bash\necho two\n```"), True, "FormatError"),
        (200, chat_response("```bash\necho setup-ok\n```", "length"), False, "truncated"),
    ],
)
def test_failed_chat_test_does_not_write_config(tmp_path: Path, model_api, status, body, existing, error):
    url, routes, calls = model_api
    routes["/llama/v1/chat/completions"] = (status, body)
    path = tmp_path / "config/mini-swe-agent/mini.yaml"
    if existing:
        path.parent.mkdir(parents=True)
        path.write_text("# existing config\n")
    result = run_cli(tmp_path, "setup", "--base-url", url, "--force")
    assert result.returncode != 0
    assert error in result.stderr
    assert result.stdout == ""
    assert [call[0] for call in calls] == ["GET", "GET", "POST"]
    assert not (tmp_path / "data").exists()
    if existing:
        assert path.read_text() == "# existing config\n"
    else:
        assert not path.exists()


def test_setup_validates_but_does_not_execute_model_actions(tmp_path: Path, model_api):
    url, routes, _ = model_api
    marker = tmp_path / "must-not-exist"
    routes["/llama/v1/chat/completions"] = (200, chat_response(f"```bash\ntouch {marker}\n```"))
    result = run_cli(tmp_path, "setup", "--base-url", url)
    assert result.returncode == 0, result.stderr
    assert "tested: chat completion and Bash action format" in result.stdout
    assert (tmp_path / "config/mini-swe-agent/mini.yaml").is_file()
    assert not marker.exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(("timeout_args", "timeout"), [([], 3), (["--check-timeout", "0.2"], 0.2)])
def test_setup_chat_test_times_out_without_replacing_config(tmp_path: Path, model_api, timeout_args, timeout):
    url, routes, calls = model_api
    routes["/llama/v1/chat/completions"] = (0, {})
    path = tmp_path / "config/mini-swe-agent/mini.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("# existing config\n")
    started = time.monotonic()
    result = run_cli(tmp_path, "setup", "--base-url", url, "--force", *timeout_args)
    assert timeout - 0.1 <= time.monotonic() - started < timeout + 2
    assert result.returncode != 0
    assert f"read timeout={timeout}" in result.stderr
    assert result.stdout == ""
    assert [call[0] for call in calls] == ["GET", "GET", "POST"]
    assert path.read_text() == "# existing config\n"


@pytest.mark.parametrize(
    ("args", "env", "error"),
    [
        ([], {}, "config not found"),
        (["-t", "Do something"], {}, "config not found"),
        (["-c", "model.model_name=local"], {}, "no config file specified"),
        ([], {"MSWEA_MINI_CONFIG_PATH": "missing.yaml"}, "config not found"),
    ],
)
def test_launch_requires_a_config_file(tmp_path: Path, args: list[str], env: dict, error: str):
    result = run_cli(tmp_path, *args, env_overrides=env)
    assert result.returncode != 0
    assert error in result.stderr
    assert "mini setup" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "data").exists()


def write_model_config(tmp_path: Path, url: str) -> None:
    path = tmp_path / "config/mini-swe-agent/mini.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"base_url": url, "model_name": "test:model", "connect_timeout": 60, "request_timeout": 600},
            }
        )
    )


@pytest.mark.parametrize(
    ("status", "body", "error"),
    [
        (503, {}, "HTTP 503"),
        (401, {}, "HTTP 401"),
        (200, {"data": [{"id": "other-model"}]}, "not available"),
        (200, {"data": []}, "not available"),
        (200, {}, "missing a 'data' array"),
    ],
)
def test_launch_rejects_unavailable_model_before_prompting(tmp_path: Path, model_api, status, body, error):
    url, routes, calls = model_api
    routes["/llama/v1/models"] = (status, body)
    write_model_config(tmp_path, url)
    result = run_cli(tmp_path)
    assert result.returncode != 0
    assert error in result.stderr
    assert "What do you want to do?" not in result.stdout
    assert calls == [("GET", "/llama/v1/models", None)]
    assert not (tmp_path / "data").exists()


def test_launch_fails_quickly_when_server_is_unreachable(tmp_path: Path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        write_model_config(tmp_path, f"http://127.0.0.1:{listener.getsockname()[1]}")
        started = time.monotonic()
        result = run_cli(tmp_path)
    assert time.monotonic() - started < 3
    assert result.returncode != 0
    assert "Connection refused" in result.stderr
    assert "What do you want to do?" not in result.stdout
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("command", "timeout_args", "timeout"),
    [
        ("run", [], 3),
        ("setup", [], 3),
        ("run", ["--check-timeout", "0.2"], 0.2),
        ("setup", ["--check-timeout", "0.2"], 0.2),
        ("run", ["--check-timeout", "4"], 4),
        ("setup-props", ["--check-timeout", "0.2"], 0.2),
    ],
)
def test_model_probe_times_out_without_retries(tmp_path: Path, model_api, command: str, timeout_args, timeout):
    url, routes, calls = model_api
    routes["/llama/props" if command == "setup-props" else "/llama/v1/models"] = (0, {})
    if command == "run":
        write_model_config(tmp_path, url)
    started = time.monotonic()
    result = run_cli(tmp_path, *(["setup", "--base-url", url] if command != "run" else []), *timeout_args)
    assert timeout - 0.1 <= time.monotonic() - started < timeout + 2
    assert result.returncode != 0
    assert f"read timeout={timeout}" in result.stderr
    assert "What do you want to do?" not in result.stdout
    assert [(call[0], urlsplit(call[1]).path) for call in calls] == [("GET", "/llama/v1/models")] + (
        [("GET", "/llama/props")] if command == "setup-props" else []
    )
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("command", "value"),
    [(command, value) for command in ("run", "setup") for value in ("0", "-1", "nan", "inf", "-inf", "invalid")],
)
def test_invalid_check_timeout_fails_before_launch(tmp_path: Path, command: str, value: str):
    result = run_cli(tmp_path, *(["setup"] if command == "setup" else []), f"--check-timeout={value}")
    assert result.returncode == 2
    assert "--check-timeout" in result.stderr
    assert result.stdout == ""
    assert not (tmp_path / "config/mini-swe-agent/mini.yaml").exists()
    assert not (tmp_path / "data").exists()
