import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml


@pytest.fixture
def model_api():
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
    }
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(("GET", self.path, None))
            status, body = routes.get(urlsplit(self.path).path, (404, {}))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            calls.append(("POST", self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
                                }
                            }
                        ],
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}/llama", routes, calls
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
    assert result.stdout.splitlines() == [
        f"api: {url}/v1",
        'model: "test:model"',
        "context: 2048",
        "temperature: 0.6 (detected)",
        "max_tokens: 1024 (client reply limit)",
        f"wrote: {path}",
    ]
    assert [call[0] for call in calls] == ["GET", "GET"]
    assert parse_qs(urlsplit(calls[1][1]).query) == {"model": ["test:model"]}

    result = run_cli(tmp_path, "-t", "Finish immediately", "--yolo", "--exit-immediately")
    assert result.returncode == 0, result.stderr
    assert calls[-1][0:2] == ("POST", "/llama/v1/chat/completions")
    assert calls[-1][2]["model"] == "test:model"
    assert calls[-1][2]["temperature"] == 0.6
    assert calls[-1][2]["max_tokens"] == 1024
    trajectory = tmp_path / "data/mini-swe-agent/trajectories/latest.traj.json"
    assert json.loads(trajectory.read_text())["info"]["exit_status"] == "Submitted"


def test_setup_generic_api_and_explicit_model(tmp_path: Path, model_api):
    url, routes, _ = model_api
    model = 'model: "quoted"\nvalue'
    routes["/llama/v1/models"] = (200, {"data": [{"id": "one"}, {"id": model}]})
    routes["/llama/props"] = (404, {})
    result = run_cli(tmp_path, "setup", "--base-url", url)
    assert result.returncode != 0
    assert "select --model" in result.stderr
    assert not (tmp_path / "config/mini-swe-agent/mini.yaml").exists()
    result = run_cli(tmp_path, "setup", "--base-url", url, "--model", model)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    config = yaml.safe_load((tmp_path / "config/mini-swe-agent/mini.yaml").read_text())
    assert config["model"]["model_name"] == model
    assert config["model"]["model_kwargs"] == {"temperature": 0, "max_tokens": 4096}
    assert "context: not advertised" in result.stdout
    assert "temperature: 0 (client default)" in result.stdout


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
    result = run_cli(tmp_path, "-t", "Finish", "--yolo", "--exit-immediately", env_overrides=env)
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
