import json
import ssl
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from minisweagent._vendor import yaml
from tests.run.test_setup import chat_response, run_cli


@pytest.fixture
def self_signed_api():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.reply()

        def reply(self):
            calls.append((self.command, self.path.split("?")[0]))
            if self.command == "POST":
                body = chat_response("```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```")
            else:
                body = {"data": [{"id": "local"}]} if self.path == "/v1/models" else {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(Path(__file__).parents[1] / "test_data/self_signed.pem")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"https://127.0.0.1:{server.server_port}/v1", calls
        finally:
            server.shutdown()
            thread.join()


def test_tls_verification_is_opt_out_for_setup_and_all_run_requests(tmp_path: Path, self_signed_api):
    url, calls = self_signed_api
    config = tmp_path / "config/mini-swe-agent/mini.yaml"
    result = run_cli(tmp_path, "setup", "--base-url", url)
    assert result.returncode != 0 and "CERTIFICATE_VERIFY_FAILED" in result.stderr
    assert not config.exists() and calls == []

    result = run_cli(tmp_path, "setup", "--base-url", url, "--no-verify-ssl")
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(config.read_text())["model"]["verify_ssl"] is False
    assert "verify_ssl: false" in result.stdout
    assert calls == [("GET", "/v1/models"), ("GET", "/props"), ("POST", "/v1/chat/completions")]

    result = run_cli(tmp_path, "-t", "finish", "--exit-immediately")
    assert result.returncode == 0, result.stderr
    assert calls[-2:] == [("GET", "/v1/models"), ("POST", "/v1/chat/completions")]
    saved = json.loads((tmp_path / "data/mini-swe-agent/trajectories/latest.traj.json").read_text())
    assert saved["info"]["exit_status"] == "Submitted"
    assert saved["info"]["config"]["model"]["verify_ssl"] is False

    result = run_cli(tmp_path, "-c", "mini.yaml", "-c", str(config), "-c", "model.verify_ssl=true", "-t", "test")
    assert result.returncode != 0 and "CERTIFICATE_VERIFY_FAILED" in result.stderr
    assert len(calls) == 5
