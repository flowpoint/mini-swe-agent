import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from minisweagent.environments.podman import PodmanEnvironment, SandboxError, _network_request, _socket_profile


def _python(source: str, tmp_path: Path, *args: str, terminal: bool = False) -> subprocess.CompletedProcess:
    master, slave = os.openpty()
    try:
        return subprocess.run(
            [sys.executable, "-P", "-c", source, *args],
            env=os.environ
            | {
                "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"),
                "XDG_DATA_HOME": str(tmp_path / "data"),
                "MINI_TEST_SECRET": "host-only",
                "HTTP_PROXY": "http://host-only.invalid:9999",
            },
            stdin=slave if terminal else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize(
    ("reply", "terminal", "expected"),
    [("yes", True, True), ("", True, False), ("y", True, False), ("no", True, False), ("yes", False, False)],
)
def test_network_requires_explicit_terminal_approval(tmp_path: Path, reply: str, terminal: bool, expected: bool):
    result = _python(
        r"""
import json, sys
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from minisweagent.environments.podman import _approve_network
with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
    pipe.send_text(sys.argv[1] + '\x07')
    print(json.dumps(_approve_network('print("test")\n# \x1b[2J', 'download\x1b[2J')))
""",
        tmp_path,
        reply,
        terminal=terminal,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) is expected
    assert "\x1b" not in result.stdout  # Untrusted script/reason cannot alter the approval display.
    if terminal:
        assert "Script: 'print" in result.stdout
        assert r"download\x1b[2J" in result.stdout


def test_interrupting_network_approval_never_grants_access(tmp_path: Path):
    result = _python(
        r"""
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from minisweagent.environments.podman import _approve_network
with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
    pipe.send_text('\x03')
    _approve_network('print("test")', 'download')
    print('executed')
""",
        tmp_path,
        terminal=True,
    )
    assert result.returncode != 0
    assert "KeyboardInterrupt" in result.stderr
    assert "executed" not in result.stdout


def test_network_declarations_and_socket_bypass_policy():
    assert _network_request("\n# network: download dependency\nprint('test')") == "download dependency"
    assert _network_request("# network:\nprint('test')") == ""
    assert _network_request("print('# network: not a declaration')") is None
    assert _network_request("print('first')\n# network: too late") is None
    original = {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64"],
        "syscalls": [
            {"names": ["read", "write", "socket", "socketcall"], "action": "SCMP_ACT_ALLOW"},
            {"names": ["io_uring_setup", "io_uring_enter", "io_uring_register"], "action": "SCMP_ACT_ALLOW"},
            {"names": ["socket"], "action": "SCMP_ACT_ALLOW", "includes": {"caps": ["CAP_AUDIT_WRITE"]}},
        ],
    }
    profile = _socket_profile(original)
    assert "socketcall" in original["syscalls"][0]["names"]
    assert profile["architectures"] == original["architectures"]
    assert profile["defaultAction"] == "SCMP_ACT_ERRNO"
    assert profile["syscalls"][0] == {"names": ["read", "write"], "action": "SCMP_ACT_ALLOW"}
    assert profile["syscalls"][1:] == [
        {
            "names": ["socket"],
            "action": "SCMP_ACT_ALLOW",
            "args": [{"index": 0, "value": family, "op": "SCMP_CMP_EQ"}],
        }
        for family in (socket.AF_INET, socket.AF_INET6)
    ]
    with pytest.raises(SandboxError, match="default-deny"):
        _socket_profile(original | {"defaultAction": "SCMP_ACT_ALLOW"})


@pytest.fixture
def podman_image():
    if not (image := os.getenv("MINI_TEST_PODMAN_IMAGE")):
        pytest.skip("Set MINI_TEST_PODMAN_IMAGE to a preloaded Python image to test real Podman containers")
    assert shutil.which("podman"), "MINI_TEST_PODMAN_IMAGE requires local Podman"
    return image


@pytest.fixture
def sandbox(tmp_path: Path, podman_image: str):
    (tmp_path / "workspace").mkdir()
    return PodmanEnvironment(image=podman_image, cwd=str(tmp_path / "workspace"), output_dir=tmp_path / "outputs")


@pytest.fixture
def network_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"network-ok")

        def log_message(self, *args):
            pass

    # Determine the local interface address without sending any packets.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 80))
        host = probe.getsockname()[0]
    with ThreadingHTTPServer(("0.0.0.0", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield host, server.server_port, requests
        server.shutdown()
        thread.join()


def test_offline_files_processes_and_socket_isolation(sandbox, network_server, tmp_path: Path):
    host, port, requests = network_server
    (tmp_path / "private").write_text("outside-workspace")
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(sandbox.workspace / "host.sock"))
        listener.listen()
        result = sandbox.execute(
            {
                "command": f"""
import os, socket
from pathlib import Path
assert Path.cwd() == Path('/workspace')
assert not Path({str(tmp_path / "private")!r}).exists()
for family, address in [(socket.AF_INET, ({host!r}, {port})), (socket.AF_INET6, ('::1', {port}))]:
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        with socket.socket(family, kind) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(address)
                sock.send(b'blocked')
                sock.recv(1)
            except OSError:
                pass
            else:
                raise AssertionError('network was reachable')
try:
    socket.socket(socket.AF_UNIX).connect('/workspace/host.sock')
except PermissionError:
    pass
else:
    raise AssertionError('host Unix socket reachable')
try:
    Path('/etc/mini-test').touch()
except PermissionError:
    pass
except OSError as error:
    assert error.errno == 30
else:
    raise AssertionError('image root writable')
Path('persisted').write_text('first')
Path('/tmp/mini-ephemeral').touch()
os.environ['MINI_ACTION_VARIABLE'] = 'first'
variable = 42
print('isolated')
"""
            }
        )
    assert result["returncode"] == 0, result
    assert result["output"] == "isolated\n"
    assert requests == []
    assert (
        sandbox.execute(
            {
                "command": "from pathlib import Path; import os; assert 'variable' not in globals(); "
                "assert 'MINI_ACTION_VARIABLE' not in os.environ; assert not Path('/tmp/mini-ephemeral').exists(); "
                "print(Path('persisted').read_text())"
            }
        )["output"]
        == "first\n"
    )
    assert (sandbox.workspace / "persisted").read_text() == "first"
    with pytest.raises(SandboxError, match="inside the sandbox"):
        sandbox.execute({"command": "print('escaped')"}, cwd=str(tmp_path))
    with pytest.raises(SandboxError, match="outside the writable workspace"):
        sandbox.check_output_path(sandbox.workspace / "trajectory.json")


def test_denials_never_execute_even_with_yolo_and_audit_bypasses(tmp_path: Path, podman_image: str):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = _python(
        r"""
import sys
from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.podman import PodmanEnvironment
from minisweagent.models.test_models import DeterministicModel
sandbox = PodmanEnvironment(image=sys.argv[1], cwd=sys.argv[2])
agent = InteractiveAgent(DeterministicModel(outputs=[]), sandbox,
    system_template='test', instance_template='{{task}}', mode='human',
    audit_timeout=0, whitelist_actions=['.*'])
for policy in ('none', 'ask'):
    sandbox.config.network = policy
    for declaration in ('# network: download', '# network:'):
        action = {'command': declaration + '\nfrom pathlib import Path; Path("denied").touch()'}
        result = agent.execute_action(action)
        assert result['returncode'] == -1
        assert result['extra']['network_access']['allowed'] is False
        assert action['network_access']['requested'] is True
print('denied')
""",
        tmp_path,
        podman_image,
        str(workspace),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "denied"
    assert not (workspace / "denied").exists()


def test_controller_and_logs_cannot_overlap_the_workspace(tmp_path: Path, podman_image: str):
    with pytest.raises(SandboxError, match="must not contain mini"):
        PodmanEnvironment(image=podman_image, cwd=sys.prefix)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "linked").symlink_to(workspace, target_is_directory=True)
    for output_dir in (workspace / "logs", tmp_path / "linked/logs"):
        with pytest.raises(SandboxError, match="outside the writable workspace"):
            PodmanEnvironment(image=podman_image, cwd=str(workspace), output_dir=output_dir)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from minisweagent.environments.podman import PodmanEnvironment; "
            f"PodmanEnvironment(image={podman_image!r})",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "Controller imports must not search the sandbox workspace" in result.stderr
    assert list(workspace.iterdir()) == []


def test_approval_is_per_script_and_does_not_forward_host_secrets(tmp_path: Path, podman_image: str, network_server):
    host, port, requests = network_server
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = _python(
        r"""
import json, sys
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from minisweagent.environments.podman import PodmanEnvironment
env = PodmanEnvironment(image=sys.argv[1], cwd=sys.argv[2], network='ask')
command = ('import os, urllib.request; '
           'assert "MINI_TEST_SECRET" not in os.environ and "HTTP_PROXY" not in os.environ; '
           'print(urllib.request.urlopen(' + repr(sys.argv[3]) + ', timeout=1).read().decode())')
assert 'MINI_TEST_SECRET' not in env.get_template_vars()
with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
    pipe.send_text('yes\x07')
    approved = env.execute({'command': '# network: local test server\n' + command})
    offline = env.execute({'command': command})
    pipe.send_text('\x07')
    denied = env.execute({'command': '# network: local test server\n' + command})
    print(json.dumps([approved, offline, denied]))
""",
        tmp_path,
        podman_image,
        str(workspace),
        f"http://{host}:{port}/once",
        terminal=True,
    )
    assert result.returncode == 0, result.stderr
    approved, offline, denied = json.loads(result.stdout.splitlines()[-1])
    assert approved["returncode"] == 0, approved
    assert approved["output"] == "network-ok\n"
    assert approved["extra"]["network_access"]["allowed"] is True
    assert offline["returncode"] != 0
    assert offline["extra"]["network_access"]["allowed"] is False
    assert denied["returncode"] == -1
    assert requests == ["/once"]


@pytest.mark.parametrize(
    ("kind", "body"),
    [("cpu", "sum(range(10000))"), ("io", "Path('/proc/self/stat').read_bytes(); time.sleep(0.05)")],
)
def test_container_child_liveness_and_cleanup(sandbox, kind: str, body: str):
    source = (
        "import time\nfrom pathlib import Path\nend = time.monotonic() + 1.5\n"
        f"while time.monotonic() < end:\n    {body}\n"
        f"print({kind!r})"
    )
    sandbox.config.timeout = 0.3
    result = sandbox.execute(
        {"command": f"import subprocess, sys; subprocess.run([sys.executable, '-c', {source!r}], check=True)"}
    )
    assert result["returncode"] == 0, result
    assert result["output"] == f"{kind}\n"
    stalled = sandbox.execute({"command": "import time; print('before stall'); time.sleep(60)"})
    assert stalled["returncode"] == -1
    assert stalled["extra"]["exception_type"] == "TimeoutExpired"
    assert Path(stalled["extra"]["output_path"]).read_text() == "before stall\n"
    assert (
        subprocess.run(
            ["podman", "--remote=false", "ps", "--all", "--filter", f"volume={sandbox.workspace}", "--format={{.ID}}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        == ""
    )
