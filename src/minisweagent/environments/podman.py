"""Optional local Podman sandbox; network grants belong to one script only."""

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from contextlib import suppress
from dataclasses import replace
from functools import partial
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO, Literal

from pydantic import Field

from minisweagent import global_config_dir, global_data_dir, package_dir
from minisweagent.environments.local import LocalEnvironment, LocalEnvironmentConfig, _activity, _read_output, _run
from minisweagent.environments.targets import Target
from minisweagent.utils.serialize import recursive_merge


class SandboxError(RuntimeError):
    """Sandbox setup or enforcement failed; never fall back to local execution."""


class PodmanEnvironmentConfig(LocalEnvironmentConfig):
    image: str = Field(min_length=1)
    network: Literal["none", "ask"] = "none"


def _socket_profile(profile: dict) -> dict:
    if profile.get("defaultAction") != "SCMP_ACT_ERRNO":
        raise SandboxError("Podman must provide a default-deny seccomp profile")
    # Pathname Unix sockets bypass network namespaces; io_uring/socketcall can bypass socket filters.
    blocked = {"socket", "socketcall", "io_uring_setup", "io_uring_enter", "io_uring_register"}
    return profile | {
        "syscalls": [
            rule | {"names": names}
            for rule in profile["syscalls"]
            if (names := [name for name in rule["names"] if name not in blocked])
        ]
        + [
            {
                "names": ["socket"],
                "action": "SCMP_ACT_ALLOW",
                "args": [{"index": 0, "value": family, "op": "SCMP_CMP_EQ"}],
            }
            for family in (2, 10)
        ]
    }


def _network_request(command: str) -> str | None:
    first = command.lstrip("\r\n").partition("\n")[0]
    return first.removeprefix("# network:").strip() if first.startswith("# network:") else None


def _approve_network(command: str, reason: str) -> bool:
    if not reason or sys.stdin is None or not sys.stdin.isatty():
        return False
    print(f"Network request: {reason!r}\nScript: {command!r}", flush=True)
    return (
        import_module("minisweagent.agents.prompt_user")
        ._multiline_prompt(
            "Allow network for this script and its children once? Type yes to allow; anything else denies.\n> "
        )
        .strip()
        == "yes"
    )


def _container_activity(pidfile: Path) -> dict:
    with suppress(FileNotFoundError, ProcessLookupError):
        if pid := pidfile.read_text().strip():
            return _activity(0, namespace=(Path("/proc") / pid / "ns/pid").stat().st_ino)
    return {}


class PodmanEnvironment(LocalEnvironment):
    def __init__(self, **kwargs):
        super().__init__(config_class=PodmanEnvironmentConfig, **kwargs)
        if not (executable := shutil.which("podman")):
            raise SandboxError("Podman is required for --sandbox/--ask-network; local execution is disabled")
        self._podman = [executable, "--remote=false", "--default-mounts-file=/dev/null"]
        self.workspace = Path(self.config.cwd or Path.cwd()).resolve(strict=True)
        if not self.workspace.is_dir() or ":" in str(self.workspace):
            raise SandboxError("Sandbox cwd must be a directory whose path contains no colon")
        info = json.loads(self._control("info", "--format=json").stdout)
        security = info["host"]["security"]
        if not security.get("seccompEnabled"):
            raise SandboxError("Podman seccomp support is required for sandbox isolation")
        protected = [
            package_dir,
            Path(sys.prefix),
            global_config_dir,
            global_data_dir,
            Path("/etc"),
            Path("/usr"),
            Path("/bin"),
            Path("/sbin"),
            Path("/lib"),
            Path("/lib64"),
            Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "containers",
            Path(info["store"]["graphRoot"]),
            Path(info["store"]["runRoot"]),
            Path(executable),
            Path(security["seccompProfilePath"]),
            *[
                Path(os.environ[key])
                for key in ("CONTAINERS_CONF", "CONTAINERS_CONF_OVERRIDE", "CONTAINERS_STORAGE_CONF")
                if os.environ.get(key)
            ],
        ]
        if Path.home().is_relative_to(self.workspace) or any(
            resolved.is_relative_to(self.workspace) or self.workspace.is_relative_to(resolved)
            for path in protected
            for resolved in (path.absolute(), path.resolve())
        ):
            raise SandboxError(
                "Sandbox workspace must not contain mini, its Python installation, home, config, or container storage; install mini outside the workspace"
            )
        if any(Path(path or Path.cwd()).resolve().is_relative_to(self.workspace) for path in sys.path):
            raise SandboxError(
                "Controller imports must not search the sandbox workspace; use the installed mini command or python -P -m minisweagent"
            )
        self.check_output_path(self.config.output_dir)
        self._profile = _socket_profile(json.loads(Path(security["seccompProfilePath"]).read_text()))
        self.image_id = self._control("image", "inspect", "--format={{.Id}}", "--", self.config.image).stdout.strip()
        self.config.cwd = str(self.workspace)
        self._temporary_root = (global_data_dir / "sandboxes").resolve()
        self._temporary_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with TemporaryDirectory(dir=self._temporary_root) as directory:
            with (Path(directory) / "probe.log").open("w+b") as log:
                code = self._run_container(self.target.example, str(self.workspace), 10, log, network=False)
                log.seek(0)
                output = log.read().decode("utf-8", errors="replace")
                if code or "setup-ok\n" not in output:
                    raise SandboxError(f"Podman sandbox interpreter check failed ({code}): {output}")

    def _control(self, *args: str) -> subprocess.CompletedProcess:
        result = subprocess.run(
            [*self._podman, *args], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10
        )
        if result.returncode:
            raise SandboxError(f"Podman {' '.join(args[:2])} failed: {result.stderr.strip()}")
        return result

    def check_output_path(self, path: Path | None) -> None:
        if path is not None and any(
            candidate.is_relative_to(self.workspace) for candidate in (path.absolute(), path.resolve())
        ):
            raise SandboxError("Sandbox trajectories and command logs must be outside the writable workspace")

    @property
    def interpreter(self) -> list[str]:
        if self.config.interpreter is not None:
            return self.config.interpreter
        return ["python3", "-u", "-c"] if self.config.language == "python" else list(super().target.interpreter)

    @property
    def target(self) -> Target:
        instructions = "Actions run in a fresh Podman container at /workspace. Only workspace files persist. "
        instructions += (
            "Networking is blocked. To request network access, put '# network: <reason>' on the first line of the script; the user must explicitly approve every such script. Unmarked scripts stay offline. Do not retry a partially executed script blindly."
            if self.config.network == "ask"
            else "Networking is disabled, including network requests declared by a script."
        )
        return replace(super().target, instructions=super().target.instructions + " " + instructions)

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(), platform.uname()._asdict(), {"target": self.target, "cwd": "/workspace"}, kwargs
        )

    def execute(self, action: dict, cwd: str = "", *, timeout: float | None = None) -> dict:
        self.check_output_path(self.config.output_dir or global_data_dir / "outputs")
        command = action.get("command", "")
        reason = _network_request(command)
        access = action["network_access"] = {
            "policy": self.config.network,
            "requested": reason is not None,
            "reason": reason,
            "allowed": False,
        }
        if reason is not None:
            access["allowed"] = self.config.network == "ask" and _approve_network(command, reason)
            if not access["allowed"]:
                return {
                    "returncode": -1,
                    "output": "Network permission denied; script was not executed.",
                    "exception_info": "",
                    "extra": {"network_access": access},
                }
        return self._execute(
            command,
            cwd,
            timeout,
            partial(self._run_container, network=access["allowed"]),
            {"network_access": access, "sandbox_image": self.image_id},
        )

    def _run_container(self, command: str, cwd: str, timeout: float | None, log: BinaryIO, *, network: bool) -> int:
        directory = Path(cwd).resolve(strict=True)
        if not directory.is_relative_to(self.workspace):
            raise SandboxError("Action cwd must be inside the sandbox workspace")
        name = f"mini-{uuid.uuid4().hex}"
        with TemporaryDirectory(dir=self._temporary_root) as temporary:
            temporary = Path(temporary)
            (temporary / "hooks").mkdir()
            (temporary / "seccomp.json").write_text(json.dumps(self._profile))
            # Host defaults must not inject mounts, devices or credentials into the sandbox.
            (temporary / "containers.conf").write_text(
                "[containers]\nenv=[]\nenv_host=false\nvolumes=[]\nmounts=[]\ndevices=[]\nannotations=[]\n"
            )
            pidfile = temporary / "pid"
            argv = [
                *self._podman,
                f"--hooks-dir={temporary / 'hooks'}",
                "run",
                "--rm",
                "--pull=never",
                f"--name={name}",
                f"--pidfile={pidfile}",
                f"--network={'bridge' if network else 'none'}",
                "--pid=private",
                "--ipc=private",
                "--uts=private",
                "--privileged=false",
                "--cap-drop=all",
                "--security-opt=no-new-privileges",
                f"--security-opt=seccomp={temporary / 'seccomp.json'}",
                "--read-only",
                "--image-volume=ignore",
                "--http-proxy=false",
                "--sdnotify=ignore",
                "--log-driver=none",
                "--cgroups=disabled",
                "--userns=keep-id",
                f"--user={os.getuid()}:{os.getgid()}",
                f"--volume={self.workspace}:/workspace:rw,Z,nodev,nosuid,rprivate",
                f"--workdir={Path('/workspace') / directory.relative_to(self.workspace)}",
                "--env=HOME=/tmp",
                *[f"--env={key}={value}" for key, value in self.config.env.items()],
                f"--entrypoint={self.interpreter[0]}",
                "--",
                self.image_id,
                *self.interpreter[1:],
                command,
            ]
            try:
                code = _run(
                    argv,
                    str(self.workspace),
                    os.environ | {"CONTAINERS_CONF_OVERRIDE": str(temporary / "containers.conf")},
                    timeout,
                    log,
                    activity=partial(_container_activity, pidfile),
                    stdin=subprocess.DEVNULL,
                )
                if code == 125:
                    raise SandboxError(
                        f"Podman sandbox failed: {_read_output(Path(log.name), self.config.max_output_bytes)}"
                    )
                return code
            finally:
                self._control("rm", "--force", "--ignore", name)
