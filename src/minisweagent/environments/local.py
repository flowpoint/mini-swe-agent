import os
import platform
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO

from pydantic import BaseModel, Field

from minisweagent import global_data_dir
from minisweagent.environments.targets import TARGETS, Target, TargetName
from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel, extra="forbid"):
    language: TargetName = "python"
    interpreter: list[str] | None = Field(default=None, min_length=1)
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: float | None = Field(default=300, gt=0, allow_inf_nan=False)
    """Inactivity seconds (output or process-group CPU/I/O); null disables the check."""
    max_output_bytes: int = Field(default=16384, ge=256)
    output_dir: Path | None = None


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """Execute each action in a fresh local interpreter."""
        self.config = config_class(**kwargs)

    @property
    def target(self) -> Target:
        return TARGETS[self.config.language]

    @property
    def interpreter(self) -> list[str]:
        return self.config.interpreter if self.config.interpreter is not None else list(self.target.interpreter)

    def execute(self, action: dict, cwd: str = "", *, timeout: float | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        directory = (self.config.output_dir or global_data_dir / "outputs").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=directory, prefix="command-", suffix=".log", delete=False) as log:
            output = {"returncode": -1, "exception_info": "", "extra": {"output_path": log.name}}
            try:
                output["returncode"] = _run(
                    [*self.interpreter, command],
                    cwd,
                    os.environ | self.config.env,
                    self.config.timeout if timeout is None else timeout,
                    log,
                )
            except (OSError, subprocess.TimeoutExpired) as e:
                output["exception_info"] = (
                    f"Command inactive: {e}" if isinstance(e, subprocess.TimeoutExpired) else f"Command failed: {e}"
                )
                output["extra"].update(exception_type=type(e).__name__, exception=str(e))
        output["output"] = _read_output(Path(log.name), self.config.max_output_bytes)
        self._check_finished(output)
        return output

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission, **output.get("extra", {})},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(), platform.uname()._asdict(), os.environ, {"target": self.target}, kwargs
        )

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json") | {"interpreter": self.interpreter},
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


def _read_output(path: Path, limit: int) -> str:
    with path.open("rb") as f:
        head = f.read(limit + 1)
        if len(head) <= limit:
            return head.decode("utf-8", errors="replace")
        f.seek(-(limit // 2), os.SEEK_END)
        return (
            head[: limit // 2].decode("utf-8", errors="replace")
            + f"\n[output truncated; full log: {path}]\n"
            + f.read(limit // 2).decode("utf-8", errors="replace")
        )


def _activity(group: int) -> dict:
    """Linux process-group CPU ticks and I/O counters, including quiet children."""
    activity = {}
    for path in Path("/proc").glob("[0-9]*"):
        with suppress(FileNotFoundError, ProcessLookupError, PermissionError):
            stat = (path / "stat").read_text(errors="replace").rsplit(")", 1)[1].split()
            if int(stat[2]) == group:
                activity[path.name] = tuple(stat[11:15])
                with suppress(PermissionError):
                    activity[path.name] += ((path / "io").read_text(),)
    return activity


def _run(argv: list[str], cwd: str, env: dict[str, str], timeout: float | None, log: BinaryIO) -> int:
    """Run until exit or inactivity; kill the process group on interruption/failure."""
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    previous, last_activity = None, time.monotonic()
    try:
        while process.poll() is None:
            if timeout is not None:
                current = (os.fstat(log.fileno()).st_size, _activity(process.pid))
                if current != previous:
                    previous, last_activity = current, time.monotonic()
                elif time.monotonic() - last_activity >= timeout:
                    raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(min(0.1, timeout / 4) if timeout else 0.1)
    finally:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return process.returncode
