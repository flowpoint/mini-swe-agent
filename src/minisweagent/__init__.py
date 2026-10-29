"""
This file provides:

- Path settings for global config file & relative directories
- Version numbering
- Protocols for the core components of mini-swe-agent.
  By the magic of protocols & duck typing, you can pretty much ignore them,
  unless you want the static type checking.
"""

__version__ = "2.4.6"

import os
from pathlib import Path
from typing import Any, Protocol

from minisweagent.utils.log import logger

package_dir = Path(__file__).resolve().parent


_config_home = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
global_config_dir = Path(os.getenv("MSWEA_GLOBAL_CONFIG_DIR") or _config_home / "mini-swe-agent")
global_config_dir.mkdir(parents=True, exist_ok=True)

_data_home = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
global_data_dir = (_data_home if _data_home.is_absolute() else Path.home() / ".local/share") / "mini-swe-agent"

# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Agent(Protocol):
    """Protocol for agents."""

    config: Any

    def run(self, task: str, **kwargs) -> dict: ...

    def save(self, path: Path | None, *extra_dicts) -> dict: ...


__all__ = [
    "Agent",
    "Model",
    "Environment",
    "package_dir",
    "__version__",
    "global_config_dir",
    "logger",
]
