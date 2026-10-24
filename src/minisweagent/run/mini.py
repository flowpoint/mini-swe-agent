#!/usr/bin/env python3

"""Run mini-SWE-agent in your local environment. This is the default executable `mini`."""
# Read this first: https://mini-swe-agent.com/latest/usage/mini/  (usage)

import argparse
import os
import sys
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from tempfile import mkstemp
from typing import Any

from rich.console import Console

from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec, get_user_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run import setup
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "mini.yaml"
_data_home = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local/share")
if not _data_home.is_absolute():
    _data_home = Path.home() / ".local/share"
DEFAULT_OUTPUT_FILE = _data_home / "mini-swe-agent" / "trajectories" / "latest.traj.json"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

Configure a server: mini setup --base-url http://host:8080/v1

More information about the usage:
https://mini-swe-agent.com/latest/usage/mini/
"""

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

IMPORTANT: If you set this option, the default config file will not be used.
So you need to explicitly set it, for example with -c mini.yaml <other options>.

Multiple configs will be recursively merged.

Examples:

-c model.model_kwargs.temperature=0  You forgot to add the default config file! See above.

-c mini.yaml -c model.model_kwargs.temperature=0.5

-c swebench.yaml agent.mode=yolo
"""

console = Console(highlight=False)


def _multiline_prompt() -> str:
    return import_module("minisweagent.agents.utils.prompt_user")._multiline_prompt()


def _new_output_file() -> Path:
    DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = mkstemp(
        prefix=f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-",
        suffix=".traj.json",
        dir=DEFAULT_OUTPUT_FILE.parent,
    )
    os.close(fd)
    return Path(filename)


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=_HELP_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--model", dest="model_name", help="Model to use")
    parser.add_argument("--model-class", help="Model class to use (supported: 'llamacpp')")
    parser.add_argument(
        "--agent-class",
        help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')",
    )
    parser.add_argument("--environment-class", help="Environment class to use: 'local' or 'docker'")
    parser.add_argument("-t", "--task", help="Task/problem statement")
    parser.add_argument("-y", "--yolo", action="store_true", help="Run without confirmation")
    parser.add_argument("-l", "--cost-limit", type=float, help="Cost limit. Set to 0 to disable.")
    parser.add_argument("-c", "--config", dest="config_spec", action="append", help=_CONFIG_SPEC_HELP_TEXT)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output trajectory file (default: a new file in {DEFAULT_OUTPUT_FILE.parent}, linked as latest.traj.json)",
    )
    parser.add_argument(
        "--exit-immediately",
        action="store_true",
        help="Exit immediately when the agent wants to finish instead of prompting.",
    )
    return parser


def main(
    model_name: str | None = None,
    model_class: str | None = None,
    agent_class: str | None = None,
    environment_class: str | None = None,
    task: str | None = None,
    yolo: bool = False,
    cost_limit: float | None = None,
    config_spec: list[str] | None = None,
    output: Path | None = DEFAULT_OUTPUT_FILE,
    exit_immediately: bool = False,
) -> Any:
    # Build the config from the command line arguments
    if config_spec is None:
        config_spec = [str(DEFAULT_CONFIG_FILE)]
        user_config = get_user_config_path()
        if user_config.is_file() or os.getenv("MSWEA_MINI_CONFIG_PATH"):
            config_spec.append(str(user_config))
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
        "run": {
            "task": task or UNSET,
        },
        "agent": {
            "agent_class": agent_class or UNSET,
            "mode": "yolo" if yolo else UNSET,
            "cost_limit": cost_limit if cost_limit is not None else UNSET,
            "confirm_exit": False if exit_immediately else UNSET,
            "output_path": output or UNSET,
        },
        "model": {
            "model_class": model_class or UNSET,
            "model_name": model_name or UNSET,
        },
        "environment": {
            "environment_class": environment_class or UNSET,
        },
    })
    config = recursive_merge(*configs)

    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    model = get_model(config=config.get("model", {}))
    env = get_environment(config.get("environment", {}), default_type="local")
    if output == DEFAULT_OUTPUT_FILE:
        config["agent"]["output_path"] = _new_output_file()
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    if output == DEFAULT_OUTPUT_FILE:
        output_path = config["agent"]["output_path"]
        agent.save(output_path)
        link = output_path.with_suffix(".link")
        link.symlink_to(output_path.name)
        link.replace(DEFAULT_OUTPUT_FILE)
    agent.run(run_task)
    if (output_path := config.get("agent", {}).get("output_path")):
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


def app(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Parse command-line arguments and run the agent."""
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["setup"]:
        return setup.app(args[1:], prog=f"{prog or 'mini'} setup")
    main(**vars(_build_parser(prog).parse_args(args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
