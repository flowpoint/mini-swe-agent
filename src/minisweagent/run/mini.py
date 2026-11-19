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

from minisweagent import global_data_dir
from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec, get_user_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.environments.targets import TARGETS
from minisweagent.models import get_model
from minisweagent.models.llamacpp_model import DEFAULT_CHECK_TIMEOUT, LlamaCppModel
from minisweagent.run import nonnegative_float, positive_float, setup
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_CONFIG_FILE = builtin_config_dir / "mini.yaml"
DEFAULT_OUTPUT_FILE = global_data_dir / "trajectories" / "latest.traj.json"


_HELP_TEXT = """Run mini-SWE-agent in your local environment.

Commands run in the same environment as mini. For container isolation, run mini inside the container.

Create a config before running: mini setup
For a remote server: mini setup --base-url http://host:8080/v1

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

-c mini.yaml -c agent.audit_timeout=0
"""

console = Console(highlight=False)


def _multiline_prompt() -> str:
    return import_module("minisweagent.agents.prompt_user")._multiline_prompt()


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
    parser.add_argument("--language", choices=TARGETS, help="Action language (default: config, otherwise python)")
    parser.add_argument(
        "--check-timeout",
        type=positive_float,
        default=DEFAULT_CHECK_TIMEOUT,
        help="Startup model check timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--agent-class",
        help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')",
    )
    parser.add_argument("-t", "--task", help="Task/problem statement")
    parser.add_argument(
        "--audit-timeout",
        type=nonnegative_float,
        help="Audit seconds per command (default: 0; inf waits for continuation)",
    )
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
    task: str | None = None,
    audit_timeout: float | None = None,
    config_spec: list[str] | None = None,
    output: Path | None = DEFAULT_OUTPUT_FILE,
    exit_immediately: bool = False,
    check_timeout: float = DEFAULT_CHECK_TIMEOUT,
    language: str | None = None,
) -> Any:
    # Build the config from the command line arguments
    if config_spec is None:
        user_config = get_user_config_path()
        if not user_config.is_file():
            raise SystemExit(f"mini: config not found: {user_config}; run mini setup or use --config PATH")
        config_spec = [str(DEFAULT_CONFIG_FILE), str(user_config)]
    if not any("=" not in spec for spec in config_spec):
        raise SystemExit("mini: no config file specified; run mini setup or use --config PATH")
    console.print(f"Building agent config from specs: [bold green]{config_spec}[/bold green]")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append(
        {
            "run": {
                "task": task or UNSET,
            },
            "agent": {
                "agent_class": agent_class or UNSET,
                "audit_timeout": audit_timeout if audit_timeout is not None else UNSET,
                "output_path": output or UNSET,
            },
            "model": {
                "model_class": model_class or UNSET,
                "model_name": model_name or UNSET,
            },
            "environment": {"language": language if language is not None else UNSET},
        }
    )
    config = recursive_merge(*configs)
    if unknown := config.keys() - {"agent", "environment", "model", "run"}:
        raise ValueError(f"Unknown config sections: {', '.join(sorted(unknown))}")
    for section, settings in config.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Config section {section!r} must be a mapping")
    if unknown := config.get("run", {}).keys() - {"task"}:
        raise ValueError(f"Unknown run settings: {', '.join(sorted(unknown))}")
    if "task" in config.get("run", {}) and not isinstance(config["run"]["task"], str):
        raise ValueError("run.task must be text")

    environment_config = config.get("environment", {})
    if environment_config.pop("environment_class", "local") != "local":
        raise SystemExit("mini: only local execution is supported; run mini itself inside your container")
    env = LocalEnvironment(**environment_config)
    model = get_model(config=config.get("model", {}))
    agent = get_agent(model, env, config.get("agent", {}), default_type="interactive")
    if exit_immediately and hasattr(agent.config, "confirm_exit"):
        agent.config.confirm_exit = False
    if isinstance(model, LlamaCppModel):
        model.check_connection(timeout=check_timeout)
    if (run_task := config.get("run", {}).get("task", UNSET)) is UNSET:
        console.print("[bold yellow]What do you want to do?")
        run_task = _multiline_prompt()
        console.print("[bold green]Got that, thanks![/bold green]")

    if output == DEFAULT_OUTPUT_FILE:
        agent.config.output_path = _new_output_file()
    if env.config.output_dir is None and agent.config.output_path:
        env.config.output_dir = agent.config.output_path.with_suffix(".outputs")
    if output == DEFAULT_OUTPUT_FILE:
        output_path = agent.config.output_path
        agent.save(output_path)
        link = output_path.with_suffix(".link")
        link.symlink_to(output_path.name)
        link.replace(DEFAULT_OUTPUT_FILE)
    agent.run(run_task)
    if output_path := agent.config.output_path:
        console.print(f"Saved trajectory to [bold green]'{output_path}'[/bold green]")
    return agent


def app(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Parse command-line arguments and run the agent."""
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["setup"]:
        return setup.app(args[1:], prog=f"{prog or 'mini'} setup")
    try:
        agent = main(**vars(_build_parser(prog).parse_args(args)))
    except KeyboardInterrupt:
        return 130
    status = agent.save(None)["info"]["exit_status"]
    return {
        "Submitted": 0,
        "LimitsExceeded": 3,
        "TimeExceeded": 3,
        "ContextLimitExceeded": 3,
        "RepeatedFormatError": 4,
        "Interrupted": 130,
    }.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(app())
