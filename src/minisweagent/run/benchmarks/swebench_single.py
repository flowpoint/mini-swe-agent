"""Run on a single SWE-Bench instance."""

import argparse
import sys
from pathlib import Path

from minisweagent import global_config_dir
from minisweagent.agents import get_agent
from minisweagent.config import builtin_config_dir, get_config_from_spec
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import (
    DATASET_MAPPING,
    SWEBENCH_DEPENDENCY_ERROR,
    get_sb_environment,
    load_dataset,
)
from minisweagent.utils.log import logger
from minisweagent.utils.serialize import UNSET, recursive_merge

DEFAULT_OUTPUT_FILE = global_config_dir / "last_swebench_single_run.traj.json"
DEFAULT_CONFIG_FILE = builtin_config_dir / "benchmarks" / "swebench.yaml"

_CONFIG_SPEC_HELP_TEXT = """Path to config files, filenames, or key-value pairs.

IMPORTANT: If you set this option, the default config file will not be used.
So you need to explicitly set it, for example with -c swebench.yaml <other options>.

Multiple configs will be recursively merged.

Examples:

-c model.model_kwargs.temperature=0  You forgot to add the default config file! See above.

-c swebench.yaml -c model.model_kwargs.temperature=0.5

-c swebench.yaml -c agent.mode=yolo
"""


def main(
    subset: str = "lite",
    split: str = "dev",
    instance_spec: str = "0",
    model_name: str | None = None,
    model_class: str | None = None,
    agent_class: str | None = None,
    yolo: bool = False,
    cost_limit: float | None = None,
    config_spec: list[str] | None = None,
    exit_immediately: bool = False,
    output: Path | None = DEFAULT_OUTPUT_FILE,
) -> None:
    """Run on a single SWE-Bench instance."""
    if config_spec is None:
        config_spec = [str(DEFAULT_CONFIG_FILE)]
    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset from {dataset_path}, split {split}...")
    try:
        instances = {
            inst["instance_id"]: inst  # type: ignore
            for inst in load_dataset(dataset_path, split=split)
        }
    except ModuleNotFoundError as e:
        if str(e) != SWEBENCH_DEPENDENCY_ERROR:
            raise
        print(SWEBENCH_DEPENDENCY_ERROR, file=sys.stderr)
        raise SystemExit(1) from None
    if instance_spec.isnumeric():
        instance_spec = sorted(instances.keys())[int(instance_spec)]
    instance: dict = instances[instance_spec]  # type: ignore

    logger.info(f"Building agent config from specs: {config_spec}")
    configs = [get_config_from_spec(spec) for spec in config_spec]
    configs.append({
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
    })
    config = recursive_merge(*configs)

    env = get_sb_environment(config, instance)
    agent = get_agent(
        get_model(config=config.get("model", {})),
        env,
        config.get("agent", {}),
        default_type="interactive",
    )
    agent.run(instance["problem_statement"])


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--subset", default="lite", help="SWE-Bench subset to use or path to a dataset")
    parser.add_argument("--split", default="dev", help="Dataset split")
    parser.add_argument("-i", "--instance", dest="instance_spec", default="0", help="SWE-Bench instance ID or index")
    parser.add_argument("-m", "--model", dest="model_name", help="Model to use")
    parser.add_argument("--model-class", help="Model class to use (e.g., 'llamacpp')")
    parser.add_argument(
        "--agent-class",
        help="Agent class to use (e.g., 'interactive' or 'minisweagent.agents.interactive.InteractiveAgent')",
    )
    parser.add_argument("-y", "--yolo", action="store_true", help="Run without confirmation")
    parser.add_argument("-l", "--cost-limit", type=float, help="Cost limit. Set to 0 to disable.")
    parser.add_argument("-c", "--config", dest="config_spec", action="append", help=_CONFIG_SPEC_HELP_TEXT)
    parser.add_argument(
        "--exit-immediately",
        action="store_true",
        help="Exit immediately when the agent wants to finish instead of prompting.",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Output trajectory file")
    return parser


def app(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Parse command-line arguments and run the single-instance benchmark."""
    main(**vars(_build_parser(prog).parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
