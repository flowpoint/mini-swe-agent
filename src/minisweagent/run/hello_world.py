"""This is the simplest possible example of how to use mini-SWE-agent with python bindings.
For a more complete example, see mini.py
"""

import argparse
import logging
import os
from pathlib import Path

from minisweagent import package_dir
from minisweagent._vendor import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.llamacpp_model import LlamaCppModel


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("-t", "--task", help="Task/problem statement")
    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        help="llama.cpp model alias (defaults to MSWEA_MODEL_NAME or local)",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="llama.cpp OpenAI API root")
    return parser


def main(
    task: str | None = None,
    model_name: str | None = None,
    base_url: str = "http://127.0.0.1:8080/v1",
) -> DefaultAgent:
    if task is None:
        task = input("Task/problem statement: ")
    if model_name is None:
        model_name = input("What model do you want to use? ") or os.getenv("MSWEA_MODEL_NAME", "local")
    logging.basicConfig(level=logging.DEBUG)
    agent = DefaultAgent(
        LlamaCppModel(
            model_name=model_name,
            base_url=base_url,
            model_kwargs={"temperature": 0, "max_tokens": 4096},
        ),
        LocalEnvironment(),
        **yaml.safe_load(Path(package_dir / "config" / "default.yaml").read_text())["agent"],
    )
    agent.run(task)
    return agent


def app(argv: list[str] | None = None, *, prog: str | None = None) -> int:
    """Parse command-line arguments and run the example agent."""
    main(**vars(_build_parser(prog).parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
