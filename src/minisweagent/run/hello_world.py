"""This is the simplest possible example of how to use mini-SWE-agent with python bindings.
For a more complete example, see mini.py
"""

import logging
import os
from pathlib import Path

import typer
import yaml

from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.llamacpp_model import LlamaCppModel

app = typer.Typer()


@app.command()
def main(
    task: str = typer.Option(..., "-t", "--task", help="Task/problem statement", show_default=False, prompt=True),
    model_name: str = typer.Option(
        os.getenv("MSWEA_MODEL_NAME", "local"),
        "-m",
        "--model",
        help="llama.cpp model alias (defaults to MSWEA_MODEL_NAME or local)",
        prompt="What model do you want to use?",
    ),
    base_url: str = typer.Option("http://127.0.0.1:8080/v1", "--base-url", help="llama.cpp OpenAI API root"),
) -> DefaultAgent:
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


if __name__ == "__main__":
    app()
