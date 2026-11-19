"""Language details shared by execution, prompts, and fenced-action parsing."""

import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

TargetName = Literal["python", "bash"]


@dataclass(frozen=True)
class Target:
    language: str
    interpreter: tuple[str, ...]
    example: str
    submit: str
    instructions: str = ""
    check: Callable[[str], object] | None = None

    @property
    def action_regex(self) -> str:
        return rf"(?m)^```{re.escape(self.language)}[ \t]*\r?\n(.*?)\r?\n```[ \t]*\r?$"


TARGETS = {
    "python": Target(
        "python",
        (sys.executable, "-u", "-c"),
        'print("setup-ok")',
        'print("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")',
        "Use print() to show results; expressions are not displayed automatically. "
        "Use pathlib for files and subprocess.run([...], check=True) for external tools and tests.",
        partial(compile, filename="<action>", mode="exec"),
    ),
    "bash": Target("bash", ("/bin/bash", "-c"), "echo setup-ok", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
}
