import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("submit_key", "newline"),
    [
        ("\x07", "\r"),
        ("\x07", "\n"),
        ("\x1b[13;2u", "\r"),
        ("\x1b[13;5u", "\r"),
        ("\x1b[27;2;13~", "\r"),
        ("\x1b[27;5;13~", "\r"),
    ],
)
def test_prompt_submission_preserves_newlines_paste_and_history(tmp_path: Path, submit_key: str, newline: str):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import json
import sys
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
    from minisweagent.agents.utils.prompt_user import _multiline_prompt, prompt_session

    pipe.send_text('first' + sys.argv[2] + 'second\x1b\rtyping\x1bb\x1b[200~\npasted\r\ntext\x07\x1b[201~' + sys.argv[1])
    message = _multiline_prompt()
    pipe.send_text('draft\x12first\x07\x07')
    draft = _multiline_prompt()
    pipe.send_text('ab\x1b[D!\x1b[C' + sys.argv[1])
    edited = _multiline_prompt()
    pipe.send_text('\r')
    confirmation = prompt_session.prompt()
    print(json.dumps([message, draft, edited, confirmation]))
""",
            submit_key,
            newline,
        ],
        env={**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["first\nsecond\ntypingb\npasted\ntext\x07", "draft", "a!b", ""]
