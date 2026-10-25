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
    from minisweagent.agents.utils.prompt_user import _multiline_prompt

    pipe.send_text('first' + sys.argv[2] + 'second\x1b\rtyping\x1bb\x1b[200~\npasted\r\ntext\x07\x1b[201~' + sys.argv[1])
    message = _multiline_prompt()
    pipe.send_text('draft\x12first\x07\x07')
    draft = _multiline_prompt()
    pipe.send_text('ab\x1b[D!\x1b[C' + sys.argv[1])
    edited = _multiline_prompt()
    pipe.send_text(sys.argv[1])
    confirmation = _multiline_prompt()
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


def test_initial_task_interruptions_and_corrections_share_input(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            r"""
import json
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
    from minisweagent.agents.interactive import InteractiveAgent
    from minisweagent.environments.local import LocalEnvironment
    from minisweagent.exceptions import Submitted, UserInterruption
    from minisweagent.models.test_models import DeterministicModel, make_output
    from minisweagent.run.mini import _multiline_prompt

    pipe.send_text('Initial\rrequest\x07')
    task = _multiline_prompt()
    agent = InteractiveAgent(
        DeterministicModel(outputs=[
            make_output('Interrupt', [{'command': 'kill -INT "$PPID"'}]),
            make_output('Finish', [{'command': 'echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'}]),
        ]),
        LocalEnvironment(), system_template='Test', instance_template='{{task}}', confirm_exit=False,
    )
    pipe.send_text('Fix\rthis\x1b\rplease\x07')
    assert agent.run(task)['exit_status'] == 'Submitted'
    interruption = next(message['content'] for message in agent.messages
                        if message.get('extra', {}).get('interrupt_type') == 'UserInterruption')

    pipe.send_text('/audit inf\rDo not run this command\x07')
    with pytest.raises(UserInterruption) as rejected:
        agent.confirm_action({'command': 'touch forbidden'})
    assert agent.config.audit_timeout == 0
    pipe.send_text('\x07')
    agent.confirm_action({'command': 'touch allowed'})

    agent.config.confirm_exit = True
    submission = Submitted({'role': 'exit', 'content': 'done'})
    pipe.send_text('Follow up\rwith tests\x07')
    with pytest.raises(UserInterruption) as followup:
        agent._check_for_new_task_or_submit(submission)
    pipe.send_text('\x07')
    with pytest.raises(Submitted):
        agent._check_for_new_task_or_submit(submission)
    print('\n' + json.dumps([task, interruption, rejected.value.messages[0]['content'], followup.value.messages[0]['content']]))
""",
        ],
        cwd=tmp_path,
        env={**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config")},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    messages = json.loads(result.stdout.splitlines()[-1])
    assert messages[0] == "Initial\nrequest"
    assert messages[1] == "Interrupted by user: Fix\nthis\nplease"
    assert messages[2].endswith("/audit inf\nDo not run this command")
    assert messages[3] == "The user added a new task: Follow up\nwith tests"
    assert not (tmp_path / "forbidden").exists()
