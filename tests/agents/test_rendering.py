import json
import re
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from minisweagent.agents.interactive import InteractiveAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicModel


@pytest.fixture
def agent():
    return InteractiveAgent(
        DeterministicModel(outputs=[]), LocalEnvironment(), system_template="Test", instance_template="{{task}}"
    )


@pytest.mark.parametrize(
    ("language", "code"),
    [
        ("bash", 'printf "%s\\n" "$HOME"'),
        ("python", 'print("héllo 世界")'),
        ("diff", "-old line\n+new line"),
        ("json", '{"enabled": true, "count": 42}'),
        ("unknown-language", "[red]literal[/red]"),
    ],
)
def test_markdown_code_rendering_preserves_content(agent, language, code):
    message = {"role": "assistant", "content": f"**Review**\n\n```{language}\n{code}\n```\n\n[red]literal[/red]"}
    output = StringIO()
    Console(file=output, force_terminal=True, color_system="standard", no_color=False, width=100).print(
        agent.render_message(message)
    )
    rendered = output.getvalue()
    styled = Text.from_ansi(rendered)
    visible = styled.plain
    assert "\x1b[" in rendered
    if language != "unknown-language":
        assert any(span.style.color is not None for span in styled.spans)
    assert "**Review**" not in visible and "Review" in visible
    assert "```" not in visible
    assert "[red]literal[/red]" in visible
    for line in code.splitlines():
        assert line in visible


def test_no_color_and_literal_tool_output(agent):
    output = StringIO()
    console = Console(file=output, force_terminal=True, no_color=True, width=100)
    console.print(agent.render_message({"role": "assistant", "content": "```bash\necho 'hello'\n```"}))
    console.print(agent.render_message({"role": "user", "content": "**literal** [red]text[/red]\n```python"}))
    assert "**literal** [red]text[/red]\n```python" in Text.from_ansi(output.getvalue()).plain
    assert not any(
        30 <= int(code) <= 49 or 90 <= int(code) <= 107
        for sequence in re.findall(r"\x1b\[([\d;]+)m", output.getvalue())
        for code in sequence.split(";")
    )


def test_piped_output_and_trajectory_preserve_original_markdown(agent, capsys, tmp_path):
    message = {"role": "assistant", "content": "**Review**\n```bash\nprintf '%s' '[red]literal[/red]'\n```"}
    agent.add_messages(message)
    output = capsys.readouterr().out
    assert message["content"] in output
    assert "\x1b[" not in output
    path = tmp_path / "trajectory.json"
    agent.save(path)
    assert json.loads(path.read_text())["messages"] == [message]
