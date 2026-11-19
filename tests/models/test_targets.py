import pytest

from minisweagent.environments.local import LocalEnvironment
from minisweagent.environments.targets import TARGETS
from minisweagent.exceptions import FormatError
from minisweagent.models.llamacpp_model import LlamaCppModel


def test_target_parser_preserves_python_source_and_indentation(tmp_path):
    source = '\nvalue = """first\n    second\n"""\nfor line in value.splitlines():\n    print(line)\n'
    model = LlamaCppModel()
    with model.session:
        actions = model.parse_actions(f"Explanation\n```python\n{source}\n```\n")
        assert actions == [{"command": source}]
        assert LocalEnvironment(output_dir=tmp_path).execute(actions[0])["output"] == "first\n    second\n"
        assert model.parse_actions("```bash\necho compatible\n```", target=TARGETS["bash"]) == [
            {"command": "echo compatible"}
        ]
        assert model.parse_actions("```python\r\nif True:\r\n    print('ok')\r\n```\r\n") == [
            {"command": "if True:\r\n    print('ok')"}
        ]


@pytest.mark.parametrize(
    ("content", "count"),
    [
        ("```bash\necho wrong\n```", 0),
        ("```python\n \t\n```", 1),
        ("```python\nprint('truncated')", 0),
        ("```python\nprint(1)\n```\n```python\nprint(2)\n```", 2),
    ],
)
def test_invalid_actions_report_the_selected_language(content, count):
    model = LlamaCppModel()
    with model.session, pytest.raises(FormatError) as error:
        model.parse_actions(content)
    assert error.value.messages[0]["extra"]["n_actions"] == count
    assert "fenced python block" in error.value.messages[0]["content"]
    assert error.value.messages[0]["extra"]["model_response"] == content
