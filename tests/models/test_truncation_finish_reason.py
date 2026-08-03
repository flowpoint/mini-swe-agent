"""Verify finish reasons are available to fenced-text format-error templates."""

import pytest

from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_text import parse_regex_actions

_TEMPLATE = "{% if finish_reason == 'length' %}cut off{% else %}{{ error }}{% endif %}"


def test_finish_reason_reported_on_zero_actions():
    with pytest.raises(FormatError) as exc:
        parse_regex_actions(
            "no action here",
            action_regex=r"```bash\n(.*?)\n```",
            format_error_template=_TEMPLATE,
            template_kwargs={"finish_reason": "length"},
        )
    assert exc.value.messages[0]["content"] == "cut off"


def test_without_template_kwargs_still_works():
    with pytest.raises(FormatError) as exc:
        parse_regex_actions(
            "nope",
            action_regex=r"```bash\n(.*?)\n```",
            format_error_template="{{ error }}",
        )
    assert "found 0" in exc.value.messages[0]["content"]
