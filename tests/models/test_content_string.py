from minisweagent.models.content_string import get_content_string


def test_plain_string_content():
    assert get_content_string({"content": "hello world"}) == "hello world"


def test_empty_or_missing_content():
    assert get_content_string({}) == ""
    assert get_content_string({"content": ""}) == ""


def test_structured_content_is_not_supported():
    assert get_content_string({"content": [{"type": "text", "text": "hello"}]}) == ""
    assert get_content_string({"content": None}) == ""
