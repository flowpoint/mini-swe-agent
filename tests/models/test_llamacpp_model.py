import json
import time
from unittest.mock import patch

import pytest
import requests

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.llamacpp_model import LlamaCppHTTPError, LlamaCppModel, LlamaCppProtocolError

_MISSING = object()


def _response(payload=_MISSING, *, status_code: int = 200, body: str = "") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = "http://127.0.0.1:8080/v1/chat/completions"
    response.reason = "test response"
    response.encoding = "utf-8"
    response.headers["Content-Type"] = "application/json"
    response._content = (json.dumps(payload) if payload is not _MISSING else body).encode()
    return response


def _success(
    content: str = """Reasoning
```bash
git status
```""",
) -> dict:
    return {
        "id": "chatcmpl-local",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        "timings": {"predicted_ms": 42.5},
    }


def _model(**overrides) -> LlamaCppModel:
    config = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model_name": "local",
        "request_timeout": 600,
        "connect_timeout": 10,
        "model_kwargs": {"temperature": 0, "max_tokens": 4096, "stream": True},
        "action_regex": r"```bash\s*\n(.*?)\n```",
    }
    config.update(overrides)
    return LlamaCppModel(**config)


def test_successful_query_sends_chat_completion_and_preserves_metadata(reset_global_stats):
    model = _model()
    response_json = _success()
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Inspect the repository."},
    ]
    started = time.time()

    with patch.object(model.session, "post", return_value=_response(response_json)) as mock_post:
        result = model.query(messages)

    assert isinstance(model.session, requests.Session)
    assert model._api_url == "http://127.0.0.1:8080/v1/chat/completions"
    call = mock_post.call_args
    assert (call.args[0] if call.args else call.kwargs["url"]) == model._api_url
    assert call.kwargs["timeout"] == (10, 600)
    assert call.kwargs["allow_redirects"] is False
    payload = call.kwargs["json"]
    assert payload["model"] == "local"
    assert payload["messages"] == messages
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 4096
    assert payload["stream"] is False

    assert result["role"] == "assistant"
    assert result["content"] == response_json["choices"][0]["message"]["content"]
    assert result["extra"]["actions"] == [{"command": "git status"}]
    assert result["extra"]["response"] == response_json
    assert result["extra"]["usage"] == response_json["usage"]
    assert result["extra"]["timings"] == response_json["timings"]
    assert result["extra"]["cost"] == 0.0
    assert started <= result["extra"]["timestamp"] <= time.time()
    assert GLOBAL_MODEL_STATS.cost == 0.0
    assert GLOBAL_MODEL_STATS.n_calls == 1


def test_missing_choices_is_a_nonretryable_protocol_error():
    model = _model()
    with patch.object(model.session, "post", return_value=_response({"usage": {}})) as mock_post:
        with pytest.raises(LlamaCppProtocolError) as exc_info:
            model.query([{"role": "user", "content": "hello"}])

    assert "choices" in str(exc_info.value).lower()
    assert mock_post.call_count == 1


def test_missing_assistant_content_is_a_nonretryable_protocol_error():
    model = _model()
    response_json = {"choices": [{"message": {"role": "assistant"}}]}
    with patch.object(model.session, "post", return_value=_response(response_json)) as mock_post:
        with pytest.raises(LlamaCppProtocolError) as exc_info:
            model.query([{"role": "user", "content": "hello"}])

    assert "content" in str(exc_info.value).lower()
    assert mock_post.call_count == 1


def test_invalid_json_is_a_nonretryable_protocol_error():
    model = _model()
    with patch.object(model.session, "post", return_value=_response(body="not json")) as mock_post:
        with pytest.raises(LlamaCppProtocolError) as exc_info:
            model.query([{"role": "user", "content": "hello"}])

    assert "json" in str(exc_info.value).lower()
    assert mock_post.call_count == 1


def test_http_400_is_not_retried_and_includes_server_detail():
    model = _model()
    response_json = {"error": {"message": "model local-missing was not found", "type": "invalid_request_error"}}
    with patch.object(model.session, "post", return_value=_response(response_json, status_code=400)) as mock_post:
        with pytest.raises(LlamaCppHTTPError) as exc_info:
            model.query([{"role": "user", "content": "hello"}])

    assert "400" in str(exc_info.value)
    assert "local-missing" in str(exc_info.value)
    assert mock_post.call_count == 1


def test_http_503_is_retried():
    model = _model()
    with (
        patch.object(
            model.session,
            "post",
            side_effect=[
                _response({"error": {"message": "server overloaded"}}, status_code=503),
                _response(_success()),
            ],
        ) as mock_post,
        patch("minisweagent.models.llamacpp_model.time.sleep", return_value=None) as mock_sleep,
    ):
        assert model.query([{"role": "user", "content": "hello"}])["extra"]["actions"] == [{"command": "git status"}]

    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_read_timeout_is_retried():
    model = _model()
    with (
        patch.object(
            model.session,
            "post",
            side_effect=[requests.exceptions.ReadTimeout("read timed out"), _response(_success())],
        ) as mock_post,
        patch("minisweagent.models.llamacpp_model.time.sleep", return_value=None) as mock_sleep,
    ):
        assert model.query([{"role": "user", "content": "hello"}])["content"].startswith("Reasoning")

    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_session_is_reused_across_queries():
    model = _model()
    session = model.session
    assert session.trust_env is False
    with patch.object(session, "post", side_effect=[_response(_success()), _response(_success())]) as mock_post:
        model.query([{"role": "user", "content": "first"}])
        model.query([{"role": "user", "content": "second"}])

    assert model.session is session
    assert mock_post.call_count == 2


def test_base_url_rejects_embedded_credentials():
    with pytest.raises(ValueError, match="must not contain credentials"):
        _model(base_url="http://secret@127.0.0.1:8080/v1")


@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:8080/v1?route=other", "http://127.0.0.1:8080/v1#other"],
)
def test_base_url_rejects_query_and_fragment(base_url):
    with pytest.raises(ValueError, match="query string or fragment"):
        _model(base_url=base_url)


def test_redirect_is_not_followed_or_retried():
    model = _model()
    response = _response(status_code=307, body="redirect")
    response.headers["Location"] = "https://example.com/v1/chat/completions"
    with patch.object(model.session, "post", return_value=response) as mock_post:
        with pytest.raises(LlamaCppHTTPError, match="HTTP 307"):
            model.query([{"role": "user", "content": "do not forward this"}])

    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["allow_redirects"] is False


def test_multiple_fenced_actions_raise_format_error_with_response_metadata():
    model = _model()
    response_json = _success("""```bash
pwd
```
```bash
git status
```""")
    with patch.object(model.session, "post", return_value=_response(response_json)) as mock_post:
        with pytest.raises(FormatError) as exc_info:
            model.query([{"role": "user", "content": "hello"}])

    extra = exc_info.value.messages[0]["extra"]
    assert extra["n_actions"] == 2
    assert extra["response"] == response_json
    assert extra["usage"] == response_json["usage"]
    assert extra["timings"] == response_json["timings"]
    assert extra["cost"] == 0.0
    assert mock_post.call_count == 1
