import logging
import time
from typing import Any
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, Field

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_text import format_observation_messages, parse_regex_actions

logger = logging.getLogger("llamacpp_model")

_RETRY_STATUS_CODES = {429, 502, 503, 504}
_RETRY_REQUEST_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
_MAX_ATTEMPTS = 3


class LlamaCppError(RuntimeError):
    """Base error raised by the local llama.cpp adapter."""


class LlamaCppHTTPError(LlamaCppError):
    """The llama.cpp server returned an unsuccessful HTTP response."""


class LlamaCppProtocolError(LlamaCppError):
    """The llama.cpp server returned a malformed successful response."""


class LlamaCppModelConfig(BaseModel):
    model_name: str = "local"
    base_url: str = Field(default="http://127.0.0.1:8080/v1", min_length=1)
    request_timeout: float = Field(default=600, gt=0, allow_inf_nan=False)
    connect_timeout: float | None = Field(default=10, gt=0, allow_inf_nan=False)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in a fenced bash block, found {{actions|length}} actions."
    )
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )


class LlamaCppModel:
    def __init__(self, *, session: requests.Session | None = None, **kwargs):
        self.config = LlamaCppModelConfig(**kwargs)
        self.session = session or requests.Session()
        # Never inherit proxy settings or credentials from the environment. This
        # adapter communicates only with the explicitly configured server.
        self.session.trust_env = False
        api_root = self.config.base_url.rstrip("/")
        parsed_url = urlsplit(api_root)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed_url.username is not None or parsed_url.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed_url.query or parsed_url.fragment:
            raise ValueError("base_url must not contain a query string or fragment")
        if not api_root.endswith("/v1"):
            api_root += "/v1"
        self._api_url = f"{api_root}/chat/completions"

    @property
    def _timeout(self) -> tuple[float, float]:
        return (self.config.connect_timeout or self.config.request_timeout, self.config.request_timeout)

    @staticmethod
    def _response_detail(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        if isinstance(body, dict) and "error" in body:
            body = body["error"]
        if isinstance(body, dict):
            body = ", ".join(f"{key}={body[key]}" for key in ("message", "type", "code") if key in body)
        detail = body if isinstance(body, str) else repr(body)
        return detail.strip().replace("\n", " ")[:1000] or response.reason or "no response body"

    def _http_error(self, response: requests.Response) -> LlamaCppHTTPError:
        return LlamaCppHTTPError(
            f"llama.cpp server returned HTTP {response.status_code}: {self._response_detail(response)}"
        )

    def _request(self, payload: dict) -> requests.Response:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self.session.post(
                    self._api_url,
                    json=payload,
                    timeout=self._timeout,
                    # Never forward prompts to a redirect target.
                    allow_redirects=False,
                )
            except _RETRY_REQUEST_EXCEPTIONS as e:
                if attempt == _MAX_ATTEMPTS:
                    raise LlamaCppError(
                        f"llama.cpp request failed after {_MAX_ATTEMPTS} attempts: {type(e).__name__}: {e}"
                    ) from e
                logger.warning(
                    "Transient llama.cpp connection failure (%s), retrying attempt %d/%d",
                    type(e).__name__,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                time.sleep(2 ** (attempt - 1))
                continue
            except requests.exceptions.RequestException as e:
                raise LlamaCppError(f"llama.cpp request failed without retry: {type(e).__name__}: {e}") from e

            if response.status_code in _RETRY_STATUS_CODES:
                if attempt == _MAX_ATTEMPTS:
                    raise self._http_error(response)
                logger.warning(
                    "Transient llama.cpp HTTP %d, retrying attempt %d/%d",
                    response.status_code,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                time.sleep(2 ** (attempt - 1))
                continue
            if not 200 <= response.status_code < 300:
                raise self._http_error(response)
            return response
        raise AssertionError("unreachable")

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict[str, str]]:
        prepared = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise LlamaCppError(f"Message {index} must be an object with a role and text content")
            role, content = message.get("role"), message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise LlamaCppError(f"Message {index} must have a supported role and text content")
            prepared.append({"role": role, "content": content})
        return prepared

    @staticmethod
    def _assistant_content(response: dict) -> tuple[str, str | None]:
        if not isinstance(response, dict):
            raise LlamaCppProtocolError("llama.cpp response must be a JSON object")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlamaCppProtocolError("llama.cpp response is missing a non-empty 'choices' array")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise LlamaCppProtocolError("llama.cpp response choices[0] is missing a 'message' object")
        message = choice["message"]
        if message.get("role", "assistant") != "assistant":
            raise LlamaCppProtocolError("llama.cpp response choices[0].message has a non-assistant role")
        if "content" not in message or not isinstance(message["content"], str):
            raise LlamaCppProtocolError("llama.cpp response choices[0].message is missing text 'content'")
        finish_reason = choice.get("finish_reason")
        return message["content"], finish_reason if isinstance(finish_reason, str) else None

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        payload = (
            self.config.model_kwargs
            | kwargs
            | {
                "model": self.config.model_name,
                "messages": self._prepare_messages_for_api(messages),
                "stream": False,
            }
        )
        response = self._request(payload)
        try:
            response_json = response.json()
        except ValueError as e:
            raise LlamaCppProtocolError(
                f"llama.cpp returned invalid JSON with HTTP {response.status_code}: {self._response_detail(response)}"
            ) from e
        content, finish_reason = self._assistant_content(response_json)
        extra = {
            "response": response_json,
            "usage": response_json.get("usage", {}),
            "timings": response_json.get("timings", {}),
            "cost": 0.0,
            "timestamp": time.time(),
        }
        GLOBAL_MODEL_STATS.add(0.0)
        try:
            actions = parse_regex_actions(
                content,
                action_regex=self.config.action_regex,
                format_error_template=self.config.format_error_template,
                template_kwargs={"finish_reason": finish_reason},
            )
        except FormatError as e:
            e.messages[0]["extra"].update(extra)
            raise
        return {"role": "assistant", "content": content, "extra": {"actions": actions, **extra}}

    def format_message(self, **kwargs) -> dict:
        if not isinstance(kwargs.get("content"), str):
            raise LlamaCppError("Messages must contain text content")
        return kwargs

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }

    def close(self) -> None:
        self.session.close()
