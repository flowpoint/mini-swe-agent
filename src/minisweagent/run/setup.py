import argparse
import json
import math

import requests
import yaml
from jinja2 import StrictUndefined, Template

from minisweagent.config import builtin_config_dir, get_config_from_spec, get_user_config_path
from minisweagent.models import get_model
from minisweagent.models.llamacpp_model import DEFAULT_CHECK_TIMEOUT, LlamaCppModelConfig, normalize_base_url
from minisweagent.models.utils.actions_text import parse_regex_actions
from minisweagent.run.utilities import positive_float
from minisweagent.utils.serialize import recursive_merge


def _get_json(session: requests.Session, url: str, *, timeout: float, optional: bool = False, **kwargs) -> dict:
    response = session.get(url, timeout=timeout, allow_redirects=False, **kwargs)
    if optional and response.status_code in {404, 405, 501}:
        return {}
    if response.status_code != 200:
        raise SystemExit(f"mini setup: GET {url}: HTTP {response.status_code}")
    data = response.json()
    if not isinstance(data, dict):
        raise SystemExit(f"mini setup: expected a JSON object from {url}")
    return data


def _test_config(content: str, *, timeout: float) -> None:
    config = recursive_merge(get_config_from_spec(builtin_config_dir / "mini.yaml"), yaml.safe_load(content))
    model = get_model(config=config["model"])
    with model.session as session:
        response = session.post(
            model._api_url,
            json=model.config.model_kwargs
            | {
                "model": model.config.model_name,
                "messages": [
                    {"role": "system", "content": config["agent"]["system_template"]},
                    {"role": "user", "content": "Reply only with one fenced bash block containing: echo setup-ok"},
                ],
                "stream": False,
            },
            timeout=timeout,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise SystemExit(f"mini setup: chat test failed: HTTP {response.status_code}")
        text, finish_reason = model._assistant_content(response.json())
    if finish_reason == "length":
        raise SystemExit("mini setup: chat test response was truncated")
    actions = parse_regex_actions(
        text,
        action_regex=model.config.action_regex,
        format_error_template=model.config.format_error_template,
        template_kwargs={"finish_reason": finish_reason},
    )
    if not actions[0]["command"]:
        raise SystemExit("mini setup: chat test returned an empty Bash action")


def main(
    base_url: str, model: str | None = None, *, force: bool = False, check_timeout: float = DEFAULT_CHECK_TIMEOUT
) -> None:
    path = get_user_config_path()
    if path.exists() and not force:
        raise SystemExit(f"mini setup: {path} already exists; use --force to replace it")
    base_url = normalize_base_url(base_url)
    with requests.Session() as session:
        session.trust_env = False
        models = _get_json(session, f"{base_url}/models", timeout=check_timeout).get("data")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] for item in models)
        ):
            raise SystemExit("mini setup: API returned no valid models")
        available = sorted((item["id"] for item in models), key=lambda name: (name.casefold(), name))
        if model is None:
            model = available[0]
        if model not in available:
            raise SystemExit(f"mini setup: model {json.dumps(model)} not found; available: {json.dumps(available)}")
        props = _get_json(
            session,
            f"{base_url.removesuffix('/v1')}/props",
            timeout=check_timeout,
            optional=True,
            params={"model": model},
        )
    settings = props.get("default_generation_settings", {})
    if not isinstance(settings, dict):
        raise SystemExit("mini setup: invalid server generation settings")
    params = settings.get("params", settings)
    if not isinstance(params, dict):
        raise SystemExit("mini setup: invalid server generation parameters")
    context = settings.get("n_ctx")
    if context is not None and (type(context) is not int or context < 2):
        raise SystemExit("mini setup: invalid server context size")
    temperature = params.get("temperature", 0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
        raise SystemExit("mini setup: invalid server temperature")
    limits = [params[key] for key in ("max_tokens", "n_predict") if key in params]
    if any(type(value) is not int for value in limits):
        raise SystemExit("mini setup: invalid server output token limit")
    max_tokens = next((value for value in limits if value > 0), 4096)
    if context:
        max_tokens = min(max_tokens, context // 2)
    content = Template(
        (builtin_config_dir / "setup.yaml.jinja").read_text(encoding="utf-8"), undefined=StrictUndefined
    ).render(
        base_url=base_url, model=model, context=context, params=params, temperature=temperature, max_tokens=max_tokens
    )
    _test_config(content, timeout=check_timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    print(f"api: {base_url}")
    print(f"model: {json.dumps(model)}")
    print(f"context: {context if context else 'not advertised'}")
    print(f"temperature: {temperature} ({'detected' if 'temperature' in params else 'client default'})")
    print(f"max_tokens: {max_tokens} (client reply limit)")
    print("tested: chat completion and Bash action format")
    print(f"wrote: {path}")


def app(argv: list[str] | None = None, *, prog: str = "mini setup") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Detect and test model settings before writing the user config."
    )
    parser.add_argument(
        "--base-url",
        default=LlamaCppModelConfig().base_url,
        help="Server URL, with or without /v1 (default: %(default)s)",
    )
    parser.add_argument("-m", "--model", help="Model name (default: alphabetically first, ignoring case)")
    parser.add_argument(
        "--check-timeout",
        type=positive_float,
        default=DEFAULT_CHECK_TIMEOUT,
        help="Timeout in seconds for each setup check (default: %(default)s)",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing config without prompting")
    main(**vars(parser.parse_args(argv)))
    return 0
