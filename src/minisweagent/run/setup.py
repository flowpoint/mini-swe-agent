import argparse
import json
import math
import os
import subprocess

import requests
from jinja2 import StrictUndefined, Template

from minisweagent._vendor import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_from_spec, get_user_config_path
from minisweagent.environments.local import LocalEnvironment
from minisweagent.environments.targets import TARGETS
from minisweagent.models import get_model
from minisweagent.models.llamacpp_model import (
    DEFAULT_CHECK_TIMEOUT,
    LlamaCppModelConfig,
    configure_session,
    normalize_base_url,
)
from minisweagent.run import positive_float
from minisweagent.utils.serialize import atomic_write, recursive_merge


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
    env = LocalEnvironment(**config["environment"])
    probe = subprocess.run(
        [*env.interpreter, env.target.example],
        cwd=env.config.cwd or None,
        env=os.environ | env.config.env,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )
    if probe.stdout.strip() != "setup-ok":
        raise SystemExit("mini setup: interpreter test failed")
    model = get_model(config=config["model"])
    agent = DefaultAgent(model, env, **config["agent"])
    with model.session as session:
        response = session.post(
            model._api_url,
            json=model.config.model_kwargs
            | {
                "model": model.config.model_name,
                "messages": [
                    {"role": "system", "content": agent._render_template(agent.config.system_template)},
                    {
                        "role": "user",
                        "content": f"Reply only with one fenced {env.target.language} block containing: {env.target.example}",
                    },
                ],
                "stream": True,
            },
            timeout=(timeout, model.config.request_timeout),
            stream=True,
            allow_redirects=False,
        )
        with response:
            if response.status_code != 200:
                raise SystemExit(f"mini setup: chat test failed: HTTP {response.status_code}")
            text, finish_reason = model._assistant_content(model._read_response(response))
    if finish_reason == "length":
        raise SystemExit("mini setup: chat test response was truncated")
    actions = model.parse_actions(text, target=env.target, finish_reason=finish_reason)
    if env.target.check:
        env.target.check(actions[0]["command"])


def main(
    base_url: str,
    model: str | None = None,
    *,
    force: bool = False,
    check_timeout: float = DEFAULT_CHECK_TIMEOUT,
    request_timeout: float = 600,
    api_key_env: str | None = None,
    verify_ssl: bool = True,
    language: str = "python",
) -> None:
    path = get_user_config_path()
    if path.exists() and not force:
        raise SystemExit(f"mini setup: {path} already exists; use --force to replace it")
    base_url = normalize_base_url(base_url)
    with configure_session(requests.Session(), api_key_env, verify_ssl=verify_ssl) as session:
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
        base_url=base_url,
        model=model,
        context=context,
        params=params,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        api_key_env=api_key_env,
        verify_ssl=verify_ssl,
        language=language,
    )
    _test_config(content, timeout=check_timeout)
    atomic_write(path, content + "\n", overwrite=force)
    print(f"api: {base_url}")
    print(f"model: {json.dumps(model)}")
    if api_key_env:
        print(f"auth: ${api_key_env}")
    if not verify_ssl:
        print("verify_ssl: false")
    print(f"context: {context if context else 'not advertised'}")
    print(f"temperature: {temperature} ({'detected' if 'temperature' in params else 'client default'})")
    print(f"max_tokens: {max_tokens} (client reply limit)")
    print(f"tested: {language} interpreter, chat completion and action format")
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
    parser.add_argument("--language", choices=TARGETS, default="python", help="Action language (default: %(default)s)")
    parser.add_argument(
        "--check-timeout",
        type=positive_float,
        default=DEFAULT_CHECK_TIMEOUT,
        help="Discovery/connect timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=600,
        help="Generation inactivity timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key-env", help="Environment variable containing the bearer token (only its name is saved)"
    )
    parser.add_argument(
        "--no-verify-ssl",
        dest="verify_ssl",
        action="store_false",
        help="Skip TLS certificate and hostname verification; save model.verify_ssl=false",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing config without prompting")
    main(**vars(parser.parse_args(argv)))
    return 0
