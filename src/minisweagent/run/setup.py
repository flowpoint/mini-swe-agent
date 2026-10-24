import argparse
import json
import math

import requests
from jinja2 import StrictUndefined, Template

from minisweagent.config import builtin_config_dir, get_user_config_path
from minisweagent.models.llamacpp_model import LlamaCppModelConfig, normalize_base_url


def _get_json(session: requests.Session, url: str, *, optional: bool = False, **kwargs) -> dict:
    response = session.get(url, timeout=(10, 30), allow_redirects=False, **kwargs)
    if optional and response.status_code in {404, 405, 501}:
        return {}
    if response.status_code != 200:
        raise SystemExit(f"mini setup: GET {url}: HTTP {response.status_code}")
    data = response.json()
    if not isinstance(data, dict):
        raise SystemExit(f"mini setup: expected a JSON object from {url}")
    return data


def main(base_url: str, model: str | None = None, *, force: bool = False) -> None:
    path = get_user_config_path()
    if path.exists() and not force:
        raise SystemExit(f"mini setup: {path} already exists; use --force to replace it")
    base_url = normalize_base_url(base_url)
    with requests.Session() as session:
        session.trust_env = False
        models = _get_json(session, f"{base_url}/models").get("data")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"] for item in models)
        ):
            raise SystemExit("mini setup: API returned no valid models")
        available = [item["id"] for item in models]
        if model is None:
            if len(available) != 1:
                raise SystemExit(f"mini setup: select --model from {json.dumps(available)}")
            model = available[0]
        if model not in available:
            raise SystemExit(f"mini setup: model {json.dumps(model)} not found; available: {json.dumps(available)}")
        props = _get_json(session, f"{base_url.removesuffix('/v1')}/props", optional=True, params={"model": model})
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    print(f"api: {base_url}")
    print(f"model: {json.dumps(model)}")
    print(f"context: {context if context else 'not advertised'}")
    print(f"temperature: {temperature} ({'detected' if 'temperature' in params else 'client default'})")
    print(f"max_tokens: {max_tokens} (client reply limit)")
    print(f"wrote: {path}")


def app(argv: list[str] | None = None, *, prog: str = "mini setup") -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Detect model settings and write the default user config.")
    parser.add_argument("--base-url", default=LlamaCppModelConfig().base_url, help="Server URL, with or without /v1")
    parser.add_argument("-m", "--model", help="Model ID (required when the API lists several models)")
    parser.add_argument("--force", action="store_true", help="Replace an existing config without prompting")
    main(**vars(parser.parse_args(argv)))
    return 0
