# FAQ

## Runtime

??? question "Which operating systems are supported?"

    This fork supports Linux with Python 3.11 or newer. Windows and macOS behavior is not supported or tested.

??? question "Which model servers are supported?"

    One local `llama-server` from llama.cpp is supported through its OpenAI-compatible Chat Completions endpoint. Set `model_name` to the server's `--alias` value. The client does not route between servers or discover models automatically.

??? question "Do I need credentials?"

    No. The default endpoint is loopback-only, the adapter does not read API keys, and URLs containing credentials are rejected. The HTTP session also ignores inherited proxy and credential settings.

??? question "Which GGUF models work?"

    The HTTP protocol works with llama.cpp models that support Chat Completions, but agent quality depends on the model's instruction following and chat template. The model must reliably emit exactly one fenced `bash` block when asked to act.

??? question "Is generation deterministic?"

    The default sends `temperature: 0`. Keeping the same model, prompt, llama.cpp build, context, and server parameters gives the most reproducible behavior, but exact bit-for-bit output is a server/runtime concern.

## Agent behavior

??? question "Why fenced bash instead of native tool calls?"

    The fork deliberately keeps one text protocol. The assistant returns text, the existing regular-expression parser extracts exactly one `bash` fence, and the local environment executes it. Native function/tool calling is out of scope.

??? question "Why is there no persistent shell session?"
    <a name="why-no-shell-session"></a>

    Each action starts a new subprocess. This makes completion, timeouts, and process-group cleanup explicit and prevents one bad command from corrupting later shell interactions. Commands such as `cd` and `export` do not persist; include them in every action that needs them.

??? question "Is local execution sandboxed?"

    No. `LocalEnvironment` runs with the current user's permissions. Confirmation mode asks before commands; `--yolo` removes that safeguard. Use a disposable checkout or the Docker environment used for SWE-bench when isolation is required.

??? question "How do I stop runaway work?"

    Use `agent.step_limit` and `agent.wall_time_limit_seconds`. Per-command execution is bounded by `environment.timeout`, and each HTTP request uses finite connect/read timeouts. Monetary cost limits are ineffective because local responses record `cost: 0.0`.

## Configuration and output

??? question "How do I change the endpoint or alias?"

    Retain the packaged base and add overrides:

    ```bash
    mini -c mini.yaml \
      -c model.base_url=http://127.0.0.1:9090/v1 \
      -c model.model_name=my-alias \
      -t "Inspect this repository"
    ```

??? question "Where is global configuration stored?"

    Runtime settings are YAML or process environment variables; this fork does not load a credential `.env` file. On Linux, the default trajectory is written under the platform configuration directory, normally `~/.config/mini-swe-agent/last_mini_run.traj.json`. Use `-o` to choose another path.

??? question "What does cost mean for a local model?"

    The adapter always records `cost: 0.0` and preserves token usage and llama.cpp timing metadata when the server returns them. There is no monetary cost calculation.

## Optional benchmarks

??? question "Why does SWE-bench ask for an extra dependency?"

    `datasets` is intentionally excluded from the normal installation. Install `mini-swe-agent[swebench]` only for SWE-bench commands; normal agent and CLI startup do not import it.

{% include-markdown "_footer.md" %}
