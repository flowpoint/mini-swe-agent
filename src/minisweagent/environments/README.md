# Environments

- `local.py` executes each fenced action in a fresh Python process (`sys.executable -u -c`). Files persist; variables, imports, and changes to cwd/environment do not. `environment.language: bash` selects Bash instead. `environment.interpreter` optionally overrides the interpreter argument list, e.g. `["/project/.venv/bin/python", "-u", "-c"]`. The resolved interpreter and language are saved in trajectories.
- The checkout's `benchmarks/docker.py` executes benchmark commands with the Docker CLI and is not installed with the core.

`get_environment` accepts only `environment_class: local`. Language details live in `targets.py`; prompts, parsing, and human command entry use that same target. `mini setup --language python` tests the interpreter and model response before saving; Python is the default for both setup and normal runs. Existing custom Bash prompts/parser overrides require `environment.language: bash` (or `--language bash`).
