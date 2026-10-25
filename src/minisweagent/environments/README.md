# Environments

- `local.py` executes fenced actions directly with Linux Bash. This is the normal agent environment.
- The checkout's `benchmarks/docker.py` executes benchmark commands with the Docker CLI and is not installed with the core.

`get_environment` accepts only `environment_class: local`.
