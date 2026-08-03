# Environments

- `local.py` executes fenced actions directly with Linux Bash. This is the normal agent environment.
- `docker.py` executes commands with the Docker CLI. It is retained for the optional SWE-bench workflow.

`get_environment` accepts only `environment_class: local` or `environment_class: docker`.
