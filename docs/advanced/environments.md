# Execution environments

The default runtime executes model actions in the existing Linux shell. Docker remains available for isolated SWE-bench jobs; other remote and platform-specific backends are outside this fork's supported workflow.

## LocalEnvironment

`LocalEnvironment` is selected by default:

```yaml
environment:
  environment_class: local
  cwd: ""
  timeout: 30
  env:
    PAGER: cat
```

- `cwd` is the command working directory. An empty value uses the directory where `mini` was started.
- `timeout` is the finite limit for each command.
- `env` adds or overrides command environment variables.

Every action starts a new shell process. `cd`, shell variables, and activated environments do not persist to the next action. Include required setup in each command.

Local execution is not a sandbox. It uses the current user's permissions, so keep the agent in `confirm` mode unless the checkout and task are trusted.

## DockerEnvironment

`DockerEnvironment` runs commands with `docker exec` inside a Linux container. It is used by the optional SWE-bench workflow, where the runner supplies the benchmark image and working directory.

A manually managed configuration requires at least an image:

```yaml
environment:
  environment_class: docker
  image: example/image:tag
  cwd: /workspace
  timeout: 60
  run_args:
    - --rm
    - --volume
    - /absolute/path/to/repository:/workspace
```

The Docker daemon, image trust, mounts, and container resource limits are operator responsibilities. The llama.cpp HTTP client still runs in the host Python process, so the default loopback `base_url` continues to address the host server.

For SWE-bench, install the optional dependency group and use the packaged benchmark command rather than hand-writing image settings:

```bash
python -m pip install 'mini-swe-agent[swebench]'
mini-extra swebench --help
```

See [LocalEnvironment](../reference/environments/local.md) and [DockerEnvironment](../reference/environments/docker.md) for API details.

--8<-- "docs/_footer.md"
