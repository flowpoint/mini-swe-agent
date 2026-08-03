# Output files

mini-SWE-agent stores each run as a JSON trajectory. The CLI prints the selected path; on Linux the default is normally `~/.config/mini-swe-agent/last_mini_run.traj.json`. Use `-o` to choose another file.

## Trajectory structure

The format identifier remains `mini-swe-agent-1.1`:

~~~~json
{
  "info": {
    "model_stats": {
      "instance_cost": 0.0,
      "api_calls": 2
    },
    "config": {
      "agent": {"step_limit": 100},
      "agent_type": "minisweagent.agents.interactive.InteractiveAgent",
      "model": {
        "model_name": "local",
        "base_url": "http://127.0.0.1:8080/v1"
      },
      "model_type": "minisweagent.models.llamacpp_model.LlamaCppModel",
      "environment": {"cwd": "", "timeout": 30},
      "environment_type": "minisweagent.environments.local.LocalEnvironment"
    },
    "mini_version": "2.4.5",
    "exit_status": "Submitted",
    "submission": "..."
  },
  "messages": [],
  "trajectory_format": "mini-swe-agent-1.1"
}
~~~~

The exact configuration mapping includes all resolved Pydantic fields, not just the abbreviated values above.

## Messages

Messages form one linear text conversation.

- `system` contains the action protocol.
- The first `user` message contains the task.
- `assistant` contains llama.cpp text and model metadata.
- Later `user` messages contain command observations or format feedback.
- `exit` records the final status and submission.

An assistant entry resembles:

~~~~json
{
  "role": "assistant",
  "content": "Checking the tree.\n\n```bash\ngit status --short\n```",
  "extra": {
    "actions": [{"command": "git status --short"}],
    "response": {"choices": [{"message": {"content": "..."}}]},
    "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    "timings": {},
    "cost": 0.0,
    "timestamp": 1706000000.0
  }
}
~~~~

`response` is the complete JSON body returned by llama.cpp. `usage` and `timings` are preserved when present and otherwise stored as empty mappings. `cost` is always zero.

An observation entry stores both rendered text and raw execution details:

```json
{
  "role": "user",
  "content": "<returncode>0</returncode>\n<output>\n...</output>",
  "extra": {
    "raw_output": "...",
    "returncode": 0,
    "timestamp": 1706000001.0,
    "exception_info": ""
  }
}
```

Malformed model responses may add the same response metadata to a format-error message so the failure remains diagnosable.

## Exit status

Common terminal statuses include `Submitted`, `LimitsExceeded`, `TimeExceeded`, `RepeatedFormatError`, and uncaught exception names. `submission` contains the model's final output when one was produced.

## Optional SWE-bench predictions

SWE-bench batch runs also create `preds.json` keyed by instance ID:

```json
{
  "owner__repo__123": {
    "model_name_or_path": "local",
    "instance_id": "owner__repo__123",
    "model_patch": "diff --git a/file.py b/file.py\n..."
  }
}
```

This output requires the optional `swebench` dependency group.

## Sensitive data

Trajectories contain complete prompts, model responses, command output, paths, and resolved configuration. Review them before sharing. The local adapter does not record API keys, but shell output can still contain project secrets.

{% include-markdown "../_footer.md" %}
