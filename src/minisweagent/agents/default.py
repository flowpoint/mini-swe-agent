"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import math
import time
import traceback
from collections import Counter
from importlib import import_module
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field, field_serializer

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import (
    ContextLimitExceeded,
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    TimeExceeded,
    UserInterruption,
)
from minisweagent.utils.serialize import atomic_write, recursive_merge


class AgentConfig(BaseModel, extra="forbid"):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = Field(default=0, ge=0)
    """Maximum number of model calls; 0 means unlimited."""
    max_context_chars: int = Field(default=65536, gt=0)
    """Bound model-visible text; the full trajectory remains saved. Override compact() to customize."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    audit_timeout: float = Field(default=0.0, ge=0)
    """Seconds to audit each command: 0 runs immediately; infinity requires explicit continuation."""

    @field_serializer("audit_timeout", when_used="json")
    def serialize_audit_timeout(self, value: float) -> float | str:
        return "inf" if math.isinf(value) else value


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except KeyboardInterrupt:
                self.add_messages({"role": "exit", "content": "Interrupted", "extra": {"exit_status": "Interrupted"}})
                raise
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        messages = self.compact(self.messages)
        self.n_calls += 1
        message = self.model.query(messages, target=self.env.target)
        self.add_messages(message)
        return message

    def compact(self, messages: list[dict]) -> list[dict]:
        """Keep the task, user corrections and recent complete turns; never edit history."""
        if sum(len(m["content"]) for m in messages) <= self.config.max_context_chars:
            return messages
        notice = {
            "role": "user",
            "content": "[Older turns omitted from context; full history remains in the trajectory.]",
        }
        groups = [messages[:2]]
        for message in messages[2:]:
            if len(groups) == 1 or message["role"] == "assistant" or message.get("extra", {}).get("interrupt_type"):
                groups.append([])
            groups[-1].append(message)
        keep = {0, len(groups) - 1} | {
            i
            for i, group in enumerate(groups)
            if any(m.get("extra", {}).get("interrupt_type", "").startswith("User") for m in group)
        }
        sizes = [sum(len(m["content"]) for m in group) for group in groups]
        remaining = self.config.max_context_chars - len(notice["content"]) - sum(sizes[i] for i in keep)
        if remaining < 0:
            raise ContextLimitExceeded(
                {
                    "role": "exit",
                    "content": "Task, corrections and latest turn exceed max_context_chars; increase it or override compact().",
                    "extra": {"exit_status": "ContextLimitExceeded", "submission": ""},
                }
            )
        for i in reversed(range(1, len(groups))):
            if i in keep:
                continue
            if sizes[i] > remaining:
                break
            keep.add(i)
            remaining -= sizes[i]
        return groups[0] + [notice] + [m for i in sorted(keep - {0}) for m in groups[i]]

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        outputs = []
        try:
            for action in message.get("extra", {}).get("actions", []):
                outputs.append(self.execute_action(action))
        finally:
            result = self.add_messages(
                *self.model.format_observation_messages(message, outputs, self.get_template_vars())
            )
        return result

    def execute_action(self, action: dict) -> dict:
        self.audit(action)
        return self.env.execute(action)

    def audit(self, action: dict) -> None:
        """Per-command hook; concurrent auditors can send SIGINT to interrupt the CLI."""
        if self.config.audit_timeout == 0:
            return
        print(f"Audit ({self.config.audit_timeout:g}s; Ctrl-C to interrupt): {action['command']}", flush=True)
        if math.isinf(self.config.audit_timeout):
            self.confirm_action(action)
        else:
            time.sleep(self.config.audit_timeout)

    def confirm_action(self, action: dict) -> None:
        reason = (
            import_module("minisweagent.agents.prompt_user")
            ._multiline_prompt("Submit an empty message to continue, or explain why this command should not run:\n> ")
            .strip()
        )
        if reason:
            raise UserInterruption(
                {
                    "role": "user",
                    "content": f"Command not executed: {reason}",
                    "extra": {"interrupt_type": "UserRejection"},
                }
            )

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        usage = Counter()
        for message in self.messages:
            if isinstance(reported := message.get("extra", {}).get("usage"), dict):
                usage.update({k: v for k, v in reported.items() if type(v) is int})
        agent_data = {
            "info": {
                "model_stats": {
                    "api_calls": self.n_calls,
                    "elapsed_seconds": int(time.time() - self._start_time),
                    "usage": dict(usage),
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            atomic_write(path, json.dumps(data, indent=2))
        return data
