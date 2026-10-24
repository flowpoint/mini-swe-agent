"""Interactive command auditing and optional human command entry."""

import re
import sys
from typing import Literal, NoReturn

from rich.console import Console
from rich.rule import Rule

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.utils.prompt_user import _multiline_prompt, prompt_session
from minisweagent.exceptions import LimitsExceeded, Submitted, TimeExceeded, UserInterruption
from minisweagent.models.utils.content_string import get_content_string

console = Console(highlight=False)


class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "agent"] = "agent"
    """Who supplies commands; audit_timeout controls review of model commands."""
    whitelist_actions: list[str] = []
    """Skip manual audit waits for commands matching these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


class InteractiveAgent(DefaultAgent):
    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0

    def _interrupt(self, content: str, *, itype: str = "UserInterruption") -> NoReturn:
        raise UserInterruption({"role": "user", "content": content, "extra": {"interrupt_type": itype}})

    def add_messages(self, *messages: dict) -> list[dict]:
        # Extend supermethod to print messages
        for msg in messages:
            role, content = msg.get("role") or msg.get("type", "unknown"), get_content_string(msg)
            if role == "assistant":
                console.print(
                    f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.n_calls}[/bold], [bold]${self.cost:.2f}[/bold]):[/red]\n",
                    end="",
                    highlight=False,
                )
            else:
                console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
            console.print(content, highlight=False, markup=False)
        return super().add_messages(*messages)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            command = self._prompt_and_handle_slash_commands("[bold yellow]>[/bold yellow] ")
            if not command.startswith("/audit "):
                msg = {
                    "role": "user",
                    "content": f"User command: \n```bash\n{command}\n```",
                    "extra": {"actions": [{"command": command}]},
                }
                self.add_messages(msg)
                return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except TimeExceeded:
            # A wall-clock limit can't be lifted by raising the step/cost limits
            # (the next query re-checks the clock and raises again), so prompting
            # would loop forever. Always stop cleanly instead.
            raise
        except LimitsExceeded:
            if not self._stdin_is_interactive():
                # No terminal to prompt for new limits -- e.g. an unattended
                # run with audit_timeout=0, or stdin redirected from /dev/null
                # (CI, a sandbox). Stop cleanly so the trajectory is saved with a
                # LimitsExceeded exit status, instead of crashing on EOFError when
                # reading input.
                raise
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.n_calls} steps, ${self.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    @staticmethod
    def _stdin_is_interactive() -> bool:
        """Whether an interactive terminal is available to prompt the user.

        Returns False for unattended runs (e.g. in CI, or inside a
        sandbox with stdin redirected from /dev/null), where calling ``input()``
        would raise ``EOFError`` and crash the run.
        """
        try:
            return sys.stdin is not None and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def step(self) -> list[dict]:
        # Override the step method to handle user interruption
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            interruption_message = self._prompt_and_handle_slash_commands(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message == "/u" or interruption_message.startswith("/audit "):
                interruption_message = "Temporary interruption caught."
            self._interrupt(f"Interrupted by user: {interruption_message}")

    def execute_actions(self, message: dict) -> list[dict]:
        try:
            return super().execute_actions(message)
        except Submitted as e:
            self._check_for_new_task_or_submit(e)

    def _add_observation_messages(self, message: dict, outputs: list[dict]) -> list[dict]:
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _check_for_new_task_or_submit(self, e: Submitted) -> NoReturn:
        """Check if user wants to add a new task or submit."""
        if self.config.confirm_exit:
            message = (
                "[bold yellow]Agent wants to finish.[/bold yellow] "
                "[bold green]Type new task[/bold green] or [bold]Enter[/bold] to quit "
                "([bold]/h[/bold] for commands)\n"
                "[bold yellow]>[/bold yellow] "
            )
            user_input = self._prompt_and_handle_slash_commands(message).strip()
            if user_input == "/u":  # directly continue
                self._interrupt("Switched to human mode.")
            elif user_input.startswith("/audit "):  # ask again
                return self._check_for_new_task_or_submit(e)
            elif user_input:
                self._interrupt(f"The user added a new task: {user_input}", itype="UserNewTask")
        raise e

    def audit(self, action: dict) -> None:
        if self.config.mode == "human" or any(re.match(r, action["command"]) for r in self.config.whitelist_actions):
            return
        super().audit(action)

    def confirm_action(self, action: dict) -> None:
        prompt = (
            "[bold yellow]Execute command?[/] [green][bold]Enter[/] to continue[/], "
            "[red]type [bold]comment[/] to reject[/], or [blue][bold]/h[/] to show available commands[/]\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_slash_commands(prompt).strip():
            case "":
                return
            case "/u":  # Skip execution action and get back to query
                self._interrupt("Commands not executed. Switching to human mode", itype="UserRejection")
            case _ if user_input.startswith("/audit "):
                self.audit(action)
            case _:
                self._interrupt(
                    f"Commands not executed. The user rejected your commands with the following message: {user_input}",
                    itype="UserRejection",
                )

    def _prompt_and_handle_slash_commands(self, prompt: str, *, _multiline: bool = False) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        if _multiline:
            return _multiline_prompt()
        user_input = prompt_session.prompt("")
        if user_input == "/m":
            return self._prompt_and_handle_slash_commands(prompt, _multiline=True)
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"Audit timeout: [bold green]{self.config.audit_timeout:g}s[/bold green]\n"
                "[bold green]/audit SECONDS[/bold green] to resume model commands with this audit window (0: immediate; inf: wait for continuation)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
                f"[bold green]/m[/bold green] to enter multiline comment",
            )
            return self._prompt_and_handle_slash_commands(prompt)
        if user_input == "/u" or user_input.startswith("/audit "):
            mode = "human" if user_input == "/u" else "agent"
            timeout = self.config.audit_timeout if mode == "human" else float(user_input.split(maxsplit=1)[1])
            if not timeout >= 0:
                raise ValueError("audit timeout must be non-negative (or inf)")
            if self.config.mode == mode and self.config.audit_timeout == timeout:
                return self._prompt_and_handle_slash_commands(
                    f"[bold red]Already in {mode} mode with audit timeout {timeout:g}s.[/bold red]\n{prompt}"
                )
            self.config.mode = mode
            self.config.audit_timeout = timeout
            console.print(f"Mode: [bold green]{mode}[/bold green]; audit timeout: [bold green]{timeout:g}s[/bold green].")
            return user_input
        return user_input
