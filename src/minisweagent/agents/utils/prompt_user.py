from prompt_toolkit.filters import is_searching
from prompt_toolkit.formatted_text.html import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import PromptSession

from minisweagent import global_config_dir

# Keep CSI-u and xterm modified Enter distinct from ordinary Enter.
ANSI_SEQUENCES.update({
    sequence: Keys.ControlG
    for modifier in (2, 5)
    for sequence in (f"\x1b[13;{modifier}u", f"\x1b[27;{modifier};13~")
})
_key_bindings = KeyBindings()


@_key_bindings.add("c-g", filter=~is_searching)
def _submit(event: KeyPressEvent) -> None:
    event.current_buffer.validate_and_handle()


_history = FileHistory(global_config_dir / "interactive_history.txt")
prompt_session = PromptSession(history=_history, key_bindings=_key_bindings)
_multiline_prompt_session = PromptSession(history=_history, key_bindings=_key_bindings, multiline=True)


def _multiline_prompt() -> str:
    return _multiline_prompt_session.prompt(
        "",
        bottom_toolbar=HTML(
            "Newline: <b>Enter</b> | Send: <b>Shift+Enter / Ctrl+Enter / Ctrl+G</b> | "
            "History: <b>Up/Down / Ctrl+R</b>"
        ),
    )
