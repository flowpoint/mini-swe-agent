"""Extract plain text from trajectory messages for display."""


def get_content_string(message: dict) -> str:
    """Return text content, or an empty string for unsupported message shapes."""
    content = message.get("content")
    return content if isinstance(content, str) else ""
