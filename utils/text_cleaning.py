"""Small text cleanup helpers shared across model-output paths."""
from __future__ import annotations

import re
from typing import Any


_THINKING_TAG_RE = re.compile(
    r"<(?:think|thinking)\b[^>]*>.*?</(?:think|thinking)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_LEADING_THINKING_RE = re.compile(
    r"^\s*<(?:think|thinking)\b[^>]*>.*?(?:\n\s*\n|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_hidden_thinking(text: Any) -> str:
    """Remove hidden-thinking tags leaked into model-visible content."""
    value = "" if text is None else str(text)
    original = value
    previous = None
    while previous != value:
        previous = value
        value = _THINKING_TAG_RE.sub("", value)
    value = _UNCLOSED_LEADING_THINKING_RE.sub("", value)
    return value.strip() if value != original else value
