from __future__ import annotations

import re


# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"new system prompt",
    r"disregard (your|all)",
    r"<\|.*?\|>",          # special tokens
    r"\[INST\]",           # Llama instruction tags
    r"###\s*instruction",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

MAX_CONTENT_LENGTH = 50_000  # characters


def sanitize_input(text: str) -> str:
    """
    Strip prompt injection patterns and enforce length limits.
    Raises ValueError if the input is clearly malicious.
    """
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH]

    for pattern in _compiled:
        if pattern.search(text):
            raise ValueError(f"Input rejected: matches injection pattern [{pattern.pattern}]")

    return text.strip()


def sanitize_output(text: str) -> str:
    """Strip any injected control tokens from model outputs before returning."""
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL)
    return text.strip()
