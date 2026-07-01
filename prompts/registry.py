from __future__ import annotations

import yaml
from pathlib import Path

from config.settings import settings

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(filename: str, key: str) -> str:
    """
    Load a prompt template from a YAML file in the prompts/ directory.
    Falls back to Langfuse prompt registry if configured.
    """
    path = _PROMPTS_DIR / filename
    with open(path) as f:
        data = yaml.safe_load(f)
    return data[key]
