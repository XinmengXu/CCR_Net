"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_configuration(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream)
    if not isinstance(configuration, dict):
        raise ValueError("configuration root must be a mapping")
    return configuration
