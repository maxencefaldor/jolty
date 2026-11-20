# Copyright 2026 Maxence Faldor
"""Configuration shared by every front end.

Each setting comes from the first source that sets it: explicit arguments, ``JOLTY_*``
environment variables, or the ``[tool.jolty]`` table of ``pyproject.toml`` in the
working directory.
"""

# Keep lightweight rule-severity types free of pydantic-settings at import time.
__lazy_modules__ = ["jolty.settings._model"]

from enum import StrEnum
from typing import Final

from jolty.settings._model import RuleKey, Settings

ALL: Final = "all"
"""Wildcard key setting the severity level for every check rule at once."""


class Level(StrEnum):
    """Severity level assigned to check rule diagnostics (matching ty naming)."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


__all__ = ["ALL", "Level", "RuleKey", "Settings"]
