# Copyright 2026 Maxence Faldor
"""Shared Hypothesis configuration and types for the parser fuzz suite.

Each test runs deterministically with a fast example budget in default CI runs so the
suite finishes well under ten seconds. Set `JOLTY_FUZZ_EXAMPLES` or pass
`--hypothesis-profile=nightly` for longer searches.
"""

import os
from typing import Final

import numpy as np
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

DEFAULT_EXAMPLES: Final = 30
"""Default per-property example budget keeping CI runtime fast and deterministic."""

SEARCH: Final = os.environ.get("JOLTY_FUZZ_EXAMPLES")
"""Override example count for ad-hoc fuzzing searches."""

type ArchiveEntries = dict[str, np.ndarray | bytes]
type JsonObject = dict[str, object]


def _fuzz_settings(default_examples: int = DEFAULT_EXAMPLES) -> settings:
    """Returns Hypothesis settings tuned for CI speed, profile overrides, or search."""
    suppressed = [HealthCheck.too_slow, HealthCheck.filter_too_much]
    if SEARCH is not None:
        return settings(
            max_examples=int(SEARCH),
            derandomize=False,
            deadline=None,
            suppress_health_check=suppressed,
        )
    if settings.get_current_profile_name() != "default":
        return settings(parent=settings.default, suppress_health_check=suppressed)
    return settings(
        max_examples=default_examples,
        derandomize=True,
        deadline=None,
        suppress_health_check=suppressed,
    )


FUZZ: Final = _fuzz_settings()
FAST_FUZZ: Final = _fuzz_settings(default_examples=20)


def one_of_strings(
    strategies: list[st.SearchStrategy[str]],
) -> st.SearchStrategy[str]:
    """Combines string strategies while preserving strict static typing."""
    combined = strategies[0]
    for strategy in strategies[1:]:
        combined = combined | strategy
    return combined
