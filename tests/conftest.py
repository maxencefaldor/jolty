# Copyright 2026 Maxence Faldor
"""Preloads local modules whose constants Hypothesis may draw during tests.

Hypothesis occasionally draws constants discovered in the source of loaded
local modules (non-stdlib modules outside site-packages, namely every `jolty`
module and benchmark script). Delayed imports would alter generated circuits
across test orders and trigger Hypothesis flakiness checks because the constant
pool grows as modules load. Preloading every local module here keeps the
constant pool identical across runs regardless of which tests execute.
"""

import importlib
import pkgutil
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

import jolty

settings.register_profile(
    "ci",
    max_examples=40,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=1000,
    derandomize=False,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

collect_ignore = ["guide"]
"""Guide output scripts executed alongside examples by `test_guide.py`.

Excluded from pytest collection so doctest discovery does not import them
without the example directory on `sys.path`.
"""

for found in pkgutil.walk_packages(jolty.__path__, f"{jolty.__name__}."):
    # The command line's entry point runs the command line as it loads.
    if not found.name.endswith(".__main__"):
        importlib.import_module(found.name)

for script in (
    "arms",
    "bench",
    "capabilities",
    "compare",
    "corpus",
    "decks",
    "deterministic.bench",
    "documented",
    "exact",
    "fixtures",
    "matrix",
    "measuring",
    "run",
    "sandbox",
    "score",
    "scoring",
    "simulators",
    "stub",
    "suite",
    "timing",
    "transcript",
    "verdict",
):
    importlib.import_module(script)


@pytest.fixture
def in_tmp_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs a test in a dedicated temporary working directory."""
    monkeypatch.chdir(tmp_path)
