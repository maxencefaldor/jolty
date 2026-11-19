# Copyright 2026 Maxence Faldor
"""Shared pytest fixtures for the parser fuzz suite."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def scratch_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Returns a shared temporary directory reused across fuzz iterations."""
    return tmp_path_factory.mktemp("fuzz_parsers")
