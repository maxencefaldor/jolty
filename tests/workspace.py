# Copyright 2026 Maxence Faldor
"""Rust workspace metadata read from `cargo metadata` for repository tests.

Returns either workspace member crates from local manifests alone or all crates
in `Cargo.lock` across platforms, including build and development dependencies.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Final, TypedDict

from pydantic import TypeAdapter

__all__ = ["ROOT", "Crate", "Dependency", "crates"]

ROOT: Final = Path(__file__).resolve().parent.parent
"""Repository and Cargo workspace root directory."""


class Dependency(TypedDict):
    """Dependency entry of a Cargo crate, with optional non-normal kind."""

    name: str
    kind: str | None


class Crate(TypedDict):
    """Metadata fields extracted from a Cargo crate manifest."""

    name: str
    version: str
    license: str | None
    rust_version: str | None
    dependencies: list[Dependency]


class _Metadata(TypedDict):
    packages: list[Crate]


_METADATA: Final = TypeAdapter(_Metadata)


def crates(*, dependencies: bool) -> list[Crate]:
    """Returns workspace crates, and `Cargo.lock` crates when `dependencies` is true."""
    cargo = shutil.which("cargo")
    assert cargo is not None, "cargo, which builds the engine, is not on the PATH"
    described = subprocess.run(  # noqa: S603 - cargo, reading the workspace's manifests
        [
            cargo,
            "metadata",
            "--format-version",
            "1",
            "--locked",
            *([] if dependencies else ["--no-deps"]),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return _METADATA.validate_json(described.stdout)["packages"]
