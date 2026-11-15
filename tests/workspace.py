# Copyright 2026 Maxence Faldor
"""The Rust workspace as `cargo metadata` describes it, for the tests that read it.

The workspace's own crates, read from their manifests alone, or every crate of
`Cargo.lock` with them: the dependencies of every platform, for development and builds
among them, whose manifests cargo downloads if it has not yet.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Final, TypedDict

from pydantic import TypeAdapter

__all__ = ["ROOT", "Crate", "Dependency", "crates"]

ROOT: Final = Path(__file__).resolve().parent.parent
"""The repository's root, the workspace's."""


class Dependency(TypedDict):
    """A crate a crate depends on, and for what, if not for its own code."""

    name: str
    kind: str | None


class Crate(TypedDict):
    """A crate, as its manifest describes it."""

    name: str
    version: str
    license: str | None
    rust_version: str | None
    dependencies: list[Dependency]


class _Metadata(TypedDict):
    packages: list[Crate]


_METADATA: Final = TypeAdapter(_Metadata)


def crates(*, dependencies: bool) -> list[Crate]:
    """The workspace's crates, and, if `dependencies`, every crate of `Cargo.lock`."""
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
