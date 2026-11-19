# Copyright 2026 Maxence Faldor
"""License compliance checks across all Rust and Python dependencies.

Each Rust crate declares an SPDX expression in its manifest, read via `cargo metadata`
across all platform, dev, and build dependencies in `Cargo.lock`. Each Python package
installed alongside jolty from `uv.lock` declares its license in distribution metadata
read via `importlib.metadata`, using `License-Expression` (PEP 639) or legacy `License`.
Dependencies without parseable SPDX metadata are documented explicitly below from their
shipped license files.
"""

import importlib.metadata
import re
from typing import Final

import pytest

from workspace import crates

LICENSES: Final = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "Apache-2.0 WITH LLVM-exception",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "MIT",
        "MIT-0",
        "MIT-CMU",
        "MPL-2.0",
        "PSF-2.0",
        "Unicode-3.0",
        "Zlib",
    }
)
"""Allowlist of dependency licenses compatible with jolty's Apache-2.0 license.

All listed licenses are permissive except the Mozilla Public License, whose copyleft
scope stops at covered files. Disjunctive expressions pass when at least one option is
allowed, and conjunctive expressions pass when every term is allowed.
"""

CLARIFIED: Final = {
    "bincode-next": "MIT OR Apache-2.0",
    "cycler": "BSD-3-Clause",
    "kiwisolver": "BSD-3-Clause",
    # The PSF License 2.0's terms, with the Matplotlib Development Team as the licensor.
    "matplotlib": "PSF-2.0",
    "rumdl": "MIT",
    "scipy": "BSD-3-Clause",
    "sortedcontainers": "Apache-2.0",
    "ty": "MIT",
}
"""Explicit license mappings for dependencies lacking valid SPDX metadata.

Each entry comes from the package's shipped license file. `cycler`, `kiwisolver`,
`matplotlib`, and `scipy` embed raw license text in metadata, `sortedcontainers` writes
`"Apache 2.0"`, and `rumdl` and `ty` provide classifiers only. `bincode-next` (a
`gungraun` dependency) uses lowercase `or`, which SPDX rejects.
"""

_TOKEN: Final = re.compile(r"[()/]|[^\s()/]+")
"""A token of an SPDX expression: a parenthesis, the slash an old manifest writes for
`OR`, which cargo still reads so, or a word, an identifier or an operator."""

_OPERATORS: Final = frozenset({"AND", "OR", "WITH", "(", ")", "/"})


def _any(tokens: list[str]) -> bool:
    """Returns whether any OR-separated branch at the front of `tokens` is allowed."""
    met = _all(tokens)
    while tokens[:1] in (["OR"], ["/"]):
        del tokens[0]
        met = _all(tokens) or met
    return met


def _all(tokens: list[str]) -> bool:
    """Returns whether every AND-joined license at the front of `tokens` is allowed."""
    met = _one(tokens)
    while tokens[:1] == ["AND"]:
        del tokens[0]
        met = _one(tokens) and met
    return met


def _one(tokens: list[str]) -> bool:
    """Returns whether the leading license atom or parenthesized group is allowed."""
    if not tokens or tokens[0] in _OPERATORS - {"("}:
        message = "a license expected"
        raise ValueError(message)
    if tokens[0] == "(":
        del tokens[0]
        met = _any(tokens)
        if tokens[:1] != [")"]:
            message = "a closing parenthesis expected"
            raise ValueError(message)
        del tokens[0]
        return met
    license_ = tokens.pop(0)
    if tokens[:1] == ["WITH"]:
        del tokens[0]
        if not tokens or tokens[0] in _OPERATORS:
            message = "an exception expected"
            raise ValueError(message)
        license_ = f"{license_} WITH {tokens.pop(0)}"
    return license_ in LICENSES


def allowed(expression: str) -> bool:
    """Returns whether `expression` is satisfiable using allowlisted licenses alone.

    Raises:
        ValueError: If `expression` is not an SPDX expression.
    """
    tokens = _TOKEN.findall(expression)
    met = _any(tokens)
    if tokens:
        message = f"{tokens[0]!r} where the expression should have ended"
        raise ValueError(message)
    return met


type Package = tuple[str, str]
"""A crate or a distribution: its name and its version."""


def _refused(packages: dict[Package, str | None]) -> list[str]:
    """Formats every package whose license is disallowed or missing."""
    refused = []
    for (name, version), stated in sorted(packages.items()):
        expression = CLARIFIED.get(name, stated)
        if expression is None:
            refused.append(f"{name} {version}: no license stated")
            continue
        try:
            met = allowed(expression)
        except ValueError as error:
            refused.append(
                f"{name} {version}: {expression!r} is no expression, {error}"
            )
            continue
        if not met:
            refused.append(f"{name} {version}: {expression}")
    return refused


CRATES: Final = {
    (crate["name"], crate["version"]): crate["license"]
    for crate in crates(dependencies=True)
}
"""Each crate of `Cargo.lock` and the license its manifest states."""

DISTRIBUTIONS: Final = {
    (distribution.name, distribution.version): (
        distribution.metadata.get("License-Expression")
        or distribution.metadata.get("License")
    )
    for distribution in importlib.metadata.distributions()
}
"""Each distribution installed and the license its metadata states."""


def test_every_crate_carries_an_allowed_license() -> None:
    refused = _refused(CRATES)
    assert not refused, "\n".join(refused)


def test_every_distribution_carries_an_allowed_license() -> None:
    refused = _refused(DISTRIBUTIONS)
    assert not refused, "\n".join(refused)


def _expression(stated: str | None) -> bool:
    """Returns whether `stated` parses as a valid SPDX license expression."""
    if stated is None:
        return False
    try:
        allowed(stated)
    except ValueError:
        return False
    return True


def test_every_clarified_dependency_states_no_expression_itself() -> None:
    stated = {
        name: license_ for (name, _), license_ in {**CRATES, **DISTRIBUTIONS}.items()
    }
    needless = [
        f"{name}: no longer a dependency"
        if name not in stated
        else f"{name}: states {stated[name]!r} itself"
        for name in CLARIFIED
        if name not in stated or _expression(stated[name])
    ]
    assert not needless, "\n".join(needless)


@pytest.mark.parametrize(
    ("expression", "met"),
    [
        ("MIT", True),
        ("GPL-3.0-only", False),
        ("MIT OR GPL-3.0-only", True),
        ("MIT AND GPL-3.0-only", False),
        ("MIT/Apache-2.0", True),
        ("(MIT OR Apache-2.0) AND Unicode-3.0", True),
        ("(MIT OR GPL-3.0-only) AND LGPL-2.1-or-later", False),
        ("Apache-2.0 WITH LLVM-exception", True),
        ("GPL-2.0-only WITH Classpath-exception-2.0", False),
        ("MIT AND BSD-3-Clause OR GPL-3.0-only", True),
        ("GPL-3.0-only OR MIT AND LGPL-2.1-or-later", False),
    ],
)
def test_an_expression_is_met_by_the_listed_licenses_alone(
    expression: str, *, met: bool
) -> None:
    assert allowed(expression) is met


@pytest.mark.parametrize(
    "expression",
    ["", "Apache 2.0", "MIT OR", "(MIT", "MIT)", "MIT WITH", "AND MIT"],
)
def test_text_that_is_no_expression_is_refused(expression: str) -> None:
    with pytest.raises(ValueError, match=r"expected|should have ended"):
        allowed(expression)
