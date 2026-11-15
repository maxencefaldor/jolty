# Copyright 2026 Maxence Faldor
"""Every dependency's license, in Rust and in Python, held to one list.

A crate states its license in its manifest, an SPDX expression that `cargo metadata`
reads, for every crate of `Cargo.lock`: the dependencies of every platform, for
development and builds among them. A Python distribution states its license in its
metadata, which `importlib.metadata` reads, for every distribution installed beside
jolty from `uv.lock`: its dependencies and the development tools. The metadata holds an
SPDX expression in `License-Expression` (PEP 639) or, written before that field, in
`License`, a free text that most distributions fill with an SPDX expression too. A
dependency whose metadata states none is clarified here, from the license file it
ships.
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
"""The licenses a dependency may carry, those jolty's Apache-2.0 can carry.

They are permissive, but for the Mozilla Public License, whose copyleft stops at the
files it covers. A dependency offering a choice of licenses passes when it offers one of
them, and one carrying several at once when it carries only these.
"""

CLARIFIED: Final = {
    "cycler": "BSD-3-Clause",
    "kiwisolver": "BSD-3-Clause",
    # The PSF License 2.0's terms, with the Matplotlib Development Team as the licensor.
    "matplotlib": "PSF-2.0",
    "rumdl": "MIT",
    "sortedcontainers": "Apache-2.0",
    "ty": "MIT",
}
"""The licenses of the dependencies whose metadata states no SPDX expression.

Each is read from the license file the dependency ships: cycler's, kiwisolver's and
matplotlib's metadata holds the file's text, sortedcontainers' says "Apache 2.0", and
rumdl's and ty's say nothing but a classifier.
"""

_TOKEN: Final = re.compile(r"[()/]|[^\s()/]+")
"""A token of an SPDX expression: a parenthesis, the slash an old manifest writes for
`OR`, which cargo still reads so, or a word, an identifier or an operator."""

_OPERATORS: Final = frozenset({"AND", "OR", "WITH", "(", ")", "/"})


def _any(tokens: list[str]) -> bool:
    """Whether the alternatives at the start of `tokens` offer an allowed license."""
    met = _all(tokens)
    while tokens[:1] in (["OR"], ["/"]):
        del tokens[0]
        met = _all(tokens) or met
    return met


def _all(tokens: list[str]) -> bool:
    """Whether every license joined at the start of `tokens` is allowed."""
    met = _one(tokens)
    while tokens[:1] == ["AND"]:
        del tokens[0]
        met = _one(tokens) and met
    return met


def _one(tokens: list[str]) -> bool:
    """Whether the license, or parenthesized expression, at the start is allowed."""
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
    """Whether an SPDX expression can be met with the listed licenses alone.

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
    """Each package whose license is not allowed, with its license, or none stated."""
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
    """Whether a stated license is an SPDX expression."""
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
