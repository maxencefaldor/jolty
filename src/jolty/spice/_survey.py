# Copyright 2026 Maxence Faldor
"""SPICE element survey metadata, model descriptions, and binning diagnostics."""

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue

from jolty.devices import Device, device
from jolty.errors import SpiceError
from jolty.spice import models
from jolty.spice._params import NO_FUNCTIONS, split_keywords
from jolty.spice._preprocess import letter
from jolty.spice.expressions import Function, evaluate, inlined, number
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, tokens

VERILOG_A: Final = "Verilog-A"
"""Survey label prefix for Verilog-A OSDI models referenced by ``N`` elements."""

SIZES: Final = {"l": "length", "w": "width"}
"""MOSFET bin-selection dimension keys mapped from SPICE names to full names."""

MULTIPLIED: Final = frozenset("rcldjmqx")
"""Element letters scaled by a subcircuit multiplier `m` when `m` is not a formal
subcircuit parameter.

Matches ngspice (``inpcom.c``, lines 4240 to 4256): resistors and inductors divide by
`m`, capacitors multiply by `m`, and diodes, transistors, and subcircuits multiply
their copy count `m`. Current sources (`I`, `F`, `G`, and `B`) are not multiplied yet
and fail when `m != 1`.
"""

UNMULTIPLIED: Final = frozenset("ifg")
"""Current-source element letters scaled by ngspice under a subcircuit multiplier and
rejected by jolty (excluding `B` sources, which check separately)."""

ELEMENT_KINDS: Final = {
    "r": "resistor",
    "c": "capacitor",
    "l": "inductor",
    "v": "voltage source",
    "i": "current source",
    "e": "controlled source",
    "f": "controlled source",
    "g": "controlled source",
    "h": "controlled source",
    "b": "behavioral source",
    "k": "mutual inductance",
    "t": "transmission line",
}
"""Readable element kind names keyed by SPICE letter when no model card applies."""

UNREAD_ELEMENTS: Final = {
    "o": "ngspice's lossy transmission line, LTRA",
    "u": "ngspice's uniform RC line, URC",
    "y": "ngspice's lossy transmission line, TXL",
    "p": "ngspice's coupled transmission lines, CPL",
}
"""Unsupported ngspice transmission-line element letters mapped to descriptions."""

BIN_SUFFIX: Final = re.compile(r"\.\d+$")
"""Numeric bin suffix appended to a binned model name, as in ``nmos_3p3.12``."""

_FAMILIES: Final = {
    "nmos": "MOSFET",
    "pmos": "MOSFET",
    "npn": "bipolar transistor",
    "pnp": "bipolar transistor",
    "d": "diode",
    "njf": "JFET",
    "pjf": "JFET",
    "r": "resistor's model card",
    "c": "capacitor's model card",
    "l": "inductor's model card",
}
"""Device family labels keyed by case-folded SPICE `.model` card type."""

_LEVELS: Final = {
    "MOSFET": {
        level: name
        for levels, name in (
            (models.BSIM3_LEVELS, "BSIM3"),
            (models.BSIM4_LEVELS, "BSIM4"),
            (models.BSIM_SOI_LEVELS, "BSIM-SOI"),
            (models.HISIM2_LEVELS, "HiSIM2"),
            (models.HISIM_HV_LEVELS, "HiSIM_HV"),
        )
        for level in levels
    },
    "bipolar transistor": {
        1: "Gummel-Poon",
        **{int(level): "VBIC" for level in models.VBIC_LEVELS},
    },
}
"""Named ngspice MOSFET and bipolar model families keyed by card level
(``inpdomod.c``)."""


@dataclass(frozen=True, slots=True)
class Use:
    """Survey record describing one subcircuit element, its model, and any error."""

    element: str
    """Hierarchical path within the subcircuit (such as ``X1/Q1``), or empty for
    subcircuit-level diagnostics."""
    model: str
    """Model description (such as ``BSIM4 4.5 (level 54)``) or element kind."""
    problem: str | None = None
    """Reason translation failed, or None when translation succeeded."""
    undefined: str | None = None
    """Undefined parameter name required by the element, when applicable."""
    middle: tuple[float, float] | None = None
    """Midpoint `(length, width)` in meters of the first model bin when unbinned."""
    placed: tuple[str, Mapping[str, float]] | None = None
    """Subcircuit name and argument dictionary for a subcircuit placement."""


class UnbinnedError(SpiceError):
    """Error raised when a binned model has no bin containing an instance's size."""

    def __init__(self, message: str, middle: tuple[float, float]) -> None:
        """Initializes an unbinned-model error with `message` and first-bin `middle`."""
        super().__init__(message)
        self.middle = middle
        """Midpoint `(length, width)` in meters of the first model bin, with width
        multiplied by finger count when width is divided per finger."""


def bin_base(key: str) -> str | None:
    """Returns the case-folded base model name for a binned key, or None."""
    base, dot, suffix = key.rpartition(".")
    if not dot or not suffix or not (suffix.isascii() and suffix.isdigit()):
        return None
    return base


def checked_device(
    statement: Statement, model: str, parameters: dict[str, JsonValue]
) -> Device:
    """Constructs a `model` device, raising `SpiceError` on validation failure."""
    try:
        return device(model, parameters)
    except ValueError as error:
        raise statement.fail(str(error)) from None


def describe_card(
    card: ModelCard,
    scope: Mapping[str, float],
    functions: Mapping[str, Function] = NO_FUNCTIONS,
) -> str:
    """Formats a human-readable model description for `card` with level and version.

    MOSFETs, bipolar transistors, and diodes include their evaluated level (such as
    ``BSIM4 4.5 (level 54)``) using `scope` and inlined `functions`.
    """
    family = _FAMILIES.get(card.kind, f"`{card.kind}`")
    if family not in {*_LEVELS, "diode"}:
        return family
    written = card.parameters.get("level")
    level = 1.0 if written is None else evaluate(inlined(written, functions), scope)
    level_key = int(level) if math.isfinite(level) and level.is_integer() else None
    named = (
        _LEVELS.get(family, {}).get(level_key, family)
        if level_key is not None
        else family
    )
    version = None if card.version is None else number(card.version)
    given = f" {version:g}" if version is not None else ""
    return f"{named}{given} (level {level:g})"


def needed_model(
    statement: Statement,
    find_card: Callable[[str], ModelCard | None],
    scope: Mapping[str, float],
    functions: Mapping[str, Function],
) -> str:
    """Identifies the model or element kind required by `statement` for survey output.

    Named model cards format via `describe_card`; ``N`` elements report their
    Verilog-A module; cardless elements report their element kind (distinguishing
    charge-based and voltage-dependent capacitors). Subcircuit placements (`X`) return
    an empty string because their inner elements report individually.
    """
    element_letter = letter(statement)
    if element_letter == "x":
        return ""
    words = tokens(statement.text)[1:]
    found = next(
        (
            card
            for word in words
            if word != "=" and (card := find_card(word)) is not None
        ),
        None,
    )
    if found is None:
        kind = ELEMENT_KINDS.get(element_letter, f"element `{element_letter.upper()}`")
        if element_letter == "c" and "q" in split_keywords(words)[1]:
            return f"{kind} of a charge"
        reads = any("v(" in word.casefold() for word in words)
        return f"{kind} whose value reads a voltage" if reads else kind
    if element_letter == "n":
        return f"{VERILOG_A} `{found.kind}`"
    return describe_card(found, scope, functions)


def raise_unbinned(
    statement: Statement,
    name: str,
    instance: Mapping[str, float],
    middle: tuple[float, float] | None,
    fingers: float,
) -> None:
    """Fails where `name` has bins, none of which holds the size of `instance`.

    The error states the size, and carries `middle` scaled by `fingers`.

    Raises:
        UnbinnedError: When `middle` is not None.
    """
    if middle is None:
        return
    length, width = middle
    sizes = [
        f"a {what} of {instance[key]!r} m"
        for key, what in SIZES.items()
        if key in instance
    ]
    held = f" holds {' and '.join(sizes)}" if sizes else " is selected without a size"
    message = f"no bin of `{name}`{held}"
    raise UnbinnedError(str(statement.fail(message)), (length, width * fingers))
