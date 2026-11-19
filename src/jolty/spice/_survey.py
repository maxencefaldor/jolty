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
"""What a survey names the model of an ``N`` element by, before its module's name: a
model ngspice loads compiled from Verilog-A, through OSDI."""

SIZES: Final = {"l": "length", "w": "width"}
"""A MOSFET's sizes its bins are selected by, by their SPICE names, and what each is."""

MULTIPLIED: Final = frozenset("rcldjmqx")
"""The elements, by letter, that a subcircuit's multiplier `m` multiplies, which a
subcircuit is given where it has no parameter of that name.

ngspice multiplies each of its elements but its V, E, H, A, K, N, O, P, S, T, U, W and Y
elements and its B sources of a voltage (``inpcom.c``, lines 4240 to 4256): a
resistor's and an inductor's value divided by it, a capacitor's multiplied, and a
diode's, a transistor's and a subcircuit instance's copies `m` multiplied. jolty does
not multiply sources of a current, I, F, G and B elements, yet, and refuses them under a
multiplier other than 1."""

UNMULTIPLIED: Final = frozenset("ifg")
"""The elements, by letter, that ngspice multiplies by a subcircuit's multiplier and
jolty does not yet: sources of a current, but for B sources, which say so themselves."""

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
"""What an element of a letter is when it names no model card."""

UNREAD_ELEMENTS: Final = {
    "o": "ngspice's lossy transmission line, LTRA",
    "u": "ngspice's uniform RC line, URC",
    "y": "ngspice's lossy transmission line, TXL",
    "p": "ngspice's coupled transmission lines, CPL",
}
"""The elements ngspice has and jolty reads not yet, by letter, and what each is."""

BIN_SUFFIX: Final = re.compile(r"\.\d+$")
"""A bin's number after its model's name, as in ``nmos_3p3.12``."""

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
"""What a model card of each type models, by its type."""

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
"""ngspice's models of a MOSFET and a bipolar transistor named apart, by their cards'
levels, of which ngspice reads level 1 by default (``inpdomod.c``)."""


@dataclass(frozen=True, slots=True)
class Use:
    """An element of a subcircuit, the model it needs, and why jolty cannot read it."""

    element: str
    """Its path within the subcircuit, such as ``X1/Q1``; empty for the subcircuit's
    own definitions."""
    model: str
    """The model it needs, such as ``BSIM4 4.5 (level 54)``, or the element's kind."""
    problem: str | None = None
    """Why jolty cannot translate it, if it cannot."""
    undefined: str | None = None
    """The parameter it reads that has no value, which an instance gives, if that is
    why it cannot be translated."""
    middle: tuple[float, float] | None = None
    """The length and the width, in meters, at the middle of the first bin of the model
    it names, if it cannot be translated as no bin holds its size."""
    placed: tuple[str, Mapping[str, float]] | None = None
    """The subcircuit it places and the parameters it gives it, if it places one."""


class UnbinnedError(SpiceError):
    """An element names a binned model, none of whose bins holds its size."""

    def __init__(self, message: str, middle: tuple[float, float]) -> None:
        """The error `message` states, and its model's first bin's `middle`."""
        super().__init__(message)
        self.middle = middle
        """The length and the width, in meters, at the middle of the ranges of the
        first bin of its model, its width multiplied by its fingers where they divide
        it."""


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
    """The model `card` makes, by its type and level, and its version if it gives one.

    A MOSFET, a bipolar transistor and a diode are named with their levels, such as
    ``BSIM4 4.5 (level 54)``, the level evaluated in `scope` with `functions` inlined.
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
    """The model the element `statement` needs, as a reader states it.

    A card's model is named by its type and level, and its version where it gives one,
    such as ``BSIM4 4.5 (level 54)``; an ``N`` element's is the Verilog-A module its
    card names; an element of no card is named by its kind, a capacitor of a charge
    ``Q`` and one whose value reads a voltage so, and a placed subcircuit needs nothing
    of its own, its elements' models being stated apart.
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
