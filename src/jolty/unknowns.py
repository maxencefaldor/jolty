# Copyright 2026 Maxence Faldor
"""The unknowns a circuit's analyses solve for, each named by its path.

A net's voltage is named by the net's path, such as ``out`` or ``amplifier/mid``, and a
branch's current or another internal unknown by its instance's path and a label, such
as ``V1:current``. [`jolty.results`][jolty.results] reads results by these names, and
[`jolty.spice.export`][jolty.spice.export] names the ngspice vector of each.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

type Nature = Literal["voltage", "current", "temperature", "state"]
"""What an unknown is: a node's voltage, a branch's current, a thermal net's
temperature above the ambient, in kelvins, or a device's dimensionless state, such as a
memristor's."""

MAX_SERIES: Final = 8
"""Default maximum number of unknowns returned by `default_traces`."""


@dataclass(frozen=True, slots=True)
class Unknown:
    """An unknown solved for: a voltage, a current, a temperature or a state."""

    name: str
    nature: Nature


def default_traces(
    unknowns: Sequence[Unknown], nets: Sequence[str] = ()
) -> tuple[str, ...]:
    """Selects the top component's named nets when no explicit traces are given.

    A net the script named at the top has no slash in its path, and no dot or colon,
    which only the names the builder and the engine give other nets carry. Voltage
    unknowns, temperature unknowns, and logic nets are eligible. Without named
    top-level nets, the first voltages and temperatures are chosen, or else the first
    logic nets. At most [`MAX_SERIES`][jolty.unknowns.MAX_SERIES] traces are returned.
    """
    nodes = [
        unknown.name
        for unknown in unknowns
        if unknown.nature in {"voltage", "temperature"}
    ]
    named = [
        name for name in [*nodes, *nets] if not any(mark in name for mark in "/.:")
    ]
    return tuple((named or nodes or list(nets))[:MAX_SERIES])
