# Copyright 2026 Maxence Faldor
"""The unknowns a circuit's analyses solve for, each named by its path.

A net's voltage is named by the net's path, such as ``out`` or ``amplifier/mid``, and a
branch's current or another internal unknown by its instance's path and a label, such
as ``V1:current``. [`jolty.results`][jolty.results] reads results by these names, and
[`jolty.spice.export`][jolty.spice.export] names the ngspice vector of each.
"""

from dataclasses import dataclass
from typing import Literal

type Quantity = Literal["voltage", "current", "temperature", "state"]
"""What an unknown is: a node's voltage, a branch's current, a thermal net's
temperature above the ambient, in kelvins, or a device's dimensionless state, such as a
memristor's."""


@dataclass(frozen=True, slots=True)
class Unknown:
    """An unknown solved for: a voltage, a current, a temperature or a state."""

    name: str
    quantity: Quantity
