# Copyright 2026 Maxence Faldor
"""Shunted MAGIC three-memristor NOR gate with ±25% pulse-amplitude margin.

A MAGIC gate wires two input memristors in parallel, in series with the output
memristor, turning the output off unless both inputs are off.
With an on-threshold magnitude twice the off-threshold, an unshunted gate works only
from 1.30 V to 1.32 V.
A 2.2 kΩ shunt resistor across the inputs diverts voltage from both-off inputs while
barely perturbing gates with an active input.
All four input combinations compute NOR from 1.14 V to 1.91 V (±25% around 1.525 V).
Simulate via ``jolty simulate docs/discovery/memristive_nor/design.py``.
"""

import candidates

from jolty import Component, Corner, Plan, Specs, Transient
from jolty.specs import value
from jolty.units import Quantity, V, ns

AMPLITUDE = 1.525
"""Nominal write pulse amplitude in volts, centered in the operating window."""
SHUNT = 2.2e3
"""Input shunt resistance in ohms."""


def build(amplitude: Quantity = AMPLITUDE * V) -> Component:
    """Builds the four-gate test bench driven by a write pulse of `amplitude`."""
    return candidates.bench(amplitude.value, SHUNT)


def plan() -> Plan:
    """Returns the transient write plan at nominal amplitude plus ±10% corners."""
    return Plan(
        {"write": Transient(13 * ns, max_step=0.025 * ns)},
        corners={
            "low": Corner(amplitude=0.9 * AMPLITUDE * V),
            "high": Corner(amplitude=1.1 * AMPLITUDE * V),
        },
    )


def specs() -> Specs:
    """Returns specs checking output states within 0.1 and inputs within 0.05."""
    end = 12.5 * ns
    requirements = []
    for index, (a, b) in enumerate(candidates.CASES, start=1):
        nor = 0.0 if a or b else 1.0
        output = value("write", f"X{index}/MO:state", at=end)
        requirements.append(output.within(1 - nor, 0.1))
        for name, bit in (("MA", a), ("MB", b)):
            state = value("write", f"X{index}/{name}:state", at=end)
            requirements.append(state.within(1 - bit, 0.05))
    return Specs(requirements)
