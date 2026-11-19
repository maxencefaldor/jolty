# Copyright 2026 Maxence Faldor
"""The best NOR gate of three memristors found: MAGIC's, its inputs shunted by 2.2 kΩ.

MAGIC's gate puts the two inputs in parallel, in series with the output, and turns the
output off unless both inputs are off. With these memristors, whose on threshold is
twice their off one, it works only between 1.30 V and 1.32 V: a gate with one input on
needs the pulse above twice the output's threshold, and a gate with both inputs off
turns them on just above their own. A resistor across the inputs takes the voltage off
both-off inputs while barely changing a gate with one input on, and 2.2 kΩ balances
the inputs' turning on against the output's turning off: every input computes NOR from
1.14 V to 1.91 V, ±25 % about 1.525 V. Simulate it with
``jolty simulate docs/discovery/memristive_nor/design.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Component, Corner, Plan, Specs, Transient
from jolty.specs import value
from jolty.units import Quantity, V, ns

AMPLITUDE = 1.525
"""The pulse's nominal amplitude, in volts: the middle of the gate's working range."""
SHUNT = 2.2e3
"""The resistance across the inputs, in ohms."""


def build(amplitude: Quantity = AMPLITUDE * V) -> Component:
    """The four gates on their bench, written by one pulse of `amplitude`."""
    return candidates.bench(amplitude.value, SHUNT)


def plan() -> Plan:
    """The write, at the nominal amplitude and 10 % above and below it."""
    return Plan(
        {"write": Transient(13 * ns, max_step=0.025 * ns)},
        corners={
            "low": Corner(amplitude=0.9 * AMPLITUDE * V),
            "high": Corner(amplitude=1.1 * AMPLITUDE * V),
        },
    )


def specs() -> Specs:
    """Each output within 0.1 of its NOR, and each input within 0.05 of its bit."""
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
