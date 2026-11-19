# Copyright 2026 Maxence Faldor
"""The full adder's plan at the spot checks' default tolerance, which one gate breaks.

`examples/full_adder.py` holds its gates to half a mean delay, as an XOR of NAND gates
needs; at a tenth, the default, a spot check of one gate violates its contract.
"""

from full_adder import SUPPLY

from jolty import Plan
from jolty.plan import Characterized, Transient
from jolty.units import ns


def plan() -> Plan:
    """Returns a plan testing eight input combinations with two spot-checked gates."""
    return Plan(
        {"count": Transient(stop=16 * ns)},
        fidelity={"adder/*": Characterized(SUPPLY, spot_checks=2)},
    )
