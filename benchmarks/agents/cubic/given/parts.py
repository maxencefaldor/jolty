# Copyright 2026 Maxence Faldor
"""Fast integrated bipolar transistor model provided for the cubic generator task.

All transistors fabricated on the same integrated die match each other, so a diode is
implemented as a diode-connected NPN transistor with base tied to collector.
"""

from jolty import devices
from jolty.units import A, V, fF, ohm, ps


def npn() -> devices.Device:
    """Returns an integrated NPN transistor with GHz transit frequency near 1 mA."""
    return devices.bipolar_transistor(
        saturation_current=1e-16 * A,
        forward_beta=100.0,
        reverse_beta=1.0,
        forward_early_voltage=50 * V,
        base_resistance=50 * ohm,
        collector_resistance=10 * ohm,
        base_emitter_capacitance=20 * fF,
        base_collector_capacitance=10 * fF,
        forward_transit_time=10 * ps,
    )
