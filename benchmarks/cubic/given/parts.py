# Copyright 2026 Maxence Faldor
"""The one transistor of the cubic function generator, of a fast bipolar process.

Every transistor of an integrated circuit made in one process matches every other, so a
diode is an NPN transistor whose base joins its collector.
"""

from jolty import devices
from jolty.units import A, V, fF, ohm, ps


def npn() -> devices.Device:
    """An integrated NPN transistor, its transit frequency gigahertz near 1 mA."""
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
