# Copyright 2026 Maxence Faldor
"""SPICE level 1 MOSFET models used across the AnalogCoder benchmark problems.

The AnalogCoder benchmark defines an NMOS with transconductance parameter 100 µA/V²
and threshold voltage 0.5 V, and a PMOS with 50 µA/V² and -0.5 V. Both models omit
channel-length modulation, body effect, and parasitic capacitance, leaving channel
width and length as design parameters.
"""

from jolty import devices
from jolty.units import Quantity, V, uA


def nmos(width: Quantity, length: Quantity) -> devices.Device:
    """Returns an N-channel level 1 MOSFET with the given `width` and `length`."""
    return devices.mosfet(
        level=1,
        channel="n",
        width=width,
        length=length,
        threshold_voltage=0.5 * V,
        transconductance_parameter=100 * uA / V**2,
    )


def pmos(width: Quantity, length: Quantity) -> devices.Device:
    """Returns a P-channel level 1 MOSFET with the given `width` and `length`."""
    return devices.mosfet(
        level=1,
        channel="p",
        width=width,
        length=length,
        threshold_voltage=-0.5 * V,
        transconductance_parameter=50 * uA / V**2,
    )
