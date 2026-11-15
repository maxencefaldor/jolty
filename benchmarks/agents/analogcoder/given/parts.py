# Copyright 2026 Maxence Faldor
"""The two transistors of AnalogCoder's problems: SPICE level 1 MOSFETs.

They are AnalogCoder's own models: an NMOS with a transconductance parameter of
100 µA/V² and a threshold of 0.5 V, and a PMOS with 50 µA/V² and -0.5 V. Neither has
channel-length modulation, body effect or capacitance; only the width and the length
are the designer's.
"""

from jolty import devices
from jolty.units import Quantity, V, uA


def nmos(width: Quantity, length: Quantity) -> devices.Device:
    """An N-channel MOSFET of `width` and `length`."""
    return devices.mosfet(
        level=1,
        channel="n",
        width=width,
        length=length,
        threshold_voltage=0.5 * V,
        transconductance_parameter=100 * uA / V**2,
    )


def pmos(width: Quantity, length: Quantity) -> devices.Device:
    """A P-channel MOSFET of `width` and `length`."""
    return devices.mosfet(
        level=1,
        channel="p",
        width=width,
        length=length,
        threshold_voltage=-0.5 * V,
        transconductance_parameter=50 * uA / V**2,
    )
