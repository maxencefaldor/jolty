# Copyright 2026 Maxence Faldor
"""The best 1 kHz sine oscillator found: a Wien bridge whose gain a JFET controls.

The JFET's channel sits in the gain leg, and its gate follows the negative peak of 43 %
of the output, so a larger swing pinches the channel and lowers the gain to exactly
three. Dividing before the detector scales the JFET's threshold and the diode's drop
alike. It settles within 100 ms to 2.02 V peak at 995.5 Hz with 0.068 % distortion,
and drifts +16 % at -40 °C and -11 % at 85 °C, the detector diode's -2 mV/K. Simulate
it with ``jolty simulate discovery/sine_oscillator/design.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Component, Corner, Plan, Specs
from jolty.plan import Transient
from jolty.specs import frequency, peak_to_peak
from jolty.units import Hz, V, celsius, ms, us


def build() -> Component:
    """The oscillator alone."""
    return candidates.bench(candidates.sensed_agc(9.5e3, 0.43, 0.1e-6))


def plan() -> Plan:
    """150 ms from power-on, at 27 °C, -40 °C and 85 °C."""
    return Plan(
        {"run": Transient(stop=150 * ms, max_step=1 * us)},
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "hot": Corner(temperature=celsius(85)),
        },
    )


def specs() -> Specs:
    """The prompt's bounds: 1 kHz within 1 %, 4 V peak to peak within 5 %, at 110 ms."""
    return Specs(
        [
            frequency("run", "out", start=110 * ms).within(1e3 * Hz, 10 * Hz),
            peak_to_peak("run", "out", start=110 * ms).within(4 * V, 0.2 * V),
        ]
    )
