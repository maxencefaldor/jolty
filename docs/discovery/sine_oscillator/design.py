# Copyright 2026 Maxence Faldor
"""JFET-controlled 1 kHz Wien bridge sine oscillator.

The JFET channel sits in the gain leg while its gate tracks the negative peak of
43% of the output, pinching off to hold closed-loop gain at three.
Dividing the output before the detector scales the JFET threshold and diode drop
together.
The oscillator settles within 100 ms to 2.02 V peak at 995.5 Hz with 0.068%
distortion, drifting +16% at -40 °C and -11% at 85 °C due to detector diode drift.
Simulate via ``jolty simulate docs/discovery/sine_oscillator/design.py``.
"""

import candidates

from jolty import Component, Corner, Plan, Specs
from jolty.plan import Transient
from jolty.specs import frequency, peak_to_peak
from jolty.units import Hz, V, celsius, ms, us


def build() -> Component:
    """Builds the sensed-AGC Wien bridge oscillator on its test bench."""
    return candidates.bench(candidates.sensed_agc(9.5e3, 0.43, 0.1e-6))


def plan() -> Plan:
    """Returns a 150 ms startup transient plan across 27 °C, -40 °C, and 85 °C."""
    return Plan(
        {"run": Transient(stop=150 * ms, max_step=1 * us)},
        corners={
            "cold": Corner(temperature=celsius(-40)),
            "hot": Corner(temperature=celsius(85)),
        },
    )


def specs() -> Specs:
    """Returns specs checking 1 kHz (±1%) and 4 V peak-to-peak (±5%) after 110 ms."""
    return Specs(
        [
            frequency("run", "out", start=110 * ms).within(1e3 * Hz, 10 * Hz),
            peak_to_peak("run", "out", start=110 * ms).within(4 * V, 0.2 * V),
        ]
    )
