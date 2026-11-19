# Copyright 2026 Maxence Faldor
"""The best 50 Hz notch found: band-pass and inverted low-pass, summed through C and R.

A Tow-Thomas biquad, Q = 1, gives a band-pass output and, two integrations on, an
inverted low-pass output exactly 90° behind it at every frequency. Summed into a
virtual ground through a capacitor and a resistor, which turn them 90° further apart,
their currents cancel at the frequency where their magnitudes meet: the parts' values
place the notch, but cannot make it shallow. Over 200 samples of 1 % resistors and 5 %
capacitors, every one notches deeper than 65 dB between 45 Hz and 55 Hz. Simulate it
with ``jolty simulate docs/discovery/notch/design.py``.
"""

import candidates

from jolty import Component, MonteCarlo, Plan, Specs
from jolty.plan import Ac, decade
from jolty.specs import gain
from jolty.units import Hz, kHz


def build() -> Component:
    """The notch on its bench."""
    return candidates.bench(candidates.summed(1.0))


def plan() -> Plan:
    """The frequency response, nominal and over 200 samples of the tolerances."""
    return Plan(
        {"bode": Ac("V1", decade(1 * Hz, 10 * kHz, 100))},
        monte_carlo=MonteCarlo(200, seed=1),
    )


def specs() -> Specs:
    """The passband within 3 dB at 25 Hz and 100 Hz, on 95 % of the samples."""
    return Specs(
        [
            gain("bode", "out", at=25 * Hz).within(0, 3),
            gain("bode", "out", at=100 * Hz).within(0, 3),
        ],
        yield_target=0.95,
    )
