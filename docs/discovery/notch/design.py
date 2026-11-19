# Copyright 2026 Maxence Faldor
"""Tolerance-immune 50 Hz notch filter summing biquad band-pass and low-pass outputs.

A Q = 1 Tow-Thomas biquad generates a band-pass output and an inverted low-pass
output exactly 90° apart at every frequency.
Summing both outputs into a virtual ground through a capacitor and resistor adds
another 90° shift, forcing exact cancellation where current magnitudes match.
Component tolerances shift notch frequency without making the notch shallow.
Across 200 Monte Carlo samples of 1% resistors and 5% capacitors, every circuit
notches deeper than 65 dB between 45 Hz and 55 Hz.
Simulate via ``jolty simulate docs/discovery/notch/design.py``.
"""

import candidates

from jolty import Component, MonteCarlo, Plan, Specs
from jolty.plan import Ac, decade
from jolty.specs import gain
from jolty.units import Hz, kHz


def build() -> Component:
    """Builds the summed Tow-Thomas notch filter on the AC test bench."""
    return candidates.bench(candidates.summed(1.0))


def plan() -> Plan:
    """Returns the AC response plan with 200 Monte Carlo tolerance samples."""
    return Plan(
        {"bode": Ac("V1", decade(1 * Hz, 10 * kHz, 100))},
        monte_carlo=MonteCarlo(200, seed=1),
    )


def specs() -> Specs:
    """Returns specs checking ±3 dB passband at 25 Hz and 100 Hz for 95% yield."""
    return Specs(
        [
            gain("bode", "out", at=25 * Hz).within(0, 3),
            gain("bode", "out", at=100 * Hz).within(0, 3),
        ],
        yield_target=0.95,
    )
