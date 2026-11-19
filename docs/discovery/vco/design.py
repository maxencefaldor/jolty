# Copyright 2026 Maxence Faldor
"""Delay-compensated MOSFET-switched integrator and Schmitt trigger VCO.

An n-channel MOSFET reverses integrator current with purely resistive channel drop
rather than a bipolar saturation offset.
A 2.2 nF speed-up capacitor across the Schmitt trigger 10 kΩ feedback accelerates
regeneration, cutting switching delay from 6 to 9 µs down to roughly 2 µs.
A 2 kΩ resistor in series with the 1 nF integrating capacitor steps the triangle at
each reversal by 2 µs worth of ramp voltage, canceling delay across frequency.
Every frequency from 0.5 V to 5 V falls within 0.1% of 996 Hz/V.
Simulate via ``jolty simulate docs/discovery/vco/design.py``.
"""

import candidates

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import frequency
from jolty.units import Hz, V, s, us
from jolty.waveforms import Piecewise

LEVELS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
"""Control voltage levels in volts, each held for eight cycles at 1 kHz/V."""

SLOPE = 996.07
"""Fitted origin-constrained tuning slope in Hz/V."""


def windows() -> list[tuple[float, float]]:
    """Returns start and end times in seconds for each stepped control level."""
    edges: list[tuple[float, float]] = []
    start = 0.0
    for volts in LEVELS:
        edges.append((start, start + 8e-3 / volts))
        start += 8e-3 / volts
    return edges


@component
def stepped(c: Builder) -> None:
    """Builds the compensated VCO driven by a piecewise-linear ramp across `LEVELS`."""
    ground = c.port("ground", Role.GROUND)
    points = []
    for (begin, _), volts in zip(windows(), LEVELS, strict=True):
        if points:
            points.append((begin * s, points[-1][1]))
        points.append(((begin + 1e-6) * s, volts * V))
    source = c.place(devices.voltage_source(Piecewise(points)), name="VC")
    oscillator = candidates.mosfet_switched(compensation=2e3, speedup=2.2e-9)
    placed = c.place(oscillator, name="X1")
    c.connect(source.positive, placed.control, name="control")
    c.connect(placed.output, name="out")
    c.connect(source.negative, placed.ground, ground)


def build() -> Component:
    """Builds the stepped-control VCO sweeping from 0.5 V to 5 V."""
    return stepped()


def plan() -> Plan:
    """Returns the transient plan spanning all six levels at 0.1 µs maximum step."""
    return Plan({"run": Transient(stop=windows()[-1][1] * s, max_step=0.1 * us)})


def specs() -> Specs:
    """Returns specs verifying each level's frequency within 1% of the fitted line."""
    requirements = []
    for (begin, end), volts in zip(windows(), LEVELS, strict=True):
        start = begin + 2e-3 / volts
        expected = SLOPE * volts
        requirements.append(
            frequency("run", "out", start=start * s, stop=end * s).within(
                expected * Hz, expected / 100 * Hz
            )
        )
    return Specs(requirements)
