# Copyright 2026 Maxence Faldor
"""The best VCO found: an integrator and a Schmitt trigger, their delays cancelled.

A MOSFET reverses the integrator's current, so its channel, a resistance, only scales
the current. A 2.2 nF capacitor across the Schmitt trigger's 10 kΩ feedback hurries
its regeneration, which cuts its delay from 6-9 µs to about 2 µs, and 2 kΩ in series
with the 1 nF integrating capacitor steps the triangle towards the next threshold by
twice its current times 2 kΩ at each reversal, the voltage it would have travelled in
2 µs: a time advance that cancels the delay at every frequency. Every frequency from
0.5 V to 5 V lies within 0.1 % of 996 Hz/V. Simulate it with
``jolty simulate docs/discovery/vco/design.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Builder, Component, Plan, Role, Specs, component, devices
from jolty.plan import Transient
from jolty.specs import frequency
from jolty.units import Hz, V, s, us
from jolty.waveforms import Piecewise

LEVELS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
"""The control's levels, in volts, each held for eight periods of 1 kHz/V."""

SLOPE = 996.07
"""The slope of the line through the origin that fits the frequencies, in Hz/V."""


def windows() -> list[tuple[float, float]]:
    """When the control steps to each level and steps away from it, in seconds."""
    edges: list[tuple[float, float]] = []
    start = 0.0
    for volts in LEVELS:
        edges.append((start, start + 8e-3 / volts))
        start += 8e-3 / volts
    return edges


@component
def stepped(c: Builder) -> None:
    """The VCO, its control stepping through LEVELS with 1 µs edges."""
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
    """The VCO, its control stepping from 0.5 V to 5 V."""
    return stepped()


def plan() -> Plan:
    """The six levels, 0.1 µs steps at most: 2000 a period at 5 V."""
    return Plan({"run": Transient(stop=windows()[-1][1] * s, max_step=0.1 * us)})


def specs() -> Specs:
    """Each level's frequency, after two periods, within 1 % of the fitted line."""
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
