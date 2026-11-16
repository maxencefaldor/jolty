# Copyright 2026 Maxence Faldor
"""Oscillators' periodic steady states against long transients."""

import importlib
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np

from jolty import (
    Builder,
    Component,
    Options,
    PeriodicSteadyState,
    Role,
    Sensitivity,
    Simulation,
    Transient,
    component,
    devices,
    results,
)
from jolty.units import F, H, V, ns, ps

EXAMPLES: Final = Path(__file__).parent.parent / "examples"
"""The examples, which import each other's modules from their folder."""


def _example(name: str) -> ModuleType:
    """The example `name`, imported as the examples import each other."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(EXAMPLES))


def _steady_ring() -> Component:
    """The examples' ring oscillator on its supply, without the pulse that kicks it.

    A transient from its balanced operating point stays there; a periodic steady state
    starts it along its fastest-growing mode.
    """
    common, example = _example("common"), _example("ring_oscillator")

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(common.SUPPLY * V), name="VDD")
        oscillator = c.place(example.ring(), name="X1")
        c.connect(supply.positive, oscillator.vdd, name="vdd")
        c.connect(oscillator.output, name="out")
        c.connect(supply.negative, oscillator.gnd, ground)

    return bench()


def _rising(waveforms: results.TransientResult, net: str, level: float) -> np.ndarray:
    """The times `net` rises through `level`, interpolated linearly between steps."""
    times, values = waveforms.times, waveforms.voltage(net)
    below = np.flatnonzero((values[:-1] <= level) & (values[1:] > level))
    fraction = (level - values[below]) / (values[below + 1] - values[below])
    return times[below] + fraction * (times[below + 1] - times[below])


def _periods(relative_tolerance: float, longest: float) -> tuple[float, float, float]:
    """The ring's period by shooting and by a long transient, its scatter beside.

    The steps are at most `longest` seconds. The transient is the example's, kicked;
    its period is the mean of its last 20, after 17 to settle, and its scatter how far
    the longest and the shortest of them lie apart, a fraction of that mean.
    """
    options = Options(relative_tolerance=relative_tolerance)
    longest_step = longest * 1e12 * ps
    steady = Simulation(_steady_ring()).run(
        PeriodicSteadyState(max_step=longest_step), options
    )
    long = Simulation(_example("ring_oscillator").bench()).run(
        Transient(40 * ns, max_step=longest_step), options
    )
    supply = _example("common").SUPPLY
    settled = np.diff(_rising(long, "out", supply / 2)[-21:])
    mean = float(np.mean(settled))
    return steady.period, mean, float(np.ptp(settled)) / mean


def test_a_ring_oscillator_runs_at_a_long_transients_settled_period() -> None:
    """Shooting finds the period a long transient settles into, within its error.

    The transient's error at each setting is its settled period's distance from the
    same transient's at a relative tolerance of 1e-9 and steps of at most 1 ps, the
    reference, together with the scatter of its own periods; at the reference's
    setting, its scatter alone. The scatter alone does not bound it: at 1e-7, the
    steps fall into a pattern each period repeats, and its periods agree to 1e-11
    while their mean lies 2e-5 off the reference.
    """
    found, reference, scatter = _periods(1e-9, 1e-12)
    assert abs(found / reference - 1) <= scatter
    for relative_tolerance in (1e-3, 1e-5, 1e-7):
        steady, mean, spread = _periods(relative_tolerance, 5e-12)
        error = abs(mean / reference - 1) + spread
        assert abs(steady / reference - 1) <= error, relative_tolerance


def test_an_oscillators_period_moves_with_its_tank_as_it_scales() -> None:
    """A tank's period scales with time, L and C alike: L ∂T/∂L + C ∂T/∂C = T.

    Its conductance feeds it inside the circle of radius 1 V, in volts and its impedance
    times its current, and drains it outside, so the equations, with that impedance
    held, are unchanged when time, L and C scale alike. The derivatives meet it as far
    as Newton's iterates meet each step's equations: to 7e-10 at the default relative
    tolerance, 1e-3, and to 1.8e-13 at 1e-6, held to 1e-12.
    """
    inductance, capacitance = 1e-6, 1e-9
    impedance = math.sqrt(inductance / capacitance)
    expression = (
        f"-{0.2 / impedance!r} * v(x) * (1 - (v(x) ^ 2 + ({impedance!r} * i(p)) ^ 2))"
    )

    @component
    def tank(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
        inductor = c.place(devices.inductor(inductance * H), name="L1")
        regulation = c.place(devices.behavioral_current_source(expression), name="B1")
        c.connect(capacitor.a, inductor.a, regulation.positive, regulation.x, name="t")
        c.connect(inductor.b, regulation.p_positive, name="l")
        c.connect(capacitor.b, regulation.negative, regulation.p_negative, ground)

    found = Simulation(tank()).run(
        Sensitivity(["t"], of=PeriodicSteadyState()),
        Options(relative_tolerance=1e-6),
    )
    assert isinstance(found.result, results.PeriodicSteadyStateResult)
    scaled = inductance * found.period_derivative(
        "L1", "inductance"
    ) + capacitance * found.period_derivative("C1", "capacitance")
    assert abs(scaled / found.result.period - 1) <= 1e-12
