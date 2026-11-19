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
    """Imports example module `name` using the examples import path."""
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(EXAMPLES))


def _steady_ring() -> Component:
    """Builds the example ring oscillator without its startup kick pulse.

    A transient from the balanced operating point remains stuck there, whereas periodic
    steady-state analysis excites the fastest-growing unstable mode.
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
    """Computes linearly interpolated rising crossings of `net` through `level`."""
    times, values = waveforms.times, waveforms.voltage(net)
    below = np.flatnonzero((values[:-1] <= level) & (values[1:] > level))
    fraction = (level - values[below]) / (values[below + 1] - values[below])
    return times[below] + fraction * (times[below + 1] - times[below])


def _periods(relative_tolerance: float, longest: float) -> tuple[float, float, float]:
    """Returns ring periods from shooting and transient runs plus transient scatter.

    Time steps cap at `longest` seconds. The kicked transient settles for 17 cycles,
    averages the last 20 cycles, and reports peak-to-peak scatter divided by that mean.
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
    """Verifies shooting period matches a settled transient within its error bound.

    At each setting, transient error combines deviation from the tight reference
    (`reltol=1e-9`, max step 1 ps) with period scatter; at the reference itself,
    scatter alone applies. Scatter alone understates bias: at 1e-7, periodic step
    locking keeps scatter near 1e-11 while the mean sits 2e-5 off the reference.
    """
    found, reference, scatter = _periods(1e-9, 1e-12)
    assert abs(found / reference - 1) <= scatter
    for relative_tolerance in (1e-3, 1e-5, 1e-7):
        steady, mean, spread = _periods(relative_tolerance, 5e-12)
        error = abs(mean / reference - 1) + spread
        assert abs(steady / reference - 1) <= error, relative_tolerance


def test_an_oscillators_period_moves_with_its_tank_as_it_scales() -> None:
    """Verifies tank period scaling `L ∂T/∂L + C ∂T/∂C = T` for an autonomous LC tank.

    Active conductance supplies energy inside the 1 V radius (measured in voltage and
    scaled current) and drains it outside, keeping equations invariant when time, L,
    and C scale together. Derivatives satisfy the scaling law to Newton convergence:
    7e-10 at default `reltol=1e-3` and 1.8e-13 at `1e-6`, asserted within 1e-12.
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
