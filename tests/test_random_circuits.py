# Copyright 2026 Maxence Faldor
"""Random circuits, simulated by jolty and by ngspice, which must agree on every one.

The circuits come from `strategies`, of every device and size, and Hypothesis shrinks
any disagreement to the smallest circuit that shows it. Each test runs the same 40
circuits every time; the environment variable `JOLTY_RANDOM_CIRCUITS` sets how many to
draw afresh instead, for a longer search.
"""

from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
from hypothesis import event, given

from differential import (
    SKIP,
    assert_ac_agrees,
    compared,
    differing,
    nearest,
    netlist,
    ngspice,
    unstable,
    vector,
)
from jolty import (
    Ac,
    JoltyError,
    OperatingPoint,
    Simulation,
    SimulationError,
    Transient,
    ir,
    results,
)
from jolty.plan import TIGHT, decade
from jolty.units import GHz, Hz, s
from strategies import SETTINGS, STEP, STOP, circuits, parasitic, pulsed

pytestmark = SKIP

type Real = npt.NDArray[np.float64]


def _spice_point(
    simulation: Simulation, start: results.OperatingPointResult | None = None
) -> dict[str, float] | None:
    """Returns ngspice's operating point keyed by shared unknown names, or `None`.

    Seeds ngspice from node voltages in `start` when provided; returns `None` if
    ngspice fails to converge.
    """
    circuit = simulation.circuit
    lines = netlist(circuit)
    if start is not None:
        shared = {unknown.name for _, unknown in compared(simulation)}
        guesses = [
            f"{vector(circuit, unknown)}={float(value)!r}"
            for unknown, value in zip(start.unknowns, start.values, strict=True)
            if unknown.nature == "voltage" and unknown.name in shared
        ]
        lines = [*lines, ".nodeset " + " ".join(guesses)]
    try:
        raw = ngspice.run(lines, "op")
    except JoltyError:
        return None
    return {
        unknown.name: float(raw.real(vector(circuit, unknown))[0])
        for _, unknown in compared(simulation)
    }


def _agrees(point: results.OperatingPointResult, spice: dict[str, float]) -> bool:
    """Returns whether jolty's operating point matches ngspice's within tolerance."""
    natures = {unknown.name: unknown.nature for unknown in point.unknowns}
    values = dict(zip(natures, point.values, strict=True))
    return not differing(values, spice, natures)


@SETTINGS
@given(circuits())
def test_operating_points_agree(circuit: ir.Circuit) -> None:
    simulation = Simulation(circuit)
    spice = _spice_point(simulation)
    if spice is None:
        event("ngspice found no operating point")
        return
    try:
        point = simulation.run(OperatingPoint(), TIGHT)
    except SimulationError as error:
        message = f"ngspice found an operating point, jolty none: {error}"
        raise AssertionError(message) from None
    if _agrees(point, spice):
        return
    # A circuit with feedback may rest in several states, and the engines may find
    # different ones: started there, ngspice must find jolty's too.
    event("another operating point")
    again = _spice_point(simulation, start=point)
    assert again is not None, "ngspice finds no operating point from jolty's"
    assert _agrees(point, again), "jolty's operating point is not one of ngspice's"


def _resting(simulation: Simulation) -> results.OperatingPointResult | None:
    """Returns the shared linearization operating point for both engines, or `None`.

    Returns `None` if ngspice fails to converge or settles on a different equilibrium.
    """
    spice = _spice_point(simulation)
    if spice is None:
        event("ngspice found no operating point")
        return None
    point = simulation.run(OperatingPoint(), TIGHT)
    if not _agrees(point, spice):
        event("another operating point")
        return None
    return point


@SETTINGS
@given(circuits())
def test_frequency_responses_agree(circuit: ir.Circuit) -> None:
    simulation = Simulation(circuit)
    if _resting(simulation) is None:
        return
    source = next(
        instance.name
        for instance in circuit.components[circuit.top].instances
        if instance.of == ir.DeviceOf(model="voltage_source")
    )
    response = simulation.run(Ac(source, decade(1 * Hz, 1 * GHz, 5)), TIGHT)
    raw = ngspice.run(netlist(circuit, excited=source), "ac dec 5 1 1g")
    assert_ac_agrees(simulation, response, raw, scale_by_nature=True)


type Waveforms = dict[str, Real]

SHORTENED: Final = (4, 16, 64)
"""The factors by which both engines shorten their longest step where they differ."""

NGSPICE_TIMEOUT: Final = 20
"""Seconds ngspice may run on a random transient before counting as stalled."""


def _jolty(simulation: Simulation, step: float, times: Real) -> Waveforms:
    """Simulates jolty transient waveforms with max step `step`, sampled at `times`."""
    plan = Transient(
        stop=STOP * s, max_step=step * s, breakpoints=[time * s for time in times[1:]]
    )
    waveforms = simulation.run(plan, TIGHT)
    taken = nearest(waveforms.times, times)
    # Breakpoint landing tolerance covering IEEE-754 minimum-step merging (`16 * eps`).
    assert np.allclose(waveforms.times[taken], times, rtol=0, atol=1e-9 * step)
    return {
        unknown.name: waveforms.values[taken, column]
        for column, unknown in compared(simulation)
    }


def _transients(
    simulation: Simulation, step: float
) -> tuple[Waveforms, Waveforms, Real] | None:
    """Runs both engines at ngspice's time points with max step `step`, or `None`.

    jolty forces breakpoints at every ngspice time point so both waveforms compare at
    solved steps rather than interpolated bends. Returns `None` if ngspice stops early.
    """
    circuit = simulation.circuit
    try:
        raw = ngspice.run(
            netlist(circuit, transient=True),
            f"tran {step!r} {STOP!r} 0 {step!r}",
            timeout=NGSPICE_TIMEOUT,
        )
    except JoltyError:
        return None
    times = raw.real("time")
    spice = {
        unknown.name: raw.real(vector(circuit, unknown))
        for _, unknown in compared(simulation)
    }
    return _jolty(simulation, step, times), spice, times


def _excused(
    simulation: Simulation,
    jolty: Waveforms,
    times: Real,
    natures: Mapping[str, str],
) -> str | None:
    """Explains any excused transient mismatch between jolty and ngspice, or `None`.

    Shrinking max step by 4x, 16x, and 64x cuts second-order error sixteenfold per
    stage to check convergence. Waveforms that still shift under tighter jolty steps
    (such as chaotic runs) have not settled and provide no valid reference.
    """
    for divisor in SHORTENED:
        step = STEP / divisor
        try:
            shorter = _transients(simulation, step)
        except SimulationError:
            return _stopped_short(simulation.circuit, step)
        if shorter is None:
            return "ngspice stopped short"
        if not differing(*shorter[:2], natures, 1e-3):
            return "agree once both take shorter steps"
    step = STEP / SHORTENED[-1]
    try:
        finest = _jolty(simulation, step, times)
    except SimulationError:
        return _stopped_short(simulation.circuit, step)
    if differing(finest, jolty, natures, 1e-3):
        return "a waveform that shorter steps move"
    return None


def _stopped_short(circuit: ir.Circuit, step: float) -> str:
    """Checks whether a transient fold explains why jolty stopped before ngspice.

    Without capacitance, resistive loops can hit a fold singularity past which no
    solution continues; ngspice may jump past the fold or halt at any step size. Adding
    10 fF from every net to ground restores continuation, so jolty must finish whenever
    ngspice does at the same step.

    jolty runs first with free step sizes: if jolty finishes, the solution continues
    regardless of ngspice. Only when jolty also stops early is ngspice queried,
    avoiding minute-long ngspice stalls on ill-conditioned circuits at tight limits.
    """
    charged = Simulation(parasitic(circuit))
    try:
        charged.run(Transient(stop=STOP * s, max_step=step * s), TIGHT)
    except SimulationError:
        try:
            found = _transients(charged, step)
        except SimulationError as error:
            message = f"jolty stopped short where ngspice finished: {error}"
            raise AssertionError(message) from None
        if found is None:
            return "neither engine finishes, without capacitance or with it"
    return "no solution continues without capacitance"


@SETTINGS
@given(circuits())
def test_transients_agree(circuit: ir.Circuit) -> None:
    simulation = Simulation(pulsed(circuit))
    if _resting(simulation) is None:
        return
    if unstable(simulation):
        # A mode that grows away from rest makes no waveform to agree on.
        event("an unstable operating point")
        return
    natures = {unknown.name: unknown.nature for _, unknown in compared(simulation)}
    try:
        found = _transients(simulation, STEP)
    except SimulationError:
        event(_stopped_short(simulation.circuit, STEP))
        return
    if found is None:
        event("ngspice stopped short")
        return
    jolty, spice, times = found
    if differing(jolty, spice, natures, 1e-3):
        excuse = _excused(simulation, jolty, times, natures)
        assert excuse is not None, "the waveforms differ"
        event(excuse)
