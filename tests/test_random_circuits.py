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
    simulation: Simulation, start: results.OperatingPoint | None = None
) -> dict[str, float] | None:
    """The operating point ngspice finds, by the name of each unknown both report.

    ngspice searches from the node voltages of `start`, if given; there is none when
    it finds none.
    """
    circuit = simulation.circuit
    lines = netlist(circuit)
    if start is not None:
        shared = {unknown.name for _, unknown in compared(simulation)}
        guesses = [
            f"{vector(circuit, unknown)}={float(value)!r}"
            for unknown, value in zip(start.unknowns, start.values, strict=True)
            if unknown.quantity == "voltage" and unknown.name in shared
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


def _agrees(point: results.OperatingPoint, spice: dict[str, float]) -> bool:
    """Whether jolty's operating point is ngspice's."""
    quantities = {unknown.name: unknown.quantity for unknown in point.unknowns}
    values = dict(zip(quantities, point.values, strict=True))
    return not differing(values, spice, quantities)


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


def _resting(simulation: Simulation) -> results.OperatingPoint | None:
    """The operating point both engines find, around which both linearize the circuit.

    There is none when ngspice finds none, or rests in another state than jolty.
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
        instance.id
        for instance in circuit.components[circuit.top].instances
        if instance.of == ir.DeviceOf(device="voltage_source")
    )
    response = simulation.run(Ac(source, decade(1 * Hz, 1 * GHz, 5)), TIGHT)
    raw = ngspice.run(netlist(circuit, excited=source), "ac dec 5 1 1g")
    shared = compared(simulation)
    jolty = {unknown.name: response.values[:, column] for column, unknown in shared}
    spice = {
        unknown.name: raw.vectors[vector(circuit, unknown)] for _, unknown in shared
    }
    quantities = {unknown.name: unknown.quantity for _, unknown in shared}
    assert not differing(jolty, spice, quantities), "the responses differ"


type Waveforms = dict[str, Real]

SHORTENED: Final = (4, 16, 64)
"""The factors by which both engines shorten their longest step where they differ."""


def _jolty(simulation: Simulation, step: float, times: Real) -> Waveforms:
    """The waveforms jolty finds with `step` the longest, read at `times`."""
    plan = Transient(
        stop=STOP * s, max_step=step * s, times=[time * s for time in times[1:]]
    )
    waveforms = simulation.run(plan, TIGHT)
    taken = nearest(waveforms.times, times)
    assert np.allclose(waveforms.times[taken], times, rtol=0, atol=1e-9 * step)
    return {
        unknown.name: waveforms.values[taken, column]
        for column, unknown in compared(simulation)
    }


def _transients(
    simulation: Simulation, step: float
) -> tuple[Waveforms, Waveforms, Real] | None:
    """The waveforms both engines find with `step` the longest, at ngspice's times.

    jolty steps onto every time ngspice took, so that both are read where each solved,
    not between their points, where a waveform may bend. There are none when ngspice
    stops short.
    """
    circuit = simulation.circuit
    try:
        raw = ngspice.run(
            netlist(circuit, transient=True), f"tran {step!r} {STOP!r} 0 {step!r}"
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
    quantities: Mapping[str, str],
) -> str | None:
    """Why the engines' waveforms at `times`, jolty's being `jolty`, may differ, if so.

    Both take steps a quarter as long, then a sixteenth and a sixty-fourth, each
    shrinking an error of the second order sixteenfold: they agree then if one had not
    settled. A waveform that
    jolty's own shorter steps still move, as a chaotic one's, has not settled at these
    steps, and is no reference for either engine.
    """
    for divisor in SHORTENED:
        step = STEP / divisor
        try:
            shorter = _transients(simulation, step)
        except SimulationError:
            return _stopped_short(simulation.circuit, step)
        if shorter is None:
            return "ngspice stopped short"
        if not differing(*shorter[:2], quantities, 1e-3):
            return "agree once both take shorter steps"
    step = STEP / SHORTENED[-1]
    try:
        finest = _jolty(simulation, step, times)
    except SimulationError:
        return _stopped_short(simulation.circuit, step)
    if differing(finest, jolty, quantities, 1e-3):
        return "a waveform that shorter steps move"
    return None


def _stopped_short(circuit: ir.Circuit, step: float) -> str:
    """Why jolty may stop short of ngspice with `step` the longest: none continues.

    Without capacitance, the equations of a net that only resistive devices hold can
    fold, as their solution moves, into a point past which none continues: ngspice
    may leap past it, or stop, whatever the step. With 10 fF from every net to the
    ground, as real nets have, the solution continues, and jolty must finish wherever
    ngspice does, at the same step.

    jolty runs first, on steps of its own: when it finishes, so does the solution,
    whatever ngspice does. Only when it stops short too is ngspice asked, which on
    such circuits can crawl for minutes at these tolerances.
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
    quantities = {unknown.name: unknown.quantity for _, unknown in compared(simulation)}
    try:
        found = _transients(simulation, STEP)
    except SimulationError:
        event(_stopped_short(simulation.circuit, STEP))
        return
    if found is None:
        event("ngspice stopped short")
        return
    jolty, spice, times = found
    if differing(jolty, spice, quantities, 1e-3):
        excuse = _excused(simulation, jolty, times, quantities)
        assert excuse is not None, "the waveforms differ"
        event(excuse)
