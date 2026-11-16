# Copyright 2026 Maxence Faldor
"""A circuit elaborated once simulates at any numbers as one elaborated with them.

A Monte Carlo sample, or a worst case's run, replaces some of a circuit's numbers. A
simulation made by
[`Simulation.from_elaborated`][jolty.simulate.Simulation.from_elaborated] from the
circuit [`Elaborated`][jolty.simulate.Elaborated] once makes its devices again from
them, where a `Simulation` elaborates the circuit with them. On the random circuits of
`strategies`, pulsed and holding capacitance at their nets, at random numbers and
temperatures, both find the same operating point, frequency response and transient, bit
for bit, their transients' work alike, or the same problems.

The numbers replaced are those the circuits write as real numbers: an integer, such as a
MOSFET's level, keeps the circuit's own form when a sample replaces it with a whole
number, which an override written as a real number does not.
"""

import dataclasses
from collections.abc import Iterator

import numpy as np
import pytest
from hypothesis import event, given
from hypothesis import strategies as st
from pydantic import JsonValue

from jolty import (
    Ac,
    Builder,
    OperatingPoint,
    Role,
    Simulation,
    SimulationError,
    Transient,
    component,
    devices,
    ir,
    results,
)
from jolty.errors import ElaborationError
from jolty.plan import TIGHT, decade
from jolty.simulate import Elaborated
from jolty.units import GHz, Hz, K, V, ohm, s
from strategies import SETTINGS, STEP, STOP, circuits, parasitic, pulsed

type Number = tuple[str, str, float]
"""A number of a device instance: the instance's path, the number's, and its value."""


def _numbers(circuit: ir.Circuit) -> list[Number]:
    """Every real number the instances of `circuit`'s top write, by path."""

    def within(path: str, value: JsonValue) -> Iterator[tuple[str, float]]:
        match value:
            case float():
                yield path, value
            case dict():
                for key, inner in value.items():
                    yield from within(f"{path}.{key}", inner)
            case list():
                for index, inner in enumerate(value):
                    yield from within(f"{path}.{index}", inner)
            case _:
                pass

    return [
        (instance.name, name, number)
        for instance in circuit.components[circuit.top].instances
        for key, value in instance.parameters.items()
        for name, number in within(key, value)
    ]


def _source(circuit: ir.Circuit) -> str:
    """The pulsed source, the first voltage source the circuit places."""
    return next(
        instance.name
        for instance in circuit.components[circuit.top].instances
        if instance.of == ir.DeviceOf(model="voltage_source")
    )


def _found(simulation: Simulation, source: str) -> list[bytes | str]:
    """Every number the simulation finds, as its bytes, or each analysis's error.

    The analyses are the operating point, the frequency response to `source` and the
    transient, whose work counts too.
    """
    found: list[bytes | str] = []
    analyses = (
        OperatingPoint(),
        Ac(source, decade(1 * Hz, 1 * GHz, 2)),
        Transient(stop=STOP * s, max_step=STEP * s),
    )
    for analysis in analyses:
        try:
            result = simulation.run(analysis, TIGHT)
        except SimulationError as error:
            found.append(str(error))
            continue
        assert dataclasses.is_dataclass(result)
        for field in dataclasses.fields(result):
            value = getattr(result, field.name)
            if isinstance(value, np.ndarray):
                found.append(value.tobytes())
            elif isinstance(value, results.Statistics):
                found.append(repr(value.work))
    return found


@SETTINGS
@given(circuits(), st.data())
def test_numbers_replaced_simulate_the_same_on_a_shared_circuit(
    circuit: ir.Circuit, data: st.DataObject
) -> None:
    loaded = parasitic(pulsed(circuit))
    numbers = _numbers(loaded)
    chosen = data.draw(st.lists(st.sampled_from(numbers), unique=True))
    factors = data.draw(st.lists(st.floats(0.5, 2), min_size=len(chosen)))
    overrides: dict[str, dict[str, float]] = {}
    for (path, name, value), factor in zip(chosen, factors, strict=False):
        overrides.setdefault(path, {})[name] = value * factor
    temperature = data.draw(st.floats(250, 400)) * K
    problems: list[str] = []
    for elaborate in (
        lambda: Simulation(loaded, temperature, overrides),
        lambda: Simulation.from_elaborated(Elaborated(loaded, temperature), overrides),
    ):
        try:
            elaborate()
        except ElaborationError as error:
            problems.append(str(error))
    if problems:
        event("numbers out of range")
        assert len(problems) == 2, problems
        assert problems[0] == problems[1]
        return
    direct = Simulation(loaded, temperature, overrides)
    shared = Simulation.from_elaborated(Elaborated(loaded, temperature), overrides)
    assert shared.unknowns == direct.unknowns
    source = _source(loaded)
    assert _found(shared, source) == _found(direct, source)


def test_one_elaboration_serves_each_structure() -> None:
    @component
    def divider(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(1 * V), name="V1")
        upper = c.place(devices.resistor(2 * ohm), name="R1")
        lower = c.place(devices.resistor(3 * ohm), name="R2")
        c.connect(source.positive, upper.a)
        c.connect(upper.b, lower.a, name="out")
        c.connect(lower.b, source.negative, ground)

    elaborated = Elaborated(divider())
    below = [
        Simulation.from_elaborated(elaborated, {"R1": {"resistance": r}})
        for r in (0.5, 0.25)
    ]
    # A resistance below 1 Ω carries its current as an unknown of its own, so both run
    # on the circuit elaborated once for that structure.
    assert all(
        "R1:current" in (unknown.name for unknown in each.unknowns) for each in below
    )
    assert len(elaborated._structures) == 1  # noqa: SLF001 - the structures elaborated
    above = Simulation.from_elaborated(elaborated, {"R1": {"resistance": 4.0}})
    assert "R1:current" not in (unknown.name for unknown in above.unknowns)
    point = above.run(OperatingPoint())
    assert point.voltage("out") == pytest.approx(3 / 7, rel=1e-12)
    with pytest.raises(ElaborationError, match="names no number among the parameters"):
        Simulation.from_elaborated(elaborated, {"R1": {"inductance": 1.0}})
    with pytest.raises(ElaborationError, match="resistance must be positive"):
        Simulation.from_elaborated(elaborated, {"R1": {"resistance": -1.0}})
