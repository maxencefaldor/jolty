# Copyright 2026 Maxence Faldor
"""Parity tests confirming a once-elaborated circuit matches fresh elaboration.

Monte Carlo and worst-case runs override numeric parameters on a fixed topology. A
simulation built via
[`Simulation.from_elaborated`][jolty.simulate.Simulation.from_elaborated] from an
[`Elaborated`][jolty.simulate.Elaborated] circuit rebuilds devices from those values,
whereas `Simulation` re-elaborates from scratch. Across pulsed random circuits with
parasitic net capacitance at random overrides and temperatures, both produce bit-for-bit
identical operating points, frequency responses, transients, work counts, and errors.

Only floating-point device parameters are overridden. Integer fields such as MOSFET
level preserve circuit structure under integer samples, whereas float overrides do not.
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
from jolty.units import GHz, H, Hz, K, V, ohm, s
from strategies import SETTINGS, STEP, STOP, circuits, parasitic, pulsed

type Number = tuple[str, str, float]
"""A number of a device instance: the instance's path, the number's, and its value."""


def _numbers(circuit: ir.Circuit) -> list[Number]:
    """Collects every floating-point parameter on top-level instances of `circuit`."""

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
    """Returns the name of the pulsed source, the circuit's first voltage source."""
    return next(
        instance.name
        for instance in circuit.components[circuit.top].instances
        if instance.of == ir.DeviceOf(model="voltage_source")
    )


def _found(simulation: Simulation, source: str) -> list[bytes | str]:
    """Returns raw byte outputs or error strings across operating point, AC, and tran.

    Includes transient step work counts alongside numeric results.
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
                if value.charge_leak is not None:
                    found.append(value.charge_leak.signed.tobytes())
                    found.append(value.charge_leak.absolute.tobytes())
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
    raw_overrides: dict[str, dict[str, float]] = {}
    for (path, name, value), factor in zip(chosen, factors, strict=False):
        raw_overrides.setdefault(path, {})[name] = value * factor
    overrides = devices.wrap_overrides(loaded, raw_overrides)
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
        Simulation.from_elaborated(elaborated, {"R1": {"resistance": r * ohm}})
        for r in (0.5, 0.25)
    ]
    # A resistance below 1 Ω adds an explicit branch-current unknown, so both runs
    # share the circuit elaborated once for that topology.
    assert all(
        "R1:current" in (unknown.name for unknown in each.unknowns) for each in below
    )
    assert len(elaborated._structures) == 1  # noqa: SLF001 - the structures elaborated
    above = Simulation.from_elaborated(elaborated, {"R1": {"resistance": 4 * ohm}})
    assert "R1:current" not in (unknown.name for unknown in above.unknowns)
    point = above.run(OperatingPoint())
    # Linear divider ratio 3 / (4 + 3): LU rounding within ~4500 eps (rel=1e-12).
    assert point.voltage("out") == pytest.approx(3 / 7, rel=1e-12)
    with pytest.raises(ElaborationError, match="names no number among the parameters"):
        Simulation.from_elaborated(elaborated, {"R1": {"inductance": 1 * H}})
    with pytest.raises(ElaborationError, match="resistance is positive"):
        Simulation.from_elaborated(elaborated, {"R1": {"resistance": -1 * ohm}})
