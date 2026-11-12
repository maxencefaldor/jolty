# Copyright 2026 Maxence Faldor
"""Random circuits, simulated by jolty and by ngspice, which must agree on every one.

Each circuit joins every node to the ground through a tree of resistors, so it always
has a DC path, then places sources and devices at random: capacitors, inductors in
series with a resistor, so that no loop of inductors and voltage sources forms, diodes,
bipolar and MOS transistors with series resistances that bound their currents, and
transconductances. Sizes run past the 32 unknowns where hand-written circuits stop, and
Hypothesis shrinks any disagreement to the smallest circuit that shows it.

The number of circuits defaults to 40; the environment variable
`JOLTY_RANDOM_CIRCUITS` sets another, for a longer run.
"""

import os
from collections import Counter
from typing import Final, Literal

import numpy as np
from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from differential import SKIP, TIGHT, compared, netlist, ngspice, vector
from jolty import (
    JoltyError,
    OperatingPoint,
    Simulation,
    SimulationError,
    devices,
    ir,
    results,
)
from jolty.settings import Level
from jolty.spice.export import CURRENT_TOLERANCE, VOLTAGE_TOLERANCE
from jolty.units import A, F, H, S, V, m, ohm

pytestmark = SKIP

CIRCUITS: Final = int(os.environ.get("JOLTY_RANDOM_CIRCUITS", "40"))
"""How many random circuits each test simulates."""


def _decades(low: float, high: float) -> st.SearchStrategy[float]:
    """Values spread evenly over the decades from `low` to `high`."""
    return st.floats(np.log10(low), np.log10(high)).map(lambda exponent: 10**exponent)


def _resistor(draw: st.DrawFn, low: float = 10, high: float = 1e6) -> devices.Device:
    return devices.resistor(draw(_decades(low, high)) * ohm)


def _capacitor(draw: st.DrawFn) -> devices.Device:
    return devices.capacitor(draw(_decades(1e-12, 1e-6)) * F)


def _current(draw: st.DrawFn) -> devices.Device:
    return devices.current_source(draw(_decades(1e-6, 1e-3)) * A)


def _diode(draw: st.DrawFn) -> devices.Device:
    return devices.diode(
        saturation_current=draw(_decades(1e-16, 1e-12)) * A,
        series_resistance=draw(_decades(1, 100)) * ohm,
    )


POLARITIES: Final[tuple[Literal["npn", "pnp"], ...]] = ("npn", "pnp")


def _bipolar(draw: st.DrawFn) -> devices.Device:
    return devices.bipolar_transistor(
        polarity=draw(st.sampled_from(POLARITIES)),
        saturation_current=draw(_decades(1e-16, 1e-14)) * A,
        forward_beta=draw(st.floats(20, 400)),
        base_resistance=draw(_decades(10, 1e3)) * ohm,
        emitter_resistance=draw(_decades(1, 100)) * ohm,
        collector_resistance=draw(_decades(1, 100)) * ohm,
    )


def _mosfet(draw: st.DrawFn) -> devices.Device:
    channel: Literal["n", "p"] = draw(st.sampled_from(["n", "p"]))
    threshold = draw(st.floats(0.3, 1.5)) * (V if channel == "n" else -V)
    return devices.mosfet(
        level=1,
        channel=channel,
        width=draw(_decades(1e-6, 1e-4)) * m,
        length=draw(_decades(1e-6, 1e-5)) * m,
        threshold_voltage=threshold,
    )


def _transconductance(draw: st.DrawFn) -> devices.Device:
    return devices.voltage_controlled_current_source(draw(_decades(1e-6, 1e-3)) * S)


DEVICES: Final = {
    "R": _resistor,
    "C": _capacitor,
    "I": _current,
    "D": _diode,
    "Q": _bipolar,
    "M": _mosfet,
    "G": _transconductance,
}
"""What each device placed at random between existing nodes is, by its prefix; an
inductor goes in series with a resistor besides."""


@st.composite
def circuits(draw: st.DrawFn) -> ir.Circuit:
    """A random circuit with a DC path from every node to the ground."""
    count = draw(st.integers(1, 40))
    nodes = ["0", *(f"n{index}" for index in range(1, count + 1))]
    instances: list[ir.Instance] = []

    def place(prefix: str, device: devices.Device, nets: list[str]) -> None:
        terminals = dict(zip(device.terminals, nets, strict=False))
        instances.append(
            ir.Instance(
                id=f"{prefix}{len(instances)}",
                of=ir.DeviceOf(device=device.model),
                parameters=device.parameters,
                terminals=terminals,
            )
        )

    for index in range(1, count + 1):
        parent = draw(st.integers(0, index - 1))
        place("R", _resistor(draw), [nodes[index], nodes[parent]])
    driven = draw(st.sets(st.integers(1, count), min_size=1, max_size=min(3, count)))
    for node in sorted(driven):
        level = draw(st.floats(-5, 5)) * V
        place("V", devices.voltage_source(level), [nodes[node], "0"])
    for _ in range(draw(st.integers(0, 2 * count))):
        # Two distinct nodes first, so that no source drives its current through
        # nothing, then any two, for a transistor's or a control's other terminals.
        node = st.sampled_from(nodes)
        ends = [
            *draw(st.lists(node, min_size=2, max_size=2, unique=True)),
            *draw(st.lists(node, min_size=2, max_size=2)),
        ]
        prefix = draw(st.sampled_from([*DEVICES, "L"]))
        if prefix == "L":
            middle = f"m{len(instances)}"
            inductance = draw(_decades(1e-9, 1e-3)) * H
            place("L", devices.inductor(inductance), [ends[0], middle])
            place("R", _resistor(draw, 1, 1e3), [middle, ends[1]])
        else:
            place(prefix, DEVICES[prefix](draw), ends)
    # A node that only the tree's resistor reaches would dangle: tie it to another.
    joined = Counter(
        net for instance in instances for net in instance.terminals.values()
    )
    for net in nodes[1:]:
        if joined[net] == 1:
            other = draw(st.sampled_from([node for node in nodes if node != net]))
            place("R", _resistor(draw), [net, other])
    circuit = ir.Circuit(
        top="random",
        components={
            "random": ir.Component(
                ports=(ir.Port(name="0", role=ir.Role.GROUND),),
                instances=tuple(instances),
            )
        },
    )
    simulation = Simulation(circuit)
    assume(all(found.level is not Level.ERROR for found in simulation.check()))
    return circuit


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
        raw = ngspice(lines, "op")
    except JoltyError:
        return None
    return {
        unknown.name: float(raw.real(vector(circuit, unknown))[0])
        for _, unknown in compared(simulation)
    }


def _agrees(point: results.OperatingPoint, spice: dict[str, float]) -> bool:
    """Whether jolty's operating point is ngspice's.

    Each unknown agrees within a millionth of the largest of its quantity in the
    circuit: both engines resolve an equation to a millionth of its largest term, so
    a small current beside large ones only that far.
    """
    quantities = {unknown.name: unknown.quantity for unknown in point.unknowns}
    scales = dict.fromkeys(("voltage", "current"), 0.0)
    for name, value in spice.items():
        scales[quantities[name]] = max(scales[quantities[name]], abs(value))
    return all(
        abs(value - spice[unknown.name])
        <= 1e-6 * scales[unknown.quantity] + 2 * _noise(unknown)
        for unknown, value in zip(point.unknowns, point.values, strict=True)
        if unknown.name in spice
    )


def _noise(unknown: results.Unknown) -> float:
    """The engines' absolute tolerance on an unknown of its quantity."""
    return CURRENT_TOLERANCE if unknown.quantity == "current" else VOLTAGE_TOLERANCE


SETTINGS: Final = settings(
    max_examples=CIRCUITS,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


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
