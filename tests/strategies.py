# Copyright 2026 Maxence Faldor
"""Random circuits drawn by Hypothesis, for the tests that simulate them.

Each circuit joins every node to the ground through a tree of resistors, so it always
has a DC path, then places sources and devices at random: capacitors, inductors in
series with a resistor, so that no loop of inductors and voltage sources forms, diodes,
bipolar and MOS transistors with series resistances that bound their currents, each
MOSFET's body on its source, and transconductances. Sizes run past the 32 unknowns
where hand-written circuits stop.
"""

import os
from collections import Counter
from typing import Final, Literal

import numpy as np
from hypothesis import HealthCheck, assume, settings
from hypothesis import strategies as st

from jolty import Simulation, devices, ir
from jolty.settings import Level
from jolty.units import A, F, H, S, V, m, ohm, us
from jolty.waveforms import Pulse

__all__ = ["PARASITIC", "SETTINGS", "STEP", "STOP", "circuits", "parasitic", "pulsed"]

SEARCH: Final = os.environ.get("JOLTY_RANDOM_CIRCUITS")
"""How many circuits each test draws afresh, if a search is asked for."""

SETTINGS: Final = settings(
    max_examples=40 if SEARCH is None else int(SEARCH),
    derandomize=SEARCH is None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
"""How each test of random circuits runs: on the same 40 circuits every time, or on as
many as `JOLTY_RANDOM_CIRCUITS` sets, drawn afresh."""


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
        drain_resistance=draw(_decades(1, 100)) * ohm,
        source_resistance=draw(_decades(1, 100)) * ohm,
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
            if prefix == "M":
                # Its body on its source, as a discrete MOSFET's is, so that no junction
                # to it lies across a source with nothing to bound its current.
                ends[3] = ends[2]
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


STOP: Final = 2e-6
"""How long each random transient runs, in seconds: a pulse of 1 V on the first source,
rising after 0.2 µs, for 0.8 µs."""

STEP: Final = STOP / 200
"""The longest step of each random transient."""

PARASITIC: Final = 10e-15
"""The capacitance every net holds to the ground where a transient asks for it, in
farads."""


def pulsed(circuit: ir.Circuit) -> ir.Circuit:
    """The circuit with its first voltage source pulsed 1 V above its level and back."""
    instances = list(circuit.components[circuit.top].instances)
    index = next(
        index
        for index, instance in enumerate(instances)
        if instance.of == ir.DeviceOf(device="voltage_source")
    )
    waveform = instances[index].parameters["waveform"]
    assert isinstance(waveform, dict)
    level = waveform["dc"]
    assert isinstance(level, float)
    pulse = Pulse(
        initial=level * V,
        pulsed=(level + 1) * V,
        delay=0.2 * us,
        rise=0.1 * us,
        fall=0.1 * us,
        width=0.8 * us,
    )
    instances[index] = instances[index].model_copy(
        update={"parameters": devices.voltage_source(pulse).parameters}
    )
    return _placing(circuit, instances)


def parasitic(circuit: ir.Circuit) -> ir.Circuit:
    """The circuit with `PARASITIC` from every net to the ground, as real nets have."""
    instances = circuit.components[circuit.top].instances
    nets = sorted(
        {net for instance in instances for net in instance.terminals.values()} - {"0"}
    )
    capacitor = devices.capacitor(PARASITIC * F)
    return _placing(
        circuit,
        [
            *instances,
            *(
                ir.Instance(
                    id=f"C{net}",
                    of=ir.DeviceOf(device=capacitor.model),
                    parameters=capacitor.parameters,
                    terminals={"a": net, "b": "0"},
                )
                for net in nets
            ),
        ],
    )


def _placing(circuit: ir.Circuit, instances: list[ir.Instance]) -> ir.Circuit:
    """The circuit with its top placing `instances` instead of its own."""
    top = circuit.components[circuit.top].model_copy(
        update={"instances": tuple(instances)}
    )
    return circuit.model_copy(update={"components": {circuit.top: top}})
