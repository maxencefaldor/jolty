# Copyright 2026 Maxence Faldor
"""Random circuits drawn by Hypothesis, for the tests that simulate them.

Each circuit joins every node to the ground through a tree of resistors, so it always
has a DC path, then places sources and devices at random: capacitors, inductors and
pairs of coupled windings each in series with a resistor, so that no loop of inductors
and voltage sources forms, diodes, bipolar and MOS transistors with series resistances
that bound their currents, each MOSFET's body on its source, some of them at
temperatures of their own, transconductances, and current-controlled switches whose
control port is in series with a resistor too. Sizes run past the 32 unknowns where
hand-written circuits stop.
"""

import os
from collections import Counter
from collections.abc import Iterable
from typing import Final, Literal

import numpy as np
from hypothesis import HealthCheck, assume, settings
from hypothesis import strategies as st

from jolty import Simulation, devices, ir
from jolty.settings import Level
from jolty.units import A, F, H, Hz, K, S, V, m, ohm, s, uC, us
from jolty.waveforms import Pulse

__all__ = [
    "PARASITIC",
    "SETTINGS",
    "STEP",
    "STOP",
    "capacitive",
    "circuits",
    "parasitic",
    "placing",
    "pulsed",
]

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


def _apart(draw: st.DrawFn, device: devices.Device) -> devices.Device:
    """`device`, or the same at a temperature of its own, offset from the circuit's."""
    if draw(st.booleans()):
        return device
    return device.warmer_by(draw(st.floats(-40, 80)) * K)


def _diode(draw: st.DrawFn) -> devices.Device:
    return _apart(
        draw,
        devices.diode(
            saturation_current=draw(_decades(1e-16, 1e-12)) * A,
            series_resistance=draw(_decades(1, 100)) * ohm,
        ),
    )


POLARITIES: Final[tuple[Literal["npn", "pnp"], ...]] = ("npn", "pnp")


def _bipolar(draw: st.DrawFn) -> devices.Device:
    return _apart(
        draw,
        devices.bipolar_transistor(
            polarity=draw(st.sampled_from(POLARITIES)),
            saturation_current=draw(_decades(1e-16, 1e-14)) * A,
            forward_beta=draw(st.floats(20, 400)),
            base_resistance=draw(_decades(10, 1e3)) * ohm,
            emitter_resistance=draw(_decades(1, 100)) * ohm,
            collector_resistance=draw(_decades(1, 100)) * ohm,
        ),
    )


def _mosfet(draw: st.DrawFn) -> devices.Device:
    channel: Literal["n", "p"] = draw(st.sampled_from(["n", "p"]))
    threshold = draw(st.floats(0.3, 1.5)) * (V if channel == "n" else -V)
    return _apart(
        draw,
        devices.mosfet(
            level=1,
            channel=channel,
            width=draw(_decades(1e-6, 1e-4)) * m,
            length=draw(_decades(1e-6, 1e-5)) * m,
            threshold_voltage=threshold,
            drain_resistance=draw(_decades(1, 100)) * ohm,
            source_resistance=draw(_decades(1, 100)) * ohm,
        ),
    )


def _transconductance(draw: st.DrawFn) -> devices.Device:
    return devices.voltage_controlled_current_source(draw(_decades(1e-6, 1e-3)) * S)


def _coupled(draw: st.DrawFn) -> devices.Device:
    windings = [devices.inductor(draw(_decades(1e-9, 1e-3)) * H) for _ in range(2)]
    return devices.coupled_inductors(windings, {(1, 2): draw(st.floats(-0.99, 0.99))})


def _switch(draw: st.DrawFn) -> devices.Device:
    on = draw(_decades(1e-6, 1e-3))
    return devices.current_controlled_switch(
        on_resistance=draw(_decades(1, 1e3)) * ohm,
        off_resistance=draw(_decades(1e5, 1e7)) * ohm,
        on_current=on * A,
        off_current=draw(st.floats(-1, 0.5)) * on * A,
    )


def _voltage_switch(draw: st.DrawFn) -> devices.Device:
    off = draw(st.floats(-5, 5))
    return devices.voltage_controlled_switch(
        on_resistance=draw(_decades(1, 1e3)) * ohm,
        off_resistance=draw(_decades(1e5, 1e7)) * ohm,
        on_voltage=(off + draw(st.sampled_from([-1, 1])) * draw(st.floats(0.1, 5))) * V,
        off_voltage=off * V,
    )


def _photocell(draw: st.DrawFn) -> devices.Device:
    return devices.photocell(
        draw(_decades(100, 1e5)) * ohm,
        gamma=draw(st.floats(0.5, 1)),
        dark_resistance=draw(_decades(1e6, 1e8)) * ohm,
        response_time=draw(st.sampled_from([0, 1e-7])) * s,
    )


def _memristor(draw: st.DrawFn) -> devices.Device:
    on = draw(_decades(100, 1e4))
    off = on * draw(_decades(2, 100))
    law = "exponential" if draw(st.booleans()) else "linear"
    state = draw(st.floats(0, 1))
    if draw(st.booleans()):
        return devices.voltage_controlled_memristor(
            on_resistance=on * ohm,
            off_resistance=off * ohm,
            on_voltage=-draw(st.floats(0.2, 2)) * V,
            off_voltage=draw(st.floats(0.2, 2)) * V,
            on_rate=draw(_decades(1e3, 1e6)) * Hz,
            off_rate=draw(_decades(1e3, 1e6)) * Hz,
            law=law,
            state=state,
        )
    return devices.current_controlled_memristor(
        on_resistance=on * ohm,
        off_resistance=off * ohm,
        charge=draw(_decades(1e-3, 1)) * uC,
        law=law,
        state=state,
    )


DEVICES: Final = {
    "R": _resistor,
    "C": _capacitor,
    "I": _current,
    "D": _diode,
    "Q": _bipolar,
    "M": _mosfet,
    "G": _transconductance,
    "S": _voltage_switch,
    "P": _photocell,
    "X": _memristor,
}
"""What each device placed at random between existing nodes is, by its prefix; an
inductor, a pair of coupled windings and a switch's control port go in series with a
resistor besides, and a behavioral source reads the voltage of a control terminal."""


def _placements(
    draw: st.DrawFn, prefix: str, ends: list[str], fresh: str
) -> list[tuple[str, devices.Device, list[str]]]:
    """What a device of `prefix` placed between `ends` places.

    Each is a prefix, a device and its nets: the device, and a resistor in series where
    it needs one, `fresh` naming a net of its own.
    """
    match prefix:
        case "L":
            inductance = draw(_decades(1e-9, 1e-3)) * H
            return [
                ("L", devices.inductor(inductance), [ends[0], fresh]),
                ("R", _resistor(draw, 1, 1e3), [fresh, ends[1]]),
            ]
        case "K":
            second = f"{fresh}b"
            return [
                ("K", _coupled(draw), [ends[0], fresh, ends[2], second]),
                ("R", _resistor(draw, 1, 1e3), [fresh, ends[1]]),
                ("R", _resistor(draw, 1, 1e3), [second, ends[3]]),
            ]
        case "W":
            return [
                ("W", _switch(draw), [ends[0], ends[1], ends[2], fresh]),
                ("R", _resistor(draw, 10, 1e5), [fresh, ends[3]]),
            ]
        case "B":
            # A transconductance as an expression of its own control terminal.
            gain = draw(_decades(1e-6, 1e-3))
            source = devices.behavioral_current_source(f"{gain!r}*v(control)")
            return [("B", source, ends[:3])]
        case "M":
            # Its body on its source, as a discrete MOSFET's is, so that no junction to
            # it lies across a source with nothing to bound its current.
            return [("M", _mosfet(draw), [*ends[:3], ends[2]])]
    return [(prefix, DEVICES[prefix](draw), ends)]


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
                name=f"{prefix}{len(instances)}",
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
        prefix = draw(st.sampled_from([*DEVICES, "L", "K", "W", "B"]))
        for placed in _placements(draw, prefix, ends, f"m{len(instances)}"):
            place(*placed)
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
    return placing(circuit, instances)


def capacitive(circuit: ir.Circuit) -> list[str]:
    """The nets that hold `PARASITIC`, sorted.

    They are every net but the ground, and but the negative one of each
    current-controlled switch's control port. The port holds its two nets at one
    voltage, as one net, which holds one capacitance, on the port's positive net. With
    one on each, the port would carry the current that charges the second, which the
    switch's own output can drive through the first: its control would still move with
    no charge to slow it, and its equations could fold.
    """
    instances = circuit.components[circuit.top].instances
    joined = {
        instance.terminals["control_negative"]
        for instance in instances
        if instance.of == ir.DeviceOf(device="current_controlled_switch")
    }
    nets = {net for instance in instances for net in instance.terminals.values()}
    return sorted(nets - joined - {"0"})


def parasitic(circuit: ir.Circuit, nets: Iterable[str] | None = None) -> ir.Circuit:
    """The circuit with `PARASITIC` from each of `nets` to the ground, as nets have.

    Without `nets`, it is each net that `capacitive` names, as in a real circuit.
    Without capacitance, a regenerative loop of devices that hold no charge, such as a
    switch whose output moves its own control, can fold as its solution moves, into a
    point past which none continues: a transient's equations then have no solution, and
    an engine stops there, or leaps by chance to another. With it, every loop charges a
    capacitance, and the solution continues.
    """
    instances = circuit.components[circuit.top].instances
    if nets is None:
        nets = capacitive(circuit)
    capacitor = devices.capacitor(PARASITIC * F)
    return placing(
        circuit,
        [
            *instances,
            *(
                ir.Instance(
                    name=f"C{net}",
                    of=ir.DeviceOf(device=capacitor.model),
                    parameters=capacitor.parameters,
                    terminals={"a": net, "b": "0"},
                )
                for net in nets
            ),
        ],
    )


def placing(circuit: ir.Circuit, instances: Iterable[ir.Instance]) -> ir.Circuit:
    """The circuit with its top placing `instances` instead of its own."""
    top = circuit.components[circuit.top].model_copy(
        update={"instances": tuple(instances)}
    )
    return circuit.model_copy(update={"components": {circuit.top: top}})
