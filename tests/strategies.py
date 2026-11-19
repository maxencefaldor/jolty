# Copyright 2026 Maxence Faldor
"""Hypothesis strategies generating random circuits for simulation tests.

Each generated circuit connects every node to ground through a spanning tree of
resistors to guarantee a DC path, then adds random sources and devices.
Inductors, coupled windings, and current-controlled switch control ports include
series resistors to prevent voltage-source/inductor loops. Diodes, BJTs, and
MOSFETs include current-limiting series resistances, with MOSFET bodies tied to
sources and optional per-instance temperature offsets. Circuit sizes extend
beyond the 32-unknown scale of hand-written tests.
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
"""Optional fresh-example count read from `JOLTY_RANDOM_CIRCUITS`."""

SETTINGS: Final = settings(
    max_examples=40 if SEARCH is None else int(SEARCH),
    derandomize=SEARCH is None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
"""Hypothesis settings for random-circuit tests (40 derandomized or `SEARCH` fresh)."""


def _decades(low: float, high: float) -> st.SearchStrategy[float]:
    """Returns a strategy for values distributed log-uniformly from `low` to `high`."""
    return st.floats(np.log10(low), np.log10(high)).map(lambda exponent: 10**exponent)


def _resistor(draw: st.DrawFn, low: float = 10, high: float = 1e6) -> devices.Device:
    return devices.resistor(draw(_decades(low, high)) * ohm)


def _capacitor(draw: st.DrawFn) -> devices.Device:
    return devices.capacitor(draw(_decades(1e-12, 1e-6)) * F)


def _current(draw: st.DrawFn) -> devices.Device:
    return devices.current_source(draw(_decades(1e-6, 1e-3)) * A)


def _apart(draw: st.DrawFn, device: devices.Device) -> devices.Device:
    """Returns `device` unchanged or with a random per-instance temperature offset."""
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
"""Mapping from SPICE prefix to device generator for direct placements.

Inductors, coupled windings, and current-controlled switch control ports add a
series resistor in `_placements`, and behavioral sources sense a control net.
"""


def _placements(
    draw: st.DrawFn, prefix: str, ends: list[str], fresh: str
) -> list[tuple[str, devices.Device, list[str]]]:
    """Returns `(prefix, device, nets)` entries for a random placement between `ends`.

    Inserts a series resistor and intermediate net `fresh` when the device
    requires series impedance.
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
            # Transconductance written as an expression of the control terminal voltage.
            gain = draw(_decades(1e-6, 1e-3))
            source = devices.behavioral_current_source(f"{gain!r}*v(control)")
            return [("B", source, ends[:3])]
        case "M":
            # Tie the MOSFET body to the source (like a discrete MOSFET) so no
            # bulk junction spans a voltage source without current limiting.
            return [("M", _mosfet(draw), [*ends[:3], ends[2]])]
    return [(prefix, DEVICES[prefix](draw), ends)]


@st.composite
def circuits(draw: st.DrawFn) -> ir.Circuit:
    """Generates a random valid circuit with a DC path from every node to ground."""
    count = draw(st.integers(1, 40))
    nodes = ["0", *(f"n{index}" for index in range(1, count + 1))]
    instances: list[ir.Instance] = []

    def place(prefix: str, device: devices.Device, nets: list[str]) -> None:
        terminals = dict(zip(device.terminals, nets, strict=False))
        instances.append(
            ir.Instance(
                name=f"{prefix}{len(instances)}",
                of=ir.DeviceOf(model=device.model),
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
        # Pick two distinct nodes first to avoid shorted two-terminal sources,
        # then two arbitrary nodes for extra transistor or control terminals.
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
"""Duration of each random transient in seconds (1 V pulse from 0.2 µs for 0.8 µs)."""

STEP: Final = STOP / 200
"""Maximum time step of each random transient in seconds."""

PARASITIC: Final = 10e-15
"""Default shunt capacitance to ground added for transient tests, in farads."""


def pulsed(circuit: ir.Circuit) -> ir.Circuit:
    """Returns `circuit` with the first voltage source pulsed 1 V above DC and back."""
    instances = list(circuit.components[circuit.top].instances)
    index = next(
        index
        for index, instance in enumerate(instances)
        if instance.of == ir.DeviceOf(model="voltage_source")
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
    """Returns sorted net names in `circuit` that receive `PARASITIC` shunt capacitors.

    Includes every net except ground and the negative terminal of each
    current-controlled switch control port. A control port forces both
    terminals to the same potential, so placing capacitance on both nets lets
    the switch output drive the second capacitor through the port without
    slowing the control current and allows the equations to fold.
    """
    instances = circuit.components[circuit.top].instances
    joined = {
        instance.terminals["control_negative"]
        for instance in instances
        if instance.of == ir.DeviceOf(model="current_controlled_switch")
    }
    nets = {net for instance in instances for net in instance.terminals.values()}
    return sorted(nets - joined - {"0"})


def parasitic(circuit: ir.Circuit, nets: Iterable[str] | None = None) -> ir.Circuit:
    """Returns `circuit` with `PARASITIC` capacitance from each net in `nets` to ground.

    Defaults to the nets returned by `capacitive`. Without shunt capacitance, a
    memoryless regenerative loop (such as a switch driving the switch control)
    can fold to a singular point where a transient stalls or jumps branches.
    Adding shunt capacitance ensures every feedback loop integrates charge.
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
                    of=ir.DeviceOf(model=capacitor.model),
                    parameters=capacitor.parameters,
                    terminals={"a": net, "b": "0"},
                )
                for net in nets
            ),
        ],
    )


def placing(circuit: ir.Circuit, instances: Iterable[ir.Instance]) -> ir.Circuit:
    """Returns a copy of `circuit` with the top component's instances replaced."""
    top = circuit.components[circuit.top].model_copy(
        update={"instances": tuple(instances)}
    )
    return circuit.model_copy(update={"components": {circuit.top: top}})
