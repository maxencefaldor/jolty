# Copyright 2026 Maxence Faldor
"""jolty's own hard circuits, each written with jolty's builder.

Each circuit gives a fixture for its operating point, judged against every DC solution
jolty finds, and most a transient, judged by the DC solutions its sources' last levels
leave, by the time its solution ends, or by its completion alone. ngspice runs each on
the netlist jolty's exporter writes.
"""

import importlib
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from corpus import (
    ROBUSTNESS,
    Built,
    Completes,
    Expectation,
    Fixture,
    Impasse,
    Kind,
    Oscillates,
    Point,
    Settles,
    Solution,
    Solutions,
    Spiced,
    Transient,
)

from jolty import Builder, Component, Role, _core, circuit, component, devices, ir
from jolty.plan import Options
from jolty.results import Unknown
from jolty.simulate import Simulation
from jolty.spice import export
from jolty.units import A, Quantity, V, fF, mA, nF, ns, ohm, pF, ps, um, us
from jolty.waveforms import Piecewise, Pulse, Waveform

BENCHMARKS: Final = ROBUSTNESS.parent
"""The ``benchmarks`` directory."""

ROOT: Final = BENCHMARKS.parent
"""The repository's root."""

for _shared in (
    ROOT / "examples",
    BENCHMARKS / "speed" / "circuits",
    ROOT / "discovery" / "memristive_nor",
):
    if str(_shared) not in sys.path:
        sys.path.append(str(_shared))

import buck  # noqa: E402 - the speed suite's fixture's script, beside its fixtures
import sram_cell  # noqa: E402 - the example's script
from common import SUPPLY, inverter, mos  # noqa: E402 - the examples' CMOS process

FOLDED: Final = (
    ROOT / "crates" / "jolty-engine" / "tests" / "data" / "folded_sources.json"
)
"""The random circuit whose operating point source stepping folds short of."""


@dataclass(frozen=True)
class WrittenCircuit:
    """A circuit jolty's builder writes, as each simulator reads it."""

    build: Callable[[], Component | ir.Circuit]
    transient: bool = False
    """Whether ngspice runs a transient of it, in whose netlist a memristor moves."""
    settle: Callable[[], Component | ir.Circuit] | None = None
    """The circuit with every source at its last level."""

    def jolty(self) -> Built:
        """The circuit at 27 °C and jolty's default options."""
        return _built(self.build())

    def spice(self) -> Spiced:
        """The netlist jolty's exporter writes of the circuit, and its nets' vectors."""
        built = self.jolty()
        lines = export.netlist(built.circuit, transient=self.transient)
        vectors = {
            net: export.vector(built.circuit, Unknown(net, "voltage"))
            for net in built.nets
        }
        return Spiced(tuple(lines), vectors)

    def settled(self) -> Built | None:
        """The circuit with every source at its last level, if given."""
        return None if self.settle is None else _built(self.settle())

    def temperature(self) -> float:
        """27 °C, in kelvins."""
        return _core.REFERENCE_TEMPERATURE


def _built(made: Component | ir.Circuit) -> Built:
    """`made` as jolty simulates it, every net's voltage compared."""
    written = made if isinstance(made, ir.Circuit) else circuit(made)
    unknowns = Simulation(written).unknowns
    nets = tuple(
        unknown.name
        for unknown in unknowns
        if unknown.nature == "voltage"
        and ":" not in unknown.name
        and "." not in unknown.name.rsplit("/", 1)[-1]
    )
    return Built(written, _core.REFERENCE_TEMPERATURE, Options(), nets)


def _level(drive: Waveform | float) -> Waveform | Quantity:
    """A source's level: a waveform, or a constant in volts."""
    return drive if not isinstance(drive, float | int) else float(drive) * V


# Schmitt triggers.

SCHMITT_SUPPLY: Final = 5.0
"""The bipolar Schmitt trigger's supply, in volts."""


def npn() -> devices.Device:
    """An npn transistor of Gummel and Poon's model: 10 fA and a gain of 100."""
    return devices.bipolar_transistor(
        polarity="npn", saturation_current=1e-14 * A, forward_beta=100
    )


@component
def emitter_coupled(c: Builder) -> None:
    """A Schmitt trigger of two npn transistors sharing an emitter resistor.

    The first's collector drives the second's base through a divider, so the second
    conducts until the input lifts the shared emitter past it: the input's rising
    threshold lies above its falling one.
    """
    vcc, ground = c.port("vcc", Role.SUPPLY), c.port("ground", Role.GROUND)
    source, output = c.port("input", Role.INPUT), c.port("output", Role.OUTPUT)
    first, second = c.place(npn(), name="Q1"), c.place(npn(), name="Q2")
    load = c.place(devices.resistor(1e3 * ohm), name="RC1")
    upper = c.place(devices.resistor(1e3 * ohm), name="R1")
    lower = c.place(devices.resistor(4e3 * ohm), name="R2")
    pull = c.place(devices.resistor(470 * ohm), name="RC2")
    shared = c.place(devices.resistor(1e3 * ohm), name="RE")
    stray = [
        c.place(devices.capacitor(1 * pF), name=f"C{index}") for index in (1, 2, 3)
    ]
    c.connect(vcc, load.a, pull.a)
    c.connect(source, first.base)
    c.connect(first.collector, load.b, upper.a, stray[0].a, name="c1")
    c.connect(upper.b, lower.a, second.base, stray[1].a, name="b2")
    c.connect(first.emitter, second.emitter, shared.a, name="e")
    c.connect(second.collector, pull.b, stray[2].a, output)
    c.connect(lower.b, shared.b, *(each.b for each in stray), ground)


def schmitt_bjt(drive: Waveform | float) -> Component:
    """The bipolar Schmitt trigger on its supply, its input at `drive`."""
    level = _level(drive)

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(SCHMITT_SUPPLY * V), name="VCC")
        given = c.place(devices.voltage_source(level), name="VIN")
        trigger = c.place(emitter_coupled(), name="X1")
        c.connect(supply.positive, trigger.vcc, name="vcc")
        c.connect(given.positive, trigger.input, name="in")
        c.connect(trigger.output, name="out")
        c.connect(supply.negative, given.negative, trigger.ground, ground)

    return bench()


@component
def cmos_schmitt(c: Builder) -> None:
    """A Schmitt trigger of six level 1 MOSFETs: two stacks, each with a feedback.

    The input drives two n-channel transistors in series down to the ground and two
    p-channel ones up to the supply; the output's feedback transistors hold each
    stack's middle node, so the output switches late in either direction.
    """
    vdd, ground = c.port("vdd", Role.SUPPLY), c.port("ground", Role.GROUND)
    source, output = c.port("input", Role.INPUT), c.port("output", Role.OUTPUT)
    n1, n2, n3 = (c.place(mos("n"), name=name) for name in ("MN1", "MN2", "MN3"))
    p1, p2, p3 = (c.place(mos("p"), name=name) for name in ("MP1", "MP2", "MP3"))
    load = c.place(devices.capacitor(10 * fF), name="CL")
    c.connect(source, n1.gate, n2.gate, p1.gate, p2.gate)
    c.connect(n1.drain, n2.source, n3.source, name="xn")
    c.connect(p1.drain, p2.source, p3.source, name="xp")
    c.connect(n2.drain, p2.drain, n3.gate, p3.gate, load.a, output)
    c.connect(n3.drain, p1.source, p1.bulk, p2.bulk, p3.bulk, vdd)
    c.connect(n1.source, n1.bulk, n2.bulk, n3.bulk, p3.drain, load.b, ground)


def schmitt_cmos(drive: Waveform | float) -> Component:
    """The CMOS Schmitt trigger on 3.3 V, its input at `drive`."""
    level = _level(drive)

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
        given = c.place(devices.voltage_source(level), name="VIN")
        trigger = c.place(cmos_schmitt(), name="X1")
        c.connect(supply.positive, trigger.vdd, name="vdd")
        c.connect(given.positive, trigger.input, name="in")
        c.connect(trigger.output, name="out")
        c.connect(supply.negative, given.negative, trigger.ground, ground)

    return bench()


BJT_BAND: Final = 2.75
"""A level of the bipolar Schmitt trigger's input between its two thresholds, in
volts, where it has three DC solutions."""

CMOS_BAND: Final = 1.65
"""A level of the CMOS Schmitt trigger's input between its thresholds, in volts."""


def triangle(low: float, high: float, middle: float, edge: float) -> Piecewise:
    """From `low` up to `high` and down to `middle`, each ramp `edge` microseconds."""
    return Piecewise(
        [
            (0 * us, low * V),
            (edge * us, high * V),
            (1.5 * edge * us, high * V),
            (2.5 * edge * us, middle * V),
        ]
    )


def highest(net: str) -> Callable[[Solution, Sequence[Solution]], bool]:
    """Whether a solution holds `net` the highest of every solution."""

    def chosen(solution: Solution, every: Sequence[Solution]) -> bool:
        return solution[net] == max(each[net] for each in every)

    return chosen


def lowest(net: str) -> Callable[[Solution, Sequence[Solution]], bool]:
    """Whether a solution holds `net` the lowest of every solution."""

    def chosen(solution: Solution, every: Sequence[Solution]) -> bool:
        return solution[net] == min(each[net] for each in every)

    return chosen


# A latch, an SRAM cell and a ring.


@component
def latch(c: Builder) -> None:
    """Two CMOS inverters, each the other's input, `q` loaded twice as much as `qb`."""
    vdd, ground = c.port("vdd", Role.SUPPLY), c.port("ground", Role.GROUND)
    left, right = c.place(inverter(), name="X1"), c.place(inverter(), name="X2")
    heavy = c.place(devices.capacitor(20 * fF), name="CQ")
    light = c.place(devices.capacitor(10 * fF), name="CQB")
    c.connect(left.vdd, right.vdd, vdd)
    c.connect(left.y, right.a, heavy.a, name="q")
    c.connect(right.y, left.a, light.a, name="qb")
    c.connect(left.gnd, right.gnd, heavy.b, light.b, ground)


def powered(supply: Waveform | float, part: Callable[[], Component]) -> Component:
    """`part` on a supply of `supply`, a level in volts or a waveform."""
    level = _level(supply)

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        source = c.place(devices.voltage_source(level), name="VDD")
        placed = c.place(part(), name="X1")
        c.connect(source.positive, placed.vdd, name="vdd")
        c.connect(source.negative, placed.ground, ground)

    return bench()


RAMP: Final = Piecewise([(0 * ns, 0 * V), (1 * ns, SUPPLY * V)])
"""A supply ramped from rest to 3.3 V in a nanosecond."""


def stored(solution: Solution, _: Sequence[Solution]) -> bool:
    """Whether a latch's solution stores a bit: `q` and `qb` on either side of half."""
    return (solution["X1/q"] - 0.5 * SUPPLY) * (solution["X1/qb"] - 0.5 * SUPPLY) < 0


@component
def ring(c: Builder) -> None:
    """Five CMOS inverters in a ring, the first stage loaded twice as much as the rest.

    The unequal load breaks the ring's symmetry, so that its solution from rest leaves
    the balanced state, where every stage sits at its threshold, which is unstable.
    """
    vdd, ground = c.port("vdd", Role.SUPPLY), c.port("ground", Role.GROUND)
    gates = [c.place(inverter(), name=f"X{index}") for index in range(1, 6)]
    nets = [c.net(f"n{index}") for index in range(5)]
    for index, gate in enumerate(gates):
        load = c.place(
            devices.capacitor((20 if index == 0 else 10) * fF), name=f"C{index}"
        )
        c.connect(gate.vdd, vdd)
        c.connect(gate.gnd, load.b, ground)
        c.connect(gate.a, nets[index - 1])
        c.connect(gate.y, load.a, nets[index])


@component
def sram(c: Builder) -> None:
    """The example's SRAM cell, its word line low and its bit lines at rest."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
    word = c.place(devices.voltage_source(0 * V), name="VWL")
    bit = c.place(devices.voltage_source(0 * V), name="VBL")
    bit_bar = c.place(devices.voltage_source(SUPPLY * V), name="VBLB")
    memory = c.place(sram_cell.cell(), name="X1")
    c.connect(supply.positive, memory.vdd, name="vdd")
    c.connect(word.positive, memory.word, name="wl")
    c.connect(bit.positive, memory.bit, name="bl")
    c.connect(bit_bar.positive, memory.bit_bar, name="blb")
    c.connect(
        supply.negative,
        word.negative,
        bit.negative,
        bit_bar.negative,
        memory.gnd,
        ground,
    )


def holds_zero(solution: Solution, _: Sequence[Solution]) -> bool:
    """Whether the SRAM cell's solution holds a 0: `q` low, `q_bar` high."""
    return solution["X1/q"] < 0.5 * SUPPLY < solution["X1/q_bar"]


# Nets with no capacitance.


def bare(channel: str) -> devices.Device:
    """A level 1 MOSFET of the examples' process with no capacitance at all."""
    p = channel == "p"
    return devices.mosfet(
        channel="p" if p else "n",
        width=(2 if p else 1) * um,
        length=1 * um,
        threshold_voltage=(-0.7 if p else 0.7) * V,
        channel_length_modulation=0.05 / V,
    )


def chain(drive: Waveform | float) -> Component:
    """Three CMOS inverters of MOSFETs with no capacitance, driven by `drive`."""
    level = _level(drive)

    @component
    def bench(c: Builder) -> None:
        ground = c.port("ground", Role.GROUND)
        supply = c.place(devices.voltage_source(SUPPLY * V), name="VDD")
        given = c.place(devices.voltage_source(level), name="VIN")
        vdd = c.net("vdd")
        nets = [c.net(name) for name in ("in", "n1", "n2", "out")]
        for index in range(3):
            up = c.place(bare("p"), name=f"MP{index + 1}")
            down = c.place(bare("n"), name=f"MN{index + 1}")
            c.connect(up.gate, down.gate, nets[index])
            c.connect(up.drain, down.drain, nets[index + 1])
            c.connect(up.source, up.bulk, vdd)
            c.connect(down.source, down.bulk, ground)
        c.connect(supply.positive, vdd)
        c.connect(given.positive, nets[0])
        c.connect(supply.negative, given.negative, ground)

    return bench()


# An impasse point, after L. O. Chua and A.-C. Deng, "Impasse points. Part I:
# numerical aspects", International Journal of Circuit Theory and Applications 17
# (1989): a capacitor across a resistor in series with an N-shaped conductance.

IMPASSE_RESISTANCE: Final = 10e3
"""The resistor in series with the N-shaped conductance, in ohms."""

IMPASSE_SCALE: Final = 1e-3
"""The conductance's current, `k·u·(u - 1)·(u - 2)`, its `k` in amperes per cubic
volt: the engine's test of an N-shaped load line."""

IMPASSE_CHARGE: Final = 1e-3
"""The current that charges the capacitor, in amperes, more than the conductance's
largest on its lower branch."""

IMPASSE_CAPACITANCE: Final = 1e-9
"""The capacitor, in farads."""

IMPASSE_RISE: Final = 1e-9
"""The charging current's rise from zero, in seconds."""


@component
def impasse_circuit(c: Builder) -> None:
    """A capacitor charged across a resistor in series with an N-shaped conductance.

    The capacitor's voltage `v` is the resistor's drop plus the conductance's voltage
    `u`, so `v = u + R·g(u)`, which folds where `1 + R·g'(u) = 0`: charged past the
    fold, the circuit's equations have no solution near the one they had.
    """
    ground = c.port("ground", Role.GROUND)
    charge = Pulse(
        initial=0 * mA,
        pulsed=IMPASSE_CHARGE * A,
        rise=IMPASSE_RISE * 1e9 * ns,
        fall=1 * ns,
        width=1 * us,
    )
    source = c.place(devices.current_source(charge), name="I1")
    held = c.place(devices.capacitor(IMPASSE_CAPACITANCE * 1e9 * nF), name="C1")
    series = c.place(devices.resistor(IMPASSE_RESISTANCE * ohm), name="R1")
    expression = f"{IMPASSE_SCALE!r} * v(u) * (v(u) - 1) * (v(u) - 2)"
    conductance = c.place(devices.behavioral_current_source(expression), name="B1")
    c.connect(source.negative, held.a, series.a, name="v")
    c.connect(series.b, conductance.positive, conductance.u, name="u")
    c.connect(source.positive, held.b, conductance.negative, ground)


def impasse_time() -> float:
    """When the impasse circuit's solution ends, in seconds, from its closed form.

    Along the lower branch `C·dv/dt = I - g(u)` and `dv = (1 + R·g'(u))·du`, so the
    time to the fold `u₁`, the smaller root of `1 + R·g'(u) = 0`, is the integral of
    `C·(1 + R·g'(u)) / (I - g(u))` from 0 to `u₁`, by 64-point Gauss-Legendre
    quadrature of its smooth integrand, plus half the current's rise, during which it
    charges the capacitor as half its level would.
    """
    k, r = IMPASSE_SCALE, IMPASSE_RESISTANCE
    # g'(u) = k·(3u² - 6u + 2), so 1 + R·g'(u) = 0 at u = 1 ± sqrt(1/3 - 1/(3kR)).
    fold = 1 - math.sqrt(1 / 3 - 1 / (3 * k * r))
    nodes, weights = np.polynomial.legendre.leggauss(64)
    u = 0.5 * fold * (nodes + 1)
    g = k * u * (u - 1) * (u - 2)
    slope = k * (3 * u * u - 6 * u + 2)
    integrand = IMPASSE_CAPACITANCE * (1 + r * slope) / (IMPASSE_CHARGE - g)
    return float(0.5 * fold * np.sum(weights * integrand)) + 0.5 * IMPASSE_RISE


# The memristive NOR, at its high corner.

NOR_AMPLITUDE: Final = 1.1 * 1.525
"""The pulse at 10 % above the design's 1.525 V, the corner ngspice cannot finish."""

NOR_SHUNT: Final = 2.2e3
"""The design's resistance across the inputs, in ohms."""


HARD: Final = {
    "schmitt_bjt/op": "A bipolar Schmitt trigger between its thresholds: three DC "
    "solutions",
    "schmitt_cmos/op": "A CMOS Schmitt trigger between its thresholds: three DC "
    "solutions",
    "cmos_latch/op": "Two cross-coupled CMOS inverters: two stored states and a "
    "balance between them",
    "sram_cell/op": "The example's SRAM cell, its word line low: two stored states and "
    "a balance",
    "ring/op": "An odd ring of five CMOS inverters: one DC solution, every stage at "
    "its threshold",
    "folded_sources/op": "A random circuit of 47 devices whose operating point source "
    "stepping folds short of",
    "bare_chain/op": "Three CMOS inverters with no capacitance at any net",
    "buck/op": "The benchmarks' synchronous buck converter, its body diodes at rest",
    "memristive_nor/op": "Four memristive NOR gates of VTEAM memristors, held at "
    "their states",
    "schmitt_bjt/tran": "The bipolar Schmitt trigger driven past its upper threshold "
    "and back into its band over 25 µs",
    "schmitt_cmos/tran": "The CMOS Schmitt trigger driven past its upper threshold and "
    "back into its band over 2.5 µs",
    "cmos_latch/power_up": "The latch from rest, its supply ramped on in 1 ns: it "
    "must fall into a stored state",
    "sram_cell/tran": "The example's SRAM cell written 1 then 0, with 100 ps edges",
    "ring/from_rest": "The ring from rest, its supply ramped on in 1 ns, no kick: "
    "only its asymmetry starts it",
    "bare_chain/tran": "The chain with no capacitance under a pulse of 100 ps edges: "
    "every net algebraic",
    "impasse/tran": "A capacitor charged until its circuit's equations fold: no "
    "solution goes on",
    "buck/tran": "The benchmarks' buck converter over 2 ms: four gate edges a period, "
    "a body diode on at each",
    "memristive_nor/tran": "The memristive NOR at 10 % above its design's pulse, "
    "whose states race to their bounds",
}
"""What makes each fixture hard, by its name."""


def _pulse() -> Pulse:
    """A pulse from 0 to 3.3 V of 100 ps edges, from 1 ns to 6.1 ns."""
    return Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        delay=1 * ns,
        rise=100 * ps,
        fall=100 * ps,
        width=5 * ns,
    )


def _folded() -> ir.Circuit:
    """The random circuit whose operating point source stepping folds short of."""
    return ir.Circuit.model_validate_json(FOLDED.read_text(encoding="utf-8"))


NOR: Final = ROOT / "discovery" / "memristive_nor"
"""The memristive NOR discovery, whose candidates' module writes its bench."""


def _nor() -> Component:
    """The memristive NOR's four gates, written by a pulse at its high corner.

    The discovery's ``candidates`` module is imported from its directory, as its design
    imports it; every discovery has a module of that name.
    """
    if str(NOR) not in sys.path:
        sys.path.insert(0, str(NOR))
    candidates = importlib.import_module("candidates")
    written = candidates.bench(NOR_AMPLITUDE, NOR_SHUNT)
    if not isinstance(written, Component):
        message = f"the memristive NOR's bench is a {type(written).__name__}"
        raise TypeError(message)
    return written


def fixtures() -> list[Fixture]:
    """The fixtures of jolty's own hard circuits."""
    points = {
        "schmitt_bjt/op": WrittenCircuit(lambda: schmitt_bjt(BJT_BAND)),
        "schmitt_cmos/op": WrittenCircuit(lambda: schmitt_cmos(CMOS_BAND)),
        "cmos_latch/op": WrittenCircuit(lambda: powered(SUPPLY, latch)),
        "sram_cell/op": WrittenCircuit(sram),
        "ring/op": WrittenCircuit(lambda: powered(SUPPLY, ring)),
        "folded_sources/op": WrittenCircuit(_folded),
        "bare_chain/op": WrittenCircuit(lambda: chain(0.0)),
        "buck/op": WrittenCircuit(buck.build),
        "memristive_nor/op": WrittenCircuit(_nor),
    }
    found = [
        Fixture(
            name, Kind.DC, "jolty", HARD[name], fixture_circuit, Point(), Solutions()
        )
        for name, fixture_circuit in points.items()
    ]
    transients: dict[str, tuple[WrittenCircuit, Transient, Expectation]] = {
        "schmitt_bjt/tran": (
            WrittenCircuit(
                lambda: schmitt_bjt(triangle(0, SCHMITT_SUPPLY, BJT_BAND, 10)),
                transient=True,
                settle=lambda: schmitt_bjt(BJT_BAND),
            ),
            Transient(0.1e-6, 25e-6),
            Settles("the output high, on the branch it came down", highest("out")),
        ),
        "schmitt_cmos/tran": (
            WrittenCircuit(
                lambda: schmitt_cmos(triangle(0, SUPPLY, CMOS_BAND, 1)),
                transient=True,
                settle=lambda: schmitt_cmos(CMOS_BAND),
            ),
            Transient(0.01e-6, 2.5e-6),
            Settles("the output low, on the branch it came down", lowest("out")),
        ),
        "cmos_latch/power_up": (
            WrittenCircuit(
                lambda: powered(RAMP, latch),
                transient=True,
                settle=lambda: powered(SUPPLY, latch),
            ),
            Transient(5e-12, 10e-9, 5e-12),
            Settles("a stored state", stored),
        ),
        "sram_cell/tran": (
            WrittenCircuit(sram_cell.build, transient=True, settle=sram),
            Transient(10e-12, 20e-9, 10e-12),
            Settles("the 0 written last", holds_zero),
        ),
        "ring/from_rest": (
            WrittenCircuit(
                lambda: powered(RAMP, ring),
                transient=True,
                settle=lambda: powered(SUPPLY, ring),
            ),
            Transient(5e-12, 20e-9, 5e-12),
            Oscillates(),
        ),
        "bare_chain/tran": (
            WrittenCircuit(
                lambda: chain(_pulse()), transient=True, settle=lambda: chain(0.0)
            ),
            Transient(10e-12, 10e-9),
            Settles("the input's low level through three inversions"),
        ),
        "impasse/tran": (
            WrittenCircuit(impasse_circuit, transient=True),
            Transient(50e-9, 10e-6),
            Impasse(impasse_time()),
        ),
        "buck/tran": (
            WrittenCircuit(buck.build, transient=True),
            Transient(2e-3 / 50, 2e-3),
            Completes(),
        ),
        "memristive_nor/tran": (
            WrittenCircuit(_nor, transient=True),
            Transient(0.025e-9, 13e-9, 0.025e-9),
            Completes(),
        ),
    }
    found.extend(
        Fixture(
            name,
            Kind.TRANSIENT,
            "jolty",
            HARD[name],
            fixture_circuit,
            analysis,
            expectation,
        )
        for name, (fixture_circuit, analysis, expectation) in transients.items()
    )
    return found
