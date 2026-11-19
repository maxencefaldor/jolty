# Copyright 2026 Maxence Faldor
"""Built-in hard circuits for the robustness corpus constructed with jolty's builder.

Each circuit defines an operating-point fixture scored against all jolty DC solutions,
and most also define a transient fixture judged by final settling equilibria, fold
impasse timing, or completion. ngspice executes each circuit on the exported netlist.
"""

import importlib
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

sys.path.append(str(Path(__file__).resolve().parents[2] / "examples"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "speed" / "circuits"))
sys.path.append(
    str(Path(__file__).resolve().parents[2] / "docs" / "discovery" / "memristive_nor")
)

import buck
import numpy as np
import sram_cell
from common import SUPPLY, inverter, mos
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

FOLDED: Final = (
    ROOT / "crates" / "jolty-engine" / "tests" / "data" / "folded_sources.json"
)
"""The random circuit whose operating point source stepping folds short of."""


@dataclass(frozen=True)
class WrittenCircuit:
    """Builder-authored robustness circuit providing jolty and SPICE netlists."""

    build: Callable[[], Component | ir.Circuit]
    transient: bool = False
    """Flag enabling transient memristor dynamics in exported ngspice netlists."""
    settle: Callable[[], Component | ir.Circuit] | None = None
    """Optional builder creating the circuit with sources held at final levels."""

    def jolty(self) -> Built:
        """Builds the circuit at 27 °C with jolty's default options."""
        return _built(self.build())

    def spice(self) -> Spiced:
        """Exports the circuit as a SPICE netlist paired with voltage vector names."""
        built = self.jolty()
        lines = export.netlist(built.circuit, transient=self.transient)
        vectors = {
            net: export.vector(built.circuit, Unknown(net, "voltage"))
            for net in built.nets
        }
        return Spiced(tuple(lines), vectors)

    def settled(self) -> Built | None:
        """Builds the final-level settling circuit when `settle` is set, else None."""
        return None if self.settle is None else _built(self.settle())

    def temperature(self) -> float:
        """Returns nominal reference temperature (27 °C) in kelvins."""
        return _core.REFERENCE_TEMPERATURE


def _built(made: Component | ir.Circuit) -> Built:
    """Wraps `made` in a `Built` record collecting all comparable node voltages."""
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
    """Converts `drive` into a `Waveform` or constant voltage `Quantity`."""
    return drive if not isinstance(drive, float | int) else float(drive) * V


# Schmitt triggers.

SCHMITT_SUPPLY: Final = 5.0
"""Supply voltage of the bipolar Schmitt trigger in volts."""


def npn() -> devices.Device:
    """Creates an NPN Gummel-Poon BJT (10 fA saturation current, beta 100)."""
    return devices.bipolar_transistor(
        polarity="npn", saturation_current=1e-14 * A, forward_beta=100
    )


@component
def emitter_coupled(c: Builder) -> None:
    """Builds an emitter-coupled bipolar Schmitt trigger using two NPN transistors.

    Couples the first collector to the second base through a divider so the shared
    emitter produces hysteresis with a rising threshold above the falling threshold.
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
    """Builds the bipolar Schmitt trigger test bench driven at `drive`."""
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
    """Builds a six-transistor level-1 CMOS Schmitt trigger with dual feedback stacks.

    Connects two stacked NMOS devices to ground and two stacked PMOS devices to supply,
    with output-driven feedback devices biasing each stack midpoint for hysteresis.
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
    """Builds the 3.3 V CMOS Schmitt trigger test bench driven at `drive`."""
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
"""Bipolar Schmitt input inside the hysteresis band with three DC solutions."""

CMOS_BAND: Final = 1.65
"""Input voltage inside the CMOS Schmitt trigger hysteresis band in volts."""


def triangle(low: float, high: float, middle: float, edge: float) -> Piecewise:
    """Builds a piecewise-linear ramp `low` -> `high` -> `middle` (`edge` us)."""
    return Piecewise(
        [
            (0 * us, low * V),
            (edge * us, high * V),
            (1.5 * edge * us, high * V),
            (2.5 * edge * us, middle * V),
        ]
    )


def highest(net: str) -> Callable[[Solution, Sequence[Solution]], bool]:
    """Creates a predicate matching the DC solution with maximum `net` voltage."""

    def chosen(solution: Solution, every: Sequence[Solution]) -> bool:
        return solution[net] == max(each[net] for each in every)

    return chosen


def lowest(net: str) -> Callable[[Solution, Sequence[Solution]], bool]:
    """Creates a predicate matching the DC solution with minimum `net` voltage."""

    def chosen(solution: Solution, every: Sequence[Solution]) -> bool:
        return solution[net] == min(each[net] for each in every)

    return chosen


# Cross-coupled latch, SRAM cell, and ring oscillator.


@component
def latch(c: Builder) -> None:
    """Builds cross-coupled CMOS inverters with asymmetric load (`q` twice `qb`)."""
    vdd, ground = c.port("vdd", Role.SUPPLY), c.port("ground", Role.GROUND)
    left, right = c.place(inverter(), name="X1"), c.place(inverter(), name="X2")
    heavy = c.place(devices.capacitor(20 * fF), name="CQ")
    light = c.place(devices.capacitor(10 * fF), name="CQB")
    c.connect(left.vdd, right.vdd, vdd)
    c.connect(left.y, right.a, heavy.a, name="q")
    c.connect(right.y, left.a, light.a, name="qb")
    c.connect(left.gnd, right.gnd, heavy.b, light.b, ground)


def powered(supply: Waveform | float, part: Callable[[], Component]) -> Component:
    """Wraps `part` in a test bench powered by voltage level or waveform `supply`."""
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
"""Supply waveform ramping linearly from 0 V to 3.3 V over 1 ns."""


def stored(solution: Solution, _: Sequence[Solution]) -> bool:
    """Returns whether a latch solution stores a valid logic bit across `q` and `qb`."""
    return (solution["X1/q"] - 0.5 * SUPPLY) * (solution["X1/qb"] - 0.5 * SUPPLY) < 0


@component
def ring(c: Builder) -> None:
    """Builds a five-stage CMOS ring oscillator with doubled load on stage 1.

    Uses unequal stage loading to break symmetry so a ramp from rest departs the
    unstable midpoint equilibrium and starts oscillating.
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
    """Builds the example SRAM cell bench with word line low and resting bit lines."""
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
    """Returns whether an SRAM solution stores logic 0 (`q` low and `q_bar` high)."""
    return solution["X1/q"] < 0.5 * SUPPLY < solution["X1/q_bar"]


# Zero-capacitance algebraic MOSFET chain.


def bare(channel: str) -> devices.Device:
    """Creates a level-1 MOSFET with zero intrinsic or parasitic capacitance."""
    p = channel == "p"
    return devices.mosfet(
        channel="p" if p else "n",
        width=(2 if p else 1) * um,
        length=1 * um,
        threshold_voltage=(-0.7 if p else 0.7) * V,
        channel_length_modulation=0.05 / V,
    )


def chain(drive: Waveform | float) -> Component:
    """Builds three zero-capacitance CMOS inverters driven by `drive`."""
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
    """Builds a capacitor charged across a series resistor and N-shaped conductance.

    Satisfies `v = u + R·g(u)`, which folds where `1 + R·g'(u) = 0` so no nearby
    solution exists once charging crosses the fold.
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
    """Computes the exact fold impasse time in seconds via Gauss-Legendre quadrature.

    Along the lower branch, `C·dv/dt = I - g(u)` and `dv = (1 + R·g'(u))·du`.
    Integrates `C·(1 + R·g'(u)) / (I - g(u))` from 0 to the smaller root `u₁` of
    `1 + R·g'(u) = 0`, then adds half the current rise time.
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
"""Summaries of the numerical or modeling difficulty of each built-in fixture."""


def _pulse() -> Pulse:
    """Creates a 0 V to 3.3 V pulse with 100 ps edges from 1 ns to 6.1 ns."""
    return Pulse(
        initial=0 * V,
        pulsed=SUPPLY * V,
        delay=1 * ns,
        rise=100 * ps,
        fall=100 * ps,
        width=5 * ns,
    )


def _folded() -> ir.Circuit:
    """Loads the 47-device circuit whose operating point folds on source stepping."""
    return ir.Circuit.model_validate_json(FOLDED.read_text(encoding="utf-8"))


NOR: Final = ROOT / "docs" / "discovery" / "memristive_nor"
"""Directory of the memristive NOR discovery supplying the `candidates` module."""


def _nor() -> Component:
    """Builds the four-gate memristive NOR bench driven at its high-amplitude corner.

    Imports the discovery's local ``candidates`` module dynamically from `NOR`.
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
    """Returns all built-in hard circuit fixtures for the robustness corpus."""
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
