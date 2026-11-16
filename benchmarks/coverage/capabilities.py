# Copyright 2026 Maxence Faldor
"""The coverage matrix's rows, each with the fixture that exercises it and its answer.

A row is a capability a user can ask a simulator for: an analysis, a device model at
its level, a feature of SPICE's syntax, or an output. Its fixture is the smallest
circuit and plan that exercise it: what jolty runs, through its builder or through its
SPICE reader, and the deck the SPICE simulators run, with each one's commands and the
vectors it reads. Each fixture reads a few numbers, named alike in every simulator, and
its known answer gives them: a closed form, or, for a model that another implementation
defines, that implementation's run (`Known.reference`).

A simulator that has no statement asking for a row has no fixture for it: its
`simulators.Unsupported` names where its documentation, or jolty's code, says what it
offers instead.
"""

import cmath
import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
from simulators import Simulator, Unsupported

from jolty import Builder, Component, Role, circuit, devices, ir, results
from jolty.builder import Net
from jolty.errors import JoltyError
from jolty.plan import (
    TIGHT,
    Ac,
    Analysis,
    Axis,
    DcSweep,
    Grid,
    Method,
    MonteCarlo,
    Noise,
    OperatingPoint,
    OperatingPoints,
    PoleZero,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
    decade,
)
from jolty.simulate import Simulation
from jolty.specs import Specs, crossing, evaluate
from jolty.spice import Library, export, ngspice
from jolty.units import (
    A,
    Hz,
    K,
    V,
    W,
    kohm,
    nF,
    ns,
    ohm,
    s,
    us,
)
from jolty.units import (
    m as m_,
)
from jolty.waveforms import Piecewise, Sine

type Values = dict[str, complex]
"""What a fixture reads, by name: a voltage such as ``v(out)``, a current such as
``i(v1)``, either at a point such as ``v(out)@2``, or another number an analysis
gives."""


SOLVED: Final = 1e-6
"""The relative tolerance a solved value agrees within, of the largest value of its
kind in the row: the relative tolerance every simulator runs with, within which the
differential tests compare operating points, sweeps and responses."""

INTEGRATED: Final = 1e-3
"""The relative tolerance a value of a transient agrees within: the differential tests'
on transients, a thousandth of the largest value, which interpolation between two
simulators' steps and their truncation errors stay within."""

ROOM: Final = 300.15
"""The temperature every fixture runs at, 27 °C in kelvins, SPICE's nominal one."""

BOLTZMANN: Final = 1.380649e-23
"""Boltzmann's constant, exact in the SI since 2019, in joules per kelvin."""

CHARGE: Final = 1.602176634e-19
"""The elementary charge, exact in the SI since 2019, in coulombs."""

THERMAL_VOLTAGE: Final = BOLTZMANN * ROOM / CHARGE
"""The thermal voltage at `ROOM`, which every simulator runs at once its temperature
is set where its own constants give this one."""


class Category(StrEnum):
    """What a row is: an analysis, a device model, a feature of syntax, an output."""

    ANALYSIS = "analysis"
    DEVICE = "device"
    SYNTAX = "syntax"
    OUTPUT = "output"


class Documented(Exception):  # noqa: N818 - a finding, not a failure
    """A simulator offers what a row asks, by its manual, but no fixture runs it."""


@dataclass(frozen=True, kw_only=True)
class Known:
    """A fixture's known answer: its values, where they come from, and the tolerance.

    Without `values`, the answer is the run of the simulator `reference` names, the
    implementation that defines the model, such as ngspice for SPICE3's models.
    """

    values: Values | None = None
    source: str
    tolerance: float = SOLVED
    reference: Simulator | None = None


@dataclass(frozen=True, kw_only=True)
class Read:
    """A number a SPICE simulator's results hold, by its name in each.

    `ngspice` names a vector of the plot the `plot`-th command wrote, and `xyce` a
    column of what Xyce prints; `index` picks a point of a sweep or a response, and
    `time` a time a transient's values are interpolated at. Xyce prints a complex
    value's real and imaginary parts apart, and its noise as a root density, which
    `power` raises.
    """

    ngspice: str
    xyce: str | None = None
    index: int = 0
    time: float | None = None
    plot: int = 0
    complex: bool = False
    power: int = 1
    printed: bool = True
    """Whether Xyce prints the column because the ``.print`` names it, rather than
    of its own, as it prints sensitivities."""


@dataclass(frozen=True, kw_only=True)
class Spice:
    """A fixture's deck for the SPICE simulators, and what each runs and reads of it.

    `deck` holds the circuit's lines, its title first, without analyses, options or
    an end. `ngspice` holds the commands ngspice runs on it, one per plot, and `xyce`
    the statements Xyce runs it with and the kind of its ``.print``; either is a
    `Unsupported` when that simulator has no statement for the row. `temperature` is
    the kelvins every simulator runs at, each where its constants give jolty's
    thermal voltage there.
    """

    deck: tuple[str, ...]
    ngspice: tuple[str, ...] | Unsupported | Documented
    xyce: tuple[tuple[str, ...], str] | Unsupported | Documented
    reads: Mapping[str, Read]
    options: str = ""
    temperature: float = ROOM
    derive: Callable[[Values], Values] | None = None
    """What the row's numbers are, from what the reads found, when they differ."""
    thermal: bool = True
    """Whether Xyce runs where its thermal voltage is jolty's, rather than at the
    temperature itself, for a fixture whose answer is a law of the temperature alone."""


@dataclass(frozen=True, kw_only=True)
class Fixture:
    """A capability, the fixture that exercises it, and its known answer."""

    key: str
    category: Category
    name: str
    known: Known | None
    jolty: Callable[[], Values] | Unsupported
    spice: Spice | Unsupported


# Where each simulator says what it offers, for the rows it has no statement for.

NGSPICE_ANALYSES: Final = (
    "ngspice 47 manual, §11.3 Analyses, lists .AC, .DC, .DISTO, .NOISE, .OP, .PZ, "
    ".SENS, .SP, .TF, .TRAN, transient noise and .PSS"
)
"""The ngspice manual's list of its analyses."""

NGSPICE_SENSITIVITY: Final = (
    "ngspice 47 manual, §11.3.7 .SENS: DC or Small-Signal AC Sensitivity Analysis"
)
"""Where ngspice's manual says which sensitivities it computes."""

XYCE_COMMANDS: Final = (
    "Xyce Reference Guide 7.10, Netlist Commands: .AC, .DC, .HB, .LIN, .MPDE, "
    ".NOISE, .OP, .PCE, .SAMPLING, .EMBEDDEDSAMPLING, .SENS, .STEP, .TRAN and others; "
    "no .PZ, .TF, .DISTO, loop gain or worst case"
)
"""The Xyce Reference Guide's list of its analysis commands."""

JOLTY_ANALYSES: Final = (
    "jolty.plan.Analysis: OperatingPoint, OperatingPoints, DcSweep, Ac, Transient, "
    "Sensitivity, Stability, Noise, PoleZero, TransferFunction"
)
"""jolty's analyses, as `jolty.plan.Analysis` lists them (`Analysis` is checked
against this list by the tests)."""

UNNOISY_TEMPERATURE: Final = "jolty.plan.Analysis has no analysis of noise over time"


def _rejected(error: JoltyError) -> Unsupported:
    """Jolty's refusal of a row, its message on one line."""
    return Unsupported(" ".join(str(error).split()))


# jolty's side.


def deck_circuit(lines: Sequence[str]) -> ir.Circuit:
    """A deck's elements read by jolty's SPICE reader, as the top of a circuit.

    The deck's title is left out, and its other lines become a subcircuit's body.

    Raises:
        Unsupported: When the reader refuses a line, with its message.
    """
    definitions: list[str] = []
    elements: list[str] = []
    within = False
    for line in lines[1:]:
        keyword = line.split()[0].casefold() if line.split() else ""
        within = within or keyword == ".subckt"
        (definitions if within else elements).append(line)
        within = within and keyword != ".ends"
    library = Library()
    try:
        library.read("\n".join(definitions), "fixture.cir")
        library.read("\n".join([".subckt deck", *elements, ".ends"]), "fixture.cir")
        return circuit(library.component("deck"))
    except JoltyError as error:
        raise _rejected(error) from None


def simulation(built: ir.Circuit, temperature: float = ROOM) -> Simulation:
    """A simulation of `built` at `temperature`, in kelvins."""
    return Simulation(built, temperature * K)


def read(found: object, name: str) -> object:
    """The unknown `name` of a result, ``v(net)`` or ``i(source)``, in jolty."""
    if not isinstance(
        found,
        results.OperatingPointResult
        | results.DcSweepResult
        | results.AcResult
        | results.TransientResult,
    ):
        message = f"no unknowns to read in {type(found).__name__}"
        raise TypeError(message)
    kind, inside = name[0], name[2:-1]
    if kind == "i":
        return found.current(inside.upper())
    return found.voltage(inside)


def solved(built: ir.Circuit, analysis: Analysis, names: Sequence[str]) -> Values:
    """Each unknown `names` gives of `built` at its operating point, or each point.

    A name ``v(out)@2`` reads the third point of a sweep or a response.
    """
    found = simulation(built).run(analysis, TIGHT)
    values: Values = {}
    for name in names:
        unknown, _, at = name.partition("@")
        value = np.asarray(read(found, unknown))
        values[name] = complex(value.reshape(-1)[int(at) if at else 0])
    return values


def sampled(waveforms: results.TransientResult, names: Sequence[str]) -> Values:
    """Each ``v(net)@time`` of `waveforms`, interpolated at its time."""
    values: Values = {}
    for name in names:
        unknown, _, at = name.partition("@")
        trace = np.asarray(read(waveforms, unknown))
        values[name] = complex(np.interp(float(at), waveforms.times, trace))
    return values


def exported(built: ir.Circuit, excited: str | None = None) -> tuple[str, ...]:
    """Jolty's netlist of `built` for ngspice, the deck both SPICE simulators run."""
    transient = excited == "transient"
    lines = export.netlist(
        built, excited=None if transient else excited, transient=transient
    )
    return tuple(lines)


def _reads(names: Sequence[str], **given: object) -> dict[str, Read]:
    """The reads of `names`, each ``v(net)`` or ``i(source)`` and maybe ``@index``."""
    found: dict[str, Read] = {}
    for name in names:
        unknown, _, at = name.partition("@")
        found[name] = Read(
            ngspice=unknown,
            xyce=unknown.upper(),
            index=int(at) if at else 0,
            **given,  # ty: ignore[invalid-argument-type] - the fixtures give Read's fields
        )
    return found


# The circuits.


def divider_deck(supply: float = 5.0) -> tuple[str, ...]:
    """A divider: 1 kΩ over 3 kΩ from a source, its output 3/4 of it."""
    return (
        "* divider",
        f"V1 in 0 {supply!r} AC 1",
        "R1 in out 1k",
        "R2 out 0 3k",
    )


def divider_out(supply: float = 5.0, upper: float = 1e3, lower: float = 3e3) -> float:
    """The divider's output, exactly."""
    return supply * lower / (upper + lower)


@dataclass(frozen=True)
class Op:
    """The same fixture in every simulator: a deck's operating point, read alike."""

    deck: tuple[str, ...]
    names: tuple[str, ...]

    def jolty(self) -> Values:
        """The fixture in jolty, its deck read by the SPICE reader."""
        return solved(deck_circuit(self.deck), OperatingPoint(), self.names)

    def spice(self, **given: object) -> Spice:
        """The fixture in the SPICE simulators."""
        return Spice(
            deck=self.deck,
            ngspice=("op",),
            xyce=((".op",), "dc"),
            reads=_reads(self.names),
            **given,  # ty: ignore[invalid-argument-type] - Spice's own fields
        )


def op_fixture(  # noqa: PLR0913, PLR0917 - a row's names, its fixture and its answer, as a table of them reads
    key: str,
    category: Category,
    name: str,
    deck: tuple[str, ...],
    known: Values | None,
    source: str,
    *,
    reference: Simulator | None = None,
    xyce: Unsupported | Documented | None = None,
) -> Fixture:
    """A row whose fixture is a deck's operating point, in every simulator.

    `xyce` replaces Xyce's run of the deck, where Xyce reads its card otherwise.
    """
    fixture = Op(deck, tuple(known or {}) if reference is None else ())
    spice = fixture.spice()
    if xyce is not None:
        spice = dataclasses.replace(spice, xyce=xyce)
    return Fixture(
        key=key,
        category=category,
        name=name,
        known=Known(values=known, source=source, reference=reference),
        jolty=fixture.jolty,
        spice=spice,
    )


def _builder(name: str) -> tuple[Builder, Net]:
    """A builder of a component `name`, and its ground."""
    c = Builder(name)
    return c, c.port("ground", Role.GROUND)


def rc_step() -> Component:
    """A source stepped from 0 V to 1 V in 10 ns at 1 µs, into 1 kΩ and 1 nF."""
    c, ground = _builder("rc_step")
    step = Piecewise([(1 * us, 0 * V), (1 * us + 10 * ns, 1 * V)])
    source = c.place(devices.voltage_source(step), id="V1")
    resistor = c.place(devices.resistor(1 * kohm), id="R1")
    capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)
    return c.finish()


RC_TAU: Final = 1e3 * 1e-9
"""The RC circuits' time constant, 1 kΩ times 1 nF, in seconds."""

RAMP: Final = (1e-6, 10e-9)
"""When `rc_step`'s step starts, and how long it rises, in seconds."""


def ramped(t: float, tau: float = RC_TAU) -> float:
    """`rc_step`'s output at `t`: an RC's exact response to a linear ramp of 1 V.

    Before the ramp it rests at 0; during it, ``(t' - τ(1 - e^(-t'/τ)))/tr``; after
    it, ``1 - τ(e^(tr/τ) - 1) e^(-t'/τ)/tr``, with ``t'`` the time since it began.
    """
    start, rise = RAMP
    since = t - start
    if since <= 0:
        return 0.0
    if since <= rise:
        return (since - tau * (1 - math.exp(-since / tau))) / rise
    return 1 - tau * (math.exp(rise / tau) - 1) * math.exp(-since / tau) / rise


STEP_TIMES: Final = (1.5e-6, 2e-6, 3e-6, 5e-6)
"""The times the transients read `rc_step`'s output at, in seconds."""


def transient_fixture(key: str, name: str, method: Method) -> Fixture:
    """A row of a transient integrated by `method`, on `rc_step`."""
    built = circuit(rc_step())
    names = tuple(f"v(out)@{time!r}" for time in STEP_TIMES)
    known = {
        name: complex(ramped(time))
        for name, time in zip(names, STEP_TIMES, strict=True)
    }
    analysis = Transient(
        6 * us, max_step=10 * ns, method=method, times=[t * s for t in STEP_TIMES]
    )

    def run() -> Values:
        found = simulation(built).run(analysis, TIGHT)
        return sampled(found, names)

    ngspice_method, ngspice_order = {
        "trapezoidal": ("trap", 2),
        "gear2": ("gear", 2),
        "backward_euler": ("gear", 1),
    }[method]
    xyce_method = "trap" if method == "trapezoidal" else "gear"
    xyce_order = 1 if method == "backward_euler" else 2
    reads = {
        name: Read(ngspice="v(out)", xyce="V(OUT)", time=time)
        for name, time in zip(names, STEP_TIMES, strict=True)
    }
    return Fixture(
        key=key,
        category=Category.ANALYSIS,
        name=name,
        known=Known(
            values=known,
            source="closed form: an RC's response to a 10 ns ramp",
            tolerance=INTEGRATED,
        ),
        jolty=run,
        spice=Spice(
            deck=exported(built, "transient"),
            ngspice=("tran 10n 6u 0 10n",),
            xyce=(
                (
                    ".tran 10n 6u 0 10n",
                    (
                        f".options timeint method={xyce_method} maxord={xyce_order} "
                        f"minord={xyce_order} reltol=1e-6"
                    ),
                ),
                "tran",
            ),
            reads=reads,
            options=f".options method={ngspice_method} maxord={ngspice_order}",
        ),
    )


# The rows.


def analyses() -> list[Fixture]:
    """The analyses' rows."""
    rows: list[Fixture] = []
    deck = divider_deck()
    out = divider_out()
    rows.append(
        op_fixture(
            "op",
            Category.ANALYSIS,
            "Operating point",
            deck,
            {"v(out)": out, "i(v1)": -5.0 / 4e3},
            "closed form: a divider's ratio",
        )
    )

    # Every DC solution: a behavioral current of 1 mA·(v³ - v) out of a node, whose
    # solutions are -1, 0 and 1 V.
    cubic = ("* cubic", "B1 x 0 I={1m*(V(x)*V(x)*V(x) - V(x))}", "R1 x 0 1g")
    conductance = 1e-9

    def every() -> Values:
        found = simulation(deck_circuit(cubic)).run(OperatingPoints(most=8), TIGHT)
        roots = sorted(float(v) for v in found.voltages("x"))
        return {f"v(x)#{index}": complex(root) for index, root in enumerate(roots)}

    # 1e-3·(v³ - v) + 1e-9·v = 0: v = 0 or v² = 1 - 1e-6.
    edge = math.sqrt(1 - conductance / 1e-3)
    rows.append(
        Fixture(
            key="op-every",
            category=Category.ANALYSIS,
            name="Every DC solution",
            known=Known(
                values={"v(x)#0": -edge, "v(x)#1": 0.0, "v(x)#2": edge},
                source="closed form: the roots of a cubic current",
            ),
            jolty=every,
            spice=Spice(
                deck=cubic,
                ngspice=Unsupported(NGSPICE_ANALYSES + "; each finds one solution"),
                xyce=Unsupported(XYCE_COMMANDS + "; .OP finds one solution"),
                reads={},
            ),
        )
    )

    levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    sweep_names = tuple(f"v(out)@{index}" for index in range(len(levels)))
    rows.append(
        Fixture(
            key="dc-source",
            category=Category.ANALYSIS,
            name="DC sweep of a source",
            known=Known(
                values={
                    n: divider_out(level)
                    for n, level in zip(sweep_names, levels, strict=True)
                },
                source="closed form: a divider's ratio",
            ),
            jolty=lambda: solved(
                deck_circuit(deck),
                DcSweep("V1", [level * V for level in levels]),
                sweep_names,
            ),
            spice=Spice(
                deck=deck,
                ngspice=("dc V1 0 5 1",),
                xyce=((".dc V1 0 5 1",), "dc"),
                reads=_reads(sweep_names),
            ),
        )
    )

    resistances = [1e3, 2e3, 3e3]
    swept = tuple(f"v(out)@{index}" for index in range(len(resistances)))
    rows.append(
        Fixture(
            key="dc-resistance",
            category=Category.ANALYSIS,
            name="DC sweep of a resistance",
            known=Known(
                values={
                    n: divider_out(lower=r)
                    for n, r in zip(swept, resistances, strict=True)
                },
                source="closed form: a divider's ratio",
            ),
            jolty=lambda: solved(
                deck_circuit(deck),
                DcSweep("R2", [r * ohm for r in resistances], parameter="resistance"),
                swept,
            ),
            spice=Spice(
                deck=deck,
                ngspice=("dc R2 1k 3k 1k",),
                xyce=((".dc R2:R 1k 3k 1k",), "dc"),
                reads=_reads(swept),
            ),
        )
    )

    rows.extend(sweeps())
    return rows


def sweeps() -> list[Fixture]:
    """The rows of sweeps over the temperature and over two sources."""
    rows: list[Fixture] = []
    # A resistor of 1 kΩ drifting 1 %/K over a fixed 3 kΩ, at -40, 25 and 90 °C.
    # A resistor's drift is a law of its temperature, not of a thermal voltage: Xyce,
    # whose constants are furthest from jolty's, sweeps the temperatures themselves.
    drifting = ("* drifting", "V1 in 0 5", "R1 in out 1k TC1=0.01", "R2 out 0 3k")
    temperatures = [-40.0, 25.0, 90.0]
    hot = tuple(f"v(out)@{index}" for index in range(len(temperatures)))
    kelvins = [t + 273.15 for t in temperatures]

    def over_temperature() -> Values:
        built = deck_circuit(drifting)
        grid = Grid([Axis.temperature([k * K for k in kelvins])])
        found = simulation(built).batch(
            {"op": OperatingPoint()}, grid, traces=["out"], options=TIGHT
        )
        values = np.asarray(found.traces[0].values_of("out"))
        return {name: complex(values[index][0]) for index, name in enumerate(hot)}

    first, last = (ngspice.spice_celsius(k) for k in (kelvins[0], kelvins[-1]))
    step = (last - first) / 2
    rows.append(
        Fixture(
            key="dc-temperature",
            category=Category.ANALYSIS,
            name="DC sweep of the temperature",
            known=Known(
                values={
                    n: divider_out(upper=1e3 * (1 + 0.01 * (t - 27.0)))
                    for n, t in zip(hot, temperatures, strict=True)
                },
                source="closed form: a divider whose upper resistor drifts linearly",
            ),
            jolty=over_temperature,
            spice=Spice(
                deck=drifting,
                ngspice=(f"dc temp {first!r} {last!r} {step!r}",),
                xyce=((".dc temp -40 90 65",), "dc"),
                reads=_reads(hot),
                thermal=False,
            ),
        )
    )

    # Two sources summed by equal resistors: the output is their mean.
    summed = ("* summed", "V1 a 0 0", "V2 b 0 0", "R1 a out 1k", "R2 b out 1k")
    pairs = [(v1, v2) for v1 in (0.0, 1.0, 2.0) for v2 in (0.0, 1.0)]
    names = tuple(f"v(out)@{index}" for index in range(len(pairs)))

    def nested() -> Values:
        grid = Grid(
            [
                Axis("V1", "waveform.dc", [0 * V, 1 * V, 2 * V]),
                Axis("V2", "waveform.dc", [0 * V, 1 * V]),
            ]
        )
        found = simulation(deck_circuit(summed)).batch(
            {"op": OperatingPoint()}, grid, traces=["out"], options=TIGHT
        )
        values = np.asarray(found.traces[0].values_of("out"))
        return {name: complex(values[index][0]) for index, name in enumerate(names)}

    # ngspice and Xyce sweep the first source fastest; jolty's grid its last axis.
    spice_order = [i * 2 + j for j in range(2) for i in range(3)]
    rows.append(
        Fixture(
            key="dc-two-sources",
            category=Category.ANALYSIS,
            name="DC sweep of two sources, nested",
            known=Known(
                values={n: (a + b) / 2 for n, (a, b) in zip(names, pairs, strict=True)},
                source="closed form: the mean of two sources",
            ),
            jolty=nested,
            spice=Spice(
                deck=summed,
                ngspice=("dc V1 0 2 1 V2 0 1 1",),
                xyce=((".dc V1 0 2 1 V2 0 1 1",), "dc"),
                reads={
                    name: Read(
                        ngspice="v(out)", xyce="V(OUT)", index=spice_order.index(index)
                    )
                    for index, name in enumerate(names)
                },
            ),
        )
    )
    rows.extend(small_signal())
    return rows


RC_DECK: Final = ("* low pass", "V1 in 0 0 AC 1", "R1 in out 1k", "C1 out 0 1n")
"""A low pass of 1 kΩ and 1 nF, its corner at 159 kHz, excited by `V1`."""

CORNER: Final = 1 / (2 * math.pi * RC_TAU)
"""`RC_DECK`'s corner frequency, in hertz."""

FREQUENCIES: Final = (1e4, 1e5, 1e6)
"""The frequencies the small-signal fixtures read, a decade apart."""


def low_pass(frequency: float) -> complex:
    """`RC_DECK`'s response at `frequency`: 1/(1 + jωRC)."""
    return 1 / (1 + 1j * 2 * math.pi * frequency * RC_TAU)


def small_signal() -> list[Fixture]:
    """The small-signal analyses' rows."""
    rows: list[Fixture] = []
    names = tuple(f"v(out)@{index}" for index in range(len(FREQUENCIES)))
    frequencies = decade(1e4 * Hz, 1e6 * Hz, 1)
    rows.append(
        Fixture(
            key="ac",
            category=Category.ANALYSIS,
            name="AC",
            known=Known(
                values={
                    n: low_pass(f) for n, f in zip(names, FREQUENCIES, strict=True)
                },
                source="closed form: a low pass's 1/(1 + jωRC)",
            ),
            jolty=lambda: solved(deck_circuit(RC_DECK), Ac("V1", frequencies), names),
            spice=Spice(
                deck=RC_DECK,
                ngspice=("ac dec 1 1e4 1e6",),
                xyce=((".ac dec 1 1e4 1e6",), "ac"),
                reads=_reads(names, complex=True),
            ),
        )
    )

    # The RC's output noise: the resistor's 4kTR, filtered: 4kTR/(1 + (ωRC)²).
    def noise() -> Values:
        found = simulation(deck_circuit(RC_DECK)).run(
            Noise("out", frequencies, source="V1"), TIGHT
        )
        return {
            n: complex(d) for n, d in zip(noise_names, found.densities, strict=True)
        }

    noise_names = tuple(f"onoise@{index}" for index in range(len(FREQUENCIES)))
    rows.append(
        Fixture(
            key="noise",
            category=Category.ANALYSIS,
            name="Noise",
            known=Known(
                values={
                    n: 4 * BOLTZMANN * ROOM * 1e3 * abs(low_pass(f)) ** 2
                    for n, f in zip(noise_names, FREQUENCIES, strict=True)
                },
                source="closed form: a resistor's 4kTR through a low pass",
            ),
            jolty=noise,
            spice=Spice(
                deck=RC_DECK,
                ngspice=(
                    "set sqrnoise\nnoise v(out) V1 dec 1 1e4 1e6\nsetplot previous",
                ),
                xyce=((".noise V(out) V1 dec 1 1e4 1e6",), "noise"),
                reads={
                    name: Read(ngspice="onoise_spectrum", xyce="ONOISE", index=index)
                    for index, name in enumerate(noise_names)
                },
            ),
        )
    )

    def transfer() -> Values:
        found = simulation(deck_circuit(divider_deck())).run(
            TransferFunction("V1", "out"), TIGHT
        )
        return {
            "gain": complex(found.gain),
            "r(in)": complex(found.input_resistance),
            "r(out)": complex(found.output_resistance),
        }

    rows.append(
        Fixture(
            key="tf",
            category=Category.ANALYSIS,
            name="Transfer function",
            known=Known(
                values={"gain": 0.75, "r(in)": 4e3, "r(out)": 750.0},
                source="closed form: a divider's ratio and resistances",
            ),
            jolty=transfer,
            spice=Spice(
                deck=divider_deck(),
                ngspice=("tf v(out) V1",),
                xyce=Unsupported(XYCE_COMMANDS),
                reads={
                    "gain": Read(ngspice="v(transfer_function)"),
                    "r(in)": Read(ngspice="v(v1#input_impedance)"),
                    "r(out)": Read(ngspice="v(output_impedance_at_v(out))"),
                },
            ),
        )
    )

    # A series RLC of 10 Ω, 1 µH and 1 nF: poles at -R/2L ± j√(1/LC - (R/2L)²).
    rlc = ("* rlc", "V1 in 0 0 AC 1", "R1 in mid 10", "L1 mid out 1u", "C1 out 0 1n")
    sigma = 10 / (2 * 1e-6)
    omega = math.sqrt(1 / (1e-6 * 1e-9) - sigma**2)

    def poles() -> Values:
        found = simulation(deck_circuit(rlc)).run(
            PoleZero(source="V1", output="out"), TIGHT
        )
        upper = [p for p in found.poles if p.imag >= 0]
        zeros = 0 if found.zeros is None else len(found.zeros)
        return {"pole": complex(upper[0]), "zeros": complex(zeros)}

    def pair(values: Values) -> Values:
        poles = [v for n, v in values.items() if n.startswith("pole") and v.imag >= 0]
        return {"pole": poles[0] if poles else complex("nan"), "zeros": complex(0)}

    rows.append(
        Fixture(
            key="pz",
            category=Category.ANALYSIS,
            name="Poles and zeros",
            known=Known(
                values={"pole": complex(-sigma, omega), "zeros": 0},
                source="closed form: a series RLC's poles, and no finite zero",
            ),
            jolty=poles,
            spice=Spice(
                deck=rlc,
                ngspice=("pz in 0 out 0 vol pz",),
                xyce=Unsupported(XYCE_COMMANDS),
                reads={
                    "pole1": Read(ngspice="v(pole(1))"),
                    "pole2": Read(ngspice="v(pole(2))"),
                },
                derive=pair,
            ),
        )
    )

    # A loop: an amplifier of gain 10 whose pole is at 1/(2π·1 µs), fed back by 1/2
    # through the probe VP: T = 5/(1 + jωτ).
    loop = (
        "* loop",
        "E1 a 0 0 fb 10",
        "R1 a b 1k",
        "C1 b 0 1n",
        "E2 o 0 b 0 1",
        "VP p o 0",
        "R2 p fb 1k",
        "R3 fb 0 1k",
    )

    def stable() -> Values:
        found = simulation(deck_circuit(loop)).run(Stability("VP", frequencies), TIGHT)
        return {f"t@{i}": complex(g) for i, g in enumerate(found.gains)}

    rows.append(
        Fixture(
            key="stability",
            category=Category.ANALYSIS,
            name="Loop gain (stability)",
            known=Known(
                values={f"t@{i}": 5 * low_pass(f) for i, f in enumerate(FREQUENCIES)},
                source="closed form: a single-pole loop's 5/(1 + jωτ)",
            ),
            jolty=stable,
            spice=Spice(
                deck=loop,
                ngspice=Unsupported(NGSPICE_ANALYSES + "; none finds a loop gain"),
                xyce=Unsupported(XYCE_COMMANDS),
                reads={},
            ),
        )
    )
    rows.extend(sensitivities())
    rows.extend(transients())
    rows.append(monte_carlo())
    return rows


def sensitivities() -> list[Fixture]:
    """The rows of sensitivities: at DC, over frequency and over time."""
    rows: list[Fixture] = []
    deck = divider_deck()

    def at_dc() -> Values:
        found = simulation(deck_circuit(deck)).run(Sensitivity(["out"]), TIGHT)
        return {
            "d(out)/d(r1)": complex(found.derivative("out", "R1", "resistance")),
            "d(out)/d(r2)": complex(found.derivative("out", "R2", "resistance")),
        }

    rows.append(
        Fixture(
            key="sens-dc",
            category=Category.ANALYSIS,
            name="Sensitivities at DC",
            known=Known(
                values={
                    "d(out)/d(r1)": -5 * 3e3 / 4e3**2,
                    "d(out)/d(r2)": 5 * 1e3 / 4e3**2,
                },
                source="closed form: a divider's derivatives",
            ),
            jolty=at_dc,
            spice=Spice(
                deck=deck,
                ngspice=("sens v(out)",),
                xyce=(
                    (".dc V1 5 5 1", ".sens objfunc={V(out)} param=R1:R,R2:R"),
                    "sens",
                ),
                reads={
                    "d(out)/d(r1)": Read(
                        ngspice="v(r1)", xyce="D_{V(OUT)}/D_R1:R_ADJ", printed=False
                    ),
                    "d(out)/d(r2)": Read(
                        ngspice="v(r2)", xyce="D_{V(OUT)}/D_R2:R_ADJ", printed=False
                    ),
                },
            ),
        )
    )

    # d/dR of 1/(1 + jωRC) is -jωC/(1 + jωRC)².
    def d_ac(frequency: float) -> complex:
        jw = 1j * 2 * math.pi * frequency
        return -jw * 1e-9 / (1 + jw * RC_TAU) ** 2

    names = tuple(f"d(out)/d(r1)@{i}" for i in range(len(FREQUENCIES)))

    def over_frequency() -> Values:
        found = simulation(deck_circuit(RC_DECK)).run(
            Sensitivity(["out"], of=Ac("V1", decade(1e4 * Hz, 1e6 * Hz, 1))), TIGHT
        )
        derivative = np.asarray(found.derivative("out", "R1", "resistance"))
        return {n: complex(d) for n, d in zip(names, derivative, strict=True)}

    rows.append(
        Fixture(
            key="sens-ac",
            category=Category.ANALYSIS,
            name="Sensitivities over frequency",
            known=Known(
                values={n: d_ac(f) for n, f in zip(names, FREQUENCIES, strict=True)},
                source="closed form: a low pass's derivative in its resistance",
            ),
            jolty=over_frequency,
            spice=Spice(
                deck=RC_DECK,
                ngspice=("sens v(out) ac dec 1 1e4 1e6",),
                xyce=Documented(
                    "Xyce Reference Guide 7.10, .SENS: AC sensitivities with "
                    "acobjfunc; no fixture runs it here yet"
                ),
                reads={
                    name: Read(ngspice="v(r1)", index=index)
                    for index, name in enumerate(names)
                },
            ),
        )
    )

    # dv/dR of the ramp response: v depends on τ = RC, so dv/dR = C·dv/dτ.
    def d_tran(time: float) -> float:
        step = 1e-6 * RC_TAU
        return (
            1e-9
            * (ramped(time, RC_TAU + step) - ramped(time, RC_TAU - step))
            / (2 * step)
        )

    times = STEP_TIMES
    tran_names = tuple(f"d(out)/d(r1)@{t!r}" for t in times)

    def over_time() -> Values:
        built = circuit(rc_step())
        analysis = Transient(6 * us, max_step=10 * ns, times=[t * s for t in times])
        found = simulation(built).run(Sensitivity(["out"], of=analysis), TIGHT)
        derivative = np.asarray(found.derivative("out", "R1", "resistance"))
        if not isinstance(found.result, results.TransientResult):
            message = "a transient's sensitivities follow its waveforms"
            raise TypeError(message)
        trace = found.result.times
        return {
            n: complex(np.interp(t, trace, derivative))
            for n, t in zip(tran_names, times, strict=True)
        }

    rows.append(
        Fixture(
            key="sens-tran",
            category=Category.ANALYSIS,
            name="Sensitivities over time",
            known=Known(
                values={n: d_tran(t) for n, t in zip(tran_names, times, strict=True)},
                source="closed form: the ramp response's derivative in its time "
                "constant, by a central difference of 1e-6 of it",
                tolerance=INTEGRATED,
            ),
            jolty=over_time,
            spice=Spice(
                deck=exported(circuit(rc_step()), "transient"),
                ngspice=Unsupported(NGSPICE_SENSITIVITY),
                xyce=Documented(
                    "Xyce Reference Guide 7.10, .SENS: transient sensitivities, direct "
                    "and adjoint; no fixture runs it here yet"
                ),
                reads={},
            ),
        )
    )
    return rows


def transients() -> list[Fixture]:
    """The rows of transients, and of the analyses built on them."""
    rows = [
        transient_fixture("tran-trapezoidal", "Transient, trapezoidal", "trapezoidal"),
        transient_fixture("tran-gear2", "Transient, Gear of order 2", "gear2"),
        transient_fixture(
            "tran-backward-euler", "Transient, backward Euler", "backward_euler"
        ),
    ]

    # A low pass driven by a 1 MHz sine of 1 V: its periodic steady state's amplitude
    # at the output is |H(f)|, and its phase H's.
    def driven() -> Component:
        c, ground = _builder("driven")
        sine = Sine(offset=0 * V, amplitude=1 * V, frequency=1e6 * Hz)
        source = c.place(devices.voltage_source(sine), id="V1")
        resistor = c.place(devices.resistor(1 * kohm), id="R1")
        capacitor = c.place(devices.capacitor(1 * nF), id="C1")
        c.connect(source.positive, resistor.a, name="in")
        c.connect(resistor.b, capacitor.a, name="out")
        c.connect(capacitor.b, source.negative, ground)
        return c.finish()

    response = low_pass(1e6)
    quarter = (0.25e-6, 0.5e-6, 0.75e-6, 1e-6)
    names = tuple(f"v(out)@{t!r}" for t in quarter)

    def periodic() -> Values:
        built = circuit(driven())
        analysis = Transient(
            1 * us, max_step=1 * ns, times=[t * s for t in quarter], periodic=True
        )
        found = simulation(built).run(analysis, TIGHT)
        start = float(found.times[0])
        values = np.asarray(found.voltage("out"))
        return {
            n: complex(np.interp(start + t, found.times, values))
            for n, t in zip(names, quarter, strict=True)
        }

    rows.append(
        Fixture(
            key="pss",
            category=Category.ANALYSIS,
            name="Periodic steady state",
            known=Known(
                values={
                    n: abs(response)
                    * math.sin(2 * math.pi * 1e6 * t + cmath.phase(response))
                    for n, t in zip(names, quarter, strict=True)
                },
                source="closed form: a sine through a low pass, settled",
                tolerance=INTEGRATED,
            ),
            jolty=periodic,
            spice=Spice(
                deck=exported(circuit(driven()), "transient"),
                ngspice=Unsupported(
                    "ngspice 47 manual, §11.3.12 .PSS: periodic steady state of "
                    "autonomous circuits, oscillators, only"
                ),
                xyce=Documented(
                    "Xyce Reference Guide 7.10, .HB: harmonic balance; no fixture "
                    "runs it here yet"
                ),
                reads={},
            ),
        )
    )
    return rows


THREE: Final = 3
"""The node voltages a forced device's fixture reads its currents at: its terminals'."""

DECKS: Final = Path(__file__).resolve().parent / "decks"
"""The files the syntax fixtures' decks read."""


def thermal(volts: float) -> float:
    """`e^(v/Vt) - 1` at `ROOM`."""
    return math.expm1(volts / THERMAL_VOLTAGE)


def bjt_currents(
    vbe: float, vbc: float, saturation: float, forward: float, reverse: float
) -> tuple[float, float]:
    """The Ebers and Moll transport currents into the collector and the base."""
    collector = saturation * (
        thermal(vbe) - thermal(vbc)
    ) - saturation / reverse * thermal(vbc)
    base = saturation / forward * thermal(vbe) + saturation / reverse * thermal(vbc)
    return collector, base


def coupled_secondary(frequency: float) -> complex:
    """The coupled inductors' open secondary's voltage, solved exactly."""
    jw = 2j * math.pi * frequency
    r1, l1, l2, r2 = 1.0, 1e-6, 4e-6, 1e9
    mutual = 0.5 * math.sqrt(l1 * l2)
    # (r1 + jωL1)·I1 + jωM·I2 = 1 and jωM·I1 + (r2 + jωL2)·I2 = 0, I2 into the
    # secondary's dotted end; its voltage is -r2·I2.
    determinant = (r1 + jw * l1) * (r2 + jw * l2) - (jw * mutual) ** 2
    secondary = -jw * mutual / determinant
    return -r2 * secondary


def device_fixtures() -> list[Fixture]:
    """The device models' rows."""
    rows: list[Fixture] = []
    at = 1e6
    jw = 2j * math.pi * at
    rows.append(
        ac_fixture(
            "capacitor",
            "Capacitor",
            ("* capacitor", "V1 a 0 0 AC 1", "C1 a 0 1n"),
            {"i(v1)": -jw * 1e-9},
            "closed form: jωC",
        )
    )
    rows.append(
        ac_fixture(
            "inductor",
            "Inductor",
            ("* inductor", "V1 a 0 0 AC 1", "R1 a b 1", "L1 b 0 1u"),
            {"i(v1)": -1 / (1 + jw * 1e-6)},
            "closed form: 1/(R + jωL)",
        )
    )
    rows.append(
        ac_fixture(
            "coupled-inductors",
            "Coupled inductors",
            (
                "* coupled",
                "V1 a 0 0 AC 1",
                "R1 a b 1",
                "L1 b 0 1u",
                "L2 c 0 4u",
                "R2 c 0 1g",
                "K1 L1 L2 0.5",
            ),
            {"v(c)": coupled_secondary(at)},
            "closed form: a transformer's two meshes",
        )
    )
    for key, name, deck, known in (
        ("isource", "Current source", ("* isource", "I1 0 o 1m", "R1 o 0 1k"), 1.0),
        (
            "vcvs",
            "Voltage-controlled voltage source",
            ("* vcvs", "V1 a 0 2", "R1 a 0 1k", "E1 o 0 a 0 3", "R2 o 0 1k"),
            6.0,
        ),
        (
            "vccs",
            "Voltage-controlled current source",
            ("* vccs", "V1 a 0 2", "R1 a 0 1k", "G1 0 o a 0 1m", "R2 o 0 1k"),
            2.0,
        ),
        (
            "cccs",
            "Current-controlled current source",
            ("* cccs", "V1 a 0 2", "R1 a b 1k", "VS b 0 0", "F1 0 o VS 3", "R2 o 0 1k"),
            6.0,
        ),
        (
            "ccvs",
            "Current-controlled voltage source",
            (
                "* ccvs",
                "V1 a 0 2",
                "R1 a b 1k",
                "VS b 0 0",
                "H1 o 0 VS 500",
                "R2 o 0 1k",
            ),
            1.0,
        ),
        (
            "behavioral",
            "Behavioral source",
            (
                "* behavioral",
                "V1 a 0 0.5",
                "R1 a 0 1k",
                "B1 o 0 V={sin(V(a)) + V(a)*V(a)}",
                "R2 o 0 1k",
            ),
            math.sin(0.5) + 0.25,
        ),
    ):
        rows.append(
            op_fixture(key, Category.DEVICE, name, deck, {"v(o)": known}, "closed form")
        )

    diode = ("* diode", "V1 a 0 0.7", "D1 a 0 dd", ".model dd D(IS=1e-14 N=1.2)")
    current = 1e-14 * math.expm1(0.7 / (1.2 * THERMAL_VOLTAGE))
    rows.append(
        op_fixture(
            "diode",
            Category.DEVICE,
            "Diode",
            diode,
            {"i(v1)": -current},
            "closed form: Shockley's law",
        )
    )

    gummel = (
        "* gummel-poon",
        "VC c 0 2",
        "VB b 0 0.7",
        "Q1 c b 0 qq",
        ".model qq NPN(IS=1e-15 BF=100 BR=2)",
    )
    collector, base = bjt_currents(0.7, -1.3, 1e-15, 100, 2)
    rows.append(
        op_fixture(
            "bjt",
            Category.DEVICE,
            "Bipolar transistor, Gummel-Poon",
            gummel,
            {"i(vc)": -collector, "i(vb)": -base},
            (
                "closed form: Ebers and Moll's transport currents, no Early effect or "
                "high injection"
            ),
        )
    )

    vbic = (
        "* vbic",
        "VC c 0 2",
        "VB b 0 0.7",
        "Q1 c b 0 qv",
        (
            ".model qv NPN(LEVEL=4 IS=1e-16 NF=1 NR=1 IBEI=1e-18 NEI=1 IBCI=1e-16 "
            "NCI=1 TNOM=27)"
        ),
    )
    rows.append(
        Fixture(
            key="vbic",
            category=Category.DEVICE,
            name="Bipolar transistor, VBIC 1.2",
            known=Known(
                source=(
                    "none yet: Ebers and Moll's transport form with VBIC's constants "
                    "misses the run of both by 2e-6, to be explained"
                )
            ),
            jolty=Op(vbic, ("i(vc)", "i(vb)")).jolty,
            spice=dataclasses.replace(
                Op(vbic, ("i(vc)", "i(vb)")).spice(),
                xyce=Documented(
                    "Xyce Reference Guide 7.10, BJT levels 11 and 12: VBIC 1.3, "
                    "not 1.2, its card's level not ngspice's 4; the dialect of "
                    "branch xyce writes it"
                ),
            ),
        )
    )

    jfet = (
        "* jfet",
        "VD d 0 5",
        "VG g 0 -1",
        "J1 d g 0 jj",
        ".model jj NJF(VTO=-2 BETA=1e-3 LAMBDA=0.01)",
    )
    rows.append(
        op_fixture(
            "jfet",
            Category.DEVICE,
            "JFET",
            jfet,
            {"i(vd)": -1e-3 * 1.05},
            "closed form: Shichman and Hodges's saturation current",
        )
    )

    square = 1e-4 / 2 * 10 / 2 * 1.3**2
    mos1 = (
        "* mos1",
        "VD d 0 3",
        "VG g 0 2",
        "M1 d g 0 0 mm W=10u L=2u",
        ".model mm NMOS(LEVEL=1 VTO=0.7 KP=1e-4 LAMBDA=0.02)",
    )
    rows.append(
        op_fixture(
            "mos1",
            Category.DEVICE,
            "MOSFET, level 1",
            mos1,
            {"i(vd)": -square * 1.06},
            "closed form: the square law in saturation",
        )
    )
    mos3 = (
        "* mos3",
        "VD d 0 3",
        "VG g 0 2",
        "M1 d g 0 0 mm W=10u L=2u",
        (
            ".model mm NMOS(LEVEL=3 VTO=0.7 KP=1e-4 GAMMA=0 THETA=0 ETA=0 DELTA=0 "
            "VMAX=0 NFS=0)"
        ),
    )
    rows.append(
        op_fixture(
            "mos3",
            Category.DEVICE,
            "MOSFET, level 3",
            mos3,
            {"i(vd)": -square},
            (
                "closed form: level 3 with no body effect, mobility degradation or "
                "velocity saturation, and no substrate doping for its channel-length "
                "modulation, is the square law"
            ),
        )
    )
    rows.extend(built_devices())
    return rows


def ac_fixture(
    key: str, name: str, deck: tuple[str, ...], known: Values, source: str
) -> Fixture:
    """A device's row whose fixture is its response at 1 MHz."""
    names = tuple(known)
    return Fixture(
        key=key,
        category=Category.DEVICE,
        name=name,
        known=Known(values=known, source=source),
        jolty=lambda: solved(deck_circuit(deck), Ac("V1", [1e6 * Hz]), names),
        spice=Spice(
            deck=deck,
            ngspice=("ac lin 1 1e6 1e6",),
            xyce=((".ac lin 1 1e6 1e6",), "ac"),
            reads=_reads(names, complex=True),
        ),
    )


def forced(name: str, device: object, terminals: Mapping[str, float]) -> Component:
    """`device` with each terminal held by a source, ``V<terminal>``, at its level."""
    c, ground = _builder(name)
    placed = c.place(device, id="X1")  # ty: ignore[invalid-argument-type] - a device
    for terminal, level in terminals.items():
        source = c.place(devices.voltage_source(level * V), id=f"V{terminal.upper()}")
        c.connect(source.positive, placed[terminal], name=terminal)
        c.connect(source.negative, ground)
    return c.finish()


def built_fixture(  # noqa: PLR0913 - a row's names, its circuit and what reads it
    key: str,
    name: str,
    built: Component,
    names: tuple[str, ...],
    known: Known,
    *,
    ngspice_note: Unsupported | None = None,
    xyce: Unsupported | Documented,
    temperature: float = ROOM,
) -> Fixture:
    """A row of a device jolty's builder places, its deck jolty's export of it."""
    flat = circuit(built)
    try:
        deck = exported(flat)
    except JoltyError as error:
        deck = ()
        ngspice_note = ngspice_note or _rejected(error)
    return Fixture(
        key=key,
        category=Category.DEVICE,
        name=name,
        known=known,
        jolty=lambda: solved(flat, OperatingPoint(), names),
        spice=Spice(
            deck=deck,
            ngspice=ngspice_note or ("op",),
            xyce=xyce,
            # jolty's export writes a source `VD` as the element `VVD`.
            reads={name: Read(ngspice=f"i(v{name[2:]}") for name in names},
            temperature=temperature,
        ),
    )


def built_devices() -> list[Fixture]:
    """The rows of the devices jolty's SPICE reader does not read, built instead."""
    rows: list[Fixture] = []
    vdmos = devices.vdmos(
        threshold_voltage=2 * V, transconductance_parameter=0.5 * A / V**2
    )
    rows.append(
        built_fixture(
            "vdmos",
            "VDMOS",
            forced("vdmos", vdmos, {"drain": 5.0, "gate": 4.0, "source": 0.0}),
            ("i(vdrain)",),
            Known(
                source="ngspice 47's VDMOS, the model's defining implementation",
                reference=Simulator.NGSPICE_SPARSE,
            ),
            xyce=Documented(
                "Xyce Reference Guide 7.10, MOSFET level 18 (VDMOS), whose card "
                "is not ngspice's; the dialect of branch xyce writes it"
            ),
        )
    )
    bsim3 = devices.bsim3(channel="n", width=10e-6 * m_, length=1e-6 * m_)
    rows.append(
        built_fixture(
            "bsim3",
            "BSIM3v3.3",
            forced(
                "bsim3", bsim3, {"drain": 1.5, "gate": 1.2, "source": 0.0, "bulk": 0.0}
            ),
            ("i(vdrain)",),
            Known(
                source=(
                    "ngspice 47's BSIM3v3.3, Berkeley's code, the model's defining "
                    "implementation"
                ),
                reference=Simulator.NGSPICE_SPARSE,
            ),
            xyce=Documented(
                "Xyce Reference Guide 7.10, MOSFET level 9 (BSIM3v3.2.2); no "
                "fixture runs it: another version than jolty's 3.3"
            ),
            # BSIM3 computes with constants of its own: ngspice runs at the kelvins
            # themselves, as the differential tests run it.
            temperature=ngspice.same_kelvins(ROOM),
        )
    )
    on, off = 100.0, 16e3
    for key, title, device in (
        (
            "memristor-vteam",
            "Memristor, VTEAM",
            devices.voltage_controlled_memristor(
                on_resistance=on * ohm,
                off_resistance=off * ohm,
                on_voltage=-1 * V,
                off_voltage=1 * V,
                on_rate=1e3 / s,
                off_rate=1e3 / s,
                state=0.25,
            ),
        ),
        (
            "memristor-hp",
            "Memristor, HP's linear ion drift",
            devices.current_controlled_memristor(
                on_resistance=on * ohm,
                off_resistance=off * ohm,
                charge=1e-6 * A * s,
                state=0.25,
            ),
        ),
    ):
        resistance = on + 0.25 * (off - on)
        rows.append(
            built_fixture(
                key,
                title,
                forced(key, device, {"positive": 0.5, "negative": 0.0}),
                ("i(vpositive)",),
                Known(
                    values={"i(vpositive)": -0.5 / resistance},
                    source=(
                        "closed form: its resistance at its state, held at an "
                        "operating point"
                    ),
                ),
                ngspice_note=Unsupported(
                    "ngspice 47 manual, §8.2.29 Memristor: XSPICE's memristor "
                    "code model, another law; jolty's export writes it as a "
                    "behavioral current"
                ),
                xyce=Unsupported(
                    "Xyce Reference Guide 7.10, Memristor: levels 2 (TEAM), 3 "
                    "(Yakopcic) and 4 (PEM), not this law"
                ),
            )
        )
    cell = devices.photocell(10e3 * ohm)
    lit = 1e-6 + (5 / 10) ** 0.6 / 10e3
    rows.append(
        built_fixture(
            "photocell",
            "Photocell",
            forced(
                "photocell",
                cell,
                {"a": 1.0, "b": 0.0, "light_positive": 5.0, "light_negative": 0.0},
            ),
            ("i(va)",),
            Known(
                values={"i(va)": -lit}, source="closed form: 1/r_dark + (E/E0)^gamma/r"
            ),
            ngspice_note=Unsupported(
                "ngspice 47 manual, chapter 7 Device Models: no photocell; "
                "jolty's export writes it as a behavioral current"
            ),
            xyce=Unsupported("Xyce Reference Guide 7.10, its devices: no photocell"),
        )
    )

    c, ground = _builder("heat")
    heat = c.place(devices.heat_source(1 * W), id="P1")
    sink = c.place(devices.thermal_resistance(50 * K / W), id="RTH")
    c.connect(heat.b, sink.a, name="hot")
    c.connect(heat.a, sink.b, ground)
    heated = circuit(c.finish())

    def rise() -> Values:
        found = simulation(heated).run(OperatingPoint(), TIGHT)
        return {"rise(hot)": complex(found.rise("hot"))}

    rows.append(
        Fixture(
            key="thermal",
            category=Category.DEVICE,
            name="Thermal network",
            known=Known(values={"rise(hot)": 50.0}, source="closed form: P·R_th"),
            jolty=rise,
            spice=Spice(
                deck=(),
                ngspice=Unsupported(
                    "ngspice 47 manual, chapter 7 Device Models: no thermal "
                    "elements; self-heating inside its VBIC and BSIM models alone"
                ),
                xyce=Unsupported(
                    "Xyce Reference Guide 7.10, its devices: no thermal elements"
                ),
                reads={},
            ),
        )
    )
    rows.extend(missing_devices())
    return rows


def ngspice_only(  # noqa: PLR0913, PLR0917 - a row's names, its deck and its answer, as a table of them reads
    key: str,
    name: str,
    deck: tuple[str, ...],
    ngspice_commands: tuple[str, ...],
    reads: Mapping[str, Read],
    known: Values | None,
    source: str,
    *,
    xyce: tuple[tuple[str, ...], str] | Unsupported | Documented,
    category: Category = Category.DEVICE,
    tolerance: float = SOLVED,
) -> Fixture:
    """A row jolty's reader is given the deck of, as ngspice and Xyce are."""
    return Fixture(
        key=key,
        category=category,
        name=name,
        known=Known(values=known, source=source, tolerance=tolerance)
        if known is not None
        else Known(source=source),
        jolty=lambda: solved(deck_circuit(deck), OperatingPoint(), ()),
        spice=Spice(deck=deck, ngspice=ngspice_commands, xyce=xyce, reads=reads),
    )


def _forced_deck(
    title: str, element: str, card: str, drain: str = "d"
) -> tuple[str, ...]:
    """A transistor's deck, `element` first for a reader to name, at 3 V and 2 V."""
    return (f"* {title}", element, card, f"VD {drain} 0 3", "VG g 0 2")


def missing_devices() -> list[Fixture]:
    """The device rows of ngspice's that jolty lacks, its reader given their decks."""
    rows: list[Fixture] = []
    unjudged = "none yet: the model's QA reference results are to come"
    current = {"i(vd)": Read(ngspice="i(vd)", xyce="I(VD)")}
    for key, name, element, card, xyce in (
        (
            "mos2",
            "MOSFET, level 2",
            "M1 d g 0 0 mm W=10u L=2u",
            ".model mm NMOS(LEVEL=2 VTO=0.7 KP=1e-4)",
            ((".op",), "dc"),
        ),
        (
            "mos6",
            "MOSFET, level 6",
            "M1 d g 0 0 mm W=10u L=2u",
            ".model mm NMOS(LEVEL=6 VTO=0.7)",
            ((".op",), "dc"),
        ),
        (
            "mos9",
            "MOSFET, level 9",
            "M1 d g 0 0 mm W=10u L=2u",
            ".model mm NMOS(LEVEL=9 VTO=0.7 KP=1e-4)",
            Documented("Xyce Reference Guide 7.10: level 9 is BSIM3 in Xyce"),
        ),
        (
            "bsim4",
            "BSIM4 4.5",
            "M1 d g 0 0 mm W=10u L=1u",
            ".model mm NMOS(LEVEL=14 VERSION=4.5)",
            Documented(
                "Xyce Reference Guide 7.10, MOSFET level 14: BSIM4 4.6.1, 4.7.0 "
                "and 4.8.2, not 4.5"
            ),
        ),
        (
            "bsimsoi",
            "BSIM-SOI 4",
            "M1 d g 0 0 mm W=10u L=1u",
            ".model mm NMOS(LEVEL=10)",
            Documented("Xyce Reference Guide 7.10, MOSFET levels 10 and 70: BSIM-SOI"),
        ),
        (
            "hisim2",
            "HiSIM2",
            "M1 d g 0 0 mm W=10u L=1u",
            ".model mm NMOS(LEVEL=68)",
            Unsupported("Xyce Reference Guide 7.10, its MOSFET levels: no HiSIM"),
        ),
        (
            "hicum",
            "HICUM/L2",
            "Q1 d g 0 qq",
            ".model qq NPN(LEVEL=8)",
            Documented(
                "Xyce Reference Guide 7.10, BJT level 234: HICUM/L2 through ADMS"
            ),
        ),
        (
            "mesfet",
            "MESFET, Statz",
            "Z1 d g 0 zz",
            ".model zz NMF(LEVEL=1 VTO=-2 BETA=1e-3)",
            Documented("Xyce Reference Guide 7.10, MESFET (Z) level 1"),
        ),
        (
            "jfet2",
            "JFET, level 2 (Parker and Skellern)",
            "J1 d g 0 jj",
            ".model jj NJF(LEVEL=2)",
            Documented(
                "Xyce Reference Guide 7.10, JFET level 2 (Shockley), another model"
            ),
        ),
    ):
        rows.append(
            ngspice_only(
                key,
                name,
                _forced_deck(name, element, card),
                ("op",),
                current,
                None,
                unjudged,
                xyce=xyce,
            )
        )

    # A pulse through a matched line of 50 Ω and 5 ns: half the pulse, 5 ns late.
    def pulse(t: float) -> float:
        return (
            min(max(t / 1e-9, 0.0), 1.0)
            if t < 11e-9
            else max(1 - (t - 11e-9) / 1e-9, 0.0)
        )

    times = (5.5e-9, 8e-9, 14e-9)
    line_reads = {
        f"v(out)@{t!r}": Read(ngspice="v(out)", xyce="V(OUT)", time=t) for t in times
    }
    line_known: Values = {f"v(out)@{t!r}": 0.5 * pulse(t - 5e-9) for t in times}
    source = ("V1 s 0 PULSE(0 1 0 1n 1n 10n 100n)", "R1 s in 50", "R2 out 0 50")
    for key, name, element, extra in (
        ("tline", "Lossless transmission line", "T1 in 0 out 0 Z0=50 TD=5n", ()),
        (
            "ltra",
            "Lossy transmission line (LTRA), lossless",
            "O1 in 0 out 0 lmod",
            (".model lmod LTRA(R=0 G=0 L=250n C=100p LEN=1)",),
        ),
    ):
        rows.append(
            ngspice_only(
                key,
                name,
                (f"* {name}", element, *source, *extra),
                ("tran 0.05n 20n 0 0.05n",),
                line_reads,
                line_known,
                "closed form: a matched line delays half its source's pulse",
                xyce=((".tran 0.05n 20n 0 0.05n",), "tran"),
                tolerance=INTEGRATED,
            )
        )
    rows.append(
        ngspice_only(
            "urc",
            "Distributed RC line (URC)",
            (
                "* urc",
                "U1 a out 0 umod L=1m",
                ".model umod URC(RPERL=1k CPERL=1p)",
                "V1 a 0 1",
                "R2 out 0 1",
            ),
            ("op",),
            {"v(out)": Read(ngspice="v(out)", xyce="V(OUT)")},
            {"v(out)": 0.5},
            "closed form: the line's DC resistance, RPERL·L, over a load of the same",
            xyce=Unsupported("Xyce Reference Guide 7.10, its devices: no URC"),
        )
    )
    rows.append(
        ngspice_only(
            "xspice",
            "XSPICE code models",
            (
                "* xspice",
                "A1 a o g1",
                ".model g1 gain(gain=3)",
                "V1 a 0 1",
                "R1 o 0 1k",
            ),
            ("op",),
            {"v(o)": Read(ngspice="v(o)", xyce="V(O)")},
            {"v(o)": 3.0},
            "closed form: a gain block's output",
            xyce=Unsupported("Xyce Reference Guide 7.10: no XSPICE code models"),
        )
    )
    rows.append(
        Fixture(
            key="osdi",
            category=Category.DEVICE,
            name="Verilog-A models through OSDI",
            known=None,
            jolty=lambda: solved(
                deck_circuit(("* osdi", "N1 a 0 resistor_va", "V1 a 0 1")),
                OperatingPoint(),
                (),
            ),
            spice=Spice(
                deck=(),
                ngspice=Documented(
                    "ngspice 47 manual, §9.2 OSDI/OpenVAF: a model compiled by "
                    "OpenVAF, which is not installed here"
                ),
                xyce=Unsupported(
                    "Xyce Reference Guide 7.10: no OSDI; Verilog-A only as "
                    "ADMS-compiled C++ models built into it"
                ),
                reads={},
            ),
        )
    )
    return rows


def syntax() -> list[Fixture]:
    """The rows of SPICE's syntax, each a deck every simulator reads."""
    out = {"v(out)": 2.0}
    return [
        op_fixture(
            "subckt",
            Category.SYNTAX,
            "Subcircuits with parameters",
            (
                "* subcircuit",
                ".subckt half a b params: r=1k",
                "R1 a b {r}",
                "R2 b 0 {r}",
                ".ends",
                "V1 in 0 4",
                "X1 in out half r=2k",
            ),
            out,
            "closed form: a halving divider",
        ),
        op_fixture(
            "param",
            Category.SYNTAX,
            ".param reading other parameters",
            (
                "* parameters",
                ".param upper=1k lower={3*upper}",
                "V1 in 0 {8/3}",
                "R1 in out {upper}",
                "R2 out 0 {lower}",
            ),
            out,
            "closed form: a divider",
        ),
        op_fixture(
            "expressions",
            Category.SYNTAX,
            "Expressions with functions",
            (
                "* expressions",
                "V1 in 0 {sqrt(16)}",
                "R1 in out {exp(0)*1k}",
                "R2 out 0 {max(1k, 0.5k)}",
            ),
            out,
            "closed form: a divider",
        ),
        op_fixture(
            "func",
            Category.SYNTAX,
            ".func",
            (
                "* functions",
                ".func twice(x) {2*x}",
                "V1 in 0 {twice(2)}",
                "R1 in out 1k",
                "R2 out 0 1k",
            ),
            out,
            "closed form: a divider",
        ),
        op_fixture(
            "lib",
            Category.SYNTAX,
            ".lib sections",
            (
                "* sections",
                f".lib '{DECKS / 'sections.lib'}' low",
                "V1 in 0 {8/3}",
                "R1 in out 1k",
                "R2 out 0 {lower}",
            ),
            out,
            "closed form: a divider",
        ),
        op_fixture(
            "include",
            Category.SYNTAX,
            ".include",
            (
                "* includes",
                f".include '{DECKS / 'included.cir'}'",
                "V1 in 0 {8/3}",
                "R1 in out 1k",
                "R2 out 0 {lower}",
            ),
            out,
            "closed form: a divider",
        ),
        op_fixture(
            "poly",
            Category.SYNTAX,
            "POLY sources",
            (
                "* poly",
                "V1 a 0 0.5",
                "R1 a 0 1k",
                "E1 out 0 POLY(1) a 0 1 2 3",
                "R2 out 0 1k",
            ),
            {"v(out)": 2.75},
            "closed form: 1 + 2v + 3v²",
        ),
        op_fixture(
            "table",
            Category.SYNTAX,
            "TABLE sources",
            (
                "* table",
                "V1 a 0 0.5",
                "R1 a 0 1k",
                "E1 out 0 TABLE {V(a)} = (0,0) (1,4)",
                "R2 out 0 1k",
            ),
            out,
            "closed form: a straight line between two points",
        ),
        op_fixture(
            "ako",
            Category.SYNTAX,
            "Model cards inheriting by AKO",
            (
                "* ako",
                "V1 a 0 0.7",
                "D1 a 0 d2",
                ".model d1 D(IS=1e-14 N=1)",
                ".model d2 AKO:d1 D(N=1.2)",
            ),
            {"i(v1)": -1e-14 * math.expm1(0.7 / (1.2 * THERMAL_VOLTAGE))},
            "closed form: Shockley's law with the inherited saturation current",
        ),
    ]


def outputs() -> list[Fixture]:
    """The rows of outputs: SPICE's raw file, and a measurement of a waveform."""
    rows = [
        Fixture(
            key="raw",
            category=Category.OUTPUT,
            name="SPICE raw files",
            known=Known(
                values={"v(out)": divider_out()},
                source="closed form: a divider's ratio, read back from the raw file",
            ),
            jolty=Unsupported(
                "jolty.files writes results files as NumPy .npz archives, its "
                "module says; no SPICE raw file"
            ),
            spice=Spice(
                deck=divider_deck(),
                ngspice=("op",),
                xyce=Documented(
                    "Xyce Reference Guide 7.10, the command line's -r: a SPICE "
                    "raw file; no fixture reads it here, the fixtures read its CSV"
                ),
                reads=_reads(("v(out)",)),
            ),
        )
    ]

    # When `rc_step`'s output first rises through 0.5 V: the root of its closed form.
    low, high = RAMP[0], RAMP[0] + 10 * RC_TAU
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if ramped(middle) < 0.5 else (low, middle)
    crossed = (low + high) / 2
    built = circuit(rc_step())

    def measured() -> Values:
        found = simulation(built).run(Transient(6 * us, max_step=10 * ns), TIGHT)
        requirement = crossing("step", "out", 0.5 * V, edge="rising") >= 0 * s
        (verdict,) = evaluate(Specs([requirement]), ["step"], {"step": found})
        return {"t(cross)": complex(verdict.value or math.nan)}

    rows.append(
        Fixture(
            key="measure",
            category=Category.OUTPUT,
            name="Measurements of a waveform",
            known=Known(
                values={"t(cross)": crossed},
                source="closed form: when an RC's ramp response crosses half its step",
                tolerance=INTEGRATED,
            ),
            jolty=measured,
            spice=Spice(
                deck=exported(built, "transient"),
                ngspice=("tran 10n 6u 0 10n\nmeas tran tcross when v(out)=0.5 rise=1",),
                xyce=(
                    (
                        ".tran 10n 6u 0 10n",
                        ".measure tran tcross when v(out)=0.5 rise=1",
                    ),
                    "tran",
                ),
                reads={
                    "t(cross)": Read(ngspice="printed:tcross", xyce="printed:tcross")
                },
            ),
        )
    )
    return rows


def monte_carlo() -> Fixture:
    """The row of Monte Carlo samples: each one's divider, at its own resistances."""
    c, ground = _builder("toleranced")
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.1), id="R1")
    lower = c.place(devices.resistor(3 * kohm, tolerance=0.1), id="R2")
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)
    built = circuit(c.finish())
    samples = 20

    def sampled_runs() -> Values:
        found = simulation(built).batch(
            {"op": OperatingPoint()},
            MonteCarlo(samples, seed=1),
            traces=["out"],
            options=TIGHT,
        )
        outs = np.asarray(found.traces[0].values_of("out"))[:, 0]
        uppers, lowers = (
            found.parameter("R1 resistance"),
            found.parameter("R2 resistance"),
        )
        closed = 5 * lowers / (uppers + lowers)
        outside = np.sum(np.abs(uppers / 1e3 - 1) > 0.1) + np.sum(
            np.abs(lowers / 3e3 - 1) > 0.1
        )
        return {
            "v(out)-closed": complex(np.max(np.abs(outs - closed))),
            "outside": complex(int(outside)),
            "repeated": complex(samples - len(set(uppers.tolist()))),
        }

    return Fixture(
        key="monte-carlo",
        category=Category.ANALYSIS,
        name="Monte Carlo",
        known=Known(
            values={"v(out)-closed": 0.0, "outside": 0.0, "repeated": 0.0},
            source="closed form: each sample's divider at the resistances it drew, "
            "each within its 10 % and every sample's apart",
        ),
        jolty=sampled_runs,
        spice=Spice(
            deck=(),
            ngspice=Documented(
                "ngspice 47 manual, §18.5 Monte-Carlo Simulation: through its "
                "control language; no fixture runs it here yet"
            ),
            xyce=Documented(
                "Xyce Reference Guide 7.10, .SAMPLING; no fixture runs it here yet"
            ),
            reads={},
        ),
    )


def fixtures() -> list[Fixture]:
    """Every row of the matrix, in its order."""
    return [*analyses(), *device_fixtures(), *syntax(), *outputs()]
