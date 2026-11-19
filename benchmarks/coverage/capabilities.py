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
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np
from simulators import Simulator, Unsupported

from jolty import Builder, Component, Role, circuit, devices, files, ir, results
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
    PeriodicSteadyState,
    PoleZero,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
    decade,
)
from jolty.report import OperatingPointReport
from jolty.runner import Built, Job
from jolty.runner import run as run_plan
from jolty.settings import Settings
from jolty.simulate import Simulation
from jolty.specs import Specs, crossing, distortion, evaluate
from jolty.spice import Library, export, gf180mcu, kits, ngspice, sg13g2, sky130
from jolty.spice.library import FUNCTION, INITIAL
from jolty.units import (
    A,
    Hz,
    K,
    V,
    W,
    fF,
    kHz,
    kohm,
    ms,
    nF,
    ns,
    ohm,
    ps,
    s,
    us,
)
from jolty.units import (
    m as m_,
)
from jolty.waveforms import Piecewise, Pulse, Sine, Waveform

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

VBIC_BOLTZMANN: Final = 1.380662e-23
"""VBIC 1.2's Boltzmann constant, in joules per kelvin, fixed by its reference code."""

VBIC_CHARGE: Final = 1.602189e-19
"""VBIC 1.2's elementary charge, in coulombs, paired with `VBIC_BOLTZMANN`."""

VBIC_INTRINSIC_RESISTANCE: Final = 0.1
"""VBIC 1.2's default intrinsic resistances (`RCI`, `RBI`, `RBP`), in ohms."""


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
    compatibility: ngspice.Compatibility | None = None
    """ngspice compatibility mode (``ps`` or ``hsa``), when a fixture needs it."""
    libraries: tuple[Path, ...] = ()
    """Files ngspice includes via ``.include`` before the deck's body."""
    sections: tuple[tuple[Path, str], ...] = ()
    """Library sections ngspice reads via ``.lib`` before the deck's body."""


@dataclass(frozen=True, kw_only=True)
class Fixture:
    """A capability, the fixture that exercises it, and its known answer."""

    key: str
    category: Category
    name: str
    known: Known | None
    jolty: Callable[[], Values] | Unsupported | Documented
    spice: Spice | Unsupported


# Where each simulator says what it offers, for the rows it has no statement for.

NGSPICE_GUIDE: Final = "ngspice 47 manual"
"""Citation prefix for ngspice 47 manual entries."""

NGSPICE_ANALYSES: Final = (
    f"{NGSPICE_GUIDE}, §11.3 Analyses, lists .AC, .DC, .DISTO, .NOISE, .OP, .PZ, "
    ".SENS, .SP, .TF, .TRAN, transient noise and .PSS"
)
"""The ngspice manual's list of its analyses."""

NGSPICE_SENSITIVITY: Final = (
    f"{NGSPICE_GUIDE}, §11.3.7 .SENS: DC or Small-Signal AC Sensitivity Analysis"
)
"""Where ngspice's manual says which sensitivities it computes."""

XYCE_GUIDE: Final = "Xyce Reference Guide 7.10"
"""Citation prefix for Xyce Reference Guide entries."""

XYCE_COMMANDS: Final = (
    f"{XYCE_GUIDE}, Netlist Commands: .AC, .DC, .HB, .LIN, .MPDE, "
    ".NOISE, .OP, .PCE, .SAMPLING, .EMBEDDEDSAMPLING, .SENS, .STEP, .TRAN and others; "
    "no .PZ, .TF, .DISTO, loop gain or worst case"
)
"""The Xyce Reference Guide's list of its analysis commands."""

_PENDING: Final = "; no fixture runs it here yet"


def _ngspice_documented(detail: str, *, pending: bool = False) -> Documented:
    """Formats an ngspice manual `Documented` note with optional pending suffix."""
    return Documented(f"{NGSPICE_GUIDE}{detail}{_PENDING if pending else ''}")


def _ngspice_unsupported(detail: str) -> Unsupported:
    """Formats an ngspice manual `Unsupported` note."""
    return Unsupported(f"{NGSPICE_GUIDE}{detail}")


def _xyce_documented(detail: str, *, pending: bool = False) -> Documented:
    """Formats a Xyce Reference Guide `Documented` note with optional pending suffix."""
    return Documented(f"{XYCE_GUIDE}{detail}{_PENDING if pending else ''}")


def _xyce_unsupported(detail: str) -> Unsupported:
    """Formats a Xyce Reference Guide `Unsupported` note."""
    return Unsupported(f"{XYCE_GUIDE}{detail}")


def _unrun_spice(
    ngspice_note: Unsupported | Documented, xyce_note: Unsupported | Documented
) -> Spice:
    """Returns a `Spice` entry for rows not executed in ngspice or Xyce."""
    return Spice(deck=(), ngspice=ngspice_note, xyce=xyce_note, reads={})


JOLTY_ANALYSES: Final = (
    "jolty.plan.Analysis: OperatingPoint, OperatingPoints, DcSweep, Ac, Transient, "
    "PeriodicSteadyState, Sensitivity, Mismatch, Stability, Noise, PoleZero, "
    "TransferFunction"
)
"""jolty's analyses, as `jolty.plan.Analysis` lists them (`Analysis` is checked
against this list by the tests)."""

UNNOISY_TEMPERATURE: Final = "jolty.plan.Analysis has no analysis of noise over time"


def _rejected(error: JoltyError) -> Unsupported:
    """Jolty's refusal of a row, its message on one line."""
    return Unsupported(" ".join(str(error).split()))


# jolty's side.


def deck_with_state(
    lines: Sequence[str],
) -> tuple[ir.Circuit, dict[str, float], dict[str, float]]:
    """Reads `lines` into a circuit plus its top-level ``.ic`` and ``.nodeset`` maps.

    The deck's title is left out, its subcircuits and the statements a library reads
    outside them, its functions, initial conditions and nodesets, stay outside, and its
    other lines become a subcircuit's body.

    Raises:
        Unsupported: When the reader refuses a line, with its message.
    """
    definitions: list[str] = []
    elements: list[str] = []
    within = False
    for line in lines[1:]:
        keyword = line.split()[0].casefold() if line.split() else ""
        within = within or keyword == ".subckt"
        outside = within or keyword in {FUNCTION, *INITIAL}
        (definitions if outside else elements).append(line)
        within = within and keyword != ".ends"
    library = Library()
    try:
        library.read("\n".join(definitions), "fixture.cir")
        library.read("\n".join([".subckt deck", *elements, ".ends"]), "fixture.cir")
        return (
            circuit(library.component("deck")),
            dict(library.initial_conditions),
            dict(library.nodesets),
        )
    except JoltyError as error:
        raise _rejected(error) from None


def deck_circuit(lines: Sequence[str]) -> ir.Circuit:
    """A deck's elements read by jolty's SPICE reader, as the top of a circuit.

    Raises:
        Unsupported: When the reader refuses a line, with its message.
    """
    built, _, _ = deck_with_state(lines)
    return built


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
    temperature: float = ROOM,
) -> Fixture:
    """A row whose fixture is a deck's operating point, in every simulator.

    `xyce` replaces Xyce's run of the deck, where Xyce reads its card otherwise.
    """
    fixture = Op(deck, tuple(known or {}) if reference is None else ())
    spice = fixture.spice(temperature=temperature)
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


def _rc_circuit(name: str, waveform: Waveform) -> Component:
    """A source driven by `waveform` into 1 kΩ and 1 nF."""
    c, ground = _builder(name)
    source = c.place(devices.voltage_source(waveform), name="V1")
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)
    return c.finish()


def rc_step() -> Component:
    """A source stepped from 0 V to 1 V in 10 ns at 1 µs, into 1 kΩ and 1 nF."""
    return _rc_circuit(
        "rc_step", Piecewise([(1 * us, 0 * V), (1 * us + 10 * ns, 1 * V)])
    )


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
        6 * us, max_step=10 * ns, method=method, breakpoints=[t * s for t in STEP_TIMES]
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
    deck = divider_deck()
    # Every DC solution: a behavioral current of 1 mA·(v³ - v) out of a node, whose
    # solutions are -1, 0 and 1 V.
    cubic = ("* cubic", "B1 x 0 I={1m*(V(x)*V(x)*V(x) - V(x))}", "R1 x 0 1g")
    conductance = 1e-9

    def every() -> Values:
        found = simulation(deck_circuit(cubic)).run(
            OperatingPoints(max_solutions=8), TIGHT
        )
        roots = sorted(float(v) for v in found.voltages("x"))
        return {f"v(x)#{index}": complex(root) for index, root in enumerate(roots)}

    # 1e-3·(v³ - v) + 1e-9·v = 0: v = 0 or v² = 1 - 1e-6.
    edge = math.sqrt(1 - conductance / 1e-3)
    levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    sweep_names = tuple(f"v(out)@{index}" for index in range(len(levels)))
    resistances = [1e3, 2e3, 3e3]
    swept = tuple(f"v(out)@{index}" for index in range(len(resistances)))
    return [
        op_fixture(
            "op",
            Category.ANALYSIS,
            "Operating point",
            deck,
            {"v(out)": divider_out(), "i(v1)": -5.0 / 4e3},
            "closed form: a divider's ratio",
        ),
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
        ),
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
        ),
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
        ),
        *sweeps(),
    ]


def _batch_out(deck: tuple[str, ...], grid: Grid, names: tuple[str, ...]) -> Values:
    """Runs a batched operating point over `grid` on `deck` and reads ``out``."""
    found = simulation(deck_circuit(deck)).batch(
        {"op": OperatingPoint()}, grid, traces=["out"], options=TIGHT
    )
    values = np.asarray(found.traces[0].values_of("out"))
    return {name: complex(values[index][0]) for index, name in enumerate(names)}


def sweeps() -> list[Fixture]:
    """The rows of sweeps over the temperature and over two sources."""
    # A resistor of 1 kΩ drifting 1 %/K over a fixed 3 kΩ, at -40, 25 and 90 °C.
    # A resistor's drift is a law of its temperature, not of a thermal voltage: Xyce,
    # whose constants are furthest from jolty's, sweeps the temperatures themselves.
    drifting = ("* drifting", "V1 in 0 5", "R1 in out 1k TC1=0.01", "R2 out 0 3k")
    temperatures = [-40.0, 25.0, 90.0]
    hot = tuple(f"v(out)@{index}" for index in range(len(temperatures)))
    kelvins = [t + 273.15 for t in temperatures]
    first, last = (ngspice.option_temperature(k) for k in (kelvins[0], kelvins[-1]))
    step = (last - first) / 2
    # Two sources summed by equal resistors: the output is their mean.
    summed = ("* summed", "V1 a 0 0", "V2 b 0 0", "R1 a out 1k", "R2 b out 1k")
    pairs = [(v1, v2) for v1 in (0.0, 1.0, 2.0) for v2 in (0.0, 1.0)]
    names = tuple(f"v(out)@{index}" for index in range(len(pairs)))
    # ngspice and Xyce sweep the first source fastest; jolty's grid its last axis.
    spice_order = [i * 2 + j for j in range(2) for i in range(3)]
    return [
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
            jolty=lambda: _batch_out(
                drifting, Grid([Axis.temperature([k * K for k in kelvins])]), hot
            ),
            spice=Spice(
                deck=drifting,
                ngspice=(f"dc temp {first!r} {last!r} {step!r}",),
                xyce=((".dc temp -40 90 65",), "dc"),
                reads=_reads(hot),
                thermal=False,
            ),
        ),
        Fixture(
            key="dc-two-sources",
            category=Category.ANALYSIS,
            name="DC sweep of two sources, nested",
            known=Known(
                values={n: (a + b) / 2 for n, (a, b) in zip(names, pairs, strict=True)},
                source="closed form: the mean of two sources",
            ),
            jolty=lambda: _batch_out(
                summed,
                Grid(
                    [
                        Axis("V1", "waveform.dc", [0 * V, 1 * V, 2 * V]),
                        Axis("V2", "waveform.dc", [0 * V, 1 * V]),
                    ]
                ),
                names,
            ),
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
        ),
        *small_signal(),
    ]


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
    names = tuple(f"v(out)@{index}" for index in range(len(FREQUENCIES)))
    frequencies = decade(1e4 * Hz, 1e6 * Hz, 1)
    noise_names = tuple(f"onoise@{index}" for index in range(len(FREQUENCIES)))

    # The RC's output noise: the resistor's 4kTR, filtered: 4kTR/(1 + (ωRC)²).
    def noise() -> Values:
        found = simulation(deck_circuit(RC_DECK)).run(
            Noise("out", frequencies, source="V1"), TIGHT
        )
        return {
            n: complex(d) for n, d in zip(noise_names, found.densities, strict=True)
        }

    def transfer() -> Values:
        found = simulation(deck_circuit(divider_deck())).run(
            TransferFunction("V1", "out"), TIGHT
        )
        return {
            "gain": complex(found.gain),
            "r(in)": complex(found.input_resistance),
            "r(out)": complex(found.output_resistance),
        }

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

    return [
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
        ),
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
        ),
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
        ),
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
        ),
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
        ),
        *sensitivities(),
        *transients(),
        monte_carlo(),
        *extended_analyses(),
    ]


def sensitivities() -> list[Fixture]:
    """The rows of sensitivities: at DC, over frequency and over time."""
    deck = divider_deck()

    def at_dc() -> Values:
        found = simulation(deck_circuit(deck)).run(Sensitivity(["out"]), TIGHT)
        return {
            "d(out)/d(r1)": complex(found.derivative("out", "R1", "resistance")),
            "d(out)/d(r2)": complex(found.derivative("out", "R2", "resistance")),
        }

    # d/dR of 1/(1 + jωRC) is -jωC/(1 + jωRC)².
    def derivative_ac(frequency: float) -> complex:
        jw = 1j * 2 * math.pi * frequency
        return -jw * 1e-9 / (1 + jw * RC_TAU) ** 2

    names = tuple(f"d(out)/d(r1)@{i}" for i in range(len(FREQUENCIES)))

    def over_frequency() -> Values:
        found = simulation(deck_circuit(RC_DECK)).run(
            Sensitivity(["out"], of=Ac("V1", decade(1e4 * Hz, 1e6 * Hz, 1))), TIGHT
        )
        derivative = np.asarray(found.derivative("out", "R1", "resistance"))
        return {n: complex(d) for n, d in zip(names, derivative, strict=True)}

    # dv/dR of the ramp response: v depends on τ = RC, so dv/dR = C·dv/dτ.
    def derivative_transient(time: float) -> float:
        step = 1e-6 * RC_TAU
        return (
            1e-9
            * (ramped(time, RC_TAU + step) - ramped(time, RC_TAU - step))
            / (2 * step)
        )

    times = STEP_TIMES
    transient_names = tuple(f"d(out)/d(r1)@{t!r}" for t in times)

    def over_time() -> Values:
        built = circuit(rc_step())
        analysis = Transient(
            6 * us, max_step=10 * ns, breakpoints=[t * s for t in times]
        )
        found = simulation(built).run(Sensitivity(["out"], of=analysis), TIGHT)
        derivative = np.asarray(found.derivative("out", "R1", "resistance"))
        if not isinstance(found.result, results.TransientResult):
            message = "a transient's sensitivities follow its waveforms"
            raise TypeError(message)
        trace = found.result.times
        return {
            n: complex(np.interp(t, trace, derivative))
            for n, t in zip(transient_names, times, strict=True)
        }

    return [
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
        ),
        Fixture(
            key="sens-ac",
            category=Category.ANALYSIS,
            name="Sensitivities over frequency",
            known=Known(
                values={
                    n: derivative_ac(f) for n, f in zip(names, FREQUENCIES, strict=True)
                },
                source="closed form: a low pass's derivative in its resistance",
            ),
            jolty=over_frequency,
            spice=Spice(
                deck=RC_DECK,
                ngspice=("sens v(out) ac dec 1 1e4 1e6",),
                xyce=_xyce_documented(
                    ", .SENS: AC sensitivities with acobjfunc", pending=True
                ),
                reads={
                    name: Read(ngspice="v(r1)", index=index)
                    for index, name in enumerate(names)
                },
            ),
        ),
        Fixture(
            key="sens-tran",
            category=Category.ANALYSIS,
            name="Sensitivities over time",
            known=Known(
                values={
                    n: derivative_transient(t)
                    for n, t in zip(transient_names, times, strict=True)
                },
                source="closed form: the ramp response's derivative in its time "
                "constant, by a central difference of 1e-6 of it",
                tolerance=INTEGRATED,
            ),
            jolty=over_time,
            spice=Spice(
                deck=exported(circuit(rc_step()), "transient"),
                ngspice=Unsupported(NGSPICE_SENSITIVITY),
                xyce=_xyce_documented(
                    ", .SENS: transient sensitivities, direct and adjoint",
                    pending=True,
                ),
                reads={},
            ),
        ),
    ]


def transients() -> list[Fixture]:
    """The rows of transients, and of the analyses built on them."""

    # A low pass driven by a 1 MHz sine of 1 V: its periodic steady state's amplitude
    # at the output is |H(f)|, and its phase H's.
    def driven() -> Component:
        return _rc_circuit(
            "driven", Sine(offset=0 * V, amplitude=1 * V, frequency=1e6 * Hz)
        )

    response = low_pass(1e6)
    quarter = (0.25e-6, 0.5e-6, 0.75e-6, 1e-6)
    names = tuple(f"v(out)@{t!r}" for t in quarter)

    def periodic() -> Values:
        built = circuit(driven())
        analysis = PeriodicSteadyState(
            1 * us, max_step=1 * ns, breakpoints=[t * s for t in quarter]
        )
        found = simulation(built).run(analysis, TIGHT)
        start = float(found.times[0])
        values = np.asarray(found.voltage("out"))
        return {
            n: complex(np.interp(start + t, found.times, values))
            for n, t in zip(names, quarter, strict=True)
        }

    return [
        transient_fixture("tran-trapezoidal", "Transient, trapezoidal", "trapezoidal"),
        transient_fixture("tran-gear2", "Transient, Gear of order 2", "gear2"),
        transient_fixture(
            "tran-backward-euler", "Transient, backward Euler", "backward_euler"
        ),
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
                ngspice=_ngspice_unsupported(
                    ", §11.3.12 .PSS: periodic steady state of autonomous circuits, "
                    "oscillators, only"
                ),
                xyce=_xyce_documented(", .HB: harmonic balance", pending=True),
                reads={},
            ),
        ),
    ]


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


def vbic_currents(  # noqa: PLR0913 - VBIC bias, saturation currents, temperature and intrinsic resistances
    collector_voltage: float,
    base_voltage: float,
    *,
    saturation_current: float,
    ideal_base_emitter_current: float,
    ideal_base_collector_current: float,
    temperature: float = ROOM,
    nominal_temperature: float = ROOM,
    intrinsic_base_resistance: float = VBIC_INTRINSIC_RESISTANCE,
    intrinsic_collector_resistance: float = VBIC_INTRINSIC_RESISTANCE,
    parasitic_base_resistance: float = VBIC_INTRINSIC_RESISTANCE,
    minimum_conductance: float = 0.0,
) -> tuple[float, float]:
    """Exact VBIC 1.2 operating-point collector and base currents (emitter/substrate 0).

    Accounts for the exact model details that caused naive Ebers-Moll with VBIC's
    physical constants to miss the `vbic` fixture by +2.114e-6:

    1. VBIC 1.2 uses its own physical constants (`k = 1.380662e-23 J/K`,
       `q = 1.602189e-19 C`) rather than 2019 SI values.
    2. VBIC 1.2 mandates positive intrinsic resistances whose defaults are non-zero:
       `RBI = RCI = RBP = 0.1 ohm` (ngspice `vbicsetup.c`, `vbic.rs`). At `V_c = 2 V`,
       `V_b = 0.7 V`, `I_b ~= 5.670e-7 A` drops `V_rbi = I_b * RBI ~= 5.670e-8 V`
       across `RBI`, lowering the intrinsic `V_bei` and reducing both `I_c` and `I_b`
       by `V_rbi / Vt_VBIC = 2.192e-6`.
    3. VBIC smooths `q1` above zero even when Early voltages are infinite (`q1z = 1`):
       `q1 = (sqrt((1 - 1e-4)^2 + 1e-8) + (1 - 1e-4)) / 2 + 1e-4 = 1 + 2.50025e-9`,
       lowering `I_tzf = I_fi / q1` by `2.500e-9` (`-2.195e-6` total at `gmin = 0`).
    4. Optional junction minimum conductance (`gmin = 1e-12 S` in SPICE/jolty defaults)
       shifts `I_c` by `+4.60 pA` (`+8.111e-8` of `I_c`, explaining
       `-2.195e-6 + 0.081e-6 = -2.114e-6`) and `I_b` by `-1.20 pA`
       (`-2.116e-8` of `I_c`).

    The coupled KCL equations are solved in voltage-drop coordinates (`base_drop =
    V_bx - V_bi`, `collector_drop = V_cx - V_ci`, `parasitic_drop = V_bp - V_cx`) so
    there is no floating-point cancellation when dividing small drops by `0.1 ohm`.
    """
    ratio = temperature / nominal_temperature
    thermal_voltage = temperature * VBIC_BOLTZMANN / VBIC_CHARGE
    arrhenius = (ratio**3) * math.exp(1.12 * (ratio - 1.0) / thermal_voltage)
    scaled_saturation_current = saturation_current * arrhenius
    scaled_base_emitter_saturation = ideal_base_emitter_current * arrhenius
    scaled_base_collector_saturation = ideal_base_collector_current * arrhenius
    shifted = 1.0 - 1e-4
    base_charge = (math.hypot(shifted, 1e-4) + shifted) * 0.5 + 1e-4
    base_collector_external = base_voltage - collector_voltage
    base_drop = 0.0
    collector_drop = 0.0
    parasitic_drop = 0.0
    forward_transport = 0.0
    reverse_transport = 0.0
    base_emitter_current = 0.0
    base_collector_current = 0.0
    for _ in range(30):
        intrinsic_base_emitter_voltage = base_voltage - base_drop
        intrinsic_base_collector_voltage = (
            base_collector_external - base_drop + collector_drop
        )
        extrinsic_base_collector_voltage = base_collector_external - base_drop
        base_emitter_current = (
            scaled_base_emitter_saturation
            * math.expm1(intrinsic_base_emitter_voltage / thermal_voltage)
            + intrinsic_base_emitter_voltage * minimum_conductance
        )
        base_collector_current = (
            scaled_base_collector_saturation
            * math.expm1(intrinsic_base_collector_voltage / thermal_voltage)
            + intrinsic_base_collector_voltage * minimum_conductance
        )
        forward_transport = (
            scaled_saturation_current
            * math.expm1(intrinsic_base_emitter_voltage / thermal_voltage)
            / base_charge
        )
        reverse_transport = (
            scaled_saturation_current
            * math.expm1(intrinsic_base_collector_voltage / thermal_voltage)
            / base_charge
        )
        parasitic_drop = (
            (base_collector_external - collector_voltage)
            * minimum_conductance
            / (1.0 / parasitic_base_resistance + 2.0 * minimum_conductance)
        )
        base_drop = (
            (base_emitter_current + base_collector_current)
            * intrinsic_base_resistance
            / base_charge
        )
        collector_drop = (
            (forward_transport - reverse_transport - base_collector_current)
            - (intrinsic_base_collector_voltage + extrinsic_base_collector_voltage)
            * minimum_conductance
        ) / (1.0 / intrinsic_collector_resistance + minimum_conductance)
    extrinsic_base_emitter_leakage = base_voltage * minimum_conductance
    parasitic_base_emitter_leakage = (
        base_collector_external - parasitic_drop
    ) * minimum_conductance
    parasitic_collector_substrate_leakage = (
        -collector_voltage - parasitic_drop
    ) * minimum_conductance
    collector_current = (
        forward_transport - reverse_transport - base_collector_current
    ) - (parasitic_base_emitter_leakage + parasitic_collector_substrate_leakage)
    base_current = (
        base_emitter_current
        + base_collector_current
        + extrinsic_base_emitter_leakage
        + parasitic_base_emitter_leakage
    )
    return collector_current, base_current


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


@dataclass(frozen=True, slots=True)
class _OpSpec:
    """Declarative entry for a deck-based operating-point fixture."""

    key: str
    name: str
    deck: tuple[str, ...]
    known: Values
    source: str
    xyce: Unsupported | Documented | None = None
    temperature: float = ROOM


@dataclass(frozen=True, slots=True)
class _BuiltSpec:
    """Declarative entry for a device built through jolty's builder."""

    key: str
    name: str
    device: object
    terminals: Mapping[str, float]
    reads: tuple[str, ...]
    known: Known
    xyce: Unsupported | Documented
    ngspice_note: Unsupported | None = None
    temperature: float = ROOM


def device_fixtures() -> list[Fixture]:
    """The device models' rows."""
    at = 1e6
    jw = 2j * math.pi * at
    rows: list[Fixture] = [
        ac_fixture(key, name, deck, known, source)
        for key, name, deck, known, source in (
            (
                "capacitor",
                "Capacitor",
                ("* capacitor", "V1 a 0 0 AC 1", "C1 a 0 1n"),
                {"i(v1)": -jw * 1e-9},
                "closed form: jωC",
            ),
            (
                "inductor",
                "Inductor",
                ("* inductor", "V1 a 0 0 AC 1", "R1 a b 1", "L1 b 0 1u"),
                {"i(v1)": -1 / (1 + jw * 1e-6)},
                "closed form: 1/(R + jωL)",
            ),
            (
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
            ),
        )
    ]
    rows.extend(
        op_fixture(key, Category.DEVICE, name, deck, {"v(o)": known}, "closed form")
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
                (
                    "* cccs",
                    "V1 a 0 2",
                    "R1 a b 1k",
                    "VS b 0 0",
                    "F1 0 o VS 3",
                    "R2 o 0 1k",
                ),
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
        )
    )
    collector, base = bjt_currents(0.7, -1.3, 1e-15, 100, 2)
    vbic_collector, vbic_base = vbic_currents(
        2.0,
        0.7,
        saturation_current=1e-16,
        ideal_base_emitter_current=1e-18,
        ideal_base_collector_current=1e-16,
    )
    square = 1e-4 / 2 * 10 / 2 * 1.3**2
    semiconductors = (
        _OpSpec(
            "diode",
            "Diode",
            ("* diode", "V1 a 0 0.7", "D1 a 0 dd", ".model dd D(IS=1e-14 N=1.2)"),
            {"i(v1)": -1e-14 * math.expm1(0.7 / (1.2 * THERMAL_VOLTAGE))},
            "closed form: Shockley's law",
        ),
        _OpSpec(
            "bjt",
            "Bipolar transistor, Gummel-Poon",
            (
                "* gummel-poon",
                "VC c 0 2",
                "VB b 0 0.7",
                "Q1 c b 0 qq",
                ".model qq NPN(IS=1e-15 BF=100 BR=2)",
            ),
            {"i(vc)": -collector, "i(vb)": -base},
            (
                "closed form: Ebers and Moll's transport currents, no Early effect or "
                "high injection"
            ),
        ),
        # VBIC computes its thermal voltage with constants of its own in both engines:
        # ngspice must run at 27.0 C (300.15 K) itself so Arrhenius scaling from
        # TNOM=27 does not shift currents by +6.53e-6.
        _OpSpec(
            "vbic",
            "Bipolar transistor, VBIC 1.2",
            (
                "* vbic",
                "VC c 0 2",
                "VB b 0 0.7",
                "Q1 c b 0 qv",
                (
                    ".model qv NPN(LEVEL=4 IS=1e-16 NF=1 NR=1 IBEI=1e-18 NEI=1 "
                    "IBCI=1e-16 NCI=1 TNOM=27)"
                ),
            ),
            {"i(vc)": -vbic_collector, "i(vb)": -vbic_base},
            (
                "closed form: VBIC 1.2 transport and base currents with VBIC "
                "constants (k=1.380662e-23, q=1.602189e-19), default intrinsic "
                "resistances (RBI=RCI=RBP=0.1 ohm, causing -2.19e-6 base drop), "
                "and q1 smoothing floor (1 + 2.5e-9)"
            ),
            xyce=_xyce_documented(
                ", BJT levels 11 and 12: VBIC 1.3, not 1.2, its card's level not "
                "ngspice's 4; the dialect of branch xyce writes it"
            ),
            temperature=ngspice.same_kelvins(ROOM),
        ),
        _OpSpec(
            "jfet",
            "JFET",
            (
                "* jfet",
                "VD d 0 5",
                "VG g 0 -1",
                "J1 d g 0 jj",
                ".model jj NJF(VTO=-2 BETA=1e-3 LAMBDA=0.01)",
            ),
            {"i(vd)": -1e-3 * 1.05},
            "closed form: Shichman and Hodges's saturation current",
        ),
        _OpSpec(
            "mos1",
            "MOSFET, level 1",
            (
                "* mos1",
                "VD d 0 3",
                "VG g 0 2",
                "M1 d g 0 0 mm W=10u L=2u",
                ".model mm NMOS(LEVEL=1 VTO=0.7 KP=1e-4 LAMBDA=0.02)",
            ),
            {"i(vd)": -square * 1.06},
            "closed form: the square law in saturation",
        ),
        _OpSpec(
            "mos3",
            "MOSFET, level 3",
            (
                "* mos3",
                "VD d 0 3",
                "VG g 0 2",
                "M1 d g 0 0 mm W=10u L=2u",
                (
                    ".model mm NMOS(LEVEL=3 VTO=0.7 KP=1e-4 GAMMA=0 THETA=0 ETA=0 "
                    "DELTA=0 VMAX=0 NFS=0)"
                ),
            ),
            {"i(vd)": -square},
            (
                "closed form: level 3 with no body effect, mobility degradation or "
                "velocity saturation, and no substrate doping for its channel-length "
                "modulation, is the square law"
            ),
        ),
    )
    rows.extend(
        op_fixture(
            spec.key,
            Category.DEVICE,
            spec.name,
            spec.deck,
            spec.known,
            spec.source,
            xyce=spec.xyce,
            temperature=spec.temperature,
        )
        for spec in semiconductors
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
    placed = c.place(device, name="X1")  # ty: ignore[invalid-argument-type] - a device
    for terminal, level in terminals.items():
        source = c.place(devices.voltage_source(level * V), name=f"V{terminal.upper()}")
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
    on, off = 100.0, 16e3
    memristor_known = Known(
        values={"i(vpositive)": -0.5 / (on + 0.25 * (off - on))},
        source="closed form: its resistance at its state, held at an operating point",
    )
    memristor_ngspice = _ngspice_unsupported(
        ", §8.2.29 Memristor: XSPICE's memristor code model, another law; "
        "jolty's export writes it as a behavioral current"
    )
    memristor_xyce = _xyce_unsupported(
        ", Memristor: levels 2 (TEAM), 3 (Yakopcic) and 4 (PEM), not this law"
    )
    lit = 1e-6 + (5 / 10) ** 0.6 / 10e3
    # Default BSIM3v3.3 (tox=15 nm) has Vth ~= 1.815 V; (1.5 V, 1.2 V) sits in deep
    # weak inversion (~1.95 pA, 77% gmin leakage). At (3.0 V, 3.0 V) it is solidly in
    # strong-inversion saturation (overdrive +1.19 V, Vdsat ~= 0.58 V, Id ~= 0.328 mA).
    specs = (
        _BuiltSpec(
            "vdmos",
            "VDMOS",
            devices.vdmos(
                threshold_voltage=2 * V, transconductance_parameter=0.5 * A / V**2
            ),
            {"drain": 5.0, "gate": 4.0, "source": 0.0},
            ("i(vdrain)",),
            Known(
                source="ngspice 47's VDMOS, the model's defining implementation",
                reference=Simulator.NGSPICE_SPARSE,
            ),
            _xyce_documented(
                ", MOSFET level 18 (VDMOS), whose card is not ngspice's; the dialect "
                "of branch xyce writes it"
            ),
        ),
        _BuiltSpec(
            "bsim3",
            "BSIM3v3.3",
            devices.bsim3(channel="n", width=10e-6 * m_, length=1e-6 * m_),
            {"drain": 3.0, "gate": 3.0, "source": 0.0, "bulk": 0.0},
            ("i(vdrain)",),
            Known(
                source=(
                    "ngspice 47's BSIM3v3.3, Berkeley's code, the model's defining "
                    "implementation"
                ),
                reference=Simulator.NGSPICE_SPARSE,
            ),
            _xyce_documented(
                ", MOSFET level 9 (BSIM3v3.2.2); no fixture runs it: another version "
                "than jolty's 3.3"
            ),
            # BSIM3 computes with constants of its own: ngspice runs at the kelvins
            # themselves, as the differential tests run it.
            temperature=ngspice.same_kelvins(ROOM),
        ),
        _BuiltSpec(
            "bsim4",
            "BSIM4 4.5",
            devices.bsim4(channel="n", width=10e-6 * m_, length=1e-6 * m_),
            {"drain": 1.5, "gate": 1.2, "source": 0.0, "bulk": 0.0},
            ("i(vdrain)",),
            Known(
                source=(
                    "ngspice 47's BSIM4v5, Berkeley's code, the model's defining "
                    "implementation"
                ),
                reference=Simulator.NGSPICE_SPARSE,
            ),
            _xyce_documented(
                ", MOSFET level 14: BSIM4 4.6.1, 4.7.0 and 4.8.2, not 4.5"
            ),
        ),
        _BuiltSpec(
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
            {"positive": 0.5, "negative": 0.0},
            ("i(vpositive)",),
            memristor_known,
            memristor_xyce,
            ngspice_note=memristor_ngspice,
        ),
        _BuiltSpec(
            "memristor-hp",
            "Memristor, HP's linear ion drift",
            devices.current_controlled_memristor(
                on_resistance=on * ohm,
                off_resistance=off * ohm,
                charge=1e-6 * A * s,
                state=0.25,
            ),
            {"positive": 0.5, "negative": 0.0},
            ("i(vpositive)",),
            memristor_known,
            memristor_xyce,
            ngspice_note=memristor_ngspice,
        ),
        _BuiltSpec(
            "photocell",
            "Photocell",
            devices.photocell(10e3 * ohm),
            {"a": 1.0, "b": 0.0, "light_positive": 5.0, "light_negative": 0.0},
            ("i(va)",),
            Known(
                values={"i(va)": -lit}, source="closed form: 1/r_dark + (E/E0)^gamma/r"
            ),
            _xyce_unsupported(", its devices: no photocell"),
            ngspice_note=_ngspice_unsupported(
                ", chapter 7 Device Models: no photocell; jolty's export writes it as "
                "a behavioral current"
            ),
        ),
    )

    c, ground = _builder("heat")
    heat = c.place(devices.heat_source(1 * W), name="P1")
    sink = c.place(devices.thermal_resistance(50 * K / W), name="RTH")
    c.connect(heat.b, sink.a, name="hot")
    c.connect(heat.a, sink.b, ground)
    heated = circuit(c.finish())

    def rise() -> Values:
        found = simulation(heated).run(OperatingPoint(), TIGHT)
        return {"rise(hot)": complex(found.rise("hot"))}

    return [
        *(
            built_fixture(
                spec.key,
                spec.name,
                forced(spec.key, spec.device, spec.terminals),
                spec.reads,
                spec.known,
                ngspice_note=spec.ngspice_note,
                xyce=spec.xyce,
                temperature=spec.temperature,
            )
            for spec in specs
        ),
        Fixture(
            key="thermal",
            category=Category.DEVICE,
            name="Thermal network",
            known=Known(values={"rise(hot)": 50.0}, source="closed form: P·R_th"),
            jolty=rise,
            spice=_unrun_spice(
                _ngspice_unsupported(
                    ", chapter 7 Device Models: no thermal elements; self-heating "
                    "inside its VBIC and BSIM models alone"
                ),
                _xyce_unsupported(", its devices: no thermal elements"),
            ),
        ),
        _line_fixture(
            "tline", "Lossless transmission line", "T1 in 0 out 0 Z0=50 TD=5n"
        ),
        *switch_and_logic_devices(),
        *missing_devices(),
        *kit_devices(),
    ]


def _switch_fixture(
    key: str,
    name: str,
    *,
    subcircuit: str,
    levels: Sequence[float],
    xyce: Documented,
) -> Fixture:
    """A PSpice-model switch row sweeping through OFF, midpoint, and ON."""
    library_path = DECKS / "switches.lib"
    deck = (
        f"* {key}",
        "V1 in 0 5",
        "VC ctrl 0 0",
        f"X1 in out ctrl {subcircuit}",
    )
    names = tuple(f"v(out)@{index}" for index in range(len(levels)))
    expected = (5 * 10e3 / 11e3, 2.5, 5 * 100 / 1100)
    step = levels[1] - levels[0]
    ngspice_command = f"dc VC {levels[0]:g} {levels[-1]:g} {step:g}"

    def run() -> Values:
        library = Library()
        library.include(library_path)
        library.read(
            "\n".join([".subckt bench", *deck[1:], ".ends"]),
            "fixture.cir",
        )
        built = circuit(library.component("bench"))
        return solved(built, DcSweep("VC", [level * V for level in levels]), names)

    return Fixture(
        key=key,
        category=Category.DEVICE,
        name=name,
        known=Known(
            values=dict(zip(names, expected, strict=True)),
            source=(
                "closed form: divider with switch at `ROFF=10 kohm`, "
                "`sqrt(RON * ROFF) = 1 kohm`, and `RON=100 ohm`"
            ),
        ),
        jolty=run,
        spice=Spice(
            deck=deck,
            ngspice=(ngspice_command,),
            xyce=xyce,
            reads=_reads(names),
            compatibility="psa",
            libraries=(library_path,),
        ),
    )


def switch_and_logic_devices() -> list[Fixture]:
    """The rows for voltage- and current-controlled switches and logic devices."""

    def run_logic() -> Values:
        builder, ground = _builder("logic")
        input_a = builder.place(
            devices.logic_source(0, [(1 * ns, 1)], transition=10 * ps), name="A"
        )
        input_b = builder.place(devices.logic_source(1, transition=10 * ps), name="B")
        nand_gate = builder.place(
            devices.logic_gate("nand", rise=2 * ns, fall=1 * ns, transition=10 * ps),
            name="U1",
        )
        not_gate = builder.place(
            devices.logic_gate("not", rise=2 * ns, fall=1 * ns, transition=10 * ps),
            name="U2",
        )
        builder.connect(input_a.output, nand_gate.a, name="a")
        builder.connect(input_b.output, nand_gate.b, name="b")
        builder.connect(nand_gate.y, not_gate.a, name="n")
        builder.connect(not_gate.y, name="y")

        supply = builder.place(devices.voltage_source(1.8 * V), name="VDD")
        edge = Pulse(
            initial=0 * V,
            pulsed=1.8 * V,
            delay=1 * ns,
            rise=1 * ns,
            fall=1 * ns,
            width=10 * ns,
        )
        stimulus = builder.place(devices.voltage_source(edge), name="V1")
        electrical_inverter = builder.place(
            devices.logic_gate(
                "not",
                rise=500 * ps,
                transition=100 * ps,
                electrical=True,
                input_capacitance=2 * fF,
            ),
            name="U3",
        )
        load = builder.place(devices.capacitor(5 * fF), name="C1")
        builder.connect(supply.positive, electrical_inverter.supply, name="vdd")
        builder.connect(stimulus.positive, electrical_inverter.a, name="in")
        builder.connect(electrical_inverter.y, load.a, name="out")
        builder.connect(
            supply.negative,
            stimulus.negative,
            electrical_inverter.ground,
            load.b,
            ground,
        )
        built = circuit(builder.finish())
        waveforms = simulation(built).run(Transient(6 * ns), TIGHT)
        nand_times = waveforms.logic_level("n").times
        inverter_times = waveforms.logic_level("y").times
        output = np.asarray(waveforms.voltage("out"))
        return {
            "t(n_fall)": complex(float(nand_times[-1])),
            "t(y_rise)": complex(float(inverter_times[-1])),
            "v(out)@5e-10": complex(float(np.interp(0.5e-9, waveforms.times, output))),
            "v(out)@2e-09": complex(float(np.interp(2.0e-9, waveforms.times, output))),
            "v(out)@3e-09": complex(float(np.interp(3.0e-9, waveforms.times, output))),
        }

    return [
        *(
            _switch_fixture(
                f"switch-{control}",
                f"{control.capitalize()}-controlled switch ({card})",
                subcircuit=f"{card.lower()}_stage",
                levels=levels,
                xyce=_xyce_documented(
                    f", {card}: smooth {control}-controlled switch with log-cubic "
                    "transition",
                    pending=True,
                ),
            )
            for control, card, levels in (
                ("voltage", "VSWITCH", (0.0, 0.5, 1.0)),
                ("current", "ISWITCH", (0.0, 1.0, 2.0)),
            )
        ),
        Fixture(
            key="logic",
            category=Category.DEVICE,
            name="Event-driven logic and mixed-signal bridge",
            known=Known(
                values={
                    "t(n_fall)": 2e-9,
                    "t(y_rise)": 4e-9,
                    "v(out)@5e-10": 1.8,
                    "v(out)@2e-09": 0.9,
                    "v(out)@3e-09": 0.0,
                },
                source=(
                    "closed form: event-driven NAND fall at 1 ns + 1 ns = 2 ns, "
                    "inverter rise at 2 ns + 2 ns = 4 ns, and electrical inverter "
                    "midpoint crossing at 1.5 ns + 0.5 ns = 2.0 ns"
                ),
            ),
            jolty=run_logic,
            spice=_unrun_spice(
                _ngspice_documented(
                    ", chapter 8 XSPICE: digital logic code models and ADC/DAC bridges",
                    pending=True,
                ),
                _xyce_documented(
                    ", Behavioral Digital Devices (U): mixed-signal digital gate "
                    "primitives",
                    pending=True,
                ),
            ),
        ),
    ]


def kit_devices() -> list[Fixture]:
    """The rows for the pinned open foundry process design kits.

    Real PDK archives are tens of megabytes fetched from GitHub and are verified by
    `tests/test_kits.py`; these rows stay `Documented` so the coverage matrix and its
    offline unit tests never require network downloads.
    """
    prefix = "docs/guide/foundry.md & tests/test_kits.py: "
    offline = "; external archive not fetched offline"
    entries = (
        (
            "kit-sky130",
            "SKY130",
            sky130.VERSION[:8],
            "(251 of 263 devices native, BSIM4 4.5, scale=1u, hsa compatibility)",
            "with ngbehavior=hsa",
            (
                "open_pdks SKY130: supported via open_pdks Xyce netlists or XDM "
                "translation"
            ),
        ),
        (
            "kit-gf180mcu",
            "GF180MCU",
            gf180mcu.VERSION[:8],
            "(65 of 72 devices native, BSIM4 4.5, Gummel-Poon, diode level 3)",
            "natively",
            "open_pdks GF180MCU: BSIM4 models run with XDM translation",
        ),
        (
            "kit-sg13g2",
            "SG13G2",
            sg13g2.VERSION[:8],
            (
                "(36 of 53 devices native including self-heated VBIC HBTs, "
                "14 Verilog-A models)"
            ),
            "native devices and requires OpenVAF/OSDI for Verilog-A MOSFETs",
            (
                "IHP-Open-PDK SG13G2: ships dedicated Xyce model libraries for MOS, "
                "HBT and passive devices"
            ),
        ),
    )
    return [
        Fixture(
            key=key,
            category=Category.DEVICE,
            name=f"Process design kit: {kit_name}",
            known=None,
            jolty=Documented(
                f"{prefix}{kit_name} pinned at {version} {jolty_detail}{offline}"
            ),
            spice=_unrun_spice(
                Documented(
                    f"{prefix}ngspice 47 runs {kit_name} ({version}) "
                    f"{ngspice_detail}{offline}"
                ),
                Documented(f"{xyce_detail}{offline}"),
            ),
        )
        for key, kit_name, version, jolty_detail, ngspice_detail, xyce_detail in entries
    ]


def _line_fixture(
    key: str,
    name: str,
    element: str,
    extra: tuple[str, ...] = (),
) -> Fixture:
    """A matched 50 ohm, 5 ns line driven by a pulse: half the pulse, 5 ns late."""

    def pulse(t: float) -> float:
        return (
            min(max(t / 1e-9, 0.0), 1.0)
            if t < 11e-9
            else max(1 - (t - 11e-9) / 1e-9, 0.0)
        )

    times = (5.5e-9, 8e-9, 14e-9)
    reads = {
        f"v(out)@{t!r}": Read(ngspice="v(out)", xyce="V(OUT)", time=t) for t in times
    }
    known: Values = {f"v(out)@{t!r}": 0.5 * pulse(t - 5e-9) for t in times}
    source = ("V1 s 0 PULSE(0 1 0 1n 1n 10n 100n)", "R1 s in 50", "R2 out 0 50")
    deck = (f"* {name}", element, *source, *extra)

    def run() -> Values:
        waveforms = simulation(deck_circuit(deck)).run(
            Transient(20 * ns, max_step=0.05 * ns), TIGHT
        )
        return sampled(waveforms, tuple(reads))

    return Fixture(
        key=key,
        category=Category.DEVICE,
        name=name,
        known=Known(
            values=known,
            source="closed form: a matched line delays half its source's pulse",
            tolerance=INTEGRATED,
        ),
        jolty=run,
        spice=Spice(
            deck=deck,
            ngspice=("tran 0.05n 20n 0 0.05n",),
            xyce=((".tran 0.05n 20n 0 0.05n",), "tran"),
            reads=reads,
        ),
    )


def ngspice_only(  # noqa: PLR0913, PLR0917 - a row's names, its deck and its answer, as a table of them reads
    key: str,
    name: str,
    deck: tuple[str, ...],
    ngspice_commands: tuple[str, ...],
    reads: Mapping[str, Read],
    known: Mapping[str, complex] | None,
    source: str,
    *,
    xyce: tuple[tuple[str, ...], str] | Unsupported | Documented,
    category: Category = Category.DEVICE,
    tolerance: float = SOLVED,
) -> Fixture:
    """A row jolty's reader is given the deck of, as ngspice and Xyce are."""
    names = tuple(reads)
    return Fixture(
        key=key,
        category=category,
        name=name,
        known=Known(values=dict(known), source=source, tolerance=tolerance)
        if known is not None
        else Known(source=source),
        jolty=lambda: solved(deck_circuit(deck), OperatingPoint(), names),
        spice=Spice(deck=deck, ngspice=ngspice_commands, xyce=xyce, reads=reads),
    )


def _forced_deck(
    title: str, element: str, card: str, drain: str = "d"
) -> tuple[str, ...]:
    """A transistor's deck, `element` first for a reader to name, at 3 V and 2 V."""
    return (f"* {title}", element, card, f"VD {drain} 0 3", "VG g 0 2")


def missing_devices() -> list[Fixture]:
    """The device rows of ngspice's that jolty lacks, its reader given their decks."""
    unjudged = "none yet: the model's QA reference results are to come"
    current = {"i(vd)": Read(ngspice="i(vd)", xyce="I(VD)")}
    mos_2u = "M1 d g 0 0 mm W=10u L=2u"
    mos_1u = "M1 d g 0 0 mm W=10u L=1u"
    forced_specs: tuple[
        tuple[
            str, str, str, str, tuple[tuple[str, ...], str] | Unsupported | Documented
        ],
        ...,
    ] = (
        (
            "mos2",
            "MOSFET, level 2",
            mos_2u,
            ".model mm NMOS(LEVEL=2 VTO=0.7 KP=1e-4)",
            ((".op",), "dc"),
        ),
        (
            "mos6",
            "MOSFET, level 6",
            mos_2u,
            ".model mm NMOS(LEVEL=6 VTO=0.7)",
            ((".op",), "dc"),
        ),
        (
            "mos9",
            "MOSFET, level 9",
            mos_2u,
            ".model mm NMOS(LEVEL=9 VTO=0.7 KP=1e-4)",
            _xyce_documented(": level 9 is BSIM3 in Xyce"),
        ),
        (
            "bsimsoi",
            "BSIM-SOI 4",
            mos_1u,
            ".model mm NMOS(LEVEL=10)",
            _xyce_documented(", MOSFET levels 10 and 70: BSIM-SOI"),
        ),
        (
            "hisim2",
            "HiSIM2",
            mos_1u,
            ".model mm NMOS(LEVEL=68)",
            _xyce_unsupported(", its MOSFET levels: no HiSIM"),
        ),
        (
            "hicum",
            "HICUM/L2",
            "Q1 d g 0 qq",
            ".model qq NPN(LEVEL=8)",
            _xyce_documented(", BJT level 234: HICUM/L2 through ADMS"),
        ),
        (
            "mesfet",
            "MESFET, Statz",
            "Z1 d g 0 zz",
            ".model zz NMF(LEVEL=1 VTO=-2 BETA=1e-3)",
            _xyce_documented(", MESFET (Z) level 1"),
        ),
        (
            "jfet2",
            "JFET, level 2 (Parker and Skellern)",
            "J1 d g 0 jj",
            ".model jj NJF(LEVEL=2)",
            _xyce_documented(", JFET level 2 (Shockley), another model"),
        ),
    )
    return [
        *(
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
            for key, name, element, card, xyce in forced_specs
        ),
        _line_fixture(
            "ltra",
            "Lossy transmission line (LTRA), lossless",
            "O1 in 0 out 0 lmod",
            (".model lmod LTRA(R=0 G=0 L=250n C=100p LEN=1)",),
        ),
        *(
            ngspice_only(
                key,
                name,
                deck,
                ("op",),
                _reads(tuple(known)),
                known,
                source,
                xyce=_xyce_unsupported(xyce_detail),
            )
            for key, name, deck, known, source, xyce_detail in (
                (
                    "urc",
                    "Distributed RC line (URC)",
                    (
                        "* urc",
                        "U1 a out 0 umod L=1m",
                        ".model umod URC(RPERL=1k CPERL=1p)",
                        "V1 a 0 1",
                        "R2 out 0 1",
                    ),
                    {"v(out)": 0.5},
                    (
                        "closed form: the line's DC resistance, RPERL·L, over a load "
                        "of the same"
                    ),
                    ", its devices: no URC",
                ),
                (
                    "xspice",
                    "XSPICE code models",
                    (
                        "* xspice",
                        "A1 a o g1",
                        ".model g1 gain(gain=3)",
                        "V1 a 0 1",
                        "R1 o 0 1k",
                    ),
                    {"v(o)": 3.0},
                    "closed form: a gain block's output",
                    ": no XSPICE code models",
                ),
            )
        ),
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
            spice=_unrun_spice(
                _ngspice_documented(
                    ", §9.2 OSDI/OpenVAF: a model compiled by OpenVAF, which is not "
                    "installed here"
                ),
                _xyce_unsupported(
                    ": no OSDI; Verilog-A only as ADMS-compiled C++ models built "
                    "into it"
                ),
            ),
        ),
    ]


def syntax() -> list[Fixture]:
    """The rows of SPICE's syntax, each a deck every simulator reads."""
    out: Values = {"v(out)": 2.0}
    divider = "closed form: a divider"
    specs: tuple[tuple[str, str, tuple[str, ...], Values, str], ...] = (
        (
            "subckt",
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
        (
            "param",
            ".param reading other parameters",
            (
                "* parameters",
                ".param upper=1k lower={3*upper}",
                "V1 in 0 {8/3}",
                "R1 in out {upper}",
                "R2 out 0 {lower}",
            ),
            out,
            divider,
        ),
        (
            "expressions",
            "Expressions with functions",
            (
                "* expressions",
                "V1 in 0 {sqrt(16)}",
                "R1 in out {exp(0)*1k}",
                "R2 out 0 {max(1k, 0.5k)}",
            ),
            out,
            divider,
        ),
        (
            "func",
            ".func",
            (
                "* functions",
                ".func twice(x) {2*x}",
                "V1 in 0 {twice(2)}",
                "R1 in out 1k",
                "R2 out 0 1k",
            ),
            out,
            divider,
        ),
        (
            "lib",
            ".lib sections",
            (
                "* sections",
                f".lib '{DECKS / 'sections.lib'}' low",
                "V1 in 0 {8/3}",
                "R1 in out 1k",
                "R2 out 0 {lower}",
            ),
            out,
            divider,
        ),
        (
            "include",
            ".include",
            (
                "* includes",
                f".include '{DECKS / 'included.cir'}'",
                "V1 in 0 {8/3}",
                "R1 in out 1k",
                "R2 out 0 {lower}",
            ),
            out,
            divider,
        ),
        (
            "poly",
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
        (
            "table",
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
        (
            "ako",
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
    )
    return [
        *(
            op_fixture(key, Category.SYNTAX, name, deck, known, source)
            for key, name, deck, known, source in specs
        ),
        *_state_and_kit_syntax(),
    ]


def _state_and_kit_syntax() -> list[Fixture]:
    """The rows for ``.ic``, ``.nodeset``, and offline process-kit library syntax."""
    initial_condition_deck = ("* ic", "R1 out 0 1k", "C1 out 0 1n", ".ic v(out)=1")
    decay_times = (0.0, 0.5e-6, 1e-6, 2e-6, 3e-6)
    decay_names = tuple(f"v(out)@{time!r}" for time in decay_times)

    def run_initial_conditions() -> Values:
        built, initial, _ = deck_with_state(initial_condition_deck)
        waveforms = simulation(built).run(
            Transient(
                4 * us,
                max_step=10 * ns,
                initial_conditions={net: level * V for net, level in initial.items()},
                breakpoints=[time * s for time in decay_times if time > 0],
            ),
            TIGHT,
        )
        return sampled(waveforms, decay_names)

    nodeset_deck = (
        "* nodeset",
        "B1 0 x I={2m * tanh(V(x))}",
        "R1 x 0 1k",
        ".nodeset v(x)=1.5",
    )
    latch_root = 1.9
    for _ in range(50):
        latch_root -= (latch_root - 2.0 * math.tanh(latch_root)) / (
            1.0 - 2.0 / math.cosh(latch_root) ** 2
        )

    def run_nodeset() -> Values:
        built, _, nodesets = deck_with_state(nodeset_deck)
        options = dataclasses.replace(
            TIGHT, nodesets={net: level * V for net, level in nodesets.items()}
        )
        found = simulation(built).run(OperatingPoint(), options)
        return {"v(x)": complex(found.voltage("x"))}

    kit_directory = DECKS / "pdk"
    globals_path = kit_directory / "globals.spice"
    corners_path = kit_directory / "corners.lib"
    library_deck = (
        "* pdk-library",
        "V1 in 0 5",
        "X1 in out 0 kit_div ratio=3",
        "VD d 0 3",
        "VG g 0 2",
        "M1 d g 0 0 nfet W=10 L=2",
        "VB db 0 1.5",
        "VGB gb 0 1.2",
        "M2 db gb 0 0 nbin W=4 L=0.5",
    )
    synthetic_kit = kits.Kit(
        name="synthetic",
        version="1",
        files=(),
        includes=(PurePosixPath("globals.spice"),),
        corners=(kits.Corners(PurePosixPath("corners.lib"), ("tt", "ss")),),
        compatibility="hsa",
    )
    transistor_current = 1e-4 / 2.0 * (10.0 / 2.0) * 1.3**2

    def run_kit_library() -> Values:
        library = kits.read(
            synthetic_kit,
            ((PurePosixPath("corners.lib"), "tt"),),
            settings=Settings(pdk_root=kit_directory),
        )
        library.read(
            "\n".join([".subckt bench", *library_deck[1:], ".ends"]),
            "fixture.cir",
        )
        built = circuit(library.component("bench"))
        return solved(built, OperatingPoint(), ("v(out)", "i(vd)"))

    return [
        Fixture(
            key="ic",
            category=Category.SYNTAX,
            name="Initial conditions (.ic)",
            known=Known(
                values={
                    name: math.exp(-time / RC_TAU)
                    for name, time in zip(decay_names, decay_times, strict=True)
                },
                source=(
                    "closed form: RC discharge `exp(-t / (R * C))` from "
                    "`v(out)(0) = 1 V`"
                ),
                tolerance=INTEGRATED,
            ),
            jolty=run_initial_conditions,
            spice=Spice(
                deck=initial_condition_deck,
                ngspice=("tran 10n 4u 0 10n",),
                xyce=_xyce_documented(
                    ", .IC: transient initial node voltages", pending=True
                ),
                reads={
                    name: Read(ngspice="v(out)", xyce="V(OUT)", time=time)
                    for name, time in zip(decay_names, decay_times, strict=True)
                },
            ),
        ),
        Fixture(
            key="nodeset",
            category=Category.SYNTAX,
            name="Operating-point guess (.nodeset)",
            known=Known(
                values={"v(x)": latch_root},
                source=(
                    "closed form: positive root of `v = 2 * tanh(v)`, selected over "
                    "the trivial root `v = 0` by `.nodeset v(x)=1.5`"
                ),
            ),
            jolty=run_nodeset,
            spice=Spice(
                deck=nodeset_deck,
                ngspice=("op",),
                xyce=_xyce_documented(
                    ", .NODESET: operating-point initial guesses", pending=True
                ),
                reads=_reads(("v(x)",)),
            ),
        ),
        Fixture(
            key="pdk-library",
            category=Category.SYNTAX,
            name="Process kit library (.lib, scale, .if, binning, temper, agauss)",
            known=Known(
                values={"v(out)": divider_out(), "i(vd)": -transistor_current},
                source=(
                    "closed form: synthetic PDK exercising `.include`, `.lib tt`, "
                    "`.option scale=1u`, `.if`, `temper`, `agauss`, and BSIM3 "
                    "binning (`M2` requires `scale=1u` to select `nbin.1`)"
                ),
            ),
            jolty=run_kit_library,
            spice=Spice(
                deck=library_deck,
                ngspice=("op",),
                xyce=_xyce_documented(
                    ", .LIB, .INCLUDE, .OPTIONS, .PARAM and binned models",
                    pending=True,
                ),
                reads=_reads(("v(out)", "i(vd)")),
                # Default ROOM keeps ngspice temp == tnom so M1's LEVEL=1 KP and VTO
                # have zero temperature-ratio shift; the +1.02e-4 K offset in `temper`
                # shifts kit_div's divider voltage by only -2.55e-8 (40x inside 1e-6).
                compatibility="hsa",
                libraries=(globals_path,),
                sections=((corners_path, "tt"),),
            ),
        ),
    ]


def raw_divider() -> Values:
    """The divider's output, written by jolty as SPICE's binary raw file, read back."""
    point = simulation(deck_circuit(divider_deck())).run(OperatingPoint(), TIGHT)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "divider.raw"
        files.save_raw(path, {"op": point})
        (plot,) = ngspice.read(path.read_bytes())
    return {"v(out)": complex(plot.vectors["v(out)"][0])}


def outputs() -> list[Fixture]:
    """The rows of outputs: SPICE's raw file, and a measurement of a waveform."""
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

    fourier_deck = (
        "* four",
        "V1 n1 0 SIN(0 1 1k)",
        "V2 n2 n1 SIN(0 0.03 2k)",
        "V3 out n2 SIN(0 0.04 3k)",
        "R1 out 0 1k",
    )

    def run_fourier() -> Values:
        waveforms = simulation(deck_circuit(fourier_deck)).run(
            Transient(2 * ms, max_step=2 * us), TIGHT
        )
        requirement = distortion("tran", "out", fundamental=1 * kHz, harmonics=9) <= 0
        (verdict,) = evaluate(Specs([requirement]), ["tran"], {"tran": waveforms})
        distortion_decibels = verdict.value if verdict.value is not None else math.nan
        return {"thd": complex(100.0 * 10.0 ** (distortion_decibels / 20.0))}

    return [
        Fixture(
            key="raw",
            category=Category.OUTPUT,
            name="SPICE raw files",
            known=Known(
                values={"v(out)": divider_out()},
                source="closed form: a divider's ratio, read back from the raw file",
            ),
            jolty=raw_divider,
            spice=Spice(
                deck=divider_deck(),
                ngspice=("op",),
                xyce=_xyce_documented(
                    ", the command line's -r: a SPICE raw file; no fixture reads it "
                    "here, the fixtures read its CSV"
                ),
                reads=_reads(("v(out)",)),
            ),
        ),
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
        ),
        Fixture(
            key="four",
            category=Category.OUTPUT,
            name="Fourier harmonics (.four)",
            known=Known(
                values={"thd": 5.0},
                source=(
                    "closed form: 3-4-5 harmonic sum "
                    "`100 * sqrt(0.03^2 + 0.04^2) / 1.0 = 5 %`"
                ),
                tolerance=INTEGRATED,
            ),
            jolty=run_fourier,
            spice=Spice(
                deck=fourier_deck,
                ngspice=("tran 2u 2m 0 2u\nfourier 1k v(out)",),
                xyce=_xyce_documented(
                    ", .FOUR: Fourier analysis written to a .four file", pending=True
                ),
                reads={"thd": Read(ngspice="printed:THD")},
            ),
        ),
    ]


def monte_carlo() -> Fixture:
    """The row of Monte Carlo samples: each one's divider, at its own resistances."""
    c, ground = _builder("toleranced")
    source = c.place(devices.voltage_source(5 * V), name="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.1), name="R1")
    lower = c.place(devices.resistor(3 * kohm, tolerance=0.1), name="R2")
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
        spice=_unrun_spice(
            _ngspice_documented(
                ", §18.5 Monte-Carlo Simulation: through its control language",
                pending=True,
            ),
            _xyce_documented(", .SAMPLING", pending=True),
        ),
    )


_CORNERS_SCRIPT: Final = """\
from jolty import Builder, Corner, OperatingPoint, Plan, Role, Specs, component, devices
from jolty.plan import TIGHT
from jolty.specs import value
from jolty.units import K, V, celsius, kohm


@component
def divider(builder: Builder, supply=5 * V) -> None:
    ground = builder.port("ground", Role.GROUND)
    source = builder.place(devices.voltage_source(supply), name="V1")
    upper = builder.place(
        devices.resistor(1 * kohm, temperature_coefficient=0.01 / K), name="R1"
    )
    lower = builder.place(devices.resistor(3 * kohm), name="R2")
    builder.connect(source.positive, upper.a, name="in")
    builder.connect(upper.b, lower.a, name="out")
    builder.connect(lower.b, source.negative, ground)


def build(supply=5 * V):
    return divider(supply=supply)


def plan():
    return Plan(
        {"op": OperatingPoint()},
        options=TIGHT,
        corners={
            "cold_low": Corner(temperature=celsius(-40), supply=4.5 * V),
            "hot_high": Corner(temperature=celsius(90), supply=5.5 * V),
        },
    )


def specs():
    return Specs([
        value("op", "out") >= 0 * V,
        value("op", "out") <= 10 * V,
    ])
"""

_WORST_CASE_SCRIPT: Final = """\
from jolty import Builder, OperatingPoint, Plan, Role, Specs, component, devices
from jolty.plan import TIGHT, WorstCase
from jolty.specs import value
from jolty.units import V, kohm


@component
def divider(builder: Builder) -> None:
    ground = builder.port("ground", Role.GROUND)
    source = builder.place(devices.voltage_source(5 * V), name="V1")
    upper = builder.place(devices.resistor(1 * kohm, tolerance=0.1), name="R1")
    lower = builder.place(devices.resistor(3 * kohm, tolerance=0.1), name="R2")
    builder.connect(source.positive, upper.a, name="in")
    builder.connect(upper.b, lower.a, name="out")
    builder.connect(lower.b, source.negative, ground)


def build():
    return divider()


def plan():
    return Plan({"op": OperatingPoint()}, options=TIGHT, worst_case=WorstCase())


def specs():
    return Specs([
        value("op", "out") >= 3.0 * V,
        value("op", "out") <= 4.5 * V,
    ])
"""


def _run_plan_script(filename: str, source: str, label: str) -> Built:
    """Runs an inline plan script in a temporary directory and checks it built."""
    with tempfile.TemporaryDirectory() as directory:
        script_path = Path(directory) / filename
        script_path.write_text(source, encoding="utf-8")
        outcome = run_plan(
            Job(script=script_path, plan=script_path, max_unknowns=10),
            Path(directory),
        )
    if not isinstance(outcome, Built):
        message = f"{label} plan did not build: {outcome}"
        raise TypeError(message)
    return outcome


def extended_analyses() -> list[Fixture]:
    """The v2 analysis rows: corners, worst case, .disto, .sp, and transient noise."""

    def run_corners() -> Values:
        outcome = _run_plan_script("corners.py", _CORNERS_SCRIPT, "corners")
        report = outcome.analyses[0]
        if not isinstance(report, OperatingPointReport):
            message = f"expected operating point report, got {type(report).__name__}"
            raise TypeError(message)
        nominal = next(item.value for item in report.values if item.unknown == "out")
        lowest, highest = outcome.specs
        return {
            "v(out)@nominal": complex(nominal),
            "v(out)@hot_high": complex(
                lowest.value if lowest.value is not None else math.nan
            ),
            "v(out)@cold_low": complex(
                highest.value if highest.value is not None else math.nan
            ),
        }

    def run_worst_case() -> Values:
        outcome = _run_plan_script("worst_case.py", _WORST_CASE_SCRIPT, "worst-case")
        if outcome.worst_case is None:
            message = "worst-case plan produced no worst-case verdicts"
            raise RuntimeError(message)
        lowest, highest = outcome.worst_case.verdicts
        return {
            "v(out)@low": complex(
                lowest.value if lowest.value is not None else math.nan
            ),
            "v(out)@high": complex(
                highest.value if highest.value is not None else math.nan
            ),
        }

    cold_resistance = 1e3 * (1.0 + 0.01 * (-40.0 - 27.0))
    hot_resistance = 1e3 * (1.0 + 0.01 * (90.0 - 27.0))
    distortion_deck = (
        "* disto",
        "I1 0 out DC 1m DISTOF1 1m",
        "D1 out 0 dd",
        ".model dd D(IS=1e-14 N=1)",
    )
    scattering_deck = (
        "* sp",
        "V1 p1 0 DC 0 AC 1 portnum 1 z0 50",
        "R1 p1 p2 100",
        "V2 p2 0 DC 0 portnum 2 z0 50",
    )
    transient_noise_deck = (
        "* trnoise",
        "V1 out 0 DC 0 TRNOISE(1m 10n 0 0)",
        "R1 out 0 1k",
    )
    return [
        Fixture(
            key="corners",
            category=Category.ANALYSIS,
            name="Corners (temperature and supply)",
            known=Known(
                values={
                    "v(out)@nominal": divider_out(5.0, 1e3, 3e3),
                    "v(out)@hot_high": divider_out(5.5, hot_resistance, 3e3),
                    "v(out)@cold_low": divider_out(4.5, cold_resistance, 3e3),
                },
                source=(
                    "closed form: drifting divider at nominal (5 V, 27 C), "
                    "hot_high (5.5 V, 90 C), and cold_low (4.5 V, -40 C)"
                ),
            ),
            jolty=run_corners,
            spice=_unrun_spice(
                _ngspice_documented(
                    ", §18.5 / control language: corners run by altering temperature "
                    "and parameters in a control loop",
                    pending=True,
                ),
                _xyce_documented(
                    ", .STEP: sweeps parameters and temperature over combinations",
                    pending=True,
                ),
            ),
        ),
        Fixture(
            key="worst-case",
            category=Category.ANALYSIS,
            name="Worst-case tolerance analysis",
            known=Known(
                values={
                    "v(out)@low": divider_out(5.0, 1.1e3, 2.7e3),
                    "v(out)@high": divider_out(5.0, 0.9e3, 3.3e3),
                },
                source=(
                    "closed form: divider extremes at +/-10 % resistance "
                    "tolerance (R1 +10 %, R2 -10 % and R1 -10 %, R2 +10 %)"
                ),
            ),
            jolty=run_worst_case,
            spice=_unrun_spice(
                Unsupported(
                    NGSPICE_ANALYSES + "; no built-in worst-case tolerance analysis"
                ),
                Unsupported(XYCE_COMMANDS),
            ),
        ),
        Fixture(
            key="disto",
            category=Category.ANALYSIS,
            name="Small-signal distortion (.disto)",
            known=Known(
                values={
                    "v(hd2)": -THERMAL_VOLTAGE / 4.0,
                    "v(hd3)": THERMAL_VOLTAGE / 12.0,
                },
                source=(
                    "closed form: Volterra 2nd and 3rd harmonic amplitudes of "
                    "`v = Vt * ln(1 + i / Is)` driven by `I0 + I0 * cos(wt)`"
                ),
            ),
            jolty=Unsupported(
                JOLTY_ANALYSES + "; none computes small-signal distortion (.disto)"
            ),
            spice=Spice(
                deck=distortion_deck,
                # ngspice 47 crashes with an integer-underflow malloc on single-point
                # `disto lin 1 1k 1k`; `lin 2 1k 2k` runs and evaluates `1k` at index 0.
                # Its `disto2` (HD3) carries a known SPICE3f5 global sign inversion
                # (-Vt/12 instead of the exact Volterra/Fourier coefficient +Vt/12).
                ngspice=("disto lin 2 1k 2k\nsetplot disto1", "setplot disto2"),
                temperature=ngspice.same_kelvins(ROOM),
                xyce=Unsupported(XYCE_COMMANDS),
                reads={
                    "v(hd2)": Read(ngspice="v(out)", plot=0, complex=True),
                    "v(hd3)": Read(ngspice="v(out)", plot=1, complex=True),
                },
            ),
        ),
        Fixture(
            key="sp",
            category=Category.ANALYSIS,
            name="S-parameters (.sp)",
            known=Known(
                values={"s(1,1)": 0.5, "s(2,1)": 0.5, "s(1,2)": 0.5, "s(2,2)": 0.5},
                source=(
                    "closed form: 100 ohm series resistor between 50 ohm ports: "
                    "`S11 = S22 = R / (R + 2 * Z0)`, "
                    "`S21 = S12 = 2 * Z0 / (R + 2 * Z0)`"
                ),
            ),
            jolty=Unsupported(JOLTY_ANALYSES + "; none computes S-parameters (.sp)"),
            spice=Spice(
                deck=scattering_deck,
                ngspice=("sp lin 1 1Meg 1Meg",),
                xyce=_xyce_documented(
                    ", .LIN: linear multi-port S-parameter extraction with P port "
                    "devices",
                    pending=True,
                ),
                reads={
                    f"s({row},{col})": Read(ngspice=f"v(s_{row}_{col})", complex=True)
                    for row, col in ((1, 1), (2, 1), (1, 2), (2, 2))
                },
            ),
        ),
        Fixture(
            key="trnoise",
            category=Category.ANALYSIS,
            name="Transient noise",
            known=Known(
                source=(
                    "none: stochastic transient noise draw has no deterministic "
                    "per-sample closed form"
                )
            ),
            jolty=Unsupported(UNNOISY_TEMPERATURE),
            spice=Spice(
                deck=transient_noise_deck,
                ngspice=("tran 10n 1u 0 10n",),
                xyce=_xyce_unsupported(
                    ", Netlist Commands: no native TRNOISE transient noise source"
                ),
                reads={"v(out)@5e-07": Read(ngspice="v(out)", time=5e-7)},
            ),
        ),
    ]


def fixtures() -> list[Fixture]:
    """Every row of the matrix, in its order."""
    return [*analyses(), *device_fixtures(), *syntax(), *outputs()]
