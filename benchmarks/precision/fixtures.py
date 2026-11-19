# Copyright 2026 Maxence Faldor
"""Precision benchmark fixtures pairing circuits with exact analytical references.

Each fixture defines a circuit built via jolty's builder, a single analysis, and scored
outputs compared against independent references from `exact` (closed forms, Lambert's W,
and high-precision modal solutions). Comparison callbacks evaluate exact answers at the
sample points selected by each simulator alongside double-precision rounding bounds.

Reference equations consume the exact IEEE 754 doubles stored in each circuit so every
simulator is evaluated against identical parameters.
"""

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import exact
import mpmath
import numpy as np
from exact import Complex, Real
from mpmath import mpf
from mpmath.ctx_mp_python import _mpf
from simulators import Simulator

from jolty import Builder, Component, Role, _core, component, devices, units
from jolty.plan import (
    Ac,
    DcSweep,
    Noise,
    PeriodicSteadyState,
    PoleZero,
    Transient,
    decade,
    linear,
)
from jolty.units import (
    A,
    F,
    H,
    Hz,
    V,
    ohm,
    s,
)
from jolty.waveforms import Piecewise, Sine

type Score = Literal["relative", "mixed"]
"""Error metric mode: relative to exact magnitude, or mixed with a floor near zero."""

type Analysis = Transient | PeriodicSteadyState | DcSweep | Ac | Noise | PoleZero
"""Union of analysis types supported by precision fixtures."""


@dataclass(frozen=True, slots=True)
class Found:
    """Simulator outputs collected at the simulator's chosen sample coordinates.

    `points` stores timestamps, sweep levels, or frequencies (empty for pole-zero),
    while `values` maps unknown names (or ``noise`` and ``poles``) to arrays.
    """

    points: Real
    values: Mapping[str, Real | Complex]


@dataclass(frozen=True, slots=True)
class ScoredOutput:
    """Specification of one scored fixture output: metric mode, scale, and unit."""

    name: str
    score: Score
    scale: float
    unit: str


@dataclass(frozen=True, slots=True)
class Compared:
    """Paired simulated output, exact reference values, rounding bounds, and points.

    `bound` records the absolute double-precision rounding error of `exact`. Each array
    holds one entry per point (or one row per output across points) matching `points`.
    """

    simulated: Real | Complex
    exact: Real | Complex
    bound: Real
    points: Real


@dataclass(frozen=True, slots=True)
class Fixture:
    """Precision fixture combining a circuit, analysis, and exact scoring callback."""

    name: str
    describes: str
    reference: str
    """Description of how the exact reference answer is computed."""
    top: Component
    analysis: Analysis
    reads: tuple[str, ...]
    """Unknown names extracted from simulator results for comparison."""
    outputs: tuple[ScoredOutput, ...]
    compare: Callable[[Found], dict[str, Compared]]
    unsupported: Mapping[Simulator, str]
    """Mapping from simulator identifier to why it cannot execute the analysis."""


NGSPICE_PSS: Final = (
    "ngspice 47's pss searches an autonomous oscillator's frequency and returns its "
    "harmonics; it has no shooting for a circuit driven at a given period"
)
"""Explanation of why ngspice cannot run driven periodic steady-state analysis."""

PROBES: Final = 50
"""The most nodes of a network a fixture scores, spread evenly along its node
numbers: the first and the last among them."""

LADDER_SEED: Final = 1_000
"""The seed of a ladder's random values, plus its size."""

TREE_SEED: Final = 2_000
"""The seed of a random tree's shape and values, plus its size."""

DECADES: Final = 3
"""The decades each network's resistances and capacitances spread over, log-uniform:
10 Ω to 10 kΩ and 1 pF to 1 nF, so its sections' time constants spread over six."""

SETTLED: Final = 5
"""The slowest time constants a network's transient lasts, rounded up to a digit."""


def _level(points: Sequence[tuple[float, float]]) -> Piecewise | units.Quantity:
    """Converts piecewise-linear `points` into a `Piecewise` or constant voltage."""
    if len(points) == 1:
        return points[0][1] * V
    return Piecewise([(time * s, value * V) for time, value in points])


@component
def rc(
    c: Builder,
    resistance: float,
    capacitance: float,
    points: Sequence[tuple[float, float]],
) -> None:
    """Builds a series RC circuit driven by a piecewise-linear voltage source."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(_level(points)), name="V1")
    resistor = c.place(devices.resistor(resistance * ohm), name="R1")
    capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def rlc(
    c: Builder,
    resistance: float,
    inductance: float,
    capacitance: float,
    points: Sequence[tuple[float, float]],
) -> None:
    """Builds a series RLC low-pass circuit with a piecewise-linear voltage source."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(_level(points)), name="V1")
    resistor = c.place(devices.resistor(resistance * ohm), name="R1")
    inductor = c.place(devices.inductor(inductance * H), name="L1")
    capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, inductor.a, name="mid")
    c.connect(inductor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def tank(
    c: Builder,
    inductance: float,
    capacitance: float,
    points: Sequence[tuple[float, float]],
) -> None:
    """Builds an undamped parallel LC tank with a piecewise-linear current source."""
    ground = c.port("ground", Role.GROUND)
    level = Piecewise([(time * s, value * A) for time, value in points])
    source = c.place(devices.current_source(level), name="I1")
    inductor = c.place(devices.inductor(inductance * H), name="L1")
    capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
    c.connect(source.negative, inductor.a, capacitor.a, name="top")
    c.connect(source.positive, inductor.b, capacitor.b, ground)


@component
def diode_load(c: Builder, resistance: float, saturation: float) -> None:
    """Builds a DC voltage source driving a diode through a series resistor."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    resistor = c.place(devices.resistor(resistance * ohm), name="R1")
    diode = c.place(
        devices.diode(saturation_current=saturation * A, emission_coefficient=1.0),
        name="D1",
    )
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, diode.anode, name="anode")
    c.connect(diode.cathode, source.negative, ground)


@component
def driven_rc(
    c: Builder, resistance: float, capacitance: float, frequency: float
) -> None:
    """Builds a series RC circuit driven by a 1 V sinusoidal source at `frequency`."""
    ground = c.port("ground", Role.GROUND)
    sine = Sine(0 * V, 1 * V, frequency * Hz)
    source = c.place(devices.voltage_source(sine), name="V1")
    resistor = c.place(devices.resistor(resistance * ohm), name="R1")
    capacitor = c.place(devices.capacitor(capacitance * F), name="C1")
    c.connect(source.positive, resistor.a, name="in")
    c.connect(resistor.b, capacitor.a, name="out")
    c.connect(capacitor.b, source.negative, ground)


@component
def ladder_filter(c: Builder, terminations: float, values: Sequence[float]) -> None:
    """Builds a doubly terminated LC ladder of shunt capacitors and series inductors.

    Connects the first shunt capacitor to ``n1`` and the final one to ``out``.
    """
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(0 * V), name="V1")
    feed = c.place(devices.resistor(terminations * ohm), name="RS")
    c.connect(source.positive, feed.a, name="in")
    c.connect(source.negative, ground)
    shunts = (len(values) + 1) // 2
    nodes = [c.net("out" if k == shunts - 1 else f"n{k + 1}") for k in range(shunts)]
    c.connect(feed.b, nodes[0])
    for index, value in enumerate(values):
        if index % 2 == 0:
            shunt = c.place(devices.capacitor(value * F), name=f"C{index + 1}")
            c.connect(nodes[index // 2], shunt.a)
            c.connect(shunt.b, ground)
        else:
            series = c.place(devices.inductor(value * H), name=f"L{index + 1}")
            c.connect(nodes[index // 2], series.a)
            c.connect(series.b, nodes[index // 2 + 1])
    load = c.place(devices.resistor(terminations * ohm), name="RL")
    c.connect(nodes[-1], load.a)
    c.connect(load.b, ground)


@component
def network(
    c: Builder, shape: exact.Tree, points: Sequence[tuple[float, float]]
) -> None:
    """Builds an RC tree network with grounded capacitors driven by a voltage source.

    Uses a piecewise-linear source waveform and names tree nodes ``n0`` onward.
    """
    ground = c.port("ground", Role.GROUND)
    level = Piecewise([(time * s, value * V) for time, value in points])
    source = c.place(devices.voltage_source(level), name="V1")
    c.connect(source.negative, ground)
    nodes = [c.net(f"n{index}") for index in range(len(shape.capacitances))]
    feed = c.place(devices.resistor(1 / shape.source * ohm), name="R0")
    c.connect(source.positive, feed.a, name="src")
    c.connect(feed.b, nodes[0])
    for index, capacitance in enumerate(shape.capacitances):
        load = c.place(devices.capacitor(capacitance * F), name=f"C{index}")
        c.connect(nodes[index], load.a)
        c.connect(load.b, ground)
        if index:
            parent = shape.parents[index]
            branch = c.place(
                devices.resistor(1 / shape.branches[index] * ohm), name=f"R{index}"
            )
            c.connect(nodes[parent], branch.a)
            c.connect(branch.b, nodes[index])


def _transient(
    modal: Callable[[], exact.Modal], reads: Sequence[str], name: str
) -> Callable[[Found], dict[str, Compared]]:
    """Builds a callback scoring transient `reads` against `modal` under `name`."""

    def compare(found: Found) -> dict[str, Compared]:
        solved = modal()
        values, bound = solved.at(found.points)
        simulated = np.array([found.values[read] for read in reads])
        points = np.broadcast_to(found.points, values.shape)
        return {name: Compared(simulated, values, bound, points)}

    return compare


def _steps(
    name: str,
    describes: str,
    values: tuple[float, float],
    points: tuple[tuple[float, float], ...],
    stop: float,
) -> Fixture:
    """Builds a piecewise-linear RC transient fixture with `(R, C)` `values`."""
    resistance, capacitance = values

    @functools.cache
    def modal() -> exact.Modal:
        rate = 1 / (mpf(resistance) * mpf(capacitance))
        return exact.linear(["v(out)"], [[-rate]], [rate], [[mpf(1)]], points)

    return Fixture(
        name,
        describes,
        "its single mode's exact response to each straight piece of the source",
        rc(resistance, capacitance, points),
        Transient(stop=stop * s),
        ("out",),
        (ScoredOutput("v(out)", "mixed", 1.0, "V"),),
        _transient(modal, ("out",), "v(out)"),
        {},
    )


def rlc_step() -> Fixture:
    """Builds an under-damped series RLC fixture driven by a 1 V, 1 ns step."""
    resistance, inductance, capacitance = 10.0, 1e-6, 1e-9
    points = ((0.0, 0.0), (1e-9, 1.0))

    @functools.cache
    def modal() -> exact.Modal:
        r, henries, c = mpf(resistance), mpf(inductance), mpf(capacitance)
        states = [[mpf(0), 1 / c], [-1 / henries, -r / henries]]
        return exact.linear(
            ["v(out)"], states, [mpf(0), 1 / henries], [[mpf(1), mpf(0)]], points
        )

    return Fixture(
        "rlc_step",
        "a series RLC of 10 Ω, 1 µH and 1 nF, damped at 0.16 of critical, under a "
        "1 V step rising in 1 ns, for 2 µs, 10 periods of its ringing",
        "its two complex modes' exact response, by mpmath's eigenvectors",
        rlc(resistance, inductance, capacitance, points),
        Transient(stop=2e-6 * s),
        ("out",),
        (ScoredOutput("v(out)", "mixed", 1.0, "V"),),
        _transient(modal, ("out",), "v(out)"),
        {},
    )


def lc_tank() -> Fixture:
    """Builds an undamped LC tank fixture excited by a 1 ns current turn-off."""
    inductance, capacitance = 1e-6, 1e-9
    points = ((0.0, 1e-3), (1e-9, 0.0))

    @functools.cache
    def modal() -> exact.Modal:
        henries, c = mpf(inductance), mpf(capacitance)
        states = [[mpf(0), -1 / c], [1 / henries, mpf(0)]]
        read = [[mpf(1), mpf(0)], [mpf(0), mpf(1)]]
        return exact.linear(["v(top)", "i(L1)"], states, [1 / c, mpf(0)], read, points)

    def compare(found: Found) -> dict[str, Compared]:
        (voltage, current), (voltage_bound, current_bound) = modal().at(found.points)
        simulated_voltage = np.asarray(found.values["top"])
        simulated_current = np.asarray(found.values["L1:current"])
        energy = (capacitance * voltage**2 + inductance * current**2) / 2
        energy_bound = (
            capacitance * np.abs(voltage) * voltage_bound
            + inductance * np.abs(current) * current_bound
            + exact.EPSILON * energy * 2
        )
        simulated_energy = (
            capacitance * simulated_voltage**2 + inductance * simulated_current**2
        ) / 2
        free = found.points >= points[-1][0]
        angle = np.unwrap(
            np.arctan2(
                simulated_current[free] * math.sqrt(inductance),
                simulated_voltage[free] * math.sqrt(capacitance),
            )
        )
        span = found.points[free][-1] - found.points[free][0]
        frequency = (angle[-1] - angle[0]) / (2 * math.pi * span)
        natural = float(
            1 / (2 * mpmath.pi * mpmath.sqrt(mpf(inductance) * mpf(capacitance)))
        )
        return {
            "energy": Compared(simulated_energy, energy, energy_bound, found.points),
            "frequency": Compared(
                np.array([frequency]),
                np.array([natural]),
                np.array([4 * exact.EPSILON * natural]),
                np.array([found.points[-1]]),
            ),
        }

    return Fixture(
        "lc_tank",
        "an undamped tank of 1 µH and 1 nF, its inductor carrying 1 mA from a current "
        "source turned off in 1 ns, for 4 µs, 20 periods: its energy at every point, "
        "and its frequency over the run, from the phase its state turned through",
        "its two imaginary modes' exact response, by mpmath's eigenvectors",
        tank(inductance, capacitance, points),
        Transient(stop=4e-6 * s),
        ("top", "L1:current"),
        (
            ScoredOutput("energy", "relative", 0.0, "J"),
            ScoredOutput("frequency", "relative", 0.0, "Hz"),
        ),
        compare,
        {},
    )


def diode() -> Fixture:
    """Builds a DC-swept resistor-diode fixture from 0.1 V to 5 V."""
    resistance, saturation = 1e3, 1e-14
    thermal = (
        mpf(_core.BOLTZMANN)
        * mpf(_core.REFERENCE_TEMPERATURE)
        / mpf(_core.ELEMENTARY_CHARGE)
    )
    gmin = 1e-12

    def compare(found: Found) -> dict[str, Compared]:
        voltages = exact.diode(found.points, resistance, saturation, thermal, gmin)
        return {
            "v(anode)": Compared(
                np.asarray(found.values["anode"]),
                voltages,
                exact.EPSILON * np.abs(voltages),
                found.points,
            )
        }

    return Fixture(
        "diode",
        "a diode of SPICE's junction, 10 fA of saturation current, fed through 1 kΩ "
        "from a source swept from 0.1 V to 5 V in 50 steps, at 27 °C",
        "Lambert's W, the junction's gmin of 1 pS across it, in mpmath",
        diode_load(resistance, saturation),
        DcSweep("V1", linear(0.1 * V, 5 * V, 50)),
        ("anode",),
        (ScoredOutput("v(anode)", "relative", 0.0, "V"),),
        compare,
        {},
    )


def rlc_ac() -> Fixture:
    """Builds a series RLC AC fixture (Q = 31.6) swept from 1 Hz to 1 GHz."""
    resistance, inductance, capacitance = 1.0, 1e-6, 1e-9

    def form(frequency: _mpf) -> object:
        omega = 2 * mpmath.pi * frequency
        r, henries, c = mpf(resistance), mpf(inductance), mpf(capacitance)
        return 1 / (1 - omega**2 * henries * c + 1j * omega * r * c)

    def compare(found: Found) -> dict[str, Compared]:
        response = exact.closed(form, found.points)
        return {
            "v(out)": Compared(
                np.asarray(found.values["out"]),
                response,
                exact.EPSILON * np.abs(response),
                found.points,
            )
        }

    return Fixture(
        "rlc_ac",
        "the small-signal response of a series RLC low-pass of 1 Ω, 1 µH and 1 nF, "
        "its quality factor 31.6, at 20 frequencies a decade from 1 Hz to 1 GHz",
        "its transfer function, 1/(1 - ω²LC + jωRC), in mpmath",
        rlc(resistance, inductance, capacitance, ((0.0, 0.0),)),
        Ac("V1", decade(1 * Hz, 1e9 * Hz, 20)),
        ("out",),
        (ScoredOutput("v(out)", "relative", 0.0, "V"),),
        compare,
        {},
    )


def rc_noise() -> Fixture:
    """Builds a thermal-noise fixture (1 kΩ into 1 nF) from 1 Hz to 1 GHz."""
    resistance, capacitance = 1e3, 1e-9

    def form(frequency: _mpf) -> object:
        omega = 2 * mpmath.pi * frequency
        kt = mpf(_core.BOLTZMANN) * mpf(_core.REFERENCE_TEMPERATURE)
        r, c = mpf(resistance), mpf(capacitance)
        return 4 * kt * r / (1 + (omega * r * c) ** 2)

    def compare(found: Found) -> dict[str, Compared]:
        density = exact.closed(form, found.points).real
        return {
            "noise": Compared(
                np.asarray(found.values["noise"]),
                density,
                exact.EPSILON * np.abs(density),
                found.points,
            )
        }

    return Fixture(
        "rc_noise",
        "the thermal noise of 1 kΩ at the capacitor of 1 nF it feeds, at 10 "
        "frequencies a decade from 1 Hz to 1 GHz, at 27 °C",
        "4kTR/(1 + (ωRC)²), with the SI's exact k, in mpmath",
        rc(resistance, capacitance, ((0.0, 0.0),)),
        Noise("out", decade(1 * Hz, 1e9 * Hz, 10), source="V1"),
        (),
        (ScoredOutput("noise", "relative", 0.0, "V²/Hz"),),
        compare,
        {},
    )


BUTTERWORTH: Final = (5, 50.0, 2 * math.pi * 1e6)
"""The Butterworth ladder's order, its terminations in ohms, and its cutoff in radians
per second."""


def butterworth_values() -> tuple[float, ...]:
    """Computes alternating Butterworth ladder C and L values from source to load.

    Scales normalized values ``gₖ = 2 sin((2k - 1)π/2n)`` by termination resistance
    and cutoff frequency, rounded to IEEE 754 doubles matching the circuit.
    """
    order, terminations, cutoff = BUTTERWORTH
    normalized = [
        2 * math.sin((2 * k - 1) * math.pi / (2 * order)) for k in range(1, order + 1)
    ]
    return tuple(
        g / (terminations * cutoff) if k % 2 else g * terminations / cutoff
        for k, g in enumerate(normalized, start=1)
    )


def butterworth_states(values: Sequence[float]) -> list[list[_mpf]]:
    """Builds the Butterworth ladder state matrix ``A`` (``ẋ = A x``) in mpmath.

    Orders state variables as alternating capacitor voltages and inductor currents.
    """
    r = mpf(BUTTERWORTH[1])
    c1, l2, c3, l4, c5 = (mpf(value) for value in values)
    zero = mpf(0)
    return [
        [-1 / (r * c1), -1 / c1, zero, zero, zero],
        [1 / l2, zero, -1 / l2, zero, zero],
        [zero, 1 / c3, zero, -1 / c3, zero],
        [zero, zero, 1 / l4, zero, -1 / l4],
        [zero, zero, zero, 1 / c5, -1 / (r * c5)],
    ]


def butterworth_poles() -> Fixture:
    """Builds a fifth-order Butterworth LC ladder pole-zero fixture (50 Ω)."""
    terminations = BUTTERWORTH[1]
    values = butterworth_values()

    @functools.cache
    def found_poles() -> Complex:
        return exact.poles(butterworth_states(values))

    def compare(found: Found) -> dict[str, Compared]:
        expected = found_poles()
        simulated = np.asarray(found.values["poles"], dtype=np.complex128)
        if len(simulated) != len(expected):
            message = (
                f"found {len(simulated)} poles, where the circuit has {len(expected)}"
            )
            raise ValueError(message)
        remaining = list(simulated)
        matched = []
        for pole in expected:
            nearest = min(
                range(len(remaining)), key=lambda index: abs(remaining[index] - pole)
            )
            matched.append(remaining.pop(nearest))
        return {
            "poles": Compared(
                np.array(matched),
                expected,
                exact.EPSILON * np.abs(expected),
                np.arange(len(expected), dtype=np.float64),
            )
        }

    return Fixture(
        "butterworth_poles",
        "the poles of a fifth-order Butterworth LC ladder between terminations of "
        "50 Ω, cut off at 1 MHz: three capacitors and two inductors",
        "the eigenvalues of its state equations, by mpmath",
        ladder_filter(terminations, values),
        PoleZero(source="V1", output="out"),
        (),
        (ScoredOutput("poles", "relative", 0.0, "rad/s"),),
        compare,
        {},
    )


def rc_periodic() -> Fixture:
    """Builds a driven RC periodic steady-state fixture (tau = 10 periods)."""
    resistance, capacitance, frequency = 1e3, 10e-6, 1e3
    omega = 2 * mpmath.pi * mpf(frequency)
    gain = 1 / (1 + 1j * omega * mpf(resistance) * mpf(capacitance))

    def form(time: _mpf) -> object:
        return mpmath.im(gain * mpmath.exp(1j * omega * time))

    def compare(found: Found) -> dict[str, Compared]:
        waveform = exact.closed(form, found.points).real
        return {
            "v(out)": Compared(
                np.asarray(found.values["out"]),
                waveform,
                exact.EPSILON * (np.abs(waveform) + float(abs(gain))),
                found.points,
            )
        }

    return Fixture(
        "rc_periodic",
        "the periodic steady state of 1 kΩ into 10 µF driven by a sine of 1 V at "
        "1 kHz, its time constant ten periods: one period",
        "the phasor's steady state, |H| sin(ωt + arg H), in mpmath",
        driven_rc(resistance, capacitance, frequency),
        PeriodicSteadyState(period=1e-3 * s),
        ("out",),
        (ScoredOutput("v(out)", "mixed", float(abs(gain)), "V"),),
        compare,
        {Simulator.NGSPICE_SPARSE: NGSPICE_PSS, Simulator.NGSPICE_KLU: NGSPICE_PSS},
    )


def networked(kind: Literal["ladder", "tree"], size: int) -> Fixture:
    """Builds a random-valued `size`-node RC ladder or tree step-response fixture."""
    generator = np.random.default_rng(
        (LADDER_SEED if kind == "ladder" else TREE_SEED) + size
    )
    parents = tuple(
        0
        if node == 0
        else node - 1
        if kind == "ladder"
        else int(generator.integers(0, node))
        for node in range(size)
    )
    resistances = 10 ** generator.uniform(1, 1 + DECADES, size)
    capacitances = 10 ** generator.uniform(-12, -12 + DECADES, size)
    shape = exact.Tree(
        parents,
        tuple(float(1 / value) for value in resistances),
        tuple(float(value) for value in capacitances),
        float(1 / resistances[0]),
    )
    rates = _estimated_rates(shape)
    slowest = SETTLED / rates.min()
    exponent = math.floor(math.log10(slowest))
    stop = math.ceil(slowest / 10**exponent) * 10**exponent
    points = ((0.0, 0.0), (1e-9, 1.0))
    probes = tuple(
        int(index)
        for index in np.unique(np.linspace(0, size - 1, min(size, PROBES)).round())
    )
    reads = tuple(f"n{probe}" for probe in probes)

    @functools.cache
    def modal() -> exact.Modal:
        return exact.tree(shape, probes, points)

    spread = math.log10(rates.max() / rates.min())
    described = (
        f"a {'ladder' if kind == 'ladder' else 'random tree'} of {size:,} nodes, its "
        "resistances log-uniform from 10 Ω to 10 kΩ and its capacitances to the ground "
        f"from 1 pF to 1 nF, its modes' rates spread over {spread:.1f} decades, under "
        f"a step of 1 V rising in 1 ns, for {stop:g} s, five of its slowest time "
        f"constants: {len(probes)} of its nodes"
    )
    return Fixture(
        f"{kind}_{size}",
        described,
        "its symmetric pencil's eigenvalues by Newton's iteration on the tree's "
        "determinant, each placed by Sylvester's inertia, and its eigenvectors by "
        "inverse iteration, in mpmath",
        network(shape, points),
        Transient(stop=stop * s),
        reads,
        (ScoredOutput("v(nodes)", "mixed", 1.0, "V"),),
        _transient(modal, reads, "v(nodes)"),
        {},
    )


def _estimated_rates(shape: exact.Tree) -> Real:
    """Estimates modal decay rates of `shape` via double-precision LAPACK eigenvalues.

    Used solely to determine the transient simulation stop time.
    """
    count = len(shape.capacitances)
    conductances = np.zeros((count, count))
    conductances[0, 0] += shape.source
    for node in range(1, count):
        parent, g = shape.parents[node], shape.branches[node]
        conductances[node, node] += g
        conductances[parent, parent] += g
        conductances[node, parent] -= g
        conductances[parent, node] -= g
    root = 1 / np.sqrt(np.array(shape.capacitances))
    return np.linalg.eigvalsh(root[:, None] * conductances * root[None, :])


def fixtures() -> dict[str, Fixture]:
    """Returns all precision benchmark fixtures keyed by name."""
    made = [
        _steps(
            "rc_step",
            "1 kΩ into 1 nF under a 1 V step rising in 1 ns, for 5 µs, five time "
            "constants",
            (1e3, 1e-9),
            ((0.0, 0.0), (1e-9, 1.0)),
            5e-6,
        ),
        _steps(
            "rc_ramp",
            "1 kΩ into 1 nF under a ramp of 1 V over 2 µs, then held, for 6 µs",
            (1e3, 1e-9),
            ((0.0, 0.0), (2e-6, 1.0)),
            6e-6,
        ),
        _steps(
            "rc_femto",
            "1 MΩ into 1 fF, a charge of 1 fC at most, under a 1 V step rising in "
            "1 ps, for 5 ns, five time constants",
            (1e6, 1e-15),
            ((0.0, 0.0), (1e-12, 1.0)),
            5e-9,
        ),
        rlc_step(),
        lc_tank(),
        diode(),
        rlc_ac(),
        rc_noise(),
        butterworth_poles(),
        rc_periodic(),
        *(
            networked(kind, size)
            for kind in ("ladder", "tree")
            for size in (10, 100, 1000)
        ),
    ]
    return {fixture.name: fixture for fixture in made}
