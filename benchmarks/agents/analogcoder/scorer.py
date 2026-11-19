# Copyright 2026 Maxence Faldor
"""Benchmark scorer for all twenty-four AnalogCoder circuit design problems.

AnalogCoder (Lai et al., AAAI 2025; github.com/laiyao1/AnalogCoder, MIT License)
evaluates twenty-four circuits built from two level 1 MOSFET models on a 5 V supply.
The original suite verifies each circuit via ``problem_check`` simulation scripts and
verifies problems 1 through 15 via ``check_netlist`` operating-point checks. The scorer
reproduces those stimuli and tolerances on a constructed test bench and passes a
submission only when all twenty-four problems pass.

Several checks are tightened over the original SPICE-name-based scripts to enforce
intent, as documented in the task prompt:

- Every design is a named-port component with declared DC input levels and R1/Cf or
  Rf/C1 values matching AnalogCoder's check values.
- Integrators and differentiators must invert (which passive RC networks cannot),
  replacing AnalogCoder's op-amp-removal heuristic.
- Adder error tolerance is 20 % of the input sum rather than 20 % of output voltage,
  preventing dropped-input solutions from passing.
- Current sources must keep every load-step current delta within 1 µA rather than
  checking a single step.
- Oscillators start from a switched-on supply without initial conditions and must
  sustain steady oscillation across the second half of the transient.
- The voltage-controlled oscillator is tested at 3.25 V, 3.5 V, and 4.0 V (scaling
  AnalogCoder's 0.65/0.7/0.8 V on 1 V to 5 V) and requires monotonic period change.
- The phase-locked loop operates at 5 V and must lock to both 10 MHz and 12 MHz,
  ruling out free-running oscillators.
- Designs may use only the two given MOSFETs, resistors, capacitors, and DC sources.
"""

import json
import runpy
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from arms import Arm
from scoring import (
    built,
    failed,
    missing,
    modeled,
    netlist,
    script,
    unbuilt,
    unmet,
)
from verdict import Shortfall, Verdict

from jolty import Builder, Component, Role, Simulation, circuit, component, devices, ir
from jolty.errors import JoltyError
from jolty.plan import Ac, DcSweep, OperatingPoint, Transient, linear
from jolty.results import OperatingPointResult as Solution
from jolty.results import Real
from jolty.units import A, Hz, V, ns, ohm, s, um, us
from jolty.waveforms import Piecewise, Pulse

type Kind = Literal[
    "amplifier",
    "inverter",
    "mirror",
    "opamp",
    "oscillator",
    "integrator",
    "differentiator",
    "adder",
    "subtractor",
    "schmitt",
    "vco",
    "pll",
]
type Levels = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class CircuitTask:
    """Specification of one AnalogCoder problem kind and its non-supply ports."""

    kind: Kind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...] = ("vout",)


PROBLEMS = {
    1: CircuitTask("amplifier", ("vin",)),
    2: CircuitTask("amplifier", ("vin",)),
    3: CircuitTask("amplifier", ("vin",)),
    4: CircuitTask("amplifier", ("vin", "vbias")),
    5: CircuitTask("amplifier", ("vin", "vbias")),
    6: CircuitTask("inverter", ("vin",)),
    7: CircuitTask("inverter", ("vin",)),
    8: CircuitTask("mirror", ("vbias",)),
    9: CircuitTask("amplifier", ("vin",)),
    10: CircuitTask("amplifier", ("vin",)),
    11: CircuitTask("opamp", ("vinp", "vinn", "vbias"), ("voutp", "vout")),
    12: CircuitTask("mirror", ("iref",), ("iout",)),
    13: CircuitTask("opamp", ("vinp", "vinn")),
    14: CircuitTask(
        "opamp", ("vinp", "vinn", "vbias1", "vbias2", "vbias3"), ("voutp", "vout")
    ),
    15: CircuitTask(
        "opamp",
        ("vinp", "vinn", "vbias1", "vbias2", "vbias3", "vbias4"),
        ("voutp", "vout"),
    ),
    16: CircuitTask("oscillator", ()),
    17: CircuitTask("oscillator", ()),
    18: CircuitTask("integrator", ("vin",)),
    19: CircuitTask("differentiator", ("vin",)),
    20: CircuitTask("adder", ("vin1", "vin2")),
    21: CircuitTask("subtractor", ("vin1", "vin2")),
    22: CircuitTask("schmitt", ("vin",)),
    23: CircuitTask("vco", ("vin",)),
    24: CircuitTask("pll", ("clk_ref",), ("clk_p",)),
}
"""Mapping from AnalogCoder problem number to `CircuitTask`; see ``task.md``."""

SUPPLY = 5.0
"""Nominal supply voltage in volts."""
SWITCHED_ON = Piecewise(((0 * s, 0 * V), (1 * us, SUPPLY * V)))
"""Piecewise supply ramp switching from 0 V to 5 V over 1 µs for oscillators."""
RESOLUTION = 1e-6
"""Minimum voltage margin in volts used for operating-point saturation checks."""
MIN_MAGNITUDE = 1e-5
"""Minimum drain current for amplifiers, op-amps, and current sources in amperes."""
GAIN = 1e-5
"""Minimum 100 Hz gain for amplifiers and differential-mode op-amps."""
MOVES = 1e-3
"""Minimum output voltage swing in volts across a full-supply DC input sweep."""
SWING = 1.0
"""Minimum separation in volts between high and low inverter output levels."""
LOADS = (100.0, 300.0, 500.0, 750.0, 1000.0)
"""Load resistances in ohms for current-source tests; the middle load checks biasing."""
STEADY = 1e-6
"""Current-source load-step ceiling and mirror tracking floor in amperes."""
REFERENCE = 1.55e-3
"""Reference current in amperes applied to test current-mirror tracking."""
STEP = 0.5
"""Input step amplitude in volts for integrator, differentiator, and adder tests."""
TIME_CONSTANT = 10e3 * 3e-6
"""Reference RC time constant (10 kΩ * 3 µF) in seconds for problems 18 and 19."""
HALF = 10e-3
"""Half-period duration in seconds for the integrator square-wave input."""
HALVES = 20
"""Total number of input half-periods simulated in the integrator test."""
EDGE = 1e-6
"""Initial rest time and edge transition duration in seconds for integrator steps."""
INTEGRATED = STEP / TIME_CONSTANT
"""Expected integrator output ramp slope magnitude in volts per second."""
RAMP = 50e-3
"""Half-period ramp duration in seconds for the differentiator triangle wave."""
DIFFERENTIATED = TIME_CONSTANT * 2 * STEP / RAMP
"""Expected differentiator output magnitude relative to input DC level in volts."""
ADDEND = 0.125
"""Fixed voltage offset in volts holding ``vin2`` above ``vin1`` in the adder test."""
DIFFERENCES = (2.25, 1.75)
"""Start and end ``vin2 - vin1`` voltage differences in volts for subtractor sweep."""
FIT = 0.9
"""Minimum coefficient of determination (R^2) required for integrator output ramps."""
TOLERANCE = 0.2
"""Relative error tolerance for differentiator, adder, and subtractor checks."""
SLOPE_TOLERANCE = 0.3
"""Relative slope error tolerance for the integrator check."""
FLAT = 0.1
"""Maximum relative RMS ripple permitted on differentiator square-wave plateaus."""
HYSTERESIS = 0.05
"""Minimum threshold separation in volts required for a Schmitt trigger."""
NOISE = 1e-6
"""Maximum downward dip in volts tolerated in a rising Schmitt trigger sweep."""
PERIODS = 3
"""Minimum number of full oscillation cycles required in the second half of a run."""
SPREAD = 0.2
"""Maximum relative standard deviation of cycle periods for steady oscillation."""
DECAY = 0.5
"""Minimum ratio of final cycle swing to peak swing preventing decaying ringing."""
QUIET = 1e-3
"""Minimum peak-to-peak voltage swing in volts required to count as oscillating."""
BAND = 0.1
"""Hysteresis band as a fraction of swing used when detecting rising crossings."""
CONTROLS = (3.25, 3.5, 4.0)
"""Control voltage levels in volts tested on the voltage-controlled oscillator."""
TUNED = 0.01
"""Minimum relative period change required between consecutive VCO control levels."""
CLOCKS = (10e6, 12e6)
"""Reference clock frequencies in hertz that the phase-locked loop must lock onto."""
LOCKED = 0.05
"""Maximum relative period error permitted between locked PLL output and reference."""
CLOCK_EDGE = 0.02
"""Reference clock rise and fall time as a fraction of the clock period."""
CLOCK_WIDTH = 0.48
"""Reference clock high pulse width as a fraction of the clock period."""
CLOCK_BAND = 0.25
"""Hysteresis margin in volts around mid-supply for PLL clock rising-edge detection."""
LOCK_RUN = 10e-6
"""Total transient simulation duration in seconds for phase-locked loop locking."""
LOCK_WINDOW = 2e-6
"""Final observation window in seconds used to measure locked PLL output period."""
ALLOWED = {"resistor", "capacitor"}
"""Passive device models permitted in addition to the given MOSFETs and DC sources."""
RESISTIVE = frozenset({1, 2, 3, 4, 5, 13})
"""Problem numbers whose topology requires at least one resistor."""
COMMON_DRAIN = 3
"""Problem number for the source follower, whose output connects to no gate or drain."""
COMMON_GATE = 4
"""Problem number for the common-gate stage, whose input joins no gate or drain."""
MILLER = 9
"""Problem number for the two-stage amplifier with a Miller capacitor to ``vout``."""
DIODE_LOAD = 10
"""Problem number for the common-source amplifier with a diode-connected load."""


@dataclass(frozen=True, slots=True)
class Part:
    """Flattened device record storing path, model, parameters, and connected nets."""

    path: str
    model: str
    parameters: Mapping[str, object]
    nets: Mapping[str, str]


def walk(
    built: ir.Circuit, name: str, prefix: str, ports: Mapping[str, str]
) -> Iterator[Part]:
    """Yields flattened `Part` entries for every leaf device inside component `name`.

    Prefixes internal net names with `prefix` while remapping component ports via
    `ports` to match engine net naming.
    """
    for instance in built.components[name].instances:
        nets = {
            terminal: ports.get(net, f"{prefix}{net}")
            for terminal, net in instance.terminals.items()
        }
        match instance.of:
            case ir.ComponentOf(component=inner):
                yield from walk(built, inner, f"{prefix}{instance.name}/", nets)
            case ir.DeviceOf(model=model):
                yield Part(f"{prefix}{instance.name}", model, instance.parameters, nets)


def designed(built: ir.Circuit) -> list[Part]:
    """Returns all flattened `Part` instances inside the benched design ``X1``."""
    design = built.find("X1")
    if design is None or not isinstance(design.of, ir.ComponentOf):
        message = "the bench places no design"
        raise ValueError(message)
    return list(walk(built, design.of.component, "X1/", design.terminals))


def _size_free(parameters: Mapping[str, object]) -> dict[str, object]:
    return {
        name: value
        for name, value in parameters.items()
        if name not in {"width", "length"}
    }


def transistors(task: Path, arm: Arm) -> dict[object, dict[str, object]]:
    """Loads size-independent parameters of the given NMOS and PMOS models for `arm`.

    Calls ``parts.py`` factories for jolty or reads ``parts.lib`` models for ngspice,
    stripping channel width and length.
    """
    if arm is Arm.JOLTY:
        parts = runpy.run_path(str(task / "given" / "parts.py"))
        made = [parts[name](1 * um, 1 * um) for name in ("nmos", "pmos")]
    else:
        made = [modeled(task, name) for name in ("nmos", "pmos")]
    found = {}
    for each in made:
        if not isinstance(each, devices.Device):
            message = f"the given {each} is no device"
            raise TypeError(message)
        found[each.parameters["channel"]] = _size_free(each.parameters)
    return found


def disallowed(
    made: Component, given: Mapping[object, Mapping[str, object]]
) -> list[str]:
    """Returns violation messages for any device in `made` outside allowed parts."""
    built = circuit(made)
    top = built.components[built.top]
    found: list[str] = []
    for part in walk(
        built, built.top, "", {port.name: port.name for port in top.ports}
    ):
        if part.model == "mosfet":
            if _size_free(part.parameters) != given.get(part.parameters.get("channel")):
                found.append(f"{part.path} is not a transistor of parts.py")
        elif part.model in {"voltage_source", "current_source"}:
            waveform = part.parameters.get("waveform")
            if not isinstance(waveform, dict) or set(waveform) != {"dc"}:
                found.append(f"{part.path} is a source whose level changes")
        elif part.model not in ALLOWED:
            found.append(f"{part.path} is a {part.model}, which the task forbids")
    return found


def driven(port: str, level: float) -> devices.Device:
    """Creates a DC source for `port` at `level` (amperes for ``iref``, else volts)."""
    if port == "iref":
        return devices.current_source(level * A)
    return devices.voltage_source(level * V)


def held(problem: CircuitTask, levels: Levels) -> dict[str, devices.Device]:
    """Builds DC bias sources holding every input of `problem` at `levels`."""
    return {port: driven(port, levels[port]) for port in problem.inputs}


@component
def bench(
    c: Builder,
    made: Component,
    problem: CircuitTask,
    drives: Mapping[str, devices.Device],
    load: float | None = None,
) -> None:
    """Builds the test bench wrapping `made` with supply, `drives`, and optional `load`.

    Defaults ``vdd`` to 5 V unless overridden in `drives`. Connects voltage input
    sources relative to ground and current input sources from ``vdd``.
    """
    ground = c.port("ground", Role.GROUND)
    vdd = c.net("vdd")
    power = c.place(drives.get("vdd", devices.voltage_source(SUPPLY * V)), name="VDD")
    placed = c.place(made, name="X1")
    c.connect(power.positive, placed.vdd, vdd)
    c.connect(power.negative, placed.ground, ground)
    for port, source in drives.items():
        if port == "vdd":
            continue
        driver = c.place(source, name=port.upper())
        if source.model == "current_source":
            c.connect(driver.positive, vdd)
            c.connect(driver.negative, placed[port], name=port)
        else:
            c.connect(driver.positive, placed[port], name=port)
            c.connect(driver.negative, ground)
    for port in problem.outputs:
        c.connect(placed[port], name=port)
    if load is not None:
        resistor = c.place(devices.resistor(load * ohm), name="RLOAD")
        c.connect(resistor.a, vdd)
        c.connect(resistor.b, placed[problem.outputs[0]])


def volts(solution: Solution, net: str) -> float:
    """Returns the node voltage of `net` from `solution`, treating ``ground`` as 0 V."""
    return 0.0 if net == "ground" else solution.voltage(net)


def recorded(parts: list[Part], *also: str) -> OperatingPoint:
    """Builds an `OperatingPoint` recording every MOSFET in `parts` plus `also` paths.

    Recorded operating-point data is inspected by `misbiased`.
    """
    return OperatingPoint(
        record=[*(part.path for part in parts if part.model == "mosfet"), *also]
    )


def misbiased(solution: Solution, parts: list[Part], least: float = 0.0) -> list[str]:
    """Checks that every MOSFET in `parts` conducts forward above `least` amperes.

    Requires gate-source voltage above threshold and drain-source voltage positive in
    the channel sense by at least `RESOLUTION`.
    """
    found: list[str] = []
    for part in parts:
        if part.model != "mosfet":
            continue
        sense = 1.0 if part.parameters.get("channel") == "n" else -1.0
        gate, drain, source = (
            sense * volts(solution, part.nets[terminal])
            for terminal in ("gate", "drain", "source")
        )
        threshold = abs(solution.variable(part.path, "threshold_voltage"))
        current = sense * solution.current(part.path, "drain")
        if gate - source - threshold < RESOLUTION:
            found.append(
                f"{part.path}'s gate is {gate - source:.4g} V beyond its source, "
                f"not beyond its {threshold:.4g} V threshold"
            )
        if drain - source < RESOLUTION:
            found.append(
                f"{part.path}'s drain is {drain - source:.4g} V beyond its source"
            )
        elif current < least:
            found.append(f"{part.path} carries {current:.4g} A, less than {least:g}")
    return found


def kept_off(parts: list[Part], net: str) -> list[str]:
    """Returns violations if `net` connects to any MOSFET gate or drain in `parts`."""
    return [
        f"{net} joins {part.path}'s {terminal}"
        for part in parts
        if part.model == "mosfet"
        for terminal in ("gate", "drain")
        if part.nets[terminal] == net
    ]


def structure(number: int, parts: list[Part]) -> list[str]:
    """Checks topology-specific structural requirements for problem `number`."""
    found: list[str] = []
    if number in RESISTIVE and not any(part.model == "resistor" for part in parts):
        found.append("it holds no resistor")
    if number == COMMON_DRAIN:
        found += kept_off(parts, "vout")
    if number == COMMON_GATE:
        found += kept_off(parts, "vin")
    mosfets = [part for part in parts if part.model == "mosfet"]
    if number == MILLER:
        stages = {part.nets["drain"] for part in mosfets if part.nets["gate"] == "vin"}
        if not any(
            part.model == "capacitor"
            and "vout" in part.nets.values()
            and stages & (set(part.nets.values()) - {"vout"})
            for part in parts
        ):
            found.append("no capacitor joins vout to a drain whose gate is vin")
    if number == DIODE_LOAD and not any(
        part.nets["gate"] == part.nets["drain"] for part in mosfets
    ):
        found.append("no MOSFET's gate joins its drain")
    return found


def amplifier(
    number: int, made: Component, problem: CircuitTask, levels: Levels
) -> list[str]:
    """Runs operating-point, DC sweep, and 100 Hz AC gain checks for an amplifier."""
    built = circuit(bench(made, problem, held(problem, levels)))
    simulation = Simulation(built)
    parts = designed(built)
    found = structure(number, parts)
    found += misbiased(simulation.run(recorded(parts)), parts, MIN_MAGNITUDE)
    swept = simulation.run(DcSweep("VIN", linear(0 * V, SUPPLY * V, 501)))
    moved = float(np.ptp(swept.voltage("vout")))
    if moved < MOVES:
        found.append(f"vout moves {moved:.3g} V as vin sweeps the supply")
    gain = float(abs(simulation.run(Ac("VIN", [100 * Hz])).voltage("vout")[0]))
    if gain <= GAIN:
        found.append(f"its gain at 100 Hz is {gain:.3g}")
    return found


def inverter(_: int, made: Component, problem: CircuitTask, __: Levels) -> list[str]:
    """Runs high and low DC transfer checks for an inverter at 0 V and 5 V."""
    drives = {"vin": driven("vin", 0.0)}
    swept = Simulation(circuit(bench(made, problem, drives))).run(
        DcSweep("VIN", [0 * V, SUPPLY * V])
    )
    high, low = (float(level) for level in swept.voltage("vout"))
    middle = SUPPLY / 2
    if high < middle or low > middle or high - low < SWING:
        return [f"vout is {high:.4g} V with vin at 0 V and {low:.4g} V at 5 V"]
    return []


def mirror(_: int, made: Component, problem: CircuitTask, levels: Levels) -> list[str]:
    """Runs load-regulation and tracking checks for a current source or mirror."""
    found: list[str] = []
    currents = []
    middle = LOADS[len(LOADS) // 2]
    for load in LOADS:
        built = circuit(bench(made, problem, held(problem, levels), load=load))
        parts = designed(built)
        solution = Simulation(built).run(recorded(parts, "RLOAD"))
        currents.append(solution.current("RLOAD", "a"))
        if load == middle:
            found += misbiased(solution, parts)
    step = float(np.max(np.abs(np.diff(currents))))
    if step >= STEADY:
        found.append(f"its current changes by {step:.4g} A from one load to the next")
    if min(currents) <= MIN_MAGNITUDE:
        found.append(
            f"it carries {min(currents):.4g} A, not more than {MIN_MAGNITUDE:g}"
        )
    if "iref" in problem.inputs:
        drives = {"iref": driven("iref", REFERENCE)}
        built = circuit(bench(made, problem, drives, load=middle))
        followed = (
            Simulation(built)
            .run(OperatingPoint(record=["RLOAD"]))
            .current("RLOAD", "a")
        )
        if abs(followed - currents[len(LOADS) // 2]) < STEADY:
            found.append(f"its current stays {followed:.4g} A as iref turns 1.55 mA")
    return found


def opamp(
    number: int, made: Component, problem: CircuitTask, levels: Levels
) -> list[str]:
    """Runs operating-point and 100 Hz differential and common-mode gain checks.

    Derives common-mode and differential gains by combining the individual AC responses
    to ``VINP`` and ``VINN``.
    """
    found: list[str] = []
    if levels["vinp"] != levels["vinn"]:
        found.append("LEVELS holds vinp and vinn at different levels")
    built = circuit(bench(made, problem, held(problem, levels)))
    simulation = Simulation(built)
    parts = designed(built)
    found += structure(number, parts)
    found += misbiased(simulation.run(recorded(parts)), parts, MIN_MAGNITUDE)
    plus, minus = (
        complex(simulation.run(Ac(source, [100 * Hz])).voltage("vout")[0])
        for source in ("VINP", "VINN")
    )
    differential, common = abs(plus - minus), abs(plus + minus)
    if differential <= GAIN:
        found.append(f"its differential gain at 100 Hz is {differential:.3g}")
    if common >= differential - GAIN:
        found.append(
            f"its common-mode gain at 100 Hz, {common:.3g}, is not less than its "
            f"differential gain, {differential:.3g}"
        )
    return found


def rises(times: Real, values: Real, middle: float, band: float) -> Real:
    """Detects interpolated rising-crossing times through `middle` with hysteresis.

    Requires `values` to fall `band` below `middle` before arming and rise `band` above
    `middle` to confirm each crossing, rejecting noise near the threshold.
    """
    found: list[float] = []
    low = False
    crossed: float | None = None
    for index in range(1, len(values)):
        before, after = values[index - 1], values[index]
        if after < middle - band:
            low, crossed = True, None
        elif low and before < middle <= after:
            share = (middle - before) / (after - before)
            crossed = float(
                times[index - 1] + share * (times[index] - times[index - 1])
            )
        if low and crossed is not None and after > middle + band:
            found.append(crossed)
            low, crossed = False, None
    return np.asarray(found, dtype=np.float64)


def oscillation(times: Real, values: Real) -> float | str:
    """Computes the mean period of steady oscillation, or returns a failure reason."""
    swing = float(np.ptp(values))
    if swing < QUIET:
        return f"it swings {swing:.3g} V"
    middle = (float(np.max(values)) + float(np.min(values))) / 2
    found = rises(times, values, middle, BAND * swing)
    if len(found) <= PERIODS:
        return f"it rises through the middle of its swing {len(found)} times"
    periods = np.diff(found)
    period = float(np.mean(periods))
    spread = float(np.std(periods)) / period
    if spread >= SPREAD:
        return f"its period varies by {spread:.0%}"
    last = values[(times >= found[-2]) & (times <= found[-1])]
    if np.ptp(last) < DECAY * swing:
        return f"it decays: its last period swings {np.ptp(last):.3g} V of {swing:.3g}"
    return period


def transient(
    made: Component,
    problem: CircuitTask,
    drives: Mapping[str, devices.Device],
    analysis: Transient,
) -> tuple[Real, dict[str, Real]]:
    """Simulates `made` under `drives` with `analysis` and returns times and outputs."""
    waveforms = Simulation(circuit(bench(made, problem, drives))).run(analysis)
    return waveforms.times, {port: waveforms.voltage(port) for port in problem.outputs}


def powered(**drives: devices.Device) -> dict[str, devices.Device]:
    """Prepends the switched-on `SWITCHED_ON` supply source to `drives`."""
    return {"vdd": devices.voltage_source(SWITCHED_ON), **drives}


def second_half(times: Real, values: Real) -> tuple[Real, Real]:
    """Slices `times` and `values` to the second half of the simulation interval."""
    late = times >= times[-1] / 2
    return times[late], values[late]


def oscillator(_: int, made: Component, problem: CircuitTask, __: Levels) -> list[str]:
    """Checks that an oscillator sustains steady oscillation over a 10 ms transient."""
    times, outputs = transient(
        made, problem, powered(), Transient(stop=10e-3 * s, max_step=1 * us)
    )
    found = oscillation(*second_half(times, outputs["vout"]))
    return [f"vout does not oscillate: {found}"] if isinstance(found, str) else []


def ramp(times: Real, values: Real) -> tuple[float, float]:
    """Fits a line to `values` over `times` and returns the slope and R-squared fit."""
    slope, intercept = np.polyfit(times, values, 1)
    residual = values - (slope * times + intercept)
    spread = float(np.sum((values - np.mean(values)) ** 2))
    fit = 1 - float(np.sum(residual**2)) / spread if spread else 0.0
    return float(slope), fit


def integrator(
    _: int, made: Component, problem: CircuitTask, levels: Levels
) -> list[str]:
    """Checks integrator step-response ramp slopes and linearity for 0.5 V inputs.

    Steps ``vin`` +/-0.5 V around DC bias every 10 ms with 1 µs edges. Verifies that
    ``vout`` inverts and ramps near +/-0.5 V / 30 ms within 30 % and R-squared >= 0.9
    over the middle 80 % of the final two half-periods.
    """
    rest = levels["vin"]
    points = [(0.0, rest)]
    for index in range(HALVES):
        start = index * HALF + EDGE
        points += [(start, points[-1][1]), (start + EDGE, rest + STEP * _sense(index))]
    signal = Piecewise(tuple((time * s, level * V) for time, level in points))
    times, outputs = transient(
        made,
        problem,
        {"vin": devices.voltage_source(signal)},
        Transient(stop=HALVES * HALF * s, max_step=100 * us),
    )
    found: list[str] = []
    for index in (HALVES - 2, HALVES - 1):
        start = index * HALF + EDGE
        window = (times >= start + HALF / 10) & (times <= start + HALF * 9 / 10)
        slope, fit = ramp(times[window], outputs["vout"][window])
        expected = -_sense(index) * INTEGRATED
        side = "above" if _sense(index) > 0 else "below"
        if abs(slope - expected) > SLOPE_TOLERANCE * INTEGRATED:
            found.append(
                f"vout ramps at {slope:.4g} V/s while vin is {side} its level, "
                f"not {expected:.4g} V/s"
            )
        if fit < FIT:
            found.append(f"vout's ramp while vin is {side} its level fits R² {fit:.3f}")
    return found


def _sense(index: int) -> float:
    """Returns +1.0 for even integrator half-period `index` and -1.0 for odd."""
    return 1.0 if index % 2 == 0 else -1.0


def differentiator(
    _: int, made: Component, problem: CircuitTask, levels: Levels
) -> list[str]:
    """Checks differentiator square-wave levels and flatness for a triangle input.

    Ramps ``vin`` +/-0.5 V around DC bias over 50 ms rise and fall segments. Verifies
    that ``vout`` sits +/-0.6 V inverted from DC bias within 20 % and stays flat within
    10 % RMS over the final 90 % of each second-period ramp.
    """
    rest = levels["vin"]
    triangle = Pulse(
        initial=(rest - STEP) * V,
        pulsed=(rest + STEP) * V,
        rise=RAMP * s,
        fall=RAMP * s,
        width=0 * s,
        period=2 * RAMP * s,
    )
    times, outputs = transient(
        made,
        problem,
        {"vin": devices.voltage_source(triangle)},
        Transient(stop=4 * RAMP * s, max_step=100 * us),
    )
    found: list[str] = []
    for index, sense, phase in ((2, -1.0, "rises"), (3, 1.0, "falls")):
        start = index * RAMP
        window = (times >= start + RAMP / 10) & (times <= start + RAMP)
        level = outputs["vout"][window] - rest
        mean, spread = float(np.mean(level)), float(np.std(level))
        expected = sense * DIFFERENTIATED
        if abs(mean - expected) > TOLERANCE * DIFFERENTIATED:
            found.append(
                f"vout sits {mean:+.4g} V from vin's level while vin {phase}, "
                f"not {expected:+.4g} V"
            )
        elif spread > FLAT * DIFFERENTIATED:
            found.append(f"vout varies by {spread:.3g} V rms while vin {phase}")
    return found


def adder(_: int, made: Component, problem: CircuitTask, levels: Levels) -> list[str]:
    """Checks that an inverting adder tracks the negated sum of ``vin1`` and ``vin2``.

    Holds ``vin2`` 0.125 V above the ``vin1`` DC bias and sweeps ``vin1`` up by 0.5 V
    in 10 mV steps.
    """
    rest = levels["vin1"]
    drives = {"vin1": driven("vin1", rest), "vin2": driven("vin2", rest + ADDEND)}
    swept = Simulation(circuit(bench(made, problem, drives))).run(
        DcSweep("VIN1", linear(rest * V, (rest + STEP) * V, 51))
    )
    expected = rest - (swept.swept - rest) - ADDEND
    return _tracked(swept.swept, swept.voltage("vout"), expected, rest, "vin1")


def subtractor(
    _: int, made: Component, problem: CircuitTask, levels: Levels
) -> list[str]:
    """Checks that a difference amplifier outputs ``vin2 - vin1`` across a DC sweep.

    Holds ``vin2`` at DC bias and sweeps ``vin1`` from 2.25 V below to 1.75 V below in
    50 mV steps.
    """
    rest = levels["vin2"]
    drives = {"vin1": driven("vin1", rest), "vin2": driven("vin2", rest)}
    low, high = DIFFERENCES
    swept = Simulation(circuit(bench(made, problem, drives))).run(
        DcSweep("VIN1", linear((rest - low) * V, (rest - high) * V, 11))
    )
    expected = rest - swept.swept
    return _tracked(swept.swept, swept.voltage("vout"), expected, 0.0, "vin1")


def _tracked(
    inputs: Real, output: Real, expected: Real, origin: float, swept: str
) -> list[str]:
    """Checks whether `output` stays within 20 % of `expected` relative to `origin`."""
    error = np.abs(output - expected)
    allowed = TOLERANCE * np.abs(expected - origin)
    worst = int(np.argmax(error - allowed))
    if error[worst] <= allowed[worst]:
        return []
    return [
        (
            f"vout is {output[worst]:.4g} V with {swept} at {inputs[worst]:.4g} V, "
            f"not {expected[worst]:.4g} V within {allowed[worst]:.3g} V"
        )
    ]


def schmitt(_: int, made: Component, problem: CircuitTask, __: Levels) -> list[str]:
    """Checks rising and falling DC sweeps of a non-inverting Schmitt trigger.

    Sweeps ``vin`` from 0 V to 5 V and back in 100 mV steps with continuation.
    Requires monotonic output rise of at least 2.5 V in each direction and threshold
    separation above 50 mV.
    """
    found: list[str] = []
    thresholds = []
    for start, stop, direction in ((0.0, SUPPLY, "rises"), (SUPPLY, 0.0, "falls")):
        drives = {"vin": driven("vin", start)}
        swept = Simulation(circuit(bench(made, problem, drives))).run(
            DcSweep("VIN", linear(start * V, stop * V, 51))
        )
        order = np.argsort(swept.swept)
        inputs, output = swept.swept[order], swept.voltage("vout")[order]
        high = np.flatnonzero(output > SUPPLY / 2)
        if not high.size:
            return [f"vout stays below 2.5 V as vin {direction}"]
        thresholds.append(float(inputs[high[0]]))
        if output[-1] - output[0] < SUPPLY / 2:
            found.append(
                f"vout rises {output[-1] - output[0]:.4g} V as vin {direction}"
            )
        if np.min(np.diff(output)) < -NOISE:
            found.append(f"vout falls somewhere as vin {direction}")
    rising, falling = thresholds
    if abs(rising - falling) <= HYSTERESIS:
        found.append(
            f"it switches at {rising:.4g} V rising and {falling:.4g} V falling"
        )
    return found


def vco(_: int, made: Component, problem: CircuitTask, __: Levels) -> list[str]:
    """Checks voltage-controlled oscillator tuning monotonicity across control levels.

    Simulates 100 µs from supply turn-on at each level in `CONTROLS`, verifies steady
    second-half oscillation, and requires monotonic period shifts exceeding 1 % per
    step.
    """
    periods = []
    for control in CONTROLS:
        times, outputs = transient(
            made,
            problem,
            powered(vin=driven("vin", control)),
            Transient(stop=100 * us, max_step=10 * ns),
        )
        found = oscillation(*second_half(times, outputs["vout"]))
        if isinstance(found, str):
            return [f"vout does not oscillate with vin at {control:g} V: {found}"]
        periods.append(found)
    ratios = np.array(periods[1:]) / np.array(periods[:-1])
    if np.all(ratios > 1 + TUNED) or np.all(ratios < 1 - TUNED):
        return []
    listed = ", ".join(f"{period:.4g} s" for period in periods)
    return [f"its periods, {listed} at vin = 3.25, 3.5 and 4 V, are not monotonic"]


def pll(_: int, made: Component, problem: CircuitTask, __: Levels) -> list[str]:
    """Checks phase-locked loop frequency locking at 10 MHz and 12 MHz references.

    Drives ``clk_ref`` with a 0 V to 5 V clock (2 % edges, 48 % pulse width, 30 ns
    delay) over 10 µs from supply turn-on. Verifies that the mean ``clk_p`` period over
    the final 2 µs matches the reference period within 5 %.
    """
    found: list[str] = []
    for frequency in CLOCKS:
        period = 1 / frequency
        clock = Pulse(
            initial=0 * V,
            pulsed=SUPPLY * V,
            rise=CLOCK_EDGE * period * s,
            fall=CLOCK_EDGE * period * s,
            width=CLOCK_WIDTH * period * s,
            delay=30 * ns,
            period=period * s,
        )
        times, outputs = transient(
            made,
            problem,
            powered(clk_ref=devices.voltage_source(clock)),
            Transient(stop=LOCK_RUN * s, max_step=period / 100 * s),
        )
        late = times >= LOCK_RUN - LOCK_WINDOW
        edges = rises(times[late], outputs["clk_p"][late], SUPPLY / 2, CLOCK_BAND)
        megahertz = f"{frequency / 1e6:g} MHz"
        if len(edges) < 2:  # noqa: PLR2004 - a period lies between two rises
            found.append(f"clk_p does not toggle with clk_ref at {megahertz}")
            continue
        measured = float(np.mean(np.diff(edges)))
        if abs(measured - period) > LOCKED * period:
            found.append(
                f"with clk_ref at {megahertz}, clk_p's period is {measured:.4g} s"
            )
    return found


type Check = Callable[[int, Component, CircuitTask, Levels], list[str]]
CHECKS: dict[Kind, Check] = {
    "amplifier": amplifier,
    "inverter": inverter,
    "mirror": mirror,
    "opamp": opamp,
    "oscillator": oscillator,
    "integrator": integrator,
    "differentiator": differentiator,
    "adder": adder,
    "subtractor": subtractor,
    "schmitt": schmitt,
    "vco": vco,
    "pll": pll,
}
"""Mapping from problem kind to simulation verification function."""


def needed(problem: CircuitTask) -> tuple[str, ...]:
    """Returns the input port names of `problem` that must appear in declared levels."""
    match problem.kind:
        case "amplifier" | "mirror" | "opamp":
            return problem.inputs
        case "integrator" | "differentiator":
            return ("vin",)
        case "adder":
            return ("vin1",)
        case "subtractor":
            return ("vin2",)
        case _:
            return ()


LEVELS = {Arm.JOLTY: "LEVELS", Arm.NGSPICE: "levels.json"}
"""Filename or variable name declaring input bias levels for each arm."""


def loaded(
    task: Path, work: Path, number: int, arm: Arm, declared: Mapping[str, object]
) -> tuple[Component, dict[str, float]] | Shortfall:
    """Loads design ``p{number}`` and its input bias levels from `work` for `arm`.

    Reads ``LEVELS`` from ``p{number}.py`` on the jolty arm or looks up ``p{number}``
    inside `declared` from ``levels.json`` on the ngspice arm.
    """
    stem = f"p{number}"
    if arm is Arm.NGSPICE:
        made = netlist(task, work, stem)
        levels = declared.get(stem, {})
    else:
        namespace = script(work, stem)
        if isinstance(namespace, Shortfall):
            return namespace
        made = built(namespace, stem)
        levels = namespace.get("LEVELS", {})
    if isinstance(made, Shortfall):
        return made
    if not isinstance(levels, Mapping):
        return unbuilt(f"{LEVELS[arm]} gives {stem} no levels by input")
    try:
        return made, {str(port): float(level) for port, level in levels.items()}
    except (TypeError, ValueError) as error:
        return unbuilt(f"{LEVELS[arm]} gives {stem} a level that is no number: {error}")


def declarations(work: Path, arm: Arm) -> Mapping[str, object] | Shortfall:
    """Loads ``levels.json`` from `work` for the ngspice arm, returning a dict or error.

    Returns an empty mapping on the jolty arm (where scripts declare ``LEVELS``) or
    when ``levels.json`` is absent.
    """
    path = work / LEVELS[Arm.NGSPICE]
    if arm is Arm.JOLTY or not path.is_file():
        return {}
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return unbuilt(f"{path.name} is no JSON: {error}")
    if not isinstance(found, dict):
        return unbuilt(f"{path.name} holds no levels by circuit")
    return dict[str, object](found)


def judge(
    found: tuple[Component, dict[str, float]] | Shortfall,
    number: int,
    given: Mapping[object, Mapping[str, object]],
    arm: Arm,
) -> list[Shortfall]:
    """Evaluates a loaded problem `number` design and returns any `Shortfall` entries.

    Records disallowed devices under spec ``parts`` and simulation check failures
    under the problem kind such as ``amplifier``.
    """
    if isinstance(found, Shortfall):
        return [found]
    made, levels = found
    problem = PROBLEMS[number]
    ports = sorted(port.name for port in made.definition.ports)
    expected = sorted({"vdd", "ground", *problem.inputs, *problem.outputs})
    if ports != expected:
        return [unbuilt(f"its ports are {', '.join(ports)}, not {', '.join(expected)}")]
    unleveled = [port for port in needed(problem) if port not in levels]
    if unleveled:
        return [missing(f"{LEVELS[arm]} gives no level for {', '.join(unleveled)}")]
    try:
        forbidden = [unmet("parts", reason) for reason in disallowed(made, given)]
        checked = forbidden or [
            unmet(problem.kind, reason)
            for reason in CHECKS[problem.kind](number, made, problem, levels)
        ]
    except JoltyError as error:
        return [failed(error)]
    return checked


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores AnalogCoder designs ``p1`` through ``p24`` in `work` for `arm`."""
    given = transistors(task, arm)
    declared = declarations(work, arm)
    if isinstance(declared, Shortfall):
        return Verdict(met=False, shortfalls=(declared,))
    failures = {
        number: judge(loaded(task, work, number, arm, declared), number, given, arm)
        for number in PROBLEMS
    }
    measures = {"solved": float(sum(not found for found in failures.values()))}
    measures |= {f"p{number}": float(not found) for number, found in failures.items()}
    shortfalls = tuple(
        shortfall.model_copy(update={"design": f"p{number}"})
        for number, found in failures.items()
        for shortfall in found
    )
    return Verdict(met=not shortfalls, measures=measures, shortfalls=shortfalls)
