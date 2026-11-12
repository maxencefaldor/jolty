# Copyright 2026 Maxence Faldor
"""AnalogCoder's twenty-four problems, checked as AnalogCoder checks them.

AnalogCoder (Lai et al., AAAI 2025; github.com/laiyao1/AnalogCoder, MIT License) asks
for twenty-four circuits built from the same two level 1 MOSFETs on a 5 V supply, and
checks each by simulation: with its ``problem_check`` scripts, and for the first fifteen
problems with the operating point its ``check_netlist`` inspects. This scorer runs the
same checks, with the same stimuli and tolerances, on a bench it builds around each
design, and the work meets the task when every problem passes.

Where AnalogCoder's checks found the parts they changed by their SPICE names, or let
through what their problems exclude, this scorer checks what they meant, and the prompt
says so:

- A design is a component with named ports, its inputs held at the levels it declares,
  and its R1 and Cf, or Rf and C1, have the values AnalogCoder's checks gave them.
- An integrator and a differentiator must invert, which a passive RC network cannot;
  AnalogCoder removed the op-amp instead, and failed the design if it still worked.
- The adder's tolerance is 20% of the sum, where AnalogCoder's was 20% of the output
  voltage, which passes an adder that drops one of its inputs.
- A current source keeps every step of its current within 1 µA as its load changes,
  where AnalogCoder's check needed only one step to.
- An oscillator starts from its supply switched on, not from initial conditions, and
  must oscillate steadily over the second half of the run, not merely ring.
- The voltage-controlled oscillator is tested at 3.25, 3.5 and 4 V, AnalogCoder's 0.65,
  0.7 and 0.8 V of a 1 V supply scaled to 5 V, and its frequency must change
  monotonically; AnalogCoder's check wanted the middle level's period the extreme one.
- The phase-locked loop runs from 5 V, and must lock to 12 MHz as well as to 10 MHz,
  which a free-running oscillator cannot.
- Only the two transistors, resistors, capacitors and constant sources may appear.
"""

import runpy
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from verdict import Verdict

from jolty import Builder, Component, Role, Simulation, circuit, component, devices, ir
from jolty.errors import JoltyError
from jolty.plan import Ac, DcSweep, OperatingPoint, Transient, linear
from jolty.results import OperatingPoint as Solution
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
class Problem:
    """What one of AnalogCoder's problems is, and its ports besides the supply's."""

    kind: Kind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...] = ("vout",)


PROBLEMS = {
    1: Problem("amplifier", ("vin",)),
    2: Problem("amplifier", ("vin",)),
    3: Problem("amplifier", ("vin",)),
    4: Problem("amplifier", ("vin", "vbias")),
    5: Problem("amplifier", ("vin", "vbias")),
    6: Problem("inverter", ("vin",)),
    7: Problem("inverter", ("vin",)),
    8: Problem("mirror", ("vbias",)),
    9: Problem("amplifier", ("vin",)),
    10: Problem("amplifier", ("vin",)),
    11: Problem("opamp", ("vinp", "vinn", "vbias"), ("voutp", "vout")),
    12: Problem("mirror", ("iref",), ("iout",)),
    13: Problem("opamp", ("vinp", "vinn")),
    14: Problem(
        "opamp", ("vinp", "vinn", "vbias1", "vbias2", "vbias3"), ("voutp", "vout")
    ),
    15: Problem(
        "opamp",
        ("vinp", "vinn", "vbias1", "vbias2", "vbias3", "vbias4"),
        ("voutp", "vout"),
    ),
    16: Problem("oscillator", ()),
    17: Problem("oscillator", ()),
    18: Problem("integrator", ("vin",)),
    19: Problem("differentiator", ("vin",)),
    20: Problem("adder", ("vin1", "vin2")),
    21: Problem("subtractor", ("vin1", "vin2")),
    22: Problem("schmitt", ("vin",)),
    23: Problem("vco", ("vin",)),
    24: Problem("pll", ("clk_ref",), ("clk_p",)),
}
"""Each problem by AnalogCoder's number; ``task.md`` describes them."""

SUPPLY = 5.0
"""The supply, in volts."""
SWITCHED_ON = Piecewise(((0 * s, 0 * V), (1 * us, SUPPLY * V)))
"""The supply of an oscillator, switched on over a microsecond."""
RESOLUTION = 1e-6
"""The resolution, in volts, at which AnalogCoder read an operating point."""
LEAST = 1e-5
"""The least drain current of an amplifier's or an op-amp's MOSFET, in amperes, and the
least current of a current source."""
GAIN = 1e-5
"""The least gain at 100 Hz of an amplifier, and of an op-amp's differential mode."""
MOVES = 1e-3
"""How far, in volts, an amplifier's output must move as its input sweeps the supply."""
SWING = 1.0
"""How far apart, in volts, an inverter's two outputs must be."""
LOADS = (100.0, 300.0, 500.0, 750.0, 1000.0)
"""A current source's loads, in ohms; the middle one's operating point is checked."""
STEADY = 1e-6
"""How much, in amperes, a current source's current may change from one load to the
next, and how much a mirror's must change with its reference."""
REFERENCE = 1.55e-3
"""The reference current, in amperes, a mirror must follow."""
STEP = 0.5
"""How far, in volts, an integrator's, a differentiator's and an adder's input moves
from its level."""
TIME_CONSTANT = 10e3 * 3e-6
"""The integrator's and the differentiator's 10 kΩ times 3 µF, in seconds."""
HALF = 10e-3
"""How long, in seconds, an integrator's input stays above or below its level."""
HALVES = 20
"""How many times an integrator's input steps."""
EDGE = 1e-6
"""How long, in seconds, an integrator's input rests, and each of its steps lasts."""
INTEGRATED = STEP / TIME_CONSTANT
"""The slope, in volts per second, of an integrator's output."""
RAMP = 50e-3
"""How long, in seconds, a differentiator's input ramps up, and then down."""
DIFFERENTIATED = TIME_CONSTANT * 2 * STEP / RAMP
"""The level, in volts, of a differentiator's output from its input's level."""
ADDEND = 0.125
"""How far above vin1's level, in volts, an adder's vin2 stays."""
DIFFERENCES = (2.25, 1.75)
"""The first and last of the differences, in volts, a subtractor's inputs sweep."""
FIT = 0.9
"""The least coefficient of determination of an integrator's ramp."""
TOLERANCE = 0.2
"""The relative tolerance of AnalogCoder's differentiator, adder and subtractor."""
SLOPE_TOLERANCE = 0.3
"""The relative tolerance of AnalogCoder's integrator."""
FLAT = 0.1
"""How much a differentiator's level may vary, relative to the level."""
HYSTERESIS = 0.05
"""The least hysteresis, in volts, of a Schmitt trigger."""
NOISE = 1e-6
"""How far, in volts, a Schmitt trigger's output may fall back as its input rises."""
PERIODS = 3
"""The fewest periods an oscillation runs over the second half of its run."""
SPREAD = 0.2
"""How much an oscillation's period may vary, relative to its mean."""
DECAY = 0.5
"""How small an oscillation's last swing may be, relative to its largest."""
QUIET = 1e-3
"""The least swing, in volts, of an oscillation."""
BAND = 0.1
"""The hysteresis of an oscillation's rises, relative to its swing."""
CONTROLS = (3.25, 3.5, 4.0)
"""The levels, in volts, of a voltage-controlled oscillator's control."""
TUNED = 0.01
"""How much a voltage-controlled oscillator's period must change between levels."""
CLOCKS = (10e6, 12e6)
"""The reference frequencies, in hertz, a phase-locked loop must lock to."""
LOCKED = 0.05
"""How close to the reference's a phase-locked loop's period must be, relatively."""
CLOCK_EDGE = 0.02
"""How long a reference clock's edges last, relative to its period."""
CLOCK_WIDTH = 0.48
"""How long a reference clock's pulse lasts, relative to its period."""
CLOCK_BAND = 0.25
"""How far past mid-supply, in volts, a clock must swing for its rise to count."""
LOCK_RUN = 10e-6
"""How long, in seconds, a phase-locked loop runs."""
LOCK_WINDOW = 2e-6
"""How long, in seconds, at the end of its run, a phase-locked loop's period is read."""
ALLOWED = {"resistor", "capacitor"}
"""The devices, besides the two transistors and constant sources, a design may hold."""
RESISTIVE = frozenset({1, 2, 3, 4, 5, 13})
"""The problems whose design must hold a resistor."""
COMMON_DRAIN = 3
"""The source follower, whose output joins no gate or drain."""
COMMON_GATE = 4
"""The common-gate amplifier, whose input joins no gate or drain."""
MILLER = 9
"""The amplifier with a Miller capacitor, from its first stage's output to its own."""
DIODE_LOAD = 10
"""The amplifier with a diode-connected load."""


@dataclass(frozen=True, slots=True)
class Part:
    """A device of a design: its path, its model, its parameters and its nets."""

    path: str
    model: str
    parameters: Mapping[str, object]
    nets: Mapping[str, str]


def walk(
    built: ir.Circuit, name: str, prefix: str, ports: Mapping[str, str]
) -> Iterator[Part]:
    """The devices of the component `name`, their paths after `prefix`.

    A net is named as the engine names it, by its path, but a port's, which is the net
    `ports` gives.
    """
    for instance in built.components[name].instances:
        nets = {
            terminal: ports.get(net, f"{prefix}{net}")
            for terminal, net in instance.terminals.items()
        }
        match instance.of:
            case ir.ComponentOf(component=inner):
                yield from walk(built, inner, f"{prefix}{instance.id}/", nets)
            case ir.DeviceOf(device=model):
                yield Part(f"{prefix}{instance.id}", model, instance.parameters, nets)


def designed(built: ir.Circuit) -> list[Part]:
    """The devices of the design on the bench `built`."""
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


def transistors(task: Path) -> dict[object, dict[str, object]]:
    """The given transistors' parameters but their size, by channel."""
    parts = runpy.run_path(str(task / "given" / "parts.py"))
    found = {}
    for name in ("nmos", "pmos"):
        made = parts[name](1 * um, 1 * um)
        if not isinstance(made, devices.Device):
            message = f"parts.py's {name}() makes no device"
            raise TypeError(message)
        found[made.parameters["channel"]] = _size_free(made.parameters)
    return found


def disallowed(
    made: Component, given: Mapping[object, Mapping[str, object]]
) -> list[str]:
    """The devices of `made` that are not the given transistors or allowed."""
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
    """The source that holds `port` at `level`: amperes into ``iref``, else volts."""
    if port == "iref":
        return devices.current_source(level * A)
    return devices.voltage_source(level * V)


def held(problem: Problem, levels: Levels) -> dict[str, devices.Device]:
    """The sources that hold each of `problem`'s inputs at its level."""
    return {port: driven(port, levels[port]) for port in problem.inputs}


@component
def bench(
    c: Builder,
    made: Component,
    problem: Problem,
    drives: Mapping[str, devices.Device],
    load: float | None = None,
) -> None:
    """`made` with its inputs and its supply driven, and `load` from supply to output.

    ``vdd`` is 5 V unless `drives` drives it. A voltage source drives an input from the
    ground; a current source, from the supply.
    """
    ground = c.port("ground", Role.GROUND)
    vdd = c.net("vdd")
    power = c.place(drives.get("vdd", devices.voltage_source(SUPPLY * V)), id="VDD")
    placed = c.place(made, id="X1")
    c.connect(power.positive, placed.vdd, vdd)
    c.connect(power.negative, placed.ground, ground)
    for port, source in drives.items():
        if port == "vdd":
            continue
        driver = c.place(source, id=port.upper())
        if source.model == "current_source":
            c.connect(driver.positive, vdd)
            c.connect(driver.negative, placed[port], name=port)
        else:
            c.connect(driver.positive, placed[port], name=port)
            c.connect(driver.negative, ground)
    for port in problem.outputs:
        c.connect(placed[port], name=port)
    if load is not None:
        resistor = c.place(devices.resistor(load * ohm), id="RLOAD")
        c.connect(resistor.a, vdd)
        c.connect(resistor.b, placed[problem.outputs[0]])


def volts(solution: Solution, net: str) -> float:
    """The voltage of `net`, the bench's ground being 0 V."""
    return 0.0 if net == "ground" else solution.voltage(net)


def misbiased(solution: Solution, parts: list[Part], least: float = 0.0) -> list[str]:
    """Why each MOSFET of `parts` is not on, forward and carrying `least`.

    A MOSFET is on when its gate is beyond its threshold and its drain beyond its
    source, in its channel's sense, at the microvolt.
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
    """Why `net` joins a gate or a drain of `parts`."""
    return [
        f"{net} joins {part.path}'s {terminal}"
        for part in parts
        if part.model == "mosfet"
        for terminal in ("gate", "drain")
        if part.nets[terminal] == net
    ]


def structure(number: int, parts: list[Part]) -> list[str]:
    """Why the design of problem `number` is not the circuit AnalogCoder checks for."""
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
    number: int, made: Component, problem: Problem, levels: Levels
) -> list[str]:
    """AnalogCoder's checks of an amplifier: its operating point, DC sweep and gain."""
    built = circuit(bench(made, problem, held(problem, levels)))
    simulation = Simulation(built)
    parts = designed(built)
    found = structure(number, parts)
    found += misbiased(simulation.run(OperatingPoint()), parts, LEAST)
    swept = simulation.run(DcSweep("VIN", linear(0 * V, SUPPLY * V, 501)))
    moved = float(np.ptp(swept.voltage("vout")))
    if moved < MOVES:
        found.append(f"vout moves {moved:.3g} V as vin sweeps the supply")
    gain = float(abs(simulation.run(Ac("VIN", [100 * Hz])).voltage("vout")[0]))
    if gain <= GAIN:
        found.append(f"its gain at 100 Hz is {gain:.3g}")
    return found


def inverter(_: int, made: Component, problem: Problem, __: Levels) -> list[str]:
    """AnalogCoder's check of an inverter: its output with its input at 0 and at 5 V."""
    drives = {"vin": driven("vin", 0.0)}
    swept = Simulation(circuit(bench(made, problem, drives))).run(
        DcSweep("VIN", [0 * V, SUPPLY * V])
    )
    high, low = (float(level) for level in swept.voltage("vout"))
    middle = SUPPLY / 2
    if high < middle or low > middle or high - low < SWING:
        return [f"vout is {high:.4g} V with vin at 0 V and {low:.4g} V at 5 V"]
    return []


def mirror(_: int, made: Component, problem: Problem, levels: Levels) -> list[str]:
    """AnalogCoder's checks of a current source over loads, and of a mirror's input."""
    found: list[str] = []
    currents = []
    middle = LOADS[len(LOADS) // 2]
    for load in LOADS:
        built = circuit(bench(made, problem, held(problem, levels), load=load))
        solution = Simulation(built).run(OperatingPoint())
        currents.append(solution.current("RLOAD", "a"))
        if load == middle:
            found += misbiased(solution, designed(built))
    step = float(np.max(np.abs(np.diff(currents))))
    if step >= STEADY:
        found.append(f"its current changes by {step:.4g} A from one load to the next")
    if min(currents) <= LEAST:
        found.append(f"it carries {min(currents):.4g} A, not more than {LEAST:g}")
    if "iref" in problem.inputs:
        drives = {"iref": driven("iref", REFERENCE)}
        built = circuit(bench(made, problem, drives, load=middle))
        followed = Simulation(built).run(OperatingPoint()).current("RLOAD", "a")
        if abs(followed - currents[len(LOADS) // 2]) < STEADY:
            found.append(f"its current stays {followed:.4g} A as iref turns 1.55 mA")
    return found


def opamp(number: int, made: Component, problem: Problem, levels: Levels) -> list[str]:
    """AnalogCoder's checks of an op-amp: its operating point, and its gains at 100 Hz.

    The common-mode gain is the response to both inputs at once, and the differential
    one to the two in opposition, each found from the responses to each input alone.
    """
    found: list[str] = []
    if levels["vinp"] != levels["vinn"]:
        found.append("LEVELS holds vinp and vinn at different levels")
    built = circuit(bench(made, problem, held(problem, levels)))
    simulation = Simulation(built)
    parts = designed(built)
    found += structure(number, parts)
    found += misbiased(simulation.run(OperatingPoint()), parts, LEAST)
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
    """The times `values` rises through `middle`, each after falling `band` below it.

    A rise counts once `values` passes `band` above `middle`, so noise near the middle
    counts no rise.
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
    """The mean period of the steady oscillation of `values`, or why it has none."""
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
    problem: Problem,
    drives: Mapping[str, devices.Device],
    analysis: Transient,
) -> tuple[Real, dict[str, Real]]:
    """The times of `analysis` of `made`, and each output's voltage at each."""
    waveforms = Simulation(circuit(bench(made, problem, drives))).run(analysis)
    return waveforms.times, {port: waveforms.voltage(port) for port in problem.outputs}


def powered(**drives: devices.Device) -> dict[str, devices.Device]:
    """`drives`, and the supply switched on."""
    return {"vdd": devices.voltage_source(SWITCHED_ON), **drives}


def second_half(times: Real, values: Real) -> tuple[Real, Real]:
    """`times` and `values` over the second half of the run."""
    late = times >= times[-1] / 2
    return times[late], values[late]


def oscillator(_: int, made: Component, problem: Problem, __: Levels) -> list[str]:
    """AnalogCoder's check of an oscillator: a steady oscillation over 10 ms."""
    times, outputs = transient(
        made, problem, powered(), Transient(stop=10e-3 * s, max_step=1 * us)
    )
    found = oscillation(*second_half(times, outputs["vout"]))
    return [f"vout does not oscillate: {found}"] if isinstance(found, str) else []


def ramp(times: Real, values: Real) -> tuple[float, float]:
    """The slope of the line that fits `values` over `times`, and how well it fits."""
    slope, intercept = np.polyfit(times, values, 1)
    residual = values - (slope * times + intercept)
    spread = float(np.sum((values - np.mean(values)) ** 2))
    fit = 1 - float(np.sum(residual**2)) / spread if spread else 0.0
    return float(slope), fit


def integrator(_: int, made: Component, problem: Problem, levels: Levels) -> list[str]:
    """AnalogCoder's check of an integrator: its ramps as its input steps by 0.5 V.

    The input rests at its level until 1 µs, then steps 0.5 V above it, and every 10 ms
    to 0.5 V below and back, each step over 1 µs. Through 10 kΩ into 3 µF the output
    must ramp against it at 0.5 V / 30 ms, within 30%, and straight, over the last two
    half periods but their first and last tenths.
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
    """1 when the integrator's input is above its level at step `index`, else -1."""
    return 1.0 if index % 2 == 0 else -1.0


def differentiator(
    _: int, made: Component, problem: Problem, levels: Levels
) -> list[str]:
    """AnalogCoder's check of a differentiator: a square wave from a triangle.

    The input ramps from 0.5 V below its level to 0.5 V above it over 50 ms and back
    over 50 ms. Through 3 µF into 10 kΩ the output must sit 0.6 V below the input's
    level while it rises, and above it while it falls, within 20% and flat to 10%, over
    the second period but each ramp's first tenth.
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


def adder(_: int, made: Component, problem: Problem, levels: Levels) -> list[str]:
    """AnalogCoder's check of an adder, whose output is the inputs' negated sum.

    The inputs and the output are measured from vin1's level: vin2 is held 0.125 V
    above it, and vin1 swept from it to 0.5 V above it, in 10 mV steps.
    """
    rest = levels["vin1"]
    drives = {"vin1": driven("vin1", rest), "vin2": driven("vin2", rest + ADDEND)}
    swept = Simulation(circuit(bench(made, problem, drives))).run(
        DcSweep("VIN1", linear(rest * V, (rest + STEP) * V, 51))
    )
    expected = rest - (swept.levels - rest) - ADDEND
    return _tracked(swept.levels, swept.voltage("vout"), expected, rest, "vin1")


def subtractor(_: int, made: Component, problem: Problem, levels: Levels) -> list[str]:
    """AnalogCoder's check of a subtractor, whose output is vin2 less vin1.

    vin2 is held at its level, and vin1 swept from 2.25 V below it to 1.75 V below it,
    in 50 mV steps.
    """
    rest = levels["vin2"]
    drives = {"vin1": driven("vin1", rest), "vin2": driven("vin2", rest)}
    low, high = DIFFERENCES
    swept = Simulation(circuit(bench(made, problem, drives))).run(
        DcSweep("VIN1", linear((rest - low) * V, (rest - high) * V, 11))
    )
    expected = rest - swept.levels
    return _tracked(swept.levels, swept.voltage("vout"), expected, 0.0, "vin1")


def _tracked(
    inputs: Real, output: Real, expected: Real, origin: float, swept: str
) -> list[str]:
    """Why `output` misses `expected` by more than 20% of its distance from `origin`."""
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


def schmitt(_: int, made: Component, problem: Problem, __: Levels) -> list[str]:
    """AnalogCoder's check of a non-inverting Schmitt trigger, swept up and down.

    Its input sweeps from 0 to 5 V and back in 100 mV steps, each operating point
    continued from the last. Each way, its output must rise by 2.5 V, never fall, and
    first pass 2.5 V at an input more than 50 mV from where it does the other way.
    """
    found: list[str] = []
    thresholds = []
    for start, stop, direction in ((0.0, SUPPLY, "rises"), (SUPPLY, 0.0, "falls")):
        drives = {"vin": driven("vin", start)}
        swept = Simulation(circuit(bench(made, problem, drives))).run(
            DcSweep("VIN", linear(start * V, stop * V, 51))
        )
        order = np.argsort(swept.levels)
        inputs, output = swept.levels[order], swept.voltage("vout")[order]
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


def vco(_: int, made: Component, problem: Problem, __: Levels) -> list[str]:
    """The check of a voltage-controlled oscillator: its period at three controls.

    At each control level it runs 100 µs from its supply switched on, and must
    oscillate steadily over the second half; its mean periods must rise or fall by more
    than 1% from each level to the next.
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


def pll(_: int, made: Component, problem: Problem, __: Levels) -> list[str]:
    """The check of a phase-locked loop: its output's period at two references.

    clk_ref is a 0 to 5 V clock, its edges 2% of its period and its pulse 48%, from
    30 ns. The loop runs 10 µs from its supply switched on, and over the last 2 µs its
    output's mean period between rises through 2.5 V must be the reference's, within 5%.
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


type Check = Callable[[int, Component, Problem, Levels], list[str]]
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
"""Each kind of problem's check."""


def needed(problem: Problem) -> tuple[str, ...]:
    """The inputs of `problem` whose levels its design's ``LEVELS`` must give."""
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


def loaded(script: Path) -> tuple[Component, dict[str, float]] | str:
    """The design `script` builds and the levels it declares, or why it has none."""
    if not script.is_file():
        return f"there is no {script.name}"
    try:
        namespace = runpy.run_path(str(script))
        made = namespace["build"]()
        levels = {
            str(port): float(level)
            for port, level in dict(namespace.get("LEVELS", {})).items()
        }
    except Exception as error:  # noqa: BLE001 - an agent's script may raise anything
        return f"{script.name} fails: {error!r}"
    if not isinstance(made, Component):
        return "build() returns no component"
    return made, levels


def judge(
    script: Path, number: int, given: Mapping[object, Mapping[str, object]]
) -> list[str]:
    """Why the design in `script` fails problem `number`, if it does."""
    found = loaded(script)
    if isinstance(found, str):
        return [found]
    made, levels = found
    problem = PROBLEMS[number]
    ports = sorted(port.name for port in made.definition.ports)
    expected = sorted({"vdd", "ground", *problem.inputs, *problem.outputs})
    if ports != expected:
        return [f"its ports are {', '.join(ports)}, not {', '.join(expected)}"]
    missing = [port for port in needed(problem) if port not in levels]
    if missing:
        return [f"LEVELS gives no level for {', '.join(missing)}"]
    try:
        forbidden = disallowed(made, given)
        return forbidden or CHECKS[problem.kind](number, made, problem, levels)
    except JoltyError as error:
        return [f"its simulation fails: {error}"]


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the designs ``p1.py`` to ``p24.py`` in `work`."""
    sys.path.insert(0, str(work))
    given = transistors(task)
    failures = {
        number: judge(work / f"p{number}.py", number, given) for number in PROBLEMS
    }
    measurements = {"solved": float(sum(not found for found in failures.values()))}
    measurements |= {
        f"p{number}": float(not found) for number, found in failures.items()
    }
    reasons = tuple(
        f"p{number}: {reason}" for number, found in failures.items() for reason in found
    )
    return Verdict(met=not reasons, measurements=measurements, reasons=reasons)
