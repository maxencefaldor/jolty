# Copyright 2026 Maxence Faldor
"""The voltage-current converter task's scorer, which the agent never sees.

It places the agent's component on Koza's test fixture for the voltage-current converter
of US patent 6,166,529 (Genetic Programming IV, sections 15.3.1.4 and 15.3.5.4) and runs
his four fitness cases: one second each, the output current read at 31 times, 1/30 s
apart. Each case's mean absolute error against 1 mA/V times the inputs' difference is
divided by the patented circuit's on that case, as Koza measured it: 41, 36, 26 and 44
µA, which he gives in microvolts, as many across the 1 Ω load. The fitness is the four
ratios' mean, as Koza weighted it, so the patent's is 1, and the circuit genetic
programming evolved reached 0.619. The work meets the task when it builds from the
allowed parts alone, its fitness is at most 1, and its output current settles within 40
µs after ``VS`` steps by 10 mV at three points of its range, to within 44 µA, the
patent's largest mean error, of 1 mA/V times the inputs' difference: a check Koza's
fitness lacks, since its steps of 1/30 s cannot see an oscillation, and its 31 readings
may miss the output's extremes. All of it must hold from 0 °C to
70 °C with the supply within 5 % of -15 V, and on 95 % of boards of the design too,
built of resistors within 1 %, capacitors within 5 % and transistors from reels anywhere
in their datasheets' ranges, each board over the same range: margins Koza's circuits,
simulated with exact parts at 27 °C, need not keep.

Koza's added input is ``plus`` and his subtracted one ``minus``; his evolved circuit
sources a current from the first and sinks one into the second, so the output current
flows out of ``output`` when ``plus`` is the higher.
"""

import runpy
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scoring import Board, cornered, count_devices, disallowed, discrete, tolerate
from verdict import Verdict

from jolty import Builder, Component, Role, circuit, component, devices
from jolty.plan import Options, Transient
from jolty.units import Hz, V, ns, ohm, s, us
from jolty.waveforms import Piecewise, Pulse, Sine

TRANSCONDUCTANCE = 1e-3
"""The output current per volt of the inputs' difference, in amperes per volt."""
TIMES = 31
"""The times each case is read at, 1/30 s apart from 0 s to 1 s."""
PATENT = (41e-6, 36e-6, 26e-6, 44e-6)
"""The patented circuit's mean absolute error in each case, in amperes, per Koza."""
EVOLVED = 0.619
"""The fitness of the circuit genetic programming evolved, as Koza found."""
ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistors."""
SUPPLIES = ("VEE",)
"""The bench's supply, which strays within 5 % of its level in the corners."""
STEP = 10e-3
"""The step of ``VS`` the output current must settle after, in volts."""
SETTLING = ((6.0, 4.0, 0.0), (10.0, 2.0, 1.0), (2.0, 10.0, -1.0))
"""Where it must settle: ``plus``, ``minus`` and ``VS`` before the step, in volts."""
UNSETTLED = 1e-6
"""The swing of the output current, in amperes, beyond which it has not settled."""
SETTLED = 40e-6
"""The time from which the output current must have settled, in seconds."""
LOAD = Options(record=["RLOAD"])
"""Options that record the load, whose current the output is."""
SETTLED_ERROR = max(PATENT)
"""How far, in amperes, the settled output current may lie from 1 mA/V times the inputs'
difference: the patent's largest mean error over a case, so that a design no worse than
the patent on average is no worse at these points either."""


def ramp(start: float, end: float) -> devices.Device:
    """A source that moves linearly from `start` to `end` volts over one second."""
    return devices.voltage_source(Piecewise([(0 * s, start * V), (1 * s, end * V)]))


def sine(middle: float, swing: float, frequency: float) -> devices.Device:
    """A source that swings `swing` volts about `middle` at `frequency` hertz."""
    return devices.voltage_source(
        Sine(offset=middle * V, amplitude=swing * V, frequency=frequency * Hz)
    )


def cases() -> list[tuple[devices.Device, devices.Device]]:
    """Koza's four fitness cases: the ``plus`` source, then the ``minus`` source."""
    return [
        (ramp(10, 2), ramp(2, 10)),
        (sine(5.5, 3.5, 4.5), sine(6, 3.5, 6.4)),
        (sine(6, 4, 2), sine(6, 4, 2)),
        (sine(4, 1, 4.5), sine(4, 1, 6.4)),
    ]


def swing() -> devices.Device:
    """Koza's ``VS``: 1 V about 0 V at 3.2 Hz."""
    return devices.voltage_source(
        Sine(offset=0 * V, amplitude=1 * V, frequency=3.2 * Hz)
    )


def step(level: float) -> devices.Device:
    """A ``VS`` that steps from `level` to 10 mV above it at 1 µs, over 1 ns."""
    return devices.voltage_source(
        Pulse(
            initial=level * V,
            pulsed=(level + STEP) * V,
            delay=1 * us,
            rise=1 * ns,
            fall=1 * ns,
            width=1 * s,
        )
    )


@component
def bench(
    c: Builder,
    made: Component,
    inputs: tuple[devices.Device, devices.Device],
    output: devices.Device,
) -> None:
    """`made` on Koza's fixture, driven by `inputs` and `output`.

    `inputs` drive ``plus`` and ``minus``, and `output` is ``VS``.
    """
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, id="X1")
    supply = c.place(devices.voltage_source(-15 * V), id="VEE")
    c.connect(supply.positive, placed.vee)
    c.connect(supply.negative, placed.ground, ground)
    for name, source in zip(("plus", "minus"), inputs, strict=True):
        driven = c.place(source, id=f"V{name.upper()}")
        series = c.place(devices.resistor(1e-6 * ohm), id=f"R{name.upper()}")
        c.connect(driven.positive, series.a, name=name)
        c.connect(series.b, placed[name])
        c.connect(driven.negative, ground)
    held = c.place(output, id="VS")
    load = c.place(devices.resistor(1 * ohm), id="RLOAD")
    c.connect(placed.output, held.positive, name="output")
    c.connect(held.negative, load.a, name="load")
    c.connect(load.b, ground)


def measure(made: Component, board: Board) -> dict[str, float]:
    """Koza's four fitness cases on `board` of `made`, and its three settling steps.

    Each case gives its mean absolute error of the output current, in amperes, and the
    fitness weighs them. The fitness cases' steps are milliseconds long, too long to see
    an oscillation; each settling step gives the swing of the output current over the
    last 10 µs of 50, read at steps of 10 ns, and its mean's error against 1 mA/V
    times the inputs' difference.
    """
    times = np.linspace(0.0, 1.0, TIMES)
    errors = []
    for inputs in cases():
        waveforms = board.simulation(
            circuit(bench(made, inputs, swing())), SUPPLIES
        ).run(
            Transient(
                stop=1 * s,
                max_step=(1 / (10 * (TIMES - 1))) * s,
                times=[time * s for time in times[1:]],
            ),
            LOAD,
        )

        def read(values: np.ndarray, at: np.ndarray = waveforms.times) -> np.ndarray:
            return np.interp(times, at, values)

        output = read(waveforms.current("RLOAD", "a"))
        wanted = TRANSCONDUCTANCE * (
            read(waveforms.voltage("plus")) - read(waveforms.voltage("minus"))
        )
        errors.append(float(np.mean(np.abs(output - wanted))))
    measurements = {"fitness": float(np.mean(np.divide(errors, PATENT)))}
    measurements |= {
        f"error {case}": error for case, error in enumerate(errors, start=1)
    }
    for case, (plus, minus, level) in enumerate(SETTLING, start=1):
        inputs = (devices.voltage_source(plus * V), devices.voltage_source(minus * V))
        waveforms = board.simulation(
            circuit(bench(made, inputs, step(level))), SUPPLIES
        ).run(Transient(stop=50 * us, max_step=10 * ns), LOAD)
        late = waveforms.times >= SETTLED
        current = waveforms.current("RLOAD", "a")[late]
        measurements[f"swing {case}"] = float(np.ptp(current))
        wanted = TRANSCONDUCTANCE * (plus - minus)
        measurements[f"settled error {case}"] = abs(float(np.mean(current)) - wanted)
    return measurements


def shortfalls(found: dict[str, float]) -> list[str]:
    """Why the measurements `found` fall short of the task, if they do."""
    reasons = []
    if found["fitness"] > 1:
        reasons.append(
            f"the fitness is {found['fitness']:.4g}, more than the patent's 1"
        )
    for case, (plus, minus, level) in enumerate(SETTLING, start=1):
        swing = found[f"swing {case}"]
        if swing > UNSETTLED:
            reasons.append(
                f"with plus at {plus} V, minus at {minus} V and VS stepped from "
                f"{level} V, the output current still swings by {swing:.3g} A"
            )
        error = found[f"settled error {case}"]
        if error > SETTLED_ERROR:
            reasons.append(
                f"with plus at {plus} V, minus at {minus} V and VS stepped from "
                f"{level} V, the output current settles {error:.3g} A from its "
                f"{TRANSCONDUCTANCE * (plus - minus):.3g} A, more than the patent's "
                f"largest mean error, {SETTLED_ERROR:.3g} A"
            )
    return reasons


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the voltage-current converter in `work`."""
    parts = runpy.run_path(str(task / "given" / "parts.py"))
    given = [parts["npn"](), parts["pnp"]()]
    design = work / "design.py"
    if not design.is_file():
        return Verdict(met=False, reasons=(f"there is no {design}",))
    sys.path.insert(0, str(work))
    made = runpy.run_path(str(design))["build"]()
    if not isinstance(made, Component):
        return Verdict(met=False, reasons=("build() returns no component",))
    built = circuit(bench(made, cases()[0], swing()))
    reasons = disallowed(built, given, ALLOWED)
    measured = partial(measure, made)
    measurements, short = cornered(measured, shortfalls)
    reasons += short
    if not reasons:
        spread, reasons = tolerate(built, measured, shortfalls, discrete)
        measurements |= spread
    measurements |= {
        f"patent's error {case}": theirs for case, theirs in enumerate(PATENT, start=1)
    }
    measurements |= {
        "evolved fitness": EVOLVED,
        "devices": float(count_devices(built, "X1")),
    }
    return Verdict(met=not reasons, measurements=measurements, reasons=tuple(reasons))
