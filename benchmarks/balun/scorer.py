# Copyright 2026 Maxence Faldor
"""The balun task's scorer, which the agent never sees.

It places the agent's component on Koza's test fixture for the low-voltage balun of US
patent 6,265,908 (Genetic Programming IV, sections 15.3.1.1 and 15.3.5.1): a 1 V supply,
the source behind 75 Ω, each output loaded with 150 Ω to the supply. It measures his
seven fitness cases: over 41 frequencies from 1 MHz to 100 MHz, each output's gain
against half the source's and its phase against 0° or 180°, and the phase of their
difference against 180°; the DC offset between the outputs; and the difference's total
harmonic distortion, harmonics 2 to 9 of a 10 MHz sine of 10 mV over 200 ns.

Koza's fitness sums these, but his totals, 1.72 for the patented circuit and 0.429 for
the one genetic programming evolved, cannot come from the sums his text states: a gain
error of 0.60 dB, the patent's on the difference, would alone sum to some 25 over 41
frequencies. The task is met instead on the two figures he gives for both circuits: the
difference's gain within 0.60 dB of the source's on average, the patent's, where the
evolved circuit's was 0.47 dB; and a distortion of at most -33.9 dB, the patent's, where
the evolved circuit's was -65.6 dB. To be a balun at all, each output must also lie
within his fitness's tolerances at every frequency, 3 dB and 10°, and the DC offset be
at most 1.72 mV, the most the patent's total of 1.72 leaves room for at his weight of
1,000 per volt. All of it must hold from 0 °C to 70 °C with the supply within 5 % of 1
V, and on 95 % of boards of the design too, built of resistors within 1 %, capacitors
within 5 % and transistors whose current gain spans their datasheets', each board over
the same range: margins Koza's circuits, simulated with exact parts at 27 °C, need not
keep.

Koza does not state the source's DC level; the source reaches the input through 1 µF as
well as 75 Ω, so a design biases its own input rather than drawing its bias through the
source. The distortion's sine has the 10 mV amplitude of his frequency sweeps. Phases
are compared on the circle, so that -179° lies 1° from 180°.
"""

import runpy
import sys
from functools import partial
from pathlib import Path

import numpy as np
from scoring import Board, cornered, count_devices, disallowed, discrete, tolerate
from verdict import Verdict

from jolty import (
    Builder,
    Component,
    Role,
    circuit,
    component,
    devices,
    specs,
)
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.units import F, Hz, MHz, V, mV, ohm, s
from jolty.waveforms import Sine

ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistors."""
SUPPLIES = ("VCC",)
"""The bench's supply, which strays within 5 % of its level in the corners."""
GAIN = 0.60
"""The patented circuit's mean absolute error of the difference's gain, in decibels."""
DISTORTION = -33.9
"""The patented circuit's total harmonic distortion, in decibels."""
EVOLVED = {"difference gain error": 0.47, "distortion": -65.6}
"""What the circuit genetic programming evolved reached on both, as Koza found."""
TOLERANCE = {"gain": 3.0, "phase": 10.0}
"""Koza's tolerances on each output's gain, in decibels, and phase, in degrees."""
OFFSET = 1.72e-3
"""The largest DC offset between the outputs, in volts."""
BLOCKING = 1e-6
"""The capacitance that keeps the source's DC level from the input, in farads."""
FUNDAMENTAL = 10e6
"""The distortion's frequency, in hertz."""
HARMONICS = 9
"""The highest harmonic the distortion counts."""


@component
def bench(c: Builder, made: Component, *, distorting: bool = False) -> None:
    """`made` on Koza's fixture, its source a 10 MHz sine when `distorting`."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, id="X1")
    supply = c.place(devices.voltage_source(1 * V), id="VCC")
    c.connect(supply.positive, placed.vcc, name="vcc")
    c.connect(supply.negative, placed.ground, ground)
    level = (
        Sine(offset=0 * V, amplitude=10 * mV, frequency=FUNDAMENTAL * Hz)
        if distorting
        else 0 * V
    )
    source = c.place(devices.voltage_source(level), id="VIN")
    block = c.place(devices.capacitor(BLOCKING * F), id="CIN")
    series = c.place(devices.resistor(75 * ohm), id="RSRC")
    c.connect(source.positive, block.a, name="source")
    c.connect(block.b, series.a, name="coupled")
    c.connect(series.b, placed.input, name="input")
    c.connect(source.negative, ground)
    for output in ("in_phase", "out_of_phase"):
        load = c.place(devices.resistor(150 * ohm), id=f"RLOAD_{output.upper()}")
        c.connect(placed[output], load.a, name=output)
        c.connect(load.b, supply.positive)


def decibels(gain: np.ndarray) -> np.ndarray:
    """The magnitude of `gain`, in decibels."""
    return 20 * np.log10(np.maximum(np.abs(gain), 1e-30))


def off(gain: np.ndarray, phase: float) -> np.ndarray:
    """How far `gain`'s phase lies from `phase`, in degrees, on the circle."""
    return np.abs((np.angle(gain, deg=True) - phase + 180.0) % 360.0 - 180.0)


def distortion(made: Component, board: Board) -> float:
    """The difference's distortion on `board`, over its second period, in decibels."""
    period = 1 / FUNDAMENTAL
    waveforms = board.simulation(circuit(bench(made, distorting=True)), SUPPLIES).run(
        Transient(stop=2 * period * s, max_step=period / 1000 * s)
    )
    harmonics = specs.distortion(
        "distortion",
        "out_of_phase",
        reference="in_phase",
        fundamental=FUNDAMENTAL * Hz,
        harmonics=HARMONICS,
    )
    return harmonics.read(waveforms).value


def measure(made: Component, board: Board) -> dict[str, float]:
    """Koza's seven fitness cases on `board` of `made`.

    The difference's gain error is a mean over the frequencies, as Koza gives it; each
    other error over the frequencies is its worst.
    """
    simulation = board.simulation(circuit(bench(made)), SUPPLIES)
    response = simulation.run(Ac("VIN", decade(1 * MHz, 100 * MHz, 20)))
    in_phase = response.voltage("in_phase")
    out_of_phase = response.voltage("out_of_phase")
    difference = out_of_phase - in_phase
    half = decibels(np.array(0.5))
    solution = simulation.run(OperatingPoint())
    return {
        "difference gain error": float(np.mean(np.abs(decibels(difference)))),
        "distortion": distortion(made, board),
        "offset": abs(solution.voltage("out_of_phase") - solution.voltage("in_phase")),
        "worst in-phase gain error": float(np.max(np.abs(decibels(in_phase) - half))),
        "worst out-of-phase gain error": float(
            np.max(np.abs(decibels(out_of_phase) - half))
        ),
        "worst in-phase phase error": float(np.max(off(in_phase, 0.0))),
        "worst out-of-phase phase error": float(np.max(off(out_of_phase, 180.0))),
        "worst difference phase error": float(np.max(off(difference, 180.0))),
    }


def shortfalls(found: dict[str, float]) -> list[str]:
    """Why the measurements `found` fall short of the task, if they do."""
    reasons = []
    if found["difference gain error"] > GAIN:
        reasons.append(
            f"the difference's gain is {found['difference gain error']:.4g} dB off "
            f"on average, more than {GAIN} dB"
        )
    if found["distortion"] > DISTORTION:
        reasons.append(
            f"the distortion is {found['distortion']:.4g} dB, above {DISTORTION} dB"
        )
    for output in ("in-phase", "out-of-phase"):
        error = found[f"worst {output} gain error"]
        if error > TOLERANCE["gain"]:
            reasons.append(f"the {output} gain is {error:.4g} dB off at worst")
    for signal in ("in-phase", "out-of-phase", "difference"):
        error = found[f"worst {signal} phase error"]
        if error > TOLERANCE["phase"]:
            reasons.append(f"the {signal} phase is {error:.4g}° off at worst")
    if found["offset"] > OFFSET:
        reasons.append(f"the outputs' DC levels differ by {found['offset']:.4g} V")
    return reasons


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the balun in `work`."""
    parts = runpy.run_path(str(task / "given" / "parts.py"))
    design = work / "design.py"
    if not design.is_file():
        return Verdict(met=False, reasons=(f"there is no {design}",))
    sys.path.insert(0, str(work))
    made = runpy.run_path(str(design))["build"]()
    if not isinstance(made, Component):
        return Verdict(met=False, reasons=("build() returns no component",))
    built = circuit(bench(made))
    reasons = disallowed(built, [parts["npn"](), parts["pnp"]()], ALLOWED)
    measured = partial(measure, made)
    measurements, short = cornered(measured, shortfalls)
    reasons += short
    if not reasons:
        spread, reasons = tolerate(built, measured, shortfalls, discrete)
        measurements |= spread
    measurements |= {f"evolved {name}": value for name, value in EVOLVED.items()}
    measurements["devices"] = float(count_devices(built, "X1"))
    return Verdict(met=not reasons, measurements=measurements, reasons=tuple(reasons))
