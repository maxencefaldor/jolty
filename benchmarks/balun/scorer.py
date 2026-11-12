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
frequencies. The task is met
instead on the two figures he gives for both circuits: the difference's gain within
0.60 dB of the source's on average, the patent's, where the evolved circuit's was 0.47
dB; and a distortion of at most -33.9 dB, the patent's, where the evolved circuit's was
-65.6 dB. To be a balun at all, each output must also lie within his fitness's
tolerances on average, 3 dB and 10°, and the DC offset be at most 1.72 mV, the most the
patent's total of 1.72 leaves room for at his weight of 1,000 per volt.

The source's DC level, which Koza does not state, is 0 V, and the distortion's sine has
the 10 mV amplitude of his frequency sweeps. Phases are compared on the circle, so that
-179° lies 1° from 180°.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
from scoring import count_devices, disallowed
from verdict import Verdict

from jolty import Builder, Component, Role, Simulation, circuit, component, devices
from jolty.plan import Ac, OperatingPoint, Transient, decade
from jolty.units import Hz, MHz, V, mV, ohm, s
from jolty.waveforms import Sine

ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistors."""
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
FUNDAMENTAL = 10e6
"""The distortion's frequency, in hertz."""
HARMONICS = 9
"""The highest harmonic the distortion counts."""
SAMPLES = 200
"""The points of the last period the distortion's spectrum is taken over."""


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
    series = c.place(devices.resistor(75 * ohm), id="RSRC")
    c.connect(source.positive, series.a, name="source")
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


def distortion(made: Component) -> float:
    """The difference's harmonic distortion over its second period, in decibels."""
    period = 1 / FUNDAMENTAL
    waveforms = Simulation(circuit(bench(made, distorting=True))).run(
        Transient(stop=2 * period * s, max_step=period / 1000 * s)
    )
    difference = waveforms.voltage("out_of_phase") - waveforms.voltage("in_phase")
    times = period + period * np.arange(SAMPLES) / SAMPLES
    spectrum = np.fft.rfft(np.interp(times, waveforms.times, difference))
    harmonics = np.abs(spectrum[1 : HARMONICS + 1])
    total = np.sqrt(np.sum(harmonics[1:] ** 2))
    return float(20 * np.log10(max(total, 1e-300) / max(harmonics[0], 1e-300)))


def measure(made: Component) -> dict[str, float]:
    """Koza's seven fitness cases on `made`, each as a mean over its frequencies."""
    simulation = Simulation(circuit(bench(made)))
    response = simulation.run(Ac("VIN", decade(1 * MHz, 100 * MHz, 20)))
    in_phase = response.voltage("in_phase")
    out_of_phase = response.voltage("out_of_phase")
    difference = out_of_phase - in_phase
    half = decibels(np.array(0.5))
    solution = simulation.run(OperatingPoint())
    return {
        "in-phase gain error": float(np.mean(np.abs(decibels(in_phase) - half))),
        "out-of-phase gain error": float(
            np.mean(np.abs(decibels(out_of_phase) - half))
        ),
        "in-phase phase error": float(np.mean(off(in_phase, 0.0))),
        "out-of-phase phase error": float(np.mean(off(out_of_phase, 180.0))),
        "difference phase error": float(np.mean(off(difference, 180.0))),
        "difference gain error": float(np.mean(np.abs(decibels(difference)))),
        "offset": abs(solution.voltage("out_of_phase") - solution.voltage("in_phase")),
        "distortion": distortion(made),
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
        error = found[f"{output} gain error"]
        if error > TOLERANCE["gain"]:
            reasons.append(f"the {output} gain is {error:.4g} dB off on average")
    for signal in ("in-phase", "out-of-phase", "difference"):
        error = found[f"{signal} phase error"]
        if error > TOLERANCE["phase"]:
            reasons.append(f"the {signal} phase is {error:.4g}° off on average")
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
    found = measure(made)
    reasons += shortfalls(found)
    measurements = found | {f"evolved {name}": value for name, value in EVOLVED.items()}
    measurements["devices"] = float(count_devices(built, "X1"))
    return Verdict(met=not reasons, measurements=measurements, reasons=tuple(reasons))
