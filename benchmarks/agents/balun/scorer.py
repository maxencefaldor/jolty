# Copyright 2026 Maxence Faldor
"""Hidden benchmark scorer for the low-voltage active balun design task.

The scorer mounts the agent's component on Koza's test fixture for the low-voltage
balun of US patent 6,265,908 (Genetic Programming IV, sections 15.3.1.1 and 15.3.5.1).
The fixture provides a 1 V supply, a 75 Ω source impedance, and two 150 Ω output loads
tied to the supply.
Evaluation covers Koza's seven fitness cases across 41 frequencies from 1 MHz to
100 MHz: each output's gain relative to 0.5 and phase relative to 0° or 180°, plus
differential phase relative to 180°. Additional checks measure output DC offset and
differential total harmonic distortion across harmonics 2 through 9 of a 10 mV, 10 MHz
sine over 200 ns.

Koza's published total fitness figures (1.72 patented, 0.429 evolved) cannot be
reconstructed from the stated weighted sums because the patent's 0.60 dB differential
gain error alone sums to roughly 25 over 41 points. The scorer instead enforces Koza's
two summary figures directly: mean differential gain error at most 0.60 dB (evolved:
0.47 dB) and total harmonic distortion at most -33.9 dB (evolved: -65.6 dB). Each
output must also remain within Koza's 3 dB and 10° per-frequency tolerances, and output
DC offset must stay within 1.72 mV (the maximum allowed by a 1.72 total at Koza's
weight of 1,000 per volt). Every specification must hold from 0 °C to 70 °C at +/-5 %
supply voltage, both nominally and on at least 95 % of Monte Carlo boards using 1 %
resistors, 5 % capacitors, and reel-varied datasheet transistors.

A 1 µF blocking capacitor isolates the source ahead of the 75 Ω impedance because Koza
omits the source DC level, requiring the design to self-bias the input. Distortion uses
the 10 mV amplitude of Koza's sweeps, and phase errors wrap circularly so -179° sits 1°
from 180°.
"""

from pathlib import Path

import numpy as np
from arms import Arm
from scoring import Board, DesignSpec, discrete, score_design, unmet
from verdict import Shortfall, Verdict

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
"""Passive device models permitted in addition to the given discrete transistors."""
SUPPLIES = ("VCC",)
"""Top-level bench supply scaled by +/-5 % across operating corners."""
PORTS = ("input", "in_phase", "out_of_phase", "vcc", "ground")
"""Required port names exposed by the balun component."""
GAIN = 0.60
"""Patented balun mean absolute differential gain error target in decibels."""
DISTORTION = -33.9
"""Patented balun total harmonic distortion target in decibels."""
EVOLVED = {"difference gain error": 0.47, "distortion": -65.6}
"""Summary metrics reported by Koza for the genetically evolved balun."""
TOLERANCE = {"gain": 3.0, "phase": 10.0}
"""Per-frequency tolerances on single-ended gain in decibels and phase in degrees."""
OFFSET = 1.72e-3
"""Maximum permitted DC offset between outputs in volts."""
BLOCKING = 1e-6
"""Input AC-coupling capacitance in farads isolating source DC bias."""
FUNDAMENTAL = 10e6
"""Fundamental sine frequency in hertz for the harmonic distortion test."""
HARMONICS = 9
"""Highest harmonic order included in the total harmonic distortion calculation."""


@component
def bench(c: Builder, made: Component, *, distorting: bool = False) -> None:
    """Builds Koza's balun fixture for `made`, using a 10 MHz sine if `distorting`."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, name="X1")
    supply = c.place(devices.voltage_source(1 * V), name="VCC")
    c.connect(supply.positive, placed.vcc, name="vcc")
    c.connect(supply.negative, placed.ground, ground)
    level = (
        Sine(offset=0 * V, amplitude=10 * mV, frequency=FUNDAMENTAL * Hz)
        if distorting
        else 0 * V
    )
    source = c.place(devices.voltage_source(level), name="VIN")
    block = c.place(devices.capacitor(BLOCKING * F), name="CIN")
    series = c.place(devices.resistor(75 * ohm), name="RSRC")
    c.connect(source.positive, block.a, name="source")
    c.connect(block.b, series.a, name="coupled")
    c.connect(series.b, placed.input, name="input")
    c.connect(source.negative, ground)
    for output in ("in_phase", "out_of_phase"):
        load = c.place(devices.resistor(150 * ohm), name=f"RLOAD_{output.upper()}")
        c.connect(placed[output], load.a, name=output)
        c.connect(load.b, supply.positive)


def decibels(gain: np.ndarray) -> np.ndarray:
    """Converts complex gain magnitudes in `gain` to decibels."""
    return 20 * np.log10(np.maximum(np.abs(gain), 1e-30))


def off(gain: np.ndarray, phase: float) -> np.ndarray:
    """Computes wrapped phase error in degrees between `gain` and target `phase`."""
    return np.abs((np.angle(gain, deg=True) - phase + 180.0) % 360.0 - 180.0)


def distortion(made: Component, board: Board) -> float:
    """Measures differential harmonic distortion in dB on `board` over cycle two."""
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
    """Measures Koza's seven balun fitness metrics for `made` on `board`.

    Reports mean differential gain error across frequency and worst-case error across
    frequency for all other AC metrics.
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


def shortfalls(found: dict[str, float]) -> list[Shortfall]:
    """Checks balun measurements in `found` and returns any unmet shortfalls."""
    reasons = []
    if found["difference gain error"] > GAIN:
        reason = (
            f"the difference's gain is {found['difference gain error']:.4g} dB off "
            f"on average, more than {GAIN} dB"
        )
        reasons.append(unmet("difference gain error", reason))
    if found["distortion"] > DISTORTION:
        reason = (
            f"the distortion is {found['distortion']:.4g} dB, above {DISTORTION} dB"
        )
        reasons.append(unmet("distortion", reason))
    for output in ("in-phase", "out-of-phase"):
        name = f"worst {output} gain error"
        if found[name] > TOLERANCE["gain"]:
            reason = f"the {output} gain is {found[name]:.4g} dB off at worst"
            reasons.append(unmet(name, reason))
    for signal in ("in-phase", "out-of-phase", "difference"):
        name = f"worst {signal} phase error"
        if found[name] > TOLERANCE["phase"]:
            reason = f"the {signal} phase is {found[name]:.4g}° off at worst"
            reasons.append(unmet(name, reason))
    if found["offset"] > OFFSET:
        reason = f"the outputs' DC levels differ by {found['offset']:.4g} V"
        reasons.append(unmet("offset", reason))
    return reasons


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores the balun submission in `work` against `task` for `arm`."""
    return score_design(
        task,
        work,
        arm,
        DesignSpec(
            ports=PORTS,
            bench=bench,
            measure=measure,
            shortfalls=shortfalls,
            parts=("npn", "pnp"),
            allowed=ALLOWED,
            spread=discrete,
            extra_measures={
                f"evolved {name}": value for name, value in EVOLVED.items()
            },
        ),
    )
