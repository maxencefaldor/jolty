# Copyright 2026 Maxence Faldor
"""Hidden benchmark scorer for the discrete cubic function generator task.

The scorer mounts the agent's component on Koza's test fixture for the cubic function
generator of US patent 6,160,427 (Genetic Programming IV, sections 15.3.1.5 and
15.3.5.5). The fixture provides a 2 V supply and converts input voltage into a
100 µA/V current drawn into ``input``. Output current drawn into ``output`` converts at
10.5 kV/A to
match the cubed input voltage. Holding one output meter
terminal at the supply yields the same current as Koza's two-port meter. Evaluation
runs Koza's four fitness cases, each sampled at 101 points. The four stimuli are a 1 ms
ramp from 0 V to 1.26 V, one 1 ms sine cycle over that span, a 10 ms ramp down, and 1 V
held for 10 ms.

A submission passes when the mean absolute output error across all four cases is at
most 6.76 mV, matching Koza's measurement of the patented circuit built from 2N3904 and
2N3906 transistors (evolved circuit: 4.20 mV). Koza's hit rate (fraction of samples
below 7 mV error, where Koza shifts weight from 1 to 10) is reported alongside the
error (61 % patented, 93.6 % evolved). The error limit must hold from 0 °C to 70 °C at
+/-5 % supply voltage, both nominally and on at least 95 % of Monte Carlo boards built
with 1 % resistors, 5 % capacitors, and datasheet reel-varied transistors.

The sine starts at midpoint rising as in SPICE because Koza omits the initial phase.
Series 1 µΩ sense resistors match Koza's fixture on both ``input`` and ``output``.
"""

from pathlib import Path

import numpy as np
from arms import Arm
from scoring import Board, DesignSpec, discrete, score_design, unmet
from verdict import Shortfall, Verdict

from jolty import Builder, Component, Role, circuit, component, devices
from jolty.plan import Transient
from jolty.units import A, Hz, Quantity, V, ms, ohm, s
from jolty.waveforms import Piecewise, Sine

TRANSCONDUCTANCE = 100e-6
"""Input voltage-to-current conversion gain in amperes per volt."""
TRANSRESISTANCE = 10.5e3
"""Output current-to-voltage conversion gain in volts per ampere."""
TOP = 1.26
"""Maximum input voltage in volts, approximating the cube root of 2."""
TIMES = 101
"""Number of evenly spaced sample times evaluated per fitness case."""
PATENT = 6.76e-3
"""Patented circuit mean absolute output error across the four cases in volts."""
EVOLVED = 4.20e-3
"""Genetically evolved circuit mean absolute output error in volts."""
HIT = 7e-3
"""Output error threshold in volts defining a hit in Koza's fitness metric."""
ALLOWED = {"resistor", "capacitor"}
"""Passive device models permitted in addition to the given discrete transistors."""
SUPPLIES = ("VCC",)
"""Top-level bench supply scaled by +/-5 % across operating corners."""
PORTS = ("input", "output", "vcc", "ground")
"""Required port names exposed by the discrete cubic generator component."""


def cases() -> list[tuple[Piecewise | Sine | Quantity, float]]:
    """Returns Koza's four stimulus waveforms and durations in seconds."""
    return [
        (Piecewise([(0 * s, 0 * V), (1 * ms, TOP * V)]), 1e-3),
        (Sine(offset=TOP / 2 * V, amplitude=TOP / 2 * V, frequency=1e3 * Hz), 1e-3),
        (Piecewise([(0 * s, TOP * V), (10 * ms, 0 * V)]), 10e-3),
        (1 * V, 10e-3),
    ]


@component
def bench(c: Builder, made: Component, case: int) -> None:
    """Builds Koza's discrete cubic fixture around `made` for fitness `case`."""
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, name="X1")
    supply = c.place(devices.voltage_source(2 * V), name="VCC")
    c.connect(supply.positive, placed.vcc, name="vcc")
    c.connect(supply.negative, placed.ground, ground)
    level, _ = cases()[case]
    command = c.place(devices.voltage_source(level), name="VIN")
    c.connect(command.negative, ground)
    convert = c.place(
        devices.voltage_controlled_current_source(TRANSCONDUCTANCE * A / V),
        name="GIN",
    )
    c.connect(convert.control_positive, command.positive, name="command")
    c.connect(convert.control_negative, ground)
    c.connect(convert.positive, supply.positive)
    meter = c.place(devices.voltage_source(0 * V), name="VMETER")
    c.connect(meter.positive, supply.positive)
    for port, driving in (("input", convert.negative), ("output", meter.negative)):
        series = c.place(devices.resistor(1e-6 * ohm), name=f"R{port.upper()}")
        c.connect(driving, series.a)
        c.connect(series.b, placed[port], name=port)


def measure(made: Component, board: Board) -> dict[str, float]:
    """Measures mean and peak output errors and hit rate for `made` on `board`."""
    found = []
    for case, (_, duration) in enumerate(cases()):
        times = np.linspace(0.0, duration, TIMES)
        waveforms = board.simulation(circuit(bench(made, case)), SUPPLIES).run(
            Transient(
                stop=duration * s,
                max_step=duration / (10 * (TIMES - 1)) * s,
                breakpoints=[time * s for time in times[1:]],
            )
        )
        output = TRANSRESISTANCE * np.interp(
            times, waveforms.times, waveforms.current("VMETER")
        )
        command = np.interp(times, waveforms.times, waveforms.voltage("command"))
        found.append(output - command**3)
    means = [float(np.mean(np.abs(error))) for error in found]
    measures = {
        "mean absolute error": float(np.mean(means)),
        "hits": float(np.mean(np.abs(np.concatenate(found)) < HIT)),
    }
    names = ("rising ramp", "sine", "falling ramp", "constant")
    for name, mean, error in zip(names, means, found, strict=True):
        measures[f"{name} mean absolute error"] = mean
        measures[f"{name} maximum absolute error"] = float(np.max(np.abs(error)))
    return measures


def shortfalls(found: dict[str, float]) -> list[Shortfall]:
    """Checks discrete cubic error metrics in `found` and returns any shortfalls."""
    mean = found["mean absolute error"]
    if mean <= PATENT:
        return []
    reason = (
        f"the mean absolute error is {mean * 1e3:.4g} mV, more than the patent's "
        f"{PATENT * 1e3:.4g} mV"
    )
    return [unmet("mean absolute error", reason)]


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores the discrete cubic generator submission in `work` for `arm`."""
    return score_design(
        task,
        work,
        arm,
        DesignSpec(
            ports=PORTS,
            bench=lambda made: bench(made, 3),
            measure=measure,
            shortfalls=shortfalls,
            parts=("npn", "pnp"),
            allowed=ALLOWED,
            spread=discrete,
            extra_measures={
                "patent's mean absolute error": PATENT,
                "evolved mean absolute error": EVOLVED,
            },
        ),
    )
