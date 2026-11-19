# Copyright 2026 Maxence Faldor
"""The discrete cubic function generator task's scorer, which the agent never sees.

It places the agent's component on Koza's test fixture for the cubic function generator
of US patent 6,160,427 (Genetic Programming IV, sections 15.3.1.5 and 15.3.5.5): a 2 V
supply, the input voltage turned into a current of 100 µA/V drawn from the supply into
``input``, and the output current drawn from the supply into ``output`` turned into a
voltage of 10.5 kV/A, which should be the input voltage's cube. Koza meters the output
between two ports of the circuit; holding one at the supply leaves the same current. It
runs his four fitness cases, each read at 101 times: a ramp from 0 V to 1.26 V over 1
ms, one cycle of a sine between them over 1 ms, a ramp back down over 10 ms, and 1 V
held for 10 ms.

The task is met when the output's absolute error, averaged over the four cases, is at
most 6.76 mV, the patented circuit's with the 2N3904 and the 2N3906 as Koza measured it;
the circuit genetic programming evolved reached 4.20 mV. Koza's fitness weighs each
error 1 below 7 mV and 10 above; the share of errors below 7 mV, his hits, is reported
beside it: 61 % for the patent, 93.6 % for the evolved circuit. The error must stay
within the patent's from 0 °C to 70 °C with the supply within 5 % of 2 V, and on 95 % of
boards of the design too, built of resistors within 1 %, capacitors within 5 % and
transistors from reels anywhere in their datasheets' ranges, each board over the same
range: margins Koza's circuits, simulated with exact parts at 27 °C, need not keep.

Koza does not give the sine's phase; it starts at its middle and rises, as SPICE's do.
His fixture also puts 1 µΩ in series with the input and the output, as the bench does.
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
"""The input current per volt of the input, in amperes per volt."""
TRANSRESISTANCE = 10.5e3
"""The output voltage per ampere of the output current, in volts per ampere."""
TOP = 1.26
"""The input's highest level, the cube root of 2, in volts."""
TIMES = 101
"""The times each case is read at."""
PATENT = 6.76e-3
"""The patented circuit's mean absolute error over the four cases, in volts."""
EVOLVED = 4.20e-3
"""The evolved circuit's mean absolute error over the four cases, in volts."""
HIT = 7e-3
"""The error below which Koza counts a hit, in volts."""
ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistors."""
SUPPLIES = ("VCC",)
"""The bench's supply, which strays within 5 % of its level in the corners."""
PORTS = ("input", "output", "vcc", "ground")
"""The design's ports, which the bench joins."""


def cases() -> list[tuple[Piecewise | Sine | Quantity, float]]:
    """Koza's four fitness cases: the input's level, in volts, and how long, in s."""
    return [
        (Piecewise([(0 * s, 0 * V), (1 * ms, TOP * V)]), 1e-3),
        (Sine(offset=TOP / 2 * V, amplitude=TOP / 2 * V, frequency=1e3 * Hz), 1e-3),
        (Piecewise([(0 * s, TOP * V), (10 * ms, 0 * V)]), 10e-3),
        (1 * V, 10e-3),
    ]


@component
def bench(c: Builder, made: Component, case: int) -> None:
    """`made` on Koza's fixture, its input driven as in `case`."""
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
    """The output's error on `board` of `made`, over each case and over all four."""
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
    """Why the measures `found` fall short of the task, if they do."""
    mean = found["mean absolute error"]
    if mean <= PATENT:
        return []
    reason = (
        f"the mean absolute error is {mean * 1e3:.4g} mV, more than the patent's "
        f"{PATENT * 1e3:.4g} mV"
    )
    return [unmet("mean absolute error", reason)]


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """The verdict on the cubic function generator in `work`, as `arm` writes it."""
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
