# Copyright 2026 Maxence Faldor
"""The cubic task's scorer, which the agent never sees.

It places the agent's component on the task's bench and measures, in four cases, the
mean absolute error of its output current against the input's cube over the reference's
square: a DC sweep, and a sine at 1 MHz, 100 MHz and 1 GHz. Each error is divided by the
patented circuit's, ``patent.py``, on the same case, as Koza weighted his fitness cases,
and the score is their mean. The work meets the task when it builds from the allowed
parts alone and scores at most 1 from 0 °C to 70 °C with the supply within 5 % of 2 V,
each corner against the patent's in it, with exact parts and on 95 % of boards of the
design over the same range. On a board, as on one die, the resistors lie together
within 20 % of their values, and each within 0.1 % of the others; each capacitor lies
within 5 %.
"""

import runpy
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from arms import Arm
from scoring import CORNERS, Board, Corner, DesignSpec, integrated, score_design, unmet
from verdict import Shortfall, Verdict

from jolty import Builder, Component, Role, circuit, component, devices
from jolty.plan import DcSweep, Transient, linear
from jolty.units import A, Hz, V, s, uA
from jolty.waveforms import Sine

REFERENCE = 100e-6
"""The reference current, in amperes."""
FREQUENCIES = (1e6, 1e8, 1e9)
"""The sine cases' frequencies, in hertz."""
ALLOWED = {"resistor", "capacitor"}
"""The devices the design may hold besides the given transistor."""
SUPPLIES = ("VS",)
"""The bench's supply, which strays within 5 % of its level in the corners."""
PORTS = ("supply", "reference", "input", "output", "ground")
"""The design's ports, which the bench joins."""


@component
def bench(c: Builder, made: Component, frequency: float = 0.0) -> None:
    """`made` from 2 V, its input a sine at `frequency`, or 0 A to sweep when 0."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(2 * V), name="VS")
    level = (
        Sine(offset=100 * uA, amplitude=90 * uA, frequency=frequency * Hz)
        if frequency
        else 0 * A
    )
    signal = c.place(devices.current_source(level), name="IIN")
    reference = c.place(devices.current_source(REFERENCE * A), name="IREF")
    meter = c.place(devices.voltage_source(0 * V), name="VM")
    placed = c.place(made, name="X1")
    c.connect(
        supply.positive,
        signal.positive,
        reference.positive,
        meter.positive,
        placed.supply,
        name="supply",
    )
    c.connect(signal.negative, placed.input, name="input")
    c.connect(reference.negative, placed.reference, name="reference")
    c.connect(meter.negative, placed.output, name="output")
    c.connect(supply.negative, placed.ground, ground)


NAMES = ("dc", *(f"{frequency:.0e} Hz" for frequency in FREQUENCIES))
"""The four cases' names."""


def errors(made: Component, board: Board) -> list[float]:
    """Each case's mean absolute error of `made`'s output on `board`, in amperes."""
    swept = board.simulation(circuit(bench(made)), SUPPLIES).run(
        DcSweep("IIN", linear(0 * A, 200 * uA, 41))
    )
    found = [_error(swept.swept, np.abs(swept.current("VM")))]
    for frequency in FREQUENCIES:
        period = 1 / frequency
        waveforms = board.simulation(circuit(bench(made, frequency)), SUPPLIES).run(
            Transient(stop=4 * period * s, max_step=period / 200 * s)
        )
        late = waveforms.times >= 2 * period
        level = 100e-6 + 90e-6 * np.sin(2 * np.pi * frequency * waveforms.times)
        found.append(_error(level[late], np.abs(waveforms.current("VM"))[late]))
    return found


def _error(level: np.ndarray, output: np.ndarray) -> float:
    return float(np.mean(np.abs(output - level**3 / REFERENCE**2)))


def measure(
    made: Component, patent: Mapping[Corner, Sequence[float]], board: Board
) -> dict[str, float]:
    """The errors on `board` of `made`, and their score against the `patent`'s.

    The patent's are those in the board's corner.
    """
    found = errors(made, board)
    measurements = {"score": float(np.mean(np.divide(found, patent[board.corner])))}
    measurements |= {
        f"error {name}": error for name, error in zip(NAMES, found, strict=True)
    }
    return measurements


def shortfalls(found: dict[str, float]) -> list[Shortfall]:
    """Why the measurements `found` fall short of the task, if they do."""
    if found["score"] <= 1:
        return []
    return [
        unmet("score", f"the score is {found['score']:.4g}, more than the patent's 1")
    ]


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """The verdict on the cubic `work` against `task`, as `arm` writes it."""
    patented = runpy.run_path(str(task / "patent.py"))["cubic"]()
    patent = {corner: errors(patented, Board(corner=corner)) for corner in CORNERS}
    return score_design(
        task,
        work,
        arm,
        DesignSpec(
            ports=PORTS,
            bench=bench,
            measure=lambda made, board: measure(made, patent, board),
            shortfalls=shortfalls,
            parts=("npn",),
            allowed=ALLOWED,
            spread=integrated,
            extra_measurements={
                f"patent's error {name}": error
                for name, error in zip(NAMES, patent[CORNERS[0]], strict=True)
            },
        ),
    )
