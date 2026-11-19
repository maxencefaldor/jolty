# Copyright 2026 Maxence Faldor
"""Hidden benchmark scorer for the integrated cubic function generator task.

The scorer mounts the agent's component on the task test bench and evaluates mean
absolute output-current error against ``I_in^3 / I_ref^2`` across four test cases: a
DC sweep and sinusoidal inputs at 1 MHz, 100 MHz, and 1 GHz. Each case error is
normalized by the patented circuit error from ``patent.py`` at the same operating
corner, following Koza's weighting, and averaged into a composite score. A submission
passes when using only allowed parts and achieving score <= 1.0 across 0 °C to 70 °C
at +/-5 % supply voltage around 2 V, both nominally and on at least 95 % of Monte
Carlo boards. Each board applies integrated die tolerances: +/-20 % resistor lot
spread, +/-0.1 % resistor matching, and +/-5 % capacitor tolerance.
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
"""Nominal reference current ``I_ref`` in amperes."""
FREQUENCIES = (1e6, 1e8, 1e9)
"""Sinusoidal input frequencies in hertz for the three transient test cases."""
ALLOWED = {"resistor", "capacitor"}
"""Passive device models permitted alongside the given integrated NPN transistor."""
SUPPLIES = ("VS",)
"""Top-level bench supply scaled by +/-5 % across operating corners."""
PORTS = ("supply", "reference", "input", "output", "ground")
"""Required port names exposed by the integrated cubic generator component."""


@component
def bench(c: Builder, made: Component, frequency: float = 0.0) -> None:
    """Builds the 2 V cubic test bench around `made` with DC or sinusoidal input."""
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
"""Display labels for the DC sweep and three sinusoidal test cases."""


def errors(made: Component, board: Board) -> list[float]:
    """Computes mean absolute output-current errors in amperes across all cases."""
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
    """Measures per-case errors and normalized score for `made` on `board` vs `patent`.

    Normalizes each case error by the patented circuit error at the matching corner.
    """
    found = errors(made, board)
    measures = {"score": float(np.mean(np.divide(found, patent[board.corner])))}
    measures |= {
        f"error {name}": error for name, error in zip(NAMES, found, strict=True)
    }
    return measures


def shortfalls(found: dict[str, float]) -> list[Shortfall]:
    """Checks cubic generator metrics in `found` and returns any unmet shortfalls."""
    if found["score"] <= 1:
        return []
    return [
        unmet("score", f"the score is {found['score']:.4g}, more than the patent's 1")
    ]


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores the integrated cubic generator submission in `work` for `arm`."""
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
            extra_measures={
                f"patent's error {name}": error
                for name, error in zip(NAMES, patent[CORNERS[0]], strict=True)
            },
        ),
    )
