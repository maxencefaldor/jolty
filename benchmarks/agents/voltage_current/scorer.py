# Copyright 2026 Maxence Faldor
"""Hidden benchmark scorer for the voltage-to-current converter design task.

The scorer mounts the agent's component on Koza's test fixture for the voltage-current
converter of US patent 6,166,529 (Genetic Programming IV, sections 15.3.1.4 and
15.3.5.4). Evaluation runs Koza's four 1 s fitness cases, sampling output current at 31
points spaced 1/30 s apart. Each case's mean absolute error against ``1 mA/V * (V_plus
- V_minus)`` is normalized by Koza's patented circuit errors (41, 36, 26, and 44 µA
across the 1 Ω load). Averaging the four ratios gives composite fitness, where the
patented circuit scores 1.0 and the evolved circuit scores 0.619.

A submission passes when using only allowed parts, achieving fitness <= 1.0, and
settling within 40 µs after a 10 mV step on ``VS`` at three operating points to within
44 µA of ideal output current. The settling test catches high-frequency oscillation and
inter-sample spikes invisible to Koza's 1/30 s sampling interval. All requirements must
hold from 0 °C to 70 °C at +/-5 % supply around -15 V, both nominally and on at least
95 % of Monte Carlo boards with 1 % resistors, 5 % capacitors, and datasheet
reel-varied transistors.

Port ``plus`` is the non-inverting input and ``minus`` is the inverting input, so
output current flows out of ``output`` when ``plus`` exceeds ``minus``.
"""

from pathlib import Path

import numpy as np
from arms import Arm
from scoring import Board, DesignSpec, discrete, score_design, unmet
from verdict import Shortfall, Verdict

from jolty import Builder, Component, Role, circuit, component, devices
from jolty.plan import Transient
from jolty.units import Hz, V, ns, ohm, s, us
from jolty.waveforms import Piecewise, Pulse, Sine

TRANSCONDUCTANCE = 1e-3
"""Target transconductance in amperes per volt of differential input."""
TIMES = 31
"""Number of evenly spaced sample points per 1 s fitness case (1/30 s spacing)."""
PATENT = (41e-6, 36e-6, 26e-6, 44e-6)
"""Patented circuit mean absolute output current errors in amperes per Koza."""
EVOLVED = 0.619
"""Composite fitness score reported by Koza for the genetically evolved circuit."""
ALLOWED = {"resistor", "capacitor"}
"""Passive device models permitted in addition to the given discrete transistors."""
SUPPLIES = ("VEE",)
"""Top-level bench supply scaled by +/-5 % across operating corners."""
PORTS = ("plus", "minus", "output", "vee", "ground")
"""Required port names exposed by the voltage-current converter component."""
STEP = 10e-3
"""Load-side ``VS`` voltage step amplitude in volts for settling checks."""
SETTLING = ((6.0, 4.0, 0.0), (10.0, 2.0, 1.0), (2.0, 10.0, -1.0))
"""Operating points ``(plus, minus, VS)`` in volts tested for step settling."""
UNSETTLED = 1e-6
"""Maximum peak-to-peak current ripple in amperes tolerated after settling."""
SETTLED = 40e-6
"""Settling deadline in seconds after which output current must stabilize."""
LOAD = ("RLOAD",)
"""Recorded load resistor instance name whose current represents circuit output."""
SETTLED_ERROR = max(PATENT)
"""Maximum steady-state current error in amperes after settling (worst patent case)."""


def ramp(start: float, end: float) -> devices.Device:
    """Returns a source ramping linearly from `start` to `end` volts over 1 s."""
    return devices.voltage_source(Piecewise([(0 * s, start * V), (1 * s, end * V)]))


def sine(middle: float, swing: float, frequency: float) -> devices.Device:
    """Returns a sine source with amplitude `swing` around `middle` at `frequency`."""
    return devices.voltage_source(
        Sine(offset=middle * V, amplitude=swing * V, frequency=frequency * Hz)
    )


def cases() -> list[tuple[devices.Device, devices.Device]]:
    """Returns Koza's four ``(plus, minus)`` voltage-source stimulus pairs."""
    return [
        (ramp(10, 2), ramp(2, 10)),
        (sine(5.5, 3.5, 4.5), sine(6, 3.5, 6.4)),
        (sine(6, 4, 2), sine(6, 4, 2)),
        (sine(4, 1, 4.5), sine(4, 1, 6.4)),
    ]


def swing() -> devices.Device:
    """Returns Koza's ``VS`` load-side source swinging 1 V around 0 V at 3.2 Hz."""
    return devices.voltage_source(
        Sine(offset=0 * V, amplitude=1 * V, frequency=3.2 * Hz)
    )


def step(level: float) -> devices.Device:
    """Returns a ``VS`` pulse source stepping up 10 mV from `level` at 1 µs."""
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
    """Builds Koza's converter fixture around `made` with `inputs` and `output`.

    Connects `inputs` to ``plus`` and ``minus`` through 1 µΩ sense resistors and places
    `output` as load-side source ``VS``.
    """
    ground = c.port("ground", Role.GROUND)
    placed = c.place(made, name="X1")
    supply = c.place(devices.voltage_source(-15 * V), name="VEE")
    c.connect(supply.positive, placed.vee)
    c.connect(supply.negative, placed.ground, ground)
    for name, source in zip(("plus", "minus"), inputs, strict=True):
        driven = c.place(source, name=f"V{name.upper()}")
        series = c.place(devices.resistor(1e-6 * ohm), name=f"R{name.upper()}")
        c.connect(driven.positive, series.a, name=name)
        c.connect(series.b, placed[name])
        c.connect(driven.negative, ground)
    held = c.place(output, name="VS")
    load = c.place(devices.resistor(1 * ohm), name="RLOAD")
    c.connect(placed.output, held.positive, name="output")
    c.connect(held.negative, load.a, name="load")
    c.connect(load.b, ground)


def measure(made: Component, board: Board) -> dict[str, float]:
    """Measures Koza's four fitness cases and three settling steps on `board`.

    Reports per-case mean absolute current errors and composite fitness. Each settling
    step also measures output current ripple and mean error over the final 10 µs of a
    50 µs run at 10 ns max step because millisecond fitness steps cannot expose
    oscillation.
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
                breakpoints=[time * s for time in times[1:]],
                record=LOAD,
            ),
        )

        def read(values: np.ndarray, at: np.ndarray = waveforms.times) -> np.ndarray:
            return np.interp(times, at, values)

        output = read(waveforms.current("RLOAD", "a"))
        wanted = TRANSCONDUCTANCE * (
            read(waveforms.voltage("plus")) - read(waveforms.voltage("minus"))
        )
        errors.append(float(np.mean(np.abs(output - wanted))))
    measures = {"fitness": float(np.mean(np.divide(errors, PATENT)))}
    measures |= {f"error {case}": error for case, error in enumerate(errors, start=1)}
    for case, (plus, minus, level) in enumerate(SETTLING, start=1):
        inputs = (devices.voltage_source(plus * V), devices.voltage_source(minus * V))
        waveforms = board.simulation(
            circuit(bench(made, inputs, step(level))), SUPPLIES
        ).run(Transient(stop=50 * us, max_step=10 * ns, record=LOAD))
        late = waveforms.times >= SETTLED
        current = waveforms.current("RLOAD", "a")[late]
        measures[f"swing {case}"] = float(np.ptp(current))
        wanted = TRANSCONDUCTANCE * (plus - minus)
        measures[f"settled error {case}"] = abs(float(np.mean(current)) - wanted)
    return measures


def shortfalls(found: dict[str, float]) -> list[Shortfall]:
    """Checks converter fitness and settling metrics in `found` for shortfalls."""
    reasons = []
    if found["fitness"] > 1:
        reason = f"the fitness is {found['fitness']:.4g}, more than the patent's 1"
        reasons.append(unmet("fitness", reason))
    for case, (plus, minus, level) in enumerate(SETTLING, start=1):
        swing = found[f"swing {case}"]
        if swing > UNSETTLED:
            reason = (
                f"with plus at {plus} V, minus at {minus} V and VS stepped from "
                f"{level} V, the output current still swings by {swing:.3g} A"
            )
            reasons.append(unmet(f"swing {case}", reason))
        error = found[f"settled error {case}"]
        if error > SETTLED_ERROR:
            reason = (
                f"with plus at {plus} V, minus at {minus} V and VS stepped from "
                f"{level} V, the output current settles {error:.3g} A from its "
                f"{TRANSCONDUCTANCE * (plus - minus):.3g} A, more than the patent's "
                f"largest mean error, {SETTLED_ERROR:.3g} A"
            )
            reasons.append(unmet(f"settled error {case}", reason))
    return reasons


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores the voltage-current converter submission in `work` for `arm`."""
    return score_design(
        task,
        work,
        arm,
        DesignSpec(
            ports=PORTS,
            bench=lambda made: bench(made, cases()[0], swing()),
            measure=measure,
            shortfalls=shortfalls,
            parts=("npn", "pnp"),
            allowed=ALLOWED,
            spread=discrete,
            extra_measures={
                **{
                    f"patent's error {case}": theirs
                    for case, theirs in enumerate(PATENT, start=1)
                },
                "evolved fitness": EVOLVED,
            },
        ),
    )
