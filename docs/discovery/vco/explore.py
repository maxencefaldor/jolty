# Copyright 2026 Maxence Faldor
"""Each VCO's frequency against its control, and its Schmitt trigger's delay.

Each frequency is compared with the line through the origin that fits them all.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Component, Simulation, Transient, circuit
from jolty.results import TransientResult
from jolty.specs import frequency
from jolty.units import s

CONTROLS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
FINE = [float(v) for v in np.round(np.arange(0.5, 5.01, 0.25), 2)]


def run(oscillator: Component, volts: float, periods: int) -> TransientResult:
    """`periods` periods of 1 kHz/V at a control of `volts`, 4000 steps a period."""
    period = 1e-3 / volts
    simulation = Simulation(circuit(candidates.bench(oscillator, volts)))
    return simulation.run(
        Transient(stop=periods * period * s, max_step=period / 4000 * s)
    )


def oscillate(oscillator: Component, volts: float, periods: int = 16) -> float:
    """The frequency at a control of `volts`, or NaN when the triangle has stalled.

    It is read over all but the first four periods.
    """
    waveforms = run(oscillator, volts, periods)
    start = 4e-3 / volts
    late = waveforms.times >= start
    if np.ptp(waveforms.voltage("out")[late]) < 1.0:
        return float("nan")
    return frequency("run", "out", start=start * s).read(waveforms).value


def delay(oscillator: Component, volts: float) -> float:
    """The Schmitt trigger's mean delay at a control of `volts`.

    It is the time from the triangle crossing the trigger's threshold to the trigger's
    output crossing zero, after the first four periods.
    """
    waveforms = run(oscillator, volts, 8)
    late = waveforms.times >= 4e-3 / volts
    times = waveforms.times[late]
    excess = (waveforms.voltage("out") - waveforms.voltage("X1/positive"))[late]
    square = waveforms.voltage("X1/square")[late]
    crossed = times[1:][np.diff(np.sign(excess)) != 0]
    switched = times[1:][np.diff(np.sign(square)) != 0]
    pairs = min(crossed.size, switched.size)
    return float(np.mean(switched[:pairs] - crossed[:pairs]))


def study(name: str, oscillator: Component, controls: list[float] = CONTROLS) -> None:
    """Prints the oscillator's slope and its worst deviation from the line."""
    found = np.array([oscillate(oscillator, v) for v in controls])
    volts = np.array(controls)
    if np.isnan(found).any():
        stalled = ", ".join(
            f"{v} V" for v, f in zip(controls, found, strict=True) if np.isnan(f)
        )
        print(f"{name:40s} stalls at {stalled}", flush=True)
        return
    slope = float(found @ volts / (volts @ volts))
    error = found / (slope * volts) - 1
    print(
        f"{name:40s} slope {slope:7.2f} Hz/V  "
        f"linearity {100 * np.abs(error).max():.3f} %  "
        + " ".join(
            f"{v}V:{f:.1f}({100 * e:+.2f}%)"
            for v, f, e in zip(controls, found, error, strict=True)
        ),
        flush=True,
    )


if __name__ == "__main__":
    study("2N3904 switch", candidates.integrator_schmitt())
    study("MOSFET switch", candidates.mosfet_switched())
    for speedup in (0.0, 1e-9, 2.2e-9):
        for volts in (0.5, 5.0):
            late = delay(candidates.mosfet_switched(speedup=speedup), volts)
            print(f"Schmitt delay, speed-up {speedup:g} F, at {volts} V: {late:.3g} s")
    study(
        "MOSFET, 5.8 kOhm in series with C1",
        candidates.mosfet_switched(compensation=5.8e3),
    )
    for speedup in (100e-12, 1e-9):
        study(
            f"MOSFET, {speedup:g} F speed-up",
            candidates.mosfet_switched(speedup=speedup),
        )
    study(
        "MOSFET, 1e-10 F, 1000 Ohm",
        candidates.mosfet_switched(compensation=1e3, speedup=100e-12),
    )
    for compensation in (1.5e3, 2e3, 2.5e3):
        for speedup in (470e-12, 1e-9, 2.2e-9):
            study(
                f"MOSFET, {speedup:g} F, {compensation:g} Ohm",
                candidates.mosfet_switched(compensation=compensation, speedup=speedup),
                FINE,
            )
