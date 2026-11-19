# Copyright 2026 Maxence Faldor
"""Measure each oscillator: frequency, amplitude, distortion, settling and drift.

The distortion comes from the output's spectrum over its last 40 periods, resampled
evenly; each harmonic is the spectrum's magnitude at its multiple of the fundamental.
"""

from itertools import pairwise

import candidates
import numpy as np

from jolty import Component, Simulation, Transient, circuit
from jolty.units import celsius, ms, us

STOP = 0.15


def run(
    oscillator: Component, temperature: float = 27.0
) -> tuple[np.ndarray, np.ndarray]:
    """The output over 150 ms."""
    simulation = Simulation(
        circuit(candidates.bench(oscillator)), temperature=celsius(temperature)
    )
    waveforms = simulation.run(Transient(stop=STOP * 1e3 * ms, max_step=1 * us))
    return waveforms.times, waveforms.voltage("out")


def measure(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    """Frequency, peak amplitude and THD over the last 40 ms, and the settling time.

    The oscillator has settled once its envelope stays within 5 % of its final value.
    """
    window = times > STOP - 0.04
    rises = np.flatnonzero((values[:-1] < 0) & (values[1:] >= 0) & window[1:])
    crossings = times[rises] - values[rises] * (times[rises + 1] - times[rises]) / (
        values[rises + 1] - values[rises]
    )
    frequency = (len(crossings) - 1) / (crossings[-1] - crossings[0])
    start, stop = crossings[0], crossings[-1]
    grid = np.linspace(start, stop, 2**15, endpoint=False)
    samples = np.interp(grid, times, values)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    cycles = len(crossings) - 1
    fundamental = spectrum[cycles - 1 : cycles + 2].max()
    harmonics = [spectrum[k * cycles - 1 : k * cycles + 2].max() for k in range(2, 11)]
    thd = float(np.sqrt(np.sum(np.square(harmonics))) / fundamental)
    amplitude = float((values[window].max() - values[window].min()) / 2)
    edges = np.arange(0.0, STOP + 1e-12, 1e-3)
    envelope = np.array(
        [
            np.abs(values[(times >= low) & (times < high)]).max(initial=0.0)
            for low, high in pairwise(edges)
        ]
    )
    off = np.flatnonzero(np.abs(envelope - amplitude) > 0.05 * amplitude)
    settled = edges[off[-1] + 1] if off.size else 0.0
    return {
        "frequency": float(frequency),
        "amplitude": amplitude,
        "thd": thd,
        "settled": float(settled),
    }


def report(
    name: str, oscillator: Component, *, temperatures: bool = False
) -> dict[str, float]:
    """Measures the oscillator, at -40 °C and 85 °C too when `temperatures`."""
    found = measure(*run(oscillator))
    line = (
        f"{name:34s} f={found['frequency']:8.2f} Hz  A={found['amplitude']:.4f} V  "
        f"THD={100 * found['thd']:.3f} %  settled {1e3 * found['settled']:.1f} ms"
    )
    if temperatures:
        for temperature in (-40.0, 85.0):
            other = measure(*run(oscillator, temperature))
            change = other["amplitude"] / found["amplitude"] - 1
            line += f"  {temperature:+.0f}°C: A {100 * change:+.2f} %"
            found[f"amplitude {temperature:+.0f}"] = other["amplitude"]
    print(line, flush=True)
    return found


if __name__ == "__main__":
    report("diodes 12k + 10k", candidates.diodes(12e3, 10e3), temperatures=True)
    report(
        "agc linearized",
        candidates.agc(9.5e3, 0.5, linearized=True),
        temperatures=True,
    )
    report("agc plain", candidates.agc(9.5e3, 0.5, linearized=False))
    for sense, hold in ((0.45, 0.1e-6), (0.45, 0.47e-6), (0.43, 0.1e-6)):
        report(
            f"sensed {sense:.0%}, {hold:.2g} F",
            candidates.sensed_agc(9.5e3, sense, hold),
            temperatures=hold < 0.2e-6,
        )
