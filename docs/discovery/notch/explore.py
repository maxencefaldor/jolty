# Copyright 2026 Maxence Faldor
"""Monte Carlo yields of the notches.

Each sample's depth is read between 45 Hz and 55 Hz, and its passband at 25 Hz and
100 Hz.
"""

import candidates
import numpy as np

from jolty import Ac, Component, Simulation, circuit, ir, variation
from jolty.units import Hz

FREQUENCIES = [25.0, *np.linspace(45.0, 55.0, 401), 100.0]


def response(
    drawn: ir.Circuit, overrides: dict[str, dict[str, float]] | None = None
) -> np.ndarray:
    """The gain at each of FREQUENCIES, in decibels."""
    simulation = Simulation(drawn, overrides=overrides or {})
    found = simulation.run(Ac("V1", [f * Hz for f in FREQUENCIES]))
    with np.errstate(divide="ignore"):  # a balanced notch is infinitely deep
        return 20 * np.log10(np.abs(found.voltage("out")))


def judge(gains: np.ndarray) -> tuple[float, float, bool]:
    """The notch's depth and the passband's worst deviation, and whether both pass."""
    depth = float(-gains[1:-1].min())
    passband = float(max(abs(gains[0]), abs(gains[-1])))
    return depth, passband, depth >= 50 and passband <= 3


def study(name: str, notch: Component, samples: int = 200) -> None:
    """Prints the notch's nominal depth and passband, and its Monte Carlo yield."""
    drawn = circuit(candidates.bench(notch))
    nominal = judge(response(drawn))
    varied = variation.toleranced(drawn)
    outcomes = [
        judge(response(drawn, variation.sample(varied, 1, index)))
        for index in range(samples)
    ]
    depths = np.array([depth for depth, _, _ in outcomes])
    passing = sum(ok for _, _, ok in outcomes)
    print(
        f"{name:28s} nominal depth {nominal[0]:6.1f} dB, "
        f"passband {nominal[1]:.2f} dB | {samples} samples: "
        f"depth min {depths.min():5.1f} median {np.median(depths):5.1f} dB, "
        f"yield {passing}/{samples} = {100 * passing / samples:.1f} %",
        flush=True,
    )


if __name__ == "__main__":
    study("passive twin-T", candidates.twin_t())
    study("twin-T bootstrapped, Q=1", candidates.bootstrapped())
    for quality in (0.5, 1.0):
        study(f"LP/BP summed, Q={quality}", candidates.summed(quality))
