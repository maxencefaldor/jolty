# Copyright 2026 Maxence Faldor
"""The range of pulse amplitudes over which each gate computes NOR, and what bounds it.

Each amplitude writes all four inputs at once on the bench; a gate works when every
output ends within 0.1 of its NOR and no input has moved by more than 0.05. The range
is scanned in steps of 10 mV, and what fails a step beyond each end is named. Beside it
stands the range the thresholds alone allow, before any state moves: the output of a
gate with one input on must see more than 0.5 V, the output of a gate with both off
less, and the inputs of that gate less than 1 V.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import candidates

from jolty import Simulation, Transient
from jolty.units import ns

ON, OFF = 1e3, 100e3
"""The memristors' on and off resistances, in ohms."""
TURNS_OFF, TURNS_ON = 0.5, 1.0
"""Their off threshold, and their on threshold's magnitude, in volts."""


def failures(amplitude: float, shunt: float | None, width: float) -> list[str]:
    """What fails when one pulse of `amplitude` and `width` writes the four gates."""
    bench = candidates.bench(float(amplitude), shunt, width)
    stop = (width + 3) * ns
    waveforms = Simulation(bench).run(Transient(stop, max_step=width / 400 * ns))
    failed = []
    for index, (a, b) in enumerate(candidates.CASES, start=1):
        output = float(waveforms.state(f"X{index}/MO:state")[-1])
        wanted = 0.0 if a == b == 0 else 1.0
        if abs(output - wanted) > 0.1:
            failed.append(f"{a}{b} output at {output:.2f}")
        for name, bit in (("MA", a), ("MB", b)):
            moved = abs(float(waveforms.state(f"X{index}/{name}:state")[-1]) - 1 + bit)
            if moved > 0.05:
                failed.append(f"{a}{b} input {name} moved {moved:.2f}")
    return failed


def parallel(*resistances: float | None) -> float:
    """The resistances given in parallel."""
    return 1 / sum(
        1 / resistance for resistance in resistances if resistance is not None
    )


def thresholds(shunt: float | None) -> tuple[float, float]:
    """The amplitudes between which the thresholds alone let the gate work."""
    one_on, both_off = parallel(ON, OFF, shunt), parallel(OFF / 2, shunt)
    lowest = TURNS_OFF * (1 + one_on / ON)
    highest = min(TURNS_OFF * (1 + both_off / ON), TURNS_ON * (1 + ON / both_off))
    return lowest, highest


def study(name: str, shunt: float | None, width: float = candidates.WIDTH) -> None:
    """Prints the range of amplitudes the gate works over, and what bounds it."""
    amplitudes = np.round(np.arange(0.8, 2.6, 0.01), 3)
    working = [
        float(level) for level in amplitudes if not failures(level, shunt, width)
    ]
    lowest, highest = thresholds(shunt)
    static = f"thresholds alone {lowest:.3f} to {highest:.3f} V"
    if not working:
        print(f"{name:24s} {width:6g} ns: none; {static}", flush=True)
        return
    low, high = working[0], working[-1]
    margin = 100 * (high - low) / (high + low)
    print(
        f"{name:24s} {width:6g} ns: {low:.2f} to {high:.2f} V, "
        f"±{margin:.1f} % about {(low + high) / 2:.3f} V; {static}\n"
        f"{'':24s} below: {', '.join(failures(low - 0.01, shunt, width))}\n"
        f"{'':24s} above: {', '.join(failures(high + 0.01, shunt, width))}",
        flush=True,
    )


if __name__ == "__main__":
    for width in (10.0, 100.0, 1000.0):
        study("MAGIC's NOR", None, width)
    for shunt in (20e3, 5e3, 3e3, 2.5e3, 2.2e3, 2e3, 1.5e3, 1e3):
        study(f"shunted by {shunt / 1e3:g} kΩ", shunt)
