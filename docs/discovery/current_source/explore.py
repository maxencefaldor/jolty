# Copyright 2026 Maxence Faldor
"""Tuning and characterization script for two-terminal 1 mA current source candidates.

Run via ``uv run python docs/discovery/current_source/explore.py``.
"""

import math
from collections.abc import Callable

import candidates

from jolty import Component, DcSweep, OperatingPoint, Simulation, circuit
from jolty.plan import linear
from jolty.units import V, celsius

TARGET = 1e-3


def current(
    candidate: Component, volts: float = 12.0, temperature: float = 27.0
) -> float:
    """Returns the DC current flowing from port `a` to port `b` in amperes."""
    simulation = Simulation(
        circuit(candidates.bench(candidate, volts)), temperature=celsius(temperature)
    )
    return -simulation.run(OperatingPoint()).current("V1")


def tune(make: Callable[[float], Component], low: float, high: float) -> float:
    """Finds the resistance between `low` and `high` ohms that produces 1 mA at 12 V.

    Uses logarithmic bisection because every candidate's current decreases monotonically
    with resistance.
    """
    for _ in range(50):
        middle = math.sqrt(low * high)
        if current(make(middle)) > TARGET:
            low = middle
        else:
            high = middle
    return math.sqrt(low * high)


def measure(name: str, made: Component) -> dict[str, float]:
    """Measures line regulation from 3 V to 24 V and drift from -40 °C to 85 °C."""
    simulation = Simulation(circuit(candidates.bench(made, 12.0)))
    sweep = simulation.run(DcSweep("V1", linear(3 * V, 24 * V, 43)))
    flowing = -sweep.current("V1")
    nominal = current(made)
    line = float(max(abs(flowing - nominal)) / nominal)
    temperatures = [-40.0, -15.0, 0.0, 27.0, 50.0, 85.0]
    drifting = [current(made, 12.0, temperature) for temperature in temperatures]
    drift = max(abs(value - nominal) / nominal for value in drifting)
    print(
        f"{name:28s} I={nominal * 1e3:.4f} mA  line={100 * line:7.3f} %  "
        f"drift={100 * drift:7.3f} %  "
        + " ".join(
            f"{t:+.0f}°C:{v * 1e3:.4f}"
            for t, v in zip(temperatures, drifting, strict=True)
        )
    )
    return {"current": nominal, "line": line, "drift": drift}


if __name__ == "__main__":
    resistance = tune(candidates.resistor_only, 100.0, 1e6)
    measure(f"resistor {resistance:.0f} Ohm", candidates.resistor_only(resistance))
    for bias in (100e3, 1e6):
        sense = tune(lambda r, bias=bias: candidates.limiter(r, bias), 100.0, 5e3)
        measure(
            f"limiter Rs={sense:.0f} Rb={bias:.0e}", candidates.limiter(sense, bias)
        )
    resistance = tune(candidates.ptat, 5.0, 5e3)
    measure(f"ptat n=4 R={resistance:.1f}", candidates.ptat(resistance))
    sense = tune(lambda r: candidates.mirrored_limiter(r, 10.0), 100.0, 5e3)
    measure(
        f"mirrored limiter Rs={sense:.1f}", candidates.mirrored_limiter(sense, 10.0)
    )
    sense = tune(candidates.compensated, 100.0, 1e5)
    measure(f"limiter and ptat Rs={sense:.1f}", candidates.compensated(sense))
