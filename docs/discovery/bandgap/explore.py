# Copyright 2026 Maxence Faldor
"""Evaluation script measuring bandgap reference drift across temperature and supply.

Bisects the Widlar resistor for each topology to flatten output over temperature.
"""

from collections.abc import Callable

import candidates
import numpy as np

from jolty import Component, DcSweep, OperatingPoint, Simulation, circuit
from jolty.plan import linear
from jolty.units import V, celsius

TEMPERATURES = [-40.0, -20.0, 0.0, 27.0, 50.0, 75.0, 100.0, 125.0]


def output(
    reference: Component, temperature: float = 27.0, supply: float = 5.0
) -> float:
    """Returns the reference output voltage at `temperature` °C and `supply` volts."""
    simulation = Simulation(
        circuit(candidates.bench(reference, supply)), temperature=celsius(temperature)
    )
    return simulation.run(OperatingPoint()).voltage("out")


def measure(name: str, reference: Component) -> tuple[float, float, float]:
    """Measures 27 °C output voltage, temperature span in mV, and line regulation."""
    values = np.array([output(reference, t) for t in TEMPERATURES])
    simulation = Simulation(circuit(candidates.bench(reference)))
    sweep = simulation.run(DcSweep("VCC", linear(4.5 * V, 5.5 * V, 11)))
    line = float(np.ptp(sweep.voltage("out")))
    nominal = output(reference)
    span = float(np.ptp(values))
    ppm = span / nominal / 165 * 1e6
    print(
        f"{name:34s} {nominal:.4f} V  span {1e3 * span:7.2f} mV ({ppm:6.1f} ppm/K)  "
        f"line {1e3 * line:6.2f} mV/V  "
        + " ".join(
            f"{t:+.0f}:{v:.4f}" for t, v in zip(TEMPERATURES, values, strict=True)
        ),
        flush=True,
    )
    return nominal, span, line


def flatten(make: Callable[[float], Component], low: float, high: float) -> float:
    """Finds the Widlar resistor that flattens the reference built by `make`.

    Bisects between `low` and `high` ohms to 0.1 Ω until the 125 °C output matches
    the -40 °C output.
    """

    def tilt(widlar_r: float) -> float:
        reference = make(widlar_r)
        return output(reference, 125.0) - output(reference, -40.0)

    rising = tilt(high) > tilt(low)
    while high - low > 0.1:
        middle = (low + high) / 2
        if (tilt(middle) > 0) == rising:
            high = middle
        else:
            low = middle
    return round((low + high) / 2, 1)


def jfet_fed(top: float, widlar_r: float, ptat: float) -> None:
    """Measures the JFET-fed bandgap for R1 `top`, R2 `widlar_r`, and R3 `ptat`."""
    name = f"JFET-fed, R1 {top:g}, R2 {widlar_r:g}, R3 {ptat:g}"
    measure(name, candidates.widlar_jfet(top, widlar_r, ptat))


if __name__ == "__main__":
    measure("diode fed by 4.3 kOhm", candidates.diode())
    measure("Widlar, R3 6 k, R2 500, fed by 1 k", candidates.widlar(600.0, 500.0, 6e3))
    flat = flatten(lambda r: candidates.widlar(600.0, r, 3.6e3), 50.0, 1000.0)
    measure(f"Widlar, R3 3.6 k, R2 {flat}", candidates.widlar(600.0, flat, 3.6e3))
    jfet_fed(600.0, flat, 3.6e3)
    for ptat in (2.4e3, 3.6e3, 6e3):
        jfet_fed(
            600.0,
            flatten(
                lambda r, ptat=ptat: candidates.widlar_jfet(600.0, r, ptat), 50, 1e3
            ),
            ptat,
        )
    # Evaluate the final design and ±1% perturbations of each resistor.
    for top, widlar_r in ((600, 155.1), (606, 155.1), (594, 155.1)):
        jfet_fed(top, widlar_r, 2.4e3)
    for widlar_r in (156.651, 153.549):
        jfet_fed(600.0, widlar_r, 2.4e3)
