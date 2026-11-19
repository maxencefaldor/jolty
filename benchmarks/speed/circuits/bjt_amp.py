# Copyright 2026 Maxence Faldor
"""Two-stage bipolar amplifier fixtures measuring single-run and batch throughput.

A 2N3904 differential pair on a 1 mA tail with a 2N3906 current-mirror load drives a
2N3904 emitter follower under 9.1 kΩ / 1 kΩ feedback (gain near 10 from ±12 V). All
resistors carry 5 % tolerances exercised during Monte Carlo sampling. Three fixtures
share the circuit:

- ``bjt_amp``: operating point, 71-point AC sweep (10 Hz to 100 MHz), and 200 µs
  transient at 10 kHz.
- ``ac_wide``: 10,000-point AC sweep (1 Hz to 1 GHz) measuring frequency-parallel speed.
- ``monte_carlo``: 100,000-sample Monte Carlo batch of ``bjt_amp`` measuring throughput.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "cluster"))

from amplifier import TOLERANCE, amplifier, bench, build, resistor
from fixture import Fixture, write

from jolty import Component, Plan
from jolty.plan import Ac, MonteCarlo, OperatingPoint, Transient, decade
from jolty.units import GHz, Hz, MHz, us

__all__ = [
    "TOLERANCE",
    "amplifier",
    "bench",
    "build",
    "circuits",
    "fixtures",
    "plan",
    "resistor",
    "wide",
]


def plan() -> Plan:
    """Builds the default amplifier plan: OP, 71-point AC, and two sine periods."""
    return Plan(
        {
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(10 * Hz, 100 * MHz, 10)),
            "tran": Transient(stop=200 * us),
        }
    )


def wide(per_decade: int = 1111) -> Plan:
    """Builds an AC plan with `per_decade` points per decade from 1 Hz to 1 GHz.

    Defaults to 1,111 points per decade (10,000 frequencies total) for ``ac_wide``;
    ``scripts/bench --break-even min_solves_per_task`` passes smaller counts.
    """
    return Plan({"ac": Ac("V1", decade(1 * Hz, 1 * GHz, per_decade))})


def circuits() -> dict[str, Component]:
    """Returns the amplifier circuit keyed by output file stem."""
    return {"bjt_amp": build()}


def fixtures() -> dict[str, Fixture]:
    """Returns the ``bjt_amp``, ``ac_wide``, and ``monte_carlo`` speed fixtures."""
    one = plan()
    return {
        "bjt_amp": Fixture("bjt_amp", one),
        "ac_wide": Fixture("bjt_amp", wide()),
        "monte_carlo": Fixture(
            "bjt_amp", Plan(one.analyses, monte_carlo=MonteCarlo(100_000))
        ),
    }


if __name__ == "__main__":
    write(circuits(), fixtures())
