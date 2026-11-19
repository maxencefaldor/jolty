# Copyright 2026 Maxence Faldor
"""A two-stage bipolar amplifier, the cost of one small run and of many.

A differential pair of 2N3904s on a 1 mA tail, loaded by a mirror of 2N3906s, drives a
2N3904 follower, and 9.1 kΩ over 1 kΩ feed the output back to the pair, for a gain of
about 10 from ±12 V. Its resistors are within 5 %, which only a Monte Carlo analysis
reads. Three fixtures run it:

- ``bjt_amp``: the operating point, the response at 71 frequencies from 10 Hz to
  100 MHz, and 200 µs of a 10 kHz sine, the cost of one small run;
- ``ac_wide``: the response at 10,000 frequencies from 1 Hz to 1 GHz, for running
  frequencies in parallel;
- ``monte_carlo``: ``bjt_amp``'s analyses on 100,000 samples of the resistors, the
  throughput of many runs.
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
    """The operating point, 71 frequencies and two periods of the sine."""
    return Plan(
        {
            "op": OperatingPoint(),
            "ac": Ac("V1", decade(10 * Hz, 100 * MHz, 10)),
            "tran": Transient(stop=200 * us),
        }
    )


def wide(per_decade: int = 1111) -> Plan:
    """The response at `per_decade` frequencies a decade from 1 Hz to 1 GHz.

    ``ac_wide`` takes 1,111, which make 10,000 frequencies; ``scripts/bench
    --break-even min_solves_per_task`` takes fewer.
    """
    return Plan({"ac": Ac("V1", decade(1 * Hz, 1 * GHz, per_decade))})


def circuits() -> dict[str, Component]:
    """The amplifier's one circuit, by the name of its file."""
    return {"bjt_amp": build()}


def fixtures() -> dict[str, Fixture]:
    """The three fixtures of the amplifier."""
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
