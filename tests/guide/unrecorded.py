# Copyright 2026 Maxence Faldor
"""Reads a current into a resistor of a divider whose result recorded the other."""

from divider import divider

from jolty import Simulation
from jolty.plan import OperatingPoint


def main() -> None:
    """Records `R2` alone, then reads the current into `R1`'s terminal `a`."""
    point = Simulation(divider()).run(OperatingPoint(record=["R2"]))
    point.current("R1", "a")


if __name__ == "__main__":
    main()
