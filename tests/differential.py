# Copyright 2026 Maxence Faldor
"""Running ngspice or Xyce on jolty's circuits and comparing what the engines find.

A circuit is written as a SPICE netlist by `jolty.spice.export`, in ngspice's dialect or
Xyce's; a component read from a SPICE library is written as an instance of its
subcircuit, the library included. The engines run with tight tolerances, the simulator
at the temperature where its thermal voltage is jolty's. The tests skip where the
simulator is not installed, unless the environment sets ``JOLTY_REQUIRE_NGSPICE=1``, as
CI does, or ``JOLTY_REQUIRE_XYCE=1``, when they fail.
"""

import os
from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from jolty import Options, Simulation, ir, results
from jolty.plan import TIGHT
from jolty.spice import export, ngspice, xyce
from jolty.spice.ngspice import CURRENT_TOLERANCE, VOLTAGE_TOLERANCE

__all__ = [
    "REQUIRED",
    "SKIP",
    "SKIP_XYCE",
    "Values",
    "closest",
    "compare",
    "compared",
    "differing",
    "nearest",
    "netlist",
    "ngspice",
    "size",
    "unstable",
    "vector",
    "xyce",
]

REQUIRED: Final = os.environ.get("JOLTY_REQUIRE_NGSPICE") == "1"
"""Whether ngspice must run, as CI says by setting ``JOLTY_REQUIRE_NGSPICE=1``."""

SKIP = pytest.mark.skipif(
    ngspice.EXECUTABLE is None and not REQUIRED, reason="ngspice is not installed"
)
"""Marks tests that need ngspice: skipped without it, unless it is required, when they
run and fail."""

REQUIRED_XYCE: Final = os.environ.get("JOLTY_REQUIRE_XYCE") == "1"
"""Whether Xyce must run, as ``JOLTY_REQUIRE_XYCE=1`` in the environment says."""

SKIP_XYCE = pytest.mark.skipif(
    xyce.EXECUTABLE is None and not REQUIRED_XYCE, reason="Xyce is not installed"
)
"""Marks tests that need Xyce: skipped without it, unless it is required, when they
run and fail."""

type Real = npt.NDArray[np.float64]
type Values = float | Real | npt.NDArray[np.complex128]


def _libraries(circuit: ir.Circuit) -> set[str]:
    """The components the top places, read from SPICE libraries in these tests."""
    return {
        instance.of.component
        for instance in circuit.components[circuit.top].instances
        if isinstance(instance.of, ir.ComponentOf)
    }


def netlist(
    circuit: ir.Circuit,
    *,
    excited: str | None = None,
    transient: bool = False,
    dialect: export.Dialect = export.NGSPICE,
) -> list[str]:
    """The SPICE lines of a circuit whose top places devices and library subcircuits.

    They are in the `dialect` of a simulator, ngspice's unless given.
    """
    return export.netlist(
        circuit,
        excited=excited,
        subcircuits=_libraries(circuit),
        transient=transient,
        dialect=dialect,
    )


def vector(
    circuit: ir.Circuit,
    unknown: results.Unknown,
    dialect: export.Dialect = export.NGSPICE,
) -> str:
    """A simulator's vector of an unknown of such a circuit, by default ngspice's."""
    return export.vector(circuit, unknown, _libraries(circuit), dialect)


# The tests.


def compare(
    jolty: Values,
    spice: Values,
    unknown: results.Unknown,
    *,
    scale: float,
    tolerance: float,
) -> None:
    """Asserts both agree within `tolerance` times `scale`, the signal's size.

    Values within the engines' absolute tolerance for the unknown's quantity are noise,
    and agree too.
    """
    noise = CURRENT_TOLERANCE if unknown.quantity == "current" else VOLTAGE_TOLERANCE
    np.testing.assert_allclose(jolty, spice, rtol=0, atol=tolerance * scale + noise)


def closest(times: Real, values: Real, at: Real, targets: Real, window: float) -> Real:
    """The values of a waveform nearest `targets`, each within `window` of its time.

    The waveform is linear between `values` at `times`, and each target's time is in
    `at`. Two engines step through time differently, and a waveform interpolated
    between one's steps cuts the other's corners and curves; within a window of a step,
    the waveforms meet there.
    """
    starts, stops = at - window, at + window
    ends = np.stack([np.interp(starts, times, values), np.interp(stops, times, values)])
    lower, upper = ends.min(axis=0), ends.max(axis=0)
    firsts = np.searchsorted(times, starts, side="right")
    lasts = np.searchsorted(times, stops, side="left")
    for index, (first, last) in enumerate(zip(firsts, lasts, strict=True)):
        if first < last:
            inside = values[first:last]
            lower[index] = min(lower[index], inside.min())
            upper[index] = max(upper[index], inside.max())
    return np.clip(targets, lower, upper)


def size(values: Values) -> float:
    """The largest magnitude among `values`, or 1 when they are all zero."""
    return float(np.abs(np.asarray(values)).max()) or 1.0


def compared(simulation: Simulation) -> list[tuple[int, results.Unknown]]:
    """The unknowns both engines report, with their columns in jolty's results.

    Both report the nets, the top's source currents, the currents sensed and devices'
    states, which the netlist gives nodes of their own. ngspice names a device's other
    internal nodes its own way, and has none of the nets jolty adds in series with a
    source whose current a subcircuit reads.
    """
    return [
        (column, unknown)
        for column, unknown in enumerate(simulation.unknowns)
        if unknown.quantity == "state" or _shared(unknown.name)
    ]


def _shared(name: str) -> bool:
    """Whether ngspice reports the unknown `name` too."""
    *path, leaf = name.split("/")
    if ":" in leaf:
        return not path and leaf.endswith((":current", ":control"))
    return "." not in leaf


def differing(
    jolty: Mapping[str, Values],
    reference: Mapping[str, Values],
    quantities: Mapping[str, str],
    tolerance: float = 1e-6,
) -> list[str]:
    """The unknowns, by name, whose values jolty and a reference disagree on.

    The values are at a point or over a sweep. Each agrees within `tolerance`, a
    millionth by default, of the largest value of its quantity anywhere in the
    reference: an engine resolves an equation to a millionth of its largest term, so a
    small current beside large ones only that far. Values within twice the engines'
    absolute tolerance of each other agree too.
    """
    scales: dict[str, float] = {}
    for name, values in reference.items():
        quantity = quantities[name]
        scales[quantity] = max(scales.get(quantity, 0.0), size(values))
    return [
        name
        for name, values in reference.items()
        if np.max(np.abs(np.asarray(jolty[name]) - values))
        > tolerance * scales[quantities[name]] + 2 * _noise(quantities[name])
    ]


def _noise(quantity: str) -> float:
    """The engines' absolute tolerance on an unknown of `quantity`."""
    return CURRENT_TOLERANCE if quantity == "current" else VOLTAGE_TOLERANCE


def nearest(times: Real, targets: Real) -> npt.NDArray[np.intp]:
    """The index of the time nearest each target.

    jolty counts a time it was asked to step onto as reached once a step ends within
    the shortest step allowed of it.
    """
    after = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    before = after - 1
    return np.where(targets - times[before] <= times[after] - targets, before, after)


def unstable(simulation: Simulation) -> bool:
    """Whether a mode of the circuit grows away from its operating point, under `TIGHT`.

    A transient from there amplifies rounding, so two simulations of it, by two engines
    or of two ways of writing it, part however accurate each is.
    """
    return any(
        diagnostic.rule == "unstable-pole"
        for diagnostic in simulation.check_operating_point(TIGHT)
    )
