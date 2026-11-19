# Copyright 2026 Maxence Faldor
"""Differential test harness comparing jolty against ngspice and Xyce.

Each circuit is exported as a SPICE netlist by `jolty.spice.export` in the
ngspice or Xyce dialect. Components loaded from a SPICE library are exported as
subcircuit instances with their library included. Both engines run with tight
tolerances, and the reference simulator runs at the temperature matching
jolty's thermal voltage. Tests skip when the simulator is absent unless
``JOLTY_REQUIRE_NGSPICE=1`` or ``JOLTY_REQUIRE_XYCE=1`` is set, in which case
missing executables fail.
"""

import os
from collections.abc import Mapping
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from jolty import Simulation, ir, results
from jolty.plan import TIGHT
from jolty.spice import export, ngspice, xyce
from jolty.spice.ngspice import CURRENT_TOLERANCE, VOLTAGE_TOLERANCE

__all__ = [
    "REQUIRED",
    "SKIP",
    "SKIP_XYCE",
    "SpiceResult",
    "Values",
    "assert_ac_agrees",
    "assert_dc_sweep_agrees",
    "assert_noise_agrees",
    "assert_op_agrees",
    "assert_transient_agrees",
    "closest",
    "compare",
    "compared",
    "dc_command",
    "differing",
    "nearest",
    "netlist",
    "ngspice",
    "size",
    "transient_command",
    "unstable",
    "vector",
    "xyce",
]

REQUIRED: Final = os.environ.get("JOLTY_REQUIRE_NGSPICE") == "1"
"""Flag requiring ngspice tests to run, enabled by ``JOLTY_REQUIRE_NGSPICE=1``."""

SKIP = pytest.mark.skipif(
    ngspice.EXECUTABLE is None and not REQUIRED, reason="ngspice is not installed"
)
"""Pytest mark skipping ngspice tests unless ngspice is installed or required."""

REQUIRED_XYCE: Final = os.environ.get("JOLTY_REQUIRE_XYCE") == "1"
"""Flag requiring Xyce tests to run, enabled by ``JOLTY_REQUIRE_XYCE=1``."""

SKIP_XYCE = pytest.mark.skipif(
    xyce.EXECUTABLE is None and not REQUIRED_XYCE, reason="Xyce is not installed"
)
"""Pytest mark skipping Xyce tests unless Xyce is installed or required."""

type Real = npt.NDArray[np.float64]
type Values = float | Real | npt.NDArray[np.complex128]
type SpiceResult = ngspice.Result | xyce.Result


def _libraries(circuit: ir.Circuit) -> set[str]:
    """Returns names of SPICE-library subcircuits placed by the top component."""
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
    """Returns SPICE netlist lines for `circuit` in the requested simulator `dialect`.

    Defaults to the ngspice dialect and includes library subcircuits placed by
    the top component.
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
    """Returns the SPICE vector name for `unknown` in `dialect` (default ngspice)."""
    return export.vector(circuit, unknown, _libraries(circuit), dialect)


# Differential comparison helpers.


def _dialect(raw: SpiceResult) -> export.Dialect:
    """Returns the SPICE dialect corresponding to the simulator that produced `raw`."""
    return export.XYCE if isinstance(raw, xyce.Result) else export.NGSPICE


def _vector_name(
    simulation: Simulation,
    unknown: results.Unknown,
    dialect: export.Dialect,
    vector_names: Mapping[str, str] | None = None,
) -> str:
    """Returns the simulator vector name for `unknown`, applying explicit overrides."""
    if vector_names is not None and unknown.name in vector_names:
        return vector_names[unknown.name]
    return vector(simulation.circuit, unknown, dialect)


def dc_command(
    source: str,
    sweep: results.DcSweepResult,
    *,
    dialect: export.Dialect = export.NGSPICE,
) -> str:
    """Returns the SPICE DC sweep command for `sweep` on voltage source `source`."""
    first, last = float(sweep.swept[0]), float(sweep.swept[-1])
    step = (last - first) / (len(sweep.swept) - 1)
    prefix = ".dc" if dialect == export.XYCE else "dc"
    return f"{prefix} V{source} {first!r} {last!r} {step!r}"


def transient_command(
    stop: float,
    step: float,
    *,
    dialect: export.Dialect = export.NGSPICE,
    uic: bool = False,
) -> str:
    """Returns the SPICE transient command for `stop` and `step` in seconds."""
    prefix = ".tran" if dialect == export.XYCE else "tran"
    suffix = " uic" if uic else ""
    return f"{prefix} {step!r} {stop!r} 0 {step!r}{suffix}"


def compare(  # noqa: PLR0913 - compared values, unknown metadata, and explicit tolerances
    jolty: Values,
    spice: Values,
    unknown: results.Unknown,
    *,
    scale: float,
    tolerance: float,
    allowance: float = 0.0,
) -> None:
    """Asserts both agree within `tolerance` times `scale`, the signal's size.

    Values within the engines' absolute tolerance for the unknown's quantity are noise,
    and agree too. A current agrees within `allowance` too, when given.
    """
    current = unknown.nature == "current"
    noise = CURRENT_TOLERANCE if current else VOLTAGE_TOLERANCE
    np.testing.assert_allclose(
        jolty,
        spice,
        rtol=0,
        atol=tolerance * scale + noise + (allowance if current else 0.0),
        err_msg=unknown.name,
    )


def assert_op_agrees(
    simulation: Simulation,
    point: results.OperatingPointResult,
    raw: SpiceResult,
    *,
    tolerance: float = 1e-6,
    allowance: float = 0.0,
) -> None:
    """Asserts every shared unknown of an operating point agrees with SPICE.

    Defaults to `tolerance=1e-6`, matching both engines' `TIGHT` Newton `reltol=1e-6`.
    """
    dialect = _dialect(raw)
    use_size = dialect == export.XYCE
    for column, unknown in compared(simulation):
        spice_value = float(raw.real(vector(simulation.circuit, unknown, dialect))[0])
        compare(
            float(point.values[column]),
            spice_value,
            unknown,
            scale=size(spice_value) if use_size else abs(spice_value),
            tolerance=tolerance,
            allowance=allowance,
        )


def assert_dc_sweep_agrees(  # noqa: PLR0913 - simulation, results, and explicit overrides
    simulation: Simulation,
    sweep: results.DcSweepResult,
    raw: SpiceResult,
    *,
    tolerance: float = 1e-6,
    allowance: float = 0.0,
    vector_names: Mapping[str, str] | None = None,
) -> None:
    """Asserts every shared unknown of a DC sweep agrees with SPICE.

    Defaults to `tolerance=1e-6`, matching both engines' `TIGHT` Newton `reltol=1e-6`.
    """
    dialect = _dialect(raw)
    for column, unknown in compared(simulation):
        name = _vector_name(simulation, unknown, dialect, vector_names)
        spice_values = raw.real(name)
        compare(
            sweep.values[:, column],
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=tolerance,
            allowance=allowance,
        )


def assert_ac_agrees(  # noqa: PLR0913 - simulation, results, and frequency/scaling options
    simulation: Simulation,
    response: results.AcResult,
    raw: SpiceResult,
    *,
    tolerance: float = 1e-6,
    frequency_tolerance: float | None = None,
    scale_by_nature: bool = False,
) -> None:
    """Asserts every shared unknown of an AC response agrees with SPICE.

    Defaults to `tolerance=1e-6` (matching `TIGHT` Newton `reltol=1e-6`); when
    `frequency_tolerance` is given (typically `1e-9` for log-decade stepping rounding),
    checks the frequency grid too. When `scale_by_nature` is true, scales each unknown
    against the largest value of its nature across the circuit via `differing`.
    """
    if frequency_tolerance is not None:
        np.testing.assert_allclose(
            response.frequencies, raw.real("frequency"), rtol=frequency_tolerance
        )
    dialect = _dialect(raw)
    if scale_by_nature:
        shared = compared(simulation)
        jolty_values = {
            unknown.name: response.values[:, column] for column, unknown in shared
        }
        spice_values = {
            unknown.name: raw.vectors[vector(simulation.circuit, unknown, dialect)]
            for _, unknown in shared
        }
        natures = {unknown.name: unknown.nature for _, unknown in shared}
        assert not differing(jolty_values, spice_values, natures, tolerance), (
            "the responses differ"
        )
        return
    for column, unknown in compared(simulation):
        spice_vector = raw.vectors[vector(simulation.circuit, unknown, dialect)]
        compare(
            response.values[:, column],
            spice_vector,
            unknown,
            scale=size(spice_vector),
            tolerance=tolerance,
        )


def assert_noise_agrees(
    spectrum: results.NoiseResult,
    raw: SpiceResult,
    *,
    tolerance: float = 1e-5,
    frequency_tolerance: float | None = 1e-9,
    check_referred: bool = True,
) -> None:
    """Asserts output (and optionally input-referred) noise agrees with SPICE.

    ngspice reports root spectral densities (`onoise_spectrum`, `inoise_spectrum`),
    whereas Xyce reports squared densities (`onoise`, `inoise`). Defaults to
    `tolerance=1e-5` (~10x `TIGHT` `reltol=1e-6`, covering operating-point spread and
    CODATA `k_B` difference) and `frequency_tolerance=1e-9` (log-decade grid rounding).
    """
    if frequency_tolerance is not None:
        np.testing.assert_allclose(
            spectrum.frequencies, raw.real("frequency"), rtol=frequency_tolerance
        )
    squared = isinstance(raw, xyce.Result)
    output_key = "onoise" if squared else "onoise_spectrum"
    output_values = spectrum.densities if squared else np.sqrt(spectrum.densities)
    np.testing.assert_allclose(output_values, raw.real(output_key), rtol=tolerance)
    if check_referred:
        input_key = "inoise" if squared else "inoise_spectrum"
        input_values = spectrum.referred() if squared else np.sqrt(spectrum.referred())
        np.testing.assert_allclose(input_values, raw.real(input_key), rtol=tolerance)


def assert_transient_agrees(
    simulation: Simulation,
    waveforms: results.TransientResult,
    raw: SpiceResult,
    *,
    window: float,
    tolerance: float = 1e-3,
) -> None:
    """Asserts every shared transient waveform agrees with SPICE within `window`.

    Defaults to `tolerance=1e-3` of each waveform's swing, covering `O(h^2)` time
    integration error and `closest()` windowed linear interpolation onto SPICE's times.
    """
    dialect = _dialect(raw)
    times = raw.real("time")
    for column, unknown in compared(simulation):
        spice_values = raw.real(vector(simulation.circuit, unknown, dialect))
        jolty = closest(
            waveforms.times, waveforms.values[:, column], times, spice_values, window
        )
        compare(
            jolty,
            spice_values,
            unknown,
            scale=size(spice_values),
            tolerance=tolerance,
        )


def closest(times: Real, values: Real, at: Real, targets: Real, window: float) -> Real:
    """Returns piecewise-linear waveform values nearest `targets` within `window`.

    The waveform is linearly interpolated over `values` at `times`, and each
    target time is given in `at`. Linear interpolation cuts across corners when
    two engines choose different time steps; allowing a time shift of `window`
    aligns the waveforms at each target.
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
    """Returns the largest magnitude in `values`, or 1.0 when all entries are zero."""
    return float(np.abs(np.asarray(values)).max()) or 1.0


def compared(simulation: Simulation) -> list[tuple[int, results.Unknown]]:
    """Returns `(column, unknown)` pairs reported by both jolty and SPICE.

    Both engines report nets, top-level source currents, sensed currents, and
    device states assigned dedicated netlist nodes. ngspice names other
    internal device nodes differently and omits series sense nets that jolty
    inserts for subcircuit current sensing.
    """
    return [
        (column, unknown)
        for column, unknown in enumerate(simulation.unknowns)
        if unknown.nature == "state" or _shared(unknown.name)
    ]


def _shared(name: str) -> bool:
    """Returns whether ngspice also reports the unknown `name`."""
    *path, leaf = name.split("/")
    if ":" in leaf:
        return not path and leaf.endswith((":current", ":control"))
    return "." not in leaf


def differing(
    jolty: Mapping[str, Values],
    reference: Mapping[str, Values],
    natures: Mapping[str, str],
    tolerance: float = 1e-6,
) -> list[str]:
    """Returns names of unknowns where `jolty` disagrees with `reference`.

    Values may be scalars or sweep arrays. Each unknown agrees within
    `tolerance` (default 1e-6) of the largest reference magnitude of its
    physical nature, because a solver resolves each equation relative to its
    largest term. Values within twice the engine absolute tolerance also agree.
    """
    scales: dict[str, float] = {}
    for name, values in reference.items():
        nature = natures[name]
        scales[nature] = max(scales.get(nature, 0.0), size(values))
    return [
        name
        for name, values in reference.items()
        if np.max(np.abs(np.asarray(jolty[name]) - values))
        > tolerance * scales[natures[name]] + 2 * _noise(natures[name])
    ]


def _noise(nature: str) -> float:
    """Returns the engine absolute tolerance for an unknown of `nature`."""
    return CURRENT_TOLERANCE if nature == "current" else VOLTAGE_TOLERANCE


def nearest(times: Real, targets: Real) -> npt.NDArray[np.intp]:
    """Returns the index in `times` nearest each entry in `targets`.

    jolty treats a requested time point as reached once a time step lands
    within the minimum allowed step size.
    """
    after = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    before = after - 1
    return np.where(targets - times[before] <= times[after] - targets, before, after)


def unstable(simulation: Simulation) -> bool:
    """Returns whether any circuit mode grows away from the `TIGHT` operating point.

    Transients from an unstable operating point amplify rounding errors, so two
    engines or equivalent spellings diverge regardless of accuracy.
    """
    return any(
        diagnostic.rule == "unstable-pole"
        for diagnostic in simulation.check_operating_point(TIGHT)
    )
