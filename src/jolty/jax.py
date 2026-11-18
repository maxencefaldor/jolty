# Copyright 2026 Maxence Faldor
"""Run and differentiate circuits inside JAX (`jit`, `grad`, `vjp`, `vmap`).

[`function`][jolty.jax.function] wraps a circuit analysis as a pure JAX callable
from a nested mapping of parameter arrays, the same mapping
[`jolty.function`][jolty.simulate.function] takes, to a dictionary of output arrays.
Forward calls run inside `jax.pure_callback` (on CPU, outside XLA): unbatched calls
(`batch_size == 1`) run directly through the native simulator so they match
[`jolty.function`][jolty.simulate.function] bit for bit (preserving nested dissection),
and batched `jax.vmap` calls (`batch_size > 1`) dispatch into one multi-threaded engine
batch across all cores.
Reverse-mode differentiation (`jax.grad`, `jax.vjp`, `jax.jacrev`) registers
[`jolty.vjp`][jolty.simulate.vjp]'s exact adjoint pullback with `jax.custom_vjp`:
one transposed solve per cotangent regardless of parameter count, for operating
points, DC sweeps, small-signal AC responses, transients at fixed breakpoints, and
driven or autonomous periodic steady states, with matching matrix-ordering and thread
settings between forward and backward.

Because `jax.custom_vjp` saves only the packed parameter vector across the callback
boundary, each reverse-mode pullback re-runs the forward simulation from scratch:
`value_and_grad` and `jax.grad` pay twice the forward cost, and `k` pullback calls pay
`(k + 1)` times the forward cost (notably relevant for autonomous
`PeriodicSteadyState` shooting).
Only first-order reverse-mode differentiation is supported: forward-mode autodiff
(`jax.jvp`) and higher-order differentiation (`jax.grad(jax.grad(...))`,
`jax.hessian`) are refused.

Install the optional extra with ``pip install 'jolty[jax]'`` to use this module.

Examples:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jolty import Builder, Role, component, devices
    >>> from jolty.jax import function
    >>> from jolty.plan import OperatingPoint
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     upper = c.place(devices.resistor(1 * kohm), name="R1")
    ...     lower = c.place(devices.resistor(1 * kohm), name="R2")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, lower.a, name="out")
    ...     c.connect(lower.b, source.negative, ground)
    >>> solve = function(divider(), OperatingPoint())
    >>> round(float(solve({"R1": {"resistance": 3000.0}})["out"][0]), 6)
    1.25
    >>> gradient = jax.grad(
    ...     lambda parameters: jnp.sum((solve(parameters)["out"] - 2.0) ** 2)
    ... )({"R1": {"resistance": 1000.0}})
    >>> round(float(gradient["R1"]["resistance"]), 9)
    -0.00125
"""

import json
import math
import sys
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

try:
    import jax
    import jax.numpy as jnp
    from jax.typing import ArrayLike
except ModuleNotFoundError as error:
    root_module = (error.name or "").split(".", 1)[0]
    if root_module not in {"jax", "jaxlib"}:
        raise
    message = (
        "jolty.jax needs the optional `jax` dependency; "
        "install it with `pip install 'jolty[jax]'`"
    )
    raise ImportError(message) from error

from jolty import _core, checks, ir
from jolty.builder import Component
from jolty.errors import ElaborationError, SimulationError
from jolty.plan import (
    Ac,
    DcSweep,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    Transient,
)
from jolty.simulate import (
    PERIOD,
    Differentiable,
    Function,
    _elaborating,
    _engine,
    _points,
    _values,
)
from jolty.units import TIME

type JaxParameters = Mapping[str, Mapping[str, ArrayLike]]
"""Nested mapping of parameter scalars, by instance path then parameter path, in SI."""

type JaxOutputs = dict[str, jax.Array]
"""Output dictionary returned by a JAX-wrapped circuit function."""

type _ParameterName = tuple[str, str]
"""One device parameter as `(instance_path, parameter_path)`."""

type _MappedIndices = tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]
"""Aligned `(base_indices, simulator_indices)` for columns present in both orders."""

type _CotangentArray = npt.NDArray[np.float64] | npt.NDArray[np.complex128]
"""One sample's cotangent array in host memory."""

_DISABLE_DISSECTION: Final[int] = sys.maxsize
"""Upper bound on `max_unknowns_per_interior` matching `Batch::run` ordering."""


@dataclass(frozen=True, slots=True)
class _ValidatedShape:
    """Validated static-shape metadata for one JAX-wrapped analysis."""

    points: int
    complex_outputs: bool = False
    oscillates: bool = False
    breakpoints: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class _AnalysisPlan:
    """Pre-elaborated circuit and static shape metadata for a JAX-wrapped analysis."""

    native: Function
    base_simulator: _core.Simulator
    unknowns: tuple[str, ...]
    encoded_analysis: str
    encoded_settings: str
    points: int
    complex_outputs: bool
    oscillates: bool
    breakpoints: tuple[float, ...] | None


def _validate_breakpoints(
    breakpoints: tuple[float, ...], upper_bound: float, label: str
) -> None:
    """Checks that every breakpoint is finite and strictly inside `(0, upper_bound]`."""
    for point in breakpoints:
        if not math.isfinite(point) or not (0.0 < point <= upper_bound):
            message = f"{label} breakpoint {point} s must lie in (0, {upper_bound}] s"
            raise ValueError(message)


def _validate_transient(analysis: Transient) -> _ValidatedShape:
    """Validates stop time and breakpoints for a JAX transient."""
    stop = analysis.stop.to(TIME, "the stop time")
    if not math.isfinite(stop) or stop <= 0.0:
        message = f"transient stop time must be positive, got {stop} s"
        raise ValueError(message)
    if not analysis.breakpoints:
        message = (
            "a JAX transient requires explicit `breakpoints` so the "
            "output array has a fixed static shape"
        )
        raise ValueError(message)
    breakpoints = tuple(
        point.to(TIME, "a time to step onto") for point in analysis.breakpoints
    )
    _validate_breakpoints(breakpoints, stop, "transient")
    return _ValidatedShape(points=len(breakpoints), breakpoints=breakpoints)


def _validate_periodic(analysis: PeriodicSteadyState) -> _ValidatedShape:
    """Validates period and breakpoints for a JAX periodic steady state."""
    if analysis.period is None:
        if analysis.breakpoints:
            message = (
                "an autonomous periodic steady state has an unknown period "
                "and cannot take fixed `breakpoints`"
            )
            raise ValueError(message)
        return _ValidatedShape(points=1, oscillates=True)
    period = analysis.period.to(TIME, "the period")
    if not math.isfinite(period) or period <= 0.0:
        message = f"periodic steady-state period must be positive, got {period} s"
        raise ValueError(message)
    if not analysis.breakpoints:
        message = (
            "a driven JAX periodic steady state requires explicit `breakpoints` so the "
            "output array has a fixed static shape"
        )
        raise ValueError(message)
    breakpoints = tuple(
        point.to(TIME, "a time to step onto") for point in analysis.breakpoints
    )
    _validate_breakpoints(breakpoints, period, "periodic steady-state")
    return _ValidatedShape(points=len(breakpoints), breakpoints=breakpoints)


def _validate_analysis(analysis: Differentiable) -> _ValidatedShape:
    """Validates static output shape rules for `analysis`."""
    match analysis:
        case OperatingPoint():
            return _ValidatedShape(points=1)
        case DcSweep():
            if not analysis.values:
                message = "a JAX DC sweep requires at least one sweep level"
                raise ValueError(message)
            return _ValidatedShape(points=_points(analysis))
        case Ac():
            if not analysis.frequencies:
                message = "a JAX AC analysis requires at least one frequency"
                raise ValueError(message)
            return _ValidatedShape(points=_points(analysis), complex_outputs=True)
        case Transient():
            return _validate_transient(analysis)
        case PeriodicSteadyState():
            return _validate_periodic(analysis)
        case _:
            message = f"unsupported analysis for jolty.jax: {type(analysis).__name__}"
            raise ValueError(message)


def _build_analysis_plan(
    circuit_or_top: ir.Circuit | Component,
    analysis: Differentiable,
    options: Options | None,
) -> _AnalysisPlan:
    """Elaborates the circuit once via `Function` and prepares static JAX metadata."""
    validated = _validate_analysis(analysis)
    native = Function(circuit_or_top, analysis, options)
    base_simulator = native.elaborated.simulator()
    if base_simulator.foundry_devices:
        message = (
            "jolty.jax only runs circuits the native engine simulates, "
            "not foundry devices"
        )
        raise ValueError(message)

    return _AnalysisPlan(
        native=native,
        base_simulator=base_simulator,
        unknowns=tuple(name for name, _ in base_simulator.unknowns),
        encoded_analysis=native._encoded(),  # noqa: SLF001 - reuses Function encoding
        encoded_settings=native._settings(),  # noqa: SLF001 - reuses Function settings
        points=validated.points,
        complex_outputs=validated.complex_outputs,
        oscillates=native._oscillates(),  # noqa: SLF001 - reuses Function check
        breakpoints=validated.breakpoints,
    )


def _batch_settings(plan: _AnalysisPlan, batch_size: int, threads: int) -> str:
    """Returns engine settings matching `Batch::run` matrix ordering and threading."""
    payload = plan.native.options.to_json()
    raw_parallelism = payload.get("parallelism")
    parallelism: dict[str, JsonValue] = (
        dict(raw_parallelism) if isinstance(raw_parallelism, dict) else {}
    )
    if batch_size >= threads:
        parallelism["max_threads"] = 1
    if parallelism.get("max_unknowns_per_interior") is None:
        parallelism["max_unknowns_per_interior"] = _DISABLE_DISSECTION
    payload["parallelism"] = parallelism
    return json.dumps(payload, allow_nan=False)


def _validate_leaf(
    instance: str, parameter_name: object, value: ArrayLike
) -> tuple[str, jax.Array]:
    """Validates one parameter key and scalar real numeric leaf."""
    if not isinstance(parameter_name, str):
        message = (
            f"parameter name under {instance!r} must be a string, "
            f"got {type(parameter_name).__name__}"
        )
        raise TypeError(message)
    if isinstance(value, bool):
        message = (
            f"parameter `{instance}.{parameter_name}` must be a real number, not bool"
        )
        raise TypeError(message)
    leaf = jnp.asarray(value)
    if not (
        jnp.issubdtype(leaf.dtype, jnp.floating)
        or jnp.issubdtype(leaf.dtype, jnp.integer)
    ):
        message = (
            f"parameter `{instance}.{parameter_name}` must have a real numeric "
            f"dtype, got {leaf.dtype}"
        )
        raise TypeError(message)
    if leaf.ndim != 0:
        message = (
            f"parameter `{instance}.{parameter_name}` must be a scalar, "
            f"got shape {leaf.shape}"
        )
        raise ValueError(message)
    return parameter_name, leaf


def _flatten_parameters(
    parameters: JaxParameters,
) -> tuple[tuple[str, ...], tuple[_ParameterName, ...], list[jax.Array]]:
    """Extracts outer instances, parameter names, and scalar leaves in sorted order.

    Raises:
        TypeError: When `parameters` is not a two-level mapping of strings to real
            numeric scalars (rejecting `bool` and `complex` leaves).
        ValueError: When any parameter leaf is not a scalar array.
    """
    if not isinstance(parameters, Mapping):
        message = (
            "parameters must be a mapping of instance path to parameter mapping, "
            f"got {type(parameters).__name__}"
        )
        raise TypeError(message)

    validated: dict[str, dict[str, jax.Array]] = {}
    for instance, named in parameters.items():
        if not isinstance(instance, str) or not isinstance(named, Mapping):
            message = (
                "parameters must be a mapping of instance path to parameter mapping"
            )
            raise TypeError(message)
        validated[instance] = dict(
            _validate_leaf(instance, parameter_name, value)
            for parameter_name, value in named.items()
        )

    outer_instances = tuple(sorted(validated))
    names: list[_ParameterName] = []
    leaves: list[jax.Array] = []
    for instance in outer_instances:
        inner_mapping = validated[instance]
        for parameter_name in sorted(inner_mapping):
            names.append((instance, parameter_name))
            leaves.append(inner_mapping[parameter_name])
    return outer_instances, tuple(names), leaves


def _validate_parameter_names(
    plan: _AnalysisPlan,
    outer_instances: tuple[str, ...],
    names: tuple[_ParameterName, ...],
) -> None:
    """Checks at trace time that every instance and parameter exists in the circuit.

    Raises:
        ElaborationError: Naming every instance or parameter the circuit does not write.
    """
    if not outer_instances:
        return
    placeholder: dict[str, dict[str, float]] = {
        instance: {} for instance in outer_instances
    }
    for instance, parameter in names:
        placeholder[instance][parameter] = 0.0
    elaborated = plan.native.elaborated
    with _elaborating():
        elaborated._core.values(  # noqa: SLF001 - trace-time schema check without rebuild
            json.dumps(placeholder, allow_nan=False),
            elaborated._kelvins,  # noqa: SLF001 - matches Elaborated temperature
        )


def _build_overrides(
    names: tuple[_ParameterName, ...], row: npt.NDArray[np.float64]
) -> dict[str, dict[str, float]]:
    """Builds the nested overrides dict for one parameter row."""
    overrides: dict[str, dict[str, float]] = {}
    for index, (instance, parameter) in enumerate(names):
        overrides.setdefault(instance, {})[parameter] = float(row[index])
    return overrides


def _raise_batch_failure(found: _core.BatchResult, batch_size: int) -> None:
    """Raises the first elaboration or simulation failure reported by `core.batch`."""
    if found["problems"]:
        first_index = min(found["problems"])
        problems = found["problems"][first_index]
        raise ElaborationError([checks.problem(each) for each in problems])
    if found["errors"]:
        first_index = min(found["errors"])
        _, detail = found["errors"][first_index][0]
        if batch_size > 1:
            detail = f"batch run {first_index}: {detail}"
        raise SimulationError(detail)


def _outcome_times(outcome: _core.AnalysisResult) -> npt.NDArray[np.float64]:
    """Returns accepted solver time points from a transient or periodic outcome."""
    match outcome["kind"]:
        case "transient" | "periodic_steady_state":
            return outcome["times"]
        case other:
            message = f"expected transient or periodic steady-state result, got {other}"
            raise SimulationError(message)


def _mapped_columns(
    plan: _AnalysisPlan, simulator: _core.Simulator
) -> tuple[list[int | None], _MappedIndices]:
    """Returns per-base-column mapping and vectorized aligned index arrays."""
    simulator_unknowns = {
        name: column for column, (name, _) in enumerate(simulator.unknowns)
    }
    column_map = [simulator_unknowns.get(name) for name in plan.unknowns]
    pairs = [
        (base_column, simulator_column)
        for base_column, simulator_column in enumerate(column_map)
        if simulator_column is not None
    ]
    if pairs:
        base_indices = np.fromiter(
            (pair[0] for pair in pairs), dtype=np.intp, count=len(pairs)
        )
        simulator_indices = np.fromiter(
            (pair[1] for pair in pairs), dtype=np.intp, count=len(pairs)
        )
    else:
        base_indices = np.empty(0, dtype=np.intp)
        simulator_indices = np.empty(0, dtype=np.intp)
    return column_map, (base_indices, simulator_indices)


def _sample_at_breakpoints(
    times: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    breakpoints: tuple[float, ...],
    indices: _MappedIndices,
    width: int,
) -> npt.NDArray[np.float64]:
    """Samples waveform `values` at `breakpoints` via linear interpolation.

    Matches `crates/jolty-python/src/batch.rs::at` bit for bit.
    """
    base_indices, simulator_indices = indices
    sampled = np.full((len(breakpoints), width), np.nan, dtype=np.float64)
    for breakpoint_index, target_time in enumerate(breakpoints):
        after = int(np.searchsorted(times, target_time))
        if after < len(times) and times[after] == target_time:
            sampled[breakpoint_index, base_indices] = values[after, simulator_indices]
        elif 0 < after < len(times):
            before = after - 1
            fraction = (target_time - float(times[before])) / (
                float(times[after]) - float(times[before])
            )
            low = values[before, simulator_indices]
            high = values[after, simulator_indices]
            sampled[breakpoint_index, base_indices] = low + (high - low) * fraction
        else:
            message = (
                f"breakpoint {target_time} s lies outside solved time range "
                f"[{float(times[0])}, {float(times[-1])}] s"
            )
            raise SimulationError(message)
    return sampled


def _forward_single(
    plan: _AnalysisPlan,
    names: tuple[_ParameterName, ...],
    row: npt.NDArray[np.float64],
    batch_shape: tuple[int, ...],
    output_dtype: np.dtype,
) -> npt.NDArray[np.floating | np.complexfloating]:
    """Runs one unbatched sample directly via `simulator.run` matching `Function`."""
    overrides = _build_overrides(names, row)
    simulator = plan.native.elaborated.simulator(overrides)
    try:
        outcome = simulator.run(plan.encoded_analysis, plan.encoded_settings, _engine())
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None

    if plan.oscillates:
        _, period = _values(outcome)
        if period is None:
            message = "periodic steady state returned no period"
            raise SimulationError(message)
        return np.full(batch_shape, period, dtype=output_dtype)

    values, _ = _values(outcome)
    _, indices = _mapped_columns(plan, simulator)
    base_indices, simulator_indices = indices
    width = len(plan.unknowns)

    if plan.breakpoints is not None:
        times = _outcome_times(outcome)
        sampled = _sample_at_breakpoints(
            times,
            np.asarray(values, dtype=np.float64),
            plan.breakpoints,
            indices,
            width,
        )
        return sampled.astype(output_dtype, copy=False).reshape(
            (*batch_shape, plan.points, width)
        )

    if plan.complex_outputs:
        packed_complex = np.full(
            (plan.points, width), complex(np.nan, np.nan), dtype=np.complex128
        )
        packed_complex[:, base_indices] = np.asarray(values, dtype=np.complex128)[
            :, simulator_indices
        ]
        return packed_complex.astype(output_dtype, copy=False).reshape(
            (*batch_shape, plan.points, width)
        )

    packed_real = np.full((plan.points, width), np.nan, dtype=np.float64)
    packed_real[:, base_indices] = np.asarray(values, dtype=np.float64)[
        :, simulator_indices
    ]
    return packed_real.astype(output_dtype, copy=False).reshape(
        (*batch_shape, plan.points, width)
    )


def _forward_batch(
    plan: _AnalysisPlan,
    theta_host: npt.NDArray[np.floating],
    names: tuple[_ParameterName, ...],
    output_dtype: np.dtype,
) -> npt.NDArray[np.floating | np.complexfloating]:
    """Runs zero, one, or many parameter rows via the Rust engine."""
    raw = np.asarray(theta_host, dtype=np.float64)
    batch_shape = raw.shape[:-1]
    batch_size = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1
    if batch_size == 0:
        empty_shape = (
            batch_shape
            if plan.oscillates
            else (*batch_shape, plan.points, len(plan.unknowns))
        )
        return np.empty(empty_shape, dtype=output_dtype)

    rows = raw.reshape(batch_size, len(names))
    if batch_size == 1:
        return _forward_single(plan, names, rows[0], batch_shape, output_dtype)

    explicit_runs = [
        (json.dumps(_build_overrides(names, rows[index]), allow_nan=False), None)
        for index in range(batch_size)
    ]
    with _elaborating():
        variants = _core.Variants.explicit(
            plan.base_simulator.elaborated, explicit_runs
        )

    analyses = [(plan.encoded_analysis, plan.encoded_settings)]
    engine_pool = _engine()

    if plan.oscillates:
        periods = np.full(batch_size, np.nan, dtype=np.float64)

        def collect_period(
            index: int,
            _structure: _core.Simulator | None,
            _problems: list[_core.Problem] | None,
            outcomes: list[_core.AnalysisResult | str],
        ) -> None:
            if not outcomes:
                return
            first = outcomes[0]
            if isinstance(first, dict) and first["kind"] == "periodic_steady_state":
                periods[index] = float(first["period"])

        found = plan.base_simulator.batch(
            analyses,
            variants,
            (0, batch_size),
            [],
            [],
            engine_pool,
            collect_period,
        )
        _raise_batch_failure(found, batch_size)
        return periods.reshape(batch_shape).astype(output_dtype, copy=False)

    traced_columns = list(range(len(plan.unknowns)))
    found = plan.base_simulator.batch(
        analyses,
        variants,
        (0, batch_size),
        [],
        [(0, traced_columns)],
        engine_pool,
        None,
    )
    _raise_batch_failure(found, batch_size)
    packed = found["traces"][0]
    target_shape = (*batch_shape, plan.points, len(plan.unknowns))
    return np.asarray(packed, dtype=output_dtype).reshape(target_shape)


def _check_dropped_unknowns(
    plan: _AnalysisPlan,
    column_map: list[int | None],
    cotangent_sample: _CotangentArray,
) -> None:
    """Raises `SimulationError` if a non-zero or NaN cotangent hits a dropped column."""
    for base_column, simulator_column in enumerate(column_map):
        if simulator_column is None and np.any(cotangent_sample[:, base_column]):
            name = plan.unknowns[base_column]
            message = (
                f"unknown `{name}` disappeared under structural parameter change and "
                "cannot receive a non-zero cotangent"
            )
            raise SimulationError(message)


def _scatter_breakpoint_cotangents(
    times: npt.NDArray[np.float64],
    breakpoints: tuple[float, ...],
    cotangent_at_breakpoints: npt.NDArray[np.float64],
    indices: _MappedIndices,
    full_cotangent: npt.NDArray[np.float64],
) -> None:
    """Transpose of linear interpolation at `breakpoints` onto solver `times`."""
    base_indices, simulator_indices = indices
    for breakpoint_index, target_time in enumerate(breakpoints):
        after = int(np.searchsorted(times, target_time))
        row_values = cotangent_at_breakpoints[breakpoint_index, base_indices]
        if after < len(times) and times[after] == target_time:
            full_cotangent[after, simulator_indices] += row_values
        elif 0 < after < len(times):
            before = after - 1
            fraction = (target_time - float(times[before])) / (
                float(times[after]) - float(times[before])
            )
            full_cotangent[before, simulator_indices] += (1.0 - fraction) * row_values
            full_cotangent[after, simulator_indices] += fraction * row_values
        else:
            message = (
                f"breakpoint {target_time} s lies outside solved time range "
                f"[{float(times[0])}, {float(times[-1])}] s"
            )
            raise SimulationError(message)


def _build_full_cotangent(
    plan: _AnalysisPlan,
    outcome: _core.AnalysisResult,
    simulator: _core.Simulator,
    cotangent_sample: _CotangentArray,
) -> tuple[_CotangentArray, float]:
    """Maps a fixed-shape JAX cotangent into the simulator's full solver cotangent."""
    simulator_width = len(simulator.unknowns)
    if plan.oscillates:
        simulator_values, _ = _values(outcome)
        zeros = np.zeros((simulator_values.shape[0], simulator_width), dtype=np.float64)
        return zeros, float(np.asarray(cotangent_sample))

    column_map, indices = _mapped_columns(plan, simulator)
    base_indices, simulator_indices = indices

    if plan.breakpoints is not None:
        real_sample = np.asarray(cotangent_sample, dtype=np.float64)
        _check_dropped_unknowns(plan, column_map, real_sample)
        times = _outcome_times(outcome)
        full_real = np.zeros((len(times), simulator_width), dtype=np.float64)
        _scatter_breakpoint_cotangents(
            times, plan.breakpoints, real_sample, indices, full_real
        )
        return full_real, 0.0

    if plan.complex_outputs:
        # JAX custom_vjp already passes the conjugated Wirtinger cotangent
        # `dL/da - j dL/db`, matching jolty.vjp's bilinear transpose directly.
        complex_sample = np.asarray(cotangent_sample, dtype=np.complex128)
        _check_dropped_unknowns(plan, column_map, complex_sample)
        full_complex = np.zeros((plan.points, simulator_width), dtype=np.complex128)
        full_complex[:, simulator_indices] = complex_sample[:, base_indices]
        return full_complex, 0.0

    real_sample = np.asarray(cotangent_sample, dtype=np.float64)
    _check_dropped_unknowns(plan, column_map, real_sample)
    full_real = np.zeros((plan.points, simulator_width), dtype=np.float64)
    full_real[:, simulator_indices] = real_sample[:, base_indices]
    return full_real, 0.0


def _pull_one_sample(
    plan: _AnalysisPlan,
    names: tuple[_ParameterName, ...],
    sample: tuple[npt.NDArray[np.float64], _CotangentArray],
    settings: str,
    engine_pool: _core.Pool,
) -> npt.NDArray[np.float64]:
    """Pulls one sample's cotangent back to parameter gradients using `core.vjp`."""
    parameter_row, cotangent_sample = sample
    if not names or not np.any(cotangent_sample):
        return np.zeros(len(names), dtype=np.float64)

    overrides = _build_overrides(names, parameter_row)
    simulator = plan.native.elaborated.simulator(overrides)
    name_list = list(names)

    try:
        outcome, pullback = simulator.vjp(
            plan.encoded_analysis, name_list, settings, engine_pool
        )
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None

    full_cotangent, period_cotangent = _build_full_cotangent(
        plan, outcome, simulator, cotangent_sample
    )

    try:
        pulled = simulator.pull_back(
            pullback,
            full_cotangent,
            period_cotangent,
            settings,
            engine_pool,
        )
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None

    if plan.complex_outputs:
        return np.real(pulled).astype(np.float64, copy=False)
    return np.asarray(pulled, dtype=np.float64)


def _backward_batch(
    plan: _AnalysisPlan,
    theta_host: npt.NDArray[np.floating],
    cotangent_host: npt.NDArray[np.floating | np.complexfloating],
    names: tuple[_ParameterName, ...],
    real_dtype: np.dtype,
) -> npt.NDArray[np.floating]:
    """Pulls zero, one, or many cotangents back across cores (GIL released in Rust)."""
    raw_theta = np.asarray(theta_host, dtype=np.float64)
    batch_shape = raw_theta.shape[:-1]
    batch_size = int(np.prod(batch_shape, dtype=np.int64)) if batch_shape else 1
    parameter_count = len(names)
    if parameter_count == 0 or batch_size == 0:
        return np.empty((*batch_shape, parameter_count), dtype=real_dtype)

    theta_rows = raw_theta.reshape(batch_size, parameter_count)
    cotangent_dtype = np.complex128 if plan.complex_outputs else np.float64
    raw_cotangent = np.asarray(cotangent_host, dtype=cotangent_dtype)
    cotangent_rows = (
        raw_cotangent.reshape(batch_size)
        if plan.oscillates
        else raw_cotangent.reshape(batch_size, plan.points, len(plan.unknowns))
    )

    engine_pool = _engine()
    if batch_size == 1:
        gradient = _pull_one_sample(
            plan,
            names,
            (theta_rows[0], cotangent_rows[0]),
            plan.encoded_settings,
            engine_pool,
        )
        return gradient.reshape((*batch_shape, parameter_count)).astype(
            real_dtype, copy=False
        )

    settings = _batch_settings(plan, batch_size, engine_pool.threads)
    worker_count = min(batch_size, max(1, engine_pool.threads))

    def pull_indexed(index: int) -> npt.NDArray[np.float64]:
        try:
            return _pull_one_sample(
                plan,
                names,
                (theta_rows[index], cotangent_rows[index]),
                settings,
                engine_pool,
            )
        except SimulationError as error:
            message = f"batch run {index}: {error}"
            raise SimulationError(message) from None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        gradients = list(executor.map(pull_indexed, range(batch_size)))
    stacked = np.stack(gradients, axis=0)
    return stacked.reshape((*batch_shape, parameter_count)).astype(
        real_dtype, copy=False
    )


def _make_packed_runner(
    plan: _AnalysisPlan,
    names: tuple[_ParameterName, ...],
    real_dtype: np.dtype,
    output_dtype: np.dtype,
) -> Callable[[jax.Array], jax.Array]:
    """Builds one `custom_vjp` packed runner cached per `(names, dtypes)` key."""
    output_shape = () if plan.oscillates else (plan.points, len(plan.unknowns))
    output_spec = jax.ShapeDtypeStruct(output_shape, output_dtype)
    gradient_spec = jax.ShapeDtypeStruct((len(names),), real_dtype)

    def forward_callback(theta_host: npt.NDArray[np.floating]) -> object:
        return _forward_batch(plan, theta_host, names, output_dtype)

    def backward_callback(
        theta_host: npt.NDArray[np.floating],
        cotangent_host: npt.NDArray[np.floating | np.complexfloating],
    ) -> object:
        return _backward_batch(plan, theta_host, cotangent_host, names, real_dtype)

    @jax.custom_vjp
    def run_packed(theta_vector: jax.Array) -> jax.Array:
        return jnp.asarray(
            jax.pure_callback(
                forward_callback,
                output_spec,
                theta_vector,
                vmap_method="broadcast_all",
            )
        )

    def run_packed_forward(theta_vector: jax.Array) -> tuple[jax.Array, jax.Array]:
        return run_packed(theta_vector), theta_vector

    def run_packed_backward(
        theta_vector: jax.Array, cotangent: jax.Array
    ) -> tuple[jax.Array]:
        gradient_vector = jnp.asarray(
            jax.pure_callback(
                backward_callback,
                gradient_spec,
                theta_vector,
                cotangent,
                vmap_method="broadcast_all",
            )
        )
        return (gradient_vector,)

    run_packed.defvjp(run_packed_forward, run_packed_backward)

    def call_packed(theta_vector: jax.Array) -> jax.Array:
        return run_packed(theta_vector)

    return call_packed


def function(
    circuit_or_top: ir.Circuit | Component,
    analysis: Differentiable,
    options: Options | None = None,
) -> Callable[[JaxParameters], JaxOutputs]:
    """Returns a pure JAX callable running and differentiating `analysis`.

    The returned callable accepts a nested mapping ``{instance: {parameter: value}}``
    of scalar JAX arrays or Python numbers (matching
    [`jolty.function`][jolty.simulate.function]), and returns a dictionary mapping
    each unknown's name to a JAX array of fixed shape:

    - `OperatingPoint`: ``(1,)`` real array per unknown.
    - `DcSweep`: ``(len(values),)`` real array per unknown.
    - `Ac`: ``(len(frequencies),)`` complex array per unknown.
    - `Transient`: ``(len(breakpoints),)`` real array per unknown (`breakpoints` must
      be non-empty and inside `(0, stop]`).
    - `PeriodicSteadyState` with `period` set: ``(len(breakpoints),)`` real array per
      unknown (`breakpoints` must be non-empty and inside `(0, period]`).
    - `PeriodicSteadyState` with `period=None` (autonomous oscillator): returns **only**
      ``{PERIOD: scalar}`` where `PERIOD` is [`PERIOD`][jolty.simulate.PERIOD] (no
      waveform arrays, unlike [`jolty.function`][jolty.simulate.function]).

    Supports `jax.jit`, `jax.grad`, `jax.value_and_grad`, `jax.vjp`, `jax.jacrev`,
    `jax.remat`, and `jax.vmap` (including empty batches, nested `vmap`, and mixed
    mapped/unmapped `in_axes`). Respects `jax_enable_x64` via
    `jax.dtypes.canonicalize_dtype` (`float64`/`complex128` when enabled,
    `float32`/`complex64` otherwise).

    Unbatched calls (`batch_size == 1`) run directly through `simulator.run` and
    `simulator.vjp` for bit-for-bit parity with `jolty.function`, preserving nested
    dissection. Batched calls (`batch_size > 1`) run through one multi-threaded Rust
    `Simulator.batch` forward and matching `max_unknowns_per_interior` and
    `max_threads` settings backward.

    If a parameter shift changes circuit structure and removes an unknown present in
    the nominal circuit (for example crossing a resistor from `< 1 Ω` to `>= 1 Ω`),
    forward outputs place `NaN` (`NaN + NaNj` for AC) in that unknown's column, and
    the backward pass raises `SimulationError` if any non-zero or `NaN` cotangent
    targets the dropped column.

    Because the custom VJP saves only the parameter vector across the callback
    boundary, each reverse-mode pullback re-runs the forward simulation from scratch
    (twice the forward cost for one pullback, `(k + 1)` times for `k` pullbacks).
    Forward-mode autodiff (`jax.jvp`) and higher-order differentiation
    (`jax.grad(jax.grad(...))`, `jax.hessian`) are refused.

    Runtime exceptions raised inside the native simulation callback
    (`ElaborationError`, `SimulationError`, or `ValueError` for non-finite values) are
    wrapped by XLA's `jax.pure_callback` host runtime (raising
    `jax.errors.JaxRuntimeError` or `ValueError` carrying the original exception name
    and message).

    Raises:
        ElaborationError: When the circuit does not elaborate.
        ValueError: When the circuit contains foundry devices or `analysis` does not
            define a valid fixed static output shape.
    """
    plan = _build_analysis_plan(circuit_or_top, analysis, options)
    cache_lock = threading.Lock()
    runner_cache: dict[
        tuple[tuple[str, ...], tuple[_ParameterName, ...], np.dtype, np.dtype],
        Callable[[jax.Array], jax.Array],
    ] = {}

    def evaluate_circuit(parameters: JaxParameters) -> JaxOutputs:
        outer_instances, names, leaves = _flatten_parameters(parameters)
        real_dtype = np.dtype(jax.dtypes.canonicalize_dtype(np.float64))
        output_dtype = (
            np.dtype(jax.dtypes.canonicalize_dtype(np.complex128))
            if plan.complex_outputs
            else real_dtype
        )
        cache_key = (outer_instances, names, real_dtype, output_dtype)
        with cache_lock:
            run_packed = runner_cache.get(cache_key)
        if run_packed is None:
            _validate_parameter_names(plan, outer_instances, names)
            candidate = _make_packed_runner(plan, names, real_dtype, output_dtype)
            with cache_lock:
                run_packed = runner_cache.setdefault(cache_key, candidate)

        theta = (
            jnp.stack([leaf.astype(real_dtype) for leaf in leaves])
            if leaves
            else jnp.empty((0,), dtype=real_dtype)
        )
        packed = run_packed(theta)
        if plan.oscillates:
            return {PERIOD: packed}
        return {name: packed[:, column] for column, name in enumerate(plan.unknowns)}

    return evaluate_circuit


__all__ = [
    "PERIOD",
    "JaxOutputs",
    "JaxParameters",
    "function",
]
