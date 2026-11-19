# Copyright 2026 Maxence Faldor
"""Engine runtime helpers, analysis encoding, result decoding, and shapes."""

import contextlib
import contextvars
import functools
import json
import math
import time
from collections.abc import Collection, Generator, Iterable, Mapping
from typing import Final

import numpy as np
from pydantic import Field, JsonValue

from jolty import _core, checks, ir, results, variation
from jolty.devices import Device
from jolty.errors import ElaborationError, JoltyError, SimulationError
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    Mismatch,
    Noise,
    OperatingPoint,
    OperatingPoints,
    PeriodicSteadyState,
    PoleZero,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
)
from jolty.settings import Settings
from jolty.units import (
    CURRENT,
    DIMENSIONLESS,
    TEMPERATURE,
    VOLTAGE,
    Dimension,
    Quantity,
)


def kelvins(temperature: Quantity | None) -> float:
    """Converts `temperature` to kelvins, defaulting to 27 °C when None.

    Raises:
        DimensionError: When `temperature` does not carry temperature dimensions.
    """
    if temperature is None:
        return _core.REFERENCE_TEMPERATURE
    return temperature.to(TEMPERATURE, "a temperature")


@functools.cache
def pool(count: int | None = None) -> _core.Pool:
    """Returns a cached engine thread pool of `count` threads, or one per CPU core.

    Pools are cached once per process so concurrent Python threads share worker
    threads instead of spawning duplicate pools.

    Raises:
        ValueError: When `count` is not positive.
    """
    if count is not None and count < 1:
        message = f"a pool holds a thread or more, not {count}"
        raise ValueError(message)
    return _core.Pool(count)


_CHOSEN: Final = contextvars.ContextVar[_core.Pool | None]("pool", default=None)
"""Thread pool bound to the current execution context by `threads`, if any."""


@contextlib.contextmanager
def threads(count: int | None) -> Generator[None]:
    """Runs the engine on `count` threads within the block, or on one per processor.

    Outside any `threads` context, the engine uses the configured ``threads`` setting.
    Numerical results are independent of thread count.

    Raises:
        ValueError: When `count` is not positive.
    """
    token = _CHOSEN.set(pool(count))
    try:
        yield
    finally:
        _CHOSEN.reset(token)


@functools.cache
def _settings() -> Settings:
    """Loads and caches process-wide settings.

    Raises:
        JoltyError: When settings validation fails.
    """
    return Settings.load()


def engine_pool() -> _core.Pool:
    """Returns the active engine thread pool from context or configured settings.

    Raises:
        JoltyError: When settings validation fails.
    """
    return _CHOSEN.get() or pool(_settings().threads)


class Logged(ir.FrozenModel):
    """Log record appended to ``analysis_log`` when an analysis starts.

    Every single or batched analysis appends one record upon starting, whether
    invoked from the CLI, MCP server, or Python API.
    """

    analysis: str = Field(
        description="Analysis class name, such as Transient or OperatingPoint."
    )
    runs: int = Field(
        ge=1,
        description="Number of runs started: 1 for a standalone analysis or the "
        "batch size for a batch.",
    )
    time: float = Field(description="Start timestamp in seconds since the Unix epoch.")


def log_analyses(analyses: Iterable[Analysis], runs: int) -> None:
    """Appends start records for `analyses` across `runs` to the configured log.

    Raises:
        JoltyError: When settings are invalid or the log file cannot be written.
    """
    log = _settings().analysis_log
    if log is None or runs == 0:
        return
    started = time.time()
    logged = [
        Logged(analysis=type(each).__name__, runs=runs, time=started)
        for each in analyses
    ]
    lines = "".join(f"{each.model_dump_json()}\n" for each in logged)
    try:
        with log.open("a", encoding="utf-8") as file:
            file.write(lines)
    except OSError as error:
        message = f"cannot append to the analysis log {log}: {error.strerror}"
        raise JoltyError(message) from None


def substitutes_json(substitutes: Mapping[str, Device] | None) -> str:
    """Serializes `substitutes` device models keyed by instance path as JSON."""
    encoded = {
        path: {"model": device.model, "parameters": device.parameters}
        for path, device in (substitutes or {}).items()
    }
    return json.dumps(encoded, allow_nan=False)


@contextlib.contextmanager
def elaborating() -> Generator[None]:
    """Translates engine elaboration failures into `ElaborationError` with locations.

    Raises:
        ElaborationError: Lists every problem reported by the engine, naming the
            instance and build-script line.
    """
    try:
        yield
    except _core.ElaborationError as error:
        (found,) = error.args
        raise ElaborationError([checks.problem(each) for each in found]) from None


def source_dimension(built: ir.Circuit, path: str) -> Dimension | None:
    """Returns the physical dimension of the independent source at `path`, if any."""
    match built.find(path):
        case ir.Instance(of=ir.DeviceOf(model="current_source")):
            return CURRENT
        case ir.Instance(of=ir.DeviceOf(model="voltage_source")):
            return VOLTAGE
    return None


def swept_dimension(built: ir.Circuit, sweep: DcSweep) -> Dimension:
    """Returns the physical dimension of the quantity swept by `sweep` in `built`."""
    if sweep.parameter is not None:
        return sweep.values[0].dimension if sweep.values else DIMENSIONLESS
    if (dimension := source_dimension(built, sweep.source)) is not None:
        return dimension
    if not sweep.values:
        return VOLTAGE
    # Not an independent source: the engine reports it regardless of values.
    return sweep.values[0].dimension


def sweep_nature(built: ir.Circuit, sweep: DcSweep) -> results.Nature | None:
    """Returns whether `sweep` varies a voltage or current, or None for parameters."""
    if sweep.parameter is not None:
        return None
    source = source_dimension(built, sweep.source)
    if source is None:
        return None
    return "current" if source == CURRENT else "voltage"


def toleranced_payload(
    built: ir.Circuit, substituted: Collection[str]
) -> list[JsonValue]:
    """Extracts toleranced parameter payloads, skipping substituted subtrees."""
    prefixes = tuple(f"{inside}/" for inside in substituted)
    return variation.payload(
        parameter
        for parameter in variation.toleranced(built)
        if not (prefixes and parameter.instance.startswith(prefixes))
    )


def encode_analysis(
    built: ir.Circuit, substituted: Collection[str], analysis: Analysis
) -> JsonValue:
    """Encodes `analysis` as the JSON payload expected by the engine."""
    match analysis:
        case DcSweep():
            return analysis.to_json(swept_dimension(built, analysis))
        case Sensitivity(of=DcSweep() as sweep):
            return analysis.to_json(swept_dimension(built, sweep))
        case Mismatch(of=of):
            dimension = (
                swept_dimension(built, of) if isinstance(of, DcSweep) else VOLTAGE
            )
            return analysis.to_json(dimension, toleranced_payload(built, substituted))
    return analysis.to_json()


def points_of(analysis: Analysis) -> int:
    """Returns the number of traced points for `analysis` in a batch."""
    match analysis:
        case DcSweep(values=values):
            return len(values)
        case Ac(frequencies=frequencies):
            return len(frequencies)
        case Transient(breakpoints=times) | PeriodicSteadyState(breakpoints=times):
            return len(times)
    return 1


def build_recording(
    core: _core.Simulator,
    unknowns: tuple[results.Unknown, ...],
    record: str,
    *,
    regions: bool,
) -> results.Recording:
    """Decodes the engine's recording metadata for `record` and `regions`."""
    devices, summed_terminals = core.recording(record, regions)
    thermal = {name for name, heat in summed_terminals if heat}
    recorded = []
    for (
        path,
        terminals,
        heats,
        first,
        count,
        variables,
        names,
        margins,
    ) in devices:
        thermal.update(
            f"{path}.{terminal}"
            for terminal, heat in zip(terminals, heats, strict=True)
            if heat
        )
        internals = unknowns[first : first + count]
        recorded.append(
            results.RecordedDevice(
                path,
                tuple(terminals),
                tuple(unknown.name for unknown in internals),
                tuple(variables),
                tuple(names),
                tuple(margins),
            )
        )
    return results.Recording(
        tuple(recorded),
        tuple(name for name, _ in summed_terminals),
        frozenset(thermal),
    )


def recorded(
    outcome: _core.RecordedValues, recording: results.Recording
) -> results.RecordedValues:
    """Extracts recorded device arrays from `outcome` paired with `recording`."""
    return {
        "recording": recording,
        "currents": outcome["currents"],
        "charges": outcome["charges"],
        "conductances": outcome["conductances"],
        "capacitances": outcome["capacitances"],
        "variables": outcome["variables"],
    }


def statistics(outcome: _core.Statistics) -> results.Statistics:
    """Decodes transient integration statistics, profile, and charge-leak data."""
    profile = outcome["profile"]
    leak = outcome["charge_leak"]
    return results.Statistics(
        outcome["initial"],
        outcome["accepted"],
        outcome["rejected"],
        outcome["landed"],
        outcome["iterations"],
        results.Work(**outcome["work"]),
        None if profile is None else results.Profile(**profile),
        results.ChargeLeak(leak["signed"], leak["absolute"]),
    )


def decode_point(
    unknowns: tuple[results.Unknown, ...],
    nets: tuple[str, ...],
    solved: _core.SolvedPoint,
    recording: results.Recording,
) -> results.OperatingPointResult:
    """Decodes one engine operating-point payload into `OperatingPointResult`."""
    return results.OperatingPointResult(
        unknowns,
        solved["values"],
        solved["strategy"],
        solved["iterations"],
        nets,
        solved["logic"],
        **recorded(solved, recording),
    )


def decode_result(
    unknowns: tuple[results.Unknown, ...],
    nets: tuple[str, ...],
    outcome: _core.AnalysisResult,
    recording: results.Recording,
) -> results.AnalysisResult:
    """Decodes an engine `outcome` payload into its typed `AnalysisResult`."""
    match outcome["kind"]:
        case "operating_point":
            return decode_point(unknowns, nets, outcome, recording)
        case "operating_points":
            points = tuple(
                decode_point(unknowns, nets, point, recording)
                for point in outcome["points"]
            )
            return results.OperatingPointsResult(unknowns, points)
        case "dc_sweep":
            return results.DcSweepResult(
                unknowns,
                outcome["source"],
                outcome["nature"],
                outcome["swept"],
                outcome["values"],
                nets,
                outcome["logic"],
                **recorded(outcome, recording),
                parameter=outcome["parameter"],
                strategies=tuple(outcome["strategies"]),
                iterations=tuple(outcome["iterations"]),
            )
        case "ac":
            return results.AcResult(
                unknowns,
                outcome["source"],
                outcome["frequencies"],
                outcome["values"],
                decode_point(unknowns, nets, outcome["operating_point"], recording),
                recording=recording,
                currents=outcome["currents"],
            )
        case "transient" | "periodic_steady_state":
            return _decode_waveforms(unknowns, nets, outcome, recording)
        case _:
            return _decode_linear(unknowns, nets, outcome, recording)


def _decode_waveforms(
    unknowns: tuple[results.Unknown, ...],
    nets: tuple[str, ...],
    outcome: _core.TransientResult | _core.PeriodicSteadyStateResult,
    recording: results.Recording,
) -> results.TransientResult:
    """Decodes transient or periodic steady state waveforms."""
    traces = tuple(
        results.LogicTrace(times, levels) for times, levels in outcome["logic"]
    )
    fields = (
        unknowns,
        outcome["times"],
        outcome["values"],
        statistics(outcome["statistics"]),
        nets,
        traces,
    )
    recorded_values = recorded(outcome, recording)
    if outcome["kind"] == "transient":
        return results.TransientResult(*fields, **recorded_values)
    return results.PeriodicSteadyStateResult(
        *fields,
        **recorded_values,
        period=outcome["period"],
        periods=outcome["periods"],
    )


def _decode_differentiated(
    unknowns: tuple[results.Unknown, ...],
    nets: tuple[str, ...],
    outcome: _core.SensitivityResult | _core.MismatchResult,
    recording: results.Recording,
) -> tuple[results.DifferentiatedResult, results.Real | results.Complex]:
    """Decodes the differentiated inner analysis and derivative matrix."""
    inner = decode_result(unknowns, nets, outcome["result"], recording)
    if not isinstance(inner, results.DIFFERENTIATED_TYPES):
        message = f"the engine differentiated {outcome['result']['kind']}"
        raise SimulationError(message)
    derivatives = outcome["derivatives"]
    if isinstance(inner, results.OperatingPointResult):
        derivatives = derivatives[0]
    return inner, derivatives


def _decode_linear(
    unknowns: tuple[results.Unknown, ...],
    nets: tuple[str, ...],
    outcome: _core.SensitivityResult
    | _core.MismatchResult
    | _core.StabilityResult
    | _core.NoiseResult
    | _core.PoleZeroResult
    | _core.TransferFunctionResult,
    recording: results.Recording,
) -> results.AnalysisResult:
    """Decodes a small-signal, sensitivity, or mismatch result."""
    match outcome["kind"]:
        case "sensitivity":
            parameters = tuple(
                results.Parameter(instance, name, float(value))
                for (instance, name), value in zip(
                    outcome["parameters"], outcome["values"], strict=True
                )
            )
            inner, derivatives = _decode_differentiated(
                unknowns, nets, outcome, recording
            )
            return results.SensitivityResult(
                unknowns,
                tuple(outcome["outputs"]),
                parameters,
                derivatives,
                inner,
                outcome["period_derivatives"],
            )
        case "mismatch":
            random_variables = tuple(
                ir.RandomVariable(
                    instance=instance,
                    lot=lot,
                    name=name,
                    mean=mean,
                    standard_deviation=deviation,
                    derivatives=derivatives,
                )
                for instance, lot, name, mean, deviation, derivatives in outcome[
                    "random_variables"
                ]
            )
            inner, derivatives = _decode_differentiated(
                unknowns, nets, outcome, recording
            )
            return results.MismatchResult(
                unknowns,
                tuple(outcome["outputs"]),
                random_variables,
                derivatives,
                inner,
                outcome["period_derivatives"],
            )
        case "transfer_function":
            return results.TransferFunctionResult(
                unknowns,
                outcome["source"],
                outcome["input"],
                outcome["output"],
                outcome["reference"],
                outcome["gain"],
                outcome["input_resistance"],
                outcome["output_resistance"],
                decode_point(unknowns, nets, outcome["operating_point"], recording),
            )
        case "stability":
            return results.StabilityResult(
                unknowns,
                outcome["probe"],
                outcome["frequencies"],
                outcome["gains"],
                decode_point(unknowns, nets, outcome["operating_point"], recording),
            )
        case "noise":
            found = outcome["input"]
            noise_input = (
                None
                if found is None
                else results.NoiseInput(
                    found["source"], found["nature"], found["gains"]
                )
            )
            return results.NoiseResult(
                unknowns,
                outcome["output"],
                outcome["reference"],
                outcome["nature"],
                outcome["frequencies"],
                outcome["densities"],
                tuple(
                    results.NoiseSource(instance, name)
                    for instance, name in outcome["sources"]
                ),
                outcome["contributions"],
                noise_input,
                decode_point(unknowns, nets, outcome["operating_point"], recording),
            )
        case "pole_zero":
            return results.PoleZeroResult(
                unknowns,
                outcome["poles"],
                outcome["zeros"],
                outcome["source"],
                outcome["output"],
                outcome["reference"],
                decode_point(unknowns, nets, outcome["operating_point"], recording),
            )


def analysis_shape(
    built: ir.Circuit,
    unknowns: tuple[results.Unknown, ...],
    analysis: Analysis,
) -> results.AnalysisResult:
    """Builds a zero-point result carrying the shape and metadata of `analysis`."""
    width = len(unknowns)
    point = results.OperatingPointResult(unknowns, np.zeros(width), "")
    empty, rows = np.empty(0), np.empty((0, width))
    match analysis:
        case OperatingPoint():
            return point
        case OperatingPoints():
            return results.OperatingPointsResult(unknowns, ())
        case DcSweep():
            return results.DcSweepResult(
                unknowns,
                analysis.source,
                sweep_nature(built, analysis),
                empty,
                rows,
                parameter=analysis.parameter,
            )
        case Ac():
            phasors = np.empty((0, width), dtype=np.complex128)
            return results.AcResult(unknowns, analysis.source, empty, phasors, point)
        case Transient() | PeriodicSteadyState():
            empty_stats = results.Statistics("", 0, 0, 0, 0)
            return (
                results.TransientResult(unknowns, empty, rows, empty_stats)
                if isinstance(analysis, Transient)
                else results.PeriodicSteadyStateResult(
                    unknowns, empty, rows, empty_stats, period=math.nan, periods=0
                )
            )
    return _linear_shape(built, unknowns, analysis, point)


def _linear_shape(
    built: ir.Circuit,
    unknowns: tuple[results.Unknown, ...],
    analysis: Sensitivity | Mismatch | Stability | Noise | PoleZero | TransferFunction,
    point: results.OperatingPointResult,
) -> results.AnalysisResult:
    """Builds a zero-point result for a linear, sensitivity, or mismatch analysis."""
    roots = np.empty(0, dtype=np.complex128)
    match analysis:
        case Sensitivity() | Mismatch():
            result_class = (
                results.SensitivityResult
                if isinstance(analysis, Sensitivity)
                else results.MismatchResult
            )
            return result_class(
                unknowns,
                tuple(analysis.outputs),
                (),
                np.empty((0, len(unknowns))),
                point,
            )
        case Stability():
            return results.StabilityResult(
                unknowns, analysis.probe, np.empty(0), roots, point
            )
        case Noise():
            names = {unknown.name: unknown.nature for unknown in unknowns}
            given = None
            if analysis.source is not None:
                nature: results.Nature = (
                    "current"
                    if source_dimension(built, analysis.source) == CURRENT
                    else "voltage"
                )
                given = results.NoiseInput(analysis.source, nature, roots)
            empty = np.empty(0)
            return results.NoiseResult(
                unknowns,
                analysis.output,
                analysis.reference,
                names.get(analysis.output, "voltage"),
                empty,
                empty,
                (),
                np.empty((0, 0)),
                given,
                point,
            )
        case TransferFunction():
            branch = f"{analysis.source}:current"
            held = any(unknown.name == branch for unknown in unknowns)
            return results.TransferFunctionResult(
                unknowns,
                analysis.source,
                "voltage" if held else "current",
                analysis.output,
                analysis.reference,
                math.nan,
                math.nan,
                math.nan,
                point,
            )
        case PoleZero():
            transfer = analysis.source is not None and analysis.output is not None
            return results.PoleZeroResult(
                unknowns,
                roots,
                roots if transfer else None,
                analysis.source,
                analysis.output,
                analysis.reference,
                point,
            )
