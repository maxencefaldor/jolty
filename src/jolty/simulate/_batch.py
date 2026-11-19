# Copyright 2026 Maxence Faldor
"""Batched simulation execution over parameter variants, in the engine or ngspice."""

import functools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from jolty import _core, checks, foundry, ir, results, specs
from jolty.errors import JoltyError, close_names
from jolty.plan import (
    Ac,
    Analysis,
    Conditions,
    DcSweep,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    Transient,
)
from jolty.simulate._decode import (
    analysis_shape,
    encode_analysis,
    engine_pool,
    points_of,
    sweep_nature,
    swept_dimension,
)
from jolty.units import FREQUENCY, TIME, Dimension

_RAISED: Final[dict[str, type[Exception]]] = {
    "overflow": OverflowError,
    "indeterminate": ValueError,
    "memory": MemoryError,
}
"""What Python raises where the engine's arithmetic stopped a measure, by its reason,
as `math.fsum` or the allocation of a distortion's samples would."""

_ALONG: Final[dict[type, Literal["point", "swept", "frequency", "time"]]] = {
    OperatingPoint: "point",
    DcSweep: "swept",
    Ac: "frequency",
    Transient: "time",
    PeriodicSteadyState: "time",
}
"""What the points of each kind of analysis a batch traces are."""

type _Decoder = tuple[
    Callable[[str], results.Recording],
    Callable[[_core.AnalysisResult, results.Recording], results.AnalysisResult],
]
"""Pair of `(recording_for, decode_result)` bound to one simulator view."""


@dataclass(frozen=True, slots=True)
class BatchHost:
    """Simulation capabilities passed into `Planned` without private access."""

    circuit: ir.Circuit
    unknowns: tuple[results.Unknown, ...]
    conditions: Conditions
    foundry: tuple[str, ...]
    core: _core.Simulator
    substituted: frozenset[str]
    record_for: Callable[[Analysis], JsonValue]
    decoder: _Decoder
    view_for: Callable[[_core.Simulator], _Decoder]


def mismatch(seed: int, index: int) -> int:
    """The seed of the kit's mismatch of the sample `index` of a Monte Carlo analysis.

    It is the same at every corner, so a corner runs the same parts.
    """
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def labels(parameters: Sequence[tuple[str, str] | None]) -> tuple[str, ...]:
    """The name of each of `parameters`: ``R1 resistance``, or ``temperature``."""
    return tuple(
        "temperature" if parameter is None else f"{parameter[0]} {parameter[1]}"
        for parameter in parameters
    )


@dataclass(frozen=True, slots=True)
class _Reading:
    """How a batch reads one measure: by the engine, in Python, or not at all."""

    measure: specs.Measure
    resolved: specs.ResolvedMeasure | None = None
    slot: int = -1
    reason: str | None = None
    everywhere: bool = False


@dataclass(slots=True)
class Planned:
    """A batch as the simulation runs it."""

    host: BatchHost
    analyses: Mapping[str, Analysis]
    options: Options
    names: tuple[str, ...]
    name_index: dict[str, int]
    engine: dict[str, int]
    settings: list[tuple[str, str]]
    records: dict[str, str]
    refused: dict[str, str]
    readings: list[_Reading]
    traces: list[tuple[str, list[int]]]
    read: dict[tuple[int, int], specs.Reading | str]

    @classmethod
    def from_host(
        cls,
        host: BatchHost,
        analyses: Mapping[str, Analysis],
        options: Options,
        measures: Sequence[specs.Measure],
        traces: Sequence[str],
    ) -> Planned:
        """Plans a batch of `analyses` on `host`.

        Raises:
            KeyError: When `traces` names no unknown.
        """
        names = tuple(analyses)
        planned = cls(
            host,
            analyses,
            options,
            names,
            {name: index for index, name in enumerate(names)},
            {},
            [],
            {},
            {},
            [],
            [],
            {},
        )
        for name, analysis in analyses.items():
            if host.foundry:
                planned.engine[name] = len(planned.engine)
                continue
            try:
                encoded = json.dumps(
                    encode_analysis(host.circuit, host.substituted, analysis),
                    allow_nan=False,
                )
                record = host.record_for(analysis)
            except JoltyError as error:
                planned.refused[name] = str(error)
                continue
            settings = {**options.to_json(), "record": record}
            planned.engine[name] = len(planned.settings)
            planned.records[name] = json.dumps(record)
            planned.settings.append((encoded, json.dumps(settings, allow_nan=False)))
        for measure in measures:
            planned.readings.append(planned._reading(measure))
        columns: list[int] = []
        trace_errors: list[str] = []
        for trace in traces:
            try:
                columns.append(results.column(host.unknowns, trace))
            except KeyError as error:
                trace_errors.append(str(error.args[0]))
        if trace_errors:
            message = "\n\n".join(trace_errors)
            raise KeyError(message)
        planned.traces = [
            (name, columns)
            for name, analysis in analyses.items()
            if columns
            and name in planned.engine
            and (
                isinstance(analysis, OperatingPoint | DcSweep | Ac)
                or (
                    isinstance(analysis, Transient | PeriodicSteadyState)
                    and analysis.breakpoints
                )
            )
        ]
        return planned

    def _reading(self, measure: specs.Measure) -> _Reading:
        """How the batch reads `measure`."""
        name = measure.analysis
        if name not in self.analyses:
            close = close_names(self.names, name)
            reason = f"the batch runs no analysis named {name}{close}"
            return _Reading(measure, reason=reason, everywhere=True)
        if name in self.refused:
            reason = f"the analysis {name} failed"
            return _Reading(measure, reason=reason, everywhere=True)
        if measure.resolved is None or self.host.foundry:
            return _Reading(measure)
        shape = analysis_shape(
            self.host.circuit, self.host.unknowns, self.analyses[name]
        )
        try:
            resolved = measure.resolved(shape)
        except (JoltyError, KeyError) as error:
            return _Reading(measure, reason=str(error.args[0]))
        slot = sum(reading.resolved is not None for reading in self.readings)
        return _Reading(measure, resolved=resolved, slot=slot)

    def _in_python(self) -> list[int]:
        """The measures read in Python, by their indices."""
        return [
            index
            for index, reading in enumerate(self.readings)
            if reading.resolved is None and reading.reason is None
        ]

    def _readings(self, index: int, run: results.Run) -> None:
        """Reads each measure read in Python of the results of the run `index`."""
        for place in self._in_python():
            measure = self.readings[place].measure
            result = run.results.get(measure.analysis)
            if result is None:
                self.read[index, place] = f"the analysis {measure.analysis} failed"
                continue
            try:
                self.read[index, place] = measure.read(result)
            except (JoltyError, KeyError) as error:
                self.read[index, place] = str(error.args[0])

    def in_engine(
        self,
        ran: _core.Variants,
        runs: range,
        each: Callable[[results.Run], object] | None,
    ) -> _core.BatchResult:
        """What the engine finds on the runs `runs` of `ran`, each handed to `each`."""
        measures = [
            (self.engine[reading.measure.analysis], reading.resolved.measure)
            for reading in self.readings
            if reading.resolved is not None
        ]
        traces = [(self.engine[name], columns) for name, columns in self.traces]
        views: dict[int, _Decoder] = {}

        def hand(
            index: int,
            structure: _core.Simulator | None,
            problems: list[_core.Problem] | None,
            outcomes: list[_core.AnalysisResult | str],
        ) -> None:
            decoder = self.host.decoder
            if structure is not None:
                decoder = views.get(id(structure)) or views.setdefault(
                    id(structure), self.host.view_for(structure)
                )
            run = self._run(decoder, index, problems, outcomes)
            self._readings(index - runs.start, run)
            if each is not None:
                each(run)

        handing = each is not None or bool(self._in_python())
        return self.host.core.batch(
            self.settings,
            ran,
            (runs.start, runs.stop),
            measures,
            traces,
            engine_pool(),
            hand if handing else None,
        )

    def _run(
        self,
        decoder: _Decoder,
        index: int,
        problems: list[_core.Problem] | None,
        outcomes: list[_core.AnalysisResult | str],
    ) -> results.Run:
        """What the run `index` found, its outcomes read by `decoder`."""
        if problems is not None:
            return results.Run(
                index, {}, {}, tuple(checks.problem(each) for each in problems)
            )
        recording_for, decode = decoder
        found: dict[str, results.AnalysisResult] = {}
        errors = dict(self.refused)
        for name, place in self.engine.items():
            outcome = outcomes[place]
            if isinstance(outcome, str):
                errors[name] = outcome
                continue
            found[name] = decode(outcome, recording_for(self.records[name]))
        ordered = {name: errors[name] for name in self.names if name in errors}
        return results.Run(index, found, ordered)

    def in_ngspice(
        self,
        ran: _core.Variants,
        runs: range,
        each: Callable[[results.Run], object] | None,
    ) -> _core.BatchResult:
        """What ngspice finds on the runs `runs` of `ran`, each handed to `each`."""
        with ThreadPoolExecutor(engine_pool().threads) as executor:
            found = list(executor.map(functools.partial(self._ngspice, ran), runs))
        count = len(found)
        ended = np.zeros((count, len(self.names)), dtype=np.bool_)
        problems: dict[int, list[_core.Problem]] = {}
        errors: dict[int, list[tuple[int, str]]] = {}
        traces: list[results.Real | results.Complex] = []
        for name, columns in self.traces:
            unknowns = [self.host.unknowns[column] for column in columns]
            analysis = self.analyses[name]
            shape = (count, points_of(analysis), len(unknowns))
            array: results.Real | results.Complex = (
                np.empty(shape, dtype=np.complex128)
                if isinstance(analysis, Ac)
                else np.empty(shape)
            )
            for row, (run, _, _) in enumerate(found):
                array[row] = _at(run.results.get(name), analysis, unknowns)
            traces.append(array)
        for row, (run, problem, _) in enumerate(found):
            self._readings(row, run)
            if problem is not None:
                problems[row] = problem
            for name in run.results:
                ended[row, self.name_index[name]] = True
            if run.errors:
                errors[row] = [
                    (self.name_index[name], error) for name, error in run.errors.items()
                ]
            if each is not None:
                each(run)
        empty = np.empty((count, 0))
        return {
            "values": np.array([numbers for _, _, numbers in found]).reshape(
                count, len(ran.parameters)
            ),
            "parameters": ran.parameters,
            "measured": empty,
            "causes": empty.astype(np.int32),
            "reasons": [],
            "ended": ended,
            "problems": problems,
            "errors": errors,
            "traces": traces,
        }

    def _ngspice(
        self, ran: _core.Variants, index: int
    ) -> tuple[results.Run, list[_core.Problem] | None, list[float]]:
        """What ngspice finds on the run `index` of `ran`, and the values it took."""
        replaced, temperature = ran.overrides(index)
        own = self.host.conditions.overrides
        overrides: dict[str, dict[str, float]] = {
            path: {**own.get(path, {}), **replaced.get(path, {})}
            for path in {*own, *replaced}
        }
        kelvin = (
            self.host.conditions.temperature if temperature is None else temperature
        )
        numbers: list[float] = [
            kelvin
            if parameter is None
            else overrides.get(parameter[0], {}).get(parameter[1], math.nan)
            for parameter in ran.parameters
        ]
        elaborated = self.host.core.elaborated
        try:
            values = elaborated.values(json.dumps(overrides, allow_nan=False), kelvin)
            elaborated.structure_of(values)
        except _core.ElaborationError as error:
            (found,) = error.args
            problems = tuple(checks.problem(each) for each in found)
            return results.Run(index, {}, {}, problems), found, numbers
        sample = ran.sample(index)
        conditions = Conditions(
            kelvin,
            self.host.conditions.process,
            overrides,
            None if sample is None else mismatch(*sample),
        )
        solved: dict[str, results.AnalysisResult] = {}
        failed: dict[str, str] = {}
        for name, analysis in self.analyses.items():
            try:
                solved[name] = foundry.run(
                    self.host.circuit,
                    analysis,
                    self.options,
                    conditions,
                    self.host.unknowns,
                )
            except JoltyError as error:
                failed[name] = str(error)
        return results.Run(index, solved, failed), None, numbers

    def result(self, found: _core.BatchResult) -> results.BatchResult:
        """The result of the batch, from what the engine or ngspice `found`.

        Raises:
            OverflowError: Where a measure's sum overflowed on a run.
            ValueError: Where a measure's sum was indeterminate on a run.
            MemoryError: Where a distortion's samples would not fit in memory.
        """
        ended_engine = found["ended"]
        runs = int(found["values"].shape[0])
        ended = np.zeros((runs, len(self.names)), dtype=np.bool_)
        for name, place in self.engine.items():
            ended[:, self.name_index[name]] = ended_engine[:, place]
        columns = _Columns(runs, len(self.readings))
        units: list[Dimension | results.Plain | None] = []
        for place, reading in enumerate(self.readings):
            if reading.resolved is not None:
                columns.engine(place, reading, found)
                units.append(reading.resolved.unit)
            elif reading.reason is not None:
                analysis = reading.measure.analysis
                where = (
                    np.zeros(runs, dtype=np.bool_)
                    if reading.everywhere
                    else ended[:, self.name_index[analysis]]
                )
                columns.unresolved(place, reading, where)
                units.append(None)
            else:
                units.append(self._read_in_python(place, columns))
        if columns.raised:
            _, _, error = min(columns.raised, key=lambda raised: raised[:2])
            raise error
        return results.BatchResult(
            self.names,
            labels(found["parameters"]),
            np.asarray(found["values"], dtype=np.float64),
            ended,
            tuple(str(reading.measure) for reading in self.readings),
            tuple(units),
            columns.measured,
            columns.causes,
            tuple(columns.texts),
            {
                int(index): tuple(checks.problem(each) for each in problems)
                for index, problems in found["problems"].items()
            },
            self._errors(found, runs),
            self._traces(found),
        )

    def _read_in_python(
        self, place: int, columns: _Columns
    ) -> Dimension | results.Plain | None:
        """Writes what the measure at `place` read in Python of each run, or not."""
        unit: Dimension | results.Plain | None = None
        for (index, at), read in sorted(self.read.items()):
            if at != place:
                continue
            if isinstance(read, str):
                columns.causes[index, place] = columns.told(read)
                continue
            if unit is None:
                unit = read.unit
            if read.unit != unit:
                reason = f"read in {read.unit}, not {unit}"
                columns.causes[index, place] = columns.told(reason)
                continue
            columns.measured[index, place] = read.value
        return unit

    def _errors(self, found: _core.BatchResult, runs: int) -> dict[int, dict[str, str]]:
        """Why each analysis that failed on a run failed, by the run, then by name."""
        problems = found["problems"]
        failed: dict[int, dict[str, str]] = {}
        by_place = {place: name for name, place in self.engine.items()}
        for index, errors in found["errors"].items():
            named = {
                by_place.get(place, self.names[place]): error for place, error in errors
            }
            failed[int(index)] = named
        if self.refused:
            for index in range(runs):
                if index not in problems:
                    failed.setdefault(index, {}).update(self.refused)
        return {
            index: {name: errors[name] for name in self.names if name in errors}
            for index, errors in sorted(failed.items())
        }

    def _traces(self, found: _core.BatchResult) -> tuple[results.Traces, ...]:
        """Each analysis traced, from the arrays `found` holds."""
        traces = []
        for (name, columns), values in zip(self.traces, found["traces"], strict=True):
            analysis = self.analyses[name]
            source = getattr(analysis, "source", None)
            traces.append(
                results.Traces(
                    name,
                    _ALONG[type(analysis)],
                    _axis(analysis, self.host.circuit),
                    tuple(self.host.unknowns[column] for column in columns),
                    values,
                    source,
                    sweep_nature(self.host.circuit, analysis)
                    if isinstance(analysis, DcSweep)
                    else None,
                    analysis.parameter if isinstance(analysis, DcSweep) else None,
                )
            )
        return tuple(traces)


@dataclass(slots=True)
class _Columns:
    """What a batch's measures read of each run, or why not, as it is written."""

    measured: results.Real
    causes: npt.NDArray[np.int32]
    texts: dict[str, int]
    raised: list[tuple[int, int, Exception]]

    def __init__(self, runs: int, width: int) -> None:
        self.measured = np.full((runs, width), np.nan)
        self.causes = np.full((runs, width), -1, dtype=np.int32)
        self.texts = {}
        self.raised = []

    def told(self, reason: str) -> np.int32:
        """The place of `reason` among the reasons, added if it is new."""
        return np.int32(self.texts.setdefault(reason, len(self.texts)))

    def engine(self, place: int, reading: _Reading, found: _core.BatchResult) -> None:
        """Writes what the engine read of the measure at `place`, or why not."""
        if reading.resolved is None:
            return
        failed = f"the analysis {reading.measure.analysis} failed"
        column = found["causes"][:, reading.slot]
        self.measured[:, place] = found["measured"][:, reading.slot]
        table = np.full(len(found["reasons"]) + 1, -1, dtype=np.int32)
        for cause in sorted({int(each) for each in column[column >= 0]}):
            reason, numbers, message = found["reasons"][cause]
            if reason in _RAISED:
                first = int(np.argmax(column == cause))
                self.raised.append((first, place, _RAISED[reason](message)))
            elif reason == "failed":
                table[cause] = self.told(failed)
            else:
                error = _core.MeasureError(reason, numbers)
                table[cause] = self.told(reading.resolved.explain(error))
        self.causes[:, place] = table[np.where(column >= 0, column, -1)]

    def unresolved(
        self, place: int, reading: _Reading, where: npt.NDArray[np.bool_]
    ) -> None:
        """Writes why the measure at `place`, which resolves against none, read nil."""
        if reading.reason is None:
            return
        if reading.everywhere:
            if len(where):
                self.causes[:, place] = self.told(reading.reason)
            return
        failed = f"the analysis {reading.measure.analysis} failed"
        for mask, reason in ((where, reading.reason), (~where, failed)):
            if mask.any():
                self.causes[mask, place] = self.told(reason)


def _axis(analysis: Analysis, built: ir.Circuit) -> results.Real:
    """The points a batch traces `analysis` along: none at an operating point."""
    match analysis:
        case DcSweep():
            dimension = swept_dimension(built, analysis)
            return np.array(
                [value.to(dimension, "a sweep level") for value in analysis.values]
            )
        case Ac(frequencies=frequencies):
            return np.array([each.to(FREQUENCY, "a frequency") for each in frequencies])
        case Transient(breakpoints=times) | PeriodicSteadyState(breakpoints=times):
            return np.array([each.to(TIME, "a time to step onto") for each in times])
    return np.empty(0)


def _at(
    result: results.AnalysisResult | None,
    analysis: Analysis,
    unknowns: Sequence[results.Unknown],
) -> results.Real | results.Complex:
    """The values of `unknowns` at each traced point of `analysis`, in `result`."""
    points = points_of(analysis)
    if isinstance(analysis, Ac):
        phasors = np.full((points, len(unknowns)), np.nan, dtype=np.complex128)
        if isinstance(result, results.AcResult):
            _picked(result.values, result.unknowns, unknowns, phasors)
        return phasors
    found = np.full((points, len(unknowns)), np.nan)
    match result:
        case results.OperatingPointResult():
            _picked(result.values[np.newaxis, :], result.unknowns, unknowns, found)
        case results.DcSweepResult():
            _picked(result.values, result.unknowns, unknowns, found)
        case results.TransientResult() if isinstance(
            analysis, Transient | PeriodicSteadyState
        ):
            times = [
                time.to(TIME, "a time to step onto") for time in analysis.breakpoints
            ]
            rows = np.array(
                [
                    [
                        _interpolated(result.times, column, time)
                        for column in result.values.T
                    ]
                    for time in times
                ]
            ).reshape(len(times), -1)
            _picked(rows, result.unknowns, unknowns, found)
    return found


def _picked(
    rows: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
    held: Sequence[results.Unknown],
    unknowns: Sequence[results.Unknown],
    into: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> None:
    """Writes columns of `rows` matching `unknowns` from `held` into `into`."""
    named = {unknown.name: place for place, unknown in enumerate(held)}
    for place, unknown in enumerate(unknowns):
        column = named.get(unknown.name)
        if column is not None:
            into[:, place] = rows[:, column]


def _interpolated(times: results.Real, values: results.Real, time: float) -> float:
    """The value at `time`, linearly interpolated between accepted times."""
    after = int(np.searchsorted(times, time, side="left"))
    if after < len(times) and times[after] == time:
        return float(values[after])
    if after == 0 or after == len(times):
        return math.nan
    start, stop = float(times[after - 1]), float(times[after])
    low, high = float(values[after - 1]), float(values[after])
    return low + (high - low) * ((time - start) / (stop - start))
