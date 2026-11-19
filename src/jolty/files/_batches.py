# Copyright 2026 Maxence Faldor
"""Batch archive serialization and shared batch-column encoding."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final, Literal, TypedDict

import numpy as np

from jolty import ir, results
from jolty.files._results import open_validated, write_archive
from jolty.units import Dimension

_ARRAYS: Final = ("values", "ended", "measured", "causes")
"""The arrays of a batch's result that every batch has."""

_BATCHES_VERSION: Final = 4
"""The version of the batch file's format that jolty writes and reads."""

_TABLE_DIMENSIONS: Final = 2
"""Expected dimensionality of a batch's 2-D sample-by-column arrays."""

type Exponents = tuple[str, str, str, str, str]
"""A dimension's exponents of the base units, as fractions."""

type EncodedUnit = Exponents | results.Plain | None
"""A measure's unit: a dimension's exponents, a plain unit, or none."""


@dataclass(frozen=True, slots=True)
class Batch:
    """A batch of runs as a batch file holds it, with the requirements it measured.

    Each of its measures is a requirement's, which `requirements` states in words,
    bounded by the `bounds` of each, its least and its most value, in SI base units,
    None for a bound it does not state.
    """

    result: results.BatchResult
    requirements: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]


def exponents_of(dimension: Dimension) -> Exponents:
    """The exponents of `dimension`, as a file's header holds them."""
    kilogram, metre, second, ampere, kelvin = (
        str(Fraction(exponent))
        for exponent in (
            dimension.kilogram,
            dimension.metre,
            dimension.second,
            dimension.ampere,
            dimension.kelvin,
        )
    )
    return kilogram, metre, second, ampere, kelvin


def of_exponents(exponents: Exponents) -> Dimension:
    """The dimension whose exponents a file's header holds as `exponents`."""
    fractions = [Fraction(exponent) for exponent in exponents]
    return Dimension(
        *(
            int(exponent) if exponent.denominator == 1 else exponent
            for exponent in fractions
        )
    )


class _Traces(ir.FrozenModel):
    analysis: str
    along: Literal["point", "swept", "frequency", "time"]
    unknowns: tuple[results.Unknown, ...]
    source: str | None = None
    nature: results.Nature | None = None
    parameter: str | None = None


class _ColumnFields(TypedDict):
    analyses: tuple[str, ...]
    parameters: tuple[str, ...]
    measures: tuple[str, ...]
    units: tuple[EncodedUnit, ...]
    reasons: tuple[str, ...]
    problems: dict[int, tuple[str, ...]]
    errors: dict[int, dict[str, str]]
    traces: tuple[_Traces, ...]


def _column_fields(found: results.BatchResult) -> _ColumnFields:
    return {
        "analyses": found.analyses,
        "parameters": found.parameters,
        "measures": found.measures,
        "units": tuple(
            exponents_of(unit) if isinstance(unit, Dimension) else unit
            for unit in found.units
        ),
        "reasons": found.reasons,
        "problems": found.problems,
        "errors": found.errors,
        "traces": tuple(
            _Traces(
                analysis=traces.analysis,
                along=traces.along,
                unknowns=traces.unknowns,
                source=traces.source,
                nature=traces.nature,
                parameter=traces.parameter,
            )
            for traces in found.traces
        ),
    }


class Columns(ir.FrozenModel):
    """What a batch or shard file's header holds of a batch result besides arrays."""

    analyses: tuple[str, ...]
    parameters: tuple[str, ...]
    measures: tuple[str, ...]
    units: tuple[EncodedUnit, ...]
    reasons: tuple[str, ...]
    problems: dict[int, tuple[str, ...]]
    errors: dict[int, dict[str, str]]
    traces: tuple[_Traces, ...]

    @classmethod
    def from_batch(cls, found: results.BatchResult) -> Columns:
        """What the header holds of `found`."""
        return cls(**_column_fields(found))

    def result(self, name: str, data: Mapping[str, np.ndarray]) -> results.BatchResult:
        """The result whose arrays `data` holds under keys that start with `name`."""
        values, ended = data[f"{name}.values"], data[f"{name}.ended"]
        measured, causes = data[f"{name}.measured"], data[f"{name}.causes"]
        if values.ndim != _TABLE_DIMENSIONS:
            message = f"{name}.values has shape {values.shape}, not 2-D"
            raise ValueError(message)
        runs = values.shape[0]
        expected = (
            ("values", values, (runs, len(self.parameters))),
            ("ended", ended, (runs, len(self.analyses))),
            ("measured", measured, (runs, len(self.measures))),
            ("causes", causes, (runs, len(self.measures))),
        )
        for label, array, shape in expected:
            if array.shape != shape:
                message = f"{name}.{label} has shape {array.shape}, not {shape}"
                raise ValueError(message)
        traces = tuple(
            results.Traces(
                each.analysis,
                each.along,
                data[f"{name}.axis.{index}"],
                each.unknowns,
                data[f"{name}.traces.{index}"],
                each.source,
                each.nature,
                each.parameter,
            )
            for index, each in enumerate(self.traces)
        )
        units = tuple(
            of_exponents(unit) if isinstance(unit, tuple) else unit
            for unit in self.units
        )
        return results.BatchResult(
            self.analyses,
            self.parameters,
            values,
            ended,
            self.measures,
            units,
            measured,
            causes,
            self.reasons,
            dict(self.problems),
            {index: dict(each) for index, each in self.errors.items()},
            traces,
        )


def batch_arrays(name: str, found: results.BatchResult) -> dict[str, np.ndarray]:
    """The arrays of `found`, under keys that start with `name`."""
    arrays = {f"{name}.{key}": getattr(found, key) for key in _ARRAYS}
    for index, traces in enumerate(found.traces):
        arrays[f"{name}.traces.{index}"] = traces.values
        arrays[f"{name}.axis.{index}"] = traces.axis
    return arrays


class _Batch(Columns):
    requirements: tuple[str, ...]
    bounds: tuple[tuple[float | None, float | None], ...]


class _Batches(ir.FrozenModel):
    format: Literal["jolty-batches"] = "jolty-batches"
    version: Literal[4] = 4
    batches: dict[str, _Batch]


def save_batches(path: Path, batches: Mapping[str, Batch]) -> None:
    """Writes `batches`, by name, to `path`."""
    arrays: dict[str, np.ndarray] = {}
    entries = {}
    for name, batch in batches.items():
        arrays |= batch_arrays(name, batch.result)
        entries[name] = _Batch(
            **_column_fields(batch.result),
            requirements=batch.requirements,
            bounds=batch.bounds,
        )
    write_archive(path, arrays, _Batches(batches=entries))


def load_batches(path: Path) -> dict[str, Batch]:
    """Reads the batch file at `path`: each batch, by name.

    Raises:
        JoltyError: When the file is not a jolty batch file of this version.
    """
    unreadable = f"{path} is not a jolty batch file of version {_BATCHES_VERSION}"
    with open_validated(path, _Batches, unreadable) as (header, data):
        return {
            name: Batch(entry.result(name, data), entry.requirements, entry.bounds)
            for name, entry in header.batches.items()
        }
