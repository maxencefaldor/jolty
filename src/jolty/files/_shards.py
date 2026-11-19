# Copyright 2026 Maxence Faldor
"""Shard archive serialization for partitioned Monte Carlo sample ranges."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from jolty import ir, results
from jolty.errors import JoltyError
from jolty.files._batches import (
    Columns,
    Exponents,
    batch_arrays,
    exponents_of,
    of_exponents,
)
from jolty.files._results import open_validated, write_archive
from jolty.specs import Measure, MeasureError, Reading, Requirement, Specs
from jolty.units import Quantity

_SHARD_VERSION: Final = 3
"""The version of the shard file's format that jolty writes and reads."""


@dataclass(frozen=True, slots=True)
class Shard:
    """The samples from `start` up to `stop` of a plan's Monte Carlo analysis, run.

    The analysis draws `samples` samples from its `seed`, each of the circuit's
    `toleranced` parameters and the kit's mismatch of its `mismatched` foundry's
    devices, and runs them at the plan's own conditions, then at its `corners`, by
    name. `conditions` holds the batch of the shard's samples at each, row 0 its
    sample `start`, or None where the circuit does not elaborate. `analyses` names
    every analysis of the plan, and `specs` are those its samples were measured
    against, if any, each requirement's measure the batch's measure at its
    place. Shards of one run of the analysis share their `digest`, which
    [`jolty.shards`][jolty.shards] makes.
    """

    start: int
    stop: int
    digest: str
    samples: int
    seed: int
    toleranced: int
    mismatched: int
    corners: tuple[str, ...]
    analyses: tuple[str, ...]
    specs: Specs | None
    conditions: tuple[results.BatchResult | None, ...]


class _Bound(ir.FrozenModel):
    value: float
    unit: Exponents


class _Requirement(ir.FrozenModel):
    analysis: str
    description: str
    plain: results.Plain | None
    minimum: _Bound | None
    maximum: _Bound | None
    margin: _Bound | None

    @classmethod
    def from_requirement(cls, requirement: Requirement) -> _Requirement:
        measure = requirement.measure
        return cls(
            analysis=measure.analysis,
            description=measure.description,
            plain=measure.plain,
            minimum=_bound(requirement.minimum),
            maximum=_bound(requirement.maximum),
            margin=_bound(requirement.margin),
        )

    def requirement(self) -> Requirement:
        measure = Measure(
            self.analysis, self.description, self.plain, _read_by_its_shard
        )
        return Requirement(
            measure,
            _quantity(self.minimum),
            _quantity(self.maximum),
            _quantity(self.margin),
        )


def _read_by_its_shard(result: results.AnalysisResult) -> Reading:
    del result
    message = "a shard's file holds what its measures read, and reads no result"
    raise MeasureError(message)


def _bound(quantity: Quantity | None) -> _Bound | None:
    if quantity is None:
        return None
    return _Bound(value=quantity.value, unit=exponents_of(quantity.dimension))


def _quantity(bound: _Bound | None) -> Quantity | None:
    return None if bound is None else Quantity(bound.value, of_exponents(bound.unit))


class _Shard(ir.FrozenModel):
    format: Literal["jolty-shard"] = "jolty-shard"
    version: Literal[3] = 3
    start: int
    stop: int
    digest: str
    samples: int
    seed: int
    toleranced: int
    mismatched: int
    corners: tuple[str, ...]
    analyses: tuple[str, ...]
    requirements: tuple[_Requirement, ...] | None
    yield_target: float | None
    conditions: tuple[Columns | None, ...]


def save_shard(path: Path, shard: Shard) -> None:
    """Writes `shard` to `path`."""
    arrays: dict[str, np.ndarray] = {}
    for place, found in enumerate(shard.conditions):
        if found is not None:
            arrays |= batch_arrays(str(place), found)
    header = _Shard(
        start=shard.start,
        stop=shard.stop,
        digest=shard.digest,
        samples=shard.samples,
        seed=shard.seed,
        toleranced=shard.toleranced,
        mismatched=shard.mismatched,
        corners=shard.corners,
        analyses=shard.analyses,
        requirements=None
        if shard.specs is None
        else tuple(
            _Requirement.from_requirement(each) for each in shard.specs.requirements
        ),
        yield_target=None if shard.specs is None else shard.specs.yield_target,
        conditions=tuple(
            None if found is None else Columns.from_batch(found)
            for found in shard.conditions
        ),
    )
    write_archive(path, arrays, header)


def load_shard(path: Path) -> Shard:
    """Reads the shard file at `path`.

    Raises:
        JoltyError: When the file is not a jolty shard file of this version.
    """
    unreadable = f"{path} is not a jolty shard file of version {_SHARD_VERSION}"
    with open_validated(path, _Shard, unreadable) as (header, data):
        if (
            not (0 <= header.start <= header.stop <= header.samples)
            or len(header.conditions) != len(header.corners) + 1
        ):
            raise JoltyError(unreadable)
        requirements = header.requirements
        return Shard(
            start=header.start,
            stop=header.stop,
            digest=header.digest,
            samples=header.samples,
            seed=header.seed,
            toleranced=header.toleranced,
            mismatched=header.mismatched,
            corners=header.corners,
            analyses=header.analyses,
            specs=None
            if requirements is None
            else Specs(
                [each.requirement() for each in requirements], header.yield_target
            ),
            conditions=tuple(
                None if entry is None else entry.result(str(place), data)
                for place, entry in enumerate(header.conditions)
            ),
        )
