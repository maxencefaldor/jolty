# Copyright 2026 Maxence Faldor
"""Shards: a plan's Monte Carlo samples run in ranges, by any number of processes.

Every sample is a pure function of the circuit, the plan, its index and the options, so
a plan's Monte Carlo analysis splits into shards, each the samples from a start up to a
stop, which any machine runs in any order, and the merge of the shards, their batches
joined in the order of their samples, holds what one run of every sample holds, bit for
bit. A shard is ``jolty simulate SCRIPT --samples START:STOP``, a command with no state
of its own, which a failed shard runs again; it writes a
[`Shard`][jolty.files.Shard] to ``shard-<start>-<stop>.npz``, and ``jolty merge``
reports the shards of a directory as one run of every sample reports them.

Each shard carries a [`digest`][jolty.shards.digest] of what its numbers
depend on, which leaves out where the scripts that built its circuits lie, and the
merge refuses shards whose digests differ, or whose samples leave a gap or
overlap.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Final

import numpy as np
from pydantic import JsonValue

from jolty import files, report
from jolty import specs as specs_module
from jolty.errors import JoltyError
from jolty.results import BatchResult
from jolty.specs import Specs

_PACKAGE: Final = Path(__file__).resolve().parent
"""jolty's own directory, whose modules and engine a build is."""


def digest(described: Mapping[str, JsonValue]) -> str:
    """A SHA-256 digest, in hexadecimal, of what a shard's numbers depend on.

    It hashes `described`, which says what the shard ran, with jolty's version, its
    [`build`][jolty.files.build] and its [`platform`][jolty.files.platform]: the
    determinism contract holds the bits to one build on one platform.

    Examples:
        >>> len(digest({"samples": 10}))
        64
    """
    document = {
        "described": described,
        "jolty": version("jolty"),
        "build": files.build(),
        "platform": files.platform(),
    }
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def merge(paths: Sequence[Path]) -> tuple[files.Shard, tuple[Path, ...]]:
    """The shards at `paths` as one, of every sample, their batches joined in order.

    Also returns the paths, in the order of the shards' samples.

    Raises:
        JoltyError: When there is no shard, a file is not one, the shards' digests
            differ, naming the shards of each, or their samples leave a gap or overlap,
            naming the shards beside each, every one of them.
    """
    if not paths:
        message = "there is no shard to merge"
        raise JoltyError(message)
    loaded = sorted(
        ((path, files.load_shard(path)) for path in paths),
        key=lambda found: (found[1].start, found[1].stop),
    )
    _same(loaded)
    _tiled(loaded)
    first = loaded[0][1]
    conditions = []
    for place in range(len(first.conditions)):
        parts = [shard.conditions[place] for _, shard in loaded]
        given = [part for part in parts if part is not None]
        if not given:
            conditions.append(None)
            continue
        if len(given) < len(parts):
            names = _names(
                path
                for (path, _), part in zip(loaded, parts, strict=True)
                if part is None
            )
            message = (
                f"the circuit does not elaborate at {_condition(first, place)} in "
                f"{names} alone"
            )
            raise JoltyError(message)
        try:
            conditions.append(BatchResult.joined(given))
        except ValueError as error:
            order = _names(path for path, _ in loaded)
            message = f"the shards {order}, in this order, do not join: {error}"
            raise JoltyError(message) from None
    merged = files.Shard(
        start=0,
        stop=first.samples,
        digest=first.digest,
        samples=first.samples,
        seed=first.seed,
        toleranced=first.toleranced,
        mismatched=first.mismatched,
        corners=first.corners,
        analyses=first.analyses,
        specs=first.specs,
        conditions=tuple(conditions),
    )
    return merged, tuple(path for path, _ in loaded)


def _same(loaded: Sequence[tuple[Path, files.Shard]]) -> None:
    """Checks that the shards `loaded` share their digest.

    Raises:
        JoltyError: When they do not, naming the shards of each digest.
    """
    by: dict[str, list[Path]] = {}
    for path, shard in loaded:
        by.setdefault(shard.digest, []).append(path)
    if len(by) == 1:
        return
    listed = "; ".join(
        f"{_names(paths)} {'has' if len(paths) == 1 else 'have'} {found[:12]}"
        for found, paths in by.items()
    )
    message = (
        "the shards ran other circuits, plans, options, builds or platforms, as their "
        f"digests differ: {listed}"
    )
    raise JoltyError(message)


def _tiled(loaded: Sequence[tuple[Path, files.Shard]]) -> None:
    """Checks that the samples of the shards `loaded`, by their starts, tile every one.

    Raises:
        JoltyError: Naming the shards beside each gap and each overlap.
    """
    problems = []
    reach, reached = 0, None
    for path, shard in loaded:
        if shard.start > reach:
            after = "" if reached is None else f"after {reached.name} and "
            problems.append(
                f"no shard holds samples {reach} up to {shard.start}, {after}before "
                f"{path.name}"
            )
        elif shard.start < reach and reached is not None:
            problems.append(
                f"{reached.name} and {path.name} both hold samples {shard.start} up "
                f"to {min(reach, shard.stop)}"
            )
        if shard.stop > reach:
            reach, reached = shard.stop, path
    samples = loaded[0][1].samples
    if reach < samples and reached is not None:
        problems.append(
            f"no shard holds samples {reach} up to {samples}, after {reached.name}"
        )
    if problems:
        raise JoltyError("\n".join(problems))


def _names(paths: Iterable[Path]) -> str:
    """The names of `paths`, joined in words."""
    named = [path.name for path in paths]
    return named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"


def _condition(shard: files.Shard, place: int) -> str:
    """The condition at `place` among those of `shard`, in words."""
    return "the plan's own conditions" if place == 0 else shard.corners[place - 1]


def monte_carlo(
    shard: files.Shard,
) -> tuple[report.MonteCarloReport, files.Batch]:
    """What the samples of `shard`, every one of its analysis's, found, against specs.

    Returns the report, and the batch of the samples at the plan's own conditions as a
    batch file saves it.
    """
    count = shard.stop - shard.start
    failed, spreads_found, passing = spreads(
        shard.analyses, shard.conditions, count, shard.specs
    )
    analysed = report.MonteCarloReport(
        samples=shard.samples,
        seed=shard.seed,
        toleranced=shard.toleranced,
        mismatched=shard.mismatched,
        corners=shard.corners,
        failed=failed,
        spreads=spreads_found,
        yield_=passing,
    )
    return analysed, saved(shard.conditions[0], count, shard.analyses, shard.specs)


def spreads(
    planned: Sequence[str],
    found: Sequence[BatchResult | None],
    count: int,
    specs: Specs | None,
) -> tuple[int, tuple[specs_module.Spread, ...], specs_module.Yield | None]:
    """How many of `count` runs failed at any condition, the spreads and the yield.

    `found` holds the runs at each condition, None where the circuit does not
    elaborate, `planned` names every analysis of the plan, and each run's verdict on a
    requirement is its worst over the conditions.
    """
    failed = np.zeros(count, dtype=np.bool_)
    for batch in found:
        failed |= np.ones(count, dtype=np.bool_) if batch is None else batch.failed
    if specs is None:
        return int(failed.sum()), (), None
    judged = [
        specs_module.worst(
            [
                specs_module.judged(requirement, planned, batch, place, count)
                for batch in found
            ]
        )
        for place, requirement in enumerate(specs.requirements)
    ]
    spreads_found, passing = specs_module.spread_of(specs, judged)
    return int(failed.sum()), spreads_found, passing


def saved(
    found: BatchResult | None,
    count: int,
    planned: Sequence[str],
    specs: Specs | None,
) -> files.Batch:
    """The batch of `count` runs `found` at the plan's own conditions, as it is saved.

    With none found, the circuit does not elaborate there, and no run reads anything.
    """
    requirements = () if specs is None else tuple(specs.requirements)
    if found is None:
        width = len(requirements)
        found = BatchResult(
            tuple(planned),
            (),
            np.empty((count, 0)),
            np.zeros((count, len(planned)), dtype=np.bool_),
            tuple(str(each.measure) for each in requirements),
            (None,) * width,
            np.full((count, width), np.nan),
            np.zeros((count, width), dtype=np.int32),
            ("the circuit does not elaborate",),
        )
    return files.Batch(
        found,
        tuple(str(requirement) for requirement in requirements),
        tuple(
            (
                None if each.minimum is None else each.minimum.value,
                None if each.maximum is None else each.maximum.value,
            )
            for each in requirements
        ),
    )
