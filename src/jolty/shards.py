# Copyright 2026 Maxence Faldor
"""Monte Carlo shard splitting, digest verification, and merging.

Every Monte Carlo sample is a deterministic function of the circuit, plan, sample
index, and options. A Monte Carlo analysis can split into independent sample ranges
(shards) executed on any machine in any order. Merging shards produces results
bit-identical to running all samples in a single process.

Each shard runs via ``jolty simulate SCRIPT --samples START:STOP``, writing a stateless
[`Shard`][jolty.files.Shard] to ``shard-<start>-<stop>.npz`` that can be retried safely
on failure. Running ``jolty merge`` combines a shard directory and reports results
identically to a full run. Each shard stores a [`digest`][jolty.shards.digest] of its
numerical inputs (excluding build-script paths), and merging rejects mismatched
digests, gaps, or overlapping sample ranges.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pydantic import JsonValue

from jolty import files, report
from jolty import specs as specs_module
from jolty.errors import JoltyError
from jolty.results import BatchResult
from jolty.specs import Specs


def digest(described: Mapping[str, JsonValue]) -> str:
    """Computes a hexadecimal SHA-256 digest of all inputs affecting shard results.

    The digest hashes `described` with the jolty version, [`build`][jolty.files.build],
    and [`platform`][jolty.files.platform] to enforce bit-identical determinism within
    one build and platform.

    Args:
        described: JSON-serializable mapping of numerical shard inputs.

    Returns:
        Lowercase 64-character hexadecimal SHA-256 digest string.

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
    """Merges shard files at `paths` into one complete shard covering all samples.

    Returns the merged [`Shard`][jolty.files.Shard] along with the input paths sorted
    by sample range.

    Args:
        paths: Non-empty sequence of shard `.npz` file paths to validate and join.

    Returns:
        Pair of merged full-range `Shard` and sorted input shard paths.

    Raises:
        JoltyError: When `paths` is empty, a file is invalid, shard digests differ, or
            sample ranges contain gaps or overlaps.
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
    if not first.conditions or any(
        len(shard.conditions) != len(first.conditions) for _, shard in loaded
    ):
        message = "the shards do not hold the same number of conditions"
        raise JoltyError(message)
    if len(first.conditions) != len(first.corners) + 1:
        message = (
            f"the shards hold {len(first.conditions)} conditions for "
            f"{len(first.corners)} corners"
        )
        raise JoltyError(message)
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
    """Verifies that all shards in `loaded` share the same provenance digest.

    Raises:
        JoltyError: When digests differ, listing the shard filenames for each digest.
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
    """Verifies that sorted shards in `loaded` tile every sample without gaps.

    Raises:
        JoltyError: When gaps or overlapping ranges exist, naming the adjacent shards.
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
    """Joins filenames from `paths` into an English list."""
    named = [path.name for path in paths]
    return named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"


def _condition(shard: files.Shard, place: int) -> str:
    """Formats condition `place` of `shard` as nominal or its corner name."""
    return "the plan's own conditions" if place == 0 else shard.corners[place - 1]


def monte_carlo(
    shard: files.Shard,
) -> tuple[report.MonteCarloReport, files.Batch]:
    """Evaluates all samples in `shard` against specs and builds the report.

    Returns the report together with the nominal-condition batch formatted for a batch
    file.

    Args:
        shard: Merged Monte Carlo shard spanning all planned samples.

    Returns:
        Pair of aggregated `MonteCarloReport` and serializable nominal `Batch`.
    """
    if not shard.conditions:
        message = "the shard holds no conditions"
        raise JoltyError(message)
    count = shard.stop - shard.start
    failed, spreads_found, passing = spreads(
        shard.analyses, shard.conditions, count, shard.specs
    )
    analyzed = report.MonteCarloReport(
        samples=shard.samples,
        seed=shard.seed,
        toleranced=shard.toleranced,
        mismatched=shard.mismatched,
        corners=shard.corners,
        failed=failed,
        spreads=spreads_found,
        yield_=passing,
    )
    return analyzed, saved(shard.conditions[0], count, shard.analyses, shard.specs)


def spreads(
    planned: Sequence[str],
    found: Sequence[BatchResult | None],
    count: int,
    specs: Specs | None,
) -> tuple[int, tuple[specs_module.Spread, ...], specs_module.Yield | None]:
    """Computes failure count, specification spreads, and overall yield across corners.

    `found` holds each condition batch (or ``None`` when elaboration failed) and
    `planned` lists the plan analysis names. Each sample takes its worst verdict across
    all conditions for every requirement.

    Args:
        planned: Planned analysis names defined by the simulation plan.
        found: Per-condition batch results (nominal first, then corners), or ``None``.
        count: Total number of Monte Carlo samples evaluated per condition.
        specs: Optional specification requirements and yield target.

    Returns:
        Three-tuple of failed sample count, requirement `Spread` tuple, and optional
        `Yield`.
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
    """Builds the serializable nominal-condition batch for `count` runs from `found`.

    When `found` is ``None``, elaboration failed and every sample records empty
    readings with an elaboration-failure diagnostic.

    Args:
        found: Nominal-condition batch result, or ``None`` when elaboration failed.
        count: Number of runs represented by the returned batch.
        planned: Planned analysis names defined by the simulation plan.
        specs: Optional circuit specifications naming requirements and bounds.

    Returns:
        Serializable `Batch` pairing nominal readings with requirement labels and
        bounds.
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
