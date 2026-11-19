# Copyright 2026 Maxence Faldor
"""Circuit, results, batch, shard, and SPICE raw file serialization.

A circuit file stores a [`Circuit`][jolty.ir.Circuit] as JSON. A batch file stores a
plan's named [`Batch`][jolty.files.Batch] objects, combining
[`BatchResult`][jolty.results.BatchResult] arrays (such as ``monte-carlo.measured``)
with a JSON header for metadata. A shard file stores a [`Shard`][jolty.files.Shard]
covering a range of Monte Carlo samples across plan conditions. Shard arrays use
condition-indexed keys such as ``1.measured`` for the first corner, while the JSON
header records metadata and measured requirements. Batch, shard, and results files are
deterministic NumPy ``.npz`` archives whose bytes depend solely on their contents, so
re-running a shard reproduces identical output.

A results file stores named analysis results in a NumPy ``.npz`` archive with a JSON
header and per-analysis arrays. The header lists circuit unknowns, logic nets,
deduplicated device recordings shared across analyses, and analysis metadata. Array
keys are prefixed by analysis name, such as ``step.times``, ``step.values``,
``step.currents``, ``step.conductances``, ``loop.frequencies``, ``loop.gains``,
``hiss.densities``, and ``hiss.contributions``. Analysis entries also record logic
domain excursions and [`Provenance`][jolty.files.Provenance] metadata: jolty version,
build, platform, and circuit and plan digests.

Results headers use format version 20; files of any other version are rejected.
Version 20 records provenance, transistor operating regions, operating-point Newton
iteration counts, and ``summed_terminals`` for hierarchical recordings. Format version
20 also supports dedicated periodic steady-state analyses, mismatch random variables,
and integrated transient residuals (`charge_leak`).

Examples:
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from jolty import results
    >>> point = results.OperatingPointResult(
    ...     (results.Unknown("out", "voltage"),), np.array([2.5]), "direct"
    ... )
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     path = Path(directory) / "divider.results.npz"
    ...     save_results(path, {"op": point})
    ...     load_results(path)["op"].voltage("out")
    2.5
"""

from pathlib import Path
from typing import Final

from jolty import ir
from jolty.files._batches import Batch, load_batches, save_batches
from jolty.files._raw import RawAnalysisResult, load_raw, save_raw
from jolty.files._results import (
    Provenance,
    build,
    circuit_digest,
    load_provenance,
    load_results,
    plan_digest,
    platform,
    save_results,
)
from jolty.files._shards import Shard, load_shard, save_shard

CIRCUIT_SUFFIX: Final = ".jolty.json"
"""Filename suffix for circuit JSON files, such as ``rc_filter.jolty.json``."""

RESULTS_SUFFIX: Final = ".results.npz"
"""Filename suffix for analysis results archives, such as ``rc_filter.results.npz``."""

RAW_SUFFIX: Final = ".raw"
"""Filename suffix for SPICE raw files, such as ``rc_filter.raw``."""

BATCHES_SUFFIX: Final = ".batches.npz"
"""Filename suffix for batch archives, such as ``rc_filter.batches.npz``."""

SHARDS_SUFFIX: Final = ".shards"
"""Directory suffix for Monte Carlo shard directories, such as ``amp.shards``."""

SHARD_PATTERN: Final = "shard-*-*.npz"
"""Glob pattern matching shard archive filenames within a shards directory."""


def circuit_path(script: Path) -> Path:
    """Returns the default circuit JSON path ``<stem>.jolty.json`` for `script`.

    Args:
        script: Build script path whose directory and stem determine the circuit path.

    Returns:
        Sibling path ending with `.jolty.json`.
    """
    return script.with_name(script.stem + CIRCUIT_SUFFIX)


def results_path(script: Path) -> Path:
    """Returns the default results archive path ``<stem>.results.npz`` for `script`.

    Args:
        script: Build script path whose directory and stem determine the archive path.

    Returns:
        Sibling path ending with `.results.npz`.
    """
    return script.with_name(script.stem + RESULTS_SUFFIX)


def raw_path(script: Path) -> Path:
    """Returns the default SPICE raw path ``<stem>.raw`` for `script`.

    Args:
        script: Build script path whose directory and stem determine the raw path.

    Returns:
        Sibling path ending with `.raw`.
    """
    return script.with_name(script.stem + RAW_SUFFIX)


def shards_path(script: Path) -> Path:
    """Returns the default shard directory path ``<stem>.shards`` for `script`.

    Args:
        script: Build script path whose directory and stem determine the directory path.

    Returns:
        Sibling directory path ending with `.shards`.
    """
    return script.with_name(script.stem + SHARDS_SUFFIX)


def shard_name(start: int, stop: int) -> str:
    """Returns the shard filename for sample indices from `start` up to `stop`.

    Args:
        start: Inclusive starting Monte Carlo sample index.
        stop: Exclusive ending Monte Carlo sample index.

    Returns:
        Shard archive filename ``shard-<start>-<stop>.npz``.

    Examples:
        >>> shard_name(0, 1000)
        'shard-0-1000.npz'
    """
    return f"shard-{start}-{stop}.npz"


def merged_path(shards: Path) -> Path:
    """Returns the merged batch archive path next to the `shards` directory.

    Args:
        shards: Monte Carlo shards directory path.

    Returns:
        Sibling batch archive path ending with `.batches.npz`.

    Examples:
        >>> merged_path(Path("amp.shards")).name
        'amp.batches.npz'
    """
    return shards.with_name(shards.name.removesuffix(SHARDS_SUFFIX) + BATCHES_SUFFIX)


def batches_path(results: Path) -> Path:
    """Returns the sibling batch archive path ``<stem>.batches.npz`` for `results`.

    Args:
        results: Results archive path whose stem names the batch archive.

    Returns:
        Sibling batch archive path ending with `.batches.npz`.

    Examples:
        >>> batches_path(Path("rc.results.npz")).name
        'rc.batches.npz'
    """
    stem = results.name.removesuffix(RESULTS_SUFFIX).removesuffix(".npz")
    return results.with_name(stem + BATCHES_SUFFIX)


def plot_path(results: Path, analysis: str) -> Path:
    """Returns the plot path ``<stem>.<analysis>.plot.png`` next to `results`.

    Args:
        results: Results or batch archive path whose stem prefixes the plot filename.
        analysis: Analysis or batch name embedded in the plot filename.

    Returns:
        Sibling PNG path ending with `.<analysis>.plot.png`.

    Examples:
        >>> plot_path(Path("rc.results.npz"), "step").name
        'rc.step.plot.png'
        >>> plot_path(Path("rc.batches.npz"), "monte-carlo").name
        'rc.monte-carlo.plot.png'
    """
    name = results.name
    stem = name.removesuffix(RESULTS_SUFFIX).removesuffix(BATCHES_SUFFIX)
    stem = stem.removesuffix(".npz")
    return results.with_name(f"{stem}.{analysis}.plot.png")


def viewer_path(circuit: Path) -> Path:
    """Returns the 3D viewer path ``<stem>.view.html`` next to `circuit`.

    Args:
        circuit: Circuit JSON path whose stem prefixes the viewer filename.

    Returns:
        Sibling HTML path ending with `.view.html`.

    Examples:
        >>> viewer_path(Path("rc.jolty.json")).name
        'rc.view.html'
    """
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    return circuit.with_name(f"{stem}.view.html")


def _component_path(circuit: Path, instance: str, suffix: str) -> Path:
    """Returns a sibling path next to `circuit` for `instance` and `suffix`."""
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    parts = [stem, *instance.split("/")] if instance else [stem]
    return circuit.with_name(".".join([*parts, suffix]))


def schematic_path(circuit: Path, instance: str = "") -> Path:
    """Returns the schematic PNG path for the component at `instance` next to `circuit`.

    The top component uses ``<stem>.schematic.png``; nested instances join their path
    segments with dots, such as ``<stem>.X1.X2.schematic.png``. Distinct suffixes keep
    schematic files separate from plot files regardless of component or analysis names.

    Args:
        circuit: Circuit JSON path whose stem prefixes the schematic filename.
        instance: Hierarchical component instance path, or empty for the top component.

    Returns:
        Sibling PNG path ending with `.schematic.png`.

    Examples:
        >>> schematic_path(Path("rc.jolty.json")).name
        'rc.schematic.png'
        >>> schematic_path(Path("rc.jolty.json"), "X1/X2").name
        'rc.X1.X2.schematic.png'
    """
    return _component_path(circuit, instance, "schematic.png")


def netlist_path(circuit: Path, instance: str = "") -> Path:
    """Returns the KiCad netlist path for the component at `instance` next to `circuit`.

    The top component uses ``<stem>.net``; nested instances join their path segments
    with dots, such as ``<stem>.X1.X2.net``.

    Args:
        circuit: Circuit JSON path whose stem prefixes the netlist filename.
        instance: Hierarchical component instance path, or empty for the top component.

    Returns:
        Sibling KiCad netlist path ending with `.net`.

    Examples:
        >>> netlist_path(Path("amp.jolty.json")).name
        'amp.net'
        >>> netlist_path(Path("amp.jolty.json"), "X1/X2").name
        'amp.X1.X2.net'
    """
    return _component_path(circuit, instance, "net")


def save_circuit(path: Path, circuit: ir.Circuit) -> None:
    """Writes `circuit` to `path` as JSON.

    Args:
        path: Destination file path for the serialized circuit JSON.
        circuit: Circuit IR to write.
    """
    path.write_text(circuit.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_circuit(path: Path) -> ir.Circuit:
    """Reads the circuit file at `path`.

    Args:
        path: Source JSON file path to read and validate.

    Returns:
        Validated `Circuit` loaded from `path`.
    """
    return ir.Circuit.model_validate_json(path.read_bytes())


__all__ = [
    "BATCHES_SUFFIX",
    "CIRCUIT_SUFFIX",
    "RAW_SUFFIX",
    "RESULTS_SUFFIX",
    "SHARDS_SUFFIX",
    "SHARD_PATTERN",
    "Batch",
    "Provenance",
    "RawAnalysisResult",
    "Shard",
    "batches_path",
    "build",
    "circuit_digest",
    "circuit_path",
    "load_batches",
    "load_circuit",
    "load_provenance",
    "load_raw",
    "load_results",
    "load_shard",
    "merged_path",
    "netlist_path",
    "plan_digest",
    "platform",
    "plot_path",
    "raw_path",
    "results_path",
    "save_batches",
    "save_circuit",
    "save_raw",
    "save_results",
    "save_shard",
    "schematic_path",
    "shard_name",
    "shards_path",
    "viewer_path",
]
