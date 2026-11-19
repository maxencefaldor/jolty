# Copyright 2026 Maxence Faldor
"""Circuit, results, batch and shard files, and SPICE's raw files.

A circuit file holds a [`Circuit`][jolty.ir.Circuit] as JSON. A batch file holds a
plan's batches of runs, each a [`Batch`][jolty.files.Batch]: the arrays of its
[`BatchResult`][jolty.results.BatchResult] under keys such as ``monte-carlo.measured``,
and a JSON header of the rest. A shard file holds a [`Shard`][jolty.files.Shard], a
range of a plan's Monte Carlo samples: the arrays of its batch at each of the plan's
conditions under keys such as ``1.measured``, for the plan's first corner, and a JSON
header of the rest, the requirements it measured among it. Every file but a circuit's
is a NumPy ``.npz`` archive whose bytes are a function of what it holds alone, so a
shard run again writes the same file. A results file holds the
results of named analyses as a NumPy ``.npz`` archive: a JSON header that names the
circuit's unknowns and logic nets, says what the analyses recorded of its devices, each
recording once however many analyses share it, and says what each analysis is, then
the arrays of each analysis under keys such as ``step.times``, ``step.values``,
``step.currents`` and ``step.conductances``, ``loop.frequencies`` and ``loop.gains``, or
``hiss.densities`` and ``hiss.contributions``, and where each analysis used logic
devices outside their domains. Its header also states the results'
[`Provenance`][jolty.files.Provenance]: the jolty, the build and the platform that
simulated them, and the digests of the circuit and the plan they are of. The header is
in version 20 of the format, which holds that provenance, the regions of operation
each recorded transistor's model distinguishes, the Newton iterations that found each
operating point, each recording's summed terminals as its ``summed_terminals``, a
periodic steady state as an analysis of its own, a mismatch analysis's random
variables, and each transient's integrated residuals (`charge_leak`); a file of another
version is not read.

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
"""The suffix of circuit files, such as ``rc_filter.jolty.json``."""

RESULTS_SUFFIX: Final = ".results.npz"
"""The suffix of results files, such as ``rc_filter.results.npz``."""

RAW_SUFFIX: Final = ".raw"
"""The suffix of SPICE raw files, such as ``rc_filter.raw``."""

BATCHES_SUFFIX: Final = ".batches.npz"
"""The suffix of batch files, such as ``rc_filter.batches.npz``."""

SHARDS_SUFFIX: Final = ".shards"
"""The suffix of the directories that hold a script's shards, such as ``amp.shards``."""

SHARD_PATTERN: Final = "shard-*-*.npz"
"""The names of shard files, which a merge reads in their directory."""


def circuit_path(script: Path) -> Path:
    """Where a script's circuit is saved by default: as ``<stem>.jolty.json``."""
    return script.with_name(script.stem + CIRCUIT_SUFFIX)


def results_path(script: Path) -> Path:
    """Where a script's results are saved by default: as ``<stem>.results.npz``."""
    return script.with_name(script.stem + RESULTS_SUFFIX)


def raw_path(script: Path) -> Path:
    """Where a script's SPICE raw file goes by default: as ``<stem>.raw``."""
    return script.with_name(script.stem + RAW_SUFFIX)


def shards_path(script: Path) -> Path:
    """Where a script's shards are written by default: in ``<stem>.shards``."""
    return script.with_name(script.stem + SHARDS_SUFFIX)


def shard_name(start: int, stop: int) -> str:
    """The name of the file of the samples from `start` up to `stop`.

    Examples:
        >>> shard_name(0, 1000)
        'shard-0-1000.npz'
    """
    return f"shard-{start}-{stop}.npz"


def merged_path(shards: Path) -> Path:
    """Where the batch merged from the shards in `shards` goes: beside the directory.

    Examples:
        >>> merged_path(Path("amp.shards")).name
        'amp.batches.npz'
    """
    return shards.with_name(shards.name.removesuffix(SHARDS_SUFFIX) + BATCHES_SUFFIX)


def batches_path(results: Path) -> Path:
    """Where the batches go beside their plan's results: ``<stem>.batches.npz``.

    Examples:
        >>> batches_path(Path("rc.results.npz")).name
        'rc.batches.npz'
    """
    stem = results.name.removesuffix(RESULTS_SUFFIX).removesuffix(".npz")
    return results.with_name(stem + BATCHES_SUFFIX)


def plot_path(results: Path, analysis: str) -> Path:
    """Where the plot of `analysis` goes beside its results or its batches.

    It is ``<stem>.<name>.plot.png``.

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
    """Where the 3D viewer of a circuit goes beside it: ``<stem>.view.html``.

    Examples:
        >>> viewer_path(Path("rc.jolty.json")).name
        'rc.view.html'
    """
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    return circuit.with_name(f"{stem}.view.html")


def _component_path(circuit: Path, instance: str, suffix: str) -> Path:
    """Path beside `circuit` carrying `instance`'s path segments and `suffix`."""
    stem = circuit.name.removesuffix(CIRCUIT_SUFFIX).removesuffix(".json")
    parts = [stem, *instance.split("/")] if instance else [stem]
    return circuit.with_name(".".join([*parts, suffix]))


def schematic_path(circuit: Path, instance: str = "") -> Path:
    """Where the schematic of the component at `instance` goes beside its circuit.

    The top component's is ``<stem>.schematic.png``; another's carries the ids of its
    instance's path, joined by dots, as ``<stem>.X1.X2.schematic.png``. The suffixes
    keep schematics and plots apart, whatever their names.

    Examples:
        >>> schematic_path(Path("rc.jolty.json")).name
        'rc.schematic.png'
        >>> schematic_path(Path("rc.jolty.json"), "X1/X2").name
        'rc.X1.X2.schematic.png'
    """
    return _component_path(circuit, instance, "schematic.png")


def netlist_path(circuit: Path, instance: str = "") -> Path:
    """Where the KiCad netlist of the component at `instance` goes beside its circuit.

    The top component's is ``<stem>.net``; another's carries the ids of its instance's
    path, joined by dots, as ``<stem>.X1.X2.net``.

    Examples:
        >>> netlist_path(Path("amp.jolty.json")).name
        'amp.net'
        >>> netlist_path(Path("amp.jolty.json"), "X1/X2").name
        'amp.X1.X2.net'
    """
    return _component_path(circuit, instance, "net")


def save_circuit(path: Path, circuit: ir.Circuit) -> None:
    """Writes `circuit` to `path` as JSON."""
    path.write_text(circuit.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_circuit(path: Path) -> ir.Circuit:
    """Reads the circuit file at `path`."""
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
