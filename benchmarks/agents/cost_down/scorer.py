# Copyright 2026 Maxence Faldor
"""Hidden benchmark scorer for the resistor cost-down optimization task.

The scorer builds ``design.py``, runs ``jolty simulate`` using the plan and specs from
``bias.py``, verifies that the test bench outside ``X1`` matches the baseline, and
prices ``X1`` against ``catalog.json``. A submission passes when the simulation meets
the plan, all bench and catalog checks pass, and total cost is strictly lower than
baseline. Plan failures keep jolty's shortfall categories, while bench, catalog, and
cost violations map to specs ``bench``, ``catalog``, and ``cost``. For the ngspice arm,
``netlist.py`` mounts ``design.cir`` and ``tolerances.json`` onto the baseline bench.

Catalog bills omit unpriced components such as the given transistor, so cost
comparisons remain valid only when no extra unpriced parts are introduced. Each
unpriced baseline device excuses one identical device in the submission regardless of
name or hierarchy, matching model, parameters, tolerances, ratings, and part number.
"""

import runpy
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

from arms import Arm
from scoring import missing, planned, simulate, unbuilt, unmet
from verdict import Kind, Shortfall, Verdict

from jolty import Component, catalog, circuit, ir
from jolty.errors import JoltyError

INSTANCE = "X1"
"""Hierarchical instance path of the amplifier stage priced against the catalog."""


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """Scores the cost-down submission in `work` against `task` for `arm`."""
    given = task / "given"
    parts = catalog.Catalog.load(given / "catalog.json")
    reference, reference_circuit = simulate(given / "bias.py", given / "bias.py")
    original = catalog.bill(reference_circuit, parts, INSTANCE)
    baseline = original.total
    with tempfile.TemporaryDirectory() as directory:
        design = _script(task, work, arm, Path(directory))
        if not design.is_file():
            return Verdict(
                met=False,
                measures={"baseline": baseline},
                shortfalls=(missing(f"there is no {work / design.name}"),),
            )
        try:
            result, built = simulate(design, given / "bias.py")
        except JoltyError as error:
            # jolty simulate returns analysis failures in the result and raises only
            # when the script fails to build, times out, or exits abnormally.
            return Verdict(
                met=False,
                measures={"baseline": baseline},
                shortfalls=(unbuilt(f"the design does not build: {error}"),),
            )
    shortfalls = planned(result)
    if result.ok == bool(shortfalls):
        message = "jolty's verdict on the plan and the scorer's reasons disagree"
        raise RuntimeError(message)
    if not reference.ok:
        reason = "the baseline design does not meet the reference plan"
        shortfalls.append(Shortfall(kind=Kind.SCORER, reason=reason))
    shortfalls += [
        unmet("bench", each) for each in _bench_changes(reference_circuit, built)
    ]
    priced = catalog.bill(built, parts, INSTANCE)
    excused = _given(task, arm, reference_circuit, original)
    shortfalls += _unpriced(priced, built, excused)
    if priced.total >= baseline:
        reason = f"the design costs {priced.total:.6g}, not less than {baseline:.6g}"
        shortfalls.append(unmet("cost", reason))
    return Verdict(
        met=not shortfalls,
        measures={"baseline": baseline, "cost": priced.total},
        shortfalls=tuple(shortfalls),
        report=result.describe(),
    )


NETLIST = ("design.cir", "tolerances.json")
"""Submission filenames expected for the ngspice arm: stage netlist and tolerances."""


def _script(task: Path, work: Path, arm: Arm, directory: Path) -> Path:
    """Returns the build script for `arm` in `work`, staging ``netlist.py`` if needed.

    Returns ``design.py`` directly for jolty, or stages ``design.cir`` and optional
    ``tolerances.json`` through `_netlist` into `directory` for ngspice.
    """
    if arm is Arm.JOLTY:
        return work / "design.py"
    netlist, tolerances = (work / name for name in NETLIST)
    if not netlist.is_file():
        return directory / netlist.name
    return _netlist(task, netlist, tolerances, directory)


def _netlist(
    task: Path, netlist: Path, tolerances: Path | None, directory: Path
) -> Path:
    """Stages ``netlist.py`` as ``design.py`` in `directory` to wrap `netlist`.

    Copies ``bias.py``, `netlist` as ``design.cir``, and optional `tolerances` as
    ``tolerances.json`` into `directory` alongside the adapter script.
    """
    named_netlist, named_tolerances = NETLIST
    shutil.copy(task / "netlist.py", directory / "design.py")
    shutil.copy(task / "given" / "bias.py", directory / "bias.py")
    shutil.copy(netlist, directory / named_netlist)
    if tolerances is not None and tolerances.is_file():
        shutil.copy(tolerances, directory / named_tolerances)
    return directory / "design.py"


def _given(
    task: Path, arm: Arm, reference: ir.Circuit, original: catalog.Bill
) -> list[ir.Instance]:
    """Collects unpriced baseline `ir.Instance` objects excused by `original` for `arm`.

    Uses instances from `reference` for jolty and parsed instances from the given
    ``stage.cir`` netlist for ngspice.
    """
    if arm is Arm.JOLTY:
        stage = reference
    else:
        netlist = task / "ngspice" / "given" / "stage.cir"
        with tempfile.TemporaryDirectory() as directory:
            script = _netlist(task, netlist, None, Path(directory))
            made = runpy.run_path(str(script))["build"]()
        if not isinstance(made, Component):
            message = "netlist.py builds no component of the given stage"
            raise TypeError(message)
        stage = circuit(made)
    placed = dict(catalog.built(stage, stage.top))
    return [placed[each.instance] for each in original.unpriced]


def _unpriced(
    priced: catalog.Bill, built: ir.Circuit, excused: Iterable[ir.Instance]
) -> list[Shortfall]:
    """Checks unpriced parts in `priced` against `excused` baseline parts.

    Each entry in `excused` excuses one matching device regardless of instance name or
    location; remaining unpriced parts yield ``catalog`` shortfalls.
    """
    placed = dict(catalog.built(built, built.top))
    left = [_part(each) for each in excused]
    reasons = []
    for each in priced.unpriced:
        part = _part(placed[each.instance])
        if part in left:
            left.remove(part)
        else:
            reasons.append(
                unmet("catalog", f"{each.instance} is unpriced: {each.reason}")
            )
    return reasons


def _part(instance: ir.Instance) -> ir.Instance:
    """Normalizes `instance` for location-independent part equality comparison.

    Clears name, terminal connections, and source location while preserving model,
    parameters, tolerances, ratings, and part number.
    """
    return instance.model_copy(
        update={"name": "", "terminals": {}, "source_location": None}
    )


def _bench_changes(reference: ir.Circuit, built: ir.Circuit) -> list[str]:
    """Returns descriptions of modifications to the top-level bench outside ``X1``."""

    def bench(circuit: ir.Circuit) -> dict[str, ir.Instance]:
        return {
            instance.name: instance.model_copy(update={"source_location": None})
            for instance in circuit.components[circuit.top].instances
        }

    original, found = bench(reference), bench(built)
    changes = []
    for instance, expected in original.items():
        placed = found.get(instance)
        if placed is None:
            changes.append(f"the bench places no {instance}")
        elif instance == INSTANCE:
            if placed.terminals != expected.terminals:
                changes.append(f"{instance}'s ports join other nets")
        elif placed != expected:
            changes.append(f"the bench's {instance} differs")
    changes.extend(
        f"the bench adds {instance}" for instance in found if instance not in original
    )
    return changes
