# Copyright 2026 Maxence Faldor
"""The cost-down task's scorer, which the agent never sees.

It builds the agent's ``design.py``, simulates it with the plan and specs of the given
``bias.py``, checks that the bench outside ``X1`` is the original's, and prices ``X1``
with the given catalog. The work meets the task when the simulation meets its plan and
the design costs less than the original. Each way it falls short of the plan is jolty's
own reason, of its kind, such as a requirement that fails or a yield below its target;
the bench, the catalog and the cost are the specs ``bench``, ``catalog`` and ``cost``.
The ngspice arm's ``design.cir`` and ``tolerances.json`` are built as ``netlist.py``
builds them: the original bench, with the stage they define in place of the original's.

A bill leaves out the parts no part of the catalog prices, such as the given
transistor, so the design's total compares with the original's only when it leaves out
no more than the original does. Each part the original leaves out excuses one of the
design's that is the same part, as the arm's agent got it, whatever its name and
wherever it sits: of the same model, with the same parameters, tolerances, ratings and
part number.
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
"""The instance whose parts the cost-down task prices."""


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """The verdict on the cost-down `work` against `task`, as `arm` writes it."""
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
            # jolty simulate reports what it finds of a circuit, and raises only when
            # the script makes none: it fails, runs out of time or its process ends.
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
        reason = "the original does not meet its own plan"
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
"""The ngspice arm's design: its stage, and its resistors' tolerances."""


def _script(task: Path, work: Path, arm: Arm, directory: Path) -> Path:
    """The build script of the design `arm` writes in `work`, which may be missing.

    The jolty arm's is its own ``design.py``. The ngspice arm's is ``netlist.py`` in
    `directory`, as `_netlist` writes it of the agent's ``design.cir`` and
    ``tolerances.json``, or none without a ``design.cir``.
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
    """``netlist.py``, copied into `directory` as ``design.py`` to build `netlist`.

    Beside it go the given ``bias.py``, `netlist` as ``design.cir``, and `tolerances`,
    when it is a file, as ``tolerances.json``.
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
    """The parts `original`, the bill of `reference`, leaves out, as `arm` got them.

    They are the original's own, for the jolty arm, and those of the given stage
    netlist read by jolty, for the ngspice arm, at the paths the bill names.
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
    """Why each part `priced`, the bill of `built`, leaves out is not excused.

    Each of `excused` excuses one part that is the same part, whatever its name and
    wherever it sits. A part not excused falls short of the spec ``catalog``.
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
    """What `instance` is, whatever its name and wherever it sits.

    It is `instance` without its id, the nets of its terminals and the script line that
    placed it: its model, parameters, tolerances, ratings and part number.
    """
    return instance.model_copy(
        update={"name": "", "terminals": {}, "source_location": None}
    )


def _bench_changes(reference: ir.Circuit, built: ir.Circuit) -> list[str]:
    """How the bench of `built`, its top component but the stage's parts, differs."""

    def bench(circuit: ir.Circuit) -> dict[str, ir.Instance]:
        return {
            instance.name: instance.model_copy(update={"source_location": None})
            for instance in circuit.components[circuit.top].instances
        }

    original, found = bench(reference), bench(built)
    changes = []
    for id, instance in original.items():
        placed = found.get(id)
        if placed is None:
            changes.append(f"the bench places no {id}")
        elif id == INSTANCE:
            if placed.terminals != instance.terminals:
                changes.append(f"{id}'s ports join other nets")
        elif placed != instance:
            changes.append(f"the bench's {id} differs")
    changes.extend(f"the bench adds {id}" for id in found if id not in original)
    return changes
