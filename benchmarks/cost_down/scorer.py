# Copyright 2026 Maxence Faldor
"""The cost-down task's scorer, which the agent never sees.

It builds the agent's ``design.py``, simulates it with the plan and specs of the given
``bias.py``, checks that the bench outside ``X1`` is the original's, and prices ``X1``
with the given catalog. The work meets the task when the simulation meets its plan and
the design costs less than the original. The ngspice arm's ``design.cir`` and
``tolerances.json`` are built as ``netlist.py`` builds them: the original bench, with
the stage they define in place of the original's.

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
from scoring import simulate
from verdict import Verdict

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
                measurements={"baseline": baseline},
                reasons=(f"there is no {work / design.name}",),
            )
        try:
            result, built = simulate(design, given / "bias.py")
        except JoltyError as error:
            return Verdict(
                met=False,
                measurements={"baseline": baseline},
                reasons=(f"the design does not build: {error}",),
            )
    reasons = []
    if not result.ok:
        reasons.append("the design does not meet the plan of bias.py")
    if not reference.ok:
        reasons.append("the original does not meet its own plan")
    reasons.extend(_bench_changes(reference_circuit, built))
    priced = catalog.bill(built, parts, INSTANCE)
    excused = _given(task, arm, reference_circuit, original)
    reasons.extend(_unpriced(priced, built, excused))
    if priced.total >= baseline:
        reasons.append(
            f"the design costs {priced.total:.6g}, not less than {baseline:.6g}"
        )
    return Verdict(
        met=not reasons,
        measurements={"baseline": baseline, "cost": priced.total},
        reasons=tuple(reasons),
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
) -> list[str]:
    """Why each part `priced`, the bill of `built`, leaves out is not excused.

    Each of `excused` excuses one part that is the same part, whatever its name and
    wherever it sits.
    """
    placed = dict(catalog.built(built, built.top))
    left = [_part(each) for each in excused]
    reasons = []
    for each in priced.unpriced:
        part = _part(placed[each.instance])
        if part in left:
            left.remove(part)
        else:
            reasons.append(f"{each.instance} is unpriced: {each.reason}")
    return reasons


def _part(instance: ir.Instance) -> ir.Instance:
    """What `instance` is, whatever its name and wherever it sits.

    It is `instance` without its id, the nets of its terminals and the script line that
    placed it: its model, parameters, tolerances, ratings and part number.
    """
    return instance.model_copy(update={"id": "", "terminals": {}, "source": None})


def _bench_changes(reference: ir.Circuit, built: ir.Circuit) -> list[str]:
    """How the bench of `built`, its top component but the stage's parts, differs."""

    def bench(circuit: ir.Circuit) -> dict[str, ir.Instance]:
        return {
            instance.id: instance.model_copy(update={"source": None})
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
