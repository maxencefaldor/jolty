# Copyright 2026 Maxence Faldor
"""The cost-down task's scorer, which the agent never sees.

It builds the agent's ``design.py``, simulates it with the plan and specs of the given
``bias.py``, checks that the bench outside ``X1`` is the original's, and prices ``X1``
with the given catalog. The work meets the task when the simulation meets its plan and
the design costs less than the original. The ngspice arm's ``design.cir`` and
``tolerances.json`` are built as ``netlist.py`` builds them: the original bench, with
the stage they define in place of the original's.
"""

import shutil
import tempfile
from pathlib import Path

from arms import Arm
from scoring import simulate
from verdict import Verdict

from jolty import catalog, ir
from jolty.errors import JoltyError

INSTANCE = "X1"
"""The instance whose parts the cost-down task prices."""


def score(task: Path, work: Path, arm: Arm) -> Verdict:
    """The verdict on the cost-down `work` against `task`, as `arm` writes it."""
    given = task / "given"
    parts = catalog.Catalog.load(given / "catalog.json")
    reference, reference_circuit = simulate(given / "bias.py", given / "bias.py")
    baseline = catalog.bill(reference_circuit, parts, INSTANCE).total
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
    reference_unpriced = {
        each.instance
        for each in catalog.bill(reference_circuit, parts, INSTANCE).unpriced
    }
    reasons.extend(
        f"{each.instance} is unpriced: {each.reason}"
        for each in priced.unpriced
        if each.instance not in reference_unpriced
    )
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

    The jolty arm's is its own ``design.py``. The ngspice arm's is ``netlist.py``,
    copied into `directory` as ``design.py`` beside the given ``bias.py`` and the
    agent's ``design.cir`` and ``tolerances.json``, or none without a ``design.cir``.
    """
    if arm is Arm.JOLTY:
        return work / "design.py"
    netlist, tolerances = (work / name for name in NETLIST)
    if not netlist.is_file():
        return directory / netlist.name
    shutil.copy(task / "netlist.py", directory / "design.py")
    shutil.copy(task / "given" / "bias.py", directory / "bias.py")
    for path in (netlist, tolerances):
        if path.is_file():
            shutil.copy(path, directory / path.name)
    return directory / "design.py"


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
