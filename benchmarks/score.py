# Copyright 2026 Maxence Faldor
"""Score an agent's work on a benchmark task, independently of the agent.

Usage: ``uv run python benchmarks/score.py TASK WORK``, where ``TASK`` is a task's
directory, such as ``benchmarks/cost_down``, and ``WORK`` the directory the agent worked
in. It prints the verdict as JSON and exits with 0 when the work meets the task.

The cost-down task's verdict builds the agent's ``design.py``, simulates it with the
plan and specs of the task's own ``bias.py``, checks that the bench outside ``X1`` is
the original's, and prices ``X1`` with the task's catalog. The work meets the task when
the simulation meets its plan and the design costs less than the original.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from jolty import catalog, files, ir, service
from jolty.settings import Settings

INSTANCE = "X1"
"""The instance whose parts the cost-down task prices."""


class Verdict(BaseModel):
    """What scoring the work found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    met: bool = Field(description="Whether the work meets the task.")
    baseline: float = Field(
        description="The original's cost, in the catalog's currency."
    )
    cost: float | None = Field(description="The design's cost; null when unscored.")
    reasons: tuple[str, ...] = Field(
        description="Why the work falls short, if it does."
    )
    report: str = Field(description="What jolty simulate reported on the design.")


def score(task: Path, work: Path) -> Verdict:
    """The verdict on the cost-down `work` against `task`."""
    parts = catalog.Catalog.load(task / "catalog.json")
    reference, reference_circuit = _simulate(task / "bias.py", task / "bias.py")
    baseline = catalog.bill(reference_circuit, parts, INSTANCE).total
    design = work / "design.py"
    if not design.is_file():
        return Verdict(
            met=False,
            baseline=baseline,
            cost=None,
            reasons=(f"there is no {design}",),
            report="",
        )
    result, built = _simulate(design, task / "bias.py")
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
        baseline=baseline,
        cost=priced.total,
        reasons=tuple(reasons),
        report=result.describe(),
    )


def _simulate(script: Path, plan: Path) -> tuple[service.SimulateResult, ir.Circuit]:
    """What ``jolty simulate`` finds of `script` with `plan`'s plan, and its circuit."""
    with tempfile.TemporaryDirectory() as directory:
        saved = Path(directory) / "circuit.jolty.json"
        request = service.SimulateRequest(
            script=script,
            plan=plan,
            output=saved,
            results=Path(directory) / "circuit.results.npz",
            schematic=False,
            plot=False,
        )
        result = asyncio.run(service.simulate(request, Settings()))
        return result, files.load_circuit(saved)


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


def main() -> None:
    """Scores the work in the directory given against the task given."""
    task, work = (Path(argument) for argument in sys.argv[1:3])
    verdict = score(task, work)
    print(verdict.model_dump_json(indent=2))
    raise SystemExit(0 if verdict.met else 1)


if __name__ == "__main__":
    main()
