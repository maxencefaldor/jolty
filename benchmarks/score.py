# Copyright 2026 Maxence Faldor
"""Score an agent's work on a benchmark task, independently of the agent.

Usage: ``uv run python benchmarks/score.py TASK WORK [--arm ARM] [--verdict FILE]``,
where ``TASK`` is a task's directory, such as ``benchmarks/cost_down``, ``WORK`` the
directory the agent worked in, and ``ARM`` the tools it had: ``jolty``, by default,
when it wrote its design as a script, such as ``design.py``, or ``ngspice`` when it
wrote it as a netlist, such as ``design.cir``. It prints the verdict as JSON, writes it
to ``FILE`` too when given, and exits with 0 when the work meets the task.

Each task's own ``scorer.py``, which the agent never sees, defines
``score(task, work, arm)`` and returns a `verdict.Verdict`. It reads the design as the
arm writes it, then judges both arms by the same code.
"""

import argparse
import importlib.util
from pathlib import Path

from arms import Arm
from pydantic import BaseModel
from verdict import Verdict


def score(task: Path, work: Path, arm: Arm = Arm.JOLTY) -> Verdict:
    """The verdict of `task`'s own scorer on `work`, as `arm` wrote it."""
    location = task / "scorer.py"
    spec = importlib.util.spec_from_file_location(f"{task.name}_scorer", location)
    if spec is None or spec.loader is None:
        message = f"{location} is not a scorer"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verdict = module.score(task, work, arm)
    if not isinstance(verdict, Verdict):
        message = f"{location} returned no verdict"
        raise TypeError(message)
    return verdict


class Arguments(BaseModel):
    """The scorer's arguments."""

    task: Path
    work: Path
    arm: Arm
    verdict: Path | None


def main() -> None:
    """Scores the work in the directory given against the task given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task", type=Path, help="the task's directory")
    parser.add_argument("work", type=Path, help="the directory the agent worked in")
    parser.add_argument(
        "--arm", type=Arm, choices=list(Arm), default=Arm.JOLTY, help="its tools"
    )
    parser.add_argument("--verdict", type=Path, help="a file to write the verdict to")
    arguments = Arguments.model_validate(vars(parser.parse_args()))
    verdict = score(arguments.task, arguments.work, arguments.arm)
    written = verdict.model_dump_json(indent=2)
    if arguments.verdict is not None:
        arguments.verdict.write_text(written, encoding="utf-8")
    print(written)
    raise SystemExit(0 if verdict.met else 1)


if __name__ == "__main__":
    main()
