# Copyright 2026 Maxence Faldor
"""Score an agent's work on a benchmark task, independently of the agent.

Usage: ``uv run python benchmarks/score.py TASK WORK``, where ``TASK`` is a task's
directory, such as ``benchmarks/cost_down``, and ``WORK`` the directory the agent worked
in. It prints the verdict as JSON and exits with 0 when the work meets the task.

Each task's own ``scorer.py``, which the agent never sees, defines ``score(task, work)``
and returns a `verdict.Verdict`.
"""

import importlib.util
import sys
from pathlib import Path

from verdict import Verdict


def score(task: Path, work: Path) -> Verdict:
    """The verdict of `task`'s own scorer on `work`."""
    location = task / "scorer.py"
    spec = importlib.util.spec_from_file_location(f"{task.name}_scorer", location)
    if spec is None or spec.loader is None:
        message = f"{location} is not a scorer"
        raise ImportError(message)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verdict = module.score(task, work)
    if not isinstance(verdict, Verdict):
        message = f"{location} returned no verdict"
        raise TypeError(message)
    return verdict


def main() -> None:
    """Scores the work in the directory given against the task given."""
    task, work = (Path(argument) for argument in sys.argv[1:3])
    verdict = score(task, work)
    print(verdict.model_dump_json(indent=2))
    raise SystemExit(0 if verdict.met else 1)


if __name__ == "__main__":
    main()
