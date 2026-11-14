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

import importlib.util
from pathlib import Path

from arms import Arm
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsConfigDict
from verdict import Verdict

from jolty import cli


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


class Scorer(cli.CommandLine):
    """Score an agent's work on a benchmark task, independently of the agent."""

    model_config = SettingsConfigDict(cli_prog_name="benchmarks/score.py")

    task: CliPositionalArg[Path] = Field(description="The task's directory.")
    work: CliPositionalArg[Path] = Field(
        description="The directory the agent worked in."
    )
    arm: Arm = Field(default=Arm.JOLTY, description="The tools the agent had.")
    verdict: Path | None = Field(
        default=None, description="A file to write the verdict to."
    )

    def cli_cmd(self) -> None:
        """Scores the work in the directory given against the task given.

        Raises:
            SystemExit: With 0 when the work meets the task, and 1 when it falls
                short.
        """
        verdict = score(self.task, self.work, self.arm)
        written = verdict.model_dump_json(indent=2)
        if self.verdict is not None:
            self.verdict.write_text(written, encoding="utf-8")
        print(written)
        raise SystemExit(0 if verdict.met else 1)


def main(arguments: list[str] | None = None) -> None:
    """Runs the command line, `arguments` or the process's own."""
    cli.run(Scorer, arguments)


if __name__ == "__main__":
    main()
