# Copyright 2026 Maxence Faldor
"""Standalone CLI for scoring an agent submission independently of the agent.

Usage: ``uv run python benchmarks/agents/score.py TASK WORK [--arm ARM] [--verdict
FILE]``. The ``TASK`` argument points to a task directory such as
``benchmarks/agents/cost_down``, and ``WORK`` points to the agent working directory.
The ``ARM`` option selects ``jolty`` (default, Python scripts) or ``ngspice`` (SPICE
netlists). The command prints the JSON verdict, writes the JSON to ``FILE`` when
requested, and exits with status 0 if the submission passes.

Each task directory provides an unexposed ``scorer.py`` defining
``score(task, work, arm)`` and returning a `verdict.Verdict`. Both arms are loaded in
their native formats and judged by identical scoring logic.
"""

import importlib.util
from pathlib import Path

from arms import Arm
from pydantic import Field
from pydantic_settings import CliPositionalArg, SettingsConfigDict
from verdict import Verdict

from jolty import cli


def score(task: Path, work: Path, arm: Arm = Arm.JOLTY) -> Verdict:
    """Loads ``scorer.py`` in `task` and returns the `Verdict` for `work` on `arm`."""
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
    """CLI command for scoring an agent submission on a benchmark task."""

    model_config = SettingsConfigDict(cli_prog_name="benchmarks/agents/score.py")

    task: CliPositionalArg[Path] = Field(
        description="Path to the benchmark task directory."
    )
    work: CliPositionalArg[Path] = Field(
        description="Path to the agent working directory."
    )
    arm: Arm = Field(
        default=Arm.JOLTY,
        description="Benchmark arm specifying the tools available to the agent.",
    )
    verdict: Path | None = Field(
        default=None,
        description="Optional output file path for writing the JSON verdict.",
    )

    def cli_cmd(self) -> None:
        """Scores the configured working directory against the configured task.

        Raises:
            SystemExit: Exits with 0 when the submission passes, and 1 otherwise.
        """
        verdict = score(self.task, self.work, self.arm)
        written = verdict.model_dump_json(indent=2)
        if self.verdict is not None:
            self.verdict.write_text(written, encoding="utf-8")
        print(written)
        raise SystemExit(0 if verdict.met else 1)


def main(arguments: list[str] | None = None) -> None:
    """Executes the scorer CLI using `arguments` or ``sys.argv``."""
    cli.run(Scorer, arguments)


if __name__ == "__main__":
    main()
