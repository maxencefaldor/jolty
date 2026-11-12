# Copyright 2026 Maxence Faldor
"""Run one agent on a benchmark task, on its own, then score what it did.

Usage: ``uv run python benchmarks/run.py TASK [AGENT ...]``, where ``TASK`` is a task's
directory, such as ``benchmarks/cost_down``, and ``AGENT`` the command that runs the
agent, Claude Code by default. The runner copies the files of the task's ``given``
directory, and only those, into a fresh working directory, runs the agent there with the
task's ``task.md`` as its prompt and jolty's command line and MCP server on its path,
keeps what it printed in ``transcript.txt``, and scores the directory with the task's
own scorer, which the agent never sees. It prints the working directory and the
verdict, and exits with 0 when the work meets the task.

With ``-`` as the agent, it only prepares the working directory and prints it, for an
agent started some other way; ``score.py`` scores the directory afterwards.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from score import score

MCP: dict[str, object] = {"mcpServers": {"jolty": {"command": "jolty-mcp"}}}
"""The MCP configuration that gives the agent jolty's server."""


def default_agent() -> list[str]:
    """Claude Code, printing its work, running on its own, with jolty's MCP server."""
    return [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--mcp-config",
        json.dumps(MCP),
    ]


def prepare(task: Path, work: Path) -> str:
    """Copies `task`'s given files into `work`, and returns the task's prompt."""
    given = task / "given"
    if given.is_dir():
        for path in given.iterdir():
            if path.is_file():
                shutil.copy(path, work / path.name)
    return (task / "task.md").read_text(encoding="utf-8")


def run(task: Path, agent: list[str], work: Path) -> int:
    """Runs `agent` on `task` in `work`, keeps its transcript there, and scores it."""
    prompt = prepare(task, work)
    scripts = Path(sys.executable).parent
    environment = {**os.environ, "PATH": f"{scripts}{os.pathsep}{os.environ['PATH']}"}
    with (work / "transcript.txt").open("w", encoding="utf-8") as transcript:
        finished = subprocess.run(  # noqa: S603 - the agent is the command given
            [*agent, prompt],
            cwd=work,
            env=environment,
            stdout=transcript,
            stderr=subprocess.STDOUT,
            check=False,
        )
    verdict = score(task, work)
    print(f"worked in {work}, agent exited with {finished.returncode}")
    print(verdict.model_dump_json(indent=2))
    return 0 if verdict.met else 1


def main() -> None:
    """Runs the agent given, or Claude Code, on the task given."""
    task = Path(sys.argv[1]).resolve()
    agent = sys.argv[2:] or default_agent()
    work = Path(tempfile.mkdtemp(prefix=f"jolty-{task.name}-"))
    if agent == ["-"]:
        prepare(task, work)
        print(work)
        raise SystemExit(0)
    raise SystemExit(run(task, agent, work))


if __name__ == "__main__":
    main()
