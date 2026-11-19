# Copyright 2026 Maxence Faldor
"""Zero-cost stand-in agent that writes known passing designs to test the harness.

Usage: ``stub.py TASK ARM PROMPT``, invoked inside the agent working directory. The
stub ignores ``PROMPT`` and writes known passing designs for tasks with repository
references. On the jolty arm, the stub writes ``plan.py`` (an operating-point plan
that also runs a transient via the Python API when executed directly). The stub then
exercises every jolty frontend so all logging paths run: the CLI, the stdio MCP
server, and direct Python execution of ``plan.py``. On the ngspice arm, or for tasks
without a stored reference, the stub runs ``--version`` on the arm simulator.
Executables are resolved from ``PATH``, and stdout emits a Claude Code ``stream-json``
transcript with one turn per file write or command.

Supported tasks:

- ``cost_down``: uses the original bias circuit with 1 % E96 resistors instead of
  0.1 % parts (for ngspice, writes ``stage.cir`` and ``tolerances.json``).
- ``analogcoder``: writes the reference circuits from ``reference.py`` (for ngspice,
  exports each circuit as a SPICE subcircuit alongside ``levels.json``).

No designs are written for other tasks. The patented cubic generator in ``patent.py``
scores exactly 1.0 in native Python, whereas SPICE netlist round-tripping introduces
rounding drift above 1.0 in some corners.
"""

import asyncio
import json
import os
import runpy
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from arms import SUFFIXES, Arm
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import TextContent

from jolty import Component, Role, circuit
from jolty.spice import export

TOLERANCE: Final = 0.01
"""Resistor tolerance (1 % E96) written for the cost-down stub design."""
SIMULATORS: Final = {Arm.JOLTY: "jolty", Arm.NGSPICE: "ngspice"}
"""Simulator executable name per arm used for version-check fallback runs."""
PLAN: Final = "plan.py"
"""Filename of the generated simulation plan used on the jolty arm."""
PLANNED: Final = '''\
"""The operating point of {design}, and run as a script, its transient."""

import runpy

from jolty import Plan, Simulation
from jolty.plan import OperatingPoint, Transient
from jolty.units import ns


def plan() -> Plan:
    """The operating point."""
    return Plan({{"op": OperatingPoint()}})


if __name__ == "__main__":
    build = runpy.run_path({design!r})["build"]
    Simulation(build()).run(Transient(1 * ns))
'''
"""Template source text for `PLAN`, parameterized by the target design filename."""


def subcircuit(made: Component, name: str) -> str:
    """Exports `made` as a SPICE ``.subckt`` block named `name`.

    Excludes the ground port from the pin list because SPICE binds ground to node
    ``0``.
    """
    pins = [port.name for port in made.definition.ports if port.role is not Role.GROUND]
    lines = export.netlist(circuit(made))
    return "\n".join([f".subckt {name} {' '.join(pins)}", *lines, f".ends {name}", ""])


def _cost_down(task: Path, arm: Arm) -> dict[str, str]:
    if arm is Arm.JOLTY:
        bias = (task / "given" / "bias.py").read_text(encoding="utf-8")
        return {
            "design.py": bias.replace("TOLERANCE = 0.001", f"TOLERANCE = {TOLERANCE}")
        }
    stage = (task / "ngspice" / "given" / "stage.cir").read_text(encoding="utf-8")
    resistors = ("R1", "R2", "RE", "RC")
    tolerances = dict.fromkeys(resistors, TOLERANCE)
    return {"design.cir": stage, "tolerances.json": json.dumps(tolerances, indent=2)}


def _analogcoder(task: Path, arm: Arm) -> dict[str, str]:
    path = task / "reference.py"
    reference = runpy.run_path(str(path))
    numbers = range(1, 25)
    if arm is Arm.JOLTY:
        return {
            f"p{number}.py": (
                "import runpy\n"
                f"_reference = runpy.run_path({str(path)!r})\n"
                f"LEVELS = _reference['LEVELS'].get({number}, {{}})\n"
                "def build():\n"
                f"    return _reference['build']({number})\n"
            )
            for number in numbers
        }
    written = {
        f"p{number}.cir": subcircuit(reference["build"](number), f"p{number}")
        for number in numbers
    }
    levels = {f"p{number}": reference["LEVELS"].get(number, {}) for number in numbers}
    return written | {"levels.json": json.dumps(levels, indent=2)}


DESIGNS: Final = {"cost_down": _cost_down, "analogcoder": _analogcoder}
"""Mapping from task name to generator function producing known passing files."""


def _turns(task: Path, arm: Arm) -> Iterator[dict[str, object]]:
    """Yields synthetic Claude Code transcript events as each stub action completes."""
    started = time.perf_counter()
    yield {
        "type": "system",
        "subtype": "init",
        "model": "stub",
        "claude_code_version": "stub",
        "cwd": str(Path.cwd()),
    }
    write = DESIGNS.get(task.name)
    files = {} if write is None else write(task, arm)
    designs = [name for name in files if name.endswith(SUFFIXES[arm])]
    turns = 0
    for name, text in files.items():
        turns += 1
        yield from _written(f"stub-{turns}", name, text)
    if arm is Arm.JOLTY and designs:
        turns += 1
        yield from _written(f"stub-{turns}", PLAN, PLANNED.format(design=designs[0]))
        simulate = ["jolty", "simulate", designs[0], "--plan", PLAN]
        quiet = ["--no-schematic", "--no-plot"]
        turns += 1
        yield from _ran(f"stub-{turns}", [*simulate, *quiet])
        turns += 1
        yield from _called(
            f"stub-{turns}",
            "simulate_circuit",
            {"script": designs[0], "plan": PLAN, "schematic": False, "plot": False},
        )
        turns += 1
        yield from _ran(f"stub-{turns}", ["python", PLAN])
    else:
        turns += 1
        yield from _ran(f"stub-{turns}", [SIMULATORS[arm], "--version"])
    yield {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": f"I wrote the known good design, {', '.join(files) or 'none'}.",
        "num_turns": turns,
        "duration_ms": (time.perf_counter() - started) * 1e3,
        "total_cost_usd": 0.0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _written(call_id: str, name: str, text: str) -> Iterator[dict[str, object]]:
    """Writes `text` to `name` and yields assistant and result events for `call_id`."""
    Path(name).write_text(text, encoding="utf-8")
    yield _message(call_id, "Write", {"file_path": name})
    yield _result(call_id, f"wrote {name}")


def _ran(call_id: str, command: list[str]) -> Iterator[dict[str, object]]:
    """Executes `command` from ``PATH`` and yields transcript events for `call_id`."""
    yield _message(call_id, "Bash", {"command": " ".join(command)})
    ran = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603 - arm executables on prepared PATH
    yield _result(call_id, ran.stdout + ran.stderr)


def _called(
    call_id: str, tool: str, arguments: dict[str, object]
) -> Iterator[dict[str, object]]:
    """Invokes `tool` on ``jolty-mcp`` and yields transcript events for `call_id`.

    Launches ``jolty-mcp`` from ``PATH`` in the current working directory with the
    active environment, matching Claude Code's ``mcp__jolty__<tool>`` naming.
    """
    server = StdioServerParameters(
        command="jolty-mcp", env=dict(os.environ), cwd=str(Path.cwd())
    )

    async def call() -> str:
        async with Client(server) as client:
            answered = await client.call_tool(tool, arguments)
        return "".join(
            block.text for block in answered.content if isinstance(block, TextContent)
        )

    yield _message(call_id, f"mcp__jolty__{tool}", arguments)
    yield _result(call_id, asyncio.run(call()))


def _message(
    call_id: str, tool: str, arguments: dict[str, object]
) -> dict[str, object]:
    """Builds an assistant transcript event for `call_id` invoking `tool`."""
    return {
        "type": "assistant",
        "message": {
            "id": call_id,
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"{call_id}-call",
                    "name": tool,
                    "input": arguments,
                }
            ],
        },
    }


def _result(call_id: str, output: str) -> dict[str, object]:
    """Builds a user transcript event carrying `output` for tool call `call_id`."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"{call_id}-call",
                    "content": output,
                }
            ],
        },
    }


def main() -> None:
    """Writes known passing files for the task and arm arguments and prints events."""
    task, arm = Path(sys.argv[1]), Arm(sys.argv[2])
    for event in _turns(task, arm):
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
