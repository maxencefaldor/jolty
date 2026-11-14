# Copyright 2026 Maxence Faldor
"""A stand-in agent, which writes a known good design: it proves the harness for free.

Usage: ``stub.py TASK ARM PROMPT``, run in the agent's working directory, as the runner
runs an agent, with the task's directory and the arm before the prompt, which it reads
no further. For the tasks whose good designs the repository holds, it writes one as the
arm writes its designs, then runs the arm's simulator once, through the path the runner
gave it, so the runner counts it. It prints a transcript as Claude Code prints its
session with ``--output-format stream-json --verbose``, with a turn for each design it
writes and one for the simulator's run.

The designs:

- ``cost_down``: the original bias of 1 % resistors, from the E96 series, where the
  original's are 0.1 %: for the ngspice arm, the given ``stage.cir``, with each
  resistor's tolerance in ``tolerances.json``.
- ``analogcoder``: the scorer's reference designs, ``reference.py``: for the ngspice
  arm, each written as a SPICE subcircuit by jolty's own writer of netlists for ngspice,
  with the inputs' levels in ``levels.json``.

It writes no design for any other task. The patented cubic generator, ``patent.py``,
meets its task at a score of exactly 1, its own: written as a netlist and read back,
it scores 1 only within rounding, and above it in some corners.
"""

import json
import runpy
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from arms import Arm

from jolty import Component, Role, circuit
from jolty.spice import export

TOLERANCE: Final = 0.01
"""The tolerance of the cost-down design's resistors: E96 parts."""
SIMULATORS: Final = {Arm.JOLTY: "jolty", Arm.NGSPICE: "ngspice"}
"""The simulator each arm has, which the stub runs once."""


def subcircuit(made: Component, name: str) -> str:
    """`made` as a SPICE library whose subcircuit `name` it is.

    Its pins are `made`'s ports but its ground, which is SPICE's node ``0``.
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
"""How the stub writes each task's known good design, by the task's name."""


def _turns(task: Path, arm: Arm) -> Iterator[dict[str, object]]:
    """The events of the stub's session, each after what it records is done."""
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
    for number, (name, text) in enumerate(files.items(), start=1):
        Path(name).write_text(text, encoding="utf-8")
        yield _message(f"stub-{number}", "Write", {"file_path": name})
        yield _result(f"stub-{number}", f"wrote {name}")
    simulator = [SIMULATORS[arm], "--version"]
    yield {
        "type": "assistant",
        "message": {
            "id": "stub-run",
            "role": "assistant",
            "content": [{"type": "text", "text": "I check the simulator runs."}],
        },
    }
    yield _message("stub-run", "Bash", {"command": " ".join(simulator)})
    ran = subprocess.run(simulator, capture_output=True, text=True, check=False)  # noqa: S603 - the arm's simulator, on the path the runner gave
    yield _result("stub-run", ran.stdout + ran.stderr)
    yield {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": f"I wrote the known good design, {', '.join(files) or 'none'}.",
        "num_turns": len(files) + 1,
        "duration_ms": (time.perf_counter() - started) * 1e3,
        "total_cost_usd": 0.0,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _message(id: str, tool: str, arguments: dict[str, str]) -> dict[str, object]:
    """An event of the model's message `id` calling `tool` with `arguments`."""
    return {
        "type": "assistant",
        "message": {
            "id": id,
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": f"{id}-call",
                    "name": tool,
                    "input": arguments,
                }
            ],
        },
    }


def _result(id: str, output: str) -> dict[str, object]:
    """An event of the tool's result for the call of the model's message `id`."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": f"{id}-call", "content": output}
            ],
        },
    }


def main() -> None:
    """Writes the known good design of the task given, as the arm given writes it."""
    task, arm = Path(sys.argv[1]), Arm(sys.argv[2])
    for event in _turns(task, arm):
        print(json.dumps(event), flush=True)


if __name__ == "__main__":
    main()
