# Copyright 2026 Maxence Faldor
"""What the fixtures' scripts share: a fixture, and the files it is written to.

A fixture is a circuit and a plan of analyses to run on it. Its script writes the
circuit in the circuit format to ``benchmarks/speed/circuits/<circuit>.json``, and the
plan in the engine's format to ``benchmarks/speed/plans/<fixture>.json``: the analyses
in order, each by name, the options, and a Monte Carlo analysis's samples and seed. The
Rust benchmarks read both files; the Python timing reads the circuit and takes the plan
from the script.

Run a fixture's script from the repository root, so the instances' script lines are
written relative to it, such as ``uv run python benchmarks/speed/circuits/bjt_amp.py``.
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# The suites' measuring helpers lie beside the suites' directories.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measuring import format_json

from jolty import Component, Plan, circuit
from jolty.plan import Ac, OperatingPoint, Transient

SPEED: Final = Path(__file__).resolve().parent.parent
"""The speed suite's directory, ``benchmarks/speed``."""

CIRCUITS: Final = SPEED / "circuits"
"""Where the circuits are written."""

PLANS: Final = SPEED / "plans"
"""Where the plans are written."""


@dataclass(frozen=True, slots=True)
class Fixture:
    """A circuit, by the name of its file, and the plan run on it."""

    circuit: str
    plan: Plan


def engine_plan(fixture: Fixture) -> dict[str, object]:
    """The fixture's plan in the engine's format, as its plan file holds it.

    Raises:
        TypeError: When the plan runs an analysis other than an operating point, a
            small-signal response or a transient, which no fixture needs.
    """
    analyses: list[object] = []
    for name, analysis in fixture.plan.analyses.items():
        if not isinstance(analysis, OperatingPoint | Ac | Transient):
            message = f"a fixture runs no {type(analysis).__name__}"
            raise TypeError(message)
        analyses.append([name, analysis.to_json()])
    monte_carlo = fixture.plan.monte_carlo
    return {
        "circuit": fixture.circuit,
        "analyses": analyses,
        "options": fixture.plan.options.to_json(),
        "monte_carlo": None
        if monte_carlo is None
        else {"samples": monte_carlo.samples, "seed": monte_carlo.seed},
    }


def write(circuits: Mapping[str, Component], fixtures: Mapping[str, Fixture]) -> None:
    """Writes each of `circuits`, by its file's name, and each fixture's plan."""
    for name, top in circuits.items():
        written = circuit(top).model_dump_json()
        (CIRCUITS / f"{name}.json").write_text(written + "\n", encoding="utf-8")
    for name, fixture in fixtures.items():
        written = format_json(engine_plan(fixture))
        (PLANS / f"{name}.json").write_text(written + "\n", encoding="utf-8")
