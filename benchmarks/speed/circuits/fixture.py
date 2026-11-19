# Copyright 2026 Maxence Faldor
"""Shared fixture schema and serializer for speed benchmark circuit scripts.

A fixture pairs a circuit name with a simulation plan. Each circuit script writes the
serialized circuit to ``benchmarks/speed/circuits/<circuit>.json`` and the engine plan
to ``benchmarks/speed/plans/<fixture>.json`` (ordered analyses, options, and optional
Monte Carlo settings). Rust Criterion benchmarks read both files, while Python timings
load the circuit JSON and obtain the plan directly from the script.

Run fixture scripts from the repository root so recorded source paths stay relative, for
example ``uv run python benchmarks/speed/circuits/bjt_amp.py``.
"""

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Shared benchmark helper modules live in the parent benchmarks directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from measuring import format_json

from jolty import Component, Plan, circuit
from jolty.plan import Ac, OperatingPoint, Transient

SPEED: Final = Path(__file__).resolve().parent.parent
"""Root directory of the speed benchmark suite (``benchmarks/speed``)."""

CIRCUITS: Final = SPEED / "circuits"
"""Directory storing serialized circuit JSON files."""

PLANS: Final = SPEED / "plans"
"""Directory storing serialized engine plan JSON files."""


@dataclass(frozen=True, slots=True)
class Fixture:
    """Speed benchmark fixture pairing a circuit file stem with its simulation plan."""

    circuit: str
    plan: Plan


def engine_plan(fixture: Fixture) -> dict[str, object]:
    """Serializes `fixture.plan` into the JSON structure read by the Rust engine.

    Raises:
        TypeError: When the plan includes an analysis other than operating point, AC,
            or transient.
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
