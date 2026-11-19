# Copyright 2026 Maxence Faldor
"""Scoring verdict returned by every agent benchmark task scorer.

A verdict records whether the submitted work meets the task requirements, the measured
metrics, and any structured shortfalls. Each shortfall carries a `Kind` category
assigned directly by the scorer based on the failure stage: missing deliverable, build
failure, elaboration failure, circuit-check rule error, analysis failure, unmet task
specification, or internal scorer error.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Kind(StrEnum):
    """Failure category assigned to a shortfall by the scoring stage where it arose."""

    MISSING = "missing"
    """Deliverable is absent: no design file or required declaration exists."""
    UNBUILT = "unbuilt"
    """Design fails to build: the script errors or yields no component, jolty cannot
    parse the netlist or declaration, or the design does not fit the test bench."""
    UNELABORATED = "unelaborated"
    """Engine cannot elaborate the circuit, as reported by jolty elaboration issues."""
    RULE = "rule"
    """A jolty circuit-check rule reports an error in the circuit."""
    ANALYSIS = "analysis"
    """A circuit analysis fails, such as an operating point that fails to converge."""
    SPEC = "spec"
    """Design fails to meet a task requirement named by the shortfall spec field."""
    SCORER = "scorer"
    """Scorer could not judge the submission because the scorer crashed or the
    reference baseline failed."""


class Shortfall(BaseModel):
    """Single task requirement or scoring-stage failure with its category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    reason: str = Field(description="Human-readable explanation of the shortfall.")
    spec: str | None = Field(
        default=None,
        description="Task requirement name for a spec shortfall, such as a measure "
        "name, 'parts', or 'yield'.",
    )
    design: str | None = Field(
        default=None,
        description="Design identifier for multi-design tasks such as AnalogCoder's "
        "p3, or null for single-design tasks.",
    )

    def describe(self) -> str:
        """Formats the shortfall reason, prefixed with the design name if present."""
        return self.reason if self.design is None else f"{self.design}: {self.reason}"


class Verdict(BaseModel):
    """Structured outcome produced by scoring an agent submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    met: bool = Field(
        description="Boolean indicating whether the submission meets all requirements."
    )
    measures: dict[str, float] = Field(
        default_factory=dict,
        description="Named scalar measurements collected by the scorer, such as cost "
        "or error.",
    )
    shortfalls: tuple[Shortfall, ...] = Field(
        default=(),
        description="Structured shortfalls explaining every unmet requirement.",
    )
    report: str = Field(
        default="", description="Simulation report text emitted by jolty simulate."
    )

    @property
    def reasons(self) -> tuple[str, ...]:
        """Formatted shortfall descriptions, one entry per shortfall."""
        return tuple(shortfall.describe() for shortfall in self.shortfalls)
