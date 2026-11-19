# Copyright 2026 Maxence Faldor
"""What a task's scorer finds of an agent's work: the verdict every scorer returns.

A verdict says why the work falls short, if it does, as shortfalls, each of a kind the
scorer knows from where it found it, never read from its words: the work holds no
design, its design does not build, the engine cannot elaborate it, a rule finds an
error in it, an analysis of it fails, it falls short of a requirement of the task,
which the shortfall names, or the scorer itself could not judge it.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Kind(StrEnum):
    """What kind of shortfall a scorer found, by where it found it."""

    MISSING = "missing"
    """The work holds no deliverable: no design, or none of what it must declare."""
    UNBUILT = "unbuilt"
    """The design does not build: its script fails or builds no component, jolty
    cannot read its netlist or what it declares, or it does not fit its bench."""
    UNELABORATED = "unelaborated"
    """The engine cannot elaborate the design's circuit, as jolty's problems say."""
    RULE = "rule"
    """A rule of jolty's checks finds an error in the circuit."""
    ANALYSIS = "analysis"
    """An analysis of the design fails, such as an operating point no strategy finds."""
    SPEC = "spec"
    """The design falls short of a requirement of the task, which its shortfall
    names."""
    SCORER = "scorer"
    """The scorer could not judge the work: it crashed, or what it compares against
    falls short itself."""


class Shortfall(BaseModel):
    """One way the work falls short of its task, and its kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Kind
    reason: str = Field(description="Why, in words.")
    spec: str | None = Field(
        default=None,
        description="The requirement the design falls short of, for a shortfall of "
        "the kind spec, such as a measure's name, 'parts' or 'yield'.",
    )
    design: str | None = Field(
        default=None,
        description="Which of the work's designs falls short, for a task of many, "
        "such as AnalogCoder's p3; null for a task of one.",
    )

    def describe(self) -> str:
        """The shortfall in words, after the design it is of, if the task has many."""
        return self.reason if self.design is None else f"{self.design}: {self.reason}"


class Verdict(BaseModel):
    """What scoring the work found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    met: bool = Field(description="Whether the work meets the task.")
    measures: dict[str, float] = Field(
        default_factory=dict,
        description="What the scorer measured, by name, such as a cost or an error.",
    )
    shortfalls: tuple[Shortfall, ...] = Field(
        default=(), description="Why the work falls short, if it does."
    )
    report: str = Field(default="", description="What jolty simulate reported.")

    @property
    def reasons(self) -> tuple[str, ...]:
        """Why the work falls short, in words, one a shortfall."""
        return tuple(shortfall.describe() for shortfall in self.shortfalls)
