# Copyright 2026 Maxence Faldor
"""What a task's scorer finds of an agent's work: the verdict every scorer returns."""

from pydantic import BaseModel, ConfigDict, Field


class Verdict(BaseModel):
    """What scoring the work found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    met: bool = Field(description="Whether the work meets the task.")
    measurements: dict[str, float] = Field(
        default_factory=dict,
        description="What the scorer measured, by name, such as a cost or an error.",
    )
    reasons: tuple[str, ...] = Field(
        default=(), description="Why the work falls short, if it does."
    )
    report: str = Field(default="", description="What jolty simulate reported.")
