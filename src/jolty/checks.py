# Copyright 2026 Maxence Faldor
"""Structural checks: facts about a circuit's topology, found before any analysis.

The engine checks every rule on an elaborated circuit, such as a node with no DC path
to the ground or a loop of voltage sources, and returns a diagnostic for each fact it
finds, naming the instances and nets involved. A check then reports each diagnostic at
its rule's level: `error` when the fact surely breaks the circuit, `warn` when it may,
and `ignore` to leave it out. Settings and options set levels over the rules' defaults.

Examples:
    >>> rules()["no-dc-path"].level
    <Level.ERROR: 'error'>
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Final, override

from pydantic import BaseModel, ConfigDict, Field

from jolty import _core, ir
from jolty.errors import JoltyError
from jolty.settings import ALL, Level


@dataclass(frozen=True, slots=True)
class Rule:
    """One fact a check can state: its name, its default level, and its condition."""

    name: str
    level: Level
    condition: str


PHASE_MARGIN: Final = Rule(
    "phase-margin",
    Level.WARN,
    "A stability analysis's loop gain falls through 0 dB with a phase margin below "
    "45°: the loop's step response overshoots by more than about a fifth and rings, "
    "and the margin left for parts and loads that shift the phase is small. The "
    "diagnostic names the probe, the margin and the crossover frequency.",
)

GAIN_MARGIN: Final = Rule(
    "gain-margin",
    Level.WARN,
    "A stability analysis's loop gain is above -6 dB where its phase, unwrapped from "
    "the lowest frequency, first falls through -180°: less than doubling the loop's "
    "gain makes it oscillate. The diagnostic names the probe and the margin.",
)

DERATED: Final = 0.8
"""The fraction of a rating past which the rule derating states a stress."""

OVER_RATING: Final = Rule(
    "over-rating",
    Level.ERROR,
    "A stress a device's rating limits, the voltage of one of its terminals over "
    "another, the current into one, the power it dissipates or its temperature, lies "
    "beyond the rating's bounds at an operating point or at any time of a transient "
    "the plan solves: the part's maker does not warrant it there, and it may fail. "
    "Each rating raises one diagnostic per analysis and bound, at the furthest point.",
)

DERATING: Final = Rule(
    "derating",
    Level.WARN,
    "A stress a device's rating limits lies within the rating but beyond 80 % of a "
    "bound away from zero, a temperature's from 0 °C, at an operating point or at any "
    "time of a transient the plan solves: the part runs without the margin derating "
    "keeps for heat, ageing and its spread. Each rating raises one diagnostic per "
    "analysis and bound, at the furthest point.",
)

ON_RESULTS: Final = (PHASE_MARGIN, GAIN_MARGIN, OVER_RATING, DERATING)
"""The rules on analyses' results, which a simulation checks once it has them."""


@cache
def rules() -> dict[str, Rule]:
    """Every rule, by name, in the order a check reports them.

    The engine's come first: on the circuit's structure, and at its operating point.
    Those on analyses' results follow.
    """
    engine = {
        name: Rule(name, Level(level), condition)
        for name, level, condition in _core.rules()
    }
    return engine | {rule.name: rule for rule in ON_RESULTS}


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Instance(_Model):
    """An instance a diagnostic names, and the script line that placed it."""

    path: str = Field(description="The instance's path, such as 'X1/C1'.")
    source: ir.Source | None = Field(
        default=None, description="The script line that placed it."
    )

    @override
    def __str__(self) -> str:
        """The path, then the script line, such as ``X1/C1, filter.py:12 in rc()``."""
        return self.path if self.source is None else f"{self.path}, {self.source}"


class Diagnostic(_Model):
    """One fact a rule states about a circuit."""

    rule: str = Field(description="The rule's name, such as 'no-dc-path'.")
    level: Level = Field(description="'error' or 'warn'.")
    message: str = Field(description="What is true, naming instances and nets.")
    instances: tuple[Instance, ...] = Field(description="The instances involved.")
    nets: tuple[str, ...] = Field(description="The nets involved, by path.")

    def describe(self) -> str:
        """The diagnostic in a compiler's form, one instance per line."""
        lines = [f"{self.level}[{self.rule}]: {self.message}"]
        lines.extend(f"  = {instance}" for instance in self.instances)
        return "\n".join(lines)


class Report(_Model):
    """What a check found: its diagnostics, errors first, and how many of each level."""

    diagnostics: tuple[Diagnostic, ...] = Field(
        description="The diagnostics, errors first, up to the configured limit."
    )
    errors: int = Field(description="Diagnostics at the level error.")
    warnings: int = Field(description="Diagnostics at the level warn.")
    unlisted: int = Field(description="Diagnostics past the limit, left out.")

    @property
    def ok(self) -> bool:
        """Whether the check found no error."""
        return self.errors == 0

    def describe(self) -> str:
        """The report as text: each diagnostic, then the counts."""
        counts = f"{_plural(self.errors, 'error')}, {_plural(self.warnings, 'warning')}"
        if self.unlisted:
            counts += f" ({self.unlisted} not listed)"
        return "\n\n".join(
            [*(diagnostic.describe() for diagnostic in self.diagnostics), counts]
        )


def problem(found: _core.ProblemOutcome) -> str:
    """A problem the engine `found`, in a compiler's form.

    The form is what is wrong, then the instance and the script line that placed it, if
    the problem is about one.

    Examples:
        >>> source = {"file": "amp.py", "line": 9, "function": "amp"}
        >>> found = {
        ...     "message": "resistance must be positive, not -1.0",
        ...     "instance": {"path": "R1", "source": source},
        ... }
        >>> print(problem(found))
        error: resistance must be positive, not -1.0
          = R1, amp.py:9 in amp()
    """
    text = f"error: {found['message']}"
    if found["instance"] is not None:
        text += f"\n  = {_instance(found['instance'])}"
    return text


def diagnostics(found: Iterable[_core.DiagnosticOutcome]) -> tuple[Diagnostic, ...]:
    """The diagnostics the engine `found`, each at its rule's default level."""
    known = rules()
    return tuple(
        Diagnostic(
            rule=diagnostic["rule"],
            level=known[diagnostic["rule"]].level,
            message=diagnostic["message"],
            instances=tuple(
                _instance(instance) for instance in diagnostic["instances"]
            ),
            nets=tuple(diagnostic["nets"]),
        )
        for diagnostic in found
    )


def _instance(found: _core.InstanceOutcome) -> Instance:
    source = found["source"]
    return Instance(
        path=found["path"],
        source=None if source is None else ir.Source.model_validate(source),
    )


def levels(
    configured: Mapping[str, Level],
    *,
    error: Iterable[str] = (),
    warn: Iterable[str] = (),
    ignore: Iterable[str] = (),
) -> dict[str, Level]:
    """The level of every rule: its default, then the configured levels, then options.

    In `configured`, as in each option, ``all`` sets every rule, and a rule's own entry
    overrides it.

    Raises:
        JoltyError: When a name is no rule, or the options give a rule two levels.
    """
    known = rules()
    chosen = {name: rule.level for name, rule in known.items()}
    chosen = _apply(chosen, configured, "the settings")
    options = {Level.ERROR: error, Level.WARN: warn, Level.IGNORE: ignore}
    given: dict[str, Level] = {}
    for level, names in options.items():
        for name in names:
            if given.setdefault(name, level) != level:
                message = f"--{given[name]} and --{level} both name {name}"
                raise JoltyError(message)
    return _apply(chosen, given, "the options")


def _apply(
    chosen: dict[str, Level], given: Mapping[str, Level], source: str
) -> dict[str, Level]:
    unknown = sorted(set(given) - set(chosen) - {ALL})
    if unknown:
        listed, rules_ = ", ".join(unknown), ", ".join(chosen)
        message = f"{source} name {listed}, which no rule is; the rules are {rules_}"
        raise JoltyError(message)
    everything = given.get(ALL)
    return {
        name: given.get(name, everything if everything is not None else level)
        for name, level in chosen.items()
    }


def report(
    diagnostics: Iterable[Diagnostic], chosen: Mapping[str, Level], limit: int
) -> Report:
    """The report of `diagnostics` at the `chosen` levels, listing `limit` of them."""
    leveled = [
        diagnostic.model_copy(update={"level": chosen[diagnostic.rule]})
        for diagnostic in diagnostics
        if chosen[diagnostic.rule] is not Level.IGNORE
    ]
    leveled.sort(key=lambda diagnostic: diagnostic.level is not Level.ERROR)
    errors = sum(diagnostic.level is Level.ERROR for diagnostic in leveled)
    return Report(
        diagnostics=tuple(leveled[:limit]),
        errors=errors,
        warnings=len(leveled) - errors,
        unlisted=max(0, len(leveled) - limit),
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"
