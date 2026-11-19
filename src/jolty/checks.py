# Copyright 2026 Maxence Faldor
"""Structural and post-simulation circuit checks and diagnostic reporting.

The engine evaluates structural rules on an elaborated circuit, such as missing DC
paths to ground or voltage-source loops, and emits diagnostics naming the affected
instances and nets. Each diagnostic is reported at the configured level for its rule:
`error`, `warn`, or `ignore`. Settings and CLI options override rule defaults.

Examples:
    >>> rules()["no-dc-path"].level
    <Level.ERROR: 'error'>
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Final, override

from pydantic import Field

from jolty import _core, ir
from jolty.errors import JoltyError
from jolty.settings import ALL, Level


@dataclass(frozen=True, slots=True)
class Rule:
    """Metadata for a single check rule: name, default level, and condition."""

    name: str
    level: Level
    condition: str


LOW_PHASE_MARGIN: Final = Rule(
    "low-phase-margin",
    Level.WARN,
    "A stability analysis's loop gain falls through 0 dB with a phase margin below "
    "45°: the loop's step response overshoots by more than about a fifth and rings, "
    "and the margin left for parts and loads that shift the phase is small. The "
    "diagnostic names the probe, the margin and the crossover frequency.",
)

MIN_PHASE_MARGIN: Final = 45.0
"""Minimum phase margin in degrees before low-phase-margin triggers."""

LOW_GAIN_MARGIN: Final = Rule(
    "low-gain-margin",
    Level.WARN,
    "A stability analysis's loop gain is above -6 dB where its phase, unwrapped from "
    "the lowest frequency, first falls through -180°: less than doubling the loop's "
    "gain makes it oscillate. The diagnostic names the probe and the margin.",
)

MIN_GAIN_MARGIN: Final = 6.0
"""Minimum gain margin in decibels before low-gain-margin triggers."""

OVER_RATING: Final = Rule(
    "over-rating",
    Level.ERROR,
    "A stress a device's rating limits, the voltage of one of its terminals over "
    "another, the current into one, the power it dissipates or its temperature, lies "
    "beyond the rating's bounds at an operating point or at any time of a transient "
    "the plan solves: the part's maker does not warrant it there, and it may fail. "
    "Each rating raises one diagnostic per analysis and bound, at the furthest point.",
)

NEAR_RATING: Final = Rule(
    "near-rating",
    Level.WARN,
    "A stress a device's rating limits lies within the rating but beyond 80 % of a "
    "bound away from zero, a temperature's from 0 °C, at an operating point or at any "
    "time of a transient the plan solves: the part runs without the margin derating "
    "keeps for heat, ageing and its spread. Each rating raises one diagnostic per "
    "analysis and bound, at the furthest point.",
)

DERATING_FACTOR: Final = 0.8
"""Fraction of a stress rating threshold beyond which near-rating triggers."""

ON_RESULTS: Final = (LOW_PHASE_MARGIN, LOW_GAIN_MARGIN, OVER_RATING, NEAR_RATING)
"""Post-simulation check rules evaluated on analysis results."""


@cache
def rules() -> dict[str, Rule]:
    """Returns all check rules keyed by name in reporting order.

    Engine rules covering circuit topology and operating-point checks appear first,
    followed by post-simulation rules evaluated on analysis results.

    Returns:
        Ordered dictionary mapping every check rule name to its `Rule` metadata.
    """
    engine = {
        name: Rule(name, Level(level), condition)
        for name, level, condition in _core.rules()
    }
    return engine | {rule.name: rule for rule in ON_RESULTS}


class Location(ir.FrozenModel):
    """Circuit instance referenced by a diagnostic along with its source location."""

    path: str = Field(description="Hierarchical path of the instance, such as 'X1/C1'.")
    source_location: ir.SourceLocation | None = Field(
        default=None, description="Build-script source line that placed the instance."
    )

    @override
    def __str__(self) -> str:
        """Formats the path followed by optional script source line."""
        return (
            self.path
            if self.source_location is None
            else f"{self.path}, {self.source_location}"
        )


class Diagnostic(ir.FrozenModel):
    """Single rule violation reported for a circuit."""

    rule: str = Field(description="Name of the triggered rule, such as 'no-dc-path'.")
    level: Level = Field(description="Severity level: 'error' or 'warn'.")
    message: str = Field(
        description="Factual description naming the affected instances and nets."
    )
    instances: tuple[Location, ...] = Field(
        description="Circuit instances involved in the diagnostic."
    )
    nets: tuple[str, ...] = Field(
        description="Hierarchical paths of the nets involved in the diagnostic."
    )

    def describe(self) -> str:
        """Formats the diagnostic in compiler style with one instance per line.

        Returns:
            Compiler-style multi-line string listing severity, rule, message, and paths.
        """
        lines = [f"{self.level}[{self.rule}]: {self.message}"]
        lines.extend(f"  = {instance}" for instance in self.instances)
        return "\n".join(lines)


class CheckReport(ir.FrozenModel):
    """Summary of check diagnostics ordered errors-first with severity counts."""

    diagnostics: tuple[Diagnostic, ...] = Field(
        description="Reported diagnostics, errors first, up to the configured limit."
    )
    errors: int = Field(description="Total count of error-level diagnostics.")
    warnings: int = Field(description="Total count of warning-level diagnostics.")
    unlisted: int = Field(
        description="Count of omitted diagnostics beyond the configured limit."
    )

    @property
    def ok(self) -> bool:
        """Pass status indicating that zero error-level diagnostics were reported."""
        return self.errors == 0

    def describe(self) -> str:
        """Formats all listed diagnostics followed by summary severity counts.

        Returns:
            Formatted diagnostic report ending with error, warning, and omitted counts.
        """
        counts = f"{_plural(self.errors, 'error')}, {_plural(self.warnings, 'warning')}"
        if self.unlisted:
            counts += f" ({self.unlisted} not listed)"
        return "\n\n".join(
            [*(diagnostic.describe() for diagnostic in self.diagnostics), counts]
        )


def problem(found: _core.Problem) -> str:
    """Formats an engine elaboration problem `found` in compiler style.

    Output starts with the error message and appends the instance path and script
    source line when available.

    Args:
        found: Raw elaboration problem dictionary returned by the Rust engine.

    Returns:
        Compiler-style error string with optional instance path and source location.

    Examples:
        >>> source = {"file": "amp.py", "line": 9, "function": "amp"}
        >>> found = {
        ...     "message": "resistance must be positive, not -1.0",
        ...     "instance": {"path": "R1", "source_location": source},
        ... }
        >>> print(problem(found))
        error: resistance must be positive, not -1.0
          = R1, amp.py:9 in amp()
    """
    text = f"error: {found['message']}"
    if found["instance"] is not None:
        text += f"\n  = {_instance(found['instance'])}"
    return text


def diagnostics(found: Iterable[_core.Diagnostic]) -> tuple[Diagnostic, ...]:
    """Converts raw engine diagnostics `found` into models with default levels.

    Args:
        found: Raw structural diagnostic records emitted by the Rust engine.

    Returns:
        Tuple of `Diagnostic` models populated with default rule severities.
    """
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


def _instance(found: _core.Location) -> Location:
    source = found["source_location"]
    return Location(
        path=found["path"],
        source_location=None
        if source is None
        else ir.SourceLocation.model_validate(source),
    )


def levels(
    configured: Mapping[str, Level],
    *,
    error: Iterable[str] = (),
    warn: Iterable[str] = (),
    ignore: Iterable[str] = (),
) -> dict[str, Level]:
    """Resolves the effective severity level for every check rule.

    Resolution applies rule defaults first, then `configured` settings, then CLI
    options. The ``all`` wildcard applies to every rule and is overridden by explicit
    rule entries in the same layer.

    Args:
        configured: Rule severity overrides loaded from `Settings.rules`.
        error: Rule names or `'all'` promoted to `Level.ERROR` via CLI flags.
        warn: Rule names or `'all'` set to `Level.WARN` via CLI flags.
        ignore: Rule names or `'all'` silenced to `Level.IGNORE` via CLI flags.

    Returns:
        Dictionary mapping every known check rule name to its resolved `Level`.

    Raises:
        JoltyError: When a name matches no rule or CLI options assign conflicting
            levels to the same rule.
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
) -> CheckReport:
    """Builds a `CheckReport` from `diagnostics` using `chosen` levels up to `limit`.

    Args:
        diagnostics: Unfiltered diagnostics prior to severity overrides.
        chosen: Effective `Level` per rule name as resolved by `levels`.
        limit: Maximum number of non-ignored diagnostics retained in output.

    Returns:
        `CheckReport` sorted errors-first with total and unlisted counts.
    """
    leveled = [
        diagnostic.model_copy(update={"level": chosen[diagnostic.rule]})
        for diagnostic in diagnostics
        if chosen[diagnostic.rule] is not Level.IGNORE
    ]
    leveled.sort(key=lambda diagnostic: diagnostic.level is not Level.ERROR)
    errors = sum(diagnostic.level is Level.ERROR for diagnostic in leveled)
    return CheckReport(
        diagnostics=tuple(leveled[:limit]),
        errors=errors,
        warnings=len(leveled) - errors,
        unlisted=max(0, len(leveled) - limit),
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"
