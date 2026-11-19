# Copyright 2026 Maxence Faldor
"""Exception hierarchy and validation-error helpers for jolty."""

import difflib
from collections.abc import Sequence
from typing import Final, LiteralString, Never

from pydantic import ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

_SHORT_NAME_LENGTH: Final = 2
"""Maximum name length that falls back to the 0.5 fuzzy-matching cutoff."""


def fail(kind: LiteralString, message: str) -> Never:
    """Raises a Pydantic custom error holding literal `message` escaped from braces.

    Args:
        kind: Machine-readable Pydantic error code.
        message: Human-readable validation message preserved verbatim.
    """
    raise PydanticCustomError(kind, "{message}", {"message": message})


def detail(
    kind: LiteralString,
    message: str,
    value: object,
    location: tuple[int | str, ...] = (),
) -> InitErrorDetails:
    """Builds one structured validation error entry at `location` with `message`.

    Args:
        kind: Machine-readable Pydantic error code.
        message: Human-readable validation failure description.
        value: Rejected input value recorded on the error entry.
        location: Field path segments locating the offending value, empty at the root.

    Returns:
        Structured Pydantic initialization error record for `ValidationError`.
    """
    return InitErrorDetails(
        type=PydanticCustomError(kind, "{message}", {"message": message}),
        loc=location,
        input=value,
    )


def raise_if_errors(title: str, found: Sequence[InitErrorDetails]) -> None:
    """Raises `ValidationError` containing every entry in `found` when non-empty.

    Args:
        title: Model or schema title attached to the raised `ValidationError`.
        found: Collected validation error records; nothing is raised when empty.
    """
    if found:
        raise ValidationError.from_exception_data(title, list(found))


def close_names(names: Sequence[str], name: str) -> str:
    """Formats up to three close matches from `names` for `name` as a suffix.

    Matching runs `difflib.get_close_matches` at cutoff 0.6, falling back to
    case-insensitive matching when needed. Two-character names fall back further to
    cutoff 0.5 so single-character edits (such as ``V2`` for ``V1`` or ``po`` for
    ``op``) still match.

    Args:
        names: Candidate valid names in priority order.
        name: Unrecognized query name to match.

    Returns:
        Semicolon-prefixed close-names suffix, or an empty string when none match.

    Examples:
        >>> close_names(["out", "in"], "ou")
        '; close names: out'
        >>> close_names(["V1", "I1"], "V2")
        '; close names: V1'
        >>> close_names(["V1", "R1"], "v1")
        '; close names: V1'
    """
    unique = list(dict.fromkeys(names))
    close = difflib.get_close_matches(name, unique, n=3)
    if not close:
        lowered: dict[str, list[str]] = {}
        for candidate in unique:
            lowered.setdefault(candidate.casefold(), []).append(candidate)
        folded = difflib.get_close_matches(name.casefold(), list(lowered), n=3)
        if not folded and len(name) == _SHORT_NAME_LENGTH:
            folded = difflib.get_close_matches(
                name.casefold(), list(lowered), n=3, cutoff=0.5
            )
        close = [candidate for key in folded for candidate in lowered[key]][:3]
    return f"; close names: {', '.join(close)}" if close else ""


class JoltyError(Exception):
    """Base exception for all errors raised by jolty."""


class DimensionError(JoltyError, TypeError):
    """Error raised when a physical quantity has an incompatible unit dimension."""


class BuildError(JoltyError, ValueError):
    """Error raised when a circuit script describes an invalid topology."""


class SpiceError(JoltyError, ValueError):
    """Error raised when a SPICE file has unsupported syntax or missing references."""


class DownloadError(JoltyError):
    """Error raised when a file download fails or mismatches its expected digest."""


class ScriptError(JoltyError):
    """Error raised when a script lacks a required function or returns a wrong type."""


class ElaborationError(JoltyError):
    """Error raised when circuit elaboration fails, collecting every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Stores formatted elaboration `problems` and initializes the message."""
        self.problems = tuple(problems)
        super().__init__("\n\n".join(self.problems))


class NetlistError(JoltyError):
    """Error raised when a component cannot be exported to a board netlist."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Stores netlist export `problems` and formats the exception message.

        Instance-specific problems are prefixed with their hierarchical path.
        """
        self.problems = tuple(problems)
        listed = "".join(f"\n  {problem}" for problem in self.problems)
        super().__init__(f"cannot write the netlist:{listed}")


class SimulationError(JoltyError):
    """Error raised when an analysis fails to solve or converge."""


class PlanError(SimulationError, ValueError):
    """Error raised when a plan, analysis, or specification is invalid for a circuit."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Stores plan validation `problems` and initializes the exception message."""
        self.problems = tuple(problems)
        super().__init__("\n\n".join(self.problems))


class CharacterizationError(JoltyError):
    """Error raised when a component cannot be characterized as a digital logic cell."""
