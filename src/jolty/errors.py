# Copyright 2026 Maxence Faldor
"""The errors jolty raises, each stating what is wrong and where."""

import difflib
from collections.abc import Sequence
from typing import Final, LiteralString, Never

from pydantic import ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

_SHORT_NAME_LENGTH: Final = 2
"""Names of two characters score at most 0.5 for a single-character edit."""


def fail(kind: LiteralString, message: str) -> Never:
    """Raises a Pydantic custom error holding literal `message` safe from braces."""
    raise PydanticCustomError(kind, "{message}", {"message": message})


def detail(
    kind: LiteralString,
    message: str,
    value: object,
    location: tuple[int | str, ...] = (),
) -> InitErrorDetails:
    """One structured validation error entry at `location` with literal `message`."""
    return InitErrorDetails(
        type=PydanticCustomError(kind, "{message}", {"message": message}),
        loc=location,
        input=value,
    )


def raise_if_errors(title: str, found: Sequence[InitErrorDetails]) -> None:
    """Raises `ValidationError` holding every entry of `found`, if any."""
    if found:
        raise ValidationError.from_exception_data(title, list(found))


def close_names(names: Sequence[str], name: str) -> str:
    """The names among `names` closest to `name`, as the end of a message, if any.

    Uses `difflib.get_close_matches` at its default cutoff (0.6). When nothing matches
    by exact case, falls back to case-insensitive matching, and for two-character names
    to cutoff 0.5 so that a single-character edit (such as ``V2`` for ``V1`` or ``po``
    for ``op``) still matches.

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
    """The base of every error jolty raises."""


class DimensionError(JoltyError, TypeError):
    """A quantity has the wrong dimension, such as farads where ohms belong."""


class BuildError(JoltyError, ValueError):
    """A script describes a circuit that cannot exist, such as two shorted ports."""


class SpiceError(JoltyError, ValueError):
    """A SPICE file holds what jolty cannot read, or reads what it does not define."""


class DownloadError(JoltyError):
    """A file cannot be downloaded, or what was downloaded is not the file expected."""


class ScriptError(JoltyError):
    """A script lacks the function a command calls, or it returns the wrong type."""


class ElaborationError(JoltyError):
    """The engine cannot elaborate a circuit: every problem it found, as errors."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Holds `problems`, each what is wrong, then its instance and source line."""
        self.problems = tuple(problems)
        super().__init__("\n\n".join(self.problems))


class NetlistError(JoltyError):
    """A component cannot be written as a board's netlist: every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Holds `problems`: what keeps the component, or an instance, off a board.

        A problem of an instance begins with its path.
        """
        self.problems = tuple(problems)
        listed = "".join(f"\n  {problem}" for problem in self.problems)
        super().__init__(f"cannot write the netlist:{listed}")


class SimulationError(JoltyError):
    """An analysis failed, such as an operating point that no strategy finds."""


class PlanError(SimulationError, ValueError):
    """A plan, an analysis or specs cannot run on a circuit: every problem found."""

    def __init__(self, problems: Sequence[str]) -> None:
        """Holds `problems`, each an unresolved name or invalid setting."""
        self.problems = tuple(problems)
        super().__init__("\n\n".join(self.problems))


class CharacterizationError(JoltyError):
    """A component cannot be characterized as a logic cell, as it stands."""
