# Copyright 2026 Maxence Faldor
"""The errors jolty raises, each stating what is wrong and where."""

from collections.abc import Sequence


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


class CharacterizationError(JoltyError):
    """A component cannot be characterized as a logic cell, as it stands."""
