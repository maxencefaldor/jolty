# Copyright 2026 Maxence Faldor
"""Command-line interface for ``jolty``.

Each subcommand extends a request model from [`jolty.service`][jolty.service] with an
output format, so CLI flags match the MCP tool parameters and print the same text.
"""

__lazy_modules__ = ["jolty._core", "jolty.cli._app"]

import sys
from collections.abc import Sequence

from jolty._core import __version__ as _version
from jolty.cli._app import COMMANDS, CommandLine, CommandSpec, Jolty, run, run_cli

__all__ = ["COMMANDS", "CommandLine", "CommandSpec", "Jolty", "main", "run"]


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the ``jolty`` command-line entry point.

    Passing ``--version`` alone prints the package version and exits with status 0.
    Exit status 1 indicates a circuit or analysis failure: failed elaboration, check
    errors, a failed simulation, missed Monte Carlo yield target, an unpriced device,
    or a failed netlist export. Exit status 2 indicates invalid arguments or settings,
    a missing subcommand, or an unhandled command failure such as a script exception.

    Args:
        argv: Command-line tokens to parse, defaulting to ``sys.argv[1:]`` when
            ``None``.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments == ["--version"]:
        print(f"jolty {_version}")
        return
    run_cli(arguments)
