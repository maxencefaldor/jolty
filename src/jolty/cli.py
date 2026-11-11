# Copyright 2026 Maxence Faldor
"""The ``jolty`` command line.

Each subcommand extends a request model of [`jolty.service`][jolty.service] with an
output format, so the command line takes exactly the arguments the MCP tools take and
prints the text they return.
"""

import json
import os
import sys
from collections.abc import Sequence
from typing import Literal, NoReturn, override

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliSubCommand,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from jolty import __version__, service
from jolty.errors import JoltyError
from jolty.settings import Settings


class _Output(BaseModel):
    output_format: Literal["text", "json"] = Field(
        default="text", description="'text' for people and agents, 'json' for programs."
    )

    def emit(self, result: service.CommandResult) -> None:
        """Prints `result` in the requested format; exits with 1 if it is not ok."""
        if self.output_format == "json":
            print(result.model_dump_json(indent=2))
        else:
            print(result.describe())
        if not result.ok:
            raise SystemExit(1)


class BuildCommand(service.BuildRequest, _Output):
    """Run a script's build(), save the circuit it returns, and elaborate it."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.build(self, Settings.load()))


class SimulateCommand(service.SimulateRequest, _Output):
    """Build a script's circuit, then run every analysis of a plan on it."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.simulate(self, Settings.load()))


class CheckCommand(service.CheckRequest, _Output):
    """Check the structure of a saved circuit file."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.check(self, Settings.load()))


class ExplainCommand(service.ExplainRequest, _Output):
    """Explain rules: the exact condition each checks, and its level."""

    async def cli_cmd(self) -> None:
        """Runs the command."""
        self.emit(await service.explain(self, Settings.load()))


class Jolty(BaseSettings):
    """Build and simulate electronic circuits written as Python scripts."""

    model_config = SettingsConfigDict(
        cli_prog_name="jolty",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_enforce_required=True,
    )

    build: CliSubCommand[BuildCommand] = Field(
        description="Build a script's circuit and elaborate it."
    )
    simulate: CliSubCommand[SimulateCommand] = Field(
        description="Build a script's circuit and run a plan's analyses on it."
    )
    check: CliSubCommand[CheckCommand] = Field(
        description="Check the structure of a saved circuit file."
    )
    explain: CliSubCommand[ExplainCommand] = Field(
        description="Explain rules and their levels."
    )

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Reads the command line only.

        Configuration lives in [`Settings`][jolty.settings.Settings].
        """
        return (init_settings,)

    def cli_cmd(self) -> None:
        """Runs the chosen subcommand."""
        CliApp.run_subcommand(self)


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the command line.

    The exit status is 1 when a circuit does not elaborate, a check finds an error or an
    analysis fails, and 2
    when the arguments or settings are invalid, a command needs a subcommand, or a
    command fails, as when a script raises. The lone argument ``--version`` prints the
    version.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments == ["--version"]:
        print(f"jolty {__version__}")
        return
    try:
        CliApp.run(Jolty, cli_args=arguments)
    except JoltyError as error:
        _fail(str(error), as_json=_wants_json(arguments))
    except ValidationError as error:
        _fail(_invalid(error), as_json=_wants_json(arguments))
    except SystemExit as exit_:
        # A command run without its subcommand exits with an error and its usage.
        if isinstance(exit_.code, str):
            print(exit_.code, end="", file=sys.stderr)
            raise SystemExit(2) from None
        raise
    except BrokenPipeError:
        # The reader stopped reading, as `head` does: stop quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(1) from None


def _fail(message: str, *, as_json: bool) -> NoReturn:
    """Prints an error, as JSON on standard output when asked to, and exits with 2."""
    if as_json:
        print(json.dumps({"error": message}))
    else:
        print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def _wants_json(arguments: Sequence[str]) -> bool:
    pairs = zip(arguments, [*arguments[1:], ""], strict=True)
    return "--output-format=json" in arguments or ("--output-format", "json") in pairs


def _invalid(error: ValidationError) -> str:
    """Names each invalid option and says what it accepts."""
    problems = []
    for problem in error.errors():
        names = [part for part in problem["loc"] if isinstance(part, str)]
        option = "--" + names[-1].replace("_", "-") if names else "arguments"
        problems.append(f"{option}: {problem['msg']}")
    return "invalid arguments: " + "; ".join(problems)
