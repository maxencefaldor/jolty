# Copyright 2026 Maxence Faldor
"""Configuration shared by every front end.

Each setting comes from the first source that sets it: explicit arguments, ``JOLTY_*``
environment variables, or the ``[tool.jolty]`` table of ``pyproject.toml`` in the
working directory.
"""

from enum import StrEnum
from typing import Annotated, Final, Self, override

from pydantic import Field, StringConstraints, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    SettingsError,
)

from jolty.errors import JoltyError

ALL: Final = "all"
"""The key that sets the level of every rule at once."""


class Level(StrEnum):
    """How a check reports a rule's diagnostics, by the names ty gives its levels."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


type RuleKey = Annotated[str, StringConstraints(pattern=r"^[a-z]+(-[a-z]+)*$")]
"""A rule's name, such as ``no-dc-path``, or ``all`` for every rule."""


class Settings(BaseSettings):
    """Settings shared by every front end."""

    model_config = SettingsConfigDict(
        env_prefix="JOLTY_",
        env_ignore_empty=True,
        extra="forbid",
        pyproject_toml_table_header=("tool", "jolty"),
    )

    timeout: float = Field(
        default=300.0,
        gt=0,
        description="Seconds a script may run, its analyses included, "
        "before jolty stops it.",
    )
    output_limit: int = Field(
        default=20_000,
        ge=0,
        description="Characters kept from each of a script's standard output "
        "and standard error.",
    )
    listed_unknowns: int = Field(
        default=40,
        ge=1,
        description="Unknowns a report lists for each analysis; "
        "the results file holds every one.",
    )
    max_diagnostics: int = Field(
        default=20,
        ge=1,
        description="Diagnostics a check report lists; it counts the rest.",
    )
    rules: dict[RuleKey, Level] = Field(
        default_factory=dict,
        description="Level of each rule by name, or of every rule with 'all': error, "
        "warn or ignore. A rule's own entry overrides 'all', and rules left out keep "
        "their default level.",
    )

    @classmethod
    def load(cls) -> Self:
        """The configured settings.

        Raises:
            JoltyError: When a source sets a setting that does not exist, or an invalid
                value.
        """
        try:
            return cls()
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in problem['loc'])}: {problem['msg']}"
                for problem in error.errors()
            )
            message = f"invalid settings: {problems}"
            raise JoltyError(message) from None
        except SettingsError as error:
            message = f"invalid settings: {error}"
            raise JoltyError(message) from None

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
        """Read explicit values, then the environment, then ``pyproject.toml``."""
        return (
            init_settings,
            env_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
        )
