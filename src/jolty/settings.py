# Copyright 2026 Maxence Faldor
"""Configuration shared by every front end.

Each setting comes from the first source that sets it: explicit arguments, ``JOLTY_*``
environment variables, or the ``[tool.jolty]`` table of ``pyproject.toml`` in the
working directory.
"""

from typing import Self, override

from pydantic import Field, ValidationError
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    SettingsError,
)

from jolty.errors import JoltyError


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
