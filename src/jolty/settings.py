# Copyright 2026 Maxence Faldor
"""Configuration shared by every front end.

Each setting comes from the first source that sets it: explicit arguments, ``JOLTY_*``
environment variables, or the ``[tool.jolty]`` table of ``pyproject.toml`` in the
working directory.
"""

import os
from enum import StrEnum
from pathlib import Path
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
"""Wildcard key setting the severity level for every check rule at once."""


class Level(StrEnum):
    """Severity level assigned to check rule diagnostics (matching ty naming)."""

    ERROR = "error"
    WARN = "warn"
    IGNORE = "ignore"


type RuleKey = Annotated[str, StringConstraints(pattern=r"^[a-z]+(-[a-z]+)*$")]
"""Check rule name (such as ``no-dc-path``) or ``all`` for every rule."""


def _user_cache() -> Path:
    """Returns `$XDG_CACHE_HOME/jolty`, or `~/.cache/jolty` when unset."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "jolty"


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
        description="Maximum runtime in seconds for a script and its analyses before "
        "termination.",
    )
    max_output: int = Field(
        default=20_000,
        ge=0,
        description="Maximum characters retained separately from script standard "
        "output and standard error.",
    )
    max_unknowns: int = Field(
        default=40,
        ge=1,
        description="Maximum unknowns listed per analysis in reports; results files "
        "store all unknowns.",
    )
    max_diagnostics: int = Field(
        default=20,
        ge=1,
        description="Maximum diagnostics listed in a check report; remaining "
        "diagnostics are counted.",
    )
    cache: Path = Field(
        default_factory=_user_cache,
        description="Directory for cached downloads such as device models; defaults "
        "to the user cache directory.",
    )
    pdk_root: Path | None = Field(
        default=None,
        description="Directory containing open_pdks installs (sky130A, gf180mcuD, "
        "ihp-sg13g2); defaults to cached pinned releases.",
    )
    threads: int | None = Field(
        default=None,
        ge=1,
        description="Worker thread count for batches and parallel analyses; defaults "
        "to one per CPU without affecting numerical results.",
    )
    rules: dict[RuleKey, Level] = Field(
        default_factory=dict,
        description="Severity override per rule name or 'all' (error, warn, ignore). "
        "Explicit rule names override 'all', and unlisted rules keep default levels.",
    )
    analysis_log: Path | None = Field(
        default=None,
        description="Append-only JSONL file recording analysis kind, run count, and "
        "start timestamp across CLI, MCP, and scripts; disabled by default.",
    )

    @classmethod
    def load(cls) -> Self:
        """Loads and validates settings from all configured sources.

        Returns:
            Validated `Settings` instance populated from configured sources.

        Raises:
            JoltyError: When a source specifies an unknown setting or an invalid value.
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
        """Reads explicit arguments, then environment variables, then pyproject.toml.

        Args:
            settings_cls: Settings model class being loaded.
            init_settings: Explicit keyword-argument settings source.
            env_settings: Environment-variable settings source (`JOLTY_*`).
            dotenv_settings: Unused `.env` settings source supplied by Pydantic.
            file_secret_settings: Unused secrets-directory source supplied by Pydantic.

        Returns:
            Ordered tuple of active settings sources from highest to lowest precedence.
        """
        return (
            init_settings,
            env_settings,
            PyprojectTomlConfigSettingsSource(settings_cls),
        )
