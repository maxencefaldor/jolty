# Copyright 2026 Maxence Faldor
"""Shared subprocess execution and thermal helpers for external SPICE simulators."""

import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from jolty import _core
from jolty.errors import JoltyError
from jolty.units import ZERO_CELSIUS

TIMEOUT: Final = 300
"""Default seconds an external simulator may run an analysis."""

_NO_EXTRA_FILES: Final[Mapping[str, str]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Executed[Value]:
    """Captured output, exit code, runtime, and payload of one simulator run."""

    printed: str
    """Combined standard output followed by standard error from the simulator."""
    code: int
    """Process exit status code."""
    seconds: float
    """Wall-clock simulator runtime in seconds, excluding deck writing."""
    values: Value | None
    """Parsed values output written by the deck, or None if no file was produced."""


def thermal_option_temperature(
    temperature: float, simulator_thermal_voltage_per_kelvin: float
) -> float:
    """Computes Celsius temperature matching jolty thermal voltage at `temperature`.

    Args:
        temperature: Target jolty temperature in kelvins.
        simulator_thermal_voltage_per_kelvin: Simulator `k / q` constant in volts per
            kelvin.

    Returns:
        Simulator option temperature in degrees Celsius producing identical thermal
        voltage.
    """
    thermal_voltage = temperature * _core.BOLTZMANN / _core.ELEMENTARY_CHARGE
    return thermal_voltage / simulator_thermal_voltage_per_kelvin - ZERO_CELSIUS


def decode_expired(expired: subprocess.TimeoutExpired) -> str:
    """Decodes partial stdout and stderr captured on a subprocess timeout.

    Args:
        expired: Timeout exception raised by `subprocess.run`.

    Returns:
        Concatenated decoded stdout and stderr captured before timeout.
    """
    # The output so far: bytes on POSIX, text where communicate gathered it.
    return "".join(
        part.decode(errors="replace") if isinstance(part, bytes) else part or ""
        for part in (expired.stdout, expired.stderr)
    )


def probe_version(executable: str, flag: str) -> str:
    """Captures stdout from invoking `executable` with its version `flag`.

    Args:
        executable: Path to the simulator executable.
        flag: Command-line flag requesting version output (such as ``"--version"``).

    Returns:
        Standard output text printed by the version command.
    """
    return subprocess.run(  # noqa: S603 - external simulator states its version
        [executable, flag], capture_output=True, check=False, text=True
    ).stdout


def read_text_utf8(path: Path) -> str:
    """Reads `path` as UTF-8 text.

    Args:
        path: Filesystem path to read.

    Returns:
        Decoded UTF-8 file contents.
    """
    return path.read_text(encoding="utf-8")


def run_deck[Value](  # noqa: PLR0913 - simulator command, deck, output, and timeout
    name: str,
    command: Sequence[str],
    deck: Sequence[str],
    values_file: str,
    read_values: Callable[[Path], Value],
    *,
    timeout: float | None,
    timeout_error: type[JoltyError],
    extra_files: Mapping[str, str] = _NO_EXTRA_FILES,
) -> Executed[Value]:
    """Runs an external simulator on `deck` in an isolated temporary directory.

    Args:
        name: Human-readable simulator name used in timeout diagnostics.
        command: Base command tokens preceding ``"circuit.cir"``.
        deck: SPICE deck lines written to ``circuit.cir``.
        values_file: Relative filename of the simulator output file to read back.
        read_values: Reader converting the output file path into `Value`.
        timeout: Subprocess timeout in seconds, or `None` for no limit.
        timeout_error: Exception class raised when the subprocess exceeds `timeout`.
        extra_files: Optional mapping of auxiliary filenames to UTF-8 contents written
            alongside ``circuit.cir``.

    Returns:
        `Executed` record bundling combined output, exit code, runtime, and payload.

    Raises:
        JoltyError: Instance of `timeout_error` when the simulator exceeds `timeout`.
    """
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        for filename, content in extra_files.items():
            (directory / filename).write_text(content, encoding="utf-8")
        (directory / "circuit.cir").write_text("\n".join(deck) + "\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            done = subprocess.run(  # noqa: S603 - runs simulator on temporary deck
                [*command, "circuit.cir"],
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
                cwd=directory,
            )
        except subprocess.TimeoutExpired as expired:
            limit = expired.timeout if timeout is None else timeout
            printed = decode_expired(expired)
            message = f"{name} ran for more than {limit:g} s:\n{printed}"
            raise timeout_error(message) from None
        seconds = time.perf_counter() - started
        values = directory / values_file
        written = read_values(values) if values.is_file() else None
    return Executed(done.stdout + done.stderr, done.returncode, seconds, written)
