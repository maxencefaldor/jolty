# Copyright 2026 Maxence Faldor
r"""Sharded Monte Carlo launcher running sample ranges in parallel and merging outputs.

Invoke the script from the repository root::

    uv run python examples/cluster/launch.py examples/cluster/amplifier.py \\
        --samples 10000000 --shard 1000000 --workers 4

The launcher partitions the plan into shards of ``--shard`` samples (with a shorter
final shard) and invokes ``jolty simulate SCRIPT --samples START:STOP --shards
DIRECTORY`` for each shard. Up to ``--workers`` shards run concurrently, each using an
equal processor share via ``JOLTY_THREADS``. Each shard command is stateless and writes
its output file atomically. Failed shards retry up to ``--attempts`` times, and
completed shard files are skipped automatically on restart. Once all shards finish,
``jolty merge DIRECTORY`` combines the batch, writes plots next to the shard directory,
and determines the launcher exit status.

A cluster scheduler follows the same pattern with a job array for shards and a merge job
after completion, using a shared filesystem for the shard directory.

Each shard pays start-up overhead once for interpreter import, script execution, and
circuit elaboration (about 0.9 s for this amplifier on an Apple M2 Max). Steady-state
throughput scales with worker threads, averaging 80 µs per sample on 12 cores and
320 µs on 3 cores. Sizing each shard to at least 50 times start-up keeps overhead at or
below 2 % (at least 45 s, or roughly 560,000 samples on 12 threads and 140,000 on 3
threads). Creating more shards than workers lets faster workers absorb extra shards on
heterogeneous nodes. Excessively large shards increase retry cost and tail idle time.
"""

import os
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import override

from pydantic import Field, PrivateAttr
from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


def ranges(samples: int, size: int) -> list[tuple[int, int]]:
    """Returns `(start, stop)` shard index ranges of at most `size` for `samples`."""
    return [(start, min(start + size, samples)) for start in range(0, samples, size)]


def run(
    command: Sequence[str], environment: dict[str, str], attempts: int
) -> subprocess.CompletedProcess[str]:
    """Runs `command` up to `attempts` times, until it does not fail outright.

    jolty exits with 1 when a circuit's checks or specs fail, which running again does
    not change, and with 2 when a command fails, as a killed or stopped one does.
    """
    for _ in range(attempts - 1):
        done = subprocess.run(  # noqa: S603 - this interpreter, running jolty
            command, env=environment, capture_output=True, text=True, check=False
        )
        if done.returncode in {0, 1}:
            return done
    return subprocess.run(  # noqa: S603 - this interpreter, running jolty
        command, env=environment, capture_output=True, text=True, check=False
    )


class Launch(BaseSettings):
    """CLI configuration for running and merging sharded Monte Carlo samples."""

    model_config = SettingsConfigDict(
        cli_prog_name="launch.py",
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_enforce_required=True,
    )

    script: CliPositionalArg[Path] = Field(description="The script of the plan to run.")
    samples: int = Field(ge=1, description="The plan's Monte Carlo samples.")
    shard: int = Field(ge=1, description="The samples of each shard.")
    workers: int = Field(default=1, ge=1, description="The shards run at a time.")
    threads: int | None = Field(
        default=None,
        ge=1,
        description="The threads each worker runs on; by default its share of the "
        "processors.",
    )
    shards: Path | None = Field(
        default=None,
        description="The shards' directory; by default <script>.shards next to the "
        "script.",
    )
    attempts: int = Field(
        default=3, ge=1, description="The runs of a shard before giving up on it."
    )
    _status: int = PrivateAttr(default=0)

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
        """Reads the command line only."""
        return (init_settings,)

    @property
    def status(self) -> int:
        """The merge's exit status, or that of a shard that failed outright."""
        return self._status

    def cli_cmd(self) -> None:
        """Runs the shards that are missing and merges them, keeping the status."""
        directory = self.shards or self.script.with_name(f"{self.script.stem}.shards")
        threads = self.threads or max(1, (os.cpu_count() or 1) // self.workers)
        environment = {**os.environ, "JOLTY_THREADS": str(threads)}
        jolty = [sys.executable, "-m", "jolty"]
        missing = [
            (start, stop)
            for start, stop in ranges(self.samples, self.shard)
            if not (directory / f"shard-{start}-{stop}.npz").is_file()
        ]

        def shard(taken: tuple[int, int]) -> subprocess.CompletedProcess[str]:
            start, stop = taken
            command = [*jolty, "simulate", str(self.script)]
            command += ["--samples", f"{start}:{stop}", "--shards", str(directory)]
            return run(command, environment, self.attempts)

        with ThreadPoolExecutor(self.workers) as pool:
            ran = zip(missing, pool.map(shard, missing), strict=True)
            for (start, stop), done in ran:
                sys.stdout.write(
                    f"shard {start}:{stop} exited with {done.returncode}\n"
                )
                if done.returncode not in {0, 1}:
                    sys.stderr.write(done.stderr)
                    self._status = done.returncode
                    return
        sys.stdout.flush()
        merged = subprocess.run(  # noqa: S603 - this interpreter, running jolty
            [*jolty, "merge", str(directory)], check=False
        )
        self._status = merged.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the shards that are missing, merges them, and returns the merge's status.

    The status is a shard's when one fails outright.
    """
    arguments = sys.argv[1:] if argv is None else list(argv)
    return CliApp.run(Launch, cli_args=arguments).status


if __name__ == "__main__":
    raise SystemExit(main())
