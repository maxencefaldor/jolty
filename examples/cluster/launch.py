# Copyright 2026 Maxence Faldor
r"""Run a plan's Monte Carlo samples in shards over a pool of processes, then merge.

Run it from the repository's root, such as::

    uv run python examples/cluster/launch.py examples/cluster/amplifier.py \\
        --samples 10000000 --shard 1000000 --workers 4

It splits the plan's samples into shards of ``--shard`` samples, the last one shorter,
and runs each as ``jolty simulate SCRIPT --samples START:STOP --shards DIRECTORY``,
``--workers`` at a time, each on its share of the processors through ``JOLTY_THREADS``.
A shard is a command with no state of its own, whose file appears whole or not at all,
so the launcher runs again each shard that fails, up to ``--attempts`` times, and runs
none whose file the directory holds already: run it again after it stops, and it goes
on where it stopped. Once every shard is there, it merges them with ``jolty merge
DIRECTORY``, which writes the samples' batch and plots beside the directory, and exits
with the merge's status.

A cluster's scheduler does the same with an array of jobs, each the command of one
shard, and the merge once every one has ended; the shards' directory is then one that
every node writes to, such as a shared file system's.

Sizing the shards: each shard pays its start-up once, two interpreters importing jolty,
the script and the circuit's elaboration, 0.9 s for this amplifier on an Apple M2 Max,
and runs its samples at the throughput of its worker's threads, 80 µs a sample on its
twelve cores and 320 µs on three. Make a shard take 50 times its start-up or more, so
that the start-up costs it 2 % of its time or less: 45 s or more, some 560,000 samples
of the amplifier on a worker of twelve threads, or 140,000 on a worker of three. More
shards than workers let a worker that runs faster take more of them, where a worker's
cores are slower or busier than another's; shards much larger only make a failed one
cost more to run again, and leave workers idle longer while the last ones end.
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
    """The shards of `samples` samples, `size` each, the last one what remains."""
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
    """Run the samples of a plan's Monte Carlo analysis in shards, and merge them."""

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
        description="The shards' directory; by default <script>.shards beside the "
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
