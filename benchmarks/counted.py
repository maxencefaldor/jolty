# Copyright 2026 Maxence Faldor
"""Log one run of a program an agent ran, then run the program in this one's place.

Usage: ``counted.py LOG PROGRAM [ARGUMENT ...]``. The runner puts a wrapper on the
agent's path in place of each simulator it counts, ``ngspice`` or ``jolty``, a shell
script that runs this one with the real program. It appends one line to ``LOG``, a JSON
object of the program's name, the arguments it was given and the time it started, in
seconds since the epoch, then replaces its own process with ``PROGRAM`` and those
arguments, so the program's output, exit code and signals are its own. It runs isolated,
``python -I``, so nothing of the agent's environment changes what it does.
"""

import json
import os
import sys
import time
from pathlib import Path


def log(record: Path, program: str, arguments: list[str]) -> None:
    """Appends to `record` the run of `program` with `arguments`, as one JSON line.

    The line is one write to a file opened to append, so runs that end at once leave
    whole lines.
    """
    line = json.dumps(
        {"program": Path(program).name, "arguments": arguments, "time": time.time()}
    )
    with record.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")


def main() -> None:
    """Logs the run the arguments describe, then runs its program in this process."""
    record, program, *arguments = sys.argv[1:]
    log(Path(record), program, arguments)
    os.execv(program, [program, *arguments])  # noqa: S606 - the counted program runs as the agent asked


if __name__ == "__main__":
    main()
