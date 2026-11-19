# Copyright 2026 Maxence Faldor
"""Wrapper script that logs a simulator invocation and replaces itself via exec.

Usage: ``counted.py LOG PROGRAM [ARGUMENT ...]``. The benchmark runner places a shell
wrapper on ``PATH`` for each tracked simulator (``ngspice`` or ``jolty``) that invokes
``counted.py`` with the real executable. The script appends a single JSON line to
``LOG`` containing the program name, argument list, and start timestamp in Unix
seconds. The process then replaces itself with ``PROGRAM`` and the forwarded arguments
so stdout, stderr, exit status, and signals pass through directly. Running under
``python -I`` isolates the wrapper from the agent environment.
"""

import json
import os
import sys
import time
from pathlib import Path


def log(record: Path, program: str, arguments: list[str]) -> None:
    """Appends one JSON line to `record` logging `program` and `arguments`.

    Opening the log in append mode writes each entry in a single call so concurrent
    invocations do not interleave partial lines.
    """
    line = json.dumps(
        {"program": Path(program).name, "arguments": arguments, "time": time.time()}
    )
    with record.open("a", encoding="utf-8") as file:
        file.write(f"{line}\n")


def main() -> None:
    """Logs the invocation from CLI arguments and executes the target program."""
    record, program, *arguments = sys.argv[1:]
    log(Path(record), program, arguments)
    os.execv(program, [program, *arguments])  # noqa: S606 - the counted program runs as the agent asked


if __name__ == "__main__":
    main()
