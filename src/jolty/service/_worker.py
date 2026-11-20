# Copyright 2026 Maxence Faldor
"""Process lifecycle and pre-warmed standby management for ``jolty.runner``.

Every script runs in its own isolated OS process with captured stdout/stderr,
timeout enforcement, and crash isolation. In long-lived callers such as the MCP
server or repeated Python `jolty.service` invocations, a single pre-warmed standby
`jolty.runner --standby` child preloads core execution modules in the background,
runs at most one dispatched job, and exits immediately afterward so no state is
ever shared across scripts.
"""

__lazy_modules__ = ["jolty", "json", "subprocess"]

import asyncio
import atexit
import contextlib
import contextvars
import json
import os
import subprocess
import sys
import threading
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

from jolty import runner
from jolty.errors import JoltyError

_STARTUP_PREFIXES: Final[tuple[str, ...]] = (
    "PYTHON",
    "COV_",
    "COVERAGE_",
    "VIRTUAL_ENV",
    "LD_",
    "DYLD_",
)
"""Environment variable prefixes affecting interpreter startup before dispatch."""

_SHUTDOWN_TIMEOUT_SECONDS: Final = 2.0
"""Seconds to wait for an idle standby worker to exit cleanly on stdin EOF."""

_SINGLE_SHOT: Final = contextvars.ContextVar[bool]("single_shot", default=False)
"""True during one-shot CLI invocations that must not spawn an unused standby."""

type _Fingerprint = tuple[str, tuple[tuple[str, str], ...]]


@dataclass(frozen=True, slots=True)
class _StandbyEntry:
    """A pre-warmed standby runner paired with the startup fingerprint it inherited."""

    process: subprocess.Popen[bytes]
    fingerprint: _Fingerprint


def _startup_fingerprint() -> _Fingerprint:
    """Returns interpreter path and startup-affecting environment entries."""
    selected = tuple(
        sorted(
            (key, value)
            for key, value in os.environ.items()
            if key.startswith(_STARTUP_PREFIXES)
        )
    )
    return sys.executable, selected


def _spawn_standby() -> subprocess.Popen[bytes]:
    """Spawns a single-use pre-warmed standby `jolty.runner` process."""
    return subprocess.Popen(  # noqa: S603 - fixed interpreter and jolty.runner module
        [sys.executable, "-P", "-m", "jolty.runner", runner.STANDBY_FLAG],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _spawn_direct(directory: Path) -> subprocess.Popen[bytes]:
    """Spawns a direct one-shot `jolty.runner` process for `directory`."""
    return subprocess.Popen(  # noqa: S603 - fixed interpreter and jolty.runner module
        [sys.executable, "-P", "-m", "jolty.runner", str(directory)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _close_stream(stream: IO[bytes] | None) -> None:
    """Closes `stream` if present, ignoring broken-pipe shutdown errors."""
    if stream is None:
        return
    with contextlib.suppress(OSError):
        stream.close()


def _shutdown_process(process: subprocess.Popen[bytes]) -> None:
    """Shuts down an idle standby process cleanly via stdin EOF and reaps it."""
    _close_stream(process.stdin)
    try:
        if process.poll() is None:
            process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        _close_stream(process.stdout)
        _close_stream(process.stderr)


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Kills `process` if running, waits for exit, and closes its pipe handles."""
    _close_stream(process.stdin)
    if process.poll() is None:
        process.kill()
    process.wait()
    _close_stream(process.stdout)
    _close_stream(process.stderr)


class _StandbySlot:
    """Thread-safe holder for at most one pre-warmed single-use standby runner."""

    __slots__ = ("_entry", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entry: _StandbyEntry | None = None

    def warm(self) -> None:
        """Ensures a live standby runner matching the current startup fingerprint."""
        current = _startup_fingerprint()
        stale: subprocess.Popen[bytes] | None = None
        with self._lock:
            existing = self._entry
            if existing is not None:
                if existing.fingerprint == current and existing.process.poll() is None:
                    return
                stale = existing.process
                self._entry = None
            try:
                self._entry = _StandbyEntry(_spawn_standby(), current)
            except OSError:
                self._entry = None
        if stale is not None:
            _shutdown_process(stale)

    def acquire(self) -> subprocess.Popen[bytes] | None:
        """Pops and returns a matching live standby runner, or ``None`` if absent."""
        current = _startup_fingerprint()
        stale: subprocess.Popen[bytes] | None = None
        taken: subprocess.Popen[bytes] | None = None
        with self._lock:
            existing = self._entry
            if existing is not None:
                self._entry = None
                if existing.fingerprint == current and existing.process.poll() is None:
                    taken = existing.process
                else:
                    stale = existing.process
        if stale is not None:
            _shutdown_process(stale)
        return taken

    def close(self) -> None:
        """Cleanly shuts down any idle standby runner."""
        with self._lock:
            existing = self._entry
            self._entry = None
        if existing is not None:
            _shutdown_process(existing.process)


_STANDBY: Final = _StandbySlot()
atexit.register(_STANDBY.close)


@contextlib.contextmanager
def single_shot() -> Generator[None]:
    """Suppresses background standby warming for one-shot CLI invocations."""
    token = _SINGLE_SHOT.set(True)
    try:
        yield
    finally:
        _SINGLE_SHOT.reset(token)


def warm_runner() -> None:
    """Pre-warms one standby runner in the background for upcoming script jobs."""
    if not _SINGLE_SHOT.get():
        _STANDBY.warm()


def close_standby() -> None:
    """Shuts down any idle standby runner cleanly."""
    _STANDBY.close()


def _dispatch_standby(process: subprocess.Popen[bytes], directory: Path) -> bool:
    """Sends the job dispatch envelope to `process` and closes its stdin.

    Returns:
        ``True`` when the dispatch line was written cleanly, or ``False`` if the
        standby process exited before accepting the job.
    """
    payload = (
        json.dumps(
            {
                "directory": str(directory),
                "working_directory": str(Path.cwd()),
                "environment": dict(os.environ),
            }
        ).encode("utf-8")
        + b"\n"
    )
    if process.stdin is None:
        _terminate_process(process)
        return False
    try:
        process.stdin.write(payload)
        process.stdin.flush()
        process.stdin.close()
    except OSError:
        _terminate_process(process)
        return False
    return True


def _start_job_process(directory: Path) -> subprocess.Popen[bytes]:
    """Acquires a standby runner or spawns a direct runner for `directory`."""
    standby = _STANDBY.acquire()
    if not _SINGLE_SHOT.get():
        _STANDBY.warm()
    if standby is not None and _dispatch_standby(standby, directory):
        return standby
    return _spawn_direct(directory)


def _read_capped(
    stream: IO[bytes], limit: int, finished: threading.Event
) -> tuple[str, bool]:
    """Reads `stream` to EOF, keeping at most `limit` characters and a truncation flag.

    Buffered bytes are bounded by the maximum UTF-8 width of `limit` characters
    regardless of total stream volume. Signals `finished` after closing `stream`.
    """
    kept = bytearray()
    dropped = False
    try:
        while chunk := stream.read(1 << 16):
            # A character takes four bytes at most in UTF-8.
            room = 4 * limit - len(kept)
            kept += chunk[:room]
            dropped = dropped or len(chunk) > room
    finally:
        stream.close()
        finished.set()
    text = kept.decode("utf-8", errors="replace")
    return text[:limit], dropped or len(text) > limit


async def run_job(directory: Path, limit: int) -> tuple[str, str, bool, int]:
    """Runs the runner on `directory` and returns stdout, stderr, truncated, and code.

    When the caller stops waiting, through a timeout or a canceled request, the child
    is killed and reaped before cancellation propagates, so no script outlives its
    command and no pipe descriptors leak.

    Args:
        directory: Scratch directory containing `job.json`.
        limit: Maximum characters retained separately from stdout and stderr.

    Returns:
        Four-tuple of `(stdout, stderr, truncated, exit_code)`.

    Raises:
        JoltyError: When stdout or stderr capture pipes cannot be attached.
    """
    process = await asyncio.to_thread(_start_job_process, directory)
    if process.stdout is None or process.stderr is None:
        await asyncio.to_thread(_terminate_process, process)
        message = "could not capture the runner's output"
        raise JoltyError(message)
    stdout_done = threading.Event()
    stderr_done = threading.Event()
    stdout_task = asyncio.create_task(
        asyncio.to_thread(_read_capped, process.stdout, limit, stdout_done)
    )
    stderr_task = asyncio.create_task(
        asyncio.to_thread(_read_capped, process.stderr, limit, stderr_done)
    )
    try:
        (stdout, stdout_truncated), (stderr, stderr_truncated) = await asyncio.gather(
            stdout_task, stderr_task
        )
        code = await asyncio.to_thread(process.wait)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        stdout_done.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        stderr_done.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        await asyncio.shield(
            asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        )
        raise
    return stdout, stderr, stdout_truncated or stderr_truncated, code
