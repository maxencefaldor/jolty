# Copyright 2026 Maxence Faldor
"""Platform sandbox confinement restricting an agent session to assigned directories.

Agent sessions run with unrestricted tool permissions, so OS sandboxing enforces
filesystem and process isolation. On macOS, Seatbelt runs through ``sandbox-exec``; on
Linux, bubblewrap runs through ``bwrap``. Both match the underlying mechanisms used by
Claude Code for command sandboxing. Confinement covers the full session process tree,
including shell commands, MCP servers, helper scripts, and simulators.

Private locations listed in `PRIVATE` (home directories, temporary directories, and
mounted volumes) are unreadable except for explicitly granted read paths, session
directories, and required executables. System libraries, configuration files, and
binaries outside `PRIVATE` remain readable as needed by standard processes. Write
access is limited to the session directories and standard device files such as
``/dev/null``.

On macOS, Seatbelt also blocks local Unix sockets except the system DNS resolver,
blocks loopback network addresses, blocks Apple Events and Launch Services, and hides
other processes while allowing outbound network access for Claude Code. On Linux,
bubblewrap hides private paths and isolates the PID namespace, while local Unix
sockets and loopback addresses remain reachable because bubblewrap cannot filter them
by path or address.
"""

import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

type Mechanism = Literal["sandbox-exec", "bwrap"]
"""Platform sandbox backend: Seatbelt ``sandbox-exec`` (macOS) or ``bwrap`` (Linux)."""

PRIVATE: Final[dict[str, tuple[str, ...]]] = {
    "darwin": ("/Users", "/Volumes", "/private/tmp", "/private/var/folders"),
    "linux": ("/home", "/root", "/tmp", "/var/tmp", "/mnt", "/media", "/run/user"),  # noqa: S108 - hidden from a session, never written
}
"""User-data root paths hidden per platform: home, temporary, and volume directories.

On macOS, ``/tmp`` and ``/var`` resolve into ``/private``, where
``/private/var/folders`` stores per-user temporary directories and caches.
"""
DEVICES: Final = ("/dev/null", "/dev/zero", "/dev/tty", "/dev/dtracehelper")
"""Standard device files writable inside a sandbox to discard output or reach a TTY."""
RESOLVER: Final = "/private/var/run/mDNSResponder"
"""macOS DNS resolver Unix socket, the sole local service reachable from a session."""


class Sandbox(BaseModel):
    """Confinement specification recording the sandbox backend and allowed paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Mechanism
    private: tuple[str, ...] = Field(
        description="Private root paths hidden from the session unless explicitly "
        "granted."
    )
    reads: tuple[str, ...] = Field(
        description="Directories and files explicitly readable inside private roots, "
        "such as executables."
    )
    writes: tuple[str, ...] = Field(
        description="Session directories writable and readable by the confined "
        "process tree."
    )


def mechanism() -> Mechanism | None:
    """Returns the sandbox backend installed on the current platform, or null if absent.

    Checks ``PATH`` for ``sandbox-exec`` on macOS and ``bwrap`` on Linux.
    """
    match sys.platform:
        case "darwin" if shutil.which("sandbox-exec") is not None:
            return "sandbox-exec"
        case "linux" if shutil.which("bwrap") is not None:
            return "bwrap"
    return None


def sandbox(reads: Iterable[Path], writes: Iterable[Path]) -> Sandbox | None:
    """Builds a `Sandbox` permitting `reads` and `writes`, or returns null if absent.

    Symlinks are resolved to canonical filesystem paths because sandbox rules match
    real paths on disk.
    """
    found = mechanism()
    if found is None:
        return None
    return Sandbox(
        mechanism=found,
        private=PRIVATE[sys.platform],
        reads=_resolved(reads),
        writes=_resolved(writes),
    )


def _resolved(paths: Iterable[Path]) -> tuple[str, ...]:
    """Resolves symlinks in `paths` and returns unique canonical strings in order."""
    return tuple(dict.fromkeys(str(path.resolve()) for path in paths))


def confined(command: Sequence[str], within: Sandbox, work: Path) -> list[str]:
    """Wraps `command` to execute inside `within` using `work` as working directory."""
    if within.mechanism == "sandbox-exec":
        text, parameters = profile(within)
        return ["sandbox-exec", "-p", text, *parameters, *command]
    return [*bubblewrap(within, work), "--", *command]


def profile(within: Sandbox) -> tuple[str, list[str]]:
    """Builds the Seatbelt profile text and ``-D`` parameter flags for `within`.

    Paths are passed through Seatbelt parameters rather than embedded literals to
    avoid string-quoting issues.
    """
    groups = {"PRIVATE": within.private, "READ": within.reads, "WRITE": within.writes}
    parameters = [
        option
        for group, paths in groups.items()
        for index, path in enumerate(paths)
        for option in ("-D", f"{group}{index}={path}")
    ]

    def subpaths(group: str) -> str:
        return " ".join(
            f'(subpath (param "{group}{index}"))' for index in range(len(groups[group]))
        )

    devices = " ".join(f'(literal "{device}")' for device in DEVICES)
    readable = f"{subpaths('READ')} {subpaths('WRITE')}".strip()
    rules = [
        "(version 1)",
        "(allow default)",
        f"(deny file-read-data file-read-xattr {subpaths('PRIVATE')})",
        *([f"(allow file-read-data file-read-xattr {readable})"] if readable else []),
        "(deny file-write*)",
        f'(allow file-write* {subpaths("WRITE")} {devices} (regex #"^/dev/fd/"))',
        "(deny appleevent-send)",
        "(deny lsopen)",
        "(deny network-outbound (remote unix-socket))",
        f'(allow network-outbound (remote unix-socket (path-literal "{RESOLVER}")))',
        '(deny network-outbound (remote ip "localhost:*"))',
        "(deny process-info* (target others))",
    ]
    return "\n".join(rules), parameters


def bubblewrap(within: Sandbox, work: Path) -> list[str]:
    """Builds the ``bwrap`` invocation prefix for `within` starting in `work`.

    Mounts the root filesystem read-only, overlays each private directory with an
    empty tmpfs, binds granted read paths read-only, and binds session directories
    read-write. Runs the session in a separate PID namespace that terminates when the
    parent exits.
    """
    hidden = [place for place in within.private if Path(place).is_dir()]
    return [
        "bwrap",
        *("--ro-bind", "/", "/"),
        *("--dev", "/dev"),
        *("--proc", "/proc"),
        *(option for place in hidden for option in ("--tmpfs", place)),
        *(option for path in within.reads for option in ("--ro-bind", path, path)),
        *(option for path in within.writes for option in ("--bind", path, path)),
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        *("--chdir", str(work.resolve())),
    ]
