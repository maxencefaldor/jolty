# Copyright 2026 Maxence Faldor
"""Confine an agent's session to its own directories, with the platform's sandbox.

An agent runs with every permission its tools give it, so the sandbox, not the agent,
decides what it reaches: on macOS, Seatbelt, through ``sandbox-exec``, and on Linux,
bubblewrap, ``bwrap``, the mechanisms Claude Code's own sandbox uses for the commands it
runs. The whole session runs inside, the agent and everything it starts: its shell's
commands, its MCP servers and the scripts they run, and the simulators.

A session reads nothing in the places that hold people's data, the `PRIVATE` places:
the home directories, the temporary directories and mounted volumes, but for what it is
given to read there, its own directories and the programs it runs. It reads the rest of
the system, its libraries, its configuration and the programs installed outside those
places, as any program must. It writes nothing but its own directories, and the
devices every program writes, such as ``/dev/null``.

On macOS it also reaches no local service, neither a Unix socket but the system's DNS
resolver nor any address of the machine itself, and starts nothing outside the sandbox,
neither by an Apple Event nor through Launch Services; it may list no other process.
It reaches the network, as Claude Code must reach its service. On Linux, bubblewrap
hides the private places and gives the session processes of its own; it can filter no
Unix socket by its path nor any address, so the machine's other services stay within
the session's reach there.
"""

import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

type Mechanism = Literal["sandbox-exec", "bwrap"]
"""A platform's sandbox: Seatbelt's ``sandbox-exec`` on macOS, bubblewrap on Linux."""

PRIVATE: Final[dict[str, tuple[str, ...]]] = {
    "darwin": ("/Users", "/Volumes", "/private/tmp", "/private/var/folders"),
    "linux": ("/home", "/root", "/tmp", "/var/tmp", "/mnt", "/media", "/run/user"),  # noqa: S108 - hidden from a session, never written
}
"""The places that hold people's data, by platform: the home directories, the
temporary directories, and mounted volumes. ``/tmp`` and ``/var`` are links into
``/private`` on macOS, where ``/private/var/folders`` holds each user's temporary
directory and caches."""
DEVICES: Final = ("/dev/null", "/dev/zero", "/dev/tty", "/dev/dtracehelper")
"""The devices a session writes, as every program may: writes there are discarded,
or reach the terminal."""
RESOLVER: Final = "/private/var/run/mDNSResponder"
"""The Unix socket of macOS's DNS resolver, the one local service a session reaches."""


class Sandbox(BaseModel):
    """How a session is confined: by what mechanism, and what it reads and writes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Mechanism
    private: tuple[str, ...] = Field(
        description="The places whose files the session reads only where it is given."
    )
    reads: tuple[str, ...] = Field(
        description="What it reads there, its programs: each a directory or a file."
    )
    writes: tuple[str, ...] = Field(
        description="Its own directories, the only ones it writes, and reads."
    )


def mechanism() -> Mechanism | None:
    """The sandbox of this platform, if it has one installed.

    It is ``sandbox-exec`` on macOS and ``bwrap`` on Linux, found on the path.
    """
    match sys.platform:
        case "darwin" if shutil.which("sandbox-exec") is not None:
            return "sandbox-exec"
        case "linux" if shutil.which("bwrap") is not None:
            return "bwrap"
    return None


def sandbox(reads: Iterable[Path], writes: Iterable[Path]) -> Sandbox | None:
    """The sandbox that reads `reads` and writes `writes`; null without a mechanism.

    Each path is taken as it is on the disk, its links resolved, since the sandboxes
    match the paths of the files themselves.
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
    """`paths` with their links resolved, each once, in order."""
    return tuple(dict.fromkeys(str(path.resolve()) for path in paths))


def confined(command: Sequence[str], within: Sandbox, work: Path) -> list[str]:
    """`command`, run in the sandbox `within`, from `work`."""
    if within.mechanism == "sandbox-exec":
        text, parameters = profile(within)
        return ["sandbox-exec", "-p", text, *parameters, *command]
    return [*bubblewrap(within, work), "--", *command]


def profile(within: Sandbox) -> tuple[str, list[str]]:
    """The Seatbelt profile of `within`, and the ``-D`` options that give its paths.

    The profile names each path by a parameter, so no path is ever quoted in it.
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
    """The options of bubblewrap for `within`, from `work`, but the command.

    The system is read-only, each private place an empty directory but for what the
    session reads there, read-only, and its own directories, writable; the session has
    processes of its own, and dies with its runner.
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
