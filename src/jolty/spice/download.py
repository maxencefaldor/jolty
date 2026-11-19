# Copyright 2026 Maxence Faldor
"""Model files fetched from where their makers publish them, checked, and kept.

A file is named by its address and the SHA-256 digest of its contents, so what is read
is always what was checked, and is downloaded once into the cache of
[`Settings`][jolty.settings.Settings].
"""

import hashlib
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import urlparse

from jolty.errors import DownloadError
from jolty.settings import Settings

_TIMEOUT: Final = 60.0
"""Seconds a download may wait for the server."""

_DIGEST_PATTERN: Final = re.compile(r"\A[0-9a-fA-F]{64}\Z")
"""Pattern matching a 64-character hexadecimal SHA-256 digest."""

_CHUNK_BYTES: Final = 64 * 1024
"""Streaming read chunk size, in bytes (64 KiB)."""

_MAX_DOWNLOAD_BYTES: Final = 512 * 1024 * 1024
"""Maximum size accepted for a single downloaded file, in bytes (512 MiB).

Accommodates full open-source process design kit archives while bounding memory
and disk consumption on unexpected or runaway responses.
"""


class _Readable(Protocol):
    """Stream supporting chunked binary reads."""

    def read(self, size: int = -1, /) -> bytes:
        """Reads up to `size` bytes from the stream."""
        ...


def fetch(url: str, digest: str, cache: Path | None = None) -> Path:
    """The file at `url`, whose SHA-256 digest is `digest`, kept in `cache`.

    The file is downloaded the first time, and read from the cache after. The cache is
    the settings' unless one is given.

    Raises:
        DownloadError: When `digest` is malformed, `url` names no safe file, the file
            cannot be downloaded, its size exceeds `_MAX_DOWNLOAD_BYTES`, or its digest
            is not `digest`.
        JoltyError: When the settings are invalid.
    """
    if _DIGEST_PATTERN.fullmatch(digest) is None:
        message = f"a SHA-256 digest is 64 hexadecimal digits, not {digest!r}"
        raise DownloadError(message)
    expected = digest.lower()
    name = urlparse(url).path.rsplit("/", 1)[-1]
    if name in {"", ".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        message = f"download URL {url!r} names no safe file"
        raise DownloadError(message)
    kept = Settings.load().cache if cache is None else cache
    path = kept / "downloads" / expected / name
    if path.is_file() and _file_digest(path) == expected:
        return path
    try:
        with urllib.request.urlopen(  # noqa: S310 - digest verifies downloaded payload
            url, timeout=_TIMEOUT
        ) as response:
            _write_verified(response, path, expected, url)
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        message = f"cannot download {url}: {error}"
        raise DownloadError(message) from None
    return path


def _file_digest(path: Path) -> str:
    """Computes the lowercase SHA-256 hex digest of the file at `path`."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_digest(found: str, expected: str, url: str) -> None:
    """Checks that `found` equals `expected`, raising `DownloadError` otherwise."""
    if found != expected:
        message = f"{url} has SHA-256 digest {found}, not {expected}"
        raise DownloadError(message)


def _copy_hashed(response: _Readable, destination: Path, url: str) -> str:
    """Streams `response` into `destination` under the size cap and returns SHA-256."""
    hasher = hashlib.sha256()
    total = 0
    with destination.open("wb") as handle:
        while chunk := response.read(_CHUNK_BYTES):
            total += len(chunk)
            if total > _MAX_DOWNLOAD_BYTES:
                message = (
                    f"{url} exceeds maximum download size of "
                    f"{_MAX_DOWNLOAD_BYTES} bytes"
                )
                raise DownloadError(message)
            hasher.update(chunk)
            handle.write(chunk)
    return hasher.hexdigest()


def _write_verified(
    response: _Readable,
    path: Path,
    expected: str,
    url: str,
) -> None:
    """Streams `response` to a temporary file, verifies SHA-256, and replaces `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each download is written to a file of its own, then renamed onto the kept one at
    # once: fetches of one file at the same time never write into the same file, and
    # none reads a file another is still writing.
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    try:
        found = _copy_hashed(response, partial, url)
        _check_digest(found, expected, url)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
