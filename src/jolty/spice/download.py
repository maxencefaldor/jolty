# Copyright 2026 Maxence Faldor
"""Digest-verified downloading and caching of published SPICE model files.

Each download is pinned by URL and SHA-256 digest and stored once in the cache
configured by [`Settings`][jolty.settings.Settings].
"""

__lazy_modules__ = [
    "hashlib",
    "jolty.settings",
    "urllib.error",
    "urllib.parse",
    "urllib.request",
    "uuid",
]

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
    """Fetches the file at `url`, verifies SHA-256 `digest`, and caches it in `cache`.

    Subsequent calls return the verified cached file without downloading again. Uses
    the configured settings cache directory when `cache` is None.

    Args:
        url: Source URL of the file to download.
        digest: Expected 64-character hexadecimal SHA-256 digest.
        cache: Optional cache root directory overriding runtime settings.

    Returns:
        Local filesystem path to the verified cached file.

    Raises:
        DownloadError: When `digest` is malformed, `url` names an unsafe filename,
            the transfer fails, the payload exceeds `_MAX_DOWNLOAD_BYTES`, or the
            computed digest does not match `digest`.
        JoltyError: When runtime settings are invalid.
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
    # Stage into a unique temporary file before atomic rename so concurrent fetches
    # never clobber one another or expose partial downloads.
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    try:
        found = _copy_hashed(response, partial, url)
        _check_digest(found, expected, url)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
