# Copyright 2026 Maxence Faldor
"""Model files fetched from where their makers publish them, checked, and kept.

A file is named by its address and the SHA-256 digest of its contents, so what is read
is always what was checked, and is downloaded once into the cache of
[`Settings`][jolty.settings.Settings].
"""

import hashlib
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from jolty.errors import DownloadError
from jolty.settings import Settings

_TIMEOUT = 60.0
"""Seconds a download may wait for the server."""


def fetch(url: str, digest: str, cache: Path | None = None) -> Path:
    """The file at `url`, whose SHA-256 digest is `digest`, kept in `cache`.

    The file is downloaded the first time, and read from the cache after. The cache is
    the settings' unless one is given.

    Raises:
        DownloadError: When the file cannot be downloaded, or its digest is not
            `digest`.
        JoltyError: When the settings are invalid.
    """
    name = PurePosixPath(urlparse(url).path).name or "file"
    kept = Settings.load().cache if cache is None else cache
    path = kept / "downloads" / digest.lower() / name
    if path.is_file() and _digest(path.read_bytes()) == digest.lower():
        return path
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as response:
            contents = response.read()
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        message = f"cannot download {url}: {error}"
        raise DownloadError(message) from None
    found = _digest(contents)
    if found != digest.lower():
        message = f"{url} has SHA-256 digest {found}, not {digest.lower()}"
        raise DownloadError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Each download is written to a file of its own, then renamed onto the kept one at
    # once: fetches of one file at the same time never write into the same file, and
    # none reads a file another is still writing.
    partial = path.with_name(f"{name}.{uuid.uuid4().hex}.partial")
    partial.write_bytes(contents)
    partial.replace(path)
    return path


def _digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()
