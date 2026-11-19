# Copyright 2026 Maxence Faldor
"""Offline guide pages and example scripts served by the CLI and MCP server.

Documents load from a repository checkout when accessible, falling back to the packaged
``assets/docs.json`` bundle. Catalogs are cached and wrapped in immutable views so
repeated reads avoid redundant disk scans while honoring bundle changes and test hooks.
"""

import ast
import functools
import importlib.resources
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import Field
from pydantic_settings import CliPositionalArg

from jolty.errors import JoltyError
from jolty.ir import FrozenModel

type DocumentSection = Literal["guide", "examples"]
"""Whether an offline document belongs to the guide or the example scripts."""

DOCS_URI_PREFIX: Final = "jolty://docs"
"""Base URI for offline documentation resources exposed over MCP."""

GUIDE_MIME_TYPE: Final = "text/markdown"
"""MIME type of offline guide pages."""

EXAMPLE_MIME_TYPE: Final = "text/x-python"
"""MIME type of offline example scripts."""

_DOCS_SECTIONS: Final[tuple[DocumentSection, ...]] = ("guide", "examples")
_DOCS_BUNDLE: Final = "docs.json"
_MARKDOWN_LINK: Final = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MIN_PRINTABLE_ASCII: Final = 32
_QUALIFIED_SEGMENT_COUNT: Final = 2
_REPOSITORY_PARENT_INDEX: Final = 3

_DOCUMENT_SECTION_DESCRIPTION: Final = (
    "Whether the document is a 'guide' page or an 'examples' script."
)
_DOCUMENT_SLUG_DESCRIPTION: Final = (
    "The topic slug, such as 'units', 'rc_filter' or 'cluster/amplifier'."
)
_DOCUMENT_URI_DESCRIPTION: Final = (
    "The MCP resource URI, such as 'jolty://docs/guide/units'."
)
_DOCUMENT_TITLE_DESCRIPTION: Final = (
    "The heading of a guide page, or the slug of an example script."
)
_DOCUMENT_SUMMARY_DESCRIPTION: Final = "The first sentence describing the document."
_DOCUMENT_MIME_DESCRIPTION: Final = (
    "MIME type: 'text/markdown' for guide pages, 'text/x-python' for examples."
)

type _Catalog = MappingProxyType[DocumentSection, MappingProxyType[str, str]]
"""Immutable offline documentation catalog mapped by section and slug."""


class DocumentSummary(FrozenModel):
    """Metadata for one offline guide page or example script."""

    section: DocumentSection = Field(description=_DOCUMENT_SECTION_DESCRIPTION)
    slug: str = Field(description=_DOCUMENT_SLUG_DESCRIPTION)
    uri: str = Field(description=_DOCUMENT_URI_DESCRIPTION)
    title: str = Field(description=_DOCUMENT_TITLE_DESCRIPTION)
    summary: str = Field(description=_DOCUMENT_SUMMARY_DESCRIPTION)
    mime_type: str = Field(description=_DOCUMENT_MIME_DESCRIPTION)


class Document(DocumentSummary):
    """One offline document: its metadata and full UTF-8 text."""

    content: str = Field(description="The full UTF-8 text of the document.")


class ReadDocsRequest(FrozenModel):
    """List or read jolty's offline guide pages and example scripts."""

    topic: CliPositionalArg[str | None] = Field(
        default=None,
        description="Topic to read: a bare slug such as 'units' or 'rc_filter', "
        "a qualified name such as 'guide/units' or 'examples/cluster/amplifier', "
        "or a resource URI such as 'jolty://docs/guide/units'. Lists all topics "
        "when omitted.",
    )
    section: DocumentSection | None = Field(
        default=None,
        description="Limit listing or lookup to 'guide' or 'examples'.",
    )


def describe_entries(entries: tuple[DocumentSummary, ...]) -> str:
    """Formats a grouped listing of offline documentation topics."""
    sections: list[str] = []
    for name in _DOCS_SECTIONS:
        grouped = [entry for entry in entries if entry.section == name]
        if not grouped:
            continue
        noun = "page" if name == "guide" else "script"
        plural = f"{len(grouped)} {noun}{'' if len(grouped) == 1 else 's'}"
        width = max(len(entry.slug) for entry in grouped)
        header = f"{name} ({plural}, {DOCS_URI_PREFIX}/{name}/<slug>):"
        lines = [header]
        for entry in grouped:
            detail = (
                f"{entry.title}: {entry.summary}"
                if name == "guide" and entry.summary
                else (entry.summary or entry.title)
            )
            lines.append(f"  {entry.slug:<{width}}  {detail}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _repository_root() -> Path | None:
    """The repository root inferred from this module's location, if accessible."""
    try:
        parents = Path(__file__).resolve().parents
    except OSError:
        return None
    return (
        parents[_REPOSITORY_PARENT_INDEX]
        if len(parents) > _REPOSITORY_PARENT_INDEX
        else None
    )


def _load_directory_documents(directory: Path, suffix: str) -> dict[str, str]:
    """Reads every file under `directory` ending in `suffix`, keyed by POSIX slug."""
    documents: dict[str, str] = {}
    pending = [directory]
    while pending:
        current = pending.pop()
        for entry in sorted(current.iterdir(), key=lambda path: path.name):
            if entry.is_dir():
                if not entry.name.startswith(".") and entry.name != "__pycache__":
                    pending.append(entry)
                continue
            if entry.suffix == suffix and entry.is_file():
                slug = entry.relative_to(directory).with_suffix("").as_posix()
                documents[slug] = entry.read_text(encoding="utf-8")
    return dict(sorted(documents.items()))


def _load_docs_from_checkout(
    root: Path,
) -> dict[DocumentSection, dict[str, str]] | None:
    """Loads guide pages and examples from a checkout at `root`, if readable."""
    guide_directory = root / "docs" / "guide"
    examples_directory = root / "examples"
    try:
        if not (guide_directory.is_dir() and examples_directory.is_dir()):
            return None
        guide = _load_directory_documents(guide_directory, ".md")
        examples = _load_directory_documents(examples_directory, ".py")
    except OSError:
        return None
    if not guide or not examples:
        return None
    return {"guide": guide, "examples": examples}


def _parse_docs_bundle(raw: str) -> dict[DocumentSection, dict[str, str]] | None:
    """Parses a bundled ``docs.json`` payload into sorted section mappings."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    catalog_map: dict[DocumentSection, dict[str, str]] = {}
    for section in _DOCS_SECTIONS:
        entries = payload.get(section)
        if not isinstance(entries, dict):
            return None
        normalized: dict[str, str] = {}
        for slug, text in sorted(entries.items()):
            if not isinstance(slug, str) or not isinstance(text, str):
                return None
            normalized[slug] = text.replace("\r\n", "\n")
        if not normalized:
            return None
        catalog_map[section] = normalized
    return catalog_map


def _freeze_catalog(
    catalog_map: Mapping[DocumentSection, Mapping[str, str]],
) -> _Catalog:
    """Wraps a parsed docs catalog in immutable `MappingProxyType` views."""
    return MappingProxyType(
        {
            section: MappingProxyType(dict(entries))
            for section, entries in catalog_map.items()
        }
    )


@functools.lru_cache(maxsize=8)
def _cached_checkout(root: Path) -> _Catalog | None:
    """Loads and caches guide pages and examples from a checkout at `root`."""
    service_module = sys.modules.get("jolty.service")
    loader = getattr(
        service_module, "_load_docs_from_checkout", _load_docs_from_checkout
    )
    loaded = loader(root)
    return None if loaded is None else _freeze_catalog(loaded)


@functools.lru_cache(maxsize=8)
def _cached_bundle_file(
    path: Path, _mtime_ns: int, _size: int, _inode: int
) -> _Catalog | None:
    """Reads, parses, and caches a filesystem ``docs.json`` bundle by stat stamp."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    parsed = _parse_docs_bundle(raw)
    return None if parsed is None else _freeze_catalog(parsed)


@functools.lru_cache(maxsize=4)
def _cached_bundle_text(raw: str) -> _Catalog | None:
    """Parses and caches a raw ``docs.json`` payload."""
    parsed = _parse_docs_bundle(raw)
    return None if parsed is None else _freeze_catalog(parsed)


def _load_docs_from_bundle() -> _Catalog | None:
    """Loads guide pages and examples from the packaged ``assets/docs.json`` bundle."""
    try:
        resource = importlib.resources.files("jolty").joinpath("assets", _DOCS_BUNDLE)
    except ModuleNotFoundError, OSError:
        return None
    if isinstance(resource, Path):
        try:
            info = resource.stat()
        except OSError:
            return None
        return _cached_bundle_file(
            resource, info.st_mtime_ns, info.st_size, info.st_ino
        )
    try:
        raw = resource.read_text(encoding="utf-8")
    except OSError:
        return None
    return _cached_bundle_text(raw)


def _docs_catalog() -> _Catalog:
    """Loads offline guide pages and examples from the checkout or packaged bundle.

    Raises:
        JoltyError: When neither a readable checkout nor a packaged bundle exists.
    """
    service_module = sys.modules.get("jolty.service")
    find_root = getattr(service_module, "_repository_root", _repository_root)
    root = find_root()
    if root is not None:
        checkout = _cached_checkout(root)
        if checkout is not None:
            return checkout
    bundled = _load_docs_from_bundle()
    if bundled is not None:
        return bundled
    message = "offline documentation bundle is not available in this installation"
    raise JoltyError(message)


def _guide_title_and_summary(slug: str, content: str) -> tuple[str, str]:
    """Extracts the Markdown heading and first summary sentence of a guide page."""
    title = slug
    summary = ""
    seen_heading = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not seen_heading and stripped.startswith("# "):
            title = stripped.removeprefix("# ").strip()
            seen_heading = True
            continue
        if stripped.startswith("#"):
            continue
        summary = _MARKDOWN_LINK.sub(r"\1", stripped)
        break
    return title, summary


def _example_summary(content: str) -> str:
    """Extracts the first line of an example script's module docstring."""
    try:
        module = ast.parse(content)
    except SyntaxError:
        return ""
    docstring = ast.get_docstring(module)
    if docstring is None:
        return ""
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _build_summary(
    section: DocumentSection, slug: str, content: str
) -> DocumentSummary:
    """Builds metadata for one offline document."""
    uri = f"{DOCS_URI_PREFIX}/{section}/{slug}"
    if section == "guide":
        title, summary = _guide_title_and_summary(slug, content)
        mime_type = GUIDE_MIME_TYPE
    else:
        title = slug
        summary = _example_summary(content)
        mime_type = EXAMPLE_MIME_TYPE
    return DocumentSummary(
        section=section,
        slug=slug,
        uri=uri,
        title=title,
        summary=summary,
        mime_type=mime_type,
    )


def _available_topics_message(
    loaded: Mapping[DocumentSection, Mapping[str, str]],
    section: DocumentSection | None = None,
) -> str:
    """Formats the factual listing of available topics for error messages."""
    if section is not None:
        return f"the {section} topics are {', '.join(loaded[section])}"
    guide_names = ", ".join(loaded["guide"])
    example_names = ", ".join(loaded["examples"])
    return (
        f"the guide topics are {guide_names}; the examples topics are {example_names}"
    )


def _validate_segments(original: str, value: str, available: str) -> tuple[str, ...]:
    """Validates a topic path, rejecting traversal, absolute paths and control bytes.

    Raises:
        JoltyError: When `value` is empty, absolute, contains ``..``, backslashes,
            colons or control characters.
    """
    invalid_control = any(ord(character) < _MIN_PRINTABLE_ASCII for character in value)
    segments = tuple(value.split("/"))
    if (
        not value
        or value != value.strip()
        or invalid_control
        or "\\" in value
        or ":" in value
        or value.startswith("/")
        or value.endswith("/")
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        message = (
            f"invalid documentation topic {original!r}: a topic names a bare slug "
            f"with no '..', absolute path, backslash or control characters; {available}"
        )
        raise JoltyError(message)
    return segments


def _read_document_from(
    loaded: Mapping[DocumentSection, Mapping[str, str]],
    section: DocumentSection,
    slug: str,
) -> Document:
    """Reads one document from an already-loaded `loaded` catalog."""
    available = _available_topics_message(loaded, section)
    segments = _validate_segments(slug, slug, available)
    normalized = "/".join(segments)
    content = loaded[section].get(normalized)
    if content is None:
        message = f"no {section} topic is named {slug!r}; {available}"
        raise JoltyError(message)
    summary = _build_summary(section, normalized, content)
    return Document(
        section=summary.section,
        slug=summary.slug,
        uri=summary.uri,
        title=summary.title,
        summary=summary.summary,
        mime_type=summary.mime_type,
        content=content,
    )


def document_summaries(
    section: DocumentSection | None = None,
) -> tuple[DocumentSummary, ...]:
    """Lists available offline guide pages and examples in deterministic order.

    Raises:
        JoltyError: When neither a readable checkout nor a packaged bundle exists.
    """
    loaded = _docs_catalog()
    sections = _DOCS_SECTIONS if section is None else (section,)
    return tuple(
        _build_summary(current, slug, content)
        for current in sections
        for slug, content in loaded[current].items()
    )


def read_document(section: DocumentSection, slug: str) -> Document:
    """Reads one offline document from `section` by exact validated `slug`.

    Raises:
        JoltyError: When `slug` is invalid, attempts path traversal, or matches no
            document in `section`.
    """
    return _read_document_from(_docs_catalog(), section, slug)


def resolve_document(topic: str, section: DocumentSection | None = None) -> Document:
    """Resolves a bare slug, qualified name or ``jolty://docs/...`` URI to a document.

    Raises:
        JoltyError: When `topic` is invalid, mismatches `section`, is ambiguous, or
            matches no available document.
    """
    loaded = _docs_catalog()
    available = _available_topics_message(loaded, section)
    prefix = f"{DOCS_URI_PREFIX}/"
    trimmed = topic.removeprefix(prefix) if topic.startswith(prefix) else topic
    segments = _validate_segments(topic, trimmed, available)
    if len(segments) >= _QUALIFIED_SEGMENT_COUNT and segments[0] in _DOCS_SECTIONS:
        explicit: DocumentSection = "guide" if segments[0] == "guide" else "examples"
        if section is not None and section != explicit:
            message = (
                f"topic {topic!r} belongs to section {explicit!r}, not {section!r}"
            )
            raise JoltyError(message)
        return _read_document_from(loaded, explicit, "/".join(segments[1:]))
    slug = "/".join(segments)
    if section is not None:
        return _read_document_from(loaded, section, slug)
    matches = [candidate for candidate in _DOCS_SECTIONS if slug in loaded[candidate]]
    if len(matches) == 1:
        return _read_document_from(loaded, matches[0], slug)
    if len(matches) > 1:
        qualified = ", ".join(f"{candidate}/{slug}" for candidate in matches)
        message = f"topic {topic!r} is ambiguous and matches {qualified}"
        raise JoltyError(message)
    message = f"no documentation topic is named {topic!r}; {available}"
    raise JoltyError(message)
