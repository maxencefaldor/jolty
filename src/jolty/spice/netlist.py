# Copyright 2026 Maxence Faldor
r"""SPICE's text: its statements, and the tokens each statement splits into.

A line that starts with ``*`` is a comment, and ``;`` starts one anywhere, as in PSpice,
as does ``$`` after a space, as in ngspice. A line that starts with ``+`` continues the
statement before it.

Examples:
    >>> [card.text for card in statements("R1 a b 1k ; load\n+ TC=1m\n* note")]
    ['R1 a b 1k TC=1m']
    >>> tokens("GD1 1 2 TABLE {V(1,2)} = (0,0) (1m,1u)")
    ['GD1', '1', '2', 'TABLE', '{V(1,2)}', '=', '(0,0)', '(1m,1u)']
"""

from dataclasses import dataclass
from typing import Final

from jolty.errors import SpiceError


@dataclass(frozen=True, slots=True)
class Statement:
    """One statement of a SPICE file, with the file and the line it starts on."""

    file: str
    line: int
    text: str

    def fail(self, message: str) -> SpiceError:
        """An error stating `message` about this statement."""
        return SpiceError(f"{self.file}:{self.line}: {message}: `{self.text}`")


def _strip_comment(text: str) -> str:
    """Removes inline SPICE comments outside single- or double-quoted strings.

    Treats ``;`` anywhere outside quotes and ``$`` preceded by whitespace outside
    quotes as the start of an inline comment.
    """
    if ";" not in text and "$" not in text:
        return text
    quote: str | None = None
    for index, character in enumerate(text):
        if quote is not None:
            if character == quote:
                quote = None
        elif character in {"'", '"'}:
            quote = character
        elif character == ";":
            return text[:index]
        elif character == "$" and index > 0 and text[index - 1].isspace():
            return text[: index - 1]
    return text


def statements(text: str, file: str = "<text>") -> list[Statement]:
    """The statements of `text`, from the file `file`."""
    found: list[Statement] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        printable = (
            raw
            if raw.isprintable()
            else "".join(
                " " if character == "\t" else character
                for character in raw
                if character == "\t" or character.isprintable()
            )
        )
        stripped = printable.lstrip()
        if not stripped or stripped.startswith("*"):
            continue
        line = _strip_comment(printable).strip()
        if not line:
            continue
        if line.startswith("+"):
            if not found:
                message = f"{file}:{number}: a continuation line continues nothing"
                raise SpiceError(message)
            last = found[-1]
            found[-1] = Statement(
                last.file, last.line, f"{last.text} {line[1:].strip()}"
            )
        else:
            found.append(Statement(file, number, line))
    return found


_CLOSING: Final = {"(": ")", "{": "}", "'": "'", '"': '"'}


def tokens(text: str) -> list[str]:
    """The tokens of a statement: words, ``=``, and groups kept whole.

    A group is in parentheses, braces or quotes; one in parentheses joins a word it
    follows directly, as in ``POLY(2)``. Commas between tokens separate them, as spaces
    do.
    """
    found: list[str] = []
    current = ""
    position = 0
    while position < len(text):
        character = text[position]
        if character in _CLOSING:
            end = _group_end(text, position)
            group = text[position:end]
            if character == "(" and current:
                current += group
            else:
                current = _flush(found, current)
                found.append(group)
            position = end
        elif character.isspace() or character == ",":
            current = _flush(found, current)
            position += 1
        elif character == "=":
            current = _flush(found, current)
            found.append("=")
            position += 1
        else:
            current += character
            position += 1
    _flush(found, current)
    return found


def _flush(found: list[str], current: str) -> str:
    """Appends `current` to `found` if it is a token; returns an empty token."""
    if current:
        found.append(current)
    return ""


def _group_end(text: str, start: int) -> int:
    """Where the group opening at `start` ends: after its closing mark.

    A group that never closes ends with `text`.
    """
    opening = text[start]
    closing = _CLOSING[opening]
    depth = 0
    for position in range(start, len(text)):
        character = text[position]
        if opening in {"'", '"'}:
            if position > start and character == opening:
                return position + 1
        else:
            depth += (character == opening) - (character == closing)
            if depth == 0:
                return position + 1
    return len(text)
