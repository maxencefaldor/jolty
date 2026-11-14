# Copyright 2026 Maxence Faldor
"""What an agent's transcript says of its session: its turns, tool calls and report.

Claude Code, run with ``--output-format stream-json --verbose``, prints its session as
events, one JSON object a line: first a ``system`` event that names the model and the
agent's version, then each message of the model, ``assistant``, and of its tools,
``user``, and last a ``result`` event, the agent's own report of the session's length,
cost and tokens. A message the model writes may stream as several events, one per block
of its content, which share the message's id; a block of type ``tool_use`` calls a tool,
by its ``name``. A subagent's messages count with the agent's own.

A line that holds no JSON object, such as a warning the agent wrote to its error
stream, which the runner keeps in the same transcript, is not an event.
"""

import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
)

type Event = dict[str, JsonValue]
"""An event of a transcript: a JSON object."""

_JSON: Final = TypeAdapter[JsonValue](JsonValue)


class Session(BaseModel):
    """What a transcript says of the agent's session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = Field(
        default=None, description="The model the session ran, as its first event says."
    )
    agent: str | None = Field(
        default=None, description="The agent's version, as its first event says."
    )
    turns: int = Field(
        description="The messages the model wrote, each counted once, however many "
        "events it streamed in."
    )
    tool_calls: dict[str, int] = Field(
        description="How many times the model called each tool, by the tool's name."
    )
    reported_turns: int | None = Field(
        default=None, description="The turns the agent reports; null without a report."
    )
    seconds: float | None = Field(
        default=None,
        description="The session's wall time the agent reports, in seconds; null "
        "without a report.",
    )
    cost: float | None = Field(
        default=None,
        description="The session's cost the agent reports, in US dollars; null "
        "without a report.",
    )
    tokens: dict[str, int] = Field(
        default_factory=dict,
        description="The tokens the agent reports, by kind, such as output_tokens.",
    )
    outcome: str | None = Field(
        default=None,
        description="How the agent reports the session ended, such as success; null "
        "without a report.",
    )

    @property
    def calls(self) -> int:
        """How many tools the model called in all."""
        return sum(self.tool_calls.values())


def events(path: Path) -> Iterator[Event]:
    """The events of the transcript at `path`, in order."""
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                event = _JSON.validate_json(line)
            except ValidationError:
                continue
            if isinstance(event, dict):
                yield event


def read(path: Path) -> Session | None:
    """What the transcript at `path` says of its session; null when it holds no event.

    A transcript holds no event when its agent printed no stream of JSON events, as an
    agent other than Claude Code may not.
    """
    found = False
    model: str | None = None
    agent: str | None = None
    messages: set[str] = set()
    calls: dict[str, str] = {}
    report: Event = {}
    for number, event in enumerate(events(path)):
        match event:
            case {"type": "system", "subtype": "init"}:
                found = True
                model = _text(event.get("model"))
                agent = _text(event.get("claude_code_version"))
            case {"type": "assistant", "message": dict(message)}:
                found = True
                messages.add(_text(message.get("id")) or f"event {number}")
                for place, block in enumerate(_blocks(message)):
                    if block.get("type") == "tool_use":
                        key = _text(block.get("id")) or f"event {number} block {place}"
                        calls[key] = _text(block.get("name")) or "unnamed"
            case {"type": "result"}:
                found = True
                report = event
    if not found:
        return None
    return Session(
        model=model,
        agent=agent,
        turns=len(messages),
        tool_calls=dict(sorted(Counter(calls.values()).items())),
        reported_turns=_integer(report.get("num_turns")),
        seconds=_seconds(report.get("duration_ms")),
        cost=_number(report.get("total_cost_usd")),
        tokens=_tokens(report.get("usage")),
        outcome=_text(report.get("subtype")),
    )


def _blocks(message: Event) -> list[Event]:
    match message.get("content"):
        case list(blocks):
            return [block for block in blocks if isinstance(block, dict)]
    return []


def _text(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: JsonValue) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: JsonValue) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _seconds(milliseconds: JsonValue) -> float | None:
    found = _number(milliseconds)
    return None if found is None else found / 1e3


def _tokens(usage: JsonValue) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    return {
        str(kind): count
        for kind, count in usage.items()
        if isinstance(count, int) and not isinstance(count, bool)
    }


def main() -> None:
    """Prints what the transcript given says of its session, as JSON."""
    session = read(Path(sys.argv[1]))
    print("null" if session is None else session.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
