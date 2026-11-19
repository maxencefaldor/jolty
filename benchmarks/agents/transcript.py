# Copyright 2026 Maxence Faldor
"""Parser for agent session transcripts capturing turns, tool calls, and reports.

Claude Code with ``--output-format stream-json --verbose`` writes one JSON object per
line. The stream begins with a ``system`` event listing the model, agent version,
tools, MCP servers, permission mode, skills, plugins, and API key source. Subsequent
``assistant`` and ``user`` events record model messages and tool results. A final
``result`` event reports session duration, cost, token usage, outcome, errors, and
final response text.

A single model message can stream across multiple events that share a message id, one
per content block. Content blocks of type ``tool_use`` invoke a tool by ``name``, and
blocks of type ``text`` carry response prose. Subagent messages are counted alongside
top-level agent messages. Field names follow the Claude Agent SDK ``SDKSystemMessage``
and ``SDKResultMessage`` schemas.

Non-JSON lines, such as stderr warnings captured in the same transcript file, are
skipped.
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
"""Single JSON object representing a transcript event."""

_JSON: Final = TypeAdapter[JsonValue](JsonValue)


class Setup(BaseModel):
    """Runtime environment reported by the agent in the initial system event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cwd: str | None = Field(
        default=None, description="Working directory reported by the agent."
    )
    tools: tuple[str, ...] = Field(
        default=(), description="Available tool names reported by the agent."
    )
    mcp_servers: dict[str, str] = Field(
        default_factory=dict,
        description="Configured MCP servers mapped from server name to status, such "
        "as connected.",
    )
    permission_mode: str | None = Field(
        default=None,
        description="Permission mode reported by the agent, such as bypassPermissions.",
    )
    api_key_source: str | None = Field(
        default=None,
        description="API key source reported by the agent, such as user or none.",
    )
    output_style: str | None = Field(
        default=None, description="Output style configured for the agent session."
    )
    skills: tuple[str, ...] = Field(
        default=(), description="Loaded skill names reported by the agent."
    )
    agents: tuple[str, ...] = Field(
        default=(), description="Configured subagent names reported by the agent."
    )
    slash_commands: tuple[str, ...] = Field(
        default=(), description="Available slash command names reported by the agent."
    )
    plugins: tuple[str, ...] = Field(
        default=(), description="Loaded plugin names reported by the agent."
    )


class Session(BaseModel):
    """Structured summary extracted from an agent session transcript."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    setup: Setup | None = Field(
        default=None,
        description="Agent runtime environment from the initial system event, or null "
        "if absent.",
    )

    model: str | None = Field(
        default=None,
        description="Model identifier reported in the initial system event.",
    )
    agent: str | None = Field(
        default=None,
        description="Agent version string reported in the initial system event.",
    )
    turns: int = Field(
        description="Count of distinct model messages, deduplicated across streamed "
        "content blocks."
    )
    tool_calls: dict[str, int] = Field(
        description="Invocation count for each tool, keyed by tool name."
    )
    reported_turns: int | None = Field(
        default=None,
        description="Turn count reported in the final result event, or null if absent.",
    )
    seconds: float | None = Field(
        default=None,
        description="Session wall-clock duration reported by the agent in seconds, or "
        "null if absent.",
    )
    cost: float | None = Field(
        default=None,
        description="Session cost reported by the agent in US dollars, or null if "
        "absent.",
    )
    tokens: dict[str, int] = Field(
        default_factory=dict,
        description="Token counts reported by the agent, keyed by usage category such "
        "as output_tokens.",
    )
    outcome: str | None = Field(
        default=None,
        description="Session outcome subtype reported by the agent, such as success, "
        "error_max_turns, error_max_budget_usd, or error_during_execution, or null if "
        "absent.",
    )
    terminal_reason: str | None = Field(
        default=None,
        description="Loop termination reason reported by the agent, such as completed, "
        "max_turns, or api_error, or null if unstated.",
    )
    is_error: bool | None = Field(
        default=None,
        description="Flag indicating whether the agent reported an error outcome, or "
        "null if absent.",
    )
    errors: tuple[str, ...] = Field(
        default=(),
        description="Error messages reported by the agent in the final result event.",
    )
    report: str | None = Field(
        default=None,
        description="Final session report from the result event, falling back to the "
        "last text block, or null if none.",
    )

    @property
    def calls(self) -> int:
        """Total number of tool invocations across all tools."""
        return sum(self.tool_calls.values())


def events(path: Path) -> Iterator[Event]:
    """Yields JSON event objects from the transcript at `path` in order."""
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                event = _JSON.validate_json(line)
            except ValidationError:
                continue
            if isinstance(event, dict):
                yield event


def read(path: Path) -> Session | None:
    """Parses the transcript at `path` into a `Session`, or returns null if empty.

    Transcripts contain no JSON events when the agent emits plain text instead of a
    JSON event stream, as non-Claude-Code agents may do.
    """
    found = False
    setup: Setup | None = None
    model: str | None = None
    agent: str | None = None
    messages: set[str] = set()
    calls: dict[str, str] = {}
    said: str | None = None
    report: Event = {}
    for number, event in enumerate(events(path)):
        match event:
            case {"type": "system", "subtype": "init"}:
                found = True
                setup = _setup(event)
                model = _text(event.get("model"))
                agent = _text(event.get("claude_code_version"))
            case {"type": "assistant", "message": dict(message)}:
                found = True
                messages.add(_text(message.get("id")) or f"event {number}")
                for place, block in enumerate(_blocks(message)):
                    match block.get("type"):
                        case "tool_use":
                            key = (
                                _text(block.get("id"))
                                or f"event {number} block {place}"
                            )
                            calls[key] = _text(block.get("name")) or "unnamed"
                        case "text":
                            said = _text(block.get("text")) or said
            case {"type": "result"}:
                found = True
                report = event
    if not found:
        return None
    flag = report.get("is_error")
    return Session(
        setup=setup,
        model=model,
        agent=agent,
        turns=len(messages),
        tool_calls=dict(sorted(Counter(calls.values()).items())),
        reported_turns=_integer(report.get("num_turns")),
        seconds=_seconds(report.get("duration_ms")),
        cost=_number(report.get("total_cost_usd")),
        tokens=_tokens(report.get("usage")),
        outcome=_text(report.get("subtype")),
        terminal_reason=_text(report.get("terminal_reason")),
        is_error=flag if isinstance(flag, bool) else None,
        errors=_texts(report.get("errors")),
        report=_text(report.get("result")) or said,
    )


def _setup(event: Event) -> Setup:
    """Builds a `Setup` record from the initial system `event`."""
    servers = _named(event.get("mcp_servers"))
    return Setup(
        cwd=_text(event.get("cwd")),
        tools=_texts(event.get("tools")),
        mcp_servers={
            name: _text(server.get("status")) or "unstated"
            for name, server in servers.items()
        },
        permission_mode=_text(event.get("permissionMode")),
        api_key_source=_text(event.get("apiKeySource")),
        output_style=_text(event.get("output_style")),
        skills=_texts(event.get("skills")),
        agents=_texts(event.get("agents")),
        slash_commands=_texts(event.get("slash_commands")),
        plugins=tuple(_named(event.get("plugins"))),
    )


def _named(value: JsonValue) -> dict[str, Event]:
    """Maps named dictionary items in `value` by their name field."""
    found: dict[str, Event] = {}
    for each in value if isinstance(value, list) else []:
        if isinstance(each, dict):
            name = _text(each.get("name"))
            if name is not None:
                found[name] = each
    return found


def _texts(value: JsonValue) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(each for each in value if isinstance(each, str))


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
    """Prints the parsed JSON summary of the transcript path passed as an argument."""
    session = read(Path(sys.argv[1]))
    print("null" if session is None else session.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
