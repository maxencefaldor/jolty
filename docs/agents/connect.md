# Connect an agent

The page shows how an agent reaches jolty through each of its two front ends.

The command line and the MCP server derive from the same request models, so their arguments always match, and they print the same text.

| Front end | Suits |
| --- | --- |
| [Command line](#command-line) | Any harness that runs shell commands, such as Claude Code or Codex |
| [MCP server](#mcp-server) | Any harness that speaks the Model Context Protocol |

## Command line

An agent whose harness runs shell commands needs nothing more.
The command `jolty simulate` runs a script and its plan and prints the report.
Every command takes `--output-format json` to print its result as JSON for programs.
The [command line reference](../reference/cli.md) lists every command and option.

```sh
uv run --project /path/to/jolty jolty simulate circuits/filter.py
```

## MCP server

The command `jolty-mcp` serves jolty over the [Model Context Protocol](https://modelcontextprotocol.io), on stdio.
A harness starts it from a project's `.mcp.json`, which Claude Code and other harnesses read:

```json title=".mcp.json"
{
  "mcpServers": {
    "jolty": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/jolty", "jolty-mcp"]
    }
  }
}
```

With Claude Code, one command does the same:

```sh
claude mcp add jolty -- uv run --project /path/to/jolty jolty-mcp
```

The option `--project` runs the server with the clone's environment while the harness's working directory stays the agent's, so relative script paths resolve where the agent works.

The server offers one tool per command:

| Tool | Command |
| --- | --- |
| `build_circuit` | `jolty build` |
| `simulate_circuit` | `jolty simulate` |
| `check_circuit` | `jolty check` |
| `explain_rule` | `jolty explain` |

Each tool answers with the text the command prints, and with the result itself as structured content.
Scripts run in a process of their own, with the configured time limit, so nothing a script prints can reach the protocol stream.
The build and simulate tools run any Python the script holds, with the user's permissions, so a harness should ask before each call; the check and explain tools only read.
