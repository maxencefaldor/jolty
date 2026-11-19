# Connect an agent

Agents connect to jolty through either the command line or the MCP server.

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

The option `--project` runs the server in the clone's environment while keeping the agent's working directory intact so relative script paths resolve where the agent works.

The server offers one tool per command:

| Tool | Command |
| --- | --- |
| `build_circuit` | `jolty build` |
| `simulate_circuit` | `jolty simulate` |
| `merge_shards` | `jolty merge` |
| `plot_results` | `jolty plot` |
| `measure_results` | `jolty measure` |
| `draw_schematic` | `jolty draw` |
| `view_circuit` | `jolty view` |
| `check_circuit` | `jolty check` |
| `explain_rules` | `jolty explain` |
| `read_docs` | `jolty read-docs` |
| `cost_circuit` | `jolty cost` |
| `export_kicad_netlist` | `jolty export-kicad` |

Each tool answers with the text the command prints, with the result itself as structured content, and with the schematics and plots it drew as images.
The server also exposes the guide pages (`jolty://docs/guide/<slug>`) and example scripts (`jolty://docs/examples/<slug>`) offline as MCP resources, with matching URI templates (`jolty://docs/guide/{slug}` and `jolty://docs/examples/{+slug}`), backed by the same loader as `jolty read-docs` and `read_docs`.
The server's instructions derive programmatically from the code, naming every device constructor, real part, analysis, measure and offline resource URI.
Scripts run in a separate process with the configured time limit so script stdout cannot reach the protocol stream.
The build and simulate tools execute arbitrary Python in the script with the user's permissions, so a harness should ask before calling them.
The measure, check, explain, read-docs, and cost tools are read-only, while the plot, draw, view, and KiCad tools write files next to the inputs they read.
