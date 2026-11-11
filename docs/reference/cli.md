# Command line

The `jolty` command builds and simulates circuits from the shell.

Every command takes `--output-format json` to print its result as JSON for programs; the default, `text`, is for people and agents.
The command `jolty --version` prints the version.
The exit status is 0 for success, 1 when a circuit does not elaborate or an analysis fails, and 2 for invalid arguments or settings, or a failed command, as when a script raises.
Each command's options come from a request model in `jolty.service`, which the [MCP server](../agents/connect.md#mcp-server) shares, so a tool takes the same arguments as its command.

## `jolty build`

Build a script's circuit and elaborate it.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `SCRIPT` | required | Python file defining build(), which returns the top Component of the circuit. |
| `--output` |  | Where to save the circuit; defaults to &lt;script&gt;.jolty.json. |

## `jolty simulate`

Build a script's circuit and run a plan's analyses on it.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `SCRIPT` | required | Python file defining build(), which returns the top Component of the circuit. |
| `--plan` |  | Python file defining plan(), which returns a Plan; defaults to the script. |
| `--output` |  | Where to save the circuit; defaults to &lt;script&gt;.jolty.json. |
| `--results` |  | Where to save the results; defaults to &lt;script&gt;.results.npz. |
