# Command line

The `jolty` command builds and simulates circuits from the shell.

Every command takes `--output-format json` to print its result as JSON for programs; the default, `text`, is for people and agents.
The command `jolty --version` prints the version.
The exit status is 0 for success, 1 when a circuit does not elaborate, a check finds an error or an analysis fails, and 2 for invalid arguments or settings, or a failed command, as when a script raises.
Each command's options come from a request model in `jolty.service`, which the [MCP server](../agents/connect.md#mcp-server) shares, so a tool takes the same arguments as its command.

## `jolty build`

Build a script's circuit and elaborate it.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `SCRIPT` | required | Python file defining build(), which returns the top Component of the circuit. |
| `--output` |  | Where to save the circuit; defaults to &lt;script&gt;.jolty.json. |
| `--check`, `--no-check` | `True` | Check the circuit's structure after building it. |
| `--error` |  | Rules to report as errors, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--warn` |  | Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; overrides the configured levels. |
| `--ignore` |  | Rules to leave out, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--max-diagnostics` |  | Diagnostics to list at most; defaults to the configured limit. |
| `--schematic`, `--no-schematic` | `True` | Draw the top component's schematic beside the circuit, marking the instances the listed diagnostics name. |

## `jolty simulate`

Build a script's circuit and run a plan's analyses on it.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `SCRIPT` | required | Python file defining build(), which returns the top Component of the circuit. |
| `--plan` |  | Python file defining plan(), which returns a Plan; defaults to the script. |
| `--output` |  | Where to save the circuit; defaults to &lt;script&gt;.jolty.json. |
| `--results` |  | Where to save the results; defaults to &lt;script&gt;.results.npz. |
| `--check`, `--no-check` | `True` | Check the circuit's structure after building it. |
| `--error` |  | Rules to report as errors, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--warn` |  | Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; overrides the configured levels. |
| `--ignore` |  | Rules to leave out, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--max-diagnostics` |  | Diagnostics to list at most; defaults to the configured limit. |
| `--schematic`, `--no-schematic` | `True` | Draw the top component's schematic beside the circuit, marking the instances the listed diagnostics name and writing the voltages of the plan's first operating point. |
| `--plot`, `--no-plot` | `True` | Plot every analysis but operating points beside the results. |
| `--probes` |  | Unknowns to plot, such as 'out', 'X1/mid' or 'V1:current'; defaults to the nets the top component names, up to eight. |

## `jolty plot`

Plot the analyses of a saved results file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `RESULTS` | required | A .results.npz file. |
| `--analyses` |  | Analyses to plot, by name; every one but operating points by default. |
| `--probes` |  | Unknowns to plot, such as 'out', 'X1/mid' or 'V1:current'; defaults to the nets the top component names, up to eight. |

## `jolty draw`

Draw a component of a saved circuit file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--instance` |  | The path of the component instance to draw, such as 'X1' or 'X1/X2'; the top component by default. |
| `--results` |  | A .results.npz file of the circuit, whose first operating point gives the voltage written on each net. |
| `--output` |  | Where to save the image; defaults to &lt;stem&gt;.schematic.png beside the circuit, or &lt;stem&gt;.X1.X2.schematic.png for the instance X1/X2. |

## `jolty check`

Check the structure of a saved circuit file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--error` |  | Rules to report as errors, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--warn` |  | Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; overrides the configured levels. |
| `--ignore` |  | Rules to leave out, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--max-diagnostics` |  | Diagnostics to list at most; defaults to the configured limit. |

## `jolty explain`

Explain rules and their levels.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `RULES` |  | Rules to explain; every rule by default. |
