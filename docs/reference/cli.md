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
| `--plot`, `--no-plot` | `True` | Plot every analysis but operating points and sensitivities beside the results. |
| `--probes` |  | Unknowns to plot, such as 'out', 'X1/mid' or 'V1:current'; defaults to the nets the top component names, up to eight. |
| `--record` |  | Instances whose currents, charges, derivatives and variables the results keep beside the unknowns, by path or pattern, such as 'M1', 'X1/**' or '!X1/bias/**', a pattern after ! taking back what earlier ones named; replaces the plan's record. |
| `--min-solves-per-task` |  | The fewest frequencies or right-hand sides a task takes when a run spreads its small-signal analyses and sensitivities across threads; replaces the plan's. No result depends on it. |
| `--min-instances-per-task` |  | The device instances a task evaluates when a run spreads its devices' evaluation across threads; replaces the plan's. No result depends on it. |
| `--min-unknowns-inside` |  | The fewest unknowns a circuit has for a run to spread the work of its Newton iteration and its step control on each unknown across threads; replaces the plan's. No result depends on it. |
| `--max-unknowns-per-interior` |  | The most unknowns of an interior when a run factors its matrix in parallel, each large block ordered by nested dissection into interiors and the separators between them; replaces the plan's. It orders the factorization's operations, so results depend on it in their last digits, but not on the threads. |
| `--samples` |  | Run only the samples of the plan's Monte Carlo analysis from START up to STOP, written START:STOP, such as 0:1000000, as a shard of the analysis: at the plan's own conditions and at every corner, running none of its other analyses, into the directory of shards, for jolty merge to report with the others. A shard draws no schematic and no plot, and traces only the probes named. |
| `--shards` |  | The directory a shard of samples is written to, as shard-START-STOP.npz; defaults to &lt;script&gt;.shards. |

## `jolty merge`

Merge the shards of a plan's Monte Carlo analysis into one run.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `SHARDS` | required | The directory of the shards, such as amp.shards. |
| `--output` |  | Where to save the merged batch; defaults to &lt;name&gt;.batches.npz beside the directory, its name without .shards. |
| `--plot`, `--no-plot` | `True` | Plot a histogram of each requirement's values, and the envelope of each analysis the shards traced, beside the merged batch. |

## `jolty plot`

Plot the analyses of a saved results file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `RESULTS` | required | A .results.npz file. |
| `--analyses` |  | Analyses to plot, by name; every one but operating points and sensitivities by default. |
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

## `jolty view`

Write a 3D viewer of a saved circuit and its results.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--results` |  | A .results.npz file of the circuit, whose analyses the viewer plays; none shows the circuit alone. |
| `--output` |  | Where to save the page; defaults to &lt;stem&gt;.view.html beside the circuit. |
| `--title` |  | The page's title; the top component's name if empty. |

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

## `jolty cost`

Price a component of a saved circuit file with a catalog.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--catalog` | required | A catalog file of parts and their prices. |
| `--instance` |  | The path of the component instance to price, such as 'X1'; the top component by default. |

## `jolty kicad`

Write a component of a saved circuit file as a KiCad netlist.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | 'text' for people and agents, 'json' for programs. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--catalog` | required | A catalog file of parts, with their footprints and pins. |
| `--instance` |  | The path of the component instance to write, such as 'X1'; the top component by default. |
| `--output` |  | Where to save the netlist; defaults to &lt;stem&gt;.net beside the circuit, or &lt;stem&gt;.X1.X2.net for the instance X1/X2. |
