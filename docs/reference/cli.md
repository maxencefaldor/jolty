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
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `SCRIPT` | required | Python file defining build(), which returns the top Component of the circuit. |
| `--output` |  | Destination path for the saved circuit; defaults to &lt;script&gt;.jolty.json. |
| `--check`, `--no-check` | `True` | Check the circuit's structure after building it. |
| `--error` |  | Rules to report as errors, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--warn` |  | Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; overrides the configured levels. |
| `--ignore` |  | Rules to omit, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--max-diagnostics` |  | Maximum diagnostics to list; defaults to the configured limit. |
| `--schematic`, `--no-schematic` | `True` | Draw the top component's schematic next to the circuit, marking instances named in listed diagnostics. |

## `jolty simulate`

Build a script's circuit and run a plan's analyses on it.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `SCRIPT` | required | Python file defining build(), which returns the top Component of the circuit. |
| `--plan` |  | Python file defining plan(), which returns a Plan; defaults to the script. |
| `--output` |  | Destination path for the saved circuit; defaults to &lt;script&gt;.jolty.json. |
| `--results` |  | Destination path for the saved results; defaults to &lt;script&gt;.results.npz. |
| `--raw` |  | Destination path for a SPICE raw file of completed operating point, DC sweep, small-signal response, and transient analyses; omitted when unset. |
| `--raw-format` | `binary` | Encoding of the SPICE raw file named by --raw: 'binary' for little-endian doubles, or 'ascii'. |
| `--check`, `--no-check` | `True` | Check the circuit's structure after building it. |
| `--error` |  | Rules to report as errors, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--warn` |  | Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; overrides the configured levels. |
| `--ignore` |  | Rules to omit, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--max-diagnostics` |  | Maximum diagnostics to list; defaults to the configured limit. |
| `--schematic`, `--no-schematic` | `True` | Draw the top component's schematic next to the circuit, marking instances named in listed diagnostics and annotating voltages from the plan's first operating point. |
| `--plot`, `--no-plot` | `True` | Plot all non-DC analyses (excluding operating points and sensitivities) next to the results. |
| `--traces` |  | Unknowns to plot, such as 'out', 'X1/mid', or 'V1:current'; defaults to up to eight named nets on the top component. |
| `--record` |  | Instance paths or glob patterns (such as 'M1', 'X1/**', or '!X1/bias/**') whose currents, charges, derivatives, and variables are recorded in addition to unknowns. Patterns prefixed with ! exclude earlier matches; replaces the plan record list. |
| `--min-solves-per-task` |  | Minimum frequencies or right-hand sides per task when parallelizing small-signal analyses and sensitivities across threads; replaces the plan setting. Numerical results are unaffected. |
| `--min-instances-per-task` |  | Minimum nonlinear device instances per task when parallelizing device evaluation across threads (linear devices are evaluated once and excluded); replaces the plan setting. Numerical results are unaffected. |
| `--min-unknowns-per-task` |  | Minimum unknowns or stored matrix entries per task when parallelizing Newton iteration and step control across threads; replaces the plan setting. Numerical results are unaffected. |
| `--max-unknowns-per-interior` |  | Maximum unknowns per interior block during nested-dissection parallel matrix factorization; replaces the plan setting. Changing block boundaries alters operation order in the lowest floating-point bits, independently of thread count. |
| `--samples` |  | Monte Carlo sample slice formatted as START:STOP (such as 0:1000000) to run as a shard across nominal conditions and corners without other analyses. Shard runs write to the shards directory for jolty merge, skip schematics and plots, and trace only requested unknowns. |
| `--shards` |  | Directory where sample shards are written as shard-START-STOP.npz; defaults to &lt;script&gt;.shards. |

## `jolty merge`

Merge the shards of a plan's Monte Carlo analysis into one run.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `SHARDS` | required | Directory containing shard files, such as amp.shards. |
| `--output` |  | Destination path for the merged batch; defaults to &lt;name&gt;.batches.npz next to the directory with the .shards suffix removed. |
| `--plot`, `--no-plot` | `True` | Plot histograms for each requirement and trace envelopes next to the merged batch. |

## `jolty plot`

Plot the analyses of a saved results file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `RESULTS` | required | A .results.npz file. |
| `--analyses` |  | Analysis names to plot; defaults to all non-DC analyses (excluding operating points and sensitivities). |
| `--traces` |  | Unknowns to plot, such as 'out', 'X1/mid', or 'V1:current'; defaults to up to eight named nets on the top component. |

## `jolty measure`

Read a measure from a saved results file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `RESULTS` | required | A .results.npz file. |
| `--measure` | required | Measure to read from the analysis result. |
| `--analysis` |  | Name of the analysis in the results file; optional when the file contains a single analysis. |
| `--signal` |  | Unknown to read, such as 'out', 'X1/mid' or 'V1:current'; omitted for a stability loop gain, period, noise, solutions or pole-zero measure. |
| `--reference` |  | Reference net for a differential voltage measure. |
| `--over` |  | Divisor unknown for a small-signal transfer measure (gain, phase, bandwidth, crossover, phase_margin, gain_margin). |
| `--at` |  | Point along the analysis axis in SI base units: time in seconds for a transient or periodic steady state, swept value for a DC sweep, or frequency in hertz for gain, phase and noise_density. |
| `--start` |  | Window start in SI base units: time in seconds, swept value, or frequency in hertz for integrated_noise; defaults to the start of the axis. |
| `--stop` |  | Window stop in SI base units: time in seconds, swept value, or frequency in hertz for integrated_noise; defaults to the end of the axis. |
| `--drop` |  | Gain drop in decibels for bandwidth; defaults to 3.0. |
| `--lower` |  | Lower step fraction (between 0 and 1) for rise_time; defaults to 0.1. |
| `--upper` |  | Upper step fraction (between 0 and 1) for rise_time; defaults to 0.9. |
| `--band` |  | Settling band fraction (positive) for settling_time; defaults to 0.02. |
| `--threshold` |  | Threshold in SI base units of the signal (volts, amperes, kelvins or plain number) for crossing (required) or frequency (optional midpoint default). |
| `--edge` |  | Crossing direction for crossing: 'rising', 'falling' or 'either'; defaults to 'either'. |
| `--fundamental` |  | Fundamental frequency in hertz for distortion (required). |
| `--harmonics` |  | Highest harmonic counted for distortion; defaults to 9. |
| `--referred`, `--no-referred` | `False` | Input-referred flag for noise_density or integrated_noise when true. |

## `jolty draw`

Draw a component of a saved circuit file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--instance` |  | Path of the component instance to draw, such as 'X1' or 'X1/X2'; defaults to the top component. |
| `--results` |  | Saved .results.npz file whose first operating point annotates net voltages on the schematic. |
| `--output` |  | Destination path for the PNG image; defaults to &lt;stem&gt;.schematic.png next to the circuit, or &lt;stem&gt;.X1.X2.schematic.png for instance X1/X2. |

## `jolty view`

Write a 3D viewer of a saved circuit and its results.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--results` |  | Saved .results.npz file whose analyses the viewer animates; omitting it displays the circuit geometry alone. |
| `--output` |  | Destination path for the HTML page; defaults to &lt;stem&gt;.view.html next to the circuit. |
| `--title` |  | HTML page title; defaults to the top component name when empty. |

## `jolty check`

Check the structure of a saved circuit file.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--error` |  | Rules to report as errors, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--warn` |  | Rules to report as warnings, such as 'no-dc-path', or 'all' for every rule; overrides the configured levels. |
| `--ignore` |  | Rules to omit, such as 'shorted-device', or 'all' for every rule; overrides the configured levels. |
| `--max-diagnostics` |  | Maximum diagnostics to list; defaults to the configured limit. |

## `jolty explain`

Explain rules and their levels.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `RULES` |  | Rules to explain; every rule by default. |

## `jolty read-docs`

List or read the offline guide and examples.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `TOPIC` |  | Topic to read: a bare slug such as 'units' or 'rc_filter', a qualified name such as 'guide/units' or 'examples/cluster/amplifier', or a resource URI such as 'jolty://docs/guide/units'. Omitting the topic lists all available documents. |
| `--section` |  | Limit listing or lookup to 'guide' or 'examples'. |

## `jolty cost`

Price a component of a saved circuit file with a catalog.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--catalog` | required | Catalog file of parts and their prices. |
| `--instance` |  | Path of the component instance to price, such as 'X1'; defaults to the top component. |

## `jolty export-kicad`

Write a component of a saved circuit file as a KiCad netlist.

| Option | Default | Description |
| --- | --- | --- |
| `--output-format` | `text` | Output format: 'text' for human and agent readers, or 'json' for structured tooling. |
| `CIRCUIT` | required | A .jolty.json circuit file. |
| `--catalog` | required | Catalog file of parts with their footprints and pins. |
| `--instance` |  | Path of the component instance to export, such as 'X1'; defaults to the top component. |
| `--output` |  | Destination path for the netlist; defaults to &lt;stem&gt;.net next to the circuit, or &lt;stem&gt;.X1.X2.net for instance X1/X2. |
