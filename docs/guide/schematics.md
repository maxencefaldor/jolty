# Schematics

`jolty build` and `jolty simulate` draw the top component's schematic next to the circuit as `<script>.schematic.png`, and the MCP server sends it alongside text output.
Scripts contain no coordinates: jolty lays out the schematic purely from connectivity, following standard engineering conventions.

![The schematic of the RC filter's bench: a source feeding the low_pass component, drawn as a box with its arguments inside](../images/rc_filter.schematic.png)

## How it reads

Signal flows from left to right.
Each instance takes a column by its distance from the sources, or, in a component without sources, from its first signal port.

- A two-terminal device between two signal nets lies across.
- A device between a signal net and a rail hangs from the signal: down to a ground, up to a supply.
  Devices that hang from the same net stand side by side along it.
- A controlled source keeps its control leads on its left, and flips upside down rather than stand on its ground.
- A placed component is a box, its function and arguments inside, its signal ports on its sides, its supplies on top and its grounds below.
- Each instance carries its name, and each device its value.

Ground and supply nets draw no wires: a ground symbol, or a bar with the supply's name, marks each of their pins.
A net is a ground or a supply when a port on it is one, of the component drawn or of a component it places.

Every other net is a wire, routed around the symbols.
Wires of different nets cross at right angles without joining, so each wire touches only its net; dots mark three-way junctions on the same net.
A net the router cannot fit displays its name at each connected pin instead.

Signal ports of the displayed component appear along the edges as directional flags: inputs on the left, outputs on the right.

## Facts on a schematic

A schematic annotates findings from build and simulation:

- `jolty simulate` labels each net with its voltage from the plan's first operating point;
- any instance referenced by a listed diagnostic renders in red with a `!` marker, propagating the marker to enclosing component boxes.

Passing `--no-schematic` omits schematic rendering.

## Inside a component

`jolty draw` draws a saved circuit's top component, or, with `--instance`, the component an instance places, as `<script>.<path>.schematic.png`:

```sh
uv run jolty draw examples/rectifier.jolty.json --instance X1
```

![The inside of the rectifier: a diode from the input into a smoothing capacitor and a load, both to ground](../images/rectifier.X1.schematic.png)

The component's ports become flags, and the option `--results` writes the voltages of a results file's first operating point; a port net takes the voltage of the net it joins outside.
