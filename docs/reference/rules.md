# Rules

Each rule names one fact the structural checks can state about a circuit, and `jolty explain` prints the same condition as this page.
A check reports each rule at its level: `error`, `warn` or `ignore`.
The default levels below hold until settings or options set others (see [Checks](../guide/checks.md#levels)).

| Rule | Default level |
| --- | --- |
| [`unconnected-terminal`](#unconnected-terminal) | `error` |
| [`shorted-source`](#shorted-source) | `error` |
| [`shorted-device`](#shorted-device) | `warn` |
| [`voltage-loop`](#voltage-loop) | `error` |
| [`no-dc-path`](#no-dc-path) | `error` |
| [`current-cutset`](#current-cutset) | `error` |
| [`driver-conflict`](#driver-conflict) | `error` |
| [`floating-input`](#floating-input) | `error` |
| [`gmin-dependence`](#gmin-dependence) | `warn` |
| [`multiple-operating-points`](#multiple-operating-points) | `warn` |
| [`unstable-pole`](#unstable-pole) | `warn` |

## unconnected-terminal

A device's terminal or logic pin joins a net that no other terminal or pin joins and that is not the ground: nothing connects to it.

## shorted-source

A voltage or current source, independent or controlled, has both ends of its output on one node, or a current-controlled source has both ends of the port that senses its control current on one node.
A voltage source then holds its voltage across nothing, a current source drives its current through nothing, and a sensing port senses no current.

## shorted-device

A resistor, capacitor, inductor, diode or transistor has every terminal on one node, so it carries no current and does nothing.
A transistor with only some terminals joined, such as a diode-connected one, raises nothing.

## voltage-loop

Voltage sources and inductors, independent or controlled, form a loop.
At DC an inductor is a short, so the loop's sources alone fix the voltages around it: they conflict, or leave the current around the loop undefined.
Each independent loop raises one diagnostic.

## no-dc-path

A group of nodes has no path to the ground through resistors, diodes, inductors or voltage sources, and no current source drives it: it reaches the ground only through capacitors, control inputs or nothing, so its DC voltage is undefined.
A group whose every node joins a single terminal is left to unconnected-terminal.

## current-cutset

Current sources, independent or controlled, drive a group of nodes that has no DC path to the ground, as no-dc-path defines it: at DC their current has nowhere to go.
A group whose every node joins a single terminal is left to unconnected-terminal.

## driver-conflict

Two or more logic outputs drive one logic net.
Wherever they drive different levels, the net's level is unknown.

## floating-input

Logic inputs read a logic net that no output drives, so its level is always unknown.
A net that a single input joins is left to unconnected-terminal.

## gmin-dependence

Doubling gmin, the conductance the engine places across every junction to help Newton iteration converge, would move a net's voltage at the operating point, to first order, by more than a hundredth of the larger of that voltage and 1 V: little but gmin holds the net, as between reverse-biased junctions, so its voltage comes from the simulator rather than the circuit.
One diagnostic names every such net, by decreasing move.
A simulation checks it at its plan's operating point.

## multiple-operating-points

The circuit's DC equations have a solution besides its operating point, as an operating-points analysis finds them: the circuit can rest in more than one state, as a latch does, or a bias circuit that can also rest with no current, and which it rests in depends on how it starts.
The diagnostic names the nets whose voltages differ between the two by at least half the largest difference.
A simulation checks it at its plan's operating point, on circuits of at most 500 unknowns.

## unstable-pole

The circuit linearized at its operating point has a pole whose real part is above a millionth of its magnitude: a natural mode grows away from the operating point, as an oscillation that builds up or a latch that tips, so the circuit does not rest there.
A real pole, or a conjugate pair, raises one diagnostic, naming the nets where its mode's amplitude is at least half its largest.
A simulation checks it at its plan's operating point, on circuits of at most 500 unknowns.
