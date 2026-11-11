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

## unconnected-terminal

A device's terminal joins a node that no other terminal joins and that is not the ground: nothing connects to it.

## shorted-source

A voltage or current source, independent or controlled, has both ends of its output on one node.
A voltage source then holds its voltage across nothing, and a current source drives its current through nothing.

## shorted-device

A resistor, capacitor, inductor or diode has both terminals on one node, so it carries no current and does nothing.

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
