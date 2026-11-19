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
| [`non-standard-value`](#non-standard-value) | `warn` |
| [`gmin-dependence`](#gmin-dependence) | `warn` |
| [`multiple-operating-points`](#multiple-operating-points) | `warn` |
| [`unstable-pole`](#unstable-pole) | `warn` |
| [`low-phase-margin`](#low-phase-margin) | `warn` |
| [`low-gain-margin`](#low-gain-margin) | `warn` |
| [`over-rating`](#over-rating) | `error` |
| [`near-rating`](#near-rating) | `warn` |

## unconnected-terminal

A device terminal or logic pin joins a non-ground net that no other terminal or pin connects to.

## shorted-source

A voltage or current source (independent or controlled) has both output terminals on the same node, or a current-controlled source has both sensing-port terminals on the same node.
Shorting a voltage source creates an inconsistent voltage constraint, shorting a current source bypasses its output, and shorting a sensing port eliminates the sensed current.

## shorted-device

A resistor, capacitor, inductor, diode, or transistor has every terminal on the same node, carrying no current.
A transistor with only some terminals joined, such as a diode-connected transistor, does not trigger this rule.

## voltage-loop

Voltage sources and inductors (independent or controlled) form a closed loop.
At DC an inductor acts as a short circuit, so the loop's sources either conflict or leave the loop current undefined.
Each independent loop raises one diagnostic.

## no-dc-path

A group of nodes has no path to ground through resistors, diodes, inductors, or voltage sources, and no current source drives the group.
The nodes couple to ground only through capacitors or control inputs, leaving their DC voltage undefined.
Groups whose nodes each join only a single terminal are reported under unconnected-terminal.

## current-cutset

Current sources (independent or controlled) drive a group of nodes that has no DC path to ground, as defined by no-dc-path, leaving their DC current with no return path.
Groups whose nodes each join only a single terminal are reported under unconnected-terminal.

## driver-conflict

Two or more logic outputs drive the same logic net.
Whenever those outputs drive different logic levels, the net level is unknown.

## floating-input

Logic inputs read a logic net that no output drives, leaving the net level permanently unknown.
A net joined by only a single input is reported under unconnected-terminal.

## non-standard-value

A resistor value falls outside the IEC 60063 E3 through E192 series, or a capacitor or inductor value falls outside E3 through E24 (beyond rounding precision).
No standard preferred part matches the value.

## gmin-dependence

Doubling gmin (the shunt conductance placed across every junction to aid Newton convergence) moves a net's operating-point voltage to first order by more than 1% of max(|V|, 1 V).
Little other than gmin sets that net's potential, as between reverse-biased junctions, so the voltage reflects simulator regularization rather than circuit behavior.
One diagnostic lists all affected nets in decreasing order of shift during operating-point checks.

## multiple-operating-points

The circuit's DC equations admit an additional solution distinct from the operating point, as found by operating-points analysis.
Circuits such as latches or self-biased stages can settle into more than one state depending on startup.
The diagnostic names every net whose voltage differs by at least half the maximum difference, checked during operating-point analysis on circuits with at most 500 unknowns.

## unstable-pole

The circuit linearized at its operating point has a pole whose real part exceeds 1e-6 times its magnitude.
A natural mode therefore grows away from the operating point, such as an emerging oscillation or a tipping latch.
Each real pole or conjugate pair raises one diagnostic naming nets with at least half the peak mode amplitude, checked on circuits of at most 500 unknowns whose linearization fits within max_memory.

## low-phase-margin

A stability analysis's loop gain falls through 0 dB with a phase margin below 45°: the loop's step response overshoots by more than about a fifth and rings, and the margin left for parts and loads that shift the phase is small.
The diagnostic names the probe, the margin and the crossover frequency.

## low-gain-margin

A stability analysis's loop gain is above -6 dB where its phase, unwrapped from the lowest frequency, first falls through -180°: less than doubling the loop's gain makes it oscillate.
The diagnostic names the probe and the margin.

## over-rating

A stress a device's rating limits, the voltage of one of its terminals over another, the current into one, the power it dissipates or its temperature, lies beyond the rating's bounds at an operating point or at any time of a transient the plan solves: the part's maker does not warrant it there, and it may fail.
Each rating raises one diagnostic per analysis and bound, at the furthest point.

## near-rating

A stress a device's rating limits lies within the rating but beyond 80 % of a bound away from zero, a temperature's from 0 °C, at an operating point or at any time of a transient the plan solves: the part runs without the margin derating keeps for heat, ageing and its spread.
Each rating raises one diagnostic per analysis and bound, at the furthest point.
