# 3D viewer

`jolty view` writes a 3D viewer of a saved circuit and its results: one HTML page that lays the circuit out on a board and plays its simulation on it.

```sh
uv run jolty simulate examples/common_emitter.py
uv run jolty view examples/common_emitter.jolty.json --results examples/common_emitter.results.npz
```

The generated page (`<script>.view.html` next to the circuit) opens in any browser and loads three.js from jsDelivr.
Every example in the [gallery](../gallery.md) includes a 3D viewer page.

The viewer animates every [recorded](results.md#what-a-result-records) quantity: all net voltages plus currents, charges, internal variables, and conductances for recorded instances (rated instances by default).
Recording every instance via `record=["**"]` illuminates every lead, matching the gallery pages.

## The board

The board lays out one component at a time like its [schematic](schematics.md): devices stand on pins, wires render as copper traces, ground and supply pins appear as posts, and ports appear as edge flags.
Resistors display value color bands, inductors show coils, diodes display cathode bands, and transistors show packages.
Each subcircuit renders as a labeled chip; double-clicking opens that subcircuit's board, and the breadcrumb path above returns up the hierarchy.

## The simulation

Each analysis of the results plays along its axis: a transient through time, a DC sweep through its source's levels, a frequency response through its frequencies, and an operating point at its one point.
The timeline plays, pauses and scrubs through every point, at a quarter to four times its pace.

At each point:

- each trace takes its net's voltage as a color, on the scale the legend shows, and a logic net its level;
- light runs along each lead, into the part where current flows in, faster as the current grows;
- each part glows with the power it takes, the sum over its terminals of each voltage times the current in.

A plot below the board draws the voltages of the top component's named nets, a dashed line at the current point.

## The inspector

Clicking a part opens the inspector, displaying the following details at the current point:

- net name, voltage, incoming current, charge, and total dissipated power for every terminal;
- internal [device variables](results.md#device-models), such as transistor threshold voltage;
- small-signal conductances between terminals, showing how each terminal current varies with another terminal's voltage;
- terminal currents across the full analysis together with device parameters.

Clicking a trace shows its net's voltage now and over the analysis.

## Addresses and keys

The page's address chooses what it opens on, as in `common_emitter.view.html#view=X1&analysis=tone&point=60&paused`: the component instance, the analysis, the point, and whether to play.

| Key | Action |
| --- | --- |
| Space | play or pause |
| ← and → | step a point |
| Home and End | the first or last point |
| F | fit the view |
| T | view from the top |
| L | show or hide the labels |
| U or Esc | close the inspector, or go up a level |
| ? | help |

The page holds a circuit's every point for an analysis of up to 240 points, and 240 points evenly spread for a longer one.
