# 3D viewer

`jolty view` writes a 3D viewer of a saved circuit and its results: one HTML page that lays the circuit out on a board and plays its simulation on it.

```sh
uv run jolty simulate examples/common_emitter.py
uv run jolty view examples/common_emitter.jolty.json --results examples/common_emitter.results.npz
```

The page, `<script>.view.html` beside the circuit, opens in any browser and loads three.js from jsDelivr.
Every example of the [gallery](../gallery.md) opens in one.

## The board

The board lays out one component at a time, as its [schematic](schematics.md) does: each device a part standing on its pins, each wire a copper trace, each ground or supply pin a post, and each port a flag at the edge.
Resistors carry their value's color bands, inductors their coil, diodes the band at their cathode, and transistors their package.
A component instance is a chip with its name on it; double-clicking it opens its own board, and the path above the board leads back up.

## The simulation

Each analysis of the results plays along its axis: a transient through time, a DC sweep through its source's levels, a frequency response through its frequencies, and an operating point at its one point.
The timeline plays, pauses and scrubs through every point, at a quarter to four times its pace.

At each point:

- each trace takes its net's voltage as a color, on the scale the legend shows, and a logic net its level;
- light runs along each lead, into the part where current flows in, faster as the current grows;
- each part glows with the power it takes, the sum over its terminals of each voltage times the current in.

A plot below the board draws the voltages of the top component's named nets, a dashed line at the current point.

## The inspector

Clicking a part opens the inspector, which shows at the current point:

- each terminal's net, voltage, current in and charge, and the power the part takes;
- the part's [variables](results.md#devices-own-models), such as a transistor's threshold voltage;
- its small-signal conductances between its terminals, each the change of a terminal's current with another's voltage;
- its terminals' currents over the whole analysis, and its parameters.

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
