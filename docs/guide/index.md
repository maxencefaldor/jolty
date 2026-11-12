# Guide

The guide explains jolty one concept per page.

- [Components](components.md): how a script declares ports, places devices and other components, and connects them into nets.
- [Units](units.md): quantities that carry their dimension, so a value in farads cannot go where ohms belong.
- [Devices](devices.md): every device a script can place, its terminals and its parameters, and the waveforms of sources.
- [Logic](logic.md): logic devices, the event scheduler that simulates them, and how logic and analog circuits join.
- [SPICE models](spice.md): vendors' models, such as an op-amp's macromodel, read from their SPICE files as components.
- [Checks](checks.md): the structural rules every build checks, their diagnostics and their levels.
- [Schematics](schematics.md): how jolty draws a circuit from its connections alone, and the facts a schematic carries.
- [3D viewer](viewer.md): a page that lays a circuit out on a board and plays its simulation on it, every device's state at every point.
- [Simulation](simulation.md): plans, the five analyses and the numerical options, and how the engine solves a circuit.
- [Results](results.md): the report, the plots, the results file, and how to read a result by net and by instance.
- [Specs](specs.md): requirements on measurements of the results, and the verdict on each.
- [Fidelity](fidelity.md): logic cells characterized from components' circuits that stand in for them, and the spot checks that hold them to their contracts.
- [Parts and costs](parts.md): catalogs of orderable parts, the E series of preferred values, and the bill of materials of a component.

A circuit and its simulation stay apart.
The circuit says what exists: components, devices, values and connections.
The plan says how to simulate it: which analyses, over which times or frequencies, and with which tolerances.
