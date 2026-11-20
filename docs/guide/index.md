# Guide

The guide explains jolty one concept per page.

- [Components](components.md): how a script declares ports, places devices and other components, and connects them into nets.
- [Units](units.md): quantities that carry their dimension, so a value in farads cannot go where ohms belong.
- [Devices](devices.md): every device a script can place, its terminals and its parameters, and the waveforms of sources.
- [Logic](logic.md): logic devices, the event scheduler that simulates them, and how logic and analog circuits join.
- [SPICE models](spice.md): vendors' models, such as an op-amp's macromodel, read from their SPICE files as components.
- [Foundry processes](foundry.md): SKY130's transistors, which ngspice simulates with the process design kit's models, at its process corners.
- [Checks](checks.md): the structural rules every build checks, their diagnostics and their levels.
- [Schematics](schematics.md): how jolty draws a circuit from its connections alone, and the facts a schematic carries.
- [3D viewer](viewer.md): a page that lays a circuit out on a board and plays its simulation on it, every recorded device's state at every point.
- [Simulation](simulation.md): plans, the analyses and the numerical options, and how the engine solves a circuit.
- [Departures from ngspice](departures.md): every place where the engine computes otherwise than ngspice 47, each with its evidence.
- [Across machines](machines.md): a Monte Carlo analysis's samples run in shards, by any number of processes on any number of machines, and merged into one run's report.
- [Results](results.md): the report, the plots, the results file, and how to read a result by net and by instance.
- [Specs](specs.md): requirements on measures of the results, and the verdict on each.
- [Fidelity](fidelity.md): logic cells characterized from components' circuits that stand in for them, and the spot checks that hold them to their contracts.
- [Parts and costs](parts.md): real parts with ratings and manufacturer numbers, orderable catalogs, E-series preferred values, bills of materials, and KiCad netlist export.

A circuit and its simulation stay separate.
The circuit defines what exists: components, devices, values, and connections.
The plan specifies how to simulate the circuit, including the analyses, time or frequency spans, and numerical tolerances.

For step-by-step recipes covering corners, vendor SPICE models, convergence diagnostics, and gradient sizing, see the [How-to guides](../how-to/index.md).
