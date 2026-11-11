# Guide

The guide explains jolty one concept per page.

- [Components](components.md): how a script declares ports, places devices and other components, and connects them into nets.
- [Units](units.md): quantities that carry their dimension, so a value in farads cannot go where ohms belong.
- [Devices](devices.md): every device a script can place, its terminals and its parameters, and the waveforms of sources.
- [Checks](checks.md): the structural rules every build checks, their diagnostics and their levels.
- [Simulation](simulation.md): plans, the four analyses and the numerical options, and how the engine solves a circuit.
- [Results](results.md): the report, the results file, and how to read a result by net and by instance.

A circuit and its simulation stay apart.
The circuit says what exists: components, devices, values and connections.
The plan says how to simulate it: which analyses, over which times or frequencies, and with which tolerances.
