# Changelog

Every version of jolty and what it changes.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

The first release, 0.1.0.

### Added

- Circuits as Python build scripts of reusable components, compiled into a canonical, validated circuit format, with every problem named by instance path and script line.
- A native engine in Rust: modified nodal analysis with charge-oriented device models, sparse LU, Newton iteration with SPICE's convergence aids, and the operating point, DC sweep, AC, transient, periodic steady state, sensitivity, stability, noise and pole-zero analyses, and a search for every DC solution by deflation; a transient steps onto each jump of a behavioral source and restarts its integration there.
- Device models: resistors, capacitors and inductors with voltage coefficients, sources and their waveforms, controlled sources, behavioral sources, switches, the diode, the JFET, the Gummel-Poon transistor with its excess phase, level 1 and level 3 MOSFETs, a photocell, and logic devices simulated by an event scheduler beside the analog circuit.
- Vendors' SPICE macromodels read as components, and any circuit written as a SPICE netlist for ngspice.
- Ratings a part's maker states on its voltages, currents and power, on a device or a component's ports, each bounded above, below or both, held to every operating point and transient.
- Thermal networks of thermal resistances, heat capacities and heat sources, whose nets are temperatures above the ambient, solved with the circuit.
- Real parts, each a model with its maker's number and ratings: the 2N3904, 2N3906 and 1N4148, and four power MOSFETs fitted to their datasheets.
- Rules on a circuit's structure, on its values against the E series, at its operating point (a net only gmin holds, another DC solution, a mode that grows), on its loops' margins and on its ratings; schematics drawn from the connections alone, plots of every analysis, and a 3D viewer that plays any simulation on the layout.
- Specs measured on every result, at every temperature and corner, over Monte Carlo samples of tolerances, and at the extremes a worst-case analysis sets them to: on any number of a device's parameters, by path, in linear or logarithmic spreads, and shared by lots.
- Fidelity: components characterized into logic cells, run in their place, and spot-checked against their circuits.
- Exact sensitivities of every unknown to every parameter, at the operating point, along DC sweeps, over frequency and over time.
- Pole-zero analysis by the QZ algorithm on the linearized circuit's pencils, with specs on the least damping and the dominant pole, agreeing with ngspice.
- The periodic steady state of a driven circuit, found by shooting with the monodromy matrix carried along each period.
- Noise analysis by the adjoint method: every device's thermal, shot and flicker noise at an output, each source's share, the noise referred to an input, and specs on noise densities and RMS noise, all agreeing with ngspice.
- A parts catalog that prices a circuit's bill of materials: by value and tolerance, or by part number for numbered devices and components, in packages, never with an obsolete part.
- A command line, an MCP server and a Python package, every command defined once.
- Benchmarks: blind tasks for one agent, from Koza's patented circuits, AnalogCoder's problems and a cost-down, each judged by a scorer the agent never sees.
