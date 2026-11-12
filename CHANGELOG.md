# Changelog

Every version of jolty and what it changes.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## Unreleased

The first release, 0.1.0.

### Added

- Circuits as Python build scripts of reusable components, compiled into a canonical, validated circuit format, with every problem named by instance path and script line.
- A native engine in Rust: modified nodal analysis with charge-oriented device models, sparse LU, Newton iteration with SPICE's convergence aids, and the operating point, DC sweep, AC, transient, sensitivity, stability and noise analyses.
- Device models: resistors, capacitors and inductors with voltage coefficients, sources and their waveforms, controlled sources, behavioral sources, switches, the diode, the JFET, the Gummel-Poon transistor, level 1 and level 3 MOSFETs, a photocell, and logic devices simulated by an event scheduler beside the analog circuit.
- Vendors' SPICE macromodels read as components, and any circuit written as a SPICE netlist for ngspice.
- Structural rules, schematics drawn from the connections alone, plots of every analysis, and a 3D viewer that plays any simulation on the layout.
- Specs measured on every result, at every temperature and corner, and over Monte Carlo samples of tolerances: on any number of a device's parameters, by path, in linear or logarithmic spreads, and shared by lots.
- Fidelity: components characterized into logic cells, run in their place, and spot-checked against their circuits.
- Exact sensitivities of every unknown to every parameter, at the operating point, along DC sweeps, over frequency and over time.
- Noise analysis by the adjoint method: every device's thermal, shot and flicker noise at an output, each source's share, the noise referred to an input, and specs on noise densities and RMS noise, all agreeing with ngspice.
- A parts catalog that prices a circuit's bill of materials.
- A command line, an MCP server and a Python package, every command defined once.
- Benchmarks: blind tasks for one agent, from Koza's patented circuits, AnalogCoder's problems and a cost-down, each judged by a scorer the agent never sees.
