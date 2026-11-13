# jolty

A compiler and simulator for electronic circuits, built for AI agents.

An AI agent writes a circuit as a Python script of reusable components, and jolty builds it the way a compiler builds code.
Each build checks every device and connection, reports each problem with the script line behind it, and simulates the circuit with a native engine written in Rust.

jolty is on its seventh milestone, gradients and discovery: board-level circuits of passive devices, sources, diodes and transistors, simulated at the operating point, over DC sweeps, in frequency and in time, beside logic that an event scheduler simulates.
Vendors' SPICE macromodels, such as an op-amp's, read as components and agree with ngspice.
Every build checks the circuit's structure and draws its schematic, and every simulation plots its analyses, so an agent sees what it built.
Specs state what a circuit must do, and every simulation measures its results against them, at every temperature and corner its plan names, over Monte Carlo samples of its parts' tolerances, and at their worst-case extremes.
A plan can run a component as a logic cell characterized from its own circuit, as the full adder example runs its nine transistor-level NAND gates, and spot checks hold each cell to its circuit.
A sensitivity analysis gives the derivative of any unknown with respect to every parameter of every device, at the operating point, along a DC sweep, over frequency or over time.
A stability analysis gives a feedback loop's gain through a probe, by Middlebrook's double injection, and specs read its phase and gain margins.
A noise analysis gives the noise density at any output from every device's thermal, shot and flicker noise, each source's share, and the noise referred to an input, and a pole-zero analysis the circuit's natural frequencies and a transfer function's zeros.
A periodic transient finds the steady state a driven circuit settles into, such as a rectifier's or a power converter's, in a few periods by shooting, where a transient would run for many.
Every result holds the circuit's unknowns at every point, and, for each device a plan records, its terminals' currents and charges, the derivatives of its equations and its model's variables, and a 3D viewer plays any simulation on the circuit's layout.
A circuit of SKY130 transistors runs its plans in ngspice with the foundry's own models, at its process corners, and its results read as the engine's.
jolty stops at the circuit: it writes SPICE netlists for ngspice and KiCad netlists for boards, and leaves laying out boards and chips to the tools made for them.

The [documentation](https://maxencefaldor.github.io/jolty/) explains every concept and lists every command.
Its [gallery](https://maxencefaldor.github.io/jolty/gallery/) shows thirty famous circuits, analog and digital, each simulated by jolty and replicated in ngspice, and its [discovery](https://maxencefaldor.github.io/jolty/discovery/) pages six hard design problems and the circuits an agent found for them with jolty alone.

## 🚀 Quick start

Building jolty needs [uv](https://docs.astral.sh/uv/), which installs Python 3.14, and a [Rust toolchain](https://rustup.rs/).

```sh
git clone https://github.com/maxencefaldor/jolty
cd jolty
uv sync                                        # builds the Rust engine and installs jolty
uv run jolty simulate examples/rc_filter.py    # builds the filter and runs its plan
```

The command saves the circuit to `examples/rc_filter.jolty.json` and draws it to `examples/rc_filter.schematic.png`, saves the results to `examples/rc_filter.results.npz` and plots them beside it, and prints a report:

```text
step: transient to 10 µs in 67 steps, 0 rejected
              first  min        max       last
  in          0 V    0 V        1 V       1 V
  out         0 V    0 V        999.9 mV  999.9 mV
  V1:current  0 A    -999.5 µA  0 A       -119.6 nA
```

## 🤖 Use it with an agent

An agent reaches jolty through one of two front ends, which share every argument and print the same text:

- **The command line**, for any harness that runs shell commands: `uv run jolty build` and `uv run jolty simulate`.
- **The MCP server**, for any harness that speaks the Model Context Protocol: `claude mcp add jolty -- uv run --project /path/to/jolty jolty-mcp` adds it to Claude Code.

Scripts run in a process of their own, with a time limit, so nothing they print reaches the protocol stream.

## ⚡ How it works

A component is a Python function that declares ports, places devices and other components, and connects their terminals:

```python
@component
def low_pass(c: Builder, resistance: Quantity, capacitance: Quantity) -> None:
    source = c.port("input")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(resistance), id="R")
    capacitor = c.place(devices.capacitor(capacitance), id="C")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)
```

Calling `low_pass(1 * kohm, 1 * nF)` returns a component that any other component places, so circuits grow into hierarchies of boxes.
Values carry units, and a capacitance passed where a resistance belongs fails on the line that passed it.

A plan says how to simulate a circuit, apart from the circuit itself:

```python
def plan() -> Plan:
    return Plan(
        {
            "op": OperatingPoint(),
            "step": Transient(stop=10 * us),
            "bode": Ac("V1", decade(1 * kHz, 100 * MHz, 10)),
        }
    )
```

Specs say what the results must show, and the simulation states the verdict on each requirement, with the value it read:

```python
def specs() -> Specs:
    return Specs(
        [
            gain("bode", "out", at=1 * kHz).within(0, 0.1),
            (rise_time("step", "out") <= 2.5 * us).with_margin(100 * ns),
        ]
    )
```

The engine solves the charge-oriented equations `f(x) + d/dt q(x) = 0` of modified nodal analysis by Newton iteration, with SPICE's convergence aids, one sparse LU factorization shared by every analysis, and adaptive time steps controlled by the local truncation error.

## 🏗️ Architecture

The Rust workspace in `crates/` holds the engine, each crate built on the ones before it:

1. `jolty-ir`: the circuit format, its names and its validation.
1. `jolty-numerics`: sparsity patterns, sparse LU over real and complex numbers, and integration formulas.
1. `jolty-models`: the interface every device implements, and the standard library of devices.
1. `jolty-engine`: elaboration of a hierarchy into devices bound to unknowns, Newton iteration, and the analyses.
1. `jolty-python`: the Python extension module `jolty._core`.

The Python package in `src/jolty` holds everything an agent reads and writes: units, the builder, plans, results, specs, reports, the command line and the MCP server.

## 📄 License

jolty is licensed under the [Apache License 2.0](LICENSE).
