# jolty

A compiler and simulator for electronic circuits, built for AI agents.

An agent writes a circuit as a Python script.
jolty checks every device and connection, points each problem at the script line behind it, draws the schematic, and simulates the circuit with a native engine written in Rust.

## What it does

- **Builds** circuits from reusable Python components, with units checked: a capacitance passed where a resistance belongs fails on that line.
- **Simulates** operating points, DC sweeps, AC, transients, periodic steady states, noise, stability, pole-zero, transfer functions and sensitivities, beside event-driven logic.
- **Checks specs**: every simulation measures its results against the circuit's requirements, at each temperature and corner, over Monte Carlo samples and at worst-case extremes.
- **Scales** across every core and across machines, with results the same bit for bit on any number of threads.
- **Works with SPICE**: it reads vendors' SPICE models and agrees with ngspice, writes SPICE and KiCad netlists, and runs SKY130 transistors through ngspice with the foundry's models.

jolty stops at the circuit: board and chip layout is left to the tools made for it.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and a [Rust toolchain](https://rustup.rs/).

```sh
git clone https://github.com/maxencefaldor/jolty
cd jolty
uv sync                                        # builds the engine and installs jolty
uv run jolty simulate examples/rc_filter.py    # builds the filter and runs its plan
```

This saves the circuit, its schematic, its results and their plots beside the script, and prints a report:

```text
step: transient to 10 µs in 67 steps, 0 rejected
              first  min        max       last
  in          0 V    0 V        1 V       1 V
  out         0 V    0 V        999.9 mV  999.9 mV
  V1:current  0 A    -999.5 µA  0 A       -119.6 nA
```

## Use it with an agent

- **Command line**, for any harness that runs shell commands: `uv run jolty build` and `uv run jolty simulate`.
- **MCP server**, for any harness that speaks the Model Context Protocol: `claude mcp add jolty -- uv run --project /path/to/jolty jolty-mcp`.

Both take the same arguments and print the same text.

## A circuit, its plan and its specs

```python
@component
def low_pass(c: Builder, resistance: Quantity, capacitance: Quantity) -> None:
    source = c.port("input")
    output = c.port("output")
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(resistance), name="R")
    capacitor = c.place(devices.capacitor(capacitance), name="C")
    c.connect(source, resistor.a)
    c.connect(resistor.b, capacitor.a, output)
    c.connect(capacitor.b, ground)


def plan() -> Plan:
    return Plan(
        {
            "op": OperatingPoint(),
            "step": Transient(stop=10 * us),
            "bode": Ac("V1", decade(1 * kHz, 100 * MHz, 10)),
        }
    )


def specs() -> Specs:
    return Specs(
        [
            gain("bode", "out", at=1 * kHz).within(0, 0.1),
            (rise_time("step", "out") <= 2.5 * us).with_margin(100 * ns),
        ]
    )
```

A component places devices and other components, so circuits grow into hierarchies.
The plan says how to simulate the circuit, and the specs what its results must show: each simulation gives a verdict on every requirement, with the value it read.

## Learn more

- [Documentation](https://maxencefaldor.github.io/jolty/): every concept and every command.
- [Gallery](https://maxencefaldor.github.io/jolty/gallery/): thirty well-known circuits, analog and digital, each simulated by jolty and replicated in ngspice.
- [Discovery](https://maxencefaldor.github.io/jolty/discovery/): six hard design problems, and the circuits an agent found for them with jolty alone.

## Architecture

| Part | What it holds |
| --- | --- |
| `crates/jolty-ir` | The circuit format and its validation |
| `crates/jolty-numerics` | Sparse LU for circuit matrices, and integration formulas |
| `crates/jolty-models` | The device models, each a kernel evaluated in parallel |
| `crates/jolty-engine` | Elaboration, Newton iteration, the analyses and the batch executor |
| `crates/jolty-python` | The Python extension module `jolty._core` |
| `src/jolty` | What an agent reads and writes: units, the builder, plans, specs, results, reports, the command line and the MCP server |
