# jolty

A compiler and simulator for electronic circuits, built for AI agents.

An agent writes a circuit as a Python script of reusable components.
jolty checks every device and connection, reports each problem with the script line behind it, draws the schematic, and simulates the circuit with a native engine written in Rust.

## What jolty does

- **Builds** circuits from reusable Python components, with units checked: a capacitance passed where a resistance belongs fails on that line.
- **Simulates** operating points, DC sweeps, AC, transients, periodic steady states, noise, stability, pole-zero, transfer functions, sensitivities, and mismatch alongside event-driven logic.
- **Checks specs**: every simulation measures its results against the circuit requirements at each temperature and corner, over Monte Carlo samples, and at worst-case extremes.
- **Scales** across every core and across machines, producing bit-identical results on any thread count.
- **Works with SPICE**: reads vendor SPICE models, agrees with ngspice, writes SPICE and KiCad netlists, and runs SKY130 transistors through ngspice using foundry models.

jolty stops at the circuit: board and chip layout remain with dedicated physical-design tools.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and a [Rust toolchain](https://rustup.rs/).

```sh
git clone https://github.com/maxencefaldor/jolty
cd jolty
uv sync                                        # builds the engine and installs jolty
uv run jolty simulate examples/rc_filter.py    # builds the filter and runs its plan
```

Running `jolty simulate` saves the circuit, schematic, results, and plots next to the script and prints a summary report:

<!-- output of: jolty simulate rc_filter.py --no-schematic --no-plot -->

```text
step: transient to 10 µs in 73 steps, 3 rejected
              first  min        max       last
  in          0 V    0 V        1 V       1 V
  out         0 V    0 V        999.9 mV  999.9 mV
  V1:current  0 A    -999.5 µA  0 A       -119.9 nA
```

## Using jolty with an agent

- **Command line**, for any harness that runs shell commands: `uv run jolty build` and `uv run jolty simulate`.
- **MCP server**, for any harness that speaks the Model Context Protocol: `claude mcp add jolty -- uv run --project /path/to/jolty jolty-mcp`.

Both interfaces accept the same arguments and print the same text.

## A circuit, plan, and specs

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
The plan specifies how to simulate the circuit, and the specs define requirements on its results.
Each simulation reports a verdict and measured value for every requirement.

## Learn more

- [Documentation](https://maxencefaldor.github.io/jolty/): every concept and every command.
- [Gallery](https://maxencefaldor.github.io/jolty/gallery/): thirty well-known circuits, analog and digital, each simulated by jolty and replicated in ngspice.
- [Discovery](https://maxencefaldor.github.io/jolty/discovery/): six hard design problems, and the circuits an agent found for them with jolty alone.

## Architecture

| Part | Responsibility |
| --- | --- |
| `crates/jolty-derive` | Derive macros (`Lift`) converting device parameters across scalar and dual types |
| `crates/jolty-ir` | Circuit intermediate representation and validation |
| `crates/jolty-numerics` | Sparse LU factorization for circuit matrices and integration formulas |
| `crates/jolty-expressions` | Behavioral-source expression language, compiled and differentiated |
| `crates/jolty-models` | Device models, each evaluated as a parallel kernel |
| `crates/jolty-engine` | Elaboration, Newton iteration, analyses, and batch execution |
| `crates/jolty-checks` | Structural and linearized operating-point rules and diagnostics |
| `crates/jolty-veriloga` | Verilog-A compiler and compiled device kernels |
| `crates/jolty-python` | Python extension module `jolty._core` |
| `src/jolty` | Python API, units, builder, plans, specs, results, reports, CLI, and MCP server |
