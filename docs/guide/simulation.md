# Simulation

A plan names the analyses to run on a circuit and the numerical options they share.
It stays apart from the circuit: the same circuit runs under any plan.

## Plans

A script defines `plan()`, which returns a `Plan` of analyses by name:

```python
def plan() -> Plan:
    return Plan(
        {
            "op": OperatingPoint(),
            "transfer": DcSweep("V1", linear(0 * V, 5 * V, 51)),
            "bode": Ac("V1", decade(10 * Hz, 10 * MHz, 20)),
            "step": Transient(stop=10 * us),
        },
        options=Options(relative_tolerance=1e-6),
    )
```

Each name is a Python identifier and names the analysis's results.
`jolty simulate` takes the plan from the build script, or from another script given with `--plan`.

## Analyses

| Analysis | Finds |
| --- | --- |
| `OperatingPoint()` | the DC solution, every source at its level at time zero |
| `DcSweep(source, levels)` | the DC solution at each level of one independent source |
| `Ac(source, frequencies)` | the small-signal response to a unit excitation of one source, around the operating point |
| `Transient(stop, max_step=None, method="trapezoidal")` | the response over time, from the operating point until `stop` |

A source is named by its path, such as `V1` or `bias/I1`.
`linear(start, stop, points)` spaces levels evenly, and `decade(start, stop, per_decade)` spaces frequencies evenly on a log scale.
A transient's method is `"trapezoidal"`, `"gear2"` for stiff circuits that ring, or `"backward_euler"` for the most damping.
Its longest step defaults to a fiftieth of `stop`.

## Temperature

A plan's `temperature` is the temperature every device is simulated at, 27 °C unless given:

```python
Plan({"op": OperatingPoint()}, temperature=celsius(85))
```

`celsius(degrees)` gives a temperature in kelvins, as every temperature is.
Away from each device's nominal temperature, its parameters scale as [Devices](devices.md) says.

## Corners

A plan's corners run its analyses again in other conditions: another temperature, or other arguments to the build script's `build()`, which builds the circuit again with them:

```python
def build(supply: Quantity = 3.3 * V) -> Component:
    return bench(supply)


def plan() -> Plan:
    return Plan(
        {"op": OperatingPoint(), "bode": Ac("V1", decade(10 * Hz, 10 * MHz, 20))},
        corners={
            "cold": Corner(temperature=celsius(-40), supply=3.0 * V),
            "hot": Corner(temperature=celsius(125), supply=3.6 * V),
        },
    )
```

Each corner is named by a Python identifier, and runs at the plan's temperature unless it gives its own.
The report, the results file and the plots hold the plan's own run; a corner's report says whether the circuit elaborates there and which analyses failed.
[Specs](specs.md) hold at every corner.

## Options

`Options` sets the tolerances; each one left out takes SPICE's default.

| Option | Default | Meaning |
| --- | --- | --- |
| `relative_tolerance` | 1e-3 | relative accuracy of every unknown and residual |
| `voltage_tolerance` | 1e-6 V | absolute accuracy of voltages |
| `current_tolerance` | 1e-12 A | absolute accuracy of currents |
| `charge_tolerance` | 1e-14 C | absolute accuracy of charges, for the time step |
| `gmin` | 1e-12 S | the conductance across every junction |
| `dc_iterations` | 100 | Newton iterations allowed for a DC solution |
| `transient_iterations` | 10 | Newton iterations allowed for a time step |
| `truncation_tolerance` | 7 | how much the time step's truncation error may exceed the tolerances |

## How the engine solves a circuit

The engine flattens the hierarchy into devices bound to unknowns: the voltage of every node but the ground, and the current through every voltage source, inductor and voltage-controlled voltage source.
It writes the circuit's equations in charge-oriented modified nodal analysis, `f(x) + d/dt q(x) = 0`, where `f` sums the currents into each node and `q` the charges.

Newton iteration solves them, with one sparse LU factorization whose symbolic analysis every analysis shares.
An iteration converges when its last step is within tolerance of every unknown and every equation balances within tolerance of the terms it sums.
When Newton alone fails, the operating point steps the conductance across every node down from 10 mS, then ramps every source up from zero.

A transient integrates with the trapezoidal rule, Gear's second-order formula or backward Euler.
It estimates each step's truncation error from divided differences, shortens a step whose error is too large, and lengthens the next when the error allows.
It restarts with backward Euler after each corner of a waveform.

The engine's results agree with ngspice's on a corpus of reference circuits: operating points, sweeps and frequency responses to within a millionth of each signal, and transients to within a thousandth of their swing.
