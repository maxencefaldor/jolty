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
A plan's `fidelity` may choose characterized logic cells for some component instances (see [Fidelity](fidelity.md)).
`jolty simulate` takes the plan from the build script, or from another script given with `--plan`.

## Analyses

| Analysis | Finds |
| --- | --- |
| `OperatingPoint()` | the DC solution, every source at its level at time zero |
| `DcSweep(source, levels)` | the DC solution at each level of one independent source |
| `Ac(source, frequencies)` | the small-signal response to a unit excitation of one source, around the operating point |
| `Transient(stop, max_step=None, method="trapezoidal", times=())` | the response over time, from the operating point until `stop` |
| `Sensitivity(outputs)` | the derivative of each output with respect to every number of every device's parameters, at the operating point |

A source is named by its path, such as `V1` or `bias/I1`.
`linear(start, stop, points)` spaces levels evenly, and `decade(start, stop, per_decade)` spaces frequencies evenly on a log scale.
A transient's method is `"trapezoidal"`, `"gear2"` for stiff circuits that ring, or `"backward_euler"` for the most damping.
Its longest step defaults to a fiftieth of `stop`.
It steps onto each of `times` exactly, besides the times it chooses, without restarting its integration there, so every unknown and terminal current has its solved value at those times.

## Sensitivities

A sensitivity analysis differentiates unknowns, such as `out` or `V1:current`, with respect to every parameter of every device, at the operating point:

```python
Plan({"sensitivity": Sensitivity(["out", "V1:current"])})
```

A parameter is a number of a device's parameters, named by the device's path and its keys, such as `R1` and `resistance`, or `V1` and `waveform.dc` within an object; a list's entries are keyed by their index from 0, such as `waveform.piecewise.1.1`.
The derivatives come from the adjoint method: one solve with the transposed Jacobian per output serves every parameter, so many parameters cost little.
Each device's own derivative with respect to a parameter is exact: every device computes with dual numbers that carry the derivative of each value with respect to the parameter through the same equations, temperature scaling included.
A number a model reads as a choice, such as a MOSFET's level, is no real parameter and has no derivative.

The result's `derivative(output, instance, parameter)` reads one derivative in SI base units, and `gradient(output)` an output's derivative with respect to every parameter, in the order of `parameters`.

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

## Monte Carlo

A device may give its parameters tolerances: a resistor's, capacitor's or inductor's `tolerance=0.01` spreads its value evenly within 1 %, and `device.toleranced(parameter, relative, distribution)` gives any number parameter a `"uniform"` or a `"normal"` tolerance, the tolerance three standard deviations.
A plan's `monte_carlo=MonteCarlo(samples, seed=0)` runs its analyses again on each sample, at the plan's own temperature:

```python
def plan() -> Plan:
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(1000, seed=1))
```

Each sample draws every toleranced parameter of every device instance independently, so two placements of one component vary apart, and a seed draws the same samples from the same circuit.
The report counts the samples on which the circuit did not elaborate or an analysis failed; with [specs](specs.md), it states what each requirement read over the samples and the yield.

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
Every other choice the engine makes is one SPICE simulators make, save those the table marks as jolty's own:

| Choice | What the engine does | Where it comes from |
| --- | --- | --- |
| Convergence | An iteration converges when its last step is within tolerance of every unknown, every equation balances within tolerance of the terms it sums, and no device limited a voltage; an equation also balances within the rounding of the unknowns it reads. | SPICE's tests on `reltol`, `vntol` and `abstol`; the rounding floor is jolty's own, so a milliohm between two nodes at hundreds of volts can converge. |
| Limiting | A junction's voltage steps at most logarithmically above its critical voltage, and a field-effect transistor's gate and drain step a bounded distance around its threshold, each device evaluated at the limited voltage and extended linearly to the applied one. | SPICE's `pnjlim`, `fetlim` and `limvds`. |
| `gmin` | A conductance of 1e-12 S across every junction. | SPICE. |
| Operating point | Newton from zero; then a conductance from every node to the ground, stepped down from 10 mS, ten times at a time when the steps converge quickly and less otherwise; then every source ramped up from zero. | ngspice's dynamic gmin stepping and SPICE's source stepping. |
| DC sweep | Each level starts from the last solution, and a level Newton cannot reach in one step is approached in steps halved after each failure; a first level no strategy reaches from zero is approached from the circuit's own operating point. | Natural-parameter continuation. |
| Integration | The trapezoidal rule by default, or Gear's second-order formula or backward Euler, as the transient asks. | SPICE. |
| Truncation error | Each step's error is estimated from divided differences of the charges, against tolerances on the charge and on the current it carries, times `truncation_tolerance`. | SPICE's `CKTterr` and `trtol`. |
| Step length | At most a fiftieth of the run unless `max_step` says otherwise; a step whose error is too large is retried shorter, the next grows by at most twice, with a safety factor of 0.9; one whose Newton iteration fails is retried an eighth as long. | SPICE. |
| Corners | Steps land on every corner of every waveform, and on every time asked for; after a corner, the first step is a tenth of the shorter of the last step and the gap to the next corner, and takes backward Euler. | SPICE. |
| Ringing | Whenever some charge's derivative under the trapezoidal rule turns back and forth twice in a row by more than its tolerance, the next step takes backward Euler. | jolty's own, as simulators that mix the trapezoidal rule with backward Euler do: the trapezoidal rule carries an error in a derivative from step to step undamped. |
| Slow convergence | A step whose Newton iteration runs out of iterations is retried from its last iterate rather than from the last solution. | jolty's own: an iteration converging slowly, as at a transistor turning off with nothing else to hold its source, goes on where it stopped. |
| Logic | Detectors' crossings are located by linear interpolation within a step, and a step that crosses early is retried to land on the crossing; at DC the logic and the analog circuit settle each other at most four times before the detectors still changing read unknown levels. | jolty's own event scheduler, after the connect elements of mixed-signal simulators. |
| Logic loads | Each logic driver's load is the capacitance it drives, read from its small-signal current at 1 Hz at the operating point. | jolty's own. |

When no step converges, however short, the transient fails and names why: the unknowns furthest from converging, or the equation whose truncation error stayed above its tolerance.

The engine's results agree with ngspice's on a corpus of reference circuits: operating points, sweeps and frequency responses to within a millionth of each signal, and transients to within a thousandth of their swing.
