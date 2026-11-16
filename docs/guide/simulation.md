# Simulation

A plan names the analyses to run on a circuit, the numerical options they share, and what their results record.
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
| `OperatingPoints(max_solutions=8)` | every DC solution, up to `max_solutions`, the operating point first |
| `DcSweep(source, values)` | the DC solution at each level of one independent source |
| `Ac(source, frequencies)` | the small-signal response to a unit excitation of one source, around the operating point |
| `Transient(stop, max_step=None, method="trapezoidal", breakpoints=(), initial_conditions={}, operating_point=True)` | the response over time, from the operating point, or from [initial conditions](#initial-conditions-and-nodesets), until `stop` |
| `PeriodicSteadyState(period=None, max_step=None, method="trapezoidal", breakpoints=())` | one period of the [steady state](#periodic-steady-state) a circuit repeats: driven by sources of `period`, or an oscillator's, whose period it finds |
| `Sensitivity(outputs, of=None)` | the derivative of each output with respect to every number of every device's parameters but a device's [own temperature](devices.md), at the operating point or at each point of the analysis `of` |
| `Stability(probe, frequencies)` | the loop gain through a probe in series with a feedback loop, at each frequency |
| `Noise(output, frequencies, reference=None, source=None)` | the noise density at an output from every device's noise, at each frequency, and referred to an input source |
| `PoleZero(source=None, output=None, reference=None)` | the poles of the circuit linearized at its operating point, and the zeros of the transfer function from a source to an output |
| `TransferFunction(source, output, reference=None)` | SPICE's `.tf`: the gain at DC from an independent source to a net's voltage or a voltage source's current, the resistance the source sees, and the one the output sees with the source at its level |

A source is named by its path, such as `V1` or `bias/I1`.
A transfer function linearizes the circuit at its operating point and solves it twice: driven by its source, for the gain and the input resistance, and driven at its output, a unit current into the net or a volt in series with the source whose current it is, for the output resistance; a voltage source is a short to that second solve and a current source is open, as in SPICE.
A resistance through which no current can flow, such as a voltage source's that drives nothing, is infinite, where ngspice writes 10²⁰ Ω.
`linear(start, stop, points)` spaces levels evenly, and `decade(start, stop, per_decade)` spaces frequencies evenly on a log scale.
A transient's method is `"trapezoidal"`, `"gear2"` for stiff circuits that ring, or `"backward_euler"` for the most damping.
Its longest step defaults to a fiftieth of `stop`.
It steps onto each of `times` exactly, besides the times it chooses, without restarting its integration there, so every unknown and terminal current has its solved value at those times.

A simulation solves its operating point once: it remembers the last one it solved, and every analysis about the operating point, a DC sweep that starts at its source's own level, and the checks at the operating point take it again, as long as the options stay the same, save `charge_tolerance`, `max_transient_iterations`, `truncation_tolerance`, `max_memory` and `profile`, which a DC solution does not read.
Under other tolerances or another `max_dc_iterations` alone, Newton iteration goes on from where it stopped, as from zero it would, since it takes the same iterates from zero under any tolerances.

## Initial conditions and nodesets

A transient's `initial_conditions` give nets, by path, the voltages they start from, as SPICE's `.ic`: each is held at its voltage, as by an ideal source whose current takes up the sum of the net's, while the operating point the transient starts from is solved, and is left free from the first step on.
With `operating_point=False`, as SPICE's `UIC`, the transient starts from its initial conditions alone, every other unknown at zero, an inductor's current among them, so a circuit starts from rest or from a state no DC solution has, such as an oscillator kicked off its balance; the first step, by backward Euler, takes each capacitor's charge from that state, and its results' statistics give no strategy for the operating point.
The options' `nodesets` give nets, by path, voltages an operating point starts from, as SPICE's `.nodeset`: the circuit is solved from zero with each held at its voltage, then released and solved again from there, as SPICE releases them once the circuit converges with them, so that a circuit with several DC solutions, such as a latch, settles in the one they lead to; where the circuit has no solution with them held, it is solved from zero.
A net that voltage sources and inductors join to the ground, or to a net named before it in the order of the names, is not held, as two ideal sources cannot hold one voltage: the sources fix it, and an initial condition there must agree with the voltage they give it within the tolerances, or the transient states both.

```python
from jolty.plan import Options, Plan, Transient
from jolty.units import V, us

Plan(
    {"from_half": Transient(stop=10 * us, initial_conditions={"out": 0.5 * V})},
    options=Options(nodesets={"latch/q": 3.3 * V}),
)
```

## Periodic steady state

A circuit driven by periodic sources, such as a rectifier, a mixer or a power converter, settles into a steady state that repeats each period, but a transient from the operating point reaches it only after its slowest time constant has died away many times over.
A periodic steady state finds it directly, given the period of every source, as ngspice's `.pss`, Spectre's `pss` and Xyce's `.HB` are analyses of their own:

```python
Plan({"steady": PeriodicSteadyState(period=1 * us, max_step=5 * ns)})
```

The steady state is a state `x₀` that one period of the circuit's equations carries back to itself, `φ(x₀) = x₀`, and shooting solves it by Newton iteration (Aprille and Trick, *Steady-state analysis of nonlinear circuits with periodic inputs*, Proceedings of the IEEE 60(1), 1972).
Each period, simulated from the state it starts at, carries the derivatives of its solution with respect to that state along every step, by the direct method of the transient sensitivities, and ends with the monodromy matrix `Φ = ∂φ/∂x₀`; the next period starts at `x₀ + Δ`, where `(Φ − I) Δ = x₀ − φ(x₀)`.
The first starts at the operating point, and the search ends when a period ends within tolerance of every unknown where it started, or fails after 40 periods, naming the unknowns furthest from repeating.
The steady state of a nearly linear circuit is found in two or three periods however slowly it settles, where a transient would take tens or hundreds.
The result, a `PeriodicSteadyStateResult`, holds the last period's waveforms, its times from zero to the period, as a transient's result does, which measures, plots and the viewer read alike; its `period` is the period, and its `periods` count the periods shooting simulated.
The period's steps are at most `max_step`, by default a fiftieth of the period, and land on each of `breakpoints`, times within the period.

A circuit can have a neutral direction, a quantity a period carries back as it was wherever it starts: the state of a memristor its drive leaves alone, or the charge a memristor and a capacitor in series share.
Every value of it has a steady state of its own, and `Φ − I` is singular.
The step `Δ` is then the least-squares solution of least norm among those in the range of `Φ − I`, which leaves every quantity a period conserves as it was, so shooting keeps the start's place along the neutral directions and finds the steady state a transient from the operating point would settle into; where `Φ − I` is regular, it is Newton's step.
The system is measured in each unknown's tolerance, as the mismatches are, and a direction is neutral when its singular value is within the rounding of the matrix, or below the relative tolerance with the start already repeating along it within its tolerance: there the period moves the state by too little for the tolerances to place the steady state along it.
The monodromy matrix and the dense factorizations of `Φ − I` hold up to 14 numbers per unknown squared, so a circuit whose matrices would pass the `max_memory` option fails before the first period, and each period's waveforms may hold what the matrices leave of it.
A circuit with logic, whose state shooting does not carry, has no periodic steady state.

### Oscillators

Without a period, the circuit is an oscillator, such as a ring oscillator, an LC oscillator or a relaxation oscillator, whose sources hold steady and whose period is its own; shooting finds the period with the state:

```python
Plan({"ring": PeriodicSteadyState()})
```

The unknowns are then `x₀` and the period `T`, and the equations `φ(x₀, T) = x₀` are one fewer: any state along the orbit starts a period as well as any other.
The phase condition takes that freedom out, as the least-norm step takes out a neutral direction: the start moves only across the orbit, its step `Δx` orthogonal to the flow there, `ẋᵀ Δx = 0`, each unknown in units of its tolerance (the orthogonal phase condition; R. Seydel, *Practical Bifurcation and Stability Analysis*, 3rd ed., Springer, 2010, §7.1).
The flow is `∂φ/∂T`, the end's derivative with respect to the period, which at the steady state is the start's, and each period carries it along its steps as it carries `Φ`, every step's length a fixed fraction of the period: a step's formula scales its charges' terms by `1/T`, which makes the derivative exact for the steps taken.
Newton's step solves the bordered system `[[Φ − I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [Δx, ΔT] = [x₀ − φ(x₀, T), 0]`, regular where the orbit is isolated, and measured in each unknown's tolerance and the period in its relative tolerance; a neutral direction of the circuit's own is left as it was, as above.

An oscillator's operating point is an equilibrium it leaves, and a state every period repeats, so shooting cannot start there: it starts on the orbit, which a transient from the operating point finds.
The transient starts from the operating point kicked along the natural mode that grows fastest, its pole the [pole-zero analysis's](#poles-and-zeros) with the largest real part, by one tolerance in the unknown the mode moves most in units of its tolerance, the smallest step off the operating point the analysis resolves.
It returns to the hyperplane through the operating point across the kick once a period, and runs, in stretches twice as long as the one before, the first the mode's natural period, until a return comes no further from the one before than that one from its own, which was further than their tolerance: the oscillation has grown past what the analysis resolves and stopped growing; returns within tolerance of each other come nearer or further by the chance of each step's own error.
Each stretch restarts the integration with backward Euler's steps, which damp the oscillation a little, so a return is compared only with those of its own stretch, and a stretch's steps are at most a fiftieth of the period the returns before it show, as a period's own are.
Shooting starts from that return, its period the time since the one before, and converges, as the circuits of the tests do, in three or four periods.
An oscillator whose operating point no mode grows from, such as a lossless tank, does not start, and the analysis says so, as it does when shooting converges onto the operating point, a state every period repeats, where no unknown moves by its tolerance as the period moves by its own; one that has not stopped growing after 40 stretches, such as a latch, which tips into a stable state and never returns, fails, saying how long its transient ran and how often it returned.
The sources hold steady over each period, so a source that kicks a transient, such as the examples' pulses, has no place in an oscillator's circuit.
A tank whose conductance feeds it inside a circle and drains it outside, so that it is lossless on the circle, has the period `2π√(LC)` exactly, and the analysis finds it within the trapezoidal formula's phase error, `(ωh)²/12` of the period for a step `h`, and the circle's radius within the energy the period's first two steps of backward Euler drain (`crates/jolty-engine/tests/periodic.rs`).
The examples' five-stage ring oscillator's period agrees with its long transient's settled period to 2.8·10⁻⁸ at a relative tolerance of 10⁻⁹ and steps of at most 1 ps, within the 5.9·10⁻⁸ its transient's periods scatter by; at the default tolerance and steps of at most 5 ps, it lies 2.9·10⁻⁵ from that reference, where the transient's settled period lies 1.7·10⁻⁴ from it, and at 10⁻⁵, 2.7·10⁻⁴, where the transient's lies 1.6·10⁻⁴ from it (`tests/test_periodic.py`, on an Apple M2 Max).

## Several DC solutions

A circuit with feedback can have several DC solutions: a latch one per state and one balanced between them, and a self-biased reference its zero-current state beside its working one.
A search for them finds the operating point first, then each other solution its equations lead to once deflated by those found so far:

```python
Plan({"states": OperatingPoints(max_solutions=8)})
```

The deflated equations are `m(x)·f(x) = 0`, with `m(x) = Π (1/‖x − r‖² + 1)` over the solutions `r` found, the distance taken over the node voltages: each found solution becomes infinitely repelling, so Newton iteration, whose deflated step is the plain step scaled by `1 / (1 − ∇m·δ/m)`, leaves it for another (Farrell, Birkisson and Funke, *Deflation techniques for finding distinct solutions of nonlinear partial differential equations*, SIAM Journal on Scientific Computing 37(4), 2015).
Each new solution is searched from zero, by Newton iteration, then gmin stepping across the nodes and then across the junctions, then source stepping, then pseudo-transient continuation, and else by Newton iteration from the reflection of a solution found through another, or from beside one, its node voltages moved 0.1 V up and down in turn, which breaks the symmetry a balanced latch's start keeps; the search ends when none leads to a new solution, ten tolerances or more from every one found in some unknown, or at `most`.
The result's `points` hold an operating point per solution, and `voltages(net)` reads a net's voltage at each; the report lists every unknown's value at each solution, and the specs' `solutions` reads how many there are.

## Sensitivities

A sensitivity analysis differentiates unknowns, such as `out` or `V1:current`, with respect to every parameter of every device, at the operating point:

```python
Plan({"sensitivity": Sensitivity(["out", "V1:current"])})
```

Given another analysis, it differentiates the outputs at each of its points instead: each level of a DC sweep, each frequency of a small-signal response, whose phasors' derivatives are complex, each time of a transient, and each point of a periodic steady state's period, at its fraction of the period:

```python
Plan({"drift": Sensitivity(["out"], of=Transient(stop=1 * ms))})
```

A parameter is a number of a device's parameters, named by the device's path and its keys, such as `R1` and `resistance`, or `V1` and `waveform.dc` within an object; a list's entries are keyed by their index from 0, such as `waveform.piecewise.1.1`.
At a DC point, the derivatives come from the adjoint method: one solve with the transposed Jacobian per output serves every parameter, so many parameters cost little.
Over frequency, the adjoint at each frequency reads how the matrix `G + jωC` moves with each parameter, directly and through the operating point; the second part is a central difference along the operating point's own derivative, a microvolt either way, since devices give no second derivatives.
Over time, the direct method carries each parameter's derivative of the solution from step to step through the same integration formula, starting from the operating point's, with each step's Jacobian evaluated again at its solution (Hocevar, Yang, Trick and Epler, 1985); a derivative moves no step and no logic event.
Over a [periodic steady state](#periodic-steady-state), a period from the start `x₀` it repeats carries each parameter's derivatives the same way, the start held, to its end, `∂φ/∂p`; the steady state moves so that it still repeats, `(Φ − I) dx₀/dp = −∂φ/∂p`, by the converged period's monodromy matrix, in shooting's units, a neutral direction left where it was, and an oscillator's period moves with it, by the bordered system `[[Φ − I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [dx₀/dp, dT/dp] = [−∂φ/∂p, 0]`, its start still orthogonal to the flow.
A second period from `x₀` then carries each parameter's derivatives from `dx₀/dp`, each step's charge terms scaled by `−(dT/dp)/T` too, every step a fixed fraction of the period, so the derivatives along the period are at fixed fractions of it.
The three periods take the same steps, and no singular value of the bordered system counts as zero but within its rounding.
A tank whose conductance holds its radius through an impedance of its own scales with time, L and C alike, so `L ∂T/∂L + C ∂T/∂C = T`: the derivatives meet it as far as Newton's iterates meet each step's equations, to 256 ε at a relative tolerance of 10⁻¹⁰ and 7·10⁻¹⁰ at the default 10⁻³, and central differences of the period, each run settled and shot afresh, to 1.2·10⁻⁷, a square root's truncation for steps of 10⁻³ (`crates/jolty-engine/tests/periodic.rs`).
Each device's own derivative with respect to a parameter is exact: every device computes with dual numbers that carry the derivative of each value with respect to the parameter through the same equations, temperature scaling included.
A number a model reads as a choice, such as a MOSFET's level, is no real parameter and has no derivative.

The result's `derivative(output, instance, parameter)` reads one derivative in SI base units, a number at the operating point and an array over the points otherwise, and `gradient(output)` an output's derivative with respect to every parameter, in the order of `parameters`.
Its `result` holds the result of the analysis differentiated, whose levels, frequencies or times the derivatives follow, and for an oscillator's periodic steady state, `period_derivative(instance, parameter)` reads its period's derivative, in seconds per the parameter's unit, and `period_derivatives` all of them, in the order of `parameters`:

```python
Plan({"tuning": Sensitivity(["out"], of=PeriodicSteadyState())})
```

## Stability

A stability analysis finds a feedback loop's gain, from which its phase and gain margins are read.
The loop runs through a probe: a voltage source, usually of 0 V, placed in series with the loop where it carries the loop's signal on, its `positive` terminal toward the side the signal enters, as between an amplifier's output and its feedback network:

<!-- not run: a fragment of a component that places `amplifier` and `feedback` -->

```python
probe = c.place(devices.voltage_source(0 * V), name="VP")
c.connect(amplifier.output, probe.negative)
c.connect(probe.positive, feedback.a)
```

A plan then finds the loop gain through it:

```python
Plan({"loop": Stability("VP", decade(1 * Hz, 1 * GHz, 20))})
```

The engine linearizes the circuit at its operating point and, at each frequency, injects a unit voltage in series through the probe and a unit current from the ground into its `positive` terminal.
From the two, Middlebrook's formula gives the loop gain `T` however the two sides load each other, as Tian's method and Spectre's `stb` analysis do (Middlebrook, *Measurement of loop gain in feedback systems*, International Journal of Electronics 38(4), 1975).
The injected current returns through the ground, so neither of the probe's terminals may be the ground.
Negative feedback makes `T` positive at low frequencies, and divides each response it encloses by `1 + T`.

The result's `gains` hold `T` at each of its `frequencies`.
The report states it at the lowest frequency, where it first falls through 0 dB, the phase margin there, and the gain margin where its phase first falls through -180°; the specs' `crossover`, `phase_margin` and `gain_margin` read the same, and `gain` and `phase` read it at a frequency.

## Noise

A noise analysis finds the noise density at an output, at each frequency, from every device's noise:

```python
Plan({"hiss": Noise("out", decade(10 * Hz, 100 * kHz, 10), source="V1")})
```

The output is an unknown, a voltage such as `out` or a current such as `V1:current`, or a voltage over a `reference` node, for a differential output.
Devices make noise as SPICE models it: a resistance thermal noise, `4kT/R`, a junction's current shot noise, `2qI`, a field-effect transistor's channel thermal noise, `8kT·gm/3`, and a diode or transistor with a flicker coefficient `KF` flicker noise, `KF·|I|^AF/f`, of its current, a bipolar transistor's base current, or a MOSFET's drain current over its gate's area and the oxide capacitance squared, as [Devices](devices.md) says.
A resistance under 1 Ω, whose current is an unknown of its own, makes its noise as a voltage in series, `4kTR`, so a short makes none.
Frequencies are positive, as flicker noise needs.

The engine linearizes the circuit at its operating point and, at each frequency, solves the transposed system once, with the output as its right-hand side: the solution weighs every equation's effect on the output, so every noise source's share comes from one solve, and the sources being independent, their densities add.
Given a `source`, an independent source such as `V1`, the same solution gives the gain from its level to the output, and the output's density over the gain's square is the noise referred to that input, as SPICE's input noise is.

The result's `densities` are the output's power densities, in V²/Hz or A²/Hz, at each of its `frequencies`, and `contributions` what each of its `sources`, a device and the noise it makes such as `thermal` or `collector shot`, adds to them.
`referred()` gives the noise referred to the input, `contribution(instance, name=None)` what a device's noise, or one of its sources, adds, `density(at)` the density at a frequency, and `rms(start=None, stop=None)` the RMS noise over a band, each density a power law between frequencies as SPICE integrates it.
The report states the noise density at the lowest and highest frequencies, the RMS noise over them, at the output and referred to the input, and the noisiest sources with their share of the output's noise power; the plot draws the noise density with the noisiest sources', and the noise referred to the input, on log axes; the specs' `noise_density` and `integrated_noise` read them.

## Poles and zeros

A pole-zero analysis finds the circuit's poles, its natural frequencies, and, given a source and an output, the zeros of the transfer function between them:

```python
Plan({"pz": PoleZero(source="V1", output="out")})
```

Linearized at its operating point, the circuit's small-signal equations are `(G + sC) x = −u·b` at the complex frequency `s`, where `u` is the source's level.
The poles are the `s` at which `G + sC` is singular, the finite generalized eigenvalues of the pencil `(G, −C)`, and the zeros those at which the transfer function vanishes, the finite eigenvalues of the pencil bordered by the source's excitation and the output's reading.
The engine solves both pencils, dense, by the QZ algorithm (Moler and Stewart, 1973) in complex arithmetic, as LAPACK's ZGGHRD and ZHGEQZ, after scaling the frequency by the circuit's own and equilibrating their rows and columns; an eigenvalue beyond 10¹² times the circuit's frequency is infinite, and a zero within 10⁻¹⁰ times the largest pole or zero lies at the origin, as a coupling capacitor's does.
The circuit has an operating point, so no pole lies at the origin.
The pencils hold up to eight dense matrices of a row per unknown, a row more for the zeros, so an analysis whose matrices would pass the `max_memory` option fails before it begins.
A natural frequency the source cannot excite, or the output cannot see, is both a pole and a zero, as ngspice reports it too.

The result's `poles` and `zeros` are complex frequencies in radians per second, by increasing magnitude, a conjugate pair's upper member first, and `natural_frequencies(roots)` and `dampings(roots)` read each as a frequency in hertz, its magnitude over 2π, and a damping ratio, minus its real part over its magnitude: 1 for a real root in the left half-plane, between 0 and 1 for a pair that rings, and negative for a pole in the right half-plane, which grows.
The report lists each root, a pair once, with its frequency and damping; the plot draws each at its natural frequency and damping ratio; and the specs' `damping` and `dominant_pole` read the least damping ratio and the slowest pole's frequency.

## Temperature

A plan's `temperature` is the temperature every device is simulated at, 27 °C unless given:

```python
Plan({"op": OperatingPoint()}, temperature=celsius(85))
```

`celsius(degrees)` gives a temperature in kelvins, as every temperature is.
Away from each device's nominal temperature, its parameters scale as [Devices](devices.md) says.

## Corners

A plan's corners run its analyses again in other conditions: another temperature, another [process corner](foundry.md#process-corners) of a foundry's devices, or other arguments to the build script's `build()`, which builds the circuit again with them:

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

Each corner is named by a Python identifier, and runs at the plan's temperature and process corner unless it gives its own.
The corners that build the plan's own circuit, changing only its temperature, run as one [batch](#batches), each corner's temperature the values of a run of it; a corner whose arguments build the circuit again runs a batch of its own.
The report, the results file and the plots hold the plan's own run; a corner's report says whether the circuit elaborates there and which analyses failed.
[Specs](specs.md) hold at every corner.

## Monte Carlo

A device may give its parameters tolerances: a resistor's, capacitor's or inductor's `tolerance=0.01` spreads its value evenly within 1 %, and `device.toleranced(parameter, relative, distribution)` gives any number parameter a `"uniform"` or a `"normal"` tolerance, the tolerance three standard deviations.
The parameter is named by its path, so `devices.voltage_source(5 * V).toleranced("waveform.dc", 0.05)` is a supply within 5 %.
A plan's `monte_carlo=MonteCarlo(samples, seed=0)` runs its analyses again on each sample, at the plan's own conditions and at each of its corners:

```python
def plan() -> Plan:
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(1000, seed=1))
```

Each sample draws every toleranced parameter of every device instance independently, so two placements of one component vary apart, and a seed draws the same samples from the same circuit.
A parameter that sets an exponential's argument spreads in its logarithm: `toleranced("saturation_current", 0.08, "log_uniform")` keeps a transistor's saturation current within a factor of 1.08 either way, its base-emitter voltage within 2 mV, and `"log_normal"` makes that factor three standard deviations.
Parts that drift together, as resistors on one die do, share a lot: `toleranced("resistance", 0.001, lot=ir.Lot(name="die", relative=0.2))` scales the part's own factor, within 0.1 %, by the lot's, drawn once per sample for each parameter and shared by every part that names the lot, wherever it lies in the hierarchy.
A sample runs on the same parts at every corner whose circuit holds the same toleranced parameters, and each requirement's verdict on it is its worst at any of them.
A sample of a circuit of a foundry's devices draws its kit's [mismatch](foundry.md#mismatch) too.
The samples run as a [batch](#batches) at the plan's own conditions and one at each corner, on the circuit elaborated once for the conditions that share it: the engine draws sample `i` from the seed and `i` alone, and runs the samples across its threads, so they are the same however many threads run them, and however many machines, as [shards](machines.md) of the samples.
The report counts the samples on which the circuit did not elaborate or an analysis failed, at the plan's own conditions or a corner; with [specs](specs.md), it states what each requirement read over the samples and the yield.
The plots draw the samples at the plan's own conditions: a histogram of what each requirement read, against its bounds, and the envelope of each plotted net along each analysis, as [Results](results.md#plots) shows.

## Worst case

A plan's `worst_case=WorstCase(step=0.1)` judges each requirement of the [specs](specs.md) at the extremes of the tolerances that worsen it, as PSpice's worst-case analysis does:

```python
def plan() -> Plan:
    return Plan({"op": OperatingPoint()}, worst_case=WorstCase())
```

At the plan's own conditions and at each corner, a run per toleranced parameter, moved `step` of its tolerance, finds which way each requirement's value moves with it.
Then, for each bound of each requirement, every parameter goes to the extreme of its tolerance that moves the value toward the bound, and a lot's spread to the extreme its parts move it toward together, so a spread that cancels, as a matched pair's in a ratio, stays at its nominal value; runs at the same extremes are shared.
A normal tolerance's extreme is its three standard deviations, and a logarithmic one's its factor.
The runs of each circuit run as a [batch](#batches) of listed values, one for the moves and one for the extremes.
The extremes follow each value's first-order slope, as PSpice's do, so a value that bends over a tolerance may be worse inside it; a Monte Carlo analysis samples the inside.
The report states how many runs it took and each requirement's verdict, its worst over its bounds and conditions, and `jolty simulate` exits with 1 when one fails.

## Variants

A plan's `variants` runs its analyses again on many runs of the circuit, each at values of its own, at the plan's own conditions and at each corner, as a Monte Carlo analysis's samples are:

```python
def plan() -> Plan:
    return Plan(
        {"op": OperatingPoint(), "bode": Ac("V1", decade(10 * Hz, 10 * MHz, 20))},
        variants=Grid(
            [
                Axis.temperature([celsius(0), celsius(50), celsius(100)]),
                Axis("R2", "resistance", [8 * kohm, 9 * kohm, 10 * kohm]),
            ]
        ),
    )
```

Variants are the values of every run, each a pure function of the run's index:

- `Explicit([Variant(overrides, temperature), ...])` lists them, each run's numbers by instance path then by parameter path, as a [`Simulation`](#analyses)'s overrides, and its temperature;
- `Grid([Axis(instance, parameter, values), Axis.temperature(values), ...], warm=False)` takes every combination of its axes' values, the last axis fastest, its lines chains of runs each starting from the operating point of the one before, as a DC sweep's levels do, when `warm`;
- `MonteCarlo(samples, seed)` draws [samples](#monte-carlo) of the circuit's tolerances;
- `Product(first, second)` runs every run of `second` at every run of `first`, the values of `second` in place of those of `first` where both give one, so `Product(grid, MonteCarlo(100))` samples the tolerances at every point of the grid.

The report states the runs and the numbers they vary, how many failed, and, with [specs](specs.md), each requirement's spread and the yield; the plots draw the runs at the plan's own conditions as a Monte Carlo analysis's samples are.

## Batches

Every many-runs analysis runs as a batch: one call to the engine for many runs of one circuit, spread across its threads, each run a pure function of its index and of nothing else, so the same bits whatever the threads.
`Simulation.batch` runs one from Python:

<!-- not run: it batches a circuit `bench`, which the page does not build -->

```python
simulation = Simulation(bench())
found = simulation.batch(
    {"op": OperatingPoint(), "bode": Ac("V1", decade(10 * Hz, 10 * MHz, 20))},
    MonteCarlo(10_000, seed=1),
    measures=[gain("bode", "out", at=1 * kHz), value("op", "out")],
    traces=["out"],
)
gains = found.measured[:, 0]  # each run's gain at 1 kHz, in dB
resistances = found.parameter("R1 resistance")  # each run's R1
failed = found.failed  # whether each run failed; found.failure(run) says why
responses = found.traces[1].values_of("out")  # a row per run, a column per frequency
```

The result holds its columns as NumPy arrays, one row per run, which the engine writes at each run's row as it ends: the values each run took, whether each analysis ended, what each [measure](specs.md#measures) read and why one read nothing, and the values of the unknowns `traces` names at each point of an operating point, a DC sweep, a response and a transient that names its `times`, which it steps onto.
The engine reads every measure of `jolty.specs` itself, without Python; a measure a script writes is read under the interpreter's lock as each run ends, as is the function given as `each`, which sees every analysis's result.
Each run finds what a simulation at its values alone finds, bit for bit.
`runs=range(start, stop)` runs only those runs, row 0 the run at `start`: the result holds the rows of the whole batch's that the range names, and `BatchResult.joined` joins the results of ranges that tile the runs into the whole batch's, as `rows` cuts one into ranges.
A result's reasons are each named once, in the order its rows first name them, so that a range's and a join's are those of a batch of their runs.
A run whose values give the circuit another structure, as a resistance below 1 Ω carries its current as an unknown, runs on the circuit elaborated once for that structure, and its measures read the unknowns of the names they read.
A run whose values put a device out of range, or one of whose analyses fails, says why, and the batch goes on.
The threads are the engine's: the setting `threads`, or `jolty.simulate.threads(count)` for a block of code.
A circuit of a foundry's devices runs each run in ngspice instead, as many at once as the engine has threads.

## Options

`Options` sets the tolerances, each one left out taking SPICE's default, and how a run spends its iterations, memory and threads; what a run records is its plan's.

| Option | Default | Meaning |
| --- | --- | --- |
| `relative_tolerance` | 1e-3 | relative accuracy of every unknown and residual |
| `voltage_tolerance` | 1e-6 V | absolute accuracy of voltages |
| `current_tolerance` | 1e-12 A | absolute accuracy of currents |
| `charge_tolerance` | 1e-14 C | the least charge the time step's tolerance on a charge is relative to, as SPICE's `chgtol`: a charge is held to `relative_tolerance` times the larger of its magnitude and `charge_tolerance` |
| `gmin` | 1e-12 S | the conductance across every junction |
| `max_dc_iterations` | 100 | Newton iterations allowed for a DC solution |
| `max_transient_iterations` | 10 | the most Newton iterations a time step takes, SPICE's `itl4`, the iterate after the last judged as any other, where ngspice 47 allows 100 whatever `itl4` says (`niiter.c`, lines 37 to 39) |
| `truncation_tolerance` | 7 | how much the time step's truncation error, as SPICE estimates it, may exceed the tolerances, as SPICE's `TRTOL` |
| `max_memory` | 2 GiB | how much an analysis may hold at once, rather than fill the memory: a transient's results, past which it stops, and the dense matrices of a pole-zero analysis or a periodic steady state, past which it fails before it begins |
| `profile` | off | whether a transient measures where its time goes, into its statistics' `profile`; the clock adds about a tenth to a small circuit's transient, and changes no other result |
| `parallelism` | every thread | how a run spreads its own work across the engine's threads, a `Parallelism`, each threshold at its break-even; no setting changes any result but `max_unknowns_per_interior`, in the last digits |
| `nodesets` | none | the voltage each net named, by path, holds while an operating point is first solved, then [released](#initial-conditions-and-nodesets), as SPICE's `.nodeset` |

A `Parallelism` spreads each kind of a run's work over the engine's threads in tasks of at least its threshold, its least size per task, once the work fills such a task for every thread the run may use, as its break-even was measured, and keeps it on the run's own thread for less, since waking threads costs more than a small run's work repays.
Each threshold left out takes its break-even on the threads the run may use, below:

| Setting | Default | What it spreads |
| --- | --- | --- |
| `max_threads` | none | at most so many of the engine's threads, every one by default; 1 runs a run serially |
| `min_solves_per_task` | its break-even | the frequencies of a response, a noise or a stability analysis and the right-hand sides of sensitivities, in tasks of at least so many |
| `min_instances_per_task` | its break-even | the evaluation of the devices, each group of one model's instances in tasks of at least so many; a linear device, which is evaluated once a run, for its entries, counts none |
| `min_unknowns_per_task` | its break-even | the work of the Newton iteration and of a transient's step control on each unknown and each stored entry of the matrix, in tasks of at least so many, in whole chunks of 1,024 that the threads take as they free |
| `max_unknowns_per_interior` | its break-even | the factorization of the matrix: each block of more unknowns whose elimination nested dissection makes cheaper is ordered so, into interiors of at most so many, which factor at once, and the separators between them, each once the parts it separates are done |

```python
from jolty.plan import Options, Parallelism

options = Options(parallelism=Parallelism(min_solves_per_task=16))
serial = Options(parallelism=Parallelism(max_threads=1))
```

A batch runs each of its runs on one thread whenever it has a run for every thread, so the parallelism matters for one run, or for fewer runs than threads.
`jolty simulate` takes each threshold too, as `--min-solves-per-task` and its kin, in place of the plan's.

Whether to dissect a block is decided by computing what each ordering costs, not by its size: the engine counts the multiply-adds of eliminating the block by the minimum degree ordering and by nested dissection, each on the symbolic factor of the block's pattern plus its transpose, and dissects the block only where nested dissection costs fewer.
On a square grid, as of a mesh of resistors, nested dissection into interiors of at most 157 nodes costs 1.06 times the minimum degree's multiply-adds at 60 nodes a side and 0.96 times at 70, and 0.70 times at 300; on a path, as of a chain of inverters, which the minimum degree eliminates with no fill, 3.9 times at 5,000 nodes, so a chain is never dissected.
The ordering depends on the circuit and on `max_unknowns_per_interior` alone, so a run finds the same bits on the same number of threads, and a dissected factorization orders its operations otherwise than the minimum degree's, so its results differ from those of one with no block dissected in their last digits, within the factorization's rounding.
The default interiors are a break-even on the run's threads, so a circuit large enough to dissect finds other last digits on another number of threads: `rc_mesh_100`, whose mesh of 10,000 nodes the default orders by the minimum degree on one thread and dissects on two or more, finds the same 109 points in its 10 µs on 1, 2 and 4 threads of an Apple M2 Max, its voltages within 2.1e-15 V of one another's.
A batch's runs take the minimum degree's ordering unless the options set the interiors, so that they find the same bits however many threads run them.

Each break-even is the least size of a task from which spreading a kind of work over every thread made a run faster than a serial one, measured by `scripts/bench --break-even` on `cedar`, an AMD Ryzen Threadripper 3970X of 32 cores and 64 threads, on 16 November 2026, on circuits of the benchmarks' scripts at several sizes: `bjt_amp`'s response at 1 to 1,111 frequencies a decade for `min_solves_per_task`, chains of 1 to 1,000 inverters of level 1 MOSFETs for `min_instances_per_task`, chains of 100 to 2,000 inverters and meshes of resistors and capacitors of 10 to 300 nodes a side for `min_unknowns_per_task`, and meshes of 10 to 300 a side for `max_unknowns_per_interior`, while the load average, the measurement's own threads included, was 1.4 to 37 on the 64 threads; [the speed suite's page](https://github.com/maxencefaldor/jolty/blob/main/benchmarks/speed/README.md#break-evens) gives every size measured.
A run takes the break-even measured on the most threads it may use or fewer, and spreads none of a kind of work whose break-even is none, as no size measured paid; `jolty.plan.BREAK_EVENS` holds them:

| Threshold | 2 threads | 4 threads | 8 threads | 64 threads |
| --- | --- | --- | --- | --- |
| `min_solves_per_task` | 4,999 | 11 | 11 | 4 |
| `min_instances_per_task` | none | 250 | 125 | none |
| `min_unknowns_per_task` | none | none | none | none |
| `max_unknowns_per_interior` | 5,001 | 2,501 | 1,251 | 157 |

Each break-even was measured with the other thresholds spreading none of the work, and the rounds of runs alternating their order, as [the speed suite's page](https://github.com/maxencefaldor/jolty/blob/main/benchmarks/speed/README.md#break-evens) says.
A chain's devices spread paid in tasks of 250 MOSFETs on 4 threads and of 125 on 8, but on no chain up to 1,000 inverters on 2 threads, a task taking each group of MOSFETs whole, or on 64, in tasks of 31, so a run on 2 or 3 threads, or on 64 or more, evaluates its devices on its own thread.
A mesh of 100 a side, dissected into interiors of a thread's unknowns, ran 21% faster on 2 threads, 29% on 4, 41% on 8 and 20% on 64, but 51% slower on 64 threads in interiors of 5,001, its break-even on 2, so `max_unknowns_per_interior` takes its break-even on the run's threads too, and a run on one thread orders by the minimum degree.
The work on each unknown spread lost on every chain up to 2,000 inverters, by 8.0% to 39.9% at 2,000, and gained 0.7% to 1.2% on the mesh of 300 a side alone, so it spreads on no number of threads; on 2 threads, the amplifier's frequencies paid from 9,999 alone, as two tasks of 2,250 gained nothing where tasks of 900 had gained 22.6%.

The unknowns cost a transient a number each at every point, eight bytes; a device recorded costs its terminals' currents and charges, the square of its unknowns twice over for its derivatives, and its variables: 12 numbers for a resistor, and 111 for a MOSFET whose gate oxide adds three unknowns to its four terminals.
An operating point and a transient record the instances rated for a current or a power, whose ratings the checks read, and what the plan's or the analysis's [record](results.md#what-a-result-records) names, so a transient of a large circuit holds about its unknowns alone unless asked for more; recording every instance multiplies what it holds by thirty or more, and reaches `max_memory` sooner.

## How the engine solves a circuit

The engine flattens the hierarchy into devices bound to unknowns: the voltage of every node but the ground, and the current through every voltage source, inductor and voltage-controlled voltage source.
It writes the circuit's equations in charge-oriented modified nodal analysis, `f(x) + d/dt q(x) = 0`, where `f` sums the currents into each node and `q` the charges.

Newton iteration solves them, with one sparse LU solver whose ordering every analysis shares, real and complex.
Every other choice the engine makes is one SPICE simulators make, save those the table marks as jolty's own, and [Departures from ngspice](departures.md) lists every place where it computes otherwise than ngspice 47, with the evidence:

| Choice | What the engine does | Where it comes from |
| --- | --- | --- |
| Convergence | An iteration converges when its last step is within tolerance of every unknown, every equation balances within tolerance of the terms it sums, and no device limited a voltage; an equation also balances within the rounding of the unknowns it reads, and a residual within that rounding drives no step. | SPICE's tests on `reltol`, `vntol` and `abstol`; the rounding floor is jolty's own, so a milliohm between two nodes at hundreds of volts can converge, and a source's current beside a node at megavolts settles. ngspice tests each update and each device's prediction of its own current, never the residual (`niconv.c`, lines 40 to 95), and holds an iteration back for its junction limiter alone (`mos1load.c`, lines 376 to 384 and 719 to 720). |
| Limiting | A junction's voltage steps at most logarithmically above its critical voltage, and a field-effect transistor's gate and drain step a bounded distance around its threshold, each device evaluated at the limited voltage and extended linearly to the applied one. | SPICE's `pnjlim`, `fetlim` and `limvds`. ngspice 47's junction limiter is its own, which for a step just over 2vt steps backward (`devsup.c`, lines 54 to 61), and a step's first iteration limits from the last accepted point (`mos1load.c`, lines 201 to 208), where jolty's devices limit from their last evaluation. |
| Self-heating | A heated device's temperature steps at most 50 K from one iteration to the next, and never below 1 K, the device evaluated there and extended linearly to the temperature applied; its parameters are scaled to its temperature at each one. | jolty's own: SPICE heats no device but its few electrothermal models, such as VBIC's. |
| `gmin` | A conductance of 1e-12 S across every junction. | SPICE. |
| Small resistors | A resistor, or a device's series resistance, below 1 Ω carries its current as an unknown, held by Ohm's law, so that its conductance cannot swamp a node's picosiemens in floating point. | Modified nodal analysis's second group of elements (Ho, Ruehli and Brennan, 1975); the threshold is jolty's own, where gmin keeps four digits beside the conductance. |
| Operating point | Newton from zero; then a conductance from every node to the ground, stepped down from 10 mS, ten times at a time when the steps converge quickly and less otherwise; then the conductance across every junction, stepped down the same way to `gmin`; then every source ramped up from zero; then a pseudo-transient from zero, each node held to its last voltage by a conductance that starts at 1 S, a farad over a second, and shrinks as the circuit settles, until a step leaves it where it was. | ngspice's dynamic and true gmin stepping, SPICE's source stepping, and in place of ngspice's transient operating point, pseudo-transient continuation (Kelley and Keyes, 1998), which follows the circuit past the folds where its solutions turn back as the sources ramp. ngspice starts Newton with each junction at its critical voltage and each MOSFET at its threshold (`dioload.c`, lines 162 to 165; `mos1load.c`, lines 397 to 408), and takes its transient operating point, 10 µs at 100 ns steps (`init.c`, lines 77 to 93), without a check that the circuit rests there. |
| DC sweep | Each level starts from the last solution, and a level Newton cannot reach in one step is approached in steps halved after each failure; a first level no strategy reaches from zero is approached from the circuit's own operating point. | Natural-parameter continuation; ngspice solves a level that fails as a new operating point (`dctrcurv.c`, lines 306 to 313), which may land on another branch. |
| Periodic steady state | Newton iteration on the state each period starts at, with the monodromy matrix carried along the period's steps and solved densely by its singular value decomposition, each step in the range of `Φ − I`, which leaves every quantity a period conserves where it started; an oscillator's period is an unknown too, its derivative carried along the steps as the period stretches them, fixed by the orthogonal phase condition, from a start where a transient from the operating point kicked along its fastest-growing mode stops growing. | Shooting (Aprille and Trick, 1972), as Spectre's periodic steady-state analysis; the orthogonal phase condition (Seydel, 2010, §7.1); the step along neutral directions and the start from the fastest-growing mode are jolty's own, where ngspice's `.pss` asks for a guess of the frequency and a settling time (ngspice 47 manual, §11.3.12). |
| Integration | The trapezoidal rule by default, or Gear's second-order formula or backward Euler, as the transient asks. | SPICE. |
| Linear devices | A resistor or a capacitor without voltage coefficients, an inductor, coupled inductors, a controlled source, a thermal resistance and a thermal capacitance are evaluated once a run, for their conductances and capacitances, which sum once into the matrix every iteration starts from; each iteration takes their currents and charges from those, so the equations are the same, to within rounding, and every iteration evaluates only the devices whose equations are not linear. | jolty's own: ngspice loads every device at every iteration (`cktload.c`, lines 62 to 75). |
| Linear solves | The matrix's rows and columns are ordered once per circuit: each column matched with a row so that the diagonal holds an entry, the matrix split into its block triangular form, so a supply's many entries are never factored, and each block ordered by approximate minimum degree. A factorization pivots on the ordering's diagonal where it holds at least a tenth of its column's largest entry, and on the largest otherwise; the next keeps those pivots while each holds a tenth, and searches afresh from the first that does not; and a Jacobian bitwise unchanged since the last factorization keeps its factors, so a linear circuit's transient factors only when its step or its formula changes. Before it factors, the solver eliminates each unknown whose pivot is proven to be the largest entry of its column, a MOSFET's displacements, by that pivot, and recovers it after each solve: the same solution in exact arithmetic, every unknown still solved for and judged, from a matrix of the other unknowns alone. | KLU (Davis and Palamadai Natarajan, 2010), which Xyce uses; SPICE's own solver keeps its pivots from one iteration to the next too. The elimination is the static condensation of finite elements (Wilson, 1974), which neither ngspice nor Xyce does; ngspice needs no displacement unknowns, as it integrates Meyer's capacitances inside the device, which conserves no charge. The tenth is jolty's own, where KLU and SPICE take a thousandth: on the examples' matrices it fills alike but for one entry of one, and on random circuits' matrices the backward errors of its solutions stay below 10⁻¹⁴, where a thousandth's reach 10⁻⁸. |
| Truncation error | Each step's error is estimated from divided differences of the charges at the points since the last corner, as many as the step's order and two, filtered through the step's own equations, `C (G + αC)⁻¹ α`, so that a component that settles within the step, which its formula damps, is not read as an error, against the larger of a tolerance on the charge, `relative_tolerance` times the larger of its magnitude and `charge_tolerance`, and one on the current it carries, the step's length times `relative_tolerance` times the current plus `current_tolerance`, or `voltage_tolerance` for a flux, whose derivative is a voltage, both times `truncation_tolerance` and the factorial `(p+1)!` that SPICE's estimate leaves out, so that the options mean what SPICE's `reltol`, `chgtol` and `trtol` do. The first step after a corner has no such points before it: it is accepted only with the second, whose three points estimate the curvature both steps share, and taken again shorter from the corner when that shows its own error too large. Gear2's error constant is taken at the step's ratio `ω` to the one before it, `(1 + ω)²/(6ω(1 + 2ω))`, and the longest step after it, or in its place, is solved for at its own ratio, so that a Gear2 step whose length changed is held to its tolerance. | SPICE's `CKTterr`, its tolerances (ngspice 47's `cktterr.c`, lines 37 to 42) and `trtol`; the filtering, Shampine's (1980), as stiff solvers such as RADAU5 filter their estimates (Hairer and Wanner, 1996, IV.8); judging the first step after a corner by the second's points is jolty's own, so that no step but one ending at a corner goes unjudged. Every charge is judged, a MOSFET's bulk junctions' among them, where ngspice's MOSFETs judge only their gate charges (`MOS1trunc`): on the benchmarks' synchronous buck, whose switched node the junctions hold, ngspice takes 56,624 time points to jolty's 103,349 steps tried, and 143,339 when made to judge the junctions too. SPICE takes Gear2's constant-step constant, 2/9, at every step (`cktterr.c`, lines 24 to 31), which after a step cut to a tenth estimates the error 7.6 times too small. SPICE holds a charge to `reltol` times the larger of its magnitude and `chgtol` (lines 40 and 41), as jolty does; it also floors the error constant times the divided difference at `abstol` (line 69), and holds an inductor's flux to `abstol` rather than to a voltage's tolerance (lines 37 and 38); jolty does neither. |
| Step length | At most a fiftieth of the run unless `max_step` says otherwise; the next step, or a step retried for too large an error, is 0.9 of the longest every charge's error allows, by the same estimate: its error grows as the step's power one past the formula's order, against the larger of the charge's tolerance, which holds, and the current's, which grows with the step, so the longest is the longer of the steps meeting each; a step grows by at most twice and shrinks by at most ten times; one whose Newton iteration fails is retried an eighth as long; and no step is shorter than 10⁻¹¹ of the longest. A transient whose results would pass its `max_memory`, 2 GiB by default, before it stops, every unknown's and every recorded device's at each point, fails and says how long its steps were where it stopped, rather than fill the memory. | SPICE, whose step follows the charge's tolerance alone; solving for the longest step against both tolerances is jolty's own, as the current's grows with the step. The shortest step is SPICE's (`traninit.c`, line 36): a charge behind a small resistance settles within picoseconds of a corner, and resolving it to SPICE's charge tolerance can take steps shorter than the 10⁻⁹ of the longest jolty took, which stopped the benchmarks' buck at a relative tolerance of 1e-5. SPICE accepts a step when the step its error allows is longer than 0.9 of it (`dctran.c`, line 801), and takes its longest by default the print step where that is shorter than a fiftieth of the run (`traninit.c`, lines 26 to 34). |
| Corners | Steps land on every corner of every waveform, and on every time asked for; after a corner, the first step is a tenth of the shorter of the step the error allowed before it and the gap to the next corner, and the steps take backward Euler until enough points follow the corner to estimate the error of the transient's own method. | SPICE, whose saved step `CKTsaveDelta` is the step the error allowed before it was cut short to land on the corner; ngspice also halves the gap to a corner that 1.9 steps would pass (`dctran.c`, lines 588 to 598), which on the benchmarks' buck cost 11% more accepted steps and 47% more rejected ones. |
| Jumps | A behavioral source's value jumps where a comparison, `u`, `sgn`, `floor` or `ceil` it reads through the branches it takes changes; the change is located by linear interpolation of its input within a step, a step in which it comes early is retried to land on it, and the integration restarts after it as after a corner. | jolty's own, as hybrid-system simulators locate zero crossings; SPICE steps across such a jump unlocated, and ngspice may stall there. |
| Ringing | Whenever some charge's derivative under the trapezoidal rule turns back and forth twice in a row by more than its tolerance, the next step takes backward Euler. | jolty's own, as simulators that mix the trapezoidal rule with backward Euler do: the trapezoidal rule carries an error in a derivative from step to step undamped. |
| Prediction | Each step's Newton iteration starts on the line through the last two accepted points, carried on to the step's end, and at the last point after a corner, a jump or a logic driver's change. | SPICE's predictor, which extrapolates its devices' junction voltages the same way. |
| Convergence of a time step | A time step's Newton iteration takes at most `max_transient_iterations` steps, and judges the iterate after the last as any other; it measures how fast it contracts, `θ`, each step's largest move of an unknown against that unknown's tolerance over the step before's, between steps from evaluations no device limited, and takes a step as within tolerance once `θ/(1 − θ)` times it is. A first step, a step from an evaluation a device limited or after one, and a step no shorter than the one before are within tolerance when they are themselves, as in SPICE. A time step whose iteration fails is retried shorter from its last iterate rather than from the last solution. | Hairer and Wanner's test for stopping the Newton iteration of an implicit integration step ("Solving Ordinary Differential Equations II", 1996, section IV.8), as RADAU5 makes it: `θ/(1 − θ)` times the last step bounds the distance an iteration contracting by `θ` leaves to the solution (Deuflhard, "Newton Methods for Nonlinear Problems", 2004, section 2.1), more than the step itself once `θ` passes ½, which SPICE's test on the step alone takes as converged, and less below. A limited step is the limiter's rather than Newton's, so no rate is measured across it. Hairer and Wanner also stop an iteration as soon as `θ` reaches 1, or as soon as the steps left at the rate `θ` could not converge, which jolty does not: on the benchmarks' buck and chain of 500 inverters and the boost and astable examples, 91%, 92%, 100% and 60% of the iterations those tests would stop converge within 10 steps, as a full Newton iteration, unlike their simplified one, contracts faster as it nears the solution, and stopping them took the chain 6,103 steps where it takes 2,358, and stopped the buck at 41 µs. SPICE judges each step alone and stops at a count, 100 in ngspice 47 (`niiter.c`, lines 37 to 39). |
| Loop gain | A unit voltage injected in series through the probe and a unit current injected from the ground into its `positive` terminal, both solved with one factorization at each frequency, give `u = v_x − i`, the voltage entering the loop less the current the probe carries on, and the loop gain `T = −u / (1 + u)`. | Middlebrook's double injection, the loop gain of Tian's method and Spectre's `stb`. |
| Logic | Detectors' crossings are located by linear interpolation within a step, and a step that crosses early is retried to land on the crossing; at DC the logic and the analog circuit settle each other at most four times before the detectors still changing read unknown levels. | jolty's own event scheduler, after the connect elements of mixed-signal simulators. |
| Logic loads | Each logic driver's load is the capacitance it drives, read from its small-signal current at 1 Hz at the operating point. | jolty's own. |

When no step converges, however short, the transient fails and names why: the unknowns furthest from converging, or the equation whose truncation error stayed above its tolerance.
Equations singular at every attempt fail the operating point, and an analysis whose linearized circuit is singular at a frequency fails too, each naming up to three of the unknowns the equations leave undetermined: the currents of two voltage sources in parallel, of which only the sum is determined, or the voltage of a node only a capacitor joins to the circuit.

The engine's results agree with ngspice's on a corpus of reference circuits: operating points, sweeps and frequency responses to within a millionth of each signal, and transients to within a thousandth of their swing.
Random circuits agree too, of every device and up to a hundred unknowns: each operating point and frequency response within a millionth of the circuit's largest voltage or current, and each transient within a thousandth, jolty stepping onto every time ngspice took.
Where the two engines rest in different states of a circuit with several, ngspice started from jolty's finds jolty's.
Where two transients differ, both run again with steps a quarter as long, then a sixteenth and a sixty-fourth, and agree then if one had not settled; one that jolty's own shorter steps still move, as a chaotic one's, is no reference for either engine.
Where jolty stops short of ngspice, the same circuit with 10 fF from every net to the ground, as real nets have, must finish: without capacitance, the equations of a net that only resistive devices hold can fold into a point past which no solution continues, which ngspice may leap past.
An operating point ngspice reaches only by its last resort, the end of a transient of fixed length that it does not check to be one, is no reference either.
The same random circuits, written differently, simulate the same: with their nets renamed, their instances reordered, a resistor split in two in series, or a source of 0 V in series with a terminal, which reads the current into it.
Each integration method converges at its order: halving the step halves backward Euler's error and quarters the trapezoidal rule's and Gear's, against a low-pass's closed form and after a diode clamps.
