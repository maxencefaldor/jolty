# Simulation

A plan defines the analyses to run on a circuit, shared numerical options, and which instances to record.
Plans stay independent of circuits, allowing the same circuit to run under any plan.

## Plans

A build script can define `plan()`, returning a `Plan` mapping analysis names to analysis objects:

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

Each analysis name must be a Python identifier identifying its result entry.
A plan's `fidelity` mapping can substitute characterized logic cells for chosen component instances (see [Fidelity](fidelity.md)).
`jolty simulate` loads `plan()` from the build script unless `--plan` points to a separate script.

## Analyses

| Analysis | Finds |
| --- | --- |
| `OperatingPoint()` | DC solution with every source evaluated at time zero |
| `OperatingPoints(max_solutions=8)` | All discovered DC solutions up to `max_solutions`, starting with the primary operating point |
| `DcSweep(source, values)` | DC solution at each stepped value of one independent source or device parameter |
| `Ac(source, frequencies)` | Small-signal frequency response to a unit excitation of one source around the operating point |
| `Transient(stop, max_step=None, method="trapezoidal", breakpoints=(), initial_conditions={}, operating_point=True)` | Time-domain response until `stop`, starting from the operating point or [initial conditions](#initial-conditions-and-nodesets) |
| `PeriodicSteadyState(period=None, max_step=None, method="trapezoidal", breakpoints=())` | One period of a repeating [periodic steady state](#periodic-steady-state) for a driven circuit or autonomous oscillator |
| `Sensitivity(outputs, of=None)` | Derivatives of outputs with respect to all numeric device parameters (except [device temperature](devices.md)) at DC or across analysis `of` |
| `Mismatch(outputs, of=None, random_variables=())` | First-order [standard deviation](#mismatch) and variance shares from circuit tolerances and supplied random variables (such as PDK statistics) |
| `Stability(probe, frequencies)` | Loop gain across a probe inserted in series with a feedback loop at each frequency |
| `Noise(output, frequencies, reference=None, source=None)` | Output noise spectral density from all device noise sources, optionally referred to an input source |
| `PoleZero(source=None, output=None, reference=None)` | Linearized operating-point poles and optional transfer-function zeros from a source to an output |
| `TransferFunction(source, output, reference=None)` | SPICE `.tf` DC gain plus input resistance at the source and output resistance at the output |

Sources are addressed by hierarchical path, such as `V1` or `bias/I1`.
Transfer function analysis linearizes the circuit at the operating point and executes two solves: excited at `source` for gain and input resistance, then excited at `output` (unit net current or series unit voltage) for output resistance.
During the second solve, independent voltage sources act as shorts and current sources as opens, matching SPICE.
Any port unable to carry current (such as an unloaded voltage source) reports infinite resistance rather than ngspice's `10²⁰` Ω.
`linear(start, stop, points)` generates evenly spaced values, and `decade(start, stop, per_decade)` generates logarithmically spaced frequencies.
Transient integration selects `"trapezoidal"` (default), `"gear2"` (for stiff ringing circuits), or `"backward_euler"` (maximum damping).
Default `max_step` equals one fiftieth of `stop`.
Transients step exactly onto every time in `breakpoints` without restarting integration, giving solved values for every unknown and recorded terminal at those instants.

A `Simulation` caches the last operating point and reuses it across small-signal analyses, matching DC sweeps, and operating-point checks whenever DC-relevant options stay unchanged (ignoring `charge_tolerance`, `max_transient_iterations`, `truncation_tolerance`, `max_memory`, and `profile`).
If only tolerances or `max_dc_iterations` change, Newton iteration resumes from the prior iterate because starting from zero yields identical steps.

## Initial conditions and nodesets

Transient `initial_conditions` specify starting net voltages by path (SPICE `.ic`), clamping each net with an ideal source during the initial operating-point solve before releasing it at `t = 0`.
Setting `operating_point=False` (SPICE `UIC`) skips the operating point, initializes all remaining unknowns (including inductor currents) to zero, and starts from the stated voltages using backward Euler on the first step (reporting no operating-point strategy).
Option `nodesets` (SPICE `.nodeset`) temporarily clamps nets during an initial solve from zero and then releases them for a final unconstrained solve.
This guides multistable circuits like latches toward a desired branch, falling back to zero if the clamped solve fails.
Nets already constrained by voltage sources or inductors (or by an earlier named net) are not double-clamped; conflicting initial conditions fail while reporting both values.

```python
from jolty.plan import Options, Plan, Transient
from jolty.units import V, us

Plan(
    {"from_half": Transient(stop=10 * us, initial_conditions={"out": 0.5 * V})},
    options=Options(nodesets={"latch/q": 3.3 * V}),
)
```

## Periodic steady state

Circuits driven by periodic sources (such as rectifiers, mixers, or power converters) settle into a periodic steady state that time-domain transients reach only after slow time constants decay.
`PeriodicSteadyState` computes one steady-state period directly, analogous to ngspice `.pss`, Spectre `pss`, and Xyce `.HB`:

```python
Plan({"steady": PeriodicSteadyState(period=1 * us, max_step=5 * ns)})
```

Shooting solves `φ(x₀) = x₀` by Newton iteration over the period map `φ` (Aprille and Trick, *Proceedings of the IEEE* 60(1), 1972).
Each period propagates sensitivities along every step to assemble monodromy matrix `Φ = ∂φ/∂x₀` and updates `x₀` by solving `(Φ - I) Δ = x₀ - φ(x₀)`.
Shooting starts from the operating point and stops when end-of-period mismatch satisfies tolerances across all unknowns, or fails after 40 periods while naming the worst-mismatching unknowns.
Nearly linear circuits converge in two or three periods regardless of settling time, where transients require tens or hundreds of cycles.
`PeriodicSteadyStateResult` exposes one period of waveforms from `0` to `period` (compatible with measures, plots, and the 3D viewer) plus `period` and shooting iteration count `periods`.
Step lengths respect `max_step` (default `period / 50`) and land on all `breakpoints` within the period.

Some circuits contain neutral directions (conserved quantities such as an unexcited memristor state or floating series capacitor-memristor charge), making `Φ - I` singular with a continuum of steady states.
Shooting computes the minimum-norm least-squares step inside the range of `Φ - I`, preserving the starting state along neutral directions (reducing to standard Newton when `Φ - I` is regular).
The linear system is scaled by unknown tolerances, treating a direction as neutral when its singular value lies within numerical rounding or below `relative_tolerance` with the starting state already repeating within tolerance.
Monodromy and dense SVD factorizations use up to 14 floats per squared unknown, so circuits exceeding `max_memory` fail immediately before simulating a period, leaving remaining memory for period waveforms.
Circuits containing event-driven logic cannot run periodic steady state because shooting does not carry discrete logic state.

### Oscillators

Omitting `period` configures autonomous shooting for free-running oscillators (ring, LC, or relaxation oscillators with constant sources), solving simultaneously for state `x₀` and unknown period `T`:

```python
Plan({"ring": PeriodicSteadyState()})
```

Any phase shift along the orbit satisfies `φ(x₀, T) = x₀`.
An orthogonal phase condition `ẋᵀ Δx = 0` (scaled by unknown tolerances) constrains state updates perpendicular to the orbit (Seydel, *Practical Bifurcation and Stability Analysis*, 3rd ed., 2010, §7.1).
Each period propagates `∂φ/∂T` alongside `Φ` by scaling step charges by `1/T` at fixed period fractions, keeping step derivatives exact.
Newton solves the bordered system `[[Φ - I, ∂φ/∂T], [∂φ/∂Tᵀ, 0]] [Δx, ΔT] = [x₀ - φ(x₀, T), 0]` in tolerance-scaled coordinates while preserving any neutral directions.

Since an oscillator's DC operating point is an unstable equilibrium, autonomous shooting initializes on the limit cycle using a startup transient.
Startup perturbs the operating point along the fastest-growing [pole-zero](#poles-and-zeros) mode by one unit of tolerance in the dominant unknown.
The transient runs in doubling time stretches (starting from one natural period), tracking returns to the hyperplane orthogonal to the perturbation until consecutive returns stabilize above tolerance.
Each stretch restarts integration with backward Euler, so returns compare only within the same stretch using `max_step` capped at 1/50 of the observed period.
Shooting launches from that converged return and typically finishes in three or four periods.
Autonomous shooting fails explicitly if no growing mode exists (such as an ideal lossless tank), if shooting collapses onto the DC equilibrium, or if 40 stretches fail to stabilize (such as a latch).
Sources must stay constant throughout an autonomous period; pulsed kick sources belong only in transient startups.
On an exact-period `2π√(LC)` nonlinear tank, shooting matches the exact period within trapezoidal phase error `(ωh)²/12` and amplitude within backward-Euler startup damping (`crates/jolty-engine/tests/periodic.rs`).
On the five-stage ring oscillator example, shooting matches a settled transient within `2.8e-8` at `reltol = 1e-9` and `max_step = 1` ps (within transient scatter of `5.9e-8`).
At default tolerances and `5` ps steps, agreement stays within `2.9e-5` (`tests/test_periodic.py`, Apple M2 Max).

## Several DC solutions

Feedback circuits can possess multiple DC operating points: bistable latches hold two stable states plus a metastable balance point, and self-biased references often admit a zero-current state alongside the active operating point.
`OperatingPoints` finds the primary operating point first and then discovers additional solutions via deflation:

```python
Plan({"states": OperatingPoints(max_solutions=8)})
```

Deflation modifies the residual to `m(x)·f(x) = 0` with `m(x) = Π (1/‖x - r‖² + 1)` over converged voltage vectors `r`.
Each Newton step scales by `1 / (1 - ∇m·δ/m)` to repel previously found roots (Farrell, Birkisson, and Funke, *SIAM Journal on Scientific Computing* 37(4), 2015).
Each search attempts Newton, node/junction gmin stepping, source stepping, and pseudo-transient continuation from zero, followed by reflections across prior solutions and alternating ±0.1 V perturbations to break latch symmetry.
Search stops at `max_solutions` or once no candidate yields a new root separated by at least ten tolerances in some unknown.
`OperatingPointsResult.points` lists every `OperatingPointResult`, `voltages(net)` returns net voltages across roots, reports print every solution, and `specs.solutions` returns the solution count.

## Sensitivities

Every primary analysis (operating point, DC sweep, AC, stability, transient, and periodic steady state) is differentiable with respect to circuit parameters.
`jolty.function(circuit, analysis)` wraps an elaborated circuit as callable `f(parameters)`, and `jolty.jvp` and `jolty.vjp` differentiate `f` with the same calling conventions as `jax.jvp` and `jax.vjp`.
`parameters` is a nested dictionary indexed by instance path then parameter path in SI base units; only listed parameters are differentiated while all other circuit parameters remain constant.
Parameter paths use dot-separated keys (such as `resistance` or `waveform.dc`) and 0-based list indices (such as `waveform.piecewise.1.1`).
Outputs form a dictionary mapping each unknown name to its scalar or array values (complex phasors for AC, complex loop gain under `jolty.simulate.LOOP_GAIN` for stability), adding oscillator period in seconds under `jolty.simulate.PERIOD`.

`jvp(function, parameters, tangents)` computes outputs and forward directional derivatives along `tangents` via the direct method (one linear solve per analysis point).
`vjp(function, parameters)` computes outputs and a `pullback` function mapping output cotangents (defaulting omitted entries to zero) back to parameter gradients via one transposed solve per point regardless of parameter count.
Scalar loss derivatives form a cotangent whose pullback yields exact gradients across all parameters at once, illustrated below on a resistive crossbar:

```python
from jolty import Net, component, devices, function, vjp
from jolty.units import V, kohm


@component
def crossbar(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    for row, level in enumerate((0.3 * V, -0.2 * V)):
        source = c.place(devices.voltage_source(level), name=f"V{row}")
        c.connect(source.positive, name=f"in{row}")
        c.connect(source.negative, ground)
    for column in range(2):
        load = c.place(devices.resistor(1 * kohm), name=f"RL{column}")
        c.connect(load.a, name=f"out{column}")
        c.connect(load.b, ground)
        for row in range(2):
            weight = c.place(devices.resistor(1 * kohm), name=f"W{row}{column}")
            c.connect(weight.a, Net(f"in{row}"))
            c.connect(weight.b, Net(f"out{column}"))


weights = {f"W{row}{column}": {"resistance": 1e3} for row in "01" for column in "01"}
targets = {"out0": 0.05, "out1": -0.02}
outputs, pullback = vjp(function(crossbar(), OperatingPoint()), weights)
distances = {name: outputs[name] - target for name, target in targets.items()}
gradient = pullback({name: 2 * distance for name, distance in distances.items()})
```

For this squared-error loss, passing `2 * distance` as the cotangent yields `gradient["W00"]["resistance"]` (and all other weight gradients) in a single transposed solve.
In AC and stability analyses, cotangents and gradients are complex (`vᵀ dx/dp`); differentiating a real loss over phasors `a + jb` takes the real part of the pullback applied to `∂L/∂a - j ∂L/∂b`.

`f.batch(names, values)` evaluates rows of `values` across engine threads in parallel, and `f.batch_vjp(names, values)` returns batched outputs plus a batched pullback.
With optional extra `jolty[jax]` (`pip install 'jolty[jax]'`), `jolty.jax.function` exposes the map as a `jax.custom_vjp` compatible with `jax.jit`, `jax.grad`, `jax.vjp`, `jax.remat`, and `jax.vmap` (running one forward batch and one backward batch regardless of `vmap` nesting):

```python
import jax
import jax.numpy as jnp

import jolty.jax

simulate_jax = jolty.jax.function(crossbar(), OperatingPoint())


def loss(parameters: dict[str, dict[str, jax.Array]]) -> jax.Array:
    solved = simulate_jax(parameters)
    return sum(
        (jnp.sum((solved[name] - target) ** 2) for name, target in targets.items()),
        jnp.asarray(0.0),
    )


jax_gradient = jax.jit(jax.grad(loss))(
    {
        path: {name: jnp.asarray(value) for name, value in inner.items()}
        for path, inner in weights.items()
    }
)
```

Batched transients and driven periodic steady states require explicit `breakpoints` so every row returns a fixed shape; autonomous oscillators expose only `jolty.jax.PERIOD`.
Mapped calls match unmapped batch rows bit for bit; standalone `jolty.function` calls may dissect large matrices while batches default to minimum-degree ordering, causing rounding-level trailing-digit differences.
Forward passes cache up to `saved_pullbacks` (default 16) pullbacks so `jax.grad` and `jax.value_and_grad` simulate once; evicted pullbacks recompute identically on demand.
Only first-order reverse-mode differentiation is supported (`jax.jvp`, second-order `jax.grad`, and `jax.hessian` raise errors).

Forward and reverse derivatives are exact transposes of one linear computation.
Parameter partial derivatives `∂F/∂p` evaluate via dual-number device twins propagating up to eight directions per pass through identical equations including temperature scaling.
VJP packs up to eight device parameters per twin, while JVP seeds each tangent direction once per affected device regardless of how many parameters that tangent moves.
Chunked twin directions match single-direction evaluations bit for bit.
The implicit function theorem `dx/dp = -J⁻¹ ∂F/∂p` (`J = ∂F/∂x`) reuses one factorization per point for all tangents and cotangents, satisfying `⟨v, J t⟩ = ⟨Jᵀ v, t⟩` to `7.4e-16` across all analyses (`crates/jolty-engine/tests/sensitivity.rs`).
At DC, `J` is the DC Jacobian, and swept parameters have zero derivative at fixed sweep points.
In AC, every nonlinear kernel evaluates seeded dual unknowns so JVP pushes exact second derivatives `(∂G/∂x · dx₀/dp, ∂C/∂x · dx₀/dp)` directly.
AC VJP accumulates frequency weights over nonlinear matrix slots and executes one transposed DC solve per active cotangent regardless of parameter count, skipping the DC solve on purely linear circuits.
Stability JVP and VJP differentiate Middlebrook's double-injection loop gain `T = -u / (1 + u)` (`dT/du = -1 / (1 + u)²`) from the same factored `G + jωC` matrix, solving two unit transposed injections per active frequency and sharing the operating-point bias-coupling path with AC.
In transients, JVP propagates sensitivities step by step using the active integration formula (Hocevar, Yang, Trick, and Epler, 1985), while VJP executes the exact discrete adjoint backward across all steps (holding step sizes and logic events fixed).
Transient derivatives start from the initial-condition-constrained DC Jacobian (or zero under `UIC`), matching closed form `d out/dR = -(t/(R²C)) e^(-t/RC)` on an RC started at `out = 0` V under either mode.
Time-domain VJP replays steps post-run and therefore rejects logic circuits (event drivers cannot replay statelessly), and both JVP and VJP reject transmission lines (history terms are omitted from derivatives).
For [periodic steady state](#periodic-steady-state), one pass over the converged period computes `∂φ/∂p`, and the monodromy matrix (or bordered oscillator system) solves for `dx₀/dp` and `dT/dp` while keeping neutral directions fixed.
A second pass propagates sensitivities at fixed period fractions (scaling charges by `-(dT/dp)/T`), while VJP executes the exact three-stage transpose backward.
Bordered-system singular values are zeroed only within numerical rounding.
On an exact-period LC tank, period derivatives satisfy `L ∂T/∂L + C ∂T/∂C = T` within `256 ε` at `reltol = 1e-10` (`7e-10` at default `1e-3`) and match central differences within `1.2e-7` (`crates/jolty-engine/tests/periodic.rs`).
Discrete integer mode selectors (such as MOSFET `level`) are not continuous parameters and carry no derivative.

`Sensitivity` computes the full Jacobian of selected outputs against every numeric parameter in the circuit:

```python
Plan({"sensitivity": Sensitivity(["out", "V1:current"])})
```

Passing `of=` differentiates outputs at every point of a DC sweep, AC sweep, stability analysis (yielding complex derivatives, defaulting `outputs` to `["loop_gain"]`), transient, or periodic steady state:

```python
Plan({"drift": Sensitivity(["out"], of=Transient(stop=1 * ms))})
```

`Sensitivity` automatically chooses VJP or JVP to minimize total linear solves.
At DC and AC points, `k` outputs require `k` transposed solves versus `P` forward solves for `P` parameters, selecting VJP whenever `k ≤ P`.
In transients and periodic steady states, `N + 1` points cost `k (N + 1)(N + 2) / 2` backward solves versus `P (N + 1)` forward solves.
`Sensitivity` selects JVP upfront when cheaper at minimum step count (or when logic is present) and chooses dynamically after step recording otherwise.

`SensitivityResult.derivative(output, instance, parameter)` returns one SI derivative (scalar at DC, array otherwise), and `gradient(output)` returns all derivatives ordered by `parameters`.
`SensitivityResult.result` holds the differentiated base analysis, while autonomous oscillators additionally expose `period_derivative(instance, parameter)` and `period_derivatives`:

```python
Plan({"tuning": Sensitivity(["out"], of=PeriodicSteadyState())})
```

## Stability

Stability analysis computes open-loop gain around a feedback loop to determine phase and gain margins.
Insert a probe (typically a 0 V voltage source) in series with the loop with `positive` facing the downstream input (such as between an amplifier output and its feedback network):

<!-- not run: a fragment of a component that places `amplifier` and `feedback` -->

```python
probe = c.place(devices.voltage_source(0 * V), name="VP")
c.connect(amplifier.output, probe.negative)
c.connect(probe.positive, feedback.a)
```

Then configure a `Stability` analysis targeting the probe:

```python
Plan({"loop": Stability("VP", decade(1 * Hz, 1 * GHz, 20))})
```

At each frequency around the linearized operating point, the engine injects a unit series voltage across the probe and a unit shunt current from ground into `positive`.
Combining both responses via Middlebrook's double-injection formula yields the exact loop gain `T` regardless of bidirectional loading, matching Tian's method and Spectre `stb` (Middlebrook, *International Journal of Electronics* 38(4), 1975).
Injected current returns through ground, so neither probe terminal may connect to ground.
Negative feedback produces positive low-frequency `T` and attenuates closed-loop sensitivity by `1 + T`.

`StabilityResult.gains` holds complex `T` across `frequencies`, and `Stability` composes directly with `Sensitivity`, `Mismatch`, `jolty.function`, and `jolty.jax.function` under output `"loop_gain"`.
Reports list low-frequency gain, unity-gain crossover frequency, phase margin, and gain margin at -180°; specs expose matching `crossover`, `phase_margin`, `gain_margin`, `gain`, and `phase` measures.

## Noise

Noise analysis computes output noise spectral density at every frequency from all active device noise sources:

```python
Plan({"hiss": Noise("out", decade(10 * Hz, 100 * kHz, 10), source="V1")})
```

The output specifies a net voltage (optionally differential with `reference`) or a branch current such as `V1:current`.
Noise sources follow SPICE equations: resistor thermal noise `4kT/R`, junction shot noise `2qI`, FET channel thermal noise `8kT·gm/3`, and flicker noise `KF·|I|^AF/f` (using base current for BJTs or gate-area-scaled drain current for MOSFETs; see [Devices](devices.md)).
Resistances below 1 Ω stamp thermal noise as a series voltage `4kTR`, so zero-ohm shorts generate zero noise.
All noise frequencies must be strictly positive because flicker noise diverges at DC.

At each frequency around the operating point, the engine solves the transposed linearized system once with the output vector on the right-hand side.
One adjoint solve gives every device's transfer function to the output, summing independent source power densities directly.
When an independent input `source` is provided, the same solve gives source-to-output gain, dividing output density by squared gain for SPICE-compatible input-referred noise.

`NoiseResult.densities` holds output power spectral density (V²/Hz or A²/Hz), and `contributions` lists each source's contribution (`thermal`, `collector shot`, etc.).
Helper methods provide `referred()`, `contribution(instance, name=None)`, `density(at)`, and band-integrated `rms(start=None, stop=None)` using SPICE power-law log-log integration.
Reports summarize band-edge densities, output and input-referred RMS noise, and top contributors; plots graph densities on log axes, and specs provide `noise_density` and `integrated_noise`.

## Poles and zeros

Pole-zero analysis computes the linearized circuit's natural frequencies (poles) and optional transfer-function zeros between a `source` and `output`:

```python
Plan({"pz": PoleZero(source="V1", output="out")})
```

Around the operating point, small-signal equations read `(G + sC) x = -u·b` for complex frequency `s` and source amplitude `u`.
Poles are finite generalized eigenvalues of pencil `(G, -C)`, while zeros are finite eigenvalues of the pencil bordered by the source excitation and output selector.
The engine equilibrates and frequency-scales dense pencils and solves them with a complex QZ algorithm (Moler and Stewart, 1973, matching LAPACK `ZGGHRD`/`ZHGEQZ`).
Eigenvalues above `10¹²` times the circuit frequency scale are treated as infinite, and zeros within `10⁻¹⁰` of the largest finite root clamp to the origin (such as coupling-capacitor zeros).
The circuit possesses a nonsingular operating point, so no pole lies at the origin.
Dense pencils require up to eight square matrices of unknowns (plus one row/column for zeros) and fail upfront if they exceed `max_memory`.
Uncontrollable or unobservable natural modes appear as coinciding pole-zero pairs, matching ngspice.

`PoleZeroResult.poles` and `zeros` store complex angular frequencies (rad/s) sorted by magnitude (upper conjugate first).
`natural_frequencies(roots)` returns `|s| / (2π)` in hertz, and `dampings(roots)` returns `-Re(s) / |s|` (1 for negative real roots, 0 to 1 for damped complex pairs, negative for unstable right-half-plane poles).
Reports list each unique root and conjugate pair, plots graph roots by frequency and damping, and specs provide `damping` and `dominant_pole`.

## Temperature

`Plan.temperature` sets the ambient simulation temperature for all devices (default 27 °C):

```python
Plan({"op": OperatingPoint()}, temperature=celsius(85))
```

`celsius(degrees)` converts degrees Celsius into kelvins.
Device parameters scale away from nominal temperature as described in [Devices](devices.md).

## Corners

Plan `corners` re-run analyses under alternate temperatures, foundry [process corners](foundry.md#process-corners), or keyword arguments passed to `build()` to reconstruct the circuit:

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

Corner names must be Python identifiers and inherit the plan's temperature and process corner unless overridden.
Temperature-only corners execute together as one [batch](#batches) on the shared elaborated circuit, while corners modifying `build()` arguments run in a separate batch.
Reports, results archives, and plots store nominal results alongside per-corner elaboration status and analysis failures.
[Specs](specs.md) evaluate across every corner.

## Monte Carlo

Devices attach parameter tolerances via passive `tolerance=0.01` (uniform ±1%) or `device.toleranced(parameter, relative, distribution)` supporting `"uniform"` and `"normal"` (3σ = `relative`).
Parameter paths address nested fields such as `devices.voltage_source(5 * V).toleranced("waveform.dc", 0.05)` for a ±5% supply.
Setting `monte_carlo=MonteCarlo(samples, seed=0)` runs plan analyses across `samples` random draws at nominal conditions and every corner:

```python
def plan() -> Plan:
    return Plan({"op": OperatingPoint()}, monte_carlo=MonteCarlo(1000, seed=1))
```

Each sample draws every toleranced instance parameter independently (so repeated component placements vary independently), deterministically determined by `seed`.
Exponential parameters use multiplicative spreads: `"log_uniform"` keeps saturation current within `1.08×` (`±2` mV on `Vbe`), and `"log_normal"` sets that factor at 3σ.
Correlated parts (such as on-die resistors) share an `ir.Lot(name="die", relative=0.2)`, multiplying each part's individual draw by one shared per-sample lot draw regardless of hierarchy location.
Each sample reuses identical part draws across corners sharing the same toleranced parameters, taking each requirement's worst corner verdict.
Circuits with foundry devices additionally draw PDK [mismatch](foundry.md#mismatch) per sample.
Samples execute as a parallel [batch](#batches) per condition on a shared elaborated circuit, seeded solely by `(seed, sample_index)` for bit-identical outputs across any thread count or [shard](machines.md) split.
Reports count elaboration and analysis failures alongside requirement distributions and yield ([Specs](specs.md)).
Plots render nominal-condition requirement histograms and trace envelopes ([Results](results.md#plots)).

## Worst case

Setting `worst_case=WorstCase(step=0.1)` evaluates every [spec](specs.md) requirement at the tolerance extremes that degrade it most, matching PSpice worst-case analysis:

```python
def plan() -> Plan:
    return Plan({"op": OperatingPoint()}, worst_case=WorstCase())
```

At nominal conditions and every corner, one probe run per toleranced parameter (perturbed by `step` times tolerance) determines the direction in which each requirement moves.
For each requirement bound, every independent parameter moves to its worst extreme and each shared lot moves to the net worst extreme of its members (leaving canceling matched pairs nominal), deduplicating identical extreme runs.
Normal extremes sit at ±3σ and logarithmic extremes at the specified factor.
Probe runs and extreme runs execute as explicit [batches](#batches) per circuit.
Extremes extrapolate from first-order slopes as in PSpice, so strongly nonlinear optima may lie inside the tolerance box; use Monte Carlo to probe interior points.
Reports list run counts and worst verdicts across bounds and conditions, and `jolty simulate` exits with 1 if any worst-case check fails.

## Mismatch

Mismatch analysis computes first-order output standard deviations and per-source variance shares in a single pass, matching Spectre `dcmatch` and `acmatch`:

```python
def plan() -> Plan:
    return Plan({"spread": Mismatch(["out"])})
```

An `ir.RandomVariable` defines a stochastic draw by mean, standard deviation, and affected device-parameter derivatives.
Each toleranced parameter generates a zero-mean variable with standard deviation `relative / √3` (uniform) or `relative / 3` (normal) scaled by nominal value, or `1/√3` and `1/3` scaled by `value · ln(1 + relative)` for logarithmic tolerances.
Each shared lot generates one variable per parameter name, shifting all member instances together.
`random_variables` appends external sources such as PDK statistical draws from [`Library.random_variables`](foundry.md#the-open-process-kits).
Output derivatives with respect to each random variable combine [sensitivity](#sensitivities) gradients with parameter derivatives at the active run point (nominal, corner, or variant).
Random variables are independent, so total standard deviation equals the root-sum-square of individual contributions (`derivative × σ`) and each variance share equals its squared fraction.
For AC phasors, magnitude standard deviation projects derivatives along the phasor while phase standard deviation (in radians) projects orthogonally across it.

`MismatchResult` provides `standard_deviation(output, part=None)`, `contributions(output, part=None)`, and `shares(output, part=None)` in `random_variables` order (passing `part="magnitude"` or `"phase"` for AC).
Autonomous periodic steady states also expose `period_derivatives` and `period_standard_deviation()`.
Reports display peak standard deviations (phase in degrees, plus oscillator period deviation) and top variance contributors sorted largest first.
First-order mismatch matches Monte Carlo within sampling uncertainty across differential pairs, current mirrors, and PDK stages (`tests/test_mismatch.py`).

## Variants

`Plan.variants` executes analyses across deterministic parameter and temperature sweeps at nominal conditions and every corner:

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

Each variant generator defines per-run values as a pure function of run index:

- `Explicit([Variant(overrides, temperature), ...])`: explicit list of per-run parameter dictionaries (matching `Simulation` overrides) and temperatures;
- `Grid([Axis(instance, parameter, values), Axis.temperature(values), ...], warm=False)`: Cartesian grid varying the last axis fastest, optionally chaining converged operating points along inner lines when `warm=True`;
- `MonteCarlo(samples, seed)`: draws random tolerance [samples](#monte-carlo);
- `Product(first, second)`: combines every run of `second` with every run of `first` (`second` taking precedence on overlap), such as `Product(grid, MonteCarlo(100))`.

Reports summarize run counts, varied parameters, failure counts, requirement spreads, and yield ([Specs](specs.md)); plots render nominal-condition distributions matching Monte Carlo output.

## Batches

All multi-run plans execute as parallel batches inside one engine call, keeping each run a pure function of its index so outputs stay bit-identical across thread counts.
Python callers invoke batches directly through `Simulation.batch`:

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
failed = found.failed  # failure flags; found.failure(run) returns the failure message
responses = found.traces[1].values_of("out")  # a row per run, a column per frequency
```

`BatchResult` stores columnar NumPy arrays populated directly by the Rust sink: parameter values, analysis statuses, [measure](specs.md#measures) values and unmeasured reasons, and traced unknowns (evaluated at `breakpoints` for transients).
Built-in `jolty.specs` measures evaluate inside Rust without Python overhead; custom Python measures and `each` callbacks run under the GIL as runs finish.
Every batch row matches a standalone simulation bit for bit.
Passing `runs=range(start, stop)` evaluates a slice starting at row 0; `BatchResult.joined` reassembles contiguous slices, and `rows` slices an existing batch.
Deduplicated failure reasons preserve first-seen row order so sliced and joined batches match full batches identically.
Parameter values that alter matrix structure (such as a resistor crossing 1 Ω) automatically share one cached elaboration per structure.
Out-of-range parameters or failed analyses record their failure message without halting the remaining runs.
Batch concurrency follows the `threads` setting or the `jolty.simulate.threads(count)` context manager.
Foundry circuits execute batch runs concurrently in ngspice up to the configured thread limit.

## Options

`Options` configures numerical tolerances (defaulting to SPICE values), iteration caps, memory limits, and parallelism; device recording stays on `Plan` or individual analyses.

| Option | Default | Meaning |
| --- | --- | --- |
| `relative_tolerance` | 1e-3 | Relative accuracy of every unknown and residual |
| `voltage_tolerance` | 1e-6 V | Absolute accuracy of voltages |
| `current_tolerance` | 1e-12 A | Absolute accuracy of currents |
| `charge_tolerance` | 1e-14 C | Charge floor matching SPICE `chgtol`: holds charge `q` to `relative_tolerance · max(|q|, charge_tolerance)` |
| `gmin` | 1e-12 S | Minimum conductance placed across every semiconductor junction |
| `max_dc_iterations` | 100 | Maximum Newton iterations per DC operating-point or sweep solve |
| `max_transient_iterations` | 10 | Maximum Newton iterations per time step (SPICE `itl4`, judging the final iterate; ngspice 47 forces at least 100 in `niiter.c`, lines 37 to 39) |
| `truncation_tolerance` | 7 | Scaling multiplier for local truncation error bounds, matching SPICE `TRTOL` |
| `max_memory` | 2 GiB | Memory budget capping transient waveforms and dense pole-zero or periodic matrices before allocation |
| `profile` | off | Records per-phase transient timings in `statistics.profile` (roughly 10% overhead on tiny circuits; zero numerical change) |
| `parallelism` | every thread | `Parallelism` work-distribution policy with measured break-even thresholds (only `max_unknowns_per_interior` shifts rounding) |
| `nodesets` | none | Initial net voltage hints by path applied during operating-point startup then [released](#initial-conditions-and-nodesets) (SPICE `.nodeset`) |

`Parallelism` distributes each work category across engine threads in tasks of at least its threshold once total work fills one task per thread, running smaller workloads serially to avoid synchronization overhead.
Omitted thresholds default to measured break-evens for the active thread count:

| Setting | Default | Work distributed |
| --- | --- | --- |
| `max_threads` | none | Maximum engine threads used per analysis (all threads by default; `1` forces serial execution) |
| `min_solves_per_task` | break-even | AC, noise, and stability frequencies plus sensitivity right-hand sides |
| `min_instances_per_task` | break-even | Nonlinear device kernel evaluations per model group (linear devices evaluate once per run and do not count) |
| `min_unknowns_per_task` | break-even | Newton matrix assembly, convergence checks, updates, and transient step control in work-stolen 1,024-element chunks |
| `max_unknowns_per_interior` | break-even | Nested-dissection interior size limit for parallel sparse LU factorization on blocks where dissection reduces arithmetic |

```python
from jolty.plan import Options, Parallelism

options = Options(parallelism=Parallelism(min_solves_per_task=16))
serial = Options(parallelism=Parallelism(max_threads=1))
```

Batches run one circuit per thread whenever run count meets or exceeds thread count, so per-run parallelism mainly accelerates single runs or small batches.
`jolty simulate` accepts matching CLI overrides such as `--min-solves-per-task`.

Nested dissection activates strictly by cost: the engine counts symbolic elimination multiply-adds for minimum-degree versus nested dissection on the symmetrized block pattern and dissects only when nested dissection costs less.
At `max_unknowns_per_interior = 157`, nested dissection costs `1.06×` minimum-degree arithmetic on a 60 × 60 grid, `0.96×` on 70 × 70, and `0.70×` on 300 × 300, versus `3.9×` on a 5,000-node inverter chain (which stays undissected).
Ordering depends solely on circuit structure and `max_unknowns_per_interior`, so fixed thread counts yield bit-identical results while differing interior limits shift trailing digits within factorization rounding.
For example, `rc_mesh_100` (10,000 nodes) takes the same 109 steps over 10 µs on 1, 2, and 4 threads of an Apple M2 Max with voltages agreeing within `2.1e-15` volts.
Batch runs default to minimum-degree ordering unless `max_unknowns_per_interior` is set explicitly so batches stay bit-identical across thread counts.

Default break-evens were measured via `scripts/bench --break-even` on `cedar` (32-core, 64-thread AMD Ryzen Threadripper 3970X, 16 November 2026, load 1.4 to 37).
Benchmarks covered `bjt_amp` frequency sweeps, 1 to 1,000-stage inverter chains, and 10 × 10 to 300 × 300 RC meshes (see [break-evens](https://github.com/maxencefaldor/jolty/blob/main/benchmarks/speed/README.md#break-evens)).
Each run selects the break-even at or below its thread limit, disabling parallelism for categories with `none`; `jolty.plan.BREAK_EVENS` exposes these defaults:

| Threshold | 2 threads | 4 threads | 8 threads | 64 threads |
| --- | --- | --- | --- | --- |
| `min_solves_per_task` | 4,999 | 11 | 11 | 4 |
| `min_instances_per_task` | none | 250 | 125 | none |
| `min_unknowns_per_task` | none | none | none | none |
| `max_unknowns_per_interior` | 5,001 | 2,501 | 1,251 | 157 |

Each break-even was measured with all other categories serial and alternating run rounds (see [break-evens](https://github.com/maxencefaldor/jolty/blob/main/benchmarks/speed/README.md#break-evens)).
Device evaluation sped up at 250 MOSFETs/task on 4 threads and 125 on 8 threads, while failing to beat serial on 2 or 64 threads up to 1,000 inverters.
On a 100 × 100 mesh, thread-matched dissection ran 21% faster on 2 threads, 29% on 4, 41% on 8, and 20% on 64, whereas 64 threads slowed 51% under 5,001-node interiors.
Unknown-level parallelism lost 8.0% to 39.9% on inverter chains up to 2,000 stages and gained only 0.7% to 1.2% on 300 × 300 meshes, so it defaults to disabled.

Transient storage uses 8 bytes per unknown per point, while recorded devices add terminal currents/charges, two square derivative matrices, and variables (12 floats per resistor, 111 per gate-oxide MOSFET).
Operating points and transients automatically record rated instances plus paths matched by [`record`](results.md#what-a-result-records); recording every instance multiplies memory 30× or more and hits `max_memory` sooner.

## How the engine solves a circuit

The engine flattens circuit hierarchy into devices bound to unknowns: all non-ground node voltages plus branch currents for voltage sources, inductors, and controlled voltage sources.
Equations follow charge-oriented modified nodal analysis `f(x) + d/dt q(x) = 0`, where `f` sums branch currents and `q` sums terminal charges.

Newton iteration solves the nonlinear equations using a single sparse LU ordering shared across real and complex analyses.
Every numerical choice follows SPICE conventions unless noted as a jolty refinement below, with full evidence cataloged in [Departures from ngspice](departures.md):

| Choice | Engine behavior | Origin and comparison |
| --- | --- | --- |
| Convergence | Requires step updates within unknown tolerances, equation residuals within tolerance plus floating-point rounding floor, and zero active voltage limiters. Residuals within rounding drive no further update. | Extends SPICE `reltol`/`vntol`/`abstol` with a rounding floor so milliohm branches at high voltage converge. ngspice skips residual checks (`niconv.c`, lines 40 to 95) and blocks convergence only on junction limiting (`mos1load.c`, lines 376 to 384, 719 to 720). |
| Limiting | Limits junction voltages logarithmically above critical voltage and constrains FET gate/drain steps near threshold, linearizing outward from the limited point. | Standard SPICE `pnjlim`, `fetlim`, and `limvds`. ngspice 47 uses a custom backward-stepping junction limiter (`devsup.c`, lines 54 to 61) and limits first steps from the last accepted point (`mos1load.c`, lines 201 to 208). |
| Self-heating | Caps per-iteration temperature steps at 50 K and floors temperature at 1 K, linearizing from the clamped temperature with rescaled parameters. | jolty-specific: SPICE lacks general thermal-network self-heating outside dedicated models like VBIC. |
| `gmin` | Places `1e-12` S across every semiconductor junction. | Standard SPICE. |
| Small resistors | Formulates resistors and series resistances below 1 Ω with branch-current unknowns so conductances above 1 S never swamp picosiemens shunts. | Modified nodal analysis group 2 (Ho, Ruehli, and Brennan, 1975); the 1 Ω threshold keeps at least four significant digits on `gmin`. |
| Operating point | Tries direct Newton from zero, then node-to-ground gmin stepping from 10 mS, junction gmin stepping to `gmin`, source stepping, and pseudo-transient continuation from 1 S until steady state. | Combines ngspice gmin/source stepping with pseudo-transient continuation (Kelley and Keyes, 1998) to cross folds. ngspice starts at non-zero guesses (`dioload.c`, lines 162 to 165) and ends with an unchecked 10 µs transient (`init.c`, lines 77 to 93). |
| DC sweep | Warm-starts each point from the previous solution and halves step sizes on failure; failed initial points approach from the DC operating point. | Natural-parameter continuation. ngspice restarts failed points from default guesses (`dctrcurv.c`, lines 306 to 313), risking branch jumps. |
| Periodic steady state | Applies shooting Newton with monodromy SVD projecting steps into the range of `Φ - I`; oscillators add period `T` under an orthogonal phase condition launched from the fastest-growing mode. | Shooting (Aprille and Trick, 1972) plus orthogonal phase condition (Seydel, 2010, §7.1); neutral-direction projection and automatic startup replace ngspice `.pss` manual frequency/settling guesses. |
| Integration | Uses trapezoidal integration by default, with optional Gear2 or backward Euler. | Standard SPICE. |
| Linear devices | Evaluates linear passives, controlled sources, and thermal elements once per run into a base matrix, skipping them during nonlinear Newton iterations. | jolty optimization: ngspice re-evaluates every device on every Newton iteration (`cktload.c`, lines 62 to 75). |
| Linear solves | Computes one BTF + AMD ordering per circuit, pivots with threshold 0.1 (reusing valid pivots and unchanged factors), and statically condenses proven-pivot unknowns such as Meyer displacements before factoring. | KLU (Davis and Palamadai Natarajan, 2010) plus static condensation (Wilson, 1974). Threshold 0.1 keeps random-circuit backward error below `10⁻¹⁴` versus `10⁻⁸` at KLU's 0.001. |
| Truncation error | Filters divided differences through `C (G + αC)⁻¹ α`, checking every charge against `reltol · max(|q|, chgtol)` and step-scaled current tolerance (`voltage_tolerance` for flux) times `trtol · (p+1)!`. Judges first post-corner steps from second-step curvature and applies variable-step Gear2 `(1 + ω)²/(6ω(1 + 2ω))`. | Combines SPICE `CKTterr` (`cktterr.c`, lines 37 to 42) with Shampine (1980) filtering (Hairer and Wanner, 1996, IV.8) and variable-step Gear2. By contrast, ngspice skips MOSFET bulk junctions (`MOS1trunc`), uses constant 2/9 for Gear2 (7.6× under-estimate after 10× cut), floors at `abstol`, and applies `abstol` to flux. |
| Step length | Caps default `max_step` at `stop / 50` and sets next/retried steps to `0.9` of the largest step meeting both charge and current tolerances. Limits growth to 2×, shrinkage to 10× (8× on Newton failure), floors steps at `16 ε · max(t, scale)` where `scale = min(max_step, first_breakpoint)`, and respects `max_memory`. | Derives the minimum step from IEEE-754 binary64 resolution (Shampine and Reichelt, 1997, §2; Hairer, Nørsett, and Wanner, 1993, §II.4) rather than ngspice's fixed `10⁻¹¹ · max_step` (`traninit.c`, line 36). Extends SPICE step control by solving against both charge and step-scaled current tolerances; ngspice accepts whenever permitted step exceeds 0.9× current step (`dctran.c`, line 801) and defaults `max_step` to `TSTEP`. |
| Corners | Lands exactly on every waveform corner and breakpoint, restarts at 0.1× the smaller of prior permitted step and next-corner gap, and uses backward Euler until enough points build for higher order. | Matches SPICE `CKTsaveDelta`. ngspice pre-equalizes pairs within 1.9 steps of a corner (`dctran.c`, lines 588 to 598), adding 11% accepted and 47% rejected steps on the buck benchmark. |
| Jumps | Detects behavioral discontinuities (`u`, `sgn`, `floor`, `ceil`, comparisons), locates crossings by linear interpolation, lands on the crossing, and restarts integration. | Hybrid-system zero-crossing location; SPICE steps blindly across discontinuities and ngspice can stall. |
| Ringing | Tests twice-alternating trapezoidal derivatives with a variable-step harmonic restoring-force check, linearized passivity `uᵀ S G u ≥ uᵀ S (αC) u` (`F_{i,i} < 0.5` for Meyer displacements), and saddle-point projection for capacitive loops before inserting one backward Euler step. | Derived from Dahlquist (1963) stability function `R = 2F - I` and Shampine (1980) filter `F = αC(G + αC)⁻¹` (Hairer and Wanner, 1996, IV.3, IV.8), separating stiff/algebraic ringing from lossless LC oscillations (`ωh > 2`). |
| Prediction | Extrapolates node voltages linearly from the last two accepted points, falling back to constant prediction immediately after corners, jumps, or logic transitions. | Matches SPICE prediction, which extrapolates per-device junction voltages. |
| Time-step Newton | Caps iterations at `max_transient_iterations` (judging the final iterate), scales contracting un-limited steps by `θ/(1 - θ)` (using raw steps on initial, limited, or non-contracting steps), and retries failed steps from the last iterate. | Follows Hairer and Wanner (1996, IV.8) and Deuflhard (2004, §2.1) without early aborts because full Newton contracts faster near roots. Early aborts inflated 500-inverter steps from 2,358 to 6,103 and halted the buck at 41 µs. ngspice ignores `ITL4` below 100 (`niiter.c`, lines 37 to 39). |
| Loop gain | Injects unit series voltage and unit shunt current across the probe using one factorization per frequency, computing `u = v_x - i` and `T = -u / (1 + u)`. | Middlebrook double injection, matching Tian's method and Spectre `stb`. |
| Logic | Locates threshold-detector crossings by linear interpolation, lands steps on crossings, and iterates DC analog-logic coupling up to four times before marking oscillating nets unknown. | Mixed-signal event scheduler with connect elements. |
| Logic loads | Computes each logic driver's electrical load capacitance from its 1 Hz small-signal current at the operating point. | jolty-specific load extraction. |

When no time step converges down to the minimum step, transients fail while naming the worst non-converging unknowns or the equation violating truncation tolerance.
Singular DC or AC systems fail while naming up to three undetermined unknowns (such as parallel voltage-source currents or capacitor-isolated nodes).

Reference-corpus tests verify agreement with ngspice within `10⁻⁶` on operating points, sweeps, and AC responses, and within `10⁻³` of signal swing on transients.
Random circuits spanning all devices up to 100 unknowns agree to the same tolerances when jolty steps onto ngspice's time points.
When multistable circuits settle on different branches, initializing ngspice at jolty's solution confirms agreement.
Disagreeing transients are refined by 4×, 16×, and 64× smaller steps; chaotic trajectories that fail to stabilize under refinement are excluded as references.
If jolty hits a singular fold on a purely resistive net where ngspice jumps past, adding 10 fF parasitic capacitance to ground must succeed.
Operating points that ngspice reaches only via an unchecked fixed-length transient are excluded as DC references.
Metamorphic tests confirm identical simulation results under net renamings, instance reorderings, split series resistors, and inserted 0 V current-sense sources.
Order-of-accuracy tests verify 1st-order convergence for backward Euler and 2nd-order convergence for trapezoidal and Gear2 against closed forms and diode-clamped transients.
