# Agentic design

Status: rewritten in plain words on 15 November 2026, every item checked against its source and today's code; the research is `research/agentic.md`, kept whole.

Plan: ✅ 11 done · 📋 12 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

An agent designs better circuits, in fewer tries, with jolty than with any other tool.
Measured by the old roadmap's item 8: the same tasks done by the same agent with jolty and with ngspice, compared by success and by the number of simulations (item 14).

## 2. Where we stand

**What jolty gives an agent, more than any reference or published agent framework surveyed:**

- **One definition, two front ends:** every command is a request model and a function in `jolty.service`, from which the command line and the MCP server derive; the Python API serves loops in one process.
- **Every problem at once,** each naming the instance's path and the script line that placed it; 16 rules at configurable levels, each with its exact condition (`jolty explain`).
- **Images on every run:** a schematic marking what the diagnostics name, a plot of every analysis, histograms and envelopes for batches.
- **Specs and verdicts:** 23 measurements the engine reads, four verdicts with the value, its slack, its worst corner and its script line; yield with a Wilson interval; corners, Monte Carlo and worst case in the same command.
- **Ten analyses,** batches across threads, shards across machines; bounds on every output (300 s a script, 40 unknowns, 20 diagnostics).
- **Discovery:** six hard design problems and the circuits an agent found for them with jolty alone (`docs/discovery`).
- **Item 8's harness** (`benchmarks/compare.py`): both arms, a clean configuration and the platform's sandbox for each session, scorers run confined, failure kinds, Wilson intervals; every analysis counted whatever runs it (item 1); proven on a stub agent.

**None of it is measured on a real agent:** item 8's paid run waits for the owner's credential (item 14).

**Where an agent's effort goes** (the discovery scripts, the guide and the MCP server):

- **Harness code:** it writes bisections (50 steps to tune one resistor), Monte Carlo loops and measurements by hand, as jolty's sensitivities are of unknowns, not of specs, and nothing in jolty optimizes (items 9 and 11).
- **Learning the API:** the MCP instructions (792 words) name 11 of the 30 device constructors, no transistor and no real part, and never mention `specs()`, corners or Monte Carlo; in item 8's sandbox the agent can read jolty's source but not the guide (item 5).
- **Reading an operating point:** no report states a transistor's region (item 4).
- **Oscillators:** no steady state for one that runs on its own, so the agent runs long transients; oscillators are where published agents fail most (item 22).
- **Images:** a plot costs 774 to 1,849 visual tokens, about 2,800 a simulation (item 6).

**What the literature shows** (2024 to 2026): simulator feedback with computed measurements lifts success most (SPICEAssistant from 14.8% to 53.0%, AnalogPortBench from 30% to 56%); sizing is handed to numerical optimizers; agents fail on oscillators, topology changes and infeasible specs.
No published study gives the same agent two toolchains, which item 8 does.

## 3. Gaps

- What jolty gives an agent has never been measured on a real agent.
- Feedback stops at facts about the circuit: no derivative of a spec, no transistor's region, no numerical error of a measurement, no optimizer.
- The reference is not inside the session; the MCP instructions cover a third of the devices.
- Every command starts a fresh Python process, its time unmeasured (`speed.md` 8).
- An agent can still bypass the counting by calling a real program directly; the harness's sandbox is untested on Linux.

## 4. Plan

Each item says what it is, why it matters, and what "done" means; **Lead** marks an item where jolty would go beyond every reference.
Its source is the research report's row, in `research/agentic.md` §4.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. Every simulation counted, whatever runs it

- **What:** each analysis a simulation runs appends a line to a log the setting `analysis_log` names, whether the command line, the MCP server or a script runs it.
- **Result** (`dfb5d4e`): item 8's harness counts each arm's simulations from it; a stub session that simulates through all three is counted right.
- **Source:** row 1.

### ✅ 2. Provenance in every results file

- **What:** each results file states the jolty, the build and the platform that simulated its results, and the hashes of the circuit and the plan they are of.
- **Result** (`dbcbfb2`): version 9 of the format; `load_provenance` reads it; `load_results` given a circuit refuses a file whose results are of another.
- **Source:** row 23.

### ✅ 3. Item 8's open follow-ups

- **What:** the cost-down task's scorer excuses an unpriced part by what it is, not by its path; the `cubic` task's threshold, met by its known answer only to rounding; the Wilson interval exported from `jolty.specs`.
- **Result:** the first and the last were already on `main` (`52c822f`, `679b556`); the `cubic` script's known answer measures exactly 1.0 against a threshold of `≤ 1`, which holds; the netlist form, at 1 ± 3e-13, may fail by rounding, stated for the paid run.

### ✅ 4. Each transistor's region and margins at the operating point

- **What:** whether each MOSFET is off, in weak inversion, linear or saturated, and each bipolar transistor cut off, active, saturated or reversed, with its distance from each boundary (`vgs − vth`, `vds − vdsat`, a bipolar transistor's `vce` over its saturation voltage), reported by each model as its own equations define it, readable from an operating point's result and listed in the report, nearest the boundary first.
- **Why:** the first fact an analog designer reads; the published agents check it first.
- **Done when:** the report lists each transistor's region, with a test per region of every transistor model.
- **Result** (`05812c3`): each transistor model declares its regions and margins beside its kernel and reports them among its variables; `OperatingPoint.region(path)` reads one; `jolty simulate`'s report lists every transistor nearest a boundary first, as its JSON does; results files keep them (version 10).
    Recording them costs an operating point of the 500-inverter chain 0.24 to 0.84 ms, so only `jolty simulate` records them by default, through `record`, which the owner kept (15 November).
    Since `adaedc9`, regions alone (`Recorded.regions`) cost a fresh operating point of the chain nothing measurable and a remembered one 9 to 10 µs; a remembered point is no longer evaluated twice.
- **Source:** row 9.

### ✅ 4b. Margins in the designer's sign

- **What:** a PMOS's and a PNP's margins flipped, so a positive margin means on, or saturated, for every transistor, as designers write |VGS| − |VTH|; the region subagent had kept each device's signed voltages.
- **Source:** the owner, 15 November.

### 📋 5. The reference inside the session

- **What:** the MCP server's instructions derived from the code, naming every device, real part, analysis and measurement; the guide and the examples served as MCP resources and by a `jolty docs` command, so an agent reads them offline.
- **Why:** an agent in item 8's sandbox cannot read the guide; the instructions name 11 of 30 constructors.
- **Done when:** a test fails when a public constructor is missing from the instructions; the guide reads offline in item 8's sandbox.
- **Source:** row 15.

### 📋 6. An image budget

- **What:** the images each call returns chosen by the caller, each image's cost in visual tokens stated, the default set by item 14's measured effect of images.
- **Done when:** the report states each image's cost.
    **Source:** row 18.

### ✅ 7. Sensitivities over parameters named by pattern

- **What:** `Sensitivity(…, parameters=["X1/R*"])` differentiates the outputs with respect to the parameters a pattern names, and only those.
- **Done when:** that call differentiates only the matching parameters.
    **Source:** row 8.
- **Result:** superseded by item 12b: the parameters a JVP or VJP is given are those differentiated, and only they get twins.

### 📋 8. Quality and cost beside pass or fail in item 8's report

- **What:** each arm's worst slack, part count, cost, and tokens and dollars per success, beside its success rate.
- **And** (the owner, 15 November): several measures per arm, not one count: simulation calls, analyses run (a 1,000-sample Monte Carlo counts 1,000), wall and CPU time, instructions on Linux, tokens and dollars.
- **Done when:** the report states them per arm.
    **Source:** row 5.

### 📋 9. Each requirement's derivative with respect to named parameters

- **What:** the derivative of each requirement's slack (how far a measurement is from its limit) with respect to parameters the agent names, at its worst condition, and the worst case computed from them.
- **Why:** an agent tuning a design writes bisections by hand today.
- **Done when:** each agrees with a central difference; a report lists each failing requirement's largest derivatives.
- **Lead.**
    **Source:** row 7.

### 📋 10. Each transient measurement's numerical error

- **What:** a delay or a frequency measured on a waveform carries its own numerical error, from the steps' error estimates or a run at half the step.
- **Done when:** a measurement carries its error, checked against a run at a tenth of the step.
- **Lead.**
    **Source:** row 10.

### 📋 11. An optimization in the plan

- **What:** parameters with bounds or catalog values, the worst slack over requirements and corners as the objective, a deterministic gradient-based search, each iterate reported, and the best reachable slack stated when it is negative.
- **Done when:** the bandgap's flattening and the cost-down's resistors are each found in one call, with fewer solves than the hand-written loops.
- **Lead.**
    **Source:** row 11.

### ✅ 12. A scalar loss's gradient for one solve more

- **What:** one transposed solve per scalar loss at DC and over frequency, the adjoint method, giving its gradient with respect to every parameter at once.
- **Done when:** a loss's gradient costs one solve more than the analysis.
    **Source:** row 12.
- **Result:** item 12b's VJP: at a DC point, one transposed solve per loss whatever the number of parameters; over frequency, the term through the operating point is still a central difference per parameter (`device-derivatives.md` 2b).

### ✅ 13. A transient's gradient by the discrete adjoint

- **What:** the adjoint through a transient, with binomial checkpointing to bound memory, so one objective's gradient costs the same whatever the number of parameters.
- **Done when:** an integrated loss's gradient agrees with the direct method, at a cost independent of the parameters' number.
- **Lead.**
    **Source:** row 13; `research/speed.md` row M4.
    This item replaces `speed.md`'s copy of it.
- **Result** (`f0cefc5` to `29d0b63`, on `main`): the exact transpose of each accepted step, backward Euler, trapezoidal or Gear2, variable steps; 1.5 to 2.2 times the transient's own cost whatever the number of parameters, against 2 to 65 times for the direct method (a crossbar of 1 to 1,000 weights, machine C); under item 12b, the transient's VJP.
    No checkpointing: the transient keeps its waveforms.

### ⏸️ 14. Item 8's paid run

- **What:** the benchmark run on a real agent: pooled over tasks, sessions added where success is middling; ablation arms without images, without rules, with the command line only, without the reference; tasks held out of every public source, web use logged.
- **Waits for:** the owner's credential and go-ahead to spend it; not yet (the owner, 15 November).
- **Done when:** each arm's success with its interval, each feature's measured effect.
- **Lead:** no published study ablates a simulator's features.
    **Source:** rows 2, 3 and 4.

### 📋 15. Session queries

- **What:** any measurement of a saved result read without simulating again; only what changed run again.
- **Done when:** a measurement of a results file works through MCP and the command line.
    **Source:** row 17.

### 📋 16. Progress for long runs over MCP

- **Done when:** a long Monte Carlo reports its samples done as it goes.
    **Source:** row 19.

### 📋 17. A data-only path

- **What:** a saved circuit, plan and specs simulated with no Python code run.
- **Done when:** `jolty simulate` takes a circuit file and a plan file and gives the script's report.
    **Source:** row 21.

### 📋 18. The runner optionally confined

- **What:** scripts run with no network and no writes outside their directory, tested on Linux as on macOS.
- **Done when:** a script's write outside its directory fails, on macOS and Linux CI.
    **Source:** row 22.

### ✅ 12b. One derivative path: a JVP and a VJP for every analysis

- **What:** every analysis (operating point, DC sweep, AC, transient, periodic steady state) gets exactly two derivative functions in the engine, a JVP (Jacobian-vector product: a parameter nudge pushed forward to every output) and a VJP (vector-Jacobian product: a loss's sensitivity to the outputs pulled back to every parameter), both built from the device kernels' dual numbers and implicit differentiation through the solve.
    `Sensitivity`, `Mismatch`, the period derivative and the transient adjoint (item 13) become thin users of them.
    In Python, `jolty.function(circuit, analysis)` gives `f(params) -> outputs`, and `jolty.vjp` and `jolty.jvp` mirror `jax.vjp` and `jax.jvp`: parameters and outputs plain nested dicts, the parameters given being those differentiated.
- **Why:** the owner (16 November): one uniform process for all derivatives; today each feature exposes the same maths through its own API, added one analysis at a time, none in the shape JAX asks for.
    The explainer `explainers/derivatives.html` sets out the three layers.
- **Done when:** every analysis's JVP and VJP agree with each other (the dot-product test `<v, J t> = <Jᵀ v, t>`) and with finite differences; the existing features' tests pass unchanged on top of them; a transient from initial conditions differentiates correctly (the bug the adjoint agent flagged); the guide documents `function`, `jvp` and `vjp`.
- **Result** (`0820e3b`, `26b418c`, branch `derivatives`, 17 November, landing): `Simulator::jvp`, `vjp` and `pull_back` for the operating point, a DC sweep, AC, a transient and a periodic steady state (the period one more output, keyed `"the period"`); only named parameters get twins; `Sensitivity`, `Mismatch`, the period's derivatives and `Objective` are users of them, their own paths deleted; the dot-product test holds to 7.4e-16; a transient from initial conditions now differentiates correctly (it was wrong: `d out/dR` −1 mV/Ω at t = 0 for an RC from 0 V); Python `jolty.function`, `jvp`, `vjp`, parameters as floats in SI base units.
    Left: AC's term through the operating point is still a central difference per parameter, so the AC VJP costs 2 evaluations and 1 DC solve per parameter, until kernels give second derivatives (`models.md` 6); the time and period VJPs refuse logic; delay lines refused both ways.
- **Lead.**
    **Source:** the owner's call, 16 November; replaces item 7's pattern selection (the parameters passed are the selection) and gives item 19 its door.

### 🚧 19. Bridges to JAX and PyTorch

- **What:** `jax.grad`, `jax.vjp`, `jax.vmap` and `jax.jit` through a circuit's analysis, as through any JAX function: jolty solves the circuit, JAX differentiates everything around it (a network, a loss, an optimizer).
- **Why:** the owner's work trains analog circuits for transformer-like networks in a loop with JAX; today the loop passes through files, with no gradient into JAX.
- **How:**
    - `jolty.jax.function(circuit, analysis)` returns a JAX function of a nested dict of parameters, the same dict `jolty.function` takes, to a dict of outputs.
    - Its forward calls jolty through `jax.pure_callback`, so it also runs inside `jit`, on the CPU, outside XLA's compiled code.
    - Its backward rule, registered with `jax.custom_vjp`, is `jolty.vjp`'s pullback (item 12b): one transposed solve per backward pass whatever the number of parameters, including exact AC after `device-derivatives.md` 2b.
    - Its batching rule maps `jax.vmap` over inputs to one jolty batch run across every core.
    - A `custom_vjp` function refuses `jax.jvp`; forward mode is not built until a use appears (the owner agreed, 16 November).
    - PyTorch, through `torch.autograd.Function` around the same pullback, once JAX's is done.
- **Needs:** item 12b (on `main`); `jax` as an optional dependency, `jolty[jax]` (approved by the owner, 17 November); `torch` remains unapproved pending owner decision.
- **Result (JAX done, PyTorch remaining):** `src/jolty/jax.py` exposes `jolty.jax.function(circuit, analysis, options=None)` with optional dependency `jax>=0.12` (`jolty[jax]`).
    Covers all five differentiable analyses (`OperatingPoint`, `DcSweep`, `Ac` with exact complex Wirtinger cotangent convention, `Transient` with explicit validated `breakpoints`, and `PeriodicSteadyState` both driven with `breakpoints` and autonomous returning only `{PERIOD: scalar}`).
    Uses a single packed `jax.pure_callback(..., vmap_method="broadcast_all")` dispatching `batch_size == 1` directly through `simulator.run`/`vjp` (bit-for-bit identical to `jolty.function`, preserving nested dissection) and `batch_size > 1` into one multi-threaded Rust `Simulator.batch` call with matching `max_unknowns_per_interior` and `max_threads` settings across forward and backward.
    Handles empty batches (`batch_size == 0`), structural changes including dropped unknowns, `jax.remat`, `jax.jacrev`, `jax_enable_x64` (`True`/`False` via `canonicalize_dtype`) cleanly under `filterwarnings = ["error"]`, caches validated packed callbacks for eager and free-threaded `3.15t` execution without re-enabling the GIL, and locks `Elaborated._structures` narrowly in `simulate.py` (`tests/test_jax.py`).
    PyTorch (`torch.autograd.Function`) remains planned.
- **Done when:** `jax.grad` of a loss through an operating point, a DC sweep and a transient agrees with `jolty.vjp` bit for bit and with finite differences; under `jit`; and `vmap` over a batch of inputs agrees with a loop over them.
- **Source:** row 14; the owner's direction, 16 and 17 November.

### 📋 20. Large regular circuits in a few lines

- **What:** tensor ports and nets, instance arrays, and scoped supplies written explicitly into the circuit format, so a memory array or a crossbar is a few lines of a script, with every supply still visible to the checks.
- **Done when:** a crossbar of a million cells is a few lines; the checks see every supply.
- **Source:** row 24; the blueprint §1.2 and §3.4; the front end's speed at that size is `speed.md` 11.

### ✅ 21. An operating point's iterations in its result

- **What:** an operating point's Newton iterations and the strategy counts that found it, as a transient's statistics hold its own.
- **Done when:** every DC result states its iterations; the hard-circuit corpus's DC cells for jolty have them.
- **Result** (`090b521`): every operating point and DC sweep states its Newton iterations and strategy, in the report and results files (version 16); the corpus's DC cells show them.
- **Source:** the corpus's first run.

### ✅ 22. Oscillators that run on their own

- **The owner** (16 November): yes, built with the `PeriodicSteadyState` analysis, the period an unknown when no source sets it.
- **What:** the periodic steady state of an autonomous oscillator, its period an unknown, and its sensitivities.
- **Result** (`050ebf2`, `e532cca`, `a0a9152`, 16 November): `PeriodicSteadyState(period=None, ...)`, kind `periodic_steady_state`, replaces `Transient(periodic=True)` (results files version 18).
    With no period, shooting takes `x₀` and `T` as unknowns under the orthogonal phase condition (Seydel §7.1), `∂φ/∂T` carried along the steps exactly; it starts from a transient kicked by one tolerance along the fastest-growing mode, run in doubling stretches until its returns stop growing.
    An LC tank whose conductance is zero on a circle: period within 2.3e-4 of 2π√(LC), inside its 3.3e-4 bound; the examples' five-stage ring agrees with a long transient's settled period to 2.8e-8, inside its scatter of 5.9e-8.
    Driven cases bit-identical to before.
    Inverse iteration for growing modes now starts from a fixed-seed vector, as LAPACK does: all ones had no part along a symmetric ring's growing mode.
    Two errors, `NoGrowingMode` and `Unsettled`.
    The period's sensitivity (`a0a9152`): `Sensitivity(outputs, of=PeriodicSteadyState(...))`, every output's derivative along the period and an oscillator's `period_derivative(instance, parameter)`, by the converged period's bordered system per parameter; a tank meets `L·∂T/∂L + C·∂T/∂C = T` to 5.7e-14 at `reltol` 1e-10, central differences to 1.24e-7.
    Shooting that converges onto the equilibrium fails with `Stationary`.
    **Left:** `.pss` in the SPICE reader; a source that varies in time detected, not only stated; the sensitivity's columns held to `max_memory`.
- **Source:** row 20.

**Elsewhere:** timing an agent's command end to end and a warm runner are `speed.md` 8; an exact cache of results is `speed.md` 16.

## 5. Decided

- A paid session needs a credential in the environment, `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`; the owner's own login is never used.
- A session's `PATH` holds its arm's tools and the system's directories alone.
- The limits of a run (time, turns, budget) have no defaults: the owner sets them for the paid run.
- Item 8 waited for item 1: a run that undercounts one arm's tries would mislead.

## 6. Open principles

None: the research's proposals were merged into the rules of `AGENTS.md` on 15 November (`444535f`).
