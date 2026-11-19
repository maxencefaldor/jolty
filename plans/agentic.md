# Agentic design

Status: rewritten in plain prose on 15 November 2026 with every item checked against source and code, and updated on 18 November.
The archival research report lives in `research/agentic.md`, kept whole.

Plan: ✅ 13 done · 🚧 1 in progress · 📋 9 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

An agent should design better circuits in fewer tries with jolty than with any other simulator.
Progress is measured by old roadmap item 8: the same agent runs identical tasks with jolty and with ngspice, compared by success rate and simulation count (item 14).

## 2. Where we stand

**Capabilities jolty gives an agent beyond any surveyed reference or published framework:**

- **One definition, two front ends:** every command is a request model and function in `jolty.service`, driving both the CLI and the MCP server, while the Python API serves in-process loops.
- **Every problem at once:** diagnostics report each instance path and originating script line across 16 rules at configurable levels, each documented with an exact condition (`jolty explain`).
- **Images on every run:** schematics highlight diagnostic targets alongside plots for every analysis, plus histograms and envelopes for batches.
- **Specs and verdicts:** 23 engine measurements and four verdicts reporting measured value, slack, worst corner, and script line, with Wilson yield intervals and corners, Monte Carlo, and worst case in one command.
- **Twelve analyses:** operating point, operating points, DC sweep, AC, transient, periodic steady state, sensitivity, mismatch, stability, noise, pole-zero, and transfer function.
    Batches parallelize across threads, shards distribute across machines, and outputs enforce hard bounds (300 s per script, 40 unknowns, 20 diagnostics).
- **Exact derivatives:** unified JVP and VJP for every analysis, exposed in Python via `jolty.function`, `jolty.jvp`, and `jolty.vjp`, and in JAX via `jolty.jax.function` (items 12b and 19).
- **Discovery:** six hard design problems with circuits an agent discovered using jolty alone (`docs/discovery`).
- **Item 8 harness** (`benchmarks/agents/compare.py`): covers both arms with clean configuration, platform sandboxing per session, confined scorers, failure categories, and Wilson intervals.
    Every analysis is counted regardless of caller (item 1), verified on a stub agent.

**No real-agent run yet:** the paid run for item 8 awaits the owner's API credential (item 14).

**Where an agent's effort goes** (observed across discovery scripts, the guide, and the MCP server):

- **Harness code:** agents write manual bisections (50 steps to tune one resistor), Monte Carlo loops, and custom measurements because sensitivities target unknowns rather than specs and jolty lacks an optimizer (items 9 and 11).
- **Learning the API:** resolved (item 5) by deriving MCP instructions programmatically from code covering every device, real part, analysis, and measurement.
    All 17 guide pages and 33 examples are bundled offline via `jolty docs`, `read_docs`, and MCP `jolty://docs/...` resources.
- **Completed since 15 November:** transistor operating-point regions and margins appear in `OperatingPointResult.region` and `jolty simulate` (items 4 and 4b), and `PeriodicSteadyState(period=None)` solves autonomous oscillators (item 22).
- **Images:** each plot costs 774 to 1,849 visual tokens, totaling roughly 2,800 tokens per simulation (item 6).

**Literature findings** (2024 to 2026): simulator feedback with computed measurements produces the largest success gains (SPICEAssistant rises from 14.8% to 53.0%; AnalogPortBench rises from 30% to 56%).
Published agents delegate sizing to numerical optimizers and fail most often on oscillators, topology changes, and infeasible specs.
No published study evaluates the same agent across two toolchains, which item 8 tests directly.

## 3. Gaps

- Agent productivity gains with jolty have not yet been measured on a real agent.
- Feedback stops at circuit facts: no spec derivatives, no numerical error bounds on measurements, and no built-in optimizer (transistor regions and margins are done in items 4 and 4b).
- Every CLI command spawns a fresh Python process with unmeasured startup cost (`speed.md` 8), while saved `.results.npz` session queries (item 15) and offline docs plus programmatic MCP instructions (item 5) are done.
- An agent can still bypass analysis counting by invoking an external binary directly, and the harness sandbox remains untested on Linux.

## 4. Plan

Each item specifies its scope, motivation, and completion criteria; **Lead** marks an item where jolty surpasses every reference.
Each item's source references the corresponding row in `research/agentic.md` §4.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. Every simulation counted regardless of caller

- **What:** every executed analysis appends a line to the log configured by `analysis_log`, whether invoked via CLI, MCP server, or Python script.
- **Result** (`dfb5d4e`): item 8's harness counts each arm's simulations from that log, verified by a stub session simulating through all three entry points.
- **Source:** row 1.

### ✅ 2. Provenance in every results file

- **What:** every results file records the jolty version, build, and platform alongside hashes of the simulated circuit and plan.
- **Result** (`dbcbfb2`): format version 9 stores provenance, `load_provenance` reads provenance metadata, and `load_results` rejects files whose circuit hash mismatches a supplied circuit.
- **Source:** row 23.

### ✅ 3. Item 8's open follow-ups

- **What:** excuse unpriced parts in the cost-down scorer by component kind rather than path, audit the `cubic` threshold near rounding, and export Wilson intervals from `jolty.specs`.
- **Result:** the cost-down scorer fix and Wilson interval export were already on `main` (`52c822f`, `679b556`).
    The `cubic` script's known solution measures exactly 1.0 against the `≤ 1` threshold, whereas the netlist form measures `1 ± 3e-13` and may fail by rounding (documented for the paid run).

### ✅ 4. Transistor operating regions and margins at the operating point

- **What:** report each MOSFET region (off, weak inversion, linear, or saturated) and each bipolar region (cut off, active, saturated, or reversed) with distances to each boundary (`vgs − vth`, `vds − vdsat`, and bipolar `vce` above saturation).
    Each model computes regions and margins from model equations, exposed on operating-point results and sorted nearest-boundary-first in reports.
- **Why:** operating region is the first check analog designers and published circuit agents inspect.
- **Done when:** reports list every transistor's region, backed by tests covering every region of every transistor model.
- **Result** (`05812c3`): each transistor model declares regions and margins alongside the device kernel and reports them among device variables, readable via `OperatingPointResult.region(path)` (formerly `OperatingPoint.region`).
    Text and JSON reports from `jolty simulate` list transistors nearest a boundary first, and results files persist them (format version 10).
    Recording full variables adds 0.24 to 0.84 ms on the 500-inverter chain, so only `jolty simulate` enables `record` by default (approved by the owner on 15 November).
    Since `adaedc9`, recording regions alone (`Recorded.regions`) adds no measurable overhead on a fresh chain operating point and 9 to 10 µs on a cached point, eliminating duplicate evaluations.
- **Source:** row 9.

### ✅ 4b. Margins in the designer's sign convention

- **What:** flip PMOS and PNP margin signs so positive margins always indicate on or saturated operation (matching `|VGS| − |VTH|`), correcting the earlier subagent's signed-voltage output.
- **Result** (`8af6932`): PMOS and PNP margins are flipped so positive values indicate on or saturated operation for every transistor.
- **Source:** owner decision, 15 November.

### ✅ 5. Offline reference inside the session

- **What:** derive MCP instructions from code covering every device, real part, analysis, and measurement, and serve the guide and examples offline via MCP resources and `jolty docs`.
- **Why:** agents inside item 8's sandbox cannot access external docs, and the hand-written instructions listed only 11 of 33 device constructors.
- **Done when:** a test fails if any public constructor is omitted from MCP instructions, and the guide reads offline inside item 8's sandbox.
- **Result** (18 November): zero-dependency `crates/jolty-python/build.rs` generates `OUT_DIR/docs.json`, packed into wheels as `jolty/assets/docs.json` (`428,613` bytes covering 17 guide pages and 33 examples).
    Added the `jolty docs [TOPIC]` CLI command, `read_docs` MCP tool, 50 static MCP resources (`jolty://docs/guide/...`, `jolty://docs/examples/...`), and 2 URI templates.
    Replaced the hand-written MCP instructions with programmatic `build_instructions()` derived from `jolty.devices`, `jolty.parts`, `jolty.plan`, `jolty.specs`, `jolty.service`, and bundled doc summaries.
    Unit tests fail whenever any public device constructor, part, analysis, or measurement is missing.
- **Source:** row 15.

### 📋 6. Image budget

- **What:** let callers select which images each call returns, document each image's visual-token cost, and set the default using item 14's ablation data.
- **Done when:** reports state each image's visual-token cost.
    **Source:** row 18.

### ✅ 7. Sensitivities over parameters named by pattern

- **What:** `Sensitivity(..., parameters=["X1/R*"])` differentiates outputs only with respect to parameters matching the pattern.
- **Done when:** pattern-filtered `Sensitivity` differentiates only the matching parameters.
    **Source:** row 8.
- **Result:** superseded by item 12b, where JVP and VJP differentiate only the passed parameters and build twins exclusively for them.

### 📋 8. Quality and cost metrics alongside pass/fail in item 8's report

- **What:** report each arm's worst slack, part count, BOM cost, and tokens and dollars per success alongside the success rate.
- **Additional scope** (owner, 15 November): report multiple resource metrics per arm: simulation calls, analyses run (a 1,000-sample Monte Carlo counts as 1,000), wall and CPU time, Linux instruction counts, tokens, and dollars.
- **Done when:** the report lists every metric per arm.
    **Source:** row 5.

### 📋 9. Requirement derivatives with respect to named parameters

- **What:** compute the derivative of each requirement's slack (distance from its limit) with respect to agent-named parameters at the worst condition, plus linear worst-case estimates.
- **Why:** agents currently write manual bisection loops to tune designs.
- **Done when:** each derivative matches a central difference, and reports list the largest derivatives for each failing requirement.
- **Lead.**
    **Source:** row 7.

### 📋 10. Numerical error on transient measurements

- **What:** attach numerical error bounds to waveform delays and frequencies, derived from step error estimates or a half-step verification run.
- **Done when:** transient measurements carry numerical error bounds verified against a tenth-step reference run.
- **Lead.**
    **Source:** row 10.

### 📋 11. Built-in optimization in the plan

- **What:** deterministic gradient-based optimization over bounded or catalog-constrained parameters maximizing worst slack across requirements and corners, reporting each iterate and the best reachable slack when negative.
- **Done when:** bandgap flattening and cost-down resistor sizing each solve in one call using fewer solves than hand-written loops.
- **Lead.**
    **Source:** row 11.

### ✅ 12. Scalar loss gradient in one additional solve

- **What:** adjoint differentiation using one transposed solve per scalar loss in DC and frequency analyses, yielding gradients with respect to all parameters simultaneously.
- **Done when:** a loss gradient costs one additional solve beyond the forward analysis.
    **Source:** row 12.
- **Result:** delivered via item 12b's VJP with one transposed solve per loss at each point regardless of parameter count.
    In AC analysis, the operating-point coupling term is exact as well, costing one transposed DC solve per cotangent (`device-derivatives.md` 2b).

### ✅ 13. Transient gradient via the discrete adjoint

- **What:** discrete adjoint differentiation through transients, optionally with binomial checkpointing to bound memory, keeping objective gradient cost independent of parameter count.
- **Done when:** integrated loss gradients match direct differentiation at a cost independent of parameter count.
- **Lead.**
    **Source:** row 13; `research/speed.md` row M4.
    Item 13 replaces the duplicate entry in `speed.md`.
- **Result** (`f0cefc5` to `29d0b63`, on `main`): exact transpose of every accepted step across variable-step backward Euler, trapezoidal, and Gear2, exposed as the transient VJP under item 12b.
    Gradient cost stays 1.5 to 2.2 times the forward transient regardless of parameter count, versus 2 to 65 times for direct differentiation (1 to 1,000 crossbar weights on machine C).
    No checkpointing is needed because the transient retains full waveforms.

### ⏸️ 14. Item 8's paid run

- **What:** run the benchmark on a real agent pooled across tasks, adding sessions where success rates are intermediate.
    Include ablation arms without images, without rules, CLI-only, and without the reference, using tasks held out from public sources with web access logged.
- **Waits for:** the owner's API credential and approval to spend budget (deferred on 15 November).
- **Done when:** every arm reports success rate with confidence intervals and each feature's ablation effect is measured.
- **Lead:** no published study ablates simulator features systematically.
    **Source:** rows 2, 3, and 4.

### ✅ 15. Session queries

- **What:** compute any measurement from saved results without re-simulating, re-running only analyses whose inputs changed.
- **Done when:** measuring saved results files works through both MCP and the CLI.
- **Result** (18 November): added `MeasureRequest` and `measure_results()` in `jolty.service`, exposed via CLI `jolty measure RESULTS --measure ...` and MCP `measure_results` (`tests/test_service.py`, `tests/test_cli.py`, `tests/test_mcp.py`).
    Saved `.results.npz` files can now be queried without invoking the simulator across DC, AC, transient, periodic, and noise metrics.
    Supported measurements cover `value`, `dc`, `peak`, `trough`, `average`, `rms`, `ripple`, `gain`, `phase`, `bandwidth`, `crossover`, `phase_margin`, `gain_margin`, `rise_time`, `settling_time`, `crossing`, `frequency`, `distortion`, `period`, `noise_density`, and `integrated_noise`.
    **Source:** row 17.

### 📋 16. Progress reporting for long MCP runs

- **Done when:** a long Monte Carlo run reports completed samples incrementally over MCP.
    **Source:** row 19.

### 📋 17. Data-only simulation path

- **What:** simulate a saved circuit, plan, and specs without executing any Python script.
- **Done when:** `jolty simulate` accepts circuit and plan files and outputs the standard report.
    **Source:** row 21.

### 📋 18. Optional runner confinement

- **What:** execute scripts with network access disabled and writes confined to the script directory, tested on both Linux and macOS.
- **Done when:** a script attempting to write outside its directory fails on both macOS and Linux CI.
    **Source:** row 22.

### ✅ 12b. Unified derivative path: JVP and VJP for every analysis

- **What:** give every analysis (operating point, DC sweep, AC, transient, periodic steady state) exactly two engine primitives: JVP and VJP, built from kernel dual numbers and implicit differentiation.
    JVP pushes parameter perturbations forward to every output, while VJP pulls output loss sensitivities back to every parameter.
    `Sensitivity`, `Mismatch`, period derivatives, and the transient adjoint (item 13) become thin consumers of JVP and VJP.
    In Python, `jolty.function(circuit, analysis)` yields `f(params) -> outputs`, and `jolty.vjp` and `jolty.jvp` mirror `jax.vjp` and `jax.jvp` over plain nested dicts of differentiated parameters.
- **Why** (owner, 16 November): establish one uniform derivative pipeline instead of per-feature APIs added piecemeal that do not match JAX's calling conventions.
- **Done when:** every analysis's JVP and VJP satisfy the dot-product test (`<v, J t> = <Jᵀ v, t>`) and match finite differences, with all existing feature tests passing unchanged.
    Transients starting from initial conditions must differentiate accurately (fixing the bug flagged by the adjoint agent), and the guide must document `function`, `jvp`, and `vjp`.
- **Result** (`0820e3b`, `26b418c`, on `main` since 17 November): implemented `Simulator::jvp`, `vjp`, and `pull_back` for operating point, DC sweep, AC, transient, and periodic steady state (period keyed `"the period"`).
    Only named parameters construct twins, and `Sensitivity`, `Mismatch`, period derivatives, and `Objective` now delegate to the shared path with legacy code paths deleted.
    The dot-product test holds to 7.4e-16, and transients from initial conditions now differentiate accurately (previously returning `d out/dR` = −1 mV/Ω at t = 0 for an RC starting at 0 V).
    Python exposes `jolty.function`, `jolty.jvp`, and `jolty.vjp` with float parameters in SI base units.
    AC's operating-point term initially used a per-parameter central difference and became exact in `1349a11` (`device-derivatives.md` 2b).
    **Remaining:** transient and period VJPs reject logic devices, and delay lines are rejected in both directions.
- **Lead.**
    **Source:** owner decision on 16 November; replaces item 7's pattern selection (passed parameters select what is differentiated) and enables item 19.

### 🚧 19. Bridges to JAX and PyTorch

- **What:** enable `jax.grad`, `jax.vjp`, `jax.vmap`, and `jax.jit` through circuit analyses so jolty solves the circuit while JAX differentiates surrounding networks, losses, and optimizers.
- **Why:** analog circuit training for transformer-like networks runs in a JAX loop, which previously exchanged files without gradients into JAX.
- **How:**
    - `jolty.jax.function(circuit, analysis)` returns a JAX callable mapping a nested parameter dict (matching `jolty.function`) to a dict of outputs.
    - The forward pass invokes jolty through `jax.pure_callback`, running on CPU inside `jit` outside XLA-compiled code.
    - The backward rule registers `jolty.vjp`'s pullback (item 12b) via `jax.custom_vjp`, costing one transposed solve per backward pass regardless of parameter count, including exact AC (`device-derivatives.md` 2b).
    - The batching rule maps `jax.vmap` over inputs into a single multi-core jolty batch run.
    - Forward-mode `jax.jvp` is rejected by `custom_vjp` until a concrete use case emerges (owner approved, 16 November).
    - PyTorch support will wrap `torch.autograd.Function` around the same pullback once JAX lands.
- **Needs:** item 12b (on `main`); `jax` as an optional extra `jolty[jax]` (owner approved, 17 November); `torch` awaits owner approval.
- **Done: JAX** (`8fd0b77`, reworked on 18 November): `jolty.jax.function(circuit, analysis, options=None, *, saved_pullbacks=16)` lives behind optional extra `jolty[jax]` (`jax>=0.12`) as thin glue over `jolty.simulate.Function`.
    Every call, whether `jax.vmap`-mapped or unbatched, runs one `Function.batch` forward and one `BatchPullback` backward across worker threads via `Batch::map`.
    The forward pass caches up to `saved_pullbacks` pullbacks so `jax.grad` solves the circuit once; evicted pullbacks re-run the forward pass with identical bits.
    All five differentiable analyses are supported: `OperatingPoint`, `DcSweep`, `Ac` with complex cotangents, `Transient` and driven `PeriodicSteadyState` evaluated at required `breakpoints`, and autonomous oscillators returning only `PERIOD`.
    In `tests/test_jax.py`, `jax.grad` matches `jolty.vjp` bit-for-bit across every analysis and agrees with finite differences under `jit`, `remat`, `jacrev`, nested `vmap`, and empty `vmap`.
    Batch `vmap` matches an explicit loop bit-for-bit with `jax_enable_x64` both enabled and disabled, including under free-threaded Python 3.15t with the GIL disabled.
    Batched runs skip matrix dissection unless options specify interior thresholds, so large circuits where `jolty.function` dissects may differ in the last digits.
- **Not done: PyTorch:** wrapping `torch.autograd.Function` around `BatchPullback` awaits owner approval of `torch` as an optional dependency.
- **Done when:** `jax.grad` of a loss through operating point, DC sweep, and transient analyses matches `jolty.vjp` bit-for-bit and matches finite differences under `jit`.
    In addition, `vmap` over a batch of inputs must match an explicit loop bit-for-bit.
- **Source:** row 14; owner direction on 16 and 17 November.

### 📋 20. Large regular circuits in a few lines

- **What:** explicit tensor ports and nets, instance arrays, and scoped supplies in the circuit format so memory arrays and crossbars take only a few lines while keeping every supply visible to checks.
- **Done when:** a million-cell crossbar takes a few lines and checks inspect every supply.
- **Source:** row 24; scaling blueprint §1.2 and §3.4; front-end performance at that scale is tracked in `speed.md` 11.

### ✅ 21. Operating-point iterations recorded in results

- **What:** record operating-point Newton iterations and homotopy strategy counts, mirroring transient solver statistics.
- **Done when:** every DC result reports iteration counts and hard-circuit corpus DC cells display them for jolty.
- **Result** (`090b521`): every operating point and DC sweep reports Newton iterations and strategy in reports and version 16 results files, shown in the corpus DC cells.
- **Source:** first run of the corpus.

### ✅ 22. Autonomous oscillators

- **Owner direction** (16 November): integrate autonomous shooting into `PeriodicSteadyState`, solving for period as an unknown when no source fixes period.
- **What:** periodic steady state and sensitivities of autonomous oscillators with unknown period.
- **Result** (`050ebf2`, `e532cca`, `a0a9152`, 16 November): `PeriodicSteadyState(period=None, ...)` (kind `periodic_steady_state`) replaces `Transient(periodic=True)` in results format version 18.
    With no fixed period, shooting solves for `x₀` and `T` under the orthogonal phase condition (Seydel §7.1) while propagating `∂φ/∂T` exactly across steps.
    Autonomous startup perturbs a transient by one tolerance along the fastest-growing mode and runs doubling stretches until returns stop growing.
    On a lossless LC tank, the solved period falls within 2.3e-4 of `2π√(LC)`, well inside the 3.3e-4 bound.
    The example five-stage ring agrees with a long transient's settled period to 2.8e-8, inside its 5.9e-8 scatter, and driven cases remain bit-identical.
    Inverse iteration for growing modes starts from a fixed-seed vector as in LAPACK because an all-ones vector lacked a component along a symmetric ring's growing mode.
    Failure conditions report `NoGrowingMode`, `Unsettled`, and `Stationary` (when shooting collapses onto equilibrium).
    Period sensitivity (`a0a9152`) adds `Sensitivity(outputs, of=PeriodicSteadyState(...))` and `period_derivative(instance, parameter)` via the converged bordered system per parameter.
    On an LC tank at `reltol = 1e-10`, sensitivity satisfies `L·∂T/∂L + C·∂T/∂C = T` to 5.7e-14 and matches central differences to 1.24e-7.
    Follow-ups on 18 November added `.pss` SPICE reader support (`PeriodicSteadyStateCard`, `PeriodicSteadyState.from_spice`) and automatic detection of time-varying sources and behavioral `time` devices (`Waveform::varies_in_time`, `Kernel::varies_in_time`, `Runs::time_varying`, `check_periodic`).
    Memory limits (`max_memory`) are now enforced across PSS sensitivity, JVP, and VJP (`DENSE_MATRICES`, `forward_period_numbers`, `pull_period_numbers`).
- **Source:** row 20.

**Elsewhere:** end-to-end agent command timing and a warm runner live in `speed.md` 8; exact result caching lives in `speed.md` 16.

## 5. Decided

- A paid session requires an explicit environment credential (`ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN`) and never uses the owner's personal login.
- Each session's `PATH` includes only the active arm's tools and system directories.
- Run limits (time, turns, and budget) have no defaults; the owner specifies limits for each paid run.
- Item 8 waited for item 1 because undercounting one arm's tries would invalidate the comparison.

## 6. Open principles

None; the research proposals were merged into `AGENTS.md` on 15 November (`444535f`).
