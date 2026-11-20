# Device derivatives: implementation plan

Status (18 November 2026): steps 1, 2, and 2b are complete on `main`, with steps 3 and 4 deferred.
Chunk width 16 was benchmarked against width 8 for owner review (step 2.3), whole VJP over device parameters was timed with the alternating protocol (step 2.6), and the step-3 decision measurement remains held for the owner.
The implementation plan covers the first layer of jolty's derivatives: per-device derivatives from device equations through the engine's JVP and VJP.

## Why

Exact derivatives are a founding rule of jolty: Newton converges on them, and every gradient (sensitivity, mismatch, JVP, VJP, and training losses) builds on them.
Each device kernel provides two derivatives: voltage derivatives `∂F/∂x` on every Newton iteration, and parameter derivatives `∂F/∂p` whenever a gradient is requested.

Steps 1 and 2 retired two remaining debts in kernel differentiation:

1. **Inconsistent kernel implementations.**
    Most kernels compose small dual numbers at the natural width of each physical law.
    VBIC evaluated every intermediate on a single nine-wide dual number even though each law reads only one to three inputs.
    Gummel-Poon, JFET, memristor, switch, photocell, VDMOS quasi-saturation, and passive voltage coefficients carried hand-written derivatives ported from ngspice, remaining consistent only while derivation avoided errors.
1. **One evaluation per parameter for parameter derivatives.**
    A twin running on dual numbers for one parameter had to be prepared and evaluated repeatedly for each parameter, recomputing identical primal values.
    Training transistor sizes or running process-kit mismatch paid that cost per parameter per device at every point, while AC gradients still took a central difference per parameter.

The goal is to write every device law once as plain mathematics, with each quantity carrying derivatives only for the inputs read by that law and using custom derivative rules (`Dual::chain`) where required.
All parameter derivatives of a device come from one evaluation, strictly off Newton's critical path.

## How, at a high level

| Step | What | Status |
| --- | --- | --- |
| 1 | Every kernel on dual numbers composed at the natural width of each law, joined onto its terminals by the chain rule | Done |
| 2 | Parameter derivatives in chunks: one evaluation gives a device's derivatives with respect to several parameters; JVP by one evaluation per device; Newton untouched | Done |
| 2b | AC's VJP exact, and independent of the number of parameters | Done |
| 3 | One shared compile-time differentiation pass for hand-written and compiled kernels | Deferred |
| 4 | Compiler-level automatic differentiation | Deferred |

Dual numbers remain the foundation throughout steps 1 and 2 and serve as the reference against which any future method is tested.
Steps 3 and 4 would replace dual numbers only if measurements show unacceptable overhead or once stable Rust supports compiler-level automatic differentiation.

## What it changes

Each stage's completion criterion measures a concrete effect, summarized in the table below.

| Work | Simulation | Gradients |
| --- | --- | --- |
| 1.2 VBIC composed | Measured: 31 to 34% fewer instructions and 18 to 46% less time per evaluation (below) | Same gains for VBIC twins |
| 1.3 Gummel-Poon, JFET | Measured: 2 to 8% more instructions per evaluation; runtime shift stays within build-layout noise (below) | Same shifts for twins |
| 1.4 Short laws | Measured: at most 25 extra instructions per evaluation (below) | None |
| MOSFET circuits under step 1 | Unchanged because MOSFET kernels are already composed | Unchanged |
| 2 Chunk twins | Unchanged by design because Newton is untouched; every kernel keeps identical instruction counts except VDMOS (+0.9% from compiler inlining and codegen choices) | Measured on machine D: an 8-wide twin takes 2.4× (BSIM4) to 3.5× (diode at 0.69 V) fewer instructions than eight 1-wide twins (below). Whole `Simulator::vjp` + `pull_back` speeds up 1.28× to 2.16× total (1.51× to 2.45× `pull_back`) on chunk 8 vs chunk 1 with bit-identical results across all 25 cases (2.6). JVP evaluates each device once per tangent regardless of parameter count. |
| 2b AC exact | At most +2 instructions (+0.07%) on Newton `f64` kernels, -4 on VBIC, and -10 on photocell (`instructions.rs`, below) | Measured on machine E: AC VJP pullback is 49.8× faster at 1,000 parameters (107.8 to 2.17 ms; total `vjp` + `pull_back` 28.7× faster, 109.9 to 3.83 ms). Results match `61fa6e8` central differences within 9.74e-8 at 1,000 parameters (9.57e-10 on `loaded_amplifier`), with AC dot-product residual 7.15e-17. |

## Step 1: every kernel composed on dual numbers

### 1.1 Where each kernel stands

Read from `crates/jolty-models/src/kernel/` on 17 November 2026:

| Kernel | Derivatives in x | Work |
| --- | --- | --- |
| BSIM4, BSIM3, MOSFET levels 1 to 3, semi-empirical MOSFET | Composed: `Dual<R, 3>` for the channel (vgs, vds, vbs), `Dual<R, 1>` per junction | None |
| Diode, junctions | Composed: `Dual<R, 1>` | None |
| VDMOS | Composed: `Dual<R, 2>` for the channel, `Dual<R, 1>` for quasi-saturation (1.4) | None |
| Verilog-A compiled kernels | Each variable carries the probes it depends on and no others (`crates/jolty-veriloga/src/emit.rs`) | None |
| Behavioral sources and capacitors | `jolty-expressions` tape: values forward, gradients backward, suited to multi-input expressions | None |
| Linear devices | Constant matrix entries stamped once per run | None |
| VBIC | Composed: `Dual<R, 1>` to `Dual<R, 3>` for each law (1.2) | None |
| Gummel-Poon (`bjt.rs`), JFET (`jfet.rs`) | Composed: `Dual<R, 1>` per junction, `Dual<R, 2>` for intrinsic transistor laws and JFET channel, `Dual<R, 3>` for base-resistance current (1.3) | None |
| Memristor, switch, photocell, passive voltage coefficients | Composed: `Dual<R, 2>` of terminal voltage and control, control-only laws on `Dual<R, 1>` (1.4) | None |

### 1.2 VBIC composed

VBIC's equations depend on nine voltages (`VARIABLES`: base-emitter, extrinsic base-emitter, base-collector, extrinsic base-collector, parasitic transistor junctions, and internal resistance drops) and previously evaluated everything on `Dual<R, 9>`.

1. Map each law to only the voltages read by that law: junction currents and charges read a single junction voltage, while transport current and normalized base charge `qb` read intrinsic base-emitter and base-collector voltages.
    Each internal resistance current reads the resistor drop plus any modulating junction voltages.
1. Evaluate each law on `Dual<R, k>` at natural width `k` from one to three.
1. Chain each result onto the nine variables once at the end, following BSIM4 (`bsim4.rs`: `gradient: [gate, -(gate + drain + bulk), bulk]`), while the self-heated variant keeps temperature propagation through `Heating` unchanged.

**Done when:** every VBIC derivative agrees with the previous nine-wide implementation to floating-point rounding (operation order changes, preventing bit-identity), while ngspice tests and the law suite pass unchanged.
The alternating timing protocol (`scripts/bench --compare A,B`) must confirm composed VBIC is no slower than the nine-wide kernel on machine B.

**Result** (17 November): each law evaluates at natural width and joins via `Oriented::reading` and `Dual::embed` in `kernel/parts.rs` and `kernel/dual.rs`.
Weak avalanche current is separated into a dedicated current law, and both helpers support subsequent kernels.
Across 1,200 evaluations (both polarities, base charges, smoothings, temperatures, and excess-phase modes), residuals and conductances shift by at most 0.9 ε of the row maximum while charges and capacitances stay unchanged.
Every VBIC unit test, ngspice comparison, and law-suite check passes unchanged.
Instruction counts from `crates/jolty-models/benches/instructions.rs` (gungraun via Valgrind 3.25.1 Callgrind, excluding case setup and drop, against `main` `8fb1e4b`) drop across every configuration.
Defaults fall from 10,879 to 7,184 instructions (-34.0%), full models fall from 12,830 to 8,838 (-31.1%), and excess-phase models fall from 13,376 to 9,290 (-30.5%).
Criterion timings from `evaluation.rs` (two builds alternating twice on one pinned core of an AMD EPYC 7B12 at load 2 to 5, with machine B unreachable) confirm large wall-clock speedups.
Defaults drop from 1.42 to 1.51 µs down to 819 to 829 ns, and full models drop from 1.53 to 1.63 µs down to 1.02 to 1.26 µs.
With excess phase enabled, evaluation drops from 1.59 to 1.66 µs down to 1.01 to 1.05 µs.

### 1.3 Gummel-Poon and the JFET on composed dual numbers

An earlier port of both models to dual numbers ran 7 to 18% slower and was paused (`quality.md` 6), without recording the dual widths used.

1. Gummel-Poon: evaluate each junction's current, charge, and leakage on `Dual<R, 1>` over that junction voltage, and evaluate transport current, `qb`, and base-resistance modulation on `Dual<R, 2>` (base-emitter, base-collector).
    Excess phase continues running on delayed dual numbers.
1. JFET: evaluate the channel on `Dual<R, 2>` (gate-source, drain-source) and each gate junction on `Dual<R, 1>`.
1. Benchmark each kernel against the hand-written version using the alternating timing protocol on machine B.

**Done when:** no hand-written derivatives remain in either kernel, ngspice comparisons pass unchanged, and the law-suite Jacobian check holds.
Either the timing protocol shows no resolvable slowdown, or the measured cost is submitted to the owner to judge whether consistency justifies the overhead.
Until acceptance, hand-written kernels remain in a test as a reference matching the new kernels everywhere.

**Result** (17 November): each law evaluates at natural width via `Oriented::reading`, following VBIC.
Gummel-Poon reads limited junction voltages plus unlimited base-resistance drop and external base-collector voltage, making base-resistance current (`drop / r`) a three-variable law and external depletion a one-variable law.
Base-resistance shaping runs on one variable (base current) chained via `Dual::apply`, while the excess-phase filter and base-current equation contribute derivatives via `Reading::add_conductances`.
The JFET reads both limited gate junctions plus drain-source voltage between them, with the channel reading gate-source and drain-source voltages.
No hand-written derivatives remain in either model, and legacy junction helpers returning value-slope pairs have been removed.
One-time verification compared each replacement against the hand-written kernel across 6,000 Gummel-Poon evaluations (both polarities, resistance modes, substrate types, high-injection branches, and excess-phase modes).
Verification also covered 504 JFET evaluations across both channels, three doping tails, optional series resistances, and all operating regions.
Residuals, conductances, and capacitances shift by at most 3.5 ε of the row maximum while charges stay identical, and every test, ngspice check, and law-suite assertion passes.
One-time verification replaced keeping a permanent reference test so hand-written kernels do not persist as duplicate implementations.
Callgrind counts on Gummel-Poon rise from 2,203 to 2,370 at defaults (+7.6%) and from 4,500 to 4,581 with every feature (+1.8%).
Gummel-Poon with excess phase rises from 4,926 to 5,053 (+2.6%), while JFET rises from 2,409 to 2,477 at defaults (+2.8%) and 3,165 to 3,237 full (+2.3%).
Alternating Criterion timings shifted by -15% to +12%, well inside build-layout noise (unchanged VDMOS code shifted by +9% to +10% between the same builds).
The small instruction increase is submitted to the owner; consistent dual-number kernels remain unless the owner decides otherwise.

### 1.4 The short hand-written laws

The memristor, switch, photocell, VDMOS quasi-saturation, and passive voltage coefficients move to dual numbers under the same acceptance criteria as step 1.3.
Measurements confirm little runtime impact across these small laws.

**Result** (17 November): each device implements a `Dual<R, 2>` law over terminal voltage and control signal (control voltage, current, state, or light).
Control-only sub-laws (conductance, state windowing, and drain-resistance rise) evaluate on `Dual<R, 1>` and compose via `Dual::apply`, while varying series resistance uses `Series::add_varying`.
Photocell settled response retains a custom derivative rule (`γ · response / √(E² + d²)`) via `Dual::chain`, staying finite regardless of darkness where direct differentiation would divide by zero.
One-time verification over 676 DC and transient evaluations across all regions and bounds left residuals, charges, and capacitances unchanged.
Conductances shifted by at most 2.9 ε of the row maximum, and all tests pass unchanged.
Callgrind instruction counts rise by between 1 instruction (resistor) and 25 instructions (voltage-controlled memristor), at most 1.3%.

## Step 2: every parameter derivative of a device in one evaluation

### 2.1 Number types

- `Gradient<const K: usize>`: stores a value and derivatives with respect to `K` parameters, implementing `Real` with `chain` scaling all `K` slots by the slope (replacing `Gradient` as `Gradient<1>`).
- `Hyper<const K: usize>`: extends `Gradient<K>` for self-heated devices with one temperature derivative, `K` parameter derivatives, and `K` mixed derivatives (replacing `Hyper` as `Hyper<1>`).
- `Heating` preserves its mapping: `f64 → Gradient<1>` and `Gradient<K> → Hyper<K>`.
- Device kernels need no edits because kernels are already generic over `R`.

### 2.2 Chunk twins

- A chunk twin bundles a device with up to `K` named parameters, seeding slot `i` on parameter `i` during lifting.
    One `prepare` and one `evaluate` call compute all residual, charge, conductance, and capacitance derivatives across all `K` parameters.
- `jolty-models/src/group.rs`: `GroupRun::twin` accepts chunk parameter names, lifts them together, and returns a `Twin` storing `K` slots per residual, charge, and matrix entry.
- `jolty-engine/src/derivative.rs`: groups named parameters by device and chunks them so `residual_dot` and related helpers return `K` values while `pull_back` loops over chunks rather than single parameters.
    Public engine APIs (`Simulator::jvp`, `vjp`, `pull_back`) remain unchanged.
- Devices with more named parameters than the maximum chunk width split across multiple twins.

**Result** (17 November): `Gradient<K>` and `Hyper<K>` carry `K` derivatives, subsuming `Gradient<1>` and `Hyper<1>`, while `Seeds` specifies per-number seed rates (`Seeds::units` for unit seeds).
`GroupRun::twin` lifts device parameters at the smallest width in `{1, 2, 4, 8}` fitting the chunk and returns all derivatives along each seed.
In `crates/jolty-engine/src/twins.rs`, `Twins` packs up to eight parameter directions per device into each twin, used uniformly across `derivative.rs`, sensitivity, mismatch, and objective passes.
Since 18 November, `Twins` groups columns by device in a single linear pass instead of scanning prior chunks (avoiding roughly 10¹² steps on a million-weight crossbar VJP).
Duplicate parameter paths now merge once in `Seeds` (summing rates) rather than separately in both engine and `lift`.
Parameter lifting uses `Lift` (`crates/jolty-models/src/lift.rs`, derived via `jolty-derive`) instead of JSON round-tripping, which previously monomorphized serde readers for every parameter type at every width.
At single width (`ecc932d`), `jolty-models` codegen at `opt-level = 1` dropped from 392 s to 68 s (1.85M to 1.49M LLVM IR lines).
Across all four widths (`61fa6e8`), `Lift` reduced codegen time from 689 s (via JSON) down to 159 s.

### 2.3 Chunk sizes

Monomorphized widths are `1`, `2`, `4`, and `8`, choosing the smallest width fitting each chunk (for example, width 1 for a crossbar resistor weight, width 2 for transistor width and length).
Maximum chunk width is treated like any engine threshold, selected from measured break-evens via the alternating timing protocol rather than picked arbitrarily.

**Result** (18 November): Callgrind instruction counts via `instructions.rs` (Valgrind 3.25.1, machine D, glibc 2.27 `libm`, `rustc 1.100.0`, `debug = true`) measure one twin evaluation at each width.
Counts reflect fixing `diode_twin` internal node bias from `"anode"` (1 mV) to `"junction"` (0.69 V forward):

| Twin | 1 | 2 | 4 | 8 | 8 against eight of 1 |
| --- | --- | --- | --- | --- | --- |
| BSIM4 | 27,077 | 35,861 | 53,350 | 92,128 | 2.4 times fewer |
| Diode (0.69 V forward) | 1,090 | 1,360 | 1,662 | 2,482 | 3.5 times fewer |
| Resistor, with voltage coefficients | 430 | 607 | 772 | 1,170 | 2.9 times fewer |

Twin evaluation splits into fixed primal/control overhead plus per-seed cost: roughly 9,293 instructions for BSIM4, 199 for the diode, and 106 for the resistor.
BSIM4 totals sit higher under glibc 2.27 `libm` than under glibc 2.39, while per-seed increments remain identical.
Every width from 2 upward costs less than the same count of 1-wide twins, showing chunking always pays.
Beyond width 8, per-seed savings flatten while stack footprint grows, and width 8 already covers the six to eight mismatch parameters typical of PDK devices.
Padding matters when per-seed cost is large: five BSIM4 parameters take 92,128 instructions in one 8-wide twin versus 80,427 in a 4-wide plus 1-wide pair.
By contrast, five diode parameters cost 2,482 instructions in one 8-wide twin versus 2,752 split, so the policy stays one smallest-fitting twin per chunk unless whole-gradient measurements justify splitting.
Instruction counts are deterministic and independent of machine load.

**Width 16 timings** (18 November, `d98de4c` with `Seeds::MOST = 16` in an unkept copy): Criterion medians from `evaluation.rs` on one pinned core of machine D (host load 259 to 269 across 96 cores):

| Twin | 1 | 2 | 4 | 8 | 16 | 16 against two of 8 |
| --- | --- | --- | --- | --- | --- | --- |
| BSIM4 | 3.691 µs | 4.075 µs | 6.603 µs | 10.92 µs | 20.42 µs | 6.5% less |
| Diode (0.69 V forward) | 129.0 ns | 156.7 ns | 188.2 ns | 281.2 ns | 533.1 ns | 5.2% less |
| Resistor, with voltage coefficients | 59.54 ns | 60.81 ns | 88.44 ns | 149.4 ns | 228.2 ns | 24% less |

A 16-wide twin beats two 8-wide twins on all three devices, marginally on BSIM4 and diode (with overlapping intervals) and by 24% on the resistor.
Supporting width 16 lengthens `jolty-models` release builds from 172.4 s to 211.3 s (+23%) while benefiting only devices seeded along more than eight directions in one derivative (above PDK mismatch counts).
Width 8 stays the maximum (the owner, 20 November).
Criterion runs above use a single build on a different host from §2.6 alternating builds, so comparisons stay strictly within each table.

### 2.4 The JVP without chunks

A parameter tangent specifies one directional derivative.
Seeding each parameter's slot with its tangent component computes `(∂F/∂p)·t` in one `Gradient<1>` evaluation regardless of how many device parameters the tangent moves, replacing one twin per parameter.

**Result** (17 November): `Simulator::jvp` packs each tangent's components on a device into one seed column (`Simulator::tangent_twins`), sharing up to eight tangents per twin.
Each tangent therefore costs one seed per device regardless of how many parameters on that device move.

### 2.5 Newton and memory

- Newton evaluates on `R = f64` with local voltage dual numbers; twins are separate instances built only on derivative paths, keeping parameter slots completely off Newton iterations.
- Transient VJP evaluates each chunk twin at recorded steps during the backward pass, adding no memory beyond the twins because waveforms are already retained.

### 2.6 Done when

- Every chunk-twin slot matches a single-parameter twin bit-for-bit, executing identical operations in identical order (Rust does not silently contract multiply-add).
- Every derivative test passes bit-for-bit unchanged, the dot-product test holds, and finite differences agree.
- Per-device JVP twins match the sum of per-parameter twins to floating-point rounding.
- The alternating timing protocol shows a VJP over `n` device parameters scaling with chunk count rather than `n`.

**Result** (17 November): every kernel gradient test verifies bit-for-bit parity between multi-parameter twins (chunks of 8 and 3, padded to 4) and single-parameter twins, including self-heated kernels (`testing.rs`, `assert_chunks_match_seeds_alone`).
A ten-parameter diode VJP (two chunks) pulls cotangents back bit-for-bit identically to single-parameter VJPs across operating points and every transient step (`a_vjp_in_chunks_pulls_back_to_each_parameter_as_a_vjp_along_it_alone`).
Directional JVP matches combined column derivatives within dot-product bounds across every analysis (`the_jvp_and_the_vjp_are_transposes_of_each_other_at_every_analysis`), and all derivative tests pass unchanged.
Each chunk performs one evaluation, with instruction counts listed in §2.3.
Criterion timings on one pinned core of machine D (`5efde61`, 18 November, `evaluation.rs`) confirm speedups across widths 1, 2, 4, and 8.
BSIM4 evaluates in 3.057, 4.132, 5.561, and 8.884 µs across widths 1, 2, 4, and 8 (2.75× faster than eight 1-wide twins).
The 0.69 V diode evaluates in 109.9, 168.2, 159.8, and 266.8 ns (3.30× faster at width 8).
The resistor evaluates in 56.25, 54.92, 86.44, and 133.2 ns (3.38× faster at width 8).
**Whole VJP, alternating protocol** (18 November, `a3c1d6c`, release build with `lto = "thin"` and `codegen-units = 1`): compared `chunk=8` (`Seeds::MOST`) against `chunk=1` (`chunks(1)` in `crates/jolty-engine/src/twins.rs`).
Nine alternating AB/BA rounds of 5 repetitions (45 samples per arm) ran pinned to one core of machine D (AMD EPYC 9B14, host load 93.39 to 93.84 across 96 cores).
All 25 `(case, n)` pairs matched with 0 mismatched bits (`f64::to_bits`).
Medians and exact 95% binomial rank intervals follow (`total` covers `vjp` + `pull_back`; transients use 200 fixed steps with `truncation_tolerance = ∞` for identical grids):

| Case | `n` | `chunk=1` total median [95% CI] | `chunk=8` total median [95% CI] | Total speedup | `chunk=1` `pull_back` median [95% CI] | `chunk=8` `pull_back` median [95% CI] | Pull speedup | Separated? |
| --- | ---: | --- | --- | ---: | --- | --- | ---: | --- |
| `bsim4_op` | 1 | 29.70 µs [29.49 to 30.33] | 30.02 µs [29.62 to 30.38] | 0.99× | 16.23 µs [16.08 to 16.56] | 16.20 µs [16.06 to 16.47] | 1.00× | no / no |
| `bsim4_op` | 2 | 52.71 µs [52.18 to 53.01] | 37.63 µs [37.25 to 38.65] | 1.40× | 29.17 µs [29.04 to 29.36] | 20.77 µs [20.56 to 21.01] | 1.40× | yes / yes |
| `bsim4_op` | 4 | 99.48 µs [98.92 to 102.00] | 57.28 µs [56.43 to 58.40] | 1.74× | 55.70 µs [55.38 to 56.15] | 31.08 µs [30.81 to 31.45] | 1.79× | yes / yes |
| `bsim4_op` | 8 | 196.25 µs [192.64 to 198.48] | 96.02 µs [95.46 to 96.54] | 2.04× | 109.69 µs [108.96 to 111.13] | 52.83 µs [52.35 to 53.34] | 2.08× | yes / yes |
| `bsim4_op` | 16 | 382.77 µs [377.51 to 399.33] | 177.01 µs [175.64 to 181.08] | 2.16× | 215.64 µs [213.21 to 226.76] | 97.70 µs [96.52 to 99.16] | 2.21× | yes / yes |
| `bsim4_ac` | 1 | 61.22 µs [60.03 to 62.96] | 61.12 µs [59.81 to 61.91] | 1.00× | 38.49 µs [38.04 to 39.43] | 38.33 µs [38.00 to 38.93] | 1.00× | no / no |
| `bsim4_ac` | 2 | 83.91 µs [83.12 to 87.76] | 62.83 µs [62.42 to 63.43] | 1.34× | 51.97 µs [51.57 to 53.47] | 40.49 µs [40.29 to 40.85] | 1.28× | yes / yes |
| `bsim4_ac` | 4 | 130.47 µs [129.51 to 134.22] | 90.74 µs [89.14 to 92.89] | 1.44× | 78.13 µs [77.84 to 80.77] | 54.77 µs [54.32 to 56.82] | 1.43× | yes / yes |
| `bsim4_ac` | 8 | 228.06 µs [223.45 to 232.60] | 132.81 µs [130.98 to 136.85] | 1.72× | 135.38 µs [132.36 to 137.30] | 79.03 µs [78.50 to 80.71] | 1.71× | yes / yes |
| `bsim4_ac` | 16 | 421.26 µs [415.78 to 432.73] | 217.90 µs [213.59 to 234.25] | 1.93× | 244.03 µs [240.22 to 247.58] | 125.74 µs [123.94 to 128.89] | 1.94× | yes / yes |
| `bsim4_tran` | 1 | 2.842 ms [2.835 to 2.864] | 2.768 ms [2.761 to 2.785] | 1.03× | 1.259 ms [1.249 to 1.265] | 1.236 ms [1.231 to 1.252] | 1.02× | yes / no |
| `bsim4_tran` | 2 | 3.528 ms [3.517 to 3.548] | 3.022 ms [3.009 to 3.045] | 1.17× | 1.924 ms [1.920 to 1.942] | 1.482 ms [1.476 to 1.490] | 1.30× | yes / yes |
| `bsim4_tran` | 4 | 4.909 ms [4.893 to 4.957] | 3.464 ms [3.443 to 3.535] | 1.42× | 3.271 ms [3.257 to 3.312] | 1.896 ms [1.880 to 1.932] | 1.73× | yes / yes |
| `bsim4_tran` | 8 | 7.714 ms [7.689 to 7.798] | 4.290 ms [4.258 to 4.359] | 1.80× | 6.023 ms [5.994 to 6.080] | 2.661 ms [2.639 to 2.692] | 2.26× | yes / yes |
| `bsim4_tran` | 16 | 13.342 ms [13.302 to 13.425] | 6.453 ms [6.370 to 6.571] | 2.07× | 11.536 ms [11.496 to 11.593] | 4.711 ms [4.685 to 4.828] | 2.45× | yes / yes |
| `diode_op` | 1 | 7.61 µs [7.44 to 7.85] | 7.49 µs [7.42 to 7.56] | 1.02× | 3.83 µs [3.75 to 3.90] | 3.76 µs [3.72 to 3.80] | 1.02× | no / no |
| `diode_op` | 2 | 11.23 µs [11.10 to 11.39] | 8.78 µs [8.70 to 9.05] | 1.28× | 5.76 µs [5.71 to 5.86] | 4.53 µs [4.45 to 4.67] | 1.27× | yes / yes |
| `diode_op` | 4 | 19.16 µs [18.90 to 19.94] | 11.35 µs [11.19 to 11.52] | 1.69× | 10.35 µs [10.13 to 10.75] | 6.18 µs [6.05 to 6.27] | 1.67× | yes / yes |
| `diode_op` | 8 | 37.00 µs [36.36 to 39.01] | 17.90 µs [17.60 to 18.04] | 2.07× | 19.58 µs [19.23 to 19.71] | 9.69 µs [9.63 to 9.90] | 2.02× | yes / yes |
| `diode_op` | 10 | 44.52 µs [43.95 to 45.29] | 22.70 µs [22.50 to 22.99] | 1.96× | 23.64 µs [23.27 to 24.13] | 12.09 µs [12.01 to 12.27] | 1.96× | yes / yes |
| `diode_tran` | 1 | 836.14 µs [829.76 to 845.79] | 820.91 µs [816.39 to 828.02] | 1.02× | 279.22 µs [275.61 to 295.01] | 269.77 µs [268.09 to 272.86] | 1.04× | yes / yes |
| `diode_tran` | 2 | 900.80 µs [890.18 to 909.10] | 860.15 µs [851.87 to 864.28] | 1.05× | 352.13 µs [333.15 to 355.18] | 306.21 µs [301.98 to 311.35] | 1.15× | yes / yes |
| `diode_tran` | 4 | 1.008 ms [1.002 to 1.016] | 892.88 µs [889.85 to 901.83] | 1.13× | 463.43 µs [458.52 to 467.70] | 339.12 µs [335.22 to 352.10] | 1.37× | yes / yes |
| `diode_tran` | 8 | 1.296 ms [1.288 to 1.311] | 1.015 ms [1.010 to 1.044] | 1.28× | 737.22 µs [734.04 to 745.84] | 465.95 µs [454.59 to 472.85] | 1.58× | yes / yes |
| `diode_tran` | 10 | 1.441 ms [1.430 to 1.461] | 1.121 ms [1.115 to 1.129] | 1.28× | 870.32 µs [865.99 to 887.12] | 576.64 µs [567.59 to 582.24] | 1.51× | yes / yes |

At `n = 1`, both binaries execute identical `Gradient<1>` evaluations; the 2 to 4% gap on `bsim4_tran` and `diode_tran` reflects LTO code-layout variance between separately linked binaries.
From `n = 2` upward, every total and `pull_back` 95% interval separates cleanly in favor of `chunk=8`.
BSIM4 reaches 2.07× to 2.16× total speedup (2.21× to 2.45× on `pull_back`) at 16 parameters.
The diode reaches 1.28× to 2.07× total speedup (1.51× to 2.02× on `pull_back`) at 8 to 10 parameters.
Past a full chunk of 8, `diode_op` (1.96× vs 2.07×) and `diode_tran` (`pull_back` 1.51× vs 1.58×) dip slightly as parameters 9 and 10 spill into a width-2 chunk.
All acceptance criteria for step 2 are complete.

## Step 2b: AC's VJP exact and independent of the number of parameters

AC's system matrix `G + jωC` depends on parameters both directly and through the operating point `(∂J/∂x)·(dx₀/dp)`.
Previously the operating-point term used a per-parameter central difference because kernels lacked second derivatives in `x` (`models.md` 6).
The operating-point term moves to the adjoint side exactly:

```text
λᵀ (∂J/∂x · dx₀/dp) v  =  gᵀ dx₀/dp,     g = ∂/∂x (λᵀ J(x) v)
gᵀ dx₀/dp  =  -(J₀⁻ᵀ g)ᵀ ∂F₀/∂p
```

Each device contributes its share of `g` as the gradient of one scalar over its local unknowns.
One transposed solve at the operating point paired with chunk-twin `∂F₀/∂p` then produces the term for every parameter simultaneously.

The cost of exact AC VJP is requiring kernels to read voltages as `R` rather than `f64` through `Local` (carrying an optional unknown tangent) so twins can seed voltage directions.
Modifying voltage reads across every kernel made step 2b conditional until AC gradients over many parameters became a priority.

**Done when:** AC VJP eliminates finite differences, adds only one transposed solve regardless of parameter count (verified across 1 to 1,000 parameters), matches previous central differences within truncation error, and passes the frequency dot-product test.

**Result** (17 November): every kernel reads unknowns via `Local::unknown<R>` and `Local::delayed_unknown<R>`, compiled Verilog-A kernels regenerate with the same reads, and `jolty-expressions` computes second derivatives without allocation (`Expression::along`, `Unary::curvature`, `Binary::curvatures`, `Points::interpolate`).
The `Real` trait defines `SEEDS`, `unit`, `seeded`, `derivative`, and `with_value`, folding seeding helpers away at compile time when `R::SEEDS == 0` (plain `f64` Newton runs).
Self-heated kernels (`HeatedKernel`) propagate second derivatives through temperature (`Hyper<K>`) and re-prepare whenever a twin seeds unknowns, fixing a latent bug where ambient `ΔT = 0` skipped preparation and returned zero thermal derivatives.
In addition, `bsim4/charges.rs` preserves `vdseff_cv`'s drain derivative at `vds == 0` via `.with_value(0.0)` instead of zeroing the dual number, and `photocell::settled` keeps the chain rule through `illuminance`.
In `jolty-engine`, `BIAS_STEP`, `Bias`, `biases`, and `bias_derivative` have been deleted.
Forward mode (`Simulator::frequency_tangents`) uses sparse `entries_along` for deterministic directional `(∂G/∂x · s, ∂C/∂x · s)` accumulation.
Adjoint mode (`Simulator::pull_frequencies`) computes the operating-point contribution via `entries_cotangents` over sorted unique nonlinear matrix slots plus one transposed DC solve per active cotangent, skipping DC solves entirely on linear circuits.

Every model in `Library::standard()` passes `every_kernel_differentiates_its_conductances_with_respect_to_its_unknowns` against Richardson-extrapolated central differences with zero allocations on Newton's path (`a_kernels_evaluation_allocates_nothing`).
Compiled Verilog-A kernels pass the identical verification in `crates/jolty-veriloga/tests/twins.rs`.
Following step 2b, `Real` and `Kernel` were decoupled from `serde` and `schemars`, confining `Deserialize`, `Serialize`, and `JsonSchema` to `Parameters<f64>` in `Differentiable<K>` and `Thermal<K>`.
Deleting `Written` and serde/schema impls on `Gradient<K>` and `Hyper<K>` was validated by library-wide `Lift` vs `serde` path parity tests across all 30 analog models and all 8 compiled Verilog-A kernels and fixtures.
On `loaded_amplifier` over four frequencies, exact AC VJP matches `61fa6e8`'s 1 µV central difference within `9.57e-10` relative error (at `VDD.waveform.dc`) and `0` on `RG`, `CL`, and `CG`.
The AC dot-product test holds to `7.15e-17` (`0.32 ε`), keeping the worst residual across all analyses at `7.38e-16` (`crates/jolty-engine/tests/sensitivity.rs`).
New regression tests cover self-heated AC (both hot and exact ambient `ΔT = 0`), nonlinear behavioral AC, linear AC, bit-identical determinism across 1, 2, and 4 threads, and a 504-parameter nonlinear amplifier plus resistive ladder.

Alternating release timings (`lto = "thin"`, `codegen-units = 1`) compared `61fa6e8` against `ab2c08c` on one pinned core (`cpu=0`) of machine E (host load 122 to 132 across 256 logical CPUs).
Three alternating rounds of 11 repetitions (33 samples per revision) ran on a 500-rung resistive ladder driven by `loaded_amplifier` over 5 frequencies with 1 cotangent (max relative difference `9.743e-8` at `P = 1,000`, index 514):

| Parameters `P` | `61fa6e8` pullback median (range) | Step 2b pullback median (range) | Pullback speedup | Total (`vjp` + `pull_back`) median |
| --- | --- | --- | --- | --- |
| 1 | 0.423 ms (0.391 to 0.470) | 0.316 ms (0.301 to 0.348) | 1.3× | 0.763 → 0.644 ms (1.2×) |
| 10 | 1.194 ms (1.115 to 1.287) | 0.343 ms (0.319 to 0.426) | 3.5× | 1.544 → 0.695 ms (2.2×) |
| 100 | 10.064 ms (9.620 to 10.857) | 0.515 ms (0.476 to 0.567) | 19.5× | 10.501 → 0.999 ms (10.5×) |
| 1,000 | 107.795 ms (100.857 to 117.648) | 2.165 ms (1.970 to 2.657) | 49.8× | 109.933 → 3.830 ms (28.7×) |

Callgrind instruction counts from `crates/jolty-models/benches/instructions.rs` (gungraun, Valgrind 3.25.1) against `61fa6e8` on `glibc 2.39` show at most +2 instructions (+0.07%) on Newton `f64` kernels.
VBIC drops by 1 to 4 instructions and the photocell drops by 10 instructions (-0.49%).
Re-measuring on machine D (`glibc 2.27`, `ff752db`) after fixing `diode_twin` bias from `"anode"` (1 mV) to `"junction"` (0.69 V) yields `27,077 / 35,861 / 53,350 / 92,128` on `bsim4`.
On the same host, `diode` counts `1,090 / 1,360 / 1,662 / 2,482` and `resistor` counts `430 / 607 / 772 / 1,170`.

## Step 3, deferred: one shared compile-time differentiation pass

As described in `models.md` 5, step 3 would generate derivative code before run time containing only nonzero derivatives needed by outputs, matching OpenVAF's approach for Verilog-A.
jolty's Verilog-A emitter already derives per-variable dependencies; step 3 would share that engine between the emitter and a `#[kernel]` macro for hand-written Rust kernels.

**Taken up only if** the measurement below shows derivative overhead remains a substantial share of any runtime gap against ngspice after steps 1 and 2.
A Rust-subset proc macro adds real maintenance and compile-time cost, so step 3 must be justified by data.

## Step 4, deferred: compiler-level automatic differentiation

Compiler-level automatic differentiation (`std::autodiff` via Enzyme) would differentiate optimized plain `f64` kernels directly.
As of 17 November 2026, `std::autodiff` remains nightly-only and experimental (tracking issue rust-lang/rust#124509, labeled `B-experimental`, no accepted RFC) and requires fat LTO.
jolty targets stable Rust.

**Taken up only once** stable Rust supports `std::autodiff`, benchmarked against steps 1 to 3 before migrating any kernel.

## The measurement, planned and held

Measurements deciding step 3, held pending owner go-ahead:

1. Benchmark BSIM4 at one card and bias in a tight loop comparing jolty against ngspice `BSIM4v5load` on machine B (avoiding machine A's HPET distortion), recording host load.
1. Run the same loop on jolty with dual slots stripped (values only) to separate derivative overhead from base model evaluation cost.
1. Profile jolty evaluation by function across dual arithmetic, branching, temperature-dependent parameters, and matrix stamping.

## Order and dependencies

| Work | Needs | Touches |
| --- | --- | --- |
| 1.2 VBIC | Nothing | `vbic.rs` |
| 1.3 Gummel-Poon, JFET | Nothing; the owner's call if it costs speed | `bjt.rs`, `jfet.rs` |
| 1.4 Short laws | Nothing | `memristor.rs`, `switch.rs`, `photocell.rs`, `vdmos.rs`, `passive.rs` |
| 2.1 to 2.6 | The derivatives refactor (`agentic.md` 12b), on `main` | `real.rs`, `group.rs`, `derivative.rs`; no kernel |
| 2b | Step 2; a need for AC gradients over many parameters | Every kernel's voltage reads, `Local` |
| 3 | The measurement, after steps 1 and 2 | The Verilog-A emitter, a new crate |
| 4 | Rust stabilizing `std::autodiff` | Every kernel |
