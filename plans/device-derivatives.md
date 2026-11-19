# Device derivatives: implementation plan

Status (18 November 2026): steps 1, 2 and 2b done, on `main`; the largest chunk measured against a width of 16 (2.3), its choice put to the owner; a whole VJP over a device's parameters timed with the alternating protocol (2.6); steps 3 and 4 deferred; the measurement that would decide step 3 held for the owner.
This plan covers the first layer of jolty's derivatives: each device's own derivatives, from a device's law to the engine's JVP and VJP.

## Why

Exact derivatives are one of jolty's founding rules: Newton converges on them, and every gradient (sensitivity, mismatch, the JVP and VJP, a loss for training) is built from them.
Each device kernel gives two kinds: with respect to its voltages, `∂F/∂x`, at every Newton iteration; and with respect to its parameters, `∂F/∂p`, whenever a gradient is asked for.

Two debts remained in how kernels computed them, which steps 1 and 2 have paid:

1. **Not every kernel is written the same way.**
    Most compose small dual numbers at the natural width of each of their laws.
    VBIC runs every intermediate on one dual number nine wide, though each of its laws reads one to three of the nine.
    Gummel-Poon, the JFET, the memristor, the switch, the photocell, VDMOS's quasi-saturation and the passive voltage coefficients carry hand-written derivatives, ported from ngspice: consistent only as long as each was derived without a slip.
1. **Parameter derivatives cost one evaluation per parameter.**
    A twin, the device run on numbers that carry one parameter's derivative, is prepared and evaluated again for each parameter, recomputing values identical in all of them.
    Training a circuit's transistor sizes, or a process kit's mismatch, pays that once per parameter per device at every point; and AC's gradient still takes a central difference per parameter.

The goal: every device's law written once, as plain maths; each quantity carrying the derivatives of only the inputs it reads, decided by the law's structure; custom derivative rules where the maths calls for them (`Dual::chain`); and all of a device's parameter derivatives from one evaluation, never on Newton's path.

## How, at a high level

| Step | What | Status |
| --- | --- | --- |
| 1 | Every kernel on dual numbers composed at the natural width of each law, joined onto its terminals by the chain rule | Done |
| 2 | Parameter derivatives in chunks: one evaluation gives a device's derivatives with respect to several parameters; the JVP by one evaluation per device; Newton untouched | Done |
| 2b | AC's VJP exact, and independent of the number of parameters | Done |
| 3 | One shared compile-time differentiation pass for hand-written and compiled kernels | Deferred |
| 4 | Compiler-level automatic differentiation | Deferred |

Dual numbers stay throughout: they are steps 1 and 2, and the reference every other method is tested against.
Steps 3 and 4 would replace them only where a measurement shows they cost too much, or once Rust can differentiate a program itself.

## What it changes

Each stage's "done when" measures its own effect; the table gives what each measured.

| Work | Simulation | Gradients |
| --- | --- | --- |
| 1.2 VBIC composed | Measured: an evaluation 31 to 34% fewer instructions, 18 to 46% less time (below) | Likewise for its twins |
| 1.3 Gummel-Poon, JFET | Measured: an evaluation 2 to 8% more instructions, a time within what a build's layout moves (below) | Likewise for their twins |
| 1.4 Short laws | Measured: an evaluation at most 25 instructions more (below) | None |
| MOSFET circuits, under all of step 1 | Unchanged: their kernels are composed already | Unchanged |
| 2 Chunk twins | None, by design: Newton is never touched; every kernel's instruction count unchanged but VDMOS's, 0.9% more, from compiler inlining and codegen choices | Measured: a twin of eight evaluates in 2.4 (BSIM4) to 3.5 (the diode, properly biased at 0.69 V) times fewer instructions than eight twins of one on machine D (below); whole `Simulator::vjp` + `pull_back` speeds up 1.28× to 2.16× total (1.51× to 2.45× `pull_back`) on chunk 8 vs chunk 1 with bit-identical results across all 25 cases (2.6); a JVP evaluates each device once per tangent whatever the number of its parameters |
| 2b AC exact | At most +2 instructions (+0.07%) on Newton's `f64` kernels, −4 on VBIC and −10 on the photocell (`instructions.rs`, below) | Measured: AC VJP pullback 49.8× faster at 1,000 parameters on machine E (107.8 to 2.17 ms; total `vjp` + `pull_back` 28.7× faster, 109.9 to 3.83 ms); agrees with `61fa6e8`'s central difference within 9.74e-8 at 1,000 parameters (9.57e-10 on `loaded_amplifier`); AC dot-product residual 7.15e-17 |

## Step 1: every kernel composed on dual numbers

### 1.1 Where each kernel stands

Read from `crates/jolty-models/src/kernel/` on 17 November 2026:

| Kernel | Derivatives in x | Work |
| --- | --- | --- |
| BSIM4, BSIM3, MOSFET levels 1 to 3, semi-empirical MOSFET | Composed: `Dual<R, 3>` for the channel (vgs, vds, vbs), `Dual<R, 1>` per junction | None |
| Diode, junctions | Composed: `Dual<R, 1>` | None |
| VDMOS | Composed: `Dual<R, 2>` for the channel, `Dual<R, 1>` for its quasi-saturation (1.4) | None |
| Verilog-A compiled kernels | Each variable carries the probes it depends on and no others (`crates/jolty-veriloga/src/emit.rs`) | None |
| Behavioral sources and capacitors | `jolty-expressions`' tape: values forward, gradients backward, the right mode for an expression of many inputs | None |
| Linear devices | Their entries, stamped once a run | None |
| VBIC | Composed: `Dual<R, 1>` to `Dual<R, 3>` for each law (1.2) | None |
| Gummel-Poon (`bjt.rs`), JFET (`jfet.rs`) | Composed: `Dual<R, 1>` per junction, `Dual<R, 2>` for the intrinsic transistor's laws and the JFET's channel, `Dual<R, 3>` for the base resistance's current (1.3) | None |
| Memristor, switch, photocell, passive voltage coefficients | Composed: `Dual<R, 2>` of the voltage across and the control, each law of the control alone on `Dual<R, 1>` (1.4) | None |

### 1.2 VBIC composed

VBIC's equations take nine voltages (`VARIABLES`: base-emitter, extrinsic base-emitter, base-collector, extrinsic base-collector, the parasitic transistor's junctions, the internal resistances' drops), and evaluate everything on `Dual<R, 9>`.

1. Map each of its laws to the voltages it reads: each junction current and charge reads its own junction's voltage; the transport current and the normalized base charge `qb` read the intrinsic base-emitter and base-collector voltages; each internal resistance's current reads its own drop and the junction voltages its modulation reads.
1. Evaluate each law on `Dual<R, k>` at its own width `k`, from one to three.
1. Chain each result onto the nine variables once, at the end, as BSIM4 does (`bsim4.rs`: `gradient: [gate, -(gate + drain + bulk), bulk]`); the self-heated variant keeps its temperature through `Heating`, unchanged.

**Done when:** every derivative of VBIC agrees with today's nine-wide one to rounding (the operations' order changes, so not bit for bit); its tests against ngspice pass unchanged; the law suite holds; and the timing protocol (`scripts/bench --compare A,B`) shows the composed VBIC no slower than today's, on machine B.

**Result** (17 November): each law computes at its own width and joins the others through `Oriented::reading`, a law that reads some of a transistor's variables, and `Dual::embed`, a number of a few variables placed among more, both in `kernel/parts.rs` and `kernel/dual.rs` for the kernels that follow; the weak avalanche current is a current of its own.
Over 1,200 evaluations, both polarities, both base charges, both smoothings, warm and at nominal temperature, with and without an excess phase, the residuals and conductances move by at most 0.9 ε of their row's largest and the charges and capacitances not at all; every VBIC test, the ngspice comparisons and the law suite pass unchanged.
Counted by `crates/jolty-models/benches/instructions.rs` (gungraun, through Valgrind 3.25.1's Callgrind), one evaluation, the bench's own work, the same in both builds, included and the case's preparation and drop left out, against `main` (`8fb1e4b`): 7,184 instructions rather than 10,879 at its defaults (−34.0%), 8,838 rather than 12,830 with every part of its model (−31.1%), 9,290 rather than 13,376 with an excess phase (−30.5%).
Timed by `evaluation.rs`, criterion, the two builds' runs alternating twice on one pinned core of an AMD EPYC 7B12 at load 2 to 5 (machine B unreachable): 819 to 829 ns rather than 1.42 to 1.51 µs at its defaults, 1.02 to 1.26 µs rather than 1.53 to 1.63 µs with every part, 1.01 to 1.05 µs rather than 1.59 to 1.66 µs with an excess phase.

### 1.3 Gummel-Poon and the JFET on composed dual numbers

A first port of both to dual numbers ran 7 to 18% slower and was stopped (`quality.md` 6); how wide its dual numbers were is not recorded.

1. Gummel-Poon: each junction's current, charge and leakage on `Dual<R, 1>` of its own voltage; the transport current, `qb` and the base resistance's modulation on `Dual<R, 2>` (base-emitter, base-collector); the excess phase as today, on the dual numbers it delays.
1. JFET: its channel on `Dual<R, 2>` (gate-source, drain-source), each gate junction on `Dual<R, 1>`.
1. Measure each against today's hand-written kernel with the timing protocol, on machine B.

**Done when:** no hand-written derivative left in either; their tests against ngspice pass unchanged; the law suite's Jacobian check holds; and either the protocol shows no slowdown it can resolve, or the measured cost is put to the owner, who decides whether consistency is worth it.
Until then the hand-written versions stay in a test as a reference the new ones must agree with everywhere.

**Result** (17 November): each law computes at its own width through `Oriented::reading`, as VBIC's do.
Gummel-Poon's variables are its two junctions, limited, then the drop across its base resistance and the external base-collector voltage, unlimited, so that the base resistance's current, `drop / r`, is a law of three variables and the external depletion a law of one; the base resistance's shape, a law of the base current alone, runs on one variable and joins the others by the chain rule (`Dual::apply`); the excess phase's filter and the base current's own equation add their derivatives through `Reading::add_conductances`.
The JFET's variables are its two gate junctions, limited, then the drain-source voltage between them, which its channel reads with the gate-source voltage.
No derivative of either is written by hand, and the junction helpers that gave such kernels a value with its slope are gone.
Each was compared once with the hand-written kernel it replaces, over 6,000 evaluations of Gummel-Poon (both polarities, each form of the base resistance and of the series resistances, a vertical and a lateral substrate or none, both high-injection branches, with and without an excess phase) and 504 of the JFET (both channels, three doping tails, with and without series resistances), across every region: the residuals, conductances and capacitances move by at most 3.5 ε of their row's largest, the charges not at all; every test, the ngspice comparisons and the law suite pass unchanged.
That comparison stands for the reference test, which was not kept: the hand-written kernels would otherwise survive as a second way to compute the same numbers.
Counted as VBIC was, against the hand-written kernels on `main`: Gummel-Poon 2,370 instructions rather than 2,203 at its defaults (+7.6%), 4,581 rather than 4,500 with every part of its model (+1.8%), 5,053 rather than 4,926 with an excess phase (+2.6%); the JFET 2,477 rather than 2,409 at its defaults (+2.8%), 3,237 rather than 3,165 with every part of its model (+2.3%).
Their times, timed as VBIC's were, moved by −15 to +12%, within what a build's layout moves a kernel left unchanged: VDMOS's, its code the same in both builds, by +9 to +10%.
The few instructions more are put to the owner: the consistency is kept unless the owner decides otherwise.

### 1.4 The short hand-written laws

The memristor, the switch, the photocell, VDMOS's quasi-saturation and the passive voltage coefficients, each moved to dual numbers when next touched, under the same acceptance as 1.3.
Small laws, little work, and little speed at stake, which the measurement confirms.

**Result** (17 November): each is a law of the voltage across the device and of what controls it, the control voltage or current, the state or the light, on `Dual<R, 2>`; the conductance it sets, the window that slows the state and the drain resistance's rise are laws of the control alone, on one variable, joined by `Dual::apply`; a series resistance that varies, `Series::add_varying`, takes the resistance as a law of the voltage it follows; and the photocell's settled response keeps its own derivative rule, `γ · response / √(E² + d²)`, through `Dual::chain`, finite however dark, where the held light's own would divide by a light rounded to zero.
Compared once with the hand-written kernels over 676 evaluations, DC and transient, across their regions and bounds, the residuals, charges and capacitances move not at all and the conductances by at most 2.9 ε of their row's largest; every test passes unchanged.
Counted as VBIC was: from 1 more instruction an evaluation, the resistor's, to 25, the voltage-controlled memristor's, at most 1.3%.

## Step 2: every parameter derivative of a device in one evaluation

### 2.1 Number types

- `Gradient<const K: usize>`: a value and its derivatives with respect to `K` parameters, implementing `Real` as `Gradient` does; its `chain` scales all `K` slots by the slope.
    Today's `Gradient` becomes `Gradient<1>`.
- `Hyper<const K: usize>`: the warm version of `Gradient<K>`, for self-heated devices: the temperature's derivative, the `K` parameters', and the `K` mixed ones.
    Today's `Hyper` becomes `Hyper<1>`.
- `Heating` keeps its map: `f64 → Gradient<1>`, `Gradient<K> → Hyper<K>`.
- Kernels change not at all: they are generic over `R` already.

### 2.2 Chunk twins

- A twin carries a device and a chunk of up to `K` of its named parameters, slot `i` seeded on parameter `i` when its parameters are lifted; one `prepare` and one `evaluate` give every residual's, charge's, conductance's and capacitance's derivative with respect to all `K`.
- `jolty-models/src/group.rs`: `GroupRun::twin` takes the chunk's parameter names and lifts them together; its `Twin` stores `K` slots per residual, charge and entry.
- `jolty-engine/src/derivative.rs`: the named parameters are grouped by device, then into chunks; `residual_dot` and its kin return `K` numbers; `pull_back` loops over chunks instead of parameters.
    Its interface (`Simulator::jvp`, `vjp`, `pull_back`) does not change.
- A device with more named parameters than the largest chunk gets several.

**Result** (17 November): `Gradient<K>` and `Hyper<K>` carry `K` derivatives, `Gradient<1>` and `Hyper<1>` the former ones; `Seeds` says each seed's rate on each number, `Seeds::units` a unit seed on each; `GroupRun::twin` lifts a device's parameters with a chunk's seeds at the smallest width of 1, 2, 4 or 8 that holds them, and its `Twin` gives each residual, charge, conductance and capacitance with its derivative along each seed.
The engine's `Twins` (`crates/jolty-engine/src/twins.rs`) hold a derivative's columns, a column a direction of one device's parameters, each device's columns up to eight to a twin, and every pass of `derivative.rs`, the sensitivity, mismatch and objective analyses read them, so no pass sees a twin of one parameter alone.
Since 18 November, `Twins` groups the columns by device in one pass, linear in the columns and the parameters, where it searched every chunk made before for each column, about 10¹² steps for a VJP over a crossbar of a million weights; and `Seeds` merges a path given twice, which moves at the sum of its rates, in one place rather than in both the engine and `lift`.
A twin's parameters are lifted from its instance's plain ones by `Lift` (`crates/jolty-models/src/lift.rs`), derived for every type of parameters by `jolty-derive`, rather than written as JSON and read back, which compiled serde's reader of every type of parameters again at each width: at single width (`ecc932d`), `jolty-models`' code generation at `opt-level = 1` dropped from 392 s to 68 s (1.85M to 1.49M LLVM IR lines), and with all four widths (`61fa6e8`) it dropped from 689 s through JSON to 159 s through `Lift`.

### 2.3 Chunk sizes

A small set monomorphized, 1, 2, 4 and 8; each chunk takes the smallest that holds its parameters: a crossbar resistor's weight takes 1, a transistor's width and length take 2.
The largest is a threshold like every other: set from a measured break-even with the timing protocol, on machine B, its machine and load stated, before this lands; never picked.

**Result** (18 November): counted by `instructions.rs` through Valgrind 3.25.1 Callgrind on machine D (glibc 2.27's `libm`, `rustc 1.100.0`, `debug = true`), one evaluation of a twin at each width along that many of eight numbers given, after fixing `diode_twin`'s internal node bias from `"anode"` (1 mV) to `"junction"` (0.69 V forward):

| Twin | 1 | 2 | 4 | 8 | 8 against eight of 1 |
| --- | --- | --- | --- | --- | --- |
| BSIM4 | 27,077 | 35,861 | 53,350 | 92,128 | 2.4 times fewer |
| Diode (0.69 V forward) | 1,090 | 1,360 | 1,662 | 2,482 | 3.5 times fewer |
| Resistor, with voltage coefficients | 430 | 607 | 772 | 1,170 | 2.9 times fewer |

A twin's evaluation costs a part paid once, its values and its control, and a part paid per seed, about 9,293 instructions of BSIM4's, 199 of the diode's and 106 of the resistor's (BSIM4's counts are higher than those on glibc 2.39 by glibc 2.27's `libm`; the part per seed is the same).
So every width is cheaper than as many twins of one, from two on: chunks always pay.
Past eight, the line through these counts gives each seed less incremental benefit while a twin's stack footprint grows with its width; eight holds the six to eight numbers a kit's mismatch moves on a device, and stays the largest.
Padding costs where the part per seed is large: five of BSIM4's parameters cost 92,128 instructions in a twin of eight against 80,427 in a twin of four and one of one, where the diode's cost 2,482 against 2,752; the rule stays the smallest width that holds a chunk, one twin per chunk, until a measurement of whole gradients asks for more.
Instruction counts are deterministic and independent of machine load.

**Width 16, timed** (18 November, `d98de4c` with `Seeds::MOST` set to 16 in a copy, not kept): one build of `evaluation.rs`, criterion, on one pinned core of machine D at a host load of 259 to 269 over its 96 cores, medians:

| Twin | 1 | 2 | 4 | 8 | 16 | 16 against two of 8 |
| --- | --- | --- | --- | --- | --- | --- |
| BSIM4 | 3.691 µs | 4.075 µs | 6.603 µs | 10.92 µs | 20.42 µs | 6.5% less |
| Diode (0.69 V forward) | 129.0 ns | 156.7 ns | 188.2 ns | 281.2 ns | 533.1 ns | 5.2% less |
| Resistor, with voltage coefficients | 59.54 ns | 60.81 ns | 88.44 ns | 149.4 ns | 228.2 ns | 24% less |

A twin of 16 takes less time than two of 8 on all three, by little on BSIM4 and the diode, whose intervals touch, and a quarter on the resistor; a release build of `jolty-models` takes 211.3 s rather than 172.4 s (+23%), and the gain reaches only a device seeded along more than eight directions in one derivative, more than a kit's mismatch moves.
Eight stays the largest: the recommendation put to the owner, who may choose 16.
These are criterion's own runs, not the timing protocol's alternating builds, and on another server than the times of 2.6, so the two tables are compared within themselves only.

### 2.4 The JVP without chunks

A tangent on the parameters is one direction.
Seeding each parameter's single slot with its component of the tangent gives `(∂F/∂p)·t` for a device in one evaluation of a `Gradient<1>` twin, however many of its parameters the tangent moves; today that sum takes one twin per parameter.

**Result** (17 November): `Simulator::jvp` makes a column of each tangent on each device it moves, the tangent's components on that device's parameters its one seed (`Simulator::tangent_twins`), so several tangents share a device's twins, eight to a twin, and a tangent costs a device one seed however many of its parameters it moves.

### 2.5 Newton and memory

- Newton evaluates with `R = f64` and its own small dual numbers; twins are separate instances, built and evaluated only on a derivative's path, so no Newton iteration ever carries a parameter slot.
- A transient's VJP evaluates each chunk twin at each recorded step as its backward pass reaches it; the waveforms are kept anyway, so this costs no memory beyond the twins.

### 2.6 Done when

- Every slot of a chunk twin is bit for bit today's single-parameter twin seeded on that parameter: the same operations in the same order (Rust contracts no multiply and add into a fused one unasked).
- Every derivative test passes bit for bit unchanged; the refactor's dot-product test holds; finite differences agree.
- The JVP's one twin per device agrees with the sum over per-parameter twins to rounding.
- The timing protocol shows a VJP over a device's `n` parameters costing one evaluation per chunk instead of `n`.

**Result** (17 November): every kernel's gradient test checks that a twin along several of its real parameters at once finds along each, bit for bit, what a twin along that parameter alone does, in chunks of eight and of three, which pad to four (`testing.rs`, `assert_chunks_match_seeds_alone`), the heated kernels' included; a VJP along ten parameters of one diode, two chunks, pulls a cotangent back to each bit for bit as a VJP along that parameter alone does, at an operating point and at every step of a transient (`a_vjp_in_chunks_pulls_back_to_each_parameter_as_a_vjp_along_it_alone`); the JVP along a tangent agrees with its columns combined within the dot-product test's bound at every analysis (`the_jvp_and_the_vjp_are_transposes_of_each_other_at_every_analysis`); every derivative test passes unchanged.
Each chunk is one evaluation, its count of instructions in 2.3.
Timed by `evaluation.rs`, criterion, on one pinned core of machine D (`5efde61`, 18 November), a twin of 1, 2, 4 and 8 seeds evaluates in 3.057, 4.132, 5.561 and 8.884 µs for BSIM4, 109.9, 168.2, 159.8 and 266.8 ns for the diode at 0.69 V, and 56.25, 54.92, 86.44 and 133.2 ns for the resistor: eight seeds in one twin take 2.75, 3.30 and 3.38 times less time than eight twins of one.
**Whole VJP, alternating protocol** (18 November, `a3c1d6c`, release build with `lto = "thin"` and `codegen-units = 1`, `chunk=8` (`Seeds::MOST`) against `chunk=1` (`chunks(1)` in `crates/jolty-engine/src/twins.rs`), 9 alternating rounds (AB/BA) of 5 repetitions each = 45 samples per arm, pinned to one core of machine D, AMD EPYC 9B14, host load 93.39 to 93.84 over 96 cores): bit-for-bit check passes across all 25 `(case, n)` combinations with 0 mismatched bits (`f64::to_bits`).
Medians and exact 95% binomial rank intervals below (`total` is `vjp` + `pull_back`, `pull` is `pull_back` alone; transients use 200 fixed steps with `truncation_tolerance = ∞` so both arms traverse identical time grids):

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

At `n = 1`, both binaries execute identical `Gradient<1>` evaluations; the 2 to 4% gap on `bsim4_tran` and `diode_tran` is LTO code-layout variance between the two separately linked binaries.
From `n = 2` up, every total and `pull_back` 95% interval separates cleanly in favor of `chunk=8`, scaling to 2.07× to 2.16× total (2.21× to 2.45× `pull_back`) on BSIM4 at 16 parameters and 1.28× to 2.07× total (1.51× to 2.02× `pull_back`) on the diode at 8 to 10 parameters.
Notice `diode_op` (`n = 10` is 1.96× vs `n = 8` at 2.07×) and `diode_tran` (`pull_back` 1.51× vs 1.58×) dip slightly right past a full chunk of 8, where parameters 9 and 10 spill into a second chunk of width 2.
All acceptance items for step 2 are done.

## Step 2b: AC's VJP exact and independent of the number of parameters

AC's matrix `G + jωC` moves with a parameter directly and through the operating point, `(∂J/∂x)·(dx₀/dp)`; the second term was a central difference per parameter, because kernels gave no second derivatives in x (`models.md` 6).
It moves to the adjoint side exactly:

```text
λᵀ (∂J/∂x · dx₀/dp) v  =  gᵀ dx₀/dp,     g = ∂/∂x (λᵀ J(x) v)
gᵀ dx₀/dp  =  −(J₀⁻ᵀ g)ᵀ ∂F₀/∂p
```

`g` is assembled device by device, each share the gradient of one scalar over the device's few local unknowns; then one transposed solve at the operating point and the chunk twins' `∂F₀/∂p` give the term for every parameter at once.

Its price: kernels must read their voltages as `R` rather than `f64`, so that a twin can seed a direction of the voltages; `Local` carries an optional tangent of the unknowns, and every kernel's voltage reads go through it.
That interface change across every kernel is why 2b is conditional: done only once AC gradients over many parameters matter.

**Done when:** AC's VJP has no finite difference and costs one more transposed solve whatever the number of parameters, which the timing protocol shows over 1 to 1,000 parameters; it agrees with today's within the central difference's error; the dot-product test over frequency holds.

**Result** (17 November): every kernel reads its unknowns through `Local::unknown<R>` (and `Local::delayed_unknown<R>`), compiled Verilog-A kernels regenerated with the same reads, and `jolty-expressions` differentiates compiled expressions and gradients to second order without allocation (`Expression::along`, `Unary::curvature`, `Binary::curvatures`, `Points::interpolate`).
`Real` carries `SEEDS`, `unit`, `seeded`, `derivative` and `with_value`; when `R::SEEDS == 0` (plain `f64` Newton runs) the unknown-seeding helpers fold away at compile time.
Self-heated kernels (`HeatedKernel`) propagate second derivatives through temperature (`Hyper<K>`) and now re-prepare whenever a twin seeds unknowns (fixing a latent bug where ambient `ΔT = 0` skipped preparation and yielded zero thermal derivatives); `bsim4/charges.rs` now preserves `vdseff_cv`'s drain derivative at `vds == 0` via `.with_value(0.0)` instead of zeroing the dual number, and `photocell::settled` keeps the chain rule through `illuminance`.
In `jolty-engine`, `BIAS_STEP`, `Bias`, `biases` and `bias_derivative` are deleted; `Simulator::frequency_tangents` uses sparse `entries_along` (exact directional `(∂G/∂x · s, ∂C/∂x · s)` summed in deterministic device order) and `Simulator::pull_frequencies` computes the operating-point share via `entries_cotangents` over the sorted unique matrix slots touched by nonlinear devices and one transposed DC solve per active cotangent (linear circuits skip DC factorization, solves and weighting entirely).

Every model in `Library::standard()` passes `every_kernel_differentiates_its_conductances_with_respect_to_its_unknowns` against Richardson-extrapolated central differences, and zero allocations remain on Newton's path (`a_kernels_evaluation_allocates_nothing`); compiled Verilog-A kernels pass the same check (`crates/jolty-veriloga/tests/twins.rs`).
Following Step 2b, `Real` and `Kernel` were decoupled from `serde` and `schemars` (`Deserialize`, `Serialize` and `JsonSchema` are now required only on `Parameters<f64>` in `Differentiable<K>` and `Thermal<K>`), deleting `Written` and the `Gradient<K>`/`Hyper<K>` serde/schema implementations and adding library-wide `Lift` vs `serde` path parity tests across all 30 analog models and all 8 compiled Verilog-A kernels/fixtures.
On `loaded_amplifier` over four frequencies, exact AC VJP matches `61fa6e8`'s 1 µV central difference within a worst relative gap of `9.57e-10` (at `VDD.waveform.dc`, the central difference's error) and `0` on `RG`, `CL`, and `CG`.
The AC dot-product test holds to `7.15e-17` (`0.32 ε`), and the worst residual across all analyses stays `7.38e-16` (`crates/jolty-engine/tests/sensitivity.rs`), with new tests covering self-heated AC (hot and cooled to exact ambient `ΔT = 0`), nonlinear behavioral AC, linear AC, bit-for-bit determinism across 1, 2 and 4 threads, and a 504-parameter nonlinear amplifier plus resistive ladder.

Measured in release mode (`lto = "thin"`, `codegen-units = 1`) on one pinned core (`cpu=0`) of machine E (host load 122 to 132 over its 256 logical CPUs), three alternating rounds of 11 repetitions each (33 samples per revision), on a fixed 500-rung resistive ladder driven by `loaded_amplifier` over 5 frequencies with 1 cotangent, comparing `61fa6e8` against `ab2c08c` (max relative difference at `P = 1,000`: 9.743e-8 at index 514):

| Parameters `P` | `61fa6e8` pullback median (range) | Step 2b pullback median (range) | Pullback speedup | Total (`vjp` + `pull_back`) median |
| --- | --- | --- | --- | --- |
| 1 | 0.423 ms (0.391 to 0.470) | 0.316 ms (0.301 to 0.348) | 1.3× | 0.763 → 0.644 ms (1.2×) |
| 10 | 1.194 ms (1.115 to 1.287) | 0.343 ms (0.319 to 0.426) | 3.5× | 1.544 → 0.695 ms (2.2×) |
| 100 | 10.064 ms (9.620 to 10.857) | 0.515 ms (0.476 to 0.567) | 19.5× | 10.501 → 0.999 ms (10.5×) |
| 1,000 | 107.795 ms (100.857 to 117.648) | 2.165 ms (1.970 to 2.657) | 49.8× | 109.933 → 3.830 ms (28.7×) |

Counted by `crates/jolty-models/benches/instructions.rs` (gungraun, Valgrind 3.25.1 Callgrind) against `61fa6e8` on `glibc 2.39`: every Newton `f64` kernel moved by at most +2 instructions (+0.07%), with VBIC down by 1 to 4 instructions and the photocell down by 10 (−0.49%).
Re-measured on machine D (glibc 2.27, `ff752db`) after fixing `diode_twin`'s internal node bias from `"anode"` (1 mV) to `"junction"` (0.69 V): `bsim4` twin (`27,077` / `35,861` / `53,350` / `92,128`), `diode` twin (`1,090` / `1,360` / `1,662` / `2,482`), and `resistor` twin (`430` / `607` / `772` / `1,170`).

## Step 3, deferred: one shared compile-time differentiation pass

What `models.md` 5 described: the derivative code generated before run time, with only the derivatives that can be nonzero and none that no output needs, as OpenVAF does for Verilog-A.
jolty's own Verilog-A emitter already decides each variable's derivatives from its dependencies; step 3 would move that logic into a crate that both the emitter and a `#[kernel]` macro over hand-written Rust kernels use.

**Taken up only if** the measurement below shows that, after steps 1 and 2, the derivatives' share of a kernel's evaluation is a real part of a gap with ngspice: it is real machinery (a macro that understands the subset of Rust the kernels are written in, longer builds), so it must be earned.

## Step 4, deferred: compiler-level automatic differentiation

The compiler differentiating the optimized program itself, kernels written as plain `f64` code: Rust's `std::autodiff`, on Enzyme.
On 17 November 2026 it is nightly-only and experimental (tracking issue rust-lang/rust#124509, labelled `B-experimental`, no RFC accepted), and needs fat LTO.
jolty builds on stable Rust.

**Taken up only once** Rust stabilizes it; then measured against steps 1 to 3 before anything moves.

## The measurement, planned and held

What decides step 3, run only when the owner says so:

1. BSIM4 at one card and bias, evaluated in a loop by jolty's kernel and by ngspice's `BSIM4v5load`, on machine B (machine A's clock distorts profiles), the load stated.
1. The same loop for jolty with its dual slots removed, values only, which separates what the derivatives cost from what the model costs.
1. A profile of jolty's evaluation by function: the dual arithmetic, the branches, the temperature-dependent numbers, the stamping.

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
