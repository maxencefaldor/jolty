# Device derivatives: implementation plan

Status: steps 1.2 and 1.3 done; 1.4 and step 2 planned, steps 3 and 4 deferred (17 November 2026).
How derivatives flow through jolty, from a device's law to the JVP and VJP, is `derivatives.html`; this plan is about its first layer, each device's own derivatives.

## Why

Exact derivatives are one of jolty's founding rules: Newton converges on them, and every gradient (sensitivity, mismatch, the JVP and VJP, a loss for training) is built from them.
Each device kernel gives two kinds: with respect to its voltages, `∂F/∂x`, at every Newton iteration; and with respect to its parameters, `∂F/∂p`, whenever a gradient is asked for.

Two debts remain in how kernels compute them:

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
| 1 | Every kernel on dual numbers composed at the natural width of each law, joined onto its terminals by the chain rule | 1.2 and 1.3 done; 1.4 planned |
| 2 | Parameter derivatives in chunks: one evaluation gives a device's derivatives with respect to several parameters; the JVP by one evaluation per device; Newton untouched | Planned |
| 2b | AC's VJP exact, and independent of the number of parameters | Planned, conditional |
| 3 | One shared compile-time differentiation pass for hand-written and compiled kernels | Deferred |
| 4 | Compiler-level automatic differentiation | Deferred |

Dual numbers stay throughout: they are steps 1 and 2, and the reference every other method is tested against.
Steps 3 and 4 would replace them only where a measurement shows they cost too much, or once Rust can differentiate a program itself.

## What it should change, unmeasured

None of these effects is measured; each stage's "done when" measures its own with the timing protocol before it lands.

| Work | Simulation | Gradients |
| --- | --- | --- |
| 1.2 VBIC composed | Measured: an evaluation 31 to 39% faster (below) | Likewise for its twins |
| 1.3 Gummel-Poon, JFET | Measured: Gummel-Poon 2 to 7% faster at its defaults and up to 4% slower with every part of its model; the JFET 1 to 3% slower with every part, 7 to 8% slower at its defaults (below) | Likewise for their twins |
| 1.4 Short laws | None | None |
| MOSFET circuits, under all of step 1 | Unchanged: their kernels are composed already | Unchanged |
| 2 Chunk twins | None, by design: Newton is never touched | A VJP evaluates each device once per chunk instead of once per parameter: unchanged for a crossbar's resistors (one weight each), half for a transistor's width and length, a sixth to an eighth for a kit's six to eight random variables; a JVP once per device whatever the number of parameters. The share of a gradient's time this saves depends on the twins' share beside the solves, largest in a transient, where twins run at every step |
| 2b AC exact | None | AC's gradient from a central difference per parameter (two evaluations and a DC solve each) to one more transposed solve in all |

## Step 1: every kernel composed on dual numbers

### 1.1 Where each kernel stands

Read from `crates/jolty-models/src/kernel/` on 17 November 2026:

| Kernel | Derivatives in x | Work |
| --- | --- | --- |
| BSIM4, BSIM3, MOSFET levels 1 to 3, semi-empirical MOSFET | Composed: `Dual<R, 3>` for the channel (vgs, vds, vbs), `Dual<R, 1>` per junction | None |
| Diode, junctions | Composed: `Dual<R, 1>` | None |
| VDMOS | Composed `Dual<R, 2>`, but its quasi-saturation by hand | 1.4 |
| Verilog-A compiled kernels | Each variable carries the probes it depends on and no others (`crates/jolty-veriloga/src/emit.rs`) | None |
| Behavioral sources and capacitors | `jolty-expressions`' tape: values forward, gradients backward, the right mode for an expression of many inputs | None |
| Linear devices | Their entries, stamped once a run | None |
| VBIC | Composed: `Dual<R, 1>` to `Dual<R, 3>` for each law (1.2) | None |
| Gummel-Poon (`bjt.rs`), JFET (`jfet.rs`) | Composed: `Dual<R, 1>` per junction, `Dual<R, 2>` for the intrinsic transistor's laws and the JFET's channel, `Dual<R, 3>` for the base resistance's current (1.3) | None |
| Memristor, switch, photocell, passive voltage coefficients | Hand-written slopes of short laws | 1.4 |

### 1.2 VBIC composed

VBIC's equations take nine voltages (`VARIABLES`: base-emitter, extrinsic base-emitter, base-collector, extrinsic base-collector, the parasitic transistor's junctions, the internal resistances' drops), and evaluate everything on `Dual<R, 9>`.

1. Map each of its laws to the voltages it reads: each junction current and charge reads its own junction's voltage; the transport current and the normalized base charge `qb` read the intrinsic base-emitter and base-collector voltages; each internal resistance's current reads its own drop and the junction voltages its modulation reads.
1. Evaluate each law on `Dual<R, k>` at its own width `k`, from one to three.
1. Chain each result onto the nine variables once, at the end, as BSIM4 does (`bsim4.rs`: `gradient: [gate, -(gate + drain + bulk), bulk]`); the self-heated variant keeps its temperature through `Heating`, unchanged.

**Done when:** every derivative of VBIC agrees with today's nine-wide one to rounding (the operations' order changes, so not bit for bit); its tests against ngspice pass unchanged; the law suite holds; and the timing protocol (`scripts/bench --compare A,B`) shows the composed VBIC no slower than today's, on machine B.

**Result** (17 November): each law computes at its own width and joins the others through `Oriented::reading`, a law that reads some of a transistor's variables, and `Dual::embed`, a number of a few variables placed among more, both in `kernel/parts.rs` and `kernel/dual.rs` for the kernels that follow; the weak avalanche current is a current of its own.
Over 1,200 evaluations, both polarities, both base charges, both smoothings, warm and at nominal temperature, with and without an excess phase, the residuals and conductances move by at most 0.9 ε of their row's largest and the charges and capacitances not at all; every VBIC test, the ngspice comparisons and the law suite pass unchanged.
`crates/jolty-models/benches/evaluation.rs`, criterion against a saved baseline of `main`, one core of an AMD EPYC 7B12 at load 2 to 4 (machine B unreachable): 779 ns rather than 1.30 µs at its defaults (−39%), 930 ns rather than 1.42 µs with every part of its model (−32%), 887 ns rather than 1.49 µs with an excess phase (−38%).

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
`crates/jolty-models/benches/evaluation.rs`, criterion against a saved baseline of the hand-written kernels, four runs on one core of an AMD EPYC 7B12 at load 2 to 8 (machine B unreachable): Gummel-Poon 171 to 182 ns rather than 188 at its defaults (−2 to −7%), 371 to 377 ns rather than 365 with every part of its model (+2.5 to +3.5%), 401 to 417 ns rather than 400 with an excess phase (0 to +3%); the JFET 176 to 179 ns rather than 164 at its defaults (+7.5 to +8.4%), 265 to 269 ns rather than 265 with every part of its model (+1 to +3%).
The JFET's loss at its defaults, about 13 ns an evaluation, is put to the owner: the consistency is kept unless the owner decides otherwise.

### 1.4 The short hand-written laws

The memristor, the switch, the photocell, VDMOS's quasi-saturation and the passive voltage coefficients, each moved to dual numbers when next touched, under the same acceptance as 1.3.
Small laws, little work, and little speed at stake, which the measurement confirms.

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

### 2.3 Chunk sizes

A small set monomorphized, 1, 2, 4 and 8; each chunk takes the smallest that holds its parameters: a crossbar resistor's weight takes 1, a transistor's width and length take 2.
The largest is a threshold like every other: set from a measured break-even with the timing protocol, on machine B, its machine and load stated, before this lands; never picked.

### 2.4 The JVP without chunks

A tangent on the parameters is one direction.
Seeding each parameter's single slot with its component of the tangent gives `(∂F/∂p)·t` for a device in one evaluation of a `Gradient<1>` twin, however many of its parameters the tangent moves; today that sum takes one twin per parameter.

### 2.5 Newton and memory

- Newton evaluates with `R = f64` and its own small dual numbers; twins are separate instances, built and evaluated only on a derivative's path, so no Newton iteration ever carries a parameter slot.
- A transient's VJP evaluates each chunk twin at each recorded step as its backward pass reaches it; the waveforms are kept anyway, so this costs no memory beyond the twins.

### 2.6 Done when

- Every slot of a chunk twin is bit for bit today's single-parameter twin seeded on that parameter: the same operations in the same order (Rust contracts no multiply and add into a fused one unasked).
- Every derivative test passes bit for bit unchanged; the refactor's dot-product test holds; finite differences agree.
- The JVP's one twin per device agrees with the sum over per-parameter twins to rounding.
- The timing protocol shows a VJP over a device's `n` parameters costing one evaluation per chunk instead of `n`.

## Step 2b: AC's VJP exact and independent of the number of parameters

AC's matrix `G + jωC` moves with a parameter directly and through the operating point, `(∂J/∂x)·(dx₀/dp)`; the second term is a central difference per parameter today, because kernels give no second derivatives in x (`models.md` 6).
It moves to the adjoint side exactly:

```text
λᵀ (∂J/∂x · dx₀/dp) v  =  gᵀ dx₀/dp,     g = ∂/∂x (λᵀ J(x) v)
gᵀ dx₀/dp  =  −(J₀⁻ᵀ g)ᵀ ∂F₀/∂p
```

`g` is assembled device by device, each share the gradient of one scalar over the device's few local unknowns; then one transposed solve at the operating point and the chunk twins' `∂F₀/∂p` give the term for every parameter at once.

Its price: kernels must read their voltages as `R` rather than `f64`, so that a twin can seed a direction of the voltages; `Local` carries an optional tangent of the unknowns, and every kernel's voltage reads go through it.
That interface change across every kernel is why 2b is conditional: done only once AC gradients over many parameters matter.

**Done when:** AC's VJP has no finite difference and costs one more transposed solve whatever the number of parameters, which the timing protocol shows over 1 to 1,000 parameters; it agrees with today's within the central difference's error; the dot-product test over frequency holds.

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
| 1.4 Short laws | Nothing | Each kernel when touched |
| 2.1 to 2.6 | The derivatives refactor (`agentic.md` 12b), on `main` | `real.rs`, `group.rs`, `derivative.rs`; no kernel |
| 2b | Step 2; a need for AC gradients over many parameters | Every kernel's voltage reads, `Local` |
| 3 | The measurement, after steps 1 and 2 | The Verilog-A emitter, a new crate |
| 4 | Rust stabilizing `std::autodiff` | Every kernel |
