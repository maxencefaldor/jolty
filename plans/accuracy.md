# Accuracy and reliability

Status: rewritten in plain words on 15 November 2026, every item checked against its source and today's code; the research is `research/accuracy.md`, kept whole.

Plan: ✅ 6 done · 🚧 3 in progress · 📋 10 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

jolty's error against exact answers is the smallest of any simulator's at equal cost, and it finds the solution of every circuit that has one.
Measured by the precision suite, against answers computed exactly, and by the failure rate on the hard-circuit corpus (`benchmarks.md` items 7 and 8).

## 2. Where we stand

**What jolty does that ngspice does not:**

- **Exact derivatives** of every device model, by automatic differentiation, checked against finite differences in each model's regions, so the small-signal model is the exact slope of the DC one.
    ngspice's hand-written derivatives leave terms out or get them wrong in several models (the guide's departures page, `docs/guide/departures.md`); on the VDMOS buck with pulse-driven gates ngspice stalls where its Jacobian misses a slope.
- **Charge as the state:** every device's charge is a function of its voltages, except Meyer's MOSFET capacitances, which have no charge function and conserve charge in neither engine.
- **Error control on every node's charge**, the MOSFETs' junction charges included, which ngspice skips; the error filtered through the step's own equations; a judged first step after every corner of a waveform.
- **A convergence test with a floor at its rounding**, and deflation to find more than one DC solution.
- **The same bits on any number of threads**, shards or orders of completion.

**Measured for the first time on 15 November** (`benchmarks.md` §2):

- **Against exact answers** (the precision suite, 16 fixtures): jolty leads on DC (14.6 correct digits to ngspice's 8.0), noise (15.2 to 8.1), poles (15.3 to 14.7) and stiff networks (5.1–5.4 to 3.2–3.9); it loses on small RCs, held by the charge tolerance's floor (item 4), on a lossless LC tank its ringing test damps (item 16), on AC near 650 MHz (item 17) and on a 1,000-node ladder whose start needs a step below jolty's shortest (item 18).
    At the defaults both simulators get only 1.2–3.8 digits on transients.
- **On hard circuits** (the corpus, 48 cases): jolty failed none of the 40 runs it supports, ngspice 1 of 48; jolty needs gmin stepping on one deck where ngspice's Newton iteration alone succeeds, and pseudo-transient continuation on `folded_sources` (items 8 and 9).

**What the research found weaker than the documentation said:**

1. `charge_tolerance` was not SPICE's `chgtol`: jolty allowed `reltol·|q| + chgtol` of charge error per step, SPICE `reltol·max(|q|, chgtol)`, about a thousand times more on a net holding under 10 fC (item 4).
1. ngspice 47 gives each time step 100 Newton iterations, not 10: it raises any limit below 100 to 100 (`niiter.c`, lines 37–39).
    Measured (item 1).
1. Gear2's error estimate used the equal-step constant after a step change, up to 7.6 times too small (item 2, fixed).

**The buck's failed iterations at gate edges** (the old roadmap's item 18, diagnosed on 14 November, an archived note, not kept): about one time step per gate edge runs out of Newton iterations, ngspice none over 2 ms. The first iteration starts with a body diode about 3 V forward, and SPICE3's junction limiter then climbs `vt·ln(1 + Δ/vt)` per iteration, converging linearly at 0.68.
Ruled out: the prediction of node voltages, ngspice's halving of the gap to a breakpoint, the convergence test.
Item 1 measured ngspice's iteration limit; items 6 and 7 address the cause.

## 3. Gaps

- The charge tolerance's floor still holds small nets' error (item 4).
- Newton's iteration stops at a fixed count, and the junction limiter jumps at 2vt in both engines (items 6 and 7).
- The cold start needs gmin stepping where SPICE3's would not (item 8).
- No simulator reports the charge its solver leaks (item 12).
- The bits differ between macOS and Linux (item 13).

## 4. Plan

Each item says what it is, why it matters, and what "done" means; **Lead** marks an item where jolty would go beyond every reference.
Its source is the research report's row, in `research/accuracy.md` §4.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. ngspice's own iteration limit measured on the buck

- **What:** `transient_iterations=100`, the limit ngspice 47 actually applies, run on the buck to see how much of jolty's extra work it explains.
- **Result:** 35% fewer rejected steps, 19% fewer iterations, 15% less time; still 1.24× ngspice's iterations and 3× its time (raw logs, not kept).
    jolty keeps SPICE's 10 (§5); the measured runs are item 6's baseline.
- **Source:** row 5.

### ✅ 2. Gear2's error estimate at its own step ratio

- **What:** the second-order Gear formula's error constant depends on the ratio of a step to the one before it, `(1+ω)²/(6ω(1+2ω))`; both engines used its equal-step value, 2/9, after any change.
- **Result** (`d778b55`): the estimate is right at every ratio, a unit test covers ratios from 0.1 to 10, the guide states it as a departure from ngspice; the trapezoidal default's bits unchanged.
- **Kept** by the owner (15 November), at 11.6% on the buck with Gear2: the error control means what it says at every ratio.
- **Source:** row 4.

### ✅ 3. Every departure from ngspice stated in the guide

- **What:** each place where jolty computes otherwise than ngspice, with ngspice's source line or a measurement as evidence, in one page.
- **Result** (`1e3f31a`): `docs/guide/departures.md`, 39 departures in five tables, each beside the guide's description of jolty's side; two of the research's rows corrected on rereading ngspice's source.
- **Source:** row 15.

### ✅ 4. `charge_tolerance` with SPICE's meaning

- **What:** a charge's error per time step held to `reltol·max(|q|, chgtol)`, as SPICE and ngspice hold it (`cktterr.c`, lines 40–41), instead of jolty's `reltol·|q| + chgtol`: the option is named after SPICE's, so it must mean SPICE's quantity (`AGENTS.md`, "Knobs").
- **Result** (`f673b49`): every test passing, each failure the first attempt exposed fixed where it arose: the buck's stop at `reltol` 1e-5 came from a real 10 ps settling at a switch's internal source, below jolty's shortest step, now ngspice's (item 18); the NAND gate's, from a near double root at a node only gmin holds, which Newton's contraction rule now carries (item 6); a currents test's bound and the random circuits' charge tolerance derived again; the comparisons now run both engines at the shared default of 1e-14 C.
    Measured at the defaults: the 500-inverter chain takes 2,448 accepted steps instead of 749 and 5.2–5.4 s instead of 2.33 s, every extra rejection at a p-channel displacement, since jolty judges the whole oxide's charge where ngspice judges Meyer's; at equal error it still takes fewer points than ngspice (2,448 at 0.23 ns mean edge error against ngspice's 3,602 at 0.22 ns).
    The buck: 82,710 steps instead of 73,880, 0.77 s instead of 0.69 s, its settled output 3.6 mV off instead of 6.5 mV.
- **Left:** the buck at `reltol` 3e-6 and 1e-6 still stops at 1.7 ns (item 6b); the gallery's ngspice comparisons, made at `chgtol` 1e-18, to regenerate.
- **Source:** row 2; raw logs, not kept.

### 📋 5. Error judged in the unknowns, free of any charge floor

- **What:** the error filter already solves `(G + αC)⁻¹ α d`, the error each charge's estimate makes in the unknowns themselves.
    Judging that against `reltol·|x|` plus the voltage or current tolerance would hold every node's accuracy whatever its capacitance, with no extra solve; an option beside judging charges.
- **Why:** scale-free error control; on a 1 fF net no charge tolerance is right for both a femtofarad and a nanofarad.
- **Done when:** the option exists and its default is chosen by the precision suite's numbers.
- **Lead:** filtered, local and in the unknowns together, which no reference does.
    **Source:** row 3.

### 🚧 6. Newton's iteration stopped by its observed contraction

- **What:** instead of a fixed count of iterations per step (SPICE's 10), measure how fast the iteration contracts, `θ = ‖δₖ‖/‖δₖ₋₁‖`: go on while θ < 1 and the iterations it predicts still fit, stop at once when θ ≥ 1, and accept when `θ/(1−θ)·‖δ‖`, a bound on the remaining error, is within tolerance.
- **Why:** no arbitrary limit: a hopeless iteration stops at once, a slowly converging one at a gate edge goes on.
- **State** (`f673b49`): in a time step, the iteration goes on past `transient_iterations` while each step is shorter than the one before, and once it contracts by more than half accepts only when `θ/(1−θ)` times its step is within tolerance; the operating point keeps its count.
    On the chain it trades accuracy for work (0.085 ns of mean edge error at 2,705 steps without it, 0.23 ns at 2,448 with it), an owner's call, listed in `README.md`'s open decisions.
- **The owner's design** (15 November), simple and orthogonal: one option, `max_transient_iterations`, a true cap; within it the contraction tests are the convergence test itself, never switched off: accept when `θ/(1−θ)·‖δ‖` is within tolerance, cut the step when `θ ≥ 1` or when the iterations left at rate `θ` cannot reach it (Hairer and Wanner IV.8, as RADAU5).
    The cap's default is measured: SPICE's 10 if every suite passes with it, else the value found, stated.
    The rule's trade on the chain is kept: its accuracy is set by the tolerance, not by Newton's failures.
    ngspice's exact iteration behaviour is not reproduced, as its Gear2 constant is not.
- **Result, in part** (on `main`, `6e01688` and `da286de`, 16 November, raw logs, not kept): `max_transient_iterations` a true cap, 10; acceptance when `θ/(1−θ)·‖δ‖` is within tolerance, its old `max(·,1)` gone; `θ` measured only between steps no device limited; a first, limited or non-contracting step judged by SPICE's step test; the iterate after the last step judged, in DC too.
    Every suite passes at 10: the Rust tests, the precision suite's 765 runs to the same digits, the hard-circuit corpus's 40 runs.
    Chain of 500 inverters: 16% fewer iterations, 13% less time, mean edge error 0.228 → 0.229 ns; buck: +4.8% time, its output's error 21.5 → 15.2 mV.
    **Left out, for the owner:** the two cuts, at `θ ≥ 1` and when the iterations left cannot reach the tolerance.
    Counted without acting at cap 10, 91% of the buck's 9,135 iterations they would cut and 92% of the chain's 521 converged within 10 anyway; enabled, the chain took 6,103 steps for 2,448 and the buck stopped at 41 µs.
    Hairer and Wanner's tests assume a simplified Newton's constant rate; jolty's full Newton speeds up near the solution.
- **Done when:** the buck and the protocol's examples run at equal accuracy for no more work, the DC iteration too; any constant the method needs approved by the owner.
- **Lead** among SPICE-class engines.
    **Source:** row 6; Hairer and Wanner, IV.8; Deuflhard 2004.

### ✅ 6b. A floor at its rounding in Newton's step test

- **What:** the test that a Newton step is small enough asks for 1e-12 A where the equations round at 2.5e-10 A, with no floor at their rounding, so below 1e-13 s Newton cannot settle the current through two off switches.
- **Why:** the buck at `reltol` 3e-6 and 1e-6 stops at 1.7 ns, where M1's Meyer capacitance turns on with a kink.
- **Done when:** the buck runs at every `reltol` down to 1e-6.
- **Result** (`da286de`, 16 November): the diagnosis did not hold: an exact floor, the largest move no equation registers, left the buck stopping at 1.70 ns.
    The cause was a junction's depletion charge, `1 − (1 − v/φ)^(1−m)` losing its digits near 0 V, 3.5e-25 C at 1 nF, 5e-12 A through a step's weight; computed as `−expm1((1−m)·ln1p(−v/φ))`, with `ln_1p` and `exp_m1` in `Real` and `Dual`, the buck runs at 1.25e-4, 1e-5, 3e-6 and 1e-6, with 13–15% fewer iterations.
    No floor added.
- **Source:** the charge tolerance's second attempt.

### 📋 7. A junction limiter without its jump at 2vt

- **What:** SPICE3's limiter for diode-like junctions switches formula at twice the thermal voltage, a discontinuity both engines share; `vold + vt·ln(1 + Δ/vt)` for every forward step above the critical voltage is continuous.
    Limiting from the last accepted point, as ngspice does, saves 1.5% of the buck's steps.
- **Done when:** a property test shows the limited voltage monotone and continuous above the critical voltage; the buck's failures and work measured.
- **Lead.**
    **Source:** row 7.

### 📋 8. A cold start that converges

- **What:** junctions initialized on the first DC iteration as SPICE3's `MODEINITJCT` does; the DC equations solved block by block in their order, any homotopy confined to the block that needs it; a start from a given point.
- **Why:** fewer operating points falling back to gmin stepping; the corpus found jolty needing it on a deck where ngspice's Newton iteration alone succeeds.
- **Done when:** every circuit of the corpus starts without gmin stepping where SPICE3's would.
- **Source:** row 8; the scaling blueprint, an archived note, not kept §3.1.

### 📋 9. DC homotopies

- **What:** continuation along the sources' scale by Keller's pseudo-arclength method, which goes round turning points that source stepping cannot; a MOS homotopy as an option; pseudo-transient steps on the circuit's own capacitances.
- **Why:** circuits with turning points and large MOS circuits; the corpus's `folded_sources` needs pseudo-transient continuation.
- **Done when:** the corpus's failure rate falls, no circuit lost.
- **Lead** with deflation along the path.
    **Source:** row 9.

### 📋 10. Failures named as mathematics

- **What:** a transient that stops where no solution continues named an impasse point; loops of capacitors and voltage sources, and cutsets of inductors and current sources, as rules of the checks.
- **Why:** an agent learns why a run failed, not only that it did.
    The corpus's impasse circuit runs past its impasse at 5.47 µs, in both engines, naming nothing.
- **Done when:** each failure of the corpus carries a named cause.
- **Lead.**
    **Source:** row 11; Chua and Deng 1989; Estévez Schwarz and Tischendorf 2000.

### 📋 11. A charge model for the MOSFET's levels 1 and 3

- **What:** Ward and Dutton's charge partition as an option beside Meyer's capacitances, so that charge is conserved.
- **Why:** switched-capacitor and DRAM-like circuits.
- **Done when:** a closed cycle of voltages returns no net charge, within the solver's leak (item 12).
- **Lead** over ngspice.
    **Source:** row 12.

### 📋 12. The solver's charge leak reported

- **What:** each node's residual integrated over a transient, reported with the statistics, and a property test that a closed system's charge stays within it.
- **Done when:** every transient's statistics carry it.
- **Lead.**
    **Source:** row 13; Kundert and Clifford 1993.

### ⏸️ 13. The same bits on every platform

- **What:** jolty's own correctly rounded `exp`, `ln` and `pow`, used by every kernel, so macOS, Linux and Windows give the same bits, where each platform's C library rounds differently today; their cost measured.
- **Parked** by the owner (15 November): the same bits across platforms is no longer a rule, and the platforms' own functions are already within one unit of the last place, far below any tolerance, so correct rounding would cost speed for nothing a simulation can see.
- **Done when:** CI's macOS and Linux jobs hash the same waveforms; an `exp`'s cost measured.
- **Lead.**
    **Source:** row 14; `research/speed.md` row M3.
    This item replaces `speed.md`'s copy of it.

### 📋 14. Output at the times asked for, and TR-BDF2 measured

- **What:** dense output, the solution interpolated at times the user asks for instead of steps forced onto them, which the comparison harness also needs; TR-BDF2 as an option, measured on the boost converter's stiff phases; variable-order BDF only if a circuit shows a gain (Xyce removed its own in 6.8).
- **Done when:** the comparison harness stops stepping onto ngspice's times; TR-BDF2's effect on the boost measured.
- **Source:** row 16; the old roadmap's item 7, narrowed.

### 📋 15. Newton's tolerance in a time step tied to the step's error tolerance

- **What:** as DASSL and Xyce do, stop Newton's iteration within a step once its error is well below what the step's truncation error allows, not at a fixed tolerance.
- **Done when:** the owner decides on the precision suite's numbers.
- **Source:** row 17.

### 📋 16. A lossless oscillation told from numerical ringing

- **What:** the trapezoidal rule's ringing test switches to backward Euler where a charge's derivative turns back and forth twice in a row; on a lossless LC tank, whose oscillation is real, it damps it: 2.7% of its energy left after 20 periods at `reltol` 1e-2, where ngspice keeps 3.6 digits.
- **Done when:** the tank keeps its energy to the step's error at every tolerance, the ringing test still catching numerical ringing.
- **Source:** the precision suite's first run (`perf/precision/README.md`).

### ✅ 17. AC's lost digits near 650 MHz

- **What:** on the precision suite's RLC, jolty and ngspice with KLU lose three digits between 600 and 700 MHz (11.1 and 11.4 correct digits), where ngspice with Sparse keeps 14.2.
    To diagnose: the complex solve's pivoting is the first suspect.
- **Done when:** the response's digits flat over frequency, the cause stated.
- **Result** (`ae943be`, 16 November): not a band but the response's size: digits fell as `16 + log10|v(out)|`, an absolute error of ε volts, from 13.8 at 100 MHz to 11.4 at 1 GHz.
    The sweep keeps the pivots it chose at 1 Hz, where conductances rule; with them elimination computes the output as a difference of volts.
    Every solve with `G + jωC` (AC, noise, stability, AC sensitivity, pole-zero) now refines in working precision while Oettli and Prager's componentwise backward error exceeds ε and each step halves it (Higham §12.3; Skeel 1980): 15.6 to 16.5 digits from 1 Hz to 1 GHz, a test holding the RLC within 32 ε.
    Cost, interleaved against `3945c01` on this Mac: `ac_wide` +10% (20,000 solves of 22 unknowns, the worst case), `bjt_amp` +2%; measured with each complex number's `|re| + |im|` and buffers kept between solves, the modulus and allocation had cost 26–29%.
    On `ac_wide` itself ω was already 0.71 ε at the median and 2.3 ε at worst, 13% of solves refined.
- **Source:** the precision suite's first run.

### 🚧 18. The shortest time step derived, not chosen

- **What:** jolty's shortest step is 10⁻⁹ of the longest, ngspice's 10⁻¹¹, neither derived; the 1,000-node ladder's runs fail from a scaled tolerance of 5.6e-7, as the step its start needs is below jolty's 8 ps.
- **State** (`f673b49`): now ngspice's 10⁻¹¹ of the longest, which removes a departure and lets the buck resolve its 10 ps settling; still chosen, not derived.
- **Done when:** the bound derived from the arithmetic's resolution, the ladder solved at every tolerance.
- **Source:** the precision suite's first run; the guide's departures page.

### 🚧 19. The departures found on 15 November added to the guide, ngspice's and Xyce's

- **What:** three found by today's suites: ngspice 47 multiplies a bipolar transistor's base-collector saturation current by the area twice when its card gives no `IBE` or `IBC` (`bjttemp.c`, lines 168, 177–179, 198–202), so `vreg`'s output differs by 8–12 mV; ngspice's AC sensitivity is right only at the first frequency, its `lin` sweep's frequencies growing geometrically (to be checked in its source); Xyce's noise is 9.5e-6 above 4kTR, from its 1973 electron charge.
- **Xyce's, from its agreement tests:** a MOSFET's reverse bulk junction linear, SPICE3f5's, where jolty and ngspice 47 use an exponential with a cubic tail; a bipolar transistor's gmin divided by beta inside the junction current; no noise in its JFET; its JFET's temperature dependence SPICE3f5's; its own excess phase.
    Two unexplained: at 125 °C `nmos_stage`'s output 1.4e-4 from jolty's, and `cmos_inverter`'s 55 µV across its threshold.
    Xyce finds no operating point for `negative_load`, after 7,692 Newton iterations, and stalls on the buck.
- **Done when:** each stated in `docs/guide/departures.md` with its evidence, ngspice's AC sensitivity checked in its source first, Xyce's two unexplained differences explained.
- **Result, in part** (`00a9e75`, 16 November): in `departures.md`, ngspice's base-collector saturation current multiplied by the area twice (`bjttemp.c`, lines 168, 177–179, 198–202, read); its AC sensitivity's linear sweep, checked in its source: `inc_freq` tests `SENS_LINEAR` (15) against AC's `LINEAR` (3), so each frequency after the first is the one before times the step (`cktsens.c`, line 831); Xyce's departures in the SPICE guide as `tests/test_xyce.py` names them.
    **Left:** Xyce's noise 9.5e-6 above 4kTR, its electron charge to be read in its source; Xyce's two unexplained differences.
- **Source:** the corpus's and the coverage matrix's first runs.

**Elsewhere:** the precision suite and the hard-circuit corpus are `benchmarks.md` items 7 and 8; their findings are this document's items 4, 8, 9, 10 and 16 to 19.

## 5. Decided

- `truncation_tolerance` means SPICE's `TRTOL`.
- Every charge's error is bounded: on the buck jolty takes 90,774 attempts to ngspice's 56,624 points, and at equal control 37% fewer steps than ngspice.
- SPICE3's junction limiter is kept over ngspice 47's `DEVpnjlim`, which halves the buck's Newton failures but saves no work.
- A transient step starts from its prediction.
- `gmin-dependence` is a warning.
- Meyer's capacitances conserve charge in neither engine; the notes no longer call jolty's form charge-conserving.
- Every transient change follows one protocol: a probe of each rejected step first, explained before anything changes; before and after numbers of accepted, rejected and landed steps and Newton iterations on the buck, boost, ring, relaxation, 555 timer, astable and Colpitts examples at default options; ngspice on the same netlists side by side; accuracy unchanged, `test_ngspice.py`, `test_random_circuits.py`, `test_metamorphic.py` and the analytic tests passing with no tolerance loosened; and the guide's table of methods saying what changed and where it comes from.
- The old roadmap's item 7, measured and set aside: bounding repeated rejections at one time point saves too little; a damped formula for the boost's stiff mode could act only on 36% of its steps, so item 14 measures TR-BDF2 there first.
- `transient_iterations` keeps SPICE's `ITL4` of 10, though ngspice 47 raises any limit below 100 to 100, a departure the guide states: another count would be a constant picked for one fixture; the fix is item 6.

## 6. Open principles

None: the research's proposals were merged into the rules of `AGENTS.md` on 15 November (`444535f`).
