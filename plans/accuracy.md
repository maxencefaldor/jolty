# Accuracy and reliability

Status: rewritten in plain prose on 15 November 2026 with every item checked against source and code, and audited on 18 November 2026.
The archival research report lives in `research/accuracy.md`, kept whole.

Plan: ✅ 8 done · 🚧 3 in progress · 📋 8 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

jolty aims for the smallest error against exact answers of any simulator at equal cost and finds a solution for every solvable circuit.
Progress is measured by the precision suite against exact answers and by the failure rate on the hard-circuit corpus (`benchmarks.md` items 7 and 8).

## 2. Where we stand

**Capabilities jolty provides beyond ngspice:**

- **Exact derivatives** for every device model via automatic differentiation, verified against finite differences in every operating region so small-signal models match the exact DC slope.
    Hand-written derivatives in ngspice omit or miscompute terms in several models (documented in `docs/guide/departures.md`), causing ngspice to stall on the pulse-driven VDMOS buck where the Jacobian misses a slope.
- **Charge as the state:** every device's charge is a function of terminal voltages, except Meyer's MOSFET capacitances, which have no charge function and conserve charge in neither engine.
- **Filtered charge error control:** error control covers every node's charge, including MOSFET junction charges skipped by ngspice, filtered through the step equations with a judged first step after each waveform corner.
- **Rounding-aware convergence and deflation:** a convergence test floored at numerical rounding, paired with deflation to locate multiple DC solutions.
- **Bit-identical parallelism:** identical bits across any thread count, shard count, or completion order.

**First measurements on 15 November** (`benchmarks.md` §2):

- **Against exact answers** (precision suite, 16 fixtures): jolty leads on DC (14.6 digits vs ngspice's 8.0), noise (15.2 vs 8.1), and poles (15.3 vs 14.7).
    On stiff networks, jolty reaches 5.1 to 5.4 digits against ngspice's 3.2 to 3.9.
    jolty loses on small RCs due to the charge tolerance floor (item 4) and on a lossless LC tank damped by the ringing test (item 16).
    Further losses occur on AC near 650 MHz (item 17) and on a 1,000-node ladder requiring an initial step shorter than jolty's minimum step (item 18).
    At default options, both simulators achieve only 1.2 to 3.8 digits on transients.
- **On hard circuits** (corpus, 48 cases): on 15 November, jolty failed 0 of the 40 supported runs while ngspice failed 1 of 48.
    Since `models.md` 7b (`.ic`, `.nodeset`, `EXP`), jolty runs all 48 cases (25 DC, 23 transient).
    On machine D (`5efde61`, 18 November), jolty had 0 failures (25 DC agreed; 6 transient agreed, 16 solved, and 1 disagreed on `impasse/tran`, matching ngspice).
    On the same suite, ngspice failed 1 case (`folded_sources/op`) and disagreed on `vreg/dc` and `impasse/tran`.
    jolty still needs gmin stepping on one deck where ngspice's Newton iteration succeeds directly, plus pseudo-transient continuation on `folded_sources` (items 8 and 9).

**Findings weaker than prior documentation claimed:**

1. `charge_tolerance` did not match SPICE's `chgtol`: jolty allowed `reltol·|q| + chgtol` per step instead of SPICE's `reltol·max(|q|, chgtol)`, roughly 1,000 times looser on nets below 10 fC (item 4).
1. ngspice 47 grants each time step 100 Newton iterations rather than 10, raising any lower limit to 100 (`niiter.c`, lines 37-39; measured in item 1).
1. Gear2's error estimate used the equal-step constant after a step-size change, underestimating error by up to 7.6 times (item 2, fixed).

**Buck failed iterations at gate edges** (old roadmap item 18, diagnosed on 14 November; archived note not kept): roughly one time step per gate edge exhausts Newton iterations, whereas ngspice has none across 2 ms of simulation.
The first iteration starts with a body diode around 3 V forward, and SPICE3's junction limiter then climbs by `vt·ln(1 + Δ/vt)` per iteration, converging linearly at rate 0.68.
Node voltage prediction, ngspice's halving of the gap to a breakpoint, and the convergence test were ruled out.
Item 1 measured ngspice's iteration limit; items 6 and 7 address the root cause.

## 3. Gaps

- `charge_tolerance` now follows SPICE's `reltol·max(|q|, chgtol)` definition (item 4, done), while `chgtol` still acts as a fixed floor on small nets until error is judged in the unknowns (item 5).
- Newton iteration stops at a fixed count, and the junction limiter jumps at 2vt in both engines (items 6 and 7).
- Cold start still requires gmin stepping on circuits where SPICE3's initialization succeeds directly (item 8).
- Integrated solver charge leak is now reported per node on every transient (item 12, done), and lossless oscillations are distinguished from numerical trapezoidal ringing (item 16, done).
- Waveform bits still differ between macOS and Linux (item 13).

## 4. Plan

Each item specifies its scope, motivation, and completion criteria; **Lead** marks an item where jolty surpasses every reference.
Each item's source references the corresponding row in `research/accuracy.md` §4.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. ngspice's iteration limit measured on the buck

- **What:** `max_transient_iterations=100` (formerly `transient_iterations`), the limit ngspice 47 actually applies, run on the buck to measure how much of jolty's extra work that limit explains.
- **Result:** 35% fewer rejected steps, 19% fewer iterations, and 15% less time, yet still 1.24× ngspice's iterations and 3× ngspice's time (raw logs not kept).
    jolty keeps SPICE's default of 10 (§5); these measured runs form the baseline for item 6.
- **Source:** row 5.

### ✅ 2. Gear2's error estimate at variable step ratios

- **What:** the second-order Gear formula's error constant depends on the step ratio `ω` via `(1+ω)²/(6ω(1+2ω))`, whereas both engines previously used the equal-step value `2/9` after any change.
- **Result** (`d778b55`): the estimate is accurate at every ratio, with a unit test covering ratios from 0.1 to 10.
    The guide documents this behavior as a departure from ngspice, and the trapezoidal default's bits remain unchanged.
- **Kept** by the owner (15 November) despite an 11.6% cost on the buck with Gear2, ensuring error control stays accurate at every step ratio.
- **Source:** row 4.

### ✅ 3. Every departure from ngspice documented in the guide

- **What:** a single page cataloging every place jolty computes differently from ngspice, backed by ngspice source lines or measurements.
- **Result** (`1e3f31a`): `docs/guide/departures.md` listed 39 departures in five tables initially (now 55 departures in six tables), each paired with the guide's description of jolty's behavior.
    Two rows from the research report were corrected after re-reading ngspice's source.
- **Source:** row 15.

### ✅ 4. `charge_tolerance` with SPICE's meaning

- **What:** hold per-step charge error to `reltol·max(|q|, chgtol)` as in SPICE and ngspice (`cktterr.c`, lines 40-41), replacing jolty's previous `reltol·|q| + chgtol`.
    The option carries SPICE's meaning to match SPICE's name (`AGENTS.md`, "Knobs").
- **Result** (`f673b49`): every test passes, and each failure exposed by the initial attempt was fixed at its root cause.
    The buck's stop at `reltol = 1e-5` came from a real 10 ps settling event at a switch's internal source, below jolty's previous shortest step (now aligned with ngspice; item 18).
    The NAND gate's failure came from a near double root on a gmin-held node, now handled by Newton's contraction rule (item 6).
    A currents test bound and the random circuits' charge tolerance were re-derived, and comparisons now run both engines at the shared default of 1e-14 C.
    At defaults, the 500-inverter chain takes 2,448 accepted steps instead of 749 and 5.2 to 5.4 s instead of 2.33 s.
    Every extra rejection occurs at a p-channel displacement because jolty bounds full oxide charge whereas ngspice checks Meyer's capacitance.
    At equal error, jolty still takes fewer points than ngspice (2,448 points at 0.23 ns mean edge error vs ngspice's 3,602 points at 0.22 ns).
    The buck takes 82,710 steps instead of 73,880 (0.77 s vs 0.69 s), improving settled output error from 6.5 mV to 3.6 mV.
- **Since resolved:** the buck at `reltol` 3e-6 and 1e-6 was fixed by item 6b (`da286de`), and the gallery and discoveries were rebuilt on the updated engine with ngspice 47 (`0552d20`, `c766183`, 16 November).
- **Source:** row 2 (raw logs not kept).

### 📋 5. Error judged in the unknowns, free of any charge floor

- **What:** the error filter already solves `(G + αC)⁻¹ α d`, giving the error each charge estimate produces in the unknowns.
    Judging that vector against `reltol·|x|` plus the voltage or current tolerance holds every node's accuracy regardless of capacitance with no extra solve, offered as an option alongside charge-based error control.
- **Why:** scale-free error control, since no single charge tolerance fits both a 1 fF net and a 1 nF net.
- **Done when:** the option exists and the precision suite selects the default.
- **Lead:** filtered, local, and unknown-space error control combined, which no reference implements.
    **Source:** row 3.

### 🚧 6. Newton's iteration stopped by observed contraction

- **What:** replace a fixed per-step iteration count (SPICE's 10) by measuring the contraction rate `θ = ‖δₖ‖/‖δₖ₋₁‖`.
    Newton continues while `θ < 1` and predicted iterations fit, stops immediately when `θ ≥ 1`, and accepts when the remaining-error bound `θ/(1−θ)·‖δ‖` lies within tolerance.
- **Why:** removes arbitrary limits so hopeless iterations stop immediately while slowly converging gate-edge iterations continue.
- **State** (`f673b49`): inside a time step, iteration continued past `transient_iterations` (later renamed `max_transient_iterations` in `be0de7d`) while steps shrank, whereas operating points kept a fixed count.
    Once contraction exceeded a factor of two, acceptance required `θ/(1−θ)` times the step to lie within tolerance.
    On the inverter chain, that rule trades accuracy for work (0.085 ns mean edge error at 2,705 steps without contraction vs 0.23 ns at 2,448 steps with contraction), listed in `README.md` open decisions.
- **Owner design** (15 November), simple and orthogonal: `max_transient_iterations` acts as a hard cap, and contraction tests form the always-on convergence criterion within that cap.
    Newton accepts when `θ/(1−θ)·‖δ‖` falls within tolerance and cuts the step when `θ ≥ 1` or when remaining iterations at rate `θ` cannot reach tolerance (Hairer and Wanner IV.8, as in RADAU5).
    The cap default is chosen by measurement: SPICE's 10 if every suite passes, otherwise the lowest measured value documented explicitly.
    The accuracy trade on the chain is retained so tolerances, rather than Newton failures, control accuracy.
    Exact ngspice iteration behavior is not replicated, matching the decision on Gear2's constant.
- **Result, in part** (on `main`, `6e01688` and `da286de`, 16 November; raw logs not kept): `max_transient_iterations` is now a true cap of 10, and acceptance checks `θ/(1−θ)·‖δ‖` within tolerance without the old `max(·,1)` floor.
    Contraction rate `θ` is measured only across unlimited steps, while first, limited, or non-contracting steps use SPICE's step test, and the final iterate is judged in DC as well.
    Every suite passes at cap 10: all Rust tests, all 765 precision suite runs to the same digits, and all 40 hard-circuit corpus runs.
    The 500-inverter chain takes 16% fewer iterations and 13% less time (mean edge error 0.228 to 0.229 ns), while the buck takes +4.8% time with output error improving from 21.5 to 15.2 mV.
    **Omitted pending owner approval:** the two early cuts at `θ ≥ 1` and when remaining iterations cannot reach tolerance.
    Measured without acting at cap 10, 91% of the buck's 9,135 candidate cut iterations and 92% of the chain's 521 converged within 10 iterations anyway.
    When enabled, those cuts expanded the chain from 2,448 to 6,103 steps and stopped the buck at 41 µs.
    Hairer and Wanner's tests assume simplified Newton's constant convergence rate, whereas jolty's full Newton accelerates near the solution.
- **Done when:** the buck and transient-protocol examples run at equal accuracy for no extra work across transient and DC iterations, with any method constant approved by the owner.
- **Lead** among SPICE-class engines.
    **Source:** row 6; Hairer and Wanner, IV.8; Deuflhard 2004.

### ✅ 6b. Rounding floor in Newton's step test

- **What:** Newton's step test requests 1e-12 A where the equations round at 2.5e-10 A without a rounding floor.
    Below 1e-13 s, Newton therefore cannot settle the current through two off switches.
- **Why:** the buck at `reltol` 3e-6 and 1e-6 stops at 1.7 ns where M1's Meyer capacitance turns on with a kink.
- **Done when:** the buck runs at every `reltol` down to 1e-6.
- **Result** (`da286de`, 16 November): the initial diagnosis did not hold, because an exact rounding floor still left the buck stopping at 1.70 ns.
    The true cause was junction depletion charge `1 − (1 − v/φ)^(1−m)` losing precision near 0 V (3.5e-25 C at 1 nF, yielding 5e-12 A through step weighting).
    Rewriting that term as `−expm1((1−m)·ln1p(−v/φ))` via `ln_1p` and `exp_m1` on `Real` and `Dual` lets the buck run at `1.25e-4`, `1e-5`, `3e-6`, and `1e-6` with 13 to 15% fewer iterations.
    No rounding floor was added.
- **Source:** second attempt on `charge_tolerance`.

### 📋 7. Continuous junction limiter without the 2vt jump

- **What:** SPICE3's diode junction limiter switches formulas at twice the thermal voltage in both engines, whereas `vold + vt·ln(1 + Δ/vt)` stays continuous for every forward step above the critical voltage.
    Limiting from the last accepted point, as ngspice does, saves 1.5% of the buck's steps.
- **Done when:** a property test verifies monotonic, continuous limiting above the critical voltage, and buck failures and work are measured.
- **Lead.**
    **Source:** row 7.

### 📋 8. Cold start convergence

- **What:** initialize junctions on the first DC iteration like SPICE3's `MODEINITJCT`, solve DC blocks sequentially with homotopy confined to failing blocks, and support warm starts from a given point.
- **Why:** reduces fallback to gmin stepping, since the corpus exposed a deck where ngspice's Newton iteration succeeds without gmin stepping.
- **Done when:** every corpus circuit starts without gmin stepping wherever SPICE3's initialization succeeds directly.
- **Source:** row 8; scaling blueprint (archived note not kept), §3.1.

### 📋 9. DC homotopies

- **What:** source continuation via Keller's pseudo-arclength method (traversing turning points that source stepping misses), an optional MOS homotopy, and pseudo-transient continuation using the circuit's internal capacitances.
- **Why:** supports circuits with turning points and large MOS circuits, including `folded_sources` in the corpus.
- **Done when:** the corpus failure rate drops with no circuit regressing.
- **Lead** with deflation along the path.
    **Source:** row 9.

### 📋 10. Mathematical failure diagnostics

- **What:** classify a transient terminating where no solution continues as an impasse point, and add check rules for capacitor-voltage-source loops and inductor-current-source cutsets.
- **Why:** agents learn why a run failed rather than only seeing a failure.
    Both engines currently run past the corpus impasse circuit's impasse at 5.47 µs without reporting a cause.
- **Done when:** every corpus failure reports a named mathematical cause.
- **Lead.**
    **Source:** row 11; Chua and Deng 1989; Estévez Schwarz and Tischendorf 2000.

### 📋 11. Charge model for MOSFET levels 1 and 3

- **What:** Ward and Dutton's charge partition as an option alongside Meyer's capacitances to guarantee charge conservation.
- **Why:** supports switched-capacitor and DRAM-like circuits.
- **Done when:** a closed voltage cycle returns zero net charge within the solver's reported leak (item 12).
- **Lead** over ngspice.
    **Source:** row 12.

### ✅ 12. Solver charge leak reporting

- **What:** integrate each node's residual over every transient, report the result in statistics, and verify closed-system charge conservation with a property test.
- **Done when:** every transient's statistics include the charge leak.
- **Result** (18 November): every transient integrates each node's residual over accepted steps using Kundert and Clifford (1993) with Neumaier (1974) compensated summation.
    Results are exposed as `ChargeLeak` across Rust, PyO3, Python (`TransientResult.statistics.charge_leak`), text and JSON reports, and version 20 results files with strict shape and floating-dtype validation on load.
    Property tests verify closed-system charge conservation within the reported leak.
- **Lead.**
    **Source:** row 13; Kundert and Clifford 1993.

### ⏸️ 13. Bit-identical results across platforms

- **What:** dedicated correctly rounded `exp`, `ln`, and `pow` functions used across all kernels so macOS, Linux, and Windows produce identical bits despite C-library rounding differences, with runtime cost measured.
- **Parked** by the owner (15 November): bit-identical waveforms across platforms are no longer a requirement.
    Platform math libraries already agree within one ULP, far below simulation tolerances, so software rounding would sacrifice speed without any visible simulation benefit.
- **Done when:** macOS and Linux CI jobs hash identical waveforms, and `exp` runtime cost is measured.
- **Lead.**
    **Source:** row 14; `research/speed.md` row M3.
    Item 13 replaces the duplicate entry in `speed.md`.

### 📋 14. Output at requested times, and TR-BDF2 evaluation

- **What:** dense output interpolating the solution at user-requested times rather than forcing steps onto output points, which the comparison harness also requires.
    TR-BDF2 will be added as an option and measured on the boost converter's stiff phases, while variable-order BDF will be pursued only if a circuit shows a clear gain (Xyce dropped variable-order BDF in 6.8).
- **Done when:** the comparison harness stops forcing steps onto ngspice's time points, and TR-BDF2's impact on the boost converter is measured.
- **Source:** row 16; narrowed from old roadmap item 7.

### 📋 15. Newton tolerance tied to step truncation tolerance

- **What:** stop Newton iterations within a time step once Newton error falls well below the step's truncation error allowance, as in DASSL and Xyce, instead of using a fixed tolerance.
- **Done when:** the owner decides based on precision suite results.
- **Source:** row 17.

### ✅ 16. Lossless oscillation distinguished from numerical ringing

- **What:** the trapezoidal ringing test switches to backward Euler whenever a charge derivative reverses twice consecutively.
    On a lossless LC tank with real physical oscillation, that check damps energy down to 2.7% after 20 periods at `reltol = 1e-2`, where ngspice retains 3.6 digits.
- **Done when:** the LC tank preserves energy to step error at every tolerance while the ringing test still catches numerical ringing.
- **Result** (18 November): implemented a two-stage ringing detector in `crates/jolty-engine/src/ringing.rs` with 12 unit tests and 5 integration tests in `crates/jolty-engine/tests/analytic.rs`.
    Stage 1 applies an O(1), zero-solve variable-step scalar restoring-force check.
    Stage 2 reuses cached sparse LU factors (zero extra refactorizations) to separate algebraic `R = -1` trapezoidal ringing from lossless physical oscillations.
    Measured under the §5 transient protocol on machine D (`0f60efc` before vs `5b44dc6` after, alongside ngspice 47 Sparse 1.3; all 7 examples pass all specs):

| Fixture (method) | Accepted (before / after) | Rejected (before / after) | Landed (before / after) | Work iters (before / after) | Refactorizations (before / after) | Reuses (before / after) | ngspice 47 (acc / rej / iters) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `buck` (`trapezoidal`) | 43,564 / 43,564 | 416 / 416 | 0 / 0 | 57,219 / 57,219 | 98,112 / 98,112 | 1,173 / 1,273 | 43,543 / 240 / 102,205 |
| `boost` (`trapezoidal`) | 75,524 / 75,524 | 3,414 / 3,414 | 0 / 0 | 111,625 / 111,625 | 185,142 / 185,142 | 3,427 / 3,957 | 73,999 / 2,031 / 178,106 |
| `ring` (`trapezoidal`) | 4,015 / 4,015 | 0 / 0 | 0 / 0 | 8,026 / 8,022 | 11,999 / 11,995 | 39 / 279 | 4,020 / 0 / 13,184 |
| `relaxation` (`gear2`) | 4,106 / 4,106 | 89 / 89 | 0 / 0 | 5,011 / 5,011 | 9,180 / 9,180 | 7 / 7 | 4,029 / 0 / 8,239 |
| `555` (`trapezoidal`) | 11,901 / 11,948 | 48 / 45 | 265 / 262 | 14,381 / 14,409 | 26,074 / 26,122 | 330 / 370 | skipped (`logic_cell`) |
| `astable` (`trapezoidal`) | 8,927 / 8,897 | 396 / 407 | 0 / 0 | 21,454 / 21,567 | 30,465 / 30,569 | 0 / 2,378 | 8,809 / 302 / 41,921 |
| `colpitts` (`trapezoidal`) | 10,017 / 10,017 | 7 / 7 | 0 / 0 | 19,992 / 19,992 | 30,005 / 30,005 | 7 / 7 | 10,016 / 1 / 20,032 |
| `speed_buck` (`trapezoidal`) | 90,797 / 90,543 | 26,805 / 26,370 | 0 / 0 | 331,101 / 328,708 | 433,499 / 431,056 | 1,793 / 13,477 | 52,946 / 3,612 / 195,843 |

- **Precision sweep (`lc_tank` on machine D):** at `reltol = 1e-2`, `E_end / E_0` rises from `0.040785` (`0.02` digits, 95.9% energy lost) to `0.999959` (`4.38` digits).
    At `reltol = 1e-2`, accepted steps drop `85 -> 67`, rejected steps drop `39 -> 3` (-92.3%), and refactorizations drop `106 -> 28` (-73.6%).
    At `reltol = 3e-3`, `E_end / E_0` improves from `0.622130` (`0.42` digits) to `0.999950` (`4.30` digits) with rejections dropping `20 -> 3`.
    At `scaled = 1e-2`, `E_end / E_0` rises from `0.031746` (`0.01` digits) to `0.999957` (`4.37` digits), cutting rejections `41 -> 1` (-97.6%) and refactorizations `112 -> 23` (-79.5%).
    At `scaled = 3e-3`, `E_end / E_0` rises from `0.641889` (`0.45` digits) to `0.999961` (`4.41` digits) with rejections dropping `21 -> 3`.
    From `1e-3` through `1e-6` (both `reltol` and `scaled`), `lc_tank` counts and digits (`4.20` to `4.47`) are bit-identical before and after.
    `speed_buck` runs faster (`1.7667 s -> 1.7424 s`, -254 accepted, -435 rejected, -2,393 work iterations, -2,443 refactorizations), and `inverter_chain_500` takes fewer steps (`2,391 -> 2,312` accepted, `15,340 -> 14,884` refactorizations).
- **Source:** first run of the precision suite (`benchmarks/precision/README.md`).

### ✅ 17. AC's lost digits near 650 MHz

- **What:** on the precision suite's RLC circuit, jolty and ngspice KLU lose three digits between 600 and 700 MHz (11.1 and 11.4 correct digits), whereas ngspice Sparse retains 14.2.
    Complex solve pivoting was the primary suspect for diagnosis.
- **Done when:** response accuracy stays flat across frequency with the cause documented.
- **Result** (`ae943be`, 16 November): accuracy fell with response magnitude rather than frequency band, tracking `16 + log10|v(out)|` (an absolute error of ε volts, from 13.8 digits at 100 MHz to 11.4 at 1 GHz).
    The frequency sweep reused pivots selected at 1 Hz where conductances dominate, causing elimination to compute the output as a cancellation of volts.
    Every `G + jωC` solve (AC, noise, stability, AC sensitivity, pole-zero) now applies working-precision iterative refinement while Oettli and Prager's componentwise backward error exceeds ε and each step halves the error (Higham §12.3; Skeel 1980).
    RLC accuracy now stays at 15.6 to 16.5 digits from 1 Hz to 1 GHz, backed by a regression test holding the RLC within 32 ε.
    Interleaved against `3945c01` on the local Mac, cost is +10% on `ac_wide` (20,000 solves of 22 unknowns, worst case) and +2% on `bjt_amp`.
    Using `|re| + |im|` and reusing buffers across solves reduced that overhead from an initial 26 to 29% caused by complex modulus and allocation.
    On `ac_wide`, backward error ω was already 0.71 ε at the median and 2.3 ε at worst, so 13% of solves refined.
- **Source:** first run of the precision suite.

### 🚧 18. Shortest time step derived rather than chosen

- **What:** jolty's shortest step was 10⁻⁹ of the longest step and ngspice's is 10⁻¹¹, neither derived; the 1,000-node ladder fails from a scaled tolerance of 5.6e-7 because the initial step needed is below 8 ps.
- **State** (`f673b49`): aligned with ngspice at 10⁻¹¹ of the longest step, removing a departure and letting the buck resolve its 10 ps settling, though still chosen rather than derived.
- **Done when:** the lower bound is derived from floating-point resolution and the ladder solves at every tolerance.
- **Source:** first run of the precision suite; departures guide.

### 🚧 19. Departures from ngspice and Xyce found on 15 November added to the guide

- **What:** three departures uncovered by the benchmark suites require documentation.
    First, ngspice 47 multiplies a bipolar transistor's base-collector saturation current by area twice when the card omits `IBE` and `IBC` (`bjttemp.c`, lines 168, 177-179, 198-202), shifting `vreg` output by 8 to 12 mV.
    Second, ngspice's AC sensitivity is accurate only at the first frequency because `lin` sweep frequencies grow geometrically (pending source verification).
    Third, Xyce's thermal noise exceeds 4kTR by 9.5e-6 due to its 1973 electron charge value.
- **Xyce departures from agreement tests:** Xyce uses SPICE3f5's linear MOSFET reverse bulk junction where jolty and ngspice 47 use an exponential with a cubic tail.
    Xyce also divides bipolar `gmin` by beta inside junction current, omits JFET noise, uses SPICE3f5 JFET temperature dependence, and implements a distinct excess-phase formulation.
    Two differences remain unexplained: at 125 °C, `nmos_stage` output differs from jolty by 1.4e-4, and `cmos_inverter` differs by 55 µV across the switching threshold.
    Xyce finds no operating point for `negative_load` after 7,692 Newton iterations and stalls on the buck.
- **Done when:** every departure is documented in `docs/guide/departures.md` with evidence, ngspice's AC sensitivity source is verified, and both unexplained Xyce differences are resolved.
- **Result, in part** (`00a9e75`, 16 November): `departures.md` now records ngspice's double area multiplication for base-collector saturation current (`bjttemp.c`, lines 168, 177-179, 198-202).
    Source inspection confirmed the ngspice AC sensitivity linear-sweep bug: `inc_freq` compares `SENS_LINEAR` (15) against AC `LINEAR` (3), multiplying each subsequent frequency by the step (`cktsens.c`, line 831).
    Xyce's departures are documented in the SPICE guide under the names used in `tests/test_xyce.py`.
    **Remaining:** verify Xyce's electron charge in source for the 9.5e-6 noise offset, and explain the two remaining Xyce differences.
- **Source:** first runs of the corpus and coverage matrix.

**Elsewhere:** the precision suite and hard-circuit corpus live in `benchmarks.md` items 7 and 8; their findings appear in items 4, 8, 9, 10, and 16 to 19 above.

## 5. Decided

- `truncation_tolerance` means SPICE's `TRTOL`.
- Every charge's error is bounded: on the buck, jolty takes 90,774 attempts versus ngspice's 56,624 points, and 37% fewer steps than ngspice at equal error control.
- SPICE3's junction limiter is kept over ngspice 47's `DEVpnjlim`, which halves buck Newton failures without reducing total work.
- A transient step starts from its prediction.
- `gmin-dependence` is a warning.
- Meyer's capacitances conserve charge in neither engine, and documentation no longer calls jolty's form charge-conserving.
- Every transient change follows a single protocol: probe and explain every rejected step before modifying code.
    Record before-and-after accepted, rejected, and landed steps plus Newton iterations at defaults on the buck, boost, ring, relaxation, 555 timer, astable, and Colpitts examples alongside ngspice on identical netlists.
    Keep accuracy unchanged with `test_ngspice.py`, `test_random_circuits.py`, `test_metamorphic.py`, and analytic tests passing without loosened tolerances.
    Update the guide's methods table to specify each change and its source.
- Old roadmap item 7 was measured and set aside because bounding repeated rejections at a single time point saves too little.
    A damped formula for the boost converter's stiff mode would affect only 36% of steps, so item 14 evaluates TR-BDF2 there first.
- `max_transient_iterations` (formerly `transient_iterations`) keeps SPICE's `ITL4` of 10, although ngspice 47 raises lower limits to 100 (documented as a departure).
    Raising the constant for a single fixture would be arbitrary; item 6 provides the principled fix.

## 6. Open principles

None; the research proposals were merged into `AGENTS.md` on 15 November (`444535f`).
