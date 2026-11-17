# Research: accuracy and reliability

Research for `accuracy.md`, written on 15 November 2026 to `brief-research.md`.
Read: jolty's `main` at `4f39ff4` (it moved from `68c3f75` during the research; the lines cited are `4f39ff4`'s), ngspice 47's source (the two files the lead patched for probes, `mos1trun.c` and `mos1load.c`, read from the pristine tarball beside it), Xyce's `develop` at `e5b7b9b` (19 October 2026, read for ideas only, no code taken), the notes, and published papers and manuals.
Nothing was built or run but two small scripts in the scratchpad: a PDF text extractor, and an exact-arithmetic check of Gear2's error constant (`scratchpad/scripts/bdf2_constant.py`).

## 1. Summary

jolty's numerics already go further than ngspice's in kind: exact derivatives, a test of every equation's residual with a floor at its rounding, error control on every node's charge filtered through the step's own equations, a judged first step after every corner, deflation to find further DC solutions, and the same bits on any number of threads.
Nothing yet measures its error against an exact answer, and this research found three places where its error control or its robustness is weaker than its documentation says:

1. **`charge_tolerance` is not SPICE's `CHGTOL`.**
    jolty allows `reltol·|q| + chgtol` of charge error per step (`transient.rs:1246`), SPICE `reltol·max(|q|, chgtol)` (`cktterr.c:40`): on a net holding under 10 fC, as most integrated-circuit nets do, jolty allows about 1,000 times ngspice's error at default options.
1. **ngspice 47 gives each time step 100 Newton iterations, not 10:** `NIiter` raises any limit below 100 to 100 (`niiter.c:37`), so `ITL4 = 10` is never applied, while jolty stops at 10.
    This is the likeliest cause of roadmap item 18's 2,768 failures on the buck, and a one-option experiment tests it.
1. **Gear2's error estimate uses the constant-step constant 2/9** in both engines; after a step cut to a tenth the true error is 7.6 times the estimate.

The three moves that matter most: a precision suite against exact answers that reports error against cost for every simulator; error control that means what it says, judged in the unknowns, which the filter already solves for; and a Newton iteration stopped by its observed rate of convergence rather than by a count.

## 2. Where jolty stands

### 2.1 Newton iteration and its convergence test

- **Full Newton steps**, with no damping and no line search; device limiting is the only globalization (`newton.rs:324-382`).
- **Converged** when three things all hold (`newton.rs:324-382`):
    - The last update is within `reltol·max(|x|, |x − δ|) + abstol` of every unknown (`newton.rs:35-40`, `518-531`).
    - Every equation's residual is within `reltol` times its largest term, plus its absolute tolerance, plus 8 roundings of `Σ|J_ij·x_j|` (`newton.rs:11-18`, `493-497`).
    - No device limited a voltage (`newton.rs:344-351`).
- **The residual test is what Kundert and Clifford asked of "a good simulator"**: it should check Kirchhoff's current law directly rather than device currents between iterations, which lets each node leak a small current (*Achieving Accurate Results with a Circuit Simulator*, EDA&T 1993).
  ngspice tests the updates (`niconv.c`) and each device's predicted current (`MOS1convTest`, compiled through `NEWCONV`, `macros.h:19`), but never the residual.
- **The rounding floor is jolty's own.**
    It is why a milliohm between nodes at hundreds of volts converges (`analytic.rs:140`), and why jolty finishes the memristive NOR discovery that ngspice 47 cannot finish at any tolerance (handoff, "Left as it was").
- **Limits:** 100 iterations for a DC solution and 10 for a time step (`options.rs:283-284`), SPICE's documented `ITL1` and `ITL4`.
- **A gap in the test itself:** "last update within tolerance" bounds the error only when convergence is fast.
  Converging linearly at a rate θ, the error left is about θ/(1 − θ) times the last update (Hairer and Wanner, *Solving ODEs II*, 1996, IV.8; Deuflhard, *Newton Methods for Nonlinear Problems*, 2004).
  At the buck's 0.68 an iteration (handoff, item 18), that is 2.1 times the update the test accepted.
  The residual test bounds part of this, but not on a high-impedance node, where a small current is a large voltage.

### 2.2 Finding the operating point

- **The chain** (`operating_point.rs:692-765`):
  1. Newton from zero, its first iteration unlimited (`newton.rs:341-344`).
  1. A conductance from every node to the ground, stepped down from 10 mS (`operating_point.rs:28-34`, `773-851`).
  1. The same stepping on the conductance across every junction, down to `gmin`.
  1. Source stepping from zero: an increment of 0.1, doubled after a success, quartered after a failure, abandoned below 10⁻⁴ (`operating_point.rs:35-38`, `911-960`).
  1. Pseudo-transient continuation (Kelley and Keyes, *SIAM Journal on Numerical Analysis* 35(2), 1998): every node held to its last voltage by a conductance that starts at 1 S, shrinks tenfold after a quick solve and by √10 after a slow one, grows tenfold after a failure, and gives up past 10⁹ S or 1,000 steps (`operating_point.rs:39-46`, `853-909`).
- **Shown by tests:**
    - Pseudo-transient continuation passes a fold where source stepping stops, at 51% of the sources, and finds ngspice's transient operating point to its printed digits (`operating_point.rs:990-1037`, `tests/convergence.rs:69-95`).
    - Gmin stepping across the junctions finds a saturated transistor that the node shunt misses, as ngspice's true gmin stepping does (`tests/convergence.rs:19-67`).
- **Deflation finds the other DC solutions** (`newton.rs:383-418`; Farrell, Birkisson and Funke, SIAM J. Sci.
  Comput.
  37(4), 2015).
  None of the references enumerates solutions: Xyce's arc-length continuation traces them along a path, ngspice has nothing of the kind, and Spectre's is unknown.
- **The gaps:**
    - The pseudo-transient holds every node with the same conductance, 1 S at first, a farad over a step of a second, rather than through the circuit's own capacitances.
    - Its step follows a count of Newton iterations, not the residual's decrease, which is the rule Kelley and Keyes analyse (switched evolution relaxation).
    - jolty has no arc-length continuation, no MOSFET homotopy, and no `MODEINITJCT`-style start: ngspice starts each MOSFET at `vbs = −1`, `vgs = vto`, `vds = 0` (`mos1load.c:397-408`) and each diode at its critical voltage (`dioload.c:162-165`).
        `accuracy.md`'s plan 5 now holds the last one.
    - Which strategy succeeds can turn on the rounding of the equations' sums (`tests/convergence.rs:75-79`): deterministic, but fragile.
- **Reliability is tested against ngspice only:**
    - A random circuit fails its test when ngspice finds an operating point and jolty none (`tests/test_random_circuits.py:84-102`).
    - Where jolty stops short in a transient, the same circuit with 10 fF on every net must finish (`tests/test_random_circuits.py:221-246`).
    - The failure rate on a corpus of hard circuits is unknown.

### 2.3 Junction limiting and roadmap item 18

- **jolty's limiters:**
    - SPICE3's `pnjlim`, with ngspice's floor on reverse steps (`junction.rs:172-197`).
    - `fetlim` and `limvds` (`junction.rs:200-283`).
    - The MOSFET limits from its last evaluation (`mosfet.rs:1337`, `1345`), and any of the three limiting blocks convergence.
- **Item 18, as measured** (handoff, "Roadmap item 18" and "Traced side by side"):
    - On the benchmarks' buck, about one step per gate edge fails its Newton iterations: 2,768 failures.
    - The first iterate puts the switched node at −3.0 V where it settles near −0.8 V, forward-biasing M2's body diode by about 3 V.
    - `pnjlim` then raises the junction 0.08–0.1 V an iteration, a linear convergence at 0.68 an iteration.
    - At the traced edge, ngspice steps 3.5 + 3.5 ns and converges in 8 iterations, 5 of them limited; jolty steps 4 + 3 ns and fails its 10, 9 of them limited.
    - ngspice fails no Newton iteration in 2 ms: a breakpoint on `dctran.c:733`, its cut by 8, was never hit.
- **Already tried, by the lead:**
    - ngspice 47's `DEVpnjlim` halves the failures (2,768 to 1,289) but leaves the steps where they were (73,891 / 16,760 against 73,880 / 16,894).
    - Limiting from the last accepted point saves 1.5% of the steps.
    - Starting Newton from the last point instead of the prediction costs 18% more iterations.
    - ngspice's rule that equalises the two steps before a breakpoint costs more than it saves.
- **New: ngspice's limit is 100, not 10.**
    - `NIiter` opens with "some convergence issues that get resolved by increasing max iter: `if (maxIter < 100) maxIter = 100;`" (`niiter.c:37-39`).
    - `dctran.c:709` calls it with `CKTtranMaxIter`, which `cktntask.c:106` sets to 10, so every time step gets 100 iterations; the DC sweep's 50 (`cktntask.c:108`) become 100 too.
    - jolty stops at 10 (`options.rs:284`) and cuts the step by 8.
    - The trace saw one edge, which ngspice closed in 8 iterations.
        It does not rule out edges where ngspice needs 11 to 100, which is all it takes for ngspice to fail nowhere.
    - The wasted work is about 27,700 iterations, 2,768 failures of 10 each: 9% of the buck's 299,616 (raw logs, not kept).
        On top come the steps the cuts force, about three to regrow from an eighth at twice a step.
        Both figures are estimates.
    - **The experiment** needs no code: the buck and the protocol's examples with `transient_iterations=100`, counting failures, steps, iterations and time.
- **Both limiters are discontinuous at a step of 2vt:**
    - SPICE3's lets a step of just under 2vt through whole, but cuts one just over 2vt to vt·ln 3 ≈ 1.10vt.
    - ngspice 47's steps backward for Δ/vt between 2 and 2 + e⁻² ≈ 2.14, towards −∞ as Δ approaches 2vt (`devsup.c:54-61`).
        This is the review's "steps backward just above 2 vt".
    - `vold + vt·ln(1 + Δ/vt)` is exactly the voltage at which the junction's current equals what its linearization at `vold` predicted for the applied voltage; the saturation current cancels.
      Applied to every forward step above the critical voltage, with no switch at 2vt, it is monotone in the step and continuous everywhere but at the critical voltage itself, which SPICE derives rather than picks.
    - Xyce carries both SPICE3's `pnjlim` and a copy of ngspice's (`N_DEV_DeviceSupport.C:181-260`); which its devices use by default was not checked.

### 2.4 Time integration and error control (roadmap item 7)

- **Methods:** the trapezoidal rule by default, Gear2 or backward Euler (`integration.rs:14-28`).
- **Around a corner:**
    - Backward Euler until `p + 2` points follow the corner (`transient.rs:756-760`).
    - The history restarts at each corner (`transient.rs:204`), and the first step after it is judged by the second's points (`transient.rs:848-897`).
- **The estimate and the step:**
    - The estimate is filtered through `C (G + αC)⁻¹ α` (`transient.rs:1292-1327`; Shampine, SIAM J. Sci.
        Stat.
        Comput.
        1(1), 1980).
    - It is held to the larger of a charge tolerance and a current tolerance times the step, scaled by `trtol·(p+1)!` (`transient.rs:1230-1253`).
    - The next step is the longest that meets either tolerance as it scales with the step (`integration.rs:52-71`), within a factor of 0.1 to 2 and a safety factor of 0.9; a Newton failure cuts the step to an eighth (`transient.rs:25-37`).
- **Ringing and jumps:** a step of backward Euler when a charge's derivative turns back and forth twice in a row (`transient.rs:1031-1060`); behavioral sources' jumps are located.
- **Measured:**
    - At default options the examples take within 2% of ngspice's steps, and about half its Newton iterations (commit `ee503e0`).
    - On an RC low-pass, the error shrinks as the tolerance to 0.48, 0.63 and 0.64 for backward Euler, the trapezoidal rule and Gear2, against 0.5, 0.67 and 0.67 in theory (`ee503e0`; `analytic.rs:598-626`).
    - Each method converges at its order as the step halves, against a closed form and, after a diode clamps, by Richardson (`analytic.rs:766-817`).
- **Finding: the charge tolerance has another meaning than SPICE's.**
    - jolty: `relative_tolerance · |q| + charge_tolerance` (`transient.rs:1246-1247`).
    - ngspice: `reltol · max(|q|, chgtol)`, the only use of `CHGTOL` in SPICE3's engine (`cktterr.c:40-41`; XSPICE's `miftrunc.c:184` copies it).
    - With the factorials cancelled, the two criteria are otherwise the same (`cktterr.c:37-42`, `69-76`).
    - For `|q| < chgtol = 10 fC`, ngspice's floor is `reltol·chgtol` = 10⁻¹⁷ C and jolty's 10⁻¹⁴ C, a thousand times larger.
    - A net of 1 fF at 1 V holds 1 fC: ngspice lets a step's estimated error reach 7·10⁻¹⁷ C (7% of it), jolty 7·10⁻¹⁴ C (70 times it), so the charge test bounds nothing there and steps follow `max_step`, the doubling limit and other nets.
        This is an estimate from the formulas, not a measurement.
    - The guide says each option "left out" takes "SPICE's default" (`simulation.md:303`): the value is SPICE's, the meaning is not.
    - The differential tests cannot see it, since both engines run with `chgtol = 1e-18` (`tests/differential.py:52-57`, `src/jolty/spice/ngspice.py:37-38`).
    - The inverter chains' nets of a few femtofarads (`perf/circuits/inverter_chain.py:33-44`) and the logic characterizations' femtofarad loads (`tests/test_fidelity.py:54-75`) run at defaults.
    - Whether this is part of why jolty takes fewer iterations than ngspice on the chains (7,335 to 10,181) is unknown.
- **Finding: Gear2's error constant ignores the step ratio.**
    - jolty uses 2/9 (`integration.rs:84-91`), the constant of equal steps.
    - For a step `r = h/h_prev` times the last, the formula `Formula::new` builds (`integration.rs:122-150`) errs by `(1+r)²/(6r(1+2r))·h³·q'''`.
    - Checked in exact rational arithmetic on a cubic: the true error is 7.56, 3.13, 1.69, 1.00 and 0.68 times the 2/9 estimate at r = 0.1, 0.25, 0.5, 1 and 2.
    - The unit test checks equal steps only (`integration.rs:236-255`).
    - ngspice has the same constants (`cktterr.c:24-31`, `60-63`).
        Xyce's Gear forms its constant from the step history, as DASSL does (`N_TIA_Gear12.C:1077-1104`).
    - The trapezoidal rule and backward Euler are one-step methods, whose constants do not depend on the ratio, so the default method is unaffected.
- **Item 7, what the references say:**
    - **Variable-order BDF:**
        - Xyce deprecated its variable-order BDF method in 6.5 and removed it in 6.8, keeping the trapezoidal rule and Gear of orders 1 and 2, with a maximum order of 2 by default (Release Notes 6.5 and 6.8, `timetbl.tex`).
        - BDF of order 3 to 6 is not A-stable: its stability angles are 86°, 73°, 52° and 18° (Hairer and Wanner 1996, V.2), so oscillators' modes near the imaginary axis are damped or lost.
        - Every corner restarts the order at 1, and 64% of the boost's accepted steps sit at its `max_step` already (handoff, idea 5).
    - **Dense output:**
        - jolty lands on every time asked for (`transient.rs:737-751`), and the differential tests make it step onto every time ngspice took (`tests/test_random_circuits.py:168-173`).
        - The cubic Hermite interpolant of `q` and `q̇`, which jolty keeps at every point, would report any time without stepping there.
        - So the circuit where dense output helps exists: the comparison harness, and any agent that samples at fixed times.
    - **TR-BDF2** (Bank and others, IEEE Trans.
        CAD 4(4), 1985; Hosea and Shampine, Appl.
        Numer.
        Math.
        20, 1996):
        - One-step and L-stable, with an estimate built in: no history crosses a corner, and nothing rings.
        - It costs two solves a step, and its upside on the boost is the error-limited third of the steps (handoff, idea 5).

### 2.5 Charge conservation and exact Jacobians

- **Charge-oriented equations:** `f(x) + d/dt q(x) = 0` (`simulation.md:344-345`).
  Every device's charge is a function of its voltages, so a charge conserves by construction, but for Meyer's capacitances.
- **Exact derivatives:**
    - Forward automatic differentiation gives every derivative (`dual.rs:1-6`), checked against central differences in each model's regions (`testing.rs:228-260`).
    - The small-signal model is therefore the exact slope of the DC one.
- **Where ngspice's derivatives are wrong or missing:**
    - The bipolar transistor's base resistance (`devices.md:378`).
    - The MOSFET's threshold with the bulk forward-biased (`devices.md:445`).
    - BSIM3's effective bulk voltage clamp, about 3·10⁻⁴ of `gmbs`, and `capmod` 0 with poly depletion, 2.5·10⁻⁴ of the AC response (`devices.md:485`, an archived note, not kept).
    - VDMOS's quasi-saturation, its subthreshold mobility, and the sign of its reverse channel-length modulation (`devices.md:549`).
- **Exact derivatives bought robustness:** ngspice stalls with "timestep too small" on the VDMOS buck with pulse-driven gates at the corpus's tolerances, where its Jacobian leaves out a slope (review, `8995de6`).
- **Meyer's capacitances conserve nothing in either engine.**
    - They have no charge function, as `mosfet.rs:105-109` says itself, and models built on them do not conserve charge as SPICE's descendants implement them:
        - Yang, Epler and Chatterjee, IEEE J. Solid-State Circuits 18(1), 1983, which concluded that charge, not capacitance, must be the state.
        - Kundert and Clifford 1993: "all of these models as implemented in Berkeley SPICE and its descendants do not conserve charge".
    - jolty's displacement unknowns (`mosfet.rs:689-741`, `966`) give the discretized current `C(v)·dv/dt` exact derivatives, and put error control on the oxide's linear charge.
    - ngspice accumulates `q = q_n + C̄·Δv` (`mos1load.c:822-823`) and integrates it with a Jacobian of `C̄` alone (`mos1load.c:863-867`).
    - So the notes' "charge-conserving default" overstates (`accuracy.md` §2, `speed.md` plan 2, handoff decision 9): the two differ in their derivatives and in what the error control sees, not in conservation.
    - A charge-based capacitance model would conserve (Ward and Dutton, IEEE J. Solid-State Circuits 13(5), 1978; Sakallah, Yen and Greenberg, IEEE Trans.
        CAD 9(1), 1990).
    - jolty already holds VDMOS's gate-drain charge exactly, where ngspice integrates its capacitance (`vdmos.rs:49-53`); BSIM3 is charge-based in both.
- **Conservation inside the solver:**
    - Each step accepts a residual of up to `reltol·|terms| + abstol` at every node, so up to `h` times that much charge appears per step.
    - "Even simulators with charge-conserving models do not conserve charge completely" (Kundert and Clifford 1993).
    - jolty does not report this leak.

### 2.6 How accuracy is measured today

- **Closed forms:**
    - DC: dividers, a diode against its load line, controlled sources.
    - AC: an RC's corner, an RLC at resonance.
    - Transients: an RC under a ramp for every method, the order of each method, an LC tank's frequency, a sine through an RC settling to the AC amplitude.
    - Every analysis: noise, poles and zeros, the periodic steady state, and an inverter-RC oscillator's period.
    - Sources: `crates/jolty-engine/tests/analytic.rs`, `noise.rs`, `pole_zero.rs`, `periodic.rs`, `logic.rs:240`.
- **Laws:** Kirchhoff's current law on random resistor networks (`crates/jolty-engine/tests/invariants.rs`).
- **Against ngspice:**
    - Operating points and frequency responses within a millionth, transients within a thousandth of the swing, at `TIGHT` (`simulation.md:375-382`, `tests/differential.py:52-57`).
    - On a corpus, on vendors' macromodels, and on random circuits of up to 100 unknowns, with metamorphic variants.
- **Missing:**
    - The error against an exact answer at default options.
    - Error against cost.
    - Any nonlinear transient checked against anything but ngspice, but for one Richardson test.
    - A corpus of hard circuits, and a failure rate.

### 2.7 Determinism

- **The contract** is `AGENTS.md`'s "Determinism".
- **What holds it:**
    - Sums run in chunks of 1,024 in one fixed order on any number of threads (`reduce.rs:1-33`).
    - The tests run every circuit on any number of threads and any split of shards (`crates/jolty-engine/tests/determinism.rs`, `tests/test_determinism.py`, `tests/test_shards.py`).
    - The remembered operating point keeps its solver, so later analyses are the same bits whatever ran between (`c2642f1`), at 26–42% more peak memory on `rc_mesh_300` (review).
- **Where it ends:**
    - A dissected factorization changes the last digits against the serial one (`simulation.md:335`).
    - Across platforms, results agree "only within tolerance … whose math libraries round apart" (`machines.md:48`): the derandomized random circuits differ in their last digits between macOS and Linux (handoff, end of the decisions table).
    - Rust documents `exp`, `ln` and `powf` as of "non-deterministic" precision, which "varies by platform, Rust version, and can even differ within the same execution" (Rust standard library, `f64`).
        jolty's kernels call them, as at `junction.rs:151`.
- **Xyce's guide** says a Schmitt trigger's operating point "has been observed to change with the computational platform" (Xyce Users' Guide, chapter 10).

### 2.8 Departures from ngspice: the start of the guide's list

Each row is a place where jolty and ngspice 47 compute differently, with the evidence for each side.
The verdicts: **right** where jolty is more accurate or more robust and should state it; **owner** where users could reasonably choose otherwise, a knob in the making; **fix** where jolty should change.

| Area | ngspice 47 | jolty | Evidence | Verdict |
| --- | --- | --- | --- | --- |
| Charges judged | MOS levels 1–3 judge their gate charges only, never `qbd` and `qbs`; BSIM3 folds the junctions into `qb` and `qd` | Every node's charge | `mos1trun.c:24-26`, `mos2trun.c:24-26`, `mos3trun.c:21-23`, `b3ld.c:2798-2801`; `transient.rs:1195-1276`; buck 56,624 points against 90,774 attempts, 143,339 when ngspice judges the junctions | right |
| Charge tolerance | `reltol·max(\|q\|, chgtol)` | `reltol·\|q\| + chgtol` | `cktterr.c:40-41`; `transient.rs:1246-1247` | fix, or owner |
| Error filter | None | `C (G + αC)⁻¹ α` | `transient.rs:1292-1327` | right |
| Gear2's constant | 2/9 at any step ratio | 2/9 at any step ratio | `cktterr.c:24-31`; `integration.rs:84-91` | fix in both |
| First step after a corner | Judged at order 1, and order 2 tried at once, both with divided differences that reach back across the corner, since its history is never restarted | Judged by the second step's points, the history restarted at the corner | `dctran.c:489-523`, `796`, `822-831`, `cktterr.c:44-59`; `transient.rs:204`, `848-897` | right |
| Acceptance | When the new step exceeds 0.9 of the last | When the error ratio is at most 1, the next step 0.9 of the longest allowed | `dctran.c:801`; `transient.rs:1330-1337`; SPICE's rule measured worse on the buck, 73,299 / 12,527 against 65,606 / 7,807 (handoff) | right |
| Step against the tolerances | Tolerance taken at the last step's length, `del^(1/order)` | The longest step meeting each tolerance as it scales with the step | `cktterr.c:41`, `69-76`; `integration.rs:52-71` | right |
| Newton's limit per step | 100, raising `ITL4 = 10` | 10 | `niiter.c:37-39`, `dctran.c:709`, `cktntask.c:106`; `options.rs:284` | owner, after measuring |
| Newton's convergence test | Updates, and each device's predicted current | Updates, every residual with a rounding floor, and no limiting | `niconv.c`, `macros.h:19`; `newton.rs:420-531` | right |
| What counts as limited | `pnjlim` only | `pnjlim`, `fetlim` and `limvds` | `mos1load.c:376-384`, `719-720`; `mosfet.rs:1345` | owner |
| Limiting's reference | The last accepted point | The last evaluation | Handoff: 1.5% fewer steps from the accepted point | fix (planned) |
| `pnjlim` | Its own form, which steps backward just above 2vt | SPICE3's | `devsup.c:54-61`; `junction.rs:178-197`; failures 2,768 to 1,289, steps unchanged (handoff) | owner; both discontinuous |
| Prediction | Each device's terminal voltages | Node voltages | `mos1load.c` (`#ifndef PREDICTOR`); `transient.rs:458-476`; first iterates agree (handoff) | same in effect |
| After the iterations run out | The step cut by 8, restarted from the prediction | Cut by 8, resumed from the last iterate | `dctran.c:733`; `transient.rs:784-790` | right |
| Shortest step | 10⁻¹¹ of the longest | 10⁻⁹ of the longest | `traninit.c:36`; `transient.rs:27` | owner |
| Longest step by default | The print step `TSTEP` when under (tstop − tstart)/50, unless `nostepsizelimit` | A fiftieth of the run | `traninit.c:26-34`; `simulation.md:362` | owner; a comparison must set `max_step` |
| Two steps before a breakpoint | Equalised when 1.9 steps would pass it | Not equalised | `dctran.c:539-546`; at `TRTOL` 42 it costs 11% more accepted steps and 47% more rejected (handoff) | right |
| Ringing | Pure trapezoidal rule; `xmu` an option | A step of backward Euler when a derivative rings | `cktntask.c:113-119`; `transient.rs:1031-1060` | owner |
| Behavioral jumps | Stepped across unlocated | Located and landed on | `simulation.md:364` | right |
| Operating point | Newton from `MODEINITJCT`; dynamic gmin, then true gmin; Gillespie's source stepping; a 10 µs transient at 100 ns steps, not checked to be at rest | Newton from zero; node gmin, then junction gmin; source stepping; pseudo-transient continuation until a step leaves the circuit where it was | `cktop.c:57-97`, `init.c:77-93`, `optran.c:296`; `operating_point.rs:692-765` | right, but for the cold start (fix, plan 5) |
| DC sweep after a failure | A full `CKTop` from `MODEINITJCT`, which can jump to another branch | The step halved from the last level | `dctrcurv.c:306-313`; `simulation.md:357` | right |
| Several DC solutions | None | Deflation | `newton.rs:383-418` | right |
| Pivot threshold | 10⁻³ | 10⁻¹ | `cktntask.c:124`; `simulation.md:360`: backward errors below 10⁻¹⁴ against 10⁻⁸ on random circuits | right |
| Resistors under 1 Ω | A conductance | A current unknown | `simulation.md:355` | right; rounding only |
| Physical constants | CODATA 2014: k = 1.38064852e-23, q = 1.6021766208e-19 | The SI's exact values | `const.h:32`, `37`; `junction.rs:19-21`; the tests shift ngspice's temperature to match (`spice/ngspice.py:48-66`) | right |
| Meyer's capacitances | `q += C̄·Δv`, integrated; Jacobian `C̄` alone | `C(v)·dv/dt` through displacement unknowns; exact Jacobian | `mos1load.c:822-823`, `863-867`; `mosfet.rs:689-741`, `966` | owner; neither conserves |
| VDMOS gate-drain | The capacitance integrated by the trapezoidal rule | Its exact charge | `vdmos.rs:49-53`, review | right |
| Derivatives | Wrong or missing in the BJT, MOSFET, BSIM3 and VDMOS rows of §2.5 | Exact | `devices.md:378`, `445`, `485`, `549` | right |
| Excess phase in AC | A pure delay | The Bessel filter the transient integrates | `devices.md:366-368`: 5% apart at `PTF` 30° | right |
| Cards ngspice runs wrongly | VDMOS with `RQ` and `VQ` without `RD`, or `FC·VJ` above 2.5 V | Refused | `devices.md:543`, review | right |
| VBIC noise | `RBP`'s thermal noise across `RE` | Across `RBP` | `vbicnoise.c`; an archived note, not kept 4 | right |
| BSIM3's limiting flag | `pnjlim` only | Gate and drain too | an archived note, not kept 8 | owner |
| `.tf` with no current | 10²⁰ Ω | Infinite | `simulation.md:44` | right |
| XSPICE switches in AC | The negative conductance (ngspice 42) | Written as behavioral currents | `devices.md:166`, `spice.md:71` | right |
| Self-heating | Only its electrothermal models | Every semiconductor on a thermal net | `simulation.md:353` | right |

## 3. The references

### 3.1 ngspice 47

**Its strengths:**

- The broadest set of models, and a long-tested operating-point chain: dynamic gmin, true gmin, Gillespie's adaptive source stepping, and in version 47 a transient operating point by default (`cktop.c:57-97`, `init.c:77-93`).
- `MODEINITJCT` starts each diode at its critical voltage and each MOSFET at its threshold (`mos1load.c:397-408`, `dioload.c:162-165`).
- Variable-order Gear up to 6 (`cktterr.c:24-31`).
- KLU as an option.

**Where it falls short:**

- **Options that mean something else than their documentation:**
    - `ITL4 = 10` is overridden to 100 (`niiter.c:37-39`).
    - `CHGTOL` sits under `reltol` (`cktterr.c:41`).
    - `TRTOL` holds an estimate six times smaller than the trapezoidal rule's true error (`ee503e0`).
- **Error control:**
    - The MOSFETs' junction charges go unjudged (`mos1trun.c`).
    - Gear's constants ignore the step ratio.
    - After a breakpoint its estimates read points from before the corner, at order 1 and when it tries order 2, since its history is never restarted.
- **Convergence:**
    - It tests no residual (`niconv.c`), and its device tests let Kirchhoff's law slip (Kundert and Clifford 1993).
    - Its `pnjlim` is discontinuous and steps backward just above 2vt (`devsup.c:54-61`).
    - Its last resort for an operating point is a transient of fixed length, which it does not check to be at rest (`optran.c`).
- **Models:** hand-written derivatives with known errors (§2.5), and capacitances integrated as `C̄·Δv`, which loses charge.
- **Reproducibility** is not a property it states: one run on one build repeats itself, and nothing more is promised, as far as this research found.

### 3.2 Xyce

**What it does:**

- **NOX:** Newton by default, a full step without line search; interval halving, quadratic, cubic and Moré–Thuente line searches and a trust region as options (`nlstbl.tex`).
- **Newton's limits:** 200 iterations for the operating point and 20 per time step; the transient's Newton test is a weighted update norm of 0.33, DASSL's, with `reltol` 10⁻² and `abstol` 10⁻⁶ (`nlstbl.tex`).
- **LOCA continuation:**
    - Natural-parameter and arc-length continuation, which can follow turning points (Users' Guide, chapter 10).
    - A two-parameter MOSFET homotopy after Roychowdhury, of gain and of nonlinearity (Roychowdhury and Melville, IEEE Trans.
        CAD 25(1), 2006).
    - Gmin stepping; simultaneous or sequential source stepping.
    - Pseudo-transient continuation, which "has not been observed to be as successful as MOSFET-continuation for large MOSFET circuits".
- **The operating point by default:** Newton, then gmin stepping, then simultaneous source stepping.
- **Time integration:**
    - The trapezoidal rule and Gear of orders 1 and 2, its variable-order BDF removed in 6.8.
    - The error estimated on the solution, against the largest signal at the current time by default (`NEWLTE = 1`, `timetbl.tex`).
    - "A reasonable large step out of every non-DCOP breakpoint", left to the error control (`NEWBPSTEPPING`).

**Where it falls short:**

- **It accepts unconverged steps:**
    - `NLSMALLUPDATE`, on by default, takes a stalled Newton iteration with tiny updates as converged and leaves the step to the error estimate.
    - `NLNEARCONV` does the same for near misses.
        Its table says off by default and its own text says on (`timetbl.tex`).
- **It does not judge the first step after a breakpoint:** "the first step after a breakpoint isn't subjected to much error analysis" (`RESTARTSTEPSCALE`, `timetbl.tex`).
- **Small signals:** a global reference for the relative error lets them carry large relative errors.
- **Platform:** its operating points can depend on the platform (chapter 10).
- **Parallel runs:** whether they are reproducible across processor counts is not documented; unknown.

### 3.3 Commercial simulators, as their makers and users describe them

- **Spectre** (Cadence's AMS reference for `tran`, and its community forums):
    - `errpreset` sets the tolerances: `liberal`, `moderate` and `conservative`.
    - It sets the method too: `traponly` under `moderate` and `gear2only` under `conservative`, as users report.
    - `lteratio` derives the error tolerance from Newton's: 3.5 and 10 in logs users posted.
    - The homotopies run in order: gmin, source, `dptran`, `ptran`, `arclength`.
    - Users warn that `gear2only` damps physical oscillation.
- **HSPICE** (its 1998 manual, chapter 7):
    - The trapezoidal rule and Gear of orders 1 to 6.
    - Three step controls: iteration count, truncation error, and `DVDT` on node slopes.
    - Kundert and Clifford (1993) say iteration-count control "should never be used".
        HSPICE's default `LVLTIM` was not found.
- **LTspice**, from users' posts only: a "modified trap" by default; direct Newton, then gmin stepping, then source stepping in its logs.
  Its official help was not read.
- **Where they fall short:** none publishes its error against exact answers, or a failure rate.
    Their limiters and Newton termination are not described publicly; unknown.

### 3.4 The literature

- **Global convergence for DC:**
    - Homotopies converge with probability one for circuits with passivity and no-gain properties (Trajković, Melville and Fang, ISCAS 1990; Melville, Trajković, Fang and Watson, IEEE Trans.
        CAD 12(6), 1993).
    - Large mixed-signal circuits nest their turning points, which defeats plain homotopies; a MOS homotopy with decoupled parameters fixes that (Roychowdhury and Melville 2006).
    - Pseudo-transient continuation (Kelley and Keyes 1998), and for DAEs (Coffey, Kelley and Keyes, SIAM J. Sci.
        Comput.
        25(2), 2003).
    - Compound pseudo-elements that leave the Jacobian's structure alone (Jin, Wu, Niu and Inoue, IEICE Trans.
        Fundamentals E96-A(12), 2013).
    - SIMetrix makes pseudo-transient its last resort and adds capacitance to stop it oscillating, its weak point (SIMetrix reference, "DC Operating Point Algorithms").
- **Newton's termination and damping:**
    - Affine-invariant damping and convergence monitors (Deuflhard 2004).
    - Global approximate Newton methods (Bank and Rose, Numer.
        Math.
        37, 1981).
    - The contraction-based stopping rule of stiff integrators (Hairer and Wanner 1996, IV.8).
    - DASSL's rate test (Brenan, Campbell and Petzold, SIAM 1996).
- **DAEs:**
    - Circuits' index from their topology: loops of capacitors and voltage sources, cutsets of inductors and current sources (Estévez Schwarz and Tischendorf, Int.
        J. Circuit Theory Appl.
        28, 2000).
    - Impasse points, where no solution continues (Chua and Deng, Int.
        J. Circuit Theory Appl.
        17(2), 1989).
- **Charge:** Ward and Dutton 1978; Yang, Epler and Chatterjee 1983; Sakallah, Yen and Greenberg 1990.
- **Measuring accuracy:**
    - The Test Set for IVP Solvers (Mazzia and Magherini, release 2.4, University of Bari, 2008) ships circuit problems with reference solutions and work-precision diagrams in significant correct digits: a ring modulator, a charge pump (Günther, Denk and Feldmann), a transistor amplifier and others.
    - Work-precision diagrams (Hairer and Wanner 1996).
    - Code verification by manufactured solutions (Roache, J. Fluids Eng.
        124(1), 2002; Salari and Knupp, Sandia SAND2000-1444, 2000).
- **Reproducibility:**
    - Reproducible parallel summation (Demmel and Nguyen, IEEE Trans.
        Computers 64(7), 2015).
    - Correctly rounded mathematical functions: the CORE-MATH project (Zimmermann and others); LLVM's libc aims to be correctly rounded by default.

## 4. Gaps and opportunities

Sizes are estimates for one agent: S is days, M one to two weeks, L a month or more.
**Lead** marks an item where jolty would go past every reference rather than catch up.

| # | Item | Impact on the goal | Evidence | Size | Done when | |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | **The precision suite.** Exact answers four ways: linear random circuits solved exactly by their pencil's QZ (jolty's `pencil.rs`); nonlinear circuits by manufactured solutions, `b(t) = f(x*) + d/dt q(x*)` with exact derivatives injected as sources; references converged at `reltol` 10⁻¹⁰ with Richardson's check; and the IVP test set's circuits. Then error against cost for jolty and ngspice, and Xyce and LTspice once installed | Makes "most accurate" measurable; would have caught rows 2 and 4 | `accuracy.md` gaps; `benchmarks.md` 3; §2.6 | L | Every fixture and at least 20 exact-answer circuits have significant correct digits against time for each simulator at five tolerances or more | **Lead** |
| 2 | **`charge_tolerance` with SPICE's meaning**, `reltol·max(\|q\|, chgtol)`, since it changes results the owner's call, as `TRTOL`'s was (decision 4) | High on integrated-circuit nets: about 1,000 times the charge error at default options | `transient.rs:1246-1247`; `cktterr.c:40-41` | S, its re-measurement M | A 1 fF and 1 MΩ RC at default options holds its error within tolerance of the closed form; the chains and characterizations re-measured against ngspice and the exact reference | |
| 3 | **Error judged in the unknowns**, from the `(G + αC)⁻¹ α d` the filter already solves for, against `reltol·\|x\| + vntol` or `abstol`: free of any charge floor, an option beside the charges | Scale-free control at no extra solve | `transient.rs:1292-1327`; Xyce estimates its error on the solution (`NEWLTE`); Spectre derives its error tolerances from Newton's (`lteratio`) | M | The option exists, and its default is chosen by row 1's suite | **Lead**: filtered, local and in the unknowns together |
| 4 | **Gear2's constant from the step ratio**, `(1+r)²/(6r(1+2r))` | `method=gear2` runs held to their tolerance; up to 7.6 times the error today | `integration.rs:84-91`; exact check; `cktterr.c:24-31` | S | The unit test covers step ratios from 0.1 to 2 | Ahead of ngspice, level with Xyce |
| 5 | **Item 18, first experiment**: `transient_iterations=100`, ngspice's own limit, on the buck and the protocol's examples | May remove most of the 2,768 failures, about 9% of the buck's iterations | `niiter.c:37-39`, `dctran.c:709`; `options.rs:284` | S | Failures, steps, iterations and time, before and after | |
| 6 | **Newton stopped by its contraction** `θ_k = ‖δ_k‖/‖δ_{k−1}‖`: go on while θ < 1 and the iterations it predicts fit, stop at θ ≥ 1, and accept on `θ/(1−θ)·‖δ‖` | No arbitrary `ITL4`; hopeless iterations stopped at once; the convergence test bounds the error under linear convergence | Hairer and Wanner IV.8; Deuflhard 2004; DASSL; Xyce's weighted test | M | The buck and the examples at equal accuracy for no more work; any constant the method needs approved by the owner | **Lead** among SPICE-class engines |
| 7 | **A junction limiter without the switch at 2vt**: `vold + vt·ln(1 + Δ/vt)` for every forward step above the critical voltage, limited from the last accepted point | Removes the discontinuity at 2vt that both SPICE3 and ngspice have; 1.5% of the buck's steps from the reference point | `junction.rs:178-197`; `devsup.c:54-61`; handoff | S–M | A property test that the limited voltage is monotone in the applied one and continuous above the critical voltage; the buck's failures and work measured | **Lead** |
| 8 | **The cold start** of `accuracy.md` plan 5 | Fewer operating points falling back to gmin stepping | `mos1load.c:397-408`, `dioload.c:162-165` | S–M | As plan 5 says | |
| 9 | **DC homotopies:** arc-length continuation in the sources' scale (Keller's pseudo-arclength), its `∂F/∂λ` exact as the sources' vector; the MOS homotopy as an option; pseudo-transient steps by the residual's decrease and on the circuit's own capacitances | Turning points; large MOS circuits | `operating_point.rs:853-909`; Xyce's LOCA; Spectre's chain | M–L | Row 10's corpus failure rate falls, with no circuit lost | Level with Xyce; **Lead** with deflation along the path |
| 10 | **A corpus of hard circuits**, and each simulator's failure rate, DC and transient: Schmitt triggers, latches, `folded_sources.json`, impasse points, converters' gate edges, memristors, nets with no capacitance | Makes "most reliable" measurable | `vision.md`: "Unknown" | M | Failure rates and times for jolty and ngspice in `benchmarks.md` | **Lead**: no public corpus with failure rates found |
| 11 | **Failures named as mathematics**: an impasse point when a transient stops where no solution continues; loops of capacitors and voltage sources, and cutsets of inductors and current sources, as rules in `jolty-checks` | An agent learns why, not just that | `tests/test_random_circuits.py:221-246`; Chua and Deng 1989; Estévez Schwarz and Tischendorf 2000 | M | Each failure in row 10's corpus carries a named cause | **Lead** |
| 12 | **Charge-conserving MOSFET capacitances**: a charge model for levels 1 and 3 (Ward and Dutton's partition, or Sakallah's first-order model) as an option beside Meyer's; and the notes' "charge-conserving" corrected | True conservation in switched-capacitor and DRAM-like circuits | §2.5 | M | A closed cycle of voltages returns zero net charge, within the solver's leak of row 13 | **Lead** over ngspice |
| 13 | **The charge leak reported**: each node's residual integrated over a transient, and a property test that a closed system's charge stays within it | Conservation stated as a number | Kundert and Clifford 1993 | S–M | Every transient's statistics carry it | **Lead** |
| 14 | **The same bits on every platform**: a portable or correctly rounded math library for the kernels, measured for speed | Shards from macOS and Linux merge; CI compares hashes across platforms | `machines.md:48`; Rust's documentation; handoff | M | CI's macOS and Linux jobs hash the same waveforms | **Lead** |
| 15 | **The departures in the guide** (`accuracy.md` plan 2) | `AGENTS.md`'s rule met | §2.8 | S | The guide holds §2.8's table, with each row's evidence | |
| 16 | **Item 7, narrowed**: dense output for times asked for and for the comparison harness; TR-BDF2 as an option, measured on the boost; variable-order BDF only for a circuit that shows a gain | Steps no longer forced; the stiff phases | §2.4 | M each | The comparison harness stops stepping onto ngspice's times; TR-BDF2's effect on the boost measured | |
| 17 | **Newton's tolerance in a time step tied to the step's error tolerance**, as DASSL and Xyce do | Fewer iterations at equal accuracy | Xyce's `DELTAXTOL` 0.33 | M | The owner decides, on row 1's numbers | |

## 5. The long view

In two to five years jolty could:

- **Give every result an error bar.**
    - Estimate the global error of the very quantities a spec reads, a delay or an overshoot, by the adjoint method (Cao and Petzold, SIAM J. Sci.
        Comput.
        26(2), 2004), on the adjoint machinery the sensitivities already have.
    - Report "1.234 ns ± 3 ps (estimated)", and refine only until the spec's own measure meets the accuracy asked.
    - No circuit simulator offers this, to the extent this research found.
- **Converge with a certificate.**
    - State when a circuit meets the passivity and no-gain conditions under which a homotopy provably reaches a solution (Melville and others 1993).
    - Find every DC solution by homotopy and deflation together, and for small circuits verify by interval methods that none was missed.
    - Name a failure's cause in mathematics: a fold, an impasse point, an index-2 structure, divergence.
- **Prove models smooth.**
    - With automatic differentiation, a property test can check every model's current, charge and their derivatives for continuity across its region boundaries.
    - Discontinuities are the commonest cause of convergence failures that Kundert and Clifford list.
- **Conserve charge by construction**, with no Meyer capacitance by default, and the solver's charge leak reported.
- **Give one result everywhere:** the same bits on any platform, thread count, shard split or batch.
- **Publish a precision and reliability benchmark** of circuits with exact answers and hard starts, on which every simulator's work-precision curve and failure rate are measured, as the IVP test set is for ODE codes.
  jolty would publish its own numbers first.

## 6. Principles to propose

For `AGENTS.md`, if the owner adopts them:

1. **Accuracy is claimed only against exact answers:** a closed form, a manufactured solution or a reference converged past any default tolerance.
    Agreement with another simulator is agreement, not accuracy.
1. **An option named after SPICE's means SPICE's quantity**, or the guide states the formula where it differs.
    The findings on `CHGTOL` and on ngspice's `ITL4` both broke this silently.
1. **Every error estimate is tested in every condition it is used in:** each order, each step ratio, each restart.
1. **A step is accepted only when its Newton iteration has converged.**
    Iterations that stall or nearly converge are failures, never results, unlike Xyce's `NLSMALLUPDATE`.
1. **Iteration limits follow from the iteration's observed convergence, not a fixed count**, if row 6 lands.
1. **A reactive model is a charge function `q(x)`**, or its option says that it does not conserve charge.
1. **A failure is named as mathematics:** a fold, an impasse point, a singular structure, a divergence.
1. **The same bits on every platform for the same source**, if the owner wants determinism to extend that far; it needs row 14.

## 7. Sources

**jolty** (`main`, `4f39ff4`):

- `AGENTS.md`
- `docs/guide/simulation.md` (lines 303-382), `devices.md` (lines 166, 366-368, 378, 443-445, 485, 541-549), `machines.md:48`, `spice.md:71`
- `crates/jolty-engine/src/newton.rs`, `operating_point.rs`, `transient.rs`, `options.rs`
- `crates/jolty-numerics/src/integration.rs`, `reduce.rs`
- `crates/jolty-models/src/junction.rs`, `mosfet.rs`, `vdmos.rs`, `dual.rs`, `testing.rs`
- `crates/jolty-engine/tests/analytic.rs`, `convergence.rs`, `invariants.rs`
- `tests/differential.py`, `tests/test_random_circuits.py`, `tests/test_fidelity.py`, `tests/strategies.py`
- `src/jolty/spice/ngspice.py`
- `perf/circuits/inverter_chain.py`, `perf/README.md`
- Commits `ee503e0`, `8dcb19f`, `9271cf9`, `8995de6`, `c2642f1`

**The notes:**

- the session log, not kept, `accuracy.md`, `vision.md`, `speed.md`, `models.md`, `benchmarks.md`
- a handoff note, not kept: "Transient steps", "Roadmap item 18", "Decisions", "Fewer transient steps"
- an archived note, not kept, an archived note, not kept, an archived note, not kept, an archived note, not kept, an archived note, not kept
- raw logs, not kept

**ngspice 47:**

- `src/maths/ni/niiter.c`, `niconv.c`
- `src/spicelib/analysis/`: `cktterr.c`, `ckttrunc.c`, `dctran.c`, `traninit.c`, `cktop.c`, `optran.c`, `dctrcurv.c`, `cktntask.c`
- `src/frontend/init.c`
- `src/spicelib/devices/`: `devsup.c`, `mos1/mos1load.c` and `mos1trun.c` (pristine, from the tarball), `mos2trun.c`, `mos3trun.c`, `bsim3/b3trunc.c`, `b3ld.c`, `vdmos/vdmostrun.c`, `dio/dioload.c`, `dio/diotrunc.c`, `bjt/bjttrunc.c`, `jfet/jfettrun.c`, `vbic/vbictrunc.c`
- `src/include/ngspice/const.h`, `macros.h`

**Xyce** (`e5b7b9b`):

- Reference Guide: `nlstbl.tex`, `timetbl.tex`, `OPTIONS_Command.tex`
- Users' Guide, chapter 10, "Homotopy and Continuation Methods", and `Xyce_UG_TimeInt.tex`
- Release Notes 6.5, 6.6 and 6.8
- `src/TimeIntegrationPKG/N_TIA_Gear12.C:1077-1104`, `src/DeviceModelPKG/Core/N_DEV_DeviceSupport.C:181-260` (read for ideas only)

**Papers and books:**

- Bank, Coughran, Fichtner, Grosse, Rose and Smith, "Transient simulation of silicon devices and circuits", IEEE Trans.
    CAD 4(4), 1985.
- Bank and Rose, "Global approximate Newton methods", Numerische Mathematik 37, 1981.
- Brenan, Campbell and Petzold, *Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations*, SIAM, 1996.
- Cao and Petzold, "A posteriori error estimation and global error control for ordinary differential equations by the adjoint method", SIAM J. Sci.
    Comput.
    26(2), 2004.
- Chua and Deng, "Impasse points.
    Part I: Numerical aspects", Int.
    J. Circuit Theory Appl.
    17(2), 1989.
- Coffey, Kelley and Keyes, "Pseudotransient continuation and differential-algebraic equations", SIAM J. Sci.
    Comput.
    25(2), 2003.
- Demmel and Nguyen, "Parallel reproducible summation", IEEE Trans.
    Computers 64(7), 2015.
- Deuflhard, *Newton Methods for Nonlinear Problems: Affine Invariance and Adaptive Algorithms*, Springer, 2004.
- Estévez Schwarz and Tischendorf, "Structural analysis of electric circuits and consequences for MNA", Int.
    J. Circuit Theory Appl.
    28, 2000.
- Farrell, Birkisson and Funke, "Deflation techniques for finding distinct solutions of nonlinear partial differential equations", SIAM J. Sci.
    Comput.
    37(4), 2015.
- Hairer and Wanner, *Solving Ordinary Differential Equations II*, 2nd ed., Springer, 1996: IV.8 and V.2.
- Hosea and Shampine, "Analysis and implementation of TR-BDF2", Applied Numerical Mathematics 20, 1996.
- Jin, Wu, Niu and Inoue, IEICE Trans.
    Fundamentals E96-A(12), 2013, on compound-element pseudo-transient analysis.
- Kelley and Keyes, "Convergence analysis of pseudo-transient continuation", SIAM J. Numer.
    Anal.
    35(2), 1998.
- Keller, "Numerical solution of bifurcation and nonlinear eigenvalue problems", in *Applications of Bifurcation Theory*, Academic Press, 1977.
- Kundert and Clifford, "Achieving Accurate Results with a Circuit Simulator", EDA&T 1993 (kenkundert.com).
- Mazzia and Magherini, Test Set for IVP Solvers, release 2.4, University of Bari, 2008 (archimede.uniba.it/~testset).
- Melville, Trajković, Fang and Watson, "Artificial parameter homotopy methods for the DC operating point problem", IEEE Trans.
    CAD 12(6), 1993.
- Nagel, "SPICE2: A computer program to simulate semiconductor circuits", UC Berkeley ERL-M520, 1975.
- Roache, "Code verification by the method of manufactured solutions", J. Fluids Eng.
    124(1), 2002.
- Roychowdhury and Melville, "Delivering global DC convergence for large mixed-signal circuits via homotopy/continuation methods", IEEE Trans.
    CAD 25(1), 2006.
- Sakallah, Yen and Greenberg, "A first-order charge conserving MOS capacitance model", IEEE Trans.
    CAD 9(1), 1990.
- Salari and Knupp, "Code verification by the method of manufactured solutions", Sandia SAND2000-1444, 2000.
- Shampine, "Implementation of implicit formulas for the solution of ODEs", SIAM J. Sci.
    Stat.
    Comput.
    1(1), 1980.
- Trajković, Melville and Fang, "Passivity and no-gain properties establish global convergence of a homotopy method for DC operating points", ISCAS 1990.
- Ward and Dutton, "A charge-oriented model for MOS transistor capacitances", IEEE J. Solid-State Circuits 13(5), 1978.
- Yang, Epler and Chatterjee, "An investigation of the charge conservation problem for MOSFET circuit simulation", IEEE J. Solid-State Circuits 18(1), 1983.

**On the web:**

- The Rust standard library's `f64` documentation, its "Unspecified precision" notes (doc.rust-lang.org).
- Cadence's Spectre AMS reference for `tran`, and Cadence community threads on `errpreset`, `lteratio` and the homotopies.
- Star-HSPICE manual, chapter 7 (web.engr.oregonstate.edu).
- SIMetrix simulator reference, "DC Operating Point Algorithms" (help.simetrix.co.uk).
- LLVM libc's math documentation; the CORE-MATH announcement on the LLVM libc-dev list (2022).
- LTspice users' posts on its "modified trap" and its operating-point log.
