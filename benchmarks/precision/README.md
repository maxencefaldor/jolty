# Precision

The precision suite measures each simulator's error against exact answers versus runtime, generating a work-precision diagram per fixture that plots significant correct digits against engine time.
Every reference answer is computed independently of all simulators from closed-form formulas or high-precision solutions of the circuit equations; no simulator run ever acts as a reference.
The precision suite stays outside `scripts/check`, like the speed suite, because execution measures wall-clock timings.

## Running

```sh
scripts/bench --precision                       # every fixture, in jolty, ngspice, and Xyce (when installed)
scripts/bench --precision rc_step lc_tank       # selected fixtures only
scripts/bench --precision --simulator jolty     # one or more specific simulators
scripts/bench --precision --sweeps reltol       # one tolerance sweep, or both
scripts/bench --precision --repeats 9           # repeat count used for median timing per setting
scripts/bench --precision --plot RESULTS        # regenerate diagrams and tables from a results file
scripts/bench --view equal-error                # fastest runtime per simulator reaching equal digits
```

Results are written to `results/<commit>.json` (with `-dirty` for modified working trees), and later runs on the same commit merge new fixtures into that file so the suite can run incrementally.
Per-fixture diagrams, the default-settings table, and the equal-error table go to a sibling directory of the same name, which `--plot` can regenerate from the JSON file.
Computing exact reference answers for 1,000-node networks takes about a minute each on the test host below, and the full suite took 6 min 18 s across three runs.

## Fixtures

| Fixture | Circuit and analysis | Exact answer | Scored |
| --- | --- | --- | --- |
| `rc_step` | 1 kΩ into 1 nF under a 1 V step rising in 1 ns, for 5 µs | Exact single-mode response to each linear segment of the source | `v(out)` at every point, mixed |
| `rc_ramp` | Same RC network under a 1 V ramp over 2 µs, for 6 µs | Same as `rc_step` | Same as `rc_step` |
| `rc_femto` | 1 MΩ into 1 fF (at most 1 fC) under a step rising in 1 ps, for 5 ns | Same as `rc_step` | Same as `rc_step` |
| `rlc_step` | Series RLC (10 Ω, 1 µH, 1 nF), damped at 0.16 of critical, under the step for 2 µs | Two complex modes via mpmath eigenvectors | Same as `rc_step` |
| `lc_tank` | Undamped LC tank (1 µH, 1 nF) excited by a 1 mA current source turned off in 1 ns, for 4 µs (20 periods) | Two imaginary modes | Relative energy at every point, plus frequency from total state phase advance |
| `diode` | Diode (10 fA saturation current) fed through 1 kΩ, swept from 0.1 V to 5 V in 50 steps | Lambert W function including 1 pS junction gmin | `v(anode)` at every level, relative |
| `rlc_ac` | Small-signal response of a series RLC low-pass filter (Q = 31.6) at 20 frequencies per decade from 1 Hz to 1 GHz | `1/(1 - ω²LC + jωRC)` | `v(out)` at every frequency, relative |
| `rc_noise` | Thermal noise of 1 kΩ into 1 nF at 10 frequencies per decade from 1 Hz to 1 GHz | `4kTR/(1 + (ωRC)²)` with exact SI `k` | Noise density at every frequency, relative |
| `butterworth_poles` | Poles of a fifth-order Butterworth LC ladder between 50 Ω terminations, cut off at 1 MHz | Eigenvalues of the state equations | Every pole, relative |
| `rc_periodic` | Periodic steady state of 1 kΩ into 10 µF driven at 1 kHz (time constant equal to ten periods) | Phasor steady-state solution | `v(out)` over the period, mixed |
| `ladder_10`, `_100`, `_1000` | RC ladders of 10, 100, and 1,000 nodes (log-uniform 10 Ω to 10 kΩ, 1 pF to 1 nF) under a step for five slowest time constants | Eigenpairs of the symmetric pencil | 50 nodes at every point, mixed |
| `tree_10`, `_100`, `_1000` | Random RC trees of 10, 100, and 1,000 nodes with each parent drawn among earlier nodes and the same value ranges | Same as `ladder_*` | Same as `ladder_*` |

Every circuit is constructed with jolty's builder, and the reference solver reads the exact floating-point element values stored in the circuit so every simulator solves the identical network.
Network element values come from NumPy's RNG seeded with 1,000 plus ladder size or 2,000 plus tree size, guaranteeing identical fixtures across machines.
Eigenvalue rates span 4.5 decades for `tree_10` up to 10.1 decades for `ladder_1000`.
Network transients cover five slowest time constants rounded up to one significant digit, ranging from 30 µs for `tree_10` to 0.4 s for `ladder_1000`.

### How the exact answers are found

Under a piecewise-linear source, each mode of a linear circuit satisfies `żₖ = λₖ zₖ + u`.
Along a linear segment `u = uⱼ + sⱼ τ`, the exact solution is `zₖ(0) e^{λₖτ} + uⱼ τ φ₁(λₖτ) + sⱼ τ² φ₂(λₖτ)`, where `φ₁(x) = (eˣ - 1)/x` and `φ₂(x) = (eˣ - 1 - x)/x²`.
mpmath evaluates decay rates, residues, and segment-start mode states at 50-digit precision before rounding them to double precision.
The mode sum is evaluated in double precision at each simulator output point alongside a rounding-error bound that caps claimable digits (from 12.4 on `ladder_1000` to 15.7 on closed forms, well above any simulator run).
Small-circuit modes come from mpmath's eigensolver applied directly to the state equations.
RC network modes come from the symmetric pencil `C^{-1/2} G C^{-1/2}`, a tree structure whose elimination introduces zero fill-in.
Each eigenvalue uses Newton iteration on the determinant seeded from LAPACK and verified by Sylvester inertia, each eigenvector uses inverse iteration, and every node's residues are verified to sum to unit DC gain.
Diode voltages use Lambert W, AC, noise, and periodic steady state use closed forms, and Butterworth poles use state-equation eigenvalues, all evaluated in mpmath.
`tests/test_precision.py` cross-checks every reference solver against an independent method: the tree solver against mpmath's general eigensolver, modes against the RC closed form, Butterworth poles against the unit circle, and diode voltages against nodal current balance.
The test file also runs `rc_step` in jolty and in ngspice (when installed) and scores both outputs, verifying the suite's scoring pipeline inside the regular test run.

## Scoring

At each output point (time, sweep level, or frequency), the exact answer scores the worst-point significant correct digits, `-log₁₀ max |found - exact| / |exact|`, matching the IVP test set's `scd`.
Quantities crossing or approaching zero use a mixed scale added to `|exact|`: 1 V for step responses and steady-state amplitude for `rc_periodic`.
Reported scores never claim more digits than the exact-answer error bound or IEEE-754 double precision permits.
Alongside each accuracy score, the suite records median engine runtime across repeats plus transient step counts (accepted steps, rejected steps, and Newton iterations); jolty does not yet report solver counts for non-transient analyses.

## Sweeps

Each fixture runs at simulator defaults (ringed on each diagram) and along two tolerance sweeps:

- **`reltol`:** sweeps relative tolerance alone from 1e-2 to 1e-8 in quarter-decade steps, following IVP test-set diagrams while keeping other tolerances at defaults;
- **`scaled`:** scales every tolerance together by the same factor (relative tolerance 1e-2 to 1e-8), scaling `reltol`, `vntol`, `abstol`, and `chgtol` from SPICE defaults of 1e-3, 1e-6 V, 1e-12 A, and 1e-14 C.

Every simulator keeps `trtol` at 7 and caps transient step size at the default one-fiftieth of the simulation window.

Xyce uses separate tolerances from SPICE: relative, absolute, and residual tolerances for DC (`NONLIN`) and transient (`NONLIN-TRAN`) Newton iterations, plus relative and absolute time-step error tolerances (`TIMEINT`).
Along either sweep, all relative tolerances take the sweep value, while absolute and residual tolerances stay at Xyce 7.10 defaults (`suite.XYCE_DEFAULTS` from source) or scale by the same factor under `scaled`.

## Simulators

| Simulator | How it runs | Engine time |
| --- | --- | --- |
| `jolty` | `Simulation.run` on a fresh simulation of the pre-elaborated circuit after one warm-up run, matching the speed suite so every repeat re-solves DC | Wall time of `Simulation.run`, covering engine work and Python result construction |
| `ngspice-sparse` | ngspice 47 on exported `jolty.spice.export` netlists saving scored vectors, with Sparse 1.3 at the temperature matching jolty's thermal voltage via `jolty.spice.ngspice` | Delta of `Total analysis time` from `rusage`, which measures wall time (a 25 s pause added 25 s to a 4.9 s run, reporting 29.9 s) |
| `ngspice-klu` | Same as `ngspice-sparse` with `.options klu` | Same as `ngspice-sparse` |
| `xyce-serial` | Xyce 7.10 on Xyce-dialect netlists printing the same vectors at the same temperature via `jolty.spice.xyce`, using a fresh process per repeat | Xyce `Total Simulation Solvers Run Time` |

Each setting runs across simulators back to back under shared host load, recording the load average at the start of every run.
ngspice executes repeats inside one process, so the first repeat includes matrix setup and the median drops that outlier, whereas jolty repeats skip setup identically.
ngspice 47 rejects noise and pole-zero analyses under KLU, and ngspice `pss` searches autonomous oscillator frequencies rather than running driven shooting at a fixed period.
Xyce 7.10 lacks pole-zero analysis and solves periodic steady state only via harmonic balance, which this suite does not run.
Unsupported combinations are recorded as unsupported rather than failures.
Failed runs record the simulator diagnostic message and are omitted from work-precision plots.

## The first diagrams

Initial measurements ran via `scripts/precision` (now `scripts/bench --precision`) on commit `042f264` on 15 November 2026 from 16:34 to 16:41.
The host was a 12-core Apple M2 Max (8 performance, 4 efficiency, 32 GB RAM, macOS 27.2.1) under load averages of 5.9 to 11.5 from concurrent builds.
Software versions were Python 3.15.2, Rust 1.100.0, ngspice 47, and mpmath 1.4.1.
The initial run scored 2,448 configurations: 2,217 completed, 27 failed, and 204 were unsupported.
Running `scripts/bench --precision` regenerates the JSON results outside the repository, like other cross-simulator comparisons.
Accuracy digits are deterministic for a given build, so background load affects only timings, which require remeasurement on an idle machine; timing comparisons apply only at matched accuracy.

At default settings, simulators match within 0.1 digits on RC steps and RLC, while ngspice leads on five of the six RC networks by 0.1 to 0.7 digits:

| Fixture | jolty | ngspice, Sparse | ngspice, KLU |
| --- | --- | --- | --- |
| `rc_step` | 3.8 digits, 100 µs | 3.8, 120 µs | 3.8, 134 µs |
| `rlc_step` | 1.3, 197 µs | 1.2, 232 µs | 1.2, 276 µs |
| `lc_tank`, energy | 4.1, 236 µs | 3.8, 370 µs | 3.8, 389 µs |
| `lc_tank`, frequency | 1.3 | 1.3 | 1.3 |
| `rc_periodic` | 3.0, 200 µs | unsupported | unsupported |
| `ladder_100` | 2.5, 2.32 ms | 2.7, 1.32 ms | 2.7, 1.53 ms |
| `tree_1000` | 2.7, 19.5 ms | 2.9, 8.53 ms | 2.9, 11.7 ms |
| `diode` | 5.2, 113 µs | 3.6, 80 µs | 3.6, 92 µs |
| `rlc_ac` | 11.1, 140 µs | 14.2, 135 µs | 11.4, 153 µs |
| `rc_noise` | 15.2, 70 µs | 8.1, 87 µs | unsupported |
| `butterworth_poles` | 15.3, 31 µs | 14.7, 160 µs | unsupported |

Default settings yield few digits because the default maximum step (one-fiftieth of the window) caps RC steps at 3.8 digits across tolerances.
On the RLC and LC tank, both simulators' error controls accept about nine steps per period, missing the tank frequency by 5%.
Per-fixture default results go to `results/<commit>/defaults.md`, paired with `equal-error.md` listing each simulator's fastest runtime at equal digits (`--digits` or the highest accuracy reached by every simulator).
The `scripts/bench --view equal-error` command prints that equal-error speed view ([`../speed/README.md`](../speed/README.md)).

What the sweeps found:

- **Measured charge-tolerance floor** (`accuracy.md` item 3): jolty allows `reltol·|q| + chgtol` charge error per step whereas SPICE allows `reltol·max(|q|, chgtol)`, so sweeping `reltol` alone plateaus once `reltol·|q|` drops below `chgtol` (10 fC).
  On `rc_step` (1 nF), jolty takes 64 steps and plateaus at 3.8 digits from `reltol = 1e-2` to `1e-8`, whereas ngspice reaches 5.3 digits at `1e-8`.
  On `rc_femto` (1 fF), where `chgtol` alone allows 10 V at defaults, jolty stays at 3.8 digits and 64 steps even after scaling every tolerance down by 10⁵, whereas ngspice reaches 5.3 digits.
  On RC networks, jolty's `reltol`-only sweep stalls at 1.9 to 3.5 digits compared to 3.2 to 5.2 digits for ngspice.
- **Scaled tolerances on stiff networks:** with all tolerances scaled, jolty reaches 5.1 digits on `ladder_10` and `ladder_100` and 5.4 on `tree_100`, while ngspice stops at 3.4, 3.2, and 3.9 digits.
  ngspice's worst error occurs at the first 10 ps time point and does not improve across the sweeps, whereas jolty validates the initial post-corner step against the following step points.
  jolty trails on `tree_10` (4.9 versus 5.2 digits) and `ladder_1000` (4.0 versus 4.8 digits).
  On `ladder_1000`, jolty fails from scaled tolerance 5.6e-7 because fast modes request a step below the 8 ps minimum on a 0.4 s run, whereas ngspice fails from 3.2e-8.
- **Lossless tank damping at loose tolerances:** jolty's trapezoidal-ringing detector misclassifies capacitor current reversals at four samples per period as ringing and switches to backward Euler.
  At `reltol = 1e-2`, jolty retains only 2.7% of tank energy after 20 periods, whereas ngspice's plain trapezoidal rule preserves 3.6 digits.
  With all tolerances scaled to 1e-8, jolty preserves 5.5 digits of energy versus KLU's 3.9 digits, and both match at 4.9 digits of frequency.
- **DC:** jolty computes the diode voltage to 14.6 digits at tightest tolerances with a rounding-floored residual test, while ngspice stops at 8.0 digits matching the `reltol·|v| + vntol` (7.5 nV) Newton update bound.
- **Small-signal:** Sparse 1.3 resolves the RLC response to 14.2 digits, whereas jolty (11.1 digits) and ngspice KLU (11.4 digits) both lose three digits between 600 and 700 MHz where output is 5e-5 of input, warranting investigation.
- **Noise:** jolty reaches 15.2 digits while ngspice reaches 8.1 digits because ngspice uses the CODATA 2014 elementary charge (8.2e-9 below exact SI), leaving a residual shift in `kT` after thermal-voltage alignment.
- **Poles:** jolty's QZ solver finds Butterworth ladder poles to 15.3 digits compared to 14.7 digits for ngspice's Muller iteration with Sparse 1.3.
- **Cost at equal accuracy:** jolty runs faster on small circuits, taking 0.21 ms versus 0.32 ms on `rc_step` (5 digits) and 3.5 ms versus 4.8 ms on `rlc_step` (4 digits).
  On `lc_tank`, jolty reaches 4 frequency digits in 4.4 ms compared with 8.0 ms for ngspice.
  On RC networks, jolty takes roughly twice ngspice's time: 60 ms versus 30 ms for 3.5 digits on `ladder_1000`, and 5.1 ms versus 2.7 ms for 3.8 digits on `tree_100`.
- **Failures:** ngspice with Sparse 1.3 aborts 13 tank runs starting at `reltol = 3.2e-8` or `scaled = 1.8e-6` with "timestep too small" at 10 ps, whereas KLU completes them.

## Not yet

- Manufactured solutions using explicit behavioral diodes and transistors with smooth `x*(t)` and injected `b(t) = f(x*) + d q(x*)/dt` sources, pending behavioral charge `q(v)` support (`benchmarks.md` item 7, B6).
- IVP test-set circuits: ring modulator and transistor amplifier with behavioral junctions, NAND gate and charge pump once behavioral charge exists, and the two-bit adder with an independent high-precision reference (published reference holds 1e-5).
- Expanding to twenty or more fixtures and reporting which fixtures place each simulator on the Pareto accuracy-time front.
- Interleaved confidence-interval timings on an idle host (`benchmarks.md` item 3) plus solver iteration counts for non-transient jolty analyses.
