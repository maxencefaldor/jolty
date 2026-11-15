# Precision

The precision suite measures each simulator's error against exact answers, and what it costs: a work-precision diagram per fixture, the correct digits of a run against its engine time.
Every exact answer is computed independently of every simulator, as a closed form or as the circuit's own equations solved in higher precision; no simulator's run, however tight, is ever a reference.
The suite stays out of `scripts/check`, as the speed suite does: it times its runs.

## Running

```sh
scripts/bench --precision                       # every fixture, in jolty and in ngspice
scripts/bench --precision rc_step lc_tank       # only these fixtures
scripts/bench --precision --simulator jolty     # one simulator, or several
scripts/bench --precision --sweeps reltol       # one sweep of the tolerances, or both
scripts/bench --precision --repeats 9           # the runs each setting's time is the median of
scripts/bench --precision --plot RESULTS        # the diagrams and table of a results file
```

The results go to `results/<commit>.json`, `-dirty` for a working tree with changes to tracked files, and a later run on the same commit adds its fixtures to the file, so that the suite can run in pieces.
The diagrams, one per fixture, and the table of each simulator at its defaults go to the directory of the same name beside it, which `--plot` draws again from the file.
The exact answers of the networks of 1,000 nodes take about a minute each on the machine below, and the whole suite took 6 min 18 s there, in three runs of its fixtures.

## Fixtures

| Fixture | Circuit and analysis | Exact answer | Scored |
| --- | --- | --- | --- |
| `rc_step` | 1 kΩ into 1 nF under a 1 V step rising in 1 ns, for 5 µs | its mode's response to each straight piece of the source | `v(out)` at every point, mixed |
| `rc_ramp` | the same under a ramp of 1 V over 2 µs, for 6 µs | the same | the same |
| `rc_femto` | 1 MΩ into 1 fF, at most 1 fC, under a step rising in 1 ps, for 5 ns | the same | the same |
| `rlc_step` | a series RLC of 10 Ω, 1 µH and 1 nF, damped at 0.16 of critical, under the step, for 2 µs | its two complex modes, by mpmath's eigenvectors | the same |
| `lc_tank` | an undamped tank of 1 µH and 1 nF set ringing by a current source of 1 mA turned off in 1 ns, for 4 µs, 20 periods | its two imaginary modes | its energy at every point, relative, and its frequency over the run, from the phase its state turned through |
| `diode` | a diode of 10 fA of saturation current fed through 1 kΩ, swept from 0.1 V to 5 V in 50 steps | Lambert's W, with the 1 pS of gmin across the junction | `v(anode)` at every level, relative |
| `rlc_ac` | the small-signal response of a series RLC low-pass of quality factor 31.6, at 20 frequencies a decade from 1 Hz to 1 GHz | `1/(1 - ω²LC + jωRC)` | `v(out)` at every frequency, relative |
| `rc_noise` | the thermal noise of 1 kΩ at the 1 nF it feeds, at 10 frequencies a decade from 1 Hz to 1 GHz | `4kTR/(1 + (ωRC)²)`, with the SI's exact `k` | the density at every frequency, relative |
| `butterworth_poles` | the poles of a fifth-order Butterworth LC ladder between 50 Ω, cut off at 1 MHz | the eigenvalues of its state equations | every pole, relative |
| `rc_periodic` | the periodic steady state of 1 kΩ into 10 µF driven at 1 kHz, its time constant ten periods | the phasor's steady state | `v(out)` over the period, mixed |
| `ladder_10`, `_100`, `_1000` | ladders of so many nodes, their resistances log-uniform from 10 Ω to 10 kΩ and their capacitances from 1 pF to 1 nF, under the step, for five of their slowest time constants | the eigenpairs of their symmetric pencils | 50 of their nodes at every point, mixed |
| `tree_10`, `_100`, `_1000` | random trees of so many nodes, each node's parent drawn among the nodes before it, their values as the ladders' | the same | the same |

Every circuit is built with jolty's builder, and its element values are the doubles the circuit holds, which the exact answers read as they are, so that they answer the circuit both simulators read.
The networks' values come from NumPy's generator seeded with 1,000 plus a ladder's size and 2,000 plus a tree's, so the same fixtures come back on every machine; their modes' rates spread over 4.5 decades for `tree_10` to 10.1 for `ladder_1000`.
Their transients cover five of their slowest time constants, rounded up to a digit: from 30 µs for `tree_10` to 0.4 s for `ladder_1000`.

### How the exact answers are found

A linear circuit under a piecewise-linear source moves each of its modes by `żₖ = λₖ zₖ + u`, and along a straight piece of the source, `u = uⱼ + sⱼ τ`, each mode is exactly `zₖ(0) e^{λₖτ} + uⱼ τ φ₁(λₖτ) + sⱼ τ² φ₂(λₖτ)`, with `φ₁(x) = (eˣ - 1)/x` and `φ₂(x) = (eˣ - 1 - x)/x²`.
mpmath computes the rates, the residues and each mode's state where each piece begins at 50 digits, and rounds them to doubles; the sum is evaluated in double precision at the points a simulator produced, beside a bound on its rounding, which caps the digits a score may claim: from 12.4 on `ladder_1000` to 15.7 on the closed forms, far above any run's.
A small circuit's modes come from mpmath's eigensolver on its state equations.
A network's come from its symmetric pencil `C^{-1/2} G C^{-1/2}`, a tree whose elimination fills nothing: each eigenvalue by Newton's iteration on its determinant from LAPACK's estimate, checked by Sylvester's inertia to be the one its place stands for, its eigenvector by inverse iteration, and every node's residues checked to sum to its gain of one.
The diode's voltage is Lambert's W, the AC response, the noise and the periodic steady state are closed forms, and the poles are the eigenvalues of the ladder's state equations, all in mpmath.
`tests/test_precision.py` checks each against another way of finding it: the tree solver against mpmath's general one, the modes against the RC's closed form, the Butterworth ladder's poles against its circle, and the diode's voltage against the currents it balances.
It also runs `rc_step` in jolty and in ngspice, when it is installed, and scores what each found, so that the suite's own reading of both is tested where the other tests run.

## Scoring

At each point a run produced, a time, a level or a frequency, the exact answer there scores it: the significant correct digits of the worst point, `-log₁₀ max |found - exact| / |exact|`, as the IVP test set's `scd`.
A quantity that passes through or near zero is scored in a mixed form, its scale added to `|exact|`: 1 V for the steps, the steady state's amplitude for `rc_periodic`.
A score claims no more digits than the exact answer's bound allows, nor than a double holds.
Beside each score, the run's engine time, the median of its repeats, and its counts: the time points it accepted and rejected and its Newton iterations, for transients; for the other analyses, jolty counts none yet.

## Sweeps

Each fixture runs at each simulator's defaults, ringed on its diagram, and along two sweeps:

- **`reltol`:** the relative tolerance alone, from 1e-2 to 1e-8 in quarter decades, as the IVP test set's diagrams sweep theirs, the other tolerances at their defaults;
- **`scaled`:** every tolerance scaled from its default by the same factor, the relative from 1e-2 to 1e-8: `reltol`, `vntol`, `abstol` and `chgtol` from SPICE's 1e-3, 1e-6 V, 1e-12 A and 1e-14 C.

`trtol` stays at its 7, and a transient's longest step at its default, a fiftieth of the run, in both simulators.

## Simulators

| Simulator | How it runs | Its engine time |
| --- | --- | --- |
| `jolty` | `Simulation.run` on a fresh simulation of the circuit elaborated once, so that every repeat solves its operating point again, after one run to warm up, as the speed suite's timing does | the wall time of `Simulation.run`: the engine's work and its results made in Python |
| `ngspice-sparse` | ngspice 47 on the netlist `jolty.spice.export` writes, saving the vectors the fixture reads, at the temperature where its thermal voltage is jolty's, as `jolty.spice.ngspice` runs it, with Sparse 1.3, its default | the growth of its own `Total analysis time` from `rusage`, which reads a wall clock: a run paused for 25 s reported 29.9 s where it took 4.9 unpaused |
| `ngspice-klu` | the same with `.options klu` | the same |

Each setting runs in each simulator one after another, so that each runs under the load the others do, and every run records the load average as it began.
ngspice's repeats run in one process, its first repeat with the setting up of its matrix, which the median passes over as jolty's repeats pass over nothing.
ngspice 47 refuses noise and pole-zero analyses with KLU, and its `pss`, which searches an oscillator's frequency and returns harmonics, has no shooting for a circuit driven at a given period: those runs are recorded as unsupported, never as failures.
A run that fails is recorded with what the simulator said, and drawn nowhere.

## The first diagrams

Measured by `scripts/precision`, now `scripts/bench --precision`, on commit `042f264`, on 15 November 2026 from 16:34 to 16:41, on an Apple M2 Max of 12 cores, 8 for performance and 4 for efficiency, with 32 GB of memory, under macOS 27.2.1, with Python 3.15.2, Rust 1.100.0, ngspice 47 and mpmath 1.4.1, while other agents' builds kept the load average at 5.9 to 11.5.
The first run, at `042f264` on an Apple M2 Max at a load of 5.9 to 11.5, scored 2,448 runs, of which 2,217 ran, 27 failed and 204 were unsupported; `scripts/bench --precision` writes its results again, which stay out of the repository, as comparisons with other simulators do.
The digits are the same on every run of the same build, so the load moves only the times, which a quiet machine is to measure again; times are compared only between runs that reached the same digits.

At their defaults, the simulators reach the same digits on the RC steps and the RLC, to a tenth, and ngspice leads on five of the six networks by 0.1 to 0.7 of a digit:

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

Few digits at defaults: on the RC steps the longest step, a fiftieth of the run, holds both simulators at 3.8 digits whatever their tolerances ask, and on the RLC and the tank both error controls accept about nine steps a period, which misses the tank's frequency by 5%.
`results/<commit>/defaults.md` holds the table of every fixture.

What the sweeps found:

- **The charge tolerance's floor, measured** (`accuracy.md` item 3): jolty allows `reltol·|q| + chgtol` of charge error a step where SPICE allows `reltol·max(|q|, chgtol)`, so with the relative tolerance alone jolty's error stops falling once `reltol·|q|` passes under `chgtol`'s 10 fC.
  On `rc_step`, 1 nF, it takes 64 steps and reaches 3.8 digits at every relative tolerance from 1e-2 to 1e-8, where ngspice reaches 5.3 at 1e-8; on `rc_femto`, 1 fF, where `chgtol` alone allows 10 V at defaults, it stays at 3.8 digits and 64 steps even with every tolerance scaled down 10⁵ times, where ngspice reaches 5.3.
  On the networks, its sweep of the relative tolerance alone stops at 1.9 to 3.5 digits, ngspice's at 3.2 to 5.2.
- **With every tolerance scaled, jolty goes further on stiff networks:** 5.1 digits on `ladder_10` and on `ladder_100`, 5.4 on `tree_100`, where ngspice stops at 3.4, 3.2 and 3.9; ngspice's worst point there is its first time point, at 10 ps, whose error no tolerance of the sweeps changes; jolty judges its first step after a corner by the second's points, which the guide names as its own.
  It loses on `tree_10`, 4.9 digits to ngspice's 5.2, and on `ladder_1000`, 4.0 to 4.8, where its runs fail from a scaled tolerance of 5.6e-7 on: at the start, the step its fastest modes' error asks falls below its shortest, 8 ps on that run of 0.4 s; ngspice's fail from 3.2e-8 on.
- **jolty damps a lossless tank at loose tolerances:** its test of the trapezoidal rule's ringing reads the capacitor's current, which a tank sampled four times a period turns back and forth, as ringing, and takes backward Euler steps, so at a relative tolerance of 1e-2 the tank keeps 2.7% of its energy after 20 periods, where ngspice's plain trapezoidal rule keeps it to 3.6 digits.
  With every tolerance scaled to 1e-8, jolty keeps 5.5 digits of the energy to KLU's 3.9, and 4.9 of the frequency to KLU's 4.9.
- **DC:** jolty finds the diode's voltage to 14.6 digits at the tightest tolerances, its residual test floored at the rounding; ngspice to 8.0, about the update its Newton iteration stops on, `reltol·|v| + vntol`, 7.5 nV there.
- **Small-signal:** Sparse 1.3 finds the RLC's response to 14.2 digits, jolty and ngspice's KLU to 11.1 and 11.4, both losing three digits between 600 and 700 MHz, where the output is 5e-5 of the input: a loss to look into.
- **Noise:** jolty's density is right to 15.2 digits, ngspice's to 8.1, as its elementary charge is CODATA 2014's, 8.2e-9 below the SI's exact one, which the alignment of its thermal voltage leaves in `kT`.
- **Poles:** jolty's QZ finds the Butterworth ladder's poles to 15.3 digits, ngspice's Muller iteration with Sparse to 14.7.
- **Cost at equal accuracy:** on the RC step, the RLC and the tank, jolty reaches the same digits in less time: 5 digits of `rc_step` in 0.21 ms to ngspice's 0.32, 4 of `rlc_step` in 3.5 ms to 4.8, 4 of the tank's frequency in 4.4 ms to 8.0.
  On the networks it takes about twice ngspice's time for the same digits: 3.5 digits of `ladder_1000` in 60 ms to ngspice's 30, 3.8 of `tree_100` in 5.1 ms to 2.7.
- **Failures:** ngspice with Sparse aborts the tank's 13 runs from a relative tolerance of 3.2e-8 or a scaled one of 1.8e-6 on, "timestep too small" at 10 ps, which KLU completes.

## Not yet

- Manufactured solutions: behavioral diodes and square-law and exponential transistors written as explicit expressions, a smooth `x*(t)` and its injected `b(t) = f(x*) + d q(x*)/dt` as behavioral sources of time; a charge needs a behavioral `q(v)` (`benchmarks.md` item 7, B6).
- The IVP test set's circuits: the ring modulator and the transistor amplifier with behavioral junctions; the NAND gate and the charge pump once a behavioral charge exists; the two bit adding unit with a reference recomputed by an independent integrator, its published one holding 1e-5.
- Twenty fixtures or more, and per simulator the fixtures where it lies on the front of best time for each accuracy.
- Each run's time with its interval, in alternating rounds, on a quiet machine (`benchmarks.md` item 3), and the counts of the analyses other than transients, which jolty does not report yet.
