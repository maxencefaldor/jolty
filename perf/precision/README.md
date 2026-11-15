# Precision

The precision suite measures each simulator's error against exact answers, and what it costs: a work-precision diagram per fixture, the correct digits of a run against its engine time.
Every exact answer is computed independently of every simulator, as a closed form or as the circuit's own equations solved in higher precision; no simulator's run, however tight, is ever a reference.
The suite stays out of `scripts/check`, as the speed suite does: it times its runs.

## Running

```sh
scripts/precision                          # every fixture, in jolty and in ngspice
scripts/precision rc_step lc_tank          # only these fixtures
scripts/precision --simulators jolty       # one simulator configuration, or several
scripts/precision --sweeps reltol          # one sweep of the tolerances, or both
scripts/precision --repeats 9              # the runs each setting's time is the median of
scripts/precision --plot RESULTS           # the diagrams and table of a results file
```

The results go to `perf/results/precision/<commit>.json`, `-dirty` for a working tree with changes to tracked files, and a later run on the same commit adds its fixtures to the file, so that the suite can run in pieces.
The diagrams, one per fixture, and the table of each simulator at its defaults go to the directory of the same name beside it, which `--plot` draws again from the file.
The exact answers of the networks of 1,000 nodes take about a minute each on the machine below; the whole suite about ten.

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
mpmath computes the rates, the residues and each mode's state where each piece begins at 50 digits, and rounds them to doubles; the sum is evaluated in double precision at the points a simulator produced, beside a bound on its rounding, which caps the digits a score may claim, near 14 on these fixtures.
A small circuit's modes come from mpmath's eigensolver on its state equations.
A network's come from its symmetric pencil `C^{-1/2} G C^{-1/2}`, a tree whose elimination fills nothing: each eigenvalue by Newton's iteration on its determinant from LAPACK's estimate, checked by Sylvester's inertia to be the one its place stands for, its eigenvector by inverse iteration, and every node's residues checked to sum to its gain of one.
The diode's voltage is Lambert's W, the AC response, the noise and the periodic steady state are closed forms, and the poles are the eigenvalues of the ladder's state equations, all in mpmath.
`tests/test_precision.py` checks each against another way of finding it: the tree solver against mpmath's general one, the modes against the RC's closed form, the Butterworth ladder's poles against its circle, and the diode's voltage against the currents it balances.

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

| Configuration | How it runs | Its engine time |
| --- | --- | --- |
| `jolty` | `Simulation.run` on a fresh simulation of the circuit elaborated once, so that every repeat solves its operating point again | the wall time of `Simulation.run`: the engine's work and its results made in Python |
| `ngspice-sparse` | ngspice 47 on the netlist `jolty.spice.export` writes, saving the vectors the fixture reads, at the temperature where its thermal voltage is jolty's, as `jolty.spice.ngspice` runs it, with Sparse 1.3, its default | the growth of its own `Total analysis time` from `rusage`, which reads a wall clock: a run paused for 25 s reported 29.9 s where it took 4.9 unpaused |
| `ngspice-klu` | the same with `.options klu` | the same |

Each setting runs in each configuration one after another, so that each runs under the load the others do, and every run records the load average as it began.
ngspice's repeats run in one process, its first repeat with the setting up of its matrix, which the median passes over as jolty's repeats pass over nothing.
ngspice 47 refuses noise and pole-zero analyses with KLU, and its `pss`, which searches an oscillator's frequency and returns harmonics, has no shooting for a circuit driven at a given period: those runs are recorded as unsupported, never as failures.
A run that fails is recorded with what the simulator said, and drawn nowhere.
