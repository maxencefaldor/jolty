# Performance

The engine's speed is measured on a fixed set of circuits, the fixtures, so that every change can say what it does to it.
`scripts/bench` times them, in Rust and end to end in Python, writes the times by commit, and compares them with a baseline's.
The benchmarks stay out of `scripts/check`, which must stay fast and deterministic.

## Running

```sh
scripts/bench                      # the suite: a few minutes, before and after a change
scripts/bench bjt_amp buck         # only these fixtures
scripts/bench --scale              # the large fixtures, each run once
scripts/bench --ngspice            # ngspice on the fixtures' netlists, for context
scripts/bench --baseline COMMIT    # against COMMIT's times, rather than an ancestor's
```

`scripts/bench` first checks that each fixture's files are what its script writes, then times the fixtures, writes the times to `results/<commit>.json`, and prints them against a baseline's, flagging every change over 5%.
The baseline is the newest ancestor of the checked-out commit whose times `results/` holds, unless `--baseline` names a commit or a results file.
A working tree with changes to tracked files writes `results/<commit>-dirty.json`, which is never taken as a baseline.
A later run on the same commit adds its times to the file, so the suite, `--scale` and `--ngspice` can each run once.
To compare two commits, check out the older one, run the suite, then do the same on the newer one.

## Fixtures

Each fixture is a circuit, in the circuit format in `circuits/`, and a plan of analyses in the engine's format in `plans/`, both written by the script beside the circuit, which builds it with jolty's own builder.
Run a script from the repository root after changing it, such as `uv run python perf/circuits/bjt_amp.py`, and commit what it writes.

| Fixture | Circuit and plan | What it stresses | Timed by |
| --- | --- | --- | --- |
| `bjt_amp` | A differential pair of 2N3904s, a mirror of 2N3906s and a follower, fed back for a gain of 10: the operating point, 71 frequencies from 10 Hz to 100 MHz, and 200 µs of a 10 kHz sine | The cost of one small run | the suite |
| `ac_wide` | `bjt_amp` at 10,000 frequencies from 1 Hz to 1 GHz | Frequencies run in parallel | the suite |
| `buck` | A synchronous buck converter of level 1 MOSFETs at 500 kHz, from 12 V into 5 Ω: 2 ms | Many tiny time steps | the suite |
| `inverter_chain_500` | 500 CMOS inverters of level 1 MOSFETs under a pulse: 40 ns | Device evaluation and Newton iteration | the suite |
| `inverter_chain_5000`, `_50000` | The same chain of 5,000 and 50,000 inverters | The same, on one large circuit | `--scale` |
| `rc_mesh_100` | A square mesh of 10,000 nodes, 1 kΩ between neighbours and 1 pF to ground, under a pulse: 10 µs | The sparse solver's fill | the suite |
| `rc_mesh_300` | The same mesh of 90,000 nodes | The same, on one large circuit | `--scale` |
| `monte_carlo` | `bjt_amp`'s analyses on 100,000 samples of its resistors, each within 5% | Many runs of one circuit | the suite, its first 1,000 samples, in Rust as one batch and in Python; `--scale`, every one, in Python |

The MOSFETs of a level 1 model each add three unknowns of their own, the displacements of their Meyer capacitances, so the chains of 500, 5,000 and 50,000 inverters solve for 3,504, 35,004 and 350,004 unknowns.
The chains' plans let their results hold 4 GiB, as the 50,000 inverters' hold 2.1 at the transient's 800 points or so, more than the 2 the engine allows by default.

## What the times measure

| Time | What it measures |
| --- | --- |
| `rust/<fixture>/setup` | Criterion's median of a setup: the circuit read from its JSON text and elaborated, into a simulator ready to run. |
| `rust/<fixture>/shared` | Criterion's median of a setup from the circuit elaborated once and shared, as each of many runs of one circuit makes it: its devices made again, and a run's buffers. |
| `rust/<fixture>/run` | Criterion's median of a run: the setup, then every analysis of the plan in order. |
| `rust/monte_carlo/<n> samples, <t> threads` | Criterion's median of the fixture's first `n` samples run as one batch, per sample: drawn from their indices and run on the circuit elaborated once, on a pool of `t` threads, for 1, 2, 4 and 8 threads and every core, as far as there are cores. |
| `python/<fixture>` | The median of at least ten runs over at least five seconds, after one to warm up: the circuit, read once from its file, elaborated by `Simulation`, and every analysis run by `Simulation.run`, whose results are made as a script's are. An operating point and a transient also record the instances rated for a current or a power, which the checks read. |
| `python/monte_carlo/<n> samples, one core`, `..., every core` | The time per sample of the fixture's first `n` samples, run as the runner runs them, as one batch of `Simulation.batch` on the circuit elaborated once, each sample drawn by the engine from its index, the one elaboration counted: on a pool of one thread, and of a thread per core. The suite runs the first 1,000, and `--scale` all of them, on every core. |
| `ngspice/<fixture>` | ngspice 47 in batch mode on the fixture's netlist as jolty writes it, its analyses run from a control section: from its start to its exit, so its start-up and the parsing of its netlist count, for context. |

Criterion takes ten samples of each benchmark over five seconds, each sample one run or more, after a second of warming up.
The Python timings follow the same protocol.
Beside the times, the suite runs each fixture with a transient once more with the option `profile` on, and writes the transient's statistics' `work` and `profile`: its Newton iterations and factorizations, and where its time went.

## Baseline

The baseline is commit `798b73d`, whose times `results/798b73d4b40d2e7cdd04c7f9a838705d1b8ae471.json` holds.
It was measured on an Apple M2 Max of 12 cores, 8 for performance and 4 for efficiency, on one socket, with 32 GB of memory, under macOS 27.2, with Rust 1.100.0 and Python 3.15.2, while other programs ran at a load average of 4 to 6.

| Fixture | Rust setup | Rust run | Python | ngspice |
| --- | --- | --- | --- | --- |
| `bjt_amp` | 63.21 µs | 806.9 µs | 1.583 ms | 17.5 ms |
| `ac_wide` | 61.9 µs | 10.61 ms | 14.63 ms | 30.94 ms |
| `buck` | 31.78 µs | 1.169 s | 1.091 s | 218.7 ms |
| `inverter_chain_500` | 4.094 ms | 2.905 s | 2.901 s | 1.126 s |
| `rc_mesh_100` | 42.59 ms | 830.8 ms | 804.7 ms | 13.97 s |

The large fixtures, each run once:

| Fixture | Python | ngspice |
| --- | --- | --- |
| `inverter_chain_5000` | 30.6 s | 14.69 s |
| `inverter_chain_50000` | 337.3 s | 409 s |
| `rc_mesh_300` | 21.57 s | 857.4 s |

A sample of `monte_carlo` takes 932.2 µs on one core, 1,073 samples a second, and 401.5 µs on every core, 2,491 a second: 2.3 times as fast on 12 cores, a parallel efficiency of 19%.
All 100,000 samples, on every core, take 379 µs each, 37.9 s in all.
A sample costs less than `bjt_amp`'s own run: Newton iteration alone finds 196 of the first 200 samples' operating points, where the nominal circuit's needs gmin stepping, and their transients take 142 Newton iterations on average, against its 254.

The suite took 3 min 14 s, the benchmarks already built, and the `--scale` run 13 min 53 s.
Run six times over an hour on the same engine, three times at this commit and three at an earlier one that differs only in `perf/`, each time but two spread by at most 5.6% from its fastest run to its slowest: `rc_mesh_100`'s Rust run by 9.8%, and the Monte Carlo samples on every core, which compete with every other program for the cores, by 19%.
The baseline's own run was among the slower: the last run at this commit was faster in every time, by 0.6% to 4.4% but for those two.
On a machine shared with other work, a change of a few percent is within the noise; the comparison flags those over 5%.

The engine is built twice from the same code, into the benchmark and into the Python extension, and the two builds differ: `buck`'s run takes 6% to 9% longer in the benchmark's.
So each kind of time is compared only with its own.

### Where the time goes

Each transient, run once with its profile on, spent its time and did its work as follows; the rest is preparing and everything else.

| Transient | Time | Evaluating | Assembling | Factoring | Solving | Step control | Recording | The rest | Newton iterations | Factorizations, refactorizations, fallbacks, reuses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bjt_amp/tran` | 786.6 µs | 35% | 8% | 18% | 11% | 9% | 9% | 10% | 254 | 0, 257, 50, 0 |
| `buck/tran` | 1.233 s | 33% | 9% | 18% | 14% | 18% | 0% | 8% | 479,871 | 1, 621,777, 10,370, 286 |
| `inverter_chain_500/tran` | 2.9 s | 65% | 8% | 12% | 9% | 5% | 0% | 1% | 7,377 | 2, 8,093, 24, 6 |
| `rc_mesh_100/tran` | 794 ms | 18% | 3% | 61% | 10% | 8% | 0% | 1% | 170 | 1, 53, 0, 203 |
| `inverter_chain_5000/tran` | 30.28 s | 66% | 8% | 12% | 9% | 4% | 0% | 1% | 7,651 | 2, 8,371, 60, 15 |
| `inverter_chain_50000/tran` | 351.6 s | 69% | 8% | 11% | 8% | 3% | 0% | 1% | 7,812 | 2, 8,532, 60, 15 |
| `rc_mesh_300/tran` | 20.66 s | 13% | 1% | 78% | 5% | 3% | 0% | 0% | 169 | 1, 53, 0, 202 |

`bjt_amp`'s transient starts from the operating point the plan's first analysis solved, so it makes no full factorization; it records its transistors, rated for a current and a power, at every point.
A Newton iteration of `inverter_chain_500` takes 0.39 ms, two thirds of it in evaluating the devices; of `inverter_chain_50000`, 45 ms, a hundred and fifteen times as long for a hundred times the inverters.
The meshes' transients factor a new matrix only when the step changes, 54 times, and spend most of their time doing so: a factorization of the larger mesh takes 33 times as long as one of the smaller, for 9 times the nodes.

### Against the plan's starting point

The many-cores plan's figures came from a two-core machine, on circuits of their own, before the sparse solver became KLU's and before the step control changed; this baseline replaces them.
jolty's level 1 MOSFETs now carry their Meyer displacements as unknowns, so 500 inverters solve for 3,504 unknowns, where the plan's solved for 504.
ngspice keeps 52,982 points of `buck`'s 2 ms, where jolty accepts 122,587 steps and rejects 32,900, and takes a fifth of jolty's time.
It takes two fifths of jolty's time on the chain of 500 inverters and half on the chain of 5,000, but a fifth more on the chain of 50,000, and, with its default sparse solver, Sparse 1.3, 17 and 40 times jolty's on the meshes.
