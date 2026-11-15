# Performance

The engine's speed is measured on a fixed set of circuits, the fixtures, so that every change can say what it does to it.
`scripts/bench` times them, in Rust and end to end in Python, writes the times by commit, and compares them with a baseline's.
The benchmarks stay out of `scripts/check`, which must stay fast and deterministic.

## Running

```sh
scripts/bench                      # the suite: a few minutes, before and after a change
scripts/bench bjt_amp buck         # only these fixtures
scripts/bench --scale              # the large fixtures, each run once, and spread over threads
scripts/bench --scale --largest    # and the 50,000 inverters' chain spread over threads too
scripts/bench --ngspice            # ngspice on the fixtures' netlists, for context
scripts/bench --xyce               # Xyce on them, and how far it lies from jolty
scripts/bench --xyce --scale --xyce-mpi PATH  # and Xyce's parallel build on 1 to 16 ranks
scripts/bench --baseline COMMIT    # against COMMIT's times, rather than an ancestor's
scripts/bench --break-even min_unknowns_inside  # from which size a threshold pays
scripts/bench --robustness         # the corpus of hard circuits, in every simulator
scripts/bench --coverage           # the coverage matrix, each row's probe in each simulator
```

`--robustness` runs instead the corpus of hard circuits that [`robustness/`](robustness/README.md) holds, jolty's own and third-party decks fetched at a pinned commit by the manifest `external.json`, each in every simulator, and prints each simulator's outcomes and failure rates.
`--coverage` runs instead the probes of [`coverage/`](coverage/README.md), each row of analyses, device models, SPICE's syntax and outputs in jolty and ngspice, or the simulators `--simulator` names, and renders the matrix, [`coverage/matrix.md`](coverage/matrix.md).

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

The MOSFETs of a level 1 model each add three unknowns of their own, the displacements of their Meyer capacitances, so the chains of 500, 5,000 and 50,000 inverters solve for 3,504, 35,004 and 350,004 unknowns; the sparse solver eliminates the displacements before it factors, so it factors 504, 5,004 and 50,004 of them, and the buck's 17 of its 23.
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
| `xyce/<fixture>`, `xyce-mpi/<n> ranks/<fixture>` | Xyce at its defaults on the fixture's netlist in Xyce's dialect, each analysis in an Xyce of its own, printing two nets at each end of the circuit's top: from writing the deck to reading its values, summed over the analyses; serially, or by its parallel build on `n` MPI ranks. |
| `.../analysis` | The same runs' time in Xyce's analyses alone, its own "Total Simulation Solvers Run Time", summed over the analyses. |

Criterion takes ten samples of each benchmark over five seconds, each sample one run or more, after a second of warming up.
The Python timings follow the same protocol.
Beside Xyce's times, the results file's `agreements` hold how far each run's printed voltages lay from jolty's at the plan's options, each analysis's largest difference over the nets printed, as a fraction of jolty's largest magnitude or, in a transient, of its swing, jolty's waveform interpolated at Xyce's times: both simulators' errors at their defaults, so a fixture is timed as the same work only where it is small.
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

## One run spread over threads

A run does all its work on its own thread unless a threshold of `Parallelism` lets a kind of it spread, as [the guide](../docs/guide/simulation.md#options) says, and no fixture's plan sets one, so the suite times the serial paths alone.

### Break-evens

`scripts/bench --break-even NAME` measures from which size of circuit setting the threshold `NAME` makes one run faster than leaving it unset.
It sweeps circuits of the fixtures' scripts at several sizes: chains of inverters and meshes for the three thresholds a transient reads, and `ac_wide`'s response at fewer frequencies a decade for `min_solves_per_task`.
Each circuit runs its fixtures' plan with the threshold unset, on one thread, and set, on 2, 4 and 8 threads and every core, one after another, round after round, until there are three rounds at least and they have taken five seconds for each.
Each time is the median of its rounds, printed beside the load average as they began and as they ended.
A run with the threshold set must find the bits of the run left unset, or the sweep stops, but for a dissection: its interiors, of a size for each number of threads, order the factorization otherwise.
`--sizes` and `--threads` sweep other sizes and threads, and naming a script, such as `inverter_chain`, sweeps its circuits alone.

On each number of threads, the threshold is set at each size so that it spreads the circuit's work over every thread, however small the circuit:

| Threshold | Set to | So that |
| --- | --- | --- |
| `min_instances_per_task` | the circuit's devices over the threads, rounded up | its devices evaluate in a task a thread |
| `min_unknowns_inside` | 1 | its work on each unknown spreads, in chunks of 1,024 |
| `max_unknowns_per_interior` | its unknowns over the threads, rounded up | its matrix's block is dissected into an interior a thread or more |
| `min_solves_per_task` | its solves, its frequencies but the first, over the threads, rounded down | its frequencies are solved in a task a thread |

The break-even on a number of threads is the smallest size measured from which setting the threshold made every run faster.
Its value is the circuit's unknowns for `min_unknowns_inside`, which then spreads from that size on, and for each other threshold the value it was set to there: a task's devices or solves, or an interior's most unknowns.
Set to it, such a threshold spreads a circuit of that size over every thread in pieces as large as there, and a larger one in more pieces as large; it spreads a smaller circuit in fewer pieces than threads, which may lose, from two tasks' worth of solves, from one task's worth of devices, and from a block larger than an interior.

The sweeps ran on the machine of the baseline, with the engine of commit `6fe8824`, on 15 November 2026 from 11:58 to 13:12, each in pieces of at most ten minutes, their sizes given by `--sizes`, while two other agents built and ran the engine and ngspice on the same cores: the load average was 7 to 19 on the 12 cores, the measurement's own threads included.
Each change is a spread run's time against the unset run's, at the smallest size from which every larger one gained too, and at the size below it; the load is the load average as that size's rounds began and as they ended.

| Threshold | Circuits | Threads | Pays from | There | Below | Break-even | Load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `min_instances_per_task` | chains of inverters | 2 | 100 inverters, 202 devices | -3.8% | +2.0% at 70 | 101 | 10.7 to 9.5 |
| `min_instances_per_task` | chains of inverters | 4 | 200 inverters, 402 devices | -18.1% | +1.6% at 100 | 101 | 9.5 to 12.0 |
| `min_instances_per_task` | chains of inverters | 8 | 500 inverters, 1,002 devices | -1.9% | +23.1% at 300 | 126 | 12.0 to 12.2 |
| `min_instances_per_task` | chains of inverters | 12 | 1,000 inverters, 2,002 devices | -9.6% | +21.4% at 700 | 167 | 12.2 to 11.0 |
| `min_instances_per_task` | meshes | 2 | 100 a side, 29,901 devices | -8.6% | +0.5% at 70 | 14,951 | 14.8 to 12.1 |
| `min_instances_per_task` | meshes | 4 | 100 a side, 29,901 devices | -10.4% | +5.7% at 70 | 7,476 | 14.8 to 12.1 |
| `min_instances_per_task` | meshes | 8 | 100 a side, 29,901 devices | -6.3% | +18.9% at 70 | 3,738 | 14.8 to 12.1 |
| `min_instances_per_task` | meshes | 12 | 100 a side, 29,901 devices | -4.8% | +29.9% at 70 | 2,492 | 14.8 to 12.1 |
| `min_unknowns_inside` | chains of inverters | 2 | 5,000 inverters, 35,004 unknowns | -8.5% | +3.0% at 4,000 | 35,004 | 11.5 to 15.7 |
| `min_unknowns_inside` | chains of inverters | 4 | 3,000 inverters, 21,004 unknowns | -3.6% | +12.9% at 2,000 | 21,004 | 12.3 to 13.1 |
| `min_unknowns_inside` | chains of inverters | 8 | 5,000 inverters, 35,004 unknowns | -9.1% | +8.7% at 4,000 | 35,004 | 11.5 to 15.7 |
| `min_unknowns_inside` | chains of inverters | 12 | 5,000 inverters, 35,004 unknowns | -2.4% | +27.9% at 4,000 | 35,004 | 11.5 to 15.7 |
| `min_unknowns_inside` | meshes | 2 | none to 300 a side, 90,001 unknowns | | +5.8% at 300 | | 15.8 to 10.6 |
| `min_unknowns_inside` | meshes | 4 | none to 300 a side, 90,001 unknowns | | +0.1% at 300 | | 15.8 to 10.6 |
| `min_unknowns_inside` | meshes | 8 | none to 300 a side, 90,001 unknowns | | +3.2% at 300 | | 15.8 to 10.6 |
| `min_unknowns_inside` | meshes | 12 | none to 300 a side, 90,001 unknowns | | +4.4% at 300 | | 15.8 to 10.6 |
| `max_unknowns_per_interior` | chains of inverters | 2 | none to 5,000 inverters, 35,004 unknowns | | +3.8% at 5,000 | | 15.3 to 10.4 |
| `max_unknowns_per_interior` | chains of inverters | 4 | none to 5,000 inverters, 35,004 unknowns | | +6.2% at 5,000 | | 15.3 to 10.4 |
| `max_unknowns_per_interior` | chains of inverters | 8 | none to 5,000 inverters, 35,004 unknowns | | +9.5% at 5,000 | | 15.3 to 10.4 |
| `max_unknowns_per_interior` | chains of inverters | 12 | none to 5,000 inverters, 35,004 unknowns | | +45.4% at 5,000 | | 15.3 to 10.4 |
| `max_unknowns_per_interior` | meshes | 2 | 70 a side, 4,901 unknowns | -17.0% | +14.8% at 50 | 2,451 | 13.8 to 13.3 |
| `max_unknowns_per_interior` | meshes | 4 | 70 a side, 4,901 unknowns | -20.9% | +1.0% at 50 | 1,226 | 13.8 to 13.3 |
| `max_unknowns_per_interior` | meshes | 8 | 70 a side, 4,901 unknowns | -16.7% | +22.7% at 50 | 613 | 13.8 to 13.3 |
| `max_unknowns_per_interior` | meshes | 12 | 100 a side, 10,001 unknowns | -38.7% | +3.2% at 70 | 834 | 17.3 to 15.1 |
| `min_solves_per_task` | `bjt_amp`'s response | 2 | 40 a decade, 360 solves | -11.4% | +3.2% at 30 | 180 | 15.4 to 12.9 |
| `min_solves_per_task` | `bjt_amp`'s response | 4 | 30 a decade, 270 solves | -1.5% | +3.1% at 20 | 67 | 16.0 to 15.4 |
| `min_solves_per_task` | `bjt_amp`'s response | 8 | 40 a decade, 360 solves | -7.6% | +5.5% at 30 | 45 | 15.4 to 12.9 |
| `min_solves_per_task` | `bjt_amp`'s response | 12 | 40 a decade, 360 solves | -3.7% | +16.9% at 30 | 30 | 15.4 to 12.9 |

Interleaving puts both sides of a comparison under the same load, but the load changed from round to round, by a few percent of a run, so a break-even whose gain is a few percent, such as the meshes' tasks of instances, is uncertain by a size of the sweep: the meshes of 50 a side gained on 2 and 4 threads, where those of 70 lost on all four.
A chain spends 12% of its run factoring, by the profile above, and a dissection, which factors more slowly on one thread, spread that too little to pay; a mesh spends most of its run factoring, in few Newton iterations, so that spreading its work on each unknown cost more than it saved.
Each threshold was measured alone, the others unset, so a dissection of a chain whose devices spread too, which leaves the factorization a larger share of the run, may pay sooner.

### Scaling

`scripts/bench --scale` runs each large fixture once serially, its plan setting no threshold, and then once with every threshold set at its break-even, on 1, 2, 4 and 8 threads and every core.
The spread runs of `inverter_chain_50000` took 17 minutes here, so it runs them only with `--largest`; `--threads` names other numbers of threads.
It prints each run's time, its speed-up against the serial run and against the spread run on one thread, and its results' fingerprint, the first 16 hexadecimal digits of the SHA-256 of their arrays.
The spread runs must find the same bits on every number of threads, and those of the serial run unless they dissect the factorization, which orders it otherwise, or it stops.
The results file holds the serial time as the fixture's own, `python/<fixture>`, each spread time as `python/<fixture>/spread on <n> threads`, and the fingerprints.

The thresholds are the break-evens above and no others: each at the largest of its break-evens on 2, 4, 8 and 12 threads on the circuits of the fixture's script, so that it pays wherever it spreads on any of them, and left unset where none paid.
`min_solves_per_task` takes the amplifier's, though no large fixture solves at frequencies.

| Fixtures | `min_instances_per_task` | `min_unknowns_inside` | `max_unknowns_per_interior` | `min_solves_per_task` |
| --- | --- | --- | --- | --- |
| `inverter_chain_5000`, `_50000` | 167 | 35,004 | unset | 180 |
| `rc_mesh_300` | 14,951 | unset | 2,451 | 180 |

Measured by `scripts/bench --scale inverter_chain_5000 rc_mesh_300` on 15 November 2026 at 13:11, on the same machine and engine as the break-evens and under the same load, and `inverter_chain_50000` from 13:16 to 13:39 by the same runs in four processes, the serial run, the spread one on one thread, those on 2 and 4 and those on 8 and 12, so that none took more than ten minutes:

| Fixture | Threads | Time | Against serial | Against 1 thread | Load |
| --- | --- | --- | --- | --- | --- |
| `inverter_chain_5000` | serially | 29.22 s |  |  | 14.8 |
| `inverter_chain_5000` | 1 | 28.72 s | 1.02 | 1.00 | 13.4 |
| `inverter_chain_5000` | 2 | 23.2 s | 1.26 | 1.24 | 11.4 |
| `inverter_chain_5000` | 4 | 16.5 s | 1.77 | 1.74 | 12.1 |
| `inverter_chain_5000` | 8 | 15.9 s | 1.84 | 1.81 | 10.9 |
| `inverter_chain_5000` | 12 | 21.77 s | 1.34 | 1.32 | 10.1 |
| `rc_mesh_300` | serially | 19.66 s |  |  | 9.8 |
| `rc_mesh_300` | 1 | 9.716 s | 2.02 | 1.00 | 9.2 |
| `rc_mesh_300` | 2 | 5.622 s | 3.50 | 1.73 | 11.6 |
| `rc_mesh_300` | 4 | 4.27 s | 4.60 | 2.28 | 11.3 |
| `rc_mesh_300` | 8 | 4.535 s | 4.33 | 2.14 | 11.6 |
| `rc_mesh_300` | 12 | 6.293 s | 3.12 | 1.54 | 12.4 |
| `inverter_chain_50000` | serially | 300.85 s | | | 9.4 |
| `inverter_chain_50000` | 1 | 300.75 s | 1.00 | 1.00 | 4.3 |
| `inverter_chain_50000` | 2 | 236.64 s | 1.27 | 1.27 | 6.3 |
| `inverter_chain_50000` | 4 | 178.77 s | 1.68 | 1.68 | 9.1 |
| `inverter_chain_50000` | 8 | 152.51 s | 1.97 | 1.97 | 11.9 |
| `inverter_chain_50000` | 12 | 169.58 s | 1.77 | 1.77 | 11.3 |

The chains' results were `6bd38191db1d7442` and `ebcc554ba5c4164b` serially and spread on every number of threads, and the mesh's `eb34cf8cbfa7c065` serially and `2a41b88522b56dbe` dissected on every number.
On one thread a chain spreads nothing, every threshold needing two threads, while the mesh's dissection alone halves its run: its dissected factorization is faster than the minimum degree's on one thread too.
A chain's factorization and solves stay on one thread, no dissection having paid on a chain alone, and the chain of 50,000 inverters ran at most 1.97 times as fast, on 8 threads.
On 12 threads both ran slower than on 8, their threads sharing the cores with the other programs, which kept the load average at 10 to 15, so a quiet machine's times are the ones to judge the scaling by.
