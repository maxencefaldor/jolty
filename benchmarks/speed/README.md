# Speed

The engine's speed is measured on a fixed set of circuits, the fixtures, so that every change can say what it does to it.
`scripts/bench` times them, the speed suite of [the benchmarks](../README.md), in Rust and end to end in Python, writes the times by commit, and compares them with a baseline's.
The benchmarks stay out of `scripts/check`, which must stay fast and deterministic.

## Running

```sh
scripts/bench                      # the suite: a few minutes, before and after a change
scripts/bench bjt_amp buck         # only these fixtures
scripts/bench --scale              # the large fixtures, each run once, and spread over threads
scripts/bench --scale --largest    # and the 50,000 inverters' chain on each number of threads too
scripts/bench --view matched       # every simulator on the fixtures, at matched options
scripts/bench --view defaults      # every simulator on them, each at its own defaults
scripts/bench --view equal-error   # every simulator's least time to equal error, on the precision suite's fixtures
scripts/bench --view defaults --scale --xyce-mpi PATH  # the large fixtures, Xyce's parallel build on 1 to 16 ranks too
scripts/bench --baseline COMMIT    # against COMMIT's times, rather than an ancestor's
scripts/bench --break-even min_unknowns_per_task  # from which size a threshold pays
```

`scripts/bench` first checks that each fixture's files are what its script writes, then times the fixtures, writes the times to `results/<commit>.json`, and prints them against a baseline's, flagging every change over 5%.
The baseline is the newest ancestor of the checked-out commit whose times `results/` holds, unless `--baseline` names a commit or a results file.
A working tree with changes to tracked files writes `results/<commit>-dirty.json`, which is never taken as a baseline.
A later run on the same commit adds its times to the file, so the suite, `--scale` and each view can each run once.
To compare two commits, check out the older one, run the suite, then do the same on the newer one.

## Fixtures

Each fixture is a circuit, in the circuit format in `circuits/`, and a plan of analyses in the engine's format in `plans/`, both written by the script beside the circuit, which builds it with jolty's own builder.
Run a script from the repository root after changing it, such as `uv run python benchmarks/speed/circuits/bjt_amp.py`, and commit what it writes.

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
| `<view>/jolty/<fixture>` | In the view `matched` or `defaults`, the median of at least ten runs over at least five seconds, after one to warm up, of the fixture's circuit elaborated by `Simulation` and every analysis run by `Simulation.run`, at the view's options. |
| `<view>/ngspice-sparse/<fixture>`, `<view>/ngspice-klu/<fixture>` | The same of ngspice 47 in batch mode on the fixture's netlist as jolty writes it, with Sparse 1.3 or with KLU, its analyses run in turn in one ngspice, saving two nets at each end of the circuit's top: from its start to its exit, so its start-up and the parsing of its netlist count. |
| `<view>/xyce-serial/<fixture>`, `<view>/xyce-mpi/<n> ranks/<fixture>` | The same of Xyce on the fixture's netlist in Xyce's dialect, each analysis in an Xyce of its own, printing the same nets: from its start to its exit, summed over the analyses; serially, or by its parallel build on `n` MPI ranks. |
| `<view>/.../analysis` | The same runs' time in their analyses alone: jolty's `Simulation.run`, the engine's work and its results made in Python, its elaboration left out; ngspice's "Total analysis time" of `rusage`; Xyce's "Total Simulation Solvers Run Time", summed over the analyses. |

Criterion takes ten samples of each benchmark over five seconds, each sample one run or more, after a second of warming up.
The Python timings follow the same protocol, and so do the views'.
Beside the times, the suite runs each fixture with a transient once more with the option `profile` on, and writes the transient's statistics' `work` and `profile`: its Newton iterations and factorizations, and where its time went.

## Every simulator, in three views

`scripts/bench --view` times every simulator instead, jolty, ngspice 47 with Sparse 1.3 and with KLU, and Xyce 7.10 serially, each that is installed or those `--simulator` names, in the three views the owner set, and prints each view's table, a row per fixture and a column per simulator, which goes to `results/<commit>/<view>.md` beside the results file too:

| View | What each simulator runs at | Where its table comes from |
| --- | --- | --- |
| `matched` | Options as tight as jolty's tightest: jolty at `jolty.plan.TIGHT`, ngspice at `jolty.spice.ngspice.OPTIONS`, Xyce at `jolty.spice.xyce.OPTIONS`, which the table's first row states | The fixtures here, every simulator's runs |
| `defaults` | Its own defaults: no options line, but ngspice's `.options klu` with KLU | The same |
| `equal-error` | Each setting of the precision suite's sweeps, its least time to the same correct digits of an exact answer | [The precision suite](../precision/README.md)'s fixtures, run as `--precision` runs them, or the results file `--plot` names |

Every simulator runs each fixture at the temperature where its thermal voltage is jolty's, as `jolty.spice` runs them, so the simulators differ by their devices rather than their physical constants.
Each cell gives the median time end to end, then in its analyses alone, and how far each analysis lay from jolty's run at the matched options, the reference: the largest difference over the nets read, two at each end of the circuit's top, as a fraction of the reference's largest magnitude or, in a transient, of its swing, the reference interpolated at the simulator's points.
That says whether two simulators found the same solution, the condition for comparing their times; it is no error, as no simulator's run is an exact answer.
Where jolty fails at the matched options, there is no reference, and each cell says so.
A simulator whose run fails, or runs past `--timeout`, 300 s unless given, is marked with what stopped it: the first line of its message, the lines of what the simulator printed that say what went wrong, and how far an Xyce transient came; its row is never dropped.
The results file holds every cell under `views`, by view, simulator and fixture, and the times among the others, so a view's times are compared with a baseline's as every time is.
The equal-error view's target is `--digits`, or, on each output, the most digits that every simulator that runs the fixture reached; each simulator's time is its precision suite's engine time, of its fastest run that reached the target, beside that run's setting, and a simulator that never reached it is marked with the most it did.

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

A run spreads each kind of its work over the engine's threads in tasks of at least its threshold of `Parallelism`, once the work fills such a task for every thread, as [the guide](../../docs/guide/simulation.md#options) says, and no fixture's plan sets one, so the suite times each fixture with every threshold at its default, its break-even below.

### Break-evens

`scripts/bench --break-even NAME` measures from which size of circuit setting the threshold `NAME` makes one run faster than running it serially.
It sweeps circuits of the fixtures' scripts at several sizes: chains of inverters and meshes for the three thresholds a transient reads, and `ac_wide`'s response at fewer frequencies a decade for `min_solves_per_task`.
Each circuit runs its fixtures' plan serially, on one thread with every threshold set to spread none of its work, a task of all of its solves, devices or unknowns and interiors as large as the circuit, and with the threshold set and the others still spreading none, on 2, 4 and 8 threads and every core, one after another, round after round, each round in the reverse order of the one before, until there are three rounds at least and they have taken five seconds for each.
Alternating the order keeps any run from always following the same one: on `cedar`, the median of a serial run of the mesh of 50 a side was 261 ms when it always followed one on 64 threads and 225 ms when it followed one on 2, though none dissected the mesh.
Each time is the median of its rounds, printed beside the load average as they began and as they ended.
A run with the threshold set must find the bits of the serial run, or the sweep stops, but for a dissection: its interiors, of a size for each number of threads, order the factorization otherwise.
`--sizes` and `--threads` sweep other sizes and threads, and naming a script, such as `inverter_chain`, sweeps its circuits alone.

On each number of threads, the threshold is set at each size so that it spreads the circuit's work over every thread, however small the circuit:

| Threshold | Set to | So that |
| --- | --- | --- |
| `min_instances_per_task` | the circuit's devices over the threads, rounded down | each group of its devices evaluates in a task a thread, but a linear group, whose devices are evaluated once a run, in one |
| `min_unknowns_per_task` | its unknowns over the threads, rounded down | its work on each unknown spreads in a task a thread, in whole chunks of 1,024 |
| `max_unknowns_per_interior` | its unknowns over the threads, rounded up | its matrix's block is dissected into an interior a thread or more, where that costs fewer multiply-adds than the minimum degree |
| `min_solves_per_task` | its solves, its frequencies but the first, over the threads, rounded down | its frequencies are solved in a task a thread |

The break-even on a number of threads is the smallest size measured from which setting the threshold made every run faster.
Its value is the value the threshold was set to there: a task's solves, devices or unknowns, or an interior's most unknowns.
Set to it, a threshold of tasks spreads a circuit of that size over every thread in tasks as large as there, and a larger one in tasks as large or larger, but spreads none of a smaller one's work, which would make fewer tasks than threads, an arrangement the sweep does not measure; an interior's most unknowns dissects a smaller circuit's block into fewer interiors than threads, where that still costs fewer multiply-adds.

The sweeps ran on `cedar`, an AMD Ryzen Threadripper 3970X of 32 cores and 64 threads, with the engine of the change that set the defaults below, on 16 November 2026 from 05:58 to 07:02 UTC, while another user's program kept one core busy: the load average, the measurement's own threads included, was 1.4 to 37 on the 64 threads.
A chain's block is never dissected, its nested dissection costing three to four times the minimum degree's multiply-adds, so `max_unknowns_per_interior` swept chains of 10, 100 and 500 inverters alone; and `min_unknowns_per_task` swept chains of 100, 500, 1,000 and 2,000, its chain of 5,000 stopped for time, a chain's transient having doubled its steps since a step's charges are held to SPICE's charge tolerance.
Each change is a spread run's time against the serial run's, at the smallest size from which every larger one gained too, and at the size below it; the load is the load average as that size's rounds began and as they ended.

| Threshold | Circuits | Threads | Pays from | There | Below | Break-even | Load |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `min_solves_per_task` | `bjt_amp`'s response | 2 | 1,111 a decade, 9,999 solves | -3.3% | +0.9% at 500 | 4,999 | 3.42 to 3.44 |
| `min_solves_per_task` | `bjt_amp`'s response | 4 | 5 a decade, 45 solves | -0.4% | +1.8% at 2 | 11 | 6.81 to 5.17 |
| `min_solves_per_task` | `bjt_amp`'s response | 8 | 10 a decade, 90 solves | -4.4% | +2.0% at 5 | 11 | 5.17 to 7.78 |
| `min_solves_per_task` | `bjt_amp`'s response | 64 | 30 a decade, 270 solves | -7.4% | +1.8% at 20 | 4 | 5.81 to 4.64 |
| `min_instances_per_task` | chains of inverters | 2 | none to 1,000 inverters, 2,002 devices | | +14.2% at 1,000 | | 10.50 to 18.96 |
| `min_instances_per_task` | chains of inverters | 4 | 500 inverters, 1,002 devices | -2.9% | +4.2% at 300 | 250 | 15.28 to 13.94 |
| `min_instances_per_task` | chains of inverters | 8 | 500 inverters, 1,002 devices | -10.3% | +0.4% at 300 | 125 | 15.28 to 13.94 |
| `min_instances_per_task` | chains of inverters | 64 | none to 1,000 inverters, 2,002 devices | | +20.7% at 1,000 | | 10.50 to 18.96 |
| `max_unknowns_per_interior` | meshes | 2 | 100 a side, 10,001 unknowns | -21.3% | +0.5% at 70 | 5,001 | 14.67 to 15.36 |
| `max_unknowns_per_interior` | meshes | 4 | 100 a side, 10,001 unknowns | -29.3% | +0.7% at 70 | 2,501 | 14.67 to 15.36 |
| `max_unknowns_per_interior` | meshes | 8 | 100 a side, 10,001 unknowns | -41.2% | +2.7% at 70 | 1,251 | 14.67 to 15.36 |
| `max_unknowns_per_interior` | meshes | 64 | 100 a side, 10,001 unknowns | -20.5% | +46.3% at 70 | 157 | 14.67 to 15.36 |
| `max_unknowns_per_interior` | chains of inverters | 2 | none to 500 inverters, 3,504 unknowns | | +0.2% at 500 | | 3.62 to 1.39 |
| `max_unknowns_per_interior` | chains of inverters | 4 | none to 500 inverters, 3,504 unknowns | | +0.1% at 500 | | 3.62 to 1.39 |
| `max_unknowns_per_interior` | chains of inverters | 8 | none to 500 inverters, 3,504 unknowns | | +0.0% at 500 | | 3.62 to 1.39 |
| `max_unknowns_per_interior` | chains of inverters | 64 | 500 inverters, 3,504 unknowns | -0.0% | +0.3% at 100 | 55 | 3.62 to 1.39 |
| `min_unknowns_per_task` | chains of inverters | 2 | none to 2,000 inverters, 14,004 unknowns | | +8.0% at 2,000 | | 19.37 to 11.53 |
| `min_unknowns_per_task` | chains of inverters | 4 | none to 2,000 inverters, 14,004 unknowns | | +9.9% at 2,000 | | 19.37 to 11.53 |
| `min_unknowns_per_task` | chains of inverters | 8 | none to 2,000 inverters, 14,004 unknowns | | +8.5% at 2,000 | | 19.37 to 11.53 |
| `min_unknowns_per_task` | chains of inverters | 64 | none to 2,000 inverters, 14,004 unknowns | | +39.9% at 2,000 | | 19.37 to 11.53 |
| `min_unknowns_per_task` | meshes | 2 | 300 a side, 90,001 unknowns | -1.1% | +4.8% at 100 | 45,000 | 5.01 to 2.03 |
| `min_unknowns_per_task` | meshes | 4 | 300 a side, 90,001 unknowns | -1.2% | +7.4% at 100 | 22,500 | 5.01 to 2.03 |
| `min_unknowns_per_task` | meshes | 8 | 300 a side, 90,001 unknowns | -1.2% | +3.6% at 100 | 11,250 | 5.01 to 2.03 |
| `min_unknowns_per_task` | meshes | 64 | 300 a side, 90,001 unknowns | -0.7% | +19.2% at 100 | 1,406 | 5.01 to 2.03 |

A threshold left unset takes its break-even measured on the most threads a run may use or fewer, and spreads none of its work where no size of a script paid, as `jolty.plan.BREAK_EVENS` holds them; `max_unknowns_per_interior`, which orders the factorization and so its results' last digits, takes its break-even on the threads likewise, and none on one: `min_solves_per_task` 4,999, 11, 11 and 4 on 2, 4, 8 and 64 threads, `min_instances_per_task` none, 250, 125 and none, `min_unknowns_per_task` none on every number, and `max_unknowns_per_interior` 5,001, 2,501, 1,251 and 157.
Interiors of 5,001, the largest, the mesh of 100 a side's break-even on 2 threads, made its run 51% slower on 64 threads of `cedar` than serially, so the interiors too depend on the threads, and a run on another number finds other last digits wherever it dissects a block.
A mesh's devices are linear, evaluated once a run, for their entries: in a first sweep, whose rounds kept one order, which favoured the spread runs, spreading their rows lost at 100 a side on every number of threads, by 1.0% to 12.5%, and gained at 300 a side by 4.0% at most, so a linear device counts no instance toward `min_instances_per_task`, a mesh's evaluation stays on its run's thread, and the sweep measures chains alone.
A chain's devices spread paid on 4 and 8 threads, in tasks of 250 and 125 MOSFETs, but on no chain up to 1,000 inverters on 2 threads, where a task took each group of MOSFETs whole, or on 64 in tasks of 31.
On 2 threads the amplifier's frequencies paid from 9,999 alone: two tasks of 2,250 lost 0.9% at 500 a decade, where tasks of 900 had gained 22.6% at 200, as the solves at closer frequencies, which keep their pivots, cost less each.
A chain's block is never dissected, so its runs with `max_unknowns_per_interior` set did the serial run's work, and lay within 0.3% of it on every number of threads, the 0.0% of 500 inverters on 64 threads counting as a gain; a mesh's block is dissected from 100 a side at every size of interior swept, and at 70 a side only into interiors of 313 or fewer, where it lost on 64 threads; dissected into interiors of a 64th of its unknowns, the mesh of 300 a side ran 5.2 times as fast on 64 threads as serially.
The work on each unknown spread lost on every chain up to 2,000 inverters, by 8.0% to 39.9% at 2,000, and gained on the mesh of 300 a side alone, by 0.7% to 1.2%, so it spreads by default on no number of threads, as a chain would lose where a mesh barely gains.

### The suite at the defaults

Each fixture of the suite but `monte_carlo`, whose batches run each run on one thread, ran serially, on one thread with no block dissected, and at the defaults, alternately, round after round, by `bench.interleaved`, on 16 November 2026: on `cedar` on its 64 threads at 07:24 UTC, the load average at 1.2 to 2.2, and on this baseline's Apple M2 Max on 4 of its 12 cores at 07:24 UTC, while other programs kept the load average at 4 to 7; each time is the median of its rounds.

| Fixture | `cedar`, serially | `cedar`, defaults | Change | M2 Max, serially | M2 Max, defaults | Change |
| --- | --- | --- | --- | --- | --- | --- |
| `bjt_amp` | 1.594 ms | 1.59 ms | -0.2% | 1.025 ms | 1.067 ms | +4.1% |
| `ac_wide` | 21.7 ms | 9.969 ms | -54.1% | 14 ms | 7.349 ms | -47.5% |
| `buck` | 1.201 s | 1.199 s | -0.2% | 720.8 ms | 718.1 ms | -0.4% |
| `inverter_chain_500` | 8.362 s | 8.358 s | -0.0% | 5.331 s | 3.217 s | -39.7% |
| `rc_mesh_100` | 1.627 s | 1.274 s | -21.7% | 1.122 s | 482.6 ms | -57.0% |

Every fixture found the serial run's bits but `rc_mesh_100`, which the defaults dissect, its 109 points the same and its voltages within 2.1e-15 V of the serial run's.
On 64 threads a chain's devices and its work on each unknown stay on one thread, their break-evens none there, and `bjt_amp`'s 71 frequencies fill no task of 4 for each of 64 threads; on 4, they fill tasks of 11, which lost 4.1% of the amplifier's run on the M2 Max, where they gained on `cedar`'s 4 threads from 45 solves.

### Scaling

`scripts/bench --scale` runs each large fixture once serially, on one thread with no block of its matrix dissected, and then once with its plan's options, every setting of `Parallelism` at its default, on 1, 2, 4 and 8 threads and every core.
The runs of `inverter_chain_50000` took 17 minutes on 12 cores of an Apple M2 Max, so it runs them only with `--largest`, and else once at the defaults; `--threads` names other numbers of threads.
It prints each run's time, its speed-up against the serial run and against the spread run on one thread, and its results' fingerprint, the first 16 hexadecimal digits of the SHA-256 of their arrays.
The spread runs must find the bits of the serial run, or it stops, unless the circuit has more unknowns than the least of the interiors' break-evens, so that its ordering may dissect a block, otherwise on each number of threads.
The results file holds the serial time as the fixture's own, `python/<fixture>`, each spread time as `python/<fixture>/spread on <n> threads`, and the fingerprints.

Measured by `scripts/bench --scale inverter_chain_5000 rc_mesh_300 --threads "[1,2,4,8,16,32,64]"` on `cedar` on 16 November 2026 from 07:28 to 07:45 UTC, while the load average, the measurement's own threads included, was 1.1 to 9.9 on its 64 threads:

| Fixture | Threads | Time | Against serial | Against 1 thread | Results | Load |
| --- | --- | --- | --- | --- | --- | --- |
| `inverter_chain_5000` | serially | 117.2 s | | | `d6b1ef5206419ac1` | 1.19 |
| `inverter_chain_5000` | 1 | 119.1 s | 0.98 | 1.00 | `d6b1ef5206419ac1` | 1.14 |
| `inverter_chain_5000` | 2 | 120.4 s | 0.97 | 0.99 | `d6b1ef5206419ac1` | 1.13 |
| `inverter_chain_5000` | 4 | 77.59 s | 1.51 | 1.54 | `d6b1ef5206419ac1` | 1.18 |
| `inverter_chain_5000` | 8 | 64.29 s | 1.82 | 1.85 | `d6b1ef5206419ac1` | 1.92 |
| `inverter_chain_5000` | 16 | 57.87 s | 2.02 | 2.06 | `d6b1ef5206419ac1` | 3.48 |
| `inverter_chain_5000` | 32 | 60.04 s | 1.95 | 1.98 | `d6b1ef5206419ac1` | 5.04 |
| `inverter_chain_5000` | 64 | 121.5 s | 0.96 | 0.98 | `d6b1ef5206419ac1` | 9.87 |
| `rc_mesh_300` | serially | 46.73 s | | | `4998e669da8acfba` | 2.28 |
| `rc_mesh_300` | 1 | 47.2 s | 0.99 | 1.00 | `4998e669da8acfba` | 1.70 |
| `rc_mesh_300` | 2 | 15.18 s | 3.08 | 3.11 | `54b9ddd5b34d2b45` | 1.33 |
| `rc_mesh_300` | 4 | 12.3 s | 3.80 | 3.84 | `e9b2d729b36516e9` | 1.41 |
| `rc_mesh_300` | 8 | 10.31 s | 4.53 | 4.58 | `40d3afc4e97cf08b` | 1.90 |
| `rc_mesh_300` | 16 | 8.528 s | 5.48 | 5.53 | `40d3afc4e97cf08b` | 2.07 |
| `rc_mesh_300` | 32 | 7.971 s | 5.86 | 5.92 | `40d3afc4e97cf08b` | 2.13 |
| `rc_mesh_300` | 64 | 8.693 s | 5.38 | 5.43 | `53418f756c62322e` | 4.52 |

The chain found the serial run's bits on every number of threads; it spread its devices on 4 to 32 threads, from the break-evens on 4 and 8, and nothing on 1, 2 or 64, where no size measured paid, so on 64 it ran as serially.
The mesh found the serial run's bits on one thread, ordered by the minimum degree, and other last digits on each number of threads whose break-even of the interiors differs, the same on 8, 16 and 32, which take 8's; it ran 5.86 times as fast on 32 threads as serially, its dissection doing it all, as its devices are linear and its work on each unknown spreads on no number.
