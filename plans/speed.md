# Speed

Status: rewritten in plain words on 15 November 2026, every item checked against its source and today's code, and brought up to date on 18 November; the research is `research/speed.md`, kept whole.

Plan: ✅ 6 done · 🚧 3 in progress · 📋 13 planned · ⏸️ 2 parked (§4).

## 1. Goal and metric

Any circuit finishes sooner in jolty than in any reference simulator on the same machine, on one core and on all of them; many runs scale with cores and machines; and one run reaches ten million devices and more.
Measured by the speed suite of `benchmarks.md`, fixture by fixture, against ngspice 47 and Xyce 7.10 at their best configurations.

## 2. Where we stand

**Three workloads, three standings.**

- **Many runs (Monte Carlo, sweeps, batches): jolty leads.**
    A run of an already elaborated circuit sets up in 2.5 µs instead of 63 µs from JSON, with the same bits.
    Monte Carlo through Python reaches 8.9× on 12 threads of this Mac, about 96% of what its 8 performance and 4 efficiency cores allow (an estimate).
    Frequency sweeps spread their frequencies: `ac_wide` 22 ms against ngspice's 117 ms on machine A.
- **One large run: jolty leads on meshes, trails on chains of transistors.**
    On machine A, `rc_mesh_100` takes 1.14 s against ngspice's 18.2 s.
    The chain of 500 inverters takes 3.81 s against 1.76 s (before condensation, item 3, which took 11% off it).
- **One small run: jolty is about 3× slower than ngspice.**
    At defaults on machine B (16 November, `benchmarks.md` item 4's views), the buck converter takes 770 ms against ngspice's 227 ms; on this Mac 795 against 277 ms. machine A's 2.17 s for ngspice of 15 November looks mismeasured.
    The gap is each Newton iteration's cost, not their number.

**Why a small run costs more** (this Mac, 14 November, raw logs, not kept):

- **Evaluating devices:** about 200 ns a MOSFET, against 110 ns for ngspice's whole iteration per MOSFET.
- **The linear algebra:** the matrix was 7× larger than ngspice's, as each MOSFET adds three displacement unknowns (the currents through its gate oxide); condensation now eliminates them before factoring (item 3), so the 500-inverter chain factors 504 unknowns, ngspice's count.
    Since item 3c (`a73e889`, 16 November), each 0.01 Ω switch resistance's current is eliminated by its inner node's row too, so the buck factors **13 of its 23 unknowns**, ngspice's count.
- **Fixed passes per iteration:** jolty keeps the conductance matrix `G` and the capacitance matrix `C` apart for the step control's error filter, which refactors at each judged step: 380,962 refactorizations for the buck's 299,616 iterations.
    These are costs of accuracy, heaviest on small systems (item 7).

**One large run on many cores** (machine A, 32 cores, 15 November, `benchmarks.md` §2): the 50,000-inverter chain reaches 2.81× on 16 threads, short of the 6× target (item 1), because it was run without dissection, the parallel factorization; the 300 × 300 mesh, dissected, runs 2.80× faster on 32 threads than on one, and 3.7× faster than undissected on one thread.
On machine D (18 November), the mesh runs 7.07× faster on 24 threads than serially and the 5,000 chain 2.18× on 16 (item 1).

**An agent's command end to end** (machine D, 18 November, item 8): `jolty simulate bjt_amp.py` takes 951.7 ms without images, 1,336.9 ms with schematic only, and 2,101.0 ms with schematic and plots, against 3.8 ms for `ngspice -b` (553× slower at defaults); the engine itself takes 1.24 ms (0.13%), while cold `import jolty` takes 440 ms and runs twice because `jolty.runner` spawns a fresh child Python process.
**Not measured yet:** anything above 350,004 unknowns.

**Memory, the first wall at ten million devices** (counted from the code, estimates): about 1.4 KB of group state per MOSFET and 1–2 KB of global arrays; every unknown recorded at every time point, 2.1 GiB for the 50,000 chain; a heap string per unknown; 32-bit indices that cap the slab, the unknowns and the matrix entries at about 1.4 × 10⁸ MOSFETs (item 10).

## 3. Gaps

- A small run costs about three times ngspice's, in each iteration.
- A large run reaches 2.8× on 16 cores, not 6×.
- An agent's command takes 952 ms without images (2,101 ms with schematic and plots) against 3.8 ms for `ngspice -b`, dominated by Python startup running twice (`jolty.cli` plus the `jolty.runner` child process) and matplotlib plotting, not the engine (1.24 ms).
- Memory and recording break first at ten million devices; nothing is measured above 350,004 unknowns.

## 4. Plan

Each item says what it is, why it matters, and what "done" means; **Exact** means its results change by rounding at most, and **Lead** that jolty would go beyond every reference.
Its source is the research report's row, in `research/speed.md` §4, or the scaling blueprint, an archived note, not kept, kept whole.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

**General, never tuned** (the owner, 16 November): every speed-up is a general improvement that makes jolty stronger on any circuit and any setting, never a heuristic that wins one fixture, one size or one machine.
A choice between two methods is made by computing their cost for the circuit at hand, as dissection's is; a constant is derived and stated, never picked to pass a benchmark.
A fixture that gets faster while another gets slower is a finding to explain, not a result.

### 🚧 1. One large run six times faster on 16 cores

- **What:** spreading a single large simulation over many cores: the factorization of the matrix, its solves, the evaluation of devices and the assembly of the equations, each split into tasks with results bit-identical on any thread count.
- **State:** Phase 6's four commits are on `main`.
    On this Mac the 50,000-inverter chain, dissected, ran 3.40× faster on 8 threads; on machine A, undissected, 2.81× on 16.
    Each Newton iteration on 8 threads then took about 13 ms: 4.2 ms in the device kernels, 4.7 ms in passes bound by memory bandwidth (the rest of the gather 2.9 ms, forming `G + αC` 0.47 ms on any thread count, the solver's copy, permutations and update), 1.35 ms refactoring and 0.86 ms in the solves' tree.
    6× needs about 7.3 ms.
- **Measured again** (16 November, machine A at load 0.2 to 1.1, `da286de`, the default parallelism with dissection chosen by its computed cost, raw logs not kept): the 50,000 chain takes 1,234 s serially, 1,188 s on 2 threads, 723 on 4, 606 on 8, 542.5 on 16 (2.27×) and 535.7 on 32 (2.30×), below 15 November's 2.81×.
    Serially its time goes 72% to evaluating devices, 10% to assembling, 7% to factoring, 5% to solving and 5% to step control (12,252 iterations, 15,397 refactorizations), so evaluation, which spreads, is not spreading as it should: the first thing to trace.
    Its results passed the fixture's 4 GiB, from more steps since SPICE's charge tolerance; it ran with 16.
- **Traced** (16 November, machine A, the 5,000 chain profiled at each thread count, `profile-5000-by-threads.txt`): evaluating takes 63.6 s serially, 64.5 on 2 threads (it does not spread there at all: the break-even table's least instances per task for 2 threads keeps it in one task), 34.0 on 4, 23.4 on 8 and 18.3 on 16, 3.5× where 16 would allow 16; and everything else stays serial: assembling 10.5 s, factoring 7.4 (the chain is not dissected), solving 3.6, step control 5.0, the rest 1.9, 28 s of the 47 on 16 threads, which bounds the run at 3.3× however fast evaluation becomes.
    6× needs, beyond evaluation spreading well: the gather's assembly spread (item 3b shrinks it), factoring spread for chains (item 4's dissection by cost finds it unprofitable today), step control's per-unknown passes spread, and the 2-thread threshold measured again without the bench's old ordering bias.
- **Traced again** (16 November evening, machine B at load 1.5–4, the 5,000 chain to 5 ns, main `982c48b`, raw logs, not kept): evaluating takes 3.40 s serially and 1.34 s on 16 threads, 2.5×.
    Each instance costs 150 ns on one thread and about 3× that on 16 threads over both CCDs, while 8 cores of one CCD scale almost perfectly: blocks written on one core are gathered on another, and work stealing moves chunks between cores each iteration.
    The gather barely spreads (0.31 → 0.20 ms; its one supply entry of 20,000 terms is summed on one thread), and threads asleep through the serial passes take about 180 µs per evaluation to wake on 16.
    Measured and not kept: each task kept on its own thread (rayon broadcast), −4% end to end on 16 threads only, and it would stall a batch's runs on one another; spreading over only as many threads as the work fills, 6–10% faster on 16 threads but 23% slower on 32 (locality lost, SMT siblings spinning), which needs a pool of exactly that size.
    The largest lever found: spreading the per-unknown passes (`min_unknowns_per_task`, never on by default) makes the 5,000 chain 21–28% faster on 4–16 threads (assembling 0.50 → 0.22 s, step control 0.32 → 0.11 s, evaluation 1.37 → 1.15 s as threads stay awake) but the 500 chain 23–40% slower on 16–32: its break-even is a few hundred to 1,000 unknowns per task, to be measured on large circuits (machine A's sweep reached 14,004 unknowns), not "never"; and a 64-thread run never spreads evaluation, since its break-even there is none.
    Open policy: a run confined to the threads its work fills.
- **Measured on machine D** (18 November, `5efde61`, 24 cores, host load 124 to 315 over its 96): the 300 × 300 mesh runs 7.07× faster on 24 threads than serially (54.56 s to 7.713 s; 3.39× on 2, 5.18× on 4, 6.52× on 8, 6.90× on 16), the same bits on 8 threads and more; the 5,000 chain 2.18× on 16 threads (89.69 s to 41.21 s; 1.91× on 4, 2.11× on 8) and only 1.69× on 24, the same bits on every count; the 50,000 chain takes 489.7 s on 16 threads and 446.8 s on 24, not run serially there.
    On 24 threads the 50,000 chain's time goes 28% to evaluating, 23% to assembling, 18% to factoring, 15% to solving and 12% to step control: the serial passes still bound it.
- **Not done:** the target; 6× on 16 cores for the 50,000 chain is not reached on any machine.
- **Next:** dissection chosen by computed cost for chains too (item 4); the memory-bound passes cut: 32-bit offsets in the gather's map, `G + αC` formed inside the gather (which needs `α` passed into evaluation), 32-bit row entries in the matrix pattern, the solver's copy of the values avoided, the finite-value check folded into the last permutation.
- **Done when:** the 50,000 chain runs 6× faster on 16 cores of machine A than on one, bits unchanged.
- **Source:** row L1; `benchmarks.md` §2 for the measurements.

### ✅ 2. One run over threads measured, and each threshold's break-even

- **What:** `scripts/bench --scale` runs the large fixtures serially and on 1 to 64 threads, checking that the results are bit-identical; `--break-even NAME` finds from which circuit size setting a threshold of `Parallelism` makes a run faster.
- **Result:** both on `main` (`7d41cc0`); measured on this Mac under load and on machine A quiet (raw logs not kept): dissection pays on meshes from about 4,900 unknowns on 4 threads and 10,000 on 2, and on chains barely, 1.7% on 2 threads at 35,004 unknowns and nothing on 4 or more up to that size.
- **Source:** row L4's measurements; `benchmarks.md` items 1 and 2.

### ✅ 3. Each MOSFET's displacement unknowns eliminated before factoring

- **What:** a level 1 or 3 MOSFET carries three internal unknowns, the currents through its gate oxide.
    The sparse solver now eliminates them first, each by its own pivot, which provably passes the solver's pivot test (its diagonal is 1, the other entries of its column are Meyer's capacitances over the oxide's, never above 1), then factors the rest, and recovers them after each solve.
    Every unknown is still solved for, judged and recorded.
- **Result** (`1cf3b9d`): the 500-inverter chain factors 504 of its 3,504 unknowns, ngspice's count, and runs 11% faster end to end; the buck factors 17 of 23.
    Results move by rounding only: at most 3.7e-14 V on the chain and 4.1 µV on the buck.
- **Source:** row S1; blueprint §2.2, level 1; design note an archived note, not kept.

### 📋 3b. The elimination done inside each device's group, so the gather shrinks too

- **What:** item 3 eliminates the displacements in the solver, after the whole system is assembled.
    Done inside each group of devices right after evaluation, each transistor would hand the matrix 4 rows and 16 entries instead of 7 and 31, shrinking the gather and the assembly too.
- **Why:** the gather and the assembly are part of item 1's memory-bound passes.
- **Needs:** `α`, the integration history and the tolerances available at evaluation; the eliminated rows judged inside the groups; the full system kept for the analyses that read it.
- **Done when:** the chain's gather and assembly times fall by a measured share, bits as item 3's.
- **Source:** row S1; an archived note, not kept.

### ✅ 3c. The switches' small series resistances eliminated too

- **What:** a resistance below 1 Ω gets a current unknown, so that `1/R` never sits on a node's diagonal.
    That current can be eliminated with its inner node's row as pivot: −1 against a column of ±1 and −R, which always passes.
    It needs the elimination to take a pivot off the diagonal.
- **Done when:** the buck factors 13 unknowns, ngspice's.
- **Result** (`a73e889`, 16 November): eliminated by their inner nodes' rows, the condensation taking pivots off the diagonal, each kernel claiming its pivots (`Layout::pivot`); the buck factors 13 of its 23 unknowns, ngspice's count; VDMOS's quasi-saturation keeps its drain current factored, its resistance varying without bound.
- **Source:** row S1; an archived note, not kept.

### 📋 3d. A leaf subcircuit's interior eliminated once for all its instances

- **What:** in a design of many identical cells (memory arrays, crossbars), each cell's internal nodes can be eliminated in the cell, leaving only its ports, the static condensation of domain decomposition.
- **Why:** shrinks the global system of regular arrays by the ratio of internal to port nodes.
- **Done when:** a crossbar fixture factors only its ports, bits within rounding.
- **Source:** blueprint §2.2, level 2.

### ✅ 4. Parallelism on by default, and the dissection a numerical option

- **What:** today a run spreads over threads only when its options set `Parallelism`'s thresholds.
    Three changes, in one API change:
  1. The dissection's interior size, `max_unknowns_per_interior`, stays in `Parallelism` (the owner, 15 November, once results may differ by rounding across threads); `Parallelism` gains a threshold for spreading a dissected factorization.
  1. Every threshold defaults to its break-even measured on a quiet machine (item 2's, on machine A), so a run spreads where it pays and not where it does not.
  1. Whether to dissect is decided by computing the cost of the two orderings for the circuit at hand, not by a size threshold, since the break-even differs by kind of circuit: a mesh from about 4,900 unknowns, a chain barely at 35,004.
- **Also:** `min_instances_per_task` spreads at one task's worth and `min_solves_per_task` at two; they should start at the same multiple.
- **And the names, as the owner decided** (`naming.md`, choices 8 and 9): `threads` becomes `max_threads`, and the thresholds one family of least sizes per task, `min_solves_per_task`, `min_instances_per_task` (an exact size today) and `min_unknowns_per_task` (a size of the whole circuit today, `min_unknowns_inside`); the break-evens are retaken for the new definitions.
- **Done when:** a run spreads by default exactly where it pays, results bit-identical on any thread count; `rc_mesh_100`, which the inside-work threshold slows by 23–43%, not slowed.
- **Result** (`6c6d08d` to `3f1124b`): the owner's names (`max_threads`, `min_solves_per_task`, `min_instances_per_task`, `min_unknowns_per_task`, least sizes per task); every threshold defaults to its break-even on machine A per number of threads (`jolty.plan.BREAK_EVENS`, the guide's table held to it); dissection decided by the two orderings' multiply-adds.
    Serial against the defaults: `ac_wide` −54% on machine A, −48% on this Mac; `rc_mesh_100` −22% and −57%; the chain −40% on the Mac, unchanged on machine A; `bjt_amp` +4.1% on the Mac (its 70 solves spread in tasks of 11).
    The bench's interleaving now alternates its order: the serial run had always followed the 64-thread run, which made spreading look up to 16% better; the break-evens of 15 November (item 2) carry that bias.
- **Source:** row L4; `AGENTS.md`'s adopted rules on parallelism and on computed costs over thresholds.

### ✅ 4b. The buck's slowdown after Phase 6 removed

- **What:** Phase 6 made the buck's 0.2 ms transient 6% slower on one thread.
    Counted by callgrind: the solve's rows above each block and their sums had become function calls, the evaluation's per-group split a call returning twice the arguments, and Gear2's error control a power computed per charge.
- **Result** (`495b713`, `204cbf9`): 65.9 ms against 64.5 ms before Phase 6 on machine B, +2.1%, within the 5% target, bits unchanged.
    The rest, 0.66 ms of step control, is Gear2's own (§5).

### 📋 5. A MOSFET's evaluation cheaper still

- **What:** after `52e3bdb` and `4f39ff4`, a MOSFET costs about 200 ns per evaluation.
    Two known costs remain: the group's per-instance work, 51 ns (zeroing its block and gathering its 83 values), and the junction charges' `powf`, 18 ns.
- **Done when:** each measured before and after, bits within rounding.
- **Source:** row S2.

### ✅ 6. Stamp linear devices once per run

- **What:** resistors, capacitors, inductors, fixed source entries and a MOSFET's overlap capacitances contribute the same matrix entries at every Newton iteration.
    Summed once per run into a base, each iteration would start from it and add only what varies; a linear device's residual and charge come from the base times the unknowns, so it is not evaluated at all.
- **Why:** 19–23% of `rc_mesh_100`'s run is evaluation, nearly all of it linear devices; ngspice reloads everything each iteration (`cktload.c`, lines 62–75), so this would lead it.
- **Done when:** the meshes' evaluation share falls by a measured amount, bits within rounding, unchanged on any thread count.
- **Result** (`4e3499a`): a kernel declares itself linear (`Kernel::linear`), its entries evaluated once a run into a base each iteration starts from: `rc_mesh_100` 4.2% faster, its evaluation share 16.3% to 10.9%; the buck 3.4%; results moved by rounding only (3.7e-15 V on the mesh), the same on any thread count.
- **Left:** the constant entries of nonlinear devices (sources' ±1, fixed series resistances, MOSFET overlaps), which share slots with varying ones.
- **Source:** row S3; brief a brief, not kept.

### 🚧 7. The fixed passes of an iteration cheaper on small systems

- **What:** three passes cost more than their work on a circuit of a few dozen unknowns: the slab's two-stage add (devices write blocks, then a gather adds them), the `|J|·|x|` product the convergence test needs, and the error filter's refactorization at each judged step.
- **Done when:** the buck's iteration costs at most 1.5× ngspice's, with no accuracy given up.
- **Result, in part** (`3f53643`, `9277c9b`, 16 November): the refactorization's loop tighter and the error filter allocating nothing, the same bits; the buck 848 → 825 ms (−2.8%), `bjt_amp` and `ac_wide` unchanged, the 500 inverters +0.7% from the general condensation's bookkeeping (3,000 unknowns eliminated per factor).
    The buck's iteration: jolty 2,661–2,697 ns against ngspice's 1,036, evaluation 1,074–1,101 against 433, factoring 449–467 against 95, solving 351–362 against 86; jolty factors 1.35 times an iteration, its error filter refactoring once per judged step, and runs 331,020 iterations to ngspice's 196,438.
    Still 2.4× ngspice's, not 1.5×: evaluation (item 5) and the filter's factorization are what remain.
- **Source:** row S4.

### 🚧 8. An agent's command timed from start to finish, then cut

- **What:** what an agent waits for is `jolty simulate`, which starts a fresh Python process, imports jolty, builds and elaborates the circuit, and runs it.
    First measure it, through the command line and through MCP; then cut it with a runner kept warm between commands, each script still run in a process of its own.
- **Why:** on `bjt_amp` the engine takes 1.24 ms and `ngspice -b` 3.8 ms on machine D; an agent sees neither, only the command's wall time.
- **Measured** (18 November, `a3c1d6c`, machine D, AMD EPYC 9B14, ngspice 47 with KLU, host load 93.11 to 112.51 over 96 cores for cold runs and 64.87 to 89.22 for warm `service.simulate`):

    Cold subprocess wall times (median and IQR over 7 runs):

    | Command | Median | IQR | Share / notes |
    | --- | ---: | --- | --- |
    | `ngspice -b bjt_amp.cir` (`op`+`ac`+`tran`) | 3.8 ms | 3.7 to 4.0 ms | Reference batch run |
    | `python -c pass` | 13.5 ms | 13.2 to 14.4 ms | Bare interpreter |
    | `import jolty._core` | 447.6 ms | 415.7 to 460.1 ms | Executes `jolty/__init__.py` first (`_core` itself is 0.018 ms in `-X importtime`) |
    | `import jolty` | 440.0 ms | 406.8 to 535.8 ms | `jolty` eager import |
    | `import jolty.cli` | 522.4 ms | 474.0 to 565.1 ms | CLI entry point |
    | `import jolty.plots` | 791.5 ms | 650.0 to 838.0 ms | Pulls in `matplotlib` |
    | `jolty simulate bjt_amp.py --no-schematic --no-plot` | 951.7 ms | 926.3 to 975.3 ms | Spawns 2 Python processes: CLI plus `jolty.runner` child (250× `ngspice -b`) |
    | `jolty simulate bjt_amp.py --no-check --no-schematic --no-plot` | 988.5 ms | 928.5 to 1,046.9 ms | Checks cost noise-level time on `bjt_amp` |
    | `jolty simulate bjt_amp.py --no-plot` | 1,336.9 ms | 1,306.4 to 1,412.7 ms | Schematic only (+385 ms over no-images) |
    | `jolty simulate bjt_amp.py` (default: schematic + plots) | 2,101.0 ms | 2,035.7 to 2,203.0 ms | 553× `ngspice -b` (+764 ms plotting over schematic only) |

    Warm `service.simulate` (same path as an already-running MCP server, median of 5 runs):

    | Call | Median | Samples (ms) | Delta |
    | --- | ---: | --- | --- |
    | `service.simulate` (`schematic=False, plot=False`) | 504.7 ms | `478.7, 487.4, 504.7, 509.7, 521.9` | Fresh `jolty.runner` child spawned per call |
    | `service.simulate` (`schematic=True, plot=False`) | 631.7 ms | `582.5, 625.6, 631.7, 642.1, 841.3` | +127 ms warm schematic draw |
    | `service.simulate` (`schematic=True, plot=True`) | 1,294.2 ms | `1,235.2, 1,252.5, 1,294.2, 1,339.0, 1,346.5` | +663 ms warm matplotlib plotting |

    Warm in-process engine phases (median of 11 runs):

    | Phase | Median | Min |
    | --- | ---: | ---: |
    | `build()` + `circuit()` | 0.006 ms | 0.004 ms |
    | Elaborate + `Simulation()` | 0.212 ms | 0.204 ms |
    | `run_plan` (`op`+`ac`+`tran`) | 1.024 ms | 1.009 ms |
    | **Total engine work** | **1.242 ms** | 0.13% of `--no-schematic --no-plot` CLI time |

    Top cumulative import times from `python -X importtime -c "import jolty.cli"` (total 371.8 ms in that run; `jolty` 322.1 ms, `jolty._core` itself 0.018 ms): `jolty.files` 209.0 ms (`numpy` 81.8 ms, `jolty.plan` 72.8 ms, `jolty.spice` 60.0 ms, `jolty.spice.download` 43.3 ms), `jolty.devices` 80.9 ms, `jolty.service` 41.1 ms (`jolty.settings` 30.6 ms via `pydantic_settings` 28.8 ms), `pydantic` 23.4 ms, `jolty.results` 23.1 ms, `jolty.ir` 19.4 ms.
- **Not done:** cutting cold import cost (lazier imports in `jolty/__init__.py` and `jolty.files`), keeping a warm `jolty.runner` worker between MCP/CLI calls, and faster plotting or parallel image generation.
- **Done when:** the speed suite reports `jolty simulate`'s wall time on `bjt_amp`, with and without images, below `ngspice -b`'s.
- **Source:** row S5; `research/agentic.md` row 6; `research/benchmarks.md` row B15.
    This item replaces `agentic.md`'s and `benchmarks.md`'s copies of it.

### 📋 9. Exact latency: skip a device whose inputs have not changed

- **What:** a kernel is a pure function of its parameters, its memory and its local unknowns, so an instance whose inputs repeat bit for bit since its last evaluation would return the same bits: it can be skipped, and the gather can add only what changed.
    SPICE's "bypass" does this approximately, with a tolerance; this is exact.
- **Why:** evaluation is 69% of the 50,000 chain run serially, and in a large mostly quiet circuit most devices repeat; up to about 3.5× if 99% of evaluations repeat (an estimate).
- **Done when:** first, a probe counts the share of bit-identical evaluations on the chain; then, if it pays, the skipping gives the same bits as full evaluation on every fixture and thread count.
- **Source:** row L2.

### 📋 10. Memory at ten million devices

- **What:** a large run records only what it names, streaming to disk by default; names are interned or formatted on demand instead of one heap string per unknown; the copy of the solver kept for the remembered operating point is shared or optional; the slab becomes a per-thread tile without the 32-bit cap.
- **Why:** 2.1 GiB to record the 50,000 chain at about 800 points; about 1.4 KB of state per MOSFET.
- **Done when:** a million-MOSFET chain runs on this 32 GB Mac, its peak bytes per device reported.
- **Source:** row L8; blueprint §3.2 and §3.4.

### 📋 11. The front end at a million to ten million devices

- **What:** building a circuit in Python, sending it to the engine as JSON and elaborating it are untimed above 10⁴ devices.
    First time them at 10⁵ and 10⁶; then, worst first: a builder in Rust with interned names, no JSON on the way, each subcircuit definition elaborated once, and the linear scans in `Groups::add`, `Recorded::of` and the checks removed.
- **Done when:** 10⁶ devices built and elaborated, timed and profiled; 10⁷ fits in memory.
- **Source:** row L9; blueprint §1 and §3.5.
    Writing such circuits in a few lines is `agentic.md` 20.

### 📋 12. Orderings for chip-like circuits

- **What:** the nested dissection gives up below three levels, and the block triangular form removes only ideal supplies.
    A resistive supply net touching every cell, or a crossbar, defeats both.
    Needed: hub nets handled apart, multilevel separators, and a fixture of a supply grid feeding logic.
- **Done when:** that fixture dissects into interiors and scales.
- **Source:** row L5; blueprint §2.1.

### 📋 13. Iterative solves preconditioned by the last factors

- **What:** a run bound by refactorization (78% of `rc_mesh_300`'s time) could solve most iterations with a Krylov method preconditioned by the last factors, stopped at the resolution of `|J|·|x|`, its sums in a fixed order.
- **Done when:** the mesh's transient faster at equal accuracy, the same bits on any thread count.
- **Lead.**
    **Source:** row L6; blueprint §2.4.

### 📋 14. Dense kernels for the separators' fronts

- **What:** the dense blocks at the top of the dissection tree are eliminated by scalar loops; register-blocked kernels, vectorized across entries only, each entry summed in a fixed order, would be faster with the same bits at any width.
- **Done when:** the fronts eliminate faster, bits unchanged at any thread count and vector width.
- **Source:** row L7; blueprint §2.4.

### 📋 15. Samples in vector lanes

- **What:** a batch's samples evaluated several at a time in SIMD lanes, each lane taking its own steps, each bit-identical to its sample run alone.
- **Needs:** the same bits on every platform first (`accuracy.md` 13).
- **Done when:** k lanes give k single runs' bits, throughput measured.
- **Lead.**
    **Source:** row M2; blueprint §3.3.

### 📋 16. An exact cache of results

- **What:** a run's results keyed by a hash of everything it reads; a repeated run returns the cached bits.
    jolty's determinism makes this exact.
- **Done when:** a repeated run returns the cached bits, a changed input misses.
- **Lead.**
    **Source:** row M5.

### 📋 18. Approximations as knobs that report their own error

- **What:** bypass, multirate integration and partitioned relaxation, the FastSPICE methods, each a knob a run opts into, never a default, each run reporting the measured defect of the whole circuit's equations.
- **Why:** FastSPICE tools claim 10–1000× on very large mostly quiet circuits, at 10–15% error.
- **Done when:** each knob's runs state their measured defect.
- **Lead.**
    **Source:** row L3; blueprint §2.3; `AGENTS.md`, "Exact before approximate".

### ⏸️ 19. Krylov pole-zero and matrix-free shooting

- **What:** the pole-zero analysis and the periodic steady state use dense matrices (0.77 s at 500 unknowns).
- **Parked until:** a circuit whose dense matrices outgrow them.
- **Source:** the old roadmap's item 14.

### ⏸️ 20. Meyer's capacitances integrated as ngspice does, an option

- **Parked** by the owner (15 November): condensation (item 3) already brings the matrix to ngspice's size with jolty's equations, and the option would make transient sensitivities and the periodic steady state inexact (§5).

### 📋 21. The schematic and the 3D viewer at a million instances

- **What:** `jolty draw` and `jolty view` draw every instance apart; at scale they need instancing, one drawing per definition placed many times.
- **Done when:** a 10⁵-instance array draws and views within a stated time.
- **Source:** blueprint §3.6.

**Elsewhere:** the stamping of BSIM3, VDMOS and VBIC through summed derivatives is `quality.md` 5, measured for speed there; transient adjoint sensitivities are `agentic.md` 13; jolty's own correctly rounded elementary functions, for the same bits on every platform, are `accuracy.md` 13.

**The blueprint's sections and where they are:** §1, the front end and tensor buses: item 11 and `agentic.md` 20; §2.1, orderings: item 12; §2.2, condensation: items 3, 3b, 3c and 3d; §2.3, FastSPICE methods: item 18; §2.4, multifrontal and Krylov solvers: items 13 and 14; §3.1, the cold start and homotopies: `accuracy.md` 8 and 9; §3.2, slab memory: item 10; §3.3, vector lanes and bypass: items 15 and 9; §3.4, tensor stimuli and selective recording: item 10 and `agentic.md` 20; §3.5, linear scans: item 11; §3.6, the schematic's and the viewer's instancing at scale: item 21 below; §4, differentiable simulation: `agentic.md` 12, 13 and 19.

## 5. Decided

- 15 Nov: Gear2's error control left the trapezoidal default 0.66 ms slower on the buck's 0.2 ms (step control 8.10 against 7.44 ms on machine B, 1% of the run): the bound each charge's step is held by and the unit steps a multistep formula keeps.
    Within item 4b's 5%, kept as Gear2's own cost.
- 15 Nov (superseded 16 Nov by item 3c, `a73e889`, which also eliminates small series resistances' currents via off-diagonal pivots): of the internal unknowns, only the MOSFET's displacements were eliminated, as only their pivots provably pass the threshold: a small series resistance's node and current pass only while the transistor's conductance stays under 11.1 S at 0.01 Ω, which the buck's switches break while they conduct (about 33 S, an estimate from their parameters); the BJT's base node's column holds the transconductance, 0.39 S against a pivot of 0.014 S at 10 mA; a heated MOSFET's displacements reach its thermal row, which nothing bounds.
- 15 Nov: the elimination lives in the sparse solver, not in the groups as first planned: every consumer reads the whole system as before and one mechanism serves every analysis; the groups' version is item 3b.
- 15 Nov: the ngspice-style Meyer option waits behind condensation: ngspice's form needs each kernel to know which time points the transient accepts, would make transient sensitivities and the periodic analysis inexact (an archived note, not kept), and condensation already reaches ngspice's matrix size.
- Phase 6's solves sum a row of more than 1,024 entries above a block in chunks, `reduce::fold`, the same bits on any thread count; the gather and `|J|·|x|` keep their order, chunked they measured slower or no faster.
- Meyer's capacitances conserve charge in neither engine: they have no charge function.
    jolty's displacement unknowns give `C(v)·dv/dt` exact derivatives and put error control on the oxide's linear charge; ngspice averages the capacitance over the step.
    VDMOS's gate charge does conserve.
- A MOSFET's junctions and overlaps are differentiated along their own voltage: 4–12% faster evaluation, the bits moved by at most 3.6e-14 V on the chain and 2.6 µV on the buck.
- `#[inline(always)]` on the functions that sum a model's derivatives in registers and on the solve's rows above a block, each with its measured reason.
- Phase 6 partitions by nested dissection, after independent blocks and level scheduling measured 1.0–1.4×.
- The pivot threshold is a tenth, not KLU's thousandth, for accuracy without row scaling.
- The remembered operating point keeps a copy of the solver, so later analyses are the same bits whatever ran between: `rc_mesh_300`'s peak memory +26% serially, +42% dissected.
- One scheduler: a single rayon pool, owned by the caller, runs every level, so nested parallelism never oversubscribes the cores; kernels know nothing of threads (an archived note, not kept).
- mimalloc is the Python extension's allocator: Monte Carlo at 95.9% of linear on 8 threads, against 64–68%.
- How speed claims are made is `benchmarks.md`'s: references at their best configuration, engine and end-to-end times, efficiency against attainable throughput.

## 6. Open principles

None: the research's proposals were merged into the rules of `AGENTS.md` on 15 November (`444535f`).
