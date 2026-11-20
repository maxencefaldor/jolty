# Speed

Status: rewritten in plain language on 15 November 2026, checked against source documents and current code, and updated on 20 November.
Full background research remains archived in `research/speed.md`.

Plan: ✅ 6 done · 🚧 4 in progress · 📋 13 planned · ⏸️ 2 parked (§4).

## 1. Goal and metric

Any circuit finishes sooner in jolty than in any reference simulator on the same machine, on one core and on all cores.
Batched runs scale across cores and machines, and a single simulation reaches ten million devices and beyond.
Progress is measured by the speed suite in `benchmarks.md`, fixture by fixture, against ngspice 47 and Xyce 7.10 at best configurations.

## 2. Where we stand

**Three workloads, three standings.**

- **Many runs (Monte Carlo, sweeps, batches): jolty leads.**
    Re-running an already elaborated circuit initializes in 2.5 µs instead of 63 µs from JSON, producing identical bits.
    Monte Carlo through Python reaches 8.9× on 12 threads of this Mac, about 96% of the estimated capacity of 8 performance and 4 efficiency cores.
    Frequency sweeps parallelize across frequencies: `ac_wide` takes 22 ms versus 117 ms for ngspice on machine A.
- **One large run: jolty leads on meshes and trails on transistor chains.**
    On machine A, `rc_mesh_100` takes 1.14 s versus 18.2 s for ngspice.
    The 500-inverter chain takes 3.81 s versus 1.76 s (measured before item 3 condensation cut 11%).
- **One small run: jolty is about 3× slower than ngspice.**
    At defaults on machine B (16 November, `benchmarks.md` item 4 views), the buck converter takes 770 ms versus 227 ms for ngspice, and 795 ms versus 277 ms on this Mac.
    Machine A's 2.17 s ngspice timing from 15 November appears mismeasured.
    The gap lies in per-iteration Newton cost rather than iteration count.

**Why a small run costs more** (this Mac, 14 November, unarchived raw logs):

- **Evaluating devices:** roughly 200 ns per MOSFET, versus 110 ns for an entire ngspice iteration per MOSFET.
- **Linear algebra:** the matrix was previously 7× larger than ngspice's because each MOSFET added three gate-oxide displacement unknowns.
    Condensation now eliminates those unknowns before factoring (item 3), reducing the 500-inverter chain to 504 factored unknowns (matching ngspice).
    Since item 3c (`a73e889`, 16 November), inner-node pivots also eliminate each 0.01 Ω switch resistance current, so the buck factors **13 of 23 unknowns** (matching ngspice).
- **Fixed passes per iteration:** jolty keeps conductance matrix `G` and capacitance matrix `C` separate for the step-control error filter, which refactors every judged step (380,962 refactorizations across 299,616 buck iterations).
    Accuracy passes carry fixed overhead that weighs heaviest on small systems (item 7).

**One large run on many cores** (machine A, 32 cores, 15 November, `benchmarks.md` §2): the 50,000-inverter chain reached 2.81× on 16 threads without parallel dissection, below the 6× target (item 1).
On machine A, the dissected 300 × 300 mesh ran 2.80× faster on 32 threads than on 1 thread, and 3.7× faster than undissected serial execution.
On machine D (18 November), the mesh ran 7.07× faster on 24 threads than serially, and the 5,000-inverter chain reached 2.18× on 16 threads (item 1).

**End-to-end agent command** (machine D, 20 November, item 8): after PEP 810 lazy imports and a standby service runner, cold `import jolty` dropped from 440.0 ms to 13.2 ms and `import jolty.cli` from 522.4 ms to 15.5 ms, over 33× faster.
Cold `jolty simulate bjt_amp.py` takes 687.4 ms without images, 987.0 ms with schematic only, and 1,322.4 ms with schematic and plots, down from 951.7 ms, 1,336.9 ms, and 2,101.0 ms previously.
Warm `service.simulate` (MCP path) dropped from 504.7 ms to 149.9 ms without images (139.4 ms on standby hit), 266.9 ms with schematic only, and 647.7 ms with schematic and plots.
Against 3.8 ms for `ngspice -b`, cold CLI execution remains dominated by spawning the isolated `jolty.runner` child process and matplotlib plotting rather than the 1.24 ms engine run.

**Not measured yet:** circuits above 350,004 unknowns.

**Memory, the first wall at ten million devices** (code estimates): group state requires about 1.4 KB per MOSFET alongside 1 to 2 KB of global arrays.
Recording every unknown at every time point takes 2.1 GiB for the 50,000-inverter chain, plus one heap string per unknown.
32-bit indices also cap the slab, unknowns, and matrix entries near 1.4 × 10⁸ MOSFETs (item 10).

## 3. Gaps

- Small runs cost roughly three times ngspice per Newton iteration.
- Large runs reach 2.8× on 16 cores instead of the 6× target.
- A cold agent CLI command still takes 687 ms without images (1,322 ms with schematic and plots; 150 ms warm via `service.simulate`) versus 3.8 ms for `ngspice -b`, dominated by subprocess isolation and matplotlib plotting rather than the 1.24 ms engine run (item 8).
- Memory and recording hit limits first at ten million devices, and nothing is measured above 350,004 unknowns.
- PDK cache deserialization still takes 0.592 s across six SKY130 corners in a cold process (down from 4.35 s cold parse; item 22 target is under 0.5 s), while native parallel Monte Carlo already runs 100 SKY130 `tt_mm` draws on 25 transistors in 42.0 ms.

## 4. Plan

Each item describes scope, rationale, and completion criteria; **Exact** means results change by rounding at most, and **Lead** marks an item exceeding every reference.
Each source cites a row in `research/speed.md` §4 or the archived scaling blueprint.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

**General, never tuned** (the owner, 16 November): every optimization must be a general improvement across circuits and settings, never a heuristic tuned to one fixture, size, or machine.
Algorithmic choices compute cost for the circuit at hand, as dissection does, and every constant is derived and documented rather than picked to pass a benchmark.
Any optimization that speeds up one fixture while slowing another requires investigation before landing.

### 🚧 1. One large run six times faster on 16 cores

- **What:** parallelize a single large simulation across cores (factorization, solves, device evaluation, and assembly) with bit-identical results across thread counts.
- **State:** Phase 6's four commits are on `main`.
    On this Mac, the dissected 50,000-inverter chain ran 3.40× faster on 8 threads; on machine A without dissection, the chain reached 2.81× on 16 threads.
    Each 8-thread Newton iteration then took about 13 ms: 4.2 ms in device kernels, 1.35 ms refactoring, 0.86 ms in the solve tree, and 4.7 ms in memory-bound passes.
    Memory-bound passes covered the remaining gather (2.9 ms), forming `G + αC` (0.47 ms regardless of thread count), solver value copies, permutations, and updates; reaching 6× requires roughly 7.3 ms.
- **Measured again** (16 November, machine A, load 0.2 to 1.1, `da286de`, default parallelism with cost-based dissection, unarchived logs): the 50,000-inverter chain took 1,234 s serially and 1,188 s on 2 threads.
    Multi-thread times were 723 s on 4 threads, 606 s on 8, 542.5 s on 16 (2.27×), and 535.7 s on 32 (2.30×), below the 2.81× figure from 15 November.
    Serial runtime broke down to 72% device evaluation, 10% assembly, 7% factoring, 5% solving, and 5% step control across 12,252 iterations and 15,397 refactorizations.
    Evaluation dominates serial runtime yet failed to scale linearly, making evaluation scaling the first target to profile.
    Recorded waveforms exceeded the fixture's 4 GiB limit due to extra steps under SPICE charge tolerance, so the run used a 16 GiB limit.
- **Traced** (16 November, machine A, 5,000-inverter chain profiled per thread count, `profile-5000-by-threads.txt`): evaluation took 63.6 s serially and 64.5 s on 2 threads, where the break-even threshold kept work in one task.
    Evaluation fell to 34.0 s on 4 threads, 23.4 s on 8, and 18.3 s on 16 (3.5× on 16 cores).
    All non-evaluation phases stayed serial: assembly 10.5 s, undissected factoring 7.4 s, solving 3.6 s, step control 5.0 s, and overhead 1.9 s.
    The 28 s of serial work out of 47 s on 16 threads caps end-to-end speedup at 3.3× regardless of evaluation speed.
    Reaching 6× requires strong evaluation scaling, parallel gather assembly (shrunk by item 3b), chain dissection (currently unprofitable by item 4 cost model), parallel step control, and re-measuring the 2-thread threshold without ordering bias.
- **Traced again** (16 November evening, machine B, load 1.5 to 4, 5,000-inverter chain to 5 ns, `982c48b`, unarchived logs): evaluation dropped from 3.40 s serially to 1.34 s on 16 threads (2.5×).
    Per-instance evaluation costs 150 ns on 1 thread and roughly 3× more across both CCDs on 16 threads, whereas 8 cores on one CCD scale nearly linearly.
    Cross-CCD overhead arises when blocks written on one core are gathered on another and work stealing migrates chunks between iterations.
    Gather barely scaled (0.31 → 0.20 ms, with one 20,000-term supply entry summed serially), and threads idling during serial passes took ~180 µs to wake on 16 threads.
    Tested and rejected: thread pinning via rayon broadcast saved only 4% on 16 threads while stalling concurrent batch runs.
    Limiting threads to active work ran 6 to 10% faster on 16 threads but 23% slower on 32 threads due to lost locality and spinning SMT siblings, requiring a right-sized pool.
    The largest gain came from enabling per-unknown parallelism (`min_unknowns_per_task`, off by default), speeding up the 5,000-inverter chain by 21 to 28% on 4 to 16 threads.
    Phase times fell for assembly (0.50 → 0.22 s), step control (0.32 → 0.11 s), and evaluation (1.37 → 1.15 s as threads stayed awake).
    Enabling per-unknown parallelism slowed the 500-inverter chain by 23 to 40% on 16 to 32 threads.
    The true break-even lies between a few hundred and 1,000 unknowns per task on large circuits (machine A's sweep stopped at 14,004 unknowns), and 64-thread runs never parallelize evaluation because no break-even is configured.
    Open policy: confine a run to the threads its work fills.
- **Measured on machine D** (18 November, `5efde61`, 24 cores, host load 124 to 315 over 96 cores): the 300 × 300 mesh ran 7.07× faster on 24 threads than serially (54.56 s to 7.713 s).
    Mesh speedups were 3.39× on 2 threads, 5.18× on 4, 6.52× on 8, and 6.90× on 16, bit-identical from 8 threads upward.
    The 5,000-inverter chain reached 2.18× on 16 threads (89.69 s to 41.21 s; 1.91× on 4, 2.11× on 8) and 1.69× on 24 threads, bit-identical across all counts.
    The 50,000-inverter chain took 489.7 s on 16 threads and 446.8 s on 24 threads (no serial baseline on machine D).
    On 24 threads, 50,000-chain runtime splits across evaluation (28%), assembly (23%), factoring (18%), solving (15%), and step control (12%), showing serial passes still dominate.
- **Not done:** the 6× target on 16 cores for the 50,000-inverter chain remains unmet on all machines.
- **Next:** extend cost-based dissection to chains (item 4) and shrink memory-bound passes via 32-bit gather offsets and 32-bit matrix-pattern row indices.
    Further memory cuts form `G + αC` inside gather (passing `α` into evaluation), avoid solver value copies, and fold finite-value checks into the final permutation.
- **Done when:** the 50,000-inverter chain runs 6× faster on 16 cores of machine A than serially, with identical bits.
- **Source:** row L1; `benchmarks.md` §2 for measurements.

### ✅ 2. One run over threads measured, and each threshold's break-even

- **What:** `scripts/bench --scale` runs large fixtures from 1 to 64 threads verifying bit identity, and `--break-even NAME` finds the circuit size where each `Parallelism` threshold pays off.
- **Result:** both commands landed on `main` (`7d41cc0`) and were measured on a loaded Mac and quiet machine A (unarchived logs).
    Dissection pays off on meshes above ~4,900 unknowns on 4 threads and 10,000 on 2, while chains gained only 1.7% on 2 threads at 35,004 unknowns and nothing on 4 or more threads.
- **Source:** row L4 measurements; `benchmarks.md` items 1 and 2.

### ✅ 3. Each MOSFET's displacement unknowns eliminated before factoring

- **What:** level 1 and level 3 MOSFETs carry three internal gate-oxide displacement-current unknowns.
    The sparse solver eliminates each displacement unknown first via a dedicated pivot (diagonal 1 with column entries bounded by 1 as Meyer-to-oxide ratios), then factors the reduced matrix and recovers the displacements.
    Every unknown is still solved, error-checked, and recorded.
- **Result** (`1cf3b9d`): the 500-inverter chain factors 504 of 3,504 unknowns (matching ngspice) and runs 11% faster end to end; the buck factors 17 of 23.
    Results shift by rounding only: at most 3.7e-14 V on the chain and 4.1 µV on the buck.
- **Source:** row S1; blueprint §2.2 level 1; archived design note.

### 📋 3b. The elimination done inside each device's group, so the gather shrinks too

- **What:** item 3 eliminates displacements inside the solver after full system assembly.
    Eliminating displacements inside each device group right after evaluation hands the matrix 4 rows and 16 entries per transistor instead of 7 and 31, shrinking gather and assembly.
- **Why:** gather and assembly form a major share of item 1's memory-bound passes.
- **Needs:** passing `α`, integration history, and tolerances into evaluation, judging eliminated rows inside groups, and retaining the full system for analyses requiring it.
- **Done when:** chain gather and assembly times drop by a measured fraction with item 3's bit-identical results.
- **Source:** row S1; archived design note.

### ✅ 3c. The switches' small series resistances eliminated too

- **What:** resistances below 1 Ω introduce a current unknown so `1/R` never lands on a node diagonal.
    An inner-node row pivot eliminates the current unknown cleanly: -1 against a column of ±1 and -R always passes the pivot test.
    Supporting inner-node elimination requires condensation to accept off-diagonal pivots.
- **Done when:** the buck factors 13 unknowns, matching ngspice.
- **Result** (`a73e889`, 16 November): condensation now accepts off-diagonal pivots declared by kernels via `Layout::pivot`, eliminating small-resistance currents on inner-node rows.
    The buck now factors 13 of 23 unknowns (matching ngspice), while VDMOS quasi-saturation keeps its drain current factored because its resistance varies unboundedly.
- **Source:** row S1; archived design note.

### 📋 3d. A leaf subcircuit's interior eliminated once for all its instances

- **What:** in repeated-cell designs (memory arrays, crossbars), static condensation eliminates each cell's internal nodes once, leaving only port equations.
- **Why:** reduces global system size in regular arrays by the ratio of internal nodes to port nodes.
- **Done when:** a crossbar fixture factors only port nodes with results agreeing within rounding.
- **Source:** blueprint §2.2 level 2.

### ✅ 4. Parallelism on by default, and the dissection a numerical option

- **What:** runs previously parallelized only when options set `Parallelism` thresholds explicitly.
    Three changes shipped in one API update:
  1. `max_unknowns_per_interior` stays in `Parallelism` (owner decision, 15 November, once rounding differences across threads are allowed), and `Parallelism` adds a threshold for dissected factorization.
  1. Every threshold defaults to its break-even measured on quiet machine A (item 2), parallelizing only where beneficial.
  1. Dissection activates via a computed cost model comparing both orderings for the circuit, rather than a fixed size threshold, because break-evens vary by topology (~4,900 unknowns on meshes versus barely 35,004 on chains).
- **Threshold alignment:** `min_instances_per_task` triggered at one task while `min_solves_per_task` triggered at two; both should start at the same multiple.
- **Owner-approved names** (`naming.md`, choices 8 and 9): `threads` became `max_threads`, and thresholds unified as per-task minima (`min_solves_per_task`, `min_instances_per_task`, `min_unknowns_per_task`).
    Break-evens were re-measured for the revised per-task definitions (replacing exact-size `min_instances_per_task` and whole-circuit `min_unknowns_inside`).
- **Done when:** default parallelism activates only where profitable with bit-identical results across thread counts, without slowing `rc_mesh_100` (previously 23 to 43% slower under `min_unknowns_inside`).
- **Result** (`6c6d08d` to `3f1124b`): adopted owner names (`max_threads`, `min_solves_per_task`, `min_instances_per_task`, `min_unknowns_per_task`), defaulting each threshold to machine A per-thread break-evens (`jolty.plan.BREAK_EVENS`, validated against the guide).
    Dissection now compares multiply-add counts of both orderings.
    Defaults versus serial cut `ac_wide` by 54% on machine A (48% on Mac), `rc_mesh_100` by 22% (57% on Mac), and the chain by 40% on Mac (flat on machine A).
    `bjt_amp` slowed by 4.1% on Mac (70 solves split into tasks of 11).
    Benchmark interleaving now alternates run order because running serial after 64 threads biased 15 November break-evens (item 2) up to 16% toward parallel runs.
- **Source:** row L4; `AGENTS.md` rules on parallelism and cost-based choices.

### ✅ 4b. The buck's slowdown after Phase 6 removed

- **What:** Phase 6 slowed the buck 0.2 ms transient by 6% on one thread.
    Callgrind attributed the slowdown to non-inlined solve-row reductions, a per-group evaluation wrapper returning doubled arguments, and a per-charge power in Gear2 error control.
- **Result** (`495b713`, `204cbf9`): buck time fell to 65.9 ms versus 64.5 ms pre-Phase 6 on machine B (+2.1%, within the 5% budget, bit-identical).
    The remaining 0.66 ms in step control belongs to Gear2 error control (§5).

### 📋 5. A MOSFET's evaluation cheaper still

- **What:** after `52e3bdb` and `4f39ff4`, MOSFET evaluation still costs ~200 ns per instance.
    Two known costs remain: 51 ns of group overhead (zeroing blocks and gathering 83 values) and 18 ns in junction-charge `powf`.
- **Done when:** before-and-after timings are recorded with results agreeing within rounding.
- **Source:** row S2.

### ✅ 6. Stamp linear devices once per run

- **What:** resistors, capacitors, inductors, fixed source stamps, and MOSFET overlap capacitances produce constant matrix entries across Newton iterations.
    Pre-summing constant entries into a base matrix lets each iteration add only nonlinear terms, computing linear residuals and charges by multiplying the base matrix against unknowns.
- **Why:** evaluation consumes 19 to 23% of `rc_mesh_100` runtime, mostly linear devices, whereas ngspice reloads all devices every iteration (`cktload.c` lines 62 to 75).
- **Done when:** mesh evaluation share drops measurably with rounding-level accuracy and bit identity across thread counts.
- **Result** (`4e3499a`): kernels declaring `Kernel::linear` evaluate once per run into a base matrix reused each iteration.
    `rc_mesh_100` sped up 4.2% (evaluation share down from 16.3% to 10.9%) and the buck sped up 3.4%, shifting results by rounding only (3.7e-15 V on the mesh) across thread counts.
- **Left:** constant entries inside nonlinear devices (source ±1, fixed series resistances, MOSFET overlaps) that share slots with varying terms.
- **Source:** row S3; unarchived brief.

### 🚧 7. The fixed passes of an iteration cheaper on small systems

- **What:** on circuits with a few dozen unknowns, fixed overhead dominates three passes: two-stage slab gather, the `|J|·|x|` convergence product, and per-step error-filter refactorization.
- **Done when:** buck iteration cost drops to at most 1.5× ngspice without sacrificing accuracy.
- **Result, in part** (`3f53643`, `9277c9b`, 16 November): tightened the refactorization loop and eliminated allocations in the error filter with identical bits.
    Buck runtime dropped from 848 to 825 ms (-2.8%), `bjt_amp` and `ac_wide` stayed flat, and the 500-inverter chain rose +0.7% from general condensation bookkeeping (3,000 eliminated unknowns per factor).
    Per buck iteration, jolty takes 2,661 to 2,697 ns versus 1,036 ns for ngspice.
    Phase breakdown (jolty vs ngspice): evaluation 1,074 to 1,101 vs 433 ns, factoring 449 to 467 vs 95 ns, and solving 351 to 362 vs 86 ns.
    jolty averages 1.35 factorizations per iteration (one extra error-filter refactor per judged step) and runs 331,020 iterations versus 196,438 for ngspice.
    Iteration cost stays at 2.4× ngspice rather than 1.5×, leaving evaluation (item 5) and error-filter factorization to address.
- **Source:** row S4.

### 🚧 8. An agent's command timed from start to finish, then cut

- **What:** agents wait on `jolty simulate`, which launches a fresh Python process, imports jolty, builds and elaborates the circuit, and runs analyses.
    Measure end-to-end latency through both CLI and MCP, then cut overhead with a warm runner between commands while keeping each script in a separate process.
- **Why:** on machine D, `bjt_amp` takes 1.24 ms in the engine and 3.8 ms in `ngspice -b`, but agents experience full CLI wall time.
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

    Top cumulative import times from `python -X importtime -c "import jolty.cli"` totaled 371.8 ms (`jolty` 322.1 ms, `jolty._core` 0.018 ms).
    Major import contributors were `jolty.files` (209.0 ms: `numpy` 81.8 ms, `jolty.plan` 72.8 ms, `jolty.spice` 60.0 ms, `jolty.spice.download` 43.3 ms) and `jolty.devices` (80.9 ms).
    Remaining contributors were `jolty.service` (41.1 ms: `jolty.settings` 30.6 ms via `pydantic_settings` 28.8 ms), `pydantic` (23.4 ms), `jolty.results` (23.1 ms), and `jolty.ir` (19.4 ms).
- **Result, in part** (20 November, machine D, AMD EPYC 9B14, host load 60.8 to 78.2 over 96 cores): adopted PEP 810 lazy imports across `jolty/__init__.py`, `jolty.cli`, `jolty.files`, and `jolty.service`, plus a prewarmed standby `jolty.runner` worker for `service` calls.

    Updated machine D timings (median and IQR over 7 cold subprocess runs; median over 5 warm `service.simulate` runs):

    | Command / call | 18 Nov | 20 Nov | Speedup / notes |
    | --- | ---: | ---: | --- |
    | `python -c pass` | 13.5 ms | 13.1 ms | IQR 11.8 to 14.4 ms (bare interpreter) |
    | `import jolty._core` | 447.6 ms | 13.8 ms | 32× faster (IQR 12.9 to 16.5 ms) |
    | `import jolty` | 440.0 ms | 13.2 ms | 33× faster (IQR 12.5 to 13.9 ms; 0.20 ms in `-X importtime`) |
    | `import jolty.cli` | 522.4 ms | 15.5 ms | 34× faster (IQR 14.3 to 18.1 ms; 1.46 ms in `-X importtime`) |
    | `import jolty.plots` | 791.5 ms | 497.2 ms | 1.6× faster (IQR 494.8 to 502.1 ms) |
    | `jolty simulate bjt_amp.py --no-schematic --no-plot` | 951.7 ms | 687.4 ms | -28% (IQR 684.7 to 688.0 ms) |
    | `jolty simulate bjt_amp.py --no-check --no-schematic --no-plot` | 988.5 ms | 666.8 ms | -33% (IQR 663.7 to 669.6 ms) |
    | `jolty simulate bjt_amp.py --no-plot` | 1,336.9 ms | 987.0 ms | -26% (IQR 875.4 to 994.2 ms) |
    | `jolty simulate bjt_amp.py` (default: schematic + plots) | 2,101.0 ms | 1,322.4 ms | -37% (IQR 1,305.7 to 1,473.1 ms) |
    | Warm `service.simulate` (`schematic=False, plot=False`) | 504.7 ms | 149.9 ms | 3.4× faster (139.4 ms on standby hit) |
    | Warm `service.simulate` (`schematic=True, plot=False`) | 631.7 ms | 266.9 ms | 2.4× faster (257.5 to 367.1 ms) |
    | Warm `service.simulate` (`schematic=True, plot=True`) | 1,294.2 ms | 647.7 ms | 2.0× faster (642.1 to 681.5 ms) |

- **Not done:** persistent CLI runner or lightweight CLI launcher to cut remaining cold subprocess overhead, and faster or non-matplotlib plotting to approach `ngspice -b`.
- **Done when:** the speed suite reports `jolty simulate` wall time on `bjt_amp`, with and without images, below `ngspice -b`.
- **Source:** row S5; `research/agentic.md` row 6; `research/benchmarks.md` row B15.
    Item 8 replaces the duplicate entries previously in `agentic.md` and `benchmarks.md`.

### 📋 9. Exact latency: skip a device whose inputs have not changed

- **What:** each kernel is a pure function of parameters, memory, and local unknowns, so an instance whose inputs match the previous evaluation bit for bit can be skipped.
    Gather then adds only deltas from re-evaluated instances.
    Unlike approximate tolerance-based SPICE bypass, bit-identical skipping is exact.
- **Why:** evaluation takes 69% of serial runtime on the 50,000-inverter chain, and large mostly quiet circuits repeat most evaluations (up to an estimated 3.5× at 99% repeats).
- **Done when:** a probe measures bit-identical repeat rates on the chain, and skipping matches full evaluation bit for bit across all fixtures and thread counts if profitable.
- **Source:** row L2.

### 📋 10. Memory at ten million devices

- **What:** large runs record only requested signals and stream to disk by default, interning or lazily formatting unknown names instead of heap-allocating per unknown.
    The remembered operating-point solver copy becomes shared or optional, and the slab moves to per-thread tiles without a 32-bit index cap.
- **Why:** recording the 50,000-inverter chain at ~800 points takes 2.1 GiB, with ~1.4 KB of state per MOSFET.
- **Done when:** a million-MOSFET chain runs on this 32 GB Mac with peak bytes per device reported.
- **Source:** row L8; blueprint §3.2 and §3.4.

### 📋 11. The front end at a million to ten million devices

- **What:** Python circuit construction, JSON serialization, and elaboration remain untimed above 10⁴ devices.
    Profile at 10⁵ and 10⁶ devices first, then address bottlenecks in order: a Rust builder with interned names, bypassing JSON, elaborating each subcircuit definition once, and removing linear scans in `Groups::add`, `Recorded::of`, and checks.
- **Done when:** 10⁶ devices build and elaborate with full timings and profiles, and 10⁷ devices fit in memory.
- **Source:** row L9; blueprint §1 and §3.5.
    Concise authoring of large circuits is tracked in `agentic.md` 20.

### 📋 12. Orderings for chip-like circuits

- **What:** nested dissection stops below three levels, and block triangular form removes only ideal supplies.
    Resistive supply grids touching every cell and crossbar topologies defeat both techniques.
    Required additions are dedicated hub-net handling, multilevel separators, and a benchmark fixture modeling a resistive supply grid feeding logic.
- **Done when:** the supply-grid fixture dissects into interiors and scales across cores.
- **Source:** row L5; blueprint §2.1.

### 📋 13. Iterative solves preconditioned by the last factors

- **What:** runs dominated by refactorization (78% of `rc_mesh_300` runtime) can solve most iterations via a Krylov method preconditioned by previous factors, stopping at `|J|·|x|` resolution with fixed summation order.
- **Done when:** the mesh transient runs faster at equal accuracy with bit-identical results across thread counts.
- **Lead.**
    **Source:** row L6; blueprint §2.4.

### 📋 14. Dense kernels for the separators' fronts

- **What:** top-level dense separator fronts currently use scalar loops; register-blocked kernels vectorized across entries with fixed summation order would accelerate elimination with identical bits across widths.
- **Done when:** separator fronts eliminate faster with bit-identical results across thread counts and SIMD widths.
- **Source:** row L7; blueprint §2.4.

### 📋 15. Samples in vector lanes

- **What:** evaluate multiple batch samples concurrently in SIMD lanes using a structure-of-arrays sample layout, with each lane advancing independently and reproducing single-sample bits on the same host.
- **Role:** serves as the CPU AVX-512 precursor to batched H100 GPU execution (`gpu.md`), validating the exact memory layout required for batched GPU ensembles while delivering immediate multi-lane CPU throughput.
- **Needs:** cross-platform bit identity (`accuracy.md` 13, parked) only if bit identity is required across platforms as well as SIMD widths.
- **Done when:** k SIMD lanes reproduce k single runs bit for bit, with throughput measured.
- **Lead.**
    **Source:** row M2; blueprint §3.3; owner prioritization, 20 November.

### 📋 16. An exact cache of results

- **What:** key simulation results by a digest of all inputs so repeated runs return cached bits directly.
    jolty's strict determinism makes result caching exact.
- **Done when:** repeated runs return cached bits and any modified input triggers a cache miss.
- **Lead.**
    **Source:** row M5.

### 📋 18. Approximations as knobs that report measured error

- **What:** expose FastSPICE methods (bypass, multirate integration, partitioned relaxation) strictly as opt-in knobs that report measured equation defect across the whole circuit.
- **Why:** FastSPICE tools claim 10 to 1000× speedups on large quiet circuits at 10 to 15% error.
- **Done when:** every opt-in approximation run reports measured defect.
- **Lead.**
    **Source:** row L3; blueprint §2.3; `AGENTS.md`, "Exact before approximate".

### ⏸️ 19. Krylov pole-zero and matrix-free shooting

- **What:** pole-zero and periodic steady-state analyses currently use dense matrices (0.77 s at 500 unknowns).
- **Parked until:** target circuits outgrow dense matrix limits.
- **Source:** legacy roadmap item 14.

### ⏸️ 20. Meyer's capacitances integrated as ngspice does, an option

- **Parked** by the owner (15 November): condensation (item 3) already matches ngspice's matrix size with jolty's equations, and an ngspice Meyer option would render transient sensitivities and periodic steady state inexact (§5).

### 📋 21. The schematic and the 3D viewer at a million instances

- **What:** `jolty draw` and `jolty view` currently emit every instance individually; million-instance scale requires instancing one drawing per component definition.
- **Done when:** a 10⁵-instance array draws and views within a defined time budget.
- **Source:** blueprint §3.6.

### 🚧 22. Fast cached PDK subcircuit translation and multi-corner reuse

- **Priority:** P1 (Wave 2).
- **What:** cache parsed subcircuit definitions, pre-indexed MOSFET bin tables, and deterministic global parameter plans on disk per `(kit_digest, section)` keyed by kit digest and jolty version.
    Separate temperature sweeps from structural translation so sweeping `Simulation` temperature does not invalidate `Library._built` once `models.md` 24 moves resistor `TC1`/`TC2` into the engine (for devices without static `temper` `.param` references).
    Move hot `Library.component` and `Library.drawn` bin selection and `.param` evaluation into Rust via `jolty-expressions` for sub-second SKY130 Monte Carlo builds.
- **Why:** `kits.read` on SKY130 previously took 2.0 to 3.1 s per corner (12 to 18 s for six corners `tt`, `ff`, `ss`, `sf`, `fs`, `tt_mm`), changing `Library.temperature` cleared `self._built`, and building a 25-device SKY130 circuit across corners and Monte Carlo samples spent over 95% of wall time in Python AST and parameter translation while each Rust operating point took ~3 ms.
- **Needs:** `models.md` 24 (roadmap N2) to decouple temperature sweeps from structural translation (done).
- **Result, in part** (20 November, machine D, AMD EPYC 9B14, host load 46.0 to 56.6 over 96 cores):
    - Added persistent zlib-compressed typed JSON disk caching (`jolty.spice._cache`, `LibrarySnapshot`) keyed by source manifest digests, parser fingerprint, and compatibility mode, plus temperature-decoupled `Library._built` caching (`Library.temperature` preserves `temper`-free subcircuits, including `behavioral_resistor` elements with `TC1`/`TC2`).
    - Cold-process loading of all six SKY130 corners (`tt`, `ff`, `ss`, `sf`, `fs`, `tt_mm`) from a warm disk cache measured **0.592 s** median across 5 runs (0.591 to 0.595 s) versus **4.349 s** cold parse on machine D (7.3× faster, and ~25× faster than the ~15 s baseline), with circuit IR bit-identical to uncached translation.
    - Native parallel PDK draws in `MonteCarlo` (`models.md` 23b) run 100 SKY130 `tt_mm` mismatch draws and operating points across 25 transistors (`13 nfet_01v8 + 12 pfet_01v8`) in **42.0 ms** median (37.5 to 52.2 ms, 12× under the 0.5 s budget) versus **6.827 s** to build 100 × 25 devices via Python `Library.drawn` alone.
- **Left:** moving disk-cache decoding and hot `Library.component` bin/parameter evaluation into Rust to bring warm six-corner cold-process loading under 0.5 s.
- **Done when:** cold-process loading of all six SKY130 corners from a warm cache drops from ~15 s to under 0.5 s, and building 100 SKY130 `tt_mm` Monte Carlo draws of a 25-device bandgap takes under 0.5 s with circuit IR bit-identical to uncached translation.
- **Source:** owner approval, 20 November (roadmap N5).

**Elsewhere:** stamping BSIM3, VDMOS, and VBIC via summed derivatives is `quality.md` 5 (with speed measurements there).
Transient adjoint sensitivities are `agentic.md` 13, and built-in correctly rounded elementary functions for cross-platform bit identity are `accuracy.md` 13.

**Blueprint section index:** front end, tensor buses, and linear scans (§1, §3.5) map to item 11 and `agentic.md` 20; orderings (§2.1) to item 12; condensation (§2.2) to items 3, 3b, 3c, and 3d.
FastSPICE methods (§2.3) map to item 18; multifrontal and Krylov solvers (§2.4) to items 13 and 14; cold start and homotopies (§3.1) to `accuracy.md` 8 and 9; slab memory (§3.2) to item 10.
Vector lanes and bypass (§3.3) map to items 15 and 9, and tensor stimuli plus selective recording (§3.4) map to item 10 and `agentic.md` 20.
Schematic and viewer instancing (§3.6) maps to item 21, and differentiable simulation (§4) maps to `agentic.md` 12, 13, and 19.

## 5. Decided

- 15 Nov: Gear2 error control slowed the trapezoidal default by 0.66 ms on the 0.2 ms buck transient (step control 8.10 vs 7.44 ms on machine B, 1% of runtime).
    The overhead reflects per-charge error bounds and multistep unit steps, stays within item 4b's 5% budget, and is kept as Gear2's inherent cost.
- 15 Nov (superseded 16 Nov by item 3c, `a73e889`, adding off-diagonal pivots for small series resistances): initial condensation eliminated only MOSFET displacements because only those pivots provably passed the threshold.
    Small series resistances passed only while transistor conductance stayed below 11.1 S at 0.01 Ω, which conducting buck switches exceeded (estimated ~33 S).
    In BJTs, the base-node column carries transconductance (0.39 S versus a 0.014 S pivot at 10 mA), and heated MOSFET displacements couple into unbounded thermal rows.
- 15 Nov: elimination lives inside the sparse solver rather than device groups so every consumer reads the full system via one mechanism; group-level elimination is tracked in item 3b.
- 15 Nov: an ngspice-style Meyer option waits behind condensation because condensation already matches ngspice's matrix size.
    Matching ngspice Meyer integration would also require kernels to track accepted time points and would make transient sensitivities and periodic steady state inexact (archived note).
- Phase 6 solves sum rows above 1,024 entries in chunks via `reduce::fold` for bit identity across thread counts.
    Gather and `|J|·|x|` keep sequential order because chunking measured slower or flat.
- Meyer capacitances do not conserve charge in either engine because no charge function exists; VDMOS gate charge does conserve.
    jolty displacement unknowns give `C(v)·dv/dt` exact derivatives and apply error control to linear oxide charge, whereas ngspice averages capacitance over the step.
- Differentiating MOSFET junctions and overlaps along individual voltages sped up evaluation by 4 to 12%, shifting waveforms by at most 3.6e-14 V on the chain and 2.6 µV on the buck.
- `#[inline(always)]` applies to register-summed derivative stamping and above-block solve rows, each backed by measurements.
- Phase 6 uses nested dissection after independent-block and level-scheduling schemes measured only 1.0 to 1.4×.
- Pivot threshold stays at 0.1 rather than KLU's 0.001 to preserve accuracy without row scaling.
- Remembered operating points retain a solver copy so later analyses remain bit-identical regardless of intervening runs (`rc_mesh_300` peak memory +26% serially, +42% dissected).
- One scheduler: a single caller-owned rayon pool runs all parallelism levels to prevent oversubscription, and device kernels have no dependency on threads (archived note).
- mimalloc serves as the Python extension allocator, keeping Monte Carlo at 95.9% of linear scaling on 8 threads versus 64 to 68% previously.
- Speed claims follow `benchmarks.md`: references at best configuration, both engine and end-to-end timings, and efficiency relative to attainable throughput.
- 20 November (owner): adopted cached PDK translation and Rust-side bin/expression evaluation (item 22, roadmap N5) and staged SIMD sample lanes (item 15) as the CPU AVX-512 precursor to batched H100 GPU execution (`gpu.md`).

## 6. Open principles

None: proposals from the research report were merged into `AGENTS.md` on 15 November (`444535f`).
