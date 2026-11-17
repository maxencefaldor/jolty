# Research: speed

The evidence for `speed.md`: how jolty becomes the fastest simulator for one run of a small circuit, one run of a very large one (ten million devices and more), and many runs of one circuit.
Written on 15 November 2026 from `main` (the repository), the notes, ngspice 47's and Xyce's sources, and published work.
`main` moved from `68c3f75` to `4f39ff4` while this was written: two MOSFET commits, `52e3bdb` and `4f39ff4`, landed; §2.1 cites them, and line numbers are `4f39ff4`'s.
Read only: this report measures nothing; every number is a measurement already recorded, a count read from code, or an estimate marked as one.
Three subagents are working on Phase 6's 6×, the MOSFET's evaluation and the benchmarks of parallel runs; their topics appear here as strategy only.

## 1. Summary

1. jolty leads on many runs (setup from a shared circuit 2.5 µs against 63 µs from JSON; Monte Carlo 8.9× on 12 threads, about 96% of what this Mac's mixed cores allow, an estimate) and on meshes against ngspice's default solver (17–40×).
1. It is 2.4–3.4× slower than ngspice on `buck` and `inverter_chain_500`, and on 12 threads it only matches ngspice's one core on `inverter_chain_5000` (14.3 s against 13.5 s).
1. The gap is the cost of each Newton iteration, not their number (jolty takes 28% fewer on the chains): its linear algebra alone costs ngspice's whole iteration (109 µs against 110 µs on the chain), and its evaluation alone twice that, on a system 1.8–6.9× larger.
1. Move 1, exact and cheaper iterations: condense each device's internal unknowns inside its group (3,504 to 504 unknowns on the 500 chain), which keeps decision 11's charge-conserving formulation and every error check; with the MOSFET work in flight it should put the chain under ngspice (estimate).
1. Move 2, exact latency for huge circuits: skip, bit for bit, the devices whose inputs repeat exactly (the Kernels convention makes this memoization provably exact), and gather only what they touch; bypass, multirate and FastSPICE partitioning stay opt-in knobs that report their measured error.
1. Move 3, own the arithmetic: jolty's own elementary functions and vector kernels whose lanes compute alike at any width, for SIMD over instances and samples, deterministic dense fronts, GPU batches, and the same bits on every platform, which no reference offers.
1. The first walls at ten million devices are memory and recording, not speed: every unknown recorded at every point (2.1 GiB for the 50,000 chain), a `String` per unknown, about 1.4 KB of group state per MOSFET (estimate), and no fixture beyond 350,004 unknowns.
1. Two claims rest on unequal comparisons: ngspice ran its default Sparse 1.3, not its optional KLU, and nobody has measured an agent's end-to-end command, which starts a fresh Python process each time.

## 2. Where jolty stands

### 2.1 One run of a small circuit

Against ngspice 47 on one core (raw logs, not kept, a handoff note, not kept, "Against ngspice on one core"):

| Fixture | jolty | ngspice 47 | Newton iterations, jolty / ngspice | Per iteration, jolty / ngspice |
| --- | --- | --- | --- | --- |
| `buck` | 0.71 s | 0.207 s | 299,616 / 196,434 | 2.4 / 1.05 µs |
| `inverter_chain_500` | 2.70 s | 1.12 s | 7,335 / 10,181 | 368 / 110 µs |
| `inverter_chain_5000` | 28.2 s; 14.3 s on 12 threads | 13.5 s | | |
| `bjt_amp` (Phase 0 baseline) | 0.81 ms in Rust, 1.58 ms in Python | 17.5 ms, its start-up included | | |
| `ac_wide` (Phase 0 baseline) | 10.6 ms in Rust | 30.9 ms | | |

Where a transient's time goes on `main` at `6fe8824` (raw logs, not kept, one profiled run each):

| Transient | Time | Evaluating | Assembling | Factoring | Solving | Step control | Other | Work |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bjt_amp` | 495.8 µs | 36% | 6% | 13% | 9% | 10% | 23% (and 2% recording, 1% preparing) | 108 iterations, 161 refactorizations |
| `buck` | 773.5 ms | 42% | 8% | 21% | 13% | 9% | 7% | 299,616 iterations, 380,962 refactorizations |
| `inverter_chain_500` | 2.662 s | 65% | 8% | 13% | 9% | 3% | 1% | 7,335 iterations, 8,049 refactorizations |

Read per iteration, from these shares and counts:

- **`inverter_chain_500`:** evaluation 236 µs, about 240 ns a MOSFET for 1,000 MOSFETs; assembly, factoring and solving together 109 µs, which equals ngspice's whole iteration, 110 µs, about 110 ns a MOSFET.
- **`buck`:** evaluation 1.08 µs and linear algebra 1.08 µs, each about ngspice's whole iteration, 1.05 µs.
- **`bjt_amp`:** 23% of the transient is in no activity the profile names; unexplained.

**The MOSFET's evaluation.**
The kernel differentiates by forward-mode automatic differentiation with respect to its three controlling voltages, `vgs`, `vds` and `vbs`, not its seven local unknowns: `Dual<R, 3>` (`crates/jolty-models/src/mosfet.rs:538–549`, `893–894`), as BSIM3's does (`bsim3.rs:69`).
`brief-mosfet-cost.md` says every operation carries about eight numbers over seven unknowns; the code carries four, a value and three derivatives.
The derivatives are then spread onto the stamped nodes (`kernel/parts.rs:463–574`), and that spreading, not the width of the differentiation, was the largest cost.
`52e3bdb`'s message:

- before it, a MOSFET made 132 bounds-checked updates of its slab block an evaluation;
- sampled on `inverter_chain_500`, those updates took 40% of the evaluation, the group's add of the block 17%, zeroing the slab 3% and `pow` 5%.

The two commits that landed during this research:

- **`52e3bdb`:** sums the derivatives in registers and writes each of the 32 entries once: 242 to 221 ns a MOSFET an iteration, at load 9, with the same bits.
- **`4f39ff4`:** differentiates each junction and overlap along its own voltage alone: 208.3 to 199.6 ns a transistor in isolation (226.0 before either), at load 15, with bits moving within 3.6e-14 V on the chain.

After them a transistor costs about 200 ns, still about twice ngspice's whole iteration per MOSFET; the MOSFET-cost subagent continues.
The block, 83 numbers, is still added to the circuit's arrays through `u32` maps (`group.rs:1783–1812`).

**The Meyer displacement unknowns.**
Each MOSFET with an oxide adds three unknowns of its own (`mosfet.rs:685–745`, `docs/guide/devices.md:443`), so the 500 chain solves 3,504 unknowns where ngspice solves about 505 (`perf/README.md:39`).
Each displacement's equation is linear, `w − Cox·d(vg − vn)/dt = 0`, its own diagonal is 1 and every other entry of its column is at most 1 in magnitude, and it touches only its own transistor's gate and node (`mosfet.rs:713–745`).
The buck's MOSFETs also carry a drain and a source resistance of 0.01 Ω, each below `SMALL_RESISTANCE` (`kernel/parts.rs:18`) and so an internal node and a current unknown each: jolty solves 23 unknowns there against ngspice's 13 (my count from `perf/circuits/buck.json` and `mos1set.c:141–184`).

**Fixed costs of an iteration and a step.**
Each Newton iteration (`crates/jolty-engine/src/newton.rs:324–382`):

- evaluates every device into its slab block and adds the block in instance order (`group.rs:1724–1777`);
- writes `G + αC` into the Jacobian (`step_matrix`, `newton.rs:148–161`);
- in one pass over the rows (`combine`, `newton.rs:420–516`), adds each row's charge term, computes the resolution `|J|·|x|` (one more sparse product per iteration) and judges each residual against tolerance;
- factors or refactors, then solves;
- moves `x` and judges the step (`take_step`, `newton.rs:518–531`).

ngspice's iteration zeroes its matrix and right-hand side, lets each device type add straight into the matrix through cached pointers, refactors with its last pivots, solves and tests convergence (`cktload.c:62–75`, `niiter.c:113–156`, `254`, `288`).
jolty keeps `G` and `C` apart because its step control filters each error estimate through `C (G + αC)⁻¹ α` (`docs/guide/simulation.md:361`).
That filter refactors at each judged step: the buck makes 380,962 refactorizations for 299,616 iterations.
A rejected alternative, reusing Newton's last factors, saved 7% but filters poorly at switching edges (handoff, "The ammeter test's slowdown").
These are costs of accuracy jolty chose, and they weigh most on small systems.

**End to end.**
In-process, jolty's engine beats ngspice's whole batch run on short analyses (`bjt_amp` 0.81 ms against 17.5 ms, ngspice's start-up and parsing included, `perf/README.md`).
An agent's command, though, starts `python -P -m jolty.runner DIRECTORY` afresh (`src/jolty/runner.py:3–4`), imports jolty (`import jolty.cli` alone took 0.31 s after decision 8) and draws plots (AGENTS.md, "Images are feedback").
That command's wall time is unmeasured; it is likely hundreds of times the engine's run on a small circuit (estimate).

### 2.2 One run of a large circuit, and the many-cores plan

The plan is implemented (an implementation plan, not kept, Status; the session log, not kept):

| Target | Measured | Met |
| --- | --- | --- |
| `inverter_chain_50000` 6× on 16 cores | 3.7× on 12 threads, the chain at 4 ns: 36.7 s serially, 10.0 s dissected (raw logs, not kept) | No |
| `rc_mesh_300` 4× | 7.0–7.7× against serial, 3.3–3.6× against dissected on one thread | One reading |
| Monte Carlo at 90% of linear on every core | 74% on 12 threads | No |
| `bjt_amp` 1.5× per core | 1.53× | Yes |
| `inverter_chain_500` at 0.4 ms per Newton iteration | 0.36 ms | Yes |
| Same bits on any thread count and shard split | Tested | Yes |

What still bounds the chain on 12 threads (a brief, not kept):

- **The evaluation, 5–6 s:** the device groups take 2.3 s; the gathers take 2.6–3.2 s, memory-bound, 2.3 million scattered sources into 1.55 million entries and 350,000 rows.
- **The solves' tree passes, 0.73 s:** at most 3.2×, as each subtree's thread share is fixed.
- **The assembly, about 1.5 s:** `|J|·|x|` takes about 0.9 ms a call and stops improving beyond 4 threads.
- **The machine:** its four efficiency cores, and the pool's workers sleeping between short parallel regions.

Serially, evaluation is 28.6 s of 42.0 s at 4 ns (handoff, "What bounds the chain").
The dissected ordering is not free on one thread: it is 2.1× faster than the serial ordering on the mesh (18.2–18.9 s against 8.65–8.73 s) but 4% slower on the chain (36.7 s against 38.1 s).
On meshes the factorization rules: 62% of `rc_mesh_100`'s transient and 78% of `rc_mesh_300`'s.
A refactorization there costs at least about 50 solves on the larger mesh and 22 on the smaller (an estimate from the profiles' shares and counts), against about 1.3 on the 500 chain.
No threshold of `Parallelism` is set by default (`options.rs:47–64`, decision 21), so every run is serial unless its options say otherwise.

### 2.3 Many runs of one circuit

- **Compile once (Phase 2):** an `Elaborated` circuit is shared and each run patches `Values` into a reused `State`.
    Setup from it takes 2.5 µs against 63 µs from JSON on `bjt_amp`, 25×, bit-identical over 46,110 arrays (handoff, "Phase 2").
- **Batches (Phase 3):** samples are drawn from their index, each run runs on one thread when there is a run for every thread, and each finds the bits a single run finds (`docs/guide/simulation.md:274–297`).
- **Monte Carlo:** through Python, 531.1 µs a sample on one core and 59.89 µs on every core, 8.9× on 12 threads (74%), against Phase 0's 932.2 and 401.5 µs (`final-bench-2026-11-15.txt`).
    Phase 3b measured 95.9% on 8 threads.
- **Shards (Phase 4):** `jolty merge` is bit-identical to one run, and two workers ran at 99–105% of one machine.
- **Frequencies:** AC, noise and stability spread their frequencies over threads; `ac_wide` runs in 9.8 ms against ngspice's 30.9 ms.
- **The mixed cores:** if the 8 performance cores keep their 95.9% at 12 threads, the 4 efficiency cores add about 1.2×, about 0.3 of a performance core each, so 12 threads can reach about 9.2× and the 8.9× measured is about 96% of that (an estimate from two measurements at different times and loads).

### 2.4 Memory per device

Counted from the code for a level-1 MOSFET with an oxide and no series resistance; estimates, since nothing measured them:

- **The slab:** 7 local unknowns and 31 local entries, so 3 × 7 + 2 × 31 = 83 numbers, 664 B (`group.rs:745–760`, `mosfet.rs:849–881`).
- **The rest of the group's state:** about 550 B of prepared numbers, mostly the two junctions' depletion constants (`mosfet.rs:131–149`, `junction.rs:288–299`), 88 B of memory (`mosfet.rs:153–163`) and 152 B of `u32` maps: about 1.4 KB in the groups.
- **The arrays per unknown and per stored entry:** solution, residual, scale, charge, resolution, step and history; conductance, capacitance, Jacobian and LU factors.
    Probably 1–2 KB more per MOSFET (estimate).
- **A name per unknown:** every unknown carries a heap `String` (`crates/jolty-models/src/binding.rs:61–66`).
- **Recording:** a transient records every unknown at every point, 8 B each (`options.rs:257–263`): the 50,000 chain's results hold 2.1 GiB at about 800 points, more than the default 2 GiB (`perf/README.md:40`).
    A recorded MOSFET adds 111 numbers a point (`docs/guide/simulation.md:338`).
    Batches already record only what they trace.
- **Measured:** `rc_mesh_300`, 90,001 unknowns, peaks at 742 MB serially, and at 934 MB since the remembered operating point keeps a copy of the solver (decision 28): about 8–10 KB per unknown, of unknown composition.
- **32-bit caps:** the slab's places, the unknowns and the stored entries must each fit a `u32` (`group.rs:624–638`), about 1.4 × 10⁸ such MOSFETs at 31 places each.
    That is beyond ten million but a wall at a hundred million.

### 2.5 Beyond 350,004 unknowns

Nothing is measured above the 50,000 chain and the 300 mesh.
an archived note, not kept names the walls at ten million devices; §3.11 weighs it.

## 3. The references

### 3.1 The cost of a Newton iteration

- **ngspice 47:** hand-derived partial derivatives, some left out (handoff, VDMOS's review rows).
    The Meyer capacitances are integrated as states in the device, with no unknown: `DEVqmeyer`, the capacitance averaged over the two points, then `NIintegrate` (`mos1load.c:767–791`, `875–879`).
    Their companion conductances are stamped into the 4×4 block (`mos1load.c:923–935`).
    An iteration zeroes the matrix and lets devices add through cached pointers (`cktload.c:62–75`).
- **Where ngspice falls short:** its averaged capacitances do not conserve charge, and its MOSFET judges the truncation error of its gate charges alone (`MOS1trunc`, `docs/guide/simulation.md:361`).
    Its speed partly comes from computing and checking less.
- **LTspice:** its maker describes "a self-authoring, self-assembling and self-linking sparse matrix solver", assembly code generated at run time for the circuit's matrix (Engelhardt, "SPICE Differentiation", LT Journal of Analog Innovation 24(4)).
    No public figure of its gain was found.
- **What jolty could do that none does:** keep exact derivatives, conserved charge and every error check, and still bring the iteration's cost to ngspice's, by exact means: condensation (§3.3), linear parts stamped once, and stamps summed among a device's nodes before they are written (done for the MOSFET by `52e3bdb`).
  An estimate for the 500 chain:
    - **Evaluation:** a third of its 236 µs, if the MOSFET work in flight reaches 80 ns a MOSFET (about 200 ns after `4f39ff4`).
    - **Linear algebra:** roughly a quarter of its 109 µs after condensation, since the stored entries fall about fourfold (by the topology: about 17,000 to 4,000).
    - **Total:** about 130 µs an iteration, or about 0.95 s for its 7,335 iterations against ngspice's 1.12 s.
        Not measured.

    On the buck, the 1.5× extra iterations (accuracy theme, items 7 and 18) keep jolty behind even at ngspice's cost per iteration.

### 3.2 Many cores for one run

- **ngspice 47:** OpenMP over the instances of some models only, BSIM3, BSIM4, BSIM-SOI and HiSIM2 (`b4ld.c:62–90`, where the matrix is still loaded serially afterwards).
    Its matrix solvers are serial.
    A level-1 circuit therefore runs on one core.
- **Xyce:** distributes device evaluation over MPI processes and, from 10,000 unknowns in a parallel build, solves iteratively (§3.6).
    Its guide gives three modes by size: serial below 10³ unknowns, parallel load with a serial solve from 10³ to 10⁴, and parallel both above (`Xyce_UG_ch13.tex:184–207`).
- **Parallel LU for circuits:** NICSLU's level scheduling and pipelining (Chen, Wang and Yang 2013), which measured 1.0–1.4× on jolty's fixtures (handoff, "Phase 6 measured before it is built").
    Basker's hierarchical BTF and nested dissection (Booth et al. 2017).
    ShyLU's hybrid Schur complement (Rajamanickam, Boman and Heroux 2012).
- **Commercial, as their makers state:** Spectre X up to 10× faster and 5× the capacity of earlier Spectre, on up to 128 cores and several machines (Cadence, 3 June 2019), "tens of millions of active devices and billions of parasitics" (SemiWiki).
    Siemens AFS more than 5× faster than traditional SPICE and more than 2× faster than parallel SPICE, 300 million elements with eXTreme (Siemens fact sheet).
- **Where they fall short:** none promises the same result on any number of cores; Xyce's partition-dependent preconditioner makes its iterates depend on the process count (inferred from its additive Schwarz over subdomains; its guide states no reproducibility either way).
- **jolty:** the same bits on any thread count is unique among them, and Phase 6 made it work at 3.7× and 7.0–7.7×.

### 3.3 Static condensation of device-internal unknowns

- **Literature:** exact elimination of interior unknowns before the global solve is the static condensation of finite elements (Wilson 1974), the node tearing of circuit analysis (Sangiovanni-Vincentelli, Chen and Chua 1977), and what a multifrontal factorization does by algebra (Duff and Reid 1983), which jolty's `tree.rs` already implements with delayed pivots.
- **ngspice and Xyce:** neither condenses.
    ngspice's internal prime nodes are unknowns (`mos1set.c:141–184`), eliminated by the sparse LU like any other.
    ngspice needs no displacement unknowns, because it integrates Meyer's capacitances inside the device.
- **What jolty could do that none does:** condense in the group, after evaluation, the internal unknowns of each instance.
    - **What is eliminated:** each instance's local `G + αC` (or `G + jωC`) is formed and its internal rows are eliminated into its terminals' block; the internal values are recovered after the global solve.
    - **Which unknowns:** only those whose local pivot provably passes the solver's threshold.
        A displacement's does always (its diagonal is 1 and its column is otherwise at most `C/Cox ≤ 1`).
        A small series resistance's node and current condense as a 2×2 pair.
        An unknown that cannot condense stays global.
    - **Exact:** the iterates are exact in exact arithmetic, and their last bits change once, like `max_unknowns_per_interior`.
    - **Decision 11:** its formulation is untouched, since every displacement unknown keeps its equation, its value, its charge and its error check.
        The roadmap's question 11, "keeps its three displacement unknowns ...
        or condenses them", treated the two as alternatives; they are not.
    - **The condensed pattern:** each MOSFET's own 4×4 block, which is also ngspice's matrix.
        A conserving formulation at ngspice's matrix size is what no reference offers.
- **What it would save:** the 500 chain from 3,504 to 504 unknowns; the 50,000 chain from 350,004 to 50,004, whose gathers bound it on 12 threads; the buck from 23 to 9 (my counts).
- **Limits:** pole-zero, whose pencil must stay polynomial in `s`, keeps the full system.
    Sensitivities and noise need the condensed form's back-substitution.
    The step control's error filter solves through the condensed factors.
- **The blueprint's second level, condensing a leaf subcircuit's interior:** jolty's nested dissection already does that, from the graph rather than the hierarchy.
    Ordering by the hierarchy would let identical subcircuits share their symbolic work, a refinement of the ordering rather than a new solver.

### 3.4 Device bypass, and its accuracy cost

- **SPICE3's bypass:** skips a device whose voltages and predicted currents moved less than the Newton tolerances since its last evaluation.
    MOS1's test runs over `mos1load.c:277–343`, "now lets see if we can bypass (ugh)".
- **ngspice 47:** off by default (`cktntask.c:105`, `TSKbypass = 0`; `cktinit.c:54`), on with `.option bypass` (`cktsopt.c:115`).
    ngspice is therefore not faster through bypass (handoff).
- **Xyce:** removed it from its MOSFET level 1 ("I took out all the CAPBYPASS stuff", `N_DEV_MOSFET1.C:3099`).
- **Others:** Silvaco's SmartSpice turns it on in its high-performance mode (`bypass=2`, application note 1-029), and Altium's simulator lists `BYPASS` enabled by default.
- **The accuracy cost:** no reference found quantifies it.
    Its test admits changes up to the Newton tolerances, so every bypassed evaluation carries an error of that order, and convergence is judged on a stale linearization (reasoned from `mos1load.c:277–310`).
- **What jolty could do that none does: exact bypass.**
    - **Why it is exact:** AGENTS.md's Kernels convention makes a device a pure function of its prepared numbers, its memory and its local unknowns.
        When those, the analysis's `gmin`, source scale and limiting flag, and, for a kernel that reads it, the time are bit-identical to its last evaluation's, its block is bit-identical too and need not be computed.
    - **Assembly:** a target's sum is computed again, in its fixed order, only when one of its sources changed; an untouched target's last sum is bitwise what recomputing would give.
        The slab and the `GatherMap` that this needs exist.
    - **Why inputs can repeat:** jolty already zeroes a step whose residual lies within rounding (`newton.rs:493–507`), and an update smaller than half a unit in the last place leaves a node's bits unchanged.
    - **What it could save:** nodes far from activity may therefore repeat exactly.
        How many do is unknown, and a probe counting bit-identical device inputs on the 50,000 chain decides it.
        If 99% repeat, the chain's serial run could shrink by up to about 3.5×, bounded by factoring, solving and step control (an estimate from its profile: evaluation 69%, assembly 8%).
    - **The factorization:** it stays whole whenever `α` changes, which is every step size; within a step's Newton iterations a partial refactorization along the elimination tree is exact too (Chan and Brandwajn 1986).
    - **Condition:** each kernel must leave its memory unchanged when it is evaluated again at its last inputs, which a property test can check.

### 3.5 Multirate and partitioned methods, and their accuracy cost

- **Literature:**
    - Waveform relaxation (Lelarasmee, Ruehli and Sangiovanni-Vincentelli 1982), which converges for circuits with a capacitance to ground at every node but slowly on tightly coupled ones.
    - Iterated timing analysis (Saleh, Kleckner and Newton 1983).
    - Multirate formulas with error control (Gear and Wells 1984; Günther and Rentrop 1993; Günther, Kværnø and Rentrop 2001; Savcenco, Hundsdorfer and Verwer 2007).
- **FastSPICE's accuracy cost, as published:**
    - Relative error to golden SPICE "in the range of 10~15% or more" (Yang, Electronic Design, 2016), from table models, RC reduction, matrix partitioning and hierarchical simulation.
    - A Synopsys patent: "speed increases of 10×-1000× with acceptable lose [sic] of accuracy" (US 8,868,395, filed 2008).
    - Accuracy levels as a setting: PrimeSim's `runlvl` 3 to 6 (Synopsys, 10 June 2026).
- **Xyce:** has no general multirate transient; its multi-time PDE package is for RF envelopes.
- **ngspice:** has none.
- **jolty:** decided not to pursue multirate "as they trade accuracy and reproducibility for speed" (decision 13, `efe815c`).
- **Reproducibility need not be lost:** a partition fixed by the circuit's structure and a sweep in a fixed order are deterministic.
- **The accuracy cost is real:** interpolated waveforms at partition boundaries.
- **What jolty could do that none does: an approximation that reports its own error.**
    - **The method:** run the partitioned or multirate method as an opt-in knob, then evaluate the defect of the whole circuit's equations, `f(x) + dq/dt`, along the solution's interpolant at sampled times, with every device exact.
    - **The result:** report that defect against the tolerances as a fact, as defect control does for ODE solvers (Enright 1989).
    - **Its cost:** device evaluations, with no factorization.
    - **Why it fits jolty:** this follows "Feedback is facts" and `vision.md`'s proposed answer, that approximations exist only as knobs a run opts into.

### 3.6 Iterative and preconditioned linear solvers

- **Xyce:**
    - KLU in serial builds; in parallel builds, GMRES from AztecOO for systems of 10,000 unknowns or more (`N_LAS_TranSolverFactory.C:96–107`, `N_LAS_fwd.h:72`).
    - Its defaults: tolerance 1e-9, 200 iterations, a Krylov space of 50, ILUT fill 2.0 and drop 1e-3 (`N_LAS_AztecOOSolver.C:117–135`), preconditioned by a non-overlapped additive Schwarz with an incomplete LU on each subdomain (`Xyce_UG_ch13.tex:443–447`).
    - It partitions by hypergraph in parallel (`N_LAS_TransformTool.C:114–115`), and offers ShyLU and Basker (`N_LAS_TranSolverFactory.C:143–148`).
    - Its guide: direct solvers are efficient "up into the tens to hundreds of thousands of unknowns" and "rarely fail", while iterative ones depend on preconditioning (`Xyce_UG_ch13.tex:176–182`).
        When an iterative solve fails, the nonlinear solver or time integrator retries (`:351`).
- **The research behind it:** block triangular form, singleton removal and hypergraph partitioning before a block Jacobi preconditioner (Thornquist, Keiter, Hoekstra, Day and Boman 2009).
- **For power grids:** SILCA preconditioned Krylov iterations with incremental and stale factorizations, keeping SPICE accuracy (Li and Shi 2006).
- **Where they fall short:** iterative solvers are fragile on circuit matrices, their stopping tolerance is a tuned constant, and their iterates depend on the partition count.
- **jolty:** has no iterative solver (roadmap 14, Krylov pole-zero, waits).
- **What jolty could do that none does: a deterministic Krylov solve.**
    - **Preconditioned by the last factors:** for runs bound by refactorization, where a refactorization costs at least about 50 solves on `rc_mesh_300`.
    - **Its stopping test:** the residual's floating-point resolution `|J|·|x|`, which jolty already computes every iteration (`newton.rs:487–492`), instead of a tuned 1e-9.
    - **Determinism:** its reductions in `reduce::sum`'s fixed order.
    - **The result:** a solve as accurate as the direct one, the same bits on any thread count, and no refactorization while it converges.
    - **Where it does not pay:** on the chains, where a refactorization costs about 1.3 solves.

### 3.7 Dense kernels and SIMD under the determinism rule

- **The rule:** nothing that reaches a result may pick its code path by CPU at run time, and builds never use `target-cpu=native` (`many-cores-implementation-plan.md:296`).
    `exp`, `ln` and `powf` come from the platform's libm, so bits match on one platform only (`:297`), and the derandomized random circuits do differ between macOS and Linux (handoff, "Still open").
- **jolty today:** the LU is plain Rust, and a separator's front takes in its earlier steps by scalar column updates (`crates/jolty-numerics/src/lu/tree.rs:942–953`, `1151–1199`).
    faer, which picks its SIMD path at run time, is ruled out for anything that reaches a result.
    Phase 6's prototype found that the dense trailing block of the meshes' separators caps the pipelined refactorization (handoff, "Phase 6 measured before it is built").
- **The references:** ngspice 47's sources contain no SIMD intrinsics and no `omp simd` (grep of `src/`).
    Xyce's "vectorized" parameters are vector-valued, not SIMD (`N_DEV_Pars.h:115`).
    LTspice generates its solver's code (§3.1).
- **The tools:** Rust's portable SIMD is still nightly-only (`std::simd`, tracking issue 86656).
    CORE-MATH provides correctly rounded binary64 `exp`, `log` and others under the MIT licence (Sibidanov, Zimmermann and Glondu 2022).
- **What makes SIMD deterministic:** IEEE addition, multiplication, division and square root are rounded exactly in each lane.
    A kernel whose every lane runs the same operations in the same order, with no implicit fused multiply-add (Rust never contracts on its own), gives the same bits at any vector width.
    The width then needs no ban; only the order of operations does.
    - **Device evaluation:** instances in lanes, branches as selects, which needs elementary functions identical in scalar and vector form.
    - **Dense fronts:** register-blocked updates in which each entry still subtracts its products in `k` order.
    - **Samples in lanes:** §3.10.
- **What jolty could do that none does:** own its elementary functions, correctly rounded and so platform-independent by definition.
    Its results would then be the same bits on every operating system and processor, scalar or vector, CPU or GPU.
    No reference offers that.

### 3.8 Memory per device

- **ngspice 47:** a MOS1 instance holds 58 `double`s, 25 matrix pointers, 22 KLU binding pointers and its flags (`mos1defs.h:32–298`), about 1 KB (estimate).
    It also holds 17 states (`mos1defs.h:323`) in each of the 4 state vectors the trapezoidal rule allocates (`cktsetup.c:190–192`), 544 B.
- **jolty:** about 1.4 KB of group state per MOSFET, plus its share of the global arrays (§2.4, estimates); the two are of one order.
- **The commercial capacity claims:** AFS 300 million elements; Spectre X billions of parasitics; a PrimeSim case of 200 million devices, more than 95% of them RC parasitics (Synopsys, 2026).
    At such sizes the bytes per element must stay well under a kilobyte.
    FastSPICE reaches that partly by reducing RC networks (Electronic Design 2016; e.g. PRIMA, Odabasioglu, Celik and Pileggi 1998, and TICER, Sheehan 1999).
- **jolty's walls at ten million devices:**
    - recording, about 280 MB a point at 35 million unknowns (estimate);
    - names, a `String` per unknown;
    - the remembered operating point's copy of the solver, +26 to +42% on the mesh;
    - the LU's fill, unknown for chips.

### 3.9 GPUs

- **Device evaluation:** maps well.
    BSIM3's evaluation, 75% of a FastSPICE's time, moved to a GPU gave 2.36× on average over the whole simulator (Gulati, Croix, Khatri and Shastry 2009).
- **Sparse LU on GPUs:** harder; GLU (He, Tan, Wang and Shi 2016), and Chen, Ren, Wang and Yang 2015.
- **Commercial:** a 200-million-device post-layout PLL ran its transient about 3× faster with GPUs than without, and 28× faster than the customer's previous simulator, on 32 CPUs and 4 GPUs (Synopsys blog, Ramadan, Pant and Guo, 10 June 2026).
    Synopsys and NVIDIA cite up to 15× on GH200 and project 30× on GB200 (March 2025 announcements).
- **This Mac:** its GPU cannot help, because Metal has no double-precision type (PyTorch's MPS notes); the owner's other machines are unknown (the session log, not kept, "Only the owner can give").
- **jolty's kernels are GPU-ready by convention:** pure, allocation-free, no panics (AGENTS.md, Kernels).
- **Its gather in a fixed order:** ports without atomics.
- **What jolty could do that none does:** a GPU batch whose every sample has the same bits as on the CPU, given jolty's own elementary functions and no fused operations.
    CUDA compilers fuse multiply-adds unless told not to.

### 3.10 Compile once, and many runs

- **ngspice 47:** runs Monte Carlo as a control-language loop whose `mc_source` "re-source[s] the actual circuit deck" each sample (`commands.c:371–374`, `inp.c:1641–1645`).
    Every sample parses and sets up again, on one core.
- **Xyce:**
    - `.SAMPLING` parses and sets up once (`sampling.tex:630–636`).
    - `.EMBEDDEDSAMPLING` solves every sample at once as one block system of identical blocks (`sampling.tex:34–39`, `638–643`), so the samples step in lockstep and no sample is the run it would be alone.
    - Xyce has transient adjoint sensitivities for many parameters and few objectives (`Xyce_UG_ch09.tex:1750–1758`).
        jolty has the adjoint at DC and the direct method over time (`sensitivity.rs:1–28`).
- **jolty leads here already:** compile once, samples drawn from their index, each run bit-identical to a single run, shards across machines.
- **What jolty could do that none does:**
    - **Samples in SIMD lanes:** each lane keeps its own steps and its own bits, where Xyce's embedded sampling makes them share; lanes that diverge, by rejecting a step, run masked.
    - **An exact result cache:** keyed by the inputs' hash, which the determinism contract already promises makes sense (an implementation plan, not kept, "Determinism contract").
    - **A transient adjoint:** for an optimizer's gradient.
    - **A warm runner:** for an agent's command (§2.1).

### 3.11 The ten-million-device blueprint, weighed

an archived note, not kept is a useful map of where scale breaks.
Its walls are real; its numbers are not measurements; several of its remedies break jolty's rules.

**Checked against the code, and true:**

- Each placement calls `inspect.currentframe()` and `Path.cwd()` (`src/jolty/builder.py:444–458`).
- Each unknown carries a `String` name (`binding.rs:61–66`).
- A transient records every unknown at every point (`options.rs:257–263`).
- The slab covers every instance, under 32-bit caps (`group.rs:624–638`).
- The dissection gives up below three levels (`dissection.rs:114`), which a hub net reaching most vertices causes.
- One pattern holds both conductive and capacitive entries, so the 500 chain is one block of 3,496 (handoff).

**Overstated or wrong:**

- "Mathematically identical Newton iterates" holds in exact arithmetic only.
- "3 sweeps recover full monolithic SPICE accuracy (< 10⁻⁶)" has no evidence, against the published FastSPICE errors of §3.5.
- "Matching Sandia Xyce vectorized models": Xyce has no SIMD device code (§3.7).
- faer's BLAS-3 and `pulp` pick their SIMD path at run time, which the determinism contract forbids.
- "< 16 MB scratch" ignores that a target's gather needs every source of it.
- Bypass is offered with no word on its accuracy, or on ngspice's and Xyce's defaults.
- Every cost figure in its first table (55–80 GB, 12–20 minutes, 3.5 GB of JSON, 10–50× faster DC) has no measurement behind it.
- Its workload, a memristor crossbar, is not a fixture.

**Missing from it:** the KLU comparison, end-to-end latency, the mixed cores, and whether each remedy keeps the same bits.

**Worth keeping:**

- condensation (§3.3);
- selective recording;
- names interned or formatted lazily;
- elaboration by template;
- a DC ordering of its own, which on CMOS logic would split the DC Jacobian into stages, a small gain where transients dominate;
- SPICE's junction initialization, which belongs to the accuracy and reliability theme;
- the adjoint (§3.10).

## 4. Gaps and opportunities

An item's letter is its workload: S one small run, L one large run, M many runs, G GPUs, R the references themselves.
"Lead" marks an item where jolty would go beyond every reference rather than catch up.

| # | Item | Impact on the goal | Evidence | Rough size | Done when | Lead |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Condense device-internal unknowns in the groups: displacements first, then small series resistances and the BJT's internal nodes, each only where its local pivot provably passes the threshold | Small and large: 3,504 → 504 unknowns on the 500 chain, 350,004 → 50,004 on the 50,000 chain, 23 → 9 on the buck; most of the 0.80 s of 2.66 s in assembly, factoring and solving, and the parallel gathers' targets | §2.1, §3.3; `perf/README.md:39`; `final-bench-2026-11-15.txt` | M | The chains factor 504 and 50,004 unknowns; per-iteration time drops by a measured share; results within rounding of today's; every unknown still recorded and judged | Lead: a conserving formulation at ngspice's matrix size |
| S2 | A MOSFET's evaluation, cheaper (subagent running; `52e3bdb` and `4f39ff4` landed) | 65% of the 500 chain's time; about 200 ns a MOSFET now, against 110 ns for ngspice's whole iteration | §2.1 | In flight | Its own brief's | |
| S3 | Linear parts stamped once: linear devices' and linear terms' conductances and capacitances summed once per run, each iteration adding only what varies | The meshes' evaluation (19% of `rc_mesh_100`); a MOSFET's overlaps, 15 ns of its 200 before `4f39ff4` though zero on the chain | `4f39ff4`'s message; `cktload.c:62–75` reloads everything | S–M | Profiles show the share fall; results within rounding | |
| S4 | The fixed passes of an iteration and a step on small systems: the slab's two-stage add, `|J|·|x|`, the error filter's refactorization per judged step | `buck`: 2.4 µs an iteration against 1.05 µs | §2.1; 380,962 refactorizations for 299,616 iterations | M | Each pass measured; the buck's iteration within 1.5× of ngspice's, no accuracy given up | |
| S5 | An agent's command end to end: measure it, then a warm runner, still one process per script | Small runs: the engine takes 0.81 ms on `bjt_amp` against ngspice's 17.5 ms, but a command's wall time is unmeasured | `runner.py:3–4`; decision 8 | S to measure, M to cut | The speed suite reports `jolty simulate`'s wall time on `bjt_amp`, below ngspice's batch run | Lead |
| L1 | Phase 6's 6× (subagent running): the gathers, the tree passes, the assembly | 3.7× now | §2.2 | In flight | Its own brief's | |
| L2 | Exact latency: skip devices whose inputs and memory repeat bit for bit; gather only the targets they touch, in the fixed order | Large quiet circuits: evaluation is 69% of the 50,000 chain serially; up to about 3.5× if 99% of evaluations repeat (estimate) | §3.4; AGENTS.md, Kernels | S for the probe, L to build | Probe: the share of bit-identical evaluations on the chain. Build: the same bits as full evaluation on every fixture and thread count | Lead |
| L3 | Approximations only as knobs that report their own error: bypass, multirate, partitioned relaxation, each followed by the defect of the whole circuit's equations | Very large, mostly quiet circuits: FastSPICE claims 10–1000× at 10–15% error | §3.4, §3.5 | L | The owner answers `vision.md`'s question; a knob whose results state their measured defect | Lead |
| L4 | Parallelism by default, from the measured break-evens; the one setting that changes the bits, dissection, chosen by the computed cost of the two orderings rather than by a threshold | Users get 3.7× and 7.0–7.7× without knobs; dissection is 2.1× faster on the mesh on one thread but 4% slower on the chain | §2.2; decision 21; `bench-par` running | S, after L1 and `bench-par` | A run spreads by default where it pays, with the same bits on any thread count | |
| L5 | Orderings for chip-like circuits: hub nets such as a resistive supply, multilevel separators; a fixture with a supply grid feeding logic | A risk: the dissection gives up below three levels, and the BTF removes only ideal supplies | `dissection.rs:114`; §3.11 | M | That fixture dissects into interiors and scales | |
| L6 | Krylov solves preconditioned by the last factors, stopped at the resolution `|J|·|x|` | Runs bound by refactorization: 78% of `rc_mesh_300`, where a refactorization costs at least about 50 solves | §3.6; Li and Shi 2006 | M | The mesh's transient faster at equal accuracy; the same bits on any thread count | Lead |
| L7 | Dense kernels for the separators' fronts: register-blocked, each entry in `k` order, vectorized across entries only | The meshes' separators | Handoff, Phase 6's prototype | M | Fronts eliminate faster, with the same bits at any thread count and vector width | |
| L8 | Memory at scale: single runs record only what they name, names interned or formatted on demand, the remembered operating point's solver copy shared or optional, bytes per device measured | A wall from a million devices: 2.1 GiB for the 50,000 chain at about 800 points | §2.4 | M | A million-MOSFET chain runs on this 32 GB Mac, its peak bytes per device reported | |
| L9 | The front end at 10⁶–10⁷ devices: measure the builder, the JSON transport and elaboration at 10⁵ and 10⁶, then fix the worst first | Unknown: the blueprint's figures are unmeasured | `builder.py:444–458`; §3.11 | L | Building and elaborating 10⁶ devices is timed and profiled; 10⁷ fits in memory | |
| M1 | Many-runs efficiency measured against attainable throughput, the sum of each core class's single-thread rate | The 90% target cannot be met on mixed cores: 74% is about 96% of attainable (estimate) | §2.3 | S | The benchmark reports both | |
| M2 | Samples in SIMD lanes, each with its own steps, bit-identical to its single run | Throughput per core of batches (unknown) | §3.10; Xyce's lockstep | L | k lanes give k single runs' bits; throughput measured | Lead |
| M3 | jolty's own elementary functions, correctly rounded, used by the scalar, vector and GPU paths | The precondition of M2, L7 and G1; the same bits on every platform, where they differ today | §3.7; plan's determinism contract | M | The corpus is bit-identical on macOS and Linux; the cost of an `exp` measured | Lead |
| M4 | Transient adjoint sensitivities | Optimizers' gradients: one backward pass for every parameter | `Xyce_UG_ch09.tex:1750`; `sensitivity.rs:1–28` | M–L | One objective's gradient agrees with the direct method, at a cost independent of the parameters' number | |
| M5 | An exact cache of results keyed by the inputs' hash | Agents running a run again | Plan's determinism contract | S | A repeated run returns the cached bits | Lead |
| G1 | A GPU back end, batches first, on a machine with double-precision GPUs | Many runs; vendors report 3× on a 200-million-device transient | §3.9 | L, and an owner's machine | A GPU batch gives the CPU's bits | Lead |
| R1 | Like-with-like references: ngspice 47 with `option klu`, Xyce serial and parallel, end to end and engine only | The 17–40× on meshes and 1.2× on the 50,000 chain rest on Sparse 1.3 | `cktntask.c:142–144`; `perf/README.md:114` | S | `benchmarks.md` holds the numbers | |

## 5. The long view

In two to five years jolty could be past every reference on all three workloads:

- **One small run:** an iteration costs no more than ngspice's while every derivative stays exact, charge stays conserved and every charge is judged, through condensation, linear parts stamped once and cheaper kernels.
    Fewer iterations come from the accuracy theme.
    An agent's command returns in tens of milliseconds, from a warm runner.
    No reference pairs SPICE's speed with that accuracy.
- **One huge run:** ten to a hundred million devices on one machine at SPICE accuracy by default, with work proportional to activity through exact latency rather than tolerances.
    Elaboration is by template, orderings handle hub nets, and solves are hybrid, direct in the interiors and iterative on the separators, stopped at the rounding floor.
    Approximations are opt-in, each run stating its measured defect.
    Commercial FastSPICE tools are fast only by being approximate; jolty would be fast and exact, and say so with numbers.
- **Many runs:** samples in SIMD lanes and on GPUs, each bit-identical to its single run on any platform, across machines by shards, with exact caches and adjoint gradients for optimizers.
    That is the batch engine an agent's design loop needs, and none of the references has it.
- **Throughout:** the same bits on any thread count, machine, operating system, processor and back end.
    Owning the elementary functions and keeping each lane's order fixed makes that achievable; no reference promises it even across thread counts.

## 6. Principles to propose

For `AGENTS.md`, if the owner agrees:

1. **Exact before approximate.**
    Speed comes first from exact means: algebra, reuse proven by a kernel's purity, better orderings.
    An approximation is a knob a run opts into, never a default, and its results state its measured error.
    This is `vision.md`'s open question, with the error report added.
1. **Arithmetic agnostic of width.**
    A kernel may choose its vector width at run time only if every lane performs the same operations in the same order at every width and nothing fuses a multiply and an add implicitly; a test compares the widths bit for bit.
    This replaces the plan's blanket ban on runtime dispatch with the property it protects.
1. **jolty owns its elementary functions.**
    `exp`, `ln` and `pow` are jolty's own, correctly rounded, so scalar, vector and GPU paths, and every platform, give the same bits.
    This extends the determinism contract from one platform to all.
1. **Speed claims compare like with like.**
    A reference runs at its best configuration (ngspice with KLU, Xyce with its own solver), and each claim gives both the engine's time and the end-to-end time, with the accuracy each simulator reached.
1. **Efficiency against attainable throughput.**
    Parallel efficiency is measured against the sum of each core class's single-thread rate, stated with the machine.
1. **Computed costs over thresholds.**
    A choice decidable by a cost jolty can compute, such as the operations of two orderings, is decided by computing it, not by a constant.
1. **Measure at the size claimed.**
    A scaling claim names a fixture of that size; extrapolations are marked as estimates, as the blueprint's figures were not.

## 7. Sources

**jolty** (the repository from `68c3f75` to `4f39ff4`):

- `AGENTS.md`; `perf/README.md`; `perf/circuits/buck.json`.
- `crates/jolty-models/src/mosfet.rs`, `kernel.rs`, `kernel/parts.rs`, `group.rs`, `dual.rs`, `bsim3.rs`, `junction.rs`, `binding.rs`.
- `crates/jolty-engine/src/newton.rs`, `options.rs`, `sensitivity.rs`, `elaborated.rs`.
- `crates/jolty-numerics/src/lu.rs`, `lu/dissection.rs`, `lu/tree.rs`.
- `src/jolty/builder.py`, `runner.py`; `docs/guide/simulation.md`, `devices.md`.
- Commits `efe815c` (decisions 11 to 13), `52e3bdb` and `4f39ff4` (the MOSFET's stamping and its junctions' and overlaps' derivatives).

**Notes** (these plans):

- the session log, not kept, `vision.md`, `speed.md`.
- an implementation plan, not kept; a handoff note, not kept.
- raw logs, not kept, `final-bench-2026-11-15.txt`, `last-mile-check-2026-11-14.txt`.
- a brief, not kept, `brief-mosfet-cost.md`.
- an archived note, not kept.

**ngspice 47** (`src/` of the unpacked tarball):

- `spicelib/analysis/cktntask.c`, `cktsopt.c`, `cktload.c`, `cktsetup.c`.
- `spicelib/devices/cktinit.c`.
- `spicelib/devices/mos1/mos1load.c`, `mos1defs.h`, `mos1set.c`.
- `spicelib/devices/bsim4/b4ld.c`.
- `maths/ni/niiter.c`; `frontend/commands.c`, `inp.c`; `NEWS`.

**Xyce** (clone):

- `src/LinearAlgebraServicesPKG/N_LAS_TranSolverFactory.C`, `N_LAS_fwd.h`, `N_LAS_AztecOOSolver.C`, `N_LAS_TransformTool.C`.
- `src/DeviceModelPKG/OpenModels/N_DEV_MOSFET1.C`, `Core/N_DEV_Pars.h`.
- `doc/Users_Guide/Xyce_UG_ch13.tex`, `Xyce_UG_ch09.tex`, `sampling.tex`.

**Papers and books:**

- T. A. Davis and E.
    Palamadai Natarajan, "Algorithm 907: KLU, a direct sparse solver for circuit simulation problems", ACM TOMS 37(3), 2010.
- X. Chen, Y. Wang and H.
    Yang, "NICSLU: An adaptive sparse matrix solver for parallel circuit simulation", IEEE TCAD 32(2), 2013.
- J. D. Booth, N. D. Ellingwood, H. K. Thornquist and S.
    Rajamanickam, "Basker: Parallel sparse LU factorization utilizing hierarchical parallelism and data layouts", Parallel Computing 68, 2017.
- S. Rajamanickam, E. G. Boman and M. A. Heroux, "ShyLU: A hybrid-hybrid solver for multicore platforms", IPDPS 2012.
- H. K. Thornquist, E. R. Keiter, R. J. Hoekstra, D. M. Day and E. G. Boman, "A parallel preconditioning strategy for efficient transistor-level circuit simulation", ICCAD 2009, 410–417.
- Z. Li and C.-J.
    R. Shi, "SILCA: SPICE-accurate iterative linear solver for efficient analysis of large-scale power/ground networks", IEEE TCAD 25(6), 2006.
- E. L. Wilson, "The static condensation algorithm", International Journal for Numerical Methods in Engineering 8(1), 1974.
- A. Sangiovanni-Vincentelli, L.-K.
    Chen and L. O. Chua, "An efficient heuristic cluster algorithm for tearing large-scale networks", IEEE Transactions on Circuits and Systems 24(12), 1977.
- I. S. Duff and J. K. Reid, "The multifrontal solution of indefinite sparse symmetric linear equations", ACM TOMS 9(3), 1983.
- K. W. Chan and V.
    Brandwajn, "Partial matrix refactorization", IEEE Transactions on Power Systems 1(1), 1986.
- E. Lelarasmee, A. E. Ruehli and A. L. Sangiovanni-Vincentelli, "The waveform relaxation method for time-domain analysis of large scale integrated circuits", IEEE TCAD 1(3), 1982.
- R. A. Saleh, J. E. Kleckner and A. R. Newton, "Iterated timing analysis in SPLICE1", ICCAD 1983.
- C. W. Gear and D. R. Wells, "Multirate linear multistep methods", BIT 24, 1984.
- M. Günther and P.
    Rentrop, "Multirate ROW methods and latency of electric circuits", Applied Numerical Mathematics 13, 1993.
- M. Günther, A. Kværnø and P.
    Rentrop, "Multirate partitioned Runge–Kutta methods", BIT 41(3), 2001.
- V. Savcenco, W. Hundsdorfer and J. G. Verwer, "A multirate time stepping strategy for stiff ordinary differential equations", BIT 47, 2007.
- W. H. Enright, "A new error-control for initial value solvers", Applied Mathematics and Computation 31, 1989.
- K. Gulati, J. F. Croix, S. P. Khatri and R.
    Shastry, "Fast circuit simulation on graphics processing units", ASP-DAC 2009, 403–408.
- K. He, S. X.-D.
    Tan, H. Wang and G.
    Shi, "GPU-accelerated parallel sparse LU factorization method for fast circuit analysis", IEEE TVLSI 24(3), 2016.
- X. Chen, L. Ren, Y. Wang and H.
    Yang, "GPU-accelerated sparse LU factorization for circuit simulation with performance modeling", IEEE TPDS 26(3), 2015.
- A. Odabasioglu, M. Celik and L. T. Pileggi, "PRIMA: Passive reduced-order interconnect macromodeling algorithm", IEEE TCAD 17(8), 1998.
- B. N. Sheehan, "TICER: Realizable reduction of extracted RC circuits", ICCAD 1999.
- A. Sibidanov, P. Zimmermann and S.
    Glondu, "The CORE-MATH project", IEEE ARITH 2022, and the project's MIT licence (Zimmermann's FPBench 2023 talk).
- T. L. Quarles, "Analysis of performance and convergence issues for circuit simulation", UCB/ERL M89/42, 1989: a scan, whose text could not be read here.

**Makers' and users' publications:**

- Cadence, "Cadence introduces the Spectre X Simulator", press release, 3 June 2019; SemiWiki's coverage.
- Siemens EDA, Analog FastSPICE platform fact sheet, and "Analog FastSPICE eXTreme" press release, 2020.
- A. Ramadan, A. Pant and M.
    Guo, "Accelerating Accuracy: PrimeSim SPICE Powered by GPU", Synopsys blog, 10 June 2026; Synopsys and NVIDIA's GTC announcements, March 2025.
- L. Yang, "What's the difference between SPICE and FastSPICE circuit simulators?", Electronic Design, 8 March 2016.
- Synopsys, US patent 8,868,395, "Fast simulation method for integrated circuits with power management circuitry", filed 2008.
- M. Engelhardt, "SPICE Differentiation", LT Journal of Analog Innovation 24(4) (its text was not reachable; quoted from search results).
- Silvaco, SmartSpice HPP application note 1-029; Altium, "Advanced SPICE Options".
- PyTorch, MPS backend notes (no `float64` in Metal); Rust, `std::simd` documentation and tracking issue 86656.
