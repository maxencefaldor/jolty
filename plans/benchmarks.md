# Benchmarks

Status: rewritten in plain words on 15 November 2026, every item checked against its source and today's code, and brought up to date on 18 November; the research is `research/benchmarks.md`, kept whole.

Plan: ✅ 6 done · 🚧 2 in progress · 📋 7 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

Every claim of the other documents rests on a measurement anyone can repeat: of coverage, which analyses and models each simulator has; of precision, each one's error against exact answers; of reliability, each one's failure rate on hard circuits; and of speed, each one's time at the accuracy it reached, all against the references at their best configurations (`AGENTS.md`, "Claims are measured").

## 2. Where we stand

**The suites, all run by `scripts/bench`** (precision brought in and `scripts/precision` deleted by `092e86f`, `quality.md` 13):

| Suite | What it measures | Command | First results |
| --- | --- | --- | --- |
| Speed | Time of `bjt_amp`, `ac_wide`, `buck`, `inverter_chain_500`, `rc_mesh_100` and Monte Carlo, in Rust and in Python, against a committed baseline, every change over 5% flagged | `scripts/bench`, `--view` (matched, defaults or equal-error) for every simulator's times | this Mac; machine A (below) |
| Scaling | One large run on 1 to 64 threads, bits checked identical; each threshold's break-even | `--scale`, `--break-even NAME` | this Mac under load; machine A quiet (below) |
| Precision | Correct digits against exact answers, against cost, for jolty and ngspice with Sparse and KLU (and Xyce since `a3b6c78`) | `scripts/bench --precision` | this Mac (below); machine D (`5efde61`) |
| Hard circuits | Each simulator's outcome on 48 hard circuits, and its failure rate | `--robustness` | this Mac (below) |
| Coverage | Every analysis, model, syntax feature and output, each cell a probe that ran in jolty, ngspice and Xyce | `--coverage` | this Mac and machine B (below) |

**The differential tests** compare jolty with ngspice 47 on a corpus of circuits, vendors' macromodels and random circuits of every device and size: agreement, which says nothing of accuracy.

**The machines:** this Apple M2 Max, 8 performance and 4 efficiency cores, shared with other work, so its timings carry its load; three shared lab workstations, machine A (AMD Threadripper 3970X, 32 cores, 64 threads), machine B (Ryzen 9 7950X, 32 threads) and machine C (Ryzen 9 7950X3D, 32 threads), each 125 GB, shared with the lab, jolty's tools in `~/.jolty-tools`, Xyce 7.10.0 built on machine B; machine D (AMD EPYC 9B14, 24 cores of a shared 96-core server), the fixed machine of the baselines since 18 November, ngspice 47 built on it; the institution's HPC, the cluster and the cluster, which need the owner's own login.

**The final benchmark on machine A** (`444535f`, 15 November 15:02 to 16:27, raw logs not kept; the loads beside each run, as the lab's other users loaded it at times):

| Fixture | jolty engine | jolty, Python | ngspice, process | ngspice / jolty |
| --- | --- | --- | --- | --- |
| `bjt_amp` | 1.34 ms | 1.61 ms | 13.1 ms | 8.1× |
| `ac_wide` | 17.2 ms | 22.0 ms | 116.8 ms | 5.3× |
| `buck` | 1.12 s | 1.09 s | 2.17 s | 2.0× |
| `inverter_chain_500` | 3.62 s | 3.81 s | 1.76 s | 0.46× |
| `rc_mesh_100` | 1.16 s | 1.14 s | 18.2 s | 16× |

Every fixture runs 1.2 to 1.7 times longer than on this Mac, the machines' difference per core; the chain's 0.46× is before condensation, which took 11% off it.
**The buck's ngspice time is suspect:** on 16 November ngspice ran the buck in 0.23 s on machine B and 0.28 s on this Mac at its defaults (item 4's views), not 2.17 s; to be measured again on machine A.
Monte Carlo scales with the cores: a sample takes 0.81 ms on one core and 0.023 ms on 64 threads, 35×.

One large run over threads, each threshold at this Mac's break-evens, the bits the same on every count:

| Fixture | Serially | 1 thread | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `inverter_chain_5000` | 55.66 s (load 1.5) | 55.83 s | 39.7 s | 32.31 s | 25.84 s | 23.08 s (2.41×) | 23.7 s | 27.46 s (load 8.9) |
| `inverter_chain_50000` | 699.6 s (load 25) | 692.7 s (load 1.5) | 395.7 s | 334.6 s | 272.7 s | 246.3 s (2.81×) | 248.5 s | 259.3 s (load 18) |
| `rc_mesh_300` | 27.59 s (load 31) | 14.91 s (load 21) | 8.702 s | 6.783 s | 6.333 s | 5.622 s | 5.317 s (2.80×) | 5.973 s |

The many-cores plan's target, 6× on 16 cores for the 50,000 chain, is not met (`speed.md` 1): the chain ran undissected, so its factoring and solving, 16% of its time, stayed on one thread.

**The break-evens on machine A**, quiet (load 1 to 7, 16:41 to 18:22, raw logs not kept): from which size setting each threshold makes a run faster, on 2, 4, 8 and 64 threads:

| Threshold | Circuits | 2 | 4 | 8 | 64 |
| --- | --- | --- | --- | --- | --- |
| `min_solves_per_task` | `bjt_amp`'s frequencies | from 1 solve | from 5 | from 5 | from 30 |
| `min_instances_per_task` | chains | no gain up to 1,000 inverters | from 500 | from 500 | no gain |
| `min_instances_per_task` | meshes | from 50 a side | from 70 | from 70 | from 100 |
| `min_unknowns_per_task` (then `min_unknowns_inside`) | chains | no gain up to 5,000 inverters | no gain | no gain | no gain |
| `min_unknowns_per_task` (then `min_unknowns_inside`) | meshes | from 300 a side | from 300 | no gain | from 300 |
| `max_unknowns_per_interior` | chains | from 5,000 inverters, −1.7% | no gain | no gain | no gain |
| `max_unknowns_per_interior` | meshes | from 100 a side, −17% | from 70, −6% | from 50, −4% | from 100, −16% |

At 300 a side the dissected mesh takes 7.37 s on 64 threads against 27.6 s undissected, −73%.

**The hard-circuit corpus's first rates** (`6905ddf`; this Mac, load 8.1 to 11.5; raw logs, not kept): 48 cases, jolty's own 18 (Schmitt triggers, a latch from rest, the SRAM cell, a ring from rest, a chain without capacitance, an impasse circuit, `folded_sources`, the buck, the memristive NOR) and 30 from CircuitSim90's and Xyce's MCNC decks, fetched at upstream `Xyce/Xyce_Regression` commit `7bb7e98` (`benchmarks/robustness/external.json`) and never copied.

| Simulator | Kind | Failed | Runs | Rate | 95% Wilson interval |
| --- | --- | --- | --- | --- | --- |
| jolty | DC | 0 | 21 | 0% | 0–15% |
| jolty | transient | 0 | 19 | 0% | 0–17% |
| ngspice, Sparse | DC | 1 | 25 | 4% | 1–20% |
| ngspice, Sparse | transient | 0 | 23 | 0% | 0–14% |

On 15 November jolty read 11 of the 15 decks; since `models.md` 7b (`.ic`, `.nodeset`, `EXP`), it reads all 15 decks (48 of 48 runs): on machine D (`5efde61`, 18 November), jolty failed 0 of 25 DC runs (25 agreed) and 0 of 23 transients (6 agreed, 16 solved, 1 disagreed on `impasse/tran`, matching ngspice), while ngspice Sparse failed 1 DC run (`folded_sources/op`) and disagreed on `vreg/dc` and `impasse/tran`.
The intervals are wide: more cases narrow them (item 8b).

**The precision suite's first diagrams** (`3525ac5`; 16 fixtures, exact answers in mpmath at 50 digits, 2,448 runs; this Mac, load 5.9 to 11.5; `benchmarks/precision/README.md`, results kept out of the repository per `.gitignore`): jolty leads on DC (14.6 correct digits to ngspice's 8.0), noise (15.2 to 8.1, ngspice held by its 2014 electron charge), poles (15.3 to 14.7) and stiff networks with every tolerance scaled (5.1 to 5.4 against 3.2 to 3.9); it loses where `accuracy.md` 4 and 16 to 18 say.
At equal digits jolty is faster on the RC, RLC and tank, about twice as slow on the networks; at the defaults both get 1.2–3.8 digits on transients.

**The coverage matrix** (`benchmarks/coverage/coverage.json`, `2e95d37`, confirmed identically on machine D `5efde61`; 65 rows): jolty offers 51 rows (**49 agreed**, **1 solved** (`vbic`, no closed form yet), **1 disagreed** (`tline`, a defect in the matrix's `ngspice_only` probe, which runs `OperatingPoint()` instead of `Transient` so it reports `read no v(out)@...` even though jolty's `transmission_line` model meets closed forms to 3.4e-15 V)), and **14 unsupported** (`mos2`, `mos6`, `mos9`, `bsimsoi`, `hisim2`, `hicum`, `mesfet`, `jfet2`, `ltra`, `urc`, `xspice`, `osdi`, plus `lib` and `include` whose matrix probes place `.lib`/`.include` inside a subcircuit); ngspice Sparse: 41 agreed, 9 solved, 3 reference, 1 disagreed (`sens-ac`), 2 documented, 9 unsupported; ngspice KLU: 42 agreed, 9 solved, 1 failed (`sens-ac`), 2 documented, 11 unsupported; Xyce serial: 35 agreed, 2 solved, 1 disagreed (`noise`, by its 1973 electron charge), 14 documented, 13 unsupported.

**Xyce's first column** (on `main` since `6047a2d` to `dd18fa0`, 15 November; measured on machine B, Ryzen 9 7950X, 16 cores, load 0.8 to 4.3; each simulator at its own defaults; raw logs not kept): on the two fixtures where both found the same solution, `bjt_amp` (jolty 1.0 ms, Xyce 68.4 ms end to end, 4.3 ms of analysis) and `ac_wide` (12.9 ms, 214 ms and 138 ms), jolty is faster.
Elsewhere the work differs: the 500-inverter chain's edge lands 3.3% later in Xyce, mostly its own step error at its default tolerance; the meshes agree only to 1 to 2.5% at Xyce's defaults; Xyce stalls on the buck at 128.8 µs, where a body diode conducts; on the 300 × 300 mesh Xyce's MPI build takes 6.0 s on 16 ranks against jolty's 3.67 s on 16 threads.
Its agreement tests, 67 on machine B, pass at `test_ngspice.py`'s tolerances, with five departures stated (`accuracy.md` 19).

**How sound the speed suite is** (the research): timings spread up to 9.8%, and 19% for Monte Carlo on every core, where changes are flagged at 5%; the large fixtures run once; the baseline is not interleaved; no time carried an interval or the accuracy reached, which `--compare` and the views now give (items 3 and 4).

## 3. Gaps

- Xyce has columns in the speed suite's views, precision suite and coverage matrix, but not yet in the robustness corpus, and is not installed on machine D (item 11); ngspice comparisons like with like (`--view`) are done (item 4).
- The suites' second versions (items 7b, 8b, 9b).
- No machine runs the nightly suite yet (item 5); nothing deterministic is checked per platform in CI (item 5b).

## 4. Plan

Each item says what it is, why it matters, and what "done" means; **Lead** marks an item where jolty would go beyond every reference.
Its source is the research report's row, in `research/benchmarks.md` §4.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. One run over threads, and each threshold's break-even

- **What:** `scripts/bench --scale` (each large fixture serially and spread, the bits checked identical) and `--break-even NAME` (from which size a threshold of `Parallelism` pays).
- **Result:** `7d41cc0`; the measurements are §2's and `speed.md` 2's.

### 🚧 2. The final benchmark on quiet machines

- **What:** the suite, ngspice on its fixtures, and the scaling on 1 to 64 threads, on a machine nothing else loads; then the break-evens there, and the guide's thresholds stated with that machine.
- **State:** machine A's suite, ngspice and scaling done; its break-evens done (§2), and the guide and `jolty.plan.BREAK_EVENS` state them as the defaults (`speed.md` 4).
- **Source:** row B12.

### ✅ 3. A timing protocol that can decide a 5% change

- **What:** alternating rounds of the compared builds, medians with 95% rank intervals, wall and CPU time, the load before and after, a change flagged only when the intervals separate.
- **Done when:** the suite prints intervals and flags only separated changes.
- **Result** (`7f841ca` to `d38cd29`, 16 November): `scripts/bench --compare A,B` times two revisions' builds in alternating rounds, each median with its interval, and names a change only where the intervals separate (`benchmarks/speed/README.md`, "Deciding a change"); `cc48834` tests its intervals, rounds and verdicts.
- **Source:** row B1.

### ✅ 4. ngspice compared like with like

- **What:** ngspice with `.options klu` where it applies, an OpenMP build for BSIM3, its own analysis time beside its process time; each run's error against a reference beside its time; the same solution confirmed before two times are compared.
- **Three views,** the owner's (15 November): every simulator at matched options, at its own defaults, and at equal measured error; comparing like with like is the benchmark's job.
- **Done when:** each speed row carries both engines' times and errors, in the three views.
- **Result** (on `main`, `d191f1b` to `3945c01`, 16 November, raw logs, not kept): `scripts/bench --view matched | defaults | equal-error`, jolty, ngspice with Sparse and KLU and Xyce, each run's options stated in the table's first row, its time end to end and in its analyses, its distance from jolty's run at matched options; a failed or timed-out run a row saying what stopped it.
    At defaults on machine B: `bjt_amp` 1.06 ms against ngspice's 4.82 ms; `ac_wide` 13.3 against 31.0 ms; the buck 770 against 227 ms (3.4× slower); the 500 inverters 4.59 against 1.17 s; `rc_mesh_100` 645 ms against 12.6 s (Sparse) and 1.60 s (KLU); Xyce slowest on each, past 300 s on the buck.
    At matched options jolty alone finishes the buck (6.2 s; ngspice stops at 1 ns, Xyce at step 14,762).
    At equal error jolty is fastest on `rc_step`, `rc_ramp`, `lc_tank`, `butterworth_poles` and `rc_noise`, ngspice on the ladders and trees at 0.4–0.8 of jolty's time.
    **Not the same work:** on the 500 inverters every simulator at its defaults lies a whole swing from jolty's tight run, and 0.18 at matched options; on `bjt_amp` ngspice and jolty differ by 3.8e-3 at any tolerance, a model or source difference to trace.
    **Not done:** the views interleaved as `--compare` interleaves builds, and `--scale`'s large fixtures in them.
- **Source:** rows B2, B3 and B4; `research/speed.md` row R1.

### 📋 5. The suite nightly on a fixed machine

- **What:** the speed suite run each night on machine A or machine B against the previous night's results.
- **Done when:** each night's results stand beside the last's.
- **State** (`f61d0cd`, `4e5f0fc`, 16 November): `scripts/bench --nightly DIR` compares the checked-out commit's build with the previous night's and keeps each night's comparison, run from a crontab line the speed suite's page gives.
    **Not done:** no machine runs it each night yet; machine D, the fixed machine of the baselines, runs only on request.

### 📋 5b. What is deterministic checked in CI

- **What:** each fixture's iterations, steps, factorizations and error at the defaults, and the precision suite's digits on its 14 fixtures under 1,000 nodes, which are the same bits on one platform, recorded per platform and checked in `scripts/check`.
- **Done when:** a change that moves them fails until it records the move.
- **Source:** row B11; the precision suite's report.

### 📋 6. Efficiency against concurrent copies, and peak memory

- **What:** a run's speed-up on N threads beside N copies of it run at once, which is what the machine can attain; each fixture's peak memory.
- **Done when:** the suite prints both on every machine.
- **Source:** rows B10 and B16.

### ✅ 7. The precision suite

- **What:** fixtures with exact answers computed independently (closed forms; linear circuits solved in higher precision), each simulator scored by the correct digits of its worst point against cost, along sweeps of the tolerances, drawn as work-precision diagrams.
- **Result** (`3525ac5` to `273af4b`): 16 fixtures, jolty and ngspice with Sparse and KLU; findings in §2.
- **Lead:** no reference publishes a work-precision diagram.
    **Source:** rows B5; `research/accuracy.md` row 1.

### 📋 7b. The precision suite's second version

- **What:** manufactured solutions for nonlinear devices, each written as a behavioral expression with an injected source `b(t)` (charges need a behavioral `q(v)` first); the IVP test set's ring modulator and transistor amplifier, then its NAND gate and charge pump; 20 fixtures or more; each simulator's best time for each accuracy; timings on a quiet machine; a sweep with the longest step lifted.
- **Done when:** 20 or more fixtures, the IVP set's circuits among them, timed on machine A.
- **Source:** rows B5 and B6; the suite's report.

### ✅ 8. The hard-circuit corpus

- **What:** jolty's own hard circuits and third-party decks fetched by a pinned manifest, each run in each simulator, its outcome classified (solved and right, another solution, failed, timed out, unsupported), each simulator's failure rate with a Wilson interval, DC and transient apart.
- **Result** (`6905ddf` to `9d9eddf`): 48 cases; rates in §2; three SPICE reader faults found and fixed.
- **Lead:** no reference publishes a failure rate on hard circuits.
    **Source:** row B7; `research/accuracy.md` row 10.

### 📋 8b. The corpus's second version

- **What:** Xyce as a simulator; references for the decks' transients, which count as solved when they finish today; ngspice with KLU; more cases, to narrow the intervals (jolty's operating-point iterations are already recorded in the table since `090b521`, `agentic.md` 21).
- **Done when:** each of these in the corpus's table.
- **Source:** the corpus's report.

### ✅ 9. The coverage matrix

- **What:** a row for every analysis, device model and level, SPICE syntax feature and output; for each, a smallest circuit with a known answer run in jolty, ngspice and Xyce; each cell "agrees", "disagrees" with the numbers, "not offered" with the simulator's message, or "documented" from a manual's cited section for simulators that cannot run here.
- **Result:** the first version, 65 rows, on `main` since 16 November (`benchmarks/coverage`), run by `scripts/bench --coverage`; counts in §2 (known probe defect: `tline` uses `ngspice_only` with `OperatingPoint()` instead of `Transient`).
- **Lead:** no one publishes a coverage matrix whose cells ran.
    **Source:** row B8.

### 📋 9b. The coverage matrix's second version

- **What:** rows for corners and worst case, `.disto`, `.sp`, transient noise, `.four`, `.ic`, `.nodeset`, switches, logic and the kits; VBIC's row judged (its closed form misses both simulators by 2e-6, unexplained); BSIM3's probe at a stronger bias; ngspice's column checked in the tests where ngspice is installed.
- **Done when:** each in the matrix.
    **Source:** the matrix's report.

### 📋 10. The speed suite's fixtures widened

- **What:** a ring oscillator, ISCAS'85 circuits at transistor level, IBM's power grids, CircuitSim90's level 3 decks, beside today's; third-party fixtures fetched by a pinned manifest, never committed.
- **Done when:** each fixture timed in jolty and ngspice on the quiet machines.
- **Source:** rows B9 and B17.

### 🚧 11. Xyce in the suites

- **What:** Xyce 7.10, serial and MPI, as a third simulator in the speed suite, the corpus and the coverage matrix, at its own best configuration.
- **State:** built on machine B by Spack (15 November, as the owner asked whether it was in); jolty's Xyce dialect, its 67 agreement tests (`tests/test_xyce.py`) and the speed suite's first column on `main` (§2); the coverage matrix's Xyce probes run through the same library.
    **Not done:** the corpus's Xyce runner, the chain and the meshes timed at equal accuracy, the 50,000 chain, the buck's stall understood.
- **The buck:** a row marked stalled at 128.8 µs, Xyce at its defaults, not dropped (the owner, 15 November).
- **Done when:** its columns in the tables.
    **Source:** row B13.

### ⏸️ 13. A fixture of ten million devices

- **What:** such as the scaling blueprint's crossbar transformer, its build, elaboration, operating point and transient timed, their memory measured.
- **Parked until:** the front end builds one (`speed.md` 11).

**Elsewhere:** timing an agent's command end to end is `speed.md` 8.

## 5. Decided

- 15 Nov, the owner: the references are ngspice and Xyce alone; no LTspice or commercial simulator.
- **How speed claims are made** (15 November, from `research/speed.md` §6): a reference runs at its best configuration, ngspice with KLU as well as Sparse 1.3, Xyce with its own solver; each claim gives the engine's time and the end-to-end time, with the accuracy each simulator reached; parallel efficiency is measured against attainable throughput, the sum of each core class's single-thread rate, stated with the machine; a scaling claim names a fixture of its size, and extrapolations are marked as estimates.
- The break-even tool lives in the repository, as `scripts/bench --break-even`, so anyone can measure the thresholds again.
- 15 Nov: the precision suite's results stay out of the repository, as comparisons with other simulators do until the owner decides; the coverage matrix's table is committed, as it is the suite's product.

## 6. Open principles

None: the research's proposals were merged into the rules of `AGENTS.md` on 15 November (`444535f`).
