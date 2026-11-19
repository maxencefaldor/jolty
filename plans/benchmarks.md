# Benchmarks

Status: rewritten in plain prose on 15 November 2026 with every item checked against source and code, and updated on 18 November.
The archival research report lives in `research/benchmarks.md`, kept whole.

Plan: ✅ 9 done · 🚧 2 in progress · 📋 4 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

Every claim across the planning documents rests on repeatable measurements against reference simulators at their best configurations (`AGENTS.md`, "Claims are measured").
Four benchmark axes provide that evidence: feature coverage, precision against exact answers, reliability on hard circuits, and runtime at matched accuracy.

## 2. Where we stand

**Benchmark suites executed via `scripts/bench`** (precision integrated and `scripts/precision` deleted in `092e86f`, `quality.md` 13):

| Suite | What it measures | Command | First results |
| --- | --- | --- | --- |
| Speed | Time of `bjt_amp`, `ac_wide`, `buck`, `inverter_chain_500`, `rc_mesh_100`, and Monte Carlo in Rust and Python against a committed baseline, flagging shifts above 5% | `scripts/bench`; `--view` (matched, defaults, or equal-error) for simulator comparisons | local Mac; machine A (below) |
| Scaling | Single large run across 1 to 64 threads verifying identical bits, plus threshold break-evens | `--scale`, `--break-even NAME` | local Mac under load; machine A quiet (below) |
| Precision | Significant digits against exact answers versus cost for jolty, ngspice Sparse, ngspice KLU, and Xyce (since `a3b6c78`) | `scripts/bench --precision` | local Mac (below); machine D (`5efde61`) |
| Hard circuits | Outcome and failure rate of each simulator across 48 hard circuits | `--robustness` | local Mac (below) |
| Coverage | Live probes for every analysis, model, syntax feature, and output across jolty, ngspice, and Xyce | `--coverage` | local Mac and machine B (below) |

**Differential tests** compare jolty against ngspice 47 across a circuit corpus, vendor macromodels, and random circuits spanning every device and size.
Differential tests verify cross-simulator agreement rather than absolute accuracy.

**Benchmark machines:**

- **Local Mac:** Apple M2 Max (8 performance and 4 efficiency cores), shared with interactive work, so timings include host load.
- **Lab workstations (machines A, B, and C):** 125 GB RAM each, shared with the lab, with jolty tools in `~/.jolty-tools`.
    Machine A is an AMD Threadripper 3970X (32 cores, 64 threads), machine B is a Ryzen 9 7950X (32 threads, hosting Xyce 7.10.0), and machine C is a Ryzen 9 7950X3D (32 threads).
- **Machine D:** AMD EPYC 9B14 (24 cores on a shared 96-core server), serving as the fixed baseline host since 18 November with ngspice 47 built locally.
- **Institutional HPC clusters:** available only through the owner's personal login.

**Final benchmark on machine A** (`444535f`, 15 November 15:02 to 16:27, raw logs not kept; host loads recorded alongside each run as other lab users shared the machine):

| Fixture | jolty engine | jolty, Python | ngspice, process | ngspice / jolty |
| --- | --- | --- | --- | --- |
| `bjt_amp` | 1.34 ms | 1.61 ms | 13.1 ms | 8.1× |
| `ac_wide` | 17.2 ms | 22.0 ms | 116.8 ms | 5.3× |
| `buck` | 1.12 s | 1.09 s | 2.17 s | 2.0× |
| `inverter_chain_500` | 3.62 s | 3.81 s | 1.76 s | 0.46× |
| `rc_mesh_100` | 1.16 s | 1.14 s | 18.2 s | 16× |

Every fixture runs 1.2 to 1.7 times slower than on the local Mac due to per-core hardware differences; the chain's 0.46× ratio predates condensation, which cut 11% from runtime.
**Suspect ngspice buck timing:** on 16 November, ngspice ran the buck in 0.23 s on machine B and 0.28 s on the local Mac at defaults (item 4 views), rather than 2.17 s.
The buck fixture needs re-measurement on machine A.
Monte Carlo scales with cores, dropping from 0.81 ms per sample on 1 core to 0.023 ms on 64 threads (35×).

Single large runs across thread counts using local Mac break-even thresholds, with identical bits at every count:

| Fixture | Serially | 1 thread | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `inverter_chain_5000` | 55.66 s (load 1.5) | 55.83 s | 39.7 s | 32.31 s | 25.84 s | 23.08 s (2.41×) | 23.7 s | 27.46 s (load 8.9) |
| `inverter_chain_50000` | 699.6 s (load 25) | 692.7 s (load 1.5) | 395.7 s | 334.6 s | 272.7 s | 246.3 s (2.81×) | 248.5 s | 259.3 s (load 18) |
| `rc_mesh_300` | 27.59 s (load 31) | 14.91 s (load 21) | 8.702 s | 6.783 s | 6.333 s | 5.622 s | 5.317 s (2.80×) | 5.973 s |

The 6× target on 16 cores for `inverter_chain_50000` is not yet met (`speed.md` 1) because the chain ran undissected, leaving factoring and solving (16% of runtime) on a single thread.

**Break-evens on quiet machine A** (load 1 to 7, 16:41 to 18:22, raw logs not kept): circuit sizes where each threshold accelerates execution across 2, 4, 8, and 64 threads:

| Threshold | Circuits | 2 | 4 | 8 | 64 |
| --- | --- | --- | --- | --- | --- |
| `min_solves_per_task` | `bjt_amp`'s frequencies | from 1 solve | from 5 | from 5 | from 30 |
| `min_instances_per_task` | chains | no gain up to 1,000 inverters | from 500 | from 500 | no gain |
| `min_instances_per_task` | meshes | from 50 a side | from 70 | from 70 | from 100 |
| `min_unknowns_per_task` (then `min_unknowns_inside`) | chains | no gain up to 5,000 inverters | no gain | no gain | no gain |
| `min_unknowns_per_task` (then `min_unknowns_inside`) | meshes | from 300 a side | from 300 | no gain | from 300 |
| `max_unknowns_per_interior` | chains | from 5,000 inverters, −1.7% | no gain | no gain | no gain |
| `max_unknowns_per_interior` | meshes | from 100 a side, −17% | from 70, −6% | from 50, −4% | from 100, −16% |

At 300 nodes per side, the dissected mesh runs in 7.37 s on 64 threads versus 27.6 s undissected (−73%).

**Initial hard-circuit corpus rates** (`6905ddf`; local Mac, load 8.1 to 11.5; raw logs not kept): 48 total cases.
Eighteen cases come from jolty (Schmitt triggers, a latch from rest, an SRAM cell, a ring from rest, a capacitance-free chain, an impasse circuit, `folded_sources`, the buck, and the memristive NOR).
Thirty cases come from CircuitSim90 and Xyce MCNC decks, fetched from upstream `Xyce/Xyce_Regression` commit `7bb7e98` (`benchmarks/robustness/external.json`) without copying source decks into the repository.

| Simulator | Kind | Failed | Runs | Rate | 95% Wilson interval |
| --- | --- | --- | --- | --- | --- |
| jolty | DC | 0 | 21 | 0% | 0 to 15% |
| jolty | transient | 0 | 19 | 0% | 0 to 17% |
| ngspice, Sparse | DC | 1 | 25 | 4% | 1 to 20% |
| ngspice, Sparse | transient | 0 | 23 | 0% | 0 to 14% |

On 15 November, jolty parsed 11 of the 15 external decks; since `models.md` 7b (`.ic`, `.nodeset`, `EXP`), jolty parses all 15 decks (48 of 48 runs).
On machine D (`5efde61`, 18 November), jolty failed 0 of 25 DC runs (25 agreed) and 0 of 23 transients (6 agreed, 16 solved, and 1 disagreed on `impasse/tran`, matching ngspice).
On the same run, ngspice Sparse failed 1 DC case (`folded_sources/op`) and disagreed on `vreg/dc` and `impasse/tran`.
Confidence intervals remain wide and require additional cases to narrow (item 8b).

**Initial precision suite diagrams** (`3525ac5`; 16 fixtures, 50-digit mpmath exact references, 2,448 runs; local Mac, load 5.9 to 11.5; see `benchmarks/precision/README.md`, with results gitignored).
jolty leads on DC (14.6 correct digits vs ngspice's 8.0), noise (15.2 vs 8.1, limited by ngspice's 2014 electron charge), and poles (15.3 vs 14.7).
On stiff networks with all tolerances scaled, jolty reaches 5.1 to 5.4 digits against ngspice's 3.2 to 3.9, while losses follow the causes tracked in `accuracy.md` items 4 and 16 to 18.
At matched accuracy, jolty is faster on the RC, RLC, and tank fixtures and roughly twice as slow on networks; at defaults, both simulators achieve 1.2 to 3.8 digits on transients.

**Coverage matrix** (`benchmarks/coverage/coverage.json`, 80 rows after v2):

- **jolty:** **62 agreed**, **3 documented** (`kit-sky130`, `kit-gf180mcu`, `kit-sg13g2`), **0 failed**, **0 disagreed**, **0 unknown**, and **15 unsupported** (`disto`, `sp`, `trnoise`, `mos2`, `mos6`, `mos9`, `bsimsoi`, `hisim2`, `hicum`, `mesfet`, `jfet2`, `ltra`, `urc`, `xspice`, `osdi`).
    Recent additions include subcircuit `lib` and `include`, `vbic` agreeing to 8.1e-8 against exact coupled-KCL closed form, `bsim3` agreeing to 3.3e-16 in strong-inversion saturation, plus `corners`, `worst-case`, `switch-voltage`, `switch-current`, `logic`, `ic`, `nodeset`, `pdk-library`, and `four`.
- **ngspice Sparse:** 49 agreed, 3 reference, 9 solved, 2 disagreed (`sens-ac`, plus `disto` where `disto2` `HD3` returns `-Vt/12` instead of exact `+Vt/12`), 7 documented, 10 unsupported, and 0 failed.
- **ngspice KLU:** 50 agreed, 9 solved, 1 failed (`sens-ac`), 7 documented, and 13 unsupported (including `disto`, which KLU rejects).
- **Xyce serial:** 35 agreed, 2 solved, 1 disagreed (`noise`, due to its 1973 electron charge), 26 documented, and 16 unsupported (preserving all 65 prior cells identically).

**Xyce first column** (on `main` since `6047a2d` to `dd18fa0`, 15 November; machine B, Ryzen 9 7950X, 16 cores, load 0.8 to 4.3; default options; raw logs not kept).
Where both simulators find the same solution, jolty beats Xyce on `bjt_amp` (1.0 ms vs 68.4 ms end-to-end and 4.3 ms analysis).
On `ac_wide`, jolty takes 12.9 ms versus Xyce's 214 ms end-to-end and 138 ms analysis.
Elsewhere the solutions differ: Xyce's 500-inverter chain edge arrives 3.3% later due to default-tolerance step error, and meshes agree only to 1 to 2.5% at Xyce defaults.
Xyce stalls on the buck at 128.8 µs where a body diode conducts, and on the 300 × 300 mesh Xyce MPI takes 6.0 s on 16 ranks versus jolty's 3.67 s on 16 threads.
All 67 Xyce agreement tests on machine B pass at `test_ngspice.py` tolerances, with five documented departures (`accuracy.md` 19).

**Speed suite noise** (from research): timings spread up to 9.8% (19% for multi-core Monte Carlo) despite flagging 5% regressions, large fixtures ran once, and baselines were not interleaved.
Previous timings also lacked confidence intervals and accuracy metrics, now supplied by `--compare` and `--view` (items 3 and 4).

## 3. Gaps

- Xyce has columns in speed views, precision, and coverage, but not yet in the robustness corpus, and is not installed on machine D (item 11); matched ngspice comparisons (`--view`) are done (item 4).
- Second versions of the precision suite and robustness corpus remain planned (items 7b and 8b), while coverage matrix v2 is done (item 9b).
- No machine runs the nightly suite yet (item 5), though CI now verifies deterministic counts, digits, and fingerprints (item 5b).

## 4. Plan

Each item specifies its scope, motivation, and completion criteria; **Lead** marks an item where jolty surpasses every reference.
Each item's source references the corresponding row in `research/benchmarks.md` §4.

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. Single-run thread scaling and threshold break-evens

- **What:** `scripts/bench --scale` (serial and parallel scaling of large fixtures with bit-identity verification) and `--break-even NAME` (measuring the circuit size where each `Parallelism` threshold pays off).
- **Result:** `7d41cc0`; measurements appear in §2 and `speed.md` item 2.

### 🚧 2. Final benchmark on quiet machines

- **What:** run the full speed suite, ngspice comparisons, 1-to-64-thread scaling, and threshold break-evens on an unloaded machine, then publish those thresholds in the guide.
- **State:** machine A's speed suite, ngspice comparisons, scaling, and break-evens are complete (§2), and both the guide and `jolty.plan.BREAK_EVENS` publish those defaults (`speed.md` 4).
    On 18 November 2026, the full commit `430c75d` baseline was measured and committed on machine D (`benchmarks/speed/results/430c75d6b0b402812924ebcfd22ed6c91fee6378.json`).
    The committed baseline stores 43 timing entries (with per-run samples for 95% rank CIs) and 7 transient profiles, scrubbing the host string to `Linux-x86_64-with-glibc2.27` (shared-host load 145 to 224; raw view files not committed).
- **Machine D summary** (full tables in `benchmarks/speed/README.md`):
    - **Speed (Rust / Python):** `bjt_amp` 1.532 / 1.942 ms, `ac_wide` 3.069 / 9.691 ms, and `buck` 1.518 / 1.556 s.
        Larger fixtures run `inverter_chain_500` in 7.825 / 8.574 s, `rc_mesh_100` in 1.340 / 1.155 s, and `monte_carlo` (100,000 samples) at 57.92 µs/sample Rust (24 threads) and 59.72 µs/sample Python.
    - **Scaling:** `rc_mesh_300` drops from 78.84 s serial to 11.08 s on 24 threads (7.12×), while `inverter_chain_50000` runs in 1,001 s (881.9 s profiled).
        `inverter_chain_5000` drops from 250.3 s serial to 58.81 s on 16 threads (4.26×) and 70.78 s on 24 threads (3.54×), keeping bit-identical fingerprint `988bcd74633cb666` across all thread counts.
    - **Defaults view (`--view defaults` vs ngspice 47):** `bjt_amp` takes 2.029 ms vs 3.711 ms Sparse and 3.948 ms KLU, while `ac_wide` takes 12.42 ms vs 44.18 ms Sparse and 54.83 ms KLU.
        `buck` takes 1.589 s vs 358.7 / 406.7 ms, and `inverter_chain_500` takes 9.912 s vs 2.428 / 2.51 s (all 1.0× swing from the matched reference).
        `rc_mesh_100` takes 1.497 s versus 51.39 s Sparse and 5.069 s KLU.
    - **Matched view (`--view matched`, `430c75d`, 5,069 s):** `bjt_amp` takes 25.8 ms vs 8.116 / 8.724 ms, and `ac_wide` takes 18.65 ms vs 62.28 ms Sparse (3.34×) and 77.39 ms KLU (4.15×).
        `buck` finishes in 12.9 s while both ngspice Sparse and KLU fail with `Timestep too small` at `t = 9.87e-10` and `1.43e-09`.
        `inverter_chain_500` takes 127.7 s vs 56.86 / 59.04 s, and `rc_mesh_100` takes 5.742 s vs Sparse timeout (> 300 s) and KLU 16.44 s (2.86× faster).
- **Remaining for quiet hosts:** dedicated idle-machine re-run, break-evens on 24-core Zen 4, and Xyce on machine D (item 11).
- **Source:** row B12.

### ✅ 3. Timing protocol able to decide a 5% change

- **What:** alternating rounds of compared builds with medians, 95% rank intervals, wall and CPU time, before-and-after load, flagging changes only when intervals separate.
- **Done when:** the suite prints confidence intervals and flags only separated changes.
- **Result** (`7f841ca` to `d38cd29`, 16 November): `scripts/bench --compare A,B` times two revisions in alternating rounds with median confidence intervals, reporting a change only when intervals separate (`benchmarks/speed/README.md`, "Deciding a change").
    Commit `cc48834` adds tests covering intervals, rounds, and verdicts.
- **Source:** row B1.

### ✅ 4. ngspice compared like with like

- **What:** benchmark ngspice with `.options klu` where supported, an OpenMP build for BSIM3, and internal analysis time alongside process time.
    Report each run's error against a reference alongside runtime, confirming equivalent solutions before comparing timings.
- **Three views** (owner, 15 November): compare simulators at matched options, at default options, and at equal measured error.
- **Done when:** every speed row lists times and errors for both engines across all three views.
- **Result** (on `main`, `d191f1b` to `3945c01`, 16 November; raw logs not kept): `scripts/bench --view matched | defaults | equal-error` compares jolty, ngspice Sparse, ngspice KLU, and Xyce.
    Each table states simulator options in the header row, reports end-to-end and analysis runtimes plus distance from jolty's matched-options reference, and records failure or timeout reasons explicitly.
    At defaults on machine B, `bjt_amp` takes 1.06 ms vs ngspice's 4.82 ms, `ac_wide` takes 13.3 vs 31.0 ms, and `buck` takes 770 vs 227 ms (3.4× slower).
    On the same host, `inverter_chain_500` takes 4.59 vs 1.17 s, and `rc_mesh_100` takes 645 ms vs 12.6 s Sparse and 1.60 s KLU.
    Xyce is slowest on every fixture and exceeds 300 s on `buck`.
    At matched options, only jolty finishes `buck` (6.2 s; ngspice stops at 1 ns, Xyce stops at step 14,762).
    At equal error, jolty leads on `rc_step`, `rc_ramp`, `lc_tank`, `butterworth_poles`, and `rc_noise`, whereas ngspice runs ladders and trees in 0.4 to 0.8 of jolty's time.
    **Unequal work:** on `inverter_chain_500`, every simulator at defaults sits a full voltage swing away from jolty's tight reference (0.18 at matched options).
    **Traced and fixed (`bjt_amp` 3.8e-3, 18 November, machine D, ngspice 47 KLU at `TIGHT` / 300.15 K):** linear chord sag in `deviations()` via `np.interp`, not a model difference, caused the gap.
    Operating point agrees to `2.0e-11` relative (`+3.17e-12 V` on `out`), and AC agrees to `7.5e-13` relative across all 71 frequencies.
    In transient, `bjt_amp` leaves `max_step = None`, so jolty's `TIGHT` reference defaults to `stop / 50 = 4.00 µs` and keeps that step across the 10 kHz sine peak (`125 µs`) where `q''' = 0`.
    Comparing jolty's 790 accepted points against fine ngspice (`0.1 µs`) yields `max_abs = 3.288e-5 V` (`1.652e-5` of the `1.99 V` swing), while cubic-spline interpolation gives `2.686e-5 V` (`1.349e-5`).
    Fine jolty (`0.1 µs`) against fine ngspice (`0.1 µs`) yields `3.573e-5 V` (`1.795e-5`).
    Linearly interpolating jolty's `4.00 µs` chord `[122.490 µs, 126.479 µs]` across the sine peak produced sag `A ω² h² / 8 = 7.86 mV`.
    Measured sag reached `7.794e-3 V` at `t = 124.518 µs` (`3.916e-3` of swing, reported as `3.8e-3` in `--view matched`).
    Replacing `np.interp` in `benchmarks/speed/bench.py` (`deviations()`) with vectorized Huynh (1993) non-oscillatory fourth-order cubic Hermite interpolation (`interpolate_transient`) cut `bjt_amp` deviation from `3.916e-3` to `1.813e-5` against fine jolty (`0.1 µs`).
    Live `--view matched bjt_amp` against ngspice 47 dropped from `3.8e-3` to `5.9e-5` on both Sparse and KLU, with exact `0.0` overshoot on sharp switching steps and triangle corners.
    **Not done:** interleaving `--view` rounds like `--compare`, and adding `--scale` large fixtures to views.
- **Source:** rows B2, B3, and B4; `research/speed.md` row R1.

### 📋 5. Nightly speed suite on a fixed machine

- **What:** run the speed suite nightly on machine A or machine B against the previous night's results.
- **Done when:** each night's results are compared alongside the previous night's run.
- **State** (`f61d0cd`, `4e5f0fc`, 16 November): `scripts/bench --nightly DIR` compares the checked-out build against the previous night and retains nightly comparisons via a documented crontab entry.
    **Not done:** no host runs the nightly cron yet; machine D, the fixed baseline host, runs only on demand.

### ✅ 5b. Deterministic checks in CI

- **What:** record per-platform solver counts, default errors, and precision digits across the 14 fixtures under 1,000 nodes, and verify them in `scripts/check`.
- **Done when:** any change shifting deterministic metrics fails CI until the recorded baseline is updated.
- **Result** (18 November): `benchmarks/deterministic.py` checks all 5 non-scale speed fixtures (`bjt_amp`, `ac_wide`, `buck`, `inverter_chain_500`, `rc_mesh_100`) and all 14 precision fixtures under 1,000 nodes (`rc_step` through `tree_100`).
    Per-platform solver counts, significant digits, and SHA-256 fingerprints are keyed by `platform_key()` in `benchmarks/deterministic.json`, wired into `scripts/check python` and tested in `tests/test_deterministic.py`.
    Unrecorded platforms check linear and analytic fixtures tightly and chaotic nonlinear transients (`buck`, `inverter_chain_500`) within bounded cross-platform margins (`6%` counts, `0.05` digits) unless `--strict-platform` is passed.
    Recording `ladder_100` also uncovered and fixed an exact zero-pivot division (`pivots[0] == 0`) in `benchmarks/precision/exact.py::_Elimination::newton`.
- **Source:** row B11; precision suite report.

### ✅ 6. Concurrent-copy efficiency and peak memory

- **What:** compare N-thread single-run speedup against N concurrent single-thread copies representing attainable machine throughput, and record peak memory per fixture.
- **Done when:** the suite reports both metrics on every machine.
- **Result** (18 November): `benchmarks/measuring.py` adds `ru_maxrss_to_bytes`, `peak_memory_bytes` (reading `VmHWM` from `/proc/self/status` on Linux to avoid inherited parent `ru_maxrss`, and `getrusage(2)` on macOS/other), `available_memory_bytes`, and `memory` (IEC binary prefixes).
    `benchmarks/speed/timing.py` adds isolated `probe()` and sets `peak_memory_bytes` to `None` if a worker serves a second distinct fixture.
    `benchmarks/speed/bench.py` adds backwards-compatible Pydantic models (`TimingRecord`, `ConcurrentCopies`, `ThroughputEfficiency`, `SpeedResults`, `load_results`) and unclamped `throughput_efficiency` (`E(N) = T_copies / (N * T_par)` plus hybrid-core rate-sum diagnostics).
    Barrier-synchronized `concurrent_copies` runs isolated `timing.py` workers with an 85% available-memory guard.
    CLI flags `--peak-memory`/`--no-peak-memory`, `--concurrent-copies`/`--no-concurrent-copies`, and `--copy-rounds` populate peak-memory and concurrent-copy columns across speed, scaling, baseline, and `--compare` tables (`tests/test_benchmarks.py`).
- **Source:** rows B10 and B16.

### ✅ 7. Precision suite

- **What:** fixtures with independently computed exact answers (closed forms and high-precision linear solves), scoring each simulator by worst-point correct digits versus cost across tolerance sweeps as work-precision diagrams.
- **Result** (`3525ac5` to `273af4b`): 16 fixtures comparing jolty against ngspice Sparse and KLU; findings appear in §2.
- **Lead:** no reference publishes a work-precision diagram.
    **Source:** row B5; `research/accuracy.md` row 1.

### 📋 7b. Precision suite v2

- **What:** add nonlinear manufactured solutions using behavioral expressions with injected sources `b(t)` (after behavioral `q(v)` lands) plus IVP test-set circuits (ring modulator, transistor amplifier, NAND gate, and charge pump).
    Expand to at least 20 fixtures reporting each simulator's fastest runtime per accuracy level on a quiet host, plus a sweep lifting the maximum step cap.
- **Done when:** at least 20 fixtures, including the IVP test-set circuits, are timed on machine A.
- **Source:** rows B5 and B6; precision suite report.

### ✅ 8. Hard-circuit corpus

- **What:** run jolty's hard circuits and pinned-manifest third-party decks across every simulator, classifying outcomes (solved and right, alternate solution, failed, timed out, unsupported).
    Report separate DC and transient failure rates with Wilson confidence intervals per simulator.
- **Result** (`6905ddf` to `9d9eddf`): 48 cases with rates in §2; three SPICE reader faults were uncovered and fixed.
- **Lead:** no reference publishes failure rates on hard circuits.
    **Source:** row B7; `research/accuracy.md` row 10.

### 📋 8b. Hard-circuit corpus v2

- **What:** add Xyce and ngspice KLU runners, reference answers for external transient decks (currently counted simply as solved when completing), and more cases to narrow confidence intervals.
    Operating-point iteration counts for jolty are already recorded in the table since `090b521` (`agentic.md` 21).
- **Done when:** all additions appear in the corpus table.
- **Source:** corpus report.

### ✅ 9. Coverage matrix

- **What:** cover every analysis, device model, level, SPICE syntax feature, and output with a minimal known-answer circuit run in jolty, ngspice, and Xyce.
    Classify each cell as "agrees", "disagrees" (with numbers), "not offered" (with simulator diagnostics), or "documented" (citing manual sections for unavailable simulators).
- **Result:** version 1 (65 rows) landed on `main` on 16 November (`benchmarks/coverage`, via `scripts/bench --coverage`), with counts in §2.
    The `tline` probe was fixed to run `Transient` (agreeing to 4.4e-16 of scale), and `ngspice_only` now reads declared names instead of `()`.
- **Lead:** no competitor publishes an executable coverage matrix.
    **Source:** row B8.

### ✅ 9b. Coverage matrix v2

- **What:** add rows for corners, worst case, `.disto`, `.sp`, transient noise, `.four`, `.ic`, `.nodeset`, switches, logic, and PDKs.
    Resolve VBIC's unexplained 2e-6 closed-form gap, move BSIM3's probe to stronger bias, and verify the ngspice column in tests when ngspice is installed.
- **Done when:** every addition is reflected in the matrix.
- **Result** (18 November): expanded the matrix from 65 to 80 offline-safe rows (`corners`, `worst-case`, `disto`, `sp`, `trnoise`, `switch-voltage`, `switch-current`, `logic`, `kit-sky130`, `kit-gf180mcu`, `kit-sg13g2`, `ic`, `nodeset`, `pdk-library`, `four`).
    Refreshed live against ngspice 47 (`Sparse 1.3` and `KLU`) with XSPICE codemodels loaded.
    Tracing VBIC's 2.114e-6 gap showed VBIC 1.2 mandates default intrinsic resistances `RBI = RCI = RBP = 0.1 ohm`, dropping `Ib * RBI ~= 5.67e-8 V` (`-2.192e-6`).
    Additional offsets come from `q1` smoothing (`-2.50e-9`) and junction `gmin = 1e-12 S` (`+8.11e-8`).
    Exact coupled-KCL `vbic_currents()` replaces the naive Ebers-Moll estimate, matching jolty to `-0.13 ULP` (`gmin=0`) and both jolty and ngspice 47 to `8.1e-8` at default `gmin`.
    Using `same_kelvins(ROOM)` removes ngspice's 6.53e-6 Arrhenius offset.
    Shifting BSIM3v3.3 from weak inversion (`1.95 pA`, 77% `gmin`) to strong-inversion saturation (`3.0 V / 3.0 V`, `0.328 mA`) matches ngspice Sparse to `3.3e-16` in jolty and `0` in ngspice KLU.
    Subcircuit `.lib`/`.include` support flipped both `lib` and `include` from `unsupported` to `agreed` (`0` error).
    All 80 rows carry zero `unknown` cells, and all 65 prior `xyce-serial` cells remain unchanged.
    Live `ngspice-47` runs exposed three quirks: `four` requires `.control` command `fourier`, and `VSWITCH`/`ISWITCH` require whole-deck `ngbehavior=psa` with XSPICE codemodels.
    In addition, `.disto` crashes with a `malloc` underflow at `lin 1 1k 1k`, fails under KLU, and inverts `disto2` `HD3` (`-Vt/12` vs exact `+Vt/12`, marked `disagreed` under `ngspice-sparse`).
    Added gated `test_ngspice_column_is_what_its_fixture_finds` in `tests/test_coverage.py` (enforced when `JOLTY_REQUIRE_NGSPICE=1`).
- **Source:** coverage matrix report.

### 📋 10. Widened speed suite fixtures

- **What:** add a ring oscillator, transistor-level ISCAS'85 circuits, IBM power grids, and CircuitSim90 level-3 decks alongside existing fixtures, fetched via pinned manifest without committing third-party files.
- **Done when:** every new fixture is timed in jolty and ngspice on quiet machines.
- **Source:** rows B9 and B17.

### 🚧 11. Xyce in the suites

- **What:** integrate serial and MPI Xyce 7.10 at optimal configuration as a third simulator across the speed suite, robustness corpus, and coverage matrix.
- **State:** Xyce was built on machine B via Spack on 15 November; jolty's Xyce dialect, 67 agreement tests (`tests/test_xyce.py`), and the initial speed column are on `main` (§2).
    Coverage matrix Xyce probes share the same library.
    **Not done:** robustness corpus Xyce runner, equal-accuracy chain and mesh timings, `inverter_chain_50000`, and diagnosing the buck stall.
- **Buck handling** (owner, 15 November): retain the row marked stalled at 128.8 µs at Xyce defaults rather than dropping the fixture.
- **Done when:** Xyce columns populate every benchmark table.
    **Source:** row B13.

### ⏸️ 13. Ten-million-device fixture

- **What:** build a fixture such as the scaling blueprint's crossbar transformer and measure build, elaboration, operating point, transient runtime, and peak memory.
- **Parked until:** the front end supports constructing ten-million-device circuits (`speed.md` 11).

**Elsewhere:** end-to-end agent command timing is tracked in `speed.md` item 8.

## 5. Decided

- 15 November (owner): reference simulators are ngspice and Xyce exclusively, with no LTspice or commercial simulators.
- **Rules for speed claims** (15 November, from `research/speed.md` §6): run every reference at its best configuration (ngspice with KLU and Sparse 1.3; Xyce with its native solver).
    Report both engine time and end-to-end time alongside reached accuracy for every simulator.
    Measure parallel efficiency against attainable throughput (the sum of single-thread rates per core class) with the host machine stated.
    Name an actual fixture of matching size for every scaling claim, and label all extrapolations explicitly as estimates.
- The break-even tool (`scripts/bench --break-even`) remains in the repository so thresholds can be re-measured at any time.
- 15 November: precision suite outputs stay out of the repository (like other cross-simulator comparisons) pending owner decision, whereas the coverage matrix table is committed as the suite's deliverable.

## 6. Open principles

None; the research proposals were merged into `AGENTS.md` on 15 November (`444535f`).
