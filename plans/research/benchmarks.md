# Research: benchmarks

The evidence for `benchmarks.md`: suites anyone can repeat that measure coverage (which analyses and models each simulator has), precision (each one's error against exact answers) and speed (each one's time per fixture, on one core and on all of them), against the established simulators.
Written on 15 November 2026, to a brief, not kept, from the repository, the notes, ngspice 47's and Xyce's sources (`e5b7b9b`), and published work.
`main` moved from `d10fc4d` to `7d41cc0` while this was written, VBIC, Phase 6's four commits and the parallel benchmarks' three landing (`1bc7dab`, `a93a9f9`, `7d41cc0`: `scripts/bench --break-even` and `--scale` spread over threads); line numbers of `perf/` are `7d41cc0`'s, the checkout's later `9545e08` changing nothing there.
Read only: nothing was built, run or edited in the repository.
The only programs run were `git log` and `git diff`, `grep`, `strings` on the installed ngspice, the GitHub API to list a public repository, and the scratchpad's PDF text extractor on papers downloaded into `scratchpad/bench-research/`.
Every number is a measurement already recorded, a count read from code or a paper, or an estimate marked as one.
The owner approved, during the research, the lab's shared workstations for benchmarks: machine A and machine B, reachable now, with ngspice 47 being built on machine A with Homebrew's configuration (the coordinator's message).

## 1. Summary

1. jolty times 8 fixtures in Rust, in Python and in ngspice, and records each timing's load, but the protocol cannot yet support its claims: timings spread up to 9.8% and 19% where changes are flagged at 5%, large fixtures run once, the suite's baseline is not interleaved, and no timing carries an interval or the accuracy it reached.
1. Every comparison so far is with ngspice 47 at its defaults, Sparse 1.3 rather than its KLU, its process start and parsing timed against jolty's in-process engine, on a Mac shared at load averages of 2 to 19; the quiet machines now exist, machine A and machine B, approved by the owner.
1. No suite measures precision against exact answers, robustness on hard circuits, or coverage by running anything; established material exists for each: the IVP test set's five circuit problems with published references, MCNC's CircuitSim90 suite (43 decks now in Xyce's public regression repository, GPL), Barby and Guindi's CircuitSim93 methodology, CMC QA tests, IBM's power grids with their solutions.
1. None of the references publishes a work-precision diagram for a circuit simulator, a failure rate on hard circuits, or a coverage matrix checked by running: jolty can lead on all three.
1. Move 1: compare like with like: ngspice with KLU, engine and end-to-end times, interleaved runs with confidence intervals, each run's error against a reference, and the same solution checked before any time is compared.
1. Move 2: a precision suite of exact answers (closed forms, linear circuits solved exactly, manufactured solutions, the IVP circuits), reported as work-precision diagrams; jolty's errors and work counts are deterministic, so they can gate CI where timings cannot.
1. Move 3: a corpus of hard circuits, fetched at a pinned commit and never copied, with each simulator's outcome classified, and a coverage matrix whose every cell is a probe that ran.

## 2. Where jolty stands

### 2.1 The fixtures

`perf/` holds 8 fixtures, each a circuit and a plan written by a script (`perf/README.md:30–39`):

| Fixture | Size | Devices | Analyses | Timed by |
| --- | --- | --- | --- | --- |
| `bjt_amp` | a few dozen unknowns | 2N3904s, 2N3906s (Gummel-Poon) | operating point, 71 frequencies, 200 µs transient | the suite |
| `ac_wide` | the same | the same | 10,000 frequencies | the suite |
| `buck` | 23 unknowns (`research/speed.md` §2.1) | level 1 MOSFETs, L, C | 2 ms transient | the suite |
| `inverter_chain_500`, `_5000`, `_50000` | 3,504, 35,004, 350,004 unknowns | level 1 MOSFETs with Meyer displacements | 40 ns transient | the suite; `--scale` |
| `rc_mesh_100`, `_300` | 10,000 and 90,000 nodes | R, C, a pulse | 10 µs transient | the suite; `--scale` |
| `monte_carlo` | `bjt_amp` | 5% resistors | 100,000 samples; 1,000 in the suite | the suite; `--scale` |

They exercise 3 device families of jolty's 26 analog models and 4 of its analyses (operating point, AC, transient, Monte Carlo); no fixture times a DC sweep, noise, pole-zero, a transfer function, sensitivities, stability or a periodic steady state, and none uses BSIM3, VBIC, VDMOS, a JFET or a behavioral source.
Every transient runs the trapezoidal rule at default tolerances (`perf/plans/*.json`).

### 2.2 The protocol

- **Rust:** criterion's median of 10 samples over 5 s after a second of warming up, for `setup`, `shared` and `run` (`perf/README.md:44–56`, `crates/jolty-engine/benches/fixtures.rs`).
- **Python:** the median of at least 10 runs over at least 5 s after one warm-up, wall time by `perf_counter`, and the load average over the last minute as the runs began (`perf/bench.py:148–151`, `197–206`, `297–311`).
- **ngspice:** `ngspice -b` from start to exit on the netlist jolty's exporter writes, its analyses in a control section, the transient as `tran stop/50 stop 0 stop/50` (`bench.py:534–604`); the exporter writes no `.options`, so ngspice runs its defaults, and its solver is Sparse 1.3, as KLU is off unless asked (`cktntask.c:142–143`).
- **At scale:** each large fixture runs once serially, then once spread on 1, 2, 4 and 8 threads and every core with each threshold at its break-even (`perf/README.md:182–224`, `bench.py:100–109`, `331`).
- **Break-evens:** `--break-even` runs a threshold unset and set in alternating rounds, at least 3 and 5 s for each side, the median of its rounds beside the load as they began and ended (`bench.py:155`, `871–890`; `perf/README.md:122–180`).
- **Results:** per timing its median, its number of runs and its load, beside the commit and the machine: processor, kinds of core, memory, system, Python, rustc, and the spread runs' fingerprints (`bench.py:634–680`, `1072–1083`).
    No dispersion is stored.
- **Comparison:** each median against the newest ancestor's results, flagged beyond 5% (`bench.py:687–744`; AGENTS.md:99).
- **Work:** one more run with `profile` on records each transient's Newton iterations, factorizations and the shares of its time (`bench.py:422`).
- **Integrity:** the suite refuses fixtures whose files are not what their scripts write (`bench.py:226`); `--scale` hashes each run's results with SHA-256 and stops if the bits differ between thread counts (`7d41cc0`), a check no reference simulator could pass.

### 2.3 How sound it is

1. **Repeatability is below the flag.**
    Six runs over an hour spread by up to 5.6%, `rc_mesh_100`'s Rust run by 9.8% and the Monte Carlo samples on every core by 19% (`perf/README.md:86–88`), against Phase 0's target of ±3% (`many-cores-implementation-plan.md:328`) and a flag at 5%.
1. **Intervals within a run understate the variation between runs.**
    At `6fe8824`, criterion's interval for `rc_mesh_100/run` was 756.1–773.8 ms, ±1.2%, yet it reported +8.2% against its previous run (p = 0.00), while the Python timing of the same fixture was 5.5% faster than the baseline (raw logs, not kept).
    Kalibera and Jones (2013) show why: variation enters at each level, iterations, process invocations and builds, and an interval over iterations alone ignores the others.
1. **The suite's baseline is not interleaved.**
    Its comparison reads a file measured at another time and load (`bench.py:687–744`); the break-even sweep alternates its two sides (`bench.py:871–890`), and even so "the load changed from round to round, by a few percent of a run", leaving a break-even of a few percent "uncertain by a size of the sweep" (`perf/README.md:178`).
1. **One run at scale.**
    Every large time is one run, ngspice's `rc_mesh_300` at 857.4 s among them (`perf/results/798b73d….json`), and every spread run of `--scale`; on 12 threads both large fixtures ran slower than on 8, "so a quiet machine's times are the ones to judge the scaling by" (`perf/README.md:224`).
1. **The table against ngspice mixes two sessions.**
    jolty's times came from a scratch binary at `f99a247` at load 5–7, ngspice's from the `798b73d` baseline's run at load 4–6 (raw logs, not kept).
1. **Unequal configurations.**
    ngspice ran Sparse 1.3, not KLU; the installed ngspice, Homebrew's bottle, is built `--disable-openmp` (Homebrew's `ngspice.rb`), which matters for BSIM3 but not for these fixtures' level 1 and Gummel-Poon devices (`research/speed.md` §3.2).
1. **Unequal scopes.**
    ngspice's time includes its process start, its code models' loading (`src/spinit.in:19–25`) and its parsing; jolty's Rust `run` starts from JSON text in its own process; `bjt_amp`'s 0.81 ms against 17.5 ms compares different things.
    ngspice keeps its own analysis time apart from its process time (`Total analysis time`, `Transient analysis time`, load, factor and solve times, `cktsopt.c:324–339`), and the suite does not read it.
1. **Unequal, unmeasured accuracy.**
    At defaults ngspice keeps 52,982 points of the buck where jolty accepted 122,587 steps at the baseline and 73,880 now (`perf/README.md:115`, raw logs, not kept); their tolerances mean different things (`research/accuracy.md` §2.8); no fixture has a reference that says which result is more accurate.
1. **Mixed cores.**
    "Every core" is `os.cpu_count()`, 12 threads over 8 performance and 4 efficiency cores (`bench.py:324`), and macOS gives no way to pin a thread (§3.5).
1. **Two builds.**
    The benchmark's build and the Python extension's differ by 6–9% on the buck (`perf/README.md:90–91`).
1. **A shared machine.**
    Recorded load averages run from 1.9 (`final-bench`) to 19 (the break-evens, `benchmarks.md` §2).

### 2.4 The differential corpus

- **What it compares** (`tests/test_ngspice.py`, 2,947 lines): the operating points of 28 circuits, DC sweeps of 27, AC responses of 16, noise of 9, transfer functions of 5, poles and zeros of 6, transients of 22, one loop gain; BSIM3's curves, AC, its six noise models, transients, published and binned cards; VBIC's curves, AC, noise, transient and self-heating; vendors' VDMOS cards.
    A run at `87df77f` passed 331 tests in 4.10 s (raw logs, not kept).
- **Its tolerances:** DC and AC within 1e-6 of the signal, transients within 1e-3 of the swing inside a step's window, noise, transfer functions within 1e-5 relative, poles and zeros 1e-6 (`test_ngspice.py:1564–1910`, `differential.py:87–122`).
- **Its options:** both engines at `TIGHT`, `reltol` 1e-6 and `chgtol` 1e-18 (`differential.py:52–57`, `src/jolty/spice/ngspice.py:37–40`), so it never sees either engine's defaults.
- **Where it runs:** CI on Ubuntu 24.04 and macOS 15 with Homebrew's ngspice 47, required by `JOLTY_REQUIRE_NGSPICE=1` (`.github/workflows/ci.yml:45–66`); beside it, vendors' macromodels, random circuits of up to 100 unknowns and metamorphic variants (AGENTS.md:119–127).
- **What it gives the suites:** a tested translation from jolty's circuits to ngspice's netlists, and a set of circuits where both engines' equations agree to 1e-6, the condition for a speed fixture that is fair to both.
    It measures agreement, not accuracy.

### 2.5 Precision, robustness and coverage today

- **Precision:** closed forms in the engine's tests, an RC under a ramp for every method, each method's order, an LC tank, noise, poles and zeros, the periodic steady state (`crates/jolty-engine/tests/analytic.rs`, `noise.rs`, `pole_zero.rs`, `periodic.rs`).
    They are pass-or-fail thresholds; none reports error against cost, and no other simulator runs them.
- **Robustness:** convergence tests (`tests/convergence.rs`), and random circuits that fail when ngspice finds an operating point and jolty none (`research/accuracy.md` §2.2).
    No corpus and no failure rate.
- **Coverage:** a hand-written table of analyses for jolty, ngspice and Xyce (`research/models.md` §3.8) and one of compact models (§3.5), from manuals and sources; nothing executed.

### 2.6 The machines

- **This Apple M2 Max:** 8 performance and 4 efficiency cores, 32 GB, shared with other agents' work.
- **Approved for benchmarks by the owner, reachable now** (the coordinator's message, 15 November): machine A, an AMD Threadripper 3970X with 32 cores and 64 threads, and machine B, an AMD Ryzen 9 7950X with 16 cores and 32 threads, both Linux, shared with the lab (`benchmarks.md` §2).
    ngspice 47 is being built on machine A with Homebrew's configuration, which means KLU compiled in and OpenMP off (§3.6).
- **Not yet:** machine C; a PBS cluster, needing the owner's login first.

## 3. The references

### 3.1 Coverage: what each simulator offers

**Analyses.**
Cells for jolty, ngspice and Xyce are from `research/models.md` §3.8 and the sources below; LTspice's from Analog Devices' list of its directives; Spectre's from its Reference, version 23.1 (September 2023); HSPICE's from the text that could be extracted from Synopsys's quick reference of June 2015.
"None found" means the source cited lists none, not that none exists.

| Analysis | jolty | ngspice 47 | Xyce | LTspice | Spectre 23.1 | HSPICE (2015) |
| --- | --- | --- | --- | --- | --- | --- |
| Operating point | yes | yes | yes | yes | `dc` | `.OP` |
| Every DC solution | deflation | none | follows solutions by arc-length continuation | none found | unknown | unknown |
| DC sweep | one source | two sources, temperature, resistance | yes | up to three sources | `dc`, `sweep` | `.DC` |
| AC, noise | yes | yes; noise not with KLU (`noisean.c:74`) | yes | yes | `ac`, `noise` | `.AC`, `.NOISE` |
| Transient | yes | yes | yes | yes | `tran` | `.TRAN` |
| Transient noise | no | from sources | unknown | unknown | unknown | `.TRANNOISE` |
| Transfer function | `.tf` | `.tf` | none found | `.tf` | `xf` | `.TF` |
| Poles and zeros | yes | yes, "not free of bugs", not with KLU (`pzan.c:31`) | none | none found | `pz` | not in the extract |
| Loop gain | Middlebrook's | none | none | `.fra`, through a transient | `stb` | `.LSTB` |
| Sensitivities | DC, AC, transient, exact | DC, AC, by finite differences | DC, AC, transient, direct and adjoint | none found | DC and AC (datasheet) | not in the extract |
| Periodic steady state | shooting | experimental | harmonic balance, MPDE | none found | `pss`, `hb`, `qpss` | `.SN`, `.HB` |
| Periodic small-signal | no | no | unknown | no | `pac`, `pnoise`, `pxf`, `psp`, `pstb` | `.HBAC`, `.HBNOISE`, `.SNAC`, `.SNNOISE`, `.SNXF` |
| Distortion | no | `.disto`, ten SPICE3 models | through harmonic balance | `.four` after a transient | through `hb` | not in the extract |
| S-parameters | no | `.sp` | `.LIN` | `.net` (older manual, unconfirmed) | `sp` | `.LIN` |
| Monte Carlo, corners | yes, the same bits on any shard | its control language | `.SAMPLING`, `.EMBEDDEDSAMPLING`, `.PCE` | `.step` | `montecarlo`, `sweep` | `.VARIATION`, `.ALTER`, `.DATA` |
| Mismatch from one solve | no | no | no | no | `dcmatch`, `acmatch` | `.DCMATCH`, `.ACMATCH` |

- **Siemens AFS**, by its fact sheet and white paper: a circuit simulator, transient noise, and a harmonic-balance family (HB, HBAC, HBNOISE, HBSTB, HBSP, HBXF; one tone for autonomous circuits, two to four for driven ones).
    Its other analyses are not listed there.
- **Synopsys PrimeSim:** HSPICE, and FastSPICE engines whose accuracy is a setting (`runlvl`, `research/speed.md` §3.5).
- **Qucs-S** is a front end, not a simulator: it drives ngspice, Xyce, SPICE OPUS and Qucsator-RF, offers S-parameters on the three SPICE engines and harmonic balance through Xyce and Qucsator-RF (its documentation).
    Only Qucsator-RF would add an engine to the comparison.
- **Models:** `research/models.md` §3.1–3.5 counts them: jolty 26 analog models natively and VBIC; ngspice 57 device directories and any Verilog-A model through OSDI; Xyce its C++ models and 25 ADMS models; LTspice ten MOSFET levels from 1 to 14, BSIM3v3.3.0 and BSIM4.6.1 among them, and no Verilog-A; Spectre BSIM, PSP, HiSIM, VBIC, HICUM, MEXTRAM, BSIM-CMG and UTSOI built in.

**Where the published lists fall short.**
They are documents, not tests: a feature listed may be partial (ngspice's distortion covers ten models, its KLU refuses noise and pole-zero), versions change it (ngspice's KLU arrived as an option in version 42 and reached CIDER's devices in 46, its `NEWS:172–175` and `NEWS:31–35`), and nothing says whether a listed feature gives the right answer.
Xyce's 2008 log of CircuitSim90 shows decks within Xyce's documented features that still fail (§3.4).

**A coverage matrix that is fair and checkable**, as this research would define it:

- **A row is a capability a user can request**, at the granularity of its options: "AC sensitivity with respect to every parameter", "BSIM4 4.5 with `capmod` 2", not "sensitivity" or "BSIM4".
- **A cell is a probe that ran:** the smallest deck exercising the row, with a known answer (a closed form, or the defining implementation's output, such as a CMC QA reference), run in each installed simulator, the cell's status one of a fixed vocabulary: right, wrong, refused, absent, through a plug-in (OSDI, ADMS, Verilog-A), documented only (for a simulator not run, with its document's version and page), unknown.
- **The same deck for every simulator**, translated mechanically into each dialect by a recorded script, as CircuitSim93 generated its netlists from the originals with `sed` and `awk`, so that "it is a simple process to show that the netlists used by each simulator are equivalent" (Barby and Guindi 1993).
- **No weights and no total:** counts per category, and for each simulator the list of rows another has and it lacks.
    A weighting would be an opinion; the counts are facts.
- **Versioned:** the matrix regenerates for each release of each simulator and its diff shows what changed.

### 3.2 Precision: established answers, and how error against cost is shown

**The Test Set for IVP Solvers** (Lioen and de Swart, CWI report MAS-R9832, release 2.0, 1998; continued by Mazzia and Magherini at Bari, release 2.4, 2008) is the established precision benchmark of stiff ODE and DAE codes, and five of its problems are circuits, each with a reference solution at the end of its interval:

| Problem | Type, size, index | Devices | Reference |
| --- | --- | --- | --- |
| Ring modulator | ODE, 15 | diodes `γ(e^{δU} − 1)`, R, L, C, two sines | PSIDE, `atol = rtol = 1e-13` |
| Transistor amplifier | DAE, 8, index 1 | exponential transistors, R, C | PSIDE on a Cray C90 in Cray double precision, 1e-14 |
| NAND gate | IDE, 14, index 1 | two enhancement MOSFETs and a depletion one, nonlinear capacitances | PSIDE, Cray double precision, 1e-16; only node 5 scored |
| Charge pump | DAE, 9, index 2 | a MOS transistor as four charges | PSIDE at `Tol = 2e-8`; only the first component scored |
| Two bit adding unit | DAE, 350, index 1 | MOS gates | RADAU5 at `rtol = atol = 1e-5` only, without restarts at the inputs' corners |

(Figures read from the CWI report's text.)

- **Its measure:** significant correct digits, `scd = −log10 ‖relative error at the end of the interval‖∞`, and at Bari also a mixed absolute-relative `mescd`.
- **Its diagrams:** each solver's `scd` against its CPU time, over a sweep of tolerances such as `rtol = 10^−(4+m/4)`, `m = 0 … 24`, `atol = rtol`, each time the minimum of 10 runs, in the format of Hairer and Wanner (1996, pp. 166–167, 324–325).
- **Its caveat**, which any such suite must repeat: "the diagrams just show the result of runs with the prescribed input on the specified computer"; other inputs, computers or ranges of tolerance "might change the diagrams considerably".
- **Where it falls short for circuits:** it scores the end point alone; two of its five references are weak (the adder's 1e-5, the charge pump's 2e-8); it uses its own device equations, not SPICE models; and no circuit simulator's results are published on it, as far as this research found.

**Other sources of reference answers:**

- **CMC QA tests:** each standard model's `qaSpec` names DC, AC, temperature and self-heating tests, with reference outputs printed to 7 significant digits (VBIC 1.3's `qaSpec` and `reference/*.standard` in Xyce's `utils/ADMS/examples/`).
    They measure agreement with the defining implementation, which `research/models.md` item 8 already plans to harness.
- **Xyce's regression suite** (`github.com/Xyce/Xyce_Regression`, about 270 entries under `Netlists/`, GPL-3.0-or-later) holds analytic tests checked by gold-standard scripts: `DIODE_ANALYTIC`, `NMOS_ANALYTIC`, `BJT_ANALYTIC`.
- **IBM's power grid benchmarks** (Nassif, "Power grid analysis benchmarks", ASP-DAC 2008, hosted at UC Santa Barbara): SPICE netlists from about 30,600 nodes (`ibmpg1`) to 1.67 million (`ibmpg6`), each with a `.solution` file, and transient variants under pulsed currents.
    Their licence was not found.
- **Closed forms** for any analysis of small linear circuits, and **linear circuits of any size solved exactly**: a linear circuit's response to exponential, sine and piecewise-linear sources is a finite sum of matrix exponentials, which arbitrary-precision arithmetic evaluates to any number of digits.
    No published circuit benchmark uses this, as far as this research found.
- **Manufactured solutions** (Roache 2002; Salari and Knupp 2000): choose the waveforms `x*(t)`, inject `b(t) = f(x*) + d q(x*)/dt` as sources, and `x*` is the exact solution.
    No study applying it to a circuit simulator was found.

**How work-precision diagrams are made elsewhere.**
Beside the test set, the SciML benchmarks build their references with an adaptive solver at `abstol = reltol = 1e-14` (`1e-12` for some), sweep tolerances by powers of ten, and score the final value, the error at the solver's own steps or an interpolated L2 error (SciMLBenchmarks; DiffEqDevDocs, "Benchmark Suite").
Neither shows a circuit simulator; VACASK, the one open comparison of circuit simulators found, matched tolerances so that each simulator produced "roughly equal" time points rather than measuring each one's error (§3.3).

**What jolty can do beyond them:** score every point a simulator produced against a dense reference evaluated at that point, which needs no interpolation of the simulator's output and judges the whole run, not its end; and publish the diagrams for SPICE-class simulators, which no one does.

### 3.3 Speed: suites, and how simulators are compared

**Established suites:**

- **Quarles' SPICE3 benchmarks** ("Benchmark Circuits: Results for Spice3", UCB/ERL M89/47, 1989), listed by ngspice's benchmark page; `tests/general/` in ngspice's tarball holds classic circuits of that era with recorded outputs: `diffpair`, `fourbitadder`, `mosamp`, `mosmem`, `rc`, `rca3040`, `rtlinv`, `schmitt`.
- **CircuitSim90** (MCNC, W. R. Richards, 1992), "one of the very few de facto standard circuit simulator benchmark suites that is in the public domain", made of "public domain circuits from MCNC and Sandia Labs", a subset of MCNC's own regression tests (Barby and Guindi 1993); "a sort of 'industry standard' test suite for spice simulators" (ngspice's benchmark page).
    Its decks are in Xyce's regression repository at `Netlists/CircuitSim90/`: 1 bipolar (`vreg`), 18 of MOS level 2 (`ab_ac` … `toronto`), 14 large ones of level 2 (`add20`, `add32`, `chip2`, `dac`, `fadd32`, `mem_plus`, `pc_frame`, `pchip`, `ram2k`, `smult20`, `sqrt`, `sram`, `voter`, `voter25`) and 10 of level 3 (`arom`, `gm1`, `gm2`, `gm3`, `gm17`, `gm19`, `jge`, `mike2`, `rich3`, `todd3`), with the CircuitSim93 paper beside them (listed through GitHub's API at revision `7bb7e98`).
    Their largest, `chip2`, has 46,850 unknowns and 18,816 MOSFETs (Xyce's log, §3.4).
- **ISCAS'85 at transistor level:** ngspice ships `c432` to `c7552` as netlists generated by `spicegen.pl` (Jingye Xu, UIC) over PTM's 45 nm BSIM4 4.7.0 cards, in `examples/klu/Circuits/`.
    ngspice reports KLU's gain on `c7552`, 15,000 MOSFETs and 82,000 passives: "overall speedup ...
    a factor of two", the transient's "reduced by a factor of three" (ngspice's applications page); the KLU slides of Lannutti and Perticaroli (MOS-AK 2012) report up to about 100× on the linear solves and about 11–13× overall on the ISCAS'85 suite (read through search results; the slides' text could not be extracted).
- **VACASK's `c6288`** (Bűrmen; its README, AGPL-3.0): a 16 × 16 multiplier in PSP 103.4, 10,112 transistors and 25,380 nodes, on a Threadripper 7970, every simulator on KLU, OpenMP off, output off, ngspice's and Xyce's tolerances "set so that the number of output time points is roughly equal": VACASK 57.98 s, ngspice 71.81 s, Xyce 151.57 s, at 3,474–3,559 Newton iterations.
    Its own developers' benchmark; no versions, repeat counts or statistic given.
    A slide of the same author, read through search results, lists a ring oscillator with "Xyce 9.39" and "Xyce-fast 4.12"; what the fast build changes is unknown.
- **IBM's power grids** (§3.2): the established large linear benchmark, with solutions.

**How the comparison was done properly once.**
Barby and Guindi (University of Waterloo, "CircuitSim93: a circuit simulator benchmarking methodology case study", IEEE, 1993, pp. 531–535) benchmarked HSPICE H9007D and H92A, Saber 3.1a and 3.1d, and Spectre 4.2 and 4.2.1a on CircuitSim90:

- **Eight criteria:** user, system and elapsed time; swap, resident memory and temporary disk; page faults; I/O; "accuracy/correctness of response"; calibration effort; porting effort; parameter checking.
- **Equivalent netlists:** each simulator's generated by scripts from the untouched originals, a paper trail checked by `diff`, with the same node and instance names, since name lengths change memory and time.
- **Broken decks fixed by rule, not by hand:** nodes with one connection or no DC path got a 1 TΩ resistor to ground, unconnected nodes were deleted, duplicate names renamed; oscillators started with every source ramped from zero ("trip=zero").
    Spectre found every floating node "in a couple of seconds" at compile time, where HSPICE and Saber took "thousands of seconds per run" to find them one at a time.
- **Its finding:** "if a circuit had three possible operating points, the three simulators ...
    would all converge to different ones", which "invalidat[ed] the cpu performance measurements", so "each simulator would have to be calibrated for each circuit"; one CPU-month of a SPARC 10 just to rerun the timings, the calibration left for a second phase that this research did not find published.

**How vendors compare:** against their own earlier products, never against each other's: Spectre X against Spectre APS, its presets from CX, the most accurate, to VX, the fastest, mapped by Cadence onto APS's `errpreset` (Cadence's blog; LX and VX "not recommended" in its ADE guide); AFS against "traditional SPICE"; PrimeSim on GPUs against CPUs (`research/speed.md` §3.2, §3.9).
Their licences explain it: Synopsys's evaluation terms bar disclosing to a third party a comparison of its products' results with others', and EDA licences commonly forbid publishing benchmark results (Bier, BDTI, 2008; a vendor's chief executive in an open letter on Sigasi's site; posts on SemiWiki).
A third party cannot publish a comparison with Spectre, HSPICE or AFS without the vendor's consent.

**Which fixtures are fair to every simulator:**

- **The same equations:** devices every simulator models alike (R, C, L, sources, the diode, Gummel-Poon, MOSFET levels 1 and 3, BSIM3v3.3), checked at tight tolerances as the differential tests check jolty against ngspice; where formulations differ (Meyer's capacitances, `research/accuracy.md` §2.5), each simulator's error is measured against its own converged answer, or the fixture is marked unfair.
- **The same solution:** CircuitSim93's lesson; a timing counts only where the solutions agree.
- **Each simulator at its best:** ngspice with KLU and, for BSIM, OpenMP; Xyce with KLU serially, or its parallel build; jolty with its `Parallelism` defaults once they exist (the lead's decision, `benchmarks.md` §5).
- **The same outputs:** the same quantities saved at each simulator's own points, or none, as VACASK turned output off.
- **Stated accuracy:** each run's error against the reference, or the whole work-precision curve.
- **Every size and kind:** small analog (`bjt_amp`), switching (`buck`), logic (chains, ISCAS'85), linear and large (meshes, power grids), autonomous (ring oscillators), many runs (Monte Carlo).

**Engine time against end-to-end time.**
Three scopes, each measurable in every open simulator:

- **Engine:** the analyses alone, from the simulator's own counters: jolty's profile; ngspice's `Total analysis time` and its parts (`cktsopt.c:333–339`, printed with `.options acct` or `rusage`); Xyce's "Total Simulation Solvers Run Time" beside its "Total Elapsed Run Time" (`N_CIR_Xyce.C:2071–2072`).
- **In process:** parsing or reading, elaboration and the analyses, as jolty's Rust `run` and Python timings are.
- **End to end:** from the command to its exit, as an agent's command sees it: `jolty simulate` against `ngspice -b`, `Xyce`, LTspice's batch mode; jolty's command starts Python afresh and draws plots (`research/speed.md` §2.1), unmeasured.

### 3.4 Robustness: hard circuits and failure rates

- **CircuitSim90 is a hard suite by reputation and record.**
    Xyce's own log of its large decks (Eric Keiter, 19 January 2008, `Netlists/CircuitSim90/MOS2_LARGE/README` in Xyce's regression repository): `add20` fails its operating point by Newton, works with gmin stepping, then stops on "time-step-too-small" halfway; `dac` "only works if GMIN stepping is used"; `pchip` stops on a step too small; `smult20` and `sram` fail their operating points; `pc_frame` does not parse; `chip2` runs about 9 hours serially, `voter` about 4, `ram2k` 90–100 minutes.
- **CircuitSim93** needed per-pair options to make some decks run: ramped starts for HSPICE on `dac`, `smult20` and `chip2`, a nodeset for Spectre on `chip2`, a reduced step and another DC method for Saber on `vreg`, `bias` and the Schmitt triggers (Barby and Guindi 1993).
- **spice-ts**, a SPICE-style simulator checked against ngspice 47, publishes a licence-checked corpus drawn from Xyce's regression decks, CircuitSim90 among them, each deck pinned by URL, commit, SHA-256 and SPDX licence, each outcome success, failed or unsupported with matched-point errors, and "losses are published, not hidden"; one batch of 24 decks: 2 matched, 5 mismatched, 2 not compared, 15 losses (its issues #1045 and #1059, and `benchmarks/SOURCES.md`).
- **The homotopy literature** uses particular hard circuits, Schmitt triggers among them, where stepping "can fail at critical values of the continuation parameter" (Roychowdhury and Melville, DAC 1996), and counts their solutions (a Schmitt trigger with three, Trajković's group at SFU), but reports no failure rates across simulators.
- **What was not found:** any published failure rate of any simulator on any corpus.
    `research/accuracy.md` item 10 reached the same conclusion.

### 3.5 Method

**Statistics for timings on a shared machine:**

- **Hoefler and Belli** (SC15, 2015) give twelve rules; those that bear here: report the base case of a speed-up and its absolute time (1); report all results, not just the best (2); summarize costs, not ratios (3, 4); state whether measurements are deterministic and give confidence intervals otherwise (5); do not assume normality (6); compare by non-overlapping intervals or a test (7); document every varying factor (9); show upper bounds (11).
    For the median they give Le Boudec's interval by ranks, from rank `⌊(n − z√n)/2⌋` to `⌈1 + (n + z√n)/2⌉`, `z = 1.96` for 95%.
- **Kalibera and Jones** ("Rigorous benchmarking in reasonable time", ISMM 2013): repetitions are needed at each level, iterations, process invocations, builds, in numbers their cost model sets; results are effect sizes with intervals.
- **Chen and Revels** ("Robust benchmarking in noisy environments", arXiv 1608.04295, 2016): environmental noise only delays a run, so the minimum is the robust estimator of a deterministic workload's time, where the median can split between the modes of a bimodal distribution; used in Julia's continuous integration.
- **Counted work instead of time:** the Rust compiler's performance site defaults to `instructions:u`, because instruction counts catch changes of about 0.3% where wall time hides changes under about 1%, while cycles stay as noisy as time (Rust internals forum, 2019).
    jolty has a stronger version: its Newton iterations, steps and factorizations are the same bits on a platform for any thread count (AGENTS.md:94), so they are exact regression signals.
- **Interleaving:** alternating the two sides of a comparison round by round makes both share whatever load the machine carries, as `--break-even`'s sweep does (`bench.py:871–890`).

**Quiet machines:**

- **On Linux (machine A, machine B):** a known load before and after (`uptime`, and `who` for other users), threads pinned with `taskset` or `numactl`, the frequency governor and boost recorded (setting them needs root), `perf stat` for instructions and cycles.
    - machine A's 3970X has four chiplets of two four-core complexes, each complex with 16 MB of L3 (ServeTheHome's review): scaling past 4, 8 and 16 threads crosses caches, and past 32 shares cores between two threads, so each step is measured, not interpolated.
    - machine B's 7950X has two chiplets of eight cores, each with 32 MB of L3, and the second chiplet's cores boost 100–250 MHz lower than the first's (TechPowerUp's review): its 16 cores are not quite alike either, and a run on 8 threads may stay within one chiplet's cache where 9 cannot.
    - Both are shared with the lab: a run starts only below a stated load, records the load as it ends, and alternates its sides, as the statistics above ask; a night's runs repeat on a second night before a figure is taken.
- **On a PBS cluster:** whole nodes by exclusive placement, as PBS Professional offers it (its `place=excl`, recalled from its documentation and not checked against the cluster's configuration), each job recording its node and processor; whether the cluster grants exclusive nodes is unknown.
- **On this Mac:** macOS has no thread affinity on Apple silicon; placement follows quality of service, and background threads are confined to the efficiency cores (Eclectic Light's measurements, Apple's developer forums), and `taskpolicy -c background` clamps a whole process to that quality of service (its manual page).
    macOS may also leave performance cores idle and raise an efficiency cluster's frequency when it carries high-priority work (same sources).

**Parallel efficiency on mixed cores** (`research/speed.md` §2.3; the lead decided on attainable throughput, the sum of each core class's single-thread rate):

- **Measure the ceiling directly:** run N independent single-threaded copies of the same fixture at once, as SPEC's rate metrics run concurrent copies (SPEC CPU 2017's documentation), and take their aggregate throughput as what N threads can attain.
    It includes the all-core frequency, the shared caches and memory bandwidth, needs no pinning, and covers SMT on machine A (64 copies) and machine B (32), and machine B's two chiplets of unequal boost.
- **Report both:** the speed-up against one thread, with its absolute time (rule 1), and the efficiency against N copies; for interpretation, each class's single-thread rate: the efficiency cores' under `taskpolicy -c background`, the performance cores' with at most 8 busy threads.
- **Monte Carlo is that ceiling's own workload:** each sample runs on one thread (`docs/guide/simulation.md:274–297`), so its efficiency against N copies of one sample's run is exactly the many-runs measure.
- **A caveat:** copies of a memory-bound fixture contend for bandwidth more than one shared run does, so an efficiency above 100% is possible and is reported as found.

### 3.6 What can run here, and under what licence

| Tool | Licence | This Mac | machine A, machine B (Linux) | CI | What it needs from the owner |
| --- | --- | --- | --- | --- | --- |
| ngspice 47, Homebrew | BSD (modified) | Installed; KLU compiled in (its binary's messages, `configure.ac:195–196` enables KLU by default); OpenMP off (`ngspice.rb`) | Being built on machine A with Homebrew's configuration, so KLU on and OpenMP off | Installed by CI on both systems | Nothing for KLU: `.options klu` (`cktsopt.c:371–377`) |
| ngspice 47 with OpenMP | BSD | A source build (OpenMP on by default, `configure.ac:1321`) | A second build beside the Homebrew-configured one, its flags recorded | No | Nothing: a separate program, needed only for BSIM3 fixtures |
| Xyce | GPL-3.0-or-later | No Homebrew formula; a source build needs Trilinos, SuiteSparse's AMD, bison, flex, BLAS and LAPACK (`INSTALL.md`); Sandia publishes macOS, Windows and Red Hat installers (`INSTALL.md:18`) | Installer or source | Too long a build for CI; a cached build possible | The word to install it (the session log, not kept); a separate program, as AGENTS.md:70–71 requires |
| LTspice | Analog Devices' freeware, proprietary; its licence text was not read | Homebrew cask `ltspice`, native, with fewer command-line options than the Windows build (spicelib's documentation) | The Windows build under Wine | No | The word to install it, and a reading of its licence for any limit on publishing results |
| Qucs-S | GPL (version not checked) | A `.dmg` from its GitHub | Packages | No | Low priority: only Qucsator-RF adds an engine |
| VACASK | AGPL-3.0 | OpenVAF-reloaded must be built from source on macOS | Binaries for OpenVAF | No | Optional: a fast OSDI-based reference |
| Spectre, HSPICE, AFS, PrimeSim | Commercial | Not here | If licensed | No | Licences, and their terms on publishing comparisons; whether the owner's institution holds any (unknown) |
| The machines | | Shared | Approved for benchmarks, reachable (15 November) | | Nothing more for machine A and machine B; the owner's login for the cluster |

**The fixtures' own licences:** CircuitSim90's decks in Xyce's repository are GPL-3.0-or-later by its README's grant; ngspice's examples carry ngspice's licence (the ISCAS'85 netlists' generator's terms unknown); IBM's power grids' terms unknown; the IVP test set's problems are published mathematics, which jolty can write afresh; CMC QA files carry each model's licence.
AGENTS.md:70–71 governs linking, not data: whether GPL decks may sit in an Apache-2.0 repository is the owner's call, and fetching them at a pinned commit, never copying them, sidesteps it.

## 4. Gaps and opportunities

Sizes are estimates for one agent: S days, M one to two weeks, L a month or more.
"Lead" marks an item where jolty would go beyond every reference rather than catch up.

| # | Item | Impact on the goal | Evidence | Size | Done when | |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | **The timing protocol:** both commits built and run in alternating rounds; each timing's median, minimum, count and 95% interval by ranks, wall and CPU time, load before and after; a change flagged when the intervals separate and the median moves past 5% | Makes the 5% rule and every speed claim decidable on a shared machine | §2.3, items 1–3: spreads of 9.8% and 19%, ±1.2% inside a run against +8.2% between runs | S–M | Two runs of the same commit flag nothing; the results file holds the intervals | |
| B2 | **ngspice like with like:** `.options klu` but for noise and pole-zero, which KLU refuses; an OpenMP build for BSIM3 fixtures; ngspice's own analysis time beside its process time; an empty deck's cost | The meshes' 17–40×, the 50,000 chain's 1.2× and the small circuits' 2.4–3.4× measured like with like | `cktntask.c:142–143`, `noisean.c:74`, `pzan.c:31`, `cktsopt.c:324–339`; `ngspice.rb` | S | `benchmarks.md` holds every fixture against Sparse 1.3 and KLU, engine and end to end | |
| B3 | **Each timed run's error:** a reference for every speed fixture, and each run's error against it beside its time | Speed at stated accuracy, as the lead decided; exposes unequal defaults | §2.3 item 8 | M | Every row of the speed tables carries its error | |
| B4 | **The same solution before the same time:** operating points and waveforms compared with the reference, and for a circuit with several DC solutions, jolty's `OperatingPoints` naming the one each simulator found | Timings of the same work; CircuitSim93's three simulators found three solutions | §3.3; `docs/guide/simulation.md` "Several DC solutions" | S–M | A row whose solutions differ is reported as a disagreement, not a time | Lead |
| B5 | **The precision suite, first version** (§4.1.3), scored at every point a simulator produced, shown as work-precision diagrams | Makes "most accurate" measurable, `accuracy.md` item 5 | §3.2; `research/accuracy.md` row 1 | L | 20 or more exact-answer fixtures with diagrams for jolty and ngspice, Sparse and KLU | Lead |
| B6 | **A behavioral charge:** `q(v)` written as an expression, with its exact gradient, integrated and judged as every charge is | Writes the IVP set's NAND gate and charge pump, and manufactured solutions with charges | jolty's behavioral sources read voltages, currents and time only (`src/jolty/devices.py:604–640`, `crates/jolty-expressions/src/parser.rs:185`); ngspice rewrites a `Q=` capacitor into a gyrator and an inductor (`inpcom.c:6937–6951`), and its B-source `ddt()` is a difference of past values, frozen within a step (`ptfuncs.c:426–475`) | S–M | Both problems run and meet their references | |
| B7 | **The hard-circuit corpus, first version** (§4.1.4), each outcome classified, failure rates with Wilson intervals | Makes "most reliable" measurable, `accuracy.md` item 6 | §3.4: no published failure rate | M | Rates and times for jolty and ngspice in `benchmarks.md` | Lead |
| B8 | **The coverage matrix, executed** (§4.1.2) | Makes "most complete" measurable; every cell a probe that ran | §3.1: the lists are documents | M | The matrix regenerates from its probes for jolty and ngspice; commercial cells marked documented | Lead |
| B9 | **The speed suite's first version** (§4.1.5): today's fixtures plus a ring oscillator, ISCAS'85 at transistor level on level 1 and BSIM3 cards, IBM's power grids, CircuitSim90's level 3 decks | Claims rest on fixtures of every kind and size, fair to every simulator | §2.1, §3.3 | M | Each fixture timed in jolty and ngspice at one core and all, both scopes, with its error | |
| B10 | **Efficiency against N concurrent copies,** beside the speed-up against one thread | Honest efficiency on mixed cores and with SMT | §3.5 | S | The suite prints both on this Mac, machine A and machine B | |
| B11 | **Deterministic metrics in CI:** each fixture's Newton iterations, steps, factorizations and error at defaults, recorded per platform and checked in `scripts/check`; instructions counted on Linux | Speed and accuracy regressions caught exactly, with no timing noise | AGENTS.md:94, `99–102`; §3.5 | S–M | A change that moves a count or an error names it, as the 5% rule asks for times | Lead |
| B12 | **Quiet-machine runs:** machine A and machine B, approved by the owner on 15 November, at idle hours by the protocol of §3.5; PBS nodes by exclusive placement later | The break-evens and the final benchmark measured where load cannot move them, and scaling past 12 threads, to 32 and 64 | `benchmarks.md` plan 2; `perf/README.md:224` | S | The many-cores plan's targets and the break-evens measured on both machines, each run's load recorded, the guide's break-evens stated with the machine | |
| B13 | **Xyce in the suites:** a netlist dialect, a runner, its raw reader, its timers | A second open reference, with its own KLU and its parallel solvers | §3.6 | M | Speed and precision tables carry Xyce's column | |
| B14 | **LTspice in the suites** | The most used free simulator | §3.6 | M | Its column, if its licence allows publishing | |
| B15 | **An agent's command end to end** (`speed.md` item 9) | The scope agents actually see | `research/speed.md` §2.1 | S | `jolty simulate` on `bjt_amp` against `ngspice -b`, with and without images | Lead |
| B16 | **Peak memory per fixture,** every simulator | CircuitSim93's second criterion; the first wall at ten million devices | `research/speed.md` §2.4 | S | Peak resident memory beside each time | |
| B17 | **A manifest of third-party fixtures:** URL, commit, SHA-256 and SPDX licence for each, fetched into a cache and never committed | Uses CircuitSim90, ISCAS'85, IBM's grids and CMC QA without licence questions | spice-ts's `SOURCES.md`; `research/models.md` principle 5 | S | `scripts/bench` fetches and checks every external deck | |
| B18 | **Commercial columns** | Leading means beating Spectre and HSPICE too | §3.3: licences forbid publishing | Owner | Results kept private or published with the vendor's consent | |

### 4.1 A design for the suites' first versions

#### 4.1.1 What they share

- **Where they live:** in the repository, beside today's fixtures, as one tool, so anyone can repeat them:

    ```text
    perf/
      circuits/, plans/     today's speed fixtures
      precision/            fixtures with exact answers, and the scripts that compute them
      robustness/           jolty's own hard circuits
      coverage/             one probe per row of the matrix, with its known answer
      external.json         third-party decks: URL, commit, SHA-256, SPDX licence (B17)
      simulators/           one adapter per simulator: dialect, command, raw reader, timers
      results/              baselines only, as today
    scripts/bench [speed | precision | robustness | coverage] [--simulator NAME ...]
    ```

    `benchmarks/` stays the agents' tasks, as the many-cores plan kept it (an implementation plan, not kept, Phase 0).
- **Simulator configurations are named,** never implied: `jolty`, `jolty-serial`, `ngspice-sparse`, `ngspice-klu`, `ngspice-klu-omp`, `xyce-serial`, `ltspice`, each with its version and build flags recorded in every results file.
- **Every result records** the machine, the load before and after, each simulator's version, the fixture's hash, and for jolty its fingerprint, so a published jolty number can be checked bit for bit by running it again.
- **Where results go:** baselines stay in `perf/results/` as today; comparisons with other simulators, which depend on machines and on licences still unread, go to the raw logs, not kept with the command that made them, and their summaries into `benchmarks.md`; publication waits for the owner.

#### 4.1.2 Coverage, first version

- **Rows:** about 25 analyses at the granularity of §3.1's table, and the models of `research/models.md` §3.5 by version and mode set, about 60 rows to start (an estimate).
- **Probes:** for analyses, the smallest circuit with a closed form (an RC's AC response, a resistor's noise `4kTR`, an RLC's poles, a divider's `.tf`); for SPICE3's models, a card from the differential corpus with ngspice's answer; for CMC models, their QA tests.
- **Statuses** as §3.1 defines them, each with its evidence; jolty and ngspice executed, Xyce and LTspice once installed, commercial simulators documented.
- **Output:** `coverage.json` as the source, rendered as a Markdown matrix with counts per category and each simulator's list of what another has and it lacks; a third table, the share of a corpus of vendors' models the SPICE reader takes (`research/models.md` item 18).

#### 4.1.3 Precision, first version

- **Fixtures, 20 or more with exact answers:**
    - closed forms: an RC under a step and a ramp, a femtofarad RC at default options (`accuracy.md` item 3), an under-damped RLC, an LC tank's frequency and energy, a diode on a resistor by Lambert's W, an AC response, a resistor's noise, an RLC's poles, a driven RC's periodic steady state;
    - linear circuits of 10 to 1,000 unknowns, ladders and random networks with time constants spread over six decades, solved exactly in arbitrary precision;
    - manufactured solutions: behavioral diodes, square-law and exponential transistors, written as explicit expressions so every simulator evaluates the same equations, with smooth `x*(t)` and injected `b(t)`;
    - the IVP test set's circuits: the ring modulator and the transistor amplifier now, the NAND gate and the charge pump after B6, the two bit adding unit with a reference recomputed by an independent integrator, since its published one holds 1e-5;
    - jolty's own reference, its run at `reltol` 1e-10 with a Richardson check, only where an independent method agrees (principle 3).
- **Scoring:** at each point a simulator produced, the reference evaluated there; the significant correct digits of the worst point, `scd`, and its mixed form for quantities near zero; the error of each fixture's measures, a delay or an overshoot, beside it.
- **Sweeps:** two, both reported: the relative tolerance alone, from 1e-2 to 1e-8 in quarter decades, the rest at defaults; and every tolerance scaled together.
    Each run records its engine time, steps, rejections, Newton iterations and failures.
- **Output:** per fixture, a work-precision diagram, `scd` against engine time on a log axis, one line per simulator configuration, its defaults marked; a table of each simulator's `scd` and time at its defaults; per simulator, the fixtures where it lies on the front of best time for each accuracy.
    No average of ratios (Hoefler and Belli's rule 4).

#### 4.1.4 Robustness, first version

- **Circuits:**
    - jolty's own: Schmitt triggers, bipolar and CMOS; latches and an SRAM cell; `folded_sources` (`crates/jolty-engine/tests/data/folded_sources.json`); an impasse point after Chua and Deng; the buck's gate edges; the memristive NOR that ngspice cannot finish; nets with no capacitance; supply ramps; ring oscillators starting from rest;
    - fetched at Xyce regression revision `7bb7e98`: CircuitSim90's 10 level 3 decks and its bipolar `vreg`, and Xyce's `MCNC_BJT_LATCH`, `MCNC_BJT_OPAMPAL`, `MCNC_BJT_RCA` and `MCNC_BJT_SCHMITECL`;
    - waiting for MOSFET level 2, which is on the owner's list of what ngspice has (`models.md` §5): CircuitSim90's 18 level 2 decks and its 14 large ones.
- **Defects** fixed by recorded rules, as CircuitSim93 did, in both simulators alike; jolty's structural checks report each one first, as facts.
- **Outcomes:** solved and right (against the reference, or every DC solution jolty finds), solved but a different solution, failed with its message, timed out, unsupported; unsupported is never counted as a failure.
- **Output:** a table of circuits by simulator, each cell an outcome, its time and its Newton iterations; each simulator's failure rate, DC and transient apart, with a Wilson interval, as `benchmarks/compare.py` already computes for the agents' runs.

#### 4.1.5 Speed, first version

- **Fixtures:** today's eight; a level 1 ring oscillator of 101 stages; ISCAS'85 `c432`, `c880` and `c6288` at transistor level on level 1 and BSIM3 cards, built from the gate-level netlists by jolty's builder (their terms checked first, B17); IBM's `ibmpg1` and `ibmpg2` at DC and in transient, from the manifest; CircuitSim90's 10 level 3 decks; the ten-million-device fixture when the front end builds one (`benchmarks.md` plan 7).
- **Runs:** each fixture on 1 thread, on the performance cores, on every core, on machine A on 1 to 64 threads and on machine B on 1 to 32; each simulator at its named best configuration; both scopes of time; peak memory; each run's error and solution check (B3, B4).
- **Output:** per fixture, a table of configuration, threads, engine time (median, 95% interval, minimum, count), end-to-end time, error, work counts, memory and load; per scaling fixture, speed-up and efficiency against N copies, against threads, with the copies' ceiling drawn (Hoefler and Belli's rule 11).

#### 4.1.6 How they run

- **In CI** (`scripts/check`, deterministic and fast): the coverage probes for jolty and ngspice, failing when `coverage.json` is stale, as `docs/reference/` fails today (`scripts/check:40–45`); each precision fixture's error and each speed fixture's work counts, recorded per platform, a change failing until its pull request records it (B11).
- **By a person or an agent, on a named machine:** `scripts/bench speed` before and after a change, interleaved (B1); the precision diagrams and the robustness corpus at each release, or when a numerical change lands; the full comparison with every installed simulator on a quiet machine (B12).
- **Never in CI:** timings, and anything needing Xyce, LTspice or a commercial licence.

## 5. The long view

In two to five years the suites could be what the IVP test set is for ODE codes and SPEC for processors: the reference benchmark of circuit simulation.

- **A versioned public benchmark:** fixtures, exact references, adapters and protocol released together, every result reproducible from its versions and, for jolty, its fingerprint.
- **Results for every open simulator, published with their losses**, and an invitation to vendors to publish theirs on the same fixtures, which their licences let only them do.
- **Work-precision diagrams for every simulator** on every fixture, so "most accurate at equal cost" is a curve anyone can read, and a robustness corpus whose failure rates fall release by release.
- **A coverage matrix that regenerates itself** from executed probes, with the CMC's QA tests and the kits' measured silicon (`research/models.md` items 8–9) as its model rows.
- **Deterministic tracking of every commit:** work counts, errors and instruction counts on a dashboard, as the Rust compiler tracks its own, with timings on quiet machines beside them.
- **One benchmark of tools and agents:** the agents' tasks (`benchmarks/`) run on the same fixtures, measuring what an agent achieves with each simulator.

No reference simulator publishes any of this about itself.

## 6. Principles to propose

For `AGENTS.md`, if the owner agrees:

1. **A claim names its measurement.**
    A claim of speed, accuracy, reliability or coverage, in the guide or a pull request, cites the suite, the fixture, the machine, the versions and the results file.
1. **The same solution, a stated accuracy.**
    A time is compared with another simulator's only where both found the same solution within the fixture's tolerance, and each time comes with the error it reached against the reference.
    The lead's decision on how speed claims are made (`benchmarks.md` §5) is the rest of this rule.
1. **A reference is independent of what it judges:** a closed form, a manufactured solution, a computation in higher precision by another method, or a published reference; jolty's own tighter run counts only when a second method agrees.
1. **Deterministic metrics gate; timings inform.**
    Work counts and errors, the same bits on a platform, are checked in `scripts/check`; timings are measured in alternating rounds with their intervals on a named machine, and stay out of it.
1. **Third-party fixtures are pinned and fetched, never copied:** URL, commit, SHA-256 and licence in a manifest.
    This extends `research/models.md`'s principle 5 from kits to every external file.
1. **Losses are reported with wins:** every suite reports every fixture, jolty's failures and slower times included.

## 7. Sources

**jolty** (`main`, `d10fc4d` to `7d41cc0`; `perf/` cited at `7d41cc0`):

- `AGENTS.md` (lines 70–71, 87–102, 119–127).
- `perf/README.md`; `perf/bench.py`; `perf/circuits/fixture.py`, `buck.py`; `perf/plans/*.json`; `perf/results/798b73d….json`, `38e4c7b….json`, `6fe8824….json` (untracked); `crates/jolty-engine/benches/fixtures.rs`; `scripts/bench`, `scripts/check`.
- `tests/test_ngspice.py`, `tests/differential.py`; `src/jolty/spice/ngspice.py`, `export.py`; `src/jolty/devices.py`; `crates/jolty-expressions/src/parser.rs`; `crates/jolty-engine/tests/analytic.rs` and its neighbours.
- `docs/guide/simulation.md`, `machines.md`; `benchmarks/README.md`; `.github/workflows/ci.yml`.
- The parallel benchmarks' commits on `main`, read with `git log` and `git diff`: `1bc7dab`, `a93a9f9`, `7d41cc0`.

**Notes:** the coordinator's message of 15 November (`main` at `7d41cc0`; machine A and machine B approved; ngspice 47 being built on machine A); `benchmarks.md`, `vision.md`, the session log, not kept, `speed.md`, `accuracy.md`, `models.md`; `research/speed.md`, `accuracy.md`, `models.md`; an implementation plan, not kept; raw logs, not kept, `final-bench-2026-11-15.txt`, `buck-itl4-2026-11-15.txt`, `bsim3-ngspice-2026-11-14.txt`, `last-mile-check-2026-11-14.txt`.

**ngspice 47** (the scratchpad's tarball): `src/spicelib/analysis/cktntask.c:142–143`, `cktsopt.c:179–187`, `300`, `324–339`, `371–377`, `noisean.c:74`, `pzan.c:31`; `src/spicelib/parser/inpptree.c:177`, `ptfuncs.c:426–475`; `src/frontend/inpcom.c:6905–6990`; `configure.ac:195–196`, `1321`; `NEWS:1–35`, `172–175`; `ANALYSES`; `src/spinit.in:19–25`; `tests/README`, `tests/general/`; `examples/klu/Circuits/` (ISCAS'85, `technology/45nm_HP.pm`).
The installed binary, `/opt/homebrew/Cellar/ngspice/47/bin/ngspice`, read with `strings`; Homebrew's formula `Formula/n/ngspice.rb` (GitHub, `Homebrew/homebrew-core`).

**Xyce** (`e5b7b9b`, GPL, read for facts only): `INSTALL.md`; `test/ReadMe_RegressionTesting.txt`; `doc/Reference_Guide/*_Command.tex`, `linsoltbl.tex`, `commandlinetbl.tex`; `src/CircuitPKG/N_CIR_Xyce.C:2071–2072`; `utils/ADMS/examples/vbic_r1.3_prerelease/qaSpec`, `reference/`.
Xyce's regression repository at `7bb7e98f0ed3a81a7d1cf1d10b68592107ed40b2` (`github.com/Xyce/Xyce_Regression`): `README.md`, `Netlists/` (listed through GitHub's API), `Netlists/CircuitSim90/circuitsim93.pdf`, `Netlists/CircuitSim90/MOS2_LARGE/README`.

**Papers and reports:**

- W. M. Lioen and J. J. B. de Swart, *Test Set for Initial Value Problem Solvers*, release 2.0, CWI report MAS-R9832, December 1998 (ir.cwi.nl/pub/4619).
- F. Mazzia and C.
    Magherini, *Test Set for IVP Solvers*, release 2.4, University of Bari and INdAM, February 2008 (archimede.uniba.it/~testset), its problems page and the transistor amplifier's and two bit adding unit's pages.
- J. A. Barby and R.
    Guindi, "CircuitSim93: a circuit simulator benchmarking methodology case study", IEEE, 1993, pp. 531–535 (in Xyce's regression repository).
- T. L. Quarles, "Benchmark circuits: results for Spice3", UCB/ERL M89/47, 1989, as ngspice's benchmark page cites it.
- S. R. Nassif, "Power grid analysis benchmarks", ASP-DAC 2008; the files at web.ece.ucsb.edu/~lip/PGBenchmarks/.
- T. Hoefler and R.
    Belli, "Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance results", SC15, 2015.
- T. Kalibera and R.
    Jones, "Rigorous benchmarking in reasonable time", ISMM 2013.
- J. Chen and J.
    Revels, "Robust benchmarking in noisy environments", arXiv:1608.04295, 2016.
- E. Hairer and G.
    Wanner, *Solving Ordinary Differential Equations II*, 2nd ed., Springer, 1996.
- P. J. Roache, "Code verification by the method of manufactured solutions", J. Fluids Eng.
    124(1), 2002; K. Salari and P.
    Knupp, SAND2000-1444, 2000.
- J. Roychowdhury and R.
    Melville, homotopy for DC operating points, DAC 1996 (the Schmitt trigger's stepping failure).
- F. Salfelder and L.
    Hedrich, "Evaluation of a benchmark suite for formal verification of analog circuits", ICCAD 2014 talk (MCNC's suite for "numerical challenges").
- F. Lannutti and S.
    Perticaroli, KLU in ngspice, MOS-AK, San Francisco, December 2012 (figures through search results only).

**Makers', projects' and users' publications:**

- ngspice: "Benchmarks" (ngspice.sourceforge.io/bench.html) and its applications page on KLU (ngspice.sourceforge.io/applic.html).
- VACASK's README (codeberg.org/arpadbuermen/VACASK); Bűrmen's MIDEM 2025 and ESSERC 2025 slides, through search results.
- spice-ts: issues #1045 and #1059, `benchmarks/SOURCES.md` (github.com/spicets/spice-ts).
- Cadence: Spectre Circuit Simulator Reference 23.1 (September 2023), its table of contents; "Spectre Tech Tips: how to migrate to Spectre X" and forum threads on presets.
- Synopsys: HSPICE quick reference, June 2015 (its text partly extracted); its evaluation terms, through search results.
- Siemens: Analog FastSPICE platform fact sheet; "A systematic analog verification approach using Analog FastSPICE harmonic balance".
- Analog Devices: LTspice keyboard shortcuts and directives sheet; spicelib's LTspice notes (spicelib.readthedocs.io).
- Qucs-S documentation (qucs-s-help.readthedocs.io): "Choosing a simulation backend", "RF simulation".
- J. Bier, "Don't ban benchmarks", BDTI, 17 December 2008; Sigasi's open letter to EDA vendors' chief executives; SemiWiki's forum threads on benchmarks.
- SciMLBenchmarks' work-precision pages and DiffEqDevDocs' "Benchmark Suite" (docs.sciml.ai).
- Rust internals forum, "What is perf.rust-lang.org measuring and why is instructions:u the default?", 2019.
- SPEC CPU 2017 documentation on SPECspeed and SPECrate.
- Eclectic Light Company on Apple silicon's core allocation and quality of service (2022–2026); Apple developer forums, thread 674456; macOS's `taskpolicy(8)` manual page.
- ServeTheHome's review of the Threadripper 3970X; TechPowerUp's review of the Ryzen 9 7950X.
