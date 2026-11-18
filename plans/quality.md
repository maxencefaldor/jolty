# Quality

Status: rewritten in plain words on 15 November 2026, every item checked against its source and today's code, and brought up to date on 18 November; the research is `research/quality.md`, kept whole.

Plan: ✅ 15 done · 🚧 1 in progress · 📋 9 planned · ⏸️ 1 parked (§4).

## 1. Goal and metric

jolty is a sharp tool: a newcomer reads, uses and extends it easily, every convention of `AGENTS.md` holds everywhere in its spirit as well as its letter, and nothing in it is slop, no second tool for one job, no fact written twice, no name for two things or two names for one.
Measured by the review's findings, which go to zero, and by the cost of adding a model, in shared files touched.

## 2. Where we stand

**Strong:** its mechanical checks are the strictest of the simulators examined.
Every ruff and ty rule and clippy's pedantic lints are errors; rustdoc's warnings are errors; the import order is tested; each suppression names its rule and its reason; the reference pages are regenerated and diffed; the extension's stub is checked against the module; the Python tests run on free-threaded Python too, in `scripts/check` exactly as in CI.

**Short of `AGENTS.md`'s spirit:**

- **One fact in several places:** one parameter lives in six places in four files (`models.md` 4).
- **Guards a new model can skip:** the SPICE round trip does not run over every model (item 4).
- **Numbers without a source:** engine constants, and up to 340 of 428 test tolerance sites (a 15 November scan's upper bound) (items 10 and 11).
- **Names:** unified across the library (`8a4d80a` to `bc35702`, item 12; only `measure::Signal` waits on the owner) and across the benchmark suites (`092e86f`, item 13, done: `fixture` everywhere, one `benchmarks/simulators.py`).

**One way to differentiate and one way to stamp,** since 17 November: every kernel composes dual numbers at each law's own width (item 6), and writes its matrix entries through summed derivatives (item 5).

**Slop found and removed on 15 November:** mypy, brought in for its stubtest beside ty, the one type checker (`5ba0fb2`, replaced by a test of jolty's own); four ways to start ngspice and three to read its version (`5f69050`, now `jolty.spice.ngspice.execute` and `version`); three copies of the options as tight as ngspice's (now `jolty.plan.TIGHT`); the precision suite's restated defaults, option names and colours (`a43aa05`); the precision suite's committed 1.65 MB of results.

**The Python API:** small, immutable and precisely typed, but only half carries units: options, overrides and variants are plain floats in SI units (item 7).
The analyses a spec names are resolved only after every analysis has run (item 8).

**The guide:** precise, and its 84 keyword calls (counted 15 November) match the API; `tests/test_guide.py` now runs its Python blocks and checks its quoted outputs (item 14, done: currently 48 Python blocks, 38 run and 10 marked `not run`, and 18 quoted outputs), while its sentences still run long, a median of 27 words on 15 November (item 20).

## 3. Gaps

- One fact in several places: a model's facts (the benchmarks' helpers are unified, item 13).
- One concept under several names: `measure::Signal` still waits on the owner (item 12).
- Guards a new model can skip (the SPICE round trip over every model, item 4); numbers that do not say where they come from.
- Units and names checked late on several user paths.
- No how-to pages; guide sentences still run long (executable blocks and quotes are done, item 14).

## 4. Plan

Each item says what it is, why it matters, and what "done" means; **Lead** marks an item where jolty would go beyond every reference.
Its source is the research report's row, in `research/quality.md` §4.
An item that would bring a new tool waits for the owner's yes (`AGENTS.md`, "Tools").

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. Results carry their facts

- **What:** where an analysis used logic devices outside their characterized domains, its excursions, stated on each result instead of on the simulation.
- **Result** (`12072a8`): every result class has `excursions`; `Simulation` has none; results files keep them.
- **Source:** row 11.

### ✅ 2. The extension's stub held to the module

- **What:** ty reads the stub `_core.pyi`, never the extension module, so a test holds them together: the stub declares every public name of the module and no other, each class holds the names the stub gives it, a class the stub marks final cannot be subclassed, and every outcome the extension returns holds exactly the keys its `TypedDict` names.
- **Result** (`7bd9436`, then `5ba0fb2` without mypy): `tests/test_stub.py`; it found the extension's classes unmarked as final and constructed by `__new__` where the stub said `__init__`; renaming a key in the stub fails it.
- **Then** (`5ae1508`): the transfer function's outcome was missing from the stub, and the test could not see it, as it checked the stub's union against its own table of analyses, which missed it too; the table is now held to `jolty.plan.Analysis`, and ty then found `Simulation._linear` typed without it.
- **Source:** row 22.

### ✅ 3. Mechanical policy checks

- **What:** rustdoc's warnings as errors in `scripts/check`; the Python tests on free-threaded Python in `scripts/check`, the same group CI's free-threaded job runs; each release wheel installed alone in a fresh environment and run on an example before upload.
- **Result** (`391f782`): all three; nine broken or private links in the crates' docs fixed on the way.
- **Also there:** `cargo-deny`, a licence check of Rust's dependencies alone, added the same day before the owner's rule on tools; the owner's decision (15 November): it leaves for a test of jolty's own over both languages (item 22).
- **Source:** row 23.

### ✅ 4. Guards that enumerate the library

- **What:** every test that should hold for all models iterates the standard library, so a new model cannot skip it.
- **Result:** the allocation test and the Jacobian check run over one table (`crates/jolty-models/src/testing/cases.rs`, 35 cases covering 28 analog models at `1235b50`; now 45 cases covering all 31 analog models in `Library::standard()`, 30 native plus `foundry_mosfet`), and fail if a model has no case; the kernels' lints are denied once, in `kernel.rs`, for every kernel beside it.
    **Not done:** the SPICE round trip over every model (a model written and read back as itself, as another element by design, or refused by name, each decided per model).
- **Done when:** removing a model's case fails a test; one `deny` attribute in the crate.
- **Source:** row 8.

### ✅ 5. One way to stamp

- **What:** BSIM3, VDMOS and VBIC write their matrix entries through summed derivatives (`parts::Derivatives`), as the MOSFET does since 15 November.
- **Done when:** `parts.rs` holds one current and one charge function; each model's evaluation timed before and after.
- **Result** (`2ed3c8a`, `7a5ba83`, `d5067e3`): `parts.rs` holds one current and one charge function; BSIM3 11.5% faster, VBIC 27%, VDMOS 8% at their defaults; `Dual::ln` and `Dual::pow` replace two private helpers.
- **Source:** row 3.
    This item replaces `speed.md`'s copy of it.

### ✅ 6. Every kernel on dual numbers

- **What:** the diode, the JFET and the Gummel-Poon transistor computing their derivatives by automatic differentiation, as the others do, and the Jacobian check tightened to rounding.
- **Done when:** no hand-written derivative left in a kernel.
- **Result** (`5421fdf`, then `device-derivatives.md` step 1, `938220a`, `7887dbe`, `ec04308`, 17 November): every kernel composes dual numbers at the width of each of its laws; no derivative is written by hand in any kernel, Gummel-Poon's, the JFET's, the memristor's, the switch's, the photocell's, VDMOS's quasi-saturation's and the passive voltage coefficients' included.
    Gummel-Poon and the JFET cost 2 to 8% more instructions an evaluation, put to the owner (`device-derivatives.md` 1.3).
    **Not done:** the Jacobian check's 1e-5 stays: BSIM3's charge rows cancel, so a bound at rounding needs the magnitude of the terms each charge sums, which kernels do not report.
- **Source:** row 4.

### 📋 7. Quantities everywhere

- **What:** options, overrides, variants, axes and sweeps take quantities, each checked against the model's dimension, not plain floats in SI units.
- **Done when:** `Axis("R1", "resistance", [1 * nF])` fails at construction.
- **Source:** row 9.

### 📋 8. Everything a plan names resolved before running

- **What:** every analysis a spec names, every record path, sweep source and axis instance checked before the first analysis runs, each error naming its nearest valid name.
- **Done when:** a misspelt analysis fails before simulating, naming its neighbour.
- **Source:** row 10; `AGENTS.md`, "Resolve before running".

### 📋 9. Plans validated as files are, every error at once

- **Done when:** a plan with two bad options reports both.
    **Source:** row 14.

### 📋 10. Every engine constant names its source

- **What:** each constant of the engine says where it comes from, a line of a reference, a paper, a derivation or a measurement; those a user could choose, such as gmin stepping's factor and source stepping's, become options.
- **Done when:** each constant's doc cites its source.
- **Source:** row 15; `AGENTS.md`, "Constants".

### 📋 11. Every test tolerance names its source

- **What:** each tolerance a test compares with states where it comes from; a check flags a new literal tolerance without one.
- **Done when:** none of the 428 sites left uncited.
- **Source:** row 16.

### 🚧 12. One name for each concept

- **What:** the owner's rule (`AGENTS.md`, "Names"): first a glossary of every concept, its one name, its definition, and every other name it has today with each place; the owner reviews it; then the renames, everywhere at once.
    It takes in the research's distinct names for analyses and results, a `PeriodicSteadyState` analysis and `batch(plan)`.
- **Done when:** the glossary holds every concept once, the owner has reviewed it, and the renames are on `main`; the reference lists each class once.
- **State:** every rename the owner decided is on `main` (`8a4d80a` to `bc35702`, 712 renames by a table and a script, not kept); the periodic steady state is an analysis of its own (choice 6, `agentic.md` 22), and `Parallelism`'s names are the owner's (`speed.md` 4).
    `elaborate::Composite` is `SummedTerminal`, the owner's name (16 November).
    **Not done:** one name, which waits for the owner: `measure::Signal` (an output and a gain's denominator).
- **Source:** row 13; a brief, not kept.

### ✅ 13. The benchmarks' code made one

- **What:** the speed suite, the precision suite, the hard-circuit corpus and the coverage matrix share one module of measuring helpers (the load, the commit, the machine, results files), one runner per simulator (jolty, ngspice with Sparse and with KLU, Xyce), one command line (`scripts/bench`, precision brought into it), and one name for the circuit a benchmark runs.
- **Why:** each suite grew its own copies: the corpus's and the matrix's modules of runners both named `simulators`, two load readings, two ways to name a results file.
- **Done when:** no helper defined twice under `benchmarks/`; one runner per simulator; one script.
- **Result** (`092e86f`, 15 November): every suite moved under `benchmarks/<suite>` (`perf/` and `scripts/precision` deleted); shared helpers in `benchmarks/measuring.py` and one runner per simulator in `benchmarks/simulators.py`; `scripts/bench` runs all five suites (`--precision`, `--robustness`, `--coverage`, `--scale`, `--view`, `--compare`, `--nightly`); every suite calls its circuit a `fixture`.

### ✅ 14. The guide's code runs and its outputs are produced

- **What:** every Python block of the guide and the README run by the tests, and each output the guide quotes produced by a run, so the guide cannot drift from the code.
- **How:** a test of jolty's own reading the blocks, as `tests/test_plan_defaults.py` reads the guide's table of options; or, if the owner approves it, the plugin the research names.
- **Done when:** CI runs the 41 blocks; each quoted output comes from a run.
- **Result** (`dc5b2ef`, `f0c8f68`): `tests/test_guide.py` ran 30 of the guide's and the README's 40 blocks then (now 38 of 48 blocks, 10 marked `not run`), each other one marked with its reason, and produces all 18 quoted outputs; 14 quotes and 5 blocks no longer matched the code and were corrected in the guide; a quoted number matches within one unit in its last printed digit plus 4 float ulps.
- **Source:** row 17.

### ✅ 15. Coverage reported

- **What:** a coverage report per CI run, Rust and Python, reported, not gated.
- **Approved** by the owner (15 November): `cargo-llvm-cov` and coverage.py, reported and never gated.
- **Source:** row 19.

### ⏸️ 16. API stability checked against the last release

- **What:** a change to the public API of the crates or the Python package detected against the last tagged release.
- **Waits for:** a first release; the tools, `cargo-semver-checks` and `griffe`, approved by the owner (15 November).
- **Source:** row 21.

### ✅ 17. Mutation testing

- **What:** the tests' strength measured by mutating the code and counting the mutants they kill: on each change, and weekly sharded over machine A's cores.
- **Approved** by the owner (15 November): `cargo-mutants`; `mutmut` not now.
- **Lead.**
    **Source:** row 18.

### 📋 18. Fuzzing the parsers

- **What:** the expression parser, the circuit format, the SPICE reader and shard files fed generated inputs, all with Hypothesis, already a dependency, the Rust parsers through the extension; no `cargo-fuzz` (the owner, 15 November).
- **Done when:** a Hypothesis netlist generator runs nightly over the Python readers.
- **Source:** row 20.

### 📋 19. A contributors' guide to adding a model

- **What:** a page that walks through adding a model from its single declaration (`models.md` 4), and a page of the sharp bits.
- **Needs:** `models.md` 4.
    **Done when:** a model added by following the page alone.
- **Source:** row 25.

### 📋 20. Shorter sentences in the guide, and how-to pages

- **Done when:** a median sentence under 25 words; how-to pages for the common tasks.
    **Source:** row 26.

### 📋 21. The cost of a model measured

- **What:** the number of shared files each new model touches, recorded per model.
- **Done when:** recorded for every model; **Lead** once it is 1 (`models.md` 4).
- **Source:** row 7.

### ✅ 22. Every dependency's licence checked, in both languages

- **What:** a test of jolty's own reads `cargo metadata` and `importlib.metadata` and holds every Rust and Python dependency's licence to one allow-list; `cargo-deny` and `deny.toml` leave.
- **Result** (`d52e342`): `tests/test_licenses.py`; 174 crates and 55 distributions passed at `d52e342` (plus optional `jax` and transitive crates/distributions added since); the list grew by eight licences for Python (BSD-3-Clause, MPL-2.0 for Hypothesis, PSF-2.0, MIT-0, MIT-CMU, 0BSD, Zlib, CC0-1.0).
    **Source:** the owner, 15 November.

### ✅ 23. Known vulnerabilities checked

- **What:** `uv audit`, uv's own, and `cargo-audit`, RustSec's, in `scripts/check` and CI, and weekly on `main`.
- **Result** (`a13244b`): no advisory today; `paste` 1.0.15, unmaintained, warns through faer.
    **Source:** the owner, 15 November.

### ✅ 24. Rust's version the latest stable, stated once

- **What:** the toolchain pinned at the latest stable (1.100.0 on 15 November), `rust-version` equal to it, a test holding the two equal; the pin moved at each release.
- **Result:** `rust-toolchain.toml` pins 1.100.0, every crate's `rust-version` claims it, and `tests/test_rust_version.py` holds the two equal.
    **Source:** the owner, 15 November.

### ✅ 25. cargo-nextest measured

- **What:** `cargo test --workspace` against `cargo nextest run` and `cargo test --doc`, on this Mac and in CI; adopted only if it pays, with the numbers for the owner.
- **Adopted** (`1e30843`): `scripts/check` and CI run nextest and the doctests, 117 s against 209.5 s.
- **Then** (`ab2c08c`, 17 November): the three slow corpus tests sharded 16 ways and `jolty-numerics` built at `opt-level = 2` in tests; the whole suite from 302.4 s to 107.2 s on machine D; on 18 November, 655 tests took 197.5 s on 4 test threads, 81.5 s on 16 and 73.8 s on 64, flat past 16, where the slowest tests set the floor.
- **Source:** the owner, 15 November.

### ✅ 26. Every message facts only

- **What:** "Feedback is facts" covers every message, misuse of the API included: an error states what is true, such as the devices a result recorded, and never how to fix the call; the guide says how.
    `AGENTS.md` says so.
- **Done when:** no message says how to fix a call.
    **Source:** the owner, 15 November.
- **Result** (`8ac4df8`): `AGENTS.md`'s "Feedback is facts" covers jolty's own use too, the guide saying how to call it; every message string of `src/jolty` and the crates searched for advice ("use", "pass", "try", "instead", "should", "give", "did you mean" and their capitals): one found, the compiler binary's usage, now stated as what it takes.

**Elsewhere:** one law suite for every kernel is `models.md` 3; each model fact declared once and a model descriptor are `models.md` 4.

## 5. Decided

- 15 Nov, the owner: mkdocstrings-python stays, Zensical's plugin for the API pages; ported kernels keep their reference's local names inside the kernel, each function citing its source's lines.
- `jolty.cli.CommandLine` and `jolty.cli.run` are public: jolty's own command line and the repository's scripts share one configuration.
- The guide's statements the code did not back were corrected (`7edbffa`), and VBIC's section links the models reference rather than a table copied by hand.
- 15 Nov, the owner: no tool or dependency joins without the owner's yes; ty is the one type checker, zensical with its mkdocstrings plugin the one docs tool; mypy left (`5ba0fb2`).
- 15 Nov, the owner: each concept has one name, uniform, consistent and symmetric, formal and principled (`AGENTS.md`, "Names").

## 6. Open principles

None: the owner decided on 15 November that ported kernels keep their reference's names inside the kernel, that "Feedback is facts" covers every message (item 26), and that `cargo-deny` leaves (item 22).
