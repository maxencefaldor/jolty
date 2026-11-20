# Quality

Status: rewritten in plain language on 15 November 2026, checked against source documents and current code, and updated on 18 November.
Full background research remains archived in `research/quality.md`.

Plan: ✅ 23 done · 📋 2 planned (§4).

## 1. Goal and metric

jolty aims to stay easy for newcomers to read, use, and extend, upholding every convention in `AGENTS.md` in both letter and spirit.
The codebase avoids duplicate tools for one job, duplicated facts, homonyms across concepts, and synonyms for one concept.
Progress is measured by driving code-review findings to zero and minimizing the shared files touched when adding a model.

## 2. Where we stand

**Strong:** automated checks in jolty are the strictest among the examined simulators.
Every ruff rule, ty rule, pedantic clippy lint, and rustdoc warning is configured as an error.
Tests verify import order, require every suppression to cite a rule and reason, diff regenerated reference pages, and check the extension stub against the compiled module.
The Python test suite also runs under free-threaded Python in `scripts/check`, matching CI.

**Short of the spirit of `AGENTS.md`:**

- **One fact in several places:** a single model parameter currently lives in six places across four files (`models.md` 4).
- **Guards that enumerate the library:** allocation, Jacobian, physical-law, and SPICE round-trip checks all enumerate `Library::standard()` (item 4).
- **Numbers with their source:** all named constants across `jolty-engine`, `jolty-numerics`, and `jolty-checks` and all Python test tolerances cite origins and are enforced by `tests/test_tolerances.py` (items 10 and 11, done).
- **Names:** naming is unified across the library (`8a4d80a` to `bc35702`, item 12, done; the owner kept `measure::Signal` on 20 November) and across benchmark suites (`092e86f`, item 13, done: `fixture` everywhere, one `benchmarks/simulators.py`).

**One way to differentiate and one way to stamp (since 17 November):** every kernel composes dual numbers at the natural width of each physical law (item 6) and stamps matrix entries through summed derivatives (item 5).

**Redundant code removed on 15 November:** mypy was removed after being introduced solely for stubtest alongside ty (`5ba0fb2`), replaced by a dedicated stub test.
Four ngspice launch paths and three version readers were consolidated into `jolty.spice.ngspice.execute` and `version` (`5f69050`).
Three copies of ngspice-tight options collapsed into `jolty.plan.TIGHT`, restated precision-suite defaults and colors were removed (`a43aa05`), and 1.65 MB of committed precision results were deleted.

**The Python API:** small, immutable, and precisely typed.
Upfront validation of plans, analyses, options, and specs resolves every name and checks every parameter dimension before simulation starts (items 7 to 9, done).

**The guide:** technical prose is precise, and all 84 keyword calls counted on 15 November match the API.
`tests/test_guide.py` executes 38 of 48 Python code blocks (10 marked `not run`) and checks all 18 quoted outputs (item 14, done).
Guide sentences remain long, with a median of 27 words measured on 15 November (item 20).

## 3. Gaps

- Duplicated model metadata: each model's facts remain spread across several files, while benchmark helpers are now unified (item 13).
- Missing how-to pages and long guide sentences; executable guide blocks, output verification, upfront validation, and unit checking are complete (items 7 to 11, 14).

## 4. Plan

Each item describes scope, rationale, and completion criteria; **Lead** marks an item where jolty aims to exceed every reference simulator.
Each source reference points to the corresponding row in `research/quality.md` §4.
Any item introducing a new tool requires explicit owner approval (`AGENTS.md`, "Tools").

Status: ✅ done, on `main` · 🚧 in progress · 📋 planned · ⏳ waits for the owner · ⏸️ parked.

### ✅ 1. Results carry their facts

- **What:** record logic-device domain excursions on each analysis result instead of on the simulation object.
- **Result** (`12072a8`): every result class carries `excursions`, `Simulation` carries none, and result files persist excursions.
- **Source:** row 11.

### ✅ 2. The extension's stub held to the module

- **What:** a parity test keeps `_core.pyi` synchronized with the compiled extension module, which ty does not inspect directly.
    The test verifies that public module names match, every class exposes the stub attributes, final classes reject subclassing, and returned outcomes contain exact `TypedDict` keys.
- **Result** (`7bd9436`, then `5ba0fb2` without mypy): `tests/test_stub.py` caught extension classes lacking `@final` and constructed via `__new__` where the stub declared `__init__`, and fails on stub key renames.
- **Then** (`5ae1508`): the transfer-function outcome was missing from both the stub and the test's separate analysis table, hiding the omission.
    The analysis table now tracks `jolty.plan.Analysis` directly, which allowed ty to catch the incomplete typing on `Simulation._linear`.
- **Source:** row 22.

### ✅ 3. Mechanical policy checks

- **What:** treat rustdoc warnings as errors in `scripts/check`, run the CI free-threaded Python test group locally in `scripts/check`, and test each release wheel in a clean environment before upload.
- **Result** (`391f782`): all three checks landed, fixing nine broken or private links in crate documentation along the way.
- **License tool update:** `cargo-deny`, checking only Rust dependency licenses, was added before the tool policy and removed in favor of a cross-language jolty test (owner decision, 15 November; item 22).
- **Source:** row 23.

### ✅ 4. Guards that enumerate the library

- **What:** every library-wide invariant test iterates `Library::standard()`, preventing new models from skipping coverage.
- **Result:** allocation and Jacobian checks run over one table (`crates/jolty-models/src/testing/cases.rs`), expanding from 35 cases across 28 analog models (`1235b50`) to 45 cases across all 31 analog models (30 native plus `foundry_mosfet`).
    The suite fails if any model lacks a case, and kernel lints are denied once in `kernel.rs` for all kernels in the module.
    `tests/test_spice.py` (`test_every_model_has_a_spice_round_trip_case` and `test_every_model_spice_round_trip`) enumerates all 33 models in `_core.models()` (`Library::standard()`), checking electrical equivalence within 1e-9 relative for all 27 exportable analog models.
    Among the 33 models, 22 round-trip directly (2 with an auxiliary 0 V sense source) and 1 exports as a foundry subcircuit (`foundry_mosfet`).
    Five models map to behavioral sources by design (`voltage_controlled_switch`, `current_controlled_switch`, `voltage_controlled_memristor`, `current_controlled_memristor`, `photocell`), and 5 are refused by name (`heat_source`, `thermal_resistance`, `thermal_capacitance`, `logic_source`, `logic_cell`).
- **Done when:** removing any model's test case fails CI, and one `deny` attribute governs all kernels in the crate.
- **Source:** row 8.

### ✅ 5. One way to stamp

- **What:** BSIM3, VDMOS, and VBIC stamp matrix entries through summed derivatives (`parts::Derivatives`), matching the MOSFET path landed on 15 November.
- **Done when:** `parts.rs` provides one current helper and one charge helper, with before-and-after evaluation timings recorded per model.
- **Result** (`2ed3c8a`, `7a5ba83`, `d5067e3`): `parts.rs` exposes one current helper and one charge helper, while `Dual::ln` and `Dual::pow` replace two private helpers.
    Default evaluation sped up by 11.5% on BSIM3, 27% on VBIC, and 8% on VDMOS.
- **Source:** row 3.
    Item 5 replaces the duplicate entry previously in `speed.md`.

### ✅ 6. Every kernel on dual numbers

- **What:** diode, JFET, and Gummel-Poon models compute derivatives via automatic differentiation like all other kernels, and Jacobian checks tighten toward rounding precision.
- **Done when:** no hand-written derivative remains in any kernel.
- **Result** (`5421fdf`, then `device-derivatives.md` step 1, `938220a`, `7887dbe`, `ec04308`, 17 November): every kernel composes dual numbers at the natural width of each physical law.
    No hand-written derivative remains in any kernel, covering Gummel-Poon, JFET, memristors, switches, photocell, VDMOS quasi-saturation, and passive voltage coefficients.
    Gummel-Poon and JFET evaluations cost 2 to 8% more instructions, escalated to the owner (`device-derivatives.md` 1.3).
    **Not done:** the 1e-5 Jacobian tolerance remains because BSIM3 charge rows cancel; a rounding-level bound requires kernels to report constituent term magnitudes.
- **Source:** row 4.

### ✅ 7. Quantities everywhere

- **What:** options, overrides, variants, axes, and sweeps accept dimensioned quantities validated against model dimensions instead of plain SI floats.
- **Done when:** every user-facing dimensioned number is a `Quantity` validated before simulation runs, and `Axis("R1", "resistance", [1 * nF])` fails upfront against the circuit's declared model dimension.
- **Result** (20 November): `Options` tolerances (`voltage_tolerance`, `current_tolerance`, `charge_tolerance`, `gmin`), `Conditions.temperature`, `Simulation` / `Simulation.from_elaborated` / `Variant` overrides, `Axis`, parameter `DcSweep`, `Toleranced.nominal`, and `variation.{sample, moved, extreme}` all carry `Quantity` values for dimensioned parameters and plain floats for dimensionless ones.
    Context-free dimensions validate at construction while circuit-bound overrides, variants, axes, and parameter sweeps validate upfront via `devices.parameter_dimension` derived from engine metadata (`_core.parameter_dimensions`), preserving circuit-plan separation (`AGENTS.md`, "Circuit and plan").
    `Function` / `jvp` / `vjp` / `jolty.jax.function` remain SI float/array APIs.
    Exhaustive tests cover all 33 registered models (`tests/test_devices.py`) and rejection of mismatched dimensions and bare floats (`tests/test_plan_validation.py`).
- **Source:** row 9.

### ✅ 8. Everything a plan names resolved before running

- **What:** every analysis named by a spec, plus every record path, sweep source, and axis instance, is validated before simulation starts, with errors reporting the nearest valid name.
- **Done when:** a misspelled analysis name fails before simulation and reports the closest match.
- **Result** (18 November): `Simulation.validate_analysis`, `Simulation.validate_plan`, and `close_names()` validate all analysis names, source references, signal names, record patterns, initial conditions, nodesets, and axis targets upfront.
    Invalid names fail before the first analysis executes and report nearest valid names in `PlanError` (`tests/test_plan_validation.py`).
- **Source:** row 10; `AGENTS.md`, "Resolve before running".

### ✅ 9. Plans validated as files are, every error at once

- **Done when:** a plan containing two invalid options reports both errors together.
- **Result** (18 November): Pydantic multi-error validation across `src/jolty/plan.py` and `specs` collects all invalid options, analyses, and spec references into a single `PlanError` before simulation starts (`tests/test_plan_validation.py`).
    **Source:** row 14.

### ✅ 10. Every engine constant names its source

- **What:** every engine constant cites its origin (reference line, paper, derivation, or measurement), while user-tunable constants such as gmin-stepping and source-stepping factors become options.
- **Done when:** every constant's documentation cites its origin.
- **Result** (18 November): documented all 91 named `const` items (including `PERIODIC_SENSITIVITY`) across `crates/jolty-engine/src/`, `crates/jolty-numerics/src/`, and `crates/jolty-checks/src/`, enforced by AST/doc-comment tests in `tests/test_tolerances.py`.
- **Source:** row 15; `AGENTS.md`, "Constants".

### ✅ 11. Every test tolerance names its source

- **What:** every test tolerance states its justification, and an automated guard rejects any uncited literal tolerance.
- **Done when:** none of the 428 baseline sites remains uncited.
- **Result** (18 November): cited all 520 Python test tolerance sites (447 pre-existing plus 73 Wave 2 sites) across `tests/**/*.py`, enforced by an AST guard in `tests/test_tolerances.py`.
    The audit exposed two masked-assertion bugs, both fixed.
    In `tests/test_specs.py`, `pytest.approx` default `abs=1e-12` masked a `9.616e-5` relative discrepancy in `noise_density`/`integrated_noise` at `1.5 kHz` under `rel=1e-12` (fixed with `abs=0` across all 5 assertions).
    In `tests/test_spice.py`, tiny sub-pA and sub-pF `pytest.approx` assertions also lacked explicit `abs=0` and were tightened.
- **Source:** row 16.

### ✅ 12. One name for each concept

- **What:** follow the owner's naming rule (`AGENTS.md`, "Names") by writing a glossary of every concept, its canonical name, definition, and existing aliases for owner review, then applying all renames atomically.
    The scope includes distinct names for analyses and results, a dedicated `PeriodicSteadyState` analysis, and `batch(plan)`.
- **Done when:** the glossary covers every concept once, the owner approves the glossary, all renames land on `main`, and reference docs list each class once.
- **State:** every owner-approved rename landed on `main` (`8a4d80a` to `bc35702`, 712 renames applied via a temporary migration table and script).
    Periodic steady state is now a dedicated analysis (choice 6, `agentic.md` 22), `Parallelism` uses the owner's names (`speed.md` 4), and `elaborate::Composite` became `SummedTerminal` (16 November).
    **Kept** (the owner, 20 November): `measure::Signal` names both an output and a gain's denominator, so the glossary's `Output` does not fit, and the measures' `signal` parameter already matches it.
- **Source:** row 13; superseded design brief (not retained).

### ✅ 13. The benchmarks' code made one

- **What:** the speed, precision, hard-circuit, and coverage suites share one measurement-helper module (load, commit, machine, result files), one runner per simulator (jolty, ngspice Sparse/KLU, Xyce), one CLI (`scripts/bench`), and one circuit term.
- **Why:** each suite had accumulated duplicate helpers, including two simulator-runner modules both named `simulators`, two load readers, and two result-file naming conventions.
- **Done when:** no helper is duplicated under `benchmarks/`, one runner serves each simulator, and one script runs every suite.
- **Result** (`092e86f`, 15 November): all suites moved under `benchmarks/<suite>` (`perf/` and `scripts/precision` deleted), sharing `benchmarks/measuring.py` and `benchmarks/simulators.py`.
    `scripts/bench` runs all five suites (`--precision`, `--robustness`, `--coverage`, `--scale`, `--view`, `--compare`), and every suite calls a benchmark circuit a `fixture`.

### ✅ 14. The guide's code runs and its outputs are produced

- **What:** tests execute every Python block in the guide and README and regenerate each quoted output so documentation cannot drift from code.
- **How:** a dedicated test reads the blocks, matching how `tests/test_plan_defaults.py` checks the guide option table, or an owner-approved plugin from the research report.
- **Done when:** CI executes all 41 baseline blocks and produces every quoted output from a live run.
- **Result** (`dc5b2ef`, `f0c8f68`): `tests/test_guide.py` initially ran 30 of 40 guide/README blocks and now runs 38 of 48 (10 marked `not run` with reasons), verifying all 18 quoted outputs.
    The initial audit corrected 14 stale quotes and 5 stale code blocks, matching quoted numbers within one unit in the last printed digit plus 4 float ulps.
- **Source:** row 17.

### ✅ 15. Coverage reported

- **What:** report Rust and Python coverage on every CI run without gating.
- **Approved** by the owner (15 November): `cargo-llvm-cov` and coverage.py, reported and never gated.
- **Source:** row 19.

### ✅ 17. Mutation testing

- **What:** measure test strength by mutating code and counting killed mutants on each change and in weekly runs sharded across machine A cores.
- **Approved** by the owner (15 November): `cargo-mutants`; `mutmut` deferred.
- **Lead.**
    **Source:** row 18.

### ✅ 18. Fuzzing the parsers

- **What:** fuzz the expression parser, circuit format, SPICE reader, and shard files using Hypothesis (already a dependency), calling Rust parsers through the extension; no `cargo-fuzz` (owner decision, 15 November).
- **Done when:** a Hypothesis netlist generator runs nightly against the Python readers.
- **Result** (18 November): added 62 Hypothesis property and fuzz tests in `tests/test_fuzz_parsers.py` covering SPICE expressions, libraries/netlists, `jolty-expressions`, circuit JSON, results/raw files, and shard files.
    The suite uncovered and fixed 40 Python parser/loader bugs, including iterative `inlined()` with a node budget, subcircuit cycle tracking, empty-token handling, finite overflow checks, and charge-leak array validation.
    Fuzzing also fixed three Rust bugs: recursive cycle stack overflow in `crates/jolty-engine/src/variants.rs` (replaced with iterative DFS returning `TolerancedError::Recursive`) and non-finite `table()` values bypassing `partial_cmp` in `crates/jolty-expressions/src/parser.rs`.
    The third Rust fix replaced recursive descent with an iterative type-state shunting-yard parser in `parser.rs` to eliminate stack overflow on deep nesting and 5,000-term binary chains.
    Related parser hardening boxed `Points` in `Node::Table`, shrinking `Node` from 64 to 24 bytes, and made `Expression::tree()` O(N) via `Option::take`.
    Iterative implementations also replaced recursion in `tape::compile`, `Node`/`Term` `Drop`, and `jolty-python::term_to_json_string`.
- **Source:** row 20.

### 📋 19. A contributors' guide to adding a model

- **What:** write a guide walking through adding a model from a single declaration (`models.md` 4), plus a page documenting subtle pitfalls.
- **Needs:** `models.md` 4.
    **Done when:** a contributor adds a model by following the guide alone.
- **Source:** row 25.

### 📋 20. Shorter sentences in the guide, and how-to pages

- **Done when:** median guide sentence length drops under 25 words, and how-to pages cover common tasks.
    **Source:** row 26.

### ✅ 21. The cost of a model measured

- **What:** record the number of shared files touched by each device model.
- **Done when:** counts are recorded for every model; **Lead** once the count reaches 1 (`models.md` 4).
- **Result:** measured across all 33 models in `Library::standard()`, excluding per-model kernel files and inline `#[cfg(test)]` blocks, each model appears in **2 to 15** shared production files (**median 5**).
    Rust wiring spans `2` to `6` files: `crates/jolty-models/src/kernel.rs` and `lib.rs`, plus checks and engine hooks for core passives, sources, and diodes.
    Python wiring spans `1` to `8` files: `devices.py`, `spice/export.py`, `spice/library.py`, `spice/models.py`, and `kicad.py`, plus layout, schematic, catalog, foundry, or fidelity modules where specialized.
    Six primitives also appear in `src/jolty/assets/viewer.js` (`0` or `1` JavaScript file), and models appear in `1` to `3` shared guard tables (`testing/cases.rs`, `testing/laws.rs`, `tests/test_spice.py`).
    In git history, recent compact-model additions touched **6 to 7** shared production wiring files excluding kernel internals and engine-wide features.
    Observed wiring counts were `vbic` (`2017503`: 6), `bsim4` (`4fc8aac`: 6, plus `bsim3` helper exports), `bsim3` (`c60cb78`+`6fe8824`: 7), `behavioral_capacitor` (`04fb223`: 5), and `transmission_line` (`a39a72f`: 5 wiring plus 17 delay-history files).

| Model | Shared prod | Rust | Python | JS | Guard tables | Shared production files |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `resistor` | 15 | 6 | 8 | 1 | 3 | Rust: `checks/src/rules.rs`, `engine/src/kernels.rs`, `engine/src/lib.rs`, `ir/src/lib.rs`, `models/src/kernel.rs`, `models/src/lib.rs`. Python/JS: `py:assets/viewer.js`, `py:catalog.py`, `py:devices.py`, `py:files.py`, `py:kicad.py`, `py:layout.py`, `py:schematic.py`, `py:spice/export.py`, `py:spice/library.py` |
| `capacitor` | 14 | 5 | 8 | 1 | 2 | Rust: `checks/src/lib.rs`, `checks/src/rules.rs`, `engine/src/kernels.rs`, `models/src/kernel.rs`, `models/src/lib.rs`. Python/JS: `py:assets/viewer.js`, `py:catalog.py`, `py:devices.py`, `py:fidelity.py`, `py:kicad.py`, `py:layout.py`, `py:schematic.py`, `py:spice/export.py`, `py:spice/library.py` |
| `voltage_source` | 14 | 5 | 8 | 1 | 3 | Rust: `engine/src/kernels.rs`, `engine/src/lib.rs`, `ir/src/lib.rs`, `models/src/kernel.rs`, `models/src/lib.rs`. Python/JS: `py:assets/viewer.js`, `py:devices.py`, `py:fidelity.py`, `py:foundry.py`, `py:layout.py`, `py:schematic.py`, `py:simulate.py`, `py:spice/export.py`, `py:spice/library.py` |
| `current_source` | 11 | 3 | 7 | 1 | 2 | Rust: `checks/src/lib.rs`, `models/src/kernel.rs`, `models/src/lib.rs`. Python/JS: `py:assets/viewer.js`, `py:devices.py`, `py:foundry.py`, `py:layout.py`, `py:schematic.py`, `py:simulate.py`, `py:spice/export.py`, `py:spice/library.py` |
| `diode` | 11 | 3 | 7 | 1 | 3 | Rust: `engine/src/kernels.rs`, `models/src/kernel.rs`, `models/src/lib.rs`. Python/JS: `py:assets/viewer.js`, `py:devices.py`, `py:kicad.py`, `py:layout.py`, `py:schematic.py`, `py:spice/export.py`, `py:spice/library.py`, `py:spice/models.py` |
| `inductor` | 11 | 3 | 7 | 1 | 2 | Rust: `checks/src/rules.rs`, `models/src/kernel.rs`, `models/src/lib.rs`. Python/JS: `py:assets/viewer.js`, `py:catalog.py`, `py:devices.py`, `py:kicad.py`, `py:layout.py`, `py:schematic.py`, `py:spice/export.py`, `py:spice/library.py` |
| `vbic` | 7 | 2 | 5 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/library.py`, `py:spice/models.py` |
| `bipolar_transistor` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/models.py` |
| `bsim3` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/models.py` |
| `bsim4` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/models.py` |
| `coupled_inductors` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/library.py` |
| `jfet` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/models.py` |
| `mosfet` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/models.py` |
| `vdmos` | 6 | 2 | 4 | 0 | 3 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:kicad.py`, `py:spice/export.py`, `py:spice/models.py` |
| `voltage_controlled_current_source` | 6 | 2 | 4 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:layout.py`, `py:schematic.py`, `py:spice/export.py` |
| `voltage_controlled_voltage_source` | 6 | 2 | 4 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:layout.py`, `py:schematic.py`, `py:spice/export.py` |
| `behavioral_capacitor` | 5 | 2 | 3 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py`, `py:spice/library.py` |
| `behavioral_current_source` | 5 | 2 | 3 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py`, `py:spice/library.py` |
| `current_controlled_switch` | 5 | 2 | 3 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py`, `py:spice/models.py` |
| `foundry_mosfet` | 5 | 2 | 3 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:foundry.py`, `py:spice/export.py`, `py:spice/sky130.py` |
| `transmission_line` | 5 | 2 | 3 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py`, `py:spice/library.py` |
| `voltage_controlled_switch` | 5 | 2 | 3 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py`, `py:spice/models.py` |
| `behavioral_voltage_source` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py` |
| `current_controlled_current_source` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py` |
| `current_controlled_memristor` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py` |
| `current_controlled_voltage_source` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py` |
| `photocell` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py` |
| `thermal_capacitance` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/library.py` |
| `thermal_resistance` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/library.py` |
| `voltage_controlled_memristor` | 4 | 2 | 2 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py`, `py:spice/export.py` |
| `heat_source` | 3 | 2 | 1 | 0 | 2 | Rust: `models/src/kernel.rs`, `models/src/lib.rs`. Python: `py:devices.py` |
| `logic_cell` | 3 | 1 | 2 | 0 | 1 | Rust: `models/src/lib.rs`. Python: `py:devices.py`, `py:fidelity.py` |
| `logic_source` | 2 | 1 | 1 | 0 | 1 | Rust: `models/src/lib.rs`. Python: `py:devices.py` |

`py:` stands for `src/jolty/`, and Rust paths omit the leading `crates/jolty-` prefix.
Guard tables count occurrences across `crates/jolty-models/src/testing/cases.rs`, `crates/jolty-models/src/testing/laws.rs`, and `tests/test_spice.py`.

- **Source:** row 7.

### ✅ 22. Every dependency's licence checked, in both languages

- **What:** a dedicated jolty test reads `cargo metadata` and `importlib.metadata` to enforce one license allow-list across Rust and Python dependencies, replacing `cargo-deny` and `deny.toml`.
- **Result** (`d52e342`): `tests/test_licenses.py` passed for 174 crates and 55 distributions (plus optional `jax` and transitive dependencies added since).
    The allow-list gained eight Python licenses: BSD-3-Clause, MPL-2.0 for Hypothesis, PSF-2.0, MIT-0, MIT-CMU, 0BSD, Zlib, and CC0-1.0.
    **Source:** the owner, 15 November.

### ✅ 23. Known vulnerabilities checked

- **What:** run `uv audit` and RustSec `cargo-audit` in `scripts/check`, CI, and weekly on `main`.
- **Result** (`a13244b`): no active advisory; unmaintained `paste` 1.0.15 warns through faer.
    **Source:** the owner, 15 November.

### ✅ 24. Rust's version the latest stable, stated once

- **What:** pin the toolchain to the latest stable Rust release (1.100.0 on 15 November), keep `rust-version` aligned via a test, and bump the pin each release.
- **Result:** `rust-toolchain.toml` pins 1.100.0, every crate's `rust-version` matches, and `tests/test_rust_version.py` enforces equality.
    **Source:** the owner, 15 November.

### ✅ 25. cargo-nextest measured

- **What:** benchmark `cargo test --workspace` against `cargo nextest run` plus `cargo test --doc` on this Mac and in CI, adopting nextest only if measurements justify the switch.
- **Adopted** (`1e30843`): `scripts/check` and CI run nextest and doctests, cutting runtime from 209.5 s to 117 s.
- **Then** (`ab2c08c`, 17 November): sharding the three slow corpus tests 16 ways and compiling `jolty-numerics` at `opt-level = 2` in tests cut the suite from 302.4 s to 107.2 s on machine D.
    On 18 November, 655 tests took 197.5 s on 4 test threads, 81.5 s on 16, and 73.8 s on 64, flattening past 16 where the slowest tests set the floor.
- **Source:** the owner, 15 November.

### ✅ 26. Every message facts only

- **What:** "Feedback is facts" applies to every error message, including API misuse: errors report factual state, such as recorded devices, while the guide documents corrective usage.
    `AGENTS.md` codifies this rule.
- **Done when:** no error message instructs callers how to fix a call.
    **Source:** the owner, 15 November.
- **Result** (`8ac4df8`): "Feedback is facts" in `AGENTS.md` covers jolty API usage as well, leaving usage guidance to the guide.
    Auditing every message string across `src/jolty` and the Rust crates for prescriptive words ("use", "pass", "try", "instead", "should", "give", "did you mean") found one match: the compiler CLI usage text, now rewritten factually.

**Elsewhere:** the library-wide physical-law suite is `models.md` 3, and single-source model declarations plus the model descriptor are `models.md` 4.

## 5. Decided

- 15 Nov, the owner: `mkdocstrings-python` stays as the Zensical API plugin; ported kernels retain reference local names inside each kernel, with functions citing upstream source lines.
- `jolty.cli.CommandLine` and `jolty.cli.run` remain public so the jolty CLI and repository scripts share one configuration.
- Guide claims unsupported by code were corrected (`7edbffa`), and the VBIC section links the generated models reference rather than repeating a manual table.
- 20 November (owner): dropped automated API-stability diffing against prior releases (former item 16).
- 20 November (owner): `jolty_engine::measure::Signal` keeps its name, matching the measures' `signal` parameter (item 12).
- 15 Nov, the owner: adding any tool or dependency requires owner approval; ty is the sole type checker, Zensical with mkdocstrings is the sole docs tool, and mypy was removed (`5ba0fb2`).
- 15 Nov, the owner: each concept has one uniform, consistent, symmetric, formal, and principled name (`AGENTS.md`, "Names").

## 6. Open principles

None: on 15 November the owner decided that ported kernels retain reference names internally, "Feedback is facts" applies to every message (item 26), and `cargo-deny` is removed (item 22).
