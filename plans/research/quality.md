# Research: quality, the cleanest and most elegant

15 November 2026, for the lead's `quality.md`.
Read-only research: nothing was built, run or edited in the repository.
jolty's files are cited at `9545e08`, the local `main` in the repository: GitHub's `7d41cc0` (VBIC, Phase 6 and the parallel benchmarks) plus the commit that adds `NOTICE`.
`main` moved during the research, from `d10fc4d`: VBIC was rebased as `2017503`, and the notices commit added four or five lines at the top of each ported model; the citations follow `9545e08`.
ngspice is cited at 47's unpacked source and Xyce at its clone, both in the session scratchpad.
Counts come from `git ls-files`, `grep` and small read-only scripts over the sources (in the scratchpad's `q/`); an estimate is marked as one.

## 1. Summary

- jolty's mechanical discipline is the strictest of the circuit simulators examined: every ruff rule, every ty rule as an error, clippy's pedantic lints as errors, about 40 Rust lint suppressions in 57,000 lines and 196 Python ones, each with its reason, the import order checked by a test, and the reference pages generated and diffed in CI.
    ngspice's tree has no lint, coverage or fuzzing setup; Xyce's CI adds coverage and sanitizer builds.
- Where it falls short is the spirit of `AGENTS.md` rather than its letter: one job is done several ways, and one fact is written in several places.
    Three semiconductor kernels write their derivatives by hand and four on dual numbers; since 15 November the MOSFET stamps through summed `Derivatives` while BSIM3, VDMOS and VBIC stamp term by term; a parameter is written at six sites in four files, near ngspice's seven in five; VBIC's commit touched 14 shared files, ngspice's VBIC touches 6.
    Guards cover the cases someone listed: the allocation test runs 16 of the 28 analog models.
- The Python API is small, immutable and precisely typed, but half-dimensioned (options, overrides, variants and parameter sweeps take SI floats or check only against their own first value), resolves the names a spec uses only after every analysis has run, and keeps one result fact, `excursions`, as simulator state.
- The guide is precise and mostly backed by the code (all 84 keyword calls in its snippets match the API's signatures), but nothing runs its 41 Python blocks or checks its 19 quoted outputs; three of its statements are stale and VBIC, alone of the semiconductors, has no parameter table.
- Move 1: one definition per model fact.
    Each parameter's name, SPICE name, dimension, default and description is declared once with its kernel, and the Python constructor, the SPICE reader and writer, the guide's tables and every unit check derive from it or are tested against it; then one mechanism per job (every kernel on `Dual`, one stamping path).
- Move 2: guards that enumerate and numbers that cite.
    Every test of a model property iterates the library, every engine constant and test tolerance names its source, the guide's code runs, and names and dimensions resolve before the first analysis.
- Move 3 (lead): quality measured as speed is.
    Mutation testing on every diff and a weekly sharded run, coverage, fuzzing of the parsers, API-stability checks, and the cost of a model counted in shared files touched, kept as a suite with a baseline.
    No reference simulator measures itself this way.

## 2. Where jolty stands

### 2.1 What it measures today

| Check | What it holds | Evidence |
| --- | --- | --- |
| Format | ruff format, rustfmt, rumdl with one sentence per line; the Markdown's Python blocks are linted by ruff | `scripts/check:22-32`, `.rumdl.toml` |
| Lints | ruff `select = ["ALL"]` with `COM812` and `TC` off; clippy pedantic and eleven restriction lints, as errors; `unsafe_code = "forbid"`, `missing_docs`, `unreachable_pub` | `pyproject.toml` `[tool.ruff.lint]`, `Cargo.toml` `[workspace.lints]` |
| Suppressions | About 40 Rust `#[expect]` lint suppressions, each with its reason, none a bare `allow` (10 `similar_names`, 6 `too_many_lines`, 4 `expect_used`, 4 `float_cmp`, ...); 196 Python `noqa`, each with its reason; ruff's complexity rules `C901`, `PLR0912` and `PLR0915` never suppressed | a parse of every `#[expect(...)]`; `git grep noqa` |
| Types | ty with every rule an error, over the sources, tests, examples, docs, benchmarks, discoveries and fixtures | `pyproject.toml` `[tool.ty]` |
| Tests | 463 Rust `#[test]` and 27 proptest blocks; 465 Python test functions, 19 of them Hypothesis `@given`; 252 doctest lines in `src/jolty`; the 30 example scripts run | `git grep`; `scripts/check:33-37` |
| Differential | The ngspice corpus, vendors' macromodels and random circuits, a skip failing in CI | `AGENTS.md:119-126`, `.github/workflows/ci.yml` |
| Architecture | The crate and module order of `AGENTS.md` checked by a test | `tests/test_architecture.py` |
| Docs | The reference pages regenerated and diffed, the site built | `scripts/check:42-44` |
| CI | Rust and Python groups on `ubuntu-24.04` and `macos-15`, the tests again on free-threaded Python, the docs on Linux; the Python jobs take 8 to 9 minutes | `ci.yml`, `docs.yml`, `handoff-2026-11-14.md:143` |

What nothing measures: test coverage; mutation testing; fuzzing; API stability, though every crate but `jolty-python` publishes to crates.io and the package to PyPI (`release.yml`); rustdoc's own lints, since `cargo doc` never runs, so a broken intra-doc link passes; the hand-written type stub of the extension against the extension; the guide's code; the dependencies' licences, which `AGENTS.md:70-71` rules on and no tool checks; the declared `rust-version = "1.97"` against the pinned toolchain 1.100; and the wheels of five of the ten release targets, Windows, Linux on ARM and macOS on Intel, which `release.yml:44-53` builds and no job imports.

Size, for scale: 56,707 lines of Rust in `crates/*/src` (their in-file tests included) and 10,201 in `crates/*/tests` and benches; 27,645 lines of Python in `src/jolty` and 19,225 in `tests`; 2,198 lines of guide in `docs/guide`.

### 2.2 The audit against `AGENTS.md`

**Asymmetric code: one job, several ways.**

- **A1.
    Derivatives are written two ways.**
    The MOSFET levels 1 and 3, BSIM3, VDMOS and VBIC compute on `Dual<R, N>`, so their Jacobians are exact by construction (`dual.rs:1-6`).
    The diode, the JFET and the Gummel-Poon transistor carry hand-derived `(value, derivative)` pairs: the BJT's `State` holds `(R, [R; 2])` tuples whose slopes are written out formula by formula (`bjt.rs:331-430`), the JFET writes `gm` and `gds` as ngspice does (`jfet.rs:220-241`), and the junction current returns its slope by hand (`junction.rs:154-169`).
    `R: Real` makes parameter derivatives automatic in both kinds; voltage derivatives are automatic only in the second.
    The research of the first wave says "every kernel writes its equations once on `Dual` numbers" (`research/models.md` §2.2); the code does not back that for these three.
    Hand-written slopes are where jolty finds ngspice wrong (`devices.md:380`, a missed slope of the BJT's base resistance), and jolty's own are guarded only by A11's central difference.
- **A2.
    Transistors stamp three ways since 15 November.**
    `52e3bdb` and `4f39ff4` moved the MOSFET onto `parts::Derivatives`, which sums the 32 entries in registers, and onto one-voltage duals for its junctions and overlaps, after measuring that writing each derivative as it came took 40% of its evaluation, and that 40 of the 84 derivatives it summed for its junctions and overlaps were zeros (their messages).
    BSIM3 still stamps its bulk junctions as three-variable duals term by term through `Oriented::current` and `Oriented::charge` (`bsim3.rs:389-402`, `bsim3.rs:523-551`), as do VDMOS (`vdmos.rs:603-613`) and VBIC with nine variables (`vbic.rs:1882-1933`).
    `kernel/parts.rs` now holds both versions of the same contract, `current`/`charge` (lines 473-512) and `add_current`/`add_charge` (lines 518-570), kept in step by a property test.
    Whether BSIM3, VDMOS and VBIC pay what the MOSFET paid is unmeasured.
- **A3.
    Device constructors follow three naming conventions.**
    Most take English keywords (`bipolar_transistor`'s 45, `vbic`'s 103: `devices.py:986`, `devices.py:1131`).
    `bsim3` takes English keywords for the instance's geometry, SPICE's names for its selectors (`delvto`, `mulu0`, `mobmod`, `capmod`, `noimod`, `paramchk`, `xpart`), and a `card` mapping keyed by SPICE names for its 400-odd model parameters (`devices.py:2126-2146`).
    The line falls where the parameter count grew, not where a principle puts it; a VBIC card's 103 SPICE names must each be translated by hand or through the reader, and the guide gives no table for it (G4).
    The convention leaks into the exporter: `_model_card` sends BSIM3 down a path of its own (`spice/export.py:237`).
- **A4.
    Ports keep their reference's names, other kernels jolty's.**
    BSIM3's evaluation is "one pass through b3ld.c's drain current, in its order, by its names" (`bsim3/channel.rs:101-105`), with `t0`, `t1`, `tmp2` reassigned through it (`channel.rs:110-164`), while the BJT reads `forward_transport` and `base_resistance_current`.
    Both choices are defensible (a port diffable against its source; a kernel readable on its own); `AGENTS.md` states neither.
- **A5.
    The guide documents models unevenly.**
    Every semiconductor section has a parameter table of jolty's name, SPICE's name, unit and default, from 12 rows (BSIM3) to 38 (MOSFET); VBIC, with the most parameters but BSIM3, has none (`devices.md:496-516`; rows counted per section).

**One fact, many places.**

- **A6.
    A model parameter is written at six sites in four files.**
    For the BJT's `IS`: the field with its SPICE name, unit and default in prose (`bjt.rs:157-158`), the default again (`bjt.rs:265`), the keyword (`devices.py:989`), its dimension (`devices.py:1074`), its SPICE name (`spice/models.py:80`), and the guide's row (`devices.md:340`).
    The default is written three times, the unit three times, the SPICE name three times.
    Nothing checks the Python keywords against the Rust fields; a script found them equal for the diode, JFET, BJT, MOSFET, VDMOS and VBIC, and found each of the 85 numeric defaults that Rust's prose states equal to its `Default` impl, so the discipline holds today, by hand.
    ngspice writes the same parameter at seven sites in five files (§3.1); Xyce at two (§3.2).
- **A7.
    The dimension of a parameter is not data.**
    It lives only in a dictionary local to each constructor (`devices.py:1073-1099`) and in Rust prose ("in amperes").
    So every path but the constructors checks a value against nothing: `Axis` converts its values to "the dimension of the first" (`plan.py:597-650`), a `DcSweep` of a parameter likewise (`simulate.py:816-822`), and `Variant`, `Simulation(overrides=...)` and `Conditions` take SI floats (`plan.py:532-563`, `simulate.py:330-357`).
    `Axis("R1", "resistance", [1 * nF, 2 * nF])` passes Python's checks and sends 1e-9 Ω to the engine (by reading; not run).
    The units module's own promise, "a value in farads cannot go where ohms belong" (`units.py:2`), holds only at a constructor.
    This keeps the letter of `AGENTS.md:83`, which names the constructors, not its spirit.
- **A8.
    The four parallelism knobs are defined four times and checked three.**
    `plan.Parallelism` (a dataclass checked in `__post_init__`, `plan.py:351-397`), `runner.Thresholds` (pydantic, `Field(ge=1)`, `runner.py:110-118`), the service's request fields and their copy into a `Thresholds` (`service.py:323`, `service.py:534-539`), and the engine's `Parallelism` (`options.rs:74-80`, its check at `options.rs:383-384`).
    A fifth knob touches at least five places.
    `AGENTS.md`'s "One definition" covers commands only (`AGENTS.md:102-103`).
- **A9.
    The extension's contract is written twice by hand.**
    `_core.pyi` is 377 hand-written lines, among them 18 `TypedDict`s describing dictionaries that Rust builds with 117 `set_item` calls (`crates/jolty-python/src/*.rs`).
    Its 15 functions and 7 classes match the Rust ones today (a script compared names); no test compares the keys, and ty trusts the stub.

**Shared files every new model touches.**

- **A10.**
    VBIC's commit touched 15 files, 14 of them shared: `lib.rs`, the allocation test, the kernel corpus, `devices.py`, `kicad.py`, the SPICE exporter, library reader and model reader, two test modules, the guide, the generated models page, `CHANGELOG.md` and `clippy.toml` (`git show --stat 2017503`); BSIM3's two commits touched 6 and 15, with 375 lines in `devices.py`, 218 in the reader and 63 in the exporter (`c60cb78`, `6fe8824`).
    Knowledge of a model is spread by name: the string `"vbic"` branches at least nine times outside its constructor, for its SPICE letter and terminal order (`export.py:129`), its card's names and its level (`export.py:225`, `export.py:268`), its temperature convention (`export.py:259`, `export.py:447`), its node count (`spice/library.py:1026`) and its designator letter (`kicad.py:45`).
    The dispatchers grew past ruff's complexity bound and were split to pass it, "`_bipolar_device` and `_card_names`, which keep the functions that call them within ruff's complexity bound" (`2017503`'s message): the bound kept by its letter, the dispatch by model name left in place.

**Guards that cover the cases someone listed.**

- **A11.
    The Jacobian check that guards every hand-written derivative allows a relative error of 1e-5 without saying why.**
    `assert_bench_jacobians_match` compares each derivative with a central difference of step `1e-6·max(|x|, 1)` within `1e-5 · max(|analytic|, |numeric|)` plus rounding (`testing.rs:245-290`, the tolerance at line 284); its comment derives the rounding term, not the 1e-5.
    A term missing from a hand-written slope that is below a hundred-thousandth of it passes.
    On a `Dual` kernel the check is redundant but for the stamping code.
- **A12.
    The allocation test covers 16 of the 28 analog models.**
    Its `CASES` list (`crates/jolty-models/tests/kernels.rs:67-168`) leaves out the inductor, the current source, the four linear controlled sources, the voltage-controlled switch, the current-controlled memristor, the behavioral current source, the two thermal elements and the foundry placeholder; some share code with a covered model, which the test cannot know.
    `AGENTS.md:98` says "a test counts a kernel's allocations", and one does; `tests/test_spice.py:969-982` shows the stronger pattern, asserting its models are `_core.models()`.
- **A13.
    The kernels' panic lints are pasted into 22 files, and code kernels call runs without them.**
    The same eight-line `cfg_attr(not(test), deny(clippy::indexing_slicing, clippy::panic))` block opens `behavioral.rs` through `vdmos.rs` (for instance `bsim3.rs:1-8`).
    `waveform.rs`, whose `piecewise` every source kernel calls at every evaluation, has no such block and indexes slices directly on an invariant established elsewhere (`waveform.rs:230-241`); `heat.rs`, the `Thermal<K>` wrapper of every heated kernel, has none either.
    One attribute on a parent module of the kernels would cover them all and every kernel to come.

**Numbers without a stated source.**

- **A14.
    The engine's step-control and continuation constants say what they mean, not where they come from.**
    Some are ngspice's without saying so: `FAILURE_SHRINK = 0.125` is `dctran.c:733`'s `/8`, `RESTART_FRACTION = 0.1` is `dctran.c:507`'s `.1`, `FIRST_STEPPED = 1e-2` and `SMALLEST_STEPPED_DIVISOR = 1.00005` are `cktop.c:189` and `cktop.c:223-234` (`transient.rs:25-29`, `operating_point.rs:29-34`).
    Others are jolty's own and differ from ngspice's: `SMALLEST_STEP_FRACTION = 1e-9` where ngspice's transient takes `1e-11` (`traninit.c:36`), and source stepping's first increment of 0.1 and floor of 1e-4 where ngspice's starts at 0.001, grows by 1.5, and gives up below 1e-7 (`operating_point.rs:36-38`, `cktop.c:555-641`).
    ngspice makes the gmin factor an option, `gminfactor`; jolty's `STEPPED_DIVISOR = 10.0` is a constant, where "knobs over opinions" (`AGENTS.md:89-90`) asks for an option.
    The brief's "no tuned heuristics" is not in `AGENTS.md`, so no rule requires these to cite.
- **A15.
    Many test tolerances do not say where they come from.**
    A crude scan finds 428 literal tolerances in tests, 340 with no comment within the four lines above (an upper bound: a docstring further up may explain one).
    Examples that state nothing: the divider's `1e-12` and `1e-9` (`crates/jolty-engine/tests/analytic.rs:57-60`, `analytic.rs:102`, `analytic.rs:131-135`), `rel=1e-6` on a divider at any scale (`tests/test_simulate.py:128`), and `rel=1e-6`, `rel=1e-4` on windowed measurements of sampled waveforms (`tests/test_specs.py:142-150`).
    Even the ngspice corpus states its transient tolerance as "a small fraction of their swing" (`tests/test_ngspice.py:6-8`) for its `1e-3` (`test_ngspice.py:1958`).

**Rules the code does not yet meet, or contradicts.**

- **A16.
    Parallelism's defaults are "none", not measured break-evens.**
    `AGENTS.md:91-93` says each default is a measured break-even; every threshold defaults to off (`plan.py:351-382`, `simulation.md:317-319`), and the break-evens the guide gives were measured at a load average of 7 to 19 (`simulation.md:339-355`).
    the session log, not kept makes this next item 2; until it lands the rule is unmet.
- **A17.
    One parallelism knob changes the bits, against the policy's name.**
    `max_unknowns_per_interior` reorders the factorization's arithmetic (`plan.py:360-367`, `simulation.md:335`).
    The fingerprint special-cases it (`runner.py:813-836`), and the guide contradicts itself: "The parallelism of the options changes no result, and the fingerprint leaves it out" (`machines.md:50`).
    An ordering of operations is a numerical choice; inside `Parallelism` it makes "serial and fully parallel are two configurations of it" (`AGENTS.md:91`) a promise with an exception.
- **A18.
    Options report their first problem.**
    `Options::validate` "returns ...
    naming the first setting out of range" (`options.rs:341-357`), where `AGENTS.md:106` has the engine report every problem it finds; the rule is worded for circuits, so its scope for plans is open.
- **A19.
    "Feedback is facts" is unclear for misuse of the API.**
    A result asked for an unrecorded current answers "record its instance, as an analysis's or a plan's record=[...] does, or every instance, as record=['**'] does" (`results.py:316-325`): how to fix it.
    That is good API practice (rustc's "help" lines do the same); `AGENTS.md:104-107` speaks of the circuit and does not say whether it covers this.
- **A20.
    CI runs a job the pre-push hook does not.**
    The free-threaded job runs `pytest` outside `scripts/check` (`ci.yml:70-100`), so "no push reaches CI without passing it" (`AGENTS.md:34`) does not hold for it, and `AGENTS.md:33`'s description of the jobs leaves it out.

**What the audit found clean.**

- **A21.
    The docs check works.**
    At `d10fc4d` VBIC was in `Library::standard()` and `docs/reference/models.md` had 29 sections, none for `vbic`, though `docs/reference.py:298-312` writes one per `_core.models()`; the rebased commit `2017503` carries the page's 111 new lines, so the full check caught it (`scripts/check:42-43`).
    The page is generated; the guide's tables, which A5 and A6 are about, are not.
- No dead code by the scan: every top-level Python function or class but four pydantic validators is referenced elsewhere, and every `pub fn` of the crates too.
- Every suppression gives its reason; `unwrap`, `expect` and `panic` outside tests appear only under `#[expect]` with their invariant (for instance `jolty-ir/src/lib.rs:339-342`).
- Each departure from ngspice that VBIC's commit names is in the guide with its line of ngspice (`devices.md:503`, `devices.md:511`), as `AGENTS.md:87-88` asks.
- Names: mostly plain and searchable, with three exceptions.
    `Pair::add_current` stamps a current while `Oriented::add_current` sums derivatives without stamping them, in the same file (`parts.rs:155`, `parts.rs:518`); `min_unknowns_inside` names an internal type, `Inside` (`options.rs:77-80`, `options.rs:182-190`), and breaks the `min_*_per_task` pattern of its neighbours; `rise(net)` reads a thermal net's temperature (`results.py:427-429`), where a reader looks for `temperature`.

### 2.3 The Python API

**What it does well.**

- A small surface: `jolty/__init__.py` exports 44 names, the rest by module.
- Values are immutable: frozen dataclasses throughout `plan.py` and `results.py`, and modified copies by verb (`rated`, `numbered`, `toleranced`, `held_at`), as JAX's arrays and Polars' frames are.
- Precise types: `Simulation.run` is overloaded per analysis, so each returns its own result type to ty (`simulate.py:553-590`), and the results share one generic reader, `_Unknowns[T]`, so `voltage`, `current` and `state` read alike from a point, a sweep, a response or a transient (`results.py:397-452`).
- Specs are expression objects (`gain("bode", "out", at=1 * kHz) >= -3`) that the engine reads natively in Rust (`specs/__init__.py:23-28`), the design Polars gives its expressions.
- Determinism by index: "sample `i` from the seed and `i` alone" (`plan.py:505-509`), the discipline JAX imposes with explicit random keys.
- Messages name the instance and the script line (`builder.caller()`, `builder.py:444-458`); a misspelt net in a result names the close ones (`results.py:350-372`), and a misspelt terminal lists the instance's terminals (`builder.py:101-108`).

**Where a user or an agent stumbles.**

- **S1.
    Half the numbers carry no dimension** (A7): `Options`' tolerances and `memory` are floats in SI base units and gibibytes (`plan.py:414-426`), `memory`'s unit appearing only in the guide; overrides and variants are SI floats; axes and parameter sweeps check only against their own first value.
- **S2.
    Names that join a spec to its plan resolve after the run.**
    A spec naming an analysis the plan lacks becomes "unmeasured: the plan runs no analysis named bdoe" once every analysis has run (`specs/_requirements.py:797-799`), without the close names a result's reader gives.
    Polars' lazy API checks a query's schema before it reads any data; a compiler resolves names before it generates code.
- **S3.
    A result's facts live on the simulator.**
    `Simulation.excursions` holds "the last analysis's" excursions outside the logic models' domains (`simulate.py:324`, `simulate.py:603-618`), and the runner must read it right after each run (`runner.py:1140-1146`): the result depends on the order of calls, which JAX's pure functions rule out.
- **S4.
    Two public classes share each of two names.**
    `jolty.plan.OperatingPoint` and `OperatingPoints` are analyses; `jolty.results.OperatingPoint` and `OperatingPoints` their results (`plan.py:85`, `plan.py:94`, `results.py:512`, `results.py:547`), while the other results take other names (`Sweep`, `Response`, `Waveforms`, `Sensitivities`, `LoopGain`, `Transfer`, `NoiseSpectrum`, `PolesZeros`).
    Both modules render on one reference page with short names (`docs/reference/simulation.md`, `zensical.toml:123`).
- **S5.
    Devices take three naming conventions** (A3), and VBIC's 103 keywords have no table mapping them to a card's names (A5).
- **S6.
    Flags and keyword spaces that overload meaning.**
    `Transient(periodic=True)` turns `stop` into the period of a steady state (`plan.py:163-193`), where a separate analysis would say so in its name.
    `Corner(temperature, process, **arguments)` shares one keyword space between its own fields and `build()`'s arguments (`plan.py:461-495`), so a corner cannot pass a `build()` parameter named `temperature` or `process`.
- **S7.
    Like operations take unlike shapes.**
    `DcSweep.to_json(level)` takes the dimension of its levels where every other analysis's `to_json()` takes nothing (`plan.py:136`, `plan.py:88`); `Simulation.run_plan(plan)` takes a plan where `Simulation.batch(analyses, variants, options, ...)` takes its pieces (`simulate.py:621-684`).
- **S8.
    Validation is split.**
    `Parallelism` checks its fields in Python, `Options` in the engine only, first error first (A18); `Plan` checks with `isinstance` in `__post_init__` (`plan.py:860-903`), where the file formats and the service use pydantic, whose `ValidationError` reports every error at once.
- **S9.
    Typos are found by running, not by `ty`.**
    Terminals are attributes through `__getattr__` (`builder.py:101-108`) and every quantity is one `Quantity` type, so `q1.colector` and `resistor(1 * nF)` pass `ty check` and fail at build time, with good messages.

**Against other libraries.**

- **NumPy and scikit-rf** hold one data model: an `ndarray`, or a `Network` whose `s`, `z0` and `frequency` are arrays of shape (frequencies, ports, ports) for every network.
    jolty's results come close (`_Unknowns[T]`), and lose it in their names (S4).
- **JAX** asks for pure functions and explicit random keys, and lists its pitfalls on one page, "The Sharp Bits".
    jolty's batches are its `vmap` and its sensitivities its `grad` in all but name; S3 is the impurity JAX would refuse; S1, S6 and A17 are what a sharp-bits page of jolty's would list.
- **Polars** builds expressions into a plan it checks before running; jolty's specs are such expressions, unchecked until run (S2).
- **pydantic** validates at construction and reports every error; jolty uses it for its files and requests but not for plans (S8).
- **gdsfactory** builds layouts from decorated, cached component functions whose ports are named, `ref.connect("o1", other.ports["o2"])`: jolty's `@component` is the same design (`builder.py:357-381`), with attribute access for terminals.
- **PySpice** places elements by SPICE letter, `circuit.R("R1", "n1", circuit.gnd, 1 @ u_kΩ)`, nodes as strings; **Xyce's Python interface** is a ctypes wrapper with `initialize(netlist)`, `runSimulation()`, `simulateUntil(t)` and integer status codes, for co-simulation (`utils/XyceCInterface/xyce_interface.py.cmake`, 570 lines); **Cadence's OCEAN** is SKILL procedures over one global session, `simulator('spectre)`, `analysis('tran ...)`, `run()`, `selectResult('tran)`, and its users report Monte Carlo results that `selectResult` no longer selects.
    None of the three types a circuit or a result; jolty is ahead of all three, and the gaps above are against the scientific libraries, not the circuit ones.

### 2.4 The Rust crates

- The layering, the kernel contract and the ban on `unsafe` hold, and are checked (`test_architecture.py`, `Cargo.toml` `unsafe_code = "forbid"`).
- Every public item has documentation (`missing_docs` as an error), but the crates hold seven fenced examples in all, about one per crate (`git grep` of fences in `crates/*/src`), where the Rust API Guidelines ask an example of every item (C-EXAMPLE) and rustdoc runs them as tests; the crates publish to crates.io at 0.1.0 with no semver check.
- `Dual`'s value is read as `x.value.value()` 87 times in the kernels (`git grep -c`), 50 of them in BSIM3: a method would read better.
- The `Kernel` trait is well documented (`kernel.rs:1-160`), but nothing tells a newcomer how to add a model: no guide page, crate README or checklist names the 14 shared files of A10.

### 2.5 The guide

**Shape.**
Sixteen pages, one concept each (`docs/guide/index.md`), a generated reference (`docs/reference.py`), a get-started page and a gallery.
It names its sources (papers with authors, ngspice's files and lines), states every departure from ngspice with its evidence, and gives each device parameter its SPICE name, unit and default.
It is reference and explanation first; the one tutorial is `get-started.md`, and there are no how-to pages ("run a corner sweep", "read a vendor's model", "find why a transient fails"), the kind Diátaxis calls how-to guides.

**For a designer and an agent.**
Its prose is dense: a median of 27 words a sentence, a fifth of sentences above 40 words, the longest 93 (`devices.md`), by a count over every prose line of the guide; GOV.UK's style guide asks to split sentences over 25 words.
The density suits an agent that reads everything and costs a designer who scans.

**Claims the code does not back.**

- **G1.**
    "The parallelism of the options changes no result, and the fingerprint leaves it out" (`machines.md:50`): false for `max_unknowns_per_interior`, which the fingerprint keeps (`runner.py:832-836`) and the simulation page treats correctly (`simulation.md:317`, `simulation.md:335`).
- **G2.**
    "plans, the five analyses and the numerical options" (`guide/index.md:14`): the simulation page lists ten (`simulation.md:27-40`).
- **G3.**
    "`Options` sets the tolerances ...
    and what results record" (`simulation.md:303`): no option records anything; `record` belongs to the plan and to the recorded analyses (`plan.py:819-857`).
- **G4.**
    VBIC has no parameter table (A5): "Its parameters take the names of a VBIC card's meaning, such as `extrinsic_collector_resistance` for `RCX`" (`devices.md:500`) gives one of 103, so a designer with a card finds the other names only on the generated models page or in `devices.vbic`'s signature.
- **G5.**
    The quoted outputs, such as "saved low_pass.jolty.json: 2 components, 3 devices, 3 unknowns", "took 0.27 s" and "transient to 10 µs in 67 steps" (`get-started.md:85-105`, `README.md:32`), are checked by no test; the get-started run shows only the transient's report of a plan of three analyses.
- What holds: all 84 calls with keyword arguments in the guide's and the README's Python blocks name only parameters the API has (a static scan against the signatures), and its departures from ngspice cite lines (`devices.md:380`, `devices.md:447`, `devices.md:503`).

## 3. The references

### 3.1 ngspice 47: a directory of hand-written C per device

- **Layout.**
    Each device is a directory of 20 to 30 files, one per entry point of the device table: `bjt/` holds 30 C files and headers, 9,379 lines, among them `bjtload.c` (DC and transient), `bjtacld.c` (AC), `bjtpzld.c` (pole-zero), `bjtsload.c` and `bjtsacl.c` (sensitivities), `bjtdisto.c` and `bjtdset.c` (distortion), `bjtnoise.c`, `bjttemp.c`, `bjtsetup.c`, `bjtbindCSC.c` (the KLU binding), `bjtparam.c`, `bjtmpar.c`, `bjtask.c` and `bjtmask.c` (`wc -l` on the tree).
- **One matrix entry in six files.**
    The entry from the base to the internal collector is declared in `bjtdefs.h:157`, allocated in `bjtsetup.c:530`, bound three times in `bjtbindCSC.c:56`, `115` and `174`, and stamped by hand in `bjtload.c:934`, `bjtacld.c:105` and `bjtpzld.c:100-101`.
    Each analysis restates the stamp, so a model change is made in several files that must agree.
- **One parameter at seven sites in five files.**
    `IS` is in the `IFparm` table (`bjt.c:84`), the header three times, its field, its `Given` bit and its enumerator (`bjtdefs.h:406`, `542`, `698`), the setter (`bjtmpar.c:45-47`), the getter (`bjtmask.c:42-43`) and the default (`bjtsetup.c:47-48`).
- **Shared files.**
    A device also enters `src/Makefile.am`, `src/spicelib/devices/Makefile.am`, the device table `dev.c`, the level dispatch `inpdomod.c`, its element parser (`inp2q.c` for VBIC) and `configure.ac:1390`: six, beside VBIC's 21 files (a search of `src/` for `vbic` outside its directory).
- **Testing.**
    Regression decks with expected outputs and, for BSIM3, the QA references of its model (`tests/bsim3/nmos/reference`, 14 files); no coverage, fuzzing or lint configuration in its build files (a search of its `*.am`, `*.ac`, `*.sh` and `*.yml` files finds only libtool's).
- **What makes it hard to change:** every stamp restated per analysis, every parameter restated in five files, derivatives by hand (the errors `devices.md:380`, `447` and `488` list), and a copy of the code per model version (`BSIM4v5` to `BSIM4`, `research/models.md` §3.1).

### 3.2 Xyce: C++ classes, a declarative parameter table, and generated code

- **Layout.**
    A device is a C++ module of `Traits`, `Model`, `Instance` and `Master`: `N_DEV_BJT.C` is 7,053 lines and its header 1,138, with `registerLIDs`, `registerJacLIDs`, `setupPointers`, `loadDAEFVector`, `loadDAEQVector`, `loadDAEdFdx`, `loadDAEdQdx`, and the `Master`'s batched `loadDAEVectors` and `loadDAEMatrices` (`N_DEV_BJT.C:1333-4400`).
- **Sparsity by hand, per variant.**
    The BJT keeps eight static Jacobian stamps, one per combination of base, collector and emitter resistances present or collapsed (`N_DEV_BJT.C:767-774`).
    jolty decides each internal node at layout (`Series::behind`, `parts.rs:276-295`) and derives the pattern from what the kernel claims.
- **One Jacobian entry at a dozen sites.**
    `BaseEquBasePNode` appears in the constructor's initializers (`N_DEV_BJT.C:986`, `1025`, `1070`), the offset map (`1680`), the pointers (`1748`, `1781`), the instance's load (`2413`), the master's load (`4415`) and its fallback (`4546`), and three times in the header.
- **A parameter at two sites, its unit as data.**
    One `addPar("IS", 1.e-16, &BJT::Model::satCur).setUnit(U_AMP).setDescription(...)` call (`N_DEV_BJT.C:127-133`) and the member's declaration; `Xyce -doc` writes the Reference Guide's parameter tables from these calls (`doc/Reference_Guide/compareLatexTables.py:17-18`, `N_DEV_LaTexDoc.C`).
    On this point Xyce is ahead of jolty, whose units are prose and whose guide tables are hand-written.
- **Generated models.**
    Its 25 Verilog-A models are 856,148 lines of ADMS output checked in, not regenerated by the build, and patched by hand (`research/models.md` §3.2).
- **Testing.**
    Its GitLab CI runs coverage and sanitizer builds (`gitlab-ci/analysisBuilds.yml:67-74`); no mutation testing or fuzzing appears in its CI or CMake files.

### 3.3 What jolty already does better, and worse

| | ngspice | Xyce | jolty |
| --- | --- | --- | --- |
| A matrix entry | 6 files, restated per analysis | about 12 sites in 2 files | claimed once at layout; one stamp of `G` and `C` serves DC, transient, AC, pole-zero, noise and sensitivities |
| Derivatives | by hand | by hand (native), generated (ADMS) | by `Dual` in 4 kernels, by hand in 3 (A1) |
| Parameter derivatives | finite difference, `Sens_Delta = 1e-6` (`cktsens.c:24`) | analytic, functors per parameter | automatic in every kernel, through `R: Real` |
| A parameter | 7 sites, 5 files | 2 sites, unit as data, guide tables generated | 6 sites, 4 files, unit as prose (A6, A7) |
| Shared files per model | 6 | the device registry and build files (not counted) | 14 (A10) |
| Static checks | none found | coverage, sanitizers | every lint as an error, types, architecture; no coverage (§2.1) |
| Reference pages | a hand-written manual | tables generated by `Xyce -doc`, compared by a script | generated and diffed in CI |

jolty leads on what a kernel costs and on how far its static checks reach; it trails Xyce on parameter metadata and both on the cost of integrating a model, the part outside the kernel.

### 3.4 The state of the art in measuring quality

- **Mutation testing.**
    Google runs it on code review, diff by diff, skipping uncovered and "arid" lines: over 70,000 diffs, 1.1 million mutants, 150,000 actionable findings, and the usefulness developers reported rose from 20% to 80% as uninteresting mutants were filtered (Petrović and Ivanković, "State of Mutation Testing at Google", ICSE-SEIP 2018).
    `cargo-mutants` mutates Rust function bodies and operators, tests only the mutants a diff touches with `--in-diff`, and shards a full run across machines with `--shard k/n`; `mutmut` mutates Python and resumes an interrupted run.
- **Property testing.**
    Hypothesis generates and shrinks inputs (MacIver and Hatfield-Dodds, "Hypothesis: A new approach to property-based testing", JOSS 2019); jolty uses it and proptest already, more than either reference.
- **API stability.**
    `cargo-semver-checks` lints a crate's rustdoc JSON against its last release for semver breaks; `griffe check`, from the library under jolty's own mkdocstrings, does the same for a Python package against a Git reference.
- **Stubs.**
    mypy's `stubtest` imports a module and compares it with its stub.
- **Licences.**
    `cargo-deny check licenses` holds a dependency graph to a list of allowed SPDX licences.
- **Docs.**
    rustdoc runs documentation examples as tests (`cargo test --doc`); `pytest-markdown-docs` runs a Markdown file's Python fences as tests.

### 3.5 The state of the art in documentation

- **Diátaxis** (Daniele Procida, 2020; Canonical's since 2021) splits documentation into tutorials, how-to guides, reference and explanation, each for one need of the reader; a page that mixes two serves neither.
- **Rust's** documentation is the Book (tutorial and explanation) beside generated, tested API docs with an example on most items.
- **JAX's** adds a page of its pitfalls, "The Sharp Bits", beside its tutorials and reference.
- jolty's guide is reference and explanation, its API reference generated, its examples linted but not run; it has one tutorial and no how-to or pitfalls page.

## 4. Gaps and opportunities

Size: S is hours, M days, L weeks (estimates).
Lead marks an item where jolty would go past every reference rather than catch up.

| # | Item | Impact on the goal | Evidence | Size | Done when |
| --- | --- | --- | --- | --- | --- |
| 1 | Fix the guide's three stale statements | The guide is true | G1–G3 | S | `machines.md`, the guide's index and the options' opening sentence say what the code does |
| 2 | Add VBIC's parameter table, as every other semiconductor has | A designer maps a card to jolty's names in the guide | G4, A5 | S | A reader finds each VBIC card name and its keyword |
| 3 | One stamping path: BSIM3, VDMOS and VBIC onto `Derivatives` and one-voltage junction duals, or the MOSFET back, with their speeds measured | No model does one job its own way | A2 | M | `parts.rs` holds one version of `current` and `charge`; before and after numbers on a BSIM3 and a VBIC fixture |
| 4 | Every kernel on `Dual`: the diode, JFET and Gummel-Poon transistor ported, the Jacobian check then tightened to rounding | Exact by construction everywhere; one way to write a model | A1, A11 | L | No hand-written slope in a kernel; the check's tolerance derived in its comment |
| 5 | One definition per model fact: each parameter's SPICE name and dimension as data in its Rust type (a schema extension), the Python keywords, dimensions and SPICE tables generated from it or tested against it, the guide's tables generated | Adding or renaming a parameter touches one file; the guide cannot drift (Xyce has half of this, neither reference all) | A6, A7, A5, §3.2 | L | A test fails when a Python keyword, dimension, SPICE name or guide row disagrees with the Rust declaration |
| 6 | A model descriptor for what the Python layer branches on by name: SPICE letter, terminal order, node counts, temperature convention, designator letter | The SPICE, KiCad and reader code hold no model name | A10 | M | `git grep '"vbic"'` finds VBIC only in its constructor and tests |
| 7 | The cost of a model, measured: shared files touched per new model, counted on each model's commit | Where ngspice needs 6 and jolty 14 | A10, §3.1 | S, then a target | The count is recorded per model; Lead once it is 1 |
| 8 | Guards that enumerate: the allocation test, the Jacobian check and the SPICE round trip iterate `Library::standard()`; the kernels' lints set once on a parent module that holds `waveform.rs` and `heat.rs` | A new model cannot skip a guard | A12, A13 | S | Removing a model's case fails a test; one `deny` attribute in the crate |
| 9 | Quantities everywhere: `Options`, overrides, `Variant`, `Axis` and parameter sweeps take quantities, checked against the model's dimension (item 5) | "A value in farads cannot go where ohms belong" holds on every path | A7, S1 | M | `Axis("R1", "resistance", [1 * nF])` fails at construction |
| 10 | Resolve before running: every analysis a spec names, every record path, sweep source and axis instance checked against the plan and the circuit before the first analysis, with close names | A typo costs a second, not a run | S2 | M | A misspelt analysis in a spec fails before simulating, naming `bode` |
| 11 | Results carry their facts: `excursions` on each result | Results are pure values; no call-order dependence | S3 | S | `Simulation` has no `excursions` attribute |
| 12 | `Parallelism` changes no bits: `max_unknowns_per_interior` moves to the numerical options; `min_unknowns_inside` renamed; the knobs defined once and derived in runner and service | The policy keeps its promise without exception | A16, A17, A8, G1 | M | The fingerprint has no special case; one definition of each knob in Python |
| 13 | API names and shapes: distinct names for analyses and results; a `PeriodicSteadyState` analysis; corners' arguments apart from their fields; `batch(plan)` | No two public names collide; a flag never changes a field's meaning | S4, S6, S7 | M (breaking) | The reference lists each class once by its full path |
| 14 | Plans validated as the files are, every error at once | One validation mechanism, all problems reported | S8, A18 | M | A plan with two bad options reports both |
| 15 | Every engine constant names its source; the ones users could choose (gmin factor, source stepping) become options | No tuned number left unexplained | A14 | M | Each constant's doc cites a line, a paper, a derivation or a measurement |
| 16 | Every test tolerance names its source | The tests' rule holds everywhere | A15 | M | A review leaves none of the 428 uncited; a lint flags new literal tolerances without a comment |
| 17 | The guide's code runs and its outputs are produced: `pytest-markdown-docs` over the guide and the README, the quoted outputs generated by `docs/illustrate.py` or tested | The guide cannot drift from the API | G5, §3.4 | M | CI runs the 41 Python blocks; each quoted output comes from a run |
| 18 | Mutation testing: `cargo-mutants --in-diff` on every change, a weekly run sharded across machine A's cores; `mutmut` on `units`, `specs` and the SPICE reader | Finds tests that pass whatever the code computes, such as A11's 1e-5 | §3.4 | M to set up; several thousand mutants in `jolty-models` alone, from its 880 functions and about 3,600 binary operators outside tests, so tens of CPU-hours a full run (estimate) | The surviving mutants are counted per crate and each is killed or explained. Lead: no simulator reports a mutation score |
| 19 | Coverage, reported and not gated at first: `cargo-llvm-cov` and coverage.py | Shows the branches no test reaches, such as rarely chosen `capmod`s | §2.1 | S | A coverage report per CI run |
| 20 | Fuzzing of the parsers: the expression language, the circuit format, the SPICE reader, shard files | A crash becomes a message on any input | §2.1 | M | `cargo-fuzz` targets and a Hypothesis netlist generator run nightly |
| 21 | API stability: `cargo-semver-checks` on the published crates, `griffe check` on the package | Breaks are seen, then recorded in `CHANGELOG.md` | §3.4, `release.yml` | S | Both run in CI against the last tag |
| 22 | The extension's stub checked: `stubtest`, and a test of each outcome's keys against its `TypedDict` (or typed `pyclass`es in their place) | The Rust and Python halves cannot drift | A9 | S | A renamed key fails a test |
| 23 | Mechanical policy checks: `cargo-deny` for licences, rustdoc with warnings as errors, the free-threaded tests in `scripts/check`, a smoke test of every release wheel | Every rule of `AGENTS.md` that a tool can check, a tool checks | §2.1, A20 | S | Each is a step of `scripts/check` or `release.yml` |
| 24 | Laws applied to every model by enumeration: npn/pnp and n/p mirrors bit for bit, charge conservation over a cycle, continuity at region boundaries, KCL on the terminals | Catches what a reference shares with jolty: ngspice's VBIC pnp is not its npn's mirror (`devices.md:503`) | A12's pattern, `research/models.md` move 3 | M | Each law is one test over the library. Lead |
| 25 | A guide for contributors: how to add a model, generated in part from item 5's declaration, and a sharp-bits page for users | A newcomer extends jolty without reading a port | §2.4, §3.5 | S | A model added by following the page alone |
| 26 | Shorter sentences in the guide, and how-to pages | Designers who scan find what they need | §2.5 | M | A median under 25 words; how-tos for the common tasks |

## 5. The long view

- **A model is one file.**
    In two to five years, a model's equations, parameters, units, SPICE names, terminals and defaults live in its kernel, or in a Verilog-A source compiled to one (`research/models.md` move 2), and everything else is derived: the Python constructor with typed quantities, the SPICE reader and writer, the KiCad letter, the guide's table, the reference, and the tests every model must pass.
    A new model's diff is its file, its tests and its changelog line; ngspice's is 21 files and 6 shared ones, Xyce's ADMS route 83,000 generated lines for BSIM-CMG alone.
- **Exact by construction, checked by laws.**
    Every kernel's derivatives are automatic, so the Jacobian check becomes a check of the stamping code at rounding level, and every model is held to the laws that hold for all of them (mirrors, conservation, continuity, KCL), not only to ngspice, whose own models fail some (ngspice's VBIC pnp).
- **A compiler's front end.**
    Names and dimensions resolve before anything runs, and much of it before Python does: typed terminals and typed quantities generated from the model declarations, so `ty check` rejects a capacitance passed as a resistance or a misspelt terminal.
    None of the tools surveyed here (PySpice, Xyce's Python interface, OCEAN, gdsfactory) checks a circuit statically, as far as their published interfaces show.
- **A functional core.**
    `simulate(circuit, plan)` is pure and returns results that hold every fact about the run; batches and sensitivities are named as the transformations they are, as JAX names `vmap` and `grad`.
- **Quality as a suite.**
    Beside the speed suite: mutation score per crate, coverage, fuzzing hours, the guide's tests, API breaks per release, the cost of a model, and the review's open findings, each with a baseline, so that a change that lowers one says why, as one that slows a benchmark does.
- **A guide that is tested.**
    Every code block runs, every output is generated, and every behaviour it states links to the test that checks it; the departures from ngspice are generated from the tests that encode them.

## 6. Principles to propose

For the owner to adopt into `AGENTS.md`, or not; each follows from findings above.

1. **One definition per fact.**
   Each fact about a model, such as a parameter's name, SPICE name, dimension, default and description, or a model's terminals and SPICE letter, is declared once, beside its kernel.
   The Python constructors, the SPICE reader and writer, the guide's tables and the reference derive from that declaration or are tested against it.
   This extends "One definition" (`AGENTS.md:102-103`) from commands to models (A6, A7, A8, A10).
1. **One way to do one job.**
   When a better way to do something lands, such as a kernel's stamp, it replaces the old way everywhere in the same change, or the change lists every place left and a test fails until they move.
   The owner's "no asymmetric code" is in the briefs (a brief, not kept), not in `AGENTS.md` (A1, A2, A3).
1. **Every number says where it comes from.**
   Every constant of the engine, the models and the checks names its source: a line of a reference simulator, a paper, a derivation in its documentation, or a measurement with its machine.
   One that users could reasonably choose otherwise is an option, as "knobs over opinions" has it.
   This extends the tests' rule on tolerances to the code (A11, A14, A15).
1. **Guards enumerate.**
   A test of a property every model or analysis must have iterates the library or the analyses, so a new one cannot be left out (A12, A13); `research/models.md`'s third principle, a law suite every kernel passes, is one such guard.
1. **Resolve before running.**
   Every name and dimension that a script, a plan or a spec uses is resolved and checked before the first analysis runs, as a compiler resolves names before it generates code.
   Every user-facing number with a dimension is a quantity (A7, S1, S2).
1. **The docs are tested.**
   Every code block of the guide and the README runs in the tests, and every output they quote comes from a run (G5).
1. **Parallelism never changes the bits.**
   A setting that reorders the arithmetic, such as the dissected factorization's interiors, is a numerical option, not part of `Parallelism`, so that serial and fully parallel are two configurations of one result (A17, G1).

Two questions the research leaves to the owner: whether ports keep their reference's names inside the kernel (A4), and whether "Feedback is facts" covers misuse of the API, as in the message that says how to record a terminal (A19).

## 7. Sources

**jolty** (the repository at `9545e08`, with `d10fc4d` for A21): `AGENTS.md`; `Cargo.toml`, `pyproject.toml`, `clippy.toml`, `.rumdl.toml`, `zensical.toml`; `scripts/check`; `.github/workflows/ci.yml`, `docs.yml`, `release.yml`; `crates/jolty-models/src/` (`kernel.rs`, `kernel/parts.rs`, `dual.rs`, `real.rs`, `bjt.rs`, `jfet.rs`, `junction.rs`, `mosfet.rs`, `bsim3.rs`, `bsim3/channel.rs`, `vdmos.rs`, `vbic.rs`, `waveform.rs`, `heat.rs`, `testing.rs`, `lib.rs`); `crates/jolty-models/tests/kernels.rs`; `crates/jolty-engine/src/` (`options.rs`, `transient.rs`, `operating_point.rs`); `crates/jolty-engine/tests/analytic.rs`; `crates/jolty-python/src/`; `src/jolty/` (`__init__.py`, `builder.py`, `devices.py`, `plan.py`, `simulate.py`, `results.py`, `units.py`, `runner.py`, `service.py`, `kicad.py`, `_core.pyi`, `specs/`, `spice/models.py`, `spice/export.py`, `spice/library.py`); `tests/` (`test_ngspice.py`, `differential.py`, `test_spice.py`, `test_simulate.py`, `test_specs.py`); `docs/guide/*.md`, `docs/get-started.md`, `docs/reference.py`, `docs/reference/*.md`, `README.md`; commits `9545e08`, `2017503`, `a93a9f9`, `4f39ff4`, `52e3bdb`, `6fe8824`, `c60cb78`.

**Notes:** `plans/vision.md`, the session log, not kept, `quality.md`, `research/models.md`, a handoff note, not kept, an archived note, not kept, a brief, not kept.

**ngspice 47** (scratchpad): `src/spicelib/devices/bjt/` (`bjt.c`, `bjtdefs.h`, `bjtsetup.c`, `bjtbindCSC.c`, `bjtload.c`, `bjtacld.c`, `bjtpzld.c`, `bjtmpar.c`, `bjtmask.c`); `src/spicelib/devices/dev.c`, `Makefile.am`; `src/spicelib/parser/inpdomod.c`, `inp2q.c`; `configure.ac`; `src/spicelib/analysis/dctran.c`, `cktop.c`, `traninit.c`, `cktsens.c`; `tests/bsim3/`.

**Xyce** (scratchpad clone): `src/DeviceModelPKG/OpenModels/N_DEV_BJT.C`, `N_DEV_BJT.h`; `src/DeviceModelPKG/Core/N_DEV_LaTexDoc.C`; `doc/Reference_Guide/compareLatexTables.py`; `gitlab-ci/analysisBuilds.yml`; `cmake/CodeCoverage.cmake`; `utils/XyceCInterface/xyce_interface.py.cmake` and `PythonExamples/runACircuit/runACircuit.py`.

**Papers and documentation:**

- Goran Petrović and Marko Ivanković, "State of Mutation Testing at Google", ICSE-SEIP 2018, pp. 163-171, doi:10.1145/3183519.3183521: <https://research.google/pubs/state-of-mutation-testing-at-google/>
- David R. MacIver and Zac Hatfield-Dodds, "Hypothesis: A new approach to property-based testing", Journal of Open Source Software 4(43), 1891, 2019: <https://joss.theoj.org/papers/10.21105/joss.01891>
- cargo-mutants, incremental tests of modified code and sharding: <https://mutants.rs/in-diff.html>, <https://mutants.rs/shards.html>
- mutmut: <https://mutmut.readthedocs.io>
- cargo-semver-checks: <https://github.com/obi1kenobi/cargo-semver-checks>, and its 2025 review: <https://predr.ag/blog/cargo-semver-checks-2025-year-in-review/>
- Griffe, checking for API breakages: <https://mkdocstrings.github.io/griffe/checking/>
- mypy, automatic stub testing (stubtest): <https://mypy.readthedocs.io/en/latest/stubtest.html>
- cargo-deny (licences, bans, advisories): <https://embarkstudios.github.io/cargo-deny/>, example configuration in Cargo's own tree: <https://rust.googlesource.com/cargo/+/refs/heads/rust-1.97.0/deny.toml>
- The rustdoc book, documentation tests: <https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html>
- pytest-markdown-docs: <https://github.com/modal-labs/pytest-markdown-docs>
- Rust API Guidelines checklist (C-EXAMPLE, C-CASE): <https://rust-lang.github.io/api-guidelines/checklist.html>
- Daniele Procida, Diátaxis: <https://diataxis.fr>, and Canonical's adoption: <https://canonical.com/blog/diataxis-a-new-foundation-for-canonical-documentation>
- JAX, "The Sharp Bits": <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>
- Polars, lazy schemas: <https://docs.pola.rs/user-guide/lazy/schemas>
- pydantic, error handling: <https://docs.pydantic.dev/dev/errors/errors>
- scikit-rf, `Network`: <https://scikit-rf-official.readthedocs.io/en/stable/api/generated/skrf.network.Network.html>
- gdsfactory, references and ports: <https://gdsfactory.github.io/gdsfactory/notebooks/01_references.html>
- PySpice examples (third-party; the official reference was not found): <https://techoverflow.net/2022/03/10/how-to-create-resistor-in-pyspice-minimal-example>
- Cadence OCEAN Reference, as published by universities (dated): <https://www.eecis.udel.edu/~vsaxena/courses/ece5411/Handouts/oceanref.pdf>, and users' reports on Monte Carlo results: <https://community.cadence.com/cadence_technology_forums/f/custom-ic-design/15830/selecting-a-particular-iteration-result-for-transient-monte-carlo-sim-in-ocean>
- GOV.UK, clear language (sentences over 25 words): <https://guidance.publishing.service.gov.uk/writing-to-gov-uk-standards/writing-guidelines/clear-language/>
- ngspice on adding devices and OSDI: <https://ngspice.sourceforge.io/osdi.html>
