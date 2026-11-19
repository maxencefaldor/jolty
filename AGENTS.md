# Working on jolty

The repository holds jolty, a compiler and simulator for electronic circuits, built for AI agents.
The README describes what jolty does; the rules below govern changes to the codebase.

## Scope

jolty describes circuits, checks them, simulates them, measures them against specs, and exports them in formats other tools read, such as SPICE netlists for ngspice and KiCad netlists for boards.
Physical design is out of scope: placing parts and routing wires on a board or chip, as well as foundry sign-off checks before fabrication.
jolty exports to dedicated physical-design tools rather than duplicating their work.
The native engine simulates devices from the built-in library.
Foundry transistor models that the native engine cannot load run in ngspice through exported netlists.
Schematic layout exists solely to render visual feedback, not to lay out PCBs.

## Commands

```sh
scripts/check               # every check below, exactly as CI runs them
scripts/bench               # benchmark suites in benchmarks/: speed against a baseline by default, or another suite
scripts/coverage            # test coverage for Rust and Python code in target/coverage, never checked in
scripts/mutants             # Rust mutants missed by tests in mutants.out, never checked in; --all for every crate
uv sync                     # build the engine, install Python 3.15 and every tool from uv.lock
uv run pytest               # Python tests and doctests across every core, or one with -n 0; warnings are errors
uv run ruff format          # format Python
uv run ruff check           # lint Python, with every rule enabled
uv run ty check             # types
uv run rumdl check .        # Markdown
uv run python docs/reference.py  # regenerate the reference pages from the code
uv run python docs/illustrate.py # redraw the docs schematics and plots from the examples
uv run python docs/gallery.py    # rebuild the gallery and the discoveries, replicated in ngspice
cargo fmt --all             # format Rust
cargo clippy --workspace --all-targets -- -D warnings  # lint Rust
cargo nextest run --workspace  # Rust tests, each in a separate process
cargo test --workspace --doc   # Rust doctests, which nextest does not run
```

A change is complete when `scripts/check` passes.
CI runs the same script on every pushed branch, with Rust and Python groups on Linux and macOS and the docs group on Linux, requiring a green run before landing on `main`.
CI runs `scripts/coverage` and `scripts/mutants` on `main`, weekly schedules, and manual runs for reporting only, without failing builds.
Enable the pre-push hook once per clone with `git config core.hooksPath .githooks` so no push reaches CI without passing checks locally.
`uv sync` rebuilds the engine whenever a Rust source file or Markdown file included by a crate changes.
PyO3 builds against the interpreter named by `PYO3_PYTHON`, so set `PYO3_PYTHON` to `.venv/bin/python` before running cargo.

## Architecture

The Rust workspace in `crates/` is ordered.
Each crate depends only on the crates listed before it:

1. `jolty-derive`: derive macros including `Lift`, which converts plain kernel parameter structs field by field into generic numeric types.
1. `jolty-ir`: circuit intermediate representation, naming rules, and schema validation.
1. `jolty-numerics`: sparsity patterns, parallel real and complex sparse LU factorization via nested dissection, deterministic ordered assembly and summation, and integration formulas.
1. `jolty-expressions`: behavioral-source expression language, compiled and evaluated with exact gradients.
1. `jolty-models`: device kernels, serial and parallel slab evaluators, multi-direction parameter differentiation twins, and the standard device library.
1. `jolty-engine`: circuit elaboration, parallel Newton iteration with SPICE convergence aids, frequency and time-domain analyses, specification measures, and deterministic batch execution.
1. `jolty-checks`: structural checks over declared device branches and linearized operating-point diagnostics.
1. `jolty-veriloga`: Verilog-A compiler generating checked-in Rust device kernels that are never hand-edited.
1. `jolty-python`: thin PyO3 extension module (`jolty._core`) adapting the Rust engine for Python.

The Python package in `src/jolty` is ordered too.
Each module imports only from modules listed before it and from earlier modules on the same line:

1. `errors`, `settings`: user-facing error types and configuration loading.
1. `units`, `waveforms`, `ir`, `unknowns`: dimensioned quantities, time-varying source waveforms, the circuit format, and solver unknowns.
1. `devices`, `builder`, `parts`: device constructors, Python component builders, and real part definitions with manufacturer numbers and ratings.
1. `spice`: SPICE library parsing, verified model downloads, SKY130 transistor support, and ngspice netlist export.
1. `plan`, `checks`, `results`, `specs`, `foundry`, `variation`, `simulate`, `jax`, `catalog`, `fidelity`: analysis plans, diagnostic rules, results, specification checks, foundry runs, Monte Carlo sampling, simulation APIs, JAX bridge, parts catalog, and logic characterization.
1. `files`, `kicad`, `report`, `plots`, `shards`: serialization files, KiCad netlists, analysis reports, distribution plots, and digest-verified Monte Carlo shards.
1. `layout`, `schematic`, `viewer`: automatic schematic placement and routing, schematic rendering, and interactive 3D simulation playback.
1. `runner`, `service`: isolated subprocess script execution and shared command definitions.
1. `cli`, `mcp`: command-line and MCP front ends delegating directly to `service`.

`tests/test_architecture.py` reads both lists from `AGENTS.md` and fails on any dependency or import violating their order, keeping these lists the single definition of the architecture.

## Conventions

Where existing code falls short of a convention, any change touching that code brings it closer rather than further away.

- **Tools:** use the standard, official, established tool in each category.
  Python tooling uses uv (including `uv audit` against OSV), ruff, ty, rumdl, pytest (with Hypothesis and pytest-xdist), coverage.py, pydantic, pydantic-settings, numpy, matplotlib, and the official MCP SDK.
  Rust and build tooling uses maturin with PyO3, cargo, clippy, proptest, cargo-nextest, cargo-audit (RustSec), cargo-llvm-cov, cargo-mutants, criterion, and gungraun (via Valgrind).
  Core Rust libraries are faer for dense linear algebra, rayon for data parallelism, mimalloc as the Python extension allocator, and `rand` with `rand_chacha` and `rand_distr` for random sampling.
- **Dependencies:** a new dependency must be the established choice in its category, actively maintained, and compatible with Apache-2.0, as verified by `tests/test_licenses.py` for both Rust crates and Python distributions.
  Before adding a new dependency, workspace crate, or core abstraction, pause and obtain explicit owner approval to keep the codebase minimal, clean, and cohesive.
  No GPL code is linked into jolty; GPL tools run only as separate executables.
  Process kits and third-party fixtures are pinned and fetched by digest rather than copied into the repository.
- **Format and lint:** use ruff default formatting at 88 columns, every ruff rule, every ty rule as an error, rustfmt, clippy pedantic lints as errors, and one-sentence-per-line Markdown.
  A lint rule may be disabled repository-wide only in `pyproject.toml` or `Cargo.toml`, as narrowly as possible, with a rationale next to the entry.
  A single-site suppression must name the rule and state its rationale alongside it, such as `# noqa: S603 - reason` in Python or `#[expect(lint, reason = "...")]` in Rust, never a bare `allow`.
- **Python:** require Python 3.15 or newer so annotations are lazy without `from __future__ import annotations` and modules load lazily via `__lazy_modules__` (PEP 810).
  Use modern syntax including PEP 695 generics and `type` aliases.
- **Rust:** forbid `unsafe` workspace-wide; allowing `unsafe` in any crate requires updating `AGENTS.md`.
  Forbid `unwrap`, `expect`, `panic!`, and `unreachable!` outside tests and test helpers unless the immediately preceding code proves the invariant, documented with `#[expect(..., reason = "...")]`.
  Every public item requires documentation.
- **Circuit and plan:** the circuit defines what exists; the plan specifies how to simulate it.
  Simulation settings stay outside the circuit format, and a plan names instances only to excite them, record them, substitute models, or vary parameters.
- **Circuit format:** any schema change increments `jolty_ir::VERSION`, and loading a mismatched version fails while naming both versions.
- **Units:** quantities carry dimensions in Python, and every device constructor in `jolty.devices` validates each parameter dimension on entry.
  The circuit format, engine, and results store numbers in SI base units.
  Every name and dimension in a script, plan, or spec is validated before the first analysis runs, and every user-facing dimensioned number is a quantity.
- **Models:** a device model with non-elementary equations cites the paper, manual, or datasheet defining its equations and defaults.
  Every model is verified against that source, a closed form, or ngspice.
  A model's authoritative reference defines expected behavior: CMC Verilog-A releases and QA tests for CMC standards, ngspice for SPICE3 models, and the published paper, manual, or datasheet otherwise.
  A model adopts a standard name only after passing that standard's QA tests.
  Verilog-A models compile into Rust kernels and are never loaded as foreign binaries.
  A model derived from external source carries that source's copyright notice and terms in both its documentation and `NOTICE`.
  A reactive model defines a charge function `q(x)` or explicitly documents that it does not conserve charge.
- **ngspice is a reference, not the truth:** jolty agrees with ngspice where ngspice is accurate and improves on it where ngspice is not, including exact derivatives, conserved charge, and error control on every charge.
  Each departure is documented in the guide with evidence (an ngspice source line or measurement), and differential tests compare every quantity except the documented departure.
- **Knobs over opinions:** any choice users could reasonably prefer differently, including ngspice compatibility behavior, becomes an option with a documented default when expressible cleanly.
  The default selects the most accurate option.
  An option sharing a SPICE name matches the SPICE quantity, or the guide documents any difference.
- **Parallelism is a policy:** work distribution is controlled through the `Parallelism` option with defaults on every field, covering both serial and fully parallel execution.
  Each default reflects a measured break-even documented in the guide with the measurement machine.
- **Determinism:** results are a pure function of the circuit, plan, sample index, options, and hardware.
  Repeating a run on the same build, machine, and thread count produces bit-identical results regardless of task completion order.
  Across thread counts, shards, machines, and devices, results agree within floating-point rounding rather than necessarily bit for bit.
  Floating-point sums execute in a fixed order, random draws depend only on the seed and sample index, and no output depends on completion order.
  Reusing state from an earlier run (a warm process, cache, or session) produces the same bits and report as a fresh run.
- **Kernels:** a device model is a pure function of one instance's prepared parameters, memory, and local unknowns.
  A device kernel performs no heap allocation, never panics, and has no dependency on threads, batches, or the global matrix.
  Clippy denies `panic` and `indexing_slicing` in kernel modules, and tests verify zero kernel allocations using `allocation-counter`.
- **Exact before approximate:** speed comes first from exact methods.
  Approximations such as device bypass or multirate integration are opt-in options, never defaults, and report measured error.
- **Sound numerics:** a time step is accepted only after Newton iteration converges, and iteration limits derive from observed convergence rather than arbitrary counts.
  Solver failures use precise mathematical names: fold, impasse point, singular structure, or divergence.
- **Constants:** every numeric constant in the engine, models, and checks cites its origin (a reference line, paper, derivation, or machine measurement).
  Any constant users could reasonably vary becomes an option, and choices governed by a computable cost are computed directly.
- **Claims are measured:** every claim about speed, accuracy, reliability, coverage, or agent effectiveness names the suite, fixture, machine, versions, and results file.
  Accuracy is claimed only against an independent reference such as a closed form, manufactured solution, or higher-precision calculation, never against another simulator.
  Runtime comparisons against another simulator require both tools to reach the same solution, reporting accuracy alongside runtime and losses alongside wins.
- **Performance:** `scripts/bench` runs each benchmark suite in `benchmarks/` according to the chosen mode.
  By default, `scripts/bench` runs the speed suite in `benchmarks/speed/` across Rust and Python and compares timings against a baseline.
  Any pull request slowing a speed benchmark by more than 5% explains the regression.
  Benchmarks stay out of `scripts/check` so `scripts/check` remains fast and deterministic.
- **One definition:** each command consists of a request model and a function in `jolty.service`.
  Both the command line and the MCP server derive from `jolty.service` and never redefine arguments.
  Every model fact (parameter name, SPICE name, dimension, default, terminals, and SPICE letter) is declared once next to its kernel, and all other layers derive from or test against that declaration.
- **One way to do one job:** when a superior pattern lands, the change either migrates every site immediately or lists remaining sites under a failing guard test until migrated.
- **Names:** every concept has one name across the engine, Python API, CLI, MCP server, file formats, benchmarks, and documentation, and every name refers to one concept.
  Names are uniform, consistent, and symmetric: the same word denotes the same concept in every layer, and matched concepts share consistent shapes.
  Inverse operations and opposite bounds form symmetric pairs, such as `save_results` and `load_results` or `min_` and `max_`.
  Names are formal, strict, and domain-based: each name describes what an entity represents in established circuit terminology rather than how it is computed, stored, or used.
  Any name violating these rules is renamed everywhere in the change that discovers it.
- **Feedback is facts:** jolty reports what is true about the circuit and invocation without attempting automatic fixes or giving prescriptive advice.
  When a call cannot succeed, jolty states the factual state (such as which devices were recorded) while the guide documents valid usage.
  Every diagnostic names the instance path and the script line that placed the instance.
  The engine reports every problem found rather than stopping at the first.
  Every number in a message is exact or explicitly states its approximation order, such as first order.
  Any agent-requested computation, such as a requirement derivative or parameter optimum, is reported as a fact rather than an autonomous decision.
- **Problems and rules:** a problem prevents the engine from elaborating a circuit (such as a negative resistance) and always fails a build.
  A rule reports a configurable diagnostic about a circuit that elaborates (such as a node without a DC path to ground).
- **Rules, as in ty:** rule names are lowercase hyphenated words with configurable levels `error`, `warn`, or `ignore`.
  Rules whose condition necessarily breaks the circuit default to `error`; all other rules default to `warn`.
  Each rule documents its exact condition (`jolty explain`) and has tests for both triggering and passing cases.
- **Images are feedback:** when the optional `plots` extra (`jolty[plots]`, included in `jolty[mcp]`) is installed, builds render schematics, simulations plot analyses, and the MCP server returns images with text.
  When `matplotlib` is absent, builds, simulations, and shard merges skip image generation cleanly while explicit `draw` and `plot` commands report the missing `jolty[plots]` extra.
  Batch runs plot statistical distributions rather than individual runs.
  Schematics derive purely from connectivity; scripts contain no layout coordinates.
  Plots place one quantity per axis and use a fixed color order checked for color-vision deficiencies.
- **For agents:** agent documentation loads offline from jolty itself, and MCP instructions derive directly from the code.
  Every response enforces a default size budget including images; complete data goes to files or returns on explicit request.
  Every command accepts circuits and plans as structured data rather than requiring executable code.
- **Scripts run in a separate process** (`jolty.runner`), never in the CLI or MCP server: output is captured, timeouts halt hung runs, and crashes terminate only the runner.
- **Numbers in messages** use their shortest exact form (such as `-1e-9`), while reports format values with SI prefixes (such as `-1 nV`).
- **Tests:** the engine is tested against closed-form solutions and property tests of universal circuit laws such as Kirchhoff's current law.
  Differential suites test the engine against ngspice on a reference corpus (`tests/test_ngspice.py`), vendor macromodels (`tests/test_vendor.py`), and Hypothesis-shrunk random circuits across every device and size (`tests/test_random_circuits.py`).
  Differential tests skip locally when ngspice is unavailable; CI installs ngspice and sets `JOLTY_REQUIRE_NGSPICE=1` so any skip fails the build.
  Random circuits must simulate identically regardless of how the circuit is constructed (`tests/test_metamorphic.py`).
  Circuits must simulate identically across runs and within rounding regardless of thread or shard count (`tests/test_determinism.py`, `tests/test_shards.py`, `crates/jolty-engine/tests/determinism.rs`).
  Schematic layout is verified by property tests over random circuits ensuring nets never touch and wires never cross symbols.
  Every universal model or analysis property runs across the entire library so no new model can be omitted, covering charge conservation, polarity mirrors, symmetry, continuity, and finiteness.
  Every error estimate is tested across all active regimes, including each integration order and step ratio.
  Every code block in the guide and README executes in the test suite, and every quoted output matches an actual run.
  Every bug fix includes a test that fails without the fix.
  Every test tolerance documents its origin and is never loosened to make a change pass.

## Releasing

`CHANGELOG.md` records changes for each version under `Unreleased` until release.
Pushing a `vX.Y.Z` tag on `main` triggers `.github/workflows/release.yml`, which validates the tag against versions in `pyproject.toml` and `Cargo.toml`.
The release workflow builds Linux, macOS, and Windows wheels plus the source distribution, publishes them to PyPI via trusted publishing, and publishes every crate except `jolty-python` to crates.io using `CARGO_REGISTRY_TOKEN`.
Nothing publishes without a tag.
