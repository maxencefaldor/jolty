# Working on jolty

The repository holds jolty, a compiler and simulator for electronic circuits, built for AI agents.
The README says what jolty does; the rules below say how to change it.

## Scope

jolty describes circuits, checks them, simulates them, measures them against specs, and writes them in the formats other tools read, such as SPICE netlists for ngspice and KiCad netlists for boards.
Physical design is out of scope: placing parts and routing wires on a board or a chip, and a foundry's checks before a chip is made.
jolty exports to the tools that do these rather than doing them itself.
The engine simulates the devices of its own library; a foundry's transistor models, which it cannot load yet, run in ngspice on an exported netlist.
A schematic's layout only draws a circuit as feedback; it lays out no board.

## Commands

```sh
scripts/check               # every check below, exactly as CI runs them
scripts/bench               # the benchmarks of benchmarks/: the speed suite against a baseline's times, or another suite
scripts/coverage            # the tests' coverage of the Rust and the Python code, in target/coverage, never checked
scripts/mutants             # the mutants of a change that the Rust tests miss, in mutants.out, never checked; --all for every crate's
uv sync                     # build the engine, install Python 3.15 and every tool from uv.lock
uv run pytest               # Python tests and doctests over every processor, or one with -n 0; warnings are errors
uv run ruff format          # format Python
uv run ruff check           # lint Python, with every rule enabled
uv run ty check             # types
uv run rumdl check .        # Markdown
uv run python docs/reference.py  # regenerate the reference pages from the code
uv run python docs/illustrate.py # redraw the docs' schematics and plots from the examples
uv run python docs/gallery.py    # rebuild the gallery and the discoveries, replicated in ngspice
cargo fmt --all             # format Rust
cargo clippy --workspace --all-targets -- -D warnings  # lint Rust
cargo nextest run --workspace  # Rust tests, each in a process of its own
cargo test --workspace --doc   # Rust doctests, which nextest does not run
```

A change is done when `scripts/check` passes; CI runs the same script, one group per job, the Rust and Python groups on Linux and on macOS and the docs group on Linux, on every branch pushed, so a branch lands on main only once its run is green.
CI also runs `scripts/coverage`, and `scripts/mutants` on each change, which report what they measure and never fail on it.
Enable the pre-push hook once per clone, `git config core.hooksPath .githooks`, so no push reaches CI without passing it.
`uv sync` rebuilds the engine whenever a Rust source, or the Markdown a crate includes, changes.
PyO3 builds against the interpreter that `PYO3_PYTHON` names, so set it to `.venv/bin/python` before running cargo.

## Architecture

The Rust workspace, in `crates/`, is ordered.
Each crate depends only on the crates listed before it:

1. `jolty-ir`: the circuit format, its names and its validation.
1. `jolty-numerics`: sparsity patterns, sparse LU over real and complex numbers, which factors a large block by the nested dissection of it on any number of threads, the gather that assembles a system's values in a fixed order, sums of many terms in one fixed order on any number of threads, and integration formulas.
1. `jolty-expressions`: the expression language of behavioral sources, compiled and evaluated with its exact gradient.
1. `jolty-models`: the kernels that compute each device model one instance at a time, the groups that evaluate a kernel's instances together, serially or across threads, into a slab, and the twins that differentiate one instance with respect to one of its parameters, and the standard library of devices.
1. `jolty-engine`: elaboration of a hierarchy, once per structure, into a circuit that every run of it shares, each run preparing its devices' instances again for the numbers it replaces; Newton iteration with SPICE's convergence aids, which for a large circuit spreads its work on each unknown across threads; the analyses, which count their work and, when asked, time it, and spread their frequencies across threads; the measures specs read of their results; and the batch executor, which runs many runs of one circuit across threads, each a pure function of its index.
1. `jolty-checks`: the rules: structural ones, which read the elaborated circuit through the branches each device declares, and those on the circuit linearized at its operating point.
1. `jolty-veriloga`: the compiler of Verilog-A into kernels, which writes the same plain Rust a person writes, and the kernels it compiled, checked in and never edited.
1. `jolty-python`: the Python extension module `jolty._core`, which adapts the engine and nothing more.

The Python package, `src/jolty`, is ordered too.
Each module imports only from the modules listed before it, and from those listed before it on its own line:

1. `errors`, `settings`: the errors front ends show as a message, and configuration.
1. `units`, `waveforms`, `ir`, `unknowns`: dimensioned quantities, the levels of sources over time, the circuit format, and the unknowns analyses solve for.
1. `devices`, `builder`, `parts`: the devices a script places, components written as Python functions, and real parts, each a model with its maker's number and ratings.
1. `spice`: SPICE libraries read into components, model files downloaded and checked, SKY130's transistors and the process design kit that models them, and circuits written as SPICE netlists for ngspice.
1. `plan`, `checks`, `results`, `specs`, `foundry`, `simulate`, `variation`, `catalog`, `fidelity`: analyses, their options and the variants of many runs, the rules and their diagnostics, results and batches' columns, requirements measured on results, the analyses of circuits of a foundry's devices run in ngspice, the engine's Python face, its batches of runs and circuits elaborated once for many simulations, Monte Carlo samples of tolerances, orderable parts and bills of materials, and logic models characterized from components' circuits.
1. `files`, `kicad`, `report`, `plots`, `shards`: circuit, results, batch and shard files, components written as KiCad netlists, what each analysis found as data and as text, plots of results and of batches' distributions, and a Monte Carlo analysis's shards, ranges of its samples that any process runs, fingerprinted and merged in the order of their samples.
1. `layout`, `schematic`, `viewer`: where a schematic places each instance and runs each wire, the schematic drawn, and the 3D viewer that plays a simulation on the layout.
1. `runner`, `service`: scripts in child processes, and every command defined once.
1. `cli`, `mcp`: front ends that only adapt `service`.

`tests/test_architecture.py` reads both lists from this file and fails on any dependency or import that breaks their order, so the lists here are the one definition of the architecture.

## Conventions

Where the code does not meet a convention yet, a change that touches that code brings it closer, never further.

- **Tools:** use the standard, official, established tool in each category.
  That means uv, ruff, ty, rumdl, pytest with Hypothesis and pytest-xdist, pydantic and pydantic-settings, numpy for arrays, the official MCP SDK, matplotlib for plots, maturin with PyO3, cargo with clippy and proptest, cargo-nextest to run the Rust tests, cargo-audit for RustSec's advisories against the crates, as uv's own audit checks the Python packages against OSV's, cargo-llvm-cov and coverage.py for the tests' coverage, cargo-mutants for the Rust tests' strength, criterion for Rust's benchmarks' times and gungraun, through Valgrind, for their instruction counts, faer for dense linear algebra, rayon for data parallelism, mimalloc as the Python extension's allocator, and `rand` with `rand_chacha` and `rand_distr` for random sampling.
- **Dependencies:** a new one is the established choice in its category, is maintained, and has a license compatible with Apache-2.0, as `tests/test_licenses.py` lists them for the Rust crates and the Python distributions alike.
  Nothing under the GPL is linked into jolty; a GPL tool runs only as a separate program.
  A process kit or a third-party fixture is pinned and fetched with its digest, never copied into the repository.
- **Format and lint:** ruff's default format at 88 columns, every ruff rule, every ty rule as an error, rustfmt, clippy's pedantic lints as errors, and Markdown one sentence per line.
  A rule is switched off for the whole code base only in `pyproject.toml` or `Cargo.toml`, as narrowly as it can be, with its reason beside it.
  A single site is excused only by a suppression that names the rule and gives its reason beside it, such as `# noqa: S603 - reason` in Python or `#[expect(lint, reason = "...")]` in Rust, never a bare `allow`.
- **Python:** 3.15 or newer, so annotations are lazy without `from __future__ import annotations` and modules load lazily through `__lazy_modules__` (PEP 810).
  Use modern syntax: PEP 695 generics and `type` aliases.
- **Rust:** no `unsafe`; the workspace forbids it, and allowing it in a crate is a change to this file.
  No `unwrap`, `expect`, `panic!` or `unreachable!` outside tests and test helpers, except where the code itself has just established what it asserts, excused as above with the invariant as the reason.
  Every public item has documentation.
- **Circuit and plan:** the circuit says what exists; the plan says how to simulate it.
  Nothing about simulation lives in the circuit format, and a plan names instances only to excite them, to record them, to run a model in their place, or to vary their parameters.
- **Circuit format:** a change to it bumps `jolty_ir::VERSION`, and a file of another version is refused with both versions named.
- **Units:** quantities carry their dimension in Python, and every device constructor in `jolty.devices` checks the dimension of each parameter as it reads it.
  The circuit format, the engine and the results hold numbers in SI base units.
  Every name and dimension a script, a plan or a spec uses is checked before the first analysis runs, and every user-facing number with a dimension is a quantity.
- **Models:** a device model whose equations are not elementary names the paper, manual or datasheet they and its defaults come from.
  Every model is checked against that source, a closed form, or ngspice.
  A model's reference is what defines it: for a CMC standard its Verilog-A release and QA tests, for SPICE3's models ngspice, otherwise its paper, manual or datasheet; a model takes a standard's name only once it passes that standard's QA tests.
  Verilog-A is compiled into kernels, never loaded as foreign code.
  A model derived from another's source carries that source's copyright notice and terms, in its documentation and in `NOTICE`.
  A reactive model is a charge function `q(x)`, or its documentation says that it does not conserve charge.
- **ngspice is a reference, not the truth:** jolty agrees with ngspice where ngspice is right, and goes beyond it where it is not, such as with exact derivatives, conserved charge and error control on every charge.
  Each departure is stated in the guide with its evidence, a line of ngspice's source or a measurement, and the differential tests compare everything but the departure, saying why.
- **Knobs over opinions:** a choice that users could reasonably make otherwise, such as ngspice's own behaviour where it differs, is an option with a documented default, when it can be expressed cleanly.
  The default is the most accurate choice.
  An option named after SPICE's means SPICE's quantity, or the guide states where it differs.
- **Parallelism is a policy:** how a run spreads its work is an option, `Parallelism`, whose every field has a default; serial and fully parallel are two configurations of it.
  Each default is a measured break-even, stated in the guide with the machine it was measured on.
- **Determinism:** results are a pure function of the circuit, the plan, the sample index, the options and the hardware: the same run on the same build, machine and number of threads gives the same bits, whatever order its work finishes in.
  Across numbers of threads, shards, machines and devices, results agree to rounding, not always to the bit.
  Floating-point sums run in a fixed order, random draws depend only on the seed and the sample index, and nothing depends on the order work finishes in.
  Whatever a command reuses from an earlier one, a warm process, a cache or a session's state, gives a fresh run's bits and report.
- **Kernels:** a device model is a pure function of one instance's prepared parameters, its memory and its local unknowns.
  It allocates nothing, never panics, and knows nothing of threads, batches or the global matrix.
  Clippy's `panic` and `indexing_slicing` lints are denied in the kernels' modules, and a test counts a kernel's allocations with `allocation-counter`.
- **Exact before approximate:** speed comes first from exact means; an approximation, such as device bypass or multirate integration, is a knob a run opts into, never a default, and its results state their measured error.
- **Sound numerics:** a step is accepted only once its Newton iteration has converged, and iteration limits are to follow from the iteration's observed convergence rather than a count.
  A failure is named as mathematics: a fold, an impasse point, a singular structure, a divergence.
- **Constants:** each constant of the engine, the models and the checks names where it comes from: a reference's line, a paper, a derivation or a measurement with its machine.
  One that users could reasonably choose otherwise is an option, and a choice that a cost jolty can compute decides is decided by computing it.
- **Claims are measured:** a claim of speed, accuracy, reliability, coverage or help to agents names its measurement: the suite, the fixture, the machine, the versions and the results file.
  Accuracy is claimed only against an independent reference, a closed form, a manufactured solution or a computation by another method in higher precision, never against another simulator; a time is compared with another simulator's only where both found the same solution, with the error each reached; losses are reported with wins.
- **Performance:** `scripts/bench` runs the benchmarks, each suite of `benchmarks/` by its mode: by default the speed suite, the fixtures of `benchmarks/speed/` timed in Rust and in Python, whose times it compares with a baseline's.
  A change that slows any of them by more than 5% says why in its pull request.
  The benchmarks stay out of `scripts/check`, which must stay fast and deterministic.
- **One definition:** each command is a request model plus a function in `jolty.service`.
  The command line and the MCP server derive from it and never redefine arguments.
  Each fact about a model, such as a parameter's name, SPICE name, dimension and default, or its terminals and SPICE letter, is declared once beside its kernel, and everything else derives from it or is tested against it.
- **One way to do one job:** when a better way lands, it replaces the old everywhere in the change that brings it, or that change lists every place it leaves and a test fails until they move.
- **Names:** each concept has one name everywhere, in the engine, the Python API, the command line, the MCP server, the files, the benchmarks and the documentation, and each name one concept.
  Names are uniform, consistent and symmetric: the same word for the same thing in every layer, the same shape for the same kind of name, and an operation and its inverse, or a bound and its opposite, named as a pair, as `save_results` and `load_results`, `min_` and `max_`.
  They are formal, strict and principled: a name says what the thing is, in the domain's established terms, never how it happens to be computed, stored or used.
  A name that breaks this is renamed everywhere in the change that finds it, as one way to do one job requires.
- **Feedback is facts:** jolty tells the agent what is true about its circuit and its own use, and never fixes either or says how to: a call jolty cannot take states what is true, such as the devices a result recorded, and the guide says how to call it.
  A problem names the instance by its path and the script line that placed it.
  The engine reports every problem it finds, not only the first.
  A number in a message is exact, or says how it was estimated, such as to first order.
  A computation the agent asks for, such as a requirement's derivative or an optimum over parameters it names, is a fact jolty states, never a choice it makes.
- **Problems and rules:** a problem stops the engine from elaborating a circuit, such as a negative resistance, and always fails a build.
  A rule states a fact about a circuit that elaborates, such as a node with no DC path to the ground, at a level people configure.
- **Rules, as in ty:** a rule's name is lowercase words joined by hyphens, and its level is `error`, `warn` or `ignore`.
  A rule whose fact surely breaks the circuit defaults to `error`; any other defaults to `warn`.
  Each rule states its exact condition (`jolty explain`), and has a test that raises it and one that passes.
- **Images are feedback:** every build draws its schematic and every simulation plots its analyses, and the MCP server sends the images with its text.
  A batch of runs plots their distribution, not each run.
  A schematic comes from the connections alone: a script holds no coordinates.
  Plots never put two quantities on one axis, and take their colors in a fixed order checked for color-vision deficiencies.
- **For agents:** what an agent needs to use jolty reads offline from jolty itself, and the MCP instructions derive from the code.
  Every response has a budget by default, images included; the complete data go to a file or come on request.
  Every command can take its circuit and plan as data, not code.
- **Scripts run in their own process** (`jolty.runner`), never in the command line or the MCP server: their output is captured, a timeout stops them, and a crash ends only the runner.
- **Numbers in messages** are written in their shortest exact form, such as `-1e-9`; reports write them with SI prefixes, such as `-1 nV`.
- **Tests:** the engine is checked against closed-form solutions, and by property tests of the laws every circuit obeys, such as Kirchhoff's current law.
  It is checked against ngspice on a corpus of circuits (`tests/test_ngspice.py`), on vendors' macromodels (`tests/test_vendor.py`), and on random circuits of every device and size, which Hypothesis shrinks to the smallest that disagrees (`tests/test_random_circuits.py`).
  These differential tests skip when ngspice is not installed, except in CI, which installs it and sets `JOLTY_REQUIRE_NGSPICE=1` so a skip fails.
  The same random circuits must simulate the same however they are written (`tests/test_metamorphic.py`).
  The same circuits must simulate the same run after run, and to rounding however many threads or shards run them (`tests/test_determinism.py`, `tests/test_shards.py`, `crates/jolty-engine/tests/determinism.rs`).
  The layout is checked by property tests over random circuits: nets never touch, and wires never cross a symbol.
  A property every model or analysis must have is tested over the whole library, so a new one cannot be left out: a law suite of charge, polarity mirrors, symmetry, continuity and finite values among them.
  Every error estimate is tested in each condition it is used in, such as each order and step ratio.
  Every code block of the guide and the README runs in the tests, and every output they quote comes from a run.
  A bug fix comes with a test that fails without it.
  A test's tolerance says where it comes from, and is never loosened to let a change pass.

## Releasing

`CHANGELOG.md` records what each version changes, under `Unreleased` until it ships.
A tag `vX.Y.Z` on `main` releases that version: `.github/workflows/release.yml` checks the tag against the versions in `pyproject.toml` and `Cargo.toml`, builds the wheels for Linux, macOS and Windows and the source distribution, publishes them to PyPI through trusted publishing, and publishes every crate but `jolty-python` to crates.io with the `CARGO_REGISTRY_TOKEN` secret.
Nothing publishes without a tag.
