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
uv sync                     # build the engine, install Python 3.15 and every tool from uv.lock
uv run pytest               # Python tests and doctests; warnings are errors
uv run ruff format          # format Python
uv run ruff check           # lint Python, with every rule enabled
uv run ty check             # types
uv run rumdl check .        # Markdown
uv run python docs/reference.py  # regenerate the reference pages from the code
uv run python docs/illustrate.py # redraw the docs' schematics and plots from the examples
uv run python docs/gallery.py    # rebuild the gallery and the discoveries, replicated in ngspice
cargo fmt --all             # format Rust
cargo clippy --workspace --all-targets -- -D warnings  # lint Rust
cargo test --workspace      # Rust tests
```

A change is done when `scripts/check` passes; CI runs the same script, one group per job, the Rust and Python groups on Linux and on macOS and the docs group on Linux.
Enable the pre-push hook once per clone, `git config core.hooksPath .githooks`, so no push reaches CI without passing it.
`uv sync` rebuilds the engine whenever a Rust source, or the Markdown a crate includes, changes.
PyO3 builds against the interpreter that `PYO3_PYTHON` names, so set it to `.venv/bin/python` before running cargo.

## Architecture

The Rust workspace, in `crates/`, is ordered.
Each crate depends only on the crates listed before it:

1. `jolty-ir`: the circuit format, its names and its validation.
1. `jolty-numerics`: sparsity patterns, sparse LU over real and complex numbers, and integration formulas.
1. `jolty-expressions`: the expression language of behavioral sources, compiled and evaluated with its exact gradient.
1. `jolty-models`: the interface every device implements, and the standard library of devices.
1. `jolty-engine`: elaboration of a hierarchy into devices bound to unknowns, Newton iteration with SPICE's convergence aids, and the analyses.
1. `jolty-checks`: the rules: structural ones, which read the elaborated circuit through the branches each device declares, and those on the circuit linearized at its operating point.
1. `jolty-python`: the Python extension module `jolty._core`, which adapts the engine and nothing more.

The Python package, `src/jolty`, is ordered too.
Each module imports only from the modules listed before it, and from those listed before it on its own line:

1. `errors`, `settings`: the errors front ends show as a message, and configuration.
1. `units`, `waveforms`, `ir`, `unknowns`: dimensioned quantities, the levels of sources over time, the circuit format, and the unknowns analyses solve for.
1. `devices`, `builder`, `parts`: the devices a script places, components written as Python functions, and real parts, each a model with its maker's number and ratings.
1. `spice`: SPICE libraries read into components, model files downloaded and checked, SKY130's transistors and the process design kit that models them, and circuits written as SPICE netlists for ngspice.
1. `plan`, `checks`, `results`, `specs`, `foundry`, `simulate`, `variation`, `catalog`, `fidelity`: analyses and their options, the rules and their diagnostics, results, requirements measured on results, the analyses of circuits of a foundry's devices run in ngspice, the engine's Python face, Monte Carlo samples of tolerances, orderable parts and bills of materials, and logic models characterized from components' circuits.
1. `files`, `kicad`, `report`, `plots`: circuit and results files, components written as KiCad netlists, what each analysis found as data and as text, and plots of results.
1. `layout`, `schematic`, `viewer`: where a schematic places each instance and runs each wire, the schematic drawn, and the 3D viewer that plays a simulation on the layout.
1. `runner`, `service`: scripts in child processes, and every command defined once.
1. `cli`, `mcp`: front ends that only adapt `service`.

`tests/test_architecture.py` reads both lists from this file and fails on any dependency or import that breaks their order, so the lists here are the one definition of the architecture.

## Conventions

- **Tools:** use the standard, official, established tool in each category.
  That means uv, ruff, ty, rumdl, pytest with Hypothesis, pydantic and pydantic-settings, numpy for arrays, the official MCP SDK, matplotlib for plots, maturin with PyO3, cargo with clippy and proptest, and faer for dense linear algebra.
- **Dependencies:** a new one is the established choice in its category, is maintained, and has a license compatible with Apache-2.0.
  Nothing under the GPL is linked into jolty; a GPL tool runs only as a separate program.
- **Format and lint:** ruff's default format at 88 columns, every ruff rule, every ty rule as an error, rustfmt, clippy's pedantic lints as errors, and Markdown one sentence per line.
  A rule is switched off for the whole code base only in `pyproject.toml` or `Cargo.toml`, or for Markdown's code blocks in `.rumdl.toml`, which rumdl alone reads them from, as narrowly as it can be, with its reason beside it.
  A single site is excused only by a suppression that names the rule and gives its reason beside it, such as `# noqa: S603 - reason` in Python or `#[expect(lint, reason = "...")]` in Rust, never a bare `allow`.
- **Python:** 3.14 or newer, so annotations are lazy without `from __future__ import annotations`.
  Use modern syntax: PEP 695 generics and `type` aliases.
- **Rust:** no `unsafe`; the workspace forbids it, and allowing it in a crate is a change to this file.
  No `unwrap`, `expect`, `panic!` or `unreachable!` outside tests and test helpers, except where the code itself has just established what it asserts, excused as above with the invariant as the reason.
  Every public item has documentation.
- **Circuit and plan:** the circuit says what exists; the plan says how to simulate it.
  Nothing about simulation lives in the circuit format, and a plan names instances only to excite them, to record them, to run a model in their place, or to vary their parameters.
- **Circuit format:** a change to it bumps `jolty_ir::VERSION`, and a file of another version is refused with both versions named.
- **Units:** quantities carry their dimension in Python, and every device constructor in `jolty.devices` checks the dimension of each parameter as it reads it.
  The circuit format, the engine and the results hold numbers in SI base units.
- **Models:** a device model whose equations are not elementary names the paper, manual or datasheet they and its defaults come from.
  Every model is checked against that source, a closed form, or ngspice.
- **Determinism:** results are a pure function of the circuit, the plan, the sample index and the options: the same bits for any thread count, shard split or completion order, on the same build and platform.
  Floating-point sums run in a fixed order, random draws depend only on the seed and the sample index, and nothing depends on the order work finishes in.
- **One definition:** each command is a request model plus a function in `jolty.service`.
  The command line and the MCP server derive from it and never redefine arguments.
- **Feedback is facts:** jolty tells the agent what is true about its circuit and never fixes it or says how to.
  A problem names the instance by its path and the script line that placed it.
  The engine reports every problem it finds, not only the first.
  A number in a message is exact, or says how it was estimated, such as to first order.
- **Problems and rules:** a problem stops the engine from elaborating a circuit, such as a negative resistance, and always fails a build.
  A rule states a fact about a circuit that elaborates, such as a node with no DC path to the ground, at a level people configure.
- **Rules, as in ty:** a rule's name is lowercase words joined by hyphens, and its level is `error`, `warn` or `ignore`.
  A rule whose fact surely breaks the circuit defaults to `error`; any other defaults to `warn`.
  Each rule states its exact condition (`jolty explain`), and has a test that raises it and one that passes.
- **Images are feedback:** every build draws its schematic and every simulation plots its analyses, and the MCP server sends the images with its text.
  A batch of runs plots their distribution, not each run.
  A schematic comes from the connections alone: a script holds no coordinates.
  Plots never put two quantities on one axis, and take their colors in a fixed order checked for color-vision deficiencies.
- **Scripts run in their own process** (`jolty.runner`), never in the command line or the MCP server: their output is captured, a timeout stops them, and a crash ends only the runner.
- **Numbers in messages** are written in their shortest exact form, such as `-1e-9`; reports write them with SI prefixes, such as `-1 nV`.
- **Tests:** the engine is checked against closed-form solutions, and by property tests of the laws every circuit obeys, such as Kirchhoff's current law.
  It is checked against ngspice on a corpus of circuits (`tests/test_ngspice.py`), on vendors' macromodels (`tests/test_vendor.py`), and on random circuits of every device and size, which Hypothesis shrinks to the smallest that disagrees (`tests/test_random_circuits.py`).
  These differential tests skip when ngspice is not installed, except in CI, which installs it and sets `JOLTY_REQUIRE_NGSPICE=1` so a skip fails.
  The same random circuits must simulate the same however they are written (`tests/test_metamorphic.py`).
  The layout is checked by property tests over random circuits: nets never touch, and wires never cross a symbol.
  A bug fix comes with a test that fails without it.
  A test's tolerance says where it comes from, and is never loosened to let a change pass.

## Releasing

`CHANGELOG.md` records what each version changes, under `Unreleased` until it ships.
A tag `vX.Y.Z` on `main` releases that version: `.github/workflows/release.yml` checks the tag against the versions in `pyproject.toml` and `Cargo.toml`, builds the wheels for Linux, macOS and Windows and the source distribution, publishes them to PyPI through trusted publishing, and publishes every crate but `jolty-python` to crates.io with the `CARGO_REGISTRY_TOKEN` secret.
Nothing publishes without a tag.
