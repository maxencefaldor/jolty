# Benchmarks

Each suite runs a fixed set of circuit fixtures, executes their analyses, and scores the results uniformly.
`scripts/bench` provides the unified command-line entry point for every suite.
Benchmarks remain outside `scripts/check`, which stays fast and deterministic.

```sh
scripts/bench                  # speed suite: time fixtures in Rust and Python against a baseline
scripts/bench --view matched   # simulator times at matched options (also supports defaults and equal-error)
scripts/bench --precision      # precision suite: simulator error against exact answers versus runtime
scripts/bench --robustness     # robustness corpus: hard-circuit outcomes and failure rates per simulator
scripts/bench --coverage       # coverage matrix: executed feature fixtures per simulator
```

| Suite | What it measures | Fixtures and results |
| --- | --- | --- |
| [Speed](speed/README.md) | Engine runtime from a bipolar amplifier to a 90,000-node mesh in Rust and Python, plus cross-simulator comparisons at matched options, simulator defaults, and equal error | `speed/circuits`, `speed/plans`, `speed/results` |
| [Precision](precision/README.md) | Significant correct digits against exact answers versus engine runtime | `precision/fixtures.py`, `precision/results` (outside the repository) |
| [Robustness](robustness/README.md) | Simulator outcomes and failure rates on hard built-in circuits and third-party decks | `robustness/own.py`, `robustness/decks.py`, `robustness/results` (outside the repository) |
| [Coverage](coverage/README.md) | Executed capability matrix covering analyses, device models, syntax, and outputs | `coverage/capabilities.py`, `coverage/coverage.json`, `coverage/matrix.md` |
| [Agents](agents/README.md) | Autonomous circuit-design task performance comparing jolty tools against ngspice | `agents/`, executed via dedicated scripts in that directory |
| Deterministic | Solver counts, precision digits, and result digests per platform, checked in `scripts/check` | `deterministic/bench.py`, `deterministic/baseline.json` |

Every suite invokes simulators through [`simulators.py`](simulators.py), covering jolty, ngspice (Sparse 1.3 and KLU), and Xyce (serial and MPI parallel builds).
Every suite writes results through [`measuring.py`](measuring.py) to `<suite>/results/<commit>.json`, recording provenance metadata including the commit hash, any `-dirty` flag, and host machine details.
