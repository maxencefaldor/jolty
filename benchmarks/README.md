# Benchmarks

Each suite runs a fixed set of fixtures, each a circuit with its analyses and what decides its score, and scores them one way.
`scripts/bench` is every suite's one command line; the suites stay out of `scripts/check`, which must stay fast and deterministic.

```sh
scripts/bench                  # the speed suite: the fixtures timed in Rust and in Python, against a baseline's times
scripts/bench --view matched   # every simulator's times at matched options; defaults and equal-error, the other two views
scripts/bench --precision      # the precision suite: each simulator's error against exact answers, against its cost
scripts/bench --robustness     # the corpus of hard circuits: each simulator's outcome on each, and its failure rate
scripts/bench --coverage       # the coverage matrix: what each simulator offers, each cell a fixture that ran
```

| Suite | What it measures | Its fixtures and results |
| --- | --- | --- |
| [Speed](speed/README.md) | The engine's time on circuits from a bipolar amplifier to a mesh of 90,000 nodes, in Rust and in Python, and every simulator's in three views: at matched options, at its own defaults, and at equal error | `speed/circuits`, `speed/plans`, `speed/results` |
| [Precision](precision/README.md) | The correct digits of each simulator's runs against exact answers, and what they cost | `precision/fixtures.py`, `precision/results`, outside the repository |
| [Robustness](robustness/README.md) | Each simulator's outcome on hard circuits, jolty's own and third-party decks, and its failure rate | `robustness/own.py`, `robustness/decks.py`, `robustness/results`, outside the repository |
| [Coverage](coverage/README.md) | What each simulator offers, each row of analyses, models, syntax and outputs a fixture that ran | `coverage/capabilities.py`, `coverage/coverage.json`, `coverage/matrix.md` |
| [Agents](agents/README.md) | What an agent achieves on tasks of circuit design with jolty's tools, against ngspice's | `agents/`, run by its own scripts there |
| Deterministic | Solver counts, precision digits and result digests per platform, checked in `scripts/check` | `deterministic/bench.py`, `deterministic/baseline.json` |

Every suite runs a simulator by its runner of [`simulators.py`](simulators.py): jolty, ngspice with Sparse 1.3 and with KLU, and Xyce, serially or its parallel build on MPI ranks.
Every suite writes its results through [`measuring.py`](measuring.py), headed by their provenance, the commit with its `-dirty` mark and the machine, to `<suite>/results/<commit>.json`.
