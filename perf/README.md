# Performance

The engine's speed is measured on a fixed set of circuits, the fixtures, so that every change can say what it does to it.
`scripts/bench` times them, in Rust and end to end in Python, writes the times by commit, and compares them with a baseline's.
The benchmarks stay out of `scripts/check`, which must stay fast and deterministic.

## Running

```sh
scripts/bench                      # the suite: a few minutes, before and after a change
scripts/bench bjt_amp buck         # only these fixtures
scripts/bench --scale              # the large fixtures, each run once
scripts/bench --ngspice            # ngspice on the fixtures' netlists, for context
scripts/bench --baseline COMMIT    # against COMMIT's times, rather than an ancestor's
```

`scripts/bench` first checks that each fixture's files are what its script writes, then times the fixtures, writes the times to `results/<commit>.json`, and prints them against a baseline's, flagging every change over 5%.
The baseline is the newest ancestor of the checked-out commit whose times `results/` holds, unless `--baseline` names a commit or a results file.
A working tree with changes to tracked files writes `results/<commit>-dirty.json`, which is never taken as a baseline.
A later run on the same commit adds its times to the file, so the suite, `--scale` and `--ngspice` can each run once.
To compare two commits, check out the older one, run the suite, then do the same on the newer one.

## Fixtures

Each fixture is a circuit, in the circuit format in `circuits/`, and a plan of analyses in the engine's format in `plans/`, both written by the script beside the circuit, which builds it with jolty's own builder.
Run a script from the repository root after changing it, such as `uv run python perf/circuits/bjt_amp.py`, and commit what it writes.

| Fixture | Circuit and plan | What it stresses | Timed by |
| --- | --- | --- | --- |
| `bjt_amp` | A differential pair of 2N3904s, a mirror of 2N3906s and a follower, fed back for a gain of 10: the operating point, 71 frequencies from 10 Hz to 100 MHz, and 200 µs of a 10 kHz sine | The cost of one small run | the suite |
| `ac_wide` | `bjt_amp` at 10,000 frequencies from 1 Hz to 1 GHz | Frequencies run in parallel | the suite |
| `buck` | A synchronous buck converter of level 1 MOSFETs at 500 kHz, from 12 V into 5 Ω: 2 ms | Many tiny time steps | the suite |
| `inverter_chain_500` | 500 CMOS inverters of level 1 MOSFETs under a pulse: 40 ns | Device evaluation and Newton iteration | the suite |
| `inverter_chain_5000`, `_50000` | The same chain of 5,000 and 50,000 inverters | The same, on one large circuit | `--scale` |
| `rc_mesh_100` | A square mesh of 10,000 nodes, 1 kΩ between neighbours and 1 pF to ground, under a pulse: 10 µs | The sparse solver's fill | the suite |
| `rc_mesh_300` | The same mesh of 90,000 nodes | The same, on one large circuit | `--scale` |
| `monte_carlo` | `bjt_amp`'s analyses on 100,000 samples of its resistors, each within 5% | Many runs of one circuit | the suite, its first 1,000 samples; `--scale`, every one |

The MOSFETs of a level 1 model each add three unknowns of their own, the displacements of their Meyer capacitances, so the chains of 500, 5,000 and 50,000 inverters solve for 3,504, 35,004 and 350,004 unknowns.
The chains' plans let their results hold 4 GiB, as the 50,000 inverters' hold 2.1 at the transient's 800 points or so, more than the 2 the engine allows by default.

## What the times measure

| Time | What it measures |
| --- | --- |
| `rust/<fixture>/setup` | Criterion's median of a setup: the circuit read from its JSON text and elaborated, into a simulator ready to run. |
| `rust/<fixture>/run` | Criterion's median of a run: the setup, then every analysis of the plan in order. |
| `python/<fixture>` | The median of at least ten runs over at least five seconds, after one to warm up: the circuit, read once from its file, elaborated by `Simulation`, and every analysis run by `Simulation.run`, whose results are made as a script's are. An operating point and a transient also record the instances rated for a current or a power, which the checks read. |
| `python/monte_carlo/<n> samples, one core`, `..., every core` | The time per sample of the fixture's first `n` samples, drawn from its seed as the runner draws them, each elaborated with its parameters and run as above: on one thread, and on a thread per core, as the runner runs them, the engine releasing the interpreter while it simulates. The suite runs the first 1,000, and `--scale` all of them, on every core. |
| `ngspice/<fixture>` | ngspice 47 in batch mode on the fixture's netlist as jolty writes it, its analyses run from a control section: from its start to its exit, so its start-up and the parsing of its netlist count, for context. |

Criterion takes ten samples of each benchmark over five seconds, each sample one run or more, after a second of warming up.
The Python timings follow the same protocol.
Beside the times, the suite runs each fixture with a transient once more with the option `profile` on, and writes the transient's statistics' `work` and `profile`: its Newton iterations and factorizations, and where its time went.
