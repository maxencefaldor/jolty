# Benchmarks

A benchmark task gives one agent a prompt and the files it names, and the agent works on its own, as long as it needs, with jolty as its compiler: its command line, its MCP server and its Python package, and whatever code it writes beside them.
A scorer then judges what the agent left, independently of the agent and of how it worked.

| Task | The agent's objective | It meets the task when |
| --- | --- | --- |
| [`cost_down`](cost_down/task.md) | make a transistor's bias cheaper, from a catalog of resistors | its design meets the original's specs and yield and costs less |
| [`cubic`](cubic/task.md) | output the cube of an input current, from 2 V, DC to 1 GHz | its errors, against the patented circuit's (US 6,160,427), average at most 1 |
| [`voltage_current`](voltage_current/task.md) | turn two inputs' difference into a current, 1 mA/V, from a −15 V supply alone | its errors, against the patented circuit's (US 6,166,529) as Koza measured them, average at most 1 |
| [`balun`](balun/task.md) | split one input into two half-size outputs 180° apart, from a 1 V supply, 1 to 100 MHz | its difference's gain and distortion match the patented circuit's (US 6,265,908) as Koza measured them, and both outputs balance |
| [`analogcoder`](analogcoder/task.md) | design AnalogCoder's twenty-four circuits, from amplifiers to a phase-locked loop | every circuit passes AnalogCoder's own check of it |

Each task is a directory: `task.md` is the agent's prompt, `given` holds the files the agent gets, and `scorer.py`, with any reference it compares against, stays with the scorer.
The agent never sees them.

`analogcoder` is the benchmark of AnalogCoder (Lai et al., AAAI 2025; [github.com/laiyao1/AnalogCoder](https://github.com/laiyao1/AnalogCoder), MIT License), its problems in their own words and its checks run on jolty.
Its scorer's docstring lists where those checks were changed: where they found parts by their SPICE names, or let through what their problems exclude.
Its `reference.py` holds a design for each problem, which the scorer's tests must pass.

`cubic`, `voltage_current` and `balun` come from Koza's re-invention of six circuits patented after 2000 (Koza, Keane, Streeter, Mydlowec, Yu and Lanza, *Genetic Programming IV*, 2003, chapter 15).
`voltage_current` keeps his test fixture, his fitness cases and his transistors, and weighs each case by the patented circuit's error as he measured it, so its fitness is computed as his was: the patent's is 1, and his evolved circuit's 0.619.
`balun` keeps his fixture and his seven fitness cases, but is met on the two figures he published for the patented and the evolved circuit, since his fitness totals cannot come from the sums his text states.
`cubic` keeps his 2 V supply and his reference current, but tests from DC to the gigahertz the patent claims, against the patented circuit rebuilt from its text in `patent.py`.

Run an agent on a task, Claude Code unless another command is given after the task:

```sh
uv run python benchmarks/run.py benchmarks/cost_down
```

The runner copies the task's given files into a fresh directory, runs the agent there with the task's `task.md` as its prompt, keeps its transcript, and scores the directory.
With `-` as the agent, it only prepares the directory and prints its path, for an agent started some other way.
Score a directory an agent already worked in:

```sh
uv run python benchmarks/score.py benchmarks/cost_down path/to/work
```
