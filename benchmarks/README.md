# Benchmarks

A benchmark task gives one agent a prompt and the files it names, and the agent works on its own, as long as it needs, with jolty as its compiler: its command line, its MCP server and its Python package, and whatever code it writes beside them.
A scorer then judges what the agent left, independently of the agent and of how it worked.

| Task | The agent's objective | It meets the task when |
| --- | --- | --- |
| [`cost_down`](cost_down/task.md) | make a transistor's bias cheaper, from a catalog of resistors | its design meets the original's specs and yield and costs less |
| [`cubic`](cubic/task.md) | output the cube of an input current, from 2 V, DC to 1 GHz | its errors, against the patented circuit's (US 6,160,427), average at most 1 |
| [`cubic_discrete`](cubic_discrete/task.md) | output a scaled cube of an input current, from 2 V, with 2N3904s and 2N3906s and no reference | its mean error is at most the patented circuit's (US 6,160,427) as Koza measured it |
| [`voltage_current`](voltage_current/task.md) | turn two inputs' difference into a current, 1 mA/V, from a −15 V supply alone | its errors, against the patented circuit's (US 6,166,529) as Koza measured them, average at most 1 |
| [`balun`](balun/task.md) | split one input into two half-size outputs 180° apart, from a 1 V supply, 1 to 100 MHz | its difference's gain and distortion match the patented circuit's (US 6,265,908) as Koza measured them, and both outputs balance |
| [`analogcoder`](analogcoder/task.md) | design AnalogCoder's twenty-four circuits, from amplifiers to a phase-locked loop | every circuit passes AnalogCoder's own check of it |

Each task is a directory: `task.md` is the agent's prompt, `given` holds the files the agent gets, and `scorer.py`, with any reference it compares against, stays with the scorer.
The agent never sees them.

`analogcoder` is the benchmark of AnalogCoder (Lai et al., AAAI 2025; [github.com/laiyao1/AnalogCoder](https://github.com/laiyao1/AnalogCoder), MIT License), its problems in their own words and its checks run on jolty.
Its scorer's docstring lists where those checks were changed: where they found parts by their SPICE names, or let through what their problems exclude.
Its `reference.py` holds a design for each problem, which the scorer's tests must pass.

`cubic`, `cubic_discrete`, `voltage_current` and `balun` come from Koza's re-invention of six circuits patented after 2000 (Koza, Keane, Streeter, Mydlowec, Yu and Lanza, *Genetic Programming IV*, 2003, chapter 15).
`voltage_current` keeps his test fixture, his fitness cases and his transistors, and weighs each case by the patented circuit's error as he measured it, so its fitness is computed as his was: the patent's is 1, and his evolved circuit's 0.619.
`balun` keeps his seven fitness cases and his fixture, but for a capacitor that keeps the source's DC level from the input, and is met on the two figures he published for the patented and the evolved circuit, since his fitness totals cannot come from the sums his text states.
`cubic_discrete` keeps his fixture, his four slow fitness cases and his transistors, and is met at the patented circuit's mean error as he measured it.
`cubic` keeps his 2 V supply, but gives a reference current and tests from DC to the gigahertz the patent claims, against the patented circuit rebuilt from its text in `patent.py`.
Koza simulated his circuits with exact parts at 27 °C; these four tasks are met only by a design that meets its task from 0 °C to 70 °C with its supply within 5 % of its level, both with exact parts and on 95 % of 100 boards built of it, each board checked over the same range, so a design that holds only for exact values or nominal conditions falls short.
On the discrete tasks' boards, each resistor lies within 1 % of its value, each capacitor within 5 %, and each type of transistor comes from one reel anywhere in its datasheet's range of current gain and base-emitter voltage, its transistors matched as a pair from adjacent die is; on the integrated cubic's, the resistors lie together within 20 % and each within 0.1 % of the others, as on one die.
Two more of his six, the variable capacitor (US 6,013,958) and the high-current load (US 6,211,726), use power MOSFETs whose makers' SPICE models may not be redistributed; `jolty.parts` fits each of the four instead, by one stated recipe, to the figures its datasheet states.

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

## Comparing an agent with jolty and with ngspice

Roadmap item 8 asks what jolty gives an agent: each task is done in two arms, which are compared by success and by iterations.

- The **jolty arm** is the runner above: the agent has jolty's command line, MCP server and Python package, and writes `design.py`.
- The **ngspice arm** has ngspice and a Python with numpy, scipy and matplotlib, made afresh for each session, and no jolty on its path or in its Python.
  Its prompt is `task.md` with each sentence the task's `ngspice/task.toml` replaces in place, and the tools of `ngspice.toml` in place of jolty's.
  It gets the task's files but its Python scripts, which need jolty, and the SPICE files of the task's `ngspice/given` in their place: the given transistors as model cards in `parts.lib`, and the cost-down's stage in `stage.cir`, with `bias.cir` to run it.
  It writes `design.cir`, a SPICE library whose subcircuit `design` is its circuit; for AnalogCoder, `p1.cir` to `p24.cir`, with their inputs' levels in `levels.json`, and for the cost-down the subcircuit `stage`, with its resistors' tolerances in `tolerances.json`.
- Both arms are judged by the same scorers, on the same simulator.
  jolty's SPICE reader reads a netlist, after the task's `.lib` files, into the component a script would build, and a transistor is a given part when its parameters are those the reader makes of the given model.
  A netlist jolty cannot read fails, as a script that does not build does.

Each arm's simulator, `jolty` or `ngspice`, is a wrapper on the agent's path that logs every run, `counted.py`, and the runner keeps the log beside the transcript, as `simulations.jsonl`.
Claude Code prints its session as stream-json events, and `transcript.py` reads its turns, its tool calls by name, and its own report of the session's wall time, cost and tokens.

Run one session of the ngspice arm, or score its work:

```sh
uv run python benchmarks/run.py --arm ngspice benchmarks/balun
uv run python benchmarks/score.py benchmarks/balun path/to/work --arm ngspice
```

Run the comparison, five sessions per arm per task by default, so 60 paid sessions for the six tasks, on the model `--model` names or Claude Code's default:

```sh
uv run python benchmarks/compare.py --agent claude
```

It runs the sessions one at a time, the arms and the tasks taking turns in each round, each in a fresh directory outside the repository, and writes `benchmarks/results/<date>-<agent>.json`, with each session's transcript and log beside it, in `benchmarks/results/<date>-<agent>/<task>/<arm>/<repetition>`.
The report holds every session, each arm's success rate on each task with its 95 % Wilson interval, as jolty states a yield's, the median and the range of its turns, tool calls, simulator runs, wall time and cost, and the arms' difference on each task, jolty's less ngspice's, with Newcombe's interval for the success rates'.
AnalogCoder's problems count as one task, met when all twenty-four are, and each of its sessions keeps the scorer's verdict on every problem.

`--agent stub` runs `stub.py` in the agent's place, which writes a known good design of the cost-down and of AnalogCoder in either arm's form, and costs nothing:

```sh
uv run python benchmarks/compare.py --agent stub --repetitions 2 --results /tmp/stub
```
