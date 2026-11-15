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
uv run python benchmarks/agents/run.py benchmarks/agents/cost_down
```

The runner copies the task's given files into a fresh directory, runs the agent there with the task's `task.md` as its prompt, keeps its transcript, and scores the directory.
With `-` as the agent, it only prepares the directory and prints its path, for an agent started some other way.
Score a directory an agent already worked in:

```sh
uv run python benchmarks/agents/score.py benchmarks/agents/cost_down path/to/work
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
A wrapper sees only the runs of a command line, so in the jolty arm jolty logs every analysis it runs itself, whatever runs it: its command line, its MCP server, which no wrapper sees, or a script through its Python package.
Each process of a session finds the log in jolty's setting `analysis_log`, `JOLTY_ANALYSIS_LOG`, and appends a line as each analysis starts, of its kind, the runs it runs on, one alone or a batch's, and the time; the runner keeps that log beside the transcript too, as `analyses.jsonl`.
The runner scores the work in a process of its own, through `score.py`, so what an agent's scripts import or change stays there, and keeps the verdict beside them, as `verdict.json`.
Since scoring runs the agent's scripts, the scorer runs as a session does, in the sandbox and in an environment of its own, which holds no credential, and from an empty directory, so no `pyproject.toml` an agent leaves changes jolty's settings.
Claude Code prints its session as stream-json events, and `transcript.py` reads its turns, its tool calls by name, what it says it runs with, and its own report of the session's wall time, cost, tokens and end, with its last message.

Every session runs clean and confined, since Claude Code runs with every permission (`--dangerously-skip-permissions`):

- **Its own configuration.**
  The session's home, Claude Code's configuration home in it (`CLAUDE_CONFIG_DIR`) and its temporary directory are made empty for it, and its environment holds nothing of the runner's but the locale and Claude Code's credentials, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or `CLAUDE_CODE_OAUTH_TOKEN` (which `claude setup-token` makes for a subscription).
  Claude Code loads no settings file and no `CLAUDE.md` (`--setting-sources=`), and has the arm's MCP servers alone (`--strict-mcp-config`), so none of the user's settings, hooks, memory, skills, plugins or MCP servers reach it; only the settings an administrator manages, which it always applies, could, and the session records them.
  Its path holds the arm's simulator and Python, then the system's own directories.
- **A sandbox.**
  `sandbox.py` runs the whole session, the agent and everything it starts, its MCP servers and their scripts among them, in the platform's sandbox: Seatbelt (`sandbox-exec`) on macOS and bubblewrap (`bwrap`) on Linux, the mechanisms Claude Code's own sandbox uses for its commands.
  It reads nothing in the home directories, the temporary directories or mounted volumes but its own directories and the programs it runs, so neither the repository's scorers nor another session's work, and writes nothing outside its own directories.
  On macOS it also reaches no local service but DNS, and starts no program outside the sandbox, through Apple Events or Launch Services.
  It reaches the network, as Claude Code must reach its service: an agent could fetch what it likes, such as jolty from PyPI in the ngspice arm.
  On Linux, bubblewrap can filter no socket by its path nor any address, so the machine's other services stay within reach there.
- **What it ran with.**
  Each session records its command, its whole environment, a credential written as `(set)`, the managed settings, and its sandbox, and its transcript's first event says what Claude Code found: its tools, MCP servers, permission mode, skills, plugins and the source of its key.

Claude Code runs only in a sandbox: on a platform without one, `compare.py` refuses to start, and before the first session it runs `claude --version` and `claude auth status` in a session's configuration and sandbox, which cost nothing, and refuses to start when Claude Code finds no credentials there.
The stub runs in the sandbox too, wherever there is one, reading its own known good designs.

Run one session of the ngspice arm, or score its work:

```sh
uv run python benchmarks/agents/run.py --arm ngspice benchmarks/agents/balun
uv run python benchmarks/agents/score.py benchmarks/agents/balun path/to/work --arm ngspice
```

Run the comparison, five sessions per arm per task by default, so 60 paid sessions for the six tasks, on the model `--model` names or Claude Code's default:

```sh
uv run python benchmarks/agents/compare.py --agent claude
```

`--timeout SECONDS` has the runner stop a session, and `--turns N` and `--budget USD` have Claude Code stop itself, after so many of its turns or US dollars; none has a default, and the report records them.
The runner also stops whatever a session leaves running when it ends.

Before the paid comparison:

1. Put Claude Code on the path, and give the sessions a credential in the environment, `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` for a subscription: no session sees the user's own login.
1. Run the stub, below, which proves the harness and the sandbox for free.
1. Run one session, `--tasks cost_down --arms jolty --repetitions 1` with the limits chosen, and read its transcript and report: its setup should list jolty's MCP server connected and no skill or plugin, and its `analyses.jsonl` should hold a line for each analysis its calls of the MCP server ran.
   The stub starts the server with the session's environment, which is how its analyses find the log; this checks that Claude Code starts it so too.

It runs the sessions one at a time, the arms and the tasks taking turns in each round, each in a fresh directory outside the repository, and writes `benchmarks/agents/results/<date>-<agent>.json`, with each session's transcript, logs and verdict beside it, in `benchmarks/agents/results/<date>-<agent>/<task>/<arm>/<repetition>`.
The report holds every session, each arm's success rate on each task with its 95 % Wilson interval, as jolty states a yield's, the median and the range of its turns, tool calls, simulations, wall time and cost, and the arms' difference on each task, jolty's less ngspice's, with Newcombe's interval for the success rates'.
A session's simulations are counted where each arm can count them: in the jolty arm, every analysis jolty ran, a batch's analysis once however many runs it ran on, such as a Monte Carlo analysis's samples; in the ngspice arm, every run of ngspice, which may run several analyses that its wrapper cannot see.
Each session also keeps the runs of its simulator's command line by command, and in the jolty arm its analyses' runs and kinds.

It also holds the failure kinds, each from what is known, never read from words:

- **How the session ended**, as the runner saw it and the agent reports it: finished, out of time (the runner stopped it at `--timeout` seconds), out of turns (`--turns`), out of budget (`--budget`, in US dollars), or errored, when the agent reports an error, exits with other than 0 or leaves no report.
- **How the work fell short**, as its scorer found it: no deliverable (`missing`), a design that does not build or read (`unbuilt`), one the engine cannot elaborate (`unelaborated`), a rule's error (`rule`), a failed analysis (`analysis`), a requirement of the task not met (`spec`), which it names, such as a measurement, `parts`, `yield` or `cost`, or a scorer that could not judge (`scorer`).
- **The agent's last report**, its final message, beside its verdict.

Each arm's summary on a task counts its sessions by how they ended, by each kind they fell short in, and by each spec.
AnalogCoder's problems count as one task, met when all twenty-four are, and each of its sessions keeps the scorer's verdict on every problem.

`--agent stub` runs `stub.py` in the agent's place, which writes a known good design of the cost-down and of AnalogCoder in either arm's form, and costs nothing.
In the jolty arm it simulates its design through each of jolty's front ends, an analysis each, and in the ngspice arm it runs ngspice once:

```sh
uv run python benchmarks/agents/compare.py --agent stub --repetitions 2 --results /tmp/stub
```
