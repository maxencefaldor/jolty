# Benchmarks

A benchmark task gives an agent a prompt and the specified input files.
The agent works autonomously for as long as needed using jolty (the command-line tool, MCP server, and Python package) together with any helper code the agent writes.
An independent scorer then evaluates the resulting deliverables without inspecting the agent's process.

| Task | Objective | Passing criterion |
| --- | --- | --- |
| [`cost_down`](cost_down/task.md) | Reduce the cost of a transistor bias stage using a resistor catalog | The design meets the original specifications and yield at a lower cost |
| [`cubic`](cubic/task.md) | Output the cube of an input current from a 2 V supply, DC to 1 GHz | Mean normalized error against the patented circuit (US 6,160,427) is at most 1 |
| [`cubic_discrete`](cubic_discrete/task.md) | Output a scaled cube of an input current from a 2 V supply using 2N3904 and 2N3906 transistors without a reference current | Mean error does not exceed the patented circuit (US 6,160,427) as measured by Koza |
| [`voltage_current`](voltage_current/task.md) | Convert a differential input voltage to current at 1 mA/V using only a -15 V supply | Mean normalized error against the patented circuit (US 6,166,529) as measured by Koza is at most 1 |
| [`balun`](balun/task.md) | Split one input into two half-amplitude outputs 180° apart from a 1 V supply, 1 to 100 MHz | Differential gain and distortion match the patented circuit (US 6,265,908) as measured by Koza, with balanced outputs |
| [`analogcoder`](analogcoder/task.md) | Design the twenty-four AnalogCoder circuits, from amplifiers to a phase-locked loop | Every circuit passes the corresponding AnalogCoder verification check |

Each task lives in a directory containing the prompt (`task.md`), the agent's starter files (`given`), and the private `scorer.py` with any reference circuits.
The agent never sees the scorer or reference files.

`analogcoder` implements the AnalogCoder benchmark (Lai et al., AAAI 2025; [github.com/laiyao1/AnalogCoder](https://github.com/laiyao1/AnalogCoder), MIT License), preserving the original problem descriptions and running the verification checks on jolty.
The scorer docstring documents every adjustment made to the original checks, such as removing SPICE-name lookups and enforcing constraints that the original checks omitted.
The task's `reference.py` stores a reference design for every problem, and the scorer test suite verifies that every reference passes.

`cubic`, `cubic_discrete`, `voltage_current`, and `balun` come from Koza's reinvention of six circuits patented after 2000 (Koza, Keane, Streeter, Mydlowec, Yu, and Lanza, *Genetic Programming IV*, 2003, chapter 15).
`voltage_current` preserves Koza's test fixture, fitness cases, and transistors, weighting each case by Koza's measured error for the patented circuit.
Under that normalization, the patented circuit scores 1 and Koza's evolved circuit scores 0.619.
`balun` preserves Koza's seven fitness cases and test fixture except for a coupling capacitor that blocks the source DC level from the input.
Passing criteria use the two headline metrics Koza published for the patented and evolved circuits, because the published fitness totals cannot be reproduced from the stated sums.
`cubic_discrete` preserves Koza's fixture, four low-frequency fitness cases, and transistors, with the threshold set to Koza's measured mean error for the patented circuit.
`cubic` keeps Koza's 2 V supply while adding a reference current and testing from DC to the 1 GHz bandwidth claimed in the patent, scored against the patented circuit rebuilt in `patent.py`.
Koza simulated circuits with exact nominal components at 27 °C.
All four tasks require passing from 0 °C to 70 °C across a ±5 % supply range, both with nominal parts and on at least 95 % of 100 Monte Carlo boards checked across that same range.
Designs that work only with exact component values or nominal operating conditions therefore fail.
On discrete-task boards, each resistor varies within 1 %, each capacitor varies within 5 %, and each transistor type comes from one datasheet-range reel for current gain and base-emitter voltage with matched adjacent-die pairs.
On integrated `cubic` boards, resistors share a common 20 % process variation while matching within 0.1 % of each other, as on a single die.
Two additional circuits from Koza's six, the variable capacitor (US 6,013,958) and high-current load (US 6,211,726), rely on power MOSFETs whose vendor SPICE models cannot be redistributed.
Instead, `jolty.parts` fits each of the four included transistors to datasheet specifications using one documented procedure.

Run an agent on a task, defaulting to Claude Code unless another command follows the task path:

```sh
uv run python benchmarks/agents/run.py benchmarks/agents/cost_down
```

The runner copies the task's `given` files into a fresh directory, invokes the agent with `task.md` as the prompt, saves the transcript, and scores the workspace.
Passing `-` as the agent command prepares the workspace and prints the path so an external agent can run there.
Score an existing agent workspace directly:

```sh
uv run python benchmarks/agents/score.py benchmarks/agents/cost_down path/to/work
```

## Comparing an agent with jolty and with ngspice

Roadmap item 8 measures what jolty adds to an agent by running each task in two arms and comparing success rates and iteration counts.

- The **jolty arm** uses the runner above: the agent receives jolty's command line, MCP server, and Python package, and writes `design.py`.
- The **ngspice arm** provides ngspice and a fresh Python environment with numpy, scipy, and matplotlib, with no jolty on `PATH` or in Python.
  The ngspice arm builds the prompt from `task.md` by applying the sentence replacements in `ngspice/task.toml` and substituting the tool instructions from `ngspice.toml`.
  The workspace receives every file in `given` except Python scripts requiring jolty, replacing them with the SPICE files from `ngspice/given`.
  The SPICE files provide transistor models in `parts.lib`, plus `stage.cir` and `bias.cir` for the cost-down task.
  The ngspice arm expects `design.cir` with a `design` subcircuit for most tasks, `p1.cir` through `p24.cir` plus `levels.json` for AnalogCoder, and subcircuit `stage` plus `tolerances.json` for cost-down.
- Both arms are judged by the same scorers using the same simulator.
  jolty's SPICE reader loads each netlist after the task's `.lib` files into the same component structure a Python script builds, recognizing a transistor as an allowed part when parameters match the given model.
  A netlist that jolty cannot parse fails just like a Python script that fails to build.

Each arm wraps the simulator (`jolty` or `ngspice`) on `PATH` via `counted.py` to log every command-line invocation into `simulations.jsonl` alongside the transcript.
A PATH wrapper observes only CLI invocations, so the jolty arm also enables built-in analysis logging across CLI calls, MCP server calls, and Python scripts.
Each process reads the log path from jolty's `analysis_log` setting (`JOLTY_ANALYSIS_LOG`) and appends one entry per analysis with the analysis kind, run count (single or batch), and duration.
The runner stores that log as `analyses.jsonl` alongside the transcript.
The runner invokes `score.py` in a separate process so agent imports and side effects stay isolated, writing `verdict.json` alongside the logs.
Scoring executes agent-authored scripts, so the scorer runs inside the same sandbox with a credential-free environment.
The scorer also runs from an empty working directory so no `pyproject.toml` left by an agent can alter jolty settings.
Claude Code emits stream-json events, and `transcript.py` extracts turn counts, tool calls by name, reported configuration, wall time, cost, token counts, termination reason, and the final message.

Every session runs clean and confined because Claude Code executes with full permissions (`--dangerously-skip-permissions`):

- **Isolated configuration.**
  Each session receives a fresh empty home directory, `CLAUDE_CONFIG_DIR`, and temporary directory.
  The environment strips runner variables, passing only the locale and Claude Code credentials (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`).
  Claude Code skips user settings files and `CLAUDE.md` (`--setting-sources=`) and loads only the arm's MCP servers (`--strict-mcp-config`), blocking user settings, hooks, memory, skills, plugins, and unrelated MCP servers.
  Only administrator-managed settings, which Claude Code always enforces, can apply, and the session records them.
  The session `PATH` lists the arm's simulator and Python environment first, followed by standard system directories.
- **Process sandbox.**
  `sandbox.py` confines the entire session, including the agent, MCP servers, and spawned scripts, using Seatbelt (`sandbox-exec`) on macOS or bubblewrap (`bwrap`) on Linux, matching Claude Code's built-in command sandbox.
  Read access outside standard executables is restricted to the session directories, blocking repository scorers, other sessions, home directories, temporary directories, and mounted volumes; write access is restricted to the session directories.
  On macOS, the sandbox also blocks local services other than DNS and prevents launching external programs through Apple Events or Launch Services.
  Network access remains open so Claude Code can reach the API service, meaning an agent could fetch external packages such as jolty from PyPI during the ngspice arm.
  On Linux, bubblewrap cannot filter sockets by path or destination address, leaving other local host services reachable.
- **Recorded provenance.**
  Each session logs the exact command, full environment (with credentials redacted as `(set)`), managed settings, and sandbox configuration.
  The first transcript event records the tools, MCP servers, permission mode, skills, plugins, and credential source that Claude Code actually detected.

Claude Code runs only inside a sandbox, so `compare.py` refuses to start on platforms without sandbox support.
Before launching the first session, `compare.py` runs `claude --version` and `claude auth status` inside the configured sandbox at zero API cost and aborts if no valid credentials are found.
The stub agent also runs inside the sandbox whenever supported, loading known good reference designs.

Run one ngspice-arm session or score a completed workspace:

```sh
uv run python benchmarks/agents/run.py --arm ngspice benchmarks/agents/balun
uv run python benchmarks/agents/score.py benchmarks/agents/balun path/to/work --arm ngspice
```

Run the full comparison (five sessions per arm per task by default, totaling 60 paid sessions across six tasks) using `--model` or Claude Code's default model:

```sh
uv run python benchmarks/agents/compare.py --agent claude
```

The `--timeout SECONDS` flag stops a session from the runner, while `--turns N` and `--budget USD` instruct Claude Code to stop after a turn or dollar limit; none has a default, and the report records all configured limits.
The runner also terminates any leftover child processes when a session ends.

Before running a paid comparison:

1. Ensure `claude` is on `PATH` and export a session credential (`ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`), since sessions cannot see interactive logins.
1. Run the stub comparison below to verify the harness and sandbox at zero cost.
1. Run one trial session (`--tasks cost_down --arms jolty --repetitions 1`) with your chosen limits and inspect the transcript and report.
   Confirm that the setup lists the jolty MCP server with no skills or plugins, and that `analyses.jsonl` contains one entry per MCP analysis call.
   The stub launches the MCP server with the session environment so analyses find the log path; the trial run confirms Claude Code does the same.

The comparator runs sessions sequentially, alternating arms and tasks in each round inside fresh directories outside the repository.
Results go to `benchmarks/agents/results/<date>-<agent>.json`, with per-session transcripts, logs, and verdicts stored under `benchmarks/agents/results/<date>-<agent>/<task>/<arm>/<repetition>`.
The report records every session, each arm's per-task success rate with a 95 % Wilson interval (matching jolty yield intervals), and the median and range of turns, tool calls, simulations, wall time, and cost.
The report also computes the per-task success-rate difference (jolty minus ngspice) with a Newcombe confidence interval.
Simulation counts reflect what each arm can observe directly.
In the jolty arm, each executed analysis counts once regardless of batch size (such as Monte Carlo samples).
In the ngspice arm, each ngspice process invocation counts once, even if a single run executes multiple analyses invisible to the wrapper.
Each session also logs CLI simulator invocations by command, and the jolty arm logs per-analysis run counts and kinds.

The report classifies failures strictly from structured signals rather than free-text parsing:

- **Session termination:** recorded from the runner and agent metadata as `finished`, `out of time` (`--timeout`), `out of turns` (`--turns`), `out of budget` (`--budget` in US dollars), or `errored` (agent error, nonzero exit, or missing report).
- **Scorer shortfall:** classified as `missing` (no deliverable), `unbuilt` (build or parse error), `unelaborated` (elaboration error), `rule` (rule error), `analysis` (failed analysis), `spec` (named unmet requirement such as `parts`, `yield`, or `cost`), or `scorer` (scorer failure).
- **Final agent message:** stored alongside the scorer verdict.

Per-arm task summaries count sessions by termination status, shortfall category, and failed specification.
AnalogCoder counts as a single task that passes only when all twenty-four circuits pass, while each session preserves per-circuit scorer verdicts.

Passing `--agent stub` runs `stub.py` instead of a live agent, emitting known good designs for `cost_down` and `analogcoder` in either arm's format at zero cost.
In the jolty arm, the stub runs one analysis through each jolty front end; in the ngspice arm, the stub invokes ngspice once:

```sh
uv run python benchmarks/agents/compare.py --agent stub --repetitions 2 --results /tmp/stub
```
