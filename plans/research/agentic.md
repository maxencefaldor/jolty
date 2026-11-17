# Research: agentic design

The evidence for `agentic.md`: what jolty gives an agent today, how agents and people drive the reference simulators, what the published LLM-agent work for analog design assumes and where it fails, what gradient-based design needs, and what would make jolty the tool agents choose.
Written on 15 November 2026 from `main`'s checkout at `d10fc4d` in the repository (line numbers are that commit's), the notes, ngspice 47's and Xyce's sources in the session scratchpad, and published work.
Read only: nothing was built, run or measured; every number is read from code, recorded in the notes, computed here from recorded figures (marked), or an estimate (marked).

## 1. Summary

1. jolty already gives an agent more than any reference does: one request model behind the command line and the MCP server, every problem at once with its script line, 16 rules, a schematic and plots on every run, 23 measurements read by the engine, verdicts with slack at every corner, and Monte Carlo and worst case in the same command.
    None of the reference simulators or published agent frameworks surveyed here combines these.
1. None of it is measured.
    No agent session has been counted or timed; the six discoveries were not blind and kept no transcripts; item 8 waits for the owner's credential.
1. Item 8 as built would undercount the jolty arm's tries: it counts only runs of the `jolty` command line, not MCP calls or Python API runs.
    With 5 sessions per arm per task, a difference in success smaller than about 0.6 on one task, or about 0.3 pooled over the six tasks, does not separate from zero (computed here).
1. The discovery scripts show where an agent's effort goes.
    It writes bisections (50 steps for one resistor), Monte Carlo loops and measurements by hand, because jolty says what is true but not which way a requirement moves: its sensitivities are of unknowns, not of specs, and nothing in jolty optimizes.
1. The literature agrees.
    Simulator feedback with computed measurements lifts success the most (SPICEAssistant, 14.8% to 53.0%; AnalogPortBench, 30% to 56%).
    Sizing goes to numerical optimizers.
    Agents fail on oscillators, on changes of topology and on infeasible specs.
1. Move 1, measure honestly: count every simulation at the engine whatever the front end, then run item 8 with ablations of jolty's features and with tasks held out of any public source.
1. Move 2 (lead), facts that point to the next move: the derivative of every requirement's slack, each transistor's operating region, each measurement's numerical error, and an optimizer the agent calls on parameters it names.
    The derivatives and the optimizer would rest on one backward pass per scalar loss.
    Xyce differentiates no `.MEASURE` result and integrates backward once per objective time point.
1. Move 3, a loop agents can reach and afford: the reference available inside the session (instructions derived from the code, the guide and examples offline), a warm runner with an exact cache and session queries, and a sandboxed path that takes the circuit as data and runs no code.

## 2. Where jolty stands

### 2.1 The loop, end to end

| Step | What jolty gives an agent | Evidence |
| --- | --- | --- |
| Describe | A circuit is a Python script. `@component` functions place devices and other components through a `Builder`, so a design is a hierarchy of reusable functions. Every device constructor checks each parameter's dimension. There are 30 device constructors and seven real parts with their ratings. SPICE libraries are read into components, and SKY130's transistors run through ngspice | README; AGENTS.md, "Units"; `src/jolty/devices.py`; `docs/guide/parts.md:11–17` |
| Problems | Every reason a circuit does not elaborate, all at once, each naming the instance's path and the script line that placed it | AGENTS.md, "Feedback is facts"; `service.py:191–194` |
| Rules | 16 rules at configurable levels. Structural: `unconnected-terminal` to `non-standard-value`. At the operating point: `gmin-dependence`, `multiple-operating-points`, `unstable-pole`. On results: `phase-margin`, `gain-margin`, `over-rating`, `derating`. `jolty explain` states each rule's exact condition | `docs/reference/rules.md:26–103`; `docs/guide/checks.md` |
| Script errors | The innermost script line of the traceback, then its callers' script lines | `runner.py:221–242` |
| Images | A schematic on every build, marking the instances the diagnostics name and showing the operating point's voltages; a plot of every analysis; histograms and envelopes for batches. The MCP server sends them as images beside the text | AGENTS.md, "Images are feedback"; `service.py:575–592`; `mcp.py:162–173` |
| Specs and verdicts | 23 measurements the engine reads, and measurements a script writes in Python. Each requirement gets one of four verdicts (passed, marginal, failed, unmeasured), with the value read, its slack, its worst corner and its script line. Yield comes with a Wilson interval; worst case is judged at the tolerances' extremes | `docs/guide/specs.md:32–98`; `specs/_requirements.py:174–192`; `specs/_measurements.py` |
| Simulation | 10 analyses: operating point, every DC solution, DC sweep, AC, transient (and the periodic steady state of driven circuits), sensitivity, stability, noise, pole-zero and transfer function. Corners, Monte Carlo, worst case, variants, batches across threads, and shards across machines | `docs/guide/simulation.md:27–47`, `186–299` |
| Iterate | Two front ends derive from one request model: the command line (`--output-format json`) and the MCP server (text, structured content, images). The Python API (`Simulation`, `Elaborated`, `Simulation.batch`) serves in-process loops | `service.py:1–10`; `cli.py:30–43`; `mcp.py:132–180`; `simulate.py:204–295` |
| Bounds | 300 s per script, 20,000 characters kept of each output stream, 40 unknowns listed per analysis, 20 diagnostics listed | `settings.py:57–76` |

This is the "agent-computer interface" the SWE-agent paper argues for (§3.5): few, compact commands, bounded output, and guardrails that catch an error before it costs a run.
The rules at the operating point state facts that ngspice reports, if at all, only when a solve fails (§3.1).

### 2.2 What the six discoveries show

In each discovery an agent wrote `candidates.py` and `explore.py` against the Python API and kept what the numbers supported (`docs/discovery.md:3–7`).
What those scripts do by hand marks what jolty does not yet do:

- **Tuning by bisection.**
  The current source tunes each candidate to 1 mA by 50 bisections in the logarithm of a resistor, one operating point per step (`discovery/current_source/explore.py:33–45`).
  The bandgap bisects its Widlar resistor from 50 Ω to 1000 Ω down to 0.1 Ω, two operating points per step (125 °C and −40 °C), so about 14 steps and 28 solves per flattening (`discovery/bandgap/explore.py:54–73`, counted here).
  With the derivative of the tilt with respect to R2, which jolty's adjoint gives at DC (§2.4), Newton's method would take about four steps (estimate).
- **Monte Carlo by hand.**
  The notch draws 200 samples with `jolty.variation`, building a new `Simulation` for each, which elaborates the circuit from JSON again (`discovery/notch/explore.py:40–52`; `simulate.py:359–370`).
  `Simulation.batch` runs the same samples in one call across threads, from a circuit elaborated once (`docs/guide/simulation.md:272–299`), and the script does not use it.
- **Measurements by hand.**
  "No spec of jolty measures a response's minimum over a band, so the agent's own code does" (`discovery/notch/findings.md:9`).
  The sine oscillator computed its own harmonic distortion; the `distortion` measure landed about 14 hours after the discoveries (`c2be2ef`, against `ae22268`).
  The VCO did use `jolty.specs.frequency` (`discovery/vco/explore.py:18`, `44`).
- **Numerical error unknown to the agent.**
  "With 800 steps a period, the frequency at 5 V reads 0.37 % above its value with 8000" (`discovery/vco/findings.md:62`).
  The agent found that only by running again; no report states a measurement's numerical error.
- **No record of effort.**
  Every page says the agent knew the textbook answers ("not a blind result").
  No transcript, turn count or wall time was kept: the commits carry the findings only (`git log -- discovery/`).
- **Discovery drives the tool.**
  The memristive NOR's commit also changed `specs.py` (`a5722f1`), and the sine oscillator preceded the distortion measure.

### 2.3 Where an agent loses time or goes wrong

1. **Learning the API.**
   Inside a session an agent learns jolty from three sources: the MCP server's instructions, the tools' descriptions and the package's docstrings.
   The instructions run to 792 words (`mcp.py:36–110`, counted here).
   They name 11 of the 30 device constructors, no transistor and no real part, and mention neither `specs()` nor corners nor Monte Carlo.
   The simulate tool's own description does name specs, corners, Monte Carlo and variants (`service.py:269–286`).
   In item 8's sandbox the agent can read jolty's source, because the editable install puts the repository's `src` on `sys.path` (`.venv/.../jolty.pth`) and the sandbox lets it read every directory on that path (`benchmarks/run.py:255–262`).
   It cannot read the guide or the examples, which lie outside `src`, unless it fetches the website over the open network.
   `docs/agents/connect.md:51–59` lists 9 tools; the server registers 10, `merge_shards` among them (`mcp.py:201–264`).
   Estimate: a session's first several turns go to reading source.
   The literature puts a price on missing context: AnalogCoder's GPT-3.5 solves 7 tasks without its domain context and 10 with it, and SPICEPilot depends on a "pilot prompt" of PySpice practice (§3.4).
1. **Writing harness code** for tuning, sweeps, Monte Carlo and measurements (§2.2).
   Each loop is code the agent writes and debugs, paying in turns.
1. **Reading an operating point.**
   No report states each transistor's region: a search of `report.py`, `checks.py` and the guide finds none.
   For a 20-transistor op-amp the agent infers regions from node voltages, or records the devices and reads their conductances in Python.
   AnalogCoder's own check is that every MOSFET is in its region, and AnalogPortBench feeds back each transistor's operating point (§3.4).
1. **Oscillators.**
   The periodic transient needs the sources' period ("`stop` the period of every source", `simulation.md:56`), so an autonomous oscillator has no steady-state analysis.
   The agent kicks the circuit and runs long transients: 16 periods at 4,000 steps each for the VCO, 150 ms at 1 µs for the sine oscillator (`discovery/*/findings.md`).
   This is where LLM agents fail most: AnalogCoder's oscillator, Schmitt trigger, VCO and PLL score zero for every model.
1. **Sensitivities that do not speak in specs.**
   `Sensitivity(outputs)` differentiates unknowns with respect to every number of every device (`plan.py:195–206`).
   "No measure reads sensitivities" (`specs/_requirements.py:158–162`).
   Worst case finds each requirement's slope by one run per toleranced parameter (`simulation.md:239`).
1. **Images that fill the context.**
   A plot is 1200 pixels wide and 120 + 360 pixels per panel high (8 inches by 0.8 + 2.4 per panel, at 150 dpi, `plots.py:304–307`).
   Claude counts an image's visual tokens as ⌈w/28⌉·⌈h/28⌉ (Anthropic's vision documentation).
   That makes 774 visual tokens for one panel, 1,290 for two and 1,849 for three (computed here).
   A simulation with a transient of voltages and currents, a Bode plot and a schematic of at least 187 tokens costs about 2,800 visual tokens.
   Thirty such calls keep about 85,000 in the conversation (estimate; it depends on the harness).
   Images serve qualitative checks better than quantitative ones: SPICEAssistant found GPT-4o read a ripple right from images in 1 case of 5, against 2 of 5 from numbers (§3.4).
   An agent can turn both images off (`schematic`, `plot`: `service.py:309–314`), but nothing tells it what they cost.
1. **The loop's time.**
   It is unmeasured (`research/speed.md` §2.1; `benchmarks.md:48`).
   A `jolty build` of the three-device filter reports 0.27 s inside the service (`docs/get-started.md:87`, the guide's example output).
   That figure includes the runner, a fresh `python -P -m jolty.runner` that every command starts from either front end (`runner.py:3–4`; `service.py:1405–1414`).
   The command line adds `uv run` and its own Python process, whose `import jolty.cli` took 0.31 s (handoff, decision 8).
   Estimate: 0.6–2 s for a small circuit's command, against 0.81 ms for `bjt_amp` in the engine.
   For an agent a model turn takes seconds, so this matters most for long runs and for an agent's own loops.
   The guide says "the loop takes seconds" (`docs/agents/why-a-compiler.md:26`) without a measurement.

### 2.4 Sensitivities and gradients today

- **At DC:** the adjoint method, one transposed solve per output, serving every parameter (`crates/jolty-engine/src/sensitivity.rs:1–8`, `374–423`).
- **Over frequency:** the adjoint at each frequency.
    The dependence through the operating point is a central difference of 1 µV either way, since devices give no second derivatives (`sensitivity.rs:9–14`, `47`, `533–551`).
- **Over time:** the direct method (Hocevar et al., 1985).
    It factors once per step and solves once per parameter per step, and keeps a tangent of the size of the system for each parameter (`sensitivity.rs:16–23`, `189–194`, `624–671`).
- **Exact device derivatives:** every device computes with dual numbers, temperature scaling included (`docs/guide/simulation.md:106`).
    ngspice instead perturbs each parameter by 10⁻⁶ (§3.1).
- **Of what:** a twin for every real number of every device (`sensitivity.rs:319–338`), with no way to name the parameters that matter.
    Outputs are unknowns only.
- **Shooting** already carries the monodromy matrix along a period by the direct method (`simulation.md:63`).
- **The blueprint** (an archived note, not kept §4) adds three things: one transposed solve per scalar loss at DC; a discrete adjoint through the transient with Griewank and Walther's binomial checkpointing; and bridges to JAX and PyTorch, with parameters declared trainable.

What an optimizer, or an agent using one, needs beyond that:

1. Parameters named by path pattern, not every number.
1. The derivative of each requirement's measurement, and so of its slack, at the condition where it is worst.
    A crossing time t\* where x(t\*, θ) = L moves by −(∂x/∂θ)/(∂x/∂t).
    A gain in decibels moves by (20/ln 10)·Re(H̄·∂H/∂θ)/|H|².
    Bandwidth, crossover and phase margin follow by the same implicit differentiation.
1. One backward pass per scalar loss: at DC on the factors already computed, over a response one transposed solve per frequency, and over a transient about one backward step per forward step plus recomputation, which the blueprint puts at 1.3× at most (its estimate).
1. Bounds, and discrete values: E-series values and a catalog's parts, which jolty already knows (`non-standard-value`, `jolty cost`).
1. The same bits on any thread count, as every result has (AGENTS.md, "Determinism").

### 2.5 Safety and reproducibility

- **Scripts** run in their own process, with a timeout and captured output; "that process is not a sandbox" (`service.py:212–213`, `284–285`).
  The MCP server marks building and simulating as running code, destructive and open-world, so a harness asks before each call (`mcp.py:123–128`; `docs/agents/connect.md`).
- **Item 8's sandbox** confines the whole session, its MCP server and scripts among them: Seatbelt on macOS, bubblewrap on Linux (`benchmarks/README.md:63–76`; `benchmarks/sandbox.py`).
  It is untested on Linux, and the network stays open (`README.md:73`; handoff, items 99 and 105).
- **Determinism:** results are the same bits for any thread count, shard split or completion order (AGENTS.md).
  Shards carry a fingerprint of jolty's version and the platform (`shards.py:43–60`).
  A results file carries only its format's version (`files.py:231`).

### 2.6 Item 8: what it will measure, and what it will miss

Built (an archived note, not kept; `benchmarks/README.md`): six tasks in two arms, with one scorer shared by both.
Five sessions per arm per task make 60 paid sessions.
Each session runs clean and sandboxed, its transcript is read for turns and tool calls, and the results carry Wilson and Newcombe intervals and failure kinds read from verdicts.

What it would miss:

- **Simulations counted unevenly.**
  The counted runs are those of the wrapper put on the path in place of `jolty` or `ngspice` (`run.py:260`, `272`; `compare.py:149–157`).
  The jolty arm's MCP server is `jolty-mcp` (`run.py:50`), which is not wrapped, and a script that calls `Simulation` in Python is not counted either.
  The ngspice arm's agent must simulate with ngspice itself through no wrapper (`benchmarks/ngspice.toml`), so its runs go through the counted command line.
  Tool calls and turns are counted alike in both arms; the "simulations" column is not.
- **Power.**
  Per task, with 5 sessions against 5: 5/5 against 2/5 gives a Newcombe interval of [0.03, 0.88], and 4/5 against 2/5 gives [−0.16, 0.73].
  Pooled over the six tasks, 30 against 30: 24/30 against 16/30 gives [0.03, 0.47].
  So about 0.6 per task and 0.27–0.30 pooled are the smallest differences that separate from zero (computed here with the Wilson and Newcombe formulas; pooling treats the tasks as alike).
- **Contamination.**
  Four tasks come from Koza's book (2003).
  AnalogCoder's problems and checks are public on GitHub.
  The network is open in the sandbox, so an agent could fetch reference designs (`README.md:73`).
  Its web tools would show by name in the transcript, which `transcript.py` counts.
- **No attribution.**
  The run compares jolty with ngspice but not jolty's features with one another (images, rules, specs, MCP against the command line).
  That is what SWE-agent's ablations did (§3.5).
- **Pass or fail only.**
  Each verdict keeps the scorer's measurements and jolty's report, slacks included, and each transcript the session's cost (`verdict.py:61–74`; `transcript.py`), but the report's measure is pass or fail.
- **Free in the paid run:** every jolty report ends with "took X s" (`service.py:259`, `474`), so the transcripts will time each command inside the service.

## 3. The references

### 3.1 How agents and people drive ngspice

- **Batch and the control language.**
  A netlist with a `.control` block (`alter`, `foreach`, `meas`, `let`, `wrdata`), run with `ngspice -b`.
  Monte Carlo is a control-language loop that re-sources the deck for every sample (`research/speed.md` §3.10, `commands.c:371–374`).
  Item 8's ngspice arm works this way, with numpy, scipy and matplotlib and no wrapper (`benchmarks/ngspice.toml`).
- **The shared library**, `libngspice` (`src/include/ngspice/sharedspice.h:1–120`): `ngSpice_Init` with callbacks for printed output, status, data and a background thread's state; `ngSpice_Command` for any interactive command; `ngSpice_Circ` for a netlist sent as lines; `ngGet_Vec_Info` for direct access to vectors.
  "Some commands are rejected (e.g. 'plot', because there is no graphics interface)."
- **PySpice**, a CFFI binding to the shared library.
  Its last release is 1.5, its last push August 2024, and its maintainer works on it in his free time; InSpice is a 2025 fork.
  It is the interface nearly every LLM-agent paper assumes (§3.4).
- **spicelib** (Nuno Brum) runs LTspice, ngspice, QSPICE and Xyce in batch and reads their raw files.
    It is how scripts drive LTspice, which has no other API.
- **MCP servers for ngspice and LTspice** exist: ngspice-mcp (in D, over the shared library), SPICEBridge (hosted), mcp-spice (npm, batch), circuit-sim-mcp (PySpice), CircuitMCP and ltspice-mcp.
  Going by their own descriptions, they take a netlist and return vectors or measurements; none describes structural checks with provenance, spec verdicts or images.
  Their maturity is unknown.
- **Feedback** comes at solve time and names a node, not a netlist line or an instance's origin: "singular matrix: check node %s" (`src/maths/ni/niiter.c:134–136`), and "Timestep too small" through `CKTtrouble` (`src/spicelib/analysis/dctran.c:907`).
- **Sensitivities:** `.sens` at DC and in AC by the direct method, each parameter's effect on the matrix taken by perturbing it by 10⁻⁶ of its value (`src/spicelib/analysis/cktsens.c:24`, `80–92`, `571–575`).
    There is no transient sensitivity and no adjoint.
- **Oscillators:** a periodic steady-state analysis, `PSS`, with `oscnode` and `fguess` for autonomous oscillators (`src/spicelib/analysis/psssetp.c:26–59`; written 2010–2012, `dcpss.c:1–5`).
- **Falls short for agents:** text to parse; results in raw files; no checks before solving; every loop, measurement and statistic written by the agent; the Python binding barely maintained.

### 3.2 Xyce

Read for ideas only (GPL).

- **A persistent simulator object** behind a C interface, with a ctypes Python wrapper whose methods include `initialize`, `runSimulation`, `simulateUntil`, `setCircuitParameter`, `getCircuitValue` and `obtainResponse`, and DAC and ADC maps for mixed-signal co-simulation (`utils/XyceCInterface/README`; `xyce_interface.py.cmake:12–564`).
  A Flask REST server keeps such objects between HTTP requests (`utils/XyceCInterface/XyceRest.py:64–284`).
- **Devices in Python**, a TensorFlow model among them (`utils/PythonModelInterface/TFModel.py`).
- **`.SENS objfunc={expression} param=…`**, direct and adjoint, at DC, in AC and over a transient (`doc/Users_Guide/Xyce_UG_ch09.tex:1595–1700`).
  The transient adjoint "will consider every time step to be a unique, separate objective, and will need to perform a backward integration for each one" (`Xyce_UG_ch09.tex:1815–1820`).
  According to Xyce's user forum there is no sensitivity of a `.MEASURE` result, and no event-based objective such as a transient signal's maximum.
- **Optimization and uncertainty:** Dakota, as a black box or linked in (`src/DakotaLinkPKG`); Xyce's own guide says the black-box form involves file I/O and scripting "which can be error-prone" (`doc/Users_Guide/sampling.tex:619`).
  It also has Trilinos ROL optimizers for DC, AC and transient analyses, behind a build option (`src/AnalysisPKG/N_ANP_ROL_*_Optimization.C`; `cmake/tps.cmake:360–363`), plus `.SAMPLING`, `.EMBEDDEDSAMPLING` and `.PCE`.
- **Falls short:** it cannot be linked into jolty; its build is heavy; it gives no agent-oriented feedback; and it has no gradients of measurements.

### 3.3 Commercial tools' AI, as their makers describe it

| Maker | Feature | What it does, in its maker's words | State |
| --- | --- | --- | --- |
| Cadence | Cerebrus (July 2021) | "Reinforcement ML" that drives the digital flow from RTL to signoff; up to 10× productivity and 20% better PPA claimed | Shipping |
| Cadence | Cerebrus AI Studio (8 May 2025) | Agentic, multi-block, multi-user SoC implementation; time to market 5× faster claimed | Shipping |
| Cadence | Virtuoso Studio (April 2023) | AI "Design Space Optimization" to navigate competing specs and centre a design after migration; TSMC migration up to 3× faster; MediaTek reports 30% productivity on 2 nm analog IP (January 2025) | Shipping |
| Cadence | ADE GXL and Assembler | Local optimizers (BFGS, conjugate gradient, Brent-Powell, Hooke-Jeeves) and a global one; a "large-signal" sensitivity of any measurement by perturbation, distinct from Spectre's small-signal `sens` (Cadence blog and forum) | Shipping |
| Cadence | ViraStack AI Super Agent (CadenceLIVE, April 2026) | Agents for schematic creation, testbench development, circuit optimization and layout migration, in Virtuoso | Early engagements with development partners |
| Synopsys | DSO.ai (early 2020) | Reinforcement learning over "trillions of design recipes"; more than 100 production tape-outs (Tom's Hardware, 2026) | Shipping |
| Synopsys | ASO.ai | Reinforcement learning narrows an analog design space across hundreds of PVT corners with post-layout parasitics; RF migration from N6RF+ to N4P with Keysight (June 2025) | Shipping |
| Synopsys | AgentEngineer (March 2026) | "L4" multi-agent workflows; an AMS flow built on Custom Compiler Layout Synthesis, "up to 3X" productivity (DAC, July 2026) | Evaluation; availability planned for the second half of 2026 |
| Siemens | Solido Design Environment | Adaptive AI for verification at 6σ and beyond, additive AI for another 10–100×, assistive AI for PPA and yield; variation-aware sensitivity analysis and optimization | Shipping; no agentic claim found |
| Synopsys | HSPICE `.OPTIMIZE` | `.PARAM` variables with bounds, `.MEASURE` goals, a `.MODEL … OPT` statement with methods such as bisection and pass/fail (Star-HSPICE manual, 1998; arXiv 2301.10157) | Shipping; the gradient algorithm unconfirmed |

Scripting is how an outside agent reaches these tools: Ocean, a subset of SKILL, drives ADE and Spectre from the command line (Adelaide's tutorial; Cadence forums).
**Where they fall short for agents:** they are closed, licensed and GUI-first.
Their AI optimizes sizing, flows and verification over a fixed topology with black-box learners (reinforcement learning, Bayesian optimization, ML surrogates).
Their agentic products were in early engagements or evaluation in 2026, and every figure above is the vendor's own.

### 3.4 LLM agents for analog design, 2024–2026

| Work | Interface | Feedback | Evaluation | What fails |
| --- | --- | --- | --- | --- |
| AnalogCoder (Lai, Lee, Chen, Poddar, Hu, Pan, Luo; AAAI 2025) | Python with PySpice | Four checks: required nodes and parts; simulation and each MOSFET's region; a DC sweep; a functional test. At most three generations | Pass@1 and pass@5 from 15 to 30 samples a task, on 24 tasks: with GPT-4o and its library of subcircuits, pass@1 66.1 and 20 solved; 15 without the library | Tasks 17 and 22–24 (an oscillator, the Schmitt trigger, the VCO, the PLL) score zero for every model; "further attempts are unlikely to succeed" |
| AnalogCoder-Pro (Lai et al., 2025) | PySpice code | Error messages and waveform images in a diagnose-and-repair loop; Bayesian optimization for sizing | 13 circuit types, 28 circuits designed | |
| SPICEPilot (Vungarala, Alam, Ghosh, Angizi; 2024) | PySpice | A "pilot prompt" of hardware knowledge and PySpice practice | A PySpice dataset; better code accuracy than AnalogCoder, by its authors | Lack of hardware knowledge |
| SPICEAssistant (Nau, Krummenauer, Zimmermann; 2025) | LTspice, with Python tools that compute the mean output, ripple, switching frequency and settling time | Measured values after each simulation | 256 questions on three switched-mode supplies: solve rate 14.8% for GPT-4o alone, 53.0% with the tools; gains plateau after about five simulations | Ripple read right from numbers in 2 of 5 cases, from images in 1 of 5; netlist pin errors; topology changes |
| AnalogPortBench (Rajabi, Moussa, Saxena, Yang, Patnaik; October 2026) | Netlist in and out; ngspice with SKY130 | Measured metrics, limits, pass or fail, and each transistor's operating point, but no diagnosis; three attempts | 50 feasible tasks: the strongest model 30% best of three, 56% with feedback | Infeasible specs seldom flagged; headroom-bound amplifiers rarely redesigned; scored at the typical corner only |
| ADO-LLM (Yin, Wang, Xu, Li; 2024) | Simulator in the loop | The LLM proposes points; a Gaussian-process Bayesian optimizer explores | Two circuit types | |
| AmpAgent (Liu et al.; 2024) | Three agents: literature, mathematics in SymPy, sizing | A conventional optimizer with the LLM | Seven amplifiers; 1.32–4× fewer iterations than the optimizer alone (from summaries) | |
| LEDRO (IBM; 2024), EEsizer (2025), AnalogSAGE (December 2025), AutoSizer (Yu, Torbunov, Mandal, Ren; ICML 2026) | LLM and optimizer: TuRBO; Bayesian optimization with ngspice; two loops on AMS-SizingBench's 24 SKY130 circuits | Simulation results | Solution quality and convergence against optimizers | |

The patterns:

1. The interface is PySpice, or netlists run in batch through ngspice, LTspice or HSPICE.
1. The feedback is ad hoc text from the authors' own checks: regions, sweeps, functional tests.
1. Sizing goes to numerical optimizers; the LLM proposes topologies and priors.
1. Success is pass@k or a solve rate over 10 to 256 tasks, with iterations or time beside it.
1. The failures are netlist syntax and connections, floating nodes, bias regions, oscillators and mixed-signal circuits, topology changes, infeasibility and corners.
1. I found no study that gives the same agent two toolchains and compares them, as item 8 does.

### 3.5 Interfaces for agents, outside circuits

- **SWE-agent** (Yang, Jimenez, Wettig, Lieret, Yao, Narasimhan, Press; arXiv 2405.15793, 2024): interface design alone moves resolve rates by several points.
  On SWE-bench Lite with GPT-4 Turbo, the default resolved 18.0%.
  It fell to 15.0% without linting on edit, to 12.0% with iterative search in place of summarized search, to 12.7% with whole files in place of a 100-line window, and to 15.0% with the full history in place of the last five observations.
  Its principles: simple actions, compact actions, informative but concise feedback, guardrails, and concise context.
- **Anthropic, "Writing effective tools for agents"** (2025): return only high-signal information; paginate, filter or truncate ("For Claude Code, we restrict tool responses to 25,000 tokens by default"); offer a `response_format` of concise or detailed; make error responses actionable; evaluate tools with agents, tracking success, tool calls, runtime, tokens and error types.
- **MCP 2025-11-25:** experimental tasks for long-running requests, with polling and deferred results; input validation errors returned as tool errors "to enable model self-correction".
  jolty's server uses structured content and images (`mcp.py:162–173`) and sends no progress.

jolty already follows most of this: all problems at once, bounded lists, structured results, tools of few arguments.
What it lacks: a response format chosen per call, a stated cost of images, progress on long runs, and an evaluation with agents (item 8).

### 3.6 Gradient-based design

| Tool | DC | AC | Transient | Differentiates | Optimizer |
| --- | --- | --- | --- | --- | --- |
| ngspice 47 | Direct method, parameters perturbed by 10⁻⁶ | Same | None | Vectors | None built in |
| Xyce | Direct and adjoint | Adjoint | Direct, or adjoint with one backward integration per objective time point | Expressions of outputs, not `.MEASURE` results | Dakota; ROL as a build option |
| Spectre and ADE | `sens`, small-signal | `sens` | ADE perturbs each parameter, runs again and measures | Any measurement, by perturbation | BFGS, conjugate gradient, Brent-Powell, Hooke-Jeeves; global |
| HSPICE | Not checked here | | | `.MEASURE` goals | `.OPTIMIZE`, algorithm unconfirmed |
| DELIGHT.SPICE (Nye, Riley, Sangiovanni-Vincentelli, Tits; IEEE TCAD 7(4), 1988) | Sensitivities | Sensitivities | Sensitivities | Design objectives | DELIGHT's interactive optimizer |
| JiffyTune (Conn, Coulman, Haring, Morrill, Visweswariah, Wu; IEEE TCAD 17(12), 1998) | | | Adjoint Lagrangian: "all the gradients necessary for one iteration of optimization in a single adjoint analysis" | Timing objectives | LANCELOT; tuned a microprocessor's circuits |
| Circulax (gdsfactory; Apache-2.0) | JAX autodiff through the solvers | Same | Same, and harmonic balance | Anything in JAX | JAX's |
| SpiceX (Mocz; 2026) | JAX | | | | Early, "not yet ready" |
| jolty today | Adjoint, per output | Adjoint per output and frequency, bias by difference | Direct, per parameter per step | Unknowns; every number of every device | None |

- **Where each falls short:**
    - ngspice differentiates by finite perturbation and not over time.
    - Xyce cannot differentiate a measurement, and its transient adjoint costs a backward integration per time point.
    - Commercial tools perturb and re-run for each parameter.
    - JAX simulators differentiate through the solver: their memory grows with the steps, their device libraries are small, and their validation against SPICE is unknown.
    - DELIGHT.SPICE and JiffyTune proved the method decades ago, in tools that did not survive.
- **What none does:** gradients of every requirement's slack, through crossing times and bandwidths, at its worst corner, from exact device derivatives, with the same bits on any thread count; one backward pass per scalar loss over a whole transient; and an optimizer that knows the E series and a catalog of parts and says when no value of the named parameters can meet a spec.

## 4. Gaps and opportunities

"Lead" marks an item where jolty would go beyond every reference rather than catch up.
Sizes: XS under a day, S days, M a week or two, L more.

| # | Item | Impact on the goal | Evidence | Size | Done when | Lead |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Count every simulation at the engine, whatever the front end (command line, MCP, Python), into the session's log | Item 8's "simulations" column measures both arms alike; today the jolty arm's MCP and API runs go uncounted | §2.6; `run.py:50`, `260`; `compare.py:149–157` | S | A stub session that simulates through MCP and through the API logs each analysis | |
| 2 | Item 8's run, analysed pooled, with sessions added where success is middling | The first measure of the pillar; 5 against 5 resolves only about 0.6 per task | §2.6, computed | Owner's budget | The report gives each arm's success with its interval, pooled and per task | |
| 3 | Ablation arms: jolty without images, without rules, through the command line only, without the reference in the session; later with gradients and an optimizer | Says which features earn their keep, as SWE-agent's ablations did | §3.5 | M, paid | Each feature's effect on success and tries, with its interval | Lead: no published agent study compares toolchains or ablates a simulator's features |
| 4 | Tasks held out of every public source; web use logged and reported | Success that is not recall; four tasks are Koza's (2003), one is AnalogCoder's (public) | §2.6; `README.md:73` | M | A set never published, scored as the others; each session's web fetches counted | |
| 5 | Quality and cost beside pass or fail: worst slack, parts, cost, tokens and dollars per success | "Better circuits" measured, not only "working" ones | `benchmarks/verdict.py`; `transcript.py` | S | The report states them per arm | |
| 6 | An agent's command timed end to end, through the command line and MCP (`speed.md` S5) | The loop's cost known; the guide's "seconds" checked | §2.3 (7) | S | The speed suite reports both on `bjt_amp` | |
| 7 | The derivative of each requirement's slack with respect to named parameters, at its worst condition; worst case computed from them | Every verdict says which way each knob moves it; worst case costs one adjoint per requirement, not one run per parameter | §2.4; `simulation.md:239`; Xyce lacks it (§3.2) | M | Each derivative agrees with a central difference; a report lists each failing requirement's largest derivatives | Lead |
| 8 | Sensitivities over parameters named by pattern | Makes 7, 11 and 12 affordable: the transient's direct method costs a solve per parameter per step | `sensitivity.rs:319–338`, `624–671` | S | `Sensitivity(…, parameters=["X1/R*"])` differentiates only those | |
| 9 | Each transistor's region and margins at the operating point (vgs − vth, vds − vdsat; a BJT's vce over saturation), in the report and as data | The first fact an analog designer reads; the published agents' own feedback | §2.3 (3); AnalogCoder; AnalogPortBench | S | The operating-point report lists each transistor's region; a test per region | |
| 10 | Each transient measurement's numerical error stated, from the step's error estimate, or by running again at half the step on request | "A number in a message is exact, or says how it was estimated"; the VCO found 0.37% only by running again | `discovery/vco/findings.md:62`; AGENTS.md | M | A frequency or delay carries its error estimate, checked against a run at a tenth of the step | Lead |
| 11 | An optimization in the plan: parameters with bounds or catalog values; the objective the worst slack over requirements and corners, or a measurement's target; gradient-based and deterministic; each iterate reported as facts; the best reachable slack stated when it is negative | Turns hand-written bisections into one call; states infeasibility, which agents seldom detect | §2.2; §3.4 (AnalogPortBench); ADE's optimizers, JiffyTune, DELIGHT.SPICE | L | The bandgap's flattening and the cost-down's resistors found in one call each, with fewer solves than the hand-written loops | Lead |
| 12 | One transposed solve per scalar loss at DC and over frequency (the blueprint's §4.2) | A loss's gradient for one solve more | `sensitivity.rs:374–423`; blueprint §4.2 | S–M | The gradient agrees with the per-output adjoint at the cost of one solve | |
| 13 | A discrete adjoint through the transient, with binomial checkpointing (the blueprint's §4.3) | Gradients through time for any number of parameters; Xyce integrates backward once per objective time point | Blueprint §4.3; `Xyce_UG_ch09.tex:1815–1820` | L | The gradient of an integrated loss agrees with the direct method; its cost does not grow with the parameters | Lead |
| 14 | Bridges to JAX and PyTorch (the blueprint's §4.4) | Circuits inside a learning loop | Blueprint §4.4 | M, after 12–13 | `jax.grad` through an operating point and a transient agrees with 12–13 | |
| 15 | The reference inside the session: MCP instructions derived from the code (every device, part, analysis and measurement), the guide and examples shipped with the package and served as MCP resources or by `jolty docs`; `connect.md`'s table fixed | Fewer turns spent reading source; the literature's context gains | §2.3 (1); `mcp.py:36–110`; `connect.md:51–59` | S–M | A test fails when a public constructor is missing from the instructions; the guide is readable offline in item 8's sandbox | |
| 16 | A warm runner: runner processes started ahead of time, each having imported jolty and waiting for one job, then exiting, so every script still runs in a process of its own and no fork follows the engine's or numpy's threads; plus an exact cache of results keyed by the built circuit, plan and options (`speed.md` S5, M5) | Commands in tens of milliseconds, still one process per script (estimate) | `runner.py:3–4`; `research/speed.md` §3.10 | M | A small simulation returns in under 100 ms warm with the cold run's bits | Lead |
| 17 | Session queries: read any measurement, value or unknown of a saved result without simulating again; run again only what changed when only numbers change, from the shared elaboration and the last operating point | Fewer simulations per question | `simulate.py:204–295`; no command reads measurements of saved results (`cli.py:173–211`) | M | A measurement of a results file through MCP and the command line; a run with changed numbers elaborates nothing | |
| 18 | An image budget: images chosen per call (none, a summary, all) and each one's visual-token cost stated; the default set by item 3's ablation | Context kept for reasoning; about 2,800 visual tokens per simulation today (estimate) | §2.3 (6); `plots.py:304–307` | S | The report states each image's cost; the default measured | |
| 19 | Progress for long runs over MCP (progress notifications, or tasks) | An agent waiting on a Monte Carlo sees it advance and can stop it | §3.5; `mcp.py` | S | A long Monte Carlo reports samples done | |
| 20 | Autonomous oscillators: the periodic steady state with the period as an unknown, giving frequency and amplitude directly, with their sensitivities | The class of circuits where agents fail most | §2.3 (4); ngspice's `PSS` (`psssetp.c:26–59`) | M–L; it is on the list of what ngspice has, so the owner decides | An oscillator's frequency and amplitude come from one analysis and agree with a long transient | Lead with sensitivities |
| 21 | A data-only path: simulate a saved circuit (jolty's format or SPICE) with a plan and specs given as data, running no code | jolty can be offered as a service; nothing an agent sends executes | `service.py:212–213`; `plan.py` `to_json` | M | `jolty simulate` takes a circuit file and a plan file and gives the same report as the script | |
| 22 | The runner optionally confined (`benchmarks/sandbox.py` moved into jolty as a setting), with no network for scripts; tested on Linux | Unattended agents run scripts safely | §2.5; `agentic.md` gaps | M | A script's write outside its directory fails under the setting, on macOS and Linux CI | |
| 23 | Provenance in every results file: jolty's version and build, the platform, and hashes of the circuit and plan, as shards carry | An agent's claim, or a scorer's verdict, can be checked and replayed | `shards.py:43–60`; `files.py:231` | S | `load_results` returns the provenance; a merge-style check refuses mismatched files | |
| 24 | Tensor ports and nets, instance arrays and scoped supplies that stay explicit in the circuit (`agentic.md` item 3; blueprint §1.2, §3.4) | Large regular circuits in a few lines, so fewer wiring errors; the blueprint's "3× less wiring code" is its own unmeasured estimate | Blueprint §1.2; `research/speed.md` §3.11; the blueprint's example names `Role.POWER`, jolty's role is `Role.SUPPLY` (`ir.py:31–38`) | L | A crossbar of a million cells in a few lines; the checks still see every supply | |

## 5. The long view

In two to five years jolty could be the design compiler agents call first, beyond every reference:

- **Every try answers the next question.**
  A report states what is true (verdicts, slacks, regions, rules) and which way to move: the derivative of every slack with respect to the parameters the agent names, at the condition where it is worst.
  It states how sure each number is, from its numerical error, and whether any value of the named parameters can meet a spec.
  The literature's agents stall after a few tries (AnalogCoder's limit of three; SPICEAssistant's plateau after about five).
  Facts of this kind are what breaks a stall short of a new topology.
- **The agent chooses, jolty computes.**
  The agent picks topologies and objectives, which LLMs do well.
  jolty sizes by exact gradients over corners and yield, snapped to orderable parts, which numerics do well.
  The published systems reach this split with Bayesian optimization and no gradients.
  jolty would do it with one backward pass per loss and the same bits on every run, so with far fewer simulations (estimate, from the adjoint's cost against one run per parameter).
- **The loop costs nothing an agent notices.**
  A warm, cached and incremental session answers small runs in tens of milliseconds.
  Long runs report progress, and huge regular circuits take a few lines.
- **A service anyone can call.**
  The circuit and plan go in as data and run in a sandbox, and results carry provenance, so a hosted jolty is safe to share and every claim can be replayed.
- **The measure becomes the standard.**
  Item 8 grows into a public benchmark of agentic circuit design: the same agent with different tools, held-out tasks, continuous quality, cost per success.
  It would be to circuits what SWE-bench is to code, and it would measure jolty's lead rather than assert it.
- **An environment for training design agents** (a possibility): verdicts are verifiable rewards and slacks dense ones, and determinism makes every reward reproducible.
  No reference simulator is built for that.

## 6. Principles to propose

For `AGENTS.md`, if the owner agrees:

1. **Agents' gains are measured, not assumed.**
   A change made for agents states the measurement that shows it helps (success and tries at equal cost, from item 8's harness or an ablation of it), and is called unmeasured until then.
   The engine counts every analysis it runs, whatever the front end, so tries are counted alike in every arm.
1. **Warm is cold.**
   Whatever a command reuses from an earlier one (a warm process, a cache, a session's state) gives the same bits and the same report as a fresh run.
   This extends "Determinism" to the speed-ups of items 16 and 17.
1. **Every response has a budget.**
   Each output an agent reads is bounded by a stated limit by default, as `listed_unknowns` and `max_diagnostics` already bound theirs; images included.
   The complete data go to a file or come on request.
1. **The reference ships with the tool.**
   What an agent needs to use jolty can be read offline from jolty itself.
   The MCP instructions derive from the code, as the tools' arguments derive from the request models.
1. **A computation the agent asks for is a fact.**
   The derivative of a requirement, and the optimum over parameters the agent names, are facts jolty states.
   jolty never chooses an objective, edits a script or proposes a topology.
   This makes explicit how "Feedback is facts" applies to items 7 and 11.
1. **Every command can take data, not code.**
   A command that runs a script also accepts the circuit and plan as data, so jolty can serve agents without executing what they send.

## 7. Sources

**jolty** (the repository, `main` at `d10fc4d`):

- `AGENTS.md`; `README.md`; `docs/agents/connect.md`, `why-a-compiler.md`; `docs/get-started.md`; `docs/guide/simulation.md`, `specs.md`, `checks.md`, `parts.md`; `docs/reference/rules.md`; `docs/discovery.md`.
- `src/jolty/service.py`, `mcp.py`, `cli.py`, `runner.py`, `settings.py`, `plan.py`, `simulate.py`, `plots.py`, `report.py`, `files.py`, `shards.py`, `ir.py`, `devices.py`, `specs/_requirements.py`, `specs/_measurements.py`.
- `crates/jolty-engine/src/sensitivity.rs`.
- `discovery/*/prompt.md`, `candidates.py`, `explore.py`, `findings.md`; commits `ae22268`, `a5722f1`, `c2be2ef`.
- `benchmarks/README.md`, `arms.py`, `ngspice.toml`, `run.py`, `compare.py`, `counted.py`, `transcript.py`, `cost_down/task.md`, `analogcoder/task.md`.

**Notes** (these plans): `agentic.md`, `vision.md`, the session log, not kept, `benchmarks.md`, `accuracy.md`, `research/speed.md`, a handoff note, not kept, an archived note, not kept, an archived note, not kept §1.2, §3.4, §4.

**ngspice 47** (scratchpad): `src/include/ngspice/sharedspice.h`; `src/spicelib/analysis/cktsens.c`, `dctran.c`, `psssetp.c`, `dcpss.c`; `src/maths/ni/niiter.c`.

**Xyce** (scratchpad): `utils/XyceCInterface/README`, `xyce_interface.py.cmake`, `XyceRest.py`; `utils/PythonModelInterface/`; `src/DakotaLinkPKG/`; `src/AnalysisPKG/N_ANP_ROL_*`, `N_ANP_Transient.C`; `cmake/tps.cmake`; `doc/Users_Guide/Xyce_UG_ch09.tex`, `sampling.tex`.

**Papers:**

- Y. Lai, S. Lee, G. Chen, S. Poddar, M. Hu, D. Z. Pan, P. Luo, "AnalogCoder: Analog Circuit Design via Training-Free Code Generation", AAAI 2025, 39(1):379–387; arXiv 2405.14918.
- Y. Lai, S. Poddar, S. Lee, G. Chen, M. Hu, B. Yu, P. Luo, D. Z. Pan, "AnalogCoder-Pro", arXiv 2508.02518, 2025.
- D. Vungarala, M. Alam, A. Ghosh, S. Angizi, "SPICEPilot: Navigating SPICE Code Generation and Simulation with AI Guidance", arXiv 2410.20553, 2024.
- S. Nau, J. Krummenauer, A. Zimmermann, "SPICEAssistant: LLM using SPICE Simulation Tools for Schematic Design of Switched-Mode Power Supplies", arXiv 2507.10639, 2025.
- S. Rajabi, A. Moussa, V. Saxena, C. Yang, S. Patnaik, "AnalogPortBench: Benchmarking LLMs on Analog Circuit Porting", arXiv 2610.33088, 2026.
- Y. Yin, Y. Wang, B. Xu, P. Li, "ADO-LLM: Analog Design Bayesian Optimization with In-Context Learning of Large Language Models", arXiv 2406.18770, 2024.
- C. Liu et al., "AmpAgent: An LLM-based Multi-Agent System for Multi-stage Amplifier Schematic Design from Literature for Process and Performance Porting", arXiv 2409.14739, 2024 (numbers from summaries, not the full text).
- "LEDRO: LLM-Enhanced Design Space Reduction and Optimization for Analog Circuits", IBM, arXiv 2411.12930, 2024.
- "EEsizer", arXiv 2504.11497 and 2509.25510, 2025.
- "AnalogSAGE", arXiv 2512.22435, 2025.
- X. Yu, D. Torbunov, S. Mandal, Y. Ren, "AutoSizer", ICML 2026 (poster page).
- J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, O. Press, "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering", arXiv 2405.15793, 2024.
- W. Nye, D. C. Riley, A. Sangiovanni-Vincentelli, A. L. Tits, "DELIGHT.SPICE: An Optimization-Based System for the Design of Integrated Circuits", IEEE TCAD 7(4):501–519, 1988.
- A. R. Conn, P. K. Coulman, R. A. Haring, G. L. Morrill, C. Visweswariah, C. W. Wu, "JiffyTune: Circuit Optimization Using Time-Domain Sensitivities", IEEE TCAD 17(12):1292–1309, 1998.
- D. E. Hocevar, P. Yang, T. N. Trick, B. D. Epler, "Transient Sensitivity Computation for MOSFET Circuits", IEEE TCAD 4(4), 1985; Y. Cao, S. Li, L. Petzold, R. Serban, SIAM J. Sci.
    Comput.
    24(3), 2003; A. Griewank, A. Walther, "Algorithm 799: Revolve", ACM TOMS 26(1), 2000 (as the blueprint cites them).

**Web pages** (fetched or searched 15 November 2026):

- Anthropic, "Writing effective tools for agents", anthropic.com/engineering/writing-tools-for-agents; Claude vision documentation, platform.claude.com/docs/en/build-with-claude/vision ("Resolution and token cost"); Anthropic sandbox-runtime, github.com/anthropic-experimental/sandbox-runtime.
- Model Context Protocol, changelog of 2025-11-25, modelcontextprotocol.io/specification/2025-11-25/changelog.
- PySpice and InSpice on PyPI; spicelib on PyPI and readthedocs; the ngspice and LTspice MCP servers' pages (mcp.so, glama.ai, lobehub.com, hackaday.io).
- Cadence: the Cerebrus press release (Business Wire, 22 July 2021); Digital Engineering 24/7 on Cerebrus AI Studio (8 May 2025); the Virtuoso Studio launch (April 2023) and MediaTek's adoption (Business Wire, January 2025); the ViraStack AI Super Agent pages and CadenceLIVE 2026 coverage (Futurum); the Cadence blog "Keeping your circuit in tune" and forum threads on ADE sensitivity.
    Cadence's own pages returned HTTP 403; their content comes from search excerpts.
- Synopsys: DSO.ai and ASO.ai pages; the Keysight and Synopsys RF migration flow (June 2025); Synopsys news of 11 March 2026 (AgentEngineer) and 26 July 2026 (AMS agentic workflow); Tom's Hardware on Synopsys' agentic platform.
- Siemens: Solido Design Environment product page and blogs (October 2024, March 2026); Tech Design Forums (2023).
- HSPICE: Star-HSPICE manual (1998) chapter on optimization; arXiv 2301.10157 on HSPICE's optimization routines.
- Circulax (github.com/gdsfactory/circulax, PyPI); SpiceX (PyPI, readthedocs).
- Xyce user forum threads on `.SENS` (groups.google.com/g/xyce-users); Sandia's Xyce and Dakota material (osti.gov).
