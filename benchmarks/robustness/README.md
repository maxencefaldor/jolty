# Robustness

jolty aims to find the solution of every solvable circuit.
The robustness corpus tracks progress toward that goal alongside ngspice by running hard circuits, classifying every simulator outcome uniformly, and reporting per-simulator failure rates.

## Running

```sh
scripts/bench --robustness                          # every fixture, in every installed simulator
scripts/bench --robustness gm1/tran vreg/dc         # selected fixtures only
scripts/bench --robustness --simulator jolty        # one simulator only
scripts/bench --robustness --timeout 60             # stop each run after 60 s instead of 300 s
```

Each fixture executes inside a dedicated simulator subprocess enforced by the timeout.
The command streams each outcome as it finishes, then prints the fixture-by-simulator table, failure rates, and diagnostic details.
Results are saved to `results/<commit>.json` (or `--output`) with provenance metadata covering commit, host machine, simulator versions, and timeout.
The runner logs system load average at the start of each fixture; the corpus measures convergence outcomes rather than timing, so reported runtimes reflect single runs on a potentially shared host.
Results stay outside the repository per `.gitignore`, because jolty diagnostics quote deck lines and GPL decks are never copied into the repository.

## Fixtures

Each fixture tests one analysis of one circuit: an operating point, a DC sweep, or a transient.
Every circuit provides an operating-point fixture (the starting problem for all other analyses) plus one fixture per additional analysis.

### Built-in fixtures

`own.py` constructs these circuits with jolty's builder, and ngspice runs the netlists produced by jolty's SPICE exporter.

| Circuit | What makes it hard | Transient, and how it is judged |
| --- | --- | --- |
| `schmitt_bjt` | Emitter-coupled Schmitt trigger of two NPN transistors biased between thresholds (three DC solutions) | Driven above the upper threshold and back into hysteresis; must stay on the descending branch |
| `schmitt_cmos` | Schmitt trigger of six level-1 MOSFETs biased between thresholds (three DC solutions) | Same as `schmitt_bjt` |
| `cmos_latch` | Two cross-coupled CMOS inverters with two stored states and one unstable balance | `power_up`: supply ramps on in 1 ns from rest; must settle into a stored state |
| `sram_cell` | Six-transistor SRAM cell with word line low (two stored states and one balance) | Writes a 1 then a 0; must hold the 0 |
| `ring` | Odd ring of five CMOS inverters with double load on stage one (one unstable DC solution) | `from_rest`: supply ramps on in 1 ns without a kick; asymmetry must start oscillation |
| `folded_sources` | Random 47-device circuit whose operating point defeats source stepping (`crates/jolty-engine/tests/data/folded_sources.json`) | |
| `bare_chain` | Three CMOS inverters of zero-capacitance MOSFETs | Pulse with 100 ps edges; must settle to the triply inverted low input level |
| `impasse` | | Capacitor charged until circuit equations fold (after Chua and Deng); solution ends at 5.47 µs and simulation must halt there |
| `buck` | Benchmark synchronous buck converter with body diodes at rest | 2 ms transient with four gate edges per period turning on a body diode; no reference, judged by completion |
| `memristive_nor` | Four gates from the memristive NOR discovery driven 10 % above the design pulse | 13 ns transient where memristor states race to their bounds; no reference |

### Third-party decks

`decks.py` reads `../external.json` (listing URL, commit, SHA-256 digest, and licence), downloads each deck into jolty's external cache, and verifies every digest without copying decks into the repository.
The external decks include CircuitSim90's ten level-3 MOSFET decks, the bipolar voltage regulator, and four CircuitSim90 bipolar circuits held separately in Xyce's regression suite (commit `7bb7e98`, GPLv3 or later).

ngspice runs each deck directly, while jolty loads each deck through the SPICE reader by wrapping top-level elements in a dedicated subcircuit alongside deck definitions and preserving original line numbers on every instance.
Initial conditions and nodesets parsed from the deck initialize jolty's transient and operating-point analyses, matching ngspice.
Decks that jolty's SPICE reader cannot parse are recorded as unsupported with the diagnostic reason.
Third-party transients have no reference solution yet and are judged by completion.

Recorded normalization rules repair deck defects identically for every simulator, and each fixture lists every rule applied:

| Rule | What it does |
| --- | --- |
| `xyce-options` | Converts Xyce `.options` packages (such as `.options device temp=100` and `.options timeint method=gear reltol=1e-3`) to plain SPICE `.options` without the package name. |
| `output-requests` | Removes `.print`, `.plot`, `.probe`, `.save`, and `.width` lines because every simulator records all nodes and Xyce brace expressions fail SPICE parsing. |
| `element-names` | Replaces `/` characters in flattened-hierarchy element names with `_`, avoiding collisions with jolty's `/` hierarchy separator. |
| `exp-defaults` | Fills zero or omitted `EXP` waveform delays and time constants using SPICE3 and ngspice 47 rules: rise delay and both time constants equal the transient step, and fall delay adds one step. |

Every simulator runs each deck at the deck temperature using shared options (tolerances, iteration limits, and integration method) and SPICE defaults for all remaining settings.
Before simulation begins, jolty's circuit checks record factual findings in the results.

## Outcomes

Every simulator run receives one of six outcome classifications:

| Outcome | Meaning |
| --- | --- |
| `agreed` | Simulation converged and matched the reference solution. |
| `disagreed` | Simulation converged but matched none of the reference solutions. |
| `solved` | Simulation converged on a fixture with no reference solution. |
| `failed` | The simulator aborted with an error message. |
| `timed out` | Execution exceeded the configured timeout. |
| `unsupported` | The simulator cannot parse or simulate the fixture (with reason); excluded from failure counts. |

Operating-point references contain every DC solution found by jolty deflation (`OperatingPoints`) at tight differential-test tolerances (`reltol = 1e-6`), and sweep references record all solutions at each sweep level.
Settling transients are judged against DC solutions with sources held at final levels, requiring convergence to the expected branch (such as a Schmitt trigger state or any stored latch state).
Ring oscillators must avoid all unstable DC solutions, and the impasse circuit must halt within the relative tolerance of the analytical fold time.
Voltage agreement follows the differential test criterion: every node voltage must stay within the fixture's relative tolerance of the largest reference voltage plus twice the absolute voltage tolerance.

Each simulator's failure rate divides failed and timed-out runs by supported runs (separately for DC and transient), reported with a 95 % Wilson interval from `jolty.specs.wilson`.

## Simulators

Each simulator entry defines a named tool configuration whose version is logged in every results file:

| Simulator | What runs |
| --- | --- |
| `jolty` | Engine at default settings running each fixture in a dedicated Python subprocess, counting Newton iterations across operating points, DC sweep steps, and transients (including initial operating points). |
| `ngspice-sparse` | Batch-mode ngspice at defaults with Sparse 1.3 at the temperature matching jolty's thermal voltage, reading iteration counts and analysis time from `rusage all`. |

Each simulator runs through [`../simulators.py`](../simulators.py), which supplies its canonical name and version.
Additional simulators integrate by adding a timeout-bounded runner function returning node voltages to `runs.py` and registering it in `runs.RUNS`.
