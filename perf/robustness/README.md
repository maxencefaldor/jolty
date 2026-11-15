# Robustness

jolty aims to find the solution of every circuit that has one.
This corpus measures how far it is from that, and how far ngspice is: hard circuits, each simulator's outcome on each, classified alike, and each simulator's failure rate.

## Running

```sh
scripts/bench --robustness                          # every case, in every simulator installed
scripts/bench --robustness gm1/tran vreg/dc         # only these cases
scripts/bench --robustness --simulator jolty        # only this simulator
scripts/bench --robustness --timeout 60             # stop each run after 60 s rather than 300 s
```

Each case runs in each simulator in a process of its own, which the timeout stops.
The command prints each outcome as it comes, then the table of cases by simulator, each simulator's failure rates, and each outcome's detail, and writes them to `results/<commit>-robustness.json` beside the machine, each simulator's version and the timeout, unless `--output` names another file.
It prints the load average as each case began; the corpus measures outcomes, and the times beside them are what one run took on a machine that may be shared.
The results file stays out of the repository, as `.gitignore` says: jolty's messages quote the lines of the decks they are about, and the GPL's decks are never copied into it.

## Cases

A case is one analysis of one circuit: its operating point, a DC sweep or a transient.
Each circuit gives a case for its operating point, the problem every other analysis starts from, and one for each further analysis it asks for.

### jolty's own

`own.py` writes them with jolty's builder, and ngspice runs the netlist jolty's exporter writes of each.

| Circuit | What makes it hard | Transient, and how it is judged |
| --- | --- | --- |
| `schmitt_bjt` | An emitter-coupled Schmitt trigger of two npn transistors, its input between its thresholds: three DC solutions | Driven above its upper threshold and back into its band: it must stay on the branch it came down |
| `schmitt_cmos` | A Schmitt trigger of six level 1 MOSFETs, its input between its thresholds: three DC solutions | The same |
| `cmos_latch` | Two cross-coupled CMOS inverters: two stored states and the balance between them | `power_up`: from rest, its supply ramped on in 1 ns, it must fall into a stored state |
| `sram_cell` | The example's six-transistor cell, its word line low: two stored states and a balance | The example's writes of a 1 then a 0: it must hold the 0 |
| `ring` | An odd ring of five CMOS inverters, its first stage loaded twice as much: one DC solution, unstable | `from_rest`: its supply ramped on in 1 ns with no kick, only its asymmetry starts it, and it must oscillate |
| `folded_sources` | A random circuit of 47 devices whose operating point source stepping folds short of (`crates/jolty-engine/tests/data/folded_sources.json`) | |
| `bare_chain` | Three CMOS inverters of MOSFETs with no capacitance at any net | A pulse of 100 ps edges: it must end at the input's low level, inverted three times |
| `impasse` | | A capacitor charged until the circuit's equations fold, after Chua and Deng: its solution ends at 5.47 µs, and a run must stop there |
| `buck` | The benchmarks' synchronous buck converter, its body diodes at rest | Its 2 ms: four gate edges a period, a body diode turning on at each; no reference, so it is judged by its completion |
| `memristive_nor` | The memristive NOR discovery's four gates at 10 % above its design's pulse | Its 13 ns, whose memristors' states race to their bounds; no reference |

### Third-party decks

`decks.py` reads `../external.json`, a manifest of decks by address, commit, SHA-256 digest and licence, and fetches each into jolty's cache, outside the repository, checking its digest: the decks are never copied into the repository.
They are CircuitSim90's ten decks of level 3 MOSFETs and its bipolar voltage regulator, and four of its bipolar circuits that Xyce's regression suite keeps apart, all at that suite's commit `7bb7e98`, under the GPL, version 3 or later.

ngspice runs each deck itself; jolty reads it through its SPICE reader, the deck's elements wrapped in a subcircuit of their own beside its definitions, each line kept at its place so that every instance records the deck's own line.
A deck jolty's reader cannot read, or one that sets initial conditions or a nodeset, which jolty's plan cannot state, is unsupported in jolty, with the reason.
A deck's transient has no reference yet, so it is judged by its completion.

A deck's defects are fixed by recorded rules, alike for every simulator, and each case lists those its deck needed:

| Rule | What it does |
| --- | --- |
| `xyce-options` | Xyce's `.options` packages, such as `.options device temp=100` and `.options timeint method=gear reltol=1e-3`, are SPICE's `.options` without the package's name. |
| `output-requests` | Output requests, `.print`, `.plot`, `.probe`, `.save` and `.width`, are taken out: every simulator keeps every node, and Xyce's `.print` writes expressions in braces that SPICE does not read. |

Every simulator runs a deck at its temperature, with the options it sets that both have, its tolerances, iteration limits and integration method, and with SPICE's defaults for the rest.
Before it runs, jolty's checks state what they find in each circuit, as facts, and the results keep them.

## Outcomes

Each run's outcome is one of these:

| Outcome | Meaning |
| --- | --- |
| `right` | Solved, and it agrees with the reference. |
| `other` | Solved, and it agrees with no solution of the reference. |
| `solved` | Solved, where the case has no reference to judge it against. |
| `failed` | The simulator gave up, with its message. |
| `timed out` | It ran past the timeout. |
| `unsupported` | It cannot read or simulate what the case holds, with the reason; never counted as a failure. |

The reference of an operating point is every DC solution jolty finds by deflation (`OperatingPoints`) at the differential tests' tight tolerances, a relative tolerance of 1e-6; of a sweep, every one at each level.
A transient that settles is judged by the DC solutions of its circuit with each source at its last level: it must end at the one its case expects, such as the branch a Schmitt trigger came down, or any stored state of a latch.
A ring must end at none of them, as each is unstable; the impasse circuit must stop where its closed form says its solution ends, within the relative tolerance of that time.
Two results agree as the differential tests judge two engines: each voltage within the case's relative tolerance of the largest voltage of the reference, plus twice its absolute tolerance on voltages.

Each simulator's failure rate is its failed and timed-out runs over the runs it supports, DC and transient apart, with the 95 % Wilson interval that `jolty.specs.wilson` gives a yield.

## Simulators

A simulator is a named configuration of one program, its version recorded in every results file:

| Simulator | What runs |
| --- | --- |
| `jolty` | The engine at its defaults, each case in a Python process of its own; it counts a transient's Newton iterations, and not yet an operating point's. |
| `ngspice-sparse` | ngspice in batch mode at its defaults, its direct solver Sparse 1.3, at the temperature where its thermal voltage is jolty's, as the differential tests run it; its iterations and analysis time from `rusage all`. |

Another simulator joins the corpus by implementing `corpus.Simulator`, a name, its version and a way to run a case within a timeout returning the voltages the case compares, and by being listed in `simulators.every`.
