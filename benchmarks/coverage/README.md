# Coverage

jolty aims to be the most complete simulator.
What each simulator offers is known from its documents; this matrix checks it by running.
Each row is a capability a user can ask for, and each cell is the fixture of that row run in that simulator, or, for a simulator that cannot run here, its document's word.
[`matrix.md`](matrix.md) is the matrix, rendered from [`coverage.json`](coverage.json).

## Running

```sh
scripts/bench --coverage                                # every row, in jolty and ngspice, Sparse and KLU
scripts/bench --coverage op ac                          # only these rows
scripts/bench --coverage --simulator xyce-serial         # Xyce's column, where Xyce is installed
```

A run replaces the cells of the rows and simulators it runs, keeps the others, judges every cell again, and renders the matrix, so a `coverage.json` with Xyce's cells copied from where Xyce runs is rendered by the next run of any row.
Each simulator's cells record its version, the commit, the machine and its load as they ran.
The fixtures are not timed.

## Fixtures

`capabilities.py` holds the rows, each with its fixture: the smallest circuit and plan that exercise the row, and the numbers it reads, named alike in every simulator, such as `v(out)` at a frequency.
A fixture of jolty's runs through its builder, or through its SPICE reader on the same deck the SPICE simulators run.
Where jolty's reader takes the deck, the deck is the circuit; where it does not, the deck is jolty's export of the circuit its builder made, as the differential tests write it.
A row that ngspice has and jolty does not gives jolty's reader its deck, and the reader's refusal is jolty's cell.
Every simulator runs at the temperature where its own physical constants give jolty's thermal voltage at 27 °C, as `jolty.spice.ngspice` runs ngspice; a fixture whose answer is a law of the temperature alone, such as a resistor's drift, runs Xyce at the temperature itself.

The known answer is a closed form wherever one exists.
For a model another implementation defines, such as ngspice's VDMOS and Berkeley's BSIM3 as ngspice runs it, the answer is that implementation's run, its cell `reference`, as `AGENTS.md` says a SPICE3 model's reference is ngspice.
A row with no answer yet, such as BSIM4's until its QA results are fetched, is `solved` where it runs.

## Cells

| Status | Meaning |
| --- | --- |
| `agreed` | It ran, and every number lies within the row's tolerance of the known answer. |
| `disagreed` | It ran, and a number lies outside it; the evidence gives the number, the answer and the error. |
| `failed` | It took the request and did not finish, with its message. |
| `unsupported` | It refused the request, with its message, or has no statement that asks for it, with the section of its manual that lists what it has. |
| `solved` | It ran, and the row has no known answer yet. |
| `reference` | It is the implementation that defines the row's model. |
| `documented` | Its manual offers the row, by the section cited, but no fixture runs it here: a simulator not installed, or a row whose fixture is to come. |
| `unknown` | No document read says. |

A number agrees when it lies within the row's tolerance of the largest known number of its kind in the row, the kind its name begins with, such as `v` or `i`.
The tolerance is 1e-6 for solved values, the relative tolerance every simulator runs with, within which the differential tests compare operating points, sweeps and responses, and 1e-3 for transients, the differential tests' tolerance on them.
A simulator's message says whether it refused or failed: `matrix.py` lists the words of each simulator's refusals.

## Simulators

| Simulator | What runs |
| --- | --- |
| `jolty` | The engine, at the differential tests' tight tolerances. |
| `ngspice-sparse` | ngspice 47 in batch mode, Sparse 1.3, through `jolty.spice.ngspice`. |
| `ngspice-klu` | The same with `.options klu`. |
| `xyce-serial` | Xyce 7.10 with each fixture's statements and a `.print` in CSV, run where it is installed, on `dogwood`. |
| LTspice, Spectre, HSPICE, AFS | Their documents, as `documented.py` cites them. |

Each simulator that runs here runs by its runner of [`../simulators.py`](../simulators.py).

## No fixture yet

These rows of the design (`research/benchmarks.md` §4.1.2 of the lead's notes) have no fixture yet: corners and worst case, which jolty runs through a plan's script; ngspice's `.disto`, `.sp`, transient noise, `.four`, initial conditions and nodesets; CIDER and device bypass; switches and logic; the foundries' kits; and Xyce's harmonic balance, AC and transient sensitivities and sampling, cited as `documented`.
