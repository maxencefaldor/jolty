# Coverage

jolty aims to be the most complete circuit simulator.
Documentation claims what each simulator supports, and the coverage matrix verifies those claims by executing live fixtures.
Each row represents a user-facing capability, and each cell records the fixture outcome for a simulator or cites manual documentation when the simulator cannot run locally.
[`matrix.md`](matrix.md) renders the coverage matrix from [`coverage.json`](coverage.json).

## Running

```sh
scripts/bench --coverage                                # every row, in jolty and ngspice (Sparse and KLU)
scripts/bench --coverage op ac                          # selected rows only
scripts/bench --coverage --simulator xyce-serial         # Xyce column, on hosts where Xyce is installed
```

A coverage run updates only the cells for the requested rows and simulators while preserving all other recorded cells.
The runner then rejudges every cell and regenerates the matrix, so copying Xyce cells from another host into `coverage.json` updates `matrix.md` on the next run.
Each simulator entry records the binary version, repository commit, host machine, and system load at execution time.
Coverage fixtures are not timed.

## Fixtures

`capabilities.py` defines the capability rows, each with a minimal circuit, analysis plan, and uniformly named output measurements such as `v(out)` at a target frequency.
Each jolty fixture runs either through the Python circuit builder or through the SPICE reader on the same deck executed by SPICE simulators.
When the SPICE reader supports a deck directly, that deck defines the circuit; otherwise, the deck comes from jolty's SPICE exporter, matching the differential test suite.
When a row covers an ngspice feature absent from jolty, the runner passes the deck to jolty's SPICE reader and records the resulting refusal.
Each simulator runs at the adjusted temperature where physical constants match jolty's thermal voltage at 27 °C, matching `jolty.spice.ngspice`.
Fixtures whose reference answer depends purely on temperature, such as resistor thermal drift, run Xyce at the nominal temperature directly.

Reference answers use analytical closed forms whenever available.
For externally defined models such as ngspice VDMOS and Berkeley BSIM3 in ngspice, the defining implementation serves as the `reference` cell, following `AGENTS.md`.
Rows without a reference answer yet, such as BSIM4 pending QA reference data, report `solved` when simulation completes.

## Cells

| Status | Meaning |
| --- | --- |
| `agreed` | Simulation completed, and every measured value lies within the row tolerance of the known answer. |
| `disagreed` | Simulation completed, and at least one value exceeds the tolerance; the evidence records the measured value, expected answer, and error. |
| `failed` | The simulator accepted the deck but did not finish; the evidence quotes the simulator error message. |
| `unsupported` | The simulator rejected the request or lacks a corresponding statement; the evidence quotes the refusal message or manual section. |
| `solved` | Simulation completed, and the row has no known reference answer yet. |
| `reference` | The simulator provides the defining reference implementation for the row model. |
| `documented` | The cited manual section documents support, without an executed local fixture because the simulator is unavailable or the fixture is pending. |
| `unknown` | No consulted documentation specifies whether the simulator supports the feature. |

A measured value agrees when the error falls within the row tolerance scaled by the largest known magnitude of the same quantity kind (such as `v` or `i`).
The relative tolerance is 1e-6 for operating points, sweeps, and small-signal responses, and 1e-3 for transient analyses, matching the differential test suite.
`matrix.py` classifies simulator diagnostics as refusals or failures by matching per-simulator refusal patterns against output messages.

## Simulators

| Simulator | What runs |
| --- | --- |
| `jolty` | Engine evaluated at the differential test suite's tight tolerances. |
| `ngspice-sparse` | ngspice 47 in batch mode with Sparse 1.3 via `jolty.spice.ngspice`. |
| `ngspice-klu` | ngspice 47 in batch mode with `.options klu`. |
| `xyce-serial` | Xyce 7.10 with fixture statements and CSV `.print` output, executed on `dogwood` where Xyce is installed. |
| LTspice, Spectre, HSPICE, AFS | Manual documentation cited in `documented.py`. |

Each locally executed simulator runs through [`../simulators.py`](../simulators.py).

## No fixture yet

Several design rows from `plans/research/benchmarks.md` §4.1.2 do not have executed fixtures yet, starting with CIDER and device bypass.
The external open PDK archive rows (`kit-sky130`, `kit-gf180mcu`, `kit-sg13g2`) remain `documented` offline so unit tests avoid network downloads; `tests/test_kits.py` verifies them online, while `pdk-library` runs a local synthetic PDK offline.
Xyce rows for `.HB`, `.LIN`, `.STEP`, `.FOUR`, `.IC`, `.NODESET`, `VSWITCH`/`ISWITCH`, digital `U` devices, AC and transient sensitivities, and `.SAMPLING` also remain `documented`.
