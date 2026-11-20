# Results

`jolty simulate` reports what each analysis found, and saves every value to a results file.

## The report

A report states facts, one section per analysis in the plan's order:

- an operating point lists every unknown value, logic net level, convergence strategy, Newton iteration count, and each transistor's [region of operation](#regions-of-operation) with boundary margins ordered from tightest margin first;
- a search for operating points lists every unknown's value at each DC solution it found, the operating point first, and the Newton iterations that found each;
- a DC sweep states the Newton iterations of all its values, and how many values each strategy but Newton iteration alone found;
- a sweep, a frequency response or a transient lists each unknown's first, lowest, highest and last value, and each logic net's first and last level and how often it changed;
- a sensitivity analysis lists, for each output, its value and its derivative with respect to each parameter, in the circuit's order, or, along another analysis, each derivative's largest and where;
- a stability analysis states the loop gain at its lowest frequency, its crossover frequency and phase margin, and its gain margin;
- a pole-zero analysis lists each pole and zero, a conjugate pair once, with its natural frequency and damping ratio;
- a noise analysis reports band-edge noise densities, output and input-referred RMS noise, and top noise contributors with RMS levels and variance shares;
- failed analyses state the failure reason.

Frequency responses report magnitudes per unit excitation (such as volts per volt for a voltage source).
Reports print up to `max_unknowns` unknowns, logic nets, operating-point transistors, and sensitivity parameters; the results file stores all of them.
With `--output-format json`, the command prints the same report as structured JSON.

## Plots

`jolty simulate` saves plots next to results as `<script>.<analysis>.plot.png` for every analysis except operating points, operating-point searches, and sensitivities, and the MCP server returns them with text output.
Transients plot against time, DC sweeps plot against swept values, and AC and stability analyses render Bode plots (marking 0 dB and -180° for margin readings).
Noise analyses plot output noise density, top contributors, and input-referred noise on log axes, while pole-zero analyses plot poles and zeros by natural frequency and damping ratio against the zero-damping instability line:

![The step response of the RC low-pass filter: the input steps to 1 V at 1 µs and the output charges toward it](../images/rc_filter.step.plot.png)

![The Bode plot of the RC low-pass filter: flat to its corner at 159 kHz, then falling 20 dB a decade as its phase falls to -90°](../images/rc_filter.bode.plot.png)

Voltages, currents, temperatures, and device states never share an axis: each physical nature gets a dedicated panel, and frequency responses split magnitude (dB) and phase into two panels.
A legend names every series, and up to four curves carry end labels:

![The half-wave rectifier over three periods: the output holds near the peak of the input, and the source delivers a pulse of current at each peak](../images/rectifier.ripple.plot.png)

Logic nets share a dedicated panel with one lane per net and hatched unknown regions (see [Logic](logic.md#levels-in-results)).

By default, plots render up to eight named top-component nets (voltages followed by logic nets).
Passing `--traces` selects custom unknowns such as `X1/mid` or `V1:current`, and `jolty plot` re-renders saved results with different traces without re-simulating:

```sh
uv run jolty plot rectifier.results.npz --analyses ripple --traces out --traces V1:current
```

[Monte Carlo](simulation.md#monte-carlo) and [variant](simulation.md#variants) plans plot nominal-condition distributions instead of individual runs: requirement histograms against bounds (`<script>.monte-carlo.plot.png` or `<script>.variant-runs.plot.png`) and min/median/max trace envelopes (`<script>.monte-carlo.<analysis>.plot.png`).

Passing `--no-plot` omits plot generation.

## Unknowns

Every result stores the circuit's unknowns, named by path:

- net voltages relative to ground, such as `out` or `X1/mid`;
- thermal-net temperature rises above ambient in kelvins, such as `junction`;
- branch currents through voltage sources, inductors, or voltage-controlled voltage sources, such as `V1:current`;
- internal device unknowns such as a diode junction voltage behind series resistance (`D1:junction`).

## What a result records

In addition to unknowns, results record devices and component instances matched by the `record` list on the plan or analysis at every solved point:

```python
analyses = {"op": OperatingPoint(), "step": Transient(stop=1 * ms)}
Plan(analyses, record=["M1", "X1"])  # the transistor M1, and the ports of X1
Plan(analyses, record=["X1/**", "!X1/bias/**"])  # inside X1 but its bias network
Plan(analyses, record=["**"])  # every instance
Transient(stop=1 * ms, record=["D1"])  # this analysis records D1, whatever the plan's
```

Each entry is an exact instance path or a glob pattern matching [fidelity](fidelity.md) scopes (`*` within one hierarchy level, `**` across levels).
Entries prefixed with `!` exclude prior matches, and the last matching pattern wins just like `.gitignore`.
The plan's `record` list applies to operating point, DC sweep, AC, and transient analyses unless overridden per analysis; omitting `record` records no devices.
Recorded devices store terminal currents and charges, small-signal matrices, and internal variables.
Recorded components and logic devices store port currents and charges summed from underlying device terminals without storing internal device records.
Requesting an unknown path or unmatched pattern fails the analysis while listing the closest valid paths.

Regardless of `record`, operating points and transients automatically record instances carrying current or power ratings for [rating checks](checks.md#on-analysis-results).
Nominal `jolty simulate` operating points also record compact [regions](#regions-of-operation) and margins for unrecorded transistors; specs, plots, schematics, and operating-point rules read unknowns alone.
Corners, Monte Carlo samples, worst-case runs, and spot checks record nothing extra beyond rating checks because only specs inspect them.

Recording consumes memory at every solved point: a recorded MOSFET stores 48 floats (114 with three gate-oxide unknowns), all bounded by `max_memory` (see [Options](simulation.md#options)).
Accessing an unrecorded device or terminal raises `KeyError` listing the closest recorded names and showing how to enable recording:

<!-- output of: python unrecorded.py -->

```text
KeyError: "the result recorded no terminal named 'R1.a'; close names: R2.a; record its instance, as an analysis's or a plan's record=['R1'] does, or every instance, as record=['**'] does"
```

Mistyping a terminal on a recorded instance reports that instance and suggests only its valid terminals (for example, `'R1', which the result recorded, has no terminal named 'R1.aa'; close names: R1.a`).

`result.recording` describes recorded metadata: `devices` (each listing `path`, `terminals`, `internals`, and `variables`) and `summed_terminals` (such as `X1.vdd`).
`terminals` lists column names for `currents` and `charges`, ordering device terminals before summed terminals.

## Terminals

Recorded terminal currents are exact: the engine evaluates each recorded device with separate terminal slots so terminals sharing a net or ground retain individual currents.
In transients, terminal currents include charge time derivatives evaluated with the active integration formula, so currents entering any net sum to its converged Newton residual.
Frequency responses store complex phasors for each recorded terminal current.

Component ports (such as `X1.vdd`) sum all underlying device terminals connected to that port net at any hierarchy depth.
Externally shorted ports still keep separate currents.
Logic pins on electrical nets (plus supply and ground pins) report currents from their connect elements: threshold-detector capacitive loads and logic-driver output stages.

## Device models

At every solved point, results expose exact small-signal matrices for each recorded device: derivatives of device equations with respect to device unknowns at the converged solution.
Device unknowns cover external terminals plus internal nodes or branch currents (such as `M1:drain` behind a series resistance or `V1:current`).
`conductance(equation, unknown)` returns the residual derivative, and `capacitance(equation, unknown)` returns the charge derivative.
Between terminals, the first argument selects terminal current and the second selects terminal voltage: `conductance("M1.drain", "M1.gate")` gives transconductance, `conductance("M1.drain", "M1.drain")` gives output conductance, and `-capacitance("M1.gate", "M1.drain")` gives gate-drain capacitance.
Internal unknown equations correspond to internal KCL nodes or voltage-source branch constraints.
Columns of `conductances` and `capacitances` store row-major entries over each recorded device's unknowns so each block reshapes directly into a square matrix.

Device variables expose internal model quantities beyond terminal flows and small-signal derivatives.
`variable(instance, name)` reads one variable, and each recorded device's `variables` attribute lists available names.

| Model | Variables |
| --- | --- |
| `mosfet` | Polarity-signed `threshold_voltage` and `saturation_voltage`; Meyer `gate_source_capacitance`, `gate_drain_capacitance`, and `gate_bulk_capacitance`; region and margins |
| `bsim3` | Polarity-signed `threshold_voltage` and `saturation_voltage`; region and margins |
| `vdmos` | Polarity-signed `threshold_voltage` and `saturation_voltage`; `gate_source_capacitance`, `gate_drain_capacitance`, body-diode `drain_source_capacitance`; region and margin |
| `jfet`, `bipolar_transistor`, `vbic` | Operating region and margins |
| `voltage_controlled_memristor`, `current_controlled_memristor` | State-determined `resistance` and state `rate` (positive toward off, in full ranges per second) |

<!-- not run: it reads a simulation of a circuit with a MOSFET `M1`, and its results, which the page does not build -->

```python
point = simulation.run(OperatingPoint(record=["M1"]))
point.conductance("M1.drain", "M1.gate")  # gm, in siemens
point.variable("M1", "threshold_voltage")  # von, with the body effect, in volts
results["step"].variable("M1", "saturation_voltage")  # one per time point
```

## Regions of operation

Each transistor reports its operating region (per its model equations) and voltage margins to region boundaries, where each margin equals zero on a boundary and flips sign across it.
The numeric `region` variable indexes into the recorded device's `regions` tuple, while `margins` names its margin variables; `region(instance)` returns the region string, and `simulation.transistors` lists all transistor paths:

<!-- not run: it reads a simulation of a circuit with a MOSFET `M1`, which the page does not build -->

```python
point = simulation.run(OperatingPoint(record=simulation.transistors))
point.region("M1")  # such as "saturated"
point.variable("M1", "saturation_margin")  # vds - vdsat, in volts
regions = point.recording.devices[0].regions  # ("off", ..., "saturated")
```

Setting `regions=True` on an analysis keeps only the region and margins (three floats instead of a full 48-float MOSFET record) for unrecorded transistors, readable via `region` and `variable`; `jolty simulate` enables `regions=True` on nominal operating points:

<!-- not run: it reads a simulation of a circuit with a MOSFET `M1`, which the page does not build -->

```python
point = simulation.run(OperatingPoint(regions=True))
point.region("M1")  # such as "saturated"
kept = point.recording.devices[0].variables  # ("region", "overdrive_voltage", ...)
```

| Model | Regions | Margins |
| --- | --- | --- |
| `mosfet` | `off` at or below its threshold, or `weak_inversion` below it at level 3 with fast surface states; then `linear` with the drain below saturation, `saturated` at or past it, as `mos1load.c` and `mos3load.c` switch | `overdrive_voltage`, `vgs - von`, and `saturation_margin`, `vds - vdsat` |
| `jfet` | `off` at or below its threshold, then `linear` with the drain within `vgs - vto`, `saturated` past it, as `jfetload.c` switches | `overdrive_voltage`, `vgs - vto`, and `saturation_margin`, `vds - max(vgs - vto, 0)` |
| `bsim3` | `weak_inversion` below its threshold, then `linear` below its saturation voltage, `saturated` at or past it | `overdrive_voltage`, `Vgs_eff - Vth`, and `saturation_margin`, `Vds - Vdsat` |
| `vdmos` | `linear` with the drain below its softened overdrive, `saturated` at or past it, as `VDMOSload` switches | `saturation_margin`, `vds` less the softened overdrive over the triode's multiplier |
| `bipolar_transistor`, `vbic` | `cut_off`, neither junction injecting, `active`, the base-emitter junction alone, `saturated`, both, and `reversed`, the base-collector junction alone | `base_emitter_voltage` and `base_collector_voltage`, its intrinsic junctions' |

Margins follow n-channel/NPN sign conventions regardless of polarity (reversing p-channel/PNP terminal voltages), so positive margins always signify on or saturated (`|vgs| - |vth|`, or `veb` for a PNP).
When an FET's drain drops below its source, the equations swap drain and source and evaluate `vds` as `vs - vd`.
A bipolar junction injects when ideal current `Is · (e^{v / (N · Vt)} - 1)` is positive (`v > 0`), so saturation begins when base-collector voltage turns positive (`vce < vbe`).
Gummel-Poon and VBIC define no separate saturation voltage.
BSIM3 smooths transitions through `Vgsteff` and `Vdseff` without hard region switches, so regions follow the manual's `Vth` and `Vdsat` boundaries.
VDMOS uses softened overdrive that conducts continuously below threshold, so VDMOS distinguishes only `linear` and `saturated`.
Devices lacking model-defined regions (such as resistors) raise `KeyError` instead of guessing, and ngspice-simulated foundry circuits record no regions.

The report lists each transistor's region and margins, the transistor with the margin nearest zero first, such as the [common-emitter amplifier](../gallery/common-emitter.md)'s:

<!-- output of: jolty simulate common_emitter.py --no-schematic --no-plot -->

```text
op: operating point, found by Newton iteration in 11 iterations
  ...
  regions, nearest a boundary first:
    X1/Q1  active  base_emitter_voltage 663.5 mV  base_collector_voltage -5.594 V
```

## The results file

`jolty simulate` saves results next to the script as `<script>.results.npz`, or at the path given by `--results`.
The archive stores every completed analysis; if all analyses fail, any prior results file at that path is removed so stale data is never left behind.

`jolty.files.save_raw(path, results, binary=True)` exports operating points, DC sweeps, AC responses, and transients as binary or ASCII SPICE raw files compatible with ngspice and waveform viewers.
Raw vectors use path names for nets, `V1#branch` for source currents, `temperature` type for thermal unknowns, and `notype` for dimensionless states.
`jolty.files.load_raw(path)` parses binary or ASCII SPICE raw files back into typed results (`OperatingPointResult`, `DcSweepResult`, `AcResult`, `TransientResult`) keyed by plot title with exact binary IEEE-754 bits, and `jolty.spice.ngspice.read` exposes raw vectors directly.
From CLI or MCP, `jolty simulate --raw out.raw` (with `--raw-format binary` or `ascii`) writes raw output alongside `.results.npz`.

Batch outputs save next to results as `<script>.batches.npz`, which `jolty.files.load_batches` restores as [batch results](simulation.md#batches) with measured requirements.

`jolty.files.load_results` restores `.results.npz` as a dictionary keyed by analysis name:

<!-- not run: it reads the results file of a script `filter.py`, which the page does not write -->

```python
results = load_results(Path("filter.results.npz"))
point = results["op"]
point.voltage("out")  # a float, in volts
point.current("V1")  # a float, in amperes
point.current("R1", "a")  # the current into the terminal a of R1, if recorded
point.charge("C1", "a")  # the charge on the terminal a of C1, in coulombs
results["step"].voltage("out", reference="in")  # an array, one value per time
```

| Result | Arrays | Each value is |
| --- | --- | --- |
| `OperatingPointResult` | `values`, `currents`, `charges`, `conductances`, `capacitances`, `variables` | a float |
| `DcSweepResult` | `swept`, and the operating point's arrays | one per value |
| `AcResult` | `frequencies`, `values`, `currents`, and the `operating_point` | a complex phasor, one per frequency |
| `TransientResult` | `times`, the operating point's arrays, and the step `statistics` | one per accepted time point |
| `PeriodicSteadyStateResult` | a `TransientResult`'s over one period, its `period` and the `periods` shooting took | one per accepted time point of the period |
| `OperatingPointsResult` | `points`, an `OperatingPointResult` per DC solution | read with `voltages(net)`, one per solution |
| `SensitivityResult` | `parameters`, `derivatives`, and the `result` differentiated | one per output and parameter, at each point |
| `StabilityResult` | `frequencies`, `gains`, and the `operating_point` | a complex loop gain, one per frequency |
| `NoiseResult` | `frequencies`, `densities`, `contributions` per noise source, and an `input` with its `gains` | a density, one per frequency |
| `PoleZeroResult` | `poles`, `zeros`, and the `operating_point` | a complex frequency, one per root |

An operating point's `strategy` records whether direct Newton (`direct`) or a continuation aid succeeded, and `iterations` sums Newton iterations across the winning strategy, prior failed attempts, and logic-settling re-solves.
A DC sweep's `strategies` and `iterations` record the same statistics at every swept value.
Transient `statistics` count accepted steps, rejected steps, event retries, and accepted-step Newton iterations.
`statistics.work` counts all Newton iterations (including operating-point solve unless cached) plus sparse solver full factorizations, refactorizations, pivot fallbacks, and factor reuses.
When `profile` is enabled, `statistics.profile` breaks runtime down in seconds across preparation, device evaluation, matrix assembly, factoring, solving, step control, recording, and overhead.
`statistics.charge_leak` integrates each equation's converged residual over accepted steps using the active integration formulas, ordered by `unknowns`.
`signed` stores the net integral (coulombs for electrical nodes, webers for branches, joules for thermal nets) and `absolute` accumulates step magnitudes without cancellation, readable via `result.charge_leak(unknown, absolute=...)`.

Accessor methods read individual quantities by name: `voltage(net, reference=None)` for net voltage, `rise(net)` for thermal rise above ambient, `state(name)` for dimensionless device state (such as `M1:state`), and `logic_level(net)` for logic nets.
Current and charge accessors include `current(instance)` for branch currents, `current(instance, terminal)` (watts on thermal terminals) and `charge(instance, terminal)` for recorded terminals, `conductance`, `capacitance`, and `variable` for recorded device models, and `region(instance)` for transistor operating regions.
Dictionary-indexed lookups run in O(1) regardless of circuit size.
Requesting an unknown name raises `KeyError` suggesting the closest valid names.

Results archives use NumPy `.npz` containers with a JSON header listing unknowns, logic nets, deduplicated recording schemas (paths, terminals, internals, variables, regions, margins, and summed terminals), and per-analysis metadata followed by keys like `step.times`.
Any tool supporting NumPy `.npz` files can inspect the archive directly.
The header uses format version 20; earlier format versions are rejected on load and regenerated by re-simulating.
