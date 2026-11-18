# Results

`jolty simulate` reports what each analysis found, and saves every value to a results file.

## The report

A report states facts, one section per analysis in the plan's order:

- an operating point lists every unknown's value and every logic net's level, the strategy that found it and its Newton iterations, and each transistor's [region of operation](#regions-of-operation) with its margins to the regions' boundaries, those nearest a boundary first;
- a search for operating points lists every unknown's value at each DC solution it found, the operating point first, and the Newton iterations that found each;
- a DC sweep states the Newton iterations of all its levels, and how many levels each strategy but Newton iteration alone found;
- a sweep, a frequency response or a transient lists each unknown's first, lowest, highest and last value, and each logic net's first and last level and how often it changed;
- a sensitivity analysis lists, for each output, its value and its derivative with respect to each parameter, in the circuit's order, or, along another analysis, each derivative's largest and where;
- a stability analysis states the loop gain at its lowest frequency, its crossover frequency and phase margin, and its gain margin;
- a pole-zero analysis lists each pole and zero, a conjugate pair once, with its natural frequency and damping ratio;
- a noise analysis states the noise density at its lowest and highest frequencies, the RMS noise over them at the output and referred to its input, and the noisiest sources with their RMS noise and share of the output's noise power;
- a failed analysis says why it failed.

A frequency response lists magnitudes per unit excitation: volts per volt, for a voltage source.
A report lists the first unknowns, then the first logic nets, up to the setting `max_unknowns`, and an operating point as many transistors and a sensitivity analysis as many parameters; the results file holds them all.
With `--output-format json`, the command prints the same report as data.

## Plots

`jolty simulate` plots every analysis but operating points, searches for them and sensitivities beside its results, as `<script>.<analysis>.plot.png`, and the MCP server sends the images with its text.
A transient plots over time, a sweep along its source's levels, and a frequency response as a Bode plot, as is a loop gain, against 0 dB and -180°, where its margins are read; a noise analysis plots its output's noise density, with its noisiest sources', and the noise referred to its input on log axes; and a pole-zero analysis plots each pole and zero at its natural frequency and damping ratio, against the line at zero below which a pole grows:

![The step response of the RC low-pass filter: the input steps to 1 V at 1 µs and the output charges toward it](../images/rc_filter.step.plot.png)

![The Bode plot of the RC low-pass filter: flat to its corner at 159 kHz, then falling 20 dB a decade as its phase falls to -90°](../images/rc_filter.bode.plot.png)

Voltages, currents, temperatures and devices' states never share an axis: each quantity has a panel of its own, and a frequency response has two, its magnitude in decibels and its phase.
A legend names the series, and up to four are also labeled at their end:

![The half-wave rectifier over three periods: the output holds near the peak of the input, and the source delivers a pulse of current at each peak](../images/rectifier.ripple.plot.png)

Logic nets share a panel of their own, one lane each, with unknown levels hatched (see [Logic](logic.md#levels-in-results)).

By default a plot draws the nets the top component names, voltages then logic nets, up to eight.
The option `--traces` chooses other unknowns, such as `X1/mid` or `V1:current`, and `jolty plot` draws a saved results file again, with other probes, without simulating:

```sh
uv run jolty plot rectifier.results.npz --analyses ripple --traces out --traces V1:current
```

A plan's [Monte Carlo analysis](simulation.md#monte-carlo) and [variants](simulation.md#variants) plot the distribution of their runs at the plan's own conditions, not each run: a histogram of what each requirement read, against its bounds, as `<script>.monte-carlo.plot.png` or `<script>.variant-runs.plot.png`, and the envelope of each plotted net along each analysis, its least, median and greatest value at each point, as `<script>.monte-carlo.<analysis>.plot.png`.

The option `--no-plot` leaves plots out.

## Unknowns

Every result holds the circuit's unknowns, named by path:

- the voltage of a net, such as `out` or `X1/mid`, relative to the ground;
- the temperature of a thermal net above the ambient, in kelvins, such as `junction`;
- the current through a voltage source, inductor or voltage-controlled voltage source, such as `V1:current`;
- the junction voltage of a diode with a series resistance, such as `D1:junction`.

## What a result records

Beside the unknowns, a result records what its plan's `record`, or its analysis's own, names of the circuit's devices and component instances, at every point it solved:

```python
analyses = {"op": OperatingPoint(), "step": Transient(stop=1 * ms)}
Plan(analyses, record=["M1", "X1"])  # the transistor M1, and the ports of X1
Plan(analyses, record=["X1/**", "!X1/bias/**"])  # inside X1 but its bias network
Plan(analyses, record=["**"])  # every instance
Transient(stop=1 * ms, record=["D1"])  # this analysis records D1, whatever the plan's
```

Each entry is an instance's path, or a pattern of paths as a plan's [fidelity](fidelity.md) scopes are written, where `*` matches within a level of the hierarchy and `**` across levels.
An entry after `!` takes back what the entries before it named, and the last entry that matches an instance decides, as in a `.gitignore`.
The plan's record holds for each of its analyses that records, an operating point, a sweep, a small-signal response or a transient, unless the analysis names its own; left out, it records nothing.
A device recorded keeps its terminals' currents and charges, its own small-signal model and its variables.
A component instance or a logic device recorded keeps its terminals' currents and charges, each summed by the engine from the device terminals behind it, which it keeps no further.
An analysis fails when its record names a path the circuit does not have, saying the closest paths it has, or holds a pattern that matches no instance.

Whatever the record names, an operating point and a transient also record the instances rated for a current or a power, whose currents the [checks of ratings](checks.md#on-analyses-results) read, and the plan's own operating points, as `jolty simulate` runs them, the [region](#regions-of-operation) and margins alone of each transistor the record leaves out, which their report lists; specs, plots, schematics and the operating point's rules read the unknowns alone.
A plan's corners, Monte Carlo samples, worst-case runs and spot checks record no more than that, since only specs read their results.

Recording costs memory at every point: a MOSFET recorded keeps 48 numbers, or 114 where its gate oxide adds three unknowns, while its nets hold a few unknowns, and the `max_memory` option bounds them all (see [Options](simulation.md#options)).
Reading what a result did not record raises a `KeyError` that suggests the closest names it holds and how to record the instance:

<!-- output of: python unrecorded.py -->

```text
KeyError: "the result recorded no terminal named 'R1.a'; close names: R2.a; record its instance, as an analysis's or a plan's record=['R1'] does, or every instance, as record=['**'] does"
```

A name mistaken for one of a recorded instance's says so, and suggests the closest names alone, such as `'R1', which the result recorded, has no terminal named 'R1.aa'; close names: R1.a`.

A result's `recording` says what it holds: `devices`, each with its `path`, `terminals`, `internals` and `variables`, and `summed_terminals`, such as `X1.vdd`.
`terminals` names the columns of `currents` and `charges`: each device's terminals, then each summed terminal.

## Terminals

A recorded terminal's current is exact: the engine evaluates each recorded device alone at the solution, each terminal on a node of its own, so terminals that share a net, or join the ground, keep their own currents.
In a transient, a current includes its charge's derivative, taken by the very integration formula the step was solved with, so the currents into a net sum to its equation's residual, which Newton iteration drives within tolerance.
A frequency response holds each recorded terminal current's phasor.

A component instance's terminals, its ports such as `X1.vdd`, sum the device terminals inside it that join each port's net there, at any depth.
Ports tied together outside keep their own currents.
A logic device's pins on electrical nets, and its supply and ground, carry the currents of its connect elements: the threshold detector that loads an input with its capacitance, and the driver that holds an output at its level's fraction of the supply.

## Devices' own models

At every point it solved, a result holds each recorded device's small-signal model exactly: the derivative of each of the device's equations with respect to each of its unknowns, taken at the solution as Newton iteration takes them.
A device's unknowns are its terminals, each on a node of its own, and its internal unknowns, such as a transistor's internal drain behind its series resistance, `M1:drain`, or a source's branch current, `V1:current`.
`conductance(equation, unknown)` reads the derivative of an equation's residual, and `capacitance(equation, unknown)` that of its charge.
Between two terminals, the first is the current into the first terminal, the second the voltage at the second: a transistor's transconductance is `conductance("M1.drain", "M1.gate")`, its output conductance `conductance("M1.drain", "M1.drain")`, and its gate-drain capacitance, where a charge describes it, `-capacitance("M1.gate", "M1.drain")`.
An internal unknown's equation is its own: Kirchhoff's current law at an internal node, or the branch equation that holds a source's voltage.
The columns of `conductances` and `capacitances` hold, for each recorded device, every pair of its terminals and internal unknowns, row by row, so a device's block reshapes into its square matrices.

A device's variables are the quantities it computes on its way to its currents that neither its terminals, nor its unknowns, nor their derivatives show.
`variable(instance, name)` reads one, and each recorded device's `variables` names its own.

| Model | Variables |
| --- | --- |
| `mosfet` | `threshold_voltage` and `saturation_voltage`, in the transistor's own polarity; Meyer's `gate_source_capacitance`, `gate_drain_capacitance` and `gate_bulk_capacitance`; its region and margins |
| `bsim3` | `threshold_voltage` and `saturation_voltage`, in the transistor's own polarity; its region and margins |
| `vdmos` | `threshold_voltage` and `saturation_voltage`, in the transistor's own polarity; `gate_source_capacitance`, `gate_drain_capacitance` and its body diode's `drain_source_capacitance`; its region and margin |
| `jfet`, `bipolar_transistor`, `vbic` | its region and margins |
| `voltage_controlled_memristor`, `current_controlled_memristor` | `resistance`, the one its state sets, and `rate`, how fast its state moves, toward off when positive, in full ranges per second |

<!-- not run: it reads a simulation of a circuit with a MOSFET `M1`, and its results, which the page does not build -->

```python
point = simulation.run(OperatingPoint(record=["M1"]))
point.conductance("M1.drain", "M1.gate")  # gm, in siemens
point.variable("M1", "threshold_voltage")  # von, with the body effect, in volts
results["step"].variable("M1", "saturation_voltage")  # one per time point
```

## Regions of operation

Each transistor states its region of operation, as its model's equations distinguish it, and its margins to the regions' boundaries, in volts: each margin is zero on a boundary, where the equations switch from one region to the next, and changes sign across it.
Its variable `region` numbers its region among those its recorded device's `regions` names, and its device's `margins` names the variables that are its margins; `region(instance)` reads the region by name, and a simulation's `transistors` are the paths of its transistors:

<!-- not run: it reads a simulation of a circuit with a MOSFET `M1`, which the page does not build -->

```python
point = simulation.run(OperatingPoint(record=simulation.transistors))
point.region("M1")  # such as "saturated"
point.variable("M1", "saturation_margin")  # vds - vdsat, in volts
regions = point.recording.devices[0].regions  # ("off", ..., "saturated")
```

An analysis's `regions` keeps, of each transistor its record leaves out, its region and its margins alone, three of a MOSFET's 48 numbers, which `region` and `variable` read as they read a transistor recorded whole; `jolty simulate` asks for them at the plan's own operating points:

<!-- not run: it reads a simulation of a circuit with a MOSFET `M1`, which the page does not build -->

```python
point = simulation.run(OperatingPoint(regions=True))
point.region("M1")  # such as "saturated"
kept = point.recording.devices[0].variables  # ("region", "overdrive_voltage", ...)
```

| Model | Regions | Margins |
| --- | --- | --- |
| `mosfet` | `off` at or below its threshold, or `weak_inversion` below it at level 3 with fast surface states; then `linear` with the drain below saturation, `saturated` at or past it, as `mos1load.c` and `mos3load.c` switch | `overdrive_voltage`, `vgs − von`, and `saturation_margin`, `vds − vdsat` |
| `jfet` | `off` at or below its threshold, then `linear` with the drain within `vgs − vto`, `saturated` past it, as `jfetload.c` switches | `overdrive_voltage`, `vgs − vto`, and `saturation_margin`, `vds − max(vgs − vto, 0)` |
| `bsim3` | `weak_inversion` below its threshold, then `linear` below its saturation voltage, `saturated` at or past it | `overdrive_voltage`, `Vgs_eff − Vth`, and `saturation_margin`, `Vds − Vdsat` |
| `vdmos` | `linear` with the drain below its softened overdrive, `saturated` at or past it, as `VDMOSload` switches | `saturation_margin`, `vds` less the softened overdrive over the triode's multiplier |
| `bipolar_transistor`, `vbic` | `cut_off`, neither junction injecting, `active`, the base-emitter junction alone, `saturated`, both, and `reversed`, the base-collector junction alone | `base_emitter_voltage` and `base_collector_voltage`, its intrinsic junctions' |

A margin is in an n-channel's or an NPN's sign whatever the transistor's polarity, a p-channel's or a PNP's taken at its voltages reversed, so a positive margin means on, or saturated, for every transistor, as designers write `|vgs| − |vth|`: a p-channel's overdrive is positive above its threshold, and a PNP's base-emitter voltage, its emitter's over its base's, positive where its junction injects.
Where a field-effect transistor's drain lies below its source, its equations swap the two, and so do its margins: `vds` is then the source's voltage over the drain's.
A junction injects where its ideal current, `Is · (e^{v / (N · Vt)} − 1)`, is positive, at a positive voltage, so a bipolar transistor saturates as its base-collector voltage turns positive, where `vce` falls below `vbe`: Gummel-Poon's and VBIC's equations define no saturation voltage of their own.
BSIM3's equations blend one region into the next, through `Vgsteff` and `Vdseff`, and switch at neither boundary; its regions are those its manual derives its current in, bounded by its own threshold and saturation voltages.
VDMOS's softened overdrive keeps it conducting below its threshold, so its equations distinguish no region there.
No region is guessed: a device whose model distinguishes none, such as a resistor, raises a `KeyError` that says so, and a circuit of a foundry's devices, which ngspice simulates, records no device and states no region.

The report lists each transistor's region and margins, the transistor with the margin nearest zero first, such as the [common-emitter amplifier](../gallery/common-emitter.md)'s:

<!-- output of: jolty simulate common_emitter.py --no-schematic --no-plot -->

```text
op: operating point, found by Newton iteration in 11 iterations
  ...
  regions, nearest a boundary first:
    X1/Q1  active  base_emitter_voltage 663.5 mV  base_collector_voltage -5.594 V
```

## The results file

`jolty simulate` saves the results beside the script as `<script>.results.npz`, or where `--results` says.
The file holds the analyses that ran to their end; when none did, the command removes an earlier results file at that path, so it never describes another circuit.

`jolty.files.save_raw(path, results, binary=True)` writes results of operating points, DC sweeps, small-signal responses and transients, by analysis name, as SPICE's raw file, binary or ASCII, a plot each, in the format ngspice writes and other tools read, such as waveform viewers and ngspice itself: a net's vector named by its path, a source's current as ngspice names it, `V1#branch`, a thermal unknown with type `temperature`, and a device's state with `notype`.
`jolty.files.load_raw(path)` reads either binary or ASCII SPICE raw files back into a dictionary of `OperatingPointResult`, `DcSweepResult`, `AcResult`, and `TransientResult` keyed by plot title, preserving exact IEEE-754 bits for binary files; `jolty.spice.ngspice.read` also reads raw vectors directly.
From the command line or MCP server, `jolty simulate --raw out.raw` (with `--raw-format binary` or `ascii`) writes every completed raw-writable analysis to `out.raw` alongside the `.results.npz` archive.

A plan's batches go beside it as `<script>.batches.npz`, which `jolty.files.load_batches` reads back as each batch's [result](simulation.md#batches), with the requirements it measured.

`jolty.files.load_results` reads it back as a dictionary of results by analysis name:

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
| `DcSweepResult` | `swept`, and the operating point's arrays | one per level |
| `AcResult` | `frequencies`, `values`, `currents`, and the `operating_point` | a complex phasor, one per frequency |
| `TransientResult` | `times`, the operating point's arrays, and the step `statistics` | one per accepted time point |
| `PeriodicSteadyStateResult` | a `TransientResult`'s over one period, its `period` and the `periods` shooting took | one per accepted time point of the period |
| `OperatingPointsResult` | `points`, an `OperatingPointResult` per DC solution | read with `voltages(net)`, one per solution |
| `SensitivityResult` | `parameters`, `derivatives`, and the `result` differentiated | one per output and parameter, at each point |
| `StabilityResult` | `frequencies`, `gains`, and the `operating_point` | a complex loop gain, one per frequency |
| `NoiseResult` | `frequencies`, `densities`, `contributions` per noise source, and an `input` with its `gains` | a density, one per frequency |
| `PoleZeroResult` | `poles`, `zeros`, and the `operating_point` | a complex frequency, one per root |

An operating point's `strategy` says how the engine found it, by Newton iteration alone, `direct`, or by the aid that took over where it failed, and its `iterations` count the Newton iterations that found it: the strategy's that found it, those of every strategy tried before it, and those of each solve again as the logic settled with it, as solving it afresh runs them, whatever the simulation remembered.
A DC sweep's `strategies` and `iterations` say the same of the operating point at each level, from the level before it.
A transient's `statistics` count its steps, accepted, rejected or tried again to land on an event, and the Newton iterations of those accepted.
Its `work` counts every Newton iteration it ran, its operating point's included unless the simulation remembered it, and its sparse solver's factorizations: those that searched for every pivot, those that kept the last's, those that fell back to a search from a pivot become too small, and those given the very matrix factored last.
With the `profile` option, its `profile` says where its time went, in seconds: preparing, evaluating the devices, assembling each Newton iteration's matrix, factoring, solving, step control, recording, and the rest.
Its `charge_leak` holds each equation's converged residual integrated over the accepted steps by those steps' own integration formulas, in the order of `unknowns`: `signed` for the net integral (coulombs for electrical nodes, webers for current branches, joules for thermal nets) and `absolute` for the cumulative sum of step leak magnitudes without cancellation; `result.charge_leak(unknown)` and `result.charge_leak(unknown, absolute=True)` read them by name.

`voltage(net, reference=None)` reads a net's voltage, relative to another net when given, `rise(net)` a thermal net's temperature above the ambient, `state(name)` a device's dimensionless state, such as a memristor's `M1:state`, `current(instance)` reads the current through an instance, `current(instance, terminal)` and `charge(instance, terminal)` read a recorded terminal's, the current into a thermal terminal being the heat that flows in, in watts, `conductance`, `capacitance` and `variable` read a recorded device's own model, `region(instance)` a recorded transistor's region of operation at an operating point, and `level(net)` reads a logic net's level.
Each finds its column by name in a dictionary, so a read costs the same however large the circuit.
A name that is not an unknown, a recorded terminal, derivative or variable, or a logic net raises a `KeyError` that suggests the closest names.

The file is a NumPy archive: a JSON header names the unknowns and the logic nets, lists each recording the analyses made once, however many share it, as each recorded device's path, terminals, internal unknowns, variables, regions and margins, and the summed terminals, and says what each analysis is and which recording it made, then each analysis's arrays follow under keys such as `step.times`.
Any program that reads NumPy files reads it.
The header is in version 20 of the format; a file of an earlier version is not read, and simulating again writes it anew.
