# Results

`jolty simulate` reports what each analysis found, and saves every value to a results file.

## The report

A report states facts, one section per analysis in the plan's order:

- an operating point lists every unknown's value and every logic net's level, and the strategy that found it;
- a sweep, a frequency response or a transient lists each unknown's first, lowest, highest and last value, and each logic net's first and last level and how often it changed;
- a sensitivity analysis lists, for each output, its value and its derivative with respect to each parameter, in the circuit's order;
- a failed analysis says why it failed.

A frequency response lists magnitudes per unit excitation: volts per volt, for a voltage source.
A report lists the first unknowns, then the first logic nets, up to the setting `listed_unknowns`, and a sensitivity analysis as many parameters; the results file holds them all.
With `--output-format json`, the command prints the same report as data.

## Plots

`jolty simulate` plots every analysis but operating points and sensitivities beside its results, as `<script>.<analysis>.plot.png`, and the MCP server sends the images with its text.
A transient plots over time, a sweep along its source's levels, and a frequency response as a Bode plot:

![The step response of the RC low-pass filter: the input steps to 1 V at 1 µs and the output charges toward it](../images/rc_filter.step.plot.png)

![The Bode plot of the RC low-pass filter: flat to its corner at 159 kHz, then falling 20 dB a decade as its phase falls to -90°](../images/rc_filter.bode.plot.png)

Voltages and currents never share an axis: each quantity has a panel of its own, and a frequency response has two, its magnitude in decibels and its phase.
A legend names the series, and up to four are also labeled at their end:

![The half-wave rectifier over three periods: the output holds near the peak of the input, and the source delivers a pulse of current at each peak](../images/rectifier.ripple.plot.png)

Logic nets share a panel of their own, one lane each, with unknown levels hatched (see [Logic](logic.md#levels-in-results)).

By default a plot draws the nets the top component names, voltages then logic nets, up to eight.
The option `--probes` chooses other unknowns, such as `X1/mid` or `V1:current`, and `jolty plot` draws a saved results file again, with other probes, without simulating:

```sh
uv run jolty plot rectifier.results.npz --analyses ripple --probes out --probes V1:current
```

The option `--no-plot` leaves plots out.

## Unknowns

Every result holds the circuit's unknowns, named by path:

- the voltage of a net, such as `out` or `X1/mid`, relative to the ground;
- the current through a voltage source, inductor or voltage-controlled voltage source, such as `V1:current`;
- the junction voltage of a diode with a series resistance, such as `D1:junction`.

## Terminals

Every result also holds, at every point it solved, the current into every terminal of every device and the charge that terminal holds, such as `X1/R1.a`, the terminal `a` of the resistor `X1/R1`.
A current is exact: the engine evaluates each device alone at the solution, each terminal on a node of its own, so terminals that share a net, or join the ground, keep their own currents.
In a transient, a current includes its charge's derivative, taken by the very integration formula the step was solved with, so the currents into a net sum to its equation's residual, which Newton iteration drives within tolerance.
A frequency response holds each terminal current's phasor.

A component instance's terminals, its ports such as `X1.vdd`, sum the device terminals inside it that join each port's net there, at any depth.
Ports tied together outside keep their own currents.
A logic device's pins on electrical nets, and its supply and ground, carry the currents of its connect elements: the threshold detector that loads an input with its capacitance, and the driver that holds an output at its level's fraction of the supply.
`composites` lists each of these terminals with the device terminals it sums.

## Devices' own models

At every point it solved, a result holds each device's small-signal model exactly: the derivative of each of the device's equations with respect to each of its unknowns, taken at the solution as Newton iteration takes them.
A device's unknowns are its terminals, each on a node of its own, and its internal unknowns, such as a transistor's internal drain behind its series resistance, `M1:drain`, or a source's branch current, `V1:current`.
`conductance(equation, unknown)` reads the derivative of an equation's residual, and `capacitance(equation, unknown)` that of its charge.
Between two terminals, the first is the current into the first terminal, the second the voltage at the second: a transistor's transconductance is `conductance("M1.drain", "M1.gate")`, its output conductance `conductance("M1.drain", "M1.drain")`, and its gate-drain capacitance, where a charge describes it, `-capacitance("M1.gate", "M1.drain")`.
An internal unknown's equation is its own: Kirchhoff's current law at an internal node, or the branch equation that holds a source's voltage.
`derivatives` lists every pair of every device's terminals and internal unknowns, row by row, so a device's block reshapes into its square matrices.

A device's variables are the quantities it computes on its way to its currents that neither its terminals, nor its unknowns, nor their derivatives show.
`variable(instance, name)` reads one, and `variables` lists them all by device and name.

| Model | Variables |
| --- | --- |
| `mosfet` | `threshold_voltage` and `saturation_voltage`, in the transistor's own polarity; Meyer's `gate_source_capacitance`, `gate_drain_capacitance` and `gate_bulk_capacitance` |

```python
point.conductance("M1.drain", "M1.gate")  # gm, in siemens
point.variable("M1", "threshold_voltage")  # von, with the body effect, in volts
results["step"].variable("M1", "saturation_voltage")  # one per time point
```

## The results file

`jolty simulate` saves the results beside the script as `<script>.results.npz`, or where `--results` says.
The file holds the analyses that ran to their end; when none did, the command removes an earlier results file at that path, so it never describes another circuit.

`jolty.files.load_results` reads it back as a dictionary of results by analysis name:

```python
results = load_results(Path("filter.results.npz"))
point = results["op"]
point.voltage("out")  # a float, in volts
point.current("V1")  # a float, in amperes
point.current("R1", "a")  # the current into the terminal a of R1
point.charge("C1", "a")  # the charge on the terminal a of C1, in coulombs
results["step"].voltage("out", reference="in")  # an array, one value per time
```

| Result | Arrays | Each value is |
| --- | --- | --- |
| `OperatingPoint` | `values`, `currents`, `charges`, `conductances`, `capacitances`, `readings` | a float |
| `Sweep` | `levels`, and the operating point's arrays | one per level |
| `Response` | `frequencies`, `values`, `currents`, and the `operating_point` | a complex phasor, one per frequency |
| `Waveforms` | `times`, the operating point's arrays, and the step `statistics` | one per accepted time point |

`voltage(net, reference=None)` reads a net's voltage, relative to another net when given, `current(instance)` reads the current through an instance, `current(instance, terminal)` and `charge(instance, terminal)` read a terminal's, `conductance`, `capacitance` and `variable` read a device's own model, and `level(net)` reads a logic net's level.
`terminals` names the columns of `currents` and `charges`, `derivatives` those of `conductances` and `capacitances`, and `variables` those of `readings`.
A name that is not an unknown, a terminal, a derivative, a variable or a logic net raises a `KeyError` that suggests the closest names.

The file is a NumPy archive: a JSON header names the unknowns, the logic nets, the terminals, the derivatives and the variables, and says what each analysis is, then each analysis's arrays follow under keys such as `step.times`.
Any program that reads NumPy files reads it.
