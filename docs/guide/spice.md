# SPICE models

Vendors publish their parts as SPICE models: an op-amp as a macromodel subcircuit, a transistor as a model card.
`jolty.spice` reads SPICE files into a library whose subcircuits each become a component to place, so a circuit can use the part its maker modelled.

## Loading a library

`spice.load(*paths)` reads SPICE files into a `Library`, and `library.component(name, **parameters)` returns a subcircuit as a component, its parameters' defaults replaced by those given:

<!-- not run: it reads Texas Instruments' model file, which `tests/test_vendor.py` downloads and checks -->

```python
from jolty import Builder, Role, devices, spice
from jolty.units import V

library = spice.load("OPA171.txt")
amplifier = library.component("OPA171")

c = Builder("follower")
ground = c.port("ground", Role.GROUND)
supply = c.place(devices.voltage_source(15 * V), name="VP")
part = c.place(amplifier, name="U1")
c.connect(supply.positive, part["VCC"], name="supply")
c.connect(supply.negative, part["VEE"], part[spice.GROUND], ground)
```

A subcircuit's pins become its ports, in order.
A subcircuit that reaches SPICE's global ground, node `0`, has one more port, `spice.GROUND`, which the circuit joins to its own ground.
Each instance records the file and the line of the statement that placed it, so a problem in an imported part names where the part says it.

`spice.fetch(url, digest)` downloads a model file from where its maker publishes it, checks its SHA-256 digest, and keeps it in the [`cache`](../reference/configuration.md) directory, where later calls read it.

## What jolty reads

jolty reads a file as PSpice and ngspice read it:

- resistors, capacitors and inductors, with PSpice's model cards' scales and their copies, `M`, SPICE3's cards of a resistor or a capacitor, of type `R` or `C`, which size an element of no value of its own by its width and length as ngspice does, its sheet resistance or its capacitances per area and per perimeter, its narrowing and shortening, and its default size and value, and mutual inductances, `K`, which couple inductors;
- independent sources at their DC level, or with their `PULSE`, `SIN`, `PWL` or `EXP` waveform over time, its arguments in parentheses or, as SPICE2 wrote them, to the statement's end, whose level at zero every other analysis reads;
- diodes of level 1, JFETs, bipolar transistors and level 1 and 3 MOSFETs, with their model cards, positive areas and multipliers, and models that inherit from one another through `AKO:`;
- a resistor whose value reads a voltage or a current, as ngspice makes it a behavioral current, `m · V(a, b) / R`, divided by `1 + TC1 · ΔT + TC2 · ΔT²`, ΔT the temperature the library is read at above 27 °C, without noise, as ngspice's has none;
- a capacitor of a charge, `Q=` an expression, which ngspice makes the current of the charge's rate of change, as a [behavioral capacitor](devices.md#behavioral-sources) of that charge, times its copies `m`; and a capacitor whose value reads its own voltage, `C=` an expression, which ngspice makes a current `m · C(v) · dv/dt`, as a behavioral capacitor of its charge, the value's integral from 0 to its voltage, in closed form where the value, its other terms evaluated, is a polynomial of the voltage, plus numbers times `exp`, `sinh`, `cosh`, `tanh`, `sin` or `cos` of a line of it; a value that reads another voltage or a current, which makes no charge, or has no closed integral is refused by name, as are temperature and voltage coefficients and sizes on either, and its own temperature, which ngspice drops, changes nothing; ngspice's 1e15 Ω across such a capacitor, which keeps a node it alone joins from floating, is left out, and its current is not delayed by the 1 fs of ngspice's differentiation of the voltage ([departures](departures.md));
- linear controlled sources, and `POLY`, `VALUE` and `TABLE` ones;
- `B` sources, `V=` or `I=` an expression, which may read the current of any source of a voltage, their own included;
- switches of PSpice's `VSWITCH` and `ISWITCH` models, `S` and `W`, a `W` switch reading the current of a source of a voltage as a current-controlled source does;
- subcircuit instances, with their parameters, and a multiplier `m`, where the subcircuit has no parameter of that name, which multiplies each element it holds as ngspice's: a resistor's and an inductor's copies divide its value, a capacitor's multiply it, and a diode's, a transistor's and an instance's copies are multiplied by it, sources of a current under a multiplier being errors; and `.param` statements;
- what process design kits write, as the [foundry's guide](foundry.md#the-open-process-kits) tells: `.include` and `.lib` sections, `.option scale`, a subcircuit's `.if` blocks and its own parameters, which its instances may give, `temper`, and the statistical functions `agauss`, `gauss`, `aunif`, `unif` and two-argument `limit`, at their nominal values unless a placement is drawn in a sample;
- `.ic` and `.nodeset` outside subcircuits, each node's voltage written `V(node)=value`, kept by the library as `initial_conditions` and `nodesets`, by node as written, for a plan's transient and options to take, as the [simulation guide](simulation.md#initial-conditions-and-nodesets) tells.

Parameters may read one another in any order, and expressions know PSpice's `TEMP`, `VT` and `GMIN`.
A `.func name(parameters) {expression}` outside subcircuits defines a function every parameter, element value and behavioral expression may call, written out in place of each call; a function may call another, but one that calls itself, directly or through others, is an error that names each, as jolty's expressions have no recursion.
Expressions are read with PSpice's functions and ngspice's meanings, written in [jolty's expression language](../reference/expressions.md): `PWR(x, y)` is `|x|^y`, `POW(x, y)` and `x ** y` keep a negative base's sign to an odd constant power, as ngspice's, and are otherwise of the magnitude of the base, a `TABLE` within an expression extends beyond its ends, and a `TABLE` source holds beyond its ends with its corners rounded, as ngspice's XSPICE `pwl` model rounds them.
A switch follows ngspice's log-cubic law between its on and off voltages or currents.
A waveform must give what SPICE would take from the transient analysis: a `PULSE` its rise, fall and width, positive, a `SIN` its frequency, and an `EXP` its two delays and two time constants, positive, since ngspice 47 takes each that is 0 from the transient's step (measured: `EXP(0 1 0 1n 6n 2n)` holds 0 V until the step, 0.5 ns, under `tran 0.5n 12n`), and a circuit is read apart from any analysis; a source's DC level beside its waveform must be the waveform's level at zero, as jolty's source has one level there for every analysis; and `SFFM` and `AM` waveforms, which jolty has not, are errors.
An element or a node whose name holds a `/`, as a flattened hierarchy writes them, is an error, as jolty's paths separate their levels with `/`.
A bipolar transistor has a substrate, as in SPICE: the ground unless its instance names a fourth node, at its collector for an NPN and at its base for a PNP unless `SUBS` builds it otherwise.

An element that reads the current through a voltage source reads it through a port placed in series with the source, as the nets `V1.sense1`, `V1.sense2` and so on for a source `V1`.
The inductors that mutual inductances couple, directly or through one another, become one device of [coupled inductors](devices.md#coupled-inductors), its windings in the order the file writes them, which takes the id and the line of the first mutual inductance: `K1 L1 L2 0.99` gives `K1`, whose currents are `K1:current1` and `K1:current2`.
A mutual inductance may name any number of inductors, as PSpice's does, each pair coupled by its coefficient; one with a magnetic core's model is an error.

Temperature parameters become the devices' own: a model's `TNOM`, or PSpice's `T_MEASURED`, is its nominal temperature, and a resistor's, capacitor's or inductor's `TC=tc1,tc2`, `TC1`, `TC2` and `TCE` drift its value.
A resistor's or capacitor's `VC1` and `VC2`, as HSPICE and Spectre write them, bend its value with its voltage; ngspice has no such parameters, so a netlist written for it gives that resistor or capacitor as an expression.
A device's own temperature comes from its model card's `T_ABS` or `T_REL_GLOBAL`, PSpice's, or its element's `TEMP` or `DTEMP`, ngspice's, as [`held_at` and `warmer_by`](devices.md) set it: `T_ABS=-273.15`, with which macromodels keep resistors quiet, holds a resistor at 0 K, where it makes no noise.
PSpice's `T_REL_LOCAL`, relative to the card a card inherits from, is an error, as is a device that sets its own temperature twice; a netlist written for ngspice gives each device its `TEMP` or `DTEMP`.
Expressions read `TEMP` and `VT` at 27 °C, whatever the temperature the circuit is simulated at.
A diode's, transistor's or resistor's `KF` and `AF` give its flicker noise; PSpice's `NLEV` and `GDSNOI`, which choose other noise equations, are left out, as ngspice leaves them; and a capacitor's or inductor's `KF`, flicker noise neither makes, is an error unless zero.
Any other parameter or element that jolty does not model is an error that names the file, the line and the statement; nothing is silently dropped.

## Agreement with ngspice

jolty's tests read vendors' macromodels from ngspice's own examples and compare every net and current of each with ngspice's, on the same file:

| Part | Maker | Analyses |
| --- | --- | --- |
| TL072 | Texas Instruments | operating point, DC sweep, AC, transient |
| MCP6041 | Microchip | operating point, DC sweep, AC, transient |
| AD22057 | Analog Devices | operating point, DC sweep, AC, transient |
| OPA171 | Texas Instruments | operating point |

ngspice runs PSpice's switches as XSPICE's, which stamp the negative of their conductance in an AC analysis, so the OPA171's frequency responses, which its ESD switches load, differ from jolty's by that error.
Every part's operating point also agrees at -40 °C and at 125 °C, but the AD22057's at 125 °C: ngspice scales a transistor's base-collector saturation current by its area twice, and the leakage of the AD22057's large transistors at that temperature makes the error visible.

## Writing SPICE

`jolty.spice.export` writes a circuit the other way, as a SPICE netlist that ngspice reads, to check jolty against the reference simulator:

<!-- not run: it exports a circuit `bench`, which the page does not build, and runs ngspice -->

```python
from jolty import circuit
from jolty.spice import export, ngspice

lines = export.netlist(circuit(bench()), excited="V1")
raw = ngspice.run(lines, "ac dec 20 10 1meg")
raw.vectors[export.vector(drawn, unknown)]  # ngspice's values of a jolty unknown
```

The netlist flattens the hierarchy into devices whose names and nets join their paths with underscores, as `RX1_R1` and `x1_mid`; components read from SPICE libraries stay instances of their subcircuits, named in `subcircuits`.
Behavioral sources become ngspice's `B` sources, and so do switches, each a current of its own law; coupled inductors become inductors and their `K` elements, a foundry's transistors instances of their [kit's](foundry.md) subcircuits, and each model card carries the device's parameters in SPICE's units.
ngspice reads a pulse's width of 0 as the transient's stop time, so a pulse with no width is written with a width of 1e-40 s, which vanishes beside its rise.
Logic devices have no SPICE element, so a circuit that holds one cannot be written.
`jolty.spice.ngspice` runs it: `run` runs one analysis with tolerances as tight as jolty's, at the temperature where ngspice's thermal voltage is jolty's, and reads the raw file it writes.
The [gallery](../gallery.md) replicates every example without logic devices this way.

`export.netlist(..., dialect=export.XYCE)` writes the netlist for Xyce 7.10.0 instead, Sandia's simulator, which `jolty.spice.xyce.run` runs as a separate program, as its GPL asks, reading the values it prints and the times it reports.
The dialect states only where Xyce departs from ngspice, each fact read from Xyce's sources: the parameters each of its models takes, a source's DC level, which Xyce takes as 0 where none is given, a pulse of no width, and temperatures where Xyce's thermal voltage, from SPICE3's constants, is jolty's.
What Xyce lacks or defines otherwise is refused by name, never dropped: BSIM3, whose Xyce version is 3.2.2, VDMOS and VBIC, other models in Xyce, excess phase, a parameter its model has not, and, for now, every device jolty writes as an expression.
Where Xyce's models compute otherwise, `tests/test_xyce.py` compares everything but the departure and names it: its MOSFETs of levels 1 and 3 keep SPICE3f5's reverse-biased bulk junction, a conductance `Is/vt`, where jolty's and ngspice's current settles on `−Is`; its JFET makes no noise and keeps SPICE3f5's temperature dependence; its bipolar transistor adds gmin to each junction's current before beta divides it, where jolty and ngspice add it to the base current undivided.
Xyce finds no operating point of one test circuit, `negative_load`, after 7,692 Newton iterations, and two disagreements of level 1 MOSFETs at 125 °C and across an inverter's threshold, up to 1.4e-4 of their values, are not explained yet.
