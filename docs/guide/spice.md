# SPICE models

Vendors publish their parts as SPICE models: an op-amp as a macromodel subcircuit, a transistor as a model card.
`jolty.spice` reads SPICE files into a library whose subcircuits each become a component to place, so a circuit can use the part its maker modelled.

## Loading a library

`spice.load(*paths)` reads SPICE files into a `Library`, and `library.component(name, **parameters)` returns a subcircuit as a component, its parameters' defaults replaced by those given:

```python
from jolty import Builder, Role, devices, spice
from jolty.units import V

library = spice.load("OPA171.txt")
amplifier = library.component("OPA171")

c = Builder("follower")
ground = c.port("ground", Role.GROUND)
supply = c.place(devices.voltage_source(15 * V), id="VP")
part = c.place(amplifier, id="U1")
c.connect(supply.positive, part["VCC"], name="supply")
c.connect(supply.negative, part["VEE"], part[spice.GROUND], ground)
```

A subcircuit's pins become its ports, in order.
A subcircuit that reaches SPICE's global ground, node `0`, has one more port, `spice.GROUND`, which the circuit joins to its own ground.
Each instance records the file and the line of the statement that placed it, so a problem in an imported part names where the part says it.

`spice.fetch(url, sha256)` downloads a model file from where its maker publishes it, checks its SHA-256 digest, and keeps it in the [`cache`](../reference/configuration.md) directory, where later calls read it.

## What jolty reads

jolty reads a file as PSpice and ngspice read it:

- resistors, capacitors and inductors, with their model cards' scales;
- independent sources at their DC level;
- diodes, JFETs, bipolar transistors and level 1 MOSFETs, with their model cards, areas and multipliers, and models that inherit from one another through `AKO:`;
- linear controlled sources, and `POLY`, `VALUE` and `TABLE` ones;
- `B` sources, `V=` or `I=` an expression;
- switches of PSpice's `VSWITCH` model;
- subcircuit instances, with their parameters, and `.param` statements.

Expressions are read with PSpice's functions and ngspice's meanings, written in [jolty's expression language](../reference/expressions.md): `PWR(x, y)` is `|x|^y`, a `TABLE` within an expression extends beyond its ends, and a `TABLE` source holds beyond its ends with its corners rounded, as ngspice's XSPICE `pwl` model rounds them.
A switch follows ngspice's log-cubic law between its on and off voltages.
A bipolar transistor has a substrate, as in SPICE: the ground unless its instance names a fourth node, at its collector for an NPN and at its base for a PNP unless `SUBS` builds it otherwise.

An element that reads the current through a voltage source reads it through a port placed in series with the source, as the nets `V1.sense1`, `V1.sense2` and so on for a source `V1`.

Parameters that change nothing at jolty's temperature, the nominal 27 °C, such as temperature coefficients, and those that only noise analyses read, are left out.
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
