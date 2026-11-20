# SPICE models

Vendors publish their parts as SPICE models: an op-amp as a macromodel subcircuit, a transistor as a model card.
`jolty.spice` reads SPICE files into a library whose subcircuits each become a component to place, so a circuit can use the part its maker modeled.

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

A subcircuit's pins become its ports in declared order.
Subcircuits referencing SPICE global ground (node `0`) expose an additional `spice.GROUND` port for connection to the circuit ground.
Each imported instance records the SPICE file and line number so diagnostics pinpoint the exact netlist statement.

`spice.fetch(url, digest)` downloads a vendor model file, verifies its SHA-256 digest, and caches it under [`cache`](../reference/configuration.md) for reuse.
`spice.load` and `kits.read` cache parsed SPICE libraries under [`cache`](../reference/configuration.md), validating the SHA-256 digest of every transitively included file before reuse.

## What jolty reads

jolty parses SPICE files following PSpice and ngspice conventions:

- Resistors, capacitors, and inductors supporting PSpice model scales, multiplicity `M`, mutual inductances `K`, and SPICE3 `R`/`C` model cards.
  SPICE3 geometric cards size elements without explicit values from width, length, sheet resistance or area/perimeter capacitances, narrowing/shortening offsets, and defaults matching ngspice.
- Independent sources with DC levels or `PULSE`, `SIN`, `PWL`, and `EXP` waveforms (in parenthesized or SPICE2 trailing-argument syntax), using the time-zero waveform value for non-transient analyses.
- Level 1 diodes, JFETs, BJTs, and level 1 and 3 MOSFETs with model cards, positive areas and multipliers, and `AKO:` model inheritance.
- Voltage- or current-dependent resistors mapped to ngspice-equivalent behavioral currents `m · V(a, b) / R` scaled by `1 + TC1 · ΔT + TC2 · ΔT²` (referenced to 27 °C, noiseless as in ngspice).
- Charge-defined `Q=` capacitors mapped directly to charge-conserving [behavioral capacitors](devices.md#behavioral-sources) scaled by multiplier `m`.
  Voltage-dependent `C=` capacitors integrate `C(v)` analytically from 0 to `v` into a behavioral charge when `C(v)` is a polynomial plus scaled `exp`, `sinh`, `cosh`, `tanh`, `sin`, or `cos` of affine voltage terms.
  Expressions depending on other voltages or currents, non-integrable expressions, and unsupported temperature/voltage/geometry modifiers are rejected by name; ngspice's 1e15 Ω shunt and 1 fs differentiation lag are omitted ([departures](departures.md)).
- Linear controlled sources along with `POLY`, `VALUE`, and `TABLE` variants.
- `B` behavioral sources (`V=` or `I=`) capable of referencing any voltage-source current, including self-currents.
- PSpice `VSWITCH` (`S`) and `ISWITCH` (`W`) switches, with `W` sensing voltage-source currents identically to current-controlled sources.
- Subcircuit instances with parameter overrides, `.param` statements, and instance multipliers `m` (when not shadowed by a subcircuit parameter).
  Subcircuit multipliers propagate as in ngspice: dividing resistances and inductances, multiplying capacitances, scaling semiconductor/subcircuit multiplicities, and rejecting multiplied current sources.
- Open-PDK directives detailed in the [foundry guide](foundry.md#the-open-process-kits): top-level and subcircuit `.include`/`.lib` sections (relative paths with cycle checks), `.option scale`, `.if` blocks, subcircuit parameters, `temper`, and statistical functions (`agauss`, `gauss`, `aunif`, `unif`, two-argument `limit`).
- Top-level and subcircuit `.ic` and `.nodeset` declarations (`V(node)=value`) alongside device `IC=` on linear/behavioral capacitors and inductors.
  Top-level declarations populate `initial_conditions` and `nodesets`, while `library.initial_conditions_of(...)` and `library.nodesets_of(...)` resolve hierarchical subcircuit voltages onto parent nets.
  With `operating_point=False` (`UIC`), capacitor `IC=` voltages solve alongside `.ic` while rejecting contradictions and non-zero inductor `IC=` currents (`library.initial_currents_of(...)` reports scaled inductor initial currents, naming coupled windings `K1:current1` and onward).

Parameters may reference one another in any order, and expressions recognize PSpice constants `TEMP`, `VT`, and `GMIN`.
Top-level and subcircuit-scoped `.func name(parameters) {expression}` definitions (including inside `.if` blocks) expand inline across parameters, `.if` conditions, elements, expressions, `TABLE` sources, instance arguments, and `.model` cards (covering `AKO:`, binned MOSFETs, geometric `R`/`C`, and VBIC self-heating).
Local `.func` definitions stay scoped to their subcircuit and shadow global functions without mutating the library.
Functions may call other functions, while direct or mutual recursion fails with every participating function named.
Expressions translate PSpice functions with ngspice semantics into [jolty's expression language](../reference/expressions.md): `PWR(x, y)` evaluates `|x|^y`, and `POW(x, y)` and `x ** y` preserve sign for odd integer powers of negative bases.
Inline `TABLE` expressions extrapolate beyond endpoints, whereas `TABLE` sources clamp at endpoints with rounded corners matching ngspice XSPICE `pwl`.
Switches follow ngspice's log-cubic transition law between on and off thresholds.
Waveforms must specify parameters SPICE otherwise borrows from the transient step: positive `PULSE` rise/fall/width, `SIN` frequency, and positive `EXP` delays and time constants.
For example, `EXP(0 1 0 1n 6n 2n)` holds 0 V until the 0.5 ns step under `tran 0.5n 12n`.
A source specifying both a DC value and a waveform must match the waveform's time-zero value, and unsupported `SFFM` and `AM` waveforms raise errors.
Element or node names containing `/` are rejected because jolty uses `/` as its hierarchy separator.
Bipolar transistors connect their substrate to ground unless a fourth instance node is given, attaching to collector (NPN) or base (PNP) unless overridden by `SUBS`.

Elements sensing a voltage-source current connect through series sense ports (`V1.sense1`, `V1.sense2`, and so on for `V1`).
Inductors linked by mutual inductances merge into a single [coupled inductors](devices.md#coupled-inductors) device preserving file winding order and taking the first `K` element's name and line (`K1 L1 L2 0.99` yields `K1` with currents `K1:current1` and `K1:current2`).
Mutual inductances accept any number of inductors (as in PSpice), coupling every pair by the given coefficient; magnetic-core models raise an error.

Temperature parameters map directly to device fields: `TNOM` (PSpice `T_MEASURED`, and HSPICE `TREF` on diodes, bipolar transistors, and resistors) sets nominal temperature, and `TC=tc1,tc2`, `TC1`, `TC2`, and `TCE` set passive temperature drift.
Resistor and capacitor voltage coefficients (`VC1`, `VC2` from HSPICE and Spectre) export to ngspice as behavioral expressions because ngspice lacks those parameters.
Per-device temperatures come from model cards (`T_ABS`, `T_REL_GLOBAL`) or elements (`TEMP`, `DTEMP`), mapping to [`held_at` and `warmer_by`](devices.md) (for example, `T_ABS=-273.15` holds macromodel resistors noiseless at 0 K).
PSpice `T_REL_LOCAL` and duplicate per-device temperature declarations fail; ngspice exports emit `TEMP` or `DTEMP`.
SPICE expressions evaluate `TEMP` and `VT` at 27 °C regardless of simulation temperature.
`KF` and `AF` configure flicker noise on diodes, transistors, and resistors; PSpice `NLEV` and `GDSNOI` are ignored (matching ngspice), and non-zero `KF` on capacitors or inductors raises an error.
Any unrecognized parameter or element fails with file, line, and statement details; nothing is silently dropped.

## Agreement with ngspice

Differential tests load vendor macromodels from ngspice examples and compare every node voltage and branch current against ngspice on the same file:

| Part | Maker | Analyses |
| --- | --- | --- |
| TL072 | Texas Instruments | operating point, DC sweep, AC, transient |
| MCP6041 | Microchip | operating point, DC sweep, AC, transient |
| AD22057 | Analog Devices | operating point, DC sweep, AC, transient |
| OPA171 | Texas Instruments | operating point |

ngspice runs PSpice switches through XSPICE, which negates small-signal AC conductance; OPA171 frequency responses loaded by ESD switches differ from jolty by that ngspice defect.
Operating points also agree at -40 °C and 125 °C except AD22057 at 125 °C, where ngspice scales BJT base-collector saturation current by area twice and amplifies high-temperature leakage.

## Writing SPICE

`jolty.spice.export` exports jolty circuits as SPICE netlists for ngspice validation:

<!-- not run: it exports a circuit `bench`, which the page does not build, and runs ngspice -->

```python
from jolty import circuit
from jolty.spice import export, ngspice

lines = export.netlist(circuit(bench()), excited="V1")
raw = ngspice.run(lines, "ac dec 20 10 1meg")
raw.vectors[export.vector(drawn, unknown)]  # ngspice's values of a jolty unknown
```

Export flattens hierarchy into underscore-joined names (`RX1_R1`, `x1_mid`), while preserving imported SPICE subcircuit instances listed in `subcircuits`.
Behavioral sources and switches export as ngspice `B` sources implementing the exact switch law; coupled inductors export as `L` and `K` elements, foundry transistors instantiate [kit](foundry.md) subcircuits, and model cards convert parameters to SPICE units.
ngspice interprets zero pulse width as the transient stop time, so zero-width pulses export as `1e-40` s (negligible compared to rise time).
Logic devices have no SPICE equivalent, so circuits containing logic cannot export to SPICE.
`jolty.spice.ngspice.run` executes one analysis with matched tolerances at the thermal-voltage-matched temperature and parses the raw output.
The [gallery](../gallery.md) replicates every analog example in ngspice using this export pipeline.

`export.netlist(..., dialect=export.XYCE)` emits netlists for Sandia's Xyce 7.10.0, executed as a separate GPL process via `jolty.spice.xyce.run` to collect printed vectors and reported runtimes.
The Xyce dialect encodes exact source-verified differences from ngspice: model parameter names, default 0 DC source levels, zero-width pulses, and SPICE3 thermal-voltage temperature offsets.
Unsupported or differently modeled features fail by name without silent omission: BSIM3 (Xyce uses 3.2.2), VDMOS, VBIC, excess phase, unrecognized parameters, and behavioral-expression exports.
`tests/test_xyce.py` compares every shared feature while documenting known Xyce model departures: level 1/3 MOSFET reverse bulk conductance `Is/vt` (vs `-Is` saturation), noiseless SPICE3f5 JFETs, and pre-beta BJT gmin injection.
Xyce fails to find an operating point on `negative_load` after 7,692 Newton iterations, and two level 1 MOSFET discrepancies up to `1.4e-4` (at 125 °C and across an inverter threshold) remain under investigation.
