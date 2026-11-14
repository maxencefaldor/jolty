# Devices

The module `jolty.devices` holds every device a script can place.
Each function checks its parameters' units and returns a device ready to place; parameters left out take the engine's defaults, which the [models reference](../reference/models.md) lists with every parameter of every model.

Devices are simulated at the plan's temperature, 27 °C unless it gives one, and each device's parameters were measured at its `nominal_temperature`, 27 °C unless given.
Away from it, passive values drift by their temperature coefficients, and semiconductors scale as ngspice scales them, as the sections below say.
A device that [heats](#self-heating) runs at its own temperature instead, which the power it dissipates raises.

Any device may run at a temperature of its own, as on a hot board: `device.held_at(celsius(85))` holds it there whatever the plan's, as SPICE's `TEMP` and PSpice's `T_ABS` do, and `device.warmer_by(20 * K)` holds it that far above the plan's, below where negative, as SPICE's `DTEMP` and PSpice's `T_REL_GLOBAL` do.
Its parameters scale to that temperature, its noise is at it, a heated device warms from it, and its temperature rating holds it there, though sensitivities do not differentiate it.
A resistor held at 0 K makes no noise, while a junction needs a temperature above zero.

Logic devices, whose pins join logic nets, have a page of their own: [Logic](logic.md).

## Passive devices

| Function | Terminals | Parameter |
| --- | --- | --- |
| `resistor(resistance)` | `a`, `b` | a positive resistance |
| `capacitor(capacitance)` | `a`, `b` | a capacitance, zero or more |
| `inductor(inductance)` | `a`, `b` | a positive inductance |

An inductor adds its current to the unknowns, as `L1:current` for an inductor `L1`: the current from `a` through the inductor to `b`.
So does a resistor below 1 Ω, as `R1:current`: beside its conductance, a node's picosiemens, such as gmin's, would round away in floating point, and the current unknown keeps them, as modified nodal analysis does for the elements of its second group.

Each value drifts with temperature by `1 + tc1 · ΔT + tc2 · ΔT²`, where `ΔT` is the temperature less the nominal one, as SPICE's do:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `temperature_coefficient` | `TC1` | K⁻¹ | 0 |
| `quadratic_temperature_coefficient` | `TC2` | K⁻² | 0 |
| `exponential_temperature_coefficient` | `TCE` | percent per kelvin, a number | none: a resistor drifts by `1.01^(tce · ΔT)` instead |
| `nominal_temperature` | `TNOM` | K | 27 °C |

Only a resistor has an exponential coefficient, and a resistance the coefficients scale to zero is a problem.

A resistor's or capacitor's value also bends with the voltage `v` from `a` to `b`, by `1 + vc1 · v + vc2 · v²`, as ngspice's expressions would write it:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `voltage_coefficient` | `VC1` | V⁻¹ | 0 |
| `quadratic_voltage_coefficient` | `VC2` | V⁻² | 0 |

A resistor then carries `v / (r · (1 + vc1 · v + vc2 · v²))`, and a capacitor holds the charge `c · (v + vc1 · v² / 2 + vc2 · v³ / 3)`, whose slope is its bent capacitance.
The formula holds at every voltage, as in ngspice: where the coefficients bend a value below zero, it is negative there.

In a [noise analysis](simulation.md#noise), a resistor makes thermal noise, `4kT/R` A²/Hz at its resistance there, or a voltage of `4kT·R` V²/Hz in series below 1 Ω, as a switch, a photocell and a memristor do at theirs; capacitors and inductors make none.
Given a flicker coefficient `KF`, `flicker_coefficient`, a resistor's current `I` also makes flicker noise of `KF · |I|^AF / f` A²/Hz across it, `AF` its `flicker_exponent`, 1 unless given, as ngspice's resistor does.

A `tolerance`, such as `0.01` for 1 %, spreads each value evenly within it, and `device.toleranced(parameter, relative, distribution, lot)` gives any device's number parameter a `"uniform"`, `"normal"`, `"log_uniform"` or `"log_normal"` tolerance, by its path, such as `waveform.dc` for a source's DC level, and optionally a lot it shares a spread with; [Monte Carlo analyses](simulation.md#monte-carlo) sample them, and [worst-case analyses](simulation.md#worst-case) set them at their extremes.

### Coupled inductors

`coupled_inductors(windings, couplings)` links inductors' fluxes, as a transformer's windings or a common-mode choke's: each winding is an `inductor(...)`, with its tolerance and drift, and winding `k`, from 1, joins terminals `a{k}` and `b{k}`, its current the unknown `current{k}`.
`couplings` gives the coefficient between pairs of windings, by their numbers, from -1 to 1; their mutual inductance is the coefficient times the root of their inductances' product, and currents into both `a` terminals add their fluxes where it is positive, as SPICE's `K` couples its inductors' first nodes.

```python
transformer = c.place(
    devices.coupled_inductors(
        [devices.inductor(1 * mH), devices.inductor(4 * mH)], {(1, 2): 0.99}
    ),
    id="T1",
)
```

Each winding is a short at DC, so a winding across a voltage source is a loop the [checks](checks.md) flag.

## Ratings

Any device may state the limits its maker warrants it within, as `device.rated(stress, maximum, minimum=..., terminals=...)`: a `"voltage"` of one of its terminals over another, a `"current"` into one, the `"power"` it dissipates, or its `"temperature"`.
A voltage is of the device's first two terminals by default, in order, and a current into its first; the power is what its electrical terminals take, and the temperature the simulation's, raised by its thermal net's where it [heats](#self-heating).
A rating bounds its stress from above, from below or both, as its maker states it; a power's only from above, and a temperature's in kelvins, as `celsius(150)`.

```python
devices.resistor(100 * ohm).rated("power", 0.25 * W)
devices.capacitor(10 * uF).rated("voltage", 25 * V, minimum=-25 * V)
devices.diode().rated("voltage", minimum=-100 * V)
switch = devices.bipolar_transistor().rated(
    "voltage", 40 * V, terminals=("collector", "emitter")
)
switch.rated("power", 0.625 * W).rated("temperature", celsius(150))
```

A component rates its ports the same way, `component.rated(...)`, and its power is all it dissipates.
`device.numbered("2N3904")` and `component.numbered(...)` name the part an instance stands for, which a [bill of materials](parts.md#bills-of-materials) prices it by; [`jolty.parts`](parts.md#real-parts) holds real parts, each with its model, its number and its ratings.
A simulation holds every rating to every operating point and transient its plan solves, as the [checks](checks.md#on-analyses-results) describe.

## Independent sources

| Function | Terminals | Parameter |
| --- | --- | --- |
| `voltage_source(waveform)` | `positive`, `negative` | a voltage, or a waveform of voltages |
| `current_source(waveform)` | `positive`, `negative` | a current, or a waveform of currents |

A voltage source holds `positive` minus `negative` at its level.
Its current, from `positive` through the source to `negative`, is the unknown `V1:current` for a source `V1`: negative while the source delivers power.

A current source drives its level through itself from `positive` to `negative`, so the current enters the circuit at `negative`.

A plan names a source to sweep it or to excite it for a frequency response; see [Simulation](simulation.md).

## Waveforms

A source's level is a constant quantity or a waveform from `jolty.waveforms`.
Every analysis but a transient reads the level at time zero.

- `Dc(level)`: a constant level, as passing the quantity itself.
- `Pulse(initial, pulsed, rise, fall, width, delay=0 s, period=None)`: a trapezoidal pulse from `initial` to `pulsed` and back, once, or every `period`.
- `Sine(offset, amplitude, frequency, delay=0 s, damping=0 Hz, phase=0)`: `offset + amplitude·exp(-damping·t')·sin(2π·frequency·t' + phase)`, where `t'` is the time since the delay and the phase is in radians; before the delay the level holds its value at `t' = 0`.
- `Piecewise(points)`: straight lines through `(time, level)` points, holding the first level before them and the last after.

The transient analysis lands a time step on every corner of a waveform, so a pulse's edges are never stepped over.

## Controlled sources

| Function | Terminals | Parameter |
| --- | --- | --- |
| `voltage_controlled_voltage_source(gain)` | `positive`, `negative`, `control_positive`, `control_negative` | a gain, a plain number |
| `voltage_controlled_current_source(transconductance)` | the same | a transconductance |
| `current_controlled_current_source(gain)` | the same | a gain, a plain number |
| `current_controlled_voltage_source(transresistance)` | the same | a transresistance |

A voltage-controlled source's control voltage is `control_positive` minus `control_negative`, and its control terminals draw no current.
A current-controlled source's control port, as in Modelica, is a short circuit from `control_positive` to `control_negative` that senses the current through it: placed in series with a branch, it senses that branch's current, which it adds to the unknowns as `F1:control` for a source `F1`.

A voltage-controlled or current-controlled voltage source holds its output at the gain or transresistance times its control, and adds its current to the unknowns as a voltage source does.
A current source of either kind drives the transconductance or gain times its control from `positive` through itself to `negative`.

## Behavioral sources

`behavioral_voltage_source(expression)` holds `positive` minus `negative` at an expression, and `behavioral_current_source(expression)` drives an expression from `positive` through itself to `negative`.
The expression reads `v(a)`, the voltage of terminal `a`, `v(a, b)`, that of `a` over `b`, `i(p)`, the current through a sensing port `p`, and `time`, in a [language](../reference/expressions.md) of arithmetic, comparisons, conditionals and functions:

```python
gain = devices.behavioral_voltage_source("limit(1e5 * v(inp, inn), -12, 12)")
amplifier = c.place(gain)
c.connect(amplifier.inp, signal)
```

A source's terminals are `positive` and `negative`, then each terminal its expression reads, then `p_positive` and `p_negative` for each port `p`.
A port, as a current-controlled source's control port, is a short circuit whose current, from `p_positive` to `p_negative`, it adds to the unknowns as `B1:p` for a source `B1`.
A behavioral voltage source adds its own current to the unknowns as a voltage source does.
The engine takes every derivative of the expression exactly, so Newton's iterations and frequency responses see the expression's true slope.

Written for ngspice, a behavioral source's expression is read into its tree and written in ngspice's language, every operation in parentheses, so ngspice computes what jolty does: ngspice groups `^` to the left and raises a negative number's magnitude, so a power is written with ngspice's `pow`, or `pwr` for an odd constant exponent, which keeps the base's sign; `limit` is written with `min` and `max`, ngspice's own `limit` being another function; and a table, which ngspice's behavioral sources lack, is written as the choice between its segments' lines and its corners' parabolas.

## Switches

`voltage_controlled_switch(...)` is a resistance between `positive` and `negative` that the voltage of `control_positive` over `control_negative` moves, in PSpice's model, from `off_resistance` to `on_resistance`: logarithmically and smoothly, with zero slope at each end, as the control moves from `off_voltage` to `on_voltage`.
Beyond either voltage, the resistance holds.
The control terminals draw no current.

| Parameter | PSpice | Unit | Default |
| --- | --- | --- | --- |
| `on_resistance`, `off_resistance` | `RON`, `ROFF` | Ω | 1 Ω, 1 MΩ |
| `on_voltage`, `off_voltage` | `VON`, `VOFF` | V | 1 V, 0 V |

`current_controlled_switch(...)` is the same switch moved by a current: its control port, from `control_positive` to `control_negative`, is a short circuit whose current is an unknown labelled `control`, as a current-controlled source's is, and moves the resistance from `off_current` to `on_current`, as PSpice's `ISWITCH` does.

| Parameter | PSpice | Unit | Default |
| --- | --- | --- | --- |
| `on_resistance`, `off_resistance` | `RON`, `ROFF` | Ω | 1 Ω, 1 MΩ |
| `on_current`, `off_current` | `ION`, `IOFF` | A | 1 mA, 0 A |

A netlist written for ngspice gives each switch as a behavioral current of its own law, reading a current-controlled switch's control through a zero-volt source: ngspice's XSPICE switches follow the same law, but ngspice 42 gives their small-signal conductance the opposite sign of their DC one.

## Photocell

`photocell(resistance, ...)` is a light-dependent resistor between `a` and `b`, as its maker's datasheet states one: its resistance at a reference illuminance and how steeply it falls with light.
The voltage of `light_positive` over `light_negative` is the illuminance on it, 1 V for each lux, and those terminals draw no current, so a source, or a behavioral source that turns a lamp's or an LED's current into light, drives it.
Its conductance is `1/r_dark + s/r`, where `s = (E/E₀)^γ` at the illuminance `E`, held at zero or above and rounded off over a microlux so its slope stays finite in darkness.
With a response time `τ`, `s` lags the light, `τ · ds/dt = (E/E₀)^γ − s`, as a cadmium sulphide cell's conductance does over milliseconds, and is a state labelled `response`, as `P1:response`, which results read with `state(name)`.

| Parameter | Unit | Default |
| --- | --- | --- |
| `resistance` | Ω, at the reference illuminance | none |
| `illuminance` | lux, a number | 10 |
| `gamma` | a number | 0.7 |
| `dark_resistance` | Ω | 1 MΩ |
| `response_time` | s | 0: the cell follows the light at once |

SPICE has no photocell; written for ngspice, it is a behavioral current whose response lags through 1 Ω and a capacitance of its response time.

## Memristors

A memristor is a resistance between `positive` and `negative` that remembers what drove it.
Its state, from 0, fully on, at `on_resistance`, to 1, fully off, at `off_resistance`, moves only while it is driven, and holds where the drive left it.
The resistance follows the state by a `law`: `"linear"`, `r_on + (r_off − r_on) · x`, as two resistances in series whose boundary moves, or `"exponential"`, `r_on · (r_off / r_on)^x`, as a tunnel barrier whose width moves.
The state is an unknown labelled `state`, as `M1:state`, which results read with `state(name)`, and the resistance and the rate at which the state moves, toward off when positive, in full ranges per second, are the [variables](results.md#devices-own-models) `resistance` and `rate`.
A window slows the state as it nears the bound it moves toward, by `1 − e^(−d/w)` at a distance `d` from it, `w` the `window`, so it never passes either bound and leaves either at once when driven away.

Only time moves a state.
At an operating point, and in the DC sweeps, small-signal responses and noise about one, a memristor's state is the one it was made in, its `state`, as SPICE's initial conditions hold a node: the memristor is a resistor there, and its `rate` says how fast a transient would move it.
A transient starts from that state.
A [periodic](simulation.md#periodic-steady-state) transient finds one periodic steady state: when the drive returns a memristor's state each period, as a symmetric one does, any state it starts from repeats, and the one found need not be the state it was made in.

`voltage_controlled_memristor(...)` is VTEAM's memristor (Kvatinsky, Ramadan, Friedman and Kolodny, 2015), whose state a voltage moves only past a threshold, as a memory's cell holds under the voltage that reads it.
A voltage `v` of `positive` over `negative` above `off_voltage` moves the state toward off at `off_rate · (v/v_off − 1)^α_off`, and one below `on_voltage`, a negative voltage, toward on at `on_rate · (v/v_on − 1)^α_on`; between them, the state holds.
Each rate is the state's speed at twice its threshold, in full ranges per second: VTEAM's `k_off`, or `−k_on`, over the range of its state variable, `w_off − w_on`.

| Parameter | VTEAM | Unit | Default |
| --- | --- | --- | --- |
| `on_resistance`, `off_resistance` | `R_ON`, `R_OFF` | Ω | none |
| `on_voltage`, `off_voltage` | `v_on`, `v_off` | V | none |
| `on_rate`, `off_rate` | `−k_on`, `k_off` over `w_off − w_on` | 1/s | none |
| `on_exponent`, `off_exponent` | `α_on`, `α_off` | a number, at least 1 | 1 |
| `law` | linear or exponential | | `"linear"` |
| `state` | `(w − w_on) / (w_off − w_on)` | from 0 to 1 | 1, off |
| `window` | | a fraction of the state's range | 0.01 |

`current_controlled_memristor(...)` is the linear ion drift memristor of HP's titanium dioxide cell (Strukov, Snider, Stewart and Williams, 2008), whose state the charge through it moves, as its dopants drift.
A charge `q` from `positive` to `negative` moves the state toward off by `q/Q`, where `Q`, its `charge`, carries it across its whole range; with a linear law, it is Chua's ideal memristor, whose flux is a function of its charge alone.
Its `on_resistance`, `off_resistance`, `law`, `state` and `window` are as above.

Written for ngspice, a memristor is a behavioral current whose state is the voltage of a node of its own.
In a transient's netlist, that node is over 1 F, charged by a behavioral current at the state's rate, with an initial condition that starts the transient from the state it was made in; in any other analysis's netlist, a source holds the node at that state, as jolty holds it at an operating point and in the analyses about one.

## Thermal networks

A thermal network carries heat as a circuit carries current: its nets are temperatures above the ambient, in kelvins, its flows heat, in watts, and the ground is the ambient itself.
Its devices have thermal terminals, which join thermal nets or the ground, and never a net an electrical terminal or a logic pin joins.

| Function | Terminals | Parameter |
| --- | --- | --- |
| `thermal_resistance(resistance)` | `a`, `b` | K/W: heat flows from the hotter terminal to the colder at their difference over it |
| `thermal_capacitance(capacitance)` | `a`, `b` | J/K: heat it stores, in proportion to their difference |
| `heat_source(power)` | `a`, `b` | W: a constant flow from `a` into `b` |

```python
heat = c.place(devices.heat_source(1 * W), id="P1")
sink = c.place(devices.thermal_resistance(50 * K / W), id="RTH")
c.connect(heat.b, sink.a, name="hot")
c.connect(heat.a, sink.b, ground)
```

A thermal net's unknown is its temperature above the ambient, which results read with `rise(net)`: 50 K here.
Results read the heat flowing into a thermal terminal as its current, `current(instance, terminal)`, in watts.

### Self-heating

`device.heated()` heats a resistor, a diode, a JFET, a bipolar transistor or a MOSFET by the power it dissipates: a terminal `thermal` joins it to a thermal net, or to the ground, the ambient.
The device runs at the simulation's temperature raised by that net's, every parameter scaled to it as the sections below say, and the power its electrical terminals take, each one's voltage times the current into it, flows into the net as heat.
So a transistor whose junction a thermal resistance holds to the ambient settles where the heat it makes flows away, and a thermal capacitance beside it sets how fast it gets there:

```python
transistor = c.place(devices.bipolar_transistor().heated(), id="Q1")
mount = c.place(devices.thermal_resistance(200 * K / W), id="RTH")
c.connect(transistor.thermal, mount.a, name="junction")
c.connect(mount.b, ground)
```

Every analysis sees the heating: an operating point balances it, a transient follows it, and a small-signal response carries its thermal poles, below which a resistor with a temperature coefficient takes a resistance of its own.
A heated device keeps the form its value took at the simulation's temperature: a resistor below 1 Ω there carries its current as an unknown however hot it runs.
ngspice heats no such device, so a circuit that holds one is not written as a SPICE netlist.

## Diode

`diode(...)` is a junction diode from `anode` to `cathode`, in the SPICE model.
Every parameter is optional:

| Parameter | Unit | Default |
| --- | --- | --- |
| `saturation_current` | A | 10 fA |
| `emission_coefficient` | none | 1 |
| `series_resistance` | Ω | 0 |
| `junction_capacitance` | F | 0 |
| `junction_potential` | V | 1 V |
| `grading_coefficient` | none | 0.5 |
| `depletion_coefficient` | none | 0.5 |
| `transit_time` | s | 0 |
| `breakdown_voltage` | V | none: no breakdown |
| `breakdown_current` | A | 1 mA |
| `breakdown_emission_coefficient` | none | the emission coefficient |
| `breakdown_temperature_coefficient` | V/K | 0 |
| `energy_gap` | V | 1.11 V, silicon's |
| `saturation_current_temperature_exponent` | none | 3 |
| `nominal_temperature` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | none | 0: no flicker noise, 1 |

A diode with a series resistance adds the voltage of its junction to the unknowns, as `D1:junction` for a diode `D1`; below 1 Ω, the resistance adds its current too, as `D1:junction_current`, as a small resistor does.

The junction follows SPICE as ngspice implements it, its reverse current settling on the saturation current below three thermal voltages of reverse bias.
Given a breakdown voltage, the reverse current grows exponentially beyond it; ngspice moves the voltage slightly so the current there is the breakdown current, and jolty does the same.

Away from the nominal temperature `Tn`, the saturation current scales by `(T/Tn)^(XTI/n) · exp((T/Tn − 1) · Eg / (n · Vt))`, with `Eg` the energy gap, `XTI` its temperature exponent and `Vt` the thermal voltage at `T`.
The junction potential and capacitance follow silicon's band gap, and the breakdown voltage falls by its temperature coefficient, as in ngspice.

In a [noise analysis](simulation.md#noise), the series resistance makes thermal noise, `4kT/R` A²/Hz, and the junction's current `I` shot noise, `2q·|I|`, and flicker noise, `KF·|I|^AF/f`, `KF` the flicker coefficient and `AF` its exponent, `KF` and `AF` in SPICE.

## JFET

`jfet(...)` is a junction field-effect transistor between `drain`, `gate` and `source`, in the SPICE model with Sydney University's doping tail, as ngspice implements it.
Two junctions join the gate to the channel, and the drain current follows the square law above the threshold voltage, its drain and source swapping when the drain falls below the source.
Every parameter is optional:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `channel` | `NJF`, `PJF` | `"n"` or `"p"` | `"n"` |
| `threshold_voltage` | `VTO` | V | -2 V |
| `transconductance_parameter` | `BETA` | A/V² | 100 µA/V² |
| `channel_length_modulation` | `LAMBDA` | V⁻¹ | 0 |
| `drain_resistance` | `RD` | Ω | 0 |
| `source_resistance` | `RS` | Ω | 0 |
| `gate_source_capacitance` | `CGS` | F | 0 |
| `gate_drain_capacitance` | `CGD` | F | 0 |
| `junction_potential` | `PB` | V | 1 V |
| `saturation_current` | `IS` | A | 10 fA |
| `depletion_coefficient` | `FC` | none | 0.5 |
| `doping_tail` | `B` | none | 1, the plain square law |
| `threshold_temperature_coefficient` | `VTOTC`, or `−TCV` | V/K | 0 |
| `beta_temperature_exponent` | `BEX` | none | 0 |
| `energy_gap` | `EG` | V | 1.11 V, silicon's |
| `saturation_current_temperature_exponent` | `XTI` | none | 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |

A drain or source resistance adds an internal node to the unknowns, as `J1:drain` or `J1:source` for a JFET `J1`; one below 1 Ω adds its current too, as `J1:drain_current` or `J1:source_current`, as a small resistor does.

Away from the nominal temperature, the gate junctions' saturation current scales with the energy gap as a diode's, and their potential and capacitances follow silicon's band gap; the threshold moves by its temperature coefficient, and the transconductance parameter scales as `(T/Tn)^BEX`.
ngspice scales a JFET's saturation current by its temperature exponent only when given, so it defaults to 0.

In a [noise analysis](simulation.md#noise), the resistances make thermal noise, the channel thermal noise of `8kT·gm/3` A²/Hz, and the drain current `Id` flicker noise of `KF·|Id|^AF/f`, as in ngspice.

## Bipolar transistor

`bipolar_transistor(...)` is a bipolar junction transistor between `collector`, `base` and `emitter`, in the Gummel-Poon model of SPICE, as ngspice implements it.
The transport current falls with the base charge, which models the Early effect and high injection, and the base resistance falls from `base_resistance` towards `minimum_base_resistance` as the base charge or the base current grows.
Every parameter is optional; an Early voltage, knee current or transit-time voltage left out is infinite:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `polarity` | `NPN`, `PNP` | `"npn"` or `"pnp"` | `"npn"` |
| `saturation_current` | `IS` | A | 0.1 fA |
| `forward_beta`, `reverse_beta` | `BF`, `BR` | none | 100, 1 |
| `forward_emission_coefficient`, `reverse_emission_coefficient` | `NF`, `NR` | none | 1, 1 |
| `forward_early_voltage`, `reverse_early_voltage` | `VAF`, `VAR` | V | infinite |
| `forward_knee_current`, `reverse_knee_current` | `IKF`, `IKR` | A | infinite |
| `base_emitter_leakage_current`, `base_emitter_leakage_emission_coefficient` | `ISE`, `NE` | A, none | 0, 1.5 |
| `base_collector_leakage_current`, `base_collector_leakage_emission_coefficient` | `ISC`, `NC` | A, none | 0, 2 |
| `base_resistance`, `minimum_base_resistance` | `RB`, `RBM` | Ω | 0, the base resistance |
| `base_resistance_current` | `IRB` | A | none: the resistance falls with the base charge |
| `emitter_resistance`, `collector_resistance` | `RE`, `RC` | Ω | 0 |
| `base_emitter_capacitance`, `base_emitter_potential`, `base_emitter_grading` | `CJE`, `VJE`, `MJE` | F, V, none | 0, 0.75 V, 0.33 |
| `forward_transit_time` | `TF` | s | 0 |
| `transit_time_bias_coefficient`, `transit_time_voltage`, `transit_time_current` | `XTF`, `VTF`, `ITF` | none, V, A | 0, infinite, 0 |
| `base_collector_capacitance`, `base_collector_potential`, `base_collector_grading` | `CJC`, `VJC`, `MJC` | F, V, none | 0, 0.75 V, 0.33 |
| `internal_base_fraction` | `XCJC` | none | 1 |
| `reverse_transit_time` | `TR` | s | 0 |
| `depletion_coefficient` | `FC` | none | 0.5 |
| `excess_phase` | `PTF` | degrees | 0 |
| `substrate` | `SUBS`, 1 or −1 | `"vertical"` or `"lateral"` | none: no substrate |
| `substrate_saturation_current`, `substrate_emission_coefficient` | `ISS`, `NS` | A, none | none: no diode, 1 |
| `substrate_capacitance`, `substrate_potential`, `substrate_grading` | `CJS`, `VJS`, `MJS` | F, V, none | 0, 0.75 V, 0 |
| `energy_gap` | `EG` | V | 1.11 V, silicon's |
| `saturation_current_temperature_exponent`, `beta_temperature_exponent` | `XTI`, `XTB` | none | 3, 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |

A base, collector or emitter resistance adds an internal node to the unknowns, as `Q1:base`, `Q1:collector` or `Q1:emitter` for a transistor `Q1`; one below 1 Ω, the base's at its minimum, adds its current too, as `Q1:base_current`, `Q1:collector_current` or `Q1:emitter_current`, as a small resistor does.

An excess phase delays the forward transport current, `Ibe / qb`, by `td = PTF · π / 180 · TF` through the second-order Bessel filter `3 / (s² td² + 3 s td + 3)`, the filter ngspice integrates in a transient, so its phase at `1 / (2π · TF)` is nearly `PTF`.
The filter adds two unknowns, `Q1:delayed_transport` and `Q1:delayed_transport_slope`, and holds in every analysis, so a pole-zero analysis finds its two poles.
ngspice's AC analysis delays the transconductance by a pure `exp(−jω·td)` instead, so at `1 / (2π · TF)` the two engines' gains differ by the filter's magnitude there, 5 % at a `PTF` of 30°.

A transistor built `"vertical"` or `"lateral"` has a fourth terminal, `substrate`, joined by a junction to its internal collector if vertical or to its internal base if lateral.
The junction conducts the minimum conductance, as ngspice's does, and a diode's current given a substrate saturation current; its depletion capacitance grows linearly in forward bias.
The substrate parameters need a substrate.

Away from the nominal temperature, the saturation currents scale with the energy gap as a diode's, the leakage currents with their own emission coefficients; the betas scale as `(T/Tn)^XTB`, dividing the leakage currents too, and the junctions' potentials and capacitances follow silicon's band gap, as in ngspice.

In a [noise analysis](simulation.md#noise), the resistances make thermal noise, the base resistance at its bias, the collector current shot noise, and the base current `Ib` shot noise and flicker noise of `KF·|Ib|^AF/f`, as in ngspice.

ngspice leaves out of its small-signal model and its Newton steps how a falling base resistance depends on the junction voltages; jolty keeps every derivative, so its frequency response is the exact slope of its DC model, and its Newton steps converge faster to the same operating point.

## MOSFET

`mosfet(...)` is a metal-oxide-semiconductor field-effect transistor between `drain`, `gate`, `source` and `bulk`, in SPICE's level 1 model, Shichman and Hodges's square law with Meyer's gate capacitances, or with `level=3` in SPICE's semi-empirical level 3 model, as ngspice implements them.
Above the threshold, the drain current follows the square law, which the body effect raises as the source rises above the bulk; the drain and the source swap when the drain falls below the source.
Two junctions join the bulk to the drain and the source, and the gate holds overlap charges and, given an oxide thickness, Meyer's capacitances.
With an oxide thickness and a substrate doping, the threshold, the body effect and the surface potential left out derive from the process, as in SPICE.
Every parameter is optional:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `channel` | `NMOS`, `PMOS` | `"n"` or `"p"` | `"n"` |
| `width`, `length` | `W`, `L` | m | 100 µm, 100 µm |
| `multiplier` | `M` | none | 1 |
| `threshold_voltage` | `VTO` | V | 0, or derived |
| `transconductance_parameter` | `KP` | A/V² | 20 µA/V², or the surface mobility times the oxide capacitance per area |
| `body_effect` | `GAMMA` | √V, as a number | 0, or derived |
| `surface_potential` | `PHI` | V | 0.6 V, or derived |
| `channel_length_modulation` | `LAMBDA` | V⁻¹ | 0, at level 1 only |
| `lateral_diffusion` | `LD` | m | 0 |
| `oxide_thickness` | `TOX` | m | none: no Meyer's capacitances |
| `surface_mobility` | `UO` | m²/(V·s) | 0.06 m²/(V·s), SPICE's 600 cm²/(V·s) |
| `substrate_doping` | `NSUB` | m⁻³ | none |
| `surface_state_density` | `NSS` | m⁻² | 0 |
| `gate_material` | `TPG` | `"opposite_doping"`, `"same_doping"` or `"aluminum"` | `"opposite_doping"` |
| `drain_resistance`, `source_resistance` | `RD`, `RS` | Ω | the sheet resistance's |
| `sheet_resistance`, `drain_squares`, `source_squares` | `RSH`, `NRD`, `NRS` | Ω, none, none | 0, 1, 1 |
| `saturation_current`, `saturation_current_density` | `IS`, `JS` | A, A/m² | 10 fA, 0 |
| `drain_area`, `source_area` | `AD`, `AS` | m² | 0 |
| `drain_perimeter`, `source_perimeter` | `PD`, `PS` | m | 0 |
| `junction_potential` | `PB` | V | 0.8 V |
| `bulk_drain_capacitance`, `bulk_source_capacitance` | `CBD`, `CBS` | F | the bottom capacitance's |
| `bottom_capacitance`, `bottom_grading` | `CJ`, `MJ` | F/m², none | 0, 0.5 |
| `sidewall_capacitance`, `sidewall_grading` | `CJSW`, `MJSW` | F/m, none | 0, 0.5 at level 1 |
| `depletion_coefficient` | `FC` | none | 0.5 |
| `gate_source_overlap`, `gate_drain_overlap`, `gate_bulk_overlap` | `CGSO`, `CGDO`, `CGBO` | F/m | 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |
| `level` | `LEVEL` | 1 or 3 | 1 |

Level 3 replaces the square law with a law of short channels, takes no channel-length modulation, gives the oxide a thickness of 0.1 µm and the sidewall a grading coefficient of 0.33 unless given, and adds parameters of its own:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `max_drift_velocity` | `VMAX` | m/s | none: no velocity saturation |
| `junction_depth` | `XJ` | m | 0: no short-channel body effect |
| `fast_surface_state_density` | `NFS` | m⁻² | 0: no weak inversion |
| `static_feedback` | `ETA` | none | 0 |
| `narrow_width_factor` | `DELTA` | none | 0 |
| `mobility_modulation` | `THETA` | V⁻¹ | 0 |
| `saturation_field_factor` | `KAPPA` | none | 0.2 |
| `width_narrowing` | `WD` | m | 0 |
| `length_adjustment`, `width_adjustment` | `XL`, `XW` | m | 0 |
| `threshold_adjustment` | `DELVTO` | V | 0 |

The channel shortens beyond saturation only given a substrate doping.

SPICE writes the surface mobility, the substrate doping and the surface state density in centimeters; jolty takes SI units, as everywhere.
The saturation current density replaces the saturation current when it and both areas are nonzero.

Away from the nominal temperature, the transconductance parameter and the surface mobility fall as `(T/Tn)^−1.5`, the surface potential follows silicon's band gap and intrinsic concentration, and the threshold moves with them; the junctions' saturation currents, potential and capacitances follow the band gap too, as in ngspice.
A process derived from the substrate doping is derived at the nominal temperature.

A drain or source resistance adds an internal node to the unknowns, as `M1:drain` or `M1:source` for a transistor `M1`; one below 1 Ω adds its current too, as `M1:drain_current` or `M1:source_current`, as a small resistor does.
Meyer's capacitances depend on every terminal voltage, and no charge yields them, so each carries a current `C(v) · dv/dt` that follows a current of the transistor's own: the displacement current through the whole oxide towards the source, the drain or the bulk, the unknowns `M1:gate_source_displacement`, `M1:gate_drain_displacement` and `M1:gate_bulk_displacement`.
The frequency response holds Meyer's capacitances exactly, as ngspice's does.
With the bulk forward-biased, ngspice's small-signal model and Newton steps take the threshold's slope from the reverse-biased formula; jolty keeps every derivative exact.

In a [noise analysis](simulation.md#noise), the resistances make thermal noise, the channel thermal noise of `8kT·gm/3` A²/Hz, and the drain current `Id` flicker noise of `m·KF·(|Id|/m)^AF/(f·W·Leff·Cox²)`: each of the `m` parallel copies carries its share of the current over its own effective gate, and their noises add, with an oxide 0.1 µm thick unless given, as in SPICE.
A netlist written for ngspice gives it as ngspice's `nlev=1`, `KF/Cox` its coefficient, which ngspice 43 and later read.
