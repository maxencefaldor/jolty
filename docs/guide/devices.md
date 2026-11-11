# Devices

The module `jolty.devices` holds every device a script can place.
Each function checks its parameters' units and returns a device ready to place; parameters left out take the engine's defaults.

## Passive devices

| Function | Terminals | Parameter |
| --- | --- | --- |
| `resistor(resistance)` | `a`, `b` | a positive resistance |
| `capacitor(capacitance)` | `a`, `b` | a capacitance, zero or more |
| `inductor(inductance)` | `a`, `b` | a positive inductance |

An inductor adds its current to the unknowns, as `L1:current` for an inductor `L1`: the current from `a` through the inductor to `b`.

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

## Diode

`diode(...)` is a junction diode from `anode` to `cathode`, in the SPICE model, at 27 °C.
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

A diode with a series resistance adds the voltage of its junction to the unknowns, as `D1:junction` for a diode `D1`.

The junction follows SPICE as ngspice implements it, its reverse current settling on the saturation current below three thermal voltages of reverse bias.

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

A drain or source resistance adds an internal node to the unknowns, as `J1:drain` or `J1:source` for a JFET `J1`.

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

A base, collector or emitter resistance adds an internal node to the unknowns, as `Q1:base`, `Q1:collector` or `Q1:emitter` for a transistor `Q1`.

ngspice leaves out of its small-signal model and its Newton steps how a falling base resistance depends on the junction voltages; jolty keeps every derivative, so its frequency response is the exact slope of its DC model, and its Newton steps converge faster to the same operating point.
