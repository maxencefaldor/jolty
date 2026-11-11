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

The control voltage is `control_positive` minus `control_negative`, and the control terminals draw no current.
A voltage-controlled voltage source holds its output at the gain times the control voltage, and adds its current to the unknowns as a voltage source does.
A voltage-controlled current source drives the transconductance times the control voltage from `positive` through itself to `negative`.

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
