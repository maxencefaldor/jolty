# Models

Each model a device can be, as the engine defines it, and every parameter a device of it takes: its default, which a device that leaves it out takes, and its description, both from the engine's own parameter types.
A parameter within another is named by its path, its keys joined by dots and `*` standing for each entry of a list, such as `waveform.pulse.rise` or `windings.*.inductance`; of the parameters of one shape, such as `waveform.dc` and `waveform.pulse`, a device gives one.
Values are in SI base units, as the circuit format holds them, while the functions of `jolty.devices` that place each model take [quantities with units](../guide/units.md).

## `behavioral_current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `expression` | required | What the source's output follows: an expression of the voltages of the terminals it reads, of the currents through sensing ports of its own, and of time. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `behavioral_voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `expression` | required | What the source's output follows: an expression of the voltages of the terminals it reads, of the currents through sensing ports of its own, and of time. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `bipolar_transistor`

| Parameter | Default | Description |
| --- | --- | --- |
| `base_collector_capacitance` | `0` | The zero-bias base-collector depletion capacitance, `CJC`, in farads; default 0. |
| `base_collector_grading` | `0.33` | The base-collector grading coefficient, `MJC`, below 1; default 0.33. |
| `base_collector_leakage_current` | `0` | The base-collector leakage saturation current, `ISC`, in amperes; default 0. |
| `base_collector_leakage_emission_coefficient` | `2` | The base-collector leakage emission coefficient, `NC`; default 2. |
| `base_collector_potential` | `0.75` | The base-collector built-in potential, `VJC`, in volts; default 0.75. |
| `base_emitter_capacitance` | `0` | The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0. |
| `base_emitter_grading` | `0.33` | The base-emitter grading coefficient, `MJE`, below 1; default 0.33. |
| `base_emitter_leakage_current` | `0` | The base-emitter leakage saturation current, `ISE`, in amperes; default 0. |
| `base_emitter_leakage_emission_coefficient` | `1.5` | The base-emitter leakage emission coefficient, `NE`; default 1.5. |
| `base_emitter_potential` | `0.75` | The base-emitter built-in potential, `VJE`, in volts; default 0.75. |
| `base_resistance` | `0` | The zero-bias base resistance, `RB`, in ohms; default 0. |
| `base_resistance_current` |  | The base current, `IRB`, at which the base resistance falls halfway to its minimum, in amperes; without it, the base resistance falls with the base charge. |
| `beta_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the betas, `XTB`; default 0. |
| `collector_resistance` | `0` | The collector resistance, `RC`, in ohms; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of each junction's potential above which its depletion capacitance is extended linearly, `FC`, below 1; default 0.5. |
| `emitter_resistance` | `0` | The emitter resistance, `RE`, in ohms; default 0. |
| `energy_gap` | `1.11` | The energy gap that scales the saturation currents with temperature, `EG`, in volts; default 1.11, silicon's. |
| `excess_phase` | `0` | The excess phase at the frequency `1 / (2π · TF)`, `PTF`, in degrees, which delays the forward transport current by `PTF · π / 180 · TF`; default 0. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the base current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the base current in the flicker noise, `AF`; default 1. |
| `forward_beta` | `100` | The ideal maximum forward current gain, `BF`; default 100. |
| `forward_early_voltage` |  | The forward Early voltage, `VAF`, in volts; infinite by default. |
| `forward_emission_coefficient` | `1` | The forward emission coefficient, `NF`; default 1. |
| `forward_knee_current` |  | The forward knee current, `IKF`, where high injection sets in, in amperes; infinite by default. |
| `forward_transit_time` | `0` | The ideal forward transit time, `TF`, in seconds; default 0. |
| `internal_base_fraction` | `1` | The fraction of the base-collector capacitance on the internal base, `XCJC`, the rest on the external base; default 1. |
| `minimum_base_resistance` |  | The base resistance at high current, `RBM`, in ohms; the base resistance by default. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `polarity` | `npn` | The doping order; default `npn`. One of `npn`, `pnp`. |
| `reverse_beta` | `1` | The ideal maximum reverse current gain, `BR`; default 1. |
| `reverse_early_voltage` |  | The reverse Early voltage, `VAR`, in volts; infinite by default. |
| `reverse_emission_coefficient` | `1` | The reverse emission coefficient, `NR`; default 1. |
| `reverse_knee_current` |  | The reverse knee current, `IKR`, in amperes; infinite by default. |
| `reverse_transit_time` | `0` | The ideal reverse transit time, `TR`, in seconds; default 0. |
| `saturation_current` | `1e-16` | The transport saturation current, `IS`, in amperes; default 1e-16. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the saturation currents' scaling, `XTI`; default 3. |
| `substrate` |  | How the transistor is built, if it has a substrate, as ngspice's `SUBS` of 1 and −1 builds it; without it, the transistor has none. The parameters below, of the junction to the substrate, need one. One of `vertical`, `lateral`. |
| `substrate_capacitance` |  | The substrate junction's zero-bias depletion capacitance, `CJS`, in farads; default 0. |
| `substrate_emission_coefficient` |  | The substrate junction's emission coefficient, `NS`; default 1. |
| `substrate_grading` |  | The substrate junction's grading coefficient, `MJS`, below 1; default 0. |
| `substrate_potential` |  | The substrate junction's built-in potential, `VJS`, in volts; default 0.75. |
| `substrate_saturation_current` |  | The substrate junction's saturation current, `ISS`, in amperes; without it, the junction carries the minimum conductance only. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `transit_time_bias_coefficient` | `0` | How much the forward transit time grows with bias, `XTF`; default 0. |
| `transit_time_current` | `0` | The current, `ITF`, above which the forward transit time grows, in amperes; default 0. |
| `transit_time_voltage` |  | The base-collector voltage, `VTF`, that sets how the forward transit time grows with it, in volts; infinite by default. |

## `capacitor`

| Parameter | Default | Description |
| --- | --- | --- |
| `capacitance` | required | The capacitance, in farads, at the nominal temperature; zero or positive. |
| `nominal_temperature` | `300.15` | The temperature the capacitance was measured at, in kelvins; default 27 °C. |
| `quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `quadratic_voltage_coefficient` | `0` | The quadratic voltage coefficient `vc2`, per volt squared; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `voltage_coefficient` | `0` | The linear voltage coefficient `vc1`, per volt; default 0. |

## `coupled_inductors`

| Parameter | Default | Description |
| --- | --- | --- |
| `windings` | required | Each winding, as an inductor's parameters; at least two. |
| `windings.*.inductance` | required | The inductance, in henries, at the nominal temperature; positive. |
| `windings.*.nominal_temperature` | `300.15` | The temperature the inductance was measured at, in kelvins; default 27 °C. |
| `windings.*.quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `windings.*.temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `couplings` | `[]` | The couplings between pairs of windings. |
| `couplings.*.coefficient` | required | The coupling coefficient, from -1 to 1. |
| `couplings.*.windings` | required | The two windings, by their numbers from 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_controlled_current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `gain` | required | The ratio of the output current to the control current. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_controlled_memristor`

| Parameter | Default | Description |
| --- | --- | --- |
| `charge` | required | The charge that carries the state across its whole range, in coulombs; positive. |
| `off_resistance` | required | The resistance fully off, at a state of 1, in ohms; above the on resistance. |
| `on_resistance` | required | The resistance fully on, at a state of 0, in ohms; positive. |
| `law` | `linear` | How the resistance follows the state; default linear. One of `linear`, `exponential`. |
| `state` | `1` | The state it was made in, from 0, fully on, to 1, fully off; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `window` | `0.01` | The width of the window that slows the state at either bound, a fraction of its range; positive; default 0.01. |

## `current_controlled_switch`

| Parameter | Default | Description |
| --- | --- | --- |
| `off_current` | `0` | The control current at which the switch is off, in amperes; default 0. |
| `off_resistance` | `1000000` | The resistance with the control at or past the off current, in ohms; default 1e6. |
| `on_current` | `0.001` | The control current at which the switch is on, in amperes; default 1 mA. |
| `on_resistance` | `1` | The resistance with the control at or past the on current, in ohms; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_controlled_voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `transresistance` | required | The ratio of the output voltage to the control current, in ohms. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `waveform` | required | The source's level over time, in volts or amperes. |
| `waveform.dc` |  | A constant level. |
| `waveform.pulse` |  | A trapezoidal pulse, repeated each period, if it has one. |
| `waveform.pulse.delay` | `0` | The time the first pulse starts its rise, in seconds. |
| `waveform.pulse.fall` | required | The time each pulse takes to fall, in seconds; positive. |
| `waveform.pulse.initial` | required | The level before the delay and between pulses. |
| `waveform.pulse.period` |  | The time from one pulse's start to the next's, in seconds; without it, a single pulse. |
| `waveform.pulse.pulsed` | required | The level at the top of each pulse. |
| `waveform.pulse.rise` | required | The time each pulse takes to rise, in seconds; positive. |
| `waveform.pulse.width` | required | The time each pulse holds its top, in seconds. |
| `waveform.sine` |  | A sine, damped from its delay on. |
| `waveform.sine.amplitude` | required | The sine's amplitude at its delay. |
| `waveform.sine.damping` | `0` | The rate at which the amplitude decays from the delay on, per second. |
| `waveform.sine.delay` | `0` | The time the sine starts, in seconds, before which it holds its value then. |
| `waveform.sine.frequency` | required | The sine's frequency, in hertz. |
| `waveform.sine.offset` | required | The level the sine swings about. |
| `waveform.sine.phase` | `0` | The phase at the delay, in radians. |
| `waveform.piecewise` |  | Straight lines between points, holding the first and last levels outside them. |
| `waveform.piecewise.points` | required | Each point's time and level, in strictly increasing time; at least one. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `diode`

| Parameter | Default | Description |
| --- | --- | --- |
| `breakdown_current` | `0.001` | The current at the breakdown voltage, in amperes; default 1e-3. |
| `breakdown_emission_coefficient` |  | The breakdown emission coefficient; the emission coefficient by default. |
| `breakdown_temperature_coefficient` | `0` | How fast the breakdown voltage falls with temperature, in volts per kelvin; default 0. |
| `breakdown_voltage` |  | The reverse breakdown voltage, in volts, positive; without it, the junction never breaks down. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitance is extended linearly, below 1; default 0.5. |
| `emission_coefficient` | `1` | The emission coefficient; default 1. |
| `energy_gap` | `1.11` | The energy gap that scales the saturation current with temperature, in volts; default 1.11, silicon's. |
| `flicker_coefficient` | `0` | The flicker noise coefficient, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the current in the flicker noise, `AF`; default 1. |
| `grading_coefficient` | `0.5` | The grading coefficient of the junction, below 1; default 0.5. |
| `junction_capacitance` | `0` | The zero-bias junction capacitance, in farads; default 0. |
| `junction_potential` | `1` | The junction potential, in volts; default 1. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `saturation_current` | `1e-14` | The saturation current, in amperes; default 1e-14. |
| `saturation_current_temperature_exponent` | `3` | The exponent of the temperature ratio in the saturation current's scaling; default 3. |
| `series_resistance` | `0` | The series resistance, in ohms; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `transit_time` | `0` | The transit time, in seconds; default 0. |

## `foundry_mosfet`

| Parameter | Default | Description |
| --- | --- | --- |
| `length` | required | The channel's length, in metres; positive. |
| `model` | required | The kit's model of the transistor, such as `sky130_fd_pr__nfet_01v8`. |
| `pdk` | required | The process design kit, such as `sky130`. |
| `width` | required | The channel's width, all its fingers together, in metres; positive. |
| `fingers` | `1` | The fingers the width is split into; 1 unless given. |
| `multiplier` | `1` | The transistors in parallel; 1 unless given. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `heat_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `power` | required | The heat flow, in watts. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `inductor`

| Parameter | Default | Description |
| --- | --- | --- |
| `inductance` | required | The inductance, in henries, at the nominal temperature; positive. |
| `nominal_temperature` | `300.15` | The temperature the inductance was measured at, in kelvins; default 27 °C. |
| `quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `jfet`

| Parameter | Default | Description |
| --- | --- | --- |
| `beta_temperature_exponent` | `0` | The exponent of the temperature ratio that scales the transconductance parameter; default 0. |
| `channel` | `n` | The channel; default `n`. One of `n`, `p`. |
| `channel_length_modulation` | `0` | The channel-length modulation, `λ`, per volt; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitance is extended linearly, below 1; default 0.5. |
| `doping_tail` | `1` | The doping tail, `b`: 1 for the plain square law; default 1. |
| `drain_resistance` | `0` | The drain resistance, in ohms; default 0. |
| `energy_gap` | `1.11` | The energy gap that scales the saturation current with temperature, in volts; default 1.11, silicon's. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the drain current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the drain current in the flicker noise, `AF`; default 1. |
| `gate_drain_capacitance` | `0` | The zero-bias gate-drain capacitance, in farads; default 0. |
| `gate_source_capacitance` | `0` | The zero-bias gate-source capacitance, in farads; default 0. |
| `junction_potential` | `1` | The gate junctions' potential, `pb`, in volts; default 1. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `saturation_current` | `1e-14` | The gate junctions' saturation current, in amperes; default 1e-14. |
| `saturation_current_temperature_exponent` | `0` | The exponent of the temperature ratio in the saturation current's scaling; default 0, as ngspice scales a JFET's by it only when it is given. |
| `source_resistance` | `0` | The source resistance, in ohms; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `threshold_temperature_coefficient` | `0` | How fast the threshold voltage rises with temperature, in volts per kelvin; default 0. |
| `threshold_voltage` | `-2` | The threshold voltage, `vto`, in volts; default −2. |
| `transconductance_parameter` | `0.0001` | The transconductance parameter, `β`, in amperes per square volt; default 1e-4. |

## `logic_cell`

| Parameter | Default | Description |
| --- | --- | --- |
| `inputs` | required | The inputs' names, in the order of the truth tables' bits. |
| `outputs` | required | The outputs, each with its truth table. |
| `outputs.*.name` | required | The output's name. |
| `outputs.*.table` | required | The output's level for each combination of the inputs, a `0` or a `1` per row. Row `k` holds the inputs whose bits make `k`, the first input the most significant: for inputs `a` and `b`, the rows are `ab` = 00, 01, 10 and 11, so a NAND is `1110`. |
| `timing` | required | How the outputs follow the inputs in time. |
| `timing.fixed` |  | The same delays after any input changes, whatever the load. |
| `timing.fixed.fall` | required | The delay of an output falling to 0, in seconds. |
| `timing.fixed.rise` | required | The delay of an output rising to 1, in seconds. |
| `timing.fixed.transition` | required | The outputs' transition, the time an edge takes to swing from ground to supply, in seconds. |
| `timing.characterized` |  | Delays and transitions that depend on the arc, the input transition and the load. |
| `timing.characterized.arcs` | required | Every arc: each change of an input that changes an output. |
| `timing.characterized.arcs.*.before` | required | Every input's level before the change, a `0` or a `1` each, in the order of the inputs. |
| `timing.characterized.arcs.*.delay` | required | The delay, in seconds, for each input transition, then for each load: one row per transition. |
| `timing.characterized.arcs.*.input` | required | The input that changes. |
| `timing.characterized.arcs.*.output` | required | The output that follows. |
| `timing.characterized.arcs.*.transition` | required | The output's transition, in seconds, laid out as the delay. |
| `timing.characterized.loads` | required | The output loads the tables were measured at, in farads, rising. |
| `timing.characterized.slews` | required | The input transitions the tables were measured at, in seconds, rising. |
| `timing.characterized.supply` |  | The voltage from ground to supply the tables were measured at, in volts. |
| `electrical` |  | With them, the cell has terminals `supply` and `ground`, and its pins may join electrical nets. |
| `electrical.ground` | `ground` | The name of the ground terminal; `ground` unless given. |
| `electrical.input_capacitances` | required | The capacitance of each input, in farads, in the order of the inputs. |
| `electrical.supply` | `supply` | The name of the supply terminal; `supply` unless given. |
| `electrical.threshold` | `0.5` | The fraction of the supply, from ground, at which an input switches; one half unless given. |

## `logic_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `initial` | required | The level at time zero. One of `0`, `1`, `x`. |
| `transition` | required | The edges' transition, the time each takes to swing from ground to supply, in seconds. |
| `changes` | `[]` | Each later change: its time, in seconds, and the new level, in time order. |

## `mosfet`

| Parameter | Default | Description |
| --- | --- | --- |
| `body_effect` |  | The body-effect coefficient, `γ`, in square-root volts; default 0, or derived from the substrate doping. |
| `bottom_capacitance` | `0` | The bulk junctions' zero-bias bottom capacitance per area, `cj`, in farads per square meter; default 0. |
| `bottom_grading` | `0.5` | The bottom capacitance's grading coefficient, `mj`, below 1; default 0.5. |
| `bulk_drain_capacitance` |  | The zero-bias bulk-drain capacitance, in farads; default none, for the bottom capacitance's. |
| `bulk_source_capacitance` |  | The zero-bias bulk-source capacitance, in farads; default none, for the bottom capacitance's. |
| `channel` | `n` | The channel; default `n`. One of `n`, `p`. |
| `channel_length_modulation` |  | The channel-length modulation, `λ`, per volt, level 1's alone; default 0. |
| `depletion_coefficient` | `0.5` | The fraction of the junction potential above which the depletion capacitances are extended linearly, below 1; default 0.5. |
| `drain_area` | `0` | The drain diffusion's area, in square meters; default 0. |
| `drain_perimeter` | `0` | The drain diffusion's perimeter, in meters; default 0. |
| `drain_resistance` |  | The drain resistance, in ohms; default none, for the sheet resistance's. |
| `drain_squares` | `1` | The drain diffusion's length in squares, `nrd`; default 1. |
| `fast_surface_state_density` |  | Level 3: the fast surface state density, `nfs`, per square meter, which turns on weak inversion; default 0. |
| `flicker_coefficient` | `0` | The flicker noise coefficient of the drain current, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the drain current in the flicker noise, `AF`; default 1. |
| `gate_bulk_overlap` | `0` | The gate-bulk overlap capacitance per length, in farads per meter; default 0. |
| `gate_drain_overlap` | `0` | The gate-drain overlap capacitance per width, in farads per meter; default 0. |
| `gate_material` | `opposite_doping` | The gate's material, which sets a derived `vto`; default `opposite_doping`. One of `opposite_doping`, `same_doping`, `aluminum`. |
| `gate_source_overlap` | `0` | The gate-source overlap capacitance per width, in farads per meter; default 0. |
| `junction_depth` |  | Level 3: the junction depth, `xj`, in meters, which a substrate doping turns into short channels' body effect; default 0. |
| `junction_potential` | `0.8` | The bulk junctions' potential, `pb`, in volts; default 0.8. |
| `lateral_diffusion` | `0` | The lateral diffusion, `ld`, by which the channel falls short of its drawn length at each end, in meters; default 0. |
| `length` | `0.0001` | The channel's drawn length, `l`, in meters; default 100 µm. |
| `length_adjustment` |  | Level 3: the length the mask adds, `xl`, in meters; default 0. |
| `level` | `1` | The model's level, 1 or 3; default 1. |
| `max_drift_velocity` |  | Level 3: the carriers' maximum drift velocity, `vmax`, in meters per second; default none, for no velocity saturation. |
| `mobility_modulation` |  | Level 3: the gate field's degradation of the mobility, `theta`, per volt; default 0. |
| `multiplier` | `1` | The number of identical transistors in parallel, `m`; default 1. |
| `narrow_width_factor` |  | Level 3: the width's effect on the threshold, `delta`; default 0. |
| `nominal_temperature` | `300.15` | The temperature the parameters were measured at, in kelvins; default 27 °C. |
| `oxide_thickness` |  | The gate oxide's thickness, `tox`, in meters; default none at level 1, which leaves the gate without Meyer's capacitances, and 0.1 µm at level 3 and for flicker noise. |
| `saturation_current` | `1e-14` | The bulk junctions' saturation current, in amperes; default 1e-14. |
| `saturation_current_density` | `0` | The bulk junctions' saturation current per area, in amperes per square meter, which replaces the saturation current when it and both areas are nonzero; default 0. |
| `saturation_field_factor` |  | Level 3: the saturation field factor, `kappa`; default 0.2. |
| `sheet_resistance` | `0` | The drain and source diffusions' sheet resistance, in ohms per square; default 0. |
| `sidewall_capacitance` | `0` | The bulk junctions' zero-bias sidewall capacitance per length of perimeter, `cjsw`, in farads per meter; default 0. |
| `sidewall_grading` |  | The sidewall capacitance's grading coefficient, `mjsw`, below 1; default 0.5 at level 1 and 0.33 at level 3. |
| `source_area` | `0` | The source diffusion's area, in square meters; default 0. |
| `source_perimeter` | `0` | The source diffusion's perimeter, in meters; default 0. |
| `source_resistance` |  | The source resistance, in ohms; default none, for the sheet resistance's. |
| `source_squares` | `1` | The source diffusion's length in squares, `nrs`; default 1. |
| `static_feedback` |  | Level 3: the drain's static feedback on the threshold, `eta`; default 0. |
| `substrate_doping` |  | The substrate doping, `nsub`, per cubic meter, from which `vto`, `γ` and `φ` derive unless given; default none. It needs an oxide thickness. |
| `surface_mobility` | `0.06` | The surface mobility, `uo`, in square meters per volt-second, from which `kp` derives; default 0.06. |
| `surface_potential` |  | The surface potential, `φ`, in volts; default 0.6, or derived from the substrate doping. |
| `surface_state_density` | `0` | The surface state density, `nss`, per square meter, which shifts a derived `vto`; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `threshold_adjustment` |  | Level 3: a shift of the threshold, `delvto`, in volts; default 0. |
| `threshold_voltage` |  | The zero-bias threshold voltage, `vto`, in volts; default 0, or derived from the substrate doping. |
| `transconductance_parameter` |  | The transconductance parameter, `kp`, in amperes per square volt; default 2e-5, or the surface mobility times the oxide capacitance per area, given an oxide thickness. |
| `width` | `0.0001` | The channel's width, `w`, in meters; default 100 µm. |
| `width_adjustment` |  | Level 3: the width the mask adds, `xw`, in meters; default 0. |
| `width_narrowing` |  | Level 3: the width lost to diffusion at each side, `wd`, in meters; default 0. |

## `photocell`

| Parameter | Default | Description |
| --- | --- | --- |
| `resistance` | required | The resistance at the reference illuminance, in ohms; positive. |
| `dark_resistance` | `1000000` | The resistance in darkness, in ohms; positive; default 1e6. |
| `gamma` | `0.7` | How steeply the resistance falls with the illuminance, `γ`, the slope of its logarithm against the illuminance's; positive; default 0.7. |
| `illuminance` | `10` | The reference illuminance, in lux; positive; default 10, as datasheets give it, such as Senba's of its GL55 series. |
| `response_time` | `0` | The time constant of the response, in seconds; zero or more; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `resistor`

| Parameter | Default | Description |
| --- | --- | --- |
| `resistance` | required | The resistance, in ohms, at the nominal temperature; positive. |
| `exponential_temperature_coefficient` |  | The exponential temperature coefficient `tce`, in percent per kelvin: given, the resistance is multiplied by `1.01^(tce · ΔT)` instead of the polynomial. |
| `flicker_coefficient` | `0` | The flicker noise coefficient, `KF`; default 0, no flicker noise. |
| `flicker_exponent` | `1` | The power of the current in the flicker noise, `AF`; default 1. |
| `nominal_temperature` | `300.15` | The temperature the resistance was measured at, in kelvins; default 27 °C. |
| `quadratic_temperature_coefficient` | `0` | The quadratic temperature coefficient `tc2`, per kelvin squared; default 0. |
| `quadratic_voltage_coefficient` | `0` | The quadratic voltage coefficient `vc2`, per volt squared; default 0. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_coefficient` | `0` | The linear temperature coefficient `tc1`, per kelvin; default 0. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `thermal` | `false` | Whether the device heats by the power it dissipates, through a terminal `thermal` joined to a thermal net, whose temperature above the ambient it runs at. |
| `voltage_coefficient` | `0` | The linear voltage coefficient `vc1`, per volt; default 0. |

## `thermal_capacitance`

| Parameter | Default | Description |
| --- | --- | --- |
| `capacitance` | required | The heat capacity, in joules per kelvin; zero or positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `thermal_resistance`

| Parameter | Default | Description |
| --- | --- | --- |
| `resistance` | required | The resistance, in kelvins per watt; positive. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_controlled_current_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `transconductance` | required | The ratio of the output current to the control voltage, in siemens. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_controlled_memristor`

| Parameter | Default | Description |
| --- | --- | --- |
| `off_rate` | required | The state's speed toward off at twice the off voltage, in full ranges per second; positive. |
| `off_resistance` | required | The resistance fully off, at a state of 1, in ohms; above the on resistance. |
| `off_voltage` | required | The voltage above which the state moves toward off, in volts; positive. |
| `on_rate` | required | The state's speed toward on at twice the on voltage, in full ranges per second; positive. |
| `on_resistance` | required | The resistance fully on, at a state of 0, in ohms; positive. |
| `on_voltage` | required | The voltage below which the state moves toward on, in volts; negative. |
| `law` | `linear` | How the resistance follows the state; default linear. One of `linear`, `exponential`. |
| `off_exponent` | `1` | How steeply the speed toward off grows past the off voltage, `α_off`; at least 1; default 1. |
| `on_exponent` | `1` | How steeply the speed toward on grows past the on voltage, `α_on`; at least 1; default 1. |
| `state` | `1` | The state it was made in, from 0, fully on, to 1, fully off; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
| `window` | `0.01` | The width of the window that slows the state at either bound, a fraction of its range; positive; default 0.01. |

## `voltage_controlled_switch`

| Parameter | Default | Description |
| --- | --- | --- |
| `off_resistance` | `1000000` | The resistance with the control at or past the off voltage, in ohms; default 1e6. |
| `off_voltage` | `0` | The control voltage at which the switch is off, in volts; default 0. |
| `on_resistance` | `1` | The resistance with the control at or past the on voltage, in ohms; default 1. |
| `on_voltage` | `1` | The control voltage at which the switch is on, in volts; default 1. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_controlled_voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `gain` | required | The ratio of the output voltage to the control voltage. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |

## `voltage_source`

| Parameter | Default | Description |
| --- | --- | --- |
| `waveform` | required | The source's level over time, in volts or amperes. |
| `waveform.dc` |  | A constant level. |
| `waveform.pulse` |  | A trapezoidal pulse, repeated each period, if it has one. |
| `waveform.pulse.delay` | `0` | The time the first pulse starts its rise, in seconds. |
| `waveform.pulse.fall` | required | The time each pulse takes to fall, in seconds; positive. |
| `waveform.pulse.initial` | required | The level before the delay and between pulses. |
| `waveform.pulse.period` |  | The time from one pulse's start to the next's, in seconds; without it, a single pulse. |
| `waveform.pulse.pulsed` | required | The level at the top of each pulse. |
| `waveform.pulse.rise` | required | The time each pulse takes to rise, in seconds; positive. |
| `waveform.pulse.width` | required | The time each pulse holds its top, in seconds. |
| `waveform.sine` |  | A sine, damped from its delay on. |
| `waveform.sine.amplitude` | required | The sine's amplitude at its delay. |
| `waveform.sine.damping` | `0` | The rate at which the amplitude decays from the delay on, per second. |
| `waveform.sine.delay` | `0` | The time the sine starts, in seconds, before which it holds its value then. |
| `waveform.sine.frequency` | required | The sine's frequency, in hertz. |
| `waveform.sine.offset` | required | The level the sine swings about. |
| `waveform.sine.phase` | `0` | The phase at the delay, in radians. |
| `waveform.piecewise` |  | Straight lines between points, holding the first and last levels outside them. |
| `waveform.piecewise.points` | required | Each point's time and level, in strictly increasing time; at least one. |
| `temperature` |  | The temperature the device is held at whatever the simulation's, in kelvins; at least 0. Without it, the device is at the simulation's temperature. |
| `temperature_offset` |  | How far the device's temperature lies above the simulation's, below where negative, in kelvins; the device's temperature at least 0. Without it, the device is at the simulation's temperature. |
