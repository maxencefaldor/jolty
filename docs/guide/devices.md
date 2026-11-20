# Devices

The module `jolty.devices` provides every device constructor available in build scripts.
Each constructor validates parameter dimensions and returns a placeable device; omitted parameters take engine defaults listed in the [models reference](../reference/models.md).

Devices simulate at the plan temperature (default 27 °C), while model parameters are referenced to `nominal_temperature` (default 27 °C).
Away from nominal temperature, passive values drift by temperature coefficients and semiconductor models scale as documented in each section below.
Devices with [self-heating](#self-heating) instead run at a dynamic temperature raised by dissipated power.

Any device can run at an independent temperature: `device.held_at(celsius(85))` fixes absolute temperature regardless of plan settings (SPICE `TEMP`, PSpice `T_ABS`), while `device.warmer_by(20 * K)` offsets the plan temperature (SPICE `DTEMP`, PSpice `T_REL_GLOBAL`).
Model scaling, noise, self-heating ambient reference, and temperature ratings all honor per-device temperature overrides, though sensitivity analyses do not differentiate device temperature.
A resistor held at 0 K generates no thermal noise, whereas semiconductor junctions require temperatures strictly above 0 K.

Logic devices connect to logic nets and are documented separately under [Logic](logic.md).

The diode, JFET, bipolar transistors, level 1 and 3 MOSFETs, and VDMOS follow ngspice 47 implementations, while BSIM3 derives from BSIM3v3.3.0 (UC Berkeley BSIM Research Group); copyright notices and terms appear in `NOTICE`.
Each device section documents behavioral differences from ngspice, summarized in [Departures from ngspice](departures.md).

Thermal voltage `kT/q` uses exact SI constants (`k = 1.380649e-23` J/K, `q = 1.602176634e-19` C), whereas ngspice 47 uses CODATA 2014 values (`1.38064852e-23` J/K, `1.6021766208e-19` C in `const.h`, lines 32 and 37).
jolty's thermal voltage is therefore `3.4e-7` higher at identical kelvins, so differential tests adjust ngspice temperature via `jolty.spice.ngspice` to match thermal voltage exactly.

## Passive devices

| Function | Terminals | Parameter |
| --- | --- | --- |
| `resistor(resistance)` | `a`, `b` | a positive resistance |
| `capacitor(capacitance)` | `a`, `b` | a capacitance, zero or more |
| `inductor(inductance)` | `a`, `b` | a positive inductance |

Every inductor adds a branch-current unknown named `L1:current` (for inductor `L1`), flowing from terminal `a` through the inductor to `b`.
Resistors below 1 Ω similarly add `R1:current` as a modified nodal analysis group-2 unknown so conductances above 1 S do not swamp picosiemens shunts such as `gmin` in floating-point arithmetic.

Passive values scale with temperature difference `ΔT = T - Tnom` as `1 + tc1 · ΔT + tc2 · ΔT²`, matching SPICE:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `temperature_coefficient` | `TC1` | K⁻¹ | 0 |
| `quadratic_temperature_coefficient` | `TC2` | K⁻² | 0 |
| `exponential_temperature_coefficient` | `TCE` | percent per kelvin, a number | none: a resistor drifts by `1.01^(tce · ΔT)` instead |
| `nominal_temperature` | `TNOM` (`TREF` on resistors) | K | 27 °C |

Only resistors accept `exponential_temperature_coefficient`, and any coefficient combination scaling resistance to zero fails elaboration.

Resistor and capacitor values also vary with terminal voltage `v = v(a) - v(b)` by `1 + vc1 · v + vc2 · v²`:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `voltage_coefficient` | `VC1` | V⁻¹ | 0 |
| `quadratic_voltage_coefficient` | `VC2` | V⁻² | 0 |

A voltage-dependent resistor conducts `v / (r · (1 + vc1 · v + vc2 · v²))`.
A voltage-dependent capacitor stores charge `c · (v + vc1 · v² / 2 + vc2 · v³ / 3)`, whose derivative equals the voltage-modulated capacitance.
Both formulas apply at all voltages as in ngspice, allowing negative values if coefficients drive the polynomial below zero.

In [noise analysis](simulation.md#noise), resistors generate thermal noise `4kT/R` A²/Hz (or series voltage noise `4kT·R` V²/Hz below 1 Ω), matching switches, photocells, and memristors, whereas capacitors and inductors are noiseless.
When `flicker_coefficient` (`KF`) is set, resistor current `I` adds flicker noise `KF · |I|^AF / f` A²/Hz with `flicker_exponent` `AF` (default 1), matching ngspice.

Passing `tolerance` (such as `0.01` for 1%) adds a uniform tolerance to passive values.
`device.toleranced(parameter, relative, distribution, lot)` attaches `"uniform"`, `"normal"`, `"log_uniform"`, or `"log_normal"` tolerances (with optional shared `lot`) to any numeric parameter path (such as `waveform.dc`) for [Monte Carlo](simulation.md#monte-carlo) and [worst-case](simulation.md#worst-case) analyses.

### Coupled inductors

`coupled_inductors(windings, couplings)` couples magnetic fluxes across inductors (such as transformers or common-mode chokes).
Each winding is an `inductor(...)` with optional tolerance and drift; winding `k` (1-indexed) exposes terminals `a{k}` and `b{k}` and branch-current unknown `current{k}`.
`couplings` maps winding index pairs to coefficients in `[-1, 1]`, producing mutual inductance `k · √(L_i L_j)` where positive coefficients add fluxes for currents entering both `a` terminals (matching SPICE `K`).

```python
transformer = c.place(
    devices.coupled_inductors(
        [devices.inductor(1 * mH), devices.inductor(4 * mH)], {(1, 2): 0.99}
    ),
    name="T1",
)
```

Each winding acts as a DC short, so placing a winding directly across a voltage source triggers a loop diagnostic in [checks](checks.md).

### Transmission lines

`transmission_line(characteristic_impedance, delay)` models a lossless SPICE `T` line matching ngspice 47.
Port `k` (1 or 2) exposes terminals `positive{k}` and `negative{k}` with branch-current unknown `current{k}` flowing from `positive{k}` to `negative{k}`.
Each port enforces `v₁ - Z₀ i₁ = v₂(t - τ) + Z₀ i₂(t - τ)` (and symmetrically with ports swapped) for characteristic impedance `Z₀` and propagation delay `τ`.

```python
line = c.place(devices.transmission_line(50 * ohm, 1 * ns), name="T2")
```

At DC, the line acts as an ideal port-to-port short (`v₁ = v₂`, `i₁ = -i₂`), while AC multiplies delayed waves by `e^(-jωτ)` so a quarter-wave line inverts load impedance around `Z₀`.
Transients retain accepted points spanning the longest line delay and interpolate delayed waves at the integration order strictly within smooth intervals between corners.
Before time zero, transmission lines rest at the DC operating point.
Every waveform corner (start, breakpoint, or restart) propagates across each line and forces a transient step at arrival, matching piecewise-linear reflection lattices within floating-point rounding.
ngspice diverges in three ways: skipping corners below `REL`/`ABS` slope thresholds (`traacct.c`, lines 94 to 120), interpolating across corners (`traload.c`, lines 89 to 141), and scaling a DC term by `1 - gmin` (line 56).
jolty lands on every arriving corner, keeps interpolation inside smooth runs, and enforces exact DC short equations.
On `transmission_lines` in `tests/test_ngspice.py`, ngspice at 20 ps steps overshoots `near` by 3 mV at 11.5 ns (0.4 mV at 2 ps, 0.08 mV at 0.5 ps) while converging toward jolty's 0.809548 volts.
jolty reaches 0.809548 V within `1e-5` V at 20 ps, and differential tests compare at 2 ps.
Interpolation accuracy matches the integration order controlled by circuit charges, so purely resistive transmission-line circuits take steps up to `max_step` between corners.
Pole-zero, periodic steady state, and frequency/time sensitivities reject transmission lines (whose delays create infinitely many poles and state-history dependencies), and [checks](checks.md) skip growing-mode analysis on them.
The SPICE reader imports `T` lines via `TD` or normalized length `NL` at frequency `F` (defaulting to quarter-wave at 1 GHz) plus initial conditions, while rejecting lossy lines (`O`, `U`, `Y`, `P`) by name.

## Ratings

Any device can declare manufacturer operating limits via `device.rated(stress, maximum, minimum=..., terminals=...)` for `"voltage"`, `"current"`, `"power"`, or `"temperature"`.
Voltage ratings default across the first two terminals, current ratings measure current entering the first terminal, power sums electrical dissipation, and temperature sums ambient plus any [self-heating](#self-heating) rise.
Ratings accept upper bounds, lower bounds, or both (power accepts upper bounds only, and temperatures use kelvins such as `celsius(150)`).

```python
devices.resistor(100 * ohm).rated("power", 0.25 * W)
devices.capacitor(10 * uF).rated("voltage", 25 * V, minimum=-25 * V)
devices.diode().rated("voltage", minimum=-100 * V)
switch = devices.bipolar_transistor().rated(
    "voltage", 40 * V, terminals=("collector", "emitter")
)
switch.rated("power", 0.625 * W).rated("temperature", celsius(150))
```

Components support port and total-power ratings identically via `component.rated(...)`.
`device.numbered("2N3904")` and `component.numbered(...)` attach manufacturer part numbers for [bill of materials](parts.md#bills-of-materials) pricing, and [`jolty.parts`](parts.md#real-parts) provides pre-rated real parts.
Simulations verify every rating across all operating points and transients as described in [checks](checks.md#on-analysis-results).

## Independent sources

| Function | Terminals | Parameter |
| --- | --- | --- |
| `voltage_source(waveform)` | `positive`, `negative` | a voltage, or a waveform of voltages |
| `current_source(waveform)` | `positive`, `negative` | a current, or a waveform of currents |

A voltage source constrains `v(positive) - v(negative)` to its waveform value.
Source current (unknown `V1:current` for source `V1`) flows from `positive` through the source to `negative`, making it negative when delivering power.

A current source drives its waveform value internally from `positive` to `negative`, injecting current into the external circuit at `negative`.

Plans reference independent sources by path for DC sweeps and AC excitations; see [Simulation](simulation.md).

## Waveforms

Independent sources accept either a constant quantity or a `jolty.waveforms` object.
All non-transient analyses evaluate each waveform at time zero.

- `Dc(level)`: constant value equivalent to passing a bare quantity.
- `Pulse(initial, pulsed, rise, fall, width, delay=0 s, period=None)`: trapezoidal pulse transitioning from `initial` to `pulsed` and back once, or repeating every `period`.
- `Sine(offset, amplitude, frequency, delay=0 s, damping=0 Hz, phase=0)`: `offset + amplitude·exp(-damping·t')·sin(2π·frequency·t' + phase)` for elapsed time `t' = max(t - delay, 0)` and `phase` in radians.
- `Piecewise(points)`: linear interpolation through `(time, level)` pairs, clamping to the first level before start and the last level after end.
- `Exponential(initial, pulsed, rise_time_constant, fall_delay, fall_time_constant, rise_delay=0 s)`: SPICE `EXP` waveform holding `initial` until `rise_delay`, rising exponentially toward `pulsed` with `rise_time_constant`, and decaying after `fall_delay` with `fall_time_constant`.

Transient integration lands an exact time step on every waveform corner so pulse edges and exponential delays are never skipped.

## Controlled sources

| Function | Terminals | Parameter |
| --- | --- | --- |
| `voltage_controlled_voltage_source(gain)` | `positive`, `negative`, `control_positive`, `control_negative` | a gain, a plain number |
| `voltage_controlled_current_source(transconductance)` | the same | a transconductance |
| `current_controlled_current_source(gain)` | the same | a gain, a plain number |
| `current_controlled_voltage_source(transresistance)` | the same | a transresistance |

Voltage-controlled sources sense `v(control_positive) - v(control_negative)` with zero control input current.
Current-controlled sources model their control port as a zero-volt short from `control_positive` to `control_negative` (as in Modelica), adding branch current `F1:control` (for `F1`) to the unknowns.

Controlled voltage sources enforce output voltage equal to gain or transresistance times control and add an output branch-current unknown.
Controlled current sources drive transconductance or gain times control internally from `positive` to `negative`.

## Behavioral sources

`behavioral_voltage_source(expression)` constrains `v(positive) - v(negative)` to an expression, while `behavioral_current_source(expression)` drives an expression current from `positive` to `negative`.
Expressions read node voltages `v(a)` or `v(a, b)`, sensing-port currents `i(p)`, and `time` using the [expression language](../reference/expressions.md):

```python
gain = devices.behavioral_voltage_source("limit(1e5 * v(inp, inn), -12, 12)")
amplifier = c.place(gain, name="E1")
c.connect(amplifier.inp, name="signal")
```

Behavioral source terminals list `positive` and `negative`, followed by every referenced voltage terminal and `p_positive`/`p_negative` pairs for every sensing port `p`.
Each sensing port acts as a zero-volt short adding unknown `B1:p` (for `B1`), and behavioral voltage sources also add an output branch-current unknown.
`behavioral_capacitor(expression)` stores charge `q` (in coulombs) on `positive` against `negative` with current `dq/dt` (matching Verilog-A `I(p, n) <+ ddt(q)`), conserving charge with exact capacitance derivatives.
The engine differentiates every expression exactly so Newton iterations and small-signal analyses observe exact slopes.

When exporting to ngspice, behavioral capacitors emit `Q=` charges and all expressions translate directly from AST trees with explicit parentheses for semantic equivalence.
Power operations emit `pow` (or `pwr` for odd integer exponents to preserve sign under negative bases), `limit` expands via `min`/`max` (avoiding ngspice's different `limit` function), and `table` expands into piecewise linear and parabolic-corner branches.

## Switches

`voltage_controlled_switch(...)` varies resistance between `positive` and `negative` smoothly and logarithmically (with zero end slopes) from `off_resistance` to `on_resistance` as `v(control_positive) - v(control_negative)` moves from `off_voltage` to `on_voltage`, matching PSpice.
Resistance clamps outside `[off_voltage, on_voltage]`, and control terminals draw zero current.

| Parameter | PSpice | Unit | Default |
| --- | --- | --- | --- |
| `on_resistance`, `off_resistance` | `RON`, `ROFF` | Ω | 1 Ω, 1 MΩ |
| `on_voltage`, `off_voltage` | `VON`, `VOFF` | V | 1 V, 0 V |

`current_controlled_switch(...)` applies the same transition law to control current through a short from `control_positive` to `control_negative` (unknown `control`), sweeping from `off_current` to `on_current` like PSpice `ISWITCH`.

| Parameter | PSpice | Unit | Default |
| --- | --- | --- | --- |
| `on_resistance`, `off_resistance` | `RON`, `ROFF` | Ω | 1 Ω, 1 MΩ |
| `on_current`, `off_current` | `ION`, `IOFF` | A | 1 mA, 0 A |

SPICE export translates switches into behavioral current sources (sensing current-controlled switches via a 0 V source) because ngspice 42 XSPICE switches negate small-signal AC conductance relative to DC.

## Photocell

`photocell(resistance, ...)` models a datasheet-specified light-dependent resistor between `a` and `b` using reference resistance at reference illuminance and power-law slope `gamma`.
Terminal voltage `v(light_positive) - v(light_negative)` encodes illuminance at 1 V/lux with zero input current, allowing direct drive from voltage or behavioral sources.
Photocell conductance equals `1/r_dark + s/r`, where `s = (E/E₀)^γ` floors illuminance `E` smoothly above zero over 1 microlux so the derivative stays finite in darkness.
When `response_time` `τ` is positive, internal state `s` lags illuminance via `τ · ds/dt = (E/E₀)^γ - s` (unknown `P1:response`, readable via `state(name)`).

| Parameter | Unit | Default |
| --- | --- | --- |
| `resistance` | Ω, at the reference illuminance | none |
| `illuminance` | lux, a number | 10 |
| `gamma` | a number | 0.6 |
| `dark_resistance` | Ω | 1 MΩ |
| `response_time` | s | 0: instantaneous response |

Default `gamma` and `dark_resistance` match the Senba GL5528 cell (typically 10 to 20 kΩ at 10 lux).

SPICE has no native photocell, so ngspice export emits a behavioral current source whose response state filters through a 1 Ω / `τ` RC network.

## Memristors

A memristor models a state-dependent resistance between `positive` and `negative` with nonvolatile memory.
Internal state `x` spans from `0` (`on_resistance`) to `1` (`off_resistance`), evolving only when driven and persisting otherwise.
The `law` parameter selects `"linear"` (`r_on + (r_off - r_on) · x`, modeling a moving boundary) or `"exponential"` (`r_on · (r_off / r_on)^x`, modeling a tunnel barrier).
State is stored as unknown `M1:state` (readable via `state(name)`), while `resistance` and `rate` (positive toward off, in full ranges per second) are exposed as [device variables](results.md#device-models).
Near either boundary at distance `d`, window function `1 - e^(-d/w)` (width `w = window`) halts motion toward the boundary while allowing immediate departure when drive reverses.

Memristor state evolves only over time.
During operating points, DC sweeps, AC, and noise analyses, state remains fixed at initial `state` (behaving as a fixed resistor), while `rate` reports the instantaneous transient drift rate.
Transients begin from initial `state`.
In [periodic steady state](simulation.md#periodic-steady-state), a symmetrically driven memristor can repeat from any initial state, so shooting preserves the starting state along neutral directions.

`voltage_controlled_memristor(...)` implements the VTEAM model (Kvatinsky, Ramadan, Friedman, and Kolodny, 2015), where state moves only when voltage exceeds switching thresholds.
Voltage `v = v(positive) - v(negative)` above `off_voltage` drives state toward off at `off_rate · (v/v_off - 1)^α_off`.
Voltage below negative `on_voltage` drives state toward on at `on_rate · (v/v_on - 1)^α_on`, while voltages between thresholds hold state constant.
Rates specify state velocity at twice threshold voltage in full ranges per second (`k_off / (w_off - w_on)` and `-k_on / (w_off - w_on)`).

| Parameter | VTEAM | Unit | Default |
| --- | --- | --- | --- |
| `on_resistance`, `off_resistance` | `R_ON`, `R_OFF` | Ω | none |
| `on_voltage`, `off_voltage` | `v_on`, `v_off` | V | none |
| `on_rate`, `off_rate` | `-k_on`, `k_off` over `w_off - w_on` | 1/s | none |
| `on_exponent`, `off_exponent` | `α_on`, `α_off` | a number, at least 1 | 1 |
| `law` | linear or exponential | | `"linear"` |
| `state` | `(w - w_on) / (w_off - w_on)` | from 0 to 1 | 1, off |
| `window` | | a fraction of the state's range | 0.01 |

`current_controlled_memristor(...)` implements the HP linear ion-drift model (Strukov, Snider, Stewart, and Williams, 2008), where state shifts proportionally to transported charge.
Charge `q` flowing from `positive` to `negative` shifts state toward off by `q / Q` (full-range `charge` `Q`), recovering Chua's ideal memristor under `"linear"` law.
Parameters `on_resistance`, `off_resistance`, `law`, `state`, and `window` follow the same definitions as VTEAM.

SPICE export represents each memristor as a behavioral current source paired with a dedicated state node.
In transient netlists, a behavioral current charges that state node from initial `state` across a 10 µF capacitor (matching ngspice's current-to-voltage tolerance ratio so state rates satisfy voltage tolerance).
In DC, AC, and noise netlists, an ideal voltage source holds the state node at initial `state`.

## Thermal networks

Thermal networks obey the electrical analogy: thermal nets carry temperature rise above ambient (K), flows represent heat power (W), and ground represents ambient temperature.
Thermal terminals connect exclusively to thermal nets or ground and cannot mix with electrical or logic nets.

| Function | Terminals | Parameter |
| --- | --- | --- |
| `thermal_resistance(resistance)` | `a`, `b` | K/W: heat flows from hotter to colder terminal proportional to temperature difference |
| `thermal_capacitance(capacitance)` | `a`, `b` | J/K: stored thermal energy proportional to temperature difference |
| `heat_source(power)` | `a`, `b` | W: constant heat flow from `a` into `b` |

```python
heat = c.place(devices.heat_source(1 * W), name="P1")
sink = c.place(devices.thermal_resistance(50 * K / W), name="RTH")
c.connect(heat.b, sink.a, name="hot")
c.connect(heat.a, sink.b, ground)
```

Thermal net unknowns represent temperature rise above ambient, readable via `rise(net)` (50 K in the example above).
Heat entering a thermal terminal is readable via `current(instance, terminal)` in watts.

### Self-heating

Calling `device.heated()` enables electrothermal coupling on resistors, diodes, JFETs, BJTs, MOSFETs, BSIM3, and VDMOS by adding a `thermal` terminal.
The device evaluates all temperature-dependent parameters at simulation temperature plus thermal-net rise and injects electrical dissipation (`Σ v_i i_i`) into the thermal net as heat.
Connecting a thermal resistance to ambient determines steady-state junction temperature, while an added thermal capacitance governs thermal transient dynamics:

```python
transistor = c.place(devices.bipolar_transistor().heated(), name="Q1")
mount = c.place(devices.thermal_resistance(200 * K / W), name="RJA")
c.connect(transistor.thermal, mount.a, name="junction")
c.connect(mount.b, ground)
```

Every analysis accounts for self-heating: operating points balance electrical and thermal equilibrium, transients track heating over time, and AC responses capture thermal poles (yielding distinct low-frequency incremental resistances).
Matrix structure stays fixed to ambient resistance thresholds: a resistor below 1 Ω at ambient keeps its branch-current unknown regardless of heated resistance.
ngspice lacks general thermal-network heating (supporting only a separate Celsius-unit VDMOS network), so circuits with heated devices cannot export to SPICE.

## Diode

`diode(...)` models a junction diode from `anode` to `cathode` following ngspice 47 (SPICE levels 1 and 3).
Every parameter is optional:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `saturation_current` | `IS`, `JS` | A | 10 fA |
| `emission_coefficient` | `N` | none | 1 |
| `series_resistance` | `RS` | Ω | 0 |
| `junction_capacitance` | `CJO`, `CJ0`, `CJ` | F | 0 |
| `junction_potential` | `VJ`, `PB` | V | 1 V |
| `grading_coefficient` | `M`, `MJ` | none | 0.5 |
| `depletion_coefficient` | `FC` | none | 0.5 |
| `transit_time` | `TT` | s | 0 |
| `forward_knee_current`, `reverse_knee_current` | `IKF`, `IK`; `IKR` | A | none: no high injection |
| `recombination_saturation_current` | `ISR` | A | none: no recombination current |
| `recombination_emission_coefficient` | `NR` | none | 2 |
| `sidewall_saturation_current` | `JSW`, `ISW` | A | 0 |
| `sidewall_emission_coefficient` | `NS` | none | the emission coefficient |
| `sidewall_series_resistance` | `RSW` | Ω | none: sidewall in parallel with bottom junction |
| `sidewall_junction_capacitance` | `CJP`, `CJSW` | F | 0 |
| `sidewall_junction_potential` | `PHP`, `VJSW` | V | 1 V |
| `sidewall_grading_coefficient` | `MJSW` | none | 0.33 |
| `sidewall_depletion_coefficient` | `FCS` | none | 0.5 |
| `sidewall_forward_knee_current` | `IKP` | A | none |
| `tunneling_saturation_current`, `sidewall_tunneling_saturation_current` | `JTUN`, `JTUNSW` | A | 0: no tunneling |
| `tunneling_emission_coefficient` | `NTUN` | none | 30 |
| `tunneling_saturation_current_temperature_exponent` | `XTITUN` | none | 3 |
| `tunneling_energy_gap_factor` | `KEG` | none | 1 |
| `breakdown_voltage` | `BV`, `VB`, `VRB`, `VAR` | V | none: no breakdown |
| `breakdown_current` | `IBV`, `IB` | A | 1 mA |
| `breakdown_emission_coefficient` | `NBV`, `NZ` | none | the emission coefficient |
| `breakdown_temperature_coefficient` | `TCV`, `TBV1` | V/K at temperature model 0, /K at 1 and 2 | 0 |
| `temperature_model` | `TLEV` | 0, 1 or 2 | 0 |
| `capacitance_temperature_model` | `TLEVC` | 0 or 1 | 0 |
| `energy_gap` | `EG` | V | 1.11 V, silicon's, and 1.16 V at temperature model 2 |
| `energy_gap_alpha`, `energy_gap_beta` | `GAP1`, `GAP2` | V/K, K | 702 µV/K, 1108 K: silicon's |
| `saturation_current_temperature_exponent` | `XTI` | none | 3 |
| `junction_capacitance_temperature_coefficient`, `sidewall_junction_capacitance_temperature_coefficient` | `CTA`, `CTC`; `CTP` | /K | 0 |
| `junction_potential_temperature_coefficient`, `sidewall_junction_potential_temperature_coefficient` | `TPB`, `TVJ`; `TPHP` | V/K | 0 |
| `series_resistance_temperature_coefficient`, `series_resistance_quadratic_temperature_coefficient` | `TRS`, `TRS1`; `TRS2` | /K, /K² | 0 |
| `transit_time_temperature_coefficient`, `transit_time_quadratic_temperature_coefficient` | `TTT1`, `TTT2` | /K, /K² | 0 |
| `grading_coefficient_temperature_coefficient`, `grading_coefficient_quadratic_temperature_coefficient` | `TM1`, `TM2` | /K, /K² | 0 |
| `nominal_temperature` | `TNOM`, `TREF` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |

A non-zero series resistance adds internal unknown `D1:junction` (plus `D1:junction_current` when below 1 Ω).
A sidewall series resistance similarly adds `D1:sidewall` and, below 1 Ω, `D1:sidewall_current`.

The bottom junction matches SPICE, saturating at reverse current `-Is` below `-3 Vt`.
Recombination adds `Isr · (e^(v/(NR·Vt)) - 1) · ((1 - v/VJ)² + 0.005)^(M/2)` (clamped below `-3 NR Vt`), and tunneling subtracts `Itun · (e^(-v/(NTUN·Vt)) - 1)`.
High-injection knee currents `Ik` soften current `I` to `I / (1 + √(|I|/Ik))` above `1e-18` A in forward (`IKF`) or reverse (`IKR`) bias.
The sidewall junction models independent saturation, emission, tunneling, knee, and depletion charge in parallel with the bottom junction behind `RS`, or behind `RSW` directly from the anode (without breakdown).
When `breakdown_voltage` is specified, reverse current grows exponentially past breakdown, shifting the onset voltage identically to ngspice so combined current matches `IBV` at `BV`.

The SPICE reader scales bottom-junction currents, capacitance, and inverse resistance by `AREA · M`, sidewall terms by `PJ · M`, and breakdown current by `M` (plus `AREA` at level 3).
Level 3 elements specifying width `W` and length `L` compute area `(W + XW) · (L + XW)` and perimeter `2 · (W + XW) + 2 · (L + XW)` scaled by library `scale`.
Level 3 metal/poly overlap terms (`LM`, `LP`, `WM`, `WP`), soft reverse recovery (`VP`), and ngspice internal self-heating (`RTH0`, `CTH0`) are unsupported (use [self-heating](#self-heating) instead).

For temperature models 0 and 1, saturation currents scale by `(T/Tn)^(XTI/n) · exp((T/Tn - 1) · Eg / (n · Vt))` (and tunneling by `XTITUN`, `NTUN`, `KEG · Eg`).
Temperature model 2 applies Varshni's band gap `Eg(T) = EG - GAP1 · T² / (T + GAP2)` with exponent `Eg(Tn)/Vt(Tn) - Eg(T)/Vt`.
Capacitance temperature model 0 scales potentials and capacitances via silicon band gap (or model 2 `Eg`), while model 1 applies linear coefficients from `TNOM`.
Breakdown voltage shifts linearly (absolute at model 0, relative at models 1 and 2), and series resistances, transit time, and grading coefficients follow linear and quadratic temperature coefficients matching ngspice.
Where ngspice diverges from diode equations (omitted `NS` scaling, `RSW` sidewall bugs, doubled level 3 multiplicity, inexact recombination conductance, and hardcoded 27 °C `TLEVC = 1`), jolty follows the equations ([departures](departures.md)).

In [noise analysis](simulation.md#noise), series resistances add thermal noise `4kT/R`, while junction currents `I` (bottom plus parallel sidewall) add shot noise `2q·|I|` and flicker noise `KF·|I|^AF/f`.

## JFET

`jfet(...)` models a junction field-effect transistor across `drain`, `gate`, and `source` using the SPICE square-law model with the Sydney University doping tail, matching ngspice.
Two gate-channel junctions couple the gate, drain current follows the square law above threshold, and drain and source swap symmetrically when `vds < 0`.
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
| `threshold_voltage_temperature_coefficient` | `VTOTC`, or `-TCV` | V/K | 0 |
| `transconductance_parameter_temperature_exponent` | `BEX` | none | 0 |
| `energy_gap` | `EG` | V | 1.11 V, silicon's |
| `saturation_current_temperature_exponent` | `XTI` | none | 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |

Drain and source resistances add internal nodes `J1:drain` and `J1:source` (plus `J1:drain_current` and `J1:source_current` below 1 Ω).

Away from nominal temperature, gate-junction saturation current, potential, and capacitances scale with band gap like a diode, threshold shifts linearly by `VTOTC`, and transconductance scales as `(T/Tn)^BEX`.
ngspice scales JFET saturation current by `XTI` only when explicitly provided, so `saturation_current_temperature_exponent` defaults to 0.

In [noise analysis](simulation.md#noise), resistances generate thermal noise, the channel generates thermal noise `8kT·gm/3` A²/Hz, and drain current `Id` generates flicker noise `KF·|Id|^AF/f`.

## Bipolar transistor

`bipolar_transistor(...)` models a Gummel-Poon BJT across `collector`, `base`, and `emitter`, matching ngspice.
Transport current scales inversely with base charge (capturing Early effect and high injection), while base resistance decreases from `base_resistance` toward `min_base_resistance` with rising base charge or current.
Every parameter is optional; omitted Early voltages, knee currents, and transit-time voltages default to infinity:

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
| `base_resistance`, `min_base_resistance` | `RB`, `RBM` | Ω | 0, the base resistance |
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
| `substrate` | `SUBS`, 1 or -1 | `"vertical"` or `"lateral"` | none: no substrate |
| `substrate_saturation_current`, `substrate_emission_coefficient` | `ISS`, `NS` | A, none | none: no diode, 1 |
| `substrate_capacitance`, `substrate_potential`, `substrate_grading` | `CJS`, `VJS`, `MJS` | F, V, none | 0, 0.75 V, 0 |
| `energy_gap` | `EG` | V | 1.11 V, silicon's |
| `saturation_current_temperature_exponent`, `beta_temperature_exponent` | `XTI`, `XTB` | none | 3, 0 |
| `temperature_model`, `capacitance_temperature_model` | `TLEV`, `TLEVC` | 0 or 1 | 0, 0 |
| `high_injection_exponent` | `NKF` | none | 0.5 |
| `<parameter>_temperature_coefficient`, `<parameter>_quadratic_temperature_coefficient` | `TBF1`, `TBF2`, `TRC1`, `TRC2` and kin | 1/K, 1/K² | none for the betas, 0 for others |
| `base_emitter_capacitance_temperature_coefficient` (plus collector and substrate) | `CTE`, `CTC`, `CTS` | 1/K | 0 |
| `base_emitter_potential_temperature_coefficient` (plus collector and substrate) | `TVJE`, `TVJC`, `TVJS` | V/K | 0 |
| `nominal_temperature` | `TNOM`, `TREF` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |

Base, collector, and emitter resistances add internal nodes `Q1:base`, `Q1:collector`, and `Q1:emitter`, plus corresponding `*_current` unknowns when resistance (or `min_base_resistance`) is below 1 Ω.

Excess phase delays forward transport current `Ibe / qb` by `td = PTF · π / 180 · TF` via the second-order Bessel filter `3 / (s² td² + 3 s td + 3)`.
Filter integration matches ngspice transient behavior and produces phase shift near `PTF` at `1 / (2π · TF)`.
The Bessel filter adds unknowns `Q1:delayed_transport` and `Q1:delayed_transport_slope` across all analyses, exposing both poles in pole-zero analysis.
ngspice AC instead applies a pure delay `exp(-jω·td)`, causing AC gain at `1 / (2π · TF)` to differ by the Bessel filter magnitude (5% at `PTF = 30°`).

Setting `substrate` to `"vertical"` or `"lateral"` adds a fourth `substrate` terminal connected via a junction to the internal collector (vertical) or internal base (lateral).
The substrate junction conducts `gmin` plus diode current when `substrate_saturation_current` is set, with linear forward depletion capacitance; substrate parameters require an active substrate.

Temperature scaling matches ngspice 47 `bjttemp.c`: saturation and leakage currents scale with band gap, and betas scale as `(T/Tn)^XTB` (`TLEV = 0`) or `1 + XTB·ΔT` (`TLEV = 1`, also dividing leakage currents).
Junctions use band-gap (`TLEVC = 0`) or linear (`TLEVC = 1`) scaling.
HSPICE quadratic polynomials `1 + c1·ΔT + c2·ΔT²` scale Early voltages (`TVAF*`, `TVAR*`), knee currents (`TIKF*`, `TIKR*`), and resistances (`TRB*`, `TRM*`, `TRC*`, `TRE*`, `TIRB*`).
They also scale emission coefficients (`TN*`), transit times (`TTF*`, `TTR*`, `TITF*`), and grading coefficients (`TMJ*`, capped at 0.999).
Explicit `TBF1`/`TBF2` or `TBR1`/`TBR2` coefficients override `XTB` on beta while leaving `XTB` active on leakage currents; excess phase always uses nominal `TF`.
`TLEV = 2` and `TLEV = 3` are rejected by name (while unused `TIS1` to `TISS2` coefficients are ignored at `TLEV` 0 or 1).
`high_injection_exponent` (`NKF`, clamped at 1) sets `qb = q1 · (1 + (1 + 4·q2)^NKF) / 2`.
The SPICE reader skips undefined HSPICE parameters (`DCAP`, `GAP1`, `GAP2`, `CBCP`, `CBEP`, `CCSP`, and solitary `IBE` or `IBC`) and rejects paired `IBE`/`IBC` by name.

In [noise analysis](simulation.md#noise), terminal resistances (with bias-dependent base resistance) add thermal noise, collector current adds shot noise, and base current `Ib` adds shot noise plus flicker noise `KF·|Ib|^AF/f`.

ngspice omits bias-dependent base-resistance derivatives from Newton steps and AC models, whereas jolty differentiates all terms exactly for consistent small-signal slopes and faster Newton convergence.

## MOSFET

`mosfet(...)` models a four-terminal MOSFET (`drain`, `gate`, `source`, `bulk`) using SPICE level 1 (Shichman-Hodges square law with Meyer capacitances) or level 3 (`level=3`, semi-empirical short-channel model), matching ngspice.
Above threshold, drain current follows the channel model with body-effect threshold modulation, swapping drain and source symmetrically when `vds < 0`.
Bulk-drain and bulk-source junctions couple the substrate, and the gate holds overlap charges plus Meyer capacitances when `oxide_thickness` is given.
When `oxide_thickness` and `substrate_doping` are set, omitted `threshold_voltage`, `body_effect`, and `surface_potential` derive automatically from process physics as in SPICE.
Every parameter is optional:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `channel` | `NMOS`, `PMOS` | `"n"` or `"p"` | `"n"` |
| `width`, `length` | `W`, `L` | m | 100 µm, 100 µm |
| `multiplier` | `M` | none | 1 |
| `threshold_voltage` | `VTO` | V | 0, or derived |
| `transconductance_parameter` | `KP` | A/V² | 20 µA/V², or surface mobility times oxide capacitance per area |
| `body_effect` | `GAMMA` | √V, as a number | 0, or derived |
| `surface_potential` | `PHI` | V | 0.6 V, or derived |
| `channel_length_modulation` | `LAMBDA` | V⁻¹ | 0, at level 1 only |
| `lateral_diffusion` | `LD` | m | 0 |
| `oxide_thickness` | `TOX` | m | none: no Meyer capacitances |
| `surface_mobility` | `UO` | m²/(V·s) | 0.06 m²/(V·s), SPICE's 600 cm²/(V·s) |
| `substrate_doping` | `NSUB` | m⁻³ | none |
| `surface_state_density` | `NSS` | m⁻² | 0 |
| `gate_material` | `TPG` | `"opposite_doping"`, `"same_doping"` or `"aluminum"` | `"opposite_doping"` |
| `drain_resistance`, `source_resistance` | `RD`, `RS` | Ω | from sheet resistance |
| `sheet_resistance`, `drain_squares`, `source_squares` | `RSH`, `NRD`, `NRS` | Ω, none, none | 0, 1, 1 |
| `saturation_current`, `saturation_current_density` | `IS`, `JS` | A, A/m² | 10 fA, 0 |
| `drain_area`, `source_area` | `AD`, `AS` | m² | 0 |
| `drain_perimeter`, `source_perimeter` | `PD`, `PS` | m | 0 |
| `junction_potential` | `PB` | V | 0.8 V |
| `bulk_drain_capacitance`, `bulk_source_capacitance` | `CBD`, `CBS` | F | from bottom capacitance |
| `bottom_capacitance`, `bottom_grading` | `CJ`, `MJ` | F/m², none | 0, 0.5 |
| `sidewall_capacitance`, `sidewall_grading` | `CJSW`, `MJSW` | F/m, none | 0, 0.5 at level 1 |
| `depletion_coefficient` | `FC` | none | 0.5 |
| `gate_source_overlap`, `gate_drain_overlap`, `gate_bulk_overlap` | `CGSO`, `CGDO`, `CGBO` | F/m | 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |
| `level` | `LEVEL` | 1 or 3 | 1 |

Level 3 replaces the square law with short-channel physics, disables `channel_length_modulation`, defaults `oxide_thickness` to 0.1 µm and `sidewall_grading` to 0.33, and adds these parameters:

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

Channel shortening past saturation activates only when `substrate_doping` is provided.

SPICE expresses mobility, doping, and surface-state density in centimeters, whereas jolty uses SI units throughout.
Non-zero `saturation_current_density` overrides `saturation_current` whenever both drain and source areas are positive.

Away from nominal temperature, `KP` and mobility scale as `(T/Tn)^(-1.5)`, surface potential and threshold follow silicon band gap and intrinsic concentration, and bulk junctions scale via band gap as in ngspice.
Derived process parameters evaluate at nominal temperature.

Drain and source resistances add internal nodes `M1:drain` and `M1:source` (plus `M1:drain_current` and `M1:source_current` below 1 Ω).
Meyer capacitances derive from no scalar charge function, so jolty stamps `C(v) · dv/dt` using displacement-current unknowns `M1:gate_source_displacement`, `M1:gate_drain_displacement`, and `M1:gate_bulk_displacement`.
The sparse solver statically condenses all three displacements using unit pivots before factoring and recovers them afterward, matching ngspice's matrix dimension while still solving, error-checking, and recording displacements.
Drain and source branch currents below 1 Ω condense similarly via their internal node equations (pivot `-1`), reducing a two-switch buck converter from 23 to 13 factored unknowns (matching ngspice).
AC analyses evaluate Meyer capacitances identically to ngspice.
In transients, ngspice integrates `q₁ + C̄ · Δv` using derivative `C̄` alone (`mos1load.c`, lines 821 to 827, 863 to 868), whereas jolty solves `C(v) · dv/dt` with exact derivatives; neither conserves charge.
Transients control truncation error across all node charges including bulk junctions, whereas ngspice levels 1 to 3 check only gate charges (`mos1trun.c`, lines 24 to 26).
Under forward bulk bias, jolty evaluates exact threshold derivatives instead of ngspice's reverse-bias slope formula.

In [noise analysis](simulation.md#noise), resistances generate thermal noise, the channel generates `8kT·gm/3` A²/Hz, and `m` parallel copies generate summed flicker noise `m·KF·(|Id|/m)^AF/(f·W·Leff·Cox²)` (using 0.1 µm default oxide thickness).
SPICE export writes flicker noise using `nlev=1` and coefficient `KF/Cox`, supported by ngspice 43 and later.

## BSIM3

`bsim3(...)` models a four-terminal short-channel CMOS transistor (0.8 µm to 0.13 µm, SPICE levels 8 and 49) implementing ngspice 47 BSIM3v3.3 per Liu, Xi, et al. (UC Berkeley, 2005).
Drain current includes short-channel, narrow-width, and DIBL thresholds, poly depletion, `mobility_model` 1 to 3, velocity saturation, channel-length modulation, output resistance, `rdsw`, and drain-bulk impact ionization.
Gate charges implement `capacitance_model` 0 to 3 with `charge_partition` and overlap charges, alongside bulk-drain and bulk-source diodes with depletion charges.
The transistor is symmetric, swapping drain and source when `vds < 0`.
jolty's BSIM3 derives from BSIM3v3.3.0 (UC Berkeley BSIM Research Group, Copyright 2004 Regents of the University of California), governed by BSIM terms reproduced in `NOTICE`.

Instances specify MOSFET geometry plus a `card` dictionary of manual parameter names in SI units (`card={"vth0": 0.5 * V, "tox": 7.6 * nm, ...}`):

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `channel` | `NMOS`, `PMOS` | `"n"` or `"p"` | `"n"` |
| `width`, `length` | `W`, `L` | m | 5 µm, 5 µm |
| `multiplier` | `M` | none | 1 |
| `drain_area`, `source_area` | `AD`, `AS` | m² | 0 |
| `drain_perimeter`, `source_perimeter` | `PD`, `PS` | m | 0 |
| `drain_squares`, `source_squares` | `NRD`, `NRS` | none | 1, 1 |
| `threshold_adjustment`, `mobility_multiplier` | `DELVTO`, `MULU0` | V, none | 0, 1 |
| `mobility_model`, `capacitance_model`, `noise_model` | `MOBMOD`, `CAPMOD`, `NOIMOD` | 1 to 3, 0 to 3, 1 to 6 | 1, 3, 1 |
| `check_parameters` | `PARAMCHK = 1` | `True` or `False` | `False` |
| `charge_partition` | `XPART` | none | 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `card` | each by its name | SI | ngspice's |

Each card parameter expects a quantity matching `devices.bsim3_dimension(name)` (or a plain float when dimensionless), deriving dependent defaults (such as `vth0`, `k1`, `cgso`) identically to ngspice.
Binnable parameters accept `l`, `w`, and `p` prefixes multiplying reciprocal effective length, width, and area in meters (`lvth0` equals a `BINUNIT = 1` SPICE value times `1e-6`).
While SPICE mixes centimeter and meter conventions for `NCH`, `NGATE`, `NSUB`, and `U0`, jolty uses strict SI units while keeping `rdsw` and `prt` in ohms for a 1 µm channel as on SPICE cards.
Parameters `ll`, `lw`, `lwl`, `wl`, `ww`, `wwl` (and charge counterparts) are plain numbers scaled by exponents `lln`, `lwn`, `wln`, `wwn`, and `noia`/`noib`/`noic` use manual units.

Temperature scaling follows ngspice: mobility scales as `(T/Tn)^ute`, `ua`/`ub`/`uc`/`vsat`/`rdsw` scale via linear coefficients, threshold via `kt1`/`kt1l`/`kt2`, and junctions via `xti`, `tcj`, and `tpb`.
BSIM3 uses fixed internal physical constants, so SPICE export keeps the exact kelvin nominal temperature and both engines agree without thermal-voltage offsetting.
Heated BSIM3 instances evaluate at their thermal-net temperature ([Self-heating](#self-heating)).

Fatal ngspice parameter errors fail elaboration, while clamped values (`pb`, `pbsw`, `pbswg` floored at 0.1 V; `a2`, `rdsw`, and overlaps under `check_parameters`) match ngspice clamping.
Sheet resistance `rsh` adds internal nodes `M1:drain` and `M1:source` (plus branch currents below 1 Ω).
Any active gate, drain, or junction limiter prevents Newton convergence, whereas ngspice BSIM3 blocks convergence only on the junction limiter (`b3ld.c`, lines 187, 359 to 366, 2437 to 2438).
Transients check truncation error on every node charge, whereas ngspice checks bulk, gate, and drain charges with junctions folded in (`b3ld.c`, lines 2797 to 2802).
jolty computes exact derivatives everywhere, whereas ngspice drops forward-bias bulk-clamp slope and double-counts poly-depletion slope in `capmod = 0` AC responses.

In [noise analysis](simulation.md#noise), sheet resistances add thermal noise while channel thermal noise uses SPICE2 (`8kT·|gm + gds + gmbs|/3` at `noise_model` 1, 3, 5, 6) or BSIM3 inversion charge (2, 4).
Channel flicker noise uses SPICE2 (`kf·|Id|^af/(f^ef·Leff²·Cox)` at `noise_model` 1, 4, 5) or oxide traps (`noia`, `noib`, `noic` at 2, 3, 6), summed across all `m` parallel copies.

## BSIM4

`bsim4(...)` models a sub-130 nm CMOS MOSFET (`drain`, `gate`, `source`, `bulk`, SPICE levels 14 and 54, version 4.5) matching ngspice 47 `bsim4v5` per Xi, Dunga, et al. (UC Berkeley, 2005).
Supported modes cover all SKY130 and GF180MCU version 4.5 cards, including ngspice defaults (`mobmod=0`, `capmod=2`, `diomod=1`, `tempmod=0`, `geomod=0`, `rgeomod=0`, `igcmod=0`, `igbmod=0`, `trnqsmod=0`, `acnqsmod=0`, `wpemod=0`).
They also include `rgatemod` 0 to 3 and values 0 or 1 for `rbodymod`, `rdsmod`, `permod`, `fnoimod`, and `tnoimod`.
Cards selecting any other mode fail elaboration by name.
Drain current models short-channel, narrow-width, pocket, and DIBL thresholds, poly depletion, gate-field/Coulomb/length mobility degradation, velocity saturation and overshoot, source velocity limit, output resistance, `rdsw`, substrate impact ionization, and GIDL/GISL currents.
Gate charges implement `capmod = 2` with charge thickness, `charge_partition`, and bias-dependent overlaps alongside body-drain and body-source diodes with trap-assisted tunneling.
Length-of-diffusion stress (`SA`, `SB`) modulates threshold, `k2`, `eta0`, and saturation velocity while leaving mobility unstressed under `MULU0` to match ngspice 47 (`b4v5temp.c`, lines 1359 and 1420).
The transistor is symmetric, swapping drain and source when `vds < 0`.
jolty's BSIM4 derives from BSIM4.5.0 (UC Berkeley BSIM Research Group, Copyright 2005 Regents of the University of California), governed by BSIM4 terms reproduced in `NOTICE`.

Instances specify MOSFET geometry, fingers, and a `card` dictionary using manual names in SI units (`card={"vth0": 0.5 * V, "toxe": 4.1 * nm, ...}`):

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `channel` | `NMOS`, `PMOS` | `"n"` or `"p"` | `"n"` |
| `width`, `length` | `W`, `L` | m | 5 µm, 5 µm |
| `multiplier`, `fingers` | `M`, `NF` | none | 1, 1 |
| `drain_area`, `source_area` | `AD`, `AS` | m² | from fingers and `dmcg` |
| `drain_perimeter`, `source_perimeter` | `PD`, `PS` | m | from fingers and `dmcg` |
| `drain_squares`, `source_squares` | `NRD`, `NRS` | none | none |
| `diffusion_edge_a`, `diffusion_edge_b`, `finger_spacing` | `SA`, `SB`, `SD` | m | 0, 0, `2·dmcg` |
| `threshold_adjustment`, `mobility_multiplier` | `DELVTO`, `MULU0` | V, none | 0, 1 |
| `body_resistance_model`, `perimeter_model` | `RBODYMOD`, `PERMOD` | 0 or 1 | 0, 1 |
| `gate_resistance_model` | `RGATEMOD` | 0 to 3 | 0 |
| `check_parameters` | `PARAMCHK = 1` | `True` or `False` | `True` |
| `charge_partition` | `XPART` | none | 0 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `card` | each by its name | SI | ngspice's |

Each card parameter expects a quantity matching `devices.bsim4_dimension(name)` (or a plain float when dimensionless), deriving dependent defaults (`vth0`, `k1`, `toxp`, drain junctions from source) identically to ngspice.
Binnable parameters accept `l`, `w`, and `p` prefixes in reciprocal meters (`lvth0` scales by `1e-6` at `BINUNIT = 1` and directly at `BINUNIT = 2`).
`NDEP`, `NSD`, `NGATE`, `NSUB`, and `U0` take strict SI units (unlike SPICE centimeter conventions), while `rdsw` and `prt` stay in ohms per 1 µm channel width.
Geometry terms `ll`, `lw`, `lwl`, `wl`, `ww`, `wwl` (and charge counterparts) are plain numbers scaled by exponents, and tunneling/flicker coefficients follow manual units.
Sheet resistance `rsh` creates internal drain/source nodes (`M1:drain`, `M1:source`) when `NRD`/`NRS` is positive (`rdsmod = 0`).
Mode `rdsmod = 1` adds bias-dependent external `rdw`/`rsw` plus `rsh` (flooring missing `nrd`/`nrs` at 1 mΩ) and removes `rdsw` from channel current.
Setting `body_resistance_model = 1` adds three substrate nodes (`M1:body`, `M1:drain_body`, `M1:source_body`) linked via `rbpb`, `rbpd`, `rbps`, `rbdb`, and `rbsb`.
`gate_resistance_model` (`RGATEMOD`) inserts internal node `M1:gate` using electrode resistance `rshg·(xgw + Weffcj/(3·ngcon))/(ngcon·nf·(l + xl - xgl))` (floored at 1 mΩ) at mode 1.
Mode 2 adds series intrinsic-input conductance `xrcrg1·(xrcrg2·kT/q·β + Ids/Vdseff)·nf`, and mode 3 splits overlap capacitance onto `M1:middle_gate`.
Instance `XGW` and `NGCON` override card defaults; `ngcon < 1` and `xgl >= l + xl` fail elaboration identically to ngspice.

Temperature scaling follows `tempmod = 0`: mobility as `(T/Tn)^ute`, `ua`/`ub`/`uc`/`ud`/`vsat`/`rdsw`/`voff` linearly, threshold via `kt1`/`kt1l`/`kt2`, and junctions via `xtis`/`xtid`, `xtss*`, `tcj`, and `tpb`.
BSIM4 uses internal physical constants like BSIM3, so SPICE export preserves exact kelvin temperatures.
Fatal ngspice checks fail elaboration and warnings clamp identically (`pbs*`/`pbd*` at 0.1 V, `ckappas`/`ckappad` at 0.02 V, plus `check_parameters` bounds on `a2`, `prwg`, `rdsw`, and `rdswmin`).
Active gate, drain, body, or junction limiters prevent Newton convergence.
jolty differentiates all charges and currents exactly, whereas ngspice `capmod = 2` body capacitances diverge from charge derivatives.
On SKY130 `nfet_01v8` (1 µm × 0.15 µm, `tt`, `Vgs = Vds = 0.9` V), ngspice `cgb` misses finite differences by `1.2e-4` at +0.2 V and `3.4e-5` at -0.3 V bulk bias.
Under the same bias, jolty matches gate/drain drives within `1e-8` and body conductances within `4e-9`.

The SPICE reader loads level 8 and 49 version 3.3 cards for BSIM3 (rejecting versions 3.0 to 3.2) and selects binned cards (`name.k`) using `LMIN`/`LMAX`/`WMIN`/`WMAX` with ngspice's 1 nm tolerance.
Unused metadata (bin bounds, SOA limits like `VGS_MAX`) is ignored, while unsupported non-quasi-static or HSPICE area modes (`NQSMOD`, `ACNQSMOD`, `ACM != 0`) are rejected by name.

In [noise analysis](simulation.md#noise), drain, source, body, and gate-electrode resistances generate thermal noise (electrode only at `RGATEMOD = 2`, matching `b4v5noi.c`).
Channel thermal noise implements charge-based `TNOIMOD = 0` (`4kT·NTNOI·µeff·|Qinv|/(Leff² + µeff·|Qinv|·Rds)`) and holistic `TNOIMOD = 1` (partitioning `RNOIB`/`TNOIB` onto the source side).
Channel flicker noise implements SPICE2 (`FNOIMOD = 0`: `KF·|Id|^AF/(f^EF·Leff²·Coxe)`) and oxide-trap (`FNOIMOD = 1`: `NOIA`, `NOIB`, `NOIC` combining strong and weak inversion in parallel).
ngspice 47 injects 1 mΩ drain/source resistances in noise-only decks at `TNOIMOD = 1` (`b4v5set.c`, lines 1823 and 1870; `b4v5temp.c`, lines 1638 and 1669), while jolty leaves the circuit unchanged.
On a SKY130 `nfet_01v8` stage, output noise agrees within `4.2e-7` without series resistances and `1.4e-7` with explicit resistances.

ngspice 47 computes gate-electrode resistance from the last prepared bin's `Lnew` (`b4v5temp.c`, lines 382 and 1562) and clamps non-1/2 `ngcon` only after first preparation (`b4v5check.c`, line 262; `b4v5temp.c`, line 1825).
jolty always uses each instance's true drawn length and specified `ngcon`.
On SKY130 RF PMOS `rf_pfet_01v8_bM04W3p00L0p18` at `tt` (`Vgs = Vds = -0.9` V), ngspice output conductance misses central differences by `2.1e-4` whereas jolty matches within `1.4e-9`.
Setting `eta0 = 0` or `drout = 0` restores agreement, and RF NMOS matches within `2.2e-7`.

## VBIC

`vbic(...)` models an integrated-circuit bipolar transistor (`collector`, `base`, `emitter`, `substrate`) implementing ngspice 47 VBIC 1.2 (McAndrew et al., *IEEE JSSC*, 1996, extended to 1.2).
In addition to Gummel-Poon physics, VBIC models the parasitic substrate transistor, weak base-collector avalanche, Kull quasi-saturation, base-emitter breakdown, base-collector reach-through, and smoothed junction capacitances.
Parameters use descriptive SI names (such as `extrinsic_collector_resistance` for `RCX`) with ngspice defaults, listed completely in the [models reference](../reference/models.md#vbic).

Extrinsic collector, base, emitter, and substrate resistances add internal nodes `Q1:collector`, `Q1:base`, `Q1:emitter`, and `Q1:substrate`; intrinsic collector, intrinsic base, and parasitic base are always internal, and excess phase adds `Q1:transport_filter` and `Q1:delayed_transport`.
PNP excess phase mirrors NPN exactly, whereas ngspice 47 omits polarity on excess-phase nodes (`vbicload.c`, lines 449, 485 to 486, 1139), giving 3.01 mA DC collector current instead of the 56.7 µA mirror value.
Intrinsic collector, intrinsic base, and parasitic base resistances must be positive (floored at 10 mΩ on SPICE import, matching ngspice, and rejected below 10 mΩ on export).

VBIC uses fixed physical constants (`k = 1.380662e-23` J/K, `q = 1.602189e-19` C) in both jolty and ngspice, so the engines agree at identical kelvins.
Temperature scaling follows ngspice VBIC across activation energies, resistance/knee/epi exponents, intrinsic-carrier potentials/capacitances, and linear emission/avalanche/breakdown coefficients.
Heated VBIC instances evaluate at thermal-net temperature with power equal to total terminal power, translating ngspice self-heating elements onto the thermal net.

In [noise analysis](simulation.md#noise), every resistance adds thermal noise, transport/base/parasitic currents add shot noise, and base currents add flicker noise `KFN·I^AFN/f^BFN`.
Parasitic base-resistance thermal noise stamps across the parasitic base resistance rather than the emitter resistance as in ngspice 47 (`vbicnoise.c`).

The SPICE reader imports level 4 and level 9 NPN/PNP cards and three- to five-node `Q` elements, converting five-node heated devices to heated VBIC plus thermal elements while leaving unheated fifth nodes unconnected.
Unused fields (`VERS`, `VREF`, `EBBE`, SOA bounds) are ignored, `DTEMP` maps to temperature offset, and model levels outside 1, 4, and 9 are rejected by name.

## Power MOSFET

`vdmos(...)` models a three-terminal vertical double-diffused power MOSFET (`drain`, `gate`, `source`) matching ngspice 47 VDMOS and LTspice power-MOSFET cards.
Channel conduction uses a body-effect-free square law with softened subthreshold overdrive, gate-source capacitance is constant, gate-drain capacitance transitions smoothly between `CGDMIN` and `CGDMAX`, and a source-to-drain body diode provides reverse conduction and breakdown.
Every parameter is optional:

| Parameter | SPICE | Unit | Default |
| --- | --- | --- | --- |
| `channel` | `VDMOS`, `VDMOS pchan` | `"n"` or `"p"` | `"n"` |
| `multiplier` | `M` | none | 1 |
| `threshold_voltage` | `VTO` | V | 3 V, or -3 V for p-channel |
| `transconductance_parameter` | `KP` | A/V² | 20 A/V², or 10 A/V² for p-channel |
| `channel_length_modulation` | `LAMBDA` | V⁻¹ | 0 |
| `mobility_modulation` | `THETA` | V⁻¹ | 0 |
| `triode_conductance_multiplier` | `MTRIODE` | none | 1 |
| `subthreshold_slope`, `subthreshold_shift` | `KSUBTHRES`, `SUBSHIFT` | V | 0.1 V, 0 |
| `drain_resistance`, `source_resistance`, `gate_resistance` | `RD`, `RS`, `RG` | Ω | 0 |
| `quasi_saturation_resistance`, `quasi_saturation_voltage` | `RQ`, `VQ` | Ω, V | none: no quasi-saturation |
| `drain_source_resistance` | `RDS` | Ω | none: 1 fS from drain to source |
| `gate_source_capacitance` | `CGS` | F | 1.4 nF |
| `max_gate_drain_capacitance`, `min_gate_drain_capacitance` | `CGDMAX`, `CGDMIN` | F | 2 nF, 20 pF |
| `gate_drain_capacitance_coefficient` | `A` | V⁻¹ | 1 |
| `saturation_current`, `emission_coefficient` | `IS`, `N` | A, none | 10 fA, 1 |
| `body_resistance` | `RB` | Ω | 0 |
| `junction_capacitance`, `junction_potential`, `grading_coefficient` | `CJO`, `VJ`, `M` | F, V, none | 0.5 nF, 0.8 V, 0.5 |
| `depletion_coefficient` | `FC` | none | 0.5 |
| `transit_time` | `TT` | s | 0 |
| `breakdown_voltage` | `BV` | V | none: no breakdown |
| `breakdown_current`, `breakdown_emission_coefficient` | `IBV`, `NBV` | A, none | 0.1 nA, 1 |
| `energy_gap`, `saturation_current_temperature_exponent` | `EG`, `XTI` | V, none | 1.11 V, 3 |
| `nominal_temperature` | `TNOM` | K | 27 °C |
| `threshold_voltage_temperature_coefficient` | `TCVTH`, `VTOTC` | V/K | 0 |
| `transconductance_parameter_temperature_exponent` | `MU`, `BEX` | none | -1.5 |
| `subthreshold_slope_temperature_coefficient`, `subthreshold_slope_quadratic_temperature_coefficient` | `TKSUBTHRES1`, `TKSUBTHRES2` | K⁻¹, K⁻² | 0 |
| `drain_resistance_temperature_exponent` | `TEXP0` | none | none: use drain coefficients |
| `drain_resistance_temperature_coefficient`, `drain_resistance_quadratic_temperature_coefficient` | `TRD1`, `TRD2` | K⁻¹, K⁻² | 0 |
| `quasi_saturation_resistance_temperature_exponent` | `TEXP1` | none | 0.3 |
| `source_resistance_temperature_coefficient`, `source_resistance_quadratic_temperature_coefficient` | `TRS1`, `TRS2` | K⁻¹, K⁻² | 0 |
| `gate_resistance_temperature_coefficient`, `gate_resistance_quadratic_temperature_coefficient` | `TRG1`, `TRG2` | K⁻¹, K⁻² | 0 |
| `body_resistance_temperature_coefficient`, `body_resistance_quadratic_temperature_coefficient` | `TRB1`, `TRB2` | K⁻¹, K⁻² | 0 |
| `flicker_coefficient`, `flicker_exponent` | `KF`, `AF` | none | 0: no flicker noise, 1 |

Default threshold, transconductance, and capacitance values match ngspice (IRF540 and IRF9540).
With `vgst = vgs - vto` and subthreshold slope `k`, softened overdrive is `vgst' = k · ln(1 + exp((vgst - SUBSHIFT) / k))`.
Effective transconductance is `β' = KP · M · (1 + λ · vds) / (1 + θ · max(vgst, 0))`.
Drain current equals `β' · vgst'² / 2` in saturation (`vgst' ≤ vdss`) and `β' · vdss · (vgst' - vdss / 2)` in triode, where `vdss = vds · MTRIODE`.
When `vds < 0`, drain and source swap while channel-length modulation retains signed `1 + λ · vds` as in ngspice.
When both `RQ` and `VQ` are set alongside `RD`, forward drain resistance increases to `RD + RQ · vds / (vds + VQ)` for quasi-saturation; setting only one has no effect.

Gate-drain capacitance follows `s · atan(A · vgd) + y` for `vgd < 0` and `s · tanh(A · vgd) + y` for `vgd ≥ 0`.
Here `s = (CGDMAX - CGDMIN) / (1 + π/2)` and `y = CGDMAX - s`.
jolty integrates this capacitance analytically into a conserved gate charge, whereas ngspice applies step-by-step trapezoidal integration; both converge as time steps shrink.
The body diode connects source (behind `RB`) to drain, scaling saturation current and capacitances by multiplier `M` while keeping breakdown current unscaled as in ngspice.
Depletion thresholds `FC · VJ > 2.5` V at operating temperature are rejected to prevent ngspice's discontinuous charge jump.

Temperature scaling follows ngspice: `KP` scales as `(T/Tn)^MU`, threshold by `TCVTH`, subthreshold slope by `TKSUBTHRES*`, drain resistance by `(T/Tn)^TEXP0` or `TRD*`, quasi-saturation by `(T/Tn)^TEXP1`, remaining resistances quadratically, and the body diode like a standard diode.
Heated VDMOS instances use [self-heating](#self-heating) on a thermal net; ngspice's optional internal thermal keywords (`RTHJC`, `RTHCA`, `CTHJ`) are not imported.

Drain, source, gate, and body resistances add internal nodes `M1:drain`, `M1:source`, `M1:gate`, and `M1:body_diode` (plus `*_current` unknowns below 1 Ω).
jolty differentiates all terms exactly, whereas ngspice omits quasi-saturation and subthreshold mobility-modulation derivatives and flips reverse channel-length modulation slope in AC responses.

In [noise analysis](simulation.md#noise), drain and source resistances generate thermal noise, the channel generates `8kT·gm/3` A²/Hz, and drain current generates flicker noise `KF·|Id|^AF/(f·Cox²)` (`Cox = 0.35` mF/m²).
Gate resistance, body resistance, and the body diode remain noiseless as in ngspice.

The SPICE reader imports `VDMOS` cards (with `pchan` for p-channel) while skipping LTspice metadata (`MFG`, `VDS`, `RON`, `QG`), distortion-only `PHI`, and ngspice internal thermal parameters.
`M` elements accept three nodes (drain, gate, source) or four/five nodes (ignoring ngspice's trailing thermal pins).
