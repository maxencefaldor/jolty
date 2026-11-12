# Twenty-four analog circuits

Design each of the twenty-four circuits below from two transistors, resistors and capacitors on a 5 V supply.

## What to write

Write `p1.py` to `p24.py`, one script per circuit.
Each script's `build()` returns its circuit as a component whose ports are `vdd`, `ground`, and the circuit's inputs and outputs, named as the table names them.
A dictionary `LEVELS` beside it gives the level of each input whose check below holds it or speaks of its level: in volts, or in amperes for `iref`.
A script may import modules of your own from beside it.

The bench that checks a circuit holds `vdd` at 5 V over `ground`, holds each input with a voltage source from `ground`, or `iref` with a current from `vdd` into it, and reads each output's voltage.

Build only from `nmos(width, length)` and `pmos(width, length)` of `parts.py`, SPICE level 1 MOSFETs of any size, from resistors and capacitors, and from sources of a constant level.
The NMOS has a transconductance parameter of 100 µA/V² and a threshold of 0.5 V, and the PMOS 50 µA/V² and -0.5 V; neither has channel-length modulation, body effect or any capacitance.
No other device may appear in your circuits.

## The circuits

| # | Circuit | Inputs | Outputs |
| --- | --- | --- | --- |
| 1 | A single-stage common-source amplifier with resistive load R | `vin` | `vout` |
| 2 | A three-stage amplifier with single input and output (each stage is common-source with resistive load) | `vin` | `vout` |
| 3 | A common-drain amplifier (a.k.a. a source follower) with resistive load R (output Vout at the source) | `vin` | `vout` |
| 4 | A single-stage common-gate amplifier with resistive load R (input signal Vin must be applied at the source terminal) | `vin`, `vbias` | `vout` |
| 5 | A single-stage cascode amplifier with two NMOS transistors provides a single-ended output through a resistive load R | `vin`, `vbias` | `vout` |
| 6 | A NMOS inverter with resistive load R | `vin` | `vout` |
| 7 | A logical inverter with 1 NMOS and 1 PMOS | `vin` | `vout` |
| 8 | A simple NMOS constant current source with resistive load R | `vbias` | `vout` |
| 9 | A two-stage amplifier with a Miller compensation capacitor | `vin` | `vout` |
| 10 | A single-stage amplifier (common-source with PMOS diode-connected load (gate and drain are shorted)) | `vin` | `vout` |
| 11 | A differential opamp with an active PMOS current mirror load, a tail current source, and two outputs | `vinp`, `vinn`, `vbias` | `voutp`, `vout` |
| 12 | A cascode current mirror with 4 mosfets (2 stacked at input side with diode-connected, 2 stacked at output side), reference current source input Iref (connected to Vdd) and resistive load R | `iref` | `iout` |
| 13 | A single-stage differential common-source opamp with dual resistive loads, tail current, and a single output | `vinp`, `vinn` | `vout` |
| 14 | A two-stage differential opamp (first stage: common-source with an active load and a tail current, second stage: common-source with an active load) | `vinp`, `vinn`, `vbias1`, `vbias2`, `vbias3` | `voutp`, `vout` |
| 15 | A single-stage telescopic cascode opamp with two outputs (4 nmos as cascode input pair, 4 pmos as cascode loads, and 1 tail current) | `vinp`, `vinn`, `vbias1` to `vbias4` | `voutp`, `vout` |
| 16 | An RC phase-shift oscillator | | `vout` |
| 17 | A Wien Bridge oscillator | | `vout` |
| 18 | An Opamp integrator with resistor R1 and capacitor Cf | `vin` | `vout` |
| 19 | An Opamp differentiator with resistor Rf and capacitor C1 | `vin` | `vout` |
| 20 | An Opamp adder to make Vout=-(Vin1+Vin2) | `vin1`, `vin2` | `vout` |
| 21 | An Op-amp subtractor to make Vout=Vin2-Vin1 | `vin1`, `vin2` | `vout` |
| 22 | A non-inverting Schmitt trigger with positive feedback op-map | `vin` | `vout` |
| 23 | A voltage-controlled oscillator | `vin` | `vout` |
| 24 | A phase-locked loop | `clk_ref` | `clk_p` |

## How each is checked

A MOSFET is on and forward when its gate is beyond its threshold, and its drain beyond its source, in its channel's sense, both by more than 1 µV.

- **Amplifiers, 1 to 5, 9 and 10.**
  Every input is held.
  At the operating point every MOSFET is on and forward and carries at least 10 µA.
  `vout` moves by at least 1 mV as `vin` sweeps from 0 to 5 V, and the gain from `vin` to `vout` at 100 Hz exceeds 1e-5.
  Circuits 1 to 5 hold a resistor.
  In 3, `vout` joins no MOSFET's gate or drain; in 4, `vin` joins none.
  In 9, a capacitor joins `vout` to the drain of a MOSFET whose gate is `vin`.
  In 10, some MOSFET's gate joins its drain.
- **Inverters, 6 and 7.**
  `vout` is at least 2.5 V with `vin` at 0 V, at most 2.5 V with `vin` at 5 V, and the two differ by at least 1 V.
- **Current sources, 8 and 12.**
  Every input is held, and the bench loads the output with a resistor to `vdd` of 100, 300, 500, 750 and 1000 Ω in turn.
  The current through the load exceeds 10 µA and changes by less than 1 µA from each load to the next, and at 500 Ω every MOSFET is on and forward.
  In 12, the current changes by at least 1 µA when `iref` turns 1.55 mA.
- **Op-amps, 11, 13, 14 and 15.**
  Every input is held, `vinp` and `vinn` at the same level.
  At the operating point every MOSFET is on and forward and carries at least 10 µA.
  At 100 Hz, with `H(x)` the response of `vout` to input `x` alone, the differential gain `|H(vinp) - H(vinn)|` exceeds 1e-5, and the common-mode gain `|H(vinp) + H(vinn)|` is less than the differential gain less 1e-5.
  Circuit 13 holds a resistor.
- **Oscillators, 16 and 17.**
  The bench switches `vdd` on, from 0 to 5 V over 1 µs, and runs 10 ms: nothing else starts the oscillator, as a simulation has no noise.
  Over the second half, `vout` oscillates steadily: it swings at least 1 mV, rises at least four times through the middle of its swing, each time from 10% of the swing below it to 10% above, its period between rises varies by less than 20% of its mean, and its last period swings at least half as far as it does over the whole half.
- **Integrator, 18.**
  R1 is 10 kΩ and Cf 3 µF.
  The input `vin` is held at its level until 1 µs, then steps 0.5 V above it, and every 10 ms 0.5 V below it and back, each step over 1 µs, until 200 ms have passed.
  Over each of the last two halves of a period, but their first and last tenths, `vout` ramps against `vin` at 0.5 V / (10 kΩ · 3 µF), within 30%: falling while `vin` is above its level, rising while it is below, along a line that fits with R² of at least 0.9.
- **Differentiator, 19.**
  C1 is 3 µF and Rf 10 kΩ.
  The input `vin` ramps from 0.5 V below its level to 0.5 V above it over 50 ms, and back over 50 ms, twice.
  Over the second period, but each ramp's first tenth, `vout` sits 0.6 V below `vin`'s level while `vin` rises and 0.6 V above it while `vin` falls, within 20%, and varies by less than 60 mV rms.
- **Adder, 20.**
  Its inputs and output are measured from `vin1`'s level, `b`: `vin2` is held at `b` + 0.125 V, and `vin1` swept from `b` to `b` + 0.5 V in 10 mV steps.
  `vout` is `b - (vin1 - b) - (vin2 - b)`, within 20% of `(vin1 - b) + (vin2 - b)`.
- **Subtractor, 21.**
  `vin2` is held at its level, and `vin1` swept from 2.25 V below it to 1.75 V below it in 50 mV steps.
  `vout` is `vin2 - vin1`, within 20%.
- **Schmitt trigger, 22.**
  `vin` sweeps from 0 to 5 V and from 5 to 0 V in 100 mV steps, each operating point continued from the last.
  Either way, `vout` rises by at least 2.5 V from `vin` at 0 V to `vin` at 5 V and never falls by more than 1 µV as `vin` rises, and the levels of `vin` at which it first exceeds 2.5 V, counting from 0 V, differ between the two sweeps by more than 50 mV.
- **Voltage-controlled oscillator, 23.**
  The bench switches `vdd` on as for 16 and holds `vin` at 3.25, 3.5 and 4 V in turn, each for 100 µs.
  Over the second half of each, `vout` oscillates steadily as for 16, and its mean period changes by more than 1% from each level to the next, all in the same direction.
- **Phase-locked loop, 24.**
  `clk_ref` is a clock from 0 to 5 V at 10 MHz, then at 12 MHz: its edges take 2% of its period and its pulse 48%, from 30 ns.
  The bench switches `vdd` on as for 16 and runs 10 µs; over the last 2 µs, the mean period between `clk_p`'s rises through 2.5 V, each from 2.25 V to 2.75 V, is the clock's, within 5%.

Every circuit must pass its check.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
Work as long as you need, and verify what you claim.
