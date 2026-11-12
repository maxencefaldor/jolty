# Findings: a linear voltage-controlled oscillator

The prompt asks for 1 kHz per volt from 0.5 V to 5 V, every frequency within 1 % of the line through the origin that fits them best, with at most two op-amps.
The textbook integrator and Schmitt trigger missed the linearity by a factor of three; jolty traced the error to the Schmitt trigger's delay, and two parts that cancel it brought every frequency within 0.1 % of 996 Hz/V.

## Method

`candidates.py` writes each VCO as a component of `control`, `output` and `ground`: an integrator whose non-inverting input sits at half the control, fed through R from the control, and a switch that draws twice that current through R/2, reversed by an inverting Schmitt trigger at ±10 % of its ±13 V swing.
A 10 µA pulse at 1 µs starts the oscillation, as noise would.
`explore.py` simulates 16 periods at each control, 4000 steps a period, and reads the frequency over the last 12 from the triangle's rises, then fits the line through the origin.
It also reads the trigger's delay: the time from the triangle crossing the trigger's threshold to the trigger's output crossing zero.

## Results

| Design | Slope | Worst deviation from the line |
| --- | --- | --- |
| 2N3904 switch | 889.7 Hz/V | 3.08 % |
| MOSFET switch | 905.0 Hz/V | 8.05 % |
| MOSFET, 5.8 kΩ in series with the 1 nF capacitor | | stalls from 2 V |
| MOSFET, 100 pF across the trigger's feedback | 935.3 Hz/V | 5.35 % |
| MOSFET, 1 nF across the trigger's feedback | 963.1 Hz/V | 2.93 % |
| MOSFET, 100 pF across the feedback, 1 kΩ in series | | stalls from 4 V |
| MOSFET, 1 nF across the feedback, 2 kΩ in series with the capacitor | 994.1 Hz/V | 0.22 % |
| MOSFET, 2.2 nF across the feedback, 1.5 kΩ in series | 988.5 Hz/V | 0.71 % |
| **MOSFET, 2.2 nF across the feedback, 2 kΩ in series** | **996.1 Hz/V** | **0.096 %** |
| MOSFET, 2.2 nF across the feedback, 2.5 kΩ in series | 1003.7 Hz/V | 0.72 % |

The first six rows read six controls, 0.5 V to 5 V; the last four read nineteen, every 0.25 V.

| Trigger | Delay at 0.5 V | Delay at 5 V |
| --- | --- | --- |
| 10 kΩ feedback alone | 9.06 µs | 5.82 µs |
| 1 nF across it | 2.21 µs | 1.96 µs |
| 2.2 nF across it | 2.27 µs | 1.93 µs |

## What the experiments showed

1. **The error is a delay, not a nonlinearity.**
    Every half period, the triangle overshoots the threshold for as long as the trigger takes to switch, so the period is 4·Vth·C/I plus four delays.
    A fixed delay costs little at 0.5 V, where the period is 2 ms, and 12 % at 5 V, where it is 200 µs: the frequencies bend down from the line as they rise.
1. **The trigger's delay is its regeneration, not its slew.**
    The op-amp slews 13 V/µs, so its output crosses zero 1 µs after it starts; the rest is the time its positive feedback, 10 % of its output, takes to grow from the triangle's slow overdrive.
    The slower the triangle, the smaller the overdrive and the longer the wait: 9.1 µs at 0.5 V, 5.8 µs at 5 V.
1. **A transistor switch hides the delay behind its own error.**
    The 2N3904's saturation voltage, in series with R/2, slows the reversed current most at the lowest control, which bends the low end the other way.
    The two errors partly cancel, 3.1 %; a MOSFET's channel is a resistance, which only scales the current, and leaves the delay alone, 8.1 %.
1. **A resistor in series with the integrating capacitor advances each reversal.**
    The integrator's output steps by twice its current times the resistor, towards the next threshold: the voltage the triangle travels in R·C, a fixed time at every frequency, which cancels a fixed delay.
    5.8 kΩ, the delay at 5 V over 1 nF, should have cancelled it there.
1. **But it can stall the oscillator.**
    Through the MOSFET's channel and that resistor, the trigger's output moves its own inverting input, a negative feedback much stronger than its 10 % positive feedback when the MOSFET sits near its threshold.
    The circuit finds a steady state, its trigger at 0.76 V and its triangle at 76 mV, and holds it from 2 V up.
1. **A capacitor across the trigger's feedback resistor fixes both.**
    At the edges it feeds the whole output back, so the trigger regenerates three times faster, in about 2 µs, and its positive feedback outweighs the resistor's: with 1 nF or more, no control of the nineteen stalled.
    100 pF acts above the op-amp's bandwidth, and with 1 kΩ in series it still stalls at 4 V and 5 V; 2.2 nF holds the delay near 2 µs at every control.
1. **Matching the resistor to the remaining delay** cancels it: 2 kΩ times 1 nF is 2 µs, and every frequency lies within 0.1 % of the line, the step size's own resolution.
    1.5 kΩ leaves the low end high and 2.5 kΩ the high end, 0.7 % either way.

## Conclusion

`design.py` steps its control through 0.5, 1, 2, 3, 4 and 5 V in one transient, and each frequency lands within 0.09 % of 996.1 Hz/V, ten times inside the bound, as ngspice confirms, with two op-amps, a MOSFET, seven resistors and two capacitors.
What limits it now is the delay's remaining dependence on the triangle's slope, 2.27 µs at 0.5 V against 1.93 µs at 5 V, and the time step: with 800 steps a period, the frequency at 5 V reads 0.37 % above its value with 8000.
The slope itself, 0.4 % below 1 kHz/V, is set by R and C and the trigger's thresholds, and so would follow their tolerances in a built circuit.
