# Findings: a clean, steady sine oscillator

The prompt asks for 1 kHz within 1 %, 2 V peak within 5 % over -40 °C to 85 °C, less than 0.5 % distortion, and settling within 100 ms. A JFET-controlled Wien bridge met the frequency, the amplitude at 27 °C, the distortion by a factor of seven and the settling time; no design of the parts allowed met the drift bound, and jolty located why.

## Method

`candidates.py` builds a Wien bridge, 15.9 kΩ and 10 nF in each arm, around the op-amp macromodel, with three amplitude controls.
`explore.py` simulates 150 ms from power-on with 1 µs steps, then reads the frequency from the zero crossings of the last 40 ms, the amplitude from their extremes, and the distortion from the spectrum of the last 40 periods resampled evenly under a Hann window: the root sum of the 2nd to the 10th harmonics over the fundamental.
It runs again at -40 °C and 85 °C for the drift.

## Results

| Amplitude control | Frequency | Amplitude | Distortion | Settled | At -40 °C | At 85 °C |
| --- | --- | --- | --- | --- | --- | --- |
| Two diodes across 10 kΩ of the feedback | 991.8 Hz | 1.59 V | 3.98 % | 10 ms | +51 % | -53 % |
| JFET, gate at half the detector's output, linearized | 995.5 Hz | 7.17 V | 0.18 % | 79 ms | -10 % | +10 % |
| JFET, gate at half the detector's output | 995.5 Hz | 1.13 V | 0.037 % | over 150 ms | | |
| JFET, detector on 45 % of the output, 0.1 µF | 995.5 Hz | 1.93 V | 0.064 % | 98 ms | +15 % | -11 % |
| JFET, detector on 45 % of the output, 0.47 µF | 995.5 Hz | 0.37 V, growing | 0.010 % | over 150 ms | | |
| **JFET, detector on 43 % of the output, 0.1 µF** | **995.5 Hz** | **2.02 V** | **0.068 %** | **99 ms** | **+16 %** | **-11 %** |

## What the experiments showed

1. **Diodes limit hard.**
    They hold the amplitude within a cycle or two, settling in 10 ms, but they clip each peak, 4 % distortion, and their knee moves with temperature, the amplitude by half.
1. **A JFET in the gain leg is a resistor the loop can turn**, and a resistor does not distort: 0.04 % when its gate is steady.
    The distortion that remains is the gate's ripple, the detector recharging once a period, which a larger hold capacitor smooths.
1. **The hold capacitor trades distortion for settling.**
    At 0.47 µF the distortion falls to 0.010 %, but the loop's time constant, the capacitor over its megohm, leaves the amplitude still growing after 150 ms; at 0.1 µF it settles in 99 ms with 0.068 %.
1. **Feeding half the drain's voltage to the gate**, which linearizes a JFET used as a resistor, here shifts its operating point so far that the amplitude settles at 7.2 V, and adds distortion rather than removing it.
1. **Dividing the output before the detector** lets the divider set the amplitude without amplifying the gate's drift, as dividing after it does.
    What still drifts is the detector diode's drop, -2 mV/K: the amplitude follows (|Vgs| + Vd)/k, and the JFET's own drift nearly cancels, so the diode's 0.13 V over the range sets the 27 % swing.
1. **The frequency sits 0.45 % low** in every design: the op-amp's 1 MHz bandwidth adds phase in the loop, which the bridge corrects by moving off f₀.

## Conclusion

`design.py` meets the frequency, the 27 °C amplitude, the distortion with a factor of seven to spare and the settling time, and fails the drift bound at both corners, as its specs report.
Holding 5 % over the range needs the detector's drop out of the loop: a precision rectifier around a second op-amp, or a diode-compensated reference for the gate.
