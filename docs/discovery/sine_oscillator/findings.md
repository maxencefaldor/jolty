# Findings: a clean, steady sine oscillator

The prompt requires a 1 kHz (±1%) sine wave at 2 V peak (±5% across -40 °C to 85 °C) with under 0.5% distortion settling within 100 milliseconds.
A JFET-controlled Wien bridge met frequency, 27 °C amplitude, distortion (7× margin), and settling time; no single-op-amp design met the temperature drift bound, and jolty isolated the cause.

## Method

`candidates.py` constructs a Wien bridge (15.9 kΩ and 10 nF per arm) around the op-amp macromodel with three amplitude-control loops.
`explore.py` simulates 150 ms from startup at 1 µs steps, measuring frequency from zero crossings over the final 40 ms, amplitude from extrema, and THD across harmonics 2 through 10 on 40 Hann-windowed resampled periods.
Temperature drift is evaluated by repeating the run at -40 °C and 85 °C.

## Results

| Amplitude control | Frequency | Amplitude | Distortion | Settled | At -40 °C | At 85 °C |
| --- | --- | --- | --- | --- | --- | --- |
| Two diodes across 10 kΩ of feedback | 991.8 Hz | 1.59 V | 3.98 % | 10 ms | +51 % | -53 % |
| JFET, gate at half detector output, linearized | 995.5 Hz | 7.17 V | 0.18 % | 79 ms | -10 % | +10 % |
| JFET, gate at half detector output | 995.5 Hz | 1.13 V | 0.037 % | over 150 ms | | |
| JFET, detector on 45% of output, 0.1 µF | 995.5 Hz | 1.93 V | 0.064 % | 98 ms | +15 % | -11 % |
| JFET, detector on 45% of output, 0.47 µF | 995.5 Hz | 0.37 V, growing | 0.010 % | over 150 ms | | |
| **JFET, detector on 43% of output, 0.1 µF** | **995.5 Hz** | **2.02 V** | **0.068 %** | **99 ms** | **+16 %** | **-11 %** |

## What the experiments showed

1. **Diode clippers settle fast but distort heavily.**
    Hard-limiting diodes stabilize amplitude within 10 ms yet clip peaks to 4% THD, while diode knee drift swings amplitude by roughly 50%.
1. **A JFET in the gain leg acts as a low-distortion variable resistor**, achieving 0.04% THD under steady gate bias.
    Residual distortion comes from gate ripple as the peak detector recharges once per cycle, which a larger hold capacitor reduces.
1. **Hold capacitance trades distortion against settling time.**
    A 0.47 µF hold capacitor cuts THD to 0.010% but pushes settling past 150 ms, whereas 0.1 µF settles in 99 ms at 0.068% THD.
1. **Adding half-drain linearization shifts the operating point**, pushing amplitude to 7.2 V and increasing distortion instead of lowering it.
1. **Attenuating the output before the detector** sets amplitude cleanly without amplifying gate-voltage drift.
    Remaining temperature swing (+16% / -11%) stems from the detector diode's -2 mV/K forward-drop shift (`(|Vgs| + Vd)/k`), since JFET thermal drift nearly cancels.
1. **Oscillation frequency sits 0.45% below nominal** across all variants because the op-amp's 1 MHz bandwidth adds loop phase lag compensated below `f₀`.

## Conclusion

`design.py` meets frequency, 27 °C amplitude, distortion (7× margin), and settling bounds while failing temperature drift at both corners.
Holding ±5% across -40 °C to 85 °C requires removing the diode drop from the loop using a precision rectifier (second op-amp) or a diode-compensated gate reference.

## Limits

The agent that wrote the prompt also designed the circuit using textbook knowledge and judged the output, making this a capability showcase rather than a blind benchmark.
The JFET-stabilized Wien bridge is a classic topology.
Simulated distortion reflects nominal passive models without resistor/capacitor nonlinearity, and real 2N5457 pinch-off spread would shift settled amplitude.
