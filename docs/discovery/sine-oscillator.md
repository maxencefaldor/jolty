# A clean, steady sine oscillator

JFET-controlled 1 kHz Wien bridge sine oscillator.

The JFET channel sits in the gain leg while its gate tracks the negative peak of 43% of the output, pinching off to hold closed-loop gain at three.
Dividing the output before the detector scales the JFET threshold and diode drop together.
The oscillator settles within 100 ms to 2.02 V peak at 995.5 Hz with 0.068% distortion, drifting +16% at -40 °C and -11% at 85 °C due to detector diode drift.
Simulate via `jolty simulate docs/discovery/sine_oscillator/design.py`.

[Open in 3D](sine-oscillator.view.html#view=X1){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/docs/discovery/sine_oscillator/design.py){ .md-button } [Every experiment](https://github.com/maxencefaldor/jolty/tree/main/docs/discovery/sine_oscillator){ .md-button }

18 devices and 13 unknowns.

![The schematic](../images/gallery/sine_oscillator.inside.schematic.png)

## The prompt

> Design a 1 kHz sine oscillator around the op-amp from `examples/common.py`.
>
> - Oscillation frequency stays within 1% of 1 kHz.
> - Peak amplitude targets 2 V within 5% and holds within 5% from -40 °C to 85 °C.
> - Total harmonic distortion stays below 0.5%.
> - The oscillator self-starts and settles within 100 ms.
>
> Use as few parts as possible from `examples/common.py` (resistors, capacitors, and parts) plus a 2N5457 JFET.
> Use jolty to find the best design you can, and report what limits performance.

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

## Specs

| Requirement | jolty | ngspice | Difference |
| --- | --- | --- | --- |
| frequency of out from 110 ms in run ≥ 990 Hz and ≤ 1.01 kHz | 995.541 Hz at cold, passed | 995.541 Hz at cold, passed | 1.1e-08 |
| peak-to-peak of out from 110 ms in run ≥ 3.8 V and ≤ 4.2 V | 4.68238 V at cold, failed | 4.68203 V at cold, failed | 7.5e-05 |

## Replicated in ngspice

jolty writes the circuit as a SPICE netlist, ngspice runs every analysis of the plan, and each signal both report is compared, relative to its largest value.
A transient compares each of ngspice's points with jolty's waveform within one of jolty's steps.

A free-running oscillator's phase drifts with the least difference in two engines' numerics, so its waveforms compare through the specs' readings, its period and its swing, rather than point by point.

![The replication in ngspice](../images/gallery/sine_oscillator.run.replication.png)

## Plots

![A plot](../images/gallery/sine_oscillator.run.plot.png)
