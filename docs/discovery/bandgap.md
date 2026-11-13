# A reference without an op-amp

The best reference found: Widlar's three-transistor bandgap, fed by a JFET.

Q1, diode-connected, and Q2 over 155 Ω form a Widlar current source, so R2 holds Vt·ln(I1/I2), and Q3 holds the output at its own Vbe plus Q2's current across 2.4 kΩ: a falling voltage and a rising one, summed near the band gap.
Fed by a resistor, the cell's output moves 44 mV for each volt of supply; fed by a 2N5457 whose gate is on its source, a current source of its Idss, it moves 0.31 mV, and the JFET's own drift turns out to straighten the bandgap's curvature: 1.2414 V within 1.08 mV from -40 °C to 125 °C, 5.3 ppm/K, with seven parts.
Simulate it with `jolty simulate discovery/bandgap/design.py`.

[Open in 3D](bandgap.view.html#view=X1){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/discovery/bandgap/design.py){ .md-button } [Every experiment](https://github.com/maxencefaldor/jolty/tree/main/discovery/bandgap){ .md-button }

9 devices and 14 unknowns.

![The schematic](../images/gallery/bandgap.inside.schematic.png)

## The prompt

> Design a voltage reference near 1.2 V from a 5 V supply, with no op-amp.
>
> - Its output drifts less than 50 ppm/K from -40 °C to 125 °C: a span under 10 mV.
> - It moves less than 1 mV for each volt the supply moves between 4.5 V and 5.5 V.
> - It starts by itself, and takes ten parts or fewer: the transistors, diodes and resistors of `examples/common.py`, and the 2N5457 JFET.
>
> Use jolty to find the best design you can, and report what limits it.

The prompt asks for a reference near 1.2 V from 5 V, drifting less than 50 ppm/K from -40 °C to 125 °C and moving less than 1 mV per volt of supply, with ten parts or fewer and no op-amp.
Widlar's three-transistor bandgap fed by a JFET met both bounds with seven parts and a wide margin: 5.3 ppm/K and 0.31 mV/V.

## Method

`candidates.py` writes each reference as a component of `vcc`, `out` and `ground`.
`explore.py` reads the output at eight temperatures and over a DC sweep of the supply from 4.5 V to 5.5 V, under a 100 kΩ load.
To flatten a bandgap, it bisects on its Widlar resistor R2 until the output at 125 °C equals the output at -40 °C, for each ratio of R3 to R1.

## Results

| Reference | Parts | Output | Span, -40 °C to 125 °C | Drift | Line |
| --- | --- | --- | --- | --- | --- |
| A diode fed by 4.3 kΩ | 2 | 0.586 V | 329 mV | 3400 ppm/K | 10.4 mV/V |
| Widlar bandgap, R3 = 6 kΩ, R2 = 500 Ω, fed by 1 kΩ | 7 | 1.510 V | 10.8 mV | 43.5 ppm/K | 57.7 mV/V |
| The same, R3 = 3.6 kΩ, R2 flattened to 255 Ω | 7 | 1.422 V | 6.0 mV | 25.7 ppm/K | 44.4 mV/V |
| The same, fed by a JFET instead, R2 kept | 7 | 1.341 V | 45.7 mV | 207 ppm/K | 0.36 mV/V |
| Fed by a JFET, R3 = 2.4 kΩ, R2 flattened to 155.0 Ω | 7 | 1.2417 V | 1.12 mV | 5.5 ppm/K | 0.31 mV/V |
| **The same, R2 = 155.1 Ω** | **7** | **1.2414 V** | **1.08 mV** | **5.3 ppm/K** | **0.31 mV/V** |
| The same, R3 = 3.6 kΩ, R2 = 298 Ω | 7 | 1.261 V | 2.2 mV | 10.4 ppm/K | 0.32 mV/V |
| The same, R3 = 6 kΩ, R2 = 618 Ω | 7 | 1.299 V | 3.9 mV | 18.2 ppm/K | 0.38 mV/V |

Flattening sets the ends equal, which leaves the span 0.04 mV wider than at 155.1 Ω, the design's.
A 1 % error in R2 or R1 leaves it within 14.2 ppm/K.

## What the experiments showed

1. **A resistor-fed Widlar bandgap cannot regulate its line.**
    Its output impedance is Q3's 1/gm times the loop's PTAT gain, and Q3's current is the feed's: whatever the feed resistor, the output moves by about that gain times Vt over the headroom, 44 to 58 mV per volt here.
1. **A JFET's Idss is a current source in one part.**
    Its gate on its source, it takes 2 mA whatever its drain, within its 2.3e-3/V of channel-length modulation, and the line regulation falls 140-fold, to 0.31 mV/V.
1. **The feed's drift is part of the bandgap.**
    Q3's Vbe follows its current, so changing the feed from a resistor to a JFET moved the flat point: the R2 that flattened one leaves the other at 207 ppm/K.
    Flattened again with the JFET in place, the span fell to 1 mV: the JFET's Idss drifts so as to bend Q3's Vbe against the bandgap's curvature, which a resistor-fed cell leaves as a 6 mV bowl.
1. **The flat output depends on the PTAT ratio.**
    The smaller R3, the lower the voltage at which the drift flattens, from 1.30 V to 1.24 V, and the smaller the remaining curvature.

## Conclusion

`design.py` meets the prompt with seven parts: 1.2414 V at 27 °C, 5.3 ppm/K and 0.31 mV/V, as its specs confirm at every corner.
The curvature cancellation rests on the JFET's and the 2N3904's temperature models; a real part's spread of Idss would move the flat point, which a trimmed R2 would restore.

## Limits

The agent that wrote this prompt also designed the circuit, knowing the textbook answers, and judged the result itself: this is a showcase of what jolty lets an agent see, not a blind result.
Widlar's three-transistor cell is from 1971, and a JFET's Idss as a feed is common.
The 5.3 ppm/K comes from trimming R2 to 0.1 Ω on nominal models: the spread of Vbe and Idss between parts, and the transistors' differing temperatures, would put an untrimmed circuit well above it, and the curvature cancellation rests on two models' temperature terms.

## Specs

| Requirement | jolty | ngspice | Difference |
| --- | --- | --- | --- |
| out in op ≥ 1.237 V and ≤ 1.246 V | 1.24252 V at cold, passed | 1.24252 V at cold, passed | 1.7e-07 |
| out at 4.5 V in line ≥ 1.236 V and ≤ 1.247 V | 1.24238 V at cold, passed | 1.24238 V at cold, passed | 2.0e-08 |
| out at 5.5 V in line ≥ 1.236 V and ≤ 1.247 V | 1.24265 V at cold, passed | 1.24265 V at cold, passed | 9.3e-09 |

## Replicated in ngspice

jolty writes the circuit as a SPICE netlist, ngspice runs every analysis of the plan, and each signal both report is compared, relative to its largest value.
A transient compares each of ngspice's points with jolty's waveform within one of jolty's steps.

| Analysis | Voltages | Largest deviation | Currents | Largest deviation |
| --- | --- | --- | --- | --- |
| `op` | 5 | 3.7e-06 | 1 | 9.7e-09 |
| `line` | 5 | 6.2e-07 | 1 | 7.3e-10 |

![The replication in ngspice](../images/gallery/bandgap.line.replication.png)

## Plots

![A plot](../images/gallery/bandgap.line.plot.png)
