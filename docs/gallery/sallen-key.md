# Sallen-Key low-pass

A unity-gain Sallen-Key low-pass filter: the second-order Butterworth response.

Two 10 kΩ resistors, a 20 nF feedback capacitor, and a 10 nF shunt capacitor set ω₀ = 1/√(R₁R₂C₁C₂) and Q = √(R₁R₂C₁C₂)/((R₁ + R₂)C₂) = 1/√2.
The filter cuts off at 1.125 kHz with -3.01 dB at cutoff, no peaking, and -40 dB per decade roll-off.
A unity follower built from the shared 100 dB, 1 MHz op-amp drives the output.
Simulate with `jolty simulate examples/sallen_key.py`.

[Open in 3D](sallen-key.view.html){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/examples/sallen_key.py){ .md-button }

10 devices and 7 unknowns.

![The schematic](../images/gallery/sallen_key.schematic.png)

## Specs

| Requirement | jolty | ngspice | Difference |
| --- | --- | --- | --- |
| gain of out at 10 Hz in bode ≥ -0.001 dB and ≤ 0.001 dB | -8.59339e-05 dB, passed | -8.59339e-05 dB, passed | 1.1e-11 |
| gain of out at 1.125 kHz in bode ≥ -3.02 dB and ≤ -3 dB | -3.01192 dB, passed | -3.01192 dB, passed | 7.4e-16 |
| gain of out at 11.25 kHz in bode ≥ -40.05 dB and ≤ -39.95 dB | -40.0148 dB, passed | -40.0148 dB, passed | 3.6e-16 |
| overshoot of out in step ≥ 4.27 % and ≤ 4.37 % | 4.34556 %, passed | 4.3453 %, passed | 6.1e-05 |

## Replicated in ngspice

jolty writes the circuit as a SPICE netlist, ngspice runs every analysis of the plan, and each signal both report is compared, relative to its largest value.
A transient compares each of ngspice's points with jolty's waveform within one of jolty's steps.

| Analysis | Voltages | Largest deviation | Currents | Largest deviation |
| --- | --- | --- | --- | --- |
| `bode` | 5 | 9.2e-16 | 2 | 1.1e-15 |
| `step` | 5 | 2.5e-06 | 2 | 2.7e-06 |

![The replication in ngspice](../images/gallery/sallen_key.bode.replication.png)

![The replication in ngspice](../images/gallery/sallen_key.step.replication.png)

## Plots

![A plot](../images/gallery/sallen_key.bode.plot.png) ![A plot](../images/gallery/sallen_key.step.plot.png)
