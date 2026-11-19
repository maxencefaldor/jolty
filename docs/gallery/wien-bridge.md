# Wien bridge oscillator

A Wien bridge oscillator, its amplitude held by two diodes.

The series and parallel 10 kΩ, 10 nF RC arms pass one third of the output with zero phase shift at f₀ = 1/(2πRC) = 1.592 kHz, sustaining oscillation at a non-inverting gain of 3.
Start-up gain is 1 + 22 kΩ/10 kΩ = 3.2; as amplitude grows, two 1N4148 diodes across part of the feedback resistor reduce gain toward 3 to stabilize the swing.
A 1 µA start-up pulse initiates oscillation.
Simulate with `jolty simulate examples/wien_bridge.py`.

[Open in 3D](wien-bridge.view.html){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/examples/wien_bridge.py){ .md-button }

15 devices and 11 unknowns.

![The schematic](../images/gallery/wien_bridge.schematic.png)

## Specs

| Requirement | jolty | ngspice | Difference |
| --- | --- | --- | --- |
| frequency of out from 30 ms in start ≥ 1.56 kHz and ≤ 1.623 kHz | 1.57384 kHz, passed | 1.57392 kHz, passed | 5.5e-05 |
| peak-to-peak of out from 30 ms in start ≥ 2 V and ≤ 20 V | 3.1809 V, passed | 3.18104 V, passed | 4.5e-05 |

## Replicated in ngspice

jolty writes the circuit as a SPICE netlist, ngspice runs every analysis of the plan, and each signal both report is compared, relative to its largest value.
A transient compares each of ngspice's points with jolty's waveform within one of jolty's steps.

A free-running oscillator's phase drifts with the least difference in two engines' numerics, so its waveforms compare through the specs' readings, its period and its swing, rather than point by point.

![The replication in ngspice](../images/gallery/wien_bridge.start.replication.png)

## Plots

![A plot](../images/gallery/wien_bridge.start.plot.png)
