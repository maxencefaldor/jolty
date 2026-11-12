# Astable multivibrator

The astable multivibrator: two cross-coupled 2N3904s flipping each other in turn.

Each transistor's collector couples to the other's base through a capacitor.
When one turns on, its collector falls by nearly the supply and drives the other's base that far below zero; that base recovers through its 47 kΩ toward the supply, turning its transistor on at half way.
Each half period is RC·ln 2 = 0.326 ms with 10 nF: a period of 0.652 ms, 1.53 kHz.
Slightly unequal capacitors break the symmetry that would hold the circuit balanced.
Simulate it with `jolty simulate examples/astable_multivibrator.py`.

[Open in 3D](astable-multivibrator.view.html){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/examples/astable_multivibrator.py){ .md-button }

9 devices, 10 unknowns, simulated in 2.25 s.

![The schematic](../images/gallery/astable_multivibrator.schematic.png)

## Specs

| Requirement | jolty | ngspice | Difference |
| --- | --- | --- | --- |
| frequency of left from 2 ms in run ≥ 1.381 kHz and ≤ 1.688 kHz | 1.48984 kHz, passed | 1.48995 kHz, passed | 7.2e-05 |
| maximum of left from 2 ms in run ≥ 8.5 V and ≤ 9.1 V | 9.00002 V, passed | 9.00002 V, passed | 4.1e-08 |
| minimum of left from 2 ms in run ≤ 300 mV | 50.4814 mV, passed | 50.5049 mV, passed | 4.7e-04 |

## Replicated in ngspice

jolty writes the circuit as a SPICE netlist, ngspice runs every analysis of the plan, and each signal both report is compared, relative to its largest value.
A transient compares each of ngspice's points with jolty's waveform within one of jolty's steps.

A free-running oscillator's phase drifts with the least difference in two engines' numerics, so its waveforms compare through the specs' readings, its period and its swing, rather than point by point.

![The replication in ngspice](../images/gallery/astable_multivibrator.run.replication.png)

## Plots

![A plot](../images/gallery/astable_multivibrator.run.plot.png)
