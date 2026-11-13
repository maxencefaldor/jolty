# 555 timer, astable

The 555 timer, built from its block diagram, running astable.

Inside: three 5 kΩ resistors divide the supply into thirds, two comparators watch the threshold against two thirds and the trigger against one, an SR latch of NOR gates remembers which crossed last, and a 2N3904 discharges the timing capacitor while the output is low.
Outside: 1 kΩ and 10 kΩ charge 10 nF from 9 V to two thirds of it, and the 10 kΩ alone discharges it to one third.
The output is high for 0.693(Ra + Rb)C = 76.2 µs and low for 0.693·Rb·C = 69.3 µs: a period of 145.5 µs, 6.87 kHz.
Simulate it with `jolty simulate examples/timer_555.py`.

[Open in 3D](timer-555.view.html){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/examples/timer_555.py){ .md-button }

16 devices and 17 unknowns.

![The schematic](../images/gallery/timer_555.schematic.png)

## Specs

| Requirement | jolty |
| --- | --- |
| frequency of out from 500 µs in run ≥ 6.664 kHz and ≤ 7.076 kHz | 6.7315 kHz, passed |
| maximum of timing from 500 µs in run ≥ 5.9 V and ≤ 6.1 V | 6.00843 V, passed |

## Checked in closed form

The example holds logic devices, which SPICE has no element for, so its specs check it against its closed-form behavior: truth tables, codes and periods.

## Plots

![A plot](../images/gallery/timer_555.run.plot.png)
