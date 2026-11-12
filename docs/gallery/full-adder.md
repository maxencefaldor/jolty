# Full adder of NAND gates

A full adder of nine CMOS NAND gates, each gate four transistors.

The ladder from transistors to a full adder: the plan characterizes the NAND gate from its transistors into a logic cell, the adder runs on nine of those cells at event speed, and a spot check runs one gate again at transistor level to hold the cell to its contract.
Simulate it with `jolty simulate examples/full_adder.py`.

[Open in 3D](full-adder.view.html){ .md-button .md-button--primary } [Script](https://github.com/maxencefaldor/jolty/blob/main/examples/full_adder.py){ .md-button }

10 devices, 12 unknowns, simulated in 5.62 s.

![The schematic](../images/gallery/full_adder.schematic.png)

## Specs

The example states no specs.

## Checked in closed form

The example holds logic devices, which SPICE has no element for, so its specs check it against its closed-form behavior: truth tables, codes and periods.

## Plots

![A plot](../images/gallery/full_adder.count.plot.png)
