# A two-terminal current source

Design a two-terminal current source: two terminals, `a` and `b`, carrying 1 mA from `a` to `b` whenever 3 V to 24 V lie across them.

- The current stays within 1 % of its value at 12 V over that whole range: its line regulation.
- At 12 V, it stays within 2 % of its value at 27 °C from -40 °C to 85 °C.
- It takes no supply but the two terminals, and as few parts as it can: 2N3904 and 2N3906 transistors, 1N4148 diodes and resistors, the parts in `examples/common.py`.

Use jolty to find the best design you can.
Report the design, its line regulation and its drift, and what you learned on the way.
