# A reference without an op-amp

Design a voltage reference near 1.2 V from a 5 V supply, with no op-amp.

- Its output drifts less than 50 ppm/K from -40 °C to 125 °C: a span under 10 mV.
- It moves less than 1 mV for each volt the supply moves between 4.5 V and 5.5 V.
- It starts by itself, and takes ten parts or fewer: the transistors, diodes and resistors of `examples/common.py`, and the 2N5457 JFET.

Use jolty to find the best design you can, and report what limits it.
