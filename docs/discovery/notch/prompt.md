# A notch that survives its tolerances

Design a filter that removes 50 Hz hum: its notch at least 50 dB deep somewhere between 45 Hz and 55 Hz, and its gain within 3 dB of unity at 25 Hz and 100 Hz.

- Use 1 % resistors and 5 % capacitors, and the op-amps of `examples/common.py`.
- At least 95 % of the circuits built from those parts must meet the specification.
- Use as few parts as the yield allows.

Use jolty to find the best design you can, its yield over Monte Carlo samples of its tolerances, and what sets that yield.
