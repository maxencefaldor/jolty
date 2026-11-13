# A NOR gate of three memristors

Compute NOR inside memory, with no transistor: two memristors hold the inputs, a third holds the output, and one voltage pulse writes the output from the inputs.
A bit is a memristor's state: on, at its low resistance, is 1, and off is 0.

- The memristors are VTEAM's: 1 kΩ on and 100 kΩ off, exponential between; their state moves toward off above 0.5 V and toward on below -1 V, at 2·10⁸ full ranges a second at twice either threshold, growing as the third power of the excess.
- The output starts at 1.
    After one pulse of at most 10 ns, it must hold the NOR of the inputs, within 0.1 of a full range, for all four inputs, and neither input may move by more than 0.05.
- Resistors may be added; nothing else.
- The gate must work with the pulse's amplitude 10 % above and below its nominal, as a driver's varies.

Use jolty to find a gate that meets this, the widest range of amplitudes it works over, and what bounds that range.
