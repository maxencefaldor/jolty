# A NOR gate of three memristors

Compute NOR inside memory without transistors: two memristors store inputs, a third stores the output, and a single voltage pulse writes the output from the inputs.
Logic `1` corresponds to the low-resistance on state, and logic `0` corresponds to the high-resistance off state.

- Each memristor uses the exponential VTEAM model (1 kΩ on, 100 kΩ off), switching toward off above 0.5 V and toward on below -1 V at `2e8` full ranges per second at twice threshold with cubic overdrive dependence.
- The output starts at `1`.
    After one pulse lasting at most 10 ns, the output must reach the NOR of the inputs within 0.1 of full range across all four input pairs, while neither input shifts by more than 0.05.
- Additional resistors are permitted; no other components may be added.
- The gate must tolerate ±10% variation in pulse amplitude.

Use jolty to find a compliant gate, maximize the operating pulse-amplitude window, and identify what bounds that window.
