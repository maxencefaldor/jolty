# Cost-down: a transistor's bias

`bias.py` builds the bias of a common-emitter stage: the component `stage`, placed as `X1` on its bench, and the bench's 5 V supply `V1`.
The plan simulates 200 Monte Carlo operating-point samples over resistor tolerances, requiring the emitter within 100 mV of 1 V and the collector within 300 mV of 2.5 V at 99 % minimum yield.
The original stage uses four 0.1 % resistors.

`catalog.json` lists allowed resistors from 10 Ω to 1 MΩ with unit prices in US dollars: 0.1 % E192 parts, 1 % E96 parts, and 5 % E24 parts.
The prices are illustrative.

Write `design.py` with a `build()` function returning the same bench and a cheaper stage that meets the original specifications and yield.
Keep the bench, its supply and its nets as they are, and keep `X1`'s ports; change what you like inside the stage.
Each resistor of the stage must state a tolerance and take a value that a part of the catalog has at that tolerance.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
`jolty cost` prices a saved circuit with a catalog.
Work as long as you need, and verify what you claim.

Your design is scored by building it, simulating it with the plan and specs of the original `bias.py`, and pricing `X1` with `catalog.json`.
The design passes when every specification passes, the yield reaches the target, and the stage costs less than the original 0.2 USD.
