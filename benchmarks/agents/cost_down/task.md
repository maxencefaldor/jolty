# Cost-down: a transistor's bias

`bias.py` builds the bias of a common-emitter stage: the component `stage`, placed as `X1` on its bench, and the bench's 5 V supply `V1`.
Its plan runs the operating point over 200 Monte Carlo samples of the resistors' tolerances, and its specs hold the emitter at 1 V within 100 mV and the collector at 2.5 V within 300 mV, with a yield of at least 99 %.
Its four resistors are 0.1 % parts.

`catalog.json` lists the resistors you may build it from: 0.1 % parts of the E192 series, 1 % parts of the E96 series and 5 % parts of the E24 series, from 10 Ω to 1 MΩ, at unit prices in US dollars.
The prices are illustrative.

Write `design.py`: a build script whose `build()` returns the same bench with a cheaper stage, which meets the same specs and yield.
Keep the bench, its supply and its nets as they are, and keep `X1`'s ports; change what you like inside the stage.
Each resistor of the stage must state a tolerance and take a value that a part of the catalog has at that tolerance.

You have jolty, a compiler and simulator for electronic circuits: its command line, `jolty`, its MCP server, `jolty-mcp`, and its Python package.
`jolty cost` prices a saved circuit with a catalog.
Work as long as you need, and verify what you claim.

Your design is scored by building it, simulating it with the plan and specs of the original `bias.py`, and pricing `X1` with `catalog.json`: it meets the task when every spec passes, the yield reaches its target, and it costs less than the original's 0.2 USD.
