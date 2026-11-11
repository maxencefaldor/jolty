# Why a compiler

The page explains why jolty treats a circuit as source code to compile, rather than a drawing to edit.

## Agents write code

Language models write code fluently, and they reason well about what they wrote when a tool tells them, precisely, what is wrong with it.
A circuit written as Python has everything code has: functions, parameters, loops, imports and version control.
A component is a function, so a filter of any order or an amplifier with any gain is one call, and a design grows into a hierarchy of reusable boxes.

## Topology is the source

A script declares components, ports and connections, with no coordinates.
Everything else is derived from it: the circuit file, the simulation, the report and, later, the schematic and the plots.
An agent never moves a wire on a canvas; it changes a connection and builds again.

## Feedback is facts

A compiler states what is wrong and where, and leaves the fix to the programmer.
jolty does the same: each problem names the instance by its path and the script line that placed it, and each report states what the analyses found, never what to change.
Facts let the agent reason about the circuit itself, and they stay true whatever the agent was trying to do.

## The loop

The agent writes a script, runs `jolty simulate`, reads the report, and edits the script, until the circuit does what it should.
Each build is fast, since the engine is native code, so the loop takes seconds.
