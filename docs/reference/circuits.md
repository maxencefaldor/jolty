# Circuits

Scripts write circuits with the builder, from devices whose values carry units and from parts read from SPICE files.

::: jolty.builder
    options:
      members: [component, circuit, Builder, Component, Placed, Terminal, Net, ComponentFunction]

::: jolty.devices

::: jolty.parts

::: jolty.spice
    options:
      members: [load, Library, fetch, GROUND]

::: jolty.spice.sky130

::: jolty.spice.export
    options:
      members: [netlist, flatten, vector, element, Dialect, NGSPICE, XYCE]

::: jolty.spice.ngspice
    options:
      members: [run, runs, execute, version, read, Executed, NgspiceTimeoutError, Running, Compatibility, Raw, spice_celsius, RELATIVE_TOLERANCE, VOLTAGE_TOLERANCE, CURRENT_TOLERANCE, OPTIONS]

::: jolty.spice.xyce
    options:
      members: [run, execute, version, Result, Executed, xyce_celsius, OPTIONS]

::: jolty.waveforms

::: jolty.units
    options:
      members: [Quantity, Dimension, celsius, format_si, finite]

::: jolty.ir

::: jolty.errors
