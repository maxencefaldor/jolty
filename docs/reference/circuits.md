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
      members: [netlist, flatten, vector, element]

::: jolty.spice.ngspice
    options:
      members: [run, runs, Running, Compatibility, Raw, spice_celsius, VOLTAGE_TOLERANCE, CURRENT_TOLERANCE, OPTIONS]

::: jolty.waveforms

::: jolty.units
    options:
      members: [Quantity, Dimension, celsius, format_si, finite]

::: jolty.ir

::: jolty.errors
