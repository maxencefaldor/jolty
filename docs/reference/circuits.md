# Circuits

Scripts write circuits with the builder, from devices whose values carry units and from parts read from SPICE files.

::: jolty.builder
    options:
      members: [component, circuit, Builder, Component, Placed, Terminal, Net, ComponentFunction]

::: jolty.devices

::: jolty.spice
    options:
      members: [load, Library, fetch, GROUND]

::: jolty.spice.export
    options:
      members: [netlist, flatten, ngspice, vector, element, Raw, spice_celsius]

::: jolty.waveforms

::: jolty.units
    options:
      members: [Quantity, Dimension, celsius, format_si, finite]

::: jolty.ir

::: jolty.errors
