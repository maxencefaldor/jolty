# Components

A component is a box with ports, built from devices and other components.
A script writes each component as a Python function, and jolty turns the function into a factory of components.

## A component function

A component function takes a `Builder` first, then the component's parameters, and is decorated with `@component`:

```python
from jolty import Builder, Role, component, devices
from jolty.units import Quantity


@component
def divider(c: Builder, upper: Quantity, lower: Quantity) -> None:
    """Two resistors in series, tapped between them."""
    top = c.port("top")
    tap = c.port("tap")
    ground = c.port("ground", Role.GROUND)
    first = c.place(devices.resistor(upper), id="R1")
    second = c.place(devices.resistor(lower), id="R2")
    c.connect(top, first.a)
    c.connect(first.b, second.a, tap)
    c.connect(second.b, ground)
```

The builder offers three methods:

- `c.port(name, role)` declares a port: the net of that name inside the component, seen from outside.
  Its role is `Role.SIGNAL` by default, `Role.INPUT` or `Role.OUTPUT` for a signal in one direction, `Role.SUPPLY` for a supply rail, or `Role.GROUND` for the ground.
  A schematic draws inputs on a box's left and outputs on its right, and a component of inputs, outputs, one supply and one ground can be [characterized](fidelity.md) as a logic cell.
- `c.place(item, id=...)` places a device or another component as an instance, and returns it.
  The instance's terminals are its attributes, such as `first.a`, or items, such as `first["a"]`.
- `c.connect(*points, name=None)` joins terminals, ports and nets into one net.
  Given a name, the net takes it, and later connections join it as `Net(name)`; giving the same name to a second connection declares it twice, which is an error.

`c.port` returns the port's net, and `c.net(name)` declares a named net before anything joins it; each name is declared once, by one of these or by `connect`'s `name`.

Connections accumulate: two `connect` calls that share a point make one net.
Joining two nets that both have names, such as two ports, is an error, since it would short them.
A net nobody names takes the name of its first terminal, such as `R1.b`.

## Factories and names

Calling a component function with its parameters returns a `Component`.
Each distinct set of arguments builds one component, named after the function and its arguments, such as `divider[upper=10kΩ,lower=10kΩ]`.
Calling it again with the same arguments returns the same component, so a circuit holds each component once, however often it is placed.

## Hierarchy

A component places other components as it places devices, and their ports are its terminals:

```python
@component
def bench(c: Builder) -> None:
    """The divider, fed 5 V."""
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    stage = c.place(divider(10 * kohm, 10 * kohm), id="X1")
    c.connect(source.positive, stage.top)
    c.connect(stage.tap, name="out")
    c.connect(source.negative, stage.ground, ground)
```

The engine flattens the hierarchy and names everything by path: the resistor `R1` inside `X1` is `X1/R1`, and a net `mid` inside `X1` is `X1/mid`.
A port has no node of its own: it is the net it connects to outside.

## The top and the ground

`jolty.circuit(top)` turns a top component into a circuit: the top and every component it places, by name.
A script's `build()` returns the top component, and jolty calls `circuit` on it.
The top component has exactly one ground port, the reference node of every voltage.

## Source locations

Every instance records the line of the script that placed it.
Problems name the instance by its path and that line, as a compiler names the line of an error, so the agent knows what to edit:

```text
error: capacitance must be at least 0, not -1e-9
  = X1/C1, filter.py:12 in low_pass()
```
