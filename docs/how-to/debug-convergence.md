# Diagnose a DC or transient failure

Pinpoint structural mistakes, steer multistable DC circuits, inspect transient charge leak, and switch integration methods on stiff circuits.

## 1. Run structural and operating-point checks first

Call `simulation.check()` to catch floating nodes, missing DC paths, or voltage-source loops before simulating.
Call `simulation.check_operating_point()` to detect unstable poles, open switches, or gmin-dependent nodes.

```python
from jolty import Builder, Role, Simulation, component, devices
from jolty.units import V, nF


@component
def floating_cap(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), name="V1")
    first = c.place(devices.capacitor(1 * nF), name="C1")
    second = c.place(devices.capacitor(1 * nF), name="C2")
    c.connect(source.positive, first.a, name="in")
    c.connect(first.b, second.a, name="mid")
    c.connect(second.b, source.negative, ground)


diagnostics = Simulation(floating_cap()).check()
[diagnostic.rule for diagnostic in diagnostics]  # ['no-dc-path']
```

Run `jolty explain <rule>` on any rule name to read its exact condition.

## 2. Find or steer multiple DC solutions

Feedback circuits such as latches and bistable references admit multiple DC operating points.
Use `OperatingPoints` to discover solutions by deflation, or set `Options(nodesets=...)` to steer Newton toward a chosen state.

```python
from jolty import OperatingPoint, OperatingPoints, Options
from jolty.units import kohm


@component
def bistable(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    feedback = c.place(
        devices.behavioral_current_source("-2e-3 * tanh(v(x))"), name="B1"
    )
    c.connect(resistor.a, feedback.positive, feedback.x, name="x")
    c.connect(resistor.b, feedback.negative, ground)


simulation = Simulation(bistable())
found = simulation.run(OperatingPoints(max_solutions=4))
len(found.points)  # 3

high = simulation.run(OperatingPoint(), Options(nodesets={"x": 2 * V}))
low = simulation.run(OperatingPoint(), Options(nodesets={"x": -2 * V}))
round(high.voltage("x"), 3)  # 1.915
round(low.voltage("x"), 3)  # -1.915
```

## 3. Start transients from explicit initial conditions

Pass `initial_conditions` on `Transient` to clamp nets during initialization before releasing them at `t = 0`.
Set `operating_point=False` (SPICE `UIC`) to skip the DC solve altogether and start directly from those net voltages.

```python
from jolty import Transient
from jolty.units import us


@component
def rc_decay(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    resistor = c.place(devices.resistor(1 * kohm), name="R1")
    capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    c.connect(resistor.a, capacitor.a, name="out")
    c.connect(resistor.b, capacitor.b, ground)


decay = Simulation(rc_decay()).run(
    Transient(stop=5 * us, initial_conditions={"out": 3 * V}, operating_point=False)
)
round(float(decay.voltage("out")[0]), 6)  # 3.0
round(float(decay.voltage("out")[-1]), 3)  # 0.02
```

## 4. Inspect solver work, charge leak, and stiff methods

Every `TransientResult` reports accepted steps, rejected steps, Newton iterations, and integrated per-node `charge_leak`.
If a stiff circuit shows trapezoidal ringing, switch to `method="gear2"` or tighten `Options(relative_tolerance=...)`.

```python
from jolty.units import mV, pS, uA

strict = Simulation(rc_decay()).run(
    Transient(stop=5 * us, initial_conditions={"out": 3 * V}, method="gear2"),
    Options(
        relative_tolerance=1e-5,
        voltage_tolerance=1 * mV,
        current_tolerance=1 * uA,
        gmin=1 * pS,
    ),
)
accepted_steps = strict.statistics.accepted
leak = strict.statistics.charge_leak
max_leak = 0.0 if leak is None else float(leak.absolute.max())
```

When a transient fails with `StepTooSmall` or Newton divergence, the error names the exact time, the worst-converging unknowns, and the failing reason.
See [Simulation](../guide/simulation.md) for all solver options and continuation strategies.
