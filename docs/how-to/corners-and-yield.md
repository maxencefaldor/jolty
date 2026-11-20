# Run a corner sweep and yield check

Verify a circuit across supply and temperature corners, parameter sweeps, worst-case extremes, and Monte Carlo yield targets.

## 1. Attach tolerances to parts

Pass `tolerance` on passive constructors or call `.toleranced(...)` on any numeric parameter.
Assign a shared `ir.Lot` when matched parts on the same die track together.

```python
from jolty import Builder, Role, component, devices, ir
from jolty.units import K, Quantity, V, kohm

DIE = ir.Lot(name="die", relative=0.05)


@component
def divider(c: Builder, supply: Quantity = 5 * V) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(supply), name="V1")
    upper = c.place(
        devices.resistor(1 * kohm, temperature_coefficient=1e-4 / K).toleranced(
            "resistance", 0.01, lot=DIE
        ),
        name="R1",
    )
    lower = c.place(
        devices.resistor(1 * kohm, temperature_coefficient=1e-4 / K).toleranced(
            "resistance", 0.01, lot=DIE
        ),
        name="R2",
    )
    c.connect(source.positive, upper.a, name="in")
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)
```

## 2. Define corners, sweeps, and statistical runs in a plan

Put corners, parameter grids, Monte Carlo sampling, and worst-case verification in `plan()`.
Each corner may override `temperature`, `process`, or keyword arguments passed to `build()`.

```python
from jolty import Corner, MonteCarlo, OperatingPoint, Plan, WorstCase
from jolty.plan import Axis, Grid
from jolty.units import celsius


def plan() -> Plan:
    return Plan(
        {"op": OperatingPoint()},
        corners={
            "cold_low": Corner(temperature=celsius(-40), supply=4.75 * V),
            "hot_high": Corner(temperature=celsius(85), supply=5.25 * V),
        },
        variants=Grid(
            [
                Axis.temperature([celsius(0), celsius(27), celsius(70)]),
                Axis("R2", "resistance", [0.9 * kohm, 1.0 * kohm, 1.1 * kohm]),
            ]
        ),
        monte_carlo=MonteCarlo(samples=200, seed=0),
        worst_case=WorstCase(),
    )
```

## 3. Add specs and a yield target

Define `specs()` with bounds on analysis measures and an optional `yield_target`.
Both `jolty simulate` and `jolty check` evaluate every corner and sample against these bounds.

```python
from jolty.specs import Specs, value


def specs() -> Specs:
    return Specs(
        [value("op", "out").within(2.5 * V, 0.2 * V)],
        yield_target=0.95,
    )
```

## 4. Run a parallel batch directly from Python

Use `Simulation.batch` to evaluate any variant generator in Python and inspect columnar NumPy results.
Passing `measures` computes specs in Rust without storing full waveforms for every sample.

```python
from jolty import Simulation

simulation = Simulation(divider())
batch = simulation.batch(
    {"op": OperatingPoint()},
    MonteCarlo(samples=100, seed=1),
    measures=[value("op", "out")],
)
outputs = batch.measured[:, 0]
resistances = batch.parameter("R1 resistance")
mean_output = float(outputs.mean())
failed_runs = int(batch.failed.sum())
```

## 5. Scale large Monte Carlo runs across machines

For large sample counts, split runs into deterministic shards and merge them afterward.
See [Across machines](../guide/machines.md) for shard commands and digest verification.
