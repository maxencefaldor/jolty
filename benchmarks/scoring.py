# Copyright 2026 Maxence Faldor
"""What the tasks' scorers share: scripts simulated, and parts counted."""

import asyncio
import tempfile
from pathlib import Path

from jolty import files, ir, service
from jolty.settings import Settings


def simulate(
    script: Path, plan: Path | None = None
) -> tuple[service.SimulateResult, ir.Circuit]:
    """What ``jolty simulate`` finds of `script`, and the circuit it built.

    The script's own plan and specs apply, or `plan`'s when given.
    """
    with tempfile.TemporaryDirectory() as directory:
        saved = Path(directory) / "circuit.jolty.json"
        request = service.SimulateRequest(
            script=script,
            plan=plan,
            output=saved,
            results=Path(directory) / "circuit.results.npz",
            schematic=False,
            plot=False,
        )
        result = asyncio.run(service.simulate(request, Settings()))
        return result, files.load_circuit(saved)


def count_devices(circuit: ir.Circuit, path: str) -> int:
    """How many devices the component instance at `path` holds, at any depth."""
    component = circuit.components[circuit.top]
    for id in path.split("/"):
        instance = component.instance(id)
        if instance is None or not isinstance(instance.of, ir.ComponentOf):
            message = f"no component instance at {path}"
            raise KeyError(message)
        component = circuit.components[instance.of.component]
    return _count(circuit, component)


def _count(circuit: ir.Circuit, component: ir.Component) -> int:
    total = 0
    for instance in component.instances:
        if isinstance(instance.of, ir.ComponentOf):
            total += _count(circuit, circuit.components[instance.of.component])
        else:
            total += 1
    return total
