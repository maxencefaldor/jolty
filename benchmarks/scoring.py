# Copyright 2026 Maxence Faldor
"""What the tasks' scorers share: scripts simulated, parts counted and parts checked."""

import asyncio
import tempfile
from collections.abc import Iterable
from pathlib import Path

from jolty import devices, files, ir, service
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


def disallowed(
    built: ir.Circuit, given: Iterable[devices.Device], allowed: Iterable[str]
) -> list[str]:
    """Why each device below the top of `built` is neither `given` nor `allowed`.

    A device of a given part's model must take that part's parameters; any other device
    must be of an allowed model, such as ``resistor``.
    """
    parts = [(part.model, part.parameters) for part in given]
    models = {model for model, _ in parts}
    permitted = set(allowed)
    found = []
    for name, made in built.components.items():
        if name == built.top:
            continue
        for instance in made.instances:
            if not isinstance(instance.of, ir.DeviceOf):
                continue
            kind = instance.of.device
            if kind in models:
                if (kind, instance.parameters) not in parts:
                    found.append(f"{instance.id} is not a given part")
            elif kind not in permitted:
                found.append(f"{instance.id} is a {kind}, which the task forbids")
    return found
