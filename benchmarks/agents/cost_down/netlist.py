# Copyright 2026 Maxence Faldor
"""Adapter wrapping an ngspice cost-down netlist onto the ``bias.py`` test bench.

The scorer copies this module as ``design.py`` alongside ``bias.py``, ``design.cir``,
and ``tolerances.json``, then simulates the adapted design with the standard pipeline.
Instance ``X1`` on the ``bias.py`` bench is replaced by the ``stage`` subcircuit parsed
from ``design.cir``. Each resistor receives a uniform relative tolerance from
``tolerances.json`` matched case-insensitively to follow SPICE naming rules.
"""

import json
import runpy
from pathlib import Path
from typing import Final

from jolty import Component, ir, spice

HERE: Final = Path(__file__)
PORTS: Final = ("supply", "output", "ground")
"""Required port names of the ``stage`` subcircuit expected by the test bench."""


def toleranced(stage: Component, tolerances: object) -> Component:
    """Applies uniform resistor tolerances from `tolerances` to resistors in `stage`.

    Raises:
        TypeError: When `tolerances` is not a mapping of resistor names to numbers.
        ValueError: When `tolerances` names a resistor absent from `stage`.
    """
    if not isinstance(tolerances, dict):
        message = "tolerances.json holds no tolerances by resistor"
        raise TypeError(message)
    given: dict[str, ir.Tolerance] = {}
    for name, relative in tolerances.items():
        if isinstance(relative, bool) or not isinstance(relative, int | float):
            message = f"tolerances.json gives {name} {relative!r}, which is no number"
            raise TypeError(message)
        given[str(name).casefold()] = ir.Tolerance(relative=relative)
    resistors = {
        instance.name.casefold()
        for instance in stage.definition.instances
        if instance.of == ir.DeviceOf(model="resistor")
    }
    stray = sorted(set(given) - resistors)
    if stray:
        message = f"tolerances.json names {stray[0]}, which is no resistor of the stage"
        raise ValueError(message)
    instances = tuple(
        instance.model_copy(
            update={"tolerances": {"resistance": given[instance.name.casefold()]}}
        )
        if instance.name.casefold() in given
        else instance
        for instance in stage.definition.instances
    )
    definition = stage.definition.model_copy(update={"instances": instances})
    return Component(stage.name, definition, stage.children)


def build() -> Component:
    """Builds the ``bias.py`` bench with ``X1`` replaced from ``design.cir``.

    Raises:
        ValueError: When the parsed ``stage`` ports do not match `PORTS`.
    """
    stage = spice.load(HERE.with_name("design.cir")).component("stage")
    if all(port.name != spice.GROUND for port in stage.definition.ports):
        # Treat SPICE node 0 as ground even when no element inside stage connects to it.
        port = ir.Port(name=spice.GROUND, role=ir.Role.GROUND)
        ports = (*stage.definition.ports, port)
        stage = Component(
            stage.name,
            stage.definition.model_copy(update={"ports": ports}),
            stage.children,
        )
    ports = tuple(port.name for port in stage.definition.ports)
    if sorted(ports) != sorted(PORTS):
        message = f"the stage's ports are {', '.join(ports)}, not {', '.join(PORTS)}"
        raise ValueError(message)
    tolerances = HERE.with_name("tolerances.json")
    if tolerances.is_file():
        stage = toleranced(stage, json.loads(tolerances.read_text(encoding="utf-8")))
    bench = runpy.run_path(str(HERE.with_name("bias.py")))["build"]()
    placed = next(
        instance.of.component
        for instance in bench.definition.instances
        if instance.name == "X1" and isinstance(instance.of, ir.ComponentOf)
    )
    instances = tuple(
        instance.model_copy(update={"of": ir.ComponentOf(component=stage.name)})
        if instance.name == "X1"
        else instance
        for instance in bench.definition.instances
    )
    definition = bench.definition.model_copy(update={"instances": instances})
    children = tuple(
        stage if child.name == placed else child for child in bench.children
    )
    return Component(bench.name, definition, children)
