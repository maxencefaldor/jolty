# Copyright 2026 Maxence Faldor
"""The ngspice arm's cost-down design as a script: bias.py's bench with its stage read.

The scorer copies this script, as ``design.py``, beside a copy of the given ``bias.py``
and of the agent's ``design.cir`` and ``tolerances.json``, and simulates it as it
simulates the jolty arm's ``design.py``. Its bench is bias.py's, with ``X1`` the
subcircuit ``stage`` that jolty's SPICE reader reads of ``design.cir``. Each resistor of
the stage strays, uniformly, within the tolerance ``tolerances.json`` gives it by its
name, matched without case, as SPICE matches names.
"""

import json
import runpy
from pathlib import Path
from typing import Final

from jolty import Component, ir, spice

HERE: Final = Path(__file__)
PORTS: Final = ("supply", "output", "ground")
"""The stage's ports, which the bench joins."""


def toleranced(stage: Component, tolerances: object) -> Component:
    """`stage`, each resistor that `tolerances` names given the tolerance it gives.

    Raises:
        TypeError: When `tolerances` is no object of numbers.
        ValueError: When `tolerances` names no resistor of the stage.
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
        instance.id.casefold()
        for instance in stage.definition.instances
        if instance.of == ir.DeviceOf(device="resistor")
    }
    stray = sorted(set(given) - resistors)
    if stray:
        message = f"tolerances.json names {stray[0]}, which is no resistor of the stage"
        raise ValueError(message)
    instances = tuple(
        instance.model_copy(
            update={"tolerances": {"resistance": given[instance.id.casefold()]}}
        )
        if instance.id.casefold() in given
        else instance
        for instance in stage.definition.instances
    )
    definition = stage.definition.model_copy(update={"instances": instances})
    return Component(stage.name, definition, stage.children)


def build() -> Component:
    """bias.py's bench, its stage ``X1`` the one ``design.cir`` defines.

    Raises:
        ValueError: When the stage's ports are not those the bench joins.
    """
    stage = spice.load(HERE.with_name("design.cir")).component("stage")
    if all(port.name != spice.GROUND for port in stage.definition.ports):
        # The stage's ground is node 0, whether or not anything joins it.
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
        if instance.id == "X1" and isinstance(instance.of, ir.ComponentOf)
    )
    instances = tuple(
        instance.model_copy(update={"of": ir.ComponentOf(component=stage.name)})
        if instance.id == "X1"
        else instance
        for instance in bench.definition.instances
    )
    definition = bench.definition.model_copy(update={"instances": instances})
    children = tuple(
        stage if child.name == placed else child for child in bench.children
    )
    return Component(bench.name, definition, children)
