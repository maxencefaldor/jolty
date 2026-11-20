# Copyright 2026 Maxence Faldor
"""KiCad PCB netlist export for components built from catalog parts.

[`netlist`][jolty.kicad.netlist] exports a component as footprints and pad-level nets
compatible with the KiCad PCB editor using parts chosen by
[`jolty.catalog.bill`][jolty.catalog.bill]. Each catalog part defines a KiCad library
footprint and terminal-to-pad mapping; two-terminal resistors, capacitors, and
inductors default ``a`` to pad 1 and ``b`` to pad 2 when pins are omitted. Purely
thermal instances produce no footprint, and terminals on thermal nets map to no pad.

Export elaborates the target component standalone, requiring a ground port that names
the ground net. Other port nets use their port names, while internal nets use
hierarchical paths such as ``X1/mid``. Reference designators follow IEEE 315 device
prefixes (R, C, L, T, D, Q, or U) numbered in bill order, with fields recording the
instance path and manufacturer part number.
"""

__lazy_modules__ = ["jolty.simulate"]

import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import Field

from jolty import catalog, ir
from jolty.errors import JoltyError, NetlistError
from jolty.simulate import Simulation
from jolty.units import format_si

PREFIXES: Final[Mapping[str, str]] = {
    "resistor": "R",
    "capacitor": "C",
    "inductor": "L",
    "coupled_inductors": "T",
    "diode": "D",
    "bipolar_transistor": "Q",
    "jfet": "Q",
    "mosfet": "Q",
    "vdmos": "Q",
    "bsim3": "Q",
    "bsim4": "Q",
    "vbic": "Q",
}
"""Mapping from device model to IEEE 315 reference designator prefix (default U)."""

TWO_PADS: Final[Mapping[str, tuple[str, ...]]] = {"a": ("1",), "b": ("2",)}
"""Default two-pad mapping for passives whose catalog part omits explicit pins."""

_NAMESPACE: Final = uuid.UUID("21d190f3-3f5d-4c5c-a8d6-37c2b6468bba")
"""UUIDv5 namespace for deterministic footprint timestamps derived from paths."""


class Node(ir.FrozenModel):
    """Single footprint pad connection attached to a net."""

    reference: str = Field(
        description="Reference designator of the footprint, such as R1."
    )
    pad: str = Field(description="Pad identifier on the footprint, such as 1.")
    terminals: tuple[str, ...] = Field(
        description="Instance terminal or port names mapped to this pad, such as base."
    )


class Net(ir.FrozenModel):
    """Named board net listing all connected footprint pads."""

    name: str
    nodes: tuple[Node, ...]


class Footprint(ir.FrozenModel):
    """Board footprint representing the catalog part for one circuit instance."""

    reference: str = Field(
        description="Reference designator assigned to the footprint, such as R1."
    )
    instance: str = Field(
        description="Hierarchical path of the instance inside the exported component."
    )
    part: catalog.Part

    @property
    def value(self) -> str:
        """Formatted nominal part value (such as 4.7 kΩ), or part number."""
        part = self.part
        if part.kind is not None and part.value is not None:
            return format_si(part.value, catalog.SYMBOLS[part.kind])
        return part.part_number or ""


class Netlist(ir.FrozenModel):
    """KiCad board netlist containing footprints and pad-level nets."""

    footprints: tuple[Footprint, ...]
    nets: tuple[Net, ...] = Field(
        description="Board nets sorted alphabetically by name."
    )

    def text(self) -> str:
        """Formats the netlist as a KiCad version-E S-expression string.

        Returns:
            Trailing-newline-terminated KiCad version-E netlist text.
        """
        written = _List(
            "export",
            (
                _List("version", ("E",)),
                _List(
                    "components", tuple(_component(each) for each in self.footprints)
                ),
                _List(
                    "nets",
                    tuple(
                        _net(code, net) for code, net in enumerate(self.nets, start=1)
                    ),
                ),
            ),
        )
        return "\n".join(written.lines("")) + "\n"

    def save(self, path: Path) -> None:
        """Writes the formatted KiCad netlist to `path`.

        Args:
            path: Destination `.net` file path to write.
        """
        path.write_text(self.text(), encoding="utf-8")

    def describe(self) -> str:
        """Formats a plain-text summary of netlist counts and footprints.

        Returns:
            Multi-line summary listing footprint count, net count, and part assignments.
        """
        head = f"{len(self.footprints)} footprints on {len(self.nets)} nets"
        rows = [
            f"  {each.reference}: {each.instance}, {each.value}, {each.part.footprint}"
            for each in self.footprints
        ]
        return "\n".join([head, *rows])


def netlist(circuit: ir.Circuit, parts: catalog.Catalog, instance: str = "") -> Netlist:
    """Builds a KiCad netlist for the component at `instance` (top level by default).

    Args:
        circuit: Circuit IR containing the target component.
        parts: Parts catalog supplying footprints and pad-to-terminal pin mappings.
        instance: Hierarchical component instance path, or empty for the top component.

    Returns:
        Exported `Netlist` containing footprints and pad-level board nets.

    Raises:
        JoltyError: When no component instance exists at `instance`.
        NetlistError: When the component lacks a ground port, or when any instance
            lacks a matching part, footprint, or valid electrical pin mapping.
            The exception lists every problem found.
        ElaborationError: When the target component fails standalone elaboration.
    """
    component = _component_at(circuit, instance)
    alone = circuit.model_copy(update={"top": component})
    ground = alone.ground_port()
    if ground is None:
        message = (
            f"the component {component} has no ground port, the reference its nets "
            "are elaborated against"
        )
        raise NetlistError([message])
    reached = _Reached(Simulation(alone), ground)
    problems: list[str] = []
    footprints: list[Footprint] = []
    nodes: defaultdict[str, list[Node]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for path, placed in catalog.built(alone, component):
        if all(reached.thermal(path, terminal) for terminal in placed.terminals):
            continue
        part = parts.part_for(placed)
        if isinstance(part, str):
            problems.append(f"{path}: {part}")
            continue
        found = _pads(path, placed, part, reached)
        if isinstance(found, list):
            problems.extend(f"{path}: {problem}" for problem in found)
            continue
        prefix = (
            PREFIXES.get(placed.of.model, "U")
            if isinstance(placed.of, ir.DeviceOf)
            else "U"
        )
        counts[prefix] += 1
        reference = f"{prefix}{counts[prefix]}"
        footprints.append(Footprint(reference=reference, instance=path, part=part))
        for pad, (net, joined) in found.items():
            nodes[net].append(Node(reference=reference, pad=pad, terminals=joined))
    if problems:
        raise NetlistError(problems)
    return Netlist(
        footprints=tuple(footprints),
        nets=tuple(Net(name=name, nodes=tuple(nodes[name])) for name in sorted(nodes)),
    )


def _component_at(circuit: ir.Circuit, instance: str) -> str:
    """Resolves the component definition name at `instance`, or top when empty.

    Raises:
        JoltyError: When no component instance exists at `instance`.
    """
    if not instance:
        return circuit.top
    found = circuit.find(instance)
    if found is None or not isinstance(found.of, ir.ComponentOf):
        message = f"no component instance is at {instance!r}"
        raise JoltyError(message)
    return found.of.component


type _Simulation = Simulation


@dataclass(frozen=True, slots=True)
class _Reached:
    """Elaborated net lookup helper for terminals inside a standalone component."""

    simulation: _Simulation
    """Standalone elaboration of the exported component."""
    ground: str
    """Name of the component ground net."""

    def net(self, path: str, terminal: str) -> str:
        """Returns the net name connected to `terminal` on the instance at `path`.

        Raises:
            KeyError: When the instance terminal is not connected to any net.
        """
        found = self.simulation.joined(path, terminal)
        if found is None:
            message = f"{path}.{terminal} is on no electrical or thermal net"
            raise KeyError(message)
        return self.ground if found.node is None else found.node

    def thermal(self, path: str, terminal: str) -> bool:
        """Returns whether `terminal` on the instance at `path` is on a thermal net."""
        found = self.simulation.joined(path, terminal)
        return found is not None and found.thermal


def _pads(
    path: str, placed: ir.Instance, part: catalog.Part, reached: _Reached
) -> dict[str, tuple[str, tuple[str, ...]]] | list[str]:
    """Maps each pad of `part` for `placed` at `path` to its net and terminal names.

    Returns a list of error strings when footprint or pin definitions are invalid.
    """
    electrical = [
        terminal for terminal in placed.terminals if not reached.thermal(path, terminal)
    ]
    problems = []
    if part.footprint is None:
        problems.append(f"its part states no footprint: {part.describe()}")
    pins = _pins(placed, part)
    if pins is None:
        return [*problems, f"its part states no pins: {part.describe()}"]
    unpinned = _unpinned(placed, pins, electrical)
    problems.extend(unpinned)
    pads: dict[str, tuple[str, tuple[str, ...]]] = {}
    for terminal in electrical if not unpinned else ():
        net = reached.net(path, terminal)
        for pad in pins[terminal]:
            held, joined = pads.get(pad, (net, ()))
            if held != net:
                problems.append(f"its part's pins put pad {pad} on {held} and on {net}")
            pads[pad] = (held, (*joined, terminal))
    return problems or pads


def _pins(placed: ir.Instance, part: catalog.Part) -> catalog.Pins | None:
    """Returns explicit pins for `part`, or default two-pad pins for passive devices."""
    if part.pins is not None:
        return part.pins
    if isinstance(placed.of, ir.DeviceOf) and placed.of.model in catalog.VALUED:
        return dict(TWO_PADS)
    return None


def _unpinned(
    placed: ir.Instance, pins: catalog.Pins, electrical: Iterable[str]
) -> list[str]:
    """Validates `pins` against the terminals of `placed` and its `electrical` subset.

    Reports unknown terminals, thermal terminals assigned to pads, and omitted
    electrical terminals.
    """
    problems = []
    for terminal in pins:
        if terminal not in placed.terminals:
            problems.append(f"its part's pins name {terminal}, which it does not have")
        elif terminal not in electrical:
            problems.append(
                f"its part's pins put {terminal}, on a thermal net, on pads"
            )
    problems.extend(
        f"its part's pins leave out {terminal}"
        for terminal in electrical
        if terminal not in pins
    )
    return problems


@dataclass(frozen=True, slots=True)
class _List:
    """S-expression node consisting of a keyword and string or nested list items."""

    keyword: str
    items: tuple[str | _List, ...]

    def lines(self, indent: str) -> list[str]:
        """Formats the S-expression as indented lines starting from `indent`.

        Flat expressions stay on one line; nested expressions place the keyword and
        string atoms on the first line and indent each child list by two spaces.
        """
        if all(isinstance(item, str) or item.flat() for item in self.items):
            return [indent + self.inline()]
        head = " ".join(
            [
                f"({self.keyword}",
                *(_quoted(item) for item in self.items if isinstance(item, str)),
            ]
        )
        lines = [indent + head]
        for item in self.items:
            if isinstance(item, _List):
                lines.extend(item.lines(indent + "  "))
        lines[-1] += ")"
        return lines

    def flat(self) -> bool:
        """Returns whether every item in the S-expression is a string atom."""
        return all(isinstance(item, str) for item in self.items)

    def inline(self) -> str:
        """Formats the S-expression onto a single line."""
        items = (
            _quoted(item) if isinstance(item, str) else item.inline()
            for item in self.items
        )
        return "(" + " ".join([self.keyword, *items]) + ")"


def _quoted(text: str) -> str:
    """Returns `text` wrapped in double quotes with escaped quotes and newlines."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _field(name: str, value: str) -> _List:
    return _List("property", (_List("name", (name,)), _List("value", (value,))))


def _component(footprint: Footprint) -> _List:
    number = footprint.part.part_number
    fields = (_field("MPN", number),) if number is not None else ()
    return _List(
        "comp",
        (
            _List("ref", (footprint.reference,)),
            _List("value", (footprint.value,)),
            _List("footprint", (footprint.part.footprint or "",)),
            *fields,
            _field("Instance", footprint.instance),
            _List("sheetpath", (_List("names", ("/",)), _List("tstamps", ("/",)))),
            _List("tstamps", (str(uuid.uuid5(_NAMESPACE, footprint.instance)),)),
        ),
    )


def _net(code: int, net: Net) -> _List:
    nodes = (
        _List(
            "node",
            (
                _List("ref", (node.reference,)),
                _List("pin", (node.pad,)),
                _List("pinfunction", ("/".join(node.terminals),)),
            ),
        )
        for node in net.nodes
    )
    return _List(
        "net", (_List("code", (str(code),)), _List("name", (net.name,)), *nodes)
    )
