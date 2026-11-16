# Copyright 2026 Maxence Faldor
"""Components written as KiCad netlists, from which KiCad lays out their boards.

[`netlist`][jolty.kicad.netlist] writes the component at an instance, or the top, as the
footprints of the parts its [bill of materials][jolty.catalog.bill] takes and the nets
that join their pads, a file KiCad's PCB editor imports. Each part states its footprint,
as KiCad's libraries name it, and the pads each terminal or port of its instance joins;
a resistor, a capacitor or an inductor whose part states no pins joins ``a`` to pad 1
and ``b`` to pad 2, as KiCad's two-pad footprints number them. An instance whose every
terminal is on a thermal net, such as a thermal resistance, has no footprint, and a
terminal on a thermal net, such as a heated transistor's ``thermal``, joins no pad.

The component is elaborated alone, so it has a ground port, whose name its ground net
takes. Each other port's net takes the port's name, and each net inside its path, such
as ``X1/mid``. A footprint's reference designator is its device's letter, R, C, L, T, D
or Q, or U for any other part, numbered in the bill's order; its fields name the
instance it builds, by its path in the component, and its maker's number.
"""

import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

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
    "vbic": "Q",
}
"""The prefix of the reference designator of each device's part, the letter IEEE 315
gives it; any other part's is U."""

TWO_PADS: Final[Mapping[str, tuple[str, ...]]] = {"a": ("1",), "b": ("2",)}
"""The pads of a resistor, a capacitor or an inductor whose part states no pins."""

_NAMESPACE: Final = uuid.UUID("21d190f3-3f5d-4c5c-a8d6-37c2b6468bba")
"""The namespace of the identifiers jolty gives footprints, from their instances' paths,
so a footprint keeps its identifier from one netlist of a component to the next."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Node(_Model):
    """A footprint's pad, on a net."""

    reference: str = Field(description="The footprint's designator, such as R1.")
    pad: str = Field(description="The pad's number in the footprint, such as 1.")
    terminals: tuple[str, ...] = Field(
        description="The terminals or ports of the instance it joins, such as base."
    )


class Net(_Model):
    """A net, and the pads it joins."""

    name: str
    nodes: tuple[Node, ...]


class Footprint(_Model):
    """The footprint of the part that builds an instance."""

    reference: str = Field(description="Its reference designator, such as R1.")
    instance: str = Field(
        description="The path of the instance it builds, in the component written."
    )
    part: catalog.Part

    @property
    def value(self) -> str:
        """Its part's value and unit, such as 4.7 kΩ, or else its maker's number."""
        part = self.part
        if part.device is not None and part.value is not None:
            return format_si(part.value, catalog.SYMBOLS[part.device])
        return part.number or ""


class Netlist(_Model):
    """The footprints of a component's parts, and the nets that join their pads."""

    footprints: tuple[Footprint, ...]
    nets: tuple[Net, ...] = Field(description="Its nets, in order of their names.")

    def text(self) -> str:
        """The netlist in KiCad's S-expression format, version E."""
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
        """Writes the netlist to `path`, as KiCad's PCB editor imports it."""
        path.write_text(self.text(), encoding="utf-8")

    def describe(self) -> str:
        """The netlist as text: its size, then a line per footprint."""
        head = f"{len(self.footprints)} footprints on {len(self.nets)} nets"
        rows = [
            f"  {each.reference}: {each.instance}, {each.value}, {each.part.footprint}"
            for each in self.footprints
        ]
        return "\n".join([head, *rows])


def netlist(circuit: ir.Circuit, parts: catalog.Catalog, instance: str = "") -> Netlist:
    """The KiCad netlist of the component at `instance`, the top by default.

    Raises:
        JoltyError: When no component instance is at `instance`.
        NetlistError: When the component has no ground port, or an instance cannot be
            written: no part of `parts` builds it, or its part states no footprint, or
            its part's pins name a terminal it lacks or one on a thermal net, leave out
            one on an electrical net, or put a pad on two nets. It names every one.
        ElaborationError: When the component does not elaborate alone.
    """
    component = _component_at(circuit, instance)
    alone = circuit.model_copy(update={"top": component})
    ground = next(
        (
            port.name
            for port in alone.components[component].ports
            if port.role == ir.Role.GROUND
        ),
        None,
    )
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
    """The component the instance at `instance` places, or the top for the empty path.

    Raises:
        JoltyError: When no component instance is at `instance`.
    """
    if not instance:
        return circuit.top
    found = circuit.find(instance)
    if found is None or not isinstance(found.of, ir.ComponentOf):
        message = f"no component instance is at {instance!r}"
        raise JoltyError(message)
    return found.of.component


@dataclass(frozen=True, slots=True)
class _Reached:
    """Where the terminals of an elaborated component are."""

    simulation: Simulation
    """The component elaborated alone."""
    ground: str
    """The name of its ground net."""

    def net(self, path: str, terminal: str) -> str:
        """The name of the net `terminal` of the instance at `path` is on.

        Raises:
            KeyError: When the instance has no such terminal on a net.
        """
        found = self.simulation.joined(path, terminal)
        if found is None:
            message = f"{path}.{terminal} is on no electrical or thermal net"
            raise KeyError(message)
        return self.ground if found.node is None else found.node

    def thermal(self, path: str, terminal: str) -> bool:
        """Whether `terminal` of the instance at `path` is on a thermal net."""
        found = self.simulation.joined(path, terminal)
        return found is not None and found.thermal


def _pads(
    path: str, placed: ir.Instance, part: catalog.Part, reached: _Reached
) -> dict[str, tuple[str, tuple[str, ...]]] | list[str]:
    """Each pad of `part`, which builds `placed` at `path`, its net and its terminals.

    Or why its pads cannot be found: every problem of its part's footprint and pins.
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
    """The pins of `part`, or two pads for a resistor, capacitor or inductor's."""
    if part.pins is not None:
        return part.pins
    if isinstance(placed.of, ir.DeviceOf) and placed.of.model in catalog.VALUED:
        return dict(TWO_PADS)
    return None


def _unpinned(
    placed: ir.Instance, pins: catalog.Pins, electrical: Iterable[str]
) -> list[str]:
    """What is wrong with `pins` for the terminals of `placed`.

    The pins name a terminal it lacks or one on a thermal net, or leave out one of
    `electrical`, its terminals on no thermal net.
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
    """An S-expression: a keyword, then its items, each a string or a list."""

    keyword: str
    items: tuple[str | _List, ...]

    def lines(self, indent: str) -> list[str]:
        """The list written from `indent`, as lines.

        It takes one line when no item holds a list; otherwise its keyword and strings
        take the first, and each list lines of its own, one level deeper.
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
        """Whether every item is a string."""
        return all(isinstance(item, str) for item in self.items)

    def inline(self) -> str:
        """The list on one line."""
        items = (
            _quoted(item) if isinstance(item, str) else item.inline()
            for item in self.items
        )
        return "(" + " ".join([self.keyword, *items]) + ")"


def _quoted(text: str) -> str:
    """`text` as a quoted string, its backslashes and quotes escaped."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _field(name: str, value: str) -> _List:
    return _List("property", (_List("name", (name,)), _List("value", (value,))))


def _component(footprint: Footprint) -> _List:
    number = footprint.part.number
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
