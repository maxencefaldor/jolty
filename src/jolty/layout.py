# Copyright 2026 Maxence Faldor
"""Schematic layout: where each instance of a component goes, and how its nets run.

The layout follows the conventions engineers read. Signal flows left to right: each
instance takes a column by its distance from the sources, or from the component's first
signal port when it has none, and a two-terminal device between two signal nets lies
across its column, its earlier net on the left. A device between a signal net and the
ground hangs below the net, and one between a signal net and a supply stands above it.
Ground and supply nets draw no wires: a symbol marks each of their pins.

Every other net is routed on a grid, pin by pin, by an A* search that charges for each
step, each bend and each crossing. Wires of two nets may cross at right angles, never
run along each other or meet at a point, so a wire touches only its own net. A net the
router cannot route gets a label at each pin instead, as a draftsman would draw it.

Coordinates are grid units, one pin pitch each, with y growing upward.

Examples:
    >>> from jolty import Builder, Role, circuit, component, devices
    >>> from jolty.units import V, kohm, nF
    >>> @component
    ... def rc(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(1 * V), id="V1")
    ...     resistor = c.place(devices.resistor(1 * kohm), id="R1")
    ...     capacitor = c.place(devices.capacitor(1 * nF), id="C1")
    ...     c.connect(source.positive, resistor.a, name="in")
    ...     c.connect(resistor.b, capacitor.a, name="out")
    ...     c.connect(capacitor.b, source.negative, ground)
    >>> drawn = layout(circuit(rc()), "rc")
    >>> [(item.id, item.column) for item in drawn.instances]
    [('V1', 0), ('R1', 1), ('C1', 2)]
    >>> drawn.instances[1].orientation
    Orientation(turns=1, flipped=False)
    >>> sorted(net.name for net in drawn.nets if net.segments)
    ['in', 'out']
"""

import heapq
import itertools
import math
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

from jolty import ir

type Point = tuple[int, int]
"""A point of the grid."""

type Segment = tuple[Point, Point]
"""A straight, horizontal or vertical piece of wire."""

_COLUMN_GAP: Final = 5
"""Grid units between the widest symbols of neighboring columns, for wires to run."""

_ROW_GAP: Final = 4
"""Grid units between the lowest pin of a row and the highest of the next, for the
rails' symbols and for wires to run."""

_LANE_GAP: Final = 2
"""Grid units between the widest symbols of neighboring lanes of a column."""

_FLAG_GAP: Final = 4
"""Grid units between the outermost pins and the port flags beyond them."""

_MARGIN: Final = 4
"""Grid units around the drawing where wires may run."""

_STEP: Final = 1
_BEND: Final = 3
_CROSSING: Final = 4


@dataclass(frozen=True, slots=True)
class Shape:
    """A symbol's geometry in its own frame: its pins, and what its body covers.

    The frame is that of the symbol upright, with its first terminal at the top.
    """

    pins: Mapping[str, Point]
    body: tuple[Point, Point]
    """The lower-left and upper-right corners of the body, which no wire crosses."""


_TWO_TERMINAL: Final = {
    "resistor": ("a", "b"),
    "capacitor": ("a", "b"),
    "inductor": ("a", "b"),
    "diode": ("anode", "cathode"),
    "voltage_source": ("positive", "negative"),
    "current_source": ("positive", "negative"),
}
"""Each two-terminal model's terminals, the first at the top of the upright symbol."""

_CONTROLLED: Final = frozenset(
    {"voltage_controlled_voltage_source", "voltage_controlled_current_source"}
)

_SOURCES: Final = frozenset({"voltage_source", "current_source"})


def _shape(instance: ir.Instance, circuit: ir.Circuit) -> Shape:
    """The shape of the symbol that draws `instance`."""
    match instance.of:
        case ir.DeviceOf(device=model) if model in _TWO_TERMINAL:
            top, bottom = _TWO_TERMINAL[model]
            return Shape({top: (0, 2), bottom: (0, -2)}, ((-1, -1), (1, 1)))
        case ir.DeviceOf(device=model) if model in _CONTROLLED:
            pins = {
                "positive": (0, 2),
                "negative": (0, -2),
                "control_positive": (-4, 1),
                "control_negative": (-4, -1),
            }
            return Shape(pins, ((-3, -1), (1, 1)))
        case ir.ComponentOf(component=name) if name in circuit.components:
            ports = [(port.name, port.role) for port in circuit.components[name].ports]
            return _box(ports, name, instance.id)
        case ir.DeviceOf(device=name) | ir.ComponentOf(component=name):
            signals = [(terminal, ir.Role.SIGNAL) for terminal in instance.terminals]
            return _box(signals, name, instance.id)


_CHARACTER: Final = 0.2
"""The width of a character of a box's labels, in grid units."""


def _box(ports: list[tuple[str, ir.Role]], name: str, id: str) -> Shape:
    """A box with signal ports on its sides, supplies on top and grounds below.

    Inputs go on the left and outputs on the right, and other signals fill the sides in
    turn, the left first, to balance them. The box is wide enough to write the
    component's function, and its arguments below it, between the names of its side
    ports, and to write the instance's id between its upper-left corner and its first
    supply.
    """
    inputs = [port for port, role in ports if role is ir.Role.INPUT]
    outputs = [port for port, role in ports if role is ir.Role.OUTPUT]
    signals = [port for port, role in ports if role is ir.Role.SIGNAL]
    middle = min(
        max((len(signals) + len(outputs) - len(inputs) + 1) // 2, 0), len(signals)
    )
    left, right = inputs + signals[:middle], outputs + signals[middle:]
    supplies = [port for port, role in ports if role is ir.Role.SUPPLY]
    grounds = [port for port, role in ports if role is ir.Role.GROUND]
    title, arguments = split_name(name)
    inside = max(len(title), len(arguments))
    text = max(map(len, left), default=0) + max(map(len, right), default=0) + inside
    beside = len(supplies) + math.ceil(_CHARACTER * len(id)) if supplies else 0
    half_width = max(len(grounds), beside, math.ceil(_CHARACTER * text / 2) + 1)
    half_height = max(len(left), len(right), 1)
    pins: dict[str, Point] = {}
    for names, x in ((left, -half_width - 1), (right, half_width + 1)):
        pins.update(
            {port: (x, half_height - 1 - 2 * index) for index, port in enumerate(names)}
        )
    for names, y in ((supplies, half_height + 1), (grounds, -half_height - 1)):
        pins.update(
            {port: (2 * index - len(names) + 1, y) for index, port in enumerate(names)}
        )
    return Shape(pins, ((-half_width, -half_height), (half_width, half_height)))


def split_name(name: str) -> tuple[str, str]:
    """A component's function and its arguments, from its name.

    Examples:
        >>> split_name("divider[ratio=0.5,load=1000.0]")
        ('divider', 'ratio=0.5, load=1000.0')
        >>> split_name("rc")
        ('rc', '')
    """
    title, _, arguments = name.partition("[")
    return title, arguments.removesuffix("]").replace(",", ", ")


@dataclass(frozen=True, slots=True)
class Orientation:
    """How a symbol stands on the sheet, from upright with its first terminal on top."""

    turns: int = 0
    """Quarter turns counterclockwise: 1 lays a two-terminal symbol across with its
    first terminal on the left."""
    flipped: bool = False
    """Whether the symbol is flipped upside down before it turns."""

    def apply[N: (int, float)](self, point: tuple[N, N]) -> tuple[N, N]:
        """Where `point` of the upright symbol goes."""
        x, y = (point[0], -point[1]) if self.flipped else point
        match self.turns % 4:
            case 0:
                return x, y
            case 1:
                return -y, x
            case 2:
                return -x, -y
            case _:
                return y, -x


@dataclass(frozen=True, slots=True)
class Placed:
    """An instance placed on the sheet."""

    id: str
    instance: ir.Instance
    shape: Shape
    origin: Point
    orientation: Orientation
    column: int

    def at[N: (int, float)](self, point: tuple[N, N]) -> tuple[N, N]:
        """Where `point` of the symbol's own frame lands on the sheet."""
        x, y = self.orientation.apply(point)
        return self.origin[0] + x, self.origin[1] + y

    def pin(self, terminal: str) -> Point:
        """Where `terminal`'s pin is on the sheet."""
        return self.at(self.shape.pins[terminal])

    def body(self) -> tuple[Point, Point]:
        """The lower-left and upper-right corners of the body on the sheet."""
        corners = [self.at(corner) for corner in self.shape.body]
        xs, ys = [x for x, _ in corners], [y for _, y in corners]
        return (min(xs), min(ys)), (max(xs), max(ys))


@dataclass(frozen=True, slots=True)
class Flag:
    """A signal port of the drawn component, at the edge of the sheet."""

    name: str
    point: Point
    side: int
    """-1 on the left, pointing in; 1 on the right, pointing out."""


@dataclass(slots=True)
class Net:
    """A net as drawn: its pins, and its wires or, failing them, its labels."""

    name: str
    role: ir.Role
    pins: list[Point]
    segments: list[Segment] = field(default_factory=list)
    junctions: list[Point] = field(default_factory=list)
    labeled: bool = False
    """Whether the router found no wiring, so each pin carries the net's name."""


@dataclass(frozen=True, slots=True)
class Layout:
    """A component laid out: its instances, its port flags and its nets."""

    component: str
    instances: tuple[Placed, ...]
    flags: tuple[Flag, ...]
    nets: tuple[Net, ...]

    def bounds(self) -> tuple[Point, Point]:
        """The lower-left and upper-right corners of everything drawn."""
        points = [point for net in self.nets for point in net.pins]
        points += [flag.point for flag in self.flags]
        for placed in self.instances:
            points.extend(placed.body())
        xs, ys = [x for x, _ in points], [y for _, y in points]
        return (min(xs, default=0), min(ys, default=0)), (
            max(xs, default=0),
            max(ys, default=0),
        )


def layout(circuit: ir.Circuit, name: str) -> Layout:
    """The layout of the component `name` of `circuit`.

    Raises:
        KeyError: When the circuit has no component named `name`.
    """
    component = circuit.components[name]
    roles = _roles(circuit, component)
    instances = component.instances
    columns = _columns(component, roles)
    rows = _rows(instances, roles, columns)
    placed = _place(circuit, instances, roles, columns, rows)
    flags = _flags(component, placed, roles)
    nets = _nets(placed, flags, roles)
    _route(nets, placed)
    return Layout(name, placed, flags, tuple(nets.values()))


def _roles(circuit: ir.Circuit, component: ir.Component) -> dict[str, ir.Role]:
    """The role each net of `component` plays, by name, which sets how it is drawn.

    A port of the component sets its net's role, an input or an output a signal. Any
    other net is a ground when a port of a component placed in it is a ground on that
    net, else a supply when one is a supply there, and else a signal.
    """
    found: dict[str, set[ir.Role]] = {}
    for instance in component.instances:
        ports = (
            {port.name: port.role for port in circuit.components[name].ports}
            if isinstance(instance.of, ir.ComponentOf)
            and (name := instance.of.component) in circuit.components
            else {}
        )
        for terminal, net in instance.terminals.items():
            found.setdefault(net, set()).add(ports.get(terminal, ir.Role.SIGNAL))
    roles = {net: _rail(played) for net, played in found.items()}
    roles.update({port.name: _rail({port.role}) for port in component.ports})
    return roles


def _rail(roles: set[ir.Role]) -> ir.Role:
    """The role a net plays for the ports on it: ground, then supply, then signal.

    A signal's direction plays no part.
    """
    for role in (ir.Role.GROUND, ir.Role.SUPPLY):
        if role in roles:
            return role
    return ir.Role.SIGNAL


def _signal_nets(instance: ir.Instance, roles: Mapping[str, ir.Role]) -> set[str]:
    return {net for net in instance.terminals.values() if roles[net] is ir.Role.SIGNAL}


def _columns(component: ir.Component, roles: Mapping[str, ir.Role]) -> dict[str, int]:
    """Each instance's column: its distance, in shared signal nets, from the start.

    Signal starts at the independent sources; in a component without them, at the
    instances on its input ports, or else on its first signal port, its input by
    convention; failing those, at its first instance. The search starts again from the
    first instance it has not reached.
    """
    instances = component.instances
    on_net: dict[str, list[str]] = {}
    for instance in instances:
        for net in _signal_nets(instance, roles):
            on_net.setdefault(net, []).append(instance.id)
    by_id = {instance.id: instance for instance in instances}
    sources = [
        instance.id
        for instance in instances
        if isinstance(instance.of, ir.DeviceOf) and instance.of.device in _SOURCES
    ]
    signals = [
        port.name for port in component.ports if roles[port.name] is ir.Role.SIGNAL
    ]
    declared = [
        instance
        for port in component.ports
        if port.role is ir.Role.INPUT
        for instance in on_net.get(port.name, [])
    ]
    inputs = list(dict.fromkeys(declared)) or (
        on_net.get(signals[0], []) if signals else []
    )
    columns: dict[str, int] = {}
    starts = sources or inputs or [instance.id for instance in instances[:1]]
    while starts:
        offset = max(columns.values(), default=-1) + 1
        columns.update(dict.fromkeys(starts, offset))
        queue = deque(starts)
        while queue:
            current = queue.popleft()
            for net in sorted(_signal_nets(by_id[current], roles)):
                for neighbor in on_net[net]:
                    if neighbor not in columns:
                        columns[neighbor] = columns[current] + 1
                        queue.append(neighbor)
        unreached = [
            instance.id for instance in instances if instance.id not in columns
        ]
        starts = unreached[:1]
    return columns


def _rows(
    instances: tuple[ir.Instance, ...],
    roles: Mapping[str, ir.Role],
    columns: Mapping[str, int],
) -> dict[str, int]:
    """Each instance's row within its column, ordered to keep wires short.

    Instances start in placement order, then each column sorts by the mean row of the
    other instances on its signal nets, net by net, a few times over: an instance that
    shares two nets with another is pulled toward it twice. An instance that shares no
    signal net, such as a supply's source, sinks to the bottom of its column. Each net
    keeps the sum of its instances' rows, so a sweep reads each pin a bounded number of
    times beside sorting each column.
    """
    nets = {instance.id: _signal_nets(instance, roles) for instance in instances}
    stacks: dict[int, list[str]] = {}
    for instance in instances:
        stacks.setdefault(columns[instance.id], []).append(instance.id)
    rows = {id: index for stack in stacks.values() for index, id in enumerate(stack)}
    sums: dict[str, int] = {}
    counts: dict[str, int] = {}
    for id, shared in nets.items():
        for net in shared:
            sums[net] = sums.get(net, 0) + rows[id]
            counts[net] = counts.get(net, 0) + 1

    def barycenter(id: str) -> tuple[bool, float]:
        """An instance's sort key: whether it is alone, then its neighbors' mean row."""
        others = sum(counts[net] - 1 for net in nets[id])
        if not others:
            return True, rows[id]
        return False, sum(sums[net] - rows[id] for net in nets[id]) / others

    for _ in range(4):
        for column in sorted(stacks):
            stack = stacks[column]
            stack.sort(key=barycenter)
            for index, id in enumerate(stack):
                for net in nets[id]:
                    sums[net] += index - rows[id]
                rows[id] = index
    return rows


def _place(
    circuit: ir.Circuit,
    instances: tuple[ir.Instance, ...],
    roles: Mapping[str, ir.Role],
    columns: Mapping[str, int],
    rows: Mapping[str, int],
) -> tuple[Placed, ...]:
    """Places every instance: its column, its row and lane in it, and its orientation.

    Within a column, devices that hang from the same signal net to a rail, one after
    the other, share a row side by side, each in a lane of its own, so their net runs
    straight across them. Every instance stands on its row's line, level across the
    columns, and rows lie as far apart as their symbols need.
    """
    earliest: dict[str, int] = {}
    for instance in instances:
        for net in instance.terminals.values():
            column = columns[instance.id]
            earliest[net] = min(earliest.get(net, column), column)
    by_id = {instance.id: instance for instance in instances}
    shapes = {instance.id: _shape(instance, circuit) for instance in instances}
    orientations = {
        instance.id: _orientation(instance, roles, earliest) for instance in instances
    }
    extents = {id: _extent(shapes[id], orientations[id]) for id in by_id}
    anchors = {
        id: _anchor(by_id[id], shapes[id], orientations[id], roles) for id in by_id
    }
    slots: dict[int, list[list[str]]] = {}
    for id in sorted(by_id, key=lambda id: (columns[id], rows[id])):
        stack = slots.setdefault(columns[id], [])
        hanging = _hanging(by_id[id], roles)
        if (
            stack
            and hanging is not None
            and _hanging(by_id[stack[-1][-1]], roles) == hanging
        ):
            stack[-1].append(id)
        else:
            stack.append([id])
    lines = _lines(slots.values(), extents, anchors)
    placed = []
    x = 0
    for column in sorted(slots):
        stack = slots[column]
        width = max(_width(extents[id]) for slot in stack for id in slot)
        pitch = width + _LANE_GAP
        for row, slot in enumerate(stack):
            for lane, id in enumerate(slot):
                (left, _), _ = extents[id]
                origin = (x - left + lane * pitch, lines[row] - anchors[id])
                placed.append(
                    Placed(id, by_id[id], shapes[id], origin, orientations[id], column)
                )
        x += (max(map(len, stack)) - 1) * pitch + width + _COLUMN_GAP
    order = {instance.id: index for index, instance in enumerate(instances)}
    return tuple(sorted(placed, key=lambda item: order[item.id]))


def _lines(
    stacks: Iterable[list[list[str]]],
    extents: Mapping[str, tuple[Point, Point]],
    anchors: Mapping[str, int],
) -> list[int]:
    """The height of each row's line, the rows spaced by what stands above and below."""
    above: dict[int, int] = {}
    below: dict[int, int] = {}
    for stack in stacks:
        for row, slot in enumerate(stack):
            for id in slot:
                (_, bottom), (_, top) = extents[id]
                above[row] = max(above.get(row, 0), top - anchors[id])
                below[row] = max(below.get(row, 0), anchors[id] - bottom)
    lines: list[int] = [0]
    for row in range(1, len(above)):
        lines.append(lines[-1] - below[row - 1] - _ROW_GAP - above[row])
    return lines


def _anchor(
    instance: ir.Instance,
    symbol: Shape,
    orientation: Orientation,
    roles: Mapping[str, ir.Role],
) -> int:
    """The height, in its oriented symbol's frame, at which an instance meets its line.

    A device that hangs from a signal net meets it at its pin on that net, any other
    upright device at its top, and anything else at its center.
    """
    hanging = _hanging(instance, roles)
    if hanging is not None:
        terminal = next(
            terminal for terminal, net in instance.terminals.items() if net == hanging
        )
        return orientation.apply(symbol.pins[terminal])[1]
    if isinstance(instance.of, ir.DeviceOf) and orientation.turns % 2 == 0:
        (_, _), (_, top) = _extent(symbol, orientation)
        return top
    return 0


def _hanging(instance: ir.Instance, roles: Mapping[str, ir.Role]) -> str | None:
    """The signal net a two-terminal device hangs from to a rail, if it does."""
    if (
        not isinstance(instance.of, ir.DeviceOf)
        or instance.of.device not in _TWO_TERMINAL
    ):
        return None
    nets = [instance.terminals[name] for name in _TWO_TERMINAL[instance.of.device]]
    signals = [net for net in nets if roles[net] is ir.Role.SIGNAL]
    rails = [net for net in nets if roles[net] is not ir.Role.SIGNAL]
    return signals[0] if len(signals) == 1 and len(rails) == 1 else None


def _width(extent: tuple[Point, Point]) -> int:
    (left, _), (right, _) = extent
    return right - left


def _extent(symbol: Shape, orientation: Orientation) -> tuple[Point, Point]:
    """The lower-left and upper-right corners of a symbol's pins and body, oriented."""
    points = [
        orientation.apply(point) for point in [*symbol.pins.values(), *symbol.body]
    ]
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _orientation(
    instance: ir.Instance, roles: Mapping[str, ir.Role], earliest: Mapping[str, int]
) -> Orientation:
    """How a device stands: upright on a rail, across between two signals.

    A device keeps a terminal on the ground at its bottom and one on a supply at its
    top: a two-terminal device turns over, and a controlled source flips upside down to
    keep its control leads on its left. An independent source stands upright. Any other
    two-terminal device lies across, its terminal on the earlier net on the left.
    """
    if not isinstance(instance.of, ir.DeviceOf):
        return Orientation()
    model = instance.of.device
    if model in _CONTROLLED:
        top, bottom = instance.terminals["positive"], instance.terminals["negative"]
        return Orientation(flipped=_inverted(roles[top], roles[bottom]))
    if model not in _TWO_TERMINAL:
        return Orientation()
    top, bottom = (instance.terminals[name] for name in _TWO_TERMINAL[model])
    if _inverted(roles[top], roles[bottom]):
        return Orientation(turns=2)
    if model in _SOURCES or {roles[top], roles[bottom]} != {ir.Role.SIGNAL}:
        return Orientation()
    return Orientation(turns=1 if earliest[top] <= earliest[bottom] else 3)


def _inverted(top: ir.Role, bottom: ir.Role) -> bool:
    """Whether a symbol with these roles on its top and bottom is upside down."""
    return top is ir.Role.GROUND or bottom is ir.Role.SUPPLY


def _flags(
    component: ir.Component, placed: tuple[Placed, ...], roles: Mapping[str, ir.Role]
) -> tuple[Flag, ...]:
    """A flag for each signal port: on the left for an input, on the right otherwise.

    A port declared an input or an output is one; any other signal port is an input
    when its net reaches no further than the first half of the columns. Its flag stands
    level with its net's outermost pin on its side, below any flag already there.
    """
    pins: dict[str, list[tuple[Point, int]]] = {}
    for item in placed:
        for terminal, net in item.instance.terminals.items():
            pins.setdefault(net, []).append((item.pin(terminal), item.column))
    points = [point for entries in pins.values() for point, _ in entries]
    if not points:
        return ()
    edges = {
        -1: min(x for x, _ in points) - _FLAG_GAP,
        1: max(x for x, _ in points) + _FLAG_GAP,
    }
    last = max(item.column for item in placed)
    taken: dict[int, set[int]] = {-1: set(), 1: set()}
    flags = []
    for port in component.ports:
        if roles.get(port.name) is not ir.Role.SIGNAL:
            continue
        entries = pins.get(port.name, [])
        reach = max((column for _, column in entries), default=0)
        match port.role:
            case ir.Role.INPUT:
                side = -1
            case ir.Role.OUTPUT:
                side = 1
            case _:
                side = 1 if last > 0 and 2 * reach > last else -1
        outermost = max(
            entries, key=lambda entry: entry[0][0] * side, default=((0, 0), 0)
        )
        y = outermost[0][1]
        while y in taken[side]:
            y -= 2
        taken[side].add(y)
        flags.append(Flag(port.name, (edges[side], y), side))
    return tuple(flags)


def _nets(
    placed: tuple[Placed, ...], flags: tuple[Flag, ...], roles: Mapping[str, ir.Role]
) -> dict[str, Net]:
    nets: dict[str, Net] = {}
    for item in placed:
        for terminal, name in item.instance.terminals.items():
            nets.setdefault(name, Net(name, roles[name], [])).pins.append(
                item.pin(terminal)
            )
    for flag in flags:
        nets.setdefault(flag.name, Net(flag.name, ir.Role.SIGNAL, [])).pins.append(
            flag.point
        )
    return nets


type _Heading = tuple[int, int] | None
"""The direction a search last stepped in, or None at its start."""

type _State = tuple[Point, _Heading]

_DIRECTIONS: Final = ((1, 0), (-1, 0), (0, 1), (0, -1))

_JOINED: Final = 3
"""Wires of a net meeting at a point that make it a junction."""

_JOINED_AT_PIN: Final = 2
"""Wires of a net meeting at one of its pins that make the pin a junction."""


class _Router:
    """Routes nets on a grid, keeping each net's wires apart from every other's.

    The grid knows what occupies each point and step: the bodies of symbols, the pins of
    every net, the steps each net's wires take, the points a wire runs straight through,
    where other nets may cross it, and the points where a wire bends, branches or ends,
    which no other net may touch.
    """

    def __init__(self, nets: Iterable[Net], placed: Iterable[Placed]) -> None:
        """Prepares a grid around the pins of `nets` and the bodies of `placed`."""
        nets, placed = list(nets), list(placed)
        self.blocked: set[Point] = set()
        corners: list[Point] = [pin for net in nets for pin in net.pins]
        for item in placed:
            (left, bottom), (right, top) = item.body()
            corners += [(left, bottom), (right, top)]
            self.blocked.update(
                (x, y) for x in range(left, right + 1) for y in range(bottom, top + 1)
            )
        xs, ys = [x for x, _ in corners], [y for _, y in corners]
        self.low = (min(xs, default=0) - _MARGIN, min(ys, default=0) - _MARGIN)
        self.high = (max(xs, default=0) + _MARGIN, max(ys, default=0) + _MARGIN)
        self.pins: dict[Point, str] = {}
        for net in nets:
            for pin in net.pins:
                self.pins.setdefault(pin, net.name)
        self.steps: dict[frozenset[Point], str] = {}
        self.through: dict[Point, tuple[str, str]] = {}
        self.ends: dict[Point, str] = {}

    def route(self, net: Net) -> None:
        """Connects every pin of `net` to its first, or labels the net if it cannot."""
        ordered = sorted(set(net.pins))
        tree = {ordered[0]}
        paths: list[list[Point]] = []
        for pin in ordered[1:]:
            path = self._search(net.name, tree, pin)
            if path is None:
                for done in paths:
                    self._release(net.name, done)
                net.labeled = True
                return
            paths.append(path)
            tree.update(path)
            self._claim(net.name, path)
        net.segments = _segments(paths)
        net.junctions = _junctions(paths, set(net.pins))

    def _search(self, name: str, tree: set[Point], pin: Point) -> list[Point] | None:
        """The cheapest path for the net `name` from any point of its `tree` to `pin`.

        The search leaves from the whole tree for the one pin, so its estimate, the
        distance to the pin, costs the same at each point however large the tree grows.
        """

        def estimate(point: Point) -> int:
            return abs(point[0] - pin[0]) + abs(point[1] - pin[1])

        counter = itertools.count()
        starts: list[_State] = [(point, None) for point in sorted(tree)]
        frontier = [(estimate(state[0]), next(counter), state) for state in starts]
        heapq.heapify(frontier)
        costs: dict[_State, int] = dict.fromkeys(starts, 0)
        previous: dict[_State, _State] = {}
        while frontier:
            _, _, state = heapq.heappop(frontier)
            point, heading = state
            if point == pin:
                return _path(previous, state)
            for direction in self._directions(name, point, heading):
                following = (point[0] + direction[0], point[1] + direction[1])
                if not self._open(name, point, following, direction, pin):
                    continue
                cost = costs[state] + _STEP
                if heading is not None and direction != heading:
                    cost += _BEND
                if self._crosses(name, following):
                    cost += _CROSSING
                successor = (following, direction)
                if cost < costs.get(successor, cost + 1):
                    costs[successor] = cost
                    previous[successor] = state
                    priority = cost + estimate(following)
                    heapq.heappush(frontier, (priority, next(counter), successor))
        return None

    def _directions(
        self, name: str, point: Point, heading: _Heading
    ) -> Iterable[tuple[int, int]]:
        """The directions a path may leave `point` in: straight on when crossing."""
        if heading is not None and self._crosses(name, point):
            return (heading,)
        return _DIRECTIONS

    def _crosses(self, name: str, point: Point) -> bool:
        """Whether `point` is on a wire of another net, which a path may only cross."""
        return point in self.through and self.through[point][0] != name

    def _open(
        self,
        name: str,
        point: Point,
        following: Point,
        direction: tuple[int, int],
        pin: Point,
    ) -> bool:
        """Whether the net `name` may step from `point` to `following`."""
        x, y = following
        inside = self.low[0] <= x <= self.high[0] and self.low[1] <= y <= self.high[1]
        ours = {None, name}
        free = (
            self.steps.get(frozenset((point, following))) in ours
            and self.pins.get(following) in ours
            and self.ends.get(following) in ours
            and (following not in self.blocked or following == pin)
        )
        return inside and free and self._may_cross(name, following, direction, pin)

    def _may_cross(
        self, name: str, point: Point, direction: tuple[int, int], pin: Point
    ) -> bool:
        """Whether a step in `direction` onto `point` runs along no other wire."""
        if not self._crosses(name, point):
            return True
        way = "h" if direction[1] == 0 else "v"
        return self.through[point][1] != way and point != pin

    def _claim(self, name: str, path: list[Point]) -> None:
        """Claims `path` for the net `name`.

        A point the path runs straight through, where no wire of the net was before, is
        one other nets may cross; any other point of the path is the net's alone.
        """
        for first, second in itertools.pairwise(path):
            self.steps[frozenset((first, second))] = name
        inner = {
            point: (before, after)
            for before, point, after in zip(path, path[1:], path[2:], strict=False)
        }
        for point in path:
            before, after = inner.get(point, (point, point))
            straight = (
                before[0] == point[0] == after[0] or before[1] == point[1] == after[1]
            )
            fresh = point not in self.ends and point not in self.through
            if point in inner and straight and fresh:
                self.through[point] = (name, "h" if before[1] == point[1] else "v")
            else:
                self.through.pop(point, None)
                self.ends[point] = name

    def _release(self, name: str, path: list[Point]) -> None:
        """Frees what `path` claimed for the net `name`, which is labeled instead."""
        for first, second in itertools.pairwise(path):
            if self.steps.get(frozenset((first, second))) == name:
                del self.steps[frozenset((first, second))]
        for point in path:
            if self.ends.get(point) == name:
                del self.ends[point]
            if point in self.through and self.through[point][0] == name:
                del self.through[point]


def _route(nets: Mapping[str, Net], placed: tuple[Placed, ...]) -> None:
    """Routes every signal net with two pins or more, the smallest spans first."""
    router = _Router(nets.values(), placed)
    routable = [
        net
        for net in nets.values()
        if net.role is ir.Role.SIGNAL and len(set(net.pins)) > 1
    ]
    for net in sorted(routable, key=_span):
        router.route(net)


def _span(net: Net) -> int:
    xs, ys = [x for x, _ in net.pins], [y for _, y in net.pins]
    return max(xs) - min(xs) + max(ys) - min(ys)


def _path(previous: Mapping[_State, _State], state: _State) -> list[Point]:
    points = [state[0]]
    while state in previous:
        state = previous[state]
        points.append(state[0])
    points.reverse()
    return points


def _segments(paths: list[list[Point]]) -> list[Segment]:
    """The straight pieces of `paths`, each from bend to bend."""
    segments: list[Segment] = []
    for path in paths:
        start = path[0]
        for before, point, after in zip(path, path[1:], path[2:], strict=False):
            if (before[0] == point[0]) != (point[0] == after[0]):
                segments.append((start, point))
                start = point
        if len(path) > 1:
            segments.append((start, path[-1]))
    return segments


def _junctions(paths: list[list[Point]], pins: set[Point]) -> list[Point]:
    """Where three wires of a net meet, or two meet at one of its pins."""
    degree: dict[Point, int] = {}
    for path in paths:
        for first, second in itertools.pairwise(path):
            degree[first] = degree.get(first, 0) + 1
            degree[second] = degree.get(second, 0) + 1
    return sorted(
        point
        for point, count in degree.items()
        if count >= (_JOINED_AT_PIN if point in pins else _JOINED)
    )
