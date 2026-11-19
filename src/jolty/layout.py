# Copyright 2026 Maxence Faldor
"""Schematic layout for component instances, port flags, and routed signal nets.

Layout follows standard schematic conventions with signal flow from left to right.
Each instance is placed in a column according to its graph distance from independent
sources, or from the first signal port when no sources exist. Two-terminal devices
between two signal nets lie horizontally across their column with the earlier net on
the left. Devices between a signal net and ground hang below the net, while devices
between a signal net and a supply stand above the net. Ground and supply nets draw no
wires and instead display a rail symbol at each pin.

All other nets are routed pin by pin on an integer grid using A* search that penalizes
each step, bend, and crossing. Wires from distinct nets may cross at right angles but
never run collinearly or share an endpoint, keeping every wire isolated to a single
net. Any net the router cannot connect receives a net label at each pin.

Coordinates use grid units equal to one pin pitch, with the y-axis increasing upward.

Examples:
    >>> from jolty import Builder, Role, circuit, component, devices
    >>> from jolty.units import V, kohm, nF
    >>> @component
    ... def rc(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(1 * V), name="V1")
    ...     resistor = c.place(devices.resistor(1 * kohm), name="R1")
    ...     capacitor = c.place(devices.capacitor(1 * nF), name="C1")
    ...     c.connect(source.positive, resistor.a, name="in")
    ...     c.connect(resistor.b, capacitor.a, name="out")
    ...     c.connect(capacitor.b, source.negative, ground)
    >>> drawn = layout(circuit(rc()), "rc")
    >>> [(item.name, item.column) for item in drawn.instances]
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
"""Grid coordinate pair ``(x, y)`` in pin-pitch units."""

type Segment = tuple[Point, Point]
"""Straight horizontal or vertical wire segment between two grid points."""

_COLUMN_GAP: Final = 5
"""Grid units between the widest symbols of neighboring columns for wire routing."""

_ROW_GAP: Final = 4
"""Grid units between adjacent rows for rail symbols and horizontal wire routing."""

_LANE_GAP: Final = 2
"""Grid units between the widest symbols of neighboring lanes within a column."""

_FLAG_GAP: Final = 4
"""Grid units between the outermost pins and port flags at sheet edges."""

_MARGIN: Final = 4
"""Grid units reserved around the schematic boundary for wire routing."""

_STEP: Final = 1
_BEND: Final = 3
_CROSSING: Final = 4


@dataclass(frozen=True, slots=True)
class Shape:
    """Symbol geometry in local coordinates, specifying pin positions and body bounds.

    Local coordinates represent the symbol upright with its first terminal at the top.
    """

    pins: Mapping[str, Point]
    body: tuple[Point, Point]
    """Lower-left and upper-right corners of the symbol body blocked from routing."""


_TWO_TERMINAL: Final = {
    "resistor": ("a", "b"),
    "capacitor": ("a", "b"),
    "inductor": ("a", "b"),
    "diode": ("anode", "cathode"),
    "voltage_source": ("positive", "negative"),
    "current_source": ("positive", "negative"),
}
"""Terminal pairs for each two-terminal device model, with the first terminal on top."""

_CONTROLLED: Final = frozenset(
    {"voltage_controlled_voltage_source", "voltage_controlled_current_source"}
)

_SOURCES: Final = frozenset({"voltage_source", "current_source"})


def _shape(instance: ir.Instance, circuit: ir.Circuit) -> Shape:
    """Builds the local symbol `Shape` representing `instance`."""
    match instance.of:
        case ir.DeviceOf(model=model) if model in _TWO_TERMINAL:
            top, bottom = _TWO_TERMINAL[model]
            return Shape({top: (0, 2), bottom: (0, -2)}, ((-1, -1), (1, 1)))
        case ir.DeviceOf(model=model) if model in _CONTROLLED:
            pins = {
                "positive": (0, 2),
                "negative": (0, -2),
                "control_positive": (-4, 1),
                "control_negative": (-4, -1),
            }
            return Shape(pins, ((-3, -1), (1, 1)))
        case ir.ComponentOf(component=name) if name in circuit.components:
            ports = [(port.name, port.role) for port in circuit.components[name].ports]
            return _box(ports, name, instance.name)
        case ir.DeviceOf(model=name) | ir.ComponentOf(component=name):
            signals = [(terminal, ir.Role.SIGNAL) for terminal in instance.terminals]
            return _box(signals, name, instance.name)


_CHARACTER: Final = 0.2
"""Estimated character width for box labels, in grid units."""


def _box(ports: list[tuple[str, ir.Role]], name: str, instance: str) -> Shape:
    """Builds a box symbol with side signals, top supplies, and bottom grounds.

    Input ports go on the left, output ports on the right, and remaining signal ports
    balance both sides starting on the left. The box width accommodates the component
    title and arguments between side-port labels, as well as the instance name to the
    left of the first supply pin.
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
    beside = len(supplies) + math.ceil(_CHARACTER * len(instance)) if supplies else 0
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
    """Splits a component `name` into its base function name and formatted arguments.

    Args:
        name: Parameterized or bare component name string.

    Returns:
        Pair of base component title and comma-space-separated argument text.

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
    """Symbol rotation and vertical flip relative to the upright reference frame."""

    turns: int = 0
    """Counterclockwise quarter turns; 1 places the first terminal on the left."""
    flipped: bool = False
    """Flag indicating a vertical flip applied before quarter-turn rotation."""

    def apply[N: (int, float)](self, point: tuple[N, N]) -> tuple[N, N]:
        """Transforms a local symbol `point` by flip and rotation.

        Args:
            point: Two-dimensional local coordinate pair `(x, y)`.

        Returns:
            Transformed coordinate pair after optional vertical flip and rotation.
        """
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
    """Component or device instance placed on the schematic grid."""

    name: str
    instance: ir.Instance
    shape: Shape
    origin: Point
    orientation: Orientation
    column: int

    def at[N: (int, float)](self, point: tuple[N, N]) -> tuple[N, N]:
        """Transforms a local symbol `point` into schematic sheet coordinates.

        Args:
            point: Local symbol coordinate pair `(x, y)`.

        Returns:
            Sheet coordinate pair after orientation and origin translation.
        """
        x, y = self.orientation.apply(point)
        return self.origin[0] + x, self.origin[1] + y

    def pin(self, terminal: str) -> Point:
        """Returns the schematic sheet coordinates of the `terminal` pin.

        Args:
            terminal: Terminal or port name defined on `shape.pins`.

        Returns:
            Grid coordinate pair of `terminal` on the schematic sheet.
        """
        return self.at(self.shape.pins[terminal])

    def body(self) -> tuple[Point, Point]:
        """Returns the lower-left and upper-right body corners on the sheet.

        Returns:
            Pair of lower-left and upper-right grid corners bounding the symbol body.
        """
        corners = [self.at(corner) for corner in self.shape.body]
        xs, ys = [x for x, _ in corners], [y for _, y in corners]
        return (min(xs), min(ys)), (max(xs), max(ys))


@dataclass(frozen=True, slots=True)
class Flag:
    """Signal port flag placed at the left or right edge of the schematic sheet."""

    name: str
    point: Point
    side: int
    """Horizontal edge indicator: -1 for left (input) or 1 for right (output)."""


@dataclass(slots=True)
class Net:
    """Routed or labeled schematic net with pins, wire segments, and junctions."""

    name: str
    role: ir.Role
    pins: list[Point]
    segments: list[Segment] = field(default_factory=list)
    junctions: list[Point] = field(default_factory=list)
    labeled: bool = False
    """Flag indicating that routing failed and each pin displays a net label."""


@dataclass(frozen=True, slots=True)
class Layout:
    """Completed schematic layout containing placed instances, port flags, and nets."""

    component: str
    instances: tuple[Placed, ...]
    flags: tuple[Flag, ...]
    nets: tuple[Net, ...]

    def bounds(self) -> tuple[Point, Point]:
        """Returns lower-left and upper-right corners bounding all drawn elements.

        Returns:
            Axis-aligned lower-left and upper-right grid corners enclosing the layout.
        """
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
    """Computes the schematic `Layout` for component `name` in `circuit`.

    Args:
        circuit: Circuit IR containing the definition of `name`.
        name: Component key in `circuit.components` to lay out and route.

    Returns:
        Completed schematic `Layout` with placed instances, port flags, and nets.

    Raises:
        KeyError: When `circuit` contains no component named `name`.
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
    """Determines the electrical role of each net in `component` for layout styling.

    Declared component ports assign their net roles, with inputs and outputs treated as
    signal nets. Any remaining net is classified as ground if connected to a ground
    port, otherwise supply if connected to a supply port, and signal otherwise.
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
    """Reduces a set of port roles to the net layout role: ground, supply, or signal.

    Signal direction (input or output) is normalized to a generic signal role.
    """
    for role in (ir.Role.GROUND, ir.Role.SUPPLY):
        if role in roles:
            return role
    return ir.Role.SIGNAL


def _signal_nets(instance: ir.Instance, roles: Mapping[str, ir.Role]) -> set[str]:
    return {net for net in instance.terminals.values() if roles[net] is ir.Role.SIGNAL}


def _columns(component: ir.Component, roles: Mapping[str, ir.Role]) -> dict[str, int]:
    """Assigns each instance a left-to-right column index by signal-net graph distance.

    Breadth-first traversal starts from independent sources, falling back to instances
    on input ports, then instances on the first signal port, and finally the first
    instance. Disconnected subgraphs restart traversal from the next unreached instance.
    """
    instances = component.instances
    on_net: dict[str, list[str]] = {}
    for instance in instances:
        for net in _signal_nets(instance, roles):
            on_net.setdefault(net, []).append(instance.name)
    by_id = {instance.name: instance for instance in instances}
    sources = [
        instance.name
        for instance in instances
        if isinstance(instance.of, ir.DeviceOf) and instance.of.model in _SOURCES
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
    starts = sources or inputs or [instance.name for instance in instances[:1]]
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
            instance.name for instance in instances if instance.name not in columns
        ]
        starts = unreached[:1]
    return columns


def _rows(
    instances: tuple[ir.Instance, ...],
    roles: Mapping[str, ir.Role],
    columns: Mapping[str, int],
) -> dict[str, int]:
    """Orders instances into vertical rows within each column to minimize wire length.

    Instances start in placement order and undergo four barycenter sweeps across
    columns. Each column sorts instances by the mean row of connected neighbors across
    shared signal nets, weighting multiple shared nets proportionally. Instances with no
    shared signal nets, such as supply sources, sink to the bottom of their column.
    Running row sums per net keep each sweep linear in pin count aside from sorting.
    """
    nets = {instance.name: _signal_nets(instance, roles) for instance in instances}
    stacks: dict[int, list[str]] = {}
    for instance in instances:
        stacks.setdefault(columns[instance.name], []).append(instance.name)
    rows = {
        name: index for stack in stacks.values() for index, name in enumerate(stack)
    }
    sums: dict[str, int] = {}
    counts: dict[str, int] = {}
    for name, shared in nets.items():
        for net in shared:
            sums[net] = sums.get(net, 0) + rows[name]
            counts[net] = counts.get(net, 0) + 1

    def barycenter(name: str) -> tuple[bool, float]:
        """Computes the sort key pairing isolation status with neighbors' mean row."""
        others = sum(counts[net] - 1 for net in nets[name])
        if not others:
            return True, rows[name]
        return False, sum(sums[net] - rows[name] for net in nets[name]) / others

    for _ in range(4):
        for column in sorted(stacks):
            stack = stacks[column]
            stack.sort(key=barycenter)
            for index, name in enumerate(stack):
                for net in nets[name]:
                    sums[net] += index - rows[name]
                rows[name] = index
    return rows


def _place(
    circuit: ir.Circuit,
    instances: tuple[ir.Instance, ...],
    roles: Mapping[str, ir.Role],
    columns: Mapping[str, int],
    rows: Mapping[str, int],
) -> tuple[Placed, ...]:
    """Places every instance with its column, row, lane, origin, and orientation.

    Within a column, consecutive devices hanging from the same signal net to a rail
    share a row in adjacent lanes so the signal net runs horizontally across them. Every
    instance aligns to its row baseline across columns, and row spacing accommodates the
    tallest symbols above and below each baseline.
    """
    earliest: dict[str, int] = {}
    for instance in instances:
        for net in instance.terminals.values():
            column = columns[instance.name]
            earliest[net] = min(earliest.get(net, column), column)
    by_name = {instance.name: instance for instance in instances}
    shapes = {instance.name: _shape(instance, circuit) for instance in instances}
    orientations = {
        instance.name: _orientation(instance, roles, earliest) for instance in instances
    }
    extents = {name: _extent(shapes[name], orientations[name]) for name in by_name}
    anchors = {
        name: _anchor(by_name[name], shapes[name], orientations[name], roles)
        for name in by_name
    }
    slots: dict[int, list[list[str]]] = {}
    for name in sorted(by_name, key=lambda name: (columns[name], rows[name])):
        stack = slots.setdefault(columns[name], [])
        hanging = _hanging(by_name[name], roles)
        if (
            stack
            and hanging is not None
            and _hanging(by_name[stack[-1][-1]], roles) == hanging
        ):
            stack[-1].append(name)
        else:
            stack.append([name])
    lines = _lines(slots.values(), extents, anchors)
    placed = []
    x = 0
    for column in sorted(slots):
        stack = slots[column]
        width = max(_width(extents[name]) for slot in stack for name in slot)
        pitch = width + _LANE_GAP
        for row, slot in enumerate(stack):
            for lane, name in enumerate(slot):
                (left, _), _ = extents[name]
                origin = (x - left + lane * pitch, lines[row] - anchors[name])
                placed.append(
                    Placed(
                        name,
                        by_name[name],
                        shapes[name],
                        origin,
                        orientations[name],
                        column,
                    )
                )
        x += (max(map(len, stack)) - 1) * pitch + width + _COLUMN_GAP
    order = {instance.name: index for index, instance in enumerate(instances)}
    return tuple(sorted(placed, key=lambda item: order[item.name]))


def _lines(
    stacks: Iterable[list[list[str]]],
    extents: Mapping[str, tuple[Point, Point]],
    anchors: Mapping[str, int],
) -> list[int]:
    """Computes vertical row baseline coordinates from symbol extents and anchors."""
    above: dict[int, int] = {}
    below: dict[int, int] = {}
    for stack in stacks:
        for row, slot in enumerate(stack):
            for name in slot:
                (_, bottom), (_, top) = extents[name]
                above[row] = max(above.get(row, 0), top - anchors[name])
                below[row] = max(below.get(row, 0), anchors[name] - bottom)
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
    """Computes the local y-offset where an oriented instance aligns to its baseline.

    A rail-hanging device aligns at its signal-net pin, any other upright device aligns
    at its top extent, and all other instances align at their vertical center.
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
    """Returns the signal net when a two-terminal device connects signal to rail."""
    if (
        not isinstance(instance.of, ir.DeviceOf)
        or instance.of.model not in _TWO_TERMINAL
    ):
        return None
    nets = [instance.terminals[name] for name in _TWO_TERMINAL[instance.of.model]]
    signals = [net for net in nets if roles[net] is ir.Role.SIGNAL]
    rails = [net for net in nets if roles[net] is not ir.Role.SIGNAL]
    return signals[0] if len(signals) == 1 and len(rails) == 1 else None


def _width(extent: tuple[Point, Point]) -> int:
    (left, _), (right, _) = extent
    return right - left


def _extent(symbol: Shape, orientation: Orientation) -> tuple[Point, Point]:
    """Computes oriented lower-left and upper-right bounding corners of `symbol`."""
    points = [
        orientation.apply(point) for point in [*symbol.pins.values(), *symbol.body]
    ]
    xs, ys = [x for x, _ in points], [y for _, y in points]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _orientation(
    instance: ir.Instance, roles: Mapping[str, ir.Role], earliest: Mapping[str, int]
) -> Orientation:
    """Determines the schematic orientation for `instance` from its terminal roles.

    Ground terminals stay at the bottom and supply terminals stay at the top:
    two-terminal devices rotate 180 degrees, whereas controlled sources flip vertically
    to keep control leads on the left. Independent sources remain upright, and all other
    two-terminal devices lie horizontally with the earlier signal net on the left.
    """
    if not isinstance(instance.of, ir.DeviceOf):
        return Orientation()
    model = instance.of.model
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
    """Returns whether top and bottom terminal roles require inverting the symbol."""
    return top is ir.Role.GROUND or bottom is ir.Role.SUPPLY


def _flags(
    component: ir.Component, placed: tuple[Placed, ...], roles: Mapping[str, ir.Role]
) -> tuple[Flag, ...]:
    """Builds edge port flags for signal ports, placing inputs left and outputs right.

    Explicit input and output roles determine side placement directly, while undirected
    signal ports go on the left unless their net extends past the horizontal midpoint.
    Each flag aligns vertically with its net's outermost pin on that side, stepping
    downward to avoid overlap with earlier flags.
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
"""Last grid step direction during routing search, or `None` at search start."""

type _State = tuple[Point, _Heading]

type _Edge = tuple[Point, Point]
"""Undirected grid step between two adjacent points, stored with ordered endpoints."""


def _edge(first: Point, second: Point) -> _Edge:
    """Returns the canonical ordered pair for an undirected grid step."""
    return (first, second) if first <= second else (second, first)


_DIRECTIONS: Final = ((1, 0), (-1, 0), (0, 1), (0, -1))

_JOINED: Final = 3
"""Minimum incident wire count that turns a non-pin grid point into a junction."""

_JOINED_AT_PIN: Final = 2
"""Minimum incident wire count that turns a pin grid point into a junction."""


class _Router:
    """Grid-based A* wire router that keeps distinct nets electrically isolated.

    The routing grid tracks blocked symbol bodies, net pin locations, and claimed wire
    segments. Straight wire interiors allow orthogonal crossings by other nets, whereas
    bends, junctions, and endpoints remain exclusive to the owning net.
    """

    def __init__(self, nets: Iterable[Net], placed: Iterable[Placed]) -> None:
        """Prepares a routing grid around `nets` pins and `placed` symbol bodies."""
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
        self.steps: dict[_Edge, str] = {}
        self.through: dict[Point, tuple[str, str]] = {}
        self.crossed: dict[Point, tuple[str, str]] = {}
        self.ends: dict[Point, str] = {}

    def route(self, net: Net) -> None:
        """Connects all pins of `net` on the grid, or labels `net` if unroutable."""
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
        """Finds the lowest-cost grid path for `name` from any point in `tree` to `pin`.

        Multi-source A* starts simultaneously from the entire connected `tree`, so the
        Manhattan heuristic to `pin` stays constant-time regardless of tree size.
        """

        def estimate(point: Point) -> int:
            return abs(point[0] - pin[0]) + abs(point[1] - pin[1])

        counter = itertools.count()
        starts: list[_State] = [(point, None) for point in sorted(tree)]
        frontier = [(estimate(state[0]), next(counter), 0, state) for state in starts]
        heapq.heapify(frontier)
        costs: dict[_State, int] = dict.fromkeys(starts, 0)
        previous: dict[_State, _State] = {}
        while frontier:
            _, _, popped_cost, state = heapq.heappop(frontier)
            if popped_cost > costs[state]:
                continue
            point, heading = state
            if point == pin:
                return _path(previous, state)
            for direction in self._directions(name, point, heading):
                following = (point[0] + direction[0], point[1] + direction[1])
                if not self._open(name, point, following, direction, pin):
                    continue
                cost = popped_cost + self._step_cost(
                    name, heading, direction, following
                )
                successor = (following, direction)
                if cost < costs.get(successor, cost + 1):
                    costs[successor] = cost
                    previous[successor] = state
                    priority = cost + estimate(following)
                    heapq.heappush(frontier, (priority, next(counter), cost, successor))
        return None

    def _step_cost(
        self,
        name: str,
        heading: _Heading,
        direction: tuple[int, int],
        following: Point,
    ) -> int:
        """Returns the incremental cost for stepping onto `following`."""
        bend = _BEND if heading is not None and direction != heading else 0
        crossing = _CROSSING if self._crosses(name, following) else 0
        return _STEP + bend + crossing

    def _directions(
        self, name: str, point: Point, heading: _Heading
    ) -> Iterable[tuple[int, int]]:
        """Returns step directions from `point`, continuing straight at crossings."""
        if heading is not None and self._crosses(name, point):
            return (heading,)
        return _DIRECTIONS

    def _crosses(self, name: str, point: Point) -> bool:
        """Returns whether `point` lies on a straight wire of another net."""
        return point in self.through and self.through[point][0] != name

    def _open(
        self,
        name: str,
        point: Point,
        following: Point,
        direction: tuple[int, int],
        pin: Point,
    ) -> bool:
        """Returns whether net `name` may step from `point` onto `following`."""
        x, y = following
        if not (self.low[0] <= x <= self.high[0] and self.low[1] <= y <= self.high[1]):
            return False
        return (
            (following not in self.blocked or following == pin)
            and self.steps.get(_edge(point, following), name) == name
            and self.pins.get(following, name) == name
            and self.ends.get(following, name) == name
            and self._may_cross(name, following, direction, pin)
        )

    def _may_cross(
        self, name: str, point: Point, direction: tuple[int, int], pin: Point
    ) -> bool:
        """Returns whether stepping in `direction` onto `point` avoids wire overlap."""
        if not self._crosses(name, point):
            return True
        way = "h" if direction[1] == 0 else "v"
        return self.through[point][1] != way and point != pin

    def _claim(self, name: str, path: list[Point]) -> None:
        """Claims grid edges and points along `path` for net `name`.

        Fresh straight interior points remain crossable by orthogonal nets, while bends,
        branch points, and endpoints are marked exclusive to `name`.
        """
        for first, second in itertools.pairwise(path):
            self.steps[_edge(first, second)] = name
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
                displaced = self.through.pop(point, None)
                if displaced is not None and displaced[0] != name:
                    self.crossed[point] = displaced
                self.ends[point] = name

    def _release(self, name: str, path: list[Point]) -> None:
        """Releases grid edges and points claimed by `path` when `name` uses labels."""
        for first, second in itertools.pairwise(path):
            edge = _edge(first, second)
            if self.steps.get(edge) == name:
                del self.steps[edge]
        for point in path:
            if self.ends.get(point) == name:
                del self.ends[point]
                restored = self.crossed.pop(point, None)
                if restored is not None:
                    self.through[point] = restored
            elif point in self.through and self.through[point][0] == name:
                del self.through[point]


def _route(nets: Mapping[str, Net], placed: tuple[Placed, ...]) -> None:
    """Routes every multi-pin signal net in order of increasing Manhattan span."""
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
    """Merges collinear steps of `paths` into straight bend-to-bend wire segments."""
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
    """Returns sorted grid points where three wires meet, or two meet at a pin."""
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
