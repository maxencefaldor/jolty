# Copyright 2026 Maxence Faldor
"""Schematics: a component drawn the way engineers draw circuits.

[`jolty.layout`][jolty.layout] places the instances and routes the wires; this module
draws them with the standard symbols: zigzag resistors, plate capacitors, coiled
inductors, triangle diodes, circled sources, diamond controlled sources, and boxes for
components, with their ports as pins. Each instance carries its id and its value, a
ground symbol or a supply flag marks each pin on a rail, and a dot marks where wires
join.

A schematic can carry facts: the voltage of each net at an operating point, and the
instances that diagnostics name, drawn in red with an exclamation point.

Install the optional extra with ``pip install 'jolty[plots]'`` to use this module.

Examples:
    >>> from jolty import Builder, Role, Simulation, circuit, component, devices
    >>> from jolty.plan import OperatingPoint
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     upper = c.place(devices.resistor(1 * kohm), name="R1")
    ...     lower = c.place(devices.resistor(1 * kohm), name="R2")
    ...     c.connect(source.positive, upper.a, name="in")
    ...     c.connect(upper.b, lower.a, name="out")
    ...     c.connect(lower.b, source.negative, ground)
    >>> drawn = circuit(divider())
    >>> point = Simulation(drawn).run(OperatingPoint())
    >>> figure = draw(drawn, point=point)
    >>> "2.5 V" in [text.get_text() for text in figure.axes[0].texts]
    True
"""

import math
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final

try:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Polygon, Rectangle
except ModuleNotFoundError as error:
    root_module = (error.name or "").split(".", 1)[0]
    if root_module != "matplotlib":
        raise
    message = (
        "jolty.schematic needs the optional `matplotlib` dependency; "
        "install it with `pip install 'jolty[plots]'`"
    )
    raise ImportError(message) from error

from jolty import ir
from jolty.errors import JoltyError
from jolty.layout import Net, Placed, Point, layout, split_name
from jolty.plots import INK as _INK
from jolty.plots import MUTED as _MUTED
from jolty.plots import SURFACE as _SURFACE
from jolty.plots import save
from jolty.results import OperatingPointResult
from jolty.units import format_si

__all__ = ["draw", "save"]

_WIRE: Final = "#1f3a8a"
_MARKED: Final = "#c62828"
_VOLTAGE: Final = "#b45309"
_MINUS: Final = "\N{MINUS SIGN}"
_TIMES: Final = "\N{MULTIPLICATION SIGN}"
_SCALE: Final = 0.32
"""Inches per grid unit."""
_LINE: Final = 1.1
"""The width of every stroke, in points."""

type _Local = tuple[float, float]
"""A point in a symbol's own frame, upright with its first terminal at the top."""


def draw(
    circuit: ir.Circuit,
    path: str = "",
    *,
    point: OperatingPointResult | None = None,
    marked: Collection[str] = (),
) -> Figure:
    """The schematic of the component the instance at `path` places, or of the top.

    `point` gives the voltage to write on each net, and `marked` the paths of instances
    to draw in red, such as those diagnostics name: an instance that holds a marked one
    is marked too.

    Raises:
        JoltyError: When no component instance is at `path`.
    """
    scope = _scope(circuit, path)
    drawn = layout(circuit, scope.component)
    nets = [net.name for net in drawn.nets]
    voltages = scope.voltages(point, nets) if point is not None else {}
    marks = scope.marked(marked)
    (left, bottom), (right, top) = drawn.bounds()
    width, height = right - left + 10, top - bottom + 7
    figure = Figure(
        figsize=(max(width * _SCALE, 3.0), max(height * _SCALE, 2.0)),
        dpi=150,
        facecolor=_SURFACE,
    )
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(left - 5, right + 5)
    axes.set_ylim(bottom - 3, top + 4)
    axes.set_aspect("equal")
    axes.set_axis_off()
    flagged = {flag.point for flag in drawn.flags}
    for net in drawn.nets:
        _draw_net(axes, net, voltages, flagged)
    for placed in drawn.instances:
        _draw_instance(axes, placed, marked=placed.name in marks)
    for flag in drawn.flags:
        _draw_flag(axes, flag.name, flag.point, flag.side)
    return figure


@dataclass(frozen=True, slots=True)
class _Scope:
    """A component as an instance path places it in a circuit."""

    component: str
    prefix: str = ""
    """The instance's path and a slash, or nothing at the top."""
    ports: Mapping[str, str] = field(default_factory=dict)
    """The path in the circuit of each port's net, by port."""

    def path(self, net: str) -> str:
        """The path in the circuit of the component's `net`, which names its voltage."""
        return self.ports.get(net, self.prefix + net)

    def voltages(
        self, point: OperatingPointResult, nets: Iterable[str]
    ) -> dict[str, float]:
        """The voltage `point` holds for each of `nets` of the component, by net.

        A net `point` holds no voltage for, such as the ground, is left out.
        """
        solved = {
            unknown.name: float(value)
            for unknown, value in zip(point.unknowns, point.values, strict=True)
            if unknown.nature == "voltage"
        }
        return {net: solved[self.path(net)] for net in nets if self.path(net) in solved}

    def marked(self, paths: Collection[str]) -> set[str]:
        """The ids of the component's instances that are or hold one of `paths`."""
        return {
            path.removeprefix(self.prefix).partition("/")[0]
            for path in paths
            if path.startswith(self.prefix)
        }


def _scope(circuit: ir.Circuit, path: str) -> _Scope:
    """The scope of the component the instance at `path` places, or of the top.

    Raises:
        JoltyError: When no component instance is at `path`.
    """
    scope = _Scope(circuit.top)
    for id in path.split("/") if path else ():
        component = circuit.components.get(scope.component)
        found = component.instance(id) if component is not None else None
        reached = scope.prefix + id
        if found is None:
            message = f"no instance is at {reached}"
            raise JoltyError(message)
        if not isinstance(found.of, ir.ComponentOf):
            message = f"{reached} is a device, which has no schematic"
            raise JoltyError(message)
        ports = {port: scope.path(net) for port, net in found.terminals.items()}
        scope = _Scope(found.of.component, f"{reached}/", ports)
    if scope.component not in circuit.components:
        message = f"the circuit defines no component {scope.component}"
        raise JoltyError(message)
    return scope


def _draw_net(
    axes: Axes, net: Net, voltages: Mapping[str, float], flagged: Collection[Point]
) -> None:
    """A net's wires and junctions, its rail symbols, and its voltage.

    A signal net without wires carries its name at each pin but the `flagged` ones,
    whose flags already name it.
    """
    for (x1, y1), (x2, y2) in net.segments:
        axes.plot([x1, x2], [y1, y2], color=_WIRE, linewidth=_LINE)
    for point in net.junctions:
        axes.add_patch(Circle(point, 0.22, color=_WIRE, zorder=3))
    match net.role:
        case ir.Role.GROUND:
            for pin in net.pins:
                _draw_ground(axes, pin)
        case ir.Role.SUPPLY:
            for pin in net.pins:
                _draw_supply(axes, pin, net.name)
        case ir.Role.SIGNAL if net.labeled or len(set(net.pins)) == 1:
            for x, y in set(net.pins) - set(flagged):
                _write(axes, (x + 0.3, y + 0.3), net.name, color=_WIRE)
    if net.name in voltages and net.role is not ir.Role.GROUND and net.pins:
        _draw_voltage(axes, net, voltages[net.name])


def _draw_voltage(axes: Axes, net: Net, voltage: float) -> None:
    """A net's voltage above the middle of its longest wire, or below its first pin."""
    text = format_si(voltage, "V")
    if not net.segments:
        x, y = net.pins[0]
        _write(axes, (x + 0.3, y - 0.3), text, color=_VOLTAGE, bold=True, va="top")
        return
    (x1, y1), (x2, y2) = max(
        net.segments,
        key=lambda ends: abs(ends[1][0] - ends[0][0]) + abs(ends[1][1] - ends[0][1]),
    )
    middle = ((x1 + x2) / 2, (y1 + y2) / 2)
    if y1 == y2:
        above = (middle[0], middle[1] + 0.3)
        _write(axes, above, text, color=_VOLTAGE, bold=True, ha="center", va="bottom")
    else:
        beside = (middle[0] + 0.3, middle[1])
        _write(axes, beside, text, color=_VOLTAGE, bold=True, va="center")


def _write(  # noqa: PLR0913 - the styles of a text, each with a default
    axes: Axes,
    point: _Local,
    text: str,
    *,
    color: str = _INK,
    size: float = 7,
    bold: bool = False,
    ha: str = "left",
    va: str = "baseline",
) -> None:
    """Writes `text` at `point`, aligned by `ha` and `va`."""
    axes.text(
        point[0],
        point[1],
        text,
        color=color,
        fontsize=size,
        fontweight="bold" if bold else "normal",
        ha=ha,
        va=va,
    )


def _draw_ground(axes: Axes, pin: Point) -> None:
    """A ground symbol below `pin`: a lead and three bars, each shorter."""
    x, y = pin
    axes.plot([x, x], [y, y - 0.6], color=_INK, linewidth=_LINE)
    for index, half in enumerate((0.6, 0.4, 0.2)):
        level = y - 0.6 - 0.25 * index
        axes.plot([x - half, x + half], [level, level], color=_INK, linewidth=_LINE)


def _draw_supply(axes: Axes, pin: Point, name: str) -> None:
    """A supply symbol above `pin`: a lead and a bar, the supply's name over it."""
    x, y = pin
    axes.plot([x, x], [y, y + 0.6], color=_INK, linewidth=_LINE)
    axes.plot([x - 0.6, x + 0.6], [y + 0.6, y + 0.6], color=_INK, linewidth=_LINE)
    _write(axes, (x, y + 0.8), name, ha="center", va="bottom")


def _draw_flag(axes: Axes, name: str, point: Point, side: int) -> None:
    """A port flag pointing the way signal flows, its name beyond it.

    An input's flag, on the left, points into its net; an output's, on the right, points
    away from it.
    """
    x, y = point
    length, tip = 2.2, 0.5
    start = x - length if side < 0 else x
    outline = [(0, 0.45), (length - tip, 0.45), (length, 0), (length - tip, -0.45)]
    corners = [(start + dx, y + dy) for dx, dy in [*outline, (0, -0.45)]]
    axes.add_patch(
        Polygon(corners, closed=True, fill=False, edgecolor=_INK, linewidth=_LINE)
    )
    beyond = (x + side * (length + 0.3), y)
    _write(axes, beyond, name, ha="left" if side > 0 else "right", va="center")


@dataclass(frozen=True, slots=True)
class _Pen:
    """Draws in a placed symbol's own frame, in the symbol's color."""

    axes: Axes
    placed: Placed
    color: str

    def line(self, *points: _Local) -> None:
        """A stroke through `points`."""
        xs, ys = zip(*(self.placed.at(point) for point in points), strict=True)
        self.axes.plot(
            xs, ys, color=self.color, linewidth=_LINE, solid_joinstyle="round"
        )

    def polygon(self, *points: _Local) -> None:
        """A closed outline through `points`."""
        corners = [self.placed.at(point) for point in points]
        self.axes.add_patch(
            Polygon(
                corners, closed=True, fill=False, edgecolor=self.color, linewidth=_LINE
            )
        )

    def circle(self, center: _Local, radius: float) -> None:
        """A circle's outline."""
        self.axes.add_patch(
            Circle(
                self.placed.at(center),
                radius,
                fill=False,
                edgecolor=self.color,
                linewidth=_LINE,
            )
        )

    def sign(self, point: _Local, text: str) -> None:
        """A small sign, centered at `point`."""
        at = self.placed.at(point)
        _write(self.axes, at, text, color=self.color, size=6, ha="center", va="center")


def _resistor(pen: _Pen) -> None:
    corners = [
        (0.45 if step % 2 == 0 else -0.45, 1 - (2 * step + 1) / 6) for step in range(6)
    ]
    pen.line((0, 2), (0, 1), *corners, (0, -1), (0, -2))


def _capacitor(pen: _Pen) -> None:
    pen.line((0, 2), (0, 0.25))
    pen.line((-0.8, 0.25), (0.8, 0.25))
    pen.line((-0.8, -0.25), (0.8, -0.25))
    pen.line((0, -0.25), (0, -2))


def _inductor(pen: _Pen) -> None:
    coil = [
        (
            0.28 * math.sin(math.pi * step / 12),
            0.75 - 0.5 * hump + 0.25 * math.cos(math.pi * step / 12),
        )
        for hump in range(4)
        for step in range(13)
    ]
    pen.line((0, 2), (0, 1), *coil, (0, -1), (0, -2))


def _diode(pen: _Pen) -> None:
    pen.line((0, 2), (0, 0.5))
    pen.polygon((-0.6, 0.5), (0.6, 0.5), (0, -0.5))
    pen.line((-0.6, -0.5), (0.6, -0.5))
    pen.line((0, -0.5), (0, -2))


def _signs(pen: _Pen, height: float) -> None:
    """A plus above and a minus below the symbol's center."""
    pen.line((-0.2, height), (0.2, height))
    pen.line((0, height - 0.2), (0, height + 0.2))
    pen.line((-0.2, -height), (0.2, -height))


def _arrow(pen: _Pen, length: float) -> None:
    """An arrow down through the symbol's center: the current's direction."""
    pen.line((0, length), (0, -length))
    pen.line((-0.22, 0.28 - length), (0, -length), (0.22, 0.28 - length))


def _voltage_source(pen: _Pen) -> None:
    pen.line((0, 2), (0, 1))
    pen.circle((0, 0), 1.0)
    _signs(pen, 0.45)
    pen.line((0, -1), (0, -2))


def _current_source(pen: _Pen) -> None:
    pen.line((0, 2), (0, 1))
    pen.circle((0, 0), 1.0)
    _arrow(pen, 0.55)
    pen.line((0, -1), (0, -2))


def _controlled(pen: _Pen, inside: Callable[[_Pen], None]) -> None:
    """A diamond with `inside` drawn in it, and control leads on its left."""
    pen.line((0, 2), (0, 1))
    pen.polygon((0, 1), (1, 0), (0, -1), (-1, 0))
    inside(pen)
    pen.line((0, -1), (0, -2))
    pen.line((-4, 1), (-2.4, 1))
    pen.line((-4, -1), (-2.4, -1))
    pen.sign((-2.05, 1), "+")
    pen.sign((-2.05, -1), _MINUS)


def _controlled_voltage(pen: _Pen) -> None:
    _controlled(pen, lambda inner: _signs(inner, 0.4))


def _controlled_current(pen: _Pen) -> None:
    _controlled(pen, lambda inner: _arrow(inner, 0.5))


_SYMBOLS: Final[Mapping[str, Callable[[_Pen], None]]] = {
    "resistor": _resistor,
    "capacitor": _capacitor,
    "inductor": _inductor,
    "diode": _diode,
    "voltage_source": _voltage_source,
    "current_source": _current_source,
    "voltage_controlled_voltage_source": _controlled_voltage,
    "voltage_controlled_current_source": _controlled_current,
}


def _draw_instance(axes: Axes, placed: Placed, *, marked: bool) -> None:
    """An instance's symbol, its id and its value, or a box for a component."""
    color = _MARKED if marked else _INK
    label = f"! {placed.name}" if marked else placed.name
    x, y = placed.origin
    match placed.instance.of:
        case ir.DeviceOf(model=model) if model in _SYMBOLS:
            _SYMBOLS[model](_Pen(axes, placed, color))
        case ir.DeviceOf(model=name) | ir.ComponentOf(component=name):
            _draw_box(axes, placed, name, color)
            (left, _), (_, top) = placed.body()
            _write(
                axes, (left, top + 0.3), label, color=color, bold=marked, va="bottom"
            )
            return
    value = _value(placed.instance)
    if placed.orientation.turns % 2 == 0:
        _write(axes, (x + 1.4, y + 0.2), label, color=color, bold=marked, va="bottom")
        _write(axes, (x + 1.4, y - 0.2), value, color=_MUTED, size=6.5, va="top")
    else:
        above, below = (x, y + 1.3), (x, y - 1.3)
        _write(axes, above, label, color=color, bold=marked, ha="center", va="bottom")
        _write(axes, below, value, color=_MUTED, size=6.5, ha="center", va="top")


def _draw_box(axes: Axes, placed: Placed, name: str, color: str) -> None:
    """A component as a box: its function and its arguments inside, and its ports.

    Each port is a lead from its pin to the box, a port on a side named inside it.
    """
    (left, bottom), (right, top) = placed.body()
    axes.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor=color,
            linewidth=_LINE,
        )
    )
    title, arguments = split_name(name)
    center = ((left + right) / 2, (bottom + top) / 2)
    _write(
        axes,
        (center[0], center[1] + (0.2 if arguments else 0)),
        title,
        color=color,
        bold=True,
        ha="center",
        va="bottom" if arguments else "center",
    )
    if arguments:
        below = (center[0], center[1] - 0.2)
        _write(axes, below, arguments, color=_MUTED, size=6, ha="center", va="top")
    for port in placed.shape.pins:
        x, y = placed.pin(port)
        inner = (min(max(x, left), right), min(max(y, bottom), top))
        axes.plot([x, inner[0]], [y, inner[1]], color=color, linewidth=_LINE)
        if inner[0] in {left, right}:
            inside, align = (0.3, "left") if inner[0] == left else (-0.3, "right")
            _write(
                axes,
                (inner[0] + inside, y),
                port,
                color=_MUTED,
                size=6,
                ha=align,
                va="center",
            )


_QUANTITIES: Final = {
    "resistor": ("resistance", "Ω"),
    "capacitor": ("capacitance", "F"),
    "inductor": ("inductance", "H"),
    "voltage_controlled_current_source": ("transconductance", "S"),
}
"""The parameter that is each model's value, and its unit."""

_SOURCES: Final = {"voltage_source": "V", "current_source": "A"}
"""The unit of each independent source's waveform."""


def _real(value: object) -> float | None:
    """Returns `value` as a float when it is an int or float (excluding bool)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _value(instance: ir.Instance) -> str:
    """The instance's value as a schematic writes it, such as ``4.7 kΩ``."""
    if not isinstance(instance.of, ir.DeviceOf):
        return ""
    model, parameters = instance.of.model, instance.parameters
    if model in _QUANTITIES:
        name, unit = _QUANTITIES[model]
        number = _real(parameters.get(name))
        return format_si(number, unit) if number is not None else ""
    if model in _SOURCES:
        return _waveform(parameters.get("waveform"), _SOURCES[model])
    gain = _real(parameters.get("gain"))
    if model == "voltage_controlled_voltage_source" and gain is not None:
        return f"{_TIMES}{gain:g}"
    return ""


def _sine_label(spec: object, unit: str) -> str:
    """Formats a sine waveform label when amplitude and frequency are numeric."""
    if not isinstance(spec, Mapping):
        return ""
    amplitude = _real(spec.get("amplitude"))
    frequency = _real(spec.get("frequency"))
    if amplitude is None or frequency is None:
        return ""
    return f"{format_si(amplitude, unit)} sine, {format_si(frequency, 'Hz')}"


def _pulsed_label(kind: str, spec: object, unit: str) -> str:
    """Formats a pulse or exponential waveform label from its `pulsed` level."""
    if not isinstance(spec, Mapping):
        return ""
    pulsed = _real(spec.get("pulsed"))
    return f"{kind} to {format_si(pulsed, unit)}" if pulsed is not None else ""


def _waveform(spec: object, unit: str) -> str:
    if not isinstance(spec, Mapping):
        return ""
    if (level := _real(spec.get("dc"))) is not None:
        return format_si(level, unit)
    if "sine" in spec:
        return _sine_label(spec["sine"], unit)
    for kind in ("pulse", "exponential"):
        if kind in spec:
            return _pulsed_label(kind, spec[kind], unit)
    return "piecewise" if "piecewise" in spec else ""
