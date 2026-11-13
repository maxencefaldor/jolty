# Copyright 2026 Maxence Faldor
"""Write circuits as Python: components, their ports, what they place and how it joins.

A component is a function decorated with [`component`][jolty.builder.component]. It
receives a [`Builder`][jolty.builder.Builder], declares ports, places devices and other
components, and connects their terminals. Calling the decorated function with its
parameters returns a [`Component`][jolty.builder.Component], which another component
places, or [`circuit`][jolty.builder.circuit] turns into the circuit format with it at
the top.

Examples:
    >>> from jolty import devices
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), id="V1")
    ...     upper = c.place(devices.resistor(1 * kohm), id="R1")
    ...     lower = c.place(devices.resistor(1 * kohm), id="R2")
    ...     c.connect(source.positive, upper.a)
    ...     c.connect(upper.b, lower.a, name="out")
    ...     c.connect(lower.b, source.negative, ground)
    >>> sorted(circuit(divider()).components)
    ['divider']
"""

import contextlib
import functools
import hashlib
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, override

from pydantic import JsonValue

from jolty import ir
from jolty.devices import Device
from jolty.errors import BuildError
from jolty.ir import Role
from jolty.units import Quantity

_NAME: Final = re.compile(r"[A-Za-z0-9_+\-]+")
"""Ids and net names: no dot, which joins an instance to its terminal, and no slash."""


@dataclass(frozen=True, slots=True)
class Terminal:
    """A terminal of a placed instance."""

    instance: str
    name: str

    @override
    def __str__(self) -> str:
        """The instance and terminal, such as ``R1.a``."""
        return f"{self.instance}.{self.name}"


@dataclass(frozen=True, slots=True)
class Net:
    """A named net of the component being built: a port, or a net named on purpose."""

    name: str

    @override
    def __str__(self) -> str:
        """The net's name."""
        return self.name


type Point = Terminal | Net
"""Anything a connection joins."""


class Placed:
    """A placed device or component, whose terminals are its attributes.

    Examples:
        >>> from jolty import devices
        >>> from jolty.units import ohm
        >>> resistor = Placed("R1", devices.resistor(1 * ohm).terminals)
        >>> resistor.a
        Terminal(instance='R1', name='a')
    """

    __slots__ = ("_id", "_terminals")

    def __init__(self, id: str, terminals: tuple[str, ...]) -> None:
        """Holds the instance's `id` and the names of its `terminals`."""
        self._id = id
        self._terminals = terminals

    @property
    def id(self) -> str:
        """The instance's id."""
        return self._id

    def __getattr__(self, name: str) -> Terminal:
        """The terminal called `name`."""
        if name in self._terminals:
            return Terminal(self._id, name)
        message = (
            f"{self._id} has no terminal {name!r}; it has {', '.join(self._terminals)}"
        )
        raise AttributeError(message)

    def __getitem__(self, name: str) -> Terminal:
        """The terminal called `name`, which may not be a Python identifier."""
        try:
            return self.__getattr__(name)
        except AttributeError as error:
            raise KeyError(name) from error


@dataclass(frozen=True, eq=False)
class Component:
    """A built component: its name, its definition, and the components it places."""

    name: str
    definition: ir.Component
    children: tuple[Component, ...]


@dataclass(slots=True)
class _Placement:
    of: ir.DeviceOf | ir.ComponentOf
    parameters: dict[str, JsonValue]
    terminals: tuple[str, ...]
    source: ir.Source | None
    tolerances: dict[str, ir.Tolerance] = field(default_factory=dict)
    ratings: tuple[ir.Rating, ...] = ()


class Builder:
    """What a component function builds with: ports, placements and connections."""

    def __init__(self, name: str) -> None:
        """Starts the component called `name`, with nothing in it."""
        self._name = name
        self._ports: dict[str, Role] = {}
        self._named: set[str] = set()
        self._placements: dict[str, _Placement] = {}
        self._children: dict[str, Component] = {}
        self._parent: dict[Point, Point] = {}
        self._root_names: dict[Point, str] = {}

    def port(self, name: str, role: Role = Role.SIGNAL) -> Net:
        """Declares the port `name`, the net of that name seen from outside.

        Raises:
            BuildError: When the name is invalid or already declared.
        """
        self._declare(name, "port")
        self._ports[name] = role
        return Net(name)

    def net(self, name: str) -> Net:
        """Declares the net `name`, so connections can join it by name.

        Raises:
            BuildError: When the name is invalid or already declared.
        """
        self._declare(name, "net")
        return Net(name)

    def place(self, item: Device | Component, *, id: str) -> Placed:
        """Places a device or a component as the instance `id`.

        The instance records the line of the build script that placed it.

        Raises:
            BuildError: When the id is invalid or already taken in this component, or
                another component of the same name is placed in it.
        """
        _check_name(id, "an instance id")
        if id in self._placements:
            message = f"{self._name} already places an instance named {id!r}"
            raise BuildError(message)
        if isinstance(item, Component):
            if self._children.setdefault(item.name, item) is not item:
                message = f"two different components are named {item.name!r}"
                raise BuildError(message)
            placement = _Placement(
                ir.ComponentOf(component=item.name),
                {},
                tuple(port.name for port in item.definition.ports),
                caller(),
            )
        else:
            placement = _Placement(
                ir.DeviceOf(device=item.model),
                item.parameters,
                item.terminals,
                caller(),
                item.tolerances,
                item.ratings,
            )
        self._placements[id] = placement
        return Placed(id, placement.terminals)

    def connect(self, *points: Point, name: str | None = None) -> None:
        """Joins `points` into one net, and declares it as the net `name` if given.

        Later connections join a named net as ``Net(name)``.

        Raises:
            BuildError: When a point does not belong to this component, the
                connection would join two nets that have different names, or `name`
                is invalid or already declared.
        """
        joined = [*points, self.net(name)] if name is not None else list(points)
        for point in joined:
            self._check(point)
        for point in joined[1:]:
            self._join(joined[0], point)

    def finish(self) -> Component:
        """The component as built."""
        nets: dict[Point, str] = {}
        instances = []
        for id, placement in self._placements.items():
            terminals = {}
            for terminal in placement.terminals:
                root = self._find(Terminal(id, terminal))
                net = self._root_names.get(root) or nets.setdefault(
                    root, f"{id}.{terminal}"
                )
                terminals[terminal] = net
            instances.append(
                ir.Instance(
                    id=id,
                    of=placement.of,
                    parameters=placement.parameters,
                    terminals=terminals,
                    tolerances=placement.tolerances,
                    ratings=placement.ratings,
                    source=placement.source,
                )
            )
        ports = tuple(
            ir.Port(name=name, role=role) for name, role in self._ports.items()
        )
        definition = ir.Component(ports=ports, instances=tuple(instances))
        return Component(self._name, definition, tuple(self._children.values()))

    def _declare(self, name: str, kind: str) -> None:
        _check_name(name, f"a {kind} name")
        if name in self._named:
            message = f"{self._name} already declares a net or port named {name!r}"
            raise BuildError(message)
        self._named.add(name)

    def _check(self, point: Point) -> None:
        if isinstance(point, Net):
            known = point.name in self._named
        else:
            placement = self._placements.get(point.instance)
            known = placement is not None and point.name in placement.terminals
        if not known:
            message = f"{point} does not belong to {self._name}"
            raise BuildError(message)

    def _find(self, point: Point) -> Point:
        self._parent.setdefault(point, point)
        if isinstance(point, Net):
            self._root_names.setdefault(point, point.name)
        while (parent := self._parent[point]) != point:
            self._parent[point] = self._parent[parent]
            point = parent
        return point

    def _join(self, a: Point, b: Point) -> None:
        root_a, root_b = self._find(a), self._find(b)
        if root_a == root_b:
            return
        name_a, name_b = self._root_names.get(root_a), self._root_names.get(root_b)
        if name_a is not None and name_b is not None:
            message = f"connecting {a} and {b} joins the nets {name_a} and {name_b}"
            raise BuildError(message)
        self._parent[root_b] = root_a
        if name_b is not None:
            self._root_names[root_a] = name_b


class ComponentFunction[**P](Protocol):
    """A function that builds a component: it takes a builder, then its parameters."""

    __name__: str

    def __call__(self, c: Builder, /, *args: P.args, **kwargs: P.kwargs) -> None:
        """Builds the component with `c`."""
        ...


def component[**P](function: ComponentFunction[P]) -> Callable[P, Component]:
    """Turns a function that builds into a factory of components.

    The factory takes the function's parameters after the builder. Each set of
    arguments builds one component, named after the function and those arguments, such
    as ``rc_filter[resistance=1kΩ]``; calling it again with arguments of the same name
    returns the same component.
    """
    signature = inspect.signature(function)
    built: dict[str, Component] = {}

    @functools.wraps(function)
    def factory(*args: P.args, **kwargs: P.kwargs) -> Component:
        bound = signature.bind(None, *args, **kwargs)
        bound.apply_defaults()
        arguments = list(bound.arguments.items())[1:]
        listed = ",".join(f"{name}={_format(value)}" for name, value in arguments)
        name = f"{function.__name__}[{listed}]" if arguments else function.__name__
        if name not in built:
            builder = Builder(name)
            function(builder, *args, **kwargs)
            built[name] = builder.finish()
        return built[name]

    return factory


def circuit(top: Component) -> ir.Circuit:
    """The circuit with `top` at the top of its hierarchy.

    Raises:
        BuildError: When two different components share a name.
    """
    components: dict[str, Component] = {}
    pending = [top]
    while pending:
        current = pending.pop()
        existing = components.setdefault(current.name, current)
        if existing is not current:
            message = f"two different components are named {current.name!r}"
            raise BuildError(message)
        pending.extend(
            child
            for child in current.children
            if components.get(child.name) is not child
        )
    definitions = {name: built.definition for name, built in components.items()}
    return ir.Circuit(top=top.name, components=definitions)


def _check_name(name: str, what: str) -> None:
    if not _NAME.fullmatch(name):
        message = f"{what} uses letters, digits, '_', '+' and '-' only, not {name!r}"
        raise BuildError(message)


def _format(value: object) -> str:
    """An argument as a component's name writes it, without whitespace.

    A quantity is written to 15 significant digits with its SI prefix, a number or a
    short string as in Python, and anything else, such as a device, as its type and a
    digest of its value, so distinct arguments name distinct components.

    Examples:
        >>> from jolty.units import kohm
        >>> _format(1.00001 * kohm), _format(0.5), _format("fast")
        ('1.00001kΩ', '0.5', "'fast'")
        >>> from jolty.devices import resistor
        >>> _format(resistor(1 * kohm))
        'resistor~1c2dea91'
    """
    match value:
        case Quantity():
            return value.compact()
        case bool() | int() | float():
            return repr(value)
        case Device(model=model):
            return f"{model}~{_digest(value)}"
    written = repr(value)
    if any(character.isspace() or character == "/" for character in written):
        return f"{type(value).__name__}~{_digest(value)}"
    return written


def _digest(value: object) -> str:
    """Eight hexadecimal digits of the SHA-256 of `value`'s representation."""
    return hashlib.sha256(repr(value).encode()).hexdigest()[:8]


def caller() -> ir.Source | None:
    """The script line that called the function calling this, such as a method.

    Its file is relative to the working directory when it lies within it.
    """
    frame = inspect.currentframe()
    line = frame.f_back.f_back if frame and frame.f_back else None
    if line is None:
        return None
    path = Path(line.f_code.co_filename)
    with contextlib.suppress(ValueError):
        path = path.resolve().relative_to(Path.cwd())
    return ir.Source(
        file=path.as_posix(), line=line.f_lineno, function=line.f_code.co_name
    )
