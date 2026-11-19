# Copyright 2026 Maxence Faldor
"""Python circuit builder for components, ports, placements, and net connections.

A component is a function decorated with [`component`][jolty.builder.component]. The
decorated function receives a [`Builder`][jolty.builder.Builder] to declare ports,
place devices or child components, and connect terminals. Calling the decorated
function returns a [`Component`][jolty.builder.Component] that another component can
place or that [`circuit`][jolty.builder.circuit] converts into a top-level circuit IR.

Examples:
    >>> from jolty import devices
    >>> from jolty.units import V, kohm
    >>> @component
    ... def divider(c: Builder) -> None:
    ...     ground = c.port("ground", Role.GROUND)
    ...     source = c.place(devices.voltage_source(5 * V), name="V1")
    ...     upper = c.place(devices.resistor(1 * kohm), name="R1")
    ...     lower = c.place(devices.resistor(1 * kohm), name="R2")
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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Protocol, override

from pydantic import JsonValue

from jolty import ir
from jolty.devices import Device, rate
from jolty.errors import BuildError
from jolty.ir import Role
from jolty.units import Quantity

_NAME: Final = re.compile(r"[A-Za-z0-9_+\-]+")
"""Valid pattern for instance and net names, excluding dots and slashes."""


@dataclass(frozen=True, slots=True)
class Terminal:
    """Terminal of a placed instance."""

    instance: str
    name: str

    @override
    def __str__(self) -> str:
        """Formats the terminal as ``<instance>.<name>``, such as ``R1.a``."""
        return f"{self.instance}.{self.name}"


@dataclass(frozen=True, slots=True)
class Net:
    """Named net of a component: either a port or an explicitly declared net."""

    name: str

    @override
    def __str__(self) -> str:
        """Returns the net name."""
        return self.name


type Connectable = Terminal | Net
"""Endpoint that a net connection can join."""


class Placed:
    """Placed device or component whose terminals are exposed as attributes.

    Examples:
        >>> from jolty import devices
        >>> from jolty.units import ohm
        >>> resistor = Placed("R1", devices.resistor(1 * ohm).terminals)
        >>> resistor.a
        Terminal(instance='R1', name='a')
    """

    __slots__ = ("_name", "_terminals")

    def __init__(self, name: str, terminals: tuple[str, ...]) -> None:
        """Stores the instance `name` and its valid `terminals`."""
        self._name = name
        self._terminals = terminals

    @property
    def name(self) -> str:
        """Name of the placed instance."""
        return self._name

    def __getattr__(self, name: str) -> Terminal:
        """Returns the terminal named `name`."""
        if name in self._terminals:
            return Terminal(self._name, name)
        message = (
            f"{self._name} has no terminal {name!r}; "
            f"available terminals: {', '.join(self._terminals)}"
        )
        raise AttributeError(message)

    def __getitem__(self, name: str) -> Terminal:
        """Returns the terminal named `name`, even when not a Python identifier."""
        try:
            return self.__getattr__(name)
        except AttributeError as error:
            raise KeyError(name) from error


@dataclass(frozen=True, eq=False)
class Component:
    """Built component with its name, IR definition, and child components.

    A component can carry port stress ratings checked during simulation and a
    manufacturer part number priced as a single item in a bill of materials. Every
    placed instance of the component inherits those ratings and part number.
    """

    name: str
    definition: ir.Component
    children: tuple[Component, ...]
    ratings: tuple[ir.Rating, ...] = ()
    part: str | None = None

    @property
    def ports(self) -> tuple[str, ...]:
        """Port names of the component in declaration order."""
        return tuple(port.name for port in self.definition.ports)

    def rated(
        self,
        stress: ir.Stress,
        maximum: Quantity | None = None,
        *,
        minimum: Quantity | None = None,
        terminals: Sequence[str] | None = None,
    ) -> Component:
        """Returns a copy of the component with an added port stress rating.

        Ratings follow [`Device.rated`][jolty.devices.Device.rated] semantics: power
        is total electrical port dissipation, and temperature is ambient plus the
        ``thermal`` port net rise when present.

        Args:
            stress: Electrical or thermal stress category to bound.
            maximum: Upper operating limit in units matching `stress`.
            minimum: Lower operating limit in units matching `stress`.
            terminals: Component port names overriding default rating ports.

        Returns:
            Copy of the component carrying the appended port stress rating.

        Raises:
            DimensionError: When a bound does not match the physical dimension of
                `stress`.
            ValueError: When `terminals` are not component ports, have the wrong
                count for `stress`, or have missing or inverted bounds.
        """
        rating = rate(
            self.ports, stress, maximum=maximum, minimum=minimum, terminals=terminals
        )
        return replace(self, ratings=(*self.ratings, rating))

    def numbered(self, part: str) -> Component:
        """Returns a copy of the component tagged with manufacturer part number `part`.

        Args:
            part: Non-empty manufacturer part number without outer whitespace.

        Returns:
            Copy of the component tagged with manufacturer part number `part`.

        Raises:
            ValueError: When `part` is empty or has leading or trailing whitespace.
        """
        return replace(self, part=ir.part_number(part))

    def built_as(self, other: Component) -> bool:
        """Returns whether `other` matches this component except ratings and part.

        Args:
            other: Candidate `Component` compared for structural identity.

        Returns:
            ``True`` when `other` shares the same name, definition, and children.
        """
        return other is self or (
            other.name == self.name
            and other.definition == self.definition
            and other.children == self.children
        )


@dataclass(slots=True)
class _Placement:
    of: ir.DeviceOf | ir.ComponentOf
    parameters: dict[str, JsonValue]
    terminals: tuple[str, ...]
    source_location: ir.SourceLocation | None
    tolerances: dict[str, ir.Tolerance] = field(default_factory=dict)
    ratings: tuple[ir.Rating, ...] = ()
    part: str | None = None


class Builder:
    """Mutable builder for ports, placements, and net connections in a component."""

    def __init__(self, name: str) -> None:
        """Initializes an empty builder for the component named `name`."""
        self._name = name
        self._ports: dict[str, Role] = {}
        self._named: set[str] = set()
        self._placements: dict[str, _Placement] = {}
        self._children: dict[str, Component] = {}
        self._parent: dict[Connectable, Connectable] = {}
        self._root_names: dict[Connectable, str] = {}

    def port(self, name: str, role: Role = Role.SIGNAL) -> Net:
        """Declares external port `name` with the given `role`.

        Args:
            name: External port identifier valid within the component.
            role: Electrical or functional role assigned to the port.

        Returns:
            Named `Net` handle representing the declared external port.

        Raises:
            BuildError: When `name` is invalid or already declared.
        """
        self._declare(name, "port")
        self._ports[name] = role
        return Net(name)

    def net(self, name: str) -> Net:
        """Declares internal net `name` so connections can reference it by name.

        Args:
            name: Internal net identifier valid within the component.

        Returns:
            Named `Net` handle for wiring subsequent connections.

        Raises:
            BuildError: When `name` is invalid or already declared.
        """
        self._declare(name, "net")
        return Net(name)

    def place(self, item: Device | Component, *, name: str) -> Placed:
        """Places a device or component as instance `name`.

        The placed instance records the build-script source line that placed it.

        Args:
            item: Primitive `Device` or child `Component` to instantiate.
            name: Unique instance identifier within the current component.

        Returns:
            `Placed` handle exposing instance terminals as attributes and keys.

        Raises:
            BuildError: When `name` is invalid or already used in this component, or
                when another distinct component already uses the same name.
        """
        _check_name(name, "an instance name")
        if name in self._placements:
            message = f"{self._name} already places an instance named {name!r}"
            raise BuildError(message)
        if isinstance(item, Component):
            if not self._children.setdefault(item.name, item).built_as(item):
                message = f"two different components are named {item.name!r}"
                raise BuildError(message)
            placement = _Placement(
                ir.ComponentOf(component=item.name),
                {},
                item.ports,
                caller(),
                ratings=item.ratings,
                part=item.part,
            )
        else:
            placement = _Placement(
                ir.DeviceOf(model=item.model),
                item.parameters,
                item.terminals,
                caller(),
                item.tolerances,
                item.ratings,
                item.part,
            )
        self._placements[name] = placement
        return Placed(name, placement.terminals)

    def connect(self, *points: Connectable, name: str | None = None) -> None:
        """Joins `points` into one net and optionally declares it as `name`.

        Subsequent connections can reference a named net with ``Net(name)``.

        Args:
            *points: Placed terminals and declared nets to short together.
            name: Optional new internal net name assigned to the joined node.

        Raises:
            BuildError: When a point does not belong to this component, the
                connection joins two differently named nets, or `name` is invalid or
                already declared.
        """
        joined = [*points, self.net(name)] if name is not None else list(points)
        for point in joined:
            self._check(point)
        for point in joined[1:]:
            self._join(joined[0], point)

    def finish(self) -> Component:
        """Returns the finalized component built from all declared items.

        Returns:
            Immutable `Component` assembled from declared ports, placements, and nets.
        """
        nets: dict[Connectable, str] = {}
        instances = []
        for name, placement in self._placements.items():
            terminals = {}
            for terminal in placement.terminals:
                root = self._find(Terminal(name, terminal))
                net = self._root_names.get(root) or nets.setdefault(
                    root, f"{name}.{terminal}"
                )
                terminals[terminal] = net
            instances.append(
                ir.Instance(
                    name=name,
                    of=placement.of,
                    parameters=placement.parameters,
                    terminals=terminals,
                    tolerances=placement.tolerances,
                    ratings=placement.ratings,
                    part=placement.part,
                    source_location=placement.source_location,
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

    def _check(self, point: Connectable) -> None:
        if isinstance(point, Net):
            known = point.name in self._named
        else:
            placement = self._placements.get(point.instance)
            known = placement is not None and point.name in placement.terminals
        if not known:
            message = f"{point} does not belong to {self._name}"
            raise BuildError(message)

    def _find(self, point: Connectable) -> Connectable:
        self._parent.setdefault(point, point)
        if isinstance(point, Net):
            self._root_names.setdefault(point, point.name)
        while (parent := self._parent[point]) != point:
            self._parent[point] = self._parent[parent]
            point = parent
        return point

    def _join(self, a: Connectable, b: Connectable) -> None:
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
    """Protocol for a component builder function taking a builder and parameters."""

    __name__: str

    def __call__(self, c: Builder, /, *args: P.args, **kwargs: P.kwargs) -> None:
        """Populates the component definition using builder `c`."""
        ...


def component[**P](function: ComponentFunction[P]) -> Callable[P, Component]:
    """Wraps a component builder function into a memoized component factory.

    The factory accepts the parameters following the builder argument. Each unique
    argument set builds a component named after the function and formatted arguments,
    such as ``rc_filter[resistance=1kΩ]``. Repeated calls with identical formatted
    arguments return the cached component.

    Args:
        function: Builder callback accepting a leading `Builder` argument.

    Returns:
        Memoized component factory taking the remaining parameters of `function`.
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
    """Builds a circuit IR with `top` as the top-level component.

    Args:
        top: Root `Component` forming the top level of the hierarchy.

    Returns:
        Validated `ir.Circuit` collecting `top` and all transitive child definitions.

    Raises:
        BuildError: When two distinct components share the same name.
    """
    components: dict[str, Component] = {}
    pending = [top]
    while pending:
        current = pending.pop()
        existing = components.setdefault(current.name, current)
        if existing is not current:
            if not existing.built_as(current):
                message = f"two different components are named {current.name!r}"
                raise BuildError(message)
            continue
        pending.extend(current.children)
    definitions = {name: built.definition for name, built in components.items()}
    return ir.Circuit(top=top.name, components=definitions)


def _check_name(name: str, what: str) -> None:
    if not _NAME.fullmatch(name):
        message = f"{what} uses letters, digits, '_', '+' and '-' only, not {name!r}"
        raise BuildError(message)


def _format(value: object) -> str:
    """Formats an argument value without whitespace for component names.

    Quantities format to 15 significant digits with an SI prefix, while scalars and
    short strings use Python repr. Any other value, such as a device, formats as its
    type name plus a digest so distinct arguments produce distinct component names.

    Examples:
        >>> from jolty.units import kohm
        >>> _format(1.00001 * kohm), _format(0.5), _format("fast")
        ('1.00001kΩ', '0.5', "'fast'")
        >>> from jolty.devices import resistor
        >>> _format(resistor(1 * kohm))
        'resistor~40390410'
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
    """Returns eight hexadecimal digits of the SHA-256 digest of ``repr(value)``."""
    return hashlib.sha256(repr(value).encode()).hexdigest()[:8]


def caller() -> ir.SourceLocation | None:
    """Returns the script source location that invoked the calling function.

    The recorded file path is relative to the working directory when located inside
    that directory.

    Returns:
        `ir.SourceLocation` two frames above `caller`, or ``None`` without frames.
    """
    frame = inspect.currentframe()
    line = frame.f_back.f_back if frame and frame.f_back else None
    if line is None:
        return None
    return ir.SourceLocation(
        file=_script_path(line.f_code.co_filename, Path.cwd()),
        line=line.f_lineno,
        function=line.f_code.co_name,
    )


@functools.cache
def _script_path(filename: str, directory: Path) -> str:
    """Returns `filename` relative to `directory` when located inside `directory`.

    Caching avoids resolving the same script path repeatedly for every placement.
    """
    path = Path(filename)
    with contextlib.suppress(ValueError):
        path = path.resolve().relative_to(directory)
    return path.as_posix()
