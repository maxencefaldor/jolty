# Copyright 2026 Maxence Faldor
"""SPICE libraries read into components.

Each subcircuit becomes a component, and each of its elements a jolty device or a placed
component. A subcircuit's pins become its ports, in order, and a subcircuit that reaches
SPICE's global ground, node ``0``, gains a ground port named ``ground``. Each instance
records the file and the line of the statement that placed it.
"""

import itertools
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from jolty import ir
from jolty.builder import Component
from jolty.devices import device
from jolty.errors import SpiceError
from jolty.ir import Role
from jolty.spice import models
from jolty.spice.expressions import (
    Binary,
    Call,
    Current,
    Expression,
    Number,
    Voltage,
    evaluate,
    names,
    number,
    parse,
    translate,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, statements, tokens
from jolty.units import celsius

GROUND: Final = "ground"
"""The port through which a subcircuit reaches SPICE's global ground, node ``0``."""


@dataclass(frozen=True, slots=True)
class Subcircuit:
    """A subcircuit's definition: its pins, its parameters' defaults, and its body."""

    name: str
    pins: tuple[str, ...]
    defaults: Mapping[str, Expression]
    body: tuple[Statement, ...]
    statement: Statement = field(repr=False)


class Library:
    r"""The subcircuits, models and parameters of SPICE files.

    Each subcircuit is ready to place as a component.

    Examples:
        >>> library = Library()
        >>> library.read(
        ...     ".subckt divider in out\nR1 in out 1k\nR2 out 0 3k\n.ends", "div.lib"
        ... )
        >>> divider = library.component("divider")
        >>> [port.name for port in divider.definition.ports]
        ['in', 'out', 'ground']
    """

    def __init__(self) -> None:
        """An empty library."""
        self._subcircuits: dict[str, Subcircuit] = {}
        self._models: dict[str, ModelCard] = {}
        self._parameters: list[tuple[str, Expression]] = []
        self._built: dict[str, Component] = {}

    @property
    def subcircuits(self) -> tuple[str, ...]:
        """The names of the subcircuits, in the order they are defined."""
        return tuple(subcircuit.name for subcircuit in self._subcircuits.values())

    def read(self, text: str, file: str) -> None:
        """Adds the definitions of `text`, the contents of `file`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads.
        """
        body: list[Statement] = []
        opened: Statement | None = None
        for statement in statements(text, file):
            keyword = statement.text.split()[0].casefold()
            if opened is not None:
                if keyword == ".ends":
                    self._define(opened, body)
                    opened = None
                    body = []
                elif keyword == ".subckt":
                    message = "a subcircuit defined within another"
                    raise statement.fail(message)
                else:
                    body.append(statement)
            elif keyword == ".subckt":
                opened = statement
            elif keyword == ".model":
                card = models.read(statement, self._models)
                self._models[card.name.casefold()] = card
            elif keyword == ".param":
                self._parameters.extend(_assignments(statement))
            elif keyword != ".end":
                message = "a library defines subcircuits, models and parameters only"
                raise statement.fail(message)
        if opened is not None:
            message = "the subcircuit never ends"
            raise opened.fail(message)

    def component(self, name: str, **parameters: float) -> Component:
        """The subcircuit `name` as a component.

        The `parameters` given replace its parameters' defaults.

        Raises:
            SpiceError: When no subcircuit is named `name`, or it holds what jolty
                cannot read.
        """
        subcircuit = self._subcircuits.get(name.casefold())
        if subcircuit is None:
            message = f"no subcircuit is named `{name}`"
            raise SpiceError(message)
        scope = self._scope()
        given = {key.casefold(): value for key, value in parameters.items()}
        return self._component(subcircuit, scope, given)

    def _define(self, opened: Statement, body: list[Statement]) -> None:
        words = tokens(opened.text)
        if len(words) < 2:  # noqa: PLR2004 - `.subckt` and its name
            message = "a subcircuit has a name"
            raise opened.fail(message)
        name, rest = words[1], words[2:]
        pins, defaults = _pins_and_parameters(opened, rest)
        subcircuit = Subcircuit(name, tuple(pins), defaults, tuple(body), opened)
        self._subcircuits[name.casefold()] = subcircuit

    def _scope(self) -> dict[str, float]:
        """The global parameters' values."""
        return _resolved(dict(self._parameters), {})

    def _component(
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
    ) -> Component:
        """The component of `subcircuit` with the `given` parameters, in `scope`."""
        unknown = sorted(set(given) - set(subcircuit.defaults))
        if unknown:
            message = f"`{subcircuit.name}` has no parameter `{unknown[0]}`"
            raise subcircuit.statement.fail(message)
        defaults = {
            parameter: default
            for parameter, default in subcircuit.defaults.items()
            if parameter not in given
        }
        local = _resolved(defaults, {**scope, **given})
        values = {parameter: local[parameter] for parameter in subcircuit.defaults}
        listed = ",".join(f"{key}={value!r}" for key, value in values.items())
        name = f"{subcircuit.name}[{listed}]" if values else subcircuit.name
        if name not in self._built:
            translation = _Translation(self, subcircuit, (scope, local))
            self._built[name] = translation.component(name)
        return self._built[name]

    def model(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """The model `name`, defined in a subcircuit's body or globally."""
        key = name.casefold()
        return local.get(key) or self._models.get(key)

    def binned(
        self,
        name: str,
        local: Mapping[str, ModelCard],
        instance: Mapping[str, float],
        scope: Mapping[str, float],
    ) -> ModelCard | None:
        """The bin of the model `name` an `instance` takes, as ngspice selects it.

        A bin is a BSIM3 card named `name` then a dot and a whole number whose
        ``LMIN`` to ``LMAX`` holds the instance's length and ``WMIN`` to ``WMAX`` its
        width, both given on its element, a bound counted in; the last defined is taken
        first, in a subcircuit's body, then globally.
        """
        if "l" not in instance or "w" not in instance:
            return None
        bins = re.compile(rf"{re.escape(name.casefold())}\.\d+")
        for cards in (local, self._models):
            for key, card in reversed(list(cards.items())):
                if bins.fullmatch(key) and models.holds(
                    card, scope, (instance["l"], instance["w"])
                ):
                    return card
        return None

    def subcircuit(self, name: str) -> Subcircuit | None:
        """The subcircuit `name`."""
        return self._subcircuits.get(name.casefold())

    def place(
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
    ) -> Component:
        """The component a subcircuit instance with the `given` parameters places."""
        return self._component(subcircuit, scope, given)


def load(*paths: Path | str) -> Library:
    """The library of the SPICE files at `paths`.

    Raises:
        SpiceError: When a file holds what jolty cannot read.
    """
    library = Library()
    for path in map(Path, paths):
        library.read(path.read_text(encoding="utf-8", errors="replace"), path.name)
    return library


def _resolved(
    definitions: Mapping[str, Expression], scope: Mapping[str, float]
) -> dict[str, float]:
    """`scope` with each parameter `definitions` defines, evaluated.

    Each parameter is evaluated once those it reads are, whatever the order of their
    definitions, as ngspice sorts them.

    Raises:
        SpiceError: When parameters read one another in a cycle, or an expression is
            wrong.
    """
    values = dict(scope)
    pending = dict(definitions)
    while pending:
        ready = [
            name
            for name, expression in pending.items()
            if not (names(expression) - {name}) & pending.keys()
        ]
        if not ready:
            message = f"parameters read one another in a cycle: {', '.join(pending)}"
            raise SpiceError(message)
        for name in ready:
            values[name] = evaluate(pending.pop(name), values)
    return values


def _assignments(statement: Statement) -> list[tuple[str, Expression]]:
    """The parameters a ``.param`` statement assigns, in order."""
    return list(models.assignments(statement, tokens(statement.text)[1:]).items())


def _pins_and_parameters(
    statement: Statement, words: list[str]
) -> tuple[list[str], dict[str, Expression]]:
    """A subcircuit's pins, then its parameters' defaults.

    The defaults follow ``PARAMS:``, or start with the first ``name = value``.
    """
    split = next(
        (
            index
            for index, word in enumerate(words)
            if word.casefold() == "params:"
            or (index + 1 < len(words) and words[index + 1] == "=")
        ),
        len(words),
    )
    pins = words[:split]
    rest = (
        words[split + 1 :] if words[split : split + 1] == ["PARAMS:"] else words[split:]
    )
    rest = [word for word in rest if word.casefold() != "params:"]
    return pins, models.assignments(statement, rest)


_IDENTIFIER: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


_SENSED: Final = frozenset({"v", "e", "h", "b"})
"""The letters of the elements whose current others may read: sources of a voltage, and
B sources."""


@dataclass(slots=True)
class _Sense:
    """A source whose current elements read, through ports in series with it.

    Each port placed moves the net its positive terminal joins along the chain.
    """

    name: str
    positive: str
    count: int = 0


class _Translation:
    """One subcircuit translated into a component, with its parameters' values."""

    def __init__(
        self,
        library: Library,
        subcircuit: Subcircuit,
        scopes: tuple[Mapping[str, float], Mapping[str, float]],
    ) -> None:
        self.library = library
        self.subcircuit = subcircuit
        self.globals, self.scope = scopes
        self.parameters: dict[str, Expression] = {}
        self.models: dict[str, ModelCard] = {}
        self.nets: dict[str, str] = {}
        self.instances: list[tuple[int, ir.Instance]] = []
        self.children: dict[str, Component] = {}
        self.senses: dict[str, _Sense] = {}
        self.couplings: list[_Coupling] = []
        self.grounded = False
        for pin in subcircuit.pins:
            if pin == "0":
                message = "a pin named `0`, SPICE's global ground"
                raise subcircuit.statement.fail(message)
            self.net(pin)

    def component(self, name: str) -> Component:
        """The component, named `name`."""
        elements: list[Statement] = []
        for statement in self.subcircuit.body:
            if statement.text.startswith("."):
                self._define(statement)
            else:
                elements.append(statement)
        self.scope = _resolved(self.parameters, self.scope)
        for statement in elements:
            if _letter(statement) in _SENSED:
                self._source(statement)
        for statement in elements:
            self._element(statement)
        self._couple()
        self._join_sensed()
        ports = [
            ir.Port(name=self.nets[pin.casefold()], role=Role.SIGNAL)
            for pin in self.subcircuit.pins
        ]
        if self.grounded:
            ports.append(ir.Port(name=GROUND, role=Role.GROUND))
        instances = tuple(
            instance for _, instance in sorted(self.instances, key=lambda pair: pair[0])
        )
        definition = ir.Component(ports=tuple(ports), instances=instances)
        return Component(name, definition, tuple(self.children.values()))

    def net(self, node: str) -> str:
        """The net of SPICE node `node`.

        Node ``0`` is the ground; any other is named as first written, matched without
        case.
        """
        if node == "0":
            self.grounded = True
            return GROUND
        key = node.casefold()
        if key == GROUND:
            message = f"a node named `{node}`, which jolty names the ground port"
            raise self.subcircuit.statement.fail(message)
        return self.nets.setdefault(key, node)

    def value(self, word: str) -> float:
        """The value `word` writes: a number, or an expression of the parameters."""
        written = number(word)
        return written if written is not None else evaluate(parse(word), self.scope)

    def add(
        self,
        statement: Statement,
        model: str,
        parameters: dict[str, JsonValue],
        terminals: Mapping[str, str],
        *,
        identifier: str | None = None,
    ) -> None:
        """Places a device of `model`, the element of `statement`, or a part of it.

        Its terminals join `terminals`' nets; it takes the element's name, or
        `identifier` for a part of it.
        """
        made = device(model, parameters)
        self.instances.append(
            (
                statement.line,
                ir.Instance(
                    id=identifier or statement.text.split()[0],
                    of=ir.DeviceOf(device=model),
                    parameters=made.parameters,
                    terminals={
                        terminal: terminals[terminal] for terminal in made.terminals
                    },
                    source=ir.Source(file=statement.file, line=statement.line),
                ),
            )
        )

    def sense(self, source: str, statement: Statement) -> tuple[str, str]:
        """The nets of a port placed in series with the voltage source `source`.

        The port senses the source's current: it runs from the net the source's positive
        terminal joined to a new one, which the terminal joins instead.
        """
        sense = self.senses.get(source.casefold())
        if sense is None:
            message = f"`{source}` is no source in `{self.subcircuit.name}`"
            raise statement.fail(message)
        sense.count += 1
        positive, sense.positive = sense.positive, f"{sense.name}.sense{sense.count}"
        return positive, sense.positive

    def _define(self, statement: Statement) -> None:
        """Reads the model or the parameters `statement` defines.

        Raises:
            SpiceError: When the statement defines neither.
        """
        keyword = statement.text.split()[0].casefold()
        if keyword == ".model":
            card = models.read(statement, self.models)
            self.models[card.name.casefold()] = card
        elif keyword == ".param":
            self.parameters.update(_assignments(statement))
        else:
            message = "a subcircuit holds elements, models and parameters only"
            raise statement.fail(message)

    def _join_sensed(self) -> None:
        """Joins each source whose current an element reads to its chain's last net.

        Each port in series with a source moves its positive terminal one net along.
        """
        moved = {
            sense.name: sense.positive for sense in self.senses.values() if sense.count
        }
        self.instances = [
            (
                line,
                instance.model_copy(
                    update={
                        "terminals": {
                            **instance.terminals,
                            "positive": moved[instance.id],
                        }
                    }
                )
                if instance.id in moved
                else instance,
            )
            for line, instance in self.instances
        ]

    def _couple(self) -> None:
        """Joins the inductors that mutual inductances couple into coupled inductors.

        The inductors that couplings join, directly or through others, become the
        windings of one device, in the order the file writes them, which takes the id
        and the line of the group's first coupling.

        Raises:
            SpiceError: When a coupling names no inductor of the subcircuit, couples one
                with itself, or couples a pair twice.
        """
        inductors = {
            instance.id.casefold(): (line, instance)
            for line, instance in self.instances
            if instance.of == ir.DeviceOf(device="inductor")
        }
        for coupling in self.couplings:
            for name in coupling.inductors:
                if name.casefold() not in inductors:
                    message = f"`{name}` is no inductor in `{self.subcircuit.name}`"
                    raise coupling.statement.fail(message)
        coupled: set[str] = set()
        for group in _groups(self.couplings):
            names = sorted(
                {name.casefold() for each in group for name in each.inductors},
                key=lambda name: inductors[name][0],
            )
            windings = [inductors[name][1] for name in names]
            self.instances.append(_coupled(group, names, windings))
            coupled |= set(names)
        self.instances = [
            (line, instance)
            for line, instance in self.instances
            if instance.id.casefold() not in coupled
        ]

    def _source(self, statement: Statement) -> None:
        words = tokens(statement.text)
        if len(words) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "a source joins two nodes"
            raise statement.fail(message)
        self.senses[words[0].casefold()] = _Sense(words[0], self.net(words[1]))

    def _element(self, statement: Statement) -> None:
        letter = _letter(statement)
        translate = _ELEMENTS.get(letter)
        if translate is None:
            message = f"jolty has no element of letter `{letter.upper()}`"
            raise statement.fail(message)
        translate(self, statement)


@dataclass(frozen=True, slots=True)
class _Coupling:
    """A mutual inductance: the inductors it couples, by name, and its coefficient."""

    statement: Statement
    inductors: tuple[str, ...]
    coefficient: float


def _groups(couplings: list[_Coupling]) -> list[list[_Coupling]]:
    """The couplings grouped by the inductors they share, directly or through others.

    Each group is in the order the file writes its couplings.
    """
    groups: list[list[_Coupling]] = []
    for coupling in couplings:
        names = {name.casefold() for name in coupling.inductors}
        joined = [
            group
            for group in groups
            if names & {name.casefold() for each in group for name in each.inductors}
        ]
        groups = [group for group in groups if group not in joined]
        groups.append([*(each for group in joined for each in group), coupling])
    return [sorted(group, key=lambda each: each.statement.line) for group in groups]


def _coupled(
    group: list[_Coupling], names: list[str], windings: list[ir.Instance]
) -> tuple[int, ir.Instance]:
    """The coupled inductors of `group`'s couplings, and the line of its first.

    Their windings are `windings`, the inductors named `names`, and they run at the
    windings' own temperature, which every winding shares.

    Raises:
        SpiceError: When a coupling couples an inductor with itself, or a pair twice, or
            the windings run at different temperatures of their own.
    """
    numbers = {name: number for number, name in enumerate(names, start=1)}
    pairs: dict[frozenset[int], float] = {}
    for each in group:
        for a, b in itertools.combinations(each.inductors, 2):
            pair = frozenset({numbers[a.casefold()], numbers[b.casefold()]})
            if len(pair) == 1:
                message = f"`{a}` is coupled with itself"
                raise each.statement.fail(message)
            if pair in pairs:
                message = f"`{a}` and `{b}` are coupled twice"
                raise each.statement.fail(message)
            pairs[pair] = each.coefficient
    couplings: list[JsonValue] = []
    for pair, coefficient in pairs.items():
        numbered: list[JsonValue] = [*sorted(pair)]
        couplings.append({"windings": numbered, "coefficient": coefficient})
    first = group[0].statement
    owns = [
        {
            key: value
            for key, value in winding.parameters.items()
            if key in {"temperature", "temperature_offset"}
        }
        for winding in windings
    ]
    if any(own != owns[0] for own in owns):
        message = "the inductors it couples run at different temperatures of their own"
        raise first.fail(message)
    parameters: dict[str, JsonValue] = {
        "windings": [
            {
                key: value
                for key, value in winding.parameters.items()
                if key not in owns[0]
            }
            for winding in windings
        ],
        "couplings": couplings,
        **owns[0],
    }
    instance = ir.Instance(
        id=first.text.split()[0],
        of=ir.DeviceOf(device="coupled_inductors"),
        parameters=device("coupled_inductors", parameters).parameters,
        terminals={
            f"{end}{number}": winding.terminals[end]
            for number, winding in enumerate(windings, start=1)
            for end in ("a", "b")
        },
        source=ir.Source(file=first.file, line=first.line),
    )
    return first.line, instance


def _mutual(translation: _Translation, statement: Statement) -> None:
    """Reads a mutual inductance, ``K1 L1 L2 0.99``, which couples each pair it names.

    ngspice couples two inductors, PSpice any number; the couplings are joined into
    coupled inductors once every element is read. A magnetic core's model is an error.
    """
    words = tokens(statement.text)
    if len(words) < 4:  # noqa: PLR2004 - a name, two inductors and a coefficient
        message = "a mutual inductance couples two inductors or more by a coefficient"
        raise statement.fail(message)
    _, *inductors, coefficient = words
    if not all(name[0].casefold() == "l" for name in inductors):
        message = "jolty has no magnetic core: a mutual inductance couples inductors"
        raise statement.fail(message)
    translation.couplings.append(
        _Coupling(statement, tuple(inductors), translation.value(coefficient))
    )


def _letter(statement: Statement) -> str:
    """The letter of an element's statement, which says what kind of element it is."""
    return statement.text[0].casefold()


def _keywords(words: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """A statement's positional words, then its ``name = value`` assignments.

    Each assignment holds every value that follows it, as in ``TC=1m,2u``.
    """
    positional: list[str] = []
    assigned: dict[str, list[str]] = {}
    current: list[str] | None = None
    position = 0
    while position < len(words):
        word = words[position]
        if position + 1 < len(words) and words[position + 1] == "=":
            current = assigned.setdefault(word.casefold(), [])
            position += 2
            continue
        if current is None:
            positional.append(word)
        else:
            current.append(word)
        position += 1
    return positional, assigned


_IGNORED_KEYWORDS: Final = frozenset({"ic"})
"""Element keywords that change nothing jolty computes: initial conditions, which only
act when a transient skips its operating point."""

_DRIFT: Final = {
    "tc1": "temperature_coefficient",
    "tc2": "quadratic_temperature_coefficient",
    "tce": "exponential_temperature_coefficient",
}
"""The temperature coefficients of a resistor, capacitor or inductor, by SPICE name."""

_CURVATURE: Final = {
    "vc1": "voltage_coefficient",
    "vc2": "quadratic_voltage_coefficient",
}
"""The voltage coefficients of a resistor or capacitor, by HSPICE's name."""


def _passive(
    model: str, parameter: str, kind: str
) -> Callable[[_Translation, Statement], None]:
    """The translation of a resistor, capacitor or inductor.

    Its model card, of type `kind`, scales its value by the parameter of its letter, and
    gives its temperature coefficients and nominal temperature. The element's own
    coefficients, ``TC=tc1,tc2`` or each by name, override the card's. Only a resistor
    has an exponential coefficient, ``TCE``, and only a resistor and a capacitor voltage
    coefficients, ``VC1`` and ``VC2``.
    """
    letter = kind[0]
    curved = frozenset(_CURVATURE) if model != "inductor" else frozenset()
    drifts = frozenset(_DRIFT) if model == "resistor" else frozenset({"tc1", "tc2"})
    drifts |= curved
    noisy = frozenset({"kf", "af"}) if model == "resistor" else frozenset()

    def translated(translation: _Translation, statement: Statement) -> None:
        positional, assigned = _keywords(tokens(statement.text))
        _check_keywords(
            statement, assigned, frozenset({letter, "tc", "temp", "dtemp", *drifts})
        )
        if len(positional) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "an element joins two nodes"
            raise statement.fail(message)
        _, a, b, *rest = positional
        card: ModelCard | None = None
        values = []
        for word in rest:
            found = translation.library.model(word, translation.models)
            if found is not None and found.kind == kind:
                card = found
            else:
                values.append(word)
        if letter in assigned:
            values.extend(assigned[letter])
        if len(values) != 1:
            message = f"a {model} has one value"
            raise statement.fail(message)
        value = translation.value(values[0])
        given: dict[str, float] = {}
        if card is not None:
            scale = card.parameters.get(letter, Number(1.0))
            value *= evaluate(scale, translation.scope)
            read = (
                models.NOMINAL_TEMPERATURES | models.OWN_TEMPERATURES | drifts | noisy
            )
            _check_model(card, translation.scope, read | {letter}, statement)
            given = {
                name: evaluate(expression, translation.scope)
                for name, expression in card.parameters.items()
                if name in read
            }
        given |= _coefficients(translation, statement, assigned, drifts)
        own = {
            name: translation.value(words[0])
            for name, words in assigned.items()
            if name in {"temp", "dtemp"} and words
        }
        parameters: dict[str, JsonValue] = {parameter: value}
        parameters |= _temperatures(given)
        parameters |= models.own_temperature(given, own, statement)
        parameters |= {
            _CURVATURE[name]: amount
            for name, amount in given.items()
            if name in _CURVATURE
        }
        if given.get("kf", 0.0) != 0:
            parameters |= {
                _FLICKER[name]: amount
                for name, amount in given.items()
                if name in _FLICKER
            }
        translation.add(
            statement,
            model,
            parameters,
            {"a": translation.net(a), "b": translation.net(b)},
        )

    return translated


def _temperatures(given: Mapping[str, float]) -> dict[str, JsonValue]:
    """A resistor's, capacitor's or inductor's drift, by jolty name.

    It comes from the nominal temperature and the coefficients `given` by SPICE name.
    """
    parameters: dict[str, JsonValue] = {}
    for name, amount in given.items():
        if name in models.NOMINAL_TEMPERATURES:
            parameters["nominal_temperature"] = celsius(amount).value
        elif name in _DRIFT:
            parameters[_DRIFT[name]] = amount
    return parameters


def _coefficients(
    translation: _Translation,
    statement: Statement,
    assigned: Mapping[str, list[str]],
    drifts: frozenset[str],
) -> dict[str, float]:
    """The temperature and voltage coefficients an element gives, by SPICE name.

    It gives them as ``TC=tc1,tc2``, or each of `drifts` by name.

    Raises:
        SpiceError: When ``TC`` gives more than two.
    """
    listed = assigned.get("tc", [])
    if len(listed) > 2:  # noqa: PLR2004 - the linear and the quadratic coefficient
        message = "`TC` gives two coefficients at most"
        raise statement.fail(message)
    given = {
        name: translation.value(word)
        for name, word in zip(("tc1", "tc2"), listed, strict=False)
    }
    for name in drifts:
        words = assigned.get(name)
        if words is not None:
            given[name] = translation.value(words[0])
    return given


def _check_keywords(
    statement: Statement, assigned: Mapping[str, list[str]], allowed: frozenset[str]
) -> None:
    """Fails on any keyword neither `allowed` nor one that changes nothing."""
    for keyword in assigned:
        if keyword not in allowed and keyword not in _IGNORED_KEYWORDS:
            message = f"jolty reads no `{keyword.upper()}` here"
            raise statement.fail(message)


_FLICKER: Final = {"kf": "flicker_coefficient", "af": "flicker_exponent"}
"""A resistor's flicker noise parameters, by SPICE name."""


def _check_model(
    card: ModelCard,
    scope: Mapping[str, float],
    allowed: frozenset[str],
    statement: Statement,
) -> None:
    """Checks a resistor's, capacitor's or inductor's model card.

    A capacitor's or inductor's flicker noise exponent changes nothing, as neither
    makes flicker noise: the coefficient, unless zero, is what fails.

    Raises:
        SpiceError: When a parameter is neither `allowed` nor one that changes nothing.
    """
    for name, expression in card.parameters.items():
        if name in allowed or name == "af":
            continue
        if evaluate(expression, scope) != 0:
            raise models.unmatched(card, name, statement)


def _independent(model: str) -> Callable[[_Translation, Statement], None]:
    """The translation of an independent source of `model`, at its DC level."""

    def translated(translation: _Translation, statement: Statement) -> None:
        words = tokens(statement.text)
        _, positive, negative, *rest = words
        if rest[:1] and rest[0].casefold() == "dc":
            rest = rest[1:]
        level = (
            translation.value(rest[0]) if rest and rest[0].casefold() != "ac" else 0.0
        )
        after = rest[1:] if rest and rest[0].casefold() != "ac" else rest
        # A small-signal magnitude belongs to the plan, which excites a source by name.
        if after[:1] and after[0].casefold() == "ac":
            after = after[
                1 + sum(1 for word in after[1:3] if number(word) is not None) :
            ]
        if after:
            message = f"jolty reads a source's DC level only, not `{' '.join(after)}`"
            raise statement.fail(message)
        translation.add(
            statement,
            model,
            {"waveform": {"dc": level}},
            {
                "positive": translation.net(positive),
                "negative": translation.net(negative),
            },
        )

    return translated


def _semiconductor(
    terminals: tuple[str, ...], kinds: frozenset[str]
) -> Callable[[_Translation, Statement], None]:
    """The translation of a diode or a transistor between `terminals`.

    Its model card is of one of `kinds`.
    """

    def translated(translation: _Translation, statement: Statement) -> None:
        positional, assigned = _keywords(tokens(statement.text))
        nodes = positional[1 : 1 + len(terminals)]
        rest = positional[1 + len(terminals) :]
        library, cards = translation.library, translation.models
        if len(nodes) != len(terminals) or not rest:
            message = f"the element joins {len(terminals)} nodes, then names its model"
            raise statement.fail(message)
        instance = {
            key: translation.value(values[0])
            for key, values in assigned.items()
            if values
        }
        card = library.model(rest[0], cards)
        if card is None and "nmos" in kinds:
            card = library.binned(rest[0], cards, instance, translation.scope)
        if card is None:
            message = f"no model is named `{rest[0]}`"
            raise statement.fail(message)
        if card.kind not in kinds:
            message = f"`{card.name}` is a model of type `{card.kind}`"
            raise statement.fail(message)
        # An initial state, `OFF`, only guides the search for the operating point.
        areas = [word for word in rest[1:] if word.casefold() not in {"off", "on"}]
        if areas:
            instance["area"] = translation.value(areas[0])
        made = models.device(card, translation.scope, instance, statement)
        joined = {
            terminal: translation.net(node)
            for terminal, node in zip(terminals, nodes, strict=True)
        }
        translation.add(statement, made.model, made.parameters, joined)

    return translated


_MOSFET: Final = _semiconductor(
    ("drain", "gate", "source", "bulk"), frozenset({"nmos", "pmos"})
)
"""The translation of a level 1 or level 3 MOSFET."""

_VDMOS_TERMINALS: Final = ("drain", "gate", "source")
"""A VDMOS's terminals, in its element's order."""

_VDMOS_NODES: Final = 5
"""The most nodes a VDMOS joins: its terminals, then a thermal junction and case, which
join nothing unless it heats."""


def _transistor(translation: _Translation, statement: Statement) -> None:
    """The translation of an `M` element: a MOSFET, or a VDMOS, as its model card says.

    As in ngspice, the element names its model after its nodes, the first word from the
    fourth node on that names a model: a MOSFET joins four nodes, and a VDMOS three,
    its drain, gate and source, and a fourth and a fifth, its thermal nodes, which join
    nothing where the element does not heat by ngspice's ``thermal``, and jolty reads
    no element that does. A VDMOS takes its number of copies, ``M``, and no area.

    Raises:
        SpiceError: When no model follows the nodes, or the element is not one jolty
            reads.
    """
    positional, assigned = _keywords(tokens(statement.text))
    library, cards = translation.library, translation.models
    first = len(_VDMOS_TERMINALS) + 1
    named = next(
        (
            position
            for position in range(first, len(positional))
            if library.model(positional[position], cards) is not None
        ),
        None,
    )
    card = None if named is None else library.model(positional[named], cards)
    if named is None or card is None or card.kind not in models.VDMOS_KINDS:
        _MOSFET(translation, statement)
        return
    nodes, rest = positional[1:named], positional[named + 1 :]
    if len(nodes) > _VDMOS_NODES:
        message = f"a VDMOS joins at most {_VDMOS_NODES} nodes, then names its model"
        raise statement.fail(message)
    for word in rest:
        if word.casefold() == "thermal":
            message = (
                "jolty reads no `THERMAL` VDMOS: a vdmos heats through a thermal "
                "terminal of its own, joined to a thermal network"
            )
            raise statement.fail(message)
        # An initial state, `OFF`, only guides the search for the operating point.
        if word.casefold() != "off":
            message = f"a VDMOS takes no `{word}`"
            raise statement.fail(message)
    instance = {
        key: translation.value(values[0]) for key, values in assigned.items() if values
    }
    made = models.device(card, translation.scope, instance, statement)
    joined = {
        terminal: translation.net(node)
        for terminal, node in zip(_VDMOS_TERMINALS, nodes, strict=False)
    }
    translation.add(statement, made.model, made.parameters, joined)


_BIPOLAR_TERMINALS: Final = ("collector", "base", "emitter")
"""A bipolar transistor's terminals, in its element's order, before its substrate."""


def _bipolar(translation: _Translation, statement: Statement) -> None:
    """The translation of a `Q` element: a Gummel-Poon transistor or a VBIC, by card.

    As in ngspice, the element names its model after its nodes, the first word from the
    fourth node on that names a model: three nodes, then a substrate, the ground unless
    a fourth node gives it, and for a VBIC a fifth, its thermal node. A VBIC whose card
    heats, by a positive ``RTH`` and a ``SELFT`` of 1 or none, heats through a thermal
    terminal on that node's net, held to the ambient by its thermal resistance and
    capacitance, devices named after the element with ``.rth`` and ``.cth``; without a
    thermal node, its thermal node is the ground, as ngspice joins it, and it does not
    heat.

    Raises:
        SpiceError: When no bipolar transistor's model follows three to five nodes.
    """
    positional, assigned = _keywords(tokens(statement.text))
    library, cards = translation.library, translation.models
    first = len(_BIPOLAR_TERMINALS) + 1
    named = next(
        (
            position
            for position in range(first, len(positional))
            if library.model(positional[position], cards) is not None
        ),
        None,
    )
    card = None if named is None else library.model(positional[named], cards)
    if named is None or card is None:
        message = "the element joins three to five nodes, then names its model"
        raise statement.fail(message)
    if card.kind not in {"npn", "pnp"}:
        message = f"`{card.name}` is a model of type `{card.kind}`"
        raise statement.fail(message)
    nodes, rest = positional[1:named], positional[named + 1 :]
    instance = {
        key: translation.value(values[0]) for key, values in assigned.items() if values
    }
    # An initial state, `OFF`, only guides the search for the operating point.
    areas = [word for word in rest if word.casefold() not in {"off", "on"}]
    if areas:
        instance["area"] = translation.value(areas[0])
    made = models.device(card, translation.scope, instance, statement)
    most = 5 if made.model == "vbic" else 4
    if len(nodes) > most:
        message = f"the transistor joins at most {most} nodes, then names its model"
        raise statement.fail(message)
    collector, base, emitter, *more = nodes
    substrate = more[0] if more else "0"
    joined = {
        "collector": translation.net(collector),
        "base": translation.net(base),
        "emitter": translation.net(emitter),
        "substrate": translation.net(substrate),
    }
    heating = (
        models.vbic_heating(card, translation.scope, instance)
        if len(nodes) == most == 5  # noqa: PLR2004 - a VBIC's thermal node, its fifth
        else None
    )
    if heating is None:
        translation.add(statement, made.model, made.parameters, joined)
        return
    thermal = translation.net(nodes[4])
    joined["thermal"] = thermal
    translation.add(statement, made.model, {**made.parameters, "thermal": True}, joined)
    resistance, capacitance = heating
    name = statement.text.split()[0]
    parts: tuple[tuple[str, str, dict[str, JsonValue]], ...] = (
        ("rth", "thermal_resistance", {"resistance": resistance}),
        ("cth", "thermal_capacitance", {"capacitance": capacitance}),
    )
    between = {"a": thermal, "b": translation.net("0")}
    for suffix, model, parameters in parts:
        translation.add(
            statement, model, parameters, between, identifier=f"{name}.{suffix}"
        )


def _controlled(output: str) -> Callable[[_Translation, Statement], None]:
    """The translation of an E, F, G or H source, whose output is a `output`."""

    def translated(translation: _Translation, statement: Statement) -> None:
        words = tokens(statement.text)
        if len(words) < 4:  # noqa: PLR2004 - a name, two nodes and a control
            message = "a controlled source joins two nodes, then names its control"
            raise statement.fail(message)
        letter = words[0][0].casefold()
        positive, negative, rest = words[1], words[2], words[3:]
        first = rest[0].casefold()
        by_current = letter in {"f", "h"}
        if first.startswith("poly("):
            expression = _polynomial(statement, rest, by_current=by_current)
        elif first == "value" and not by_current:
            expression = parse(rest[2] if rest[1:2] == ["="] else rest[1])
        elif first == "table" and not by_current:
            expression = _table(statement, rest)
        elif by_current and len(rest) == 2:  # noqa: PLR2004 - a source and a gain
            _linear_current(translation, statement, (positive, negative), rest, output)
            return
        elif not by_current and len(rest) == 3:  # noqa: PLR2004 - two nodes and a gain
            control_positive, control_negative, gain = rest
            parameter = "gain" if output == "voltage" else "transconductance"
            translation.add(
                statement,
                f"voltage_controlled_{output}_source",
                {parameter: translation.value(gain)},
                {
                    "positive": translation.net(positive),
                    "negative": translation.net(negative),
                    "control_positive": translation.net(control_positive),
                    "control_negative": translation.net(control_negative),
                },
            )
            return
        else:
            message = "jolty reads a controlled source's gain, POLY, VALUE or TABLE"
            raise statement.fail(message)
        _behavioral(translation, statement, output, (positive, negative), expression)

    return translated


def _linear_current(
    translation: _Translation,
    statement: Statement,
    nodes: tuple[str, str],
    rest: list[str],
    output: str,
) -> None:
    """A current-controlled source with a gain.

    Its control port is in series with the source whose current it reads.
    """
    source, gain = rest
    control_positive, control_negative = translation.sense(source, statement)
    parameter = "gain" if output == "current" else "transresistance"
    translation.add(
        statement,
        f"current_controlled_{output}_source",
        {parameter: translation.value(gain)},
        {
            "positive": translation.net(nodes[0]),
            "negative": translation.net(nodes[1]),
            "control_positive": control_positive,
            "control_negative": control_negative,
        },
    )


_TABLE_NUMBER: Final = re.compile(
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?[a-z]*", re.IGNORECASE
)


_ROUNDING: Final = 0.1
"""How much of the shorter segment beside each corner of a ``TABLE`` source ngspice
rounds: its XSPICE ``pwl`` model's ``input_domain``."""


def _table(statement: Statement, rest: list[str]) -> Expression:
    """``TABLE {x} = (x1, y1) …``, as ngspice reads it.

    It interpolates its points, holds beyond the ends, and rounds its corners. The
    points are read leniently, as PSpice reads unbalanced parentheses.
    """
    if len(rest) < 2:  # noqa: PLR2004 - the keyword and the input
        message = "a TABLE has an input and points"
        raise statement.fail(message)
    written = [number(word) for word in _TABLE_NUMBER.findall(" ".join(rest[2:]))]
    if not written or len(written) % 2 or any(value is None for value in written):
        message = "a TABLE's points come in pairs of numbers"
        raise statement.fail(message)
    points = [Number(value) for value in written if value is not None]
    return Call("rounded_table", (parse(rest[1]), Number(_ROUNDING), *points))


def _polynomial(
    statement: Statement, rest: list[str], *, by_current: bool
) -> Expression:
    """``POLY(n)`` controls and coefficients, as SPICE2's polynomial.

    The coefficients are the constant's, the linear terms', then each product of two
    controls', then of three, in order.
    """
    match = re.fullmatch(r"poly\((\d+)\)", rest[0], re.IGNORECASE)
    if match is None:
        message = "POLY counts its controls, as POLY(2)"
        raise statement.fail(message)
    count = int(match[1])
    words = [part for word in rest[1:] for part in _split_group(word)]
    # A control is a source, or a pair of nodes.
    width = 1 if by_current else 2
    written = [words[width * k : width * (k + 1)] for k in range(count)]
    values = [number(word) for word in words[width * count :]]
    if len(words) <= width * count or any(value is None for value in values):
        message = f"POLY({count}) reads {count} controls, then numbers"
        raise statement.fail(message)
    controls: list[Expression] = [
        Current(*control) if by_current else Voltage(*control) for control in written
    ]
    numbers = [value for value in values if value is not None]
    # One coefficient of a one-dimensional polynomial is its gain, as in SPICE.
    if count == 1 and len(numbers) == 1:
        numbers = [0.0, numbers[0]]
    monomials = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(count), degree)
        for degree in itertools.count()
    )
    terms: list[Expression] = []
    for coefficient, monomial in zip(numbers, monomials, strict=False):
        if coefficient == 0:
            continue
        term: Expression = Number(coefficient)
        for control in monomial:
            term = Binary("*", term, controls[control])
        terms.append(term)
    if not terms:
        return Number(0.0)
    total = terms[0]
    for term in terms[1:]:
        total = Binary("+", total, term)
    return total


def _split_group(word: str) -> list[str]:
    """The words of a group such as ``(3,0)``, or the word itself."""
    if word.startswith("(") and word.endswith(")"):
        return [part for part in re.split(r"[\s,]+", word[1:-1]) if part]
    return [word]


def _behavioral(
    translation: _Translation,
    statement: Statement,
    output: str,
    nodes: tuple[str, str],
    expression: Expression,
) -> None:
    """A behavioral source of `output` between `nodes`, following `expression`.

    Each node it reads is a terminal of its own, and each source whose current it reads
    a port in series with that source.
    """
    terminals = {
        "positive": translation.net(nodes[0]),
        "negative": translation.net(nodes[1]),
    }
    taken = {"positive", "negative", "current"}
    read_nodes: dict[str, str] = {}
    ports: dict[str, str] = {}

    def node(name: str) -> str:
        key = name.casefold()
        if key not in read_nodes:
            terminal = _identifier(name, taken)
            read_nodes[key] = terminal
            terminals[terminal] = translation.net(name)
        return read_nodes[key]

    def current(source: str) -> str:
        key = source.casefold()
        if key not in ports:
            port = _identifier(source, taken)
            taken.update({f"{port}_positive", f"{port}_negative"})
            ports[key] = port
            sensed = translation.sense(source, statement)
            terminals[f"{port}_positive"], terminals[f"{port}_negative"] = sensed
        return ports[key]

    text = translate(expression, translation.scope, node, current)
    translation.add(
        statement, f"behavioral_{output}_source", {"expression": text}, terminals
    )


def _identifier(name: str, taken: set[str]) -> str:
    """A name that jolty's expressions read, from a SPICE node's or source's.

    It names a terminal or a port, unlike any `taken`.
    """
    base = re.sub(
        r"[^A-Za-z0-9_]", "_", name.replace("+", "_plus").replace("-", "_minus")
    )
    if not _IDENTIFIER.fullmatch(base):
        base = f"n{base}"
    candidate = base
    for suffix in itertools.count(2):
        reserved = candidate.endswith(("_positive", "_negative"))
        if candidate not in taken and not reserved:
            break
        candidate = f"{base}_{suffix}"
    taken.add(candidate)
    return candidate


def _arbitrary(translation: _Translation, statement: Statement) -> None:
    """A B source: ``V=`` or ``I=`` an expression."""
    match = re.match(
        r"\S+\s+(\S+)\s+(\S+)\s+([VI])\s*=\s*(.+)$", statement.text, re.IGNORECASE
    )
    if match is None:
        message = "a B source joins two nodes, then sets `V=` or `I=` an expression"
        raise statement.fail(message)
    positive, negative, kind, written = match.groups()
    output = "voltage" if kind.casefold() == "v" else "current"
    _behavioral(translation, statement, output, (positive, negative), parse(written))


def _switch(translation: _Translation, statement: Statement) -> None:
    """An S switch, controlled by a voltage."""
    words = [
        word for word in tokens(statement.text) if word.casefold() not in {"on", "off"}
    ]
    if len(words) != 6:  # noqa: PLR2004 - a name, four nodes and a model
        message = "a switch joins four nodes, then names its model"
        raise statement.fail(message)
    _, positive, negative, control_positive, control_negative, name = words
    card = translation.library.model(name, translation.models)
    if card is None:
        message = f"no model is named `{name}`"
        raise statement.fail(message)
    made = models.device(card, translation.scope, {}, statement)
    translation.add(
        statement,
        made.model,
        made.parameters,
        {
            "positive": translation.net(positive),
            "negative": translation.net(negative),
            "control_positive": translation.net(control_positive),
            "control_negative": translation.net(control_negative),
        },
    )


def _current_switch(translation: _Translation, statement: Statement) -> None:
    """A W switch, controlled by the current through a source of a voltage.

    Its control port is in series with the source, as a current-controlled source's is.
    """
    words = [
        word for word in tokens(statement.text) if word.casefold() not in {"on", "off"}
    ]
    if len(words) != 5:  # noqa: PLR2004 - a name, two nodes, a source and a model
        message = "a current-controlled switch joins two nodes, then names its source "
        message += "and its model"
        raise statement.fail(message)
    _, positive, negative, source, name = words
    card = translation.library.model(name, translation.models)
    if card is None:
        message = f"no model is named `{name}`"
        raise statement.fail(message)
    made = models.device(card, translation.scope, {}, statement)
    control_positive, control_negative = translation.sense(source, statement)
    translation.add(
        statement,
        made.model,
        made.parameters,
        {
            "positive": translation.net(positive),
            "negative": translation.net(negative),
            "control_positive": control_positive,
            "control_negative": control_negative,
        },
    )


def _instance(translation: _Translation, statement: Statement) -> None:
    """An X instance of a subcircuit, with its parameters."""
    words = tokens(statement.text)
    split = next(
        (
            position
            for position, word in enumerate(words)
            if word.casefold() == "params:"
            or (position + 1 < len(words) and words[position + 1] == "=")
        ),
        len(words),
    )
    positional = words[:split]
    assigned = [word for word in words[split:] if word.casefold() != "params:"]
    if len(positional) < 2:  # noqa: PLR2004 - a name and a subcircuit
        message = "an instance names its subcircuit"
        raise statement.fail(message)
    name, *nodes, target = positional
    subcircuit = translation.library.subcircuit(target)
    if subcircuit is None:
        message = f"no subcircuit is named `{target}`"
        raise statement.fail(message)
    if len(nodes) != len(subcircuit.pins):
        message = (
            f"`{subcircuit.name}` has {len(subcircuit.pins)} pins, not {len(nodes)}"
        )
        raise statement.fail(message)
    given = {
        parameter: evaluate(expression, translation.scope)
        for parameter, expression in models.assignments(statement, assigned).items()
    }
    child = translation.library.place(subcircuit, translation.globals, given)
    ports = [port.name for port in child.definition.ports]
    terminals = {
        port: translation.net(node) for port, node in zip(ports, nodes, strict=False)
    }
    if GROUND in ports:
        terminals[GROUND] = translation.net("0")
    translation.children[child.name] = child
    translation.instances.append(
        (
            statement.line,
            ir.Instance(
                id=name,
                of=ir.ComponentOf(component=child.name),
                terminals=terminals,
                source=ir.Source(file=statement.file, line=statement.line),
            ),
        )
    )


_ELEMENTS: Final[dict[str, Callable[[_Translation, Statement], None]]] = {
    "r": _passive("resistor", "resistance", "res"),
    "c": _passive("capacitor", "capacitance", "cap"),
    "l": _passive("inductor", "inductance", "ind"),
    "v": _independent("voltage_source"),
    "i": _independent("current_source"),
    "d": _semiconductor(("anode", "cathode"), frozenset({"d"})),
    "q": _bipolar,
    "j": _semiconductor(("drain", "gate", "source"), frozenset({"njf", "pjf"})),
    "m": _transistor,
    "e": _controlled("voltage"),
    "f": _controlled("current"),
    "g": _controlled("current"),
    "h": _controlled("voltage"),
    "b": _arbitrary,
    "k": _mutual,
    "s": _switch,
    "w": _current_switch,
    "x": _instance,
}
"""How each element, by its letter, becomes devices or a placed component."""
