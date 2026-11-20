# Copyright 2026 Maxence Faldor
"""Subcircuit translation state, current sensing, coupling, and thermal passes."""

import itertools
import math
import operator
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import JsonValue

from jolty import _core, ir
from jolty.builder import Component
from jolty.devices import device
from jolty.errors import SpiceError
from jolty.ir import Role
from jolty.spice import models
from jolty.spice._initial import (
    GROUND,
    INITIAL,
    Built,
    DeviceVoltage,
    InitialState,
    check_net_name,
    parse_voltages,
)
from jolty.spice._params import (
    FUNCTION,
    NO_FUNCTIONS,
    PARAMETERS,
    Drawing,
    Subcircuit,
    TrackedScope,
    parse_assignments,
    parse_function,
    resolve_into,
)
from jolty.spice._preprocess import chosen, keyword, letter, unconditional
from jolty.spice._survey import (
    ELEMENT_KINDS,
    UNMULTIPLIED,
    UNREAD_ELEMENTS,
    UnbinnedError,
    Use,
    checked_device,
    needed_model,
)
from jolty.spice.expressions import (
    Draw,
    Expression,
    Function,
    UndefinedError,
    check_core_depth,
    evaluate,
    inlined,
    nominal,
    number,
    parse,
    translate,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, tokens

type ElementTranslator = Callable[[Translation, Statement], None]
"""Translates one SPICE element statement into a `Translation`."""

_SENSED: Final = frozenset({"v", "e", "h", "b"})
"""The letters of the elements whose current others may read: sources of a voltage, and
B sources."""

_THERMAL_ELEMENTS: Final = {
    ir.DeviceOf(model="resistor"): ("thermal_resistance", "resistance"),
    ir.DeviceOf(model="capacitor"): ("thermal_capacitance", "capacitance"),
}
"""The thermal element a resistor or a capacitor on a thermal node is, and the
parameter of its value."""

_IDENTIFIER: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
"""Pattern matching a valid jolty expression identifier."""

_NON_IDENT_CHAR: Final = re.compile(r"[^A-Za-z0-9_]")
"""Character pattern replaced with ``_`` when synthesizing terminal identifiers."""


class LibraryHost(Protocol):
    """Library interface consumed during subcircuit translation."""

    @property
    def temperature(self) -> float:
        """Temperature at which the library evaluates ``temper``, in kelvins."""
        ...

    @property
    def options(self) -> Mapping[str, float]:
        """Library-level SPICE options, by case-folded name."""
        ...

    @property
    def functions(self) -> Mapping[str, Function]:
        """Library-level user functions defined by ``.func``."""
        ...

    def known_models(self, local: Mapping[str, ModelCard]) -> dict[str, ModelCard]:
        """Returns all models visible inside a subcircuit, local shadowing global.

        Args:
            local: Subcircuit-local model cards keyed by case-folded name.

        Returns:
            Combined global and local model dictionary keyed by case-folded name.
        """
        ...

    def model(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """Looks up a model card by name in local then global scope.

        Args:
            name: Model name to look up (matched case-insensitively).
            local: Subcircuit-local model cards keyed by case-folded name.

        Returns:
            Matching `ModelCard`, or `None` when undefined in both scopes.
        """
        ...

    def card(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """Looks up a model card by name or its first bin.

        Args:
            name: Exact model name or binned model base name.
            local: Subcircuit-local model cards keyed by case-folded name.

        Returns:
            Exact `ModelCard` or first matching bin card, or `None` when absent.
        """
        ...

    def binned(
        self,
        name: str,
        local: Mapping[str, ModelCard],
        instance: Mapping[str, float],
        scope: Mapping[str, float],
        *,
        functions: Mapping[str, Function] = NO_FUNCTIONS,
    ) -> ModelCard | None:
        """Selects the matching MOSFET bin for `instance`.

        Args:
            name: Base model name without numeric bin suffix.
            local: Subcircuit-local model cards keyed by case-folded name.
            instance: Evaluated MOSFET instance attributes including `l`, `w`, and `nf`.
            scope: Case-folded parameter values in scope for evaluating bin bounds.
            functions: Case-folded user-defined functions inlined before evaluation.

        Returns:
            Matching binned `ModelCard`, or `None` when no bin matches `instance`.
        """
        ...

    def fingers(self, instance: Mapping[str, float]) -> float:
        """Returns the finger divisor used when selecting a MOSFET bin.

        Args:
            instance: Evaluated MOSFET instance parameter dictionary.

        Returns:
            Effective finger divisor applied to instance width during bin lookup.
        """
        ...

    def subcircuit(self, name: str) -> Subcircuit | None:
        """Looks up a subcircuit definition by name.

        Args:
            name: Subcircuit name to look up.

        Returns:
            Parsed `Subcircuit` definition, or `None` when undefined.
        """
        ...

    def place_built(  # noqa: PLR0913 - subcircuit placement state and survey list
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
        drawing: Drawing,
        instantiating: tuple[str, ...] = (),
        *,
        uses: list[Use] | None = None,
        temperature: float | None = None,
        temperature_dependent: frozenset[str] | None = None,
    ) -> Built:
        """Builds a child subcircuit placement.

        Args:
            subcircuit: Target subcircuit definition to instantiate.
            scope: Enclosing global parameter values in scope.
            given: Evaluated parameter overrides passed on the instance line.
            drawing: Statistical sampling context and hierarchical placement path.
            instantiating: Active chain of subcircuit names used to detect recursion.
            uses: Optional list collecting survey records rather than raising on errors.
            temperature: Optional library temperature snapshot in kelvins.
            temperature_dependent: Optional set of global parameter names depending on
                library temperature.

        Returns:
            Full `Built` record bundling the component, net names, and initial state.
        """
        ...


@dataclass(frozen=True, slots=True)
class TranslationOptions:
    """Per-placement options injected into `Translation`."""

    drawing: Drawing
    elements: Mapping[str, ElementTranslator]
    multiplier: float = 1.0
    parsed_models: dict[Statement, ModelCard] | None = None
    lock: threading.Lock | None = None
    instantiating: tuple[str, ...] = ()
    uses: list[Use] | None = None
    temperature: float = _core.REFERENCE_TEMPERATURE
    temperature_dependent: frozenset[str] = frozenset()


@dataclass(slots=True)
class Sense:
    """A source whose current elements read, through ports in series with it.

    Each port placed moves the net its positive terminal joins along the chain.
    """

    name: str
    positive: str
    count: int = 0


@dataclass(frozen=True, slots=True)
class Coupling:
    """A mutual inductance: the inductors it couples, by name, coefficient and order."""

    statement: Statement
    inductors: tuple[str, ...]
    coefficient: float
    order: int


class Translation:
    """One subcircuit translated into a component, with its parameters' values."""

    def __init__(
        self,
        library: LibraryHost,
        subcircuit: Subcircuit,
        scopes: tuple[Mapping[str, float], MutableMapping[str, float]],
        options: TranslationOptions,
    ) -> None:
        """Initializes translation state for `subcircuit` under `scopes`."""
        self.library = library
        self.subcircuit = subcircuit
        self.globals, self.scope = scopes
        self.drawing = options.drawing
        self._elements = options.elements
        self.multiplier = options.multiplier
        """The subcircuit's multiplier, which multiplies each element it holds."""
        self._parsed_models = options.parsed_models
        self._cache_lock = options.lock
        self.instantiating = options.instantiating
        self.uses = options.uses
        self._temperature = options.temperature
        self._global_temperature_dependent = options.temperature_dependent
        self._temperature_dependent = False
        self.order = 0
        self.draw: Draw = nominal
        """The draws of the statistical functions of the element being translated."""
        self.parameters: dict[str, Expression] = {}
        if self._cache_lock is not None:
            with self._cache_lock:
                global_functions = dict(library.functions)
        else:
            global_functions = dict(library.functions)
        self.functions: dict[str, Function] = {
            **global_functions,
            **subcircuit.functions,
        }
        self.models: dict[str, ModelCard] = {}
        self.nets: dict[str, str] = {}
        self.instances: list[tuple[int, ir.Instance]] = []
        self.children: dict[str, Component] = {}
        self.senses: dict[str, Sense] = {}
        self.couplings: list[Coupling] = []
        self.initial_statements: list[tuple[str, Statement]] = []
        self.child_initial_conditions: list[tuple[str, float]] = []
        self.child_nodesets: list[tuple[str, float]] = []
        self.capacitor_initial_voltages: list[DeviceVoltage] = []
        self.inductor_initial_currents: dict[str, tuple[str, float]] = {}
        self.coupled_initial_currents: list[tuple[str, float]] = []
        self.child_initial_currents: list[tuple[str, float]] = []
        self.grounded = False
        for pin in subcircuit.pins:
            if pin == "0":
                message = "a pin named `0`, SPICE's global ground"
                raise subcircuit.statement.fail(message)
            self.net(pin)

    @property
    def global_temperature_dependent(self) -> frozenset[str]:
        """Global parameter names whose values depend on library temperature."""
        return self._global_temperature_dependent

    @property
    def temperature_value(self) -> float:
        """Library temperature in kelvins without marking temperature dependence."""
        return self._temperature

    def read_temperature(self) -> float:
        """Returns library temperature in kelvins and marks temperature dependence."""
        self._temperature_dependent = True
        return self._temperature

    def mark_temperature_dependent(self) -> None:
        """Marks this translation as depending on library temperature."""
        self._temperature_dependent = True

    @property
    def is_temperature_dependent(self) -> bool:
        """Whether this translation read library temperature or a dependent value."""
        return self._temperature_dependent or (
            isinstance(self.scope, TrackedScope) and self.scope.read_temperature
        )

    def translate(self, name: str) -> Built:
        """Translates the subcircuit into a `Built` component named `name`.

        Evaluates ``.if`` conditional blocks using parameters declared outside those
        blocks.

        Args:
            name: Qualified component name assigned to the built subcircuit.

        Returns:
            Translated `Built` bundle containing the component, net map, and initial
            state.
        """
        body = self.subcircuit.body
        outside_conditions = {id(statement) for statement in unconditional(body)}
        elements: list[Statement] = []
        for statement in chosen(body, self.scope, self.functions):
            directive = keyword(statement)
            if id(statement) in outside_conditions and directive in {
                FUNCTION,
                *PARAMETERS,
            }:
                continue
            if statement.text.startswith("."):
                self._define(statement)
            else:
                elements.append(statement)
        if self.parameters:
            resolve_into(
                self.parameters, self.scope, self.drawing.draws, self.functions
            )
        for statement in elements:
            if letter(statement) in _SENSED:
                self._source(statement)
        for order, statement in enumerate(elements):
            self.order = order
            self._element(statement)
        self._couple()
        self._join_sensed()
        self._heat()
        ports = [
            ir.Port(name=self.nets[pin.casefold()], role=Role.SIGNAL)
            for pin in self.subcircuit.pins
        ]
        if self.grounded:
            ports.append(ir.Port(name=GROUND, role=Role.GROUND))
        instances = tuple(
            instance
            for _, instance in sorted(self.instances, key=operator.itemgetter(0))
        )
        definition = ir.Component(ports=tuple(ports), instances=instances)
        component = Component(name, definition, tuple(self.children.values()))
        return Built(
            component,
            MappingProxyType(dict(self.nets)),
            self._initial_state(),
            temperature_dependent=self.is_temperature_dependent,
        )

    def _initial_state(self) -> InitialState:
        """Evaluates local ``.ic`` / ``.nodeset`` lines and gathers device ``IC=``."""
        initial_conditions = list(self.child_initial_conditions)
        nodesets = list(self.child_nodesets)
        for directive, statement in self.initial_statements:
            target = initial_conditions if directive == ".ic" else nodesets
            for node, voltage in parse_voltages(
                statement, self.scope, self.functions
            ).items():
                resolved = (
                    GROUND if node == "0" else self.nets.get(node.casefold(), node)
                )
                target.append((resolved, voltage))
        inductors = (
            *self.child_initial_currents,
            *self.coupled_initial_currents,
            *(
                (f"{element_name}:current", current)
                for element_name, current in self.inductor_initial_currents.values()
            ),
        )
        return InitialState(
            initial_conditions=tuple(initial_conditions),
            nodesets=tuple(nodesets),
            capacitors=tuple(self.capacitor_initial_voltages),
            inductors=inductors,
        )

    def component(self, name: str) -> Component:
        """Translates the subcircuit and returns its built `Component` named `name`.

        Args:
            name: Qualified component name assigned to the built subcircuit.

        Returns:
            Translated jolty `Component` for the subcircuit.
        """
        return self.translate(name).component

    def net(self, node: str) -> str:
        """Resolves SPICE `node` to its canonical net name in this subcircuit.

        Node ``0`` maps to `GROUND`; other nodes preserve their first spelling while
        matching case-insensitively.

        Args:
            node: Written SPICE node token to resolve.

        Returns:
            Canonical net name in the translated component (`GROUND` for node ``0``).
        """
        if node == "0":
            self.grounded = True
            return GROUND
        check_net_name(node, self.subcircuit.statement)
        return self.nets.setdefault(node.casefold(), node)

    def value(self, word: str) -> float:
        """Evaluates numeric literal or parameter expression `word` in current scope.

        Statistical calls sample using the active element's `self.draw` callback.

        Args:
            word: SPICE literal or parameter expression token to evaluate.

        Returns:
            Evaluated finite floating-point value.

        Raises:
            SpiceError: When `word` evaluates to a non-finite number.
        """
        written = number(word)
        resolved = (
            evaluate(self.parse(word), self.scope, self.draw)
            if written is None
            else written
        )
        if not math.isfinite(resolved):
            message = f"`{word}` evaluates to a non-finite value ({resolved!r})"
            raise SpiceError(message)
        return resolved

    def multiplied(self, instance: Mapping[str, float]) -> dict[str, float]:
        """Applies the enclosing subcircuit multiplier `m` to `instance` parameters.

        Multiplies instance `m` by `self.multiplier` in all compatibility modes,
        matching ngspice HSPICE mode (``inpcom.c``, lines 4257 to 4283).

        Args:
            instance: Evaluated element-level parameter dictionary.

        Returns:
            Copy of `instance` with `m` scaled by `self.multiplier`.
        """
        if self.multiplier == 1:
            return dict(instance)
        return {**instance, "m": instance.get("m", 1.0) * self.multiplier}

    def parse(self, text: str) -> Expression:
        """Parses SPICE expression `text` and inlines all visible user functions.

        Args:
            text: Raw SPICE expression string to parse and inline.

        Returns:
            Parsed `Expression` tree with user functions expanded inline.

        Raises:
            SpiceError: When `text` is invalid or invokes a recursive user function.
        """
        return inlined(parse(text), self.functions)

    def add(
        self,
        statement: Statement,
        model: str,
        parameters: dict[str, JsonValue],
        terminals: Mapping[str, str],
        *,
        identifier: str | None = None,
    ) -> None:
        """Validates and appends a device instance of `model` for `statement`.

        Connects terminals according to `terminals` and names the instance after the
        element token unless `identifier` is supplied.

        Args:
            statement: Source SPICE element statement providing location metadata.
            model: Target jolty device model name.
            parameters: SI device parameter dictionary validated by the engine.
            terminals: Mapping from jolty terminal names to canonical net names.
            identifier: Optional explicit instance name overriding the element token.
        """
        made = checked_device(statement, model, parameters)
        self.instances.append(
            (
                self.order,
                ir.Instance(
                    name=identifier or statement.text.split()[0],
                    of=ir.DeviceOf(model=model),
                    parameters=made.parameters,
                    terminals={
                        terminal: terminals[terminal] for terminal in made.terminals
                    },
                    source_location=ir.SourceLocation(
                        file=statement.file, line=statement.line
                    ),
                ),
            )
        )

    def add_behavioral(
        self,
        statement: Statement,
        model: str,
        nodes: tuple[str, str],
        expression: Expression,
        *,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> None:
        """Places a behavioral device of `model` between `nodes` using `expression`.

        Adds dedicated sense terminals for referenced node voltages and inserts series
        sensing ports for referenced source currents.

        Args:
            statement: Source SPICE element statement providing location metadata.
            model: Behavioral jolty device model name.
            nodes: Pair of `(positive_node, negative_node)` SPICE node tokens.
            expression: Inlined SPICE `Expression` governing the behavioral output.
            parameters: Optional extra device parameters merged alongside `expression`.
        """
        terminals = {
            "positive": self.net(nodes[0]),
            "negative": self.net(nodes[1]),
        }
        taken = {"positive", "negative", "current"}
        read_nodes: dict[str, str] = {}
        ports: dict[str, str] = {}

        def node(name: str) -> str:
            key = name.casefold()
            if key not in read_nodes:
                terminal = _identifier(name, taken)
                read_nodes[key] = terminal
                terminals[terminal] = self.net(name)
            return read_nodes[key]

        def current(source: str) -> str:
            key = source.casefold()
            if key not in ports:
                port = _identifier(source, taken)
                taken.update({f"{port}_positive", f"{port}_negative"})
                ports[key] = port
                sensed = self.sense(source, statement)
                terminals[f"{port}_positive"], terminals[f"{port}_negative"] = sensed
            return ports[key]

        text = translate(expression, self.scope, node, current)
        try:
            check_core_depth(text)
        except SpiceError as error:
            raise statement.fail(str(error)) from None
        combined: dict[str, JsonValue] = {"expression": text}
        if parameters is not None:
            combined |= parameters
        self.add(statement, model, combined, terminals)

    def sense(self, source: str, statement: Statement) -> tuple[str, str]:
        """Inserts a current-sensing port in series with voltage source `source`.

        Returns the `(entry, exit)` net pair spliced onto the source's positive
        terminal.

        Args:
            source: SPICE name of the sensed voltage or behavioral source.
            statement: Dependent element statement used for error attribution.

        Returns:
            Pair of `(entry_net, exit_net)` spliced in series with `source`.
        """
        found = self.senses.get(source.casefold())
        if found is None:
            message = f"`{source}` is no source in `{self.subcircuit.name}`"
            raise statement.fail(message)
        found.count += 1
        positive, found.positive = found.positive, f"{found.name}.sense{found.count}"
        return positive, found.positive

    def _read_model(self, statement: Statement) -> ModelCard:
        """Parses a local `.model`, reusing cached cards for non-`AKO:` definitions."""
        if self._parsed_models is None or models.is_derived(statement):
            return models.read(statement, self.library.known_models(self.models))
        if self._cache_lock is not None:
            with self._cache_lock:
                card = self._parsed_models.get(statement)
        else:
            card = self._parsed_models.get(statement)
        if card is None:
            card = models.read(statement, self.models)
            if self._cache_lock is not None:
                with self._cache_lock:
                    self._parsed_models[statement] = card
            else:
                self._parsed_models[statement] = card
        return card

    def _define(self, statement: Statement) -> None:
        """Reads the model, parameter, function, initial condition, or nodeset defined.

        Raises:
            SpiceError: When the statement defines none of them.
        """
        directive = keyword(statement)
        if directive == ".model":
            card = self._read_model(statement)
            self.models[card.name.casefold()] = card
        elif directive in PARAMETERS:
            self.parameters.update(parse_assignments(statement))
        elif directive == FUNCTION:
            name, function = parse_function(statement)
            self.functions[name] = function
        elif directive in INITIAL:
            self.initial_statements.append((directive, statement))
        else:
            message = (
                "a subcircuit holds elements, models, parameters, functions, "
                "initial conditions and nodesets only"
            )
            raise statement.fail(message)

    def _join_sensed(self) -> None:
        """Joins each source whose current an element reads to its chain's last net.

        Each port in series with a source moves its positive terminal one net along.
        """
        moved = {
            sensed.name: sensed.positive
            for sensed in self.senses.values()
            if sensed.count
        }
        self.instances = [
            (
                order,
                instance.model_copy(
                    update={
                        "terminals": {
                            **instance.terminals,
                            "positive": moved[instance.name],
                        }
                    }
                )
                if instance.name in moved
                else instance,
            )
            for order, instance in self.instances
        ]

    def _heat(self) -> None:
        """Converts resistors and capacitors on thermal nodes to thermal elements.

        In SPICE, nodes have no domain: a heated device's thermal node holds a
        temperature rise as voltage and takes power as current. Thermal membership
        propagates via breadth-first search across connected resistors and capacitors
        without crossing ground, so multi-stage Foster and Cauer RC ladders become
        thermal resistances in kelvins per watt and thermal capacitances in joules
        per kelvin.

        Raises:
            SpiceError: When a passive element joins a thermal node to an electrical
                node, or drifts with temperature or voltage.
        """
        thermal, electrical, adjacency, queue = _collect_thermal_graph(self.instances)
        if not queue:
            return
        _propagate_thermal(thermal, electrical, adjacency, queue)
        heated: list[tuple[int, ir.Instance]] = []
        for order, instance in self.instances:
            made = _THERMAL_ELEMENTS.get(instance.of)
            non_ground = set(instance.terminals.values()) - {GROUND}
            if made is None or not (non_ground & thermal):
                heated.append((order, instance))
                continue
            model, parameter = made
            location = instance.source_location
            place = f"{location.file}:{location.line}: " if location is not None else ""
            if non_ground & electrical:
                message = f"{place}`{instance.name}` joins a thermal node to another"
                raise SpiceError(message)
            if set(instance.parameters) != {parameter}:
                message = (
                    f"{place}`{instance.name}`, on a thermal node, drifts with "
                    "temperature or voltage, which no thermal element does"
                )
                raise SpiceError(message)
            value = {parameter: instance.parameters[parameter]}
            try:
                checked_parameters = device(model, value).parameters
            except ValueError as error:
                message = f"{place}{error}"
                raise SpiceError(message) from None
            replaced = {
                "of": ir.DeviceOf(model=model),
                "parameters": checked_parameters,
            }
            heated.append((order, instance.model_copy(update=replaced)))
        self.instances = heated

    def _couple(self) -> None:
        """Joins the inductors that mutual inductances couple into coupled inductors.

        The inductors that couplings join, directly or through others, become the
        windings of one device, in the order the expanded subcircuit writes them, which
        takes the id, source location, and expanded order of the group's first coupling.

        Raises:
            SpiceError: When a coupling names no inductor of the subcircuit, couples one
                with itself, or couples a pair twice.
        """
        inductors = {
            instance.name.casefold(): (order, instance)
            for order, instance in self.instances
            if instance.of == ir.DeviceOf(model="inductor")
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
            order, instance = _coupled(group, names, windings)
            self.instances.append((order, instance))
            for winding_index, name in enumerate(names, start=1):
                entry = self.inductor_initial_currents.pop(name, None)
                if entry is not None:
                    self.coupled_initial_currents.append(
                        (f"{instance.name}:current{winding_index}", entry[1])
                    )
            coupled |= set(names)
        self.instances = [
            (order, instance)
            for order, instance in self.instances
            if instance.name.casefold() not in coupled
        ]

    def _source(self, statement: Statement) -> None:
        """Registers a voltage or behavioral source so other elements may sense it."""
        words = tokens(statement.text)
        if len(words) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "a source joins two nodes"
            raise statement.fail(message)
        self.senses[words[0].casefold()] = Sense(words[0], self.net(words[1]))

    def _element(self, statement: Statement) -> None:
        """Translates the element `statement`, or, in a survey, states its use.

        Raises:
            SpiceError: When jolty cannot translate it, outside a survey.
        """
        name = statement.text.split()[0]
        self.draw = self.drawing.draws(name.casefold())
        if self.uses is None:
            self._translated(statement)
            return

        def find_card(word: str) -> ModelCard | None:
            return self.library.card(word, self.models)

        model = needed_model(statement, find_card, self.scope, self.functions)
        path = self.drawing.within(name).path
        try:
            self._translated(statement)
        except UndefinedError as error:
            self.uses.append(Use(path, model, str(error), error.name))
        except UnbinnedError as error:
            self.uses.append(Use(path, model, str(error), middle=error.middle))
        except SpiceError as error:
            self.uses.append(Use(path, model, str(error)))
        else:
            if model:
                self.uses.append(Use(path, model))

    def _translated(self, statement: Statement) -> None:
        """Dispatches an element statement to its registered translator."""
        name = statement.text.split()[0]
        if ir.SEPARATOR in name:
            message = (
                f"an element named `{name}`, whose `{ir.SEPARATOR}` separates the "
                "levels of jolty's paths"
            )
            raise statement.fail(message)
        element_letter = letter(statement)
        if self.multiplier != 1 and element_letter in UNMULTIPLIED:
            message = (
                f"jolty multiplies no {ELEMENT_KINDS[element_letter]} by its "
                f"subcircuit's multiplier `m`, here {self.multiplier!r}"
            )
            raise statement.fail(message)
        translator = self._elements.get(element_letter)
        if translator is None:
            kind = UNREAD_ELEMENTS.get(element_letter)
            named = "" if kind is None else f", {kind}"
            upper = element_letter.upper()
            message = f"jolty has no element of letter `{upper}`{named}"
            raise statement.fail(message)
        translator(self, statement)

    def device(
        self, card: ModelCard, instance: Mapping[str, float], statement: Statement
    ) -> models.Made:
        """Translates model `card` and `instance` parameters into a `Made` device.

        Statistical functions in `card` sample under keys ``<card_name>.<parameter>``.

        Args:
            card: Parsed SPICE `.model` card describing the device.
            instance: Evaluated element-level geometric, area, and multiplicity values.
            statement: Element statement used for error attribution.

        Returns:
            Translated jolty `Made` device pairing model name and SI parameters.
        """
        prefix = card.name.casefold()

        def draws(name: str) -> Draw:
            return self.drawing.draws(f"{prefix}.{name}")

        return models.device(
            card,
            self.scope,
            instance,
            statement,
            draws,
            functions=self.functions,
        )


type _ThermalAdjacency = dict[str, list[tuple[str, ir.Instance]]]
"""Adjacency mapping non-ground nets across passive resistor/capacitor edges."""


def _collect_thermal_graph(
    instances: Sequence[tuple[int, ir.Instance]],
) -> tuple[set[str], set[str], _ThermalAdjacency, deque[str]]:
    """Collects thermal seeds, electrical nets and passive adjacency for BFS."""
    thermal_models = {
        ir.DeviceOf(model=model) for model, _ in _THERMAL_ELEMENTS.values()
    }
    thermal: set[str] = set()
    electrical: set[str] = set()
    queue: deque[str] = deque()
    adjacency: _ThermalAdjacency = {}
    for _, instance in instances:
        if instance.of in _THERMAL_ELEMENTS:
            first, second = instance.terminals["a"], instance.terminals["b"]
            if first != GROUND and second not in {GROUND, first}:
                adjacency.setdefault(first, []).append((second, instance))
                adjacency.setdefault(second, []).append((first, instance))
            continue
        is_thermal_element = instance.of in thermal_models
        heated_device = instance.parameters.get("thermal") is True
        for terminal, net in instance.terminals.items():
            if net == GROUND:
                continue
            if is_thermal_element or (heated_device and terminal == "thermal"):
                if net not in thermal:
                    thermal.add(net)
                    queue.append(net)
            else:
                electrical.add(net)
    return thermal, electrical, adjacency, queue


def _propagate_thermal(
    thermal: set[str],
    electrical: set[str],
    adjacency: _ThermalAdjacency,
    queue: deque[str],
) -> None:
    """Propagates thermal net membership across passive edges via BFS.

    Raises:
        SpiceError: When a passive element connects a thermal net to an electrical net.
    """
    while queue:
        current = queue.popleft()
        for neighbor, instance in adjacency.get(current, ()):
            if neighbor in electrical:
                location = instance.source_location
                place = (
                    f"{location.file}:{location.line}: " if location is not None else ""
                )
                message = f"{place}`{instance.name}` joins a thermal node to another"
                raise SpiceError(message)
            if neighbor not in thermal:
                thermal.add(neighbor)
                queue.append(neighbor)


def _groups(couplings: list[Coupling]) -> list[list[Coupling]]:
    """Partitions `couplings` into connected components sharing inductor windings.

    Each returned group is sorted in expanded statement order.
    """
    groups: list[list[Coupling]] = []
    for coupling in couplings:
        names = {name.casefold() for name in coupling.inductors}
        joined = [
            group
            for group in groups
            if names & {name.casefold() for each in group for name in each.inductors}
        ]
        groups = [group for group in groups if group not in joined]
        groups.append([*(each for group in joined for each in group), coupling])
    return [sorted(group, key=operator.attrgetter("order")) for group in groups]


def _coupled(
    group: list[Coupling], names: list[str], windings: list[ir.Instance]
) -> tuple[int, ir.Instance]:
    """Builds a `coupled_inductors` device and order index for one coupling `group`.

    Combines `windings` (matching `names`) and preserves their shared instance
    temperature override.

    Raises:
        SpiceError: When a coupling self-couples an inductor, duplicates a winding
            pair, or joins windings with conflicting instance temperatures.
    """
    numbers = {name: index for index, name in enumerate(names, start=1)}
    pairs: dict[frozenset[int], float] = {}
    for each in group:
        for first, second in itertools.combinations(each.inductors, 2):
            pair = frozenset({numbers[first.casefold()], numbers[second.casefold()]})
            if len(pair) == 1:
                message = f"`{first}` is coupled with itself"
                raise each.statement.fail(message)
            if pair in pairs:
                message = f"`{first}` and `{second}` are coupled twice"
                raise each.statement.fail(message)
            pairs[pair] = each.coefficient
    couplings: list[JsonValue] = []
    for pair, coefficient in pairs.items():
        numbered: list[JsonValue] = [*sorted(pair)]
        couplings.append({"windings": numbered, "coefficient": coefficient})
    first_coupling = group[0]
    first_statement = first_coupling.statement
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
        raise first_statement.fail(message)
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
        name=first_statement.text.split()[0],
        of=ir.DeviceOf(model="coupled_inductors"),
        parameters=checked_device(
            first_statement, "coupled_inductors", parameters
        ).parameters,
        terminals={
            f"{end}{index}": winding.terminals[end]
            for index, winding in enumerate(windings, start=1)
            for end in ("a", "b")
        },
        source_location=ir.SourceLocation(
            file=first_statement.file, line=first_statement.line
        ),
    )
    return first_coupling.order, instance


def _identifier(name: str, taken: set[str]) -> str:
    """Synthesizes a unique jolty expression identifier from SPICE `name`.

    Avoids collisions with existing names in `taken` and reserved port suffixes.
    """
    base = _NON_IDENT_CHAR.sub("_", name.replace("+", "_plus").replace("-", "_minus"))
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
