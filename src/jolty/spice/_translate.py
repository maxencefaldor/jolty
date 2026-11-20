# Copyright 2026 Maxence Faldor
"""Subcircuit translation state, current sensing, coupling, and thermal passes."""

import itertools
import math
import operator
import re
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, override

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
    DualBuilt,
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
    resolve_into_dual,
)
from jolty.spice._preprocess import chosen, keyword, letter, unconditional
from jolty.spice._statistics import (
    StatisticalCompiler,
    check_condition_deterministic,
    extract_probe_behavioral,
    record_device_targets,
    record_vbic_heating_targets,
)
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
    Binary,
    Draw,
    Dual,
    DualDraw,
    Expression,
    Function,
    Number,
    Scalar,
    UndefinedError,
    check_core_depth,
    compact,
    evaluate,
    evaluate_dual,
    has_derivatives,
    inlined,
    is_finite,
    nominal,
    number,
    parse,
    primal,
    translate,
)
from jolty.spice.models import BinBounds, DualJsonValue, ModelCard
from jolty.spice.netlist import Statement, tokens


class ElementTranslator(Protocol):
    """Translates one SPICE element statement into a `Translation`."""

    def __call__[Seed](
        self, translation: Translation[Seed], statement: Statement
    ) -> None:
        """Translates `statement` into `translation`."""
        ...


class _PrimalScope[Seed](Mapping[str, float]):
    """Read-only primal-value view over a scalar or dual parameter scope."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Scalar[Seed]]) -> None:
        self._values = values

    @override
    def __getitem__(self, key: str) -> float:
        return primal(self._values[key])

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    @override
    def __len__(self) -> int:
        return len(self._values)

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._values


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

    def binned_with_bounds(
        self,
        name: str,
        local: Mapping[str, ModelCard],
        instance: Mapping[str, float],
        scope: Mapping[str, float],
        *,
        functions: Mapping[str, Function] = NO_FUNCTIONS,
    ) -> tuple[ModelCard, BinBounds] | None:
        """Selects the matching MOSFET bin card and its active `BinBounds`.

        Args:
            name: Base model name without numeric bin suffix.
            local: Subcircuit-local model cards keyed by case-folded name.
            instance: Evaluated MOSFET instance attributes including `l`, `w`, and `nf`.
            scope: Case-folded parameter values in scope for evaluating bin bounds.
            functions: Case-folded user-defined functions inlined before evaluation.

        Returns:
            Pair `(card, bounds)` for the matching bin, or `None` when none matches.
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

    def place_built(  # noqa: PLR0913 - subcircuit placement state and statistical hoisting
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
        given_symbolic: Mapping[str, Expression] | None = None,
        parent_statistics: StatisticalCompiler | None = None,
        hoist_instance: str | None = None,
        given_probe_stochastic: frozenset[str] = frozenset(),
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
            given_symbolic: Optional symbolic expressions for stochastic `given` args.
            parent_statistics: Enclosing `StatisticalCompiler` when hoisting this child.
            hoist_instance: Local instance name under `parent_statistics` when hoisting.
            given_probe_stochastic: Names in `given` that vary under `Probe`.

        Returns:
            Full `Built` record bundling the component, net names, and initial state.
        """
        ...

    def place_dual[Seed](  # noqa: PLR0913 - dual subcircuit placement state
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, Scalar[Seed]],
        drawing: Drawing,
        instantiating: tuple[str, ...] = (),
        *,
        dual_draws: Callable[[str, str], DualDraw[Seed]] | None = None,
        temperature: float | None = None,
        temperature_dependent: frozenset[str] | None = None,
    ) -> DualBuilt[Seed]:
        """Builds a child subcircuit placement propagating exact dual derivatives.

        Args:
            subcircuit: Target subcircuit definition to instantiate.
            scope: Enclosing global parameter values in scope.
            given: Scalar or dual parameter overrides passed on the instance line.
            drawing: Statistical sampling context and hierarchical placement path.
            instantiating: Active chain of subcircuit names used to detect recursion.
            dual_draws: Optional factory mapping `(path, name)` to a `DualDraw`.
            temperature: Optional library temperature snapshot in kelvins.
            temperature_dependent: Optional set of global parameter names depending on
                library temperature.

        Returns:
            `DualBuilt` bundling the translated child, derivatives, and bin bounds.
        """
        ...


@dataclass(frozen=True, slots=True)
class TranslationOptions[Seed]:
    """Per-placement options injected into `Translation`."""

    drawing: Drawing
    elements: Mapping[str, ElementTranslator]
    multiplier: Scalar[Seed] = 1.0
    parsed_models: dict[Statement, ModelCard] | None = None
    lock: threading.Lock | None = None
    instantiating: tuple[str, ...] = ()
    uses: list[Use] | None = None
    temperature: float = _core.REFERENCE_TEMPERATURE
    temperature_dependent: frozenset[str] = frozenset()
    statistics: StatisticalCompiler | None = None
    probe_compiler: StatisticalCompiler | None = None
    multiplier_symbolic: Expression | None = None
    hoisted: bool = False
    differentiating: bool = False
    dual_draws: Callable[[str, str], DualDraw[Seed]] | None = None


@dataclass(slots=True)
class Sense:
    """A source whose current elements read, through ports in series with it.

    Each port placed moves the net its positive terminal joins along the chain.
    """

    name: str
    positive: str
    count: int = 0


@dataclass(frozen=True, slots=True)
class Coupling[Seed]:
    """A mutual inductance: the inductors it couples, by name, coefficient and order."""

    statement: Statement
    inductors: tuple[str, ...]
    coefficient: Scalar[Seed]
    order: int
    coefficient_symbolic: Expression | None = None


class Translation[Seed]:
    """One subcircuit translated into a component, with its parameters' values."""

    def __init__(
        self,
        library: LibraryHost,
        subcircuit: Subcircuit,
        scopes: tuple[Mapping[str, float], Mapping[str, Scalar[Seed]]],
        options: TranslationOptions[Seed],
    ) -> None:
        """Initializes translation state for `subcircuit` under `scopes`."""
        self.library = library
        self.subcircuit = subcircuit
        self.globals, self.scope = scopes
        self.drawing = options.drawing
        self._elements = options.elements
        self.multiplier: Scalar[Seed] = options.multiplier
        """The subcircuit's multiplier, which multiplies each element it holds."""
        self._parsed_models = options.parsed_models
        self._cache_lock = options.lock
        self.instantiating = options.instantiating
        self.uses = options.uses
        self._temperature = options.temperature
        self._global_temperature_dependent = options.temperature_dependent
        self._temperature_dependent = False
        self.statistics = options.statistics
        self.probe_compiler = options.probe_compiler
        self.multiplier_symbolic = options.multiplier_symbolic
        self._hoisted = options.hoisted
        self.differentiating = options.differentiating
        self._dual_draws = options.dual_draws
        self.order = 0
        self.draw: Draw = nominal
        """The draws of the statistical functions of the element being translated."""
        self.dual_draw: DualDraw[Seed] | None = None
        """Dual statistical draw callback for the element being translated."""
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
        self.couplings: list[Coupling[Seed]] = []
        self.device_derivatives: dict[str, dict[str, dict[Seed, float]]] = {}
        self.device_bins: dict[str, BinBounds] = {}
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
    def primal_scope(self) -> Mapping[str, float]:
        """Read-only primal floating-point view over `self.scope`."""
        return _PrimalScope(self.scope)

    @property
    def dual_draw_factory(self) -> Callable[[str, str], DualDraw[Seed]] | None:
        """Factory producing dual statistical callbacks for nested placements."""
        return self._dual_draws

    def dual_draws_for(self, name: str) -> DualDraw[Seed] | None:
        """Returns the dual statistical callback for `name` at `self.drawing.path`."""
        if self._dual_draws is None:
            return None
        return self._dual_draws(self.drawing.path, name)

    def record_bin(self, instance_name: str, bin_bounds: BinBounds) -> None:
        """Records the active MOSFET `bin_bounds` for `instance_name`."""
        if self.differentiating:
            self.device_bins[instance_name] = bin_bounds

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

    def _select_body(self) -> list[Statement]:
        """Evaluates `.if` blocks, runs directives, and returns active elements."""
        body = self.subcircuit.body
        outside_conditions = {id(statement) for statement in unconditional(body)}
        elements: list[Statement] = []

        def check_condition(statement: Statement, condition: Expression) -> None:
            if self.statistics is not None:
                check_condition_deterministic(
                    condition, self.statistics, self.primal_scope, statement
                )

        selected = chosen(
            body,
            self.primal_scope,
            self.functions,
            on_condition=check_condition if self.statistics is not None else None,
        )
        for statement in selected:
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
        return elements

    def _resolve_conditional_parameters(self) -> None:
        """Resolves conditional `.param` statements into scope and active compilers."""
        if not self.parameters:
            return
        if self.differentiating:
            resolved_scope: dict[str, Scalar[Seed]] = dict(self.scope)
            resolve_into_dual(
                self.parameters,
                resolved_scope,
                self.dual_draws_for,
                self.functions,
            )
            self.scope = resolved_scope
        elif isinstance(self.scope, TrackedScope):
            resolve_into(
                self.parameters, self.scope, self.drawing.draws, self.functions
            )
        else:
            plain_scope = dict(self.primal_scope)
            resolve_into(
                self.parameters, plain_scope, self.drawing.draws, self.functions
            )
            self.scope = plain_scope
        active = self.statistics or self.probe_compiler
        if active is not None:
            active.resolve_into(self.parameters, self.primal_scope, self.functions)

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
        elements = self._select_body()
        self._resolve_conditional_parameters()
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
        statistical = (
            None
            if self.statistics is None or self._hoisted
            else self.statistics.finalize()
        )
        definition = ir.Component(
            ports=tuple(ports), instances=instances, statistical=statistical
        )
        component = Component(name, definition, tuple(self.children.values()))
        return Built(
            component,
            MappingProxyType(dict(self.nets)),
            self._initial_state(),
            temperature_dependent=self.is_temperature_dependent,
        )

    def translate_dual(self, name: str) -> DualBuilt[Seed]:
        """Translates the subcircuit returning `DualBuilt` with derivatives and bins."""
        built = self.translate(name)
        return DualBuilt(
            built,
            MappingProxyType(
                {
                    device_path: MappingProxyType(
                        {
                            parameter: MappingProxyType(dict(slopes))
                            for parameter, slopes in parameters.items()
                        }
                    )
                    for device_path, parameters in self.device_derivatives.items()
                }
            ),
            MappingProxyType(dict(self.device_bins)),
        )

    def _initial_state(self) -> InitialState:
        """Evaluates local ``.ic`` / ``.nodeset`` lines and gathers device ``IC=``."""
        initial_conditions = list(self.child_initial_conditions)
        nodesets = list(self.child_nodesets)
        for directive, statement in self.initial_statements:
            target = initial_conditions if directive == ".ic" else nodesets
            for node, voltage in parse_voltages(
                statement, self.primal_scope, self.functions
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

    def value(self, word: str) -> Scalar[Seed]:
        """Evaluates numeric literal or parameter expression `word` in current scope.

        Statistical calls sample using the active element's `self.draw` callback (or
        `self.dual_draw` when differentiating).

        Args:
            word: SPICE literal or parameter expression token to evaluate.

        Returns:
            Evaluated finite scalar or `Dual` value.

        Raises:
            SpiceError: When `word` evaluates to a non-finite value or derivative.
        """
        written = number(word)
        if written is not None:
            return written
        if not self.differentiating:
            resolved = evaluate(self.parse(word), self.primal_scope, self.draw)
            if not math.isfinite(resolved):
                message = f"`{word}` evaluates to a non-finite value ({resolved!r})"
                raise SpiceError(message)
            return resolved
        dual = evaluate_dual(self.parse(word), self.scope, self.dual_draw)
        if not is_finite(dual):
            message = (
                f"`{word}` evaluates to a non-finite value or derivative "
                f"({dual.value!r})"
            )
            raise SpiceError(message)
        return compact(dual)

    def value_symbolic(self, word: str) -> tuple[Scalar[Seed], Expression | None]:
        """Evaluates `word` and returns `(value, symbolic_expression)`.

        In dual differentiation mode, returns `(scalar_or_dual, None)`. Otherwise
        evaluates the nominal value and, when statistical compilation is active, lowers
        `word` through the element's active call counter.
        """
        if self.differentiating:
            return self.value(word), None
        written = number(word)
        if written is not None:
            if not math.isfinite(written):
                message = f"`{word}` evaluates to a non-finite value ({written!r})"
                raise SpiceError(message)
            return written, None
        parsed = self.parse(word)
        scope = self.primal_scope
        resolved = evaluate(parsed, scope, self.draw)
        if not math.isfinite(resolved):
            message = f"`{word}` evaluates to a non-finite value ({resolved!r})"
            raise SpiceError(message)
        symbolic = (
            None
            if self.statistics is None
            else self.statistics.lower_element(parsed, scope)
        )
        return resolved, symbolic

    def evaluate_expression(
        self,
        expression: Expression,
        *,
        draw_name: str | None = None,
    ) -> Scalar[Seed]:
        """Evaluates parsed `expression` in current scope in plain or dual mode."""
        resolved = inlined(expression, self.functions)
        if not self.differentiating:
            draw = self.draw if draw_name is None else self.drawing.draws(draw_name)
            return evaluate(resolved, self.primal_scope, draw)
        dual_draw = (
            self.dual_draw if draw_name is None else self.dual_draws_for(draw_name)
        )
        return compact(evaluate_dual(resolved, self.scope, dual_draw))

    def multiplied(
        self, instance: Mapping[str, Scalar[Seed]]
    ) -> dict[str, Scalar[Seed]]:
        """Applies the enclosing subcircuit multiplier `m` to `instance` parameters.

        Multiplies instance `m` by `self.multiplier` in all compatibility modes,
        matching ngspice HSPICE mode (``inpcom.c``, lines 4257 to 4283).

        Args:
            instance: Evaluated element-level parameter dictionary.

        Returns:
            Copy of `instance` with `m` scaled by `self.multiplier`.
        """
        if primal(self.multiplier) == 1.0 and not has_derivatives(self.multiplier):
            return dict(instance)
        return {**instance, "m": instance.get("m", 1.0) * self.multiplier}

    def multiplied_symbolic(
        self,
        instance: Mapping[str, Scalar[Seed]],
        instance_symbolic: Mapping[str, Expression],
    ) -> tuple[dict[str, Scalar[Seed]], dict[str, Expression]]:
        """Applies enclosing subcircuit multiplier `m` to value and symbolic maps."""
        values = self.multiplied(instance)
        symbolic = dict(instance_symbolic)
        multiplier = primal(self.multiplier)
        if multiplier == 1.0 and self.multiplier_symbolic is None:
            return values, symbolic
        own_value = primal(instance.get("m", 1.0))
        own_symbolic = instance_symbolic.get("m")
        if self.multiplier_symbolic is None:
            if own_symbolic is not None:
                symbolic["m"] = Binary("*", own_symbolic, Number(multiplier))
        elif own_symbolic is None:
            symbolic["m"] = (
                self.multiplier_symbolic
                if own_value == 1.0
                else Binary("*", Number(own_value), self.multiplier_symbolic)
            )
        else:
            symbolic["m"] = Binary("*", own_symbolic, self.multiplier_symbolic)
        return values, symbolic

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

    def add(  # noqa: PLR0913 - device placement fields and optional derivatives
        self,
        statement: Statement,
        model: str,
        parameters: Mapping[str, JsonValue],
        terminals: Mapping[str, str],
        *,
        identifier: str | None = None,
        derivatives: Mapping[str, Mapping[Seed, float]] | None = None,
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
            derivatives: Optional sparse partial derivatives keyed by dotted parameter
                path.
        """
        made = checked_device(statement, model, dict(parameters))
        instance_name = identifier or statement.text.split()[0]
        if self.differentiating and derivatives is not None:
            non_empty = {
                key: dict(slopes) for key, slopes in derivatives.items() if slopes
            }
            if non_empty:
                self.device_derivatives[instance_name] = non_empty
        self.instances.append(
            (
                self.order,
                ir.Instance(
                    name=instance_name,
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

    def add_dual(
        self,
        statement: Statement,
        model: str,
        parameters: Mapping[str, DualJsonValue[Seed]],
        terminals: Mapping[str, str],
        *,
        identifier: str | None = None,
    ) -> None:
        """Splits dual parameters into primal JSON and derivatives, then calls `add`."""
        primal_parameters, derivatives = models.split_dual_parameters(
            parameters, statement
        )
        self.add(
            statement,
            model,
            primal_parameters,
            terminals,
            identifier=identifier,
            derivatives=derivatives,
        )

    def _prepare_behavioral_parameters(
        self,
        element_name: str,
        expression: Expression,
        extra_parameters: Mapping[str, tuple[float, Expression]] | None,
        probe_extra_parameters: Mapping[str, float] | None,
    ) -> tuple[Expression, dict[str, float], dict[str, str] | None]:
        """Extracts stochastic behavioral parameters for statistics or probe mode."""
        if self.statistics is not None:
            rewritten, extracted, symbolic = self.statistics.extract_behavioral(
                element_name,
                expression,
                self.primal_scope,
                extra_parameters=extra_parameters,
            )
            return rewritten, extracted, symbolic or None
        if self.probe_compiler is not None:
            rewritten, extracted, symbolic = extract_probe_behavioral(
                expression,
                self.primal_scope,
                frozenset(self.probe_compiler.scope_symbols),
                extra_parameters=probe_extra_parameters,
            )
            return rewritten, extracted, symbolic or None
        return expression, {}, None

    def add_behavioral(  # noqa: PLR0913 - behavioral device placement and statistics
        self,
        statement: Statement,
        model: str,
        nodes: tuple[str, str],
        expression: Expression,
        *,
        parameters: Mapping[str, DualJsonValue[Seed]] | None = None,
        extra_parameters: Mapping[str, tuple[float, Expression]] | None = None,
        probe_extra_parameters: Mapping[str, float] | None = None,
    ) -> None:
        """Places a behavioral device of `model` between `nodes` using `expression`.

        Adds dedicated sense terminals for referenced node voltages and inserts series
        sensing ports for referenced source currents. When statistical compilation or
        mismatch probing is active, extracts stochastic parameter leaves as named
        behavioral device parameters (`p(<name>)`).

        Args:
            statement: Source SPICE element statement providing location metadata.
            model: Behavioral jolty device model name.
            nodes: Pair of `(positive_node, negative_node)` SPICE node tokens.
            expression: Inlined SPICE `Expression` governing the behavioral output.
            parameters: Optional extra device parameters merged alongside `expression`.
            extra_parameters: Optional `(nominal, step_expression)` map for synthesized
                stochastic factors (such as behavioral resistor copy scaling).
            probe_extra_parameters: Optional nominal map matching `extra_parameters`
                during `Probe` mismatch extraction.
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

        translated_expression, extracted_parameters, symbolic_names = (
            self._prepare_behavioral_parameters(
                statement.text.split()[0],
                expression,
                extra_parameters,
                probe_extra_parameters,
            )
        )
        text = translate(
            translated_expression,
            self.primal_scope,
            node,
            current,
            symbolic=symbolic_names,
        )
        try:
            check_core_depth(text)
        except SpiceError as error:
            raise statement.fail(str(error)) from None
        combined: dict[str, JsonValue] = {"expression": text}
        derivatives: dict[str, dict[Seed, float]] | None = None
        if parameters is not None:
            primal_parameters, derivatives = models.split_dual_parameters(
                parameters, statement
            )
            combined |= primal_parameters
        if extracted_parameters:
            combined["parameters"] = dict(extracted_parameters)
        self.add(statement, model, combined, terminals, derivatives=derivatives)

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
            winding_derivatives = [
                self.device_derivatives.pop(winding.name, {}) for winding in windings
            ]
            order, instance, coupling_expressions, coupled_derivatives = _coupled(
                group, names, windings, winding_derivatives
            )
            self.instances.append((order, instance))
            if self.statistics is not None:
                self.statistics.remap_coupled_inductors(
                    instance.name, names, windings, coupling_expressions
                )
            if self.differentiating and coupled_derivatives:
                self.device_derivatives[instance.name] = coupled_derivatives
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
        folded_name = name.casefold()
        self.draw = self.drawing.draws(folded_name)
        self.dual_draw = (
            self.dual_draws_for(folded_name) if self.differentiating else None
        )
        if self.statistics is not None:
            self.statistics.begin_element(folded_name)
        if self.uses is None:
            self._translated(statement)
            return

        def find_card(word: str) -> ModelCard | None:
            return self.library.card(word, self.models)

        model = needed_model(statement, find_card, self.primal_scope, self.functions)
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
        if (
            primal(self.multiplier) != 1.0 or self.multiplier_symbolic is not None
        ) and element_letter in UNMULTIPLIED:
            message = (
                f"jolty multiplies no {ELEMENT_KINDS[element_letter]} by its "
                f"subcircuit's multiplier `m`, here {primal(self.multiplier)!r}"
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
        self,
        card: ModelCard,
        instance: Mapping[str, Scalar[Seed]],
        statement: Statement,
        *,
        instance_symbolic: Mapping[str, Expression] | None = None,
    ) -> models.MadeDual[Seed]:
        """Translates model `card` and `instance` parameters into a `MadeDual` device.

        Statistical functions in `card` sample under keys ``<card_name>.<parameter>``.

        Args:
            card: Parsed SPICE `.model` card describing the device.
            instance: Evaluated element-level geometric, area, and multiplicity values.
            statement: Element statement used for error attribution.
            instance_symbolic: Optional symbolic expressions for stochastic instance
                parameters recorded when statistical compilation is active.

        Returns:
            Translated jolty `MadeDual` pairing model name, SI parameters, and exact
            derivatives (empty when not differentiating).
        """
        prefix = card.name.casefold()
        if self.differentiating:

            def dual_draws(name: str) -> DualDraw[Seed] | None:
                return self.dual_draws_for(f"{prefix}.{name}")

            return models.device_dual(
                card,
                self.scope,
                instance,
                statement,
                dual_draws,
                functions=self.functions,
            )

        def draws(name: str) -> Draw:
            return self.drawing.draws(f"{prefix}.{name}")

        primal_instance = {key: primal(value) for key, value in instance.items()}
        made = models.device(
            card,
            self.primal_scope,
            primal_instance,
            statement,
            draws,
            functions=self.functions,
        )
        if self.statistics is not None:
            self._record_device_statistics(
                card,
                primal_instance,
                instance_symbolic or {},
                made,
                statement,
            )
        return models.MadeDual(made.model, made.parameters, {})

    def _record_device_statistics(
        self,
        card: ModelCard,
        instance: Mapping[str, float],
        instance_symbolic: Mapping[str, Expression],
        made: models.Made,
        statement: Statement,
    ) -> None:
        """Records symbolic targets for a translated semiconductor or switch device."""
        if self.statistics is None:
            return
        prefix = card.name.casefold()
        scope = self.primal_scope
        card_symbolic: dict[str, Expression] = {}
        for parameter_name, expression in card.parameters.items():
            resolved = inlined(expression, self.functions)
            lowered = self.statistics.lower(
                resolved, scope, call_name=f"{prefix}.{parameter_name}"
            )
            if lowered is not None:
                card_symbolic[parameter_name] = lowered
        if not card_symbolic and not instance_symbolic:
            return
        card_values = {
            parameter_name: evaluate(
                inlined(expression, self.functions), scope, nominal
            )
            for parameter_name, expression in card.parameters.items()
        }
        record_device_targets(
            self.statistics,
            statement.text.split()[0],
            card,
            card_values=card_values,
            card_symbolic=card_symbolic,
            instance=instance,
            instance_symbolic=instance_symbolic,
            made=made,
            statement=statement,
        )

    def record_vbic_heating(
        self,
        card: ModelCard,
        instance: Mapping[str, Scalar[Seed]],
        instance_symbolic: Mapping[str, Expression],
        statement: Statement,
        *,
        heated: bool,
    ) -> None:
        """Records symbolic targets for 5-node VBIC self-heating `.rth`/`.cth`."""
        if self.statistics is None:
            return
        prefix = card.name.casefold()
        scope = self.primal_scope
        card_symbolic: dict[str, Expression] = {}
        card_values: dict[str, float] = {}
        for parameter_name in models.VBIC_SELF_HEATING:
            expression = card.parameters.get(parameter_name)
            if expression is None:
                continue
            resolved = inlined(expression, self.functions)
            lowered = self.statistics.lower(
                resolved, scope, call_name=f"{prefix}.{parameter_name}"
            )
            if lowered is not None:
                card_symbolic[parameter_name] = lowered
            card_values[parameter_name] = evaluate(resolved, scope, nominal)
        if card_symbolic or (
            heated and ("area" in instance_symbolic or "m" in instance_symbolic)
        ):
            primal_instance = {key: primal(value) for key, value in instance.items()}
            record_vbic_heating_targets(
                self.statistics,
                statement.text.split()[0],
                card,
                card_values=card_values,
                card_symbolic=card_symbolic,
                instance=primal_instance,
                instance_symbolic=instance_symbolic,
                heated=heated,
                statement=statement,
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


def _groups[Seed](couplings: list[Coupling[Seed]]) -> list[list[Coupling[Seed]]]:
    """Partitions `couplings` into connected components sharing inductor windings.

    Each returned group is sorted in expanded statement order.
    """
    groups: list[list[Coupling[Seed]]] = []
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


def _coupled_derivatives[Seed](
    coefficients: Sequence[Scalar[Seed]],
    winding_derivatives: Sequence[Mapping[str, Mapping[Seed, float]]],
    shared_keys: Mapping[str, JsonValue],
) -> dict[str, dict[Seed, float]]:
    """Remaps winding and coupling coefficient derivatives onto `coupled_inductors`."""
    derivatives: dict[str, dict[Seed, float]] = {}
    for coupling_index, coefficient in enumerate(coefficients):
        if isinstance(coefficient, Dual) and coefficient.derivatives:
            derivatives[f"couplings.{coupling_index}.coefficient"] = dict(
                coefficient.derivatives
            )
    for winding_index, entry in enumerate(winding_derivatives):
        for parameter, slopes in entry.items():
            if not slopes:
                continue
            if parameter in shared_keys:
                derivatives.setdefault(parameter, dict(slopes))
            else:
                derivatives[f"windings.{winding_index}.{parameter}"] = dict(slopes)
    return derivatives


def _coupled[Seed](
    group: list[Coupling[Seed]],
    names: list[str],
    windings: list[ir.Instance],
    winding_derivatives: Sequence[Mapping[str, Mapping[Seed, float]]],
) -> tuple[
    int,
    ir.Instance,
    list[tuple[int, Expression | None]],
    dict[str, dict[Seed, float]],
]:
    """Builds a `coupled_inductors` device, order, expressions, and derivatives.

    Combines `windings` (matching `names`) and preserves their shared instance
    temperature override.

    Raises:
        SpiceError: When a coupling self-couples an inductor, duplicates a winding
            pair, or joins windings with conflicting instance temperatures.
    """
    numbers = {name: index for index, name in enumerate(names, start=1)}
    pairs: dict[frozenset[int], tuple[Scalar[Seed], Expression | None]] = {}
    for each in group:
        for first, second in itertools.combinations(each.inductors, 2):
            pair = frozenset({numbers[first.casefold()], numbers[second.casefold()]})
            if len(pair) == 1:
                message = f"`{first}` is coupled with itself"
                raise each.statement.fail(message)
            if pair in pairs:
                message = f"`{first}` and `{second}` are coupled twice"
                raise each.statement.fail(message)
            pairs[pair] = (each.coefficient, each.coefficient_symbolic)
    couplings: list[JsonValue] = []
    coupling_expressions: list[tuple[int, Expression | None]] = []
    for index, (pair, (coefficient, symbolic)) in enumerate(pairs.items()):
        numbered: list[JsonValue] = [*sorted(pair)]
        couplings.append({"windings": numbered, "coefficient": primal(coefficient)})
        if symbolic is not None:
            coupling_expressions.append((index, symbolic))
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
    derivatives = _coupled_derivatives(
        [coefficient for coefficient, _ in pairs.values()],
        winding_derivatives,
        owns[0],
    )
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
    return first_coupling.order, instance, coupling_expressions, derivatives


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
