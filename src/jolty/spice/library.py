# Copyright 2026 Maxence Faldor
"""SPICE libraries read into components.

Each subcircuit becomes a component, and each of its elements a jolty device or a placed
component. A subcircuit's pins become its ports, in order, and a subcircuit that reaches
SPICE's global ground, node ``0``, gains a ground port named ``ground``. Each instance
records the file and the line of the statement that placed it.

A file read from disk includes others as ngspice does: ``.include`` the whole of a
file, ``.lib`` a section of one, which only ``.lib`` and ``.endl`` around it define; its
``.option scale`` scales its MOSFETs' sizes. A subcircuit's ``.if`` blocks hold the
statements of the branch its parameters choose. Its statistical functions, ``agauss``
and the others, take their nominal values, or draws of a Monte Carlo
[`Sample`][jolty.spice.Sample] for each placement, and ``temper`` is the
temperature the library reads at.
"""

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from jolty import _core, ir
from jolty.builder import Component
from jolty.errors import SpiceError
from jolty.spice import models
from jolty.spice._elements_active import ACTIVE_ELEMENTS
from jolty.spice._elements_passive import PASSIVE_ELEMENTS, TRANSMISSION_LINE
from jolty.spice._initial import (
    GROUND,
    INITIAL,
    Built,
    collect_node_voltages,
    parse_voltages,
    prefixed_inductors,
    remap_capacitor,
    resolve_node_voltages,
    solve_initial_voltages,
)
from jolty.spice._params import (
    FUNCTION,
    NO_FUNCTIONS,
    NO_MODELS,
    OPTIONS,
    PARAMETERS,
    STEP,
    Drawing,
    GlobalPlan,
    Probe,
    Sample,
    Subcircuit,
    build_global_plan,
    check_cycle,
    evaluate_stochastic,
    extract_multiplier,
    parse_assignments,
    parse_function,
    parse_options,
    parse_pins_and_parameters,
    resolve_into,
    standard_deviation,
)
from jolty.spice._preprocess import (
    CONDITIONS,
    expanded,
    expanded_text,
    keyword,
    unconditional,
)
from jolty.spice._pss import (
    PERIODIC_STEADY_STATE,
    IntegrationMethod,
    PeriodicSteadyStateCard,
    parse_periodic_steady_state,
)
from jolty.spice._survey import (
    BIN_SUFFIX,
    MULTIPLIED,
    SIZES,
    VERILOG_A,
    UnbinnedError,
    Use,
    bin_base,
    describe_card,
)
from jolty.spice._translate import (
    ElementTranslator,
    Translation,
    TranslationOptions,
)
from jolty.spice.expressions import (
    Expression,
    Function,
    UndefinedError,
    evaluate,
    inlined,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, tokens
from jolty.spice.ngspice import Compatibility

__all__ = [
    "CONDITIONS",
    "FUNCTION",
    "GROUND",
    "INITIAL",
    "MULTIPLIED",
    "OPTIONS",
    "PARAMETERS",
    "PERIODIC_STEADY_STATE",
    "SIZES",
    "TRANSMISSION_LINE",
    "VERILOG_A",
    "IntegrationMethod",
    "Library",
    "PeriodicSteadyStateCard",
    "Sample",
    "Subcircuit",
    "UnbinnedError",
    "Use",
    "load",
    "standard_deviation",
]

_MAX_CACHED_SCOPES: Final = 8
"""How many `(sample, temperature)` global scopes a `Library` caches at once."""

_ELEMENTS: Final[Mapping[str, ElementTranslator]] = MappingProxyType(
    {**PASSIVE_ELEMENTS, **ACTIVE_ELEMENTS}
)
"""How each element, by its letter, becomes devices or a placed component."""

type _BuiltState = tuple[
    float,
    dict[str, Function],
    tuple[tuple[str, float], ...],
    Compatibility | None,
]
"""Snapshot of mutable library state affecting nominal cached component builds."""


def _devices(placed: Component, prefix: str) -> dict[str, dict[str, float]]:
    """The numbers of every device `placed` places, within the components it places too.

    They come by the device's path, `prefix` then its path within `placed`, then by the
    number's path.
    """
    child_map = {child.name: child for child in placed.children}
    found: dict[str, dict[str, float]] = {}
    for instance in placed.definition.instances:
        path = f"{prefix}{instance.name}"
        match instance.of:
            case ir.DeviceOf():
                found[path] = ir.numbers(instance.parameters)
            case ir.ComponentOf(component=inner):
                found |= _devices(child_map[inner], f"{path}{ir.SEPARATOR}")
    return found


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

    def __init__(
        self,
        temperature: float = _core.REFERENCE_TEMPERATURE,
        compatibility: Compatibility | None = None,
    ) -> None:
        """An empty library, which reads ``temper`` at `temperature`, in kelvins.

        It reads its files as ngspice does in `compatibility`, its own unless given.
        """
        self._subcircuits: dict[str, Subcircuit] = {}
        self._models: dict[str, ModelCard] = {}
        self._bins: dict[str, tuple[ModelCard, ...]] | None = None
        self._parsed_models: dict[Statement, ModelCard] = {}
        self._cache_lock = threading.Lock()
        self._generation = 0
        self._parameters: list[tuple[str, Expression]] = []
        self._global_plan: GlobalPlan | None = None
        self._scopes: OrderedDict[tuple[Sample, float], dict[str, float]] = (
            OrderedDict()
        )
        self._built: dict[str, Built] = {}
        self._built_state: _BuiltState | None = None
        self.temperature = temperature
        self.compatibility = compatibility
        self.options: dict[str, float] = {}
        """The options its files set, such as ``scale``, by case-folded name."""
        self.functions: dict[str, Function] = {}
        """The user's functions its files define with ``.func``, by case-folded name,
        which its expressions call as SPICE's own."""
        self.initial_conditions: dict[str, float] = {}
        """The voltage, in volts, its files' ``.ic`` statements give each node, by name
        as they write it: where a plan's transient starts from, its
        `initial_conditions`."""
        self.nodesets: dict[str, float] = {}
        """The voltage, in volts, its files' ``.nodeset`` statements give each node, by
        name as they write it: where a plan's operating points start from, its options'
        `nodesets`."""
        self.periodic_steady_states: list[PeriodicSteadyStateCard] = []
        """The ``.pss`` periodic steady-state analyses its files define, in order."""
        self._uses: list[Use] | None = None

    @property
    def subcircuits(self) -> tuple[str, ...]:
        """The names of the subcircuits, in the order they are defined."""
        with self._cache_lock:
            return tuple(subcircuit.name for subcircuit in self._subcircuits.values())

    @property
    def models(self) -> tuple[str, ...]:
        """The names of the models defined outside subcircuits, in order.

        A binned model is named once, without its bins' numbers.
        """
        with self._cache_lock:
            cards = tuple(self._models.values())
        named = (BIN_SUFFIX.sub("", card.name) for card in cards)
        return tuple(dict.fromkeys(named))

    def read(self, text: str, file: str) -> None:
        """Adds the definitions of `text`, the contents of `file`.

        Any ``.include`` or ``.lib`` directive, outside or inside a subcircuit, expands
        in place relative to the directory of `file`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads, an included
                file is missing, a section is not in the file, or files include one
                another in a cycle.
        """
        found, _ = expanded_text(text, Path(file).parent, file, None, ())
        self._read(found)

    def include(self, path: Path, section: str | None = None) -> None:
        """Adds the definitions of the file at `path`, or of its section `section`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads, a file is
                missing, the section is not in the file, or files include one another
                in a cycle.
        """
        self._read(expanded(path, section, ()))

    def _read(self, found: Sequence[Statement]) -> None:
        """Adds the definitions of the statements `found`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads.
        """
        body: list[Statement] = []
        opened: Statement | None = None
        for statement in found:
            directive = keyword(statement)
            if opened is not None:
                if directive == ".ends":
                    self._define(opened, body)
                    opened = None
                    body = []
                elif directive == ".subckt":
                    message = "a subcircuit defined within another"
                    raise statement.fail(message)
                else:
                    body.append(statement)
            elif directive == ".subckt":
                opened = statement
            else:
                self._global(statement, directive)
        if opened is not None:
            message = "the subcircuit never ends"
            raise opened.fail(message)

    def _clear_built_locked(self) -> None:
        """Clears built components, parsed models and bin index under `_cache_lock`."""
        self._generation += 1
        self._built.clear()
        self._built_state = None
        self._parsed_models.clear()
        self._bins = None

    def _clear_caches_locked(self) -> None:
        """Clears all cached plans, scopes and builds under `_cache_lock`."""
        self._global_plan = None
        self._scopes.clear()
        self._clear_built_locked()

    def _invalidate_caches(self) -> None:
        """Clears cached global plans, drawn scopes, parsed models and built builds."""
        with self._cache_lock:
            self._clear_caches_locked()

    def _current_built_state_locked(self) -> _BuiltState:
        """Snapshots mutable library state affecting nominal cached component builds."""
        return (
            self.temperature,
            dict(self.functions),
            tuple(sorted(self.options.items())),
            self.compatibility,
        )

    def _read_options(self, statement: Statement) -> None:
        """Evaluates `.option` assignments and updates `self.options` under lock."""
        with self._cache_lock:
            functions_snapshot = dict(self.functions)
        parsed_options = {
            name: evaluate(inlined(value, functions_snapshot), {})
            for name, value in parse_options(statement).items()
        }
        for option_name, option_value in parsed_options.items():
            if not math.isfinite(option_value):
                message = f"option `{option_name}` is not finite ({option_value!r})"
                raise statement.fail(message)
        with self._cache_lock:
            self.options |= parsed_options
            self._clear_built_locked()

    def _global(self, statement: Statement, directive: str) -> None:
        """Adds the model, parameters or options `statement`, of `directive`, defines.

        Raises:
            SpiceError: When the statement defines none of them.
        """
        if directive == ".model":
            known: Mapping[str, ModelCard] = NO_MODELS
            if models.is_derived(statement):
                with self._cache_lock:
                    known = dict(self._models)
            card = models.read(statement, known)
            with self._cache_lock:
                self._models[card.name.casefold()] = card
                self._clear_built_locked()
        elif directive in PARAMETERS:
            assigned = parse_assignments(statement)
            with self._cache_lock:
                self._parameters.extend(assigned)
                self._clear_caches_locked()
        elif directive in OPTIONS:
            self._read_options(statement)
        elif directive == FUNCTION:
            name, function = parse_function(statement)
            with self._cache_lock:
                self.functions[name] = function
                self._clear_caches_locked()
        elif directive in INITIAL:
            given = self.initial_conditions if directive == ".ic" else self.nodesets
            with self._cache_lock:
                functions_snapshot = dict(self.functions)
            given |= parse_voltages(statement, self.values(), functions_snapshot)
        elif directive == PERIODIC_STEADY_STATE:
            with self._cache_lock:
                functions_snapshot = dict(self.functions)
            self.periodic_steady_states.append(
                parse_periodic_steady_state(
                    statement, self.values(), functions_snapshot
                )
            )
        elif directive != ".end":
            message = (
                "a library defines subcircuits, models, parameters, functions, "
                "options, initial conditions, nodesets and periodic steady states only"
            )
            raise statement.fail(message)

    def component(self, name: str, /, **parameters: float) -> Component:
        """The subcircuit `name` as a component, its statistical functions nominal.

        The `parameters` given replace its parameters' defaults.

        Raises:
            SpiceError: When no subcircuit is named `name`, or it holds what jolty
                cannot read.
        """
        return self._placed(name, Drawing(None, ""), parameters).component

    def drawn(
        self, name: str, sample: Sample, path: str, /, **parameters: float
    ) -> Component:
        """The subcircuit `name` as a component placed at `path`, in `sample`.

        Its statistical functions draw from the sample for the placement at `path`, a
        path of the circuit it is placed in, such as ``X1``; the library's own draw for
        the path ``""``. The `parameters` given replace its parameters' defaults.

        Raises:
            SpiceError: When no subcircuit is named `name`, or it holds what jolty
                cannot read.
        """
        return self._placed(name, Drawing(sample, path), parameters).component

    def _subcircuit_built(
        self, name: str, parameters: Mapping[str, float]
    ) -> tuple[Subcircuit, Built]:
        """Looks up `name` and returns its definition alongside its nominal build."""
        subcircuit = self._subcircuits.get(name.casefold())
        if subcircuit is None:
            message = f"no subcircuit is named `{name}`"
            raise SpiceError(message)
        return subcircuit, self._placed(name, Drawing(None, ""), parameters)

    def initial_conditions_of(
        self,
        name: str,
        path: str = "",
        /,
        *,
        operating_point: bool = True,
        connections: Mapping[str, str] | None = None,
        **parameters: float,
    ) -> dict[str, float]:
        """The initial node voltages of the subcircuit `name`, in volts, by net path.

        It gathers every ``.ic`` statement inside `name` and inside the subcircuits it
        instantiates, remapping subcircuit pins onto the nets they connect to and
        prefixing internal nets with the instance hierarchy. When `path` is empty,
        `name` is treated as the top circuit and top-level ``.ic`` statements
        (`self.initial_conditions`) are merged on top, with outer declarations
        overriding inner ones. When `path` is non-empty, internal nets are prefixed by
        ``f"{path}/"`` and optional `connections` map pins of `name` to external nets
        (``"0"`` or `GROUND` maps to ground).

        When `operating_point` is True (the default), device ``IC=`` parameters are
        omitted, matching SPICE when a transient solves its operating point. When
        `operating_point` is False (SPICE's ``UIC``), capacitor ``IC=`` voltages are
        solved alongside ``.ic`` into node voltages, and any non-zero inductor ``IC=``
        current is refused because `Transient.initial_conditions` accepts net voltages
        alone (inspect inductor currents via
        [`initial_currents_of`][jolty.spice.Library.initial_currents_of]).

        Raises:
            SpiceError: When no subcircuit is named `name`, `connections` names an
                unknown pin, an initial condition contradicts ground or another
                initial condition, or `operating_point` is False with a non-zero
                inductor initial current.
        """
        subcircuit, built = self._subcircuit_built(name, parameters)
        mapping, prefix, entries = collect_node_voltages(
            subcircuit,
            built,
            built.initial.initial_conditions,
            self.initial_conditions,
            path=path,
            connections=connections,
        )
        if operating_point:
            return resolve_node_voltages(entries, "an initial condition")
        for unknown_name, current in prefixed_inductors(
            built.initial.inductors, prefix
        ).items():
            if current != 0.0:
                message = (
                    f"inductor initial current `{unknown_name}` is {current!r} A; "
                    "`Transient.initial_conditions` accepts net voltages alone"
                )
                raise SpiceError(message)
        capacitors = tuple(
            remap_capacitor(capacitor, mapping, prefix)
            for capacitor in built.initial.capacitors
        )
        return solve_initial_voltages(entries, capacitors)

    def nodesets_of(
        self,
        name: str,
        path: str = "",
        /,
        *,
        connections: Mapping[str, str] | None = None,
        **parameters: float,
    ) -> dict[str, float]:
        """The nodeset voltages of the subcircuit `name`, in volts, by net path.

        It gathers every ``.nodeset`` statement inside `name` and inside the
        subcircuits it instantiates, remapping pins and prefixing internal nets as
        [`initial_conditions_of`][jolty.spice.Library.initial_conditions_of] does.
        When `path` is empty, top-level ``.nodeset`` statements (`self.nodesets`) are
        merged on top.

        Raises:
            SpiceError: When no subcircuit is named `name`, `connections` names an
                unknown pin, or a nodeset assigns a non-zero voltage to ground.
        """
        subcircuit, built = self._subcircuit_built(name, parameters)
        _, _, entries = collect_node_voltages(
            subcircuit,
            built,
            built.initial.nodesets,
            self.nodesets,
            path=path,
            connections=connections,
        )
        return resolve_node_voltages(entries, "a nodeset")

    def initial_currents_of(
        self, name: str, path: str = "", /, **parameters: float
    ) -> dict[str, float]:
        """The initial inductor currents (`IC=`) of the subcircuit `name`, in amperes.

        Each key is the engine unknown name of the inductor branch current, such as
        ``L1:current``, ``X1/L1:current``, or ``K1:current1`` for a coupled inductor
        winding, prefixed with ``f"{path}/"`` when `path` is non-empty. Currents are
        scaled by each inductor's parallel multiplier `M` and enclosing subcircuit
        multipliers.

        Raises:
            SpiceError: When no subcircuit is named `name`, or it holds what jolty
                cannot read.
        """
        built = self._placed(name, Drawing(None, ""), parameters)
        prefix = f"{path}{ir.SEPARATOR}" if path else ""
        return prefixed_inductors(built.initial.inductors, prefix)

    def random_variables(
        self, name: str, path: str, /, **parameters: float
    ) -> tuple[ir.RandomVariable, ...]:
        """The random variables of the subcircuit `name` placed at `path`.

        Each is a call of a statistical function that the placement's definitions make,
        within those it places too, and that moves its devices: its instance is the
        placement whose definitions make the call, such as `path` or ``X1/Xinner``, and
        its name the definition's name and the call's place among those it makes, such
        as ``vth0#0``, as a [`Sample`][jolty.spice.Sample] keys its draws; its mean is
        its nominal value, which [`component`][jolty.spice.Library.component] places,
        and its standard deviation as
        [`standard_deviation`][jolty.spice.standard_deviation] takes it. Each number of
        the devices' parameters that it moves has its derivative with respect to it, by
        the device's path in the circuit the subcircuit is placed in: a central
        difference of the reading, the cube root of the machine epsilon times its
        standard deviation either way, exact to rounding where the number is a
        polynomial of degree two or less in it, as every open kit writes it. A call
        whose standard deviation is 0, or that moves no device, is none; a call of the
        library's own parameters, which every placement shares, is no placement's. The
        `parameters` given replace the subcircuit's defaults, as `component` takes them.

        Raises:
            SpiceError: When no subcircuit is named `name`, it holds what jolty cannot
                read, or a call moves which devices or numbers it places.
        """
        given = {key.casefold(): value for key, value in parameters.items()}

        def read(probe: Probe) -> dict[str, dict[str, float]]:
            placed = self._placed(name, Drawing(probe, path), given).component
            return _devices(placed, f"{path}{ir.SEPARATOR}")

        recorded = Probe()
        nominal_numbers = read(recorded)
        found: list[ir.RandomVariable] = []
        for (scope, call), (function, arguments) in recorded.calls.items():
            spread = standard_deviation(function, arguments)
            if not scope or spread == 0:
                continue
            if not (math.isfinite(arguments[0]) and math.isfinite(spread)):
                message = (
                    f"the call `{call}` of `{scope}` has non-finite mean or standard "
                    f"deviation ({arguments[0]!r}, {spread!r})"
                )
                raise SpiceError(message)
            step = STEP * spread
            if not (step > 0 and math.isfinite(step)):
                continue
            up, down = (read(Probe((scope, call), sign * step)) for sign in (1.0, -1.0))
            if not up.keys() == down.keys() == nominal_numbers.keys() or any(
                up[device].keys() != down[device].keys() for device in up
            ):
                message = (
                    f"the call `{call}` of `{scope}` changes which devices or numbers "
                    f"`{name}` places, so it has no derivative"
                )
                raise SpiceError(message)
            derivatives = {
                device: moved
                for device, numbers in up.items()
                if (
                    moved := {
                        parameter_name: (value - down[device][parameter_name])
                        / (2 * step)
                        for parameter_name, value in numbers.items()
                        if value != down[device][parameter_name]
                    }
                )
            }
            if derivatives:
                found.append(
                    ir.RandomVariable(
                        instance=scope,
                        name=call,
                        mean=arguments[0],
                        standard_deviation=spread,
                        derivatives=derivatives,
                    )
                )
        return tuple(found)

    def survey(self, name: str, /, **parameters: float) -> tuple[Use, ...]:
        """Each element the subcircuit `name` holds, within those it places too.

        Elements are read as [`component`][jolty.spice.Library.component] reads them,
        with the `parameters` given, and each element states the model it needs and,
        where jolty cannot translate it, why. A placed subcircuit needs no model of its
        own: it states the parameters its placement gives it, and a problem of its
        placement, such as a parameter its subcircuit lacks; a problem of the
        subcircuit's own definitions, such as a parameter it cannot evaluate, is stated
        as a use of no element.
        """
        uses: list[Use] = []
        try:
            self._placed(name, Drawing(None, ""), parameters, uses=uses)
        except SpiceError as error:
            uses.append(Use("", "", str(error)))
        return tuple(uses)

    def survey_model(self, name: str) -> Use:
        """The use of the model `name`, or of its first bin, by a bare element.

        The element gives no parameters of its own, and the use is stated as
        [`survey`][jolty.spice.Library.survey] states an element's.

        Raises:
            SpiceError: When no model is named `name`.
        """
        card = self.card(name, {})
        if card is None:
            message = f"no model is named `{name}`"
            raise SpiceError(message)
        scope = self.values()
        with self._cache_lock:
            functions_snapshot = dict(self.functions)
        model = describe_card(card, scope, functions_snapshot)
        try:
            if card.kind in models.GEOMETRIC:
                read = models.geometric_values(
                    card, scope, functions=functions_snapshot
                )
                models.geometric(card, read, {}, card.statement)
            else:
                models.device(
                    card, scope, {}, card.statement, functions=functions_snapshot
                )
        except UndefinedError as error:
            return Use(card.name, model, str(error), error.name)
        except SpiceError as error:
            return Use(card.name, model, str(error))
        return Use(card.name, model)

    def _placed(
        self,
        name: str,
        drawing: Drawing,
        parameters: Mapping[str, float],
        *,
        uses: list[Use] | None = None,
    ) -> Built:
        """Builds a top-level placement of subcircuit `name` under `drawing`."""
        subcircuit = self._subcircuits.get(name.casefold())
        if subcircuit is None:
            message = f"no subcircuit is named `{name}`"
            raise SpiceError(message)
        if isinstance(drawing.sample, Probe):
            if drawing.sample.moved is None or drawing.sample.moved[0] == "":
                scope = self._values(Drawing(drawing.sample, ""))
            else:
                scope = self._scope(None)
        else:
            scope = self._scope(drawing.sample)
        given = {key.casefold(): value for key, value in parameters.items()}
        return self._build(subcircuit, scope, given, drawing, uses=uses)

    def _define(self, opened: Statement, body: list[Statement]) -> None:
        """Registers a `.subckt` definition and clears built component caches."""
        words = tokens(opened.text)
        if len(words) < 2:  # noqa: PLR2004 - `.subckt` and its name
            message = "a subcircuit has a name"
            raise opened.fail(message)
        name, rest = words[1], words[2:]
        pins, defaults = parse_pins_and_parameters(opened, rest)
        local_functions: dict[str, Function] = {}
        # ngspice moves a subcircuit's parameters onto its line, so an instance may
        # give any of them (`inp_reorder_params_subckt` in `inpcom.c`).
        for statement in unconditional(body):
            directive = keyword(statement)
            if directive in PARAMETERS:
                defaults |= dict(parse_assignments(statement))
            elif directive == FUNCTION:
                function_name, function = parse_function(statement)
                local_functions[function_name] = function
        subcircuit = Subcircuit(
            name,
            tuple(pins),
            defaults,
            tuple(body),
            opened,
            MappingProxyType(local_functions) if local_functions else NO_FUNCTIONS,
        )
        with self._cache_lock:
            self._subcircuits[name.casefold()] = subcircuit
            self._clear_built_locked()

    def values(self, sample: Sample | None = None) -> dict[str, float]:
        """The values of the parameters defined outside subcircuits, by name.

        Their names are case-folded, ``temper`` among them. Their statistical functions
        draw in `sample`, at the path ``""``, or take their nominal values without one.

        Raises:
            SpiceError: When parameters read one another in a cycle, or an expression is
                wrong.
        """
        return dict(self._scope(sample))

    def _plan(self) -> GlobalPlan:
        """Cached partition of global parameters for `(self.temperature, functions)`."""
        with self._cache_lock:
            cached = self._global_plan
            temperature = self.temperature
            functions_snapshot = dict(self.functions)
            if (
                cached is not None
                and cached.temperature == temperature
                and cached.functions == functions_snapshot
            ):
                return cached
            generation = self._generation
            parameters_snapshot = tuple(self._parameters)
        plan = build_global_plan(parameters_snapshot, temperature, functions_snapshot)
        with self._cache_lock:
            if (
                self._generation == generation
                and self.temperature == plan.temperature
                and self.functions == plan.functions
            ):
                if (
                    self._global_plan is None
                    or self._global_plan.temperature != plan.temperature
                    or self._global_plan.functions != plan.functions
                ):
                    self._global_plan = plan
                    self._scopes.clear()
                return self._global_plan
            return plan

    def _scope(self, sample: Sample | None) -> Mapping[str, float]:
        """Cached global parameter scope for `(sample, self.temperature)`.

        Deterministic parameters are evaluated once inside `GlobalPlan`; per-sample
        evaluation computes only the stochastic subset and its transitive dependents.
        """
        plan = self._plan()
        if sample is None or not plan.stochastic:
            return plan.nominal
        key = (sample, plan.temperature)
        with self._cache_lock:
            cached = self._scopes.get(key)
            if cached is not None:
                self._scopes.move_to_end(key)
                return cached
        values = evaluate_stochastic(plan, Drawing(sample, ""))
        with self._cache_lock:
            if self._global_plan is plan:
                self._scopes[key] = values
                self._scopes.move_to_end(key)
                while len(self._scopes) > _MAX_CACHED_SCOPES:
                    self._scopes.popitem(last=False)
        return values

    def _values(self, drawing: Drawing) -> dict[str, float]:
        """The values of the parameters defined outside subcircuits, drawn by `drawing`.

        Raises:
            SpiceError: When parameters read one another in a cycle, or an expression is
                wrong.
        """
        return evaluate_stochastic(self._plan(), drawing)

    def _build(  # noqa: PLR0913 - build context and optional survey output list
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
        drawing: Drawing,
        instantiating: tuple[str, ...] = (),
        *,
        uses: list[Use] | None = None,
    ) -> Built:
        """The built `subcircuit` with the `given` parameters, in `scope`.

        Statistical functions draw as `drawing` specifies. A multiplier `m` given to a
        subcircuit that has no parameter of that name multiplies every element it
        holds, as [`MULTIPLIED`][jolty.spice.library.MULTIPLIED] says.
        """
        key = check_cycle(subcircuit, instantiating)
        surveying = uses is not None
        if uses is not None and drawing.path:
            # A survey states each placement within, and what it gives.
            uses.append(Use(drawing.path, "", placed=(subcircuit.name, dict(given))))
        multiplier, given = extract_multiplier(subcircuit, given)
        unknown = sorted(set(given) - set(subcircuit.defaults))
        if unknown:
            message = f"`{subcircuit.name}` has no parameter `{unknown[0]}`"
            error = subcircuit.statement.fail(message)
            if uses is None:
                raise error
            # A survey states the placement's problem and reads on within it.
            uses.append(Use(drawing.path, "", str(error)))
            given = {
                key_name: value
                for key_name, value in given.items()
                if key_name not in unknown
            }
        defaults = {
            parameter: default
            for parameter, default in subcircuit.defaults.items()
            if parameter not in given
        }
        local = dict(scope)
        local.update(given)
        with self._cache_lock:
            library_functions = dict(self.functions)
        scoped_functions = (
            {**library_functions, **subcircuit.functions}
            if subcircuit.functions
            else library_functions
        )
        resolve_into(
            defaults,
            local,
            drawing.draws,
            scoped_functions,
            partial=surveying,
        )
        values = {
            parameter: local[parameter]
            for parameter in subcircuit.defaults
            if parameter in local
        }
        if multiplier != 1:
            # ngspice gives the subcircuit a parameter `m` that its elements read
            # (`inp_fix_subckt_multiplier` in `inpcom.c`, lines 4216 to 4230).
            local["m"] = values["m"] = multiplier
        listed = ",".join(f"{name}={value!r}" for name, value in values.items())
        name = f"{subcircuit.name}[{listed}]" if values else subcircuit.name
        name = drawing.named(name)
        options = TranslationOptions(
            drawing=drawing,
            elements=_ELEMENTS,
            multiplier=multiplier,
            parsed_models=self._parsed_models,
            lock=self._cache_lock,
            instantiating=(*instantiating, key),
            uses=uses,
        )
        if drawing.sample is not None or uses is not None:
            return Translation(self, subcircuit, (scope, local), options).translate(
                name
            )
        return self._translate_cached(subcircuit, scope, local, name, options)

    def _translate_cached(
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        local: dict[str, float],
        name: str,
        options: TranslationOptions,
    ) -> Built:
        """Returns the cached nominal build of `subcircuit`, translating on a miss."""
        with self._cache_lock:
            generation = self._generation
            state = self._current_built_state_locked()
            if self._built_state is None or self._built_state != state:
                self._built.clear()
                self._built_state = state
            cached = self._built.get(name)
            if cached is not None:
                return cached
        built = Translation(self, subcircuit, (scope, local), options).translate(name)
        with self._cache_lock:
            if (
                self._generation == generation
                and self._built_state == state
                and self._current_built_state_locked() == state
            ):
                return self._built.setdefault(name, built)
        return built

    def known_models(self, local: Mapping[str, ModelCard]) -> dict[str, ModelCard]:
        """All models visible inside a subcircuit, local shadowing global."""
        with self._cache_lock:
            return {**self._models, **local}

    def model(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """The model `name`, defined in a subcircuit's body or globally."""
        key = name.casefold()
        return local.get(key) or self._models.get(key)

    def _global_bins(self, base: str) -> tuple[ModelCard, ...]:
        """Returns cached global bin cards for case-folded base model name `base`."""
        with self._cache_lock:
            index = self._bins
            if index is not None:
                return index.get(base, ())
            generation = self._generation
            snapshot = tuple(self._models.items())
        grouped: dict[str, list[ModelCard]] = {}
        for key, card in snapshot:
            card_base = bin_base(key)
            if card_base is not None:
                grouped.setdefault(card_base, []).append(card)
        built = {model_name: tuple(cards) for model_name, cards in grouped.items()}
        with self._cache_lock:
            if self._generation == generation and self._bins is None:
                self._bins = built
            return (self._bins or built).get(base, ())

    def card(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """The model `name`, or its first bin, defined locally or globally."""
        found = self.model(name, local)
        if found is not None:
            return found
        base = name.casefold()
        for key, card in local.items():
            if bin_base(key) == base:
                return card
        bins = self._global_bins(base)
        return bins[0] if bins else None

    @property
    def surveying(self) -> bool:
        """Whether a [`survey`][jolty.spice.Library.survey] is being made."""
        return self._uses is not None

    def use(self, use: Use) -> None:
        """Records `use` of the survey being made."""
        if self._uses is not None:
            self._uses.append(use)

    def binned(
        self,
        name: str,
        local: Mapping[str, ModelCard],
        instance: Mapping[str, float],
        scope: Mapping[str, float],
        *,
        functions: Mapping[str, Function] = NO_FUNCTIONS,
    ) -> ModelCard | None:
        """The bin of the model `name` an `instance` takes, as ngspice selects it.

        A bin is a MOSFET's card named `name` then a dot and a whole number whose
        ``LMIN`` to ``LMAX`` holds the instance's length and ``WMIN`` to ``WMAX`` its
        width, divided among its [`fingers`][jolty.spice.Library.fingers], both given
        on its element, a bound counted in; the last defined is taken first, in a
        subcircuit's body, then globally.
        """
        if "l" not in instance or "w" not in instance:
            return None
        finger_count = self.fingers(instance)
        if not (finger_count > 0 and math.isfinite(finger_count)):
            message = (
                f"a MOSFET's fingers `NF` is positive and finite, not {finger_count!r}"
            )
            raise SpiceError(message)
        size = (instance["l"], instance["w"] / finger_count)
        base = name.casefold()
        for key, card in reversed(list(local.items())):
            if bin_base(key) == base and models.holds(
                card, scope, size, functions=functions
            ):
                return card
        for card in reversed(self._global_bins(base)):
            if models.holds(card, scope, size, functions=functions):
                return card
        return None

    def fingers(self, instance: Mapping[str, float]) -> float:
        """What a MOSFET's width is divided by where its bin is selected.

        As ngspice's ``INPgetModBin`` selects a bin (``inpgmod.c``, lines 268 to
        302), that is the instance's fingers, ``nf``, where it gives them and the
        library's option ``wnflag``, or else HSPICE's compatibility, asks for it; and
        1 otherwise. An instance's own ``wnflag``, which ngspice reads there too, only
        BSIM4's version 4.8 takes, which jolty does not model.
        """
        if "nf" not in instance:
            return 1.0
        hspice = 1.0 if self.compatibility == "hsa" else 0.0
        return instance["nf"] if self.options.get("wnflag", hspice) else 1.0

    def subcircuit(self, name: str) -> Subcircuit | None:
        """The subcircuit `name`."""
        return self._subcircuits.get(name.casefold())

    def place(  # noqa: PLR0913 - subcircuit placement state and optional survey list
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
        drawing: Drawing,
        instantiating: tuple[str, ...] = (),
        *,
        uses: list[Use] | None = None,
    ) -> Component:
        """The component a subcircuit instance with the `given` parameters places.

        Statistical functions draw as `drawing` specifies.
        """
        return self._build(
            subcircuit, scope, given, drawing, instantiating, uses=uses
        ).component

    def place_built(  # noqa: PLR0913 - subcircuit placement state and optional survey list
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
        drawing: Drawing,
        instantiating: tuple[str, ...] = (),
        *,
        uses: list[Use] | None = None,
    ) -> Built:
        """The built subcircuit (component, nets, initial state) for an instance."""
        return self._build(subcircuit, scope, given, drawing, instantiating, uses=uses)


def load(*paths: Path | str) -> Library:
    """The library of the SPICE files at `paths`, with the files they include.

    Raises:
        SpiceError: When a file holds what jolty cannot read.
    """
    library = Library()
    for path in map(Path, paths):
        library.include(path)
    return library
