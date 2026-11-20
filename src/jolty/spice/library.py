# Copyright 2026 Maxence Faldor
"""SPICE library loader translating subcircuits, models, and parameters into components.

Each subcircuit becomes a component, and each element inside it becomes a jolty device
or nested component placement. Subcircuit pins become ports in order, and any
subcircuit referencing SPICE global ground (node ``0``) gains a ground port named
``ground``. Each instance records the source file and line number of its statement.

Files loaded from disk resolve ``.include`` and ``.lib`` directives relative to the
including file, matching ngspice, and apply ``.option scale`` to MOSFET dimensions.
Conditional ``.if`` blocks select statements according to subcircuit parameters.
Statistical functions such as ``agauss`` evaluate at nominal values by default or draw
from a Monte Carlo [`Sample`][jolty.spice.Sample] per placement, while ``temper``
reflects the library temperature.
"""

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Self

from jolty import _core, ir
from jolty.builder import Component
from jolty.errors import JoltyError, SpiceError
from jolty.settings import Settings
from jolty.spice import models
from jolty.spice._cache import LibrarySnapshot, cache_key, read_cached, write_cached
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
    TrackedScope,
    build_global_plan,
    check_cycle,
    evaluate_stochastic,
    extract_multiplier,
    parse_assignments,
    parse_function,
    parse_options,
    parse_pins_and_parameters,
    resolve_into,
    retarget_global_plan,
    standard_deviation,
)
from jolty.spice._preprocess import (
    CONDITIONS,
    ManifestRecorder,
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
from jolty.spice._statistics import (
    GlobalSymbolic,
    StatisticalCompiler,
    build_global_symbolic,
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
    Name,
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
"""Maximum number of `(sample, temperature)` global scopes cached by a `Library`."""

_MAX_TEMPERATURE_BUILDS: Final = 256
"""Maximum number of `(name, temperature)` nominal builds cached by a `Library`."""

_ELEMENTS: Final[Mapping[str, ElementTranslator]] = MappingProxyType(
    {**PASSIVE_ELEMENTS, **ACTIVE_ELEMENTS}
)
"""Mapping from lowercase SPICE element letters to element translator callables."""

type _BuiltState = tuple[
    dict[str, Function],
    tuple[tuple[str, float], ...],
    Compatibility | None,
]
"""Snapshot of temperature-independent library state affecting cached nominal builds."""


@dataclass(frozen=True, slots=True)
class _IndexedBin:
    """Global binned MOSFET model card paired with optional static literal bounds."""

    card: ModelCard
    static_bounds: tuple[float, float, float, float] | None
    dynamic: bool


def _devices(placed: Component, prefix: str) -> dict[str, dict[str, float]]:
    """Collects numeric parameter dictionaries for every device inside `placed`.

    Keys are hierarchical device paths formed by prepending `prefix` to each path
    inside `placed`, mapping to numeric parameter dictionaries.
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
    r"""Collection of SPICE subcircuits, model cards, parameters, and functions.

    Each parsed subcircuit can be instantiated as a jolty component.

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
        """Initializes an empty library evaluating ``temper`` at `temperature` (K).

        File parsing follows ngspice `compatibility` mode when provided, or standard
        ngspice mode by default.
        """
        self._subcircuits: dict[str, Subcircuit] = {}
        self._models: dict[str, ModelCard] = {}
        self._bins: dict[str, tuple[_IndexedBin, ...]] | None = None
        self._parsed_models: dict[Statement, ModelCard] = {}
        self._cache_lock = threading.Lock()
        self._generation = 0
        self._parameters: list[tuple[str, Expression]] = []
        self._global_plan: GlobalPlan | None = None
        self._global_symbolic: tuple[GlobalPlan, GlobalSymbolic] | None = None
        self._scopes: OrderedDict[tuple[Sample, float], dict[str, float]] = (
            OrderedDict()
        )
        self._built: dict[str, Built] = {}
        self._built_by_temperature: OrderedDict[tuple[str, float], Built] = (
            OrderedDict()
        )
        self._built_state: _BuiltState | None = None
        self._has_evaluated_directives = False
        self.temperature = temperature
        self.compatibility = compatibility
        self.options: dict[str, float] = {}
        """Library `.option` values keyed by case-folded name (such as ``scale``)."""
        self.functions: dict[str, Function] = {}
        """Library `.func` definitions keyed by case-folded function name."""
        self.initial_conditions: dict[str, float] = {}
        """Top-level ``.ic`` node voltages in volts keyed by written node name."""
        self.nodesets: dict[str, float] = {}
        """Top-level ``.nodeset`` node voltages in volts keyed by written node name."""
        self.periodic_steady_states: list[PeriodicSteadyStateCard] = []
        """Parsed ``.pss`` periodic steady-state statements in source order."""
        self._uses: list[Use] | None = None

    @classmethod
    def _from_snapshot(
        cls,
        snapshot: LibrarySnapshot,
        temperature: float,
        compatibility: Compatibility | None,
    ) -> Self:
        """Constructs a fresh `Library` populated from a cached `LibrarySnapshot`."""
        library = cls(temperature, compatibility)
        library._subcircuits = {
            subcircuit.name.casefold(): subcircuit
            for subcircuit in snapshot.subcircuits
        }
        library._models = {card.name.casefold(): card for card in snapshot.models}
        library._parameters = list(snapshot.parameters)
        library.options = dict(snapshot.options)
        library.functions = dict(snapshot.functions)
        library.initial_conditions = dict(snapshot.initial_conditions)
        library.nodesets = dict(snapshot.nodesets)
        library.periodic_steady_states = list(snapshot.periodic_steady_states)
        library._has_evaluated_directives = snapshot.has_evaluated_directives
        return library

    def _snapshot(self) -> LibrarySnapshot:
        """Captures an immutable `LibrarySnapshot` of parsed definitions."""
        with self._cache_lock:
            return LibrarySnapshot(
                subcircuits=tuple(self._subcircuits.values()),
                models=tuple(self._models.values()),
                parameters=tuple(self._parameters),
                options=tuple(self.options.items()),
                functions=tuple(self.functions.items()),
                initial_conditions=tuple(self.initial_conditions.items()),
                nodesets=tuple(self.nodesets.items()),
                periodic_steady_states=tuple(self.periodic_steady_states),
                has_evaluated_directives=self._has_evaluated_directives,
                evaluated_temperature=(
                    self.temperature if self._has_evaluated_directives else None
                ),
            )

    @property
    def subcircuits(self) -> tuple[str, ...]:
        """Subcircuit names defined in the library in definition order."""
        with self._cache_lock:
            return tuple(subcircuit.name for subcircuit in self._subcircuits.values())

    @property
    def models(self) -> tuple[str, ...]:
        """Global model names defined outside subcircuits in definition order.

        Binned models appear once under their base name without numeric bin suffixes.
        """
        with self._cache_lock:
            cards = tuple(self._models.values())
        named = (BIN_SUFFIX.sub("", card.name) for card in cards)
        return tuple(dict.fromkeys(named))

    def read(self, text: str, file: str) -> None:
        """Adds the definitions of `text`, the contents of `file`.

        Any ``.include`` or ``.lib`` directive, outside or inside a subcircuit, expands
        in place relative to the directory of `file`.

        Args:
            text: SPICE library or netlist source text to parse.
            file: Source file path used for diagnostics and relative includes.

        Raises:
            SpiceError: When a statement is not a definition jolty reads, an included
                file is missing, a section is not in the file, or files include one
                another in a cycle.
        """
        found, _ = expanded_text(text, Path(file).parent, file, None, ())
        self._read(found)

    def include(self, path: Path, section: str | None = None) -> None:
        """Adds the definitions of the file at `path`, or of its section `section`.

        Args:
            path: Filesystem path to the SPICE file to include.
            section: Optional ``.lib`` section name to extract from `path`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads, a file is
                missing, the section is not in the file, or files include one another
                in a cycle.
        """
        self._include_recorded(path, section, None)

    def _include_recorded(
        self,
        path: Path,
        section: str | None,
        recorder: ManifestRecorder | None,
    ) -> None:
        """Includes `path` (optionally `section`) while recording read files."""
        self._read(expanded(path, section, (), recorder=recorder))

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
            message = "unterminated .subckt block (missing .ends)"
            raise opened.fail(message)

    def _clear_built_locked(self) -> None:
        """Clears built components, parsed models and bin index under `_cache_lock`."""
        self._generation += 1
        self._built.clear()
        self._built_by_temperature.clear()
        self._built_state = None
        self._parsed_models.clear()
        self._bins = None

    def _clear_caches_locked(self) -> None:
        """Clears all cached plans, scopes and builds under `_cache_lock`."""
        self._global_plan = None
        self._global_symbolic = None
        self._scopes.clear()
        self._clear_built_locked()

    def _invalidate_caches(self) -> None:
        """Clears cached global plans, drawn scopes, parsed models and built builds."""
        with self._cache_lock:
            self._clear_caches_locked()

    def _current_built_state_locked(self) -> _BuiltState:
        """Snapshots temperature-independent state affecting cached nominal builds."""
        return (
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
            self._has_evaluated_directives = True
            given = self.initial_conditions if directive == ".ic" else self.nodesets
            with self._cache_lock:
                functions_snapshot = dict(self.functions)
            given |= parse_voltages(statement, self.values(), functions_snapshot)
        elif directive == PERIODIC_STEADY_STATE:
            self._has_evaluated_directives = True
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
        """Builds subcircuit `name` as a component with nominal statistical values.

        Keyword `parameters` override the subcircuit's default parameter values.

        Args:
            name: Subcircuit name to instantiate (matched case-insensitively).
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Translated jolty `Component` evaluated at nominal parameter values.

        Raises:
            SpiceError: When `name` is undefined or contains unsupported SPICE syntax.
        """
        return self._placed(name, Drawing(None, ""), parameters).component

    def drawn(
        self, name: str, sample: Sample, path: str, /, **parameters: float
    ) -> Component:
        """Builds subcircuit `name` as a component placed at `path` under `sample`.

        Statistical functions inside `name` draw from `sample` for `path` (such as
        ``X1``), while global library parameters draw for path ``""``. Keyword
        `parameters` override the subcircuit's default parameter values.

        Args:
            name: Subcircuit name to instantiate (matched case-insensitively).
            sample: Monte Carlo sample supplying statistical draws per placement path.
            path: Hierarchical instance path identifying the placement in `sample`.
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Translated jolty `Component` evaluated with statistical draws from `sample`.

        Raises:
            SpiceError: When `name` is undefined or contains unsupported SPICE syntax.
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
        """Returns initial node voltages in volts for subcircuit `name` by net path.

        Gathers every ``.ic`` statement inside `name` and nested subcircuits, remapping
        subcircuit pins onto connected nets and prefixing internal nets with instance
        hierarchy. When `path` is empty, `name` acts as the top circuit and top-level
        `self.initial_conditions` override inner entries. When `path` is non-empty,
        internal nets receive the ``f"{path}/"`` prefix and `connections` maps pins to
        external nets (``"0"`` or `GROUND` maps to ground).

        When `operating_point` is True, device ``IC=`` parameters are omitted, matching
        SPICE transients that solve an operating point. When `operating_point` is False
        (SPICE ``UIC``), capacitor ``IC=`` voltages are solved alongside ``.ic``, while
        nonzero inductor ``IC=`` currents raise an error because
        `Transient.initial_conditions` accepts net voltages alone (use
        [`initial_currents_of`][jolty.spice.Library.initial_currents_of] for currents).

        Args:
            name: Subcircuit name whose initial conditions are gathered.
            path: Hierarchical placement path, or empty when `name` is the top circuit.
            operating_point: When `True`, omits capacitor ``IC=`` values and returns
                ``.ic`` node voltages alone.
            connections: Optional mapping from subcircuit pin names to external nets
                when `path` is non-empty.
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Mapping from circuit net paths to initial voltages in volts.

        Raises:
            SpiceError: When `name` is undefined, `connections` names an unknown pin,
                initial conditions conflict, or `operating_point` is False with a
                nonzero inductor initial current.
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
        """Returns nodeset voltages in volts for subcircuit `name` by net path.

        Gathers every ``.nodeset`` statement inside `name` and nested subcircuits,
        remapping pins and prefixing internal nets identically to
        [`initial_conditions_of`][jolty.spice.Library.initial_conditions_of]. When
        `path` is empty, top-level `self.nodesets` override inner declarations.

        Args:
            name: Subcircuit name whose nodesets are gathered.
            path: Hierarchical placement path, or empty when `name` is the top circuit.
            connections: Optional mapping from subcircuit pin names to external nets
                when `path` is non-empty.
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Mapping from circuit net paths to nodeset voltages in volts.

        Raises:
            SpiceError: When `name` is undefined, `connections` names an unknown pin,
                or a nodeset assigns a nonzero voltage to ground.
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
        """Returns initial inductor currents (`IC=`) in amperes for subcircuit `name`.

        Keys are engine unknown names such as ``L1:current``, ``X1/L1:current``, or
        ``K1:current1`` for coupled windings, prefixed with ``f"{path}/"`` when `path`
        is non-empty. Currents include each inductor's parallel multiplier `M` and
        enclosing subcircuit multipliers.

        Args:
            name: Subcircuit name whose inductor ``IC=`` currents are gathered.
            path: Hierarchical placement path prefixing returned unknown names.
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Mapping from inductor current unknown names to initial currents in amperes.

        Raises:
            SpiceError: When `name` is undefined or contains unsupported SPICE syntax.
        """
        built = self._placed(name, Drawing(None, ""), parameters)
        prefix = f"{path}{ir.SEPARATOR}" if path else ""
        return prefixed_inductors(built.initial.inductors, prefix)

    def random_variables(
        self, name: str, path: str, /, **parameters: float
    ) -> tuple[ir.RandomVariable, ...]:
        """Extracts placement random variables and parameter sensitivities for `name`.

        Inspects every statistical call made inside subcircuit `name` placed at `path`
        (and nested placements) that changes at least one device parameter. Each entry
        records its placement path, call identifier (such as ``vth0#0``), nominal mean,
        and standard deviation from
        [`standard_deviation`][jolty.spice.standard_deviation].

        Parameter derivatives are computed by central difference with step
        ``eps^(1/3) * sigma``, exact up to rounding for quadratic polynomials used by
        open PDKs. Calls with zero standard deviation, calls with no device effect, and
        shared global library parameters are excluded. Keyword `parameters` override
        subcircuit defaults.

        Args:
            name: Subcircuit name to inspect for statistical variation.
            path: Non-empty hierarchical placement path assigned to `name`.
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Random variables affecting device parameters within the placement.

        Raises:
            SpiceError: When `name` is undefined, contains unsupported syntax, or a
                statistical perturbation alters the set of emitted devices or keys.
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
        """Reports every element and translation diagnostic inside subcircuit `name`.

        Translates elements as [`component`][jolty.spice.Library.component] does with
        `parameters`, recording each element's required model and any translation error.
        Subcircuit placements report their passed parameters and placement errors.
        Subcircuit-level definition errors appear as a `Use` with an empty element path.

        Args:
            name: Subcircuit name to survey.
            **parameters: Numeric overrides for subcircuit formal parameters or `m`.

        Returns:
            Recorded element, placement, and definition `Use` entries in traversal
            order.
        """
        uses: list[Use] = []
        try:
            self._placed(name, Drawing(None, ""), parameters, uses=uses)
        except SpiceError as error:
            uses.append(Use("", "", str(error)))
        return tuple(uses)

    def survey_model(self, name: str) -> Use:
        """Reports translation compatibility for model `name` (or its first bin).

        Evaluates the model against an empty instance parameter map and formats any
        diagnostic identically to [`survey`][jolty.spice.Library.survey].

        Args:
            name: Global model or binned model base name to survey.

        Returns:
            `Use` entry recording the model description and any translation error.

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
        plan = self._plan()
        if isinstance(drawing.sample, Probe):
            if drawing.sample.moved is None or drawing.sample.moved[0] == "":
                scope: Mapping[str, float] = evaluate_stochastic(
                    plan, Drawing(drawing.sample, "")
                )
            else:
                scope = plan.nominal
        else:
            scope = self._scope_for_plan(plan, drawing.sample)
        given = {key.casefold(): value for key, value in parameters.items()}
        return self._build(
            subcircuit,
            scope,
            given,
            drawing,
            uses=uses,
            temperature=plan.temperature,
            temperature_dependent=plan.temperature_dependent,
        )

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
            MappingProxyType(defaults),
            tuple(body),
            opened,
            MappingProxyType(local_functions) if local_functions else NO_FUNCTIONS,
        )
        with self._cache_lock:
            self._subcircuits[name.casefold()] = subcircuit
            self._clear_built_locked()

    def values(self, sample: Sample | None = None) -> dict[str, float]:
        """Returns evaluated global library parameters keyed by case-folded name.

        Includes ``temper`` alongside parsed `.param` values. Statistical functions
        draw from `sample` at path ``""``, or use nominal values when `sample` is None.

        Args:
            sample: Optional Monte Carlo sample supplying global statistical draws.

        Returns:
            Evaluated global parameter dictionary keyed by case-folded name.

        Raises:
            SpiceError: When parameters form a dependency cycle or fail evaluation.
        """
        return dict(self._scope(sample))

    def _plan(self) -> GlobalPlan:
        """Returns the cached `GlobalPlan` for `(self.temperature, self.functions)`."""
        with self._cache_lock:
            cached = self._global_plan
            temperature = self.temperature
            functions_snapshot = dict(self.functions)
            if cached is not None and cached.functions == functions_snapshot:
                if cached.temperature == temperature:
                    return cached
                retargeted = retarget_global_plan(cached, temperature)
                if retargeted is not None:
                    self._global_plan = retargeted
                    return retargeted
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

    def _global_symbolic_plan(self) -> GlobalSymbolic:
        """Returns the cached `GlobalSymbolic` compiled from `self._plan()`."""
        plan = self._plan()
        with self._cache_lock:
            cached = self._global_symbolic
            if cached is not None and cached[0] is plan:
                return cached[1]
        symbolic = build_global_symbolic(plan)
        with self._cache_lock:
            if self._global_plan is plan:
                self._global_symbolic = (plan, symbolic)
        return symbolic

    def _scope(self, sample: Sample | None) -> Mapping[str, float]:
        """Returns the cached global parameter scope for `(sample, self.temperature)`.

        Deterministic parameters are evaluated once inside `GlobalPlan`; per-sample
        evaluation computes only the stochastic subset and its transitive dependents.
        """
        return self._scope_for_plan(self._plan(), sample)

    def _scope_for_plan(
        self, plan: GlobalPlan, sample: Sample | None
    ) -> Mapping[str, float]:
        """Returns the cached global scope for `(sample, plan.temperature)`."""
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
        """Evaluates global library parameters using statistical draws from `drawing`.

        Raises:
            SpiceError: When parameters form a dependency cycle or fail evaluation.
        """
        return evaluate_stochastic(self._plan(), drawing)

    def _scoped_functions(self, subcircuit: Subcircuit) -> dict[str, Function]:
        """Returns global and subcircuit-local user functions for `subcircuit`."""
        with self._cache_lock:
            library_functions = dict(self.functions)
        return (
            {**library_functions, **subcircuit.functions}
            if subcircuit.functions
            else library_functions
        )

    def _resolve_defaults(  # noqa: PLR0913 - subcircuit defaults resolution context
        self,
        defaults: Mapping[str, Expression],
        local: dict[str, float],
        drawing: Drawing,
        scoped_functions: Mapping[str, Function],
        *,
        surveying: bool,
        temperature_dependent: frozenset[str],
    ) -> bool:
        """Resolves omitted `defaults` into `local`; returns whether temperature read."""
        if not defaults:
            return False
        if drawing.sample is None and not surveying and temperature_dependent:
            tracked = TrackedScope(local, temperature_dependent)
            resolve_into(
                defaults,
                tracked,
                drawing.draws,
                scoped_functions,
                partial=False,
            )
            return tracked.read_temperature
        resolve_into(
            defaults,
            local,
            drawing.draws,
            scoped_functions,
            partial=surveying,
        )
        return False

    def _build_hoisted(  # noqa: PLR0913 - hoisted child translation context
        self,
        subcircuit: Subcircuit,
        scopes: tuple[Mapping[str, float], dict[str, float]],
        name: str,
        *,
        drawing: Drawing,
        multiplier: float,
        multiplier_symbolic: Expression | None,
        chain: tuple[str, ...],
        given: Mapping[str, float],
        symbolic_overrides: Mapping[str, Expression],
        defaults: Mapping[str, Expression],
        scoped_functions: Mapping[str, Function],
        parent_statistics: StatisticalCompiler,
        hoist_instance: str,
        temperature: float,
        temperature_dependent: frozenset[str],
    ) -> Built:
        """Translates a hoisted child placement directly into `parent_statistics`."""
        tag = parent_statistics.hoisted_tag(symbolic_overrides, multiplier_symbolic)
        hoisted_name = f"{name}~{tag}"
        saved = parent_statistics.child_state(hoist_instance, symbolic_overrides)
        try:
            parent_statistics.set_given(given, symbolic_overrides)
            if multiplier_symbolic is not None:
                parent_statistics.bind_symbol("m", multiplier_symbolic)
            parent_statistics.resolve_into(defaults, scopes[1], scoped_functions)
            options = TranslationOptions(
                drawing=drawing,
                elements=_ELEMENTS,
                multiplier=multiplier,
                parsed_models=self._parsed_models,
                lock=self._cache_lock,
                instantiating=chain,
                temperature=temperature,
                temperature_dependent=temperature_dependent,
                statistics=parent_statistics,
                multiplier_symbolic=multiplier_symbolic,
                hoisted=True,
            )
            return Translation(self, subcircuit, scopes, options).translate(
                hoisted_name
            )
        finally:
            parent_statistics.restore_state(saved)

    def _build_probed(  # noqa: PLR0913 - probed child translation context
        self,
        subcircuit: Subcircuit,
        scopes: tuple[Mapping[str, float], dict[str, float]],
        name: str,
        *,
        drawing: Drawing,
        multiplier: float,
        chain: tuple[str, ...],
        given: Mapping[str, float],
        probe_overrides: set[str],
        probe_multiplier_stochastic: bool,
        defaults: Mapping[str, Expression],
        scoped_functions: Mapping[str, Function],
        temperature: float,
        temperature_dependent: frozenset[str],
    ) -> Built:
        """Translates a subcircuit under `Probe`, extracting behavioral parameters."""
        probe_compiler = StatisticalCompiler(self._global_symbolic_plan())
        probe_compiler.set_given(given)
        for parameter_key in sorted(probe_overrides):
            probe_compiler.scope_symbols[parameter_key] = Name(
                probe_compiler.allocate_external_step()
            )
        probe_multiplier_expression: Expression | None = (
            Name(probe_compiler.allocate_external_step())
            if probe_multiplier_stochastic
            else None
        )
        if probe_multiplier_expression is not None:
            probe_compiler.scope_symbols["m"] = probe_multiplier_expression
        probe_compiler.resolve_into(defaults, scopes[1], scoped_functions)
        options = TranslationOptions(
            drawing=drawing,
            elements=_ELEMENTS,
            multiplier=multiplier,
            parsed_models=self._parsed_models,
            lock=self._cache_lock,
            instantiating=chain,
            temperature=temperature,
            temperature_dependent=temperature_dependent,
            probe_compiler=probe_compiler,
            multiplier_symbolic=probe_multiplier_expression,
        )
        return Translation(self, subcircuit, scopes, options).translate(name)

    def _filter_unknown_overrides(
        self,
        subcircuit: Subcircuit,
        given: Mapping[str, float],
        drawing: Drawing,
        uses: list[Use] | None,
    ) -> dict[str, float]:
        """Rejects unknown parameter overrides, or filters them during a survey."""
        unknown = sorted(set(given) - set(subcircuit.defaults))
        if not unknown:
            return dict(given)
        message = f"`{subcircuit.name}` has no parameter `{unknown[0]}`"
        error = subcircuit.statement.fail(message)
        if uses is None:
            raise error
        # A survey states the placement's problem and reads on within it.
        uses.append(Use(drawing.path, "", str(error)))
        return {key: value for key, value in given.items() if key not in unknown}

    def _build(  # noqa: PLR0913 - build context, survey list, and statistical hoisting
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
        """Builds `subcircuit` with `given` parameter overrides under `scope`.

        Statistical functions sample via `drawing`. A multiplier `m` supplied to a
        subcircuit without a formal `m` parameter scales eligible inner elements per
        [`MULTIPLIED`][jolty.spice.library.MULTIPLIED].
        """
        key = check_cycle(subcircuit, instantiating)
        if uses is not None and drawing.path:
            # A survey states each placement within, and what it gives.
            uses.append(Use(drawing.path, "", placed=(subcircuit.name, dict(given))))
        had_multiplier_override = "m" in given and "m" not in subcircuit.defaults
        multiplier, given = extract_multiplier(subcircuit, given)
        symbolic_overrides = {} if given_symbolic is None else dict(given_symbolic)
        multiplier_symbolic: Expression | None = (
            symbolic_overrides.pop("m", None) if had_multiplier_override else None
        )
        probe_overrides = set(given_probe_stochastic)
        probe_multiplier_stochastic = had_multiplier_override and "m" in probe_overrides
        if had_multiplier_override:
            probe_overrides.discard("m")
        given = self._filter_unknown_overrides(subcircuit, given, drawing, uses)
        symbolic_overrides = {
            name: expression
            for name, expression in symbolic_overrides.items()
            if name in given
        }
        probe_overrides &= set(given)
        defaults = {
            parameter: default
            for parameter, default in subcircuit.defaults.items()
            if parameter not in given
        }
        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_temperature_dependent = (
            self._plan().temperature_dependent
            if temperature_dependent is None
            else temperature_dependent
        )
        scoped_functions = self._scoped_functions(subcircuit)
        local = dict(scope)
        local.update(given)
        defaults_used_temperature = self._resolve_defaults(
            defaults,
            local,
            drawing,
            scoped_functions,
            surveying=uses is not None,
            temperature_dependent=resolved_temperature_dependent,
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
        elif multiplier_symbolic is not None:
            local["m"] = multiplier
        listed = ",".join(f"{name}={value!r}" for name, value in values.items())
        name = f"{subcircuit.name}[{listed}]" if values else subcircuit.name
        name = drawing.named(name)
        chain = (*instantiating, key)
        if (
            drawing.sample is None
            and uses is None
            and parent_statistics is not None
            and hoist_instance is not None
        ):
            built = self._build_hoisted(
                subcircuit,
                (scope, local),
                name,
                drawing=drawing,
                multiplier=multiplier,
                multiplier_symbolic=multiplier_symbolic,
                chain=chain,
                given=given,
                symbolic_overrides=symbolic_overrides,
                defaults=defaults,
                scoped_functions=scoped_functions,
                parent_statistics=parent_statistics,
                hoist_instance=hoist_instance,
                temperature=resolved_temperature,
                temperature_dependent=resolved_temperature_dependent,
            )
            if defaults_used_temperature and not built.temperature_dependent:
                return Built(
                    built.component,
                    built.nets,
                    built.initial,
                    temperature_dependent=True,
                )
            return built
        if isinstance(drawing.sample, Probe):
            return self._build_probed(
                subcircuit,
                (scope, local),
                name,
                drawing=drawing,
                multiplier=multiplier,
                chain=chain,
                given=given,
                probe_overrides=probe_overrides,
                probe_multiplier_stochastic=probe_multiplier_stochastic,
                defaults=defaults,
                scoped_functions=scoped_functions,
                temperature=resolved_temperature,
                temperature_dependent=resolved_temperature_dependent,
            )
        options = TranslationOptions(
            drawing=drawing,
            elements=_ELEMENTS,
            multiplier=multiplier,
            parsed_models=self._parsed_models,
            lock=self._cache_lock,
            instantiating=chain,
            uses=uses,
            temperature=resolved_temperature,
            temperature_dependent=resolved_temperature_dependent,
        )
        if drawing.sample is not None or uses is not None:
            return Translation(self, subcircuit, (scope, local), options).translate(
                name
            )
        body_temperature_keys = resolved_temperature_dependent - values.keys()
        built = self._translate_cached(
            subcircuit,
            scope,
            local,
            name,
            options,
            body_temperature_keys=body_temperature_keys,
            given=given,
            defaults=defaults,
            scoped_functions=scoped_functions,
        )
        if defaults_used_temperature and not built.temperature_dependent:
            return Built(
                built.component,
                built.nets,
                built.initial,
                temperature_dependent=True,
            )
        return built

    def _translate_cached(  # noqa: PLR0913 - nominal build context and compiler inputs
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        local: dict[str, float],
        name: str,
        options: TranslationOptions,
        *,
        body_temperature_keys: frozenset[str],
        given: Mapping[str, float],
        defaults: Mapping[str, Expression],
        scoped_functions: Mapping[str, Function],
    ) -> Built:
        """Returns the cached nominal build of `subcircuit`, translating on a miss."""
        temperature_key = (name, options.temperature)
        with self._cache_lock:
            generation = self._generation
            state = self._current_built_state_locked()
            if self._built_state is None or self._built_state != state:
                self._built.clear()
                self._built_by_temperature.clear()
                self._built_state = state
            cached = self._built.get(name)
            if cached is not None:
                return cached
            cached = self._built_by_temperature.get(temperature_key)
            if cached is not None:
                self._built_by_temperature.move_to_end(temperature_key)
                return cached
        compiler = StatisticalCompiler(self._global_symbolic_plan())
        compiler.set_given(given)
        compiler.resolve_into(defaults, local, scoped_functions)
        compiled_options = TranslationOptions(
            drawing=options.drawing,
            elements=options.elements,
            multiplier=options.multiplier,
            parsed_models=options.parsed_models,
            lock=options.lock,
            instantiating=options.instantiating,
            uses=options.uses,
            temperature=options.temperature,
            temperature_dependent=options.temperature_dependent,
            statistics=compiler,
        )
        body_scope: MutableMapping[str, float] = (
            TrackedScope(local, body_temperature_keys)
            if body_temperature_keys
            else local
        )
        built = Translation(
            self, subcircuit, (scope, body_scope), compiled_options
        ).translate(name)
        with self._cache_lock:
            if (
                self._generation == generation
                and self._built_state == state
                and self._current_built_state_locked() == state
            ):
                if built.temperature_dependent:
                    existing = self._built_by_temperature.get(temperature_key)
                    if existing is not None:
                        self._built_by_temperature.move_to_end(temperature_key)
                        return existing
                    self._built_by_temperature[temperature_key] = built
                    while len(self._built_by_temperature) > _MAX_TEMPERATURE_BUILDS:
                        self._built_by_temperature.popitem(last=False)
                    return built
                return self._built.setdefault(name, built)
        return built

    def known_models(self, local: Mapping[str, ModelCard]) -> dict[str, ModelCard]:
        """Returns all models visible inside a subcircuit, with `local` shadowing.

        Args:
            local: Subcircuit-local model cards keyed by case-folded name.

        Returns:
            Combined global and local model dictionary keyed by case-folded name.
        """
        with self._cache_lock:
            return {**self._models, **local}

    def model(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """Looks up model `name` in `local` scope first, then global library models.

        Args:
            name: Model name to look up (matched case-insensitively).
            local: Subcircuit-local model cards keyed by case-folded name.

        Returns:
            Matching `ModelCard`, or `None` when undefined in both scopes.
        """
        key = name.casefold()
        return local.get(key) or self._models.get(key)

    def _global_bins(self, base: str) -> tuple[_IndexedBin, ...]:
        """Returns cached indexed global bin cards for folded base model `base`."""
        with self._cache_lock:
            index = self._bins
            if index is not None:
                return index.get(base, ())
            generation = self._generation
            snapshot = tuple(self._models.items())
        grouped: dict[str, list[_IndexedBin]] = {}
        for key, card in snapshot:
            card_base = bin_base(key)
            if card_base is not None:
                is_static, bounds = models.static_bounds(card)
                grouped.setdefault(card_base, []).append(
                    _IndexedBin(card=card, static_bounds=bounds, dynamic=not is_static)
                )
        built = {model_name: tuple(cards) for model_name, cards in grouped.items()}
        with self._cache_lock:
            if self._generation == generation and self._bins is None:
                self._bins = built
            return (self._bins or built).get(base, ())

    def card(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """Looks up model `name` or its first bin in `local` then global scope.

        Args:
            name: Exact model name or binned model base name.
            local: Subcircuit-local model cards keyed by case-folded name.

        Returns:
            Exact `ModelCard` or first matching bin card, or `None` when absent.
        """
        found = self.model(name, local)
        if found is not None:
            return found
        base = name.casefold()
        for key, card in local.items():
            if bin_base(key) == base:
                return card
        bins = self._global_bins(base)
        return bins[0].card if bins else None

    @property
    def surveying(self) -> bool:
        """Flag indicating whether [`survey`][jolty.spice.Library.survey] is active."""
        return self._uses is not None

    def use(self, use: Use) -> None:
        """Records `use` of the survey being made.

        Args:
            use: Diagnostic or element usage record to append when surveying.
        """
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
        """Selects the binned MOSFET card for `name` matching `instance` dimensions.

        Searches cards named ``<name>.<n>`` whose ``LMIN``..``LMAX`` and
        ``WMIN``..``WMAX`` ranges contain `l` and ``w / fingers(instance)`` inclusive
        of boundaries. Checks local definitions in reverse definition order first,
        then global definitions in reverse order.

        Args:
            name: Base model name without numeric bin suffix.
            local: Subcircuit-local model cards keyed by case-folded name.
            instance: Evaluated MOSFET instance attributes including `l`, `w`, and `nf`.
            scope: Case-folded parameter values in scope for evaluating bin bounds.
            functions: Case-folded user-defined functions inlined before evaluation.

        Returns:
            Matching binned `ModelCard`, or `None` when no bin matches `instance`.
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
        for entry in reversed(self._global_bins(base)):
            if entry.dynamic:
                if models.holds(entry.card, scope, size, functions=functions):
                    return entry.card
            elif entry.static_bounds is not None and models.bounds_hold(
                entry.static_bounds, size
            ):
                return entry.card
        return None

    def fingers(self, instance: Mapping[str, float]) -> float:
        """Returns the width divisor applied when selecting a MOSFET bin.

        Matches ngspice ``INPgetModBin`` (``inpgmod.c``, lines 268 to 302): divides
        width by ``nf`` when `instance` provides ``nf`` and either ``.option wnflag``
        or HSPICE compatibility (`hsa`) is enabled, and returns 1 otherwise.
        Instance-level ``wnflag`` applies only to unmodeled BSIM4.8.

        Args:
            instance: Evaluated MOSFET instance parameter dictionary.

        Returns:
            Effective finger divisor applied to instance width during bin lookup.
        """
        if "nf" not in instance:
            return 1.0
        hspice = 1.0 if self.compatibility == "hsa" else 0.0
        return instance["nf"] if self.options.get("wnflag", hspice) else 1.0

    def subcircuit(self, name: str) -> Subcircuit | None:
        """Looks up subcircuit definition `name` by case-folded name.

        Args:
            name: Subcircuit name to look up.

        Returns:
            Parsed `Subcircuit` definition, or `None` when undefined.
        """
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
        """Builds a child `Component` for a placement of `subcircuit` with `given`.

        Statistical functions sample according to `drawing`.

        Args:
            subcircuit: Target subcircuit definition to instantiate.
            scope: Enclosing global parameter values in scope.
            given: Evaluated parameter overrides passed on the instance line.
            drawing: Statistical sampling context and hierarchical placement path.
            instantiating: Active chain of subcircuit names used to detect recursion.
            uses: Optional list collecting survey records rather than raising on errors.

        Returns:
            Translated child `Component` for the placement.
        """
        return self._build(
            subcircuit, scope, given, drawing, instantiating, uses=uses
        ).component

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
        """Builds a child `Built` placement (component, nets, and initial state).

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
        return self._build(
            subcircuit,
            scope,
            given,
            drawing,
            instantiating,
            uses=uses,
            temperature=temperature,
            temperature_dependent=temperature_dependent,
            given_symbolic=given_symbolic,
            parent_statistics=parent_statistics,
            hoist_instance=hoist_instance,
            given_probe_stochastic=given_probe_stochastic,
        )

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[tuple[Path, str | None]],
        *,
        temperature: float = _core.REFERENCE_TEMPERATURE,
        compatibility: Compatibility | None = None,
        cache: Path | None = None,
    ) -> Self:
        """Loads ordered `(path, section)` SPICE sources with persistent disk caching.

        Args:
            sources: Ordered `(path, section)` pairs to include into a new `Library`.
            temperature: Circuit temperature in kelvins assigned to SPICE ``temper``.
            compatibility: Optional ngspice compatibility mode applied to the library.
            cache: Optional root cache directory overriding `Settings.load().cache`.

        Returns:
            Populated `Library` containing all definitions read from `sources`.

        Raises:
            SpiceError: When any file is unreadable or contains unsupported syntax.
        """
        if not sources:
            return cls(temperature, compatibility)
        cache_directory = _resolve_cache_directory(cache)
        key = cache_key(sources, compatibility) if cache_directory is not None else None
        if cache_directory is not None and key is not None:
            snapshot = read_cached(
                cache_directory,
                key,
                compatibility=compatibility,
                temperature=temperature,
            )
            if snapshot is not None:
                return cls._from_snapshot(snapshot, temperature, compatibility)
        library = cls(temperature, compatibility)
        recorder = ManifestRecorder() if cache_directory is not None else None
        for path, section in sources:
            library._include_recorded(path, section, recorder)
        if (
            cache_directory is not None
            and key is not None
            and recorder is not None
            and recorder.consistent
        ):
            write_cached(
                cache_directory,
                key,
                library._snapshot(),
                recorder.files(),
                compatibility=compatibility,
            )
        return library


def _resolve_cache_directory(cache: Path | None) -> Path | None:
    """Returns `cache` when provided, or the configured settings cache directory."""
    if cache is not None:
        return cache
    try:
        return Settings.load().cache
    except JoltyError:
        return None


def load(
    *paths: Path | str,
    temperature: float = _core.REFERENCE_TEMPERATURE,
    compatibility: Compatibility | None = None,
    cache: Path | None = None,
) -> Library:
    """Loads SPICE files at `paths` and their transitive includes into a `Library`.

    Reuses a digest-validated parsed library cache inside `cache` (or the configured
    settings cache directory when omitted), invalidating automatically whenever any
    transitively included file changes.

    Args:
        *paths: Filesystem paths to SPICE library or netlist files loaded in order.
        temperature: Circuit temperature in kelvins assigned to SPICE ``temper``.
        compatibility: Optional ngspice compatibility mode applied to the library.
        cache: Optional root cache directory overriding `Settings.load().cache`.

    Returns:
        Populated `Library` containing all definitions read from `paths`.

    Raises:
        SpiceError: When any file is unreadable or contains unsupported SPICE syntax.
    """
    sources = tuple((Path(path), None) for path in paths)
    return Library.from_sources(
        sources,
        temperature=temperature,
        compatibility=compatibility,
        cache=cache,
    )
