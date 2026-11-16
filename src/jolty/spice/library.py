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

import itertools
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from jolty import _core, ir
from jolty.builder import Component
from jolty.devices import device
from jolty.errors import SpiceError
from jolty.ir import Role
from jolty.spice import models
from jolty.spice.expressions import (
    Binary,
    Call,
    Conditional,
    Current,
    Draw,
    Expression,
    Function,
    Number,
    Unary,
    UndefinedError,
    Voltage,
    evaluate,
    inlined,
    names,
    nominal,
    number,
    parse,
    translate,
)
from jolty.spice.models import ModelCard
from jolty.spice.netlist import Statement, statements, tokens
from jolty.spice.ngspice import Compatibility
from jolty.units import ZERO_CELSIUS, celsius

GROUND: Final = "ground"
"""The port through which a subcircuit reaches SPICE's global ground, node ``0``."""

PARAMETERS: Final = frozenset({".param", ".params"})
"""The statements that define parameters: ngspice reads ``.params`` as ``.param``."""

OPTIONS: Final = frozenset({".option", ".options"})
"""The statements that set options, such as ``scale``."""

FUNCTION: Final = ".func"
"""The statement that defines a user's function, which every expression may call."""

INITIAL: Final = frozenset({".ic", ".nodeset"})
"""The statements that give nodes' voltages to start from: a transient's initial
conditions, ``.ic``, and an operating point's nodesets, ``.nodeset``."""

CONDITIONS: Final = frozenset({".if", ".elseif", ".else", ".endif"})
"""The statements of a subcircuit's ``.if`` blocks."""


@dataclass(frozen=True, slots=True)
class Use:
    """An element of a subcircuit, the model it needs, and why jolty cannot read it."""

    element: str
    """Its path within the subcircuit, such as ``X1/Q1``; empty for the subcircuit's
    own definitions."""
    model: str
    """The model it needs, such as ``BSIM4 4.5 (level 54)``, or the element's kind."""
    problem: str | None = None
    """Why jolty cannot translate it, if it cannot."""
    undefined: str | None = None
    """The parameter it reads that has no value, which an instance gives, if that is
    why it cannot be translated."""
    middle: tuple[float, float] | None = None
    """The length and the width, in meters, at the middle of the first bin of the model
    it names, if it cannot be translated as no bin holds its size."""


@dataclass(frozen=True, slots=True)
class Sample:
    """A Monte Carlo sample, whose statistical functions draw from its seed and index.

    A call draws the standard variate the engine's generator of Monte Carlo samples
    draws for its seed, its index, the path of the placement whose definitions call it
    and which call it is, a parameter's name and the call's place among those its
    definition makes; so a sample draws the same alone or on any shard, and each
    placement its own.
    """

    seed: int
    index: int

    def draw(self, scope: str, name: str) -> Draw:
        """The draws of the statistical functions `name`'s definition calls in `scope`.

        Each draws as ngspice defines it (numparam's ``xpressn.c``), of a standard
        normal variate ``z`` or a uniform one ``u`` in ``[-1, 1)``: ``agauss(x, a, s)``
        is ``x + a / s * z``, ``gauss(x, r, s)`` is ``x + x * r / s * z``, each ``x``
        where ``a``, ``r`` or ``s`` is not positive, ``aunif(x, a)`` is ``x + a * u``,
        ``unif(x, r)`` is ``x + x * r * u``, and ``limit(x, a)`` is ``x + a`` where
        ``u`` is positive, ``x - a`` elsewhere.
        """
        calls = itertools.count()

        def drawn(function: str, arguments: tuple[float, ...]) -> float:
            call = f"{name}#{next(calls)}"
            nominal_value, spread, *rest = arguments
            if function in {"agauss", "gauss"}:
                sigma = rest[0]
                if spread <= 0 or sigma <= 0:
                    return nominal_value
                deviation = spread / sigma
                if function == "gauss":
                    deviation *= nominal_value
                normal = _core.variate("normal", self.seed, self.index, scope, call)
                return nominal_value + deviation * normal
            uniform = _core.variate("uniform", self.seed, self.index, scope, call)
            match function:
                case "aunif":
                    return nominal_value + spread * uniform
                case "unif":
                    return nominal_value + nominal_value * spread * uniform
            return nominal_value + (spread if uniform > 0 else -spread)

        return drawn


def standard_deviation(function: str, arguments: tuple[float, ...]) -> float:
    """The standard deviation of what the statistical function `function` draws.

    It draws of `arguments` as [`Sample.draw`][jolty.spice.Sample.draw] defines it:
    ``agauss(x, a, s)`` a normal number of standard deviation ``a / s``, and
    ``gauss(x, r, s)`` of ``|x| r / s``, each 0 where ``a``, ``r`` or ``s`` is not
    positive; ``aunif(x, a)`` a uniform one within ``|a|`` of ``x``, whose standard
    deviation is ``|a| / √3``, and ``unif(x, r)`` within ``|x r|``; ``limit(x, a)``
    ``x + a`` or ``x - a``, each half the time, so ``|a|``.

    Examples:
        >>> standard_deviation("agauss", (0.0, 0.03, 3.0))
        0.01
        >>> standard_deviation("aunif", (2.0, 0.3))
        0.17320508075688773
    """
    nominal_value, spread, *rest = arguments
    match function:
        case "agauss" | "gauss":
            sigma = rest[0]
            if spread <= 0 or sigma <= 0:
                return 0.0
            scale = abs(nominal_value) if function == "gauss" else 1.0
            return scale * spread / sigma
        case "aunif":
            return abs(spread) / math.sqrt(3)
        case "unif":
            return abs(nominal_value * spread) / math.sqrt(3)
    return abs(spread)


_STEP: Final = sys.float_info.epsilon ** (1 / 3)
"""The step of the central difference that differentiates a placement's devices with
respect to a random variable, per standard deviation of it: the cube root of the machine
epsilon, which balances the difference's truncation error, of the order of the step
squared, against its rounding error, of the order of the epsilon over the step, for a
number that varies on the scale of the random variable's standard deviation (W. H.
Press et al., *Numerical Recipes*, 3rd edition, 2007, §5.7)."""


@dataclass(slots=True)
class _Probe:
    """Statistical functions at their nominal values, but for one call, moved.

    The call `moved` takes its nominal value plus `offset`. Each call is recorded by
    where its definitions are and which call of them it is, as a
    [`Sample`][jolty.spice.Sample] keys its draws, with its function and its arguments.
    """

    moved: tuple[str, str] | None = None
    offset: float = 0.0
    calls: dict[tuple[str, str], tuple[str, tuple[float, ...]]] = field(
        default_factory=dict
    )

    def draw(self, scope: str, name: str) -> Draw:
        """The draws of the functions `name`'s definition calls in `scope`."""
        calls = itertools.count()

        def drawn(function: str, arguments: tuple[float, ...]) -> float:
            call = (scope, f"{name}#{next(calls)}")
            self.calls[call] = (function, arguments)
            nominal_value = arguments[0]
            return nominal_value + self.offset if call == self.moved else nominal_value

        return drawn


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
        self._parameters: list[tuple[str, Expression]] = []
        self._built: dict[str, Component] = {}
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
        self._uses: list[Use] | None = None

    @property
    def subcircuits(self) -> tuple[str, ...]:
        """The names of the subcircuits, in the order they are defined."""
        return tuple(subcircuit.name for subcircuit in self._subcircuits.values())

    @property
    def models(self) -> tuple[str, ...]:
        """The names of the models defined outside subcircuits, in order.

        A binned model is named once, without its bins' numbers.
        """
        named = (_BIN.sub("", card.name) for card in self._models.values())
        return tuple(dict.fromkeys(named))

    def read(self, text: str, file: str) -> None:
        """Adds the definitions of `text`, the contents of `file`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads.
        """
        self._read(statements(text, file))

    def include(self, path: Path, section: str | None = None) -> None:
        """Adds the definitions of the file at `path`, or of its section `section`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads, a file is
                missing, the section is not in the file, or files include one another
                in a cycle.
        """
        self._read(_expanded(path, section, ()))

    def _read(self, found: Sequence[Statement]) -> None:
        """Adds the definitions of the statements `found`.

        Raises:
            SpiceError: When a statement is not a definition jolty reads.
        """
        body: list[Statement] = []
        opened: Statement | None = None
        for statement in found:
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
            else:
                self._global(statement, keyword)
        if opened is not None:
            message = "the subcircuit never ends"
            raise opened.fail(message)

    def _global(self, statement: Statement, keyword: str) -> None:
        """Adds the model, parameters or options `statement`, of `keyword`, defines.

        Raises:
            SpiceError: When the statement defines none of them.
        """
        if keyword == ".model":
            card = models.read(statement, self._models)
            self._models[card.name.casefold()] = card
        elif keyword in PARAMETERS:
            self._parameters.extend(_assignments(statement))
        elif keyword in OPTIONS:
            self.options |= {
                name: evaluate(value, {}) for name, value in _options(statement).items()
            }
        elif keyword == FUNCTION:
            name, function = _function(statement)
            self.functions[name] = function
        elif keyword in INITIAL:
            given = self.initial_conditions if keyword == ".ic" else self.nodesets
            given |= _voltages(statement, self.values(), self.functions)
        elif keyword != ".end":
            message = (
                "a library defines subcircuits, models, parameters, functions, "
                "options, initial conditions and nodesets only"
            )
            raise statement.fail(message)

    def component(self, name: str, /, **parameters: float) -> Component:
        """The subcircuit `name` as a component, its statistical functions nominal.

        The `parameters` given replace its parameters' defaults.

        Raises:
            SpiceError: When no subcircuit is named `name`, or it holds what jolty
                cannot read.
        """
        return self._placed(name, _Drawing(None, ""), parameters)

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
        return self._placed(name, _Drawing(sample, path), parameters)

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

        def read(probe: _Probe) -> dict[str, dict[str, float]]:
            built, self._built = self._built, {}
            try:
                placed = self._placed(name, _Drawing(probe, path), given)
            finally:
                self._built = built
            return _devices(placed, f"{path}{ir.SEPARATOR}")

        recorded = _Probe()
        nominal_numbers = read(recorded)
        found: list[ir.RandomVariable] = []
        for (scope, call), (function, arguments) in recorded.calls.items():
            spread = standard_deviation(function, arguments)
            if not scope or spread == 0:
                continue
            step = _STEP * spread
            up, down = (
                read(_Probe((scope, call), sign * step)) for sign in (1.0, -1.0)
            )
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
                        number: (value - down[device][number]) / (2 * step)
                        for number, value in numbers.items()
                        if value != down[device][number]
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

        It is read as [`component`][jolty.spice.Library.component] reads it, with the
        `parameters` given, and each element states the model it needs and, where jolty
        cannot translate it, why. A placed subcircuit needs no model of its own, and
        states a problem of its placement, such as a parameter its subcircuit lacks; a
        problem of the subcircuit's own definitions, such as a parameter it cannot
        evaluate, is stated as a use of no element.
        """
        uses: list[Use] = []
        built, self._built, self._uses = self._built, {}, uses
        try:
            self._placed(name, _Drawing(None, ""), parameters)
        except SpiceError as error:
            uses.append(Use("", "", str(error)))
        finally:
            self._built, self._uses = built, None
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
        model = _described(card, scope)
        try:
            models.device(card, scope, {}, card.statement)
        except UndefinedError as error:
            return Use(card.name, model, str(error), error.name)
        except SpiceError as error:
            return Use(card.name, model, str(error))
        return Use(card.name, model)

    def _placed(
        self, name: str, drawing: _Drawing, parameters: Mapping[str, float]
    ) -> Component:
        subcircuit = self._subcircuits.get(name.casefold())
        if subcircuit is None:
            message = f"no subcircuit is named `{name}`"
            raise SpiceError(message)
        scope = self._values(_Drawing(drawing.sample, ""))
        given = {key.casefold(): value for key, value in parameters.items()}
        return self._component(subcircuit, scope, given, drawing)

    def _define(self, opened: Statement, body: list[Statement]) -> None:
        words = tokens(opened.text)
        if len(words) < 2:  # noqa: PLR2004 - `.subckt` and its name
            message = "a subcircuit has a name"
            raise opened.fail(message)
        name, rest = words[1], words[2:]
        pins, defaults = _pins_and_parameters(opened, rest)
        # ngspice moves a subcircuit's parameters onto its line, so an instance may
        # give any of them (`inp_reorder_params_subckt` in `inpcom.c`).
        for statement in _unconditional(body):
            if _keyword(statement) in PARAMETERS:
                defaults |= dict(_assignments(statement))
        subcircuit = Subcircuit(name, tuple(pins), defaults, tuple(body), opened)
        self._subcircuits[name.casefold()] = subcircuit

    def values(self, sample: Sample | None = None) -> dict[str, float]:
        """The values of the parameters defined outside subcircuits, by name.

        Their names are case-folded, ``temper`` among them. Their statistical functions
        draw in `sample`, at the path ``""``, or take their nominal values without one.

        Raises:
            SpiceError: When parameters read one another in a cycle, or an expression is
                wrong.
        """
        return self._values(_Drawing(sample, ""))

    def _values(self, drawing: _Drawing) -> dict[str, float]:
        """The values of the parameters defined outside subcircuits, drawn by `drawing`.

        Raises:
            SpiceError: When parameters read one another in a cycle, or an expression is
                wrong.
        """
        celsius_temperature = self.temperature - ZERO_CELSIUS
        return _resolved(
            dict(self._parameters),
            {"temper": celsius_temperature},
            drawing.draws,
            self.functions,
        )

    def _component(
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
        drawing: _Drawing,
    ) -> Component:
        """The component of `subcircuit` with the `given` parameters, in `scope`.

        Its statistical functions draw as `drawing` does. A multiplier `m` given to a
        subcircuit that has no parameter of that name multiplies every element it
        holds, as [`MULTIPLIED`][jolty.spice.library.MULTIPLIED] says.
        """
        multiplier = 1.0
        if "m" in given and "m" not in subcircuit.defaults:
            multiplier = given["m"]
            given = {key: value for key, value in given.items() if key != "m"}
        unknown = sorted(set(given) - set(subcircuit.defaults))
        if unknown:
            message = f"`{subcircuit.name}` has no parameter `{unknown[0]}`"
            error = subcircuit.statement.fail(message)
            if not self.surveying:
                raise error
            # A survey states the placement's problem and reads on within it.
            self.use(Use(drawing.path, "", str(error)))
            given = {key: value for key, value in given.items() if key not in unknown}
        defaults = {
            parameter: default
            for parameter, default in subcircuit.defaults.items()
            if parameter not in given
        }
        local = _resolved(
            defaults,
            {**scope, **given},
            drawing.draws,
            self.functions,
            partial=self.surveying,
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
        listed = ",".join(f"{key}={value!r}" for key, value in values.items())
        name = f"{subcircuit.name}[{listed}]" if values else subcircuit.name
        name = drawing.named(name)
        if name not in self._built:
            translation = _Translation(
                self, subcircuit, (scope, local), drawing, multiplier
            )
            self._built[name] = translation.component(name)
        return self._built[name]

    def model(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """The model `name`, defined in a subcircuit's body or globally."""
        key = name.casefold()
        return local.get(key) or self._models.get(key)

    def card(self, name: str, local: Mapping[str, ModelCard]) -> ModelCard | None:
        """The model `name`, or its first bin, defined locally or globally."""
        found = self.model(name, local)
        if found is not None:
            return found
        bins = _bins(name)
        for cards in (local, self._models):
            for key, card in cards.items():
                if bins.fullmatch(key):
                    return card
        return None

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
        size = (instance["l"], instance["w"] / self.fingers(instance))
        bins = _bins(name)
        for cards in (local, self._models):
            for key, card in reversed(list(cards.items())):
                if bins.fullmatch(key) and models.holds(card, scope, size):
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

    def place(
        self,
        subcircuit: Subcircuit,
        scope: Mapping[str, float],
        given: Mapping[str, float],
        drawing: _Drawing,
    ) -> Component:
        """The component a subcircuit instance with the `given` parameters places.

        Its statistical functions draw as `drawing` does.
        """
        return self._component(subcircuit, scope, given, drawing)


def load(*paths: Path | str) -> Library:
    """The library of the SPICE files at `paths`, with the files they include.

    Raises:
        SpiceError: When a file holds what jolty cannot read.
    """
    library = Library()
    for path in map(Path, paths):
        library.include(path)
    return library


_INCLUDES: Final = frozenset({".include", ".inc"})
"""The statements that include a file whole."""


def _expanded(
    path: Path, section: str | None, within: tuple[tuple[Path, str | None], ...]
) -> list[Statement]:
    """The statements of the file at `path`, or of its section `section`, expanded.

    As ngspice reads them, each ``.include`` is replaced by its file's statements and
    each ``.lib file section`` by that section's, a file's path read from the directory
    of the file that names it; a file read whole holds its statements outside its
    sections, which ``.lib name`` and ``.endl`` enclose. `within` lists the files and
    sections being read already, which include this one.

    Raises:
        SpiceError: When the file is missing, has no such section, or files include one
            another in a cycle.
    """
    reading = (path.resolve(), None if section is None else section.casefold())
    if reading in within:
        message = f"{path} includes itself, through {len(within)} files"
        raise SpiceError(message)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        message = f"cannot read {path}: {error.strerror}"
        raise SpiceError(message) from None
    found: list[Statement] = []
    opened: str | None = None
    seen = False
    for statement in statements(text, path.name):
        words = [word.strip("'\"") for word in tokens(statement.text)]
        keyword = words[0].casefold()
        if keyword == ".lib" and len(words) == 2:  # noqa: PLR2004 - `.lib` and a name
            if opened is not None:
                message = f"a section opens within section `{opened}`"
                raise statement.fail(message)
            opened = words[1].casefold()
            seen |= opened == reading[1]
        elif keyword == ".endl":
            opened = None
        elif opened == reading[1]:
            found.extend(_included(statement, words, path, (*within, reading)))
    if section is not None and not seen:
        message = f"{path} has no section `{section}`"
        raise SpiceError(message)
    return found


def _included(
    statement: Statement,
    words: Sequence[str],
    path: Path,
    within: tuple[tuple[Path, str | None], ...],
) -> list[Statement]:
    """`statement` of the file at `path`, of `words`, or what it includes, expanded.

    Raises:
        SpiceError: When it includes a file amiss.
    """
    keyword = words[0].casefold()
    if keyword == ".lib" and len(words) == 3:  # noqa: PLR2004 - a file and a section
        return _expanded(path.parent / words[1], words[2], within)
    if keyword in _INCLUDES and len(words) == 2:  # noqa: PLR2004 - and a file
        return _expanded(path.parent / words[1], None, within)
    if keyword in _INCLUDES or keyword == ".lib":
        message = "a file is included by its path, a section by the file's and name"
        raise statement.fail(message)
    return [statement]


def _options(statement: Statement) -> dict[str, Expression]:
    """The options an ``.option`` statement sets to values, by case-folded name.

    A flag it sets, a name without a value, changes nothing jolty reads.
    """
    words = tokens(statement.text)[1:]
    assigned = [
        position for position in range(len(words) - 2) if words[position + 1] == "="
    ]
    return models.assignments(
        statement,
        [word for position in assigned for word in words[position : position + 3]],
    )


@dataclass(frozen=True, slots=True)
class _Drawing:
    """How a placement's statistical functions draw.

    They draw of `sample`, at `path`, or take their nominal values without a sample, or
    as a probe sets them.
    """

    sample: Sample | _Probe | None
    path: str

    def draws(self, name: str) -> Draw:
        """The draws of the statistical functions of `name`'s definition."""
        if self.sample is None:
            return nominal
        return self.sample.draw(self.path, name)

    def named(self, name: str) -> str:
        """`name`, the placement's component's, told apart from another placement's.

        Without a sample, every placement of the same parameters places the same
        component.
        """
        match self.sample:
            case Sample(seed=seed, index=index):
                return f"{name}@{self.path}#{seed}:{index}"
            case _Probe():
                return f"{name}@{self.path}"
        return name

    def within(self, instance: str) -> _Drawing:
        """How the placement of `instance` within this one draws."""
        path = f"{self.path}/{instance}" if self.path else instance
        return _Drawing(self.sample, path)


def _devices(placed: Component, prefix: str) -> dict[str, dict[str, float]]:
    """The numbers of every device `placed` places, within the components it places too.

    They come by the device's path, `prefix` then its path within `placed`, then by the
    number's path.
    """
    children = {child.name: child for child in placed.children}
    found: dict[str, dict[str, float]] = {}
    for instance in placed.definition.instances:
        path = f"{prefix}{instance.name}"
        match instance.of:
            case ir.DeviceOf():
                found[path] = ir.numbers(instance.parameters)
            case ir.ComponentOf(component=inner):
                found |= _devices(children[inner], f"{path}{ir.SEPARATOR}")
    return found


def _resolved(
    definitions: Mapping[str, Expression],
    scope: Mapping[str, float],
    draws: Callable[[str], Draw],
    functions: Mapping[str, Function],
    *,
    partial: bool = False,
) -> dict[str, float]:
    """`scope` with each parameter `definitions` defines, evaluated.

    Each definition calls the user's `functions` as SPICE's own.

    Each parameter is evaluated once those it reads are, whatever the order of their
    definitions, as ngspice sorts them, its statistical functions drawn by `draws` of
    its name. When `partial`, a parameter that reads a name nothing defines is left
    out, as ngspice leaves a subcircuit's default unevaluated where its instance gives
    the parameter.

    Raises:
        SpiceError: When parameters read one another in a cycle, or an expression is
            wrong.
    """
    values = dict(scope)
    pending = {
        name: inlined(expression, functions) for name, expression in definitions.items()
    }
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
            try:
                values[name] = evaluate(pending.pop(name), values, draws(name))
            except UndefinedError:
                if not partial:
                    raise
    return values


_FUNCTION: Final = re.compile(
    r"\.func\s+(?P<name>[a-z_][a-z0-9_]*)\s*\((?P<parameters>[^()]*)\)\s*=?\s*"
    r"(?P<body>.+)",
    re.IGNORECASE | re.DOTALL,
)
"""A ``.func`` statement: ``.func name(a, b) {body}``, its body in braces or quotes
or bare, after an ``=`` or not."""


def _function(statement: Statement) -> tuple[str, Function]:
    """The name, case-folded, and the definition of the function `statement` defines.

    Raises:
        SpiceError: When the statement is not a function's definition, or its body not
            an expression.
    """
    found = _FUNCTION.fullmatch(statement.text.strip())
    if found is None:
        message = "a function is defined as `.func name(parameters) {expression}`"
        raise statement.fail(message)
    name, written, text = found.group("name", "parameters", "body")
    parameters = tuple(
        word.strip().casefold() for word in str(written).split(",") if word.strip()
    )
    try:
        body = parse(str(text))
    except SpiceError as error:
        raise statement.fail(str(error)) from None
    return str(name).casefold(), Function(parameters, body)


def _voltages(
    statement: Statement,
    scope: Mapping[str, float],
    functions: Mapping[str, Function],
) -> dict[str, float]:
    """The voltage each node of an ``.ic`` or ``.nodeset`` statement is given, by name.

    Each is written ``V(node)=value``, the value a number or an expression of the
    parameters `scope` holds, which may call the user's `functions`.

    Raises:
        SpiceError: When the statement gives anything but a node's voltage.
    """
    words = tokens(statement.text)[1:]
    found: dict[str, float] = {}
    for index in range(0, len(words), 3):
        written = words[index : index + 3]
        match written:
            case [voltage, "=", value] if (node := _node_of(voltage)) is not None:
                given = number(value)
                found[node] = (
                    evaluate(inlined(parse(value), functions), scope)
                    if given is None
                    else given
                )
            case _:
                message = (
                    f"a node's voltage is written `V(node)=value`, not "
                    f"`{' '.join(written)}`"
                )
                raise statement.fail(message)
    return found


def _node_of(written: str) -> str | None:
    """The node `written`, ``V(node)``, names, or None when it names no one node."""
    if written[:2].casefold() != "v(" or not written.endswith(")"):
        return None
    inside = written[2:-1].strip()
    return inside if inside and "," not in inside else None


def _assignments(statement: Statement) -> list[tuple[str, Expression]]:
    """The parameters a ``.param`` statement assigns, in order.

    As ngspice reads them, a value runs to the name of the next assignment, so it may
    be an expression written with spaces, as in ``.param a = 1 + b c = 2``.

    Raises:
        SpiceError: When the words do not pair names with values.
    """
    words = tokens(statement.text)[1:]
    signs = [position for position, word in enumerate(words) if word == "="]
    if not signs or signs[0] != 1:
        message = "a parameter is assigned a value, `name = value`"
        raise statement.fail(message)
    ends = [*(sign - 1 for sign in signs[1:]), len(words)]
    found: list[tuple[str, Expression]] = []
    for sign, end in zip(signs, ends, strict=True):
        written = " ".join(words[sign + 1 : end])
        if not written:
            message = f"`{words[sign - 1]}` is not assigned a value"
            raise statement.fail(message)
        found.append((words[sign - 1].casefold(), models.value(written)))
    return found


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
        drawing: _Drawing,
        multiplier: float = 1.0,
    ) -> None:
        self.library = library
        self.subcircuit = subcircuit
        self.globals, self.scope = scopes
        self.drawing = drawing
        self.multiplier = multiplier
        """The subcircuit's multiplier, which multiplies each element it holds."""
        self.draw: Draw = nominal
        """The draws of the statistical functions of the element being translated."""
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
        """The component, named `name`.

        Its ``.if`` blocks choose by its parameters, those defined outside them.
        """
        body = self.subcircuit.body
        unconditional = {id(statement) for statement in _unconditional(body)}
        elements: list[Statement] = []
        for statement in _chosen(body, self.scope):
            if id(statement) in unconditional and _keyword(statement) in PARAMETERS:
                continue
            if statement.text.startswith("."):
                self._define(statement)
            else:
                elements.append(statement)
        self.scope = _resolved(
            self.parameters, self.scope, self.drawing.draws, self.library.functions
        )
        for statement in elements:
            if _letter(statement) in _SENSED:
                self._source(statement)
        for statement in elements:
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
        if ir.SEPARATOR in node:
            message = (
                f"a node named `{node}`, whose `{ir.SEPARATOR}` separates the levels "
                "of jolty's paths"
            )
            raise self.subcircuit.statement.fail(message)
        return self.nets.setdefault(key, node)

    def value(self, word: str) -> float:
        """The value `word` writes: a number, or an expression of the parameters.

        Its statistical functions draw as those of the element being translated.
        """
        written = number(word)
        if written is not None:
            return written
        return evaluate(self.parse(word), self.scope, self.draw)

    def multiplied(self, instance: Mapping[str, float]) -> dict[str, float]:
        """An element's `instance` parameters, its copies `m` times the subcircuit's.

        ngspice multiplies them so in HSPICE's compatibility, and in its others gives
        the element the subcircuit's alone (``inpcom.c``, lines 4257 to 4283); jolty
        multiplies them in each, as copies of copies are.
        """
        if self.multiplier == 1:
            return dict(instance)
        return {**instance, "m": instance.get("m", 1.0) * self.multiplier}

    def parse(self, text: str) -> Expression:
        """The expression `text` writes, each call of the user's functions written out.

        Raises:
            SpiceError: When `text` is not an expression, or calls a function that
                calls itself.
        """
        return inlined(parse(text), self.library.functions)

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
        keyword = _keyword(statement)
        if keyword == ".model":
            card = models.read(statement, self.models)
            self.models[card.name.casefold()] = card
        elif keyword in PARAMETERS:
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
                            "positive": moved[instance.name],
                        }
                    }
                )
                if instance.name in moved
                else instance,
            )
            for line, instance in self.instances
        ]

    def _heat(self) -> None:
        """Makes each resistor and capacitor on a thermal node a thermal element.

        ngspice's nodes have no nature: a device's thermal node holds a temperature
        rise as its voltage and takes a power as its current, so a resistor or a
        capacitor between such nodes, or one and the ground, is a thermal resistance in
        kelvins per watt or a thermal capacitance in joules per kelvin, of its value. A
        thermal node is one a heated device's `thermal` terminal joins, or a thermal
        element's.

        Raises:
            SpiceError: When such an element joins an electrical node too, or drifts
                with temperature or voltage.
        """
        elements = {ir.DeviceOf(model=model) for model, _ in _THERMAL_ELEMENTS.values()}
        thermal = {
            net
            for _, instance in self.instances
            for terminal, net in instance.terminals.items()
            if instance.of in elements
            or (instance.parameters.get("thermal") is True and terminal == "thermal")
        }
        heated: list[tuple[int, ir.Instance]] = []
        for line, instance in self.instances:
            made = _THERMAL_ELEMENTS.get(instance.of)
            nets = set(instance.terminals.values())
            if made is None or not nets & thermal:
                heated.append((line, instance))
                continue
            model, parameter = made
            location = instance.source_location
            place = f"{location.file}:{location.line}: " if location else ""
            if not nets <= thermal | {GROUND}:
                message = f"{place}`{instance.name}` joins a thermal node to another"
                raise SpiceError(message)
            if set(instance.parameters) != {parameter}:
                message = (
                    f"{place}`{instance.name}`, on a thermal node, drifts with "
                    "temperature or voltage, which no thermal element does"
                )
                raise SpiceError(message)
            value = {parameter: instance.parameters[parameter]}
            replaced = {
                "of": ir.DeviceOf(model=model),
                "parameters": device(model, value).parameters,
            }
            heated.append((line, instance.model_copy(update=replaced)))
        self.instances = heated

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
            instance.name.casefold(): (line, instance)
            for line, instance in self.instances
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
            self.instances.append(_coupled(group, names, windings))
            coupled |= set(names)
        self.instances = [
            (line, instance)
            for line, instance in self.instances
            if instance.name.casefold() not in coupled
        ]

    def _source(self, statement: Statement) -> None:
        words = tokens(statement.text)
        if len(words) < 3:  # noqa: PLR2004 - a name and two nodes
            message = "a source joins two nodes"
            raise statement.fail(message)
        self.senses[words[0].casefold()] = _Sense(words[0], self.net(words[1]))

    def _element(self, statement: Statement) -> None:
        """Translates the element `statement`, or, in a survey, states its use.

        Raises:
            SpiceError: When jolty cannot translate it, outside a survey.
        """
        name = statement.text.split()[0]
        self.draw = self.drawing.draws(name.casefold())
        if not self.library.surveying:
            self._translated(statement)
            return
        model = _needed(self, statement)
        path = self.drawing.within(name).path
        try:
            self._translated(statement)
        except UndefinedError as error:
            self.library.use(Use(path, model, str(error), error.name))
        except UnbinnedError as error:
            self.library.use(Use(path, model, str(error), middle=error.middle))
        except SpiceError as error:
            self.library.use(Use(path, model, str(error)))
        else:
            if model:
                self.library.use(Use(path, model))

    def _translated(self, statement: Statement) -> None:
        name = statement.text.split()[0]
        if ir.SEPARATOR in name:
            message = (
                f"an element named `{name}`, whose `{ir.SEPARATOR}` separates the "
                "levels of jolty's paths"
            )
            raise statement.fail(message)
        letter = _letter(statement)
        if self.multiplier != 1 and letter in _UNMULTIPLIED:
            message = (
                f"jolty multiplies no {_ELEMENT_KINDS[letter]} by its subcircuit's "
                f"multiplier `m`, here {self.multiplier!r}"
            )
            raise statement.fail(message)
        translate = _ELEMENTS.get(letter)
        if translate is None:
            kind = _UNREAD_ELEMENTS.get(letter)
            named = "" if kind is None else f", {kind}"
            message = f"jolty has no element of letter `{letter.upper()}`{named}"
            raise statement.fail(message)
        translate(self, statement)

    def device(
        self, card: ModelCard, instance: Mapping[str, float], statement: Statement
    ) -> models.Made:
        """The device `card` makes for an element of `instance` parameters.

        Its statistical functions draw for each of the card's parameters, by the card's
        name and the parameter's.
        """
        prefix = card.name.casefold()
        return models.device(
            card,
            self.scope,
            instance,
            statement,
            lambda name: self.drawing.draws(f"{prefix}.{name}"),
        )


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
        name=first.text.split()[0],
        of=ir.DeviceOf(model="coupled_inductors"),
        parameters=device("coupled_inductors", parameters).parameters,
        terminals={
            f"{end}{number}": winding.terminals[end]
            for number, winding in enumerate(windings, start=1)
            for end in ("a", "b")
        },
        source_location=ir.SourceLocation(file=first.file, line=first.line),
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


VERILOG_A: Final = "Verilog-A"
"""What a survey names the model of an ``N`` element by, before its module's name: a
model ngspice loads compiled from Verilog-A, through OSDI."""

_BIN: Final = re.compile(r"\.\d+$")
"""A bin's number after its model's name, as in ``nmos_3p3.12``."""


def _bins(name: str) -> re.Pattern[str]:
    """The case-folded names of the bins of the model `name`: it, a dot and a number."""
    return re.compile(rf"{re.escape(name.casefold())}{_BIN.pattern}")


_FAMILIES: Final = {
    "nmos": "MOSFET",
    "pmos": "MOSFET",
    "npn": "bipolar transistor",
    "pnp": "bipolar transistor",
    "d": "diode",
    "njf": "JFET",
    "pjf": "JFET",
    "r": "resistor's model card",
    "c": "capacitor's model card",
    "l": "inductor's model card",
}
"""What a model card of each type models, by its type."""

_LEVELS: Final = {
    "MOSFET": {
        level: name
        for levels, name in (
            (models.BSIM3_LEVELS, "BSIM3"),
            (models.BSIM4_LEVELS, "BSIM4"),
            (models.BSIM_SOI_LEVELS, "BSIM-SOI"),
            (models.HISIM2_LEVELS, "HiSIM2"),
            (models.HISIM_HV_LEVELS, "HiSIM_HV"),
        )
        for level in levels
    },
    "bipolar transistor": {
        1: "Gummel-Poon",
        **{int(level): "VBIC" for level in models.VBIC_LEVELS},
    },
}
"""ngspice's models of a MOSFET and a bipolar transistor named apart, by their cards'
levels, of which ngspice reads level 1 by default (``inpdomod.c``)."""

_ELEMENT_KINDS: Final = {
    "r": "resistor",
    "c": "capacitor",
    "l": "inductor",
    "v": "voltage source",
    "i": "current source",
    "e": "controlled source",
    "f": "controlled source",
    "g": "controlled source",
    "h": "controlled source",
    "b": "behavioral source",
    "k": "mutual inductance",
    "t": "transmission line",
}
"""What an element of a letter is when it names no model card."""

MULTIPLIED: Final = frozenset("rcldjmqx")
"""The elements, by letter, that a subcircuit's multiplier `m` multiplies, which a
subcircuit is given where it has no parameter of that name.

ngspice multiplies each of its elements but its V, E, H, A, K, N, O, P, S, T, U, W and Y
elements and its B sources of a voltage (``inpcom.c``, lines 4240 to 4256): a
resistor's and an inductor's value divided by it, a capacitor's multiplied, and a
diode's, a transistor's and a subcircuit instance's copies `m` multiplied. jolty does
not multiply sources of a current, I, F, G and B elements, yet, and refuses them under a
multiplier other than 1."""

_UNMULTIPLIED: Final = frozenset("ifg")
"""The elements, by letter, that ngspice multiplies by a subcircuit's multiplier and
jolty does not yet: sources of a current, but for B sources, which say so themselves."""

_UNREAD_ELEMENTS: Final = {
    "o": "ngspice's lossy transmission line, LTRA",
    "u": "ngspice's uniform RC line, URC",
    "y": "ngspice's lossy transmission line, TXL",
    "p": "ngspice's coupled transmission lines, CPL",
}
"""The elements ngspice has and jolty reads not yet, by letter, and what each is."""


def _needed(translation: _Translation, statement: Statement) -> str:
    """The model the element `statement` needs, as a reader states it.

    A card's model is named by its type and level, and its version where it gives one,
    such as ``BSIM4 4.5 (level 54)``; an ``N`` element's is the Verilog-A module its
    card names; an element of no card is named by its kind, and a placed subcircuit
    needs nothing of its own, its elements' models being stated apart.
    """
    letter = _letter(statement)
    if letter == "x":
        return ""
    words = tokens(statement.text)[1:]
    found = next(
        (
            card
            for word in words
            if word != "="
            and (card := translation.library.card(word, translation.models)) is not None
        ),
        None,
    )
    if found is None:
        kind = _ELEMENT_KINDS.get(letter, f"element `{letter.upper()}`")
        reads = any("v(" in word.casefold() for word in words)
        return f"{kind} whose value reads a voltage" if reads else kind
    if letter == "n":
        return f"{VERILOG_A} `{found.kind}`"
    return _described(found, translation.scope)


def _described(card: ModelCard, scope: Mapping[str, float]) -> str:
    """The model `card` makes, by its type and level, and its version if it gives one.

    A MOSFET, a bipolar transistor and a diode are named with their levels, such as
    ``BSIM4 4.5 (level 54)``, the level evaluated in `scope`.
    """
    family = _FAMILIES.get(card.kind, f"`{card.kind}`")
    if family not in {*_LEVELS, "diode"}:
        return family
    written = card.parameters.get("level")
    level = 1.0 if written is None else evaluate(written, scope)
    named = _LEVELS.get(family, {}).get(int(level), family)
    version = None if card.version is None else number(card.version)
    given = f" {version:g}" if version is not None else ""
    return f"{named}{given} (level {level:g})"


def _letter(statement: Statement) -> str:
    """The letter of an element's statement, which says what kind of element it is."""
    return statement.text[0].casefold()


def _keyword(statement: Statement) -> str:
    """A statement's first word, case-folded, such as ``.param``."""
    return statement.text.split()[0].casefold()


def _unconditional(body: Sequence[Statement]) -> list[Statement]:
    """The statements of `body` outside its ``.if`` blocks."""
    depth = 0
    found: list[Statement] = []
    for statement in body:
        keyword = _keyword(statement)
        if keyword == ".if":
            depth += 1
        elif keyword == ".endif":
            depth -= 1
        elif depth == 0 and keyword not in CONDITIONS:
            found.append(statement)
    return found


@dataclass(slots=True)
class _Block:
    """An ``.if`` block being read: whether its branch is chosen, whether one was."""

    chosen: bool
    taken: bool


def _chosen(body: Sequence[Statement], scope: Mapping[str, float]) -> list[Statement]:
    """The statements of `body` its ``.if`` blocks choose, with `scope`'s values.

    A block holds the statements of its first branch whose condition, an expression
    in parentheses, is not zero, ``.if`` then each ``.elseif``, or else of its
    ``.else``, if any; blocks nest.

    Raises:
        SpiceError: When a branch or an end opens no block, or a block never ends.
    """
    blocks: list[_Block] = []
    found: list[Statement] = []
    for statement in body:
        keyword = _keyword(statement)
        within = all(block.chosen for block in blocks)
        if keyword in CONDITIONS - {".if"} and not blocks:
            message = f"`{keyword}` within no `.if` block"
            raise statement.fail(message)
        match keyword:
            case ".if":
                held = within and _condition(statement, scope)
                blocks.append(_Block(chosen=held, taken=held))
            case ".elseif" | ".else":
                block = blocks[-1]
                outer = all(each.chosen for each in blocks[:-1])
                held = (
                    outer
                    and not block.taken
                    and (keyword == ".else" or _condition(statement, scope))
                )
                block.chosen, block.taken = held, block.taken or held
            case ".endif":
                blocks.pop()
            case _ if within:
                found.append(statement)
    if blocks:
        message = "an `.if` block never ends"
        raise body[-1].fail(message)
    return found


def _condition(statement: Statement, scope: Mapping[str, float]) -> bool:
    """Whether the condition of an ``.if`` or ``.elseif`` statement holds in `scope`."""
    _, _, condition = statement.text.partition(" ")
    return evaluate(parse(condition), scope) != 0


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
    coefficients, ``VC1`` and ``VC2``. Its copies in parallel, its ``M`` times its
    subcircuit's multiplier, divide a resistor's and an inductor's value and multiply a
    capacitor's, and a resistor's flicker noise is theirs, ``m · KF · (|I| / m)^AF``,
    as ngspice adds it (``resnoise.c``, line 109).
    """
    letter = kind[0]
    curved = frozenset(_CURVATURE) if model != "inductor" else frozenset()
    drifts = frozenset(_DRIFT) if model == "resistor" else frozenset({"tc1", "tc2"})
    drifts |= curved
    noisy = frozenset({"kf", "af"}) if model == "resistor" else frozenset()

    def translated(translation: _Translation, statement: Statement) -> None:
        positional, assigned = _keywords(tokens(statement.text))
        _check_keywords(
            statement,
            assigned,
            frozenset({letter, "m", "tc", "temp", "dtemp", *drifts}),
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
        if _reads_unknowns(translation.parse(values[0])):
            message = (
                f"jolty reads no {model} whose value reads a voltage or a current, "
                "which ngspice makes a behavioral source"
            )
            raise statement.fail(message)
        value = translation.value(values[0])
        own = {
            name: translation.value(words[0])
            for name, words in assigned.items()
            if name == "m" and words
        }
        copies = translation.multiplied(own).get("m", 1.0)
        value = value * copies if model == "capacitor" else value / copies
        given: dict[str, float] = {}
        if card is not None:
            drawing, prefix = translation.drawing, card.name.casefold()
            scale = card.parameters.get(letter, Number(1.0))
            value *= evaluate(
                scale, translation.scope, drawing.draws(f"{prefix}.{letter}")
            )
            read = (
                models.NOMINAL_TEMPERATURES | models.OWN_TEMPERATURES | drifts | noisy
            )
            _check_model(card, translation.scope, read | {letter}, statement)
            given = {
                name: evaluate(
                    expression, translation.scope, drawing.draws(f"{prefix}.{name}")
                )
                for name, expression in card.parameters.items()
                if name in read
            }
        given |= _coefficients(translation, statement, assigned, drifts)
        temperatures = {
            name: translation.value(words[0])
            for name, words in assigned.items()
            if name in {"temp", "dtemp"} and words
        }
        parameters: dict[str, JsonValue] = {parameter: value}
        parameters |= _temperatures(given)
        parameters |= models.own_temperature(given, temperatures, statement)
        parameters |= {
            _CURVATURE[name]: amount
            for name, amount in given.items()
            if name in _CURVATURE
        }
        if given.get("kf", 0.0) != 0:
            given["kf"] *= copies ** (1 - given.get("af", 1.0))
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


def _reads_unknowns(expression: Expression) -> bool:
    """Whether `expression` reads a voltage or a current."""
    match expression:
        case Voltage() | Current():
            return True
        case Unary(_, operand):
            return _reads_unknowns(operand)
        case Binary(_, left, right):
            return _reads_unknowns(left) or _reads_unknowns(right)
        case Conditional(condition, then, otherwise):
            return any(map(_reads_unknowns, (condition, then, otherwise)))
        case Call(_, arguments):
            return any(map(_reads_unknowns, arguments))
    return False


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


_TRANSIENT_FUNCTIONS: Final = frozenset({"pulse", "sin", "pwl", "exp", "sffm", "am"})
"""The names of SPICE's waveforms of an independent source over time."""

_SINE_OPTIONS: Final = ("delay", "damping", "phase")
"""What a `SIN` may give after its offset, amplitude and frequency, in order."""


def _independent(model: str) -> Callable[[_Translation, Statement], None]:
    """The translation of an independent source of `model`.

    Its level is its DC level, or its waveform over time, SPICE's `PULSE`, `SIN`, `PWL`
    or `EXP`, whose level at time zero every other analysis reads, as ngspice reads it
    for a source with no DC level.
    """

    def translated(translation: _Translation, statement: Statement) -> None:
        words = tokens(statement.text)
        _, positive, negative, *rest = words
        rest, waveform = _transient_function(translation, statement, rest)
        given = rest[:1] and rest[0].casefold() == "dc"
        if given:
            rest = rest[1:]
        level = (
            translation.value(rest[0]) if rest and rest[0].casefold() != "ac" else None
        )
        after = rest[1:] if rest and rest[0].casefold() != "ac" else rest
        # A small-signal magnitude belongs to the plan, which excites a source by name.
        if after[:1] and after[0].casefold() == "ac":
            after = after[
                1 + sum(1 for word in after[1:3] if number(word) is not None) :
            ]
        if after:
            message = (
                "jolty reads a source's DC level and its PULSE, SIN, PWL or EXP "
                f"waveform only, not `{' '.join(after)}`"
            )
            raise statement.fail(message)
        if waveform is None:
            waveform = {"dc": 0.0 if level is None else level}
        elif level is not None and level != (
            at_zero := _core.level(json.dumps(waveform), 0.0)
        ):
            message = (
                f"a DC level of {level!r} beside a waveform at {at_zero!r} at time "
                "zero: jolty's source has one level at zero for every analysis"
            )
            raise statement.fail(message)
        parameters: dict[str, JsonValue] = {"waveform": waveform}
        translation.add(
            statement,
            model,
            parameters,
            {
                "positive": translation.net(positive),
                "negative": translation.net(negative),
            },
        )

    return translated


def _transient_function(
    translation: _Translation, statement: Statement, words: list[str]
) -> tuple[list[str], dict[str, JsonValue] | None]:
    """`words` without a waveform over time, and that waveform, if they hold one.

    The waveform is written ``PULSE(...)`` or ``PULSE (...)``, or as SPICE2 wrote it,
    ``PULSE`` followed by its arguments to the statement's end, and likewise ``SIN``,
    ``PWL`` and ``EXP``.

    Raises:
        SpiceError: When the waveform is one jolty has not, such as `SFFM`, or leaves
            out what SPICE takes from the transient analysis.
    """
    for index, word in enumerate(words):
        name, _, inside = word.partition("(")
        if name.casefold() not in _TRANSIENT_FUNCTIONS:
            continue
        if inside:
            arguments, rest = tokens(word[len(name) + 1 : -1]), words[index + 1 :]
        elif index + 1 < len(words) and words[index + 1].startswith("("):
            arguments, rest = tokens(words[index + 1][1:-1]), words[index + 2 :]
        else:
            arguments, rest = words[index + 1 :], []
        values = [translation.value(each) for each in arguments]
        waveform = _waveform(name.casefold(), values, statement)
        return [*words[:index], *rest], waveform
    return words, None


def _waveform(
    name: str, arguments: list[float], statement: Statement
) -> dict[str, JsonValue]:
    """The waveform SPICE's function `name` of `arguments` writes, in jolty's format.

    Raises:
        SpiceError: When jolty has no such waveform, or the function leaves out what
            SPICE takes from the transient analysis: a pulse's rise, fall or width, a
            sine's frequency, or an exponential's delays and time constants, which
            ngspice 47 takes from the transient's step wherever they are 0.
    """
    match name, arguments:
        case "pulse", [initial, pulsed, delay, rise, fall, width, *period] if (
            rise > 0 and fall > 0 and width > 0 and len(period) <= 1
        ):
            pulse: dict[str, JsonValue] = {
                "initial": initial,
                "pulsed": pulsed,
                "delay": delay,
                "rise": rise,
                "fall": fall,
                "width": width,
            }
            if period:
                pulse["period"] = period[0]
            return {"pulse": pulse}
        case "pulse", _:
            message = (
                "a PULSE without a positive rise, fall and width, which SPICE takes "
                "from the transient analysis's step and stop, and jolty's circuit "
                "from no analysis"
            )
            raise statement.fail(message)
        case "sin", [offset, amplitude, frequency, *more] if frequency > 0 and len(
            more
        ) <= len(_SINE_OPTIONS):
            delay, damping, degrees = [*more, 0.0, 0.0, 0.0][: len(_SINE_OPTIONS)]
            return {
                "sine": {
                    "offset": offset,
                    "amplitude": amplitude,
                    "frequency": frequency,
                    "delay": delay,
                    "damping": damping,
                    "phase": math.radians(degrees),
                }
            }
        case "sin", _:
            message = (
                "a SIN without a positive frequency, which SPICE takes from the "
                "transient analysis's stop, and jolty's circuit from no analysis"
            )
            raise statement.fail(message)
        case "exp", [initial, pulsed, rise_delay, rise, fall_delay, fall] if (
            min(rise_delay, rise, fall_delay, fall) > 0
        ):
            return {
                "exponential": {
                    "initial": initial,
                    "pulsed": pulsed,
                    "rise_delay": rise_delay,
                    "rise_time_constant": rise,
                    "fall_delay": fall_delay,
                    "fall_time_constant": fall,
                }
            }
        case "exp", _:
            message = (
                "an EXP without a positive rise delay, rise time constant, fall delay "
                "and fall time constant, each of which ngspice 47 takes from the "
                "transient analysis's step where it is 0 or left out, and jolty's "
                "circuit from no analysis"
            )
            raise statement.fail(message)
        case "pwl", [*pairs] if pairs and len(pairs) % 2 == 0:
            points: list[JsonValue] = [
                [pairs[index], pairs[index + 1]] for index in range(0, len(pairs), 2)
            ]
            return {"piecewise": {"points": points}}
    message = f"jolty has no `{name.upper()}` waveform of these arguments"
    raise statement.fail(message)


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
        instance = translation.multiplied(
            {
                key: translation.value(values[0])
                for key, values in assigned.items()
                if values
            }
        )
        if "nmos" in kinds:
            scale = library.options.get("scale", 1.0)
            instance = {
                key: value * scale ** _SCALED.get(key, 0)
                for key, value in instance.items()
            }
        card = library.model(rest[0], cards)
        if card is None and "nmos" in kinds:
            card = library.binned(rest[0], cards, instance, translation.scope)
            if card is None:
                _unbinned(translation, statement, rest[0], instance)
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
        made = translation.device(card, instance, statement)
        joined = {
            terminal: translation.net(node)
            for terminal, node in zip(terminals, nodes, strict=True)
        }
        translation.add(statement, made.model, made.parameters, joined)

    return translated


def _unbinned(
    translation: _Translation,
    statement: Statement,
    name: str,
    instance: Mapping[str, float],
) -> None:
    """Fails where `name` has bins, none of which holds the size of `instance`.

    The error states the size, and carries the middle of the first bin defined.

    Raises:
        UnbinnedError: When the model has bins.
    """
    library = translation.library
    card = library.card(name, translation.models)
    middle = None if card is None else models.centre(card, translation.scope)
    if middle is None:
        return
    length, width = middle
    sizes = [
        f"a {what} of {instance[key]!r} m" for key, what in _SIZES if key in instance
    ]
    held = f" holds {' and '.join(sizes)}" if sizes else " is selected without a size"
    message = f"no bin of `{name}`{held}"
    raise UnbinnedError(
        str(statement.fail(message)),
        (length, width * library.fingers(instance)),
    )


_SIZES: Final = (("l", "length"), ("w", "width"))
"""A MOSFET's sizes its bins are selected by, by their SPICE names."""


class UnbinnedError(SpiceError):
    """An element names a binned model, none of whose bins holds its size."""

    def __init__(self, message: str, middle: tuple[float, float]) -> None:
        """The error `message` states, and its model's first bin's `middle`."""
        super().__init__(message)
        self.middle = middle
        """The length and the width, in meters, at the middle of the ranges of the
        first bin of its model, its width multiplied by its fingers where they divide
        it."""


_THERMAL_ELEMENTS: Final = {
    ir.DeviceOf(model="resistor"): ("thermal_resistance", "resistance"),
    ir.DeviceOf(model="capacitor"): ("thermal_capacitance", "capacitance"),
}
"""The thermal element a resistor or a capacitor on a thermal node is, and the
parameter of its value."""

_SCALED: Final = {
    **dict.fromkeys(("w", "l", "pd", "ps", "sa", "sb", "sd", "sc"), 1),
    **dict.fromkeys(("ad", "as"), 2),
}
"""The sizes of a MOSFET's element that ngspice's ``scale`` option scales, by the power
of it each takes: a length's 1, an area's 2 (``mos1par.c``), and a BSIM4's distances to
its diffusion's edges, between its fingers and to its well's edge (``b4v5par.c``, lines
87 to 111)."""

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
    instance = translation.multiplied(
        {
            key: translation.value(values[0])
            for key, values in assigned.items()
            if values
        }
    )
    made = translation.device(card, instance, statement)
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
    instance = translation.multiplied(
        {
            key: translation.value(values[0])
            for key, values in assigned.items()
            if values
        }
    )
    # An initial state, `OFF`, only guides the search for the operating point.
    areas = [word for word in rest if word.casefold() not in {"off", "on"}]
    if areas:
        instance["area"] = translation.value(areas[0])
    made = translation.device(card, instance, statement)
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
            expression = translation.parse(rest[2] if rest[1:2] == ["="] else rest[1])
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
    if output == "current" and translation.multiplier != 1:
        message = (
            "jolty multiplies no behavioral current by its subcircuit's multiplier "
            f"`m`, here {translation.multiplier!r}"
        )
        raise statement.fail(message)
    expression = translation.parse(written)
    _behavioral(translation, statement, output, (positive, negative), expression)


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
    made = translation.device(card, {}, statement)
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
    made = translation.device(card, {}, statement)
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


TRANSMISSION_LINE: Final = {
    "z0": "characteristic_impedance",
    "zo": "characteristic_impedance",
    "td": "delay",
}
"""A T element's parameters, by the SPICE names ngspice's ``tra.c`` (lines 13 to 16)
gives them, each parameter's own name first."""

_LINE_KEYWORDS: Final = frozenset(
    {*TRANSMISSION_LINE, "f", "nl", "v1", "i1", "v2", "i2"}
)
"""What a T element gives: its parameters, the frequency at which it is a normalized
length, and initial conditions, as ``tra.c`` (lines 13 to 22) names them."""

_QUARTER_WAVE_AT_GHZ: Final = (0.25, 1e9)
"""A T element's normalized length and frequency where it gives neither, as ngspice's
``trasetup.c`` (lines 94 to 99) sets them: a quarter wavelength at 1 GHz."""


def _transmission_line(translation: _Translation, statement: Statement) -> None:
    """A T element, a lossless transmission line, between two ports of two nodes.

    Its characteristic impedance is ``Z0``, or ``ZO``. Its delay is ``TD``, or else the
    normalized length ``NL`` over the frequency ``F``, either taking ngspice's default
    where it is not given (``tratemp.c``, lines 35 to 37). Its initial conditions,
    ``IC`` or ``V1``, ``I1``, ``V2`` and ``I2``, change nothing jolty computes, as an
    element's ``IC`` does not.
    """
    positional, assigned = _keywords(tokens(statement.text))
    _check_keywords(statement, assigned, _LINE_KEYWORDS)
    if len(positional) != 5:  # noqa: PLR2004 - a name and four nodes
        message = "a transmission line joins four nodes"
        raise statement.fail(message)
    _, *nodes = positional

    def given(name: str) -> float | None:
        words = assigned.get(name)
        if words is None:
            return None
        if len(words) != 1:
            message = f"`{name.upper()}` has one value"
            raise statement.fail(message)
        return translation.value(words[0])

    read: dict[str, float] = {}
    for spice, name in TRANSMISSION_LINE.items():
        value = given(spice)
        if value is not None:
            read.setdefault(name, value)
    impedance = read.get("characteristic_impedance")
    if impedance is None:
        message = "a transmission line gives its characteristic impedance, `Z0`"
        raise statement.fail(message)
    delay = read.get("delay")
    if delay is None:
        length, frequency = given("nl"), given("f")
        default_length, default_frequency = _QUARTER_WAVE_AT_GHZ
        delay = (default_length if length is None else length) / (
            default_frequency if frequency is None else frequency
        )
    parameters: dict[str, JsonValue] = {
        "characteristic_impedance": impedance,
        "delay": delay,
    }
    # The line's terminals are in the order of SPICE's nodes, each port's in turn.
    terminals = device("transmission_line", parameters).terminals
    translation.add(
        statement,
        "transmission_line",
        parameters,
        {
            terminal: translation.net(node)
            for terminal, node in zip(terminals, nodes, strict=True)
        },
    )


def _instance(translation: _Translation, statement: Statement) -> None:
    """An X instance of a subcircuit, with its parameters."""
    words = [word for word in tokens(statement.text) if word.casefold() != "params:"]
    # Each parameter is `name = value`; the subcircuit's name is the last word besides,
    # which HSPICE's netlists, such as SKY130's, may write after the parameters.
    signs = {position for position, word in enumerate(words) if word == "="}
    paired = {position + step for position in signs for step in (-1, 0, 1)}
    positional = [word for place, word in enumerate(words) if place not in paired]
    assigned = [word for place, word in enumerate(words) if place in paired]
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
    functions = translation.library.functions
    given = translation.multiplied(
        {
            parameter: evaluate(
                inlined(expression, functions), translation.scope, translation.draw
            )
            for parameter, expression in models.assignments(statement, assigned).items()
        }
    )
    child = translation.library.place(
        subcircuit, translation.globals, given, translation.drawing.within(name)
    )
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
                name=name,
                of=ir.ComponentOf(component=child.name),
                terminals=terminals,
                source_location=ir.SourceLocation(
                    file=statement.file, line=statement.line
                ),
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
    "t": _transmission_line,
    "w": _current_switch,
    "x": _instance,
}
"""How each element, by its letter, becomes devices or a placed component."""
