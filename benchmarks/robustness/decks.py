# Copyright 2026 Maxence Faldor
"""Third-party decks, fetched at a pinned commit and never copied, read as each reads.

The manifest, ``benchmarks/robustness/external.json``, names each deck by its address at
a commit, its SHA-256 digest and its licence; `jolty.spice.fetch` downloads it once into
jolty's cache, outside the repository, and checks its digest. Each deck gives a fixture
for its operating point and one for each analysis it asks for, a DC sweep or a
transient.

ngspice runs the deck itself; jolty reads it through its SPICE reader, its elements
wrapped in a subcircuit of their own beside the deck's definitions, every line kept at
its place so that each instance records the deck's own line, and its initial conditions
and nodesets into the plan. A deck that the reader cannot read is unsupported in jolty,
with the reason.

A deck's defects are fixed by recorded rules, alike for every simulator (`RULES`); each
fixture lists those its deck needed.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Final

from corpus import (
    ROBUSTNESS,
    Analysis,
    Built,
    Completes,
    Fixture,
    Kind,
    Point,
    Solutions,
    Spiced,
    Sweep,
    Transient,
)
from simulators import Unsupported

from jolty import _core, circuit, ir
from jolty.errors import JoltyError
from jolty.plan import Method, Options
from jolty.simulate import Simulation
from jolty.spice import Library, fetch
from jolty.spice.expressions import number
from jolty.spice.netlist import Statement, statements, tokens
from jolty.units import V

MANIFEST: Final = ROBUSTNESS / "external.json"
"""The third-party decks: each one's address, commit, digest and licence."""

RULES: Final = {
    "xyce-options": (
        "Xyce's `.options` packages, such as `.options device temp=100` and "
        "`.options timeint method=gear reltol=1e-3`, are SPICE's `.options` without "
        "the package's name."
    ),
    "output-requests": (
        "Output requests, `.print`, `.plot`, `.probe`, `.save` and `.width`, are taken "
        "out: every simulator keeps every node, and Xyce's `.print` writes expressions "
        "in braces that SPICE does not read."
    ),
    "element-names": (
        "An element whose name holds a `/`, as a flattened hierarchy's do, has each "
        "written as `_`, as jolty's paths separate their levels with `/`."
    ),
    "exp-defaults": (
        "An `EXP` waveform's delays and time constants left at 0 or out are written as "
        "SPICE3 and ngspice 47 take them, from the transient's step: each time "
        "constant and the rise delay the step, the fall delay the rise delay plus it."
    ),
}
"""The rules that fix the decks' defects, alike for every simulator, by name."""

_XYCE_PACKAGES: Final = frozenset(
    {"device", "timeint", "nonlin", "nonlin-tran", "linsol", "output", "restart"}
)
"""The names of Xyce's packages of options, which SPICE's `.options` does not take."""

_OUTPUTS: Final = frozenset({".print", ".plot", ".probe", ".save", ".width"})
"""The output requests, which change no simulation."""

_ANALYSES: Final = frozenset({".op", ".dc", ".tran"})
"""The analyses a deck's fixture runs."""

_DEFINITIONS: Final = frozenset({".model", ".param"})
"""The definitions outside subcircuits, which jolty's SPICE reader reads."""

_CONDITIONS: Final = frozenset({".ic", ".nodeset"})
"""The nodes' voltages a deck starts its analyses from: initial conditions and
nodesets, which the library reads with its definitions."""

_EXP: Final = re.compile(r"\bexp\s*\(([^()]*)\)", re.IGNORECASE)
"""An `EXP` waveform's arguments in a source's statement."""

_EXP_LEVELS: Final = 2
"""The levels an `EXP` gives first, its initial and its pulsed one."""

_EXP_ARGUMENTS: Final = 6
"""An `EXP`'s arguments: its two levels, then its two delays and two time constants."""

_METHODS: Final[Mapping[str, Method]] = {
    "trap": "trapezoidal",
    "trapezoidal": "trapezoidal",
    "gear": "gear2",
}
"""SPICE's integration methods as jolty names them: ngspice's Gear takes the second
order unless ``maxord`` says otherwise."""


@dataclass(frozen=True, slots=True)
class Entry:
    """A deck of the manifest."""

    name: str
    suite: str
    url: str
    commit: str
    sha256: str
    licence: str
    hard: str


def manifest() -> list[Entry]:
    """The manifest's decks, in its order."""
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [Entry(**entry) for entry in document["decks"]]


@dataclass(frozen=True, slots=True)
class _Card:
    """A statement of a deck, the raw lines it spans, and what it is."""

    statement: Statement
    first: int
    """Its first line, from 1."""
    last: int
    """Its last line, its continuations, comments and blank lines after it included."""
    kind: str
    """``definition``, ``element``, ``analysis``, ``option``, ``output``,
    ``condition``, ``temperature`` or ``disagreed``."""

    @property
    def keyword(self) -> str:
        """Its first word, in lowercase."""
        return self.statement.text.split()[0].casefold()


def _cards(text: str, file: str) -> list[_Card]:
    """The statements of a deck, after its title and up to its ``.end``.

    Raises:
        Unsupported: When a statement cannot be split.
    """
    lines = text.splitlines()
    body = "\n".join(["", *lines[1:]])
    try:
        found = statements(body, file)
    except JoltyError as error:
        raise Unsupported(str(error)) from None
    cards: list[_Card] = []
    depth = 0
    for index, statement in enumerate(found):
        last = found[index + 1].line - 1 if index + 1 < len(found) else len(lines)
        keyword = statement.text.split()[0].casefold()
        if keyword == ".end" and depth == 0:
            break
        if depth or keyword == ".subckt":
            depth += (keyword == ".subckt") - (keyword == ".ends")
            kind = "definition"
        else:
            kind = _kind(keyword)
        cards.append(_Card(statement, statement.line, last, kind))
    return cards


_KINDS: Final = (
    (_DEFINITIONS, "definition"),
    (_ANALYSES, "analysis"),
    (frozenset({".options", ".option", ".opt"}), "option"),
    (_OUTPUTS, "output"),
    (_CONDITIONS, "condition"),
    (frozenset({".temp"}), "temperature"),
)
"""The kind of each statement outside a subcircuit, by its keyword."""


def _kind(keyword: str) -> str:
    """The kind of a statement outside a subcircuit that starts with `keyword`."""
    found = next((kind for keywords, kind in _KINDS if keyword in keywords), None)
    if found is not None:
        return found
    return "other" if keyword.startswith(".") else "element"


def _value(word: str, card: _Card) -> float:
    """The number `word` writes, in `card`.

    Raises:
        Unsupported: When it is not a number.
    """
    found = number(word)
    if found is None:
        message = f"{card.statement.file}:{card.first}: `{word}` is not a number"
        raise Unsupported(message)
    return found


def _assignments(card: _Card) -> dict[str, str]:
    """The ``name=value`` pairs of an options statement, by lowercase name."""
    words = tokens(card.statement.text)[1:]
    return {
        words[index - 1].casefold(): words[index + 1]
        for index, word in enumerate(words)
        if word == "=" and 0 < index < len(words) - 1
    }


def _analysis(card: _Card) -> Analysis:
    """The analysis an analysis card asks for.

    Raises:
        Unsupported: When the card asks for what the corpus does not run.
    """
    words = tokens(card.statement.text)
    match card.keyword:
        case ".op":
            return Point()
        case ".dc" if len(words) == 5:
            _, source, *levels = words
            start, stop, step = (_value(word, card) for word in levels)
            return Sweep(source, start, stop, step)
        case ".tran" if (
            3 <= len(words) <= 5 and "uic" not in card.statement.text.casefold()
        ):
            step, stop = _value(words[1], card), _value(words[2], card)
            longest = _value(words[4], card) if len(words) == 5 else None
            return Transient(step, stop, longest)
    where = f"{card.statement.file}:{card.first}"
    message = f"{where}: the corpus runs no `{card.statement.text}`"
    raise Unsupported(message)


@dataclass(frozen=True)
class Deck:
    """A fetched deck, split into what each simulator reads."""

    entry: Entry
    text: str
    file: str
    cards: Sequence[_Card] = field(repr=False)

    @classmethod
    def of(cls, entry: Entry) -> Deck:
        """The deck `entry` names, fetched and checked against its digest."""
        path = fetch(entry.url, entry.sha256)
        return cls.read(entry, path.read_text(encoding="utf-8", errors="replace"))

    @classmethod
    def read(cls, entry: Entry, text: str) -> Deck:
        """The deck `entry` names, whose text is `text`."""
        file = Path(entry.url).name
        return cls(entry, text, file, _cards(text, file))

    @cached_property
    def analyses(self) -> list[Analysis]:
        """The analyses its cards ask for, in order.

        Those the corpus does not run are left out.
        """
        found: list[Analysis] = []
        for card in self.cards:
            if card.kind == "analysis":
                try:
                    found.append(_analysis(card))
                except Unsupported:
                    continue
        return found

    @cached_property
    def assignments(self) -> dict[str, str]:
        """Every option it sets, by lowercase name, Xyce's packages' among them."""
        found: dict[str, str] = {}
        for card in self.cards:
            if card.kind == "option":
                found |= _assignments(card)
        return found

    @cached_property
    def fixed(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Its lines with the rules that rewrite its elements applied, and those rules.

        A statement rewritten is written on its first line, its continuations blank:
        an `EXP` that takes a time from the transient's step with that time, and an
        element whose name holds a `/` with each written as `_`, unless another element
        is named so.
        """
        lines = self.text.splitlines()
        step = next(
            (
                analysis.step
                for analysis in self.analyses
                if isinstance(analysis, Transient)
            ),
            None,
        )
        elements = [card for card in self.cards if card.kind == "element"]
        taken = {card.statement.text.split()[0].casefold() for card in elements}
        applied: list[str] = []
        for card in elements:
            text = card.statement.text
            explicit = None if step is None else _exp_defaults(text, step)
            if explicit is not None:
                text = explicit
                applied.append("exp-defaults")
            name = text.split()[0]
            renamed = name.replace(ir.SEPARATOR, "_")
            if renamed != name and renamed.casefold() not in taken:
                text = renamed + text[len(name) :]
                applied.append("element-names")
            if text != card.statement.text:
                blank = [""] * (card.last - card.first)
                lines[card.first - 1 : card.last] = [text, *blank]
        return tuple(lines), tuple(dict.fromkeys(applied))

    @property
    def lines(self) -> tuple[str, ...]:
        """Its lines, with the rules that rewrite its elements applied."""
        return self.fixed[0]

    @cached_property
    def rules(self) -> tuple[str, ...]:
        """The rules its defects need."""
        needed: list[str] = list(self.fixed[1])
        for card in self.cards:
            words = card.statement.text.split()
            if (
                card.kind == "option"
                and len(words) > 1
                and words[1].casefold() in _XYCE_PACKAGES
            ):
                needed.append("xyce-options")
            if card.kind == "output":
                needed.append("output-requests")
        return tuple(dict.fromkeys(needed))

    def options(self) -> Options:
        """The options it sets that jolty's have, by SPICE's names."""
        given = self.assignments

        def read(name: str) -> float | None:
            word = given.get(name)
            return None if word is None else number(word)

        iterations = {
            key: int(value)
            for key, value in (("itl1", read("itl1")), ("itl4", read("itl4")))
            if value is not None
        }
        return Options(
            relative_tolerance=read("reltol"),
            voltage_tolerance=read("vntol"),
            current_tolerance=read("abstol"),
            charge_tolerance=read("chgtol"),
            gmin=read("gmin"),
            max_dc_iterations=iterations.get("itl1"),
            max_transient_iterations=iterations.get("itl4"),
            truncation_tolerance=read("trtol"),
        )

    def method(self) -> Method:
        """The integration method it asks for, the trapezoidal rule by default."""
        written = self.assignments.get("method", "trap").casefold()
        return _METHODS.get(written, "trapezoidal")

    def temperature(self) -> float:
        """The temperature it runs at, in kelvins: 27 °C unless it says otherwise."""
        written: float | None = None
        for card in self.cards:
            if card.kind == "temperature":
                words = tokens(card.statement.text)
                written = _value(words[1], card) if len(words) > 1 else None
        option = self.assignments.get("temp")
        if option is not None:
            written = number(option)
        if written is None:
            return _core.REFERENCE_TEMPERATURE
        return written + 273.15

    def library(self) -> tuple[Library, str]:
        """The deck as a library of jolty's, and the name of its elements' subcircuit.

        Raises:
            Unsupported: When the SPICE reader cannot read the deck.
        """
        others = [card for card in self.cards if card.kind == "other"]
        if others:
            card = others[0]
            message = (
                f"{self.file}:{card.first}: the corpus reads no "
                f"`{card.statement.text[:120]}`"
            )
            raise Unsupported(message)
        lines = self.lines
        definitions = [""] * len(lines)
        elements = [""] * len(lines)
        for card in self.cards:
            target = (
                definitions
                if card.kind in {"definition", "condition"}
                else elements
                if card.kind == "element"
                else None
            )
            if target is None:
                continue
            for line in range(card.first, card.last + 1):
                target[line - 1] = lines[line - 1]
        library = Library()
        try:
            library.read("\n".join(definitions), self.file)
            name = "deck"
            while name in {each.casefold() for each in library.subcircuits}:
                name += "_"
            elements[0] = f".subckt {name}"
            library.read("\n".join([*elements, ".ends"]), self.file)
        except JoltyError as error:
            raise Unsupported(" ".join(str(error).split())) from None
        return library, name


@dataclass(frozen=True)
class DeckCircuit:
    """A deck's circuit, as each simulator reads it."""

    deck: Deck

    def jolty(self) -> Built:
        """The deck as jolty's reader reads it, with its conditions and nodesets.

        Its elements are a subcircuit's, beside its definitions.

        Raises:
            Unsupported: When the reader cannot read it.
        """
        library, name = self.deck.library()
        try:
            built = circuit(library.component(name))
            unknowns = Simulation(built).unknowns
        except JoltyError as error:
            raise Unsupported(" ".join(str(error).split())) from None
        nets = tuple(
            unknown.name
            for unknown in unknowns
            if unknown.nature == "voltage" and not {"/", ":", "."} & set(unknown.name)
        )
        options = replace(
            self.deck.options(),
            nodesets={node: volts * V for node, volts in library.nodesets.items()},
        )
        return Built(
            built,
            self.deck.temperature(),
            options,
            nets,
            dict(library.initial_conditions),
        )

    def spice(self) -> Spiced:
        """The deck as ngspice reads it, its rules applied and its analyses taken out.

        Its nodes are those jolty reads, when jolty reads it.
        """
        lines = self.deck.lines
        kept = [lines[0]]
        for card in self.deck.cards:
            span = lines[card.first - 1 : card.last]
            if card.kind in {"analysis", "output", "temperature"}:
                kept.extend("" for _ in span)
            elif card.kind == "option":
                kept.append(_spice_options(card))
                kept.extend("" for _ in span[1:])
            else:
                kept.extend(span)
        try:
            nets = self.jolty().nets
        except Unsupported:
            nets = ()
        vectors = {net: f"v({net.casefold()})" for net in nets}
        return Spiced(tuple(kept), vectors)

    def settled(self) -> Built | None:
        """None: a deck's transient has no reference."""
        return None

    def temperature(self) -> float:
        """The deck's temperature, in kelvins."""
        return self.deck.temperature()


def _exp_defaults(text: str, step: float) -> str | None:
    """The statement `text` with its `EXP`'s times of 0 or left out made explicit.

    They are written as SPICE3 and ngspice 47 take them from the transient's `step`;
    None when the statement has no such `EXP`, or one whose arguments are not all
    numbers.
    """
    found = _EXP.search(text)
    if found is None:
        return None
    given = [number(word) for word in tokens(found[1])]
    numbers = [value for value in given if value is not None]
    if len(numbers) != len(given) or not _EXP_LEVELS <= len(numbers) <= _EXP_ARGUMENTS:
        return None
    initial, pulsed, *times = numbers
    times += [0.0] * (_EXP_ARGUMENTS - len(numbers))
    if all(times):
        return None
    rise_delay = times[0] or step
    fall_delay = times[2] or rise_delay + step
    explicit = [
        initial,
        pulsed,
        rise_delay,
        times[1] or step,
        fall_delay,
        times[3] or step,
    ]
    arguments = " ".join(repr(value) for value in explicit)
    return f"{text[: found.start()]}EXP({arguments}){text[found.end() :]}"


def _spice_options(card: _Card) -> str:
    """An options statement as SPICE's, without Xyce's package name.

    The temperature is set apart, as both simulators run at the deck's, and the
    method, which each fixture states.
    """
    given = {
        key: value
        for key, value in _assignments(card).items()
        if key not in {"temp", "method"}
    }
    if not given:
        return ""
    return ".options " + " ".join(f"{key}={value}" for key, value in given.items())


def fixtures() -> list[Fixture]:
    """A fixture for each deck's operating point, and one for each analysis it runs.

    Raises:
        SystemExit: When a deck cannot be fetched, or is not what its digest says.
    """
    found: list[Fixture] = []
    for entry in manifest():
        try:
            deck = Deck.of(entry)
        except JoltyError as error:
            raise SystemExit(str(error)) from None
        fixture_circuit = DeckCircuit(deck)
        options = deck.options()
        found.append(
            Fixture(
                f"{entry.name}/op",
                Kind.DC,
                entry.suite,
                entry.hard,
                fixture_circuit,
                Point(),
                Solutions(),
                options,
                deck.rules,
            )
        )
        for analysis in deck.analyses:
            match analysis:
                case Sweep():
                    found.append(
                        Fixture(
                            f"{entry.name}/dc",
                            Kind.DC,
                            entry.suite,
                            entry.hard,
                            fixture_circuit,
                            analysis,
                            Solutions(),
                            options,
                            deck.rules,
                        )
                    )
                case Transient():
                    transient = Transient(
                        analysis.step, analysis.stop, analysis.max_step, deck.method()
                    )
                    found.append(
                        Fixture(
                            f"{entry.name}/tran",
                            Kind.TRANSIENT,
                            entry.suite,
                            entry.hard,
                            fixture_circuit,
                            transient,
                            Completes(),
                            options,
                            deck.rules,
                        )
                    )
                case Point():
                    continue
    return found
