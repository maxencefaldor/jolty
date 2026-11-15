# Copyright 2026 Maxence Faldor
"""Third-party decks, fetched at a pinned commit and never copied, read as each reads.

The manifest, ``benchmarks/robustness/external.json``, names each deck by its address at
a commit, its SHA-256 digest and its licence; `jolty.spice.fetch` downloads it once into
jolty's cache, outside the repository, and checks its digest. Each deck gives a fixture
for its operating point and one for each analysis it asks for, a DC sweep or a
transient.

ngspice runs the deck itself; jolty reads it through its SPICE reader, its elements
wrapped in a subcircuit of their own beside the deck's definitions, every line kept at
its place so that each instance records the deck's own line. A deck that the reader
cannot read, or that asks for what jolty's plan has not, initial conditions or a
nodeset, is unsupported in jolty, with the reason.

A deck's defects are fixed by recorded rules, alike for every simulator (`RULES`); each
fixture lists those its deck needed.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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

from jolty import _core, circuit
from jolty.errors import JoltyError
from jolty.plan import Method, Options
from jolty.simulate import Simulation
from jolty.spice import Library, fetch
from jolty.spice.expressions import number
from jolty.spice.netlist import Statement, statements, tokens

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
"""What a deck asks of the operating point, which jolty's plan cannot state."""

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
    def rules(self) -> tuple[str, ...]:
        """The rules its defects need."""
        needed: list[str] = []
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
            dc_iterations=iterations.get("itl1"),
            transient_iterations=iterations.get("itl4"),
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

    def conditions(self) -> list[_Card]:
        """Its initial conditions and nodesets."""
        return [card for card in self.cards if card.kind == "condition"]

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
        lines = self.text.splitlines()
        definitions = [""] * len(lines)
        elements = [""] * len(lines)
        for card in self.cards:
            target = (
                definitions
                if card.kind == "definition"
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
        """The deck as jolty's SPICE reader reads it, its elements a subcircuit.

        Raises:
            Unsupported: When the reader cannot read it, or it sets initial conditions
                or a nodeset, which jolty's plan cannot state.
        """
        conditions = self.deck.conditions()
        if conditions:
            card = conditions[0]
            message = (
                f"{self.deck.file}:{card.first}: jolty's plan sets no initial "
                f"conditions or nodeset: `{card.statement.text[:120]}`"
            )
            raise Unsupported(message)
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
        return Built(built, self.deck.temperature(), self.deck.options(), nets)

    def spice(self) -> Spiced:
        """The deck as ngspice reads it, its rules applied and its analyses taken out.

        Its nodes are those jolty reads, when jolty reads it.
        """
        lines = self.deck.text.splitlines()
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
