# Copyright 2026 Maxence Faldor

import corpus
import decks
import pytest

from jolty import OperatingPoint, Simulation
from jolty.plan import Options

DECK = """A divider and a pulse, as Xyce writes CircuitSim90's decks
.options timeint method=gear reltol=1e-4
.options device temp=50
v1 1 0 dc 3
r1 1 2 1k
x1 2 half
.subckt half a
r2 a 0 2k
.ends half
.op
.tran 1n 10n
.print tran {v(2)+1}
.end
r3 9 9 1k
"""
"""A deck of each kind of statement the corpus reads, and a line after its end."""


def deck() -> decks.DeckCircuit:
    entry = decks.Entry("divider", "tests", "file:divider.cir", "", "", "", "")
    return decks.DeckCircuit(decks.Deck.read(entry, DECK))


def test_a_deck_splits_into_its_analyses_options_and_rules() -> None:
    read = deck().deck
    assert read.analyses == [corpus.Point(), corpus.Transient(1e-9, 1e-8)]
    assert read.rules == ("xyce-options", "output-requests")
    assert read.options() == Options(relative_tolerance=1e-4)
    assert read.method() == "gear2"
    assert read.temperature() == pytest.approx(323.15)


def test_ngspice_reads_the_deck_without_its_analyses_or_xyce_s_packages() -> None:
    spiced = deck().spice()
    kept = [line for line in spiced.lines if line]
    assert kept == [
        "A divider and a pulse, as Xyce writes CircuitSim90's decks",
        ".options reltol=1e-4",
        "v1 1 0 dc 3",
        "r1 1 2 1k",
        "x1 2 half",
        ".subckt half a",
        "r2 a 0 2k",
        ".ends half",
    ]
    assert spiced.vectors == {"1": "v(1)", "2": "v(2)"}


def test_jolty_reads_the_deck_s_elements_at_their_lines() -> None:
    built = deck().jolty()
    top = built.circuit.components[built.circuit.top]
    lines = {
        instance.id: None if instance.source is None else instance.source.line
        for instance in top.instances
    }
    assert lines == {"v1": 4, "r1": 5, "x1": 6}
    assert built.nets == ("1", "2")
    point = Simulation(built.circuit).run(OperatingPoint())
    # 3 V across 1 kΩ over 2 kΩ.
    assert point.voltage("2") == pytest.approx(2.0)


def test_runs_are_judged_against_every_solution() -> None:
    fixture = corpus.Fixture(
        "latch/op",
        corpus.Kind.DC,
        "tests",
        "two solutions",
        deck(),
        corpus.Point(),
        corpus.Solutions(),
    )
    reference = corpus.Reference((({"q": 0.0}, {"q": 3.3}),))

    def judged(run: corpus.FixtureRun) -> tuple[str, str]:
        outcome = corpus.judge(fixture, run, reference, Options())
        return outcome.status.value, outcome.detail

    # Within 1e-3 of 3.3 V plus twice 1 µV, the default tolerances.
    near = corpus.FixtureRun(completed=True, values=({"q": 3.3032},))
    assert judged(near) == ("agreed", "solution 2 of 2")
    far = corpus.FixtureRun(completed=True, values=({"q": 3.304},))
    assert judged(far) == ("disagreed", "agrees with no DC solution")
    failed = corpus.FixtureRun(completed=False, message="no operating point")
    assert judged(failed) == ("failed", "no operating point")
    refused = corpus.FixtureRun(completed=False, unsupported=True, message="no PULSE")
    assert judged(refused) == ("unsupported", "no PULSE")


def test_failure_rates_leave_out_what_a_simulator_does_not_support() -> None:
    counts = {
        corpus.Status.AGREED: 7,
        corpus.Status.FAILED: 2,
        corpus.Status.TIMED_OUT: 1,
        corpus.Status.UNSUPPORTED: 5,
    }
    failed, supported, low, high = corpus.wilsons(counts) or (0, 0, 0.0, 0.0)
    assert (failed, supported) == (3, 10)
    assert low < 0.3 < high
