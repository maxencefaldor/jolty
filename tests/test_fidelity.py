# Copyright 2026 Maxence Faldor

from typing import Final

import pytest

from jolty import (
    Builder,
    CharacterizationError,
    Component,
    Role,
    Simulation,
    circuit,
    component,
    devices,
    fidelity,
)
from jolty.errors import JoltyError
from jolty.plan import Characterized, Explicit, OperatingPoint, Plan, Transient, Variant
from jolty.results import AnalysisResult, Run
from jolty.specs import Specs, crossing, evaluate
from jolty.units import TIME, Quantity, V, fF, format_si, nm, ns, ps, um
from jolty.waveforms import Pulse

SUPPLY = 3.3 * V


def transistor(channel: str) -> devices.Device:
    p = channel == "p"
    return devices.mosfet(
        channel="p" if p else "n",
        width=(2 if p else 1) * um,
        length=1 * um,
        threshold_voltage=(-0.7 if p else 0.7) * V,
        oxide_thickness=20 * nm,
        bulk_drain_capacitance=2 * fF,
        bulk_source_capacitance=2 * fF,
    )


@component
def inverter(c: Builder) -> None:
    """A CMOS inverter."""
    a, y = c.port("a", Role.INPUT), c.port("y", Role.OUTPUT)
    vdd, gnd = c.port("vdd", Role.SUPPLY), c.port("gnd", Role.GROUND)
    up, down = c.place(transistor("p"), id="P"), c.place(transistor("n"), id="N")
    c.connect(up.source, up.bulk, vdd)
    c.connect(down.source, down.bulk, gnd)
    c.connect(up.gate, down.gate, a)
    c.connect(up.drain, down.drain, y)


@component
def chain(c: Builder, *, load: float = 0.0) -> None:
    """A pulse through two inverters, the first loaded by `load` femtofarads."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(SUPPLY), id="VDD")
    pulse = Pulse(
        initial=0 * V,
        pulsed=SUPPLY,
        delay=1 * ns,
        rise=100 * ps,
        fall=100 * ps,
        width=3 * ns,
    )
    source = c.place(devices.voltage_source(pulse), id="V1")
    first, second = c.place(inverter(), id="X1"), c.place(inverter(), id="X2")
    capacitor = c.place(devices.capacitor(load * fF), id="C1")
    c.connect(supply.positive, first.vdd, second.vdd, name="vdd")
    c.connect(supply.negative, source.negative, first.gnd, second.gnd, ground)
    c.connect(source.positive, first.a, name="in")
    c.connect(first.y, second.a, capacitor.a, name="mid")
    c.connect(capacitor.b, ground)
    c.connect(second.y, name="out")


def characterized() -> fidelity.Characterization:
    return fidelity.characterize(
        circuit(inverter()), "inverter", Characterized(SUPPLY), None
    )


def test_a_characterization_measures_tables_capacitances_and_its_error() -> None:
    model = characterized()
    assert model.tables == {"y": "10"}
    assert model.inputs == ("a",)
    assert model.input_capacitances[0] > 0
    assert [arc.before for arc in model.arcs] == ["0", "1"]
    assert model.loads[0] == 0
    assert model.loads[1] == pytest.approx(model.input_capacitances[0])
    assert 0 < model.delay_error < 0.1 * model.mean_delay()
    assert model.device().terminals == ("a", "y", "vdd", "gnd")


def test_a_component_with_a_signal_port_is_no_logic_cell() -> None:
    @component
    def wire(c: Builder) -> None:
        c.port("a")
        c.port("vdd", Role.SUPPLY)
        c.port("gnd", Role.GROUND)

    with pytest.raises(CharacterizationError, match="not inputs, outputs"):
        fidelity.characterize(circuit(wire()), "wire", Characterized(SUPPLY), None)


def test_scopes_match_the_outermost_instances() -> None:
    built = circuit(chain())
    plan = {"X*": Characterized(SUPPLY), "**": Characterized(SUPPLY)}
    assert fidelity.scopes(built, plan) == {
        "X1": ("X*", "inverter"),
        "X2": ("X*", "inverter"),
    }
    with pytest.raises(JoltyError, match="'Y\\*' matches no component instance"):
        fidelity.scopes(built, {"Y*": Characterized(SUPPLY)})


def test_a_plan_s_characterizations_are_checked() -> None:
    with pytest.raises(ValueError, match="the slews rise above zero"):
        Characterized(SUPPLY, slews=(0, 1))
    with pytest.raises(ValueError, match="the fanouts rise from zero or more"):
        Characterized(SUPPLY, fanouts=(2, 1))
    with pytest.raises(TypeError, match="not a Characterized model"):
        Plan({"step": Transient(stop=1 * ns)}, fidelity={"X1": 1.0})  # ty: ignore[invalid-argument-type] - the wrong type is tested


PLAN: Final = Plan(
    {"step": Transient(stop=6 * ns)}, fidelity={"X1": Characterized(SUPPLY)}
)


def runs(
    top: Component,
) -> tuple[
    fidelity.Fidelity, Simulation, dict[str, AnalysisResult], dict[str, AnalysisResult]
]:
    """The fidelity of `top`, and its simulation and results with every model.

    Last come its results with X1 at its circuit's fidelity.
    """
    chosen = fidelity.resolve(circuit(top), PLAN, None, {})
    simulation = chosen.simulation()
    modeled = simulation.run_plan(PLAN)
    opened = chosen.simulation(opened="X1").run_plan(PLAN)
    return chosen, simulation, modeled, opened


def spot_check(top: Component) -> fidelity.Contract:
    chosen, _, modeled, opened = runs(top)
    return fidelity.contract(chosen, PLAN, "X1", modeled, opened)


def test_contract_violated() -> None:
    held = spot_check(chain())
    assert held.held
    assert held.describe().startswith("X1: its model agrees with its circuit within")
    # A load a hundred fanouts strong lies far beyond the tables, which hold their edge.
    violated = spot_check(chain(load=500.0))
    assert not violated.held
    assert violated.divergences[0].startswith("step: output y changes to 0 at")
    assert violated.describe().startswith("error[contract-violated]: X1's model")


def test_outside_validity() -> None:
    _, _, modeled, _ = runs(chain())
    assert modeled["step"].excursions == ()
    _, _, modeled, _ = runs(chain(load=500.0))
    watched = modeled["step"].excursions
    assert watched is not None
    (excursion,) = watched
    assert (excursion.path, excursion.bound) == ("X1", "load")
    assert excursion.value > excursion.high
    outside = fidelity.Outside(analysis="step", excursion=excursion)
    assert outside.describe().startswith("warn[outside-validity]: X1's load reads")


def test_a_result_states_its_own_excursions_whatever_runs_after_it() -> None:
    chosen = fidelity.resolve(circuit(chain(load=500.0)), PLAN, None, {})
    simulation = chosen.simulation()
    stepped = simulation.run(Transient(stop=6 * ns))
    # At the operating point no cell switches an output, so none reads its load.
    point = simulation.run(OperatingPoint())
    assert point.excursions == ()
    assert stepped.excursions is not None
    (excursion,) = stepped.excursions
    assert (excursion.path, excursion.bound) == ("X1", "load")
    assert not hasattr(simulation, "excursions")
    # A batch's runs are not watched.
    ran: list[Run] = []
    step = {"step": Transient(stop=6 * ns)}
    simulation.batch(step, Explicit([Variant()]), each=ran.append)
    (run,) = ran
    assert run.results["step"].excursions is None


def test_fidelity_dependence() -> None:
    _, _, modeled, opened = runs(chain(load=500.0))
    delay = crossing("step", "mid", SUPPLY / 2, edge="falling")
    times = [delay.read(found["step"]).value for found in (modeled, opened)]
    specs = Specs([delay <= Quantity(sum(times) / 2, TIME)])
    first = evaluate(specs, PLAN.analyses, modeled)
    second = evaluate(specs, PLAN.analyses, opened)
    (found,) = fidelity.dependences("X1", first, second)
    assert found.describe() == (
        "warn[fidelity-dependence]: first falling crossing of 1.65 V by mid in step "
        f"≤ {found.modeled.requirement.split('≤ ')[1]}: passed, reading "
        f"{format_si(times[0], 's')}, with every model, failed, reading "
        f"{format_si(times[1], 's')}, with X1 at its circuit's fidelity"
    )
    assert fidelity.dependences("X1", first, first) == []
