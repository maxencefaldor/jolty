# Copyright 2026 Maxence Faldor

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jolty import Builder, Role, Simulation, component, devices, results
from jolty.files import load_results, save_results
from jolty.plan import DcSweep, OperatingPoint, Transient
from jolty.plots import default_traces, plot
from jolty.report import report
from jolty.units import V, fF, ns, ps
from jolty.waveforms import Pulse


@component
def chain(c: Builder) -> None:
    """A source into a NAND with a 1, into an inverter."""
    c.port("ground", Role.GROUND)
    a = c.place(devices.logic_source(0, [(1 * ns, 1)], transition=10 * ps), id="A")
    b = c.place(devices.logic_source(1, transition=10 * ps), id="B")
    nand = c.place(
        devices.logic_gate("nand", rise=2 * ns, fall=1 * ns, transition=10 * ps),
        id="U1",
    )
    inverter = c.place(
        devices.logic_gate("not", rise=2 * ns, fall=1 * ns, transition=10 * ps),
        id="U2",
    )
    c.connect(a.output, nand.a, name="a")
    c.connect(b.output, nand.b, name="b")
    c.connect(nand.y, inverter.a, name="n")
    c.connect(inverter.y, name="y")


@component
def inverter(c: Builder) -> None:
    """An electrical inverter between an analog edge and a capacitor."""
    ground = c.port("ground", Role.GROUND)
    supply = c.place(devices.voltage_source(1.8 * V), id="VDD")
    edge = Pulse(
        initial=0 * V,
        pulsed=1.8 * V,
        delay=1 * ns,
        rise=1 * ns,
        fall=1 * ns,
        width=10 * ns,
    )
    stimulus = c.place(devices.voltage_source(edge), id="V1")
    gate = c.place(
        devices.logic_gate(
            "not",
            rise=500 * ps,
            transition=100 * ps,
            electrical=True,
            input_capacitance=2 * fF,
        ),
        id="U1",
    )
    load = c.place(devices.capacitor(5 * fF), id="C1")
    c.connect(supply.positive, gate.supply, name="vdd")
    c.connect(stimulus.positive, gate.a, name="in")
    c.connect(gate.y, load.a, name="out")
    c.connect(supply.negative, stimulus.negative, gate.ground, load.b, ground)


def test_logic_settles_then_follows_each_change_after_its_delay() -> None:
    simulation = Simulation(chain())
    assert simulation.unknowns == ()
    assert simulation.nets == ("a", "b", "n", "y")
    assert simulation.devices == 4
    point = simulation.run(OperatingPoint())
    assert [point.level(net) for net in simulation.nets] == [0.0, 1.0, 1.0, 0.0]
    waveforms = simulation.run(Transient(stop=10 * ns))
    assert waveforms.level("n").times.tolist() == pytest.approx([0, 2e-9])
    assert waveforms.level("y").times.tolist() == pytest.approx([0, 4e-9])
    assert waveforms.level("y").levels.tolist() == [0.0, 1.0]
    with pytest.raises(KeyError, match="no logic net named 'z'"):
        waveforms.level("z")


def test_a_logic_pin_joins_an_electrical_net_through_a_connect_element() -> None:
    simulation = Simulation(inverter())
    assert simulation.nets == ("U1:a", "U1:y")
    waveforms = simulation.run(Transient(stop=5 * ns))
    assert waveforms.level("U1:a").times.tolist() == pytest.approx([0, 1.5e-9])
    out = waveforms.voltage("out")
    falling = np.flatnonzero((out[:-1] > 0.9) & (out[1:] <= 0.9))[0]
    times = waveforms.times
    crossing = times[falling] + (times[falling + 1] - times[falling]) * (
        out[falling] - 0.9
    ) / (out[falling] - out[falling + 1])
    assert crossing == pytest.approx(2e-9, abs=1e-14)


def test_unknown_levels_read_as_nan() -> None:
    @component
    def floating(c: Builder) -> None:
        c.port("ground", Role.GROUND)
        gate = c.place(
            devices.logic_gate("and", rise=1 * ns, transition=10 * ps), id="U1"
        )
        source = c.place(devices.logic_source(None, transition=10 * ps), id="A")
        c.connect(source.output, gate.a, gate.b, name="a")
        c.connect(gate.y, name="y")

    point = Simulation(floating()).run(OperatingPoint())
    assert math.isnan(point.level("y"))


def test_a_sweep_settles_the_logic_at_each_level() -> None:
    simulation = Simulation(inverter())
    sweep = simulation.run(DcSweep("V1", [0 * V, 1.8 * V]))
    assert sweep.level("U1:y").tolist() == [1.0, 0.0]
    assert sweep.voltage("out").tolist() == pytest.approx([1.8, 0.0])


def test_results_files_keep_the_logic() -> None:
    simulation = Simulation(chain())
    analyses = {
        "op": simulation.run(OperatingPoint()),
        "step": simulation.run(Transient(stop=10 * ns)),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "chain.results.npz"
        save_results(path, analyses)
        loaded = load_results(path)
    point, step = loaded["op"], loaded["step"]
    assert isinstance(point, results.OperatingPointResult)
    assert point.nets == simulation.nets
    assert isinstance(step, results.TransientResult)
    assert step.level("y").times.tolist() == pytest.approx([0, 4e-9])


def test_reports_and_plots_state_the_logic() -> None:
    simulation = Simulation(chain())
    point = report("op", simulation.run(OperatingPoint()), 10)
    assert point.describe().splitlines()[1:] == ["  a  0", "  b  1", "  n  1", "  y  0"]
    waveforms = simulation.run(Transient(stop=10 * ns))
    step = report("step", waveforms, 3)
    assert step.describe().splitlines() == [
        "step: transient to 10 ns in 4 steps, 0 rejected",
        "     first  changes  last",
        "  a  0      1        1",
        "  b  1      0        1",
        "  n  1      1        0",
        "  ... and 1 more in the results file",
    ]
    traces = default_traces(waveforms.unknowns, waveforms.nets)
    assert traces == ("a", "b", "n", "y")
    figure = plot("step", waveforms, traces)
    (axes,) = figure.axes
    assert axes.get_ylabel() == "logic"
    assert [label.get_text() for label in axes.get_yticklabels()] == list(traces)


def test_a_logic_cell_takes_tables_or_functions() -> None:
    xor = devices.logic_cell(
        ["a", "b"],
        {"y": "0110", "z": lambda a, b: a and not b},
        rise=1 * ns,
        transition=10 * ps,
    )
    assert xor.terminals == ("a", "b", "y", "z")
    assert xor.parameters["outputs"] == [
        {"name": "y", "table": "0110"},
        {"name": "z", "table": "0010"},
    ]
    with pytest.raises(ValueError, match="only an electrical cell"):
        devices.logic_gate("not", rise=1 * ns, transition=10 * ps, threshold=0.4)
    with pytest.raises(ValueError, match="each of the 4 rows"):
        devices.logic_cell(["a", "b"], {"y": "01"}, rise=1 * ns, transition=10 * ps)
