# Copyright 2026 Maxence Faldor
"""The same circuits simulate the same however many threads run them.

On the random circuits of `strategies`, an operating point, a frequency response, the
noise at a net and the sensitivities at the operating point find the same bits on 1, 2,
3 and 8 threads, as the engine spreads each analysis's frequencies and right-hand sides
over them. A Monte Carlo sample is a function of its seed, its index and the names of
its parts alone, so it is the same drawn alone or among others, and a part added leaves
every other part's draws alone.
"""

import dataclasses

import numpy as np
import pytest
from hypothesis import given

from jolty import (
    Ac,
    Builder,
    Noise,
    OperatingPoint,
    Role,
    Sensitivity,
    Simulation,
    SimulationError,
    circuit,
    component,
    devices,
    ir,
    simulate,
)
from jolty.plan import MonteCarlo, decade
from jolty.units import GHz, Hz, V, kohm
from jolty.variation import Varied, sample, toleranced
from strategies import SETTINGS, circuits

THREADS = (1, 2, 3, 8)
"""The numbers of threads each circuit runs on."""


def _bits(found: object) -> object:
    """Every number a result holds, as bytes, and the rest of it as text."""
    if isinstance(found, np.ndarray):
        return (str(found.dtype), found.shape, found.tobytes())
    if dataclasses.is_dataclass(found) and not isinstance(found, type):
        return tuple(
            (field.name, _bits(getattr(found, field.name)))
            for field in dataclasses.fields(found)
        )
    if isinstance(found, tuple | list):
        return tuple(_bits(each) for each in found)
    if isinstance(found, float):
        return np.float64(found).tobytes()
    return repr(found)


def _simulated(built: ir.Circuit) -> list[object]:
    """What each analysis finds on `built`, as bits, or why it fails."""
    simulation = Simulation(built)
    top = built.components[built.top]
    source = next(
        instance.id
        for instance in top.instances
        if instance.of == ir.DeviceOf(device="voltage_source")
    )
    output = next(
        unknown.name for unknown in simulation.unknowns if unknown.quantity == "voltage"
    )
    frequencies = decade(1 * Hz, 1 * GHz, 10)
    analyses = (
        OperatingPoint(),
        Ac(source, frequencies),
        Noise(output, frequencies, source=source),
        Sensitivity([output]),
    )
    found: list[object] = []
    for analysis in analyses:
        try:
            found.append(_bits(simulation.run(analysis)))
        except SimulationError as error:
            found.append(str(error))
    return found


@SETTINGS
@given(circuits())
def test_the_same_circuits_simulate_the_same_on_any_threads(built: ir.Circuit) -> None:
    found = []
    for count in THREADS:
        with simulate.threads(count):
            found.append(_simulated(built))
    assert all(each == found[0] for each in found[1:])


@component
def _divider(c: Builder) -> None:
    ground = c.port("ground", Role.GROUND)
    source = c.place(devices.voltage_source(5 * V), id="V1")
    upper = c.place(devices.resistor(1 * kohm, tolerance=0.01), id="R1")
    lower = c.place(
        devices.resistor(1 * kohm).toleranced("resistance", 0.01, "normal"), id="R2"
    )
    c.connect(source.positive, upper.a)
    c.connect(upper.b, lower.a, name="out")
    c.connect(lower.b, source.negative, ground)


def test_a_sample_is_the_same_drawn_alone_or_among_others() -> None:
    varied = toleranced(circuit(_divider()))
    among = [sample(varied, 7, index) for index in range(100)]
    for index in (0, 41, 99):
        assert sample(varied, 7, index) == among[index]
    assert among[0] != among[1]
    assert sample(varied, 8, 0) != among[0]


def test_a_part_added_leaves_every_other_draw_alone() -> None:
    varied = toleranced(circuit(_divider()))
    added = Varied("R3", "resistance", 2e3, ir.Tolerance(relative=0.05))
    for index in range(20):
        alone = sample(varied, 3, index)
        more = sample((*varied, added), 3, index)
        assert more.pop("R3") is not None
        assert more == alone


def test_a_pool_holds_a_thread_or_more() -> None:
    assert simulate.pool(3).threads == 3
    for count in (0, -1):
        with pytest.raises(ValueError, match="a thread or more"):
            simulate.pool(count)


def test_a_seed_is_64_bits() -> None:
    assert MonteCarlo(1, seed=2**64 - 1).seed == 2**64 - 1
    for seed in (-1, 2**64):
        with pytest.raises(ValueError, match="seed lies from 0"):
            MonteCarlo(1, seed=seed)
