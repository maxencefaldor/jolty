# Copyright 2026 Maxence Faldor
"""The devices a build script places, each a model, its terminals and its parameters.

Each function checks the dimension of every parameter it reads, so a capacitance
passed where a resistance belongs fails on the line that passed it. Parameters left out
take the engine's defaults.

Examples:
    >>> from jolty.units import kohm
    >>> resistor(10 * kohm).parameters
    {'resistance': 10000.0}
"""

from dataclasses import dataclass

from pydantic import JsonValue

from jolty.units import (
    CAPACITANCE,
    CONDUCTANCE,
    CURRENT,
    INDUCTANCE,
    RESISTANCE,
    TIME,
    VOLTAGE,
    Quantity,
    finite,
)
from jolty.waveforms import Dc, Waveform


@dataclass(frozen=True, slots=True)
class Device:
    """A device ready to place: its model's name, its terminals and its parameters."""

    model: str
    terminals: tuple[str, ...]
    parameters: dict[str, JsonValue]


def resistor(resistance: Quantity) -> Device:
    """A linear resistor between terminals ``a`` and ``b``."""
    return Device(
        "resistor", ("a", "b"), {"resistance": resistance.to(RESISTANCE, "resistance")}
    )


def capacitor(capacitance: Quantity) -> Device:
    """A linear capacitor between terminals ``a`` and ``b``."""
    value = capacitance.to(CAPACITANCE, "capacitance")
    return Device("capacitor", ("a", "b"), {"capacitance": value})


def inductor(inductance: Quantity) -> Device:
    """A linear inductor between terminals ``a`` and ``b``.

    Its current, from ``a`` through the inductor to ``b``, is the unknown ``current``.
    """
    value = inductance.to(INDUCTANCE, "inductance")
    return Device("inductor", ("a", "b"), {"inductance": value})


def voltage_source(waveform: Waveform | Quantity) -> Device:
    """An independent voltage source, ``positive`` minus ``negative`` at its level.

    A quantity is a constant level. The source's current, from ``positive`` through
    the source to ``negative``, is the unknown ``current``: negative while it delivers
    power.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(VOLTAGE)}
    return Device("voltage_source", ("positive", "negative"), parameters)


def current_source(waveform: Waveform | Quantity) -> Device:
    """An independent current source, driving its level from ``positive`` to the other.

    The current flows through the source from ``positive`` to ``negative``, so it enters
    the circuit at the ``negative`` terminal. A quantity is a constant level.
    """
    level = Dc(waveform) if isinstance(waveform, Quantity) else waveform
    parameters: dict[str, JsonValue] = {"waveform": level.to_json(CURRENT)}
    return Device("current_source", ("positive", "negative"), parameters)


_CONTROLLED = ("positive", "negative", "control_positive", "control_negative")


def voltage_controlled_voltage_source(gain: float) -> Device:
    """A source holding ``positive`` minus ``negative`` at `gain` times a voltage.

    That control voltage is ``control_positive`` minus ``control_negative``, and the
    control terminals draw no current.
    """
    parameters: dict[str, JsonValue] = {"gain": finite(gain, "gain")}
    return Device("voltage_controlled_voltage_source", _CONTROLLED, parameters)


def voltage_controlled_current_source(transconductance: Quantity) -> Device:
    """A source driving `transconductance` times the control voltage through itself.

    The current flows from ``positive`` through the source to ``negative``; the control
    voltage is ``control_positive`` minus ``control_negative``.
    """
    value = transconductance.to(CONDUCTANCE, "transconductance")
    parameters: dict[str, JsonValue] = {"transconductance": value}
    return Device("voltage_controlled_current_source", _CONTROLLED, parameters)


def diode(  # noqa: PLR0913 - the SPICE junction model has eight parameters
    *,
    saturation_current: Quantity | None = None,
    emission_coefficient: float | None = None,
    series_resistance: Quantity | None = None,
    junction_capacitance: Quantity | None = None,
    junction_potential: Quantity | None = None,
    grading_coefficient: float | None = None,
    depletion_coefficient: float | None = None,
    transit_time: Quantity | None = None,
) -> Device:
    """A junction diode from ``anode`` to ``cathode``, in the SPICE model.

    Parameters left out take SPICE's defaults: a saturation current of 10 fA, an
    emission coefficient of 1, no series resistance, junction capacitance or transit
    time, a junction potential of 1 V, and grading and depletion coefficients of 0.5.
    """
    quantities = {
        "saturation_current": (saturation_current, CURRENT),
        "series_resistance": (series_resistance, RESISTANCE),
        "junction_capacitance": (junction_capacitance, CAPACITANCE),
        "junction_potential": (junction_potential, VOLTAGE),
        "transit_time": (transit_time, TIME),
    }
    numbers = {
        "emission_coefficient": emission_coefficient,
        "grading_coefficient": grading_coefficient,
        "depletion_coefficient": depletion_coefficient,
    }
    parameters: dict[str, JsonValue] = {
        name: value.to(dimension, name)
        for name, (value, dimension) in quantities.items()
        if value is not None
    }
    parameters |= {
        name: finite(value, name)
        for name, value in numbers.items()
        if value is not None
    }
    return Device("diode", ("anode", "cathode"), parameters)
