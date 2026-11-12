# Copyright 2026 Maxence Faldor
"""Power MOSFETs fitted to their makers' datasheets, for the Koza tasks that use them.

Koza's variable capacitor (US 6,013,958) uses the IRF511 and the IRF9230, and his
high-current load (US 6,211,726) the IRFZ44 and the MTP50P03HDL. Their SPICE models'
public copies carry licences that forbid redistributing them, so each transistor here is
instead fitted, by one stated recipe, to the figures its datasheet states, which are
facts anyone may use:

- the channel is SPICE's level 1 square law, its threshold the datasheet's, typical or
  the middle of its range, and its transconductance parameter the larger of the one
  that gives the stated on-resistance at the stated gate voltage and drain current and
  the one that gives the least transconductance stated; where the latter is larger, a
  drain resistance takes up the on-resistance its channel leaves;
- the gate-drain capacitance is the reverse transfer capacitance, the gate-source
  capacitance the input capacitance less it, both as overlaps, so constant;
- the body diode joins the source to the drain, in place of the channel's own
  junctions, whose saturation current of 1e-80 A leaves them no current below the
  diode's drop, its junction capacitance the output
  capacitance less the reverse transfer capacitance at the voltage the datasheet states
  them at, and its series resistance the one that gives the stated forward drop at the
  stated current, from a saturation current of 10 pA.

So each reproduces its datasheet's threshold, on-resistance, capacitances and diode drop
at the points the datasheet states them, and a transconductance above the datasheet's
minimum. The square law leaves out what bends a power MOSFET's curves away from those
points: the mobility's fall at high gate voltages, velocity saturation, the
gate-drain capacitance's fall with the drain voltage, and self-heating. Its channel is
also symmetric, as level 1 is: with the gate at the source and the drain driven beyond
the source by more than the threshold, the channel turns on as though drain and source
had swapped, and shares the reverse current with the body diode, which in a real
transistor carries it alone, its body tied to its source. Vendors' level-1 and level-3
models share this. Of these four, only the MTP50P03HDL's stated diode drop, 2.39 V,
lies beyond its threshold.
"""

import math
from dataclasses import dataclass
from typing import Literal

from jolty import Builder, component, devices
from jolty.units import A, F, V, m, ohm

SATURATION_CURRENT = 10e-12
"""The body diode's saturation current, in amperes."""
CHANNEL_JUNCTIONS = 1e-80
"""The saturation current of the channel's own junctions, in amperes: so small that
they carry nothing below the body diode's drop."""
THERMAL_VOLTAGE = 1.380649e-23 * 300.15 / 1.602176634e-19
"""The thermal voltage at 27 °C, in volts."""
JUNCTION_POTENTIAL = 0.7
"""The body diode's junction potential, in volts, with a grading coefficient of 0.5."""


@dataclass(frozen=True, slots=True)
class Datasheet:
    """What a power MOSFET's datasheet states, at 25 °C, every value positive.

    Voltages and currents are magnitudes, whichever the channel.
    """

    channel: Literal["n", "p"]
    threshold: float
    """VGS(th), in volts: typical, or the middle of its range."""
    on_resistance: float
    """RDS(on), typical where stated, in ohms, ..."""
    on_gate: float
    """... at this gate-source voltage, in volts, ..."""
    on_current: float
    """... and this drain current, in amperes."""
    transconductance: float
    """The least gfs, in siemens, ..."""
    transconductance_current: float
    """... at this drain current, in amperes, in saturation."""
    input_capacitance: float
    """Ciss, typical, in farads, ..."""
    output_capacitance: float
    """Coss, typical, in farads, ..."""
    reverse_capacitance: float
    """Crss, typical, in farads, ..."""
    capacitance_voltage: float
    """... at this drain-source voltage, in volts, the gate at the source."""
    diode_drop: float
    """VSD, typical where stated, in volts, ..."""
    diode_current: float
    """... at this source-drain current, in amperes."""


IRF511 = Datasheet(
    channel="n",
    threshold=3.0,
    on_resistance=0.4,
    on_gate=10.0,
    on_current=3.4,
    transconductance=1.3,
    transconductance_current=3.4,
    input_capacitance=135e-12,
    output_capacitance=80e-12,
    reverse_capacitance=20e-12,
    capacitance_voltage=25.0,
    diode_drop=2.5,
    diode_current=5.6,
)
"""80 V, 5.6 A, n-channel: Harris's IRF510-IRF513 datasheet, gfs Vishay's IRF510's."""

IRF9230 = Datasheet(
    channel="p",
    threshold=3.0,
    on_resistance=0.5,
    on_gate=10.0,
    on_current=3.5,
    transconductance=2.2,
    transconductance_current=3.5,
    input_capacitance=550e-12,
    output_capacitance=170e-12,
    reverse_capacitance=50e-12,
    capacitance_voltage=25.0,
    diode_drop=1.5,
    diode_current=6.5,
)
"""200 V, 6.5 A, p-channel: Intersil's IRF9230-IRF9233 datasheet; diode drop its max."""

IRFZ44 = Datasheet(
    channel="n",
    threshold=3.0,
    on_resistance=0.028,
    on_gate=10.0,
    on_current=31.0,
    transconductance=15.0,
    transconductance_current=31.0,
    input_capacitance=1900e-12,
    output_capacitance=920e-12,
    reverse_capacitance=170e-12,
    capacitance_voltage=25.0,
    diode_drop=2.5,
    diode_current=51.0,
)
"""60 V, 50 A, n-channel: Vishay's IRFZ44 datasheet; on-resistance and drop its max."""

MTP50P03HDL = Datasheet(
    channel="p",
    threshold=1.5,
    on_resistance=0.020,
    on_gate=5.0,
    on_current=25.0,
    transconductance=15.0,
    transconductance_current=25.0,
    input_capacitance=3500e-12,
    output_capacitance=1550e-12,
    reverse_capacitance=550e-12,
    capacitance_voltage=25.0,
    diode_drop=2.39,
    diode_current=50.0,
)
"""30 V, 50 A, logic-level p-channel: onsemi's MTP50P03HDL datasheet."""


def transconductance_parameter(sheet: Datasheet) -> float:
    """The square law's `kp · w / l`, in A/V².

    It is the larger of the one whose channel alone gives the stated on-resistance,
    below saturation, where `id = β · vds · (vov - vds / 2)` at the stated gate
    voltage's overdrive `vov`, and the one that gives the least transconductance,
    `√(2 · β · id)` in saturation at its stated current.
    """
    return max(_channel(sheet, sheet.on_resistance * sheet.on_current), _least(sheet))


def drain_resistance(sheet: Datasheet) -> float:
    """The drain resistance, in ohms, taking up the on-resistance the channel leaves.

    It is zero where the channel alone gives the on-resistance, as it does unless the
    least transconductance asks for a larger `β`.
    """
    beta = _least(sheet)
    if beta <= _channel(sheet, sheet.on_resistance * sheet.on_current):
        return 0.0
    overdrive = sheet.on_gate - sheet.threshold
    # The channel's drop at the stated current: the root of id = β · v · (vov - v / 2).
    drop = overdrive - math.sqrt(overdrive**2 - 2 * sheet.on_current / beta)
    return sheet.on_resistance - drop / sheet.on_current


def _channel(sheet: Datasheet, drop: float) -> float:
    """The `β` whose channel drops `drop` at the stated on-current and gate voltage."""
    overdrive = sheet.on_gate - sheet.threshold
    return sheet.on_current / (drop * (overdrive - drop / 2))


def _least(sheet: Datasheet) -> float:
    """The `β` that gives the least transconductance stated."""
    return sheet.transconductance**2 / (2 * sheet.transconductance_current)


def diode_resistance(sheet: Datasheet) -> float:
    """The body diode's series resistance, in ohms, that gives the stated drop."""
    junction = THERMAL_VOLTAGE * math.log1p(sheet.diode_current / SATURATION_CURRENT)
    return (sheet.diode_drop - junction) / sheet.diode_current


@component
def power_mosfet(c: Builder, sheet: Datasheet) -> None:
    """The transistor `sheet` states, its ports ``drain``, ``gate`` and ``source``."""
    drain, gate, source = (c.port(name) for name in ("drain", "gate", "source"))
    sign = 1.0 if sheet.channel == "n" else -1.0
    resistance = drain_resistance(sheet)
    channel = c.place(
        devices.mosfet(
            channel=sheet.channel,
            width=1 * m,
            length=1 * m,
            threshold_voltage=sign * sheet.threshold * V,
            transconductance_parameter=transconductance_parameter(sheet) * A / V**2,
            drain_resistance=resistance * ohm if resistance > 0 else None,
            saturation_current=CHANNEL_JUNCTIONS * A,
            gate_drain_overlap=sheet.reverse_capacitance * F / m,
            gate_source_overlap=(sheet.input_capacitance - sheet.reverse_capacitance)
            * F
            / m,
        ),
        id="M1",
    )
    drain_source = sheet.output_capacitance - sheet.reverse_capacitance
    zero_bias = (
        drain_source * (1 + sheet.capacitance_voltage / JUNCTION_POTENTIAL) ** 0.5
    )
    body = c.place(
        devices.diode(
            saturation_current=SATURATION_CURRENT * A,
            series_resistance=diode_resistance(sheet) * ohm,
            junction_capacitance=zero_bias * F,
            junction_potential=JUNCTION_POTENTIAL * V,
            grading_coefficient=0.5,
        ),
        id="D1",
    )
    anode, cathode = (source, drain) if sheet.channel == "n" else (drain, source)
    c.connect(channel.drain, drain)
    c.connect(channel.gate, gate)
    c.connect(channel.source, channel.bulk, source)
    c.connect(body.anode, anode)
    c.connect(body.cathode, cathode)
