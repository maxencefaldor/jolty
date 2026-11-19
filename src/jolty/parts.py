# Copyright 2026 Maxence Faldor
"""Real parts: models of what makers sell, each with its number and its ratings.

Each function returns a device or a component that stands for one part: its model, the
maker's part number, which a bill of materials prices it by, and the absolute maximum
ratings of its datasheet, which a simulation holds every operating point and transient
to. A rating of the power is at a case or an ambient of 25 °C, as its datasheet states,
and the rating of the temperature bounds its junction's.

Each part takes ``heated``: heated, it runs at its junction's temperature, which a
terminal or port ``thermal`` joins to a thermal net, and heats the net by the power it
dissipates; see [`Device.heated`][jolty.devices.Device.heated]. A thermal resistance
from that net to the ground, the ambient, then stands for its datasheet's junction to
ambient, or a network of them for its case and heatsink.

The small-signal parts take the SPICE parameters their makers publish:

- [`q2n3904`][jolty.parts.q2n3904], the NPN transistor, onsemi's;
- [`q2n3906`][jolty.parts.q2n3906], the PNP transistor, onsemi's;
- [`d1n4148`][jolty.parts.d1n4148], the signal diode, onsemi's.

The power MOSFETs' SPICE models carry licences that forbid redistributing them, so each
is instead fitted, by one stated recipe, to the figures its
[datasheet][jolty.parts.Datasheet] states, which are facts anyone may use; see
[`power_mosfet`][jolty.parts.power_mosfet]:

- [`irf511`][jolty.parts.irf511], 80 V and 5.6 A, n-channel;
- [`irf9230`][jolty.parts.irf9230], 200 V and 6.5 A, p-channel;
- [`irfz44`][jolty.parts.irfz44], 60 V and 50 A, n-channel;
- [`mtp50p03hdl`][jolty.parts.mtp50p03hdl], 30 V and 50 A, logic-level p-channel.

Examples:
    >>> transistor = q2n3904()
    >>> transistor.part, [rating.stress for rating in transistor.ratings][-1]
    ('2N3904', 'temperature')
"""

import math
from dataclasses import dataclass
from typing import Final, Literal

from jolty import _core, devices
from jolty.builder import Builder, Component, component
from jolty.units import A, F, V, W, celsius, m, nA, ns, ohm, pF, ps

# The small-signal parts.


def q2n3904(*, heated: bool = False) -> devices.Device:
    """A 2N3904, the small-signal NPN transistor, with its maker's SPICE parameters.

    Rated as onsemi's datasheet states: 40 V from collector to emitter, 60 V from
    collector to base and 6 V from emitter to base, 200 mA into the collector, 625 mW
    at an ambient of 25 °C, and a junction from -55 °C to 150 °C, which 200 K/W join
    to the ambient.
    """
    return (
        _heated(
            devices.bipolar_transistor(
                saturation_current=6.734e-15 * A,
                forward_beta=416.4,
                reverse_beta=0.7371,
                forward_emission_coefficient=1.0,
                forward_early_voltage=74.03 * V,
                forward_knee_current=66.78e-3 * A,
                base_emitter_leakage_current=6.734e-15 * A,
                base_emitter_leakage_emission_coefficient=1.259,
                base_collector_leakage_emission_coefficient=2.0,
                base_resistance=10 * ohm,
                collector_resistance=1 * ohm,
                base_emitter_capacitance=4.493 * pF,
                base_emitter_grading=0.2593,
                base_collector_capacitance=3.638 * pF,
                base_collector_grading=0.3085,
                forward_transit_time=301.2 * ps,
                transit_time_bias_coefficient=2.0,
                transit_time_voltage=4 * V,
                transit_time_current=0.4 * A,
                reverse_transit_time=239.5 * ns,
                beta_temperature_exponent=1.5,
            ),
            heated=heated,
        )
        .numbered("2N3904")
        .rated("voltage", 40 * V, terminals=("collector", "emitter"))
        .rated("voltage", 60 * V, terminals=("collector", "base"))
        .rated("voltage", 6 * V, terminals=("emitter", "base"))
        .rated("current", 0.2 * A, terminals=("collector",))
        .rated("power", 0.625 * W)
        .rated("temperature", celsius(150), minimum=celsius(-55))
    )


def q2n3906(*, heated: bool = False) -> devices.Device:
    """A 2N3906, the small-signal PNP transistor, with its maker's SPICE parameters.

    Rated as onsemi's datasheet states: 40 V from emitter to collector and from base
    to collector, 5 V from base to emitter, 200 mA out of the collector, 625 mW at an
    ambient of 25 °C, and a junction from -55 °C to 150 °C, which 200 K/W join to the
    ambient.
    """
    return (
        _heated(
            devices.bipolar_transistor(
                polarity="pnp",
                saturation_current=1.41e-15 * A,
                forward_beta=180.7,
                reverse_beta=4.977,
                forward_early_voltage=18.7 * V,
                forward_knee_current=80e-3 * A,
                base_emitter_leakage_emission_coefficient=1.5,
                base_collector_leakage_emission_coefficient=2.0,
                base_resistance=10 * ohm,
                collector_resistance=2.5 * ohm,
                base_emitter_capacitance=8.063 * pF,
                base_emitter_grading=0.3677,
                base_collector_capacitance=9.728 * pF,
                base_collector_grading=0.5776,
                forward_transit_time=179.3 * ps,
                transit_time_bias_coefficient=6.0,
                transit_time_voltage=4 * V,
                transit_time_current=0.4 * A,
                reverse_transit_time=33.42 * ns,
                beta_temperature_exponent=1.5,
            ),
            heated=heated,
        )
        .numbered("2N3906")
        .rated("voltage", minimum=-40 * V, terminals=("collector", "emitter"))
        .rated("voltage", minimum=-40 * V, terminals=("collector", "base"))
        .rated("voltage", minimum=-5 * V, terminals=("emitter", "base"))
        .rated("current", minimum=-0.2 * A, terminals=("collector",))
        .rated("power", 0.625 * W)
        .rated("temperature", celsius(150), minimum=celsius(-55))
    )


def d1n4148(*, heated: bool = False) -> devices.Device:
    """A 1N4148, the small-signal silicon diode, with its maker's SPICE parameters.

    Rated as onsemi's datasheet states: 100 V of repetitive reverse voltage, 300 mA of
    forward current, 500 mW at an ambient of 25 °C, and a junction from -55 °C to
    175 °C, which 300 K/W join to the ambient.
    """
    return (
        _heated(
            devices.diode(
                saturation_current=2.52 * nA,
                emission_coefficient=1.752,
                series_resistance=0.568 * ohm,
                junction_capacitance=4 * pF,
                grading_coefficient=0.4,
                transit_time=20 * ns,
            ),
            heated=heated,
        )
        .numbered("1N4148")
        .rated("voltage", minimum=-100 * V)
        .rated("current", 0.3 * A)
        .rated("power", 0.5 * W)
        .rated("temperature", celsius(175), minimum=celsius(-55))
    )


def _heated(device: devices.Device, *, heated: bool) -> devices.Device:
    """`device`, [heated][jolty.devices.Device.heated] if `heated`."""
    return device.heated() if heated else device


# The power MOSFETs.

SATURATION_CURRENT: Final = 10e-12
"""A fitted body diode's saturation current, in amperes."""
CHANNEL_JUNCTIONS: Final = 1e-80
"""The saturation current of a fitted channel's own junctions, in amperes: so small
that they carry nothing below the body diode's drop."""
THERMAL_VOLTAGE: Final = (
    _core.BOLTZMANN * _core.REFERENCE_TEMPERATURE / _core.ELEMENTARY_CHARGE
)
"""The thermal voltage at 27 °C, in volts."""
JUNCTION_POTENTIAL: Final = 0.7
"""A fitted body diode's junction potential, in volts, with a grading coefficient of
0.5."""
LEAST_JUNCTION_TEMPERATURE: Final = -55.0
"""The least junction temperature every sheet here states, in degrees Celsius."""


@dataclass(frozen=True, slots=True)
class Datasheet:
    """What a power MOSFET's datasheet states at 25 °C, its electrical values positive.

    Voltages and currents are magnitudes, whichever the channel. The absolute maximum
    ratings bound the drain over the source, the gate over the source either way, the
    current through the drain as the channel conducts it, the power at a case of
    25 °C, and the junction's temperature.
    """

    part_number: str
    """The maker's part number."""
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
    drain_source_voltage: float
    """VDSS, the most drain-source voltage, in volts."""
    gate_source_voltage: float
    """VGS, the most gate-source voltage either way, in volts."""
    drain_current: float
    """ID, the most continuous drain current at a case of 25 °C, in amperes."""
    power: float
    """PD, the most power at a case of 25 °C, in watts."""
    junction_temperature: float
    """TJ's most, in degrees Celsius; its least is -55 °C for each sheet here."""


IRF511: Final = Datasheet(
    part_number="IRF511",
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
    drain_source_voltage=80.0,
    gate_source_voltage=20.0,
    drain_current=5.6,
    power=43.0,
    junction_temperature=175.0,
)
"""80 V, 5.6 A, n-channel: Harris's IRF510-IRF513 datasheet, gfs Vishay's IRF510's."""

IRF9230: Final = Datasheet(
    part_number="IRF9230",
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
    drain_source_voltage=200.0,
    gate_source_voltage=20.0,
    drain_current=6.5,
    power=75.0,
    junction_temperature=150.0,
)
"""200 V, 6.5 A, p-channel: Intersil's IRF9230-IRF9233 datasheet; diode drop its max."""

IRFZ44: Final = Datasheet(
    part_number="IRFZ44",
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
    drain_source_voltage=60.0,
    gate_source_voltage=20.0,
    drain_current=50.0,
    power=150.0,
    junction_temperature=175.0,
)
"""60 V, 50 A, n-channel: Vishay's IRFZ44 datasheet; on-resistance and drop its max."""

MTP50P03HDL: Final = Datasheet(
    part_number="MTP50P03HDL",
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
    drain_source_voltage=30.0,
    gate_source_voltage=15.0,
    drain_current=50.0,
    power=125.0,
    junction_temperature=150.0,
)
"""30 V, 50 A, logic-level p-channel: onsemi's MTP50P03HDL datasheet."""


def irf511(*, heated: bool = False) -> Component:
    """An IRF511, fitted to [its datasheet][jolty.parts.IRF511]."""
    return power_mosfet(IRF511, heated=heated)


def irf9230(*, heated: bool = False) -> Component:
    """An IRF9230, fitted to [its datasheet][jolty.parts.IRF9230]."""
    return power_mosfet(IRF9230, heated=heated)


def irfz44(*, heated: bool = False) -> Component:
    """An IRFZ44, fitted to [its datasheet][jolty.parts.IRFZ44]."""
    return power_mosfet(IRFZ44, heated=heated)


def mtp50p03hdl(*, heated: bool = False) -> Component:
    """An MTP50P03HDL, fitted to [its datasheet][jolty.parts.MTP50P03HDL]."""
    return power_mosfet(MTP50P03HDL, heated=heated)


def power_mosfet(sheet: Datasheet, *, heated: bool = False) -> Component:
    """The power MOSFET `sheet` states: fitted to it, numbered, and rated as it states.

    Its ports are ``drain``, ``gate`` and ``source``, and ``thermal`` where `heated`,
    the junction both its channel and its body diode heat. Its model is fitted to the
    sheet by one recipe:

    - the channel is SPICE's level 1 square law, its threshold the sheet's, and its
      transconductance parameter the larger of the one that gives the stated
      on-resistance at the stated gate voltage and drain current and the one that
      gives the least transconductance stated; where the latter is larger, a drain
      resistance takes up the on-resistance its channel leaves;
    - the gate-drain capacitance is the reverse transfer capacitance, the gate-source
      capacitance the input capacitance less it, both as overlaps, so constant;
    - the body diode joins the source to the drain, in place of the channel's own
      junctions, whose saturation current of 1e-80 A leaves them no current below the
      diode's drop, its junction capacitance the output capacitance less the reverse
      transfer capacitance at the voltage the sheet states them at, and its series
      resistance the one that gives the stated forward drop at the stated current,
      from a saturation current of 10 pA.

    So it reproduces the sheet's threshold, on-resistance, capacitances and diode drop
    at the points the sheet states them, and a transconductance above the sheet's
    minimum. The square law leaves out what bends a power MOSFET's curves away from
    those points: the mobility's fall at high gate voltages, velocity saturation, the
    gate-drain capacitance's fall with the drain voltage, and self-heating. Its channel
    is also symmetric, as level 1 is: with the gate at the source and the drain driven
    beyond the source by more than the threshold, the channel turns on as though drain
    and source had swapped, and shares the reverse current with the body diode, which
    in a real transistor carries it alone, its body tied to its source. Vendors' level-1
    and level-3 models share this. Of the four here, only the MTP50P03HDL's stated
    diode drop, 2.39 V, lies beyond its threshold.
    """
    n = sheet.channel == "n"
    drain_source = sheet.drain_source_voltage * V
    drain = sheet.drain_current * A
    gate_source = sheet.gate_source_voltage * V
    return (
        fitted_mosfet(sheet, heated=heated)
        .numbered(sheet.part_number)
        .rated(
            "voltage",
            drain_source if n else None,
            minimum=None if n else -drain_source,
            terminals=("drain", "source"),
        )
        .rated(
            "voltage", gate_source, minimum=-gate_source, terminals=("gate", "source")
        )
        .rated(
            "current",
            drain if n else None,
            minimum=None if n else -drain,
            terminals=("drain",),
        )
        .rated("power", sheet.power * W)
        .rated(
            "temperature",
            celsius(sheet.junction_temperature),
            minimum=celsius(LEAST_JUNCTION_TEMPERATURE),
        )
    )


def transconductance_parameter(sheet: Datasheet) -> float:
    """The square law's `kp · w / l`, in A/V², of the MOSFET fitted to `sheet`.

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


def diode_resistance(sheet: Datasheet) -> float:
    """The body diode's series resistance, in ohms, that gives the stated drop."""
    junction = THERMAL_VOLTAGE * math.log1p(sheet.diode_current / SATURATION_CURRENT)
    return (sheet.diode_drop - junction) / sheet.diode_current


def _channel(sheet: Datasheet, drop: float) -> float:
    """The `β` whose channel drops `drop` at the stated on-current and gate voltage."""
    overdrive = sheet.on_gate - sheet.threshold
    return sheet.on_current / (drop * (overdrive - drop / 2))


def _least(sheet: Datasheet) -> float:
    """The `β` that gives the least transconductance stated."""
    return sheet.transconductance**2 / (2 * sheet.transconductance_current)


@component
def fitted_mosfet(c: Builder, sheet: Datasheet, *, heated: bool = False) -> None:
    """The model [`power_mosfet`][jolty.parts.power_mosfet] fits to `sheet`, unrated."""
    drain, gate, source = (c.port(name) for name in ("drain", "gate", "source"))
    sign = 1.0 if sheet.channel == "n" else -1.0
    resistance = drain_resistance(sheet)
    channel = c.place(
        _heated(
            devices.mosfet(
                channel=sheet.channel,
                width=1 * m,
                length=1 * m,
                threshold_voltage=sign * sheet.threshold * V,
                transconductance_parameter=transconductance_parameter(sheet) * A / V**2,
                drain_resistance=resistance * ohm if resistance > 0 else None,
                saturation_current=CHANNEL_JUNCTIONS * A,
                gate_drain_overlap=sheet.reverse_capacitance * F / m,
                gate_source_overlap=(
                    sheet.input_capacitance - sheet.reverse_capacitance
                )
                * F
                / m,
            ),
            heated=heated,
        ),
        name="M1",
    )
    drain_source = sheet.output_capacitance - sheet.reverse_capacitance
    zero_bias = (
        drain_source * (1 + sheet.capacitance_voltage / JUNCTION_POTENTIAL) ** 0.5
    )
    body = c.place(
        _heated(
            devices.diode(
                saturation_current=SATURATION_CURRENT * A,
                series_resistance=diode_resistance(sheet) * ohm,
                junction_capacitance=zero_bias * F,
                junction_potential=JUNCTION_POTENTIAL * V,
                grading_coefficient=0.5,
            ),
            heated=heated,
        ),
        name="D1",
    )
    anode, cathode = (source, drain) if sheet.channel == "n" else (drain, source)
    c.connect(channel.drain, drain)
    c.connect(channel.gate, gate)
    c.connect(channel.source, channel.bulk, source)
    c.connect(body.anode, anode)
    c.connect(body.cathode, cathode)
    if heated:
        c.connect(channel.thermal, body.thermal, c.port("thermal"))
