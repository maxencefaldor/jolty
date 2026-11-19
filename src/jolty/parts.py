# Copyright 2026 Maxence Faldor
"""Real manufacturer part models with part numbers and datasheet ratings.

Each factory builds a device or component tagged with its manufacturer part number for
bill-of-materials pricing and absolute maximum datasheet ratings checked across
operating points and transients. Power ratings apply at 25 °C case or ambient
temperature as stated on each datasheet, and temperature ratings bound junction
temperature.

Passing ``heated=True`` enables self-heating through a ``thermal`` port or terminal
that injects dissipated power into a thermal net; see
[`Device.heated`][jolty.devices.Device.heated]. Connecting a thermal resistance from
that net to ground models junction-to-ambient or heatsink thermal impedance.

Small-signal parts use manufacturer-published SPICE parameters:

- [`q2n3904`][jolty.parts.q2n3904]: onsemi NPN small-signal transistor.
- [`q2n3906`][jolty.parts.q2n3906]: onsemi PNP small-signal transistor.
- [`d1n4148`][jolty.parts.d1n4148]: onsemi small-signal silicon diode.

Power MOSFET models are fitted by a documented recipe to public
[datasheet][jolty.parts.Datasheet] figures rather than redistributing licensed vendor
SPICE models; see [`power_mosfet`][jolty.parts.power_mosfet]:

- [`irf511`][jolty.parts.irf511]: 80 V, 5.6 A n-channel power MOSFET.
- [`irf9230`][jolty.parts.irf9230]: 200 V, 6.5 A p-channel power MOSFET.
- [`irfz44`][jolty.parts.irfz44]: 60 V, 50 A n-channel power MOSFET.
- [`mtp50p03hdl`][jolty.parts.mtp50p03hdl]: 30 V, 50 A logic-level p-channel MOSFET.

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
    """Returns a 2N3904 NPN transistor using onsemi SPICE parameters.

    Datasheet ratings are 40 V collector-emitter, 60 V collector-base, 6 V
    emitter-base, 200 mA collector current, and 625 mW at 25 °C ambient. Junction
    temperature spans -55 °C to 150 °C with 200 K/W thermal resistance to ambient.
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
    """Returns a 2N3906 PNP transistor using onsemi SPICE parameters.

    Datasheet ratings are 40 V emitter-collector and base-collector, 5 V base-emitter,
    200 mA collector current, and 625 mW at 25 °C ambient. Junction temperature spans
    -55 °C to 150 °C with 200 K/W thermal resistance to ambient.
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
    """Returns a 1N4148 small-signal silicon diode using onsemi SPICE parameters.

    Datasheet ratings are 100 V repetitive reverse voltage, 300 mA forward current,
    and 500 mW at 25 °C ambient. Junction temperature spans -55 °C to 175 °C with
    300 K/W thermal resistance to ambient.
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
    """Returns `device` with [heating][jolty.devices.Device.heated] when `heated`."""
    return device.heated() if heated else device


# Power MOSFET models.

SATURATION_CURRENT: Final = 10e-12
"""Saturation current of a fitted power MOSFET body diode, in amperes."""
CHANNEL_JUNCTIONS: Final = 1e-80
"""Negligible saturation current (in amperes) suppressing intrinsic channel diodes."""
THERMAL_VOLTAGE: Final = (
    _core.BOLTZMANN * _core.REFERENCE_TEMPERATURE / _core.ELEMENTARY_CHARGE
)
"""Thermal voltage at 27 °C, in volts."""
JUNCTION_POTENTIAL: Final = 0.7
"""Built-in junction potential (in volts) of a fitted body diode at grading 0.5."""
MIN_JUNCTION_TEMPERATURE: Final = -55.0
"""Minimum rated junction temperature across built-in power MOSFET sheets, in °C."""


@dataclass(frozen=True, slots=True)
class Datasheet:
    """Power MOSFET datasheet parameters specified at 25 °C as positive magnitudes.

    Voltages and currents are stored as positive magnitudes for both n-channel and
    p-channel parts. Absolute maximum ratings bound drain-source voltage, bipolar
    gate-source voltage, continuous drain current, 25 °C case power dissipation, and
    junction temperature.
    """

    part_number: str
    """Manufacturer part number."""
    channel: Literal["n", "p"]
    threshold: float
    """Gate-source threshold voltage VGS(th) in volts (typical or range midpoint)."""
    on_resistance: float
    """Typical drain-source on-resistance RDS(on) in ohms."""
    on_gate: float
    """Gate-source voltage in volts used for the RDS(on) measurement."""
    on_current: float
    """Drain current in amperes used for the RDS(on) measurement."""
    transconductance: float
    """Minimum forward transconductance gfs in siemens."""
    transconductance_current: float
    """Saturation drain current in amperes for the gfs measurement."""
    input_capacitance: float
    """Typical input capacitance Ciss in farads."""
    output_capacitance: float
    """Typical output capacitance Coss in farads."""
    reverse_capacitance: float
    """Typical reverse transfer capacitance Crss in farads."""
    capacitance_voltage: float
    """Drain-source voltage in volts for capacitance measurements at VGS = 0 V."""
    diode_drop: float
    """Body-diode forward voltage drop VSD in volts."""
    diode_current: float
    """Source-drain current in amperes for the VSD measurement."""
    drain_source_voltage: float
    """Maximum rated drain-source voltage VDSS in volts."""
    gate_source_voltage: float
    """Maximum rated bipolar gate-source voltage VGS in volts."""
    drain_current: float
    """Maximum continuous drain current ID at 25 °C case temperature, in amperes."""
    power: float
    """Maximum power dissipation PD at 25 °C case temperature, in watts."""
    junction_temperature: float
    """Maximum rated junction temperature TJ in °C (minimum is -55 °C)."""


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
"""80 V, 5.6 A n-channel datasheet (Harris IRF510-IRF513; gfs from Vishay IRF510)."""

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
"""200 V, 6.5 A p-channel datasheet (Intersil IRF9230-IRF9233; maximum diode drop)."""

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
"""60 V, 50 A n-channel datasheet (Vishay IRFZ44; maximum on-resistance and drop)."""

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
"""30 V, 50 A logic-level p-channel datasheet (onsemi MTP50P03HDL)."""


def irf511(*, heated: bool = False) -> Component:
    """Returns an IRF511 power MOSFET fitted to [its datasheet][jolty.parts.IRF511]."""
    return power_mosfet(IRF511, heated=heated)


def irf9230(*, heated: bool = False) -> Component:
    """Returns an IRF9230 MOSFET fitted to [its datasheet][jolty.parts.IRF9230]."""
    return power_mosfet(IRF9230, heated=heated)


def irfz44(*, heated: bool = False) -> Component:
    """Returns an IRFZ44 MOSFET fitted to [its datasheet][jolty.parts.IRFZ44]."""
    return power_mosfet(IRFZ44, heated=heated)


def mtp50p03hdl(*, heated: bool = False) -> Component:
    """Returns an MTP50P03HDL MOSFET fitted to [its sheet][jolty.parts.MTP50P03HDL]."""
    return power_mosfet(MTP50P03HDL, heated=heated)


def power_mosfet(sheet: Datasheet, *, heated: bool = False) -> Component:
    """Builds a numbered, rated power MOSFET component fitted to `sheet`.

    Exposed ports are ``drain``, ``gate``, and ``source``, plus ``thermal`` when
    `heated` is true so both the channel and body diode heat the same junction. The
    model fits `sheet` with three components:

    - The channel uses SPICE level 1 with the datasheet threshold. Transconductance
      `β` takes the larger of the RDS(on) fit and the minimum `gfs` fit; when `gfs`
      dominates, series drain resistance supplies the residual RDS(on).

    - Constant overlap capacitances model gate-drain (`Crss`) and gate-source
      (`Ciss - Crss`).

    - An explicit body diode spans source to drain while intrinsic channel diodes use
      1e-80 A saturation current. Zero-bias junction capacitance matches `Coss - Crss`
      at the measurement voltage, and series resistance matches the stated forward drop
      from 10 pA saturation current.

    The fitted component reproduces threshold, RDS(on), capacitances, and body-diode
    drop at the datasheet test points while meeting minimum `gfs`. Level-1 equations
    omit high-field mobility degradation, velocity saturation, nonlinear gate-drain
    capacitance, and self-heating without a thermal network. Level-1 symmetry causes
    the channel to conduct in reverse alongside the body diode when the drain exceeds
    the source beyond threshold at VGS = 0, matching standard vendor level-1 and
    level-3 subcircuits. Among the built-in parts, only MTP50P03HDL has a stated diode
    drop (2.39 V) above its threshold.
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
            minimum=celsius(MIN_JUNCTION_TEMPERATURE),
        )
    )


def transconductance_parameter(sheet: Datasheet) -> float:
    """Computes the square-law `kp · w / l` parameter (A/V²) fitted to `sheet`.

    Selects the larger of two candidates: the linear-region `β` matching RDS(on) via
    `id = β · vds · (vov - vds / 2)` at overdrive `vov`, and the saturation `β`
    matching minimum transconductance `√(2 · β · id)`.
    """
    return max(_channel(sheet, sheet.on_resistance * sheet.on_current), _least(sheet))


def drain_resistance(sheet: Datasheet) -> float:
    """Computes the extra series drain resistance (ohms) needed beyond channel RDS(on).

    Returns zero when the channel alone achieves RDS(on), and returns the residual
    resistance when minimum transconductance requires a larger `β`.
    """
    beta = _least(sheet)
    if beta <= _channel(sheet, sheet.on_resistance * sheet.on_current):
        return 0.0
    overdrive = sheet.on_gate - sheet.threshold
    # Channel voltage drop at the stated current: root of id = β · v · (vov - v / 2).
    drop = overdrive - math.sqrt(overdrive**2 - 2 * sheet.on_current / beta)
    return sheet.on_resistance - drop / sheet.on_current


def diode_resistance(sheet: Datasheet) -> float:
    """Computes the body-diode series resistance (ohms) matching the datasheet drop."""
    junction = THERMAL_VOLTAGE * math.log1p(sheet.diode_current / SATURATION_CURRENT)
    return (sheet.diode_drop - junction) / sheet.diode_current


def _channel(sheet: Datasheet, drop: float) -> float:
    """Computes the channel `β` producing voltage `drop` at the on-resistance point."""
    overdrive = sheet.on_gate - sheet.threshold
    return sheet.on_current / (drop * (overdrive - drop / 2))


def _least(sheet: Datasheet) -> float:
    """Computes the minimum `β` required to match the datasheet transconductance."""
    return sheet.transconductance**2 / (2 * sheet.transconductance_current)


@component
def fitted_mosfet(c: Builder, sheet: Datasheet, *, heated: bool = False) -> None:
    """Builds the unrated power MOSFET subcircuit fitted to `sheet`."""
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
