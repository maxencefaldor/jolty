# Copyright 2026 Maxence Faldor
"""SPICE `.model` card definitions and conversion to jolty device parameters.

SPICE card parameters map to jolty device parameters converted into SI base units
(for example, centimeter quantities to meters and ``TNOM`` from degrees Celsius to
kelvins). Parameters with no effect on jolty simulation are ignored explicitly:
noise-equation selectors, VDMOS part annotations, VDMOS ``PHI`` and thermal networks,
and BSIM bin bounds, safe-operating-area limits, and disabled submodel parameters.
Any other unsupported parameter raises an error rather than being silently dropped.
"""

import functools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

from jolty import _core
from jolty.devices import BSIM3_BINNED, BSIM3_CARD, BSIM4_BINNED, BSIM4_CARD
from jolty.errors import SpiceError
from jolty.spice.expressions import (
    Draw,
    Expression,
    Function,
    Number,
    evaluate,
    inlined,
    nominal,
    number,
    parse,
)
from jolty.spice.netlist import Statement, tokens
from jolty.units import celsius

type Table = Mapping[str, tuple[str, float]]
"""Mapping from SPICE parameter names to `(jolty_name, si_scale_factor)`."""

_EMPTY_STRINGS: Final[Mapping[str, str]] = MappingProxyType({})
_EMPTY_SCALES: Final[Mapping[str, float]] = MappingProxyType({})
_EMPTY_TABLE: Final[Table] = MappingProxyType({})


def _table(  # noqa: PLR0913 - parameter table builder options
    names: Mapping[str, str],
    *,
    drifts: Sequence[str] = (),
    exponents: Sequence[str] = (),
    temperature_pairs: Mapping[str, str] = _EMPTY_STRINGS,
    scales: Mapping[str, float] = _EMPTY_SCALES,
    extra: Table = _EMPTY_TABLE,
) -> dict[str, tuple[str, float]]:
    """Builds a SPICE-to-jolty parameter `Table`, keeping canonical names first.

    Keys in `names` list the canonical SPICE name followed by space-separated aliases.
    `drifts` and `exponents` expand ``t{base}1``/``t{base}2`` and ``x{base}`` (plus
    optional space-separated aliases) from `names`; `temperature_pairs` expands custom
    prefixes; `scales` supplies non-unit SI factors; and `extra` appends entries last.
    """
    built: dict[str, tuple[str, float]] = {
        spice: (jolty, scales.get(spice, 1.0))
        for keys, jolty in names.items()
        for spice in keys.split()
    }
    for entry in drifts:
        base, *aliases = entry.split()
        parameter = built[base][0]
        built[f"t{base}1"] = (f"{parameter}_temperature_coefficient", 1.0)
        built[f"t{base}2"] = (f"{parameter}_quadratic_temperature_coefficient", 1.0)
        built |= dict.fromkeys(aliases, (f"{parameter}_temperature_coefficient", 1.0))
    for entry in exponents:
        base, *aliases = entry.split()
        target = (f"{built[base][0]}_temperature_exponent", 1.0)
        built |= dict.fromkeys((f"x{base}", *aliases), target)
    for prefix, parameter in temperature_pairs.items():
        built[f"{prefix}1"] = (f"{parameter}_temperature_coefficient", 1.0)
        built[f"{prefix}2"] = (f"{parameter}_quadratic_temperature_coefficient", 1.0)
    built |= extra
    return built


FLICKER: Final[Table] = {
    "kf": ("flicker_coefficient", 1.0),
    "af": ("flicker_exponent", 1.0),
}
"""SPICE flicker noise parameters ``KF`` and ``AF`` for diodes and transistors."""

PASSIVE_DRIFT: Final = {
    "tc1": "temperature_coefficient",
    "tc2": "quadratic_temperature_coefficient",
    "tce": "exponential_temperature_coefficient",
}
"""SPICE temperature coefficient names for resistors, capacitors, and inductors."""

PASSIVE_CURVATURE: Final = {
    "vc1": "voltage_coefficient",
    "vc2": "quadratic_voltage_coefficient",
}
"""HSPICE voltage coefficient names for resistors and capacitors."""

DIODE: Final[Table] = _table(
    {
        "is js": "saturation_current",
        "n": "emission_coefficient",
        "rs": "series_resistance",
        "cjo cj0 cj": "junction_capacitance",
        "vj pb": "junction_potential",
        "m mj": "grading_coefficient",
        "fc": "depletion_coefficient",
        "tt": "transit_time",
        "ikf ik": "forward_knee_current",
        "ikr": "reverse_knee_current",
        "isr": "recombination_saturation_current",
        "nr": "recombination_emission_coefficient",
        "jsw isw": "sidewall_saturation_current",
        "ns": "sidewall_emission_coefficient",
        "rsw": "sidewall_series_resistance",
        "cjp cjsw": "sidewall_junction_capacitance",
        "php vjsw": "sidewall_junction_potential",
        "mjsw": "sidewall_grading_coefficient",
        "fcs": "sidewall_depletion_coefficient",
        "ikp": "sidewall_forward_knee_current",
        "jtun": "tunneling_saturation_current",
        "jtunsw": "sidewall_tunneling_saturation_current",
        "ntun": "tunneling_emission_coefficient",
        "xtitun": "tunneling_saturation_current_temperature_exponent",
        "keg": "tunneling_energy_gap_factor",
        "bv vb vrb var": "breakdown_voltage",
        "ibv ib": "breakdown_current",
        "nbv nz": "breakdown_emission_coefficient",
        "tcv tbv1": "breakdown_temperature_coefficient",
        "tlev": "temperature_model",
        "tlevc": "capacitance_temperature_model",
        "eg": "energy_gap",
        "gap1": "energy_gap_alpha",
        "gap2": "energy_gap_beta",
        "xti": "saturation_current_temperature_exponent",
        "cta ctc": "junction_capacitance_temperature_coefficient",
        "ctp": "sidewall_junction_capacitance_temperature_coefficient",
        "tpb tvj": "junction_potential_temperature_coefficient",
        "tphp": "sidewall_junction_potential_temperature_coefficient",
        "trs": "series_resistance_temperature_coefficient",
    },
    drifts=("rs", "tt", "m"),
    extra=FLICKER,
)
"""ngspice 47 diode parameters (``dio.c``, lines 48 to 147).

Geometry and ignored safe-operating-area parameters live in `DIODE_GEOMETRY` and
`DIODE_IGNORED`.
"""

DIODE_BY_AREA: Final = (
    *("is", "js", "cjo", "cj0", "cj"),
    *("ikf", "ik", "ikr", "isr", "jtun"),
)
"""Diode bottom-junction parameters scaled linearly by area and multiplicity.

Reference: ``diosetup.c`` (lines 362 to 365) and ``diotemp.c`` (lines 120, 136, 147).
Series resistance ``RS`` divides by the same factor (``diotemp.c``, line 254).
"""

DIODE_BY_PERIMETER: Final = ("jsw", "isw", "cjp", "cjsw", "ikp", "jtunsw")
"""Diode sidewall parameters scaled linearly by perimeter and multiplicity.

Reference: ``diosetup.c`` (lines 364, 366) and ``diotemp.c`` (lines 128, 139).
Sidewall resistance ``RSW`` divides by the same factor (``diotemp.c``, line 262).
"""

DIODE_KNEES: Final = frozenset(
    name for name, (jolty, _) in DIODE.items() if jolty.endswith("knee_current")
)
"""Diode knee-current parameters disabled by ngspice below `MIN_CURRENT`."""

DIODE_SELECTORS: Final = frozenset({"tlev", "tlevc"})
"""Integer diode temperature-equation selector parameter names."""

DIODE_GEOMETRY: Final = {
    "area": 1.0,
    **dict.fromkeys(("pj", "lm", "lp", "wm", "wp", "xm", "xp", "xw"), 0.0),
    **dict.fromkeys(("xom", "xoi"), 1e4),
}
"""Default diode card geometry parameters (``diosetup.c``, lines 183-188, 281-307).

Covers default area and perimeter factors used when an instance omits them, plus
level-3 metal and polysilicon dimensions, oxide thicknesses, and mask offsets.
"""

METAL_AND_POLY: Final = ("lm", "lp", "wm", "wp")
"""Level-3 diode metal and polysilicon bulk dimension parameter names.

Nonzero metal or polysilicon dimensions are unsupported and rejected during import.
"""

DIODE_IGNORED: Final = frozenset(
    {"fv_max", "bv_max", "id_max", "te_max", "pd_max", "d", "scalm"}
)
"""Diode card parameters ignored during simulation (``inpgmod.c``, lines 157-191).

Includes safe-operating-area limits, the ``D`` type flag, and HSPICE ``SCALM``.
"""

MIN_CURRENT: Final = 1e-28
"""ngspice ``epsmin`` floor (``cktinit.c``, line 98) for diode saturation currents.

Diode knee currents below this threshold are disabled (``diosetup.c``, lines 110-130,
230).
"""

JFET: Final[Table] = _table(
    {
        "vto": "threshold_voltage",
        "beta": "transconductance_parameter",
        "lambda": "channel_length_modulation",
        "rd": "drain_resistance",
        "rs": "source_resistance",
        "cgs": "gate_source_capacitance",
        "cgd": "gate_drain_capacitance",
        "pb": "junction_potential",
        "is": "saturation_current",
        "fc": "depletion_coefficient",
        "b": "doping_tail",
        "vtotc tcv": "threshold_voltage_temperature_coefficient",
        "bex": "transconductance_parameter_temperature_exponent",
        "eg": "energy_gap",
        "xti": "saturation_current_temperature_exponent",
    },
    scales={"tcv": -1.0},
    extra=FLICKER,
)

BIPOLAR: Final[Table] = _table(
    {
        "is": "saturation_current",
        "bf": "forward_beta",
        "br": "reverse_beta",
        "nf": "forward_emission_coefficient",
        "nr": "reverse_emission_coefficient",
        "vaf va": "forward_early_voltage",
        "var vb": "reverse_early_voltage",
        "ikf ik": "forward_knee_current",
        "ikr": "reverse_knee_current",
        "ise": "base_emitter_leakage_current",
        "ne": "base_emitter_leakage_emission_coefficient",
        "isc": "base_collector_leakage_current",
        "nc": "base_collector_leakage_emission_coefficient",
        "rb": "base_resistance",
        "rbm": "min_base_resistance",
        "irb": "base_resistance_current",
        "re": "emitter_resistance",
        "rc": "collector_resistance",
        "cje": "base_emitter_capacitance",
        "vje pe": "base_emitter_potential",
        "mje me": "base_emitter_grading",
        "tf": "forward_transit_time",
        "xtf": "transit_time_bias_coefficient",
        "vtf": "transit_time_voltage",
        "itf": "transit_time_current",
        "cjc": "base_collector_capacitance",
        "vjc pc": "base_collector_potential",
        "mjc mc": "base_collector_grading",
        "xcjc": "internal_base_fraction",
        "tr": "reverse_transit_time",
        "fc": "depletion_coefficient",
        "ptf": "excess_phase",
        "iss": "substrate_saturation_current",
        "ns": "substrate_emission_coefficient",
        "cjs csub ccs": "substrate_capacitance",
        "vjs ps": "substrate_potential",
        "mjs ms": "substrate_grading",
        "eg": "energy_gap",
        "xti": "saturation_current_temperature_exponent",
        "xtb": "beta_temperature_exponent",
        "nkf nk": "high_injection_exponent",
        "cte": "base_emitter_capacitance_temperature_coefficient",
        "ctc": "base_collector_capacitance_temperature_coefficient",
        "cts": "substrate_capacitance_temperature_coefficient",
        "tvje": "base_emitter_potential_temperature_coefficient",
        "tvjc": "base_collector_potential_temperature_coefficient",
        "tvjs": "substrate_potential_temperature_coefficient",
    },
    drifts=(
        *("bf", "br", "ikf", "ikr", "vaf", "var", "rb trb", "irb", "rc trc", "re tre"),
        *("nf", "nr", "ne", "nc", "ns", "itf", "tf", "tr", "mje", "mjc", "mjs"),
    ),
    temperature_pairs={"trm": "min_base_resistance"},
    extra=FLICKER,
)

MOSFET: Final[Table] = _table(
    {
        "vto": "threshold_voltage",
        "kp": "transconductance_parameter",
        "gamma": "body_effect",
        "phi": "surface_potential",
        "lambda": "channel_length_modulation",
        "ld": "lateral_diffusion",
        "tox": "oxide_thickness",
        "uo": "surface_mobility",
        "nsub": "substrate_doping",
        "nss": "surface_state_density",
        "rd": "drain_resistance",
        "rs": "source_resistance",
        "rsh": "sheet_resistance",
        "is": "saturation_current",
        "js": "saturation_current_density",
        "pb": "junction_potential",
        "cbd": "bulk_drain_capacitance",
        "cbs": "bulk_source_capacitance",
        "cj": "bottom_capacitance",
        "cjsw": "sidewall_capacitance",
        "mj": "bottom_grading",
        "mjsw": "sidewall_grading",
        "fc": "depletion_coefficient",
        "cgso": "gate_source_overlap",
        "cgdo": "gate_drain_overlap",
        "cgbo": "gate_bulk_overlap",
        "vmax": "max_drift_velocity",
        "xj": "junction_depth",
        "nfs": "fast_surface_state_density",
        "eta": "static_feedback",
        "delta": "narrow_width_factor",
        "theta": "mobility_modulation",
        "kappa": "saturation_field_factor",
        "wd": "width_narrowing",
        "xl": "length_adjustment",
        "xw": "width_adjustment",
        "delvto delvt0": "threshold_adjustment",
    },
    scales={"uo": 1e-4, "nsub": 1e6, "nss": 1e4, "nfs": 1e4},
    extra=FLICKER,
)

MOSFET_INSTANCE: Final[Table] = _table(
    {
        "w": "width",
        "l": "length",
        "m": "multiplier",
        "nrd": "drain_squares",
        "nrs": "source_squares",
        "ad": "drain_area",
        "as": "source_area",
        "pd": "drain_perimeter",
        "ps": "source_perimeter",
    }
)

VDMOS: Final[Table] = _table(
    {
        "vto vth0": "threshold_voltage",
        "kp": "transconductance_parameter",
        "lambda": "channel_length_modulation",
        "theta": "mobility_modulation",
        "mtriode": "triode_conductance_multiplier",
        "ksubthres": "subthreshold_slope",
        "subshift": "subthreshold_shift",
        "rd": "drain_resistance",
        "rs": "source_resistance",
        "rg": "gate_resistance",
        "rq": "quasi_saturation_resistance",
        "vq": "quasi_saturation_voltage",
        "rds": "drain_source_resistance",
        "cgs": "gate_source_capacitance",
        "cgdmax": "max_gate_drain_capacitance",
        "cgdmin": "min_gate_drain_capacitance",
        "a": "gate_drain_capacitance_coefficient",
        "is": "saturation_current",
        "n": "emission_coefficient",
        "rb": "body_resistance",
        "cjo": "junction_capacitance",
        "vj": "junction_potential",
        "m": "grading_coefficient",
        "fc": "depletion_coefficient",
        "tt": "transit_time",
        "bv": "breakdown_voltage",
        "ibv": "breakdown_current",
        "nbv": "breakdown_emission_coefficient",
        "eg": "energy_gap",
        "xti": "saturation_current_temperature_exponent",
        "tcvth vtotc": "threshold_voltage_temperature_coefficient",
        "mu bex": "transconductance_parameter_temperature_exponent",
        "texp0": "drain_resistance_temperature_exponent",
        "texp1": "quasi_saturation_resistance_temperature_exponent",
    },
    drifts=("ksubthres", "rd", "rs", "rg", "rb"),
    extra=FLICKER,
)
"""ngspice VDMOS power MOSFET parameters matching LTspice model cards."""

VDMOS_INSTANCE: Final[Table] = _table({"m": "multiplier"})
"""VDMOS instance parameters (parallel device multiplier ``M``)."""

VDMOS_KINDS: Final = frozenset({"vdmosn", "vdmosp"})
"""Normalized VDMOS card type names encoding channel polarity."""

VDMOS_ANNOTATIONS: Final = frozenset({"mfg", "vds", "ron", "qg"})
"""LTspice VDMOS part metadata keys (manufacturer, ratings, gate charge) ignored by
ngspice and jolty."""

VDMOS_SELF_HEATING: Final = frozenset({"rthjc", "rthca", "cthj"})
"""VDMOS thermal-network card parameters ignored because jolty does not import
self-heating VDMOS instances."""

_NO_FUNCTIONS: Final[Mapping[str, Function]] = MappingProxyType({})
"""Empty user-function mapping used when a caller supplies none."""

_WRAPPED_PREFIX: Final = re.compile(r"[\w\s=.+-]*")
"""Parameter assignments that may precede an enclosing ``(...)`` group, as in
``NPN Level=4(...)``, parsed identically to assignments inside parentheses."""

_BARE_EQUALS: Final = re.compile(r"(?<![=!<>])=(?!=)")
"""Bare assignment ``=`` distinguishing a card parameter list from call arguments."""

_CALL_PREFIX: Final = re.compile(r"=\s*[A-Za-z_][A-Za-z0-9_]*$")
"""Function name immediately after ``=``, as in ``IS=scale()``."""

_CHANNEL: Final = re.compile(r"\b([np])chan\b", re.IGNORECASE)
"""VDMOS channel polarity flag (``nchan`` or ``pchan``) appearing anywhere in a card."""

_AKO: Final = re.compile(r"ako:\s*([^\s(]+)\s*(.*)", re.IGNORECASE | re.DOTALL)
"""Inheritance header ``AKO:`` matching base model name and remaining card body."""

_LEADING_WORD: Final = re.compile(r"\w*")
"""Leading type token on a model card before whitespace, ``=``, or ``(``."""

BSIM3_LEVELS: Final = frozenset({8, 49})
"""BSIM3 model levels: ngspice level 8 and HSPICE level 49."""

BSIM3_INSTANCE: Final[Table] = _table(
    {"delvto": "threshold_adjustment", "mulu0": "mobility_multiplier"},
    extra=MOSFET_INSTANCE,
)
"""BSIM3 instance parameters extending `MOSFET_INSTANCE` with threshold shift and
mobility multiplier."""

BSIM3_SELECTORS: Final = {
    "mobmod": "mobility_model",
    "capmod": "capacitance_model",
    "noimod": "noise_model",
}
"""Integer BSIM3 equation-selector SPICE names mapped to jolty parameter names."""

BSIM3_SETTINGS: Final = {"paramchk": "check_parameters", "xpart": "charge_partition"}
"""BSIM3 parameter-check and charge-partition SPICE names mapped to jolty names."""

BSIM3_OFF: Final = ("nqsmod", "acnqsmod", "acm")
"""Unsupported BSIM3 non-quasi-static and HSPICE area modes required to be zero."""

BSIM4_UNREAD: Final = frozenset(
    {
        *("lmin", "lmax", "wmin", "wmax"),
        *("vgs_max", "vgd_max", "vgb_max", "vds_max", "vbs_max", "vbd_max"),
        *("vgsr_max", "vgdr_max", "vgbr_max", "vbsr_max", "vbdr_max"),
    }
)
"""BSIM4 card parameters that do not affect simulation equations.

Includes bin dimension bounds used only for bin selection and safe-operating-area
limits read only by SOA checks.
"""

BSIM3_UNREAD: Final = BSIM4_UNREAD | frozenset(
    {
        *("elm", "lelm", "welm", "pelm", "calcacm", "geo"),
        *("hdif", "ldif", "ld", "rd", "rs", "rdc", "rsc", "wmlt"),
    }
)
"""BSIM3 card parameters unused when non-quasi-static and area modes are disabled.

Extends `BSIM4_UNREAD` with NQS Elmore constants and disabled area-calculation
parameters.
"""

BSIM3_DENSITIES: Final = {"nch": 1e20, "ngate": 1.000001e24}
"""BSIM3 doping parameters read per cubic centimeter up to a threshold and per cubic
meter above it (``b3mpar.c``)."""

BSIM4_INSTANCE: Final[Table] = _table(
    {
        "nf": "fingers",
        "sa": "diffusion_edge_a",
        "sb": "diffusion_edge_b",
        "sd": "finger_spacing",
        "delvto delvt0": "threshold_adjustment",
        "mulu0": "mobility_multiplier",
        **{name: name for name in ("rbdb", "rbsb", "rbpb", "rbps", "rbpd", "xgw")},
        "ngcon": "ngcon",
    },
    extra=MOSFET_INSTANCE,
)
"""BSIM4 instance parameters extending `MOSFET_INSTANCE` with finger counts, diffusion
stress distances, threshold shift, mobility multiplier, and body/gate overrides."""

BSIM4_SELECTORS: Final = {
    "rbodymod": "body_resistance_model",
    "rgatemod": "gate_resistance_model",
    "permod": "perimeter_model",
    "rdsmod": "drain_source_resistance_model",
    "fnoimod": "flicker_noise_model",
    "tnoimod": "thermal_noise_model",
}
"""Integer BSIM4 equation-selector SPICE names with multiple supported values mapped to
jolty parameter names."""

BSIM4_SETTINGS: Final = BSIM3_SETTINGS
"""BSIM4 parameter-check and charge-partition SPICE names mapped to jolty names."""

BSIM4_MODES: Final = {
    "mobmod": 0,
    "capmod": 2,
    "diomod": 1,
    "tempmod": 0,
    "geomod": 0,
    "rgeomod": 0,
    "igcmod": 0,
    "igbmod": 0,
    "trnqsmod": 0,
    "acnqsmod": 0,
    "wpemod": 0,
    "min": 0,
}
"""Fixed BSIM4 model and instance mode selectors required to match ngspice defaults.

Covers the modes selected by SKY130 and GF180MCU cards, including instance ``min`` for
even-finger drain/source count selection.
"""

BSIM4_INSTANCE_UNREAD: Final = frozenset({"sca", "scb", "scc", "sc"})
"""BSIM4 instance well-proximity parameters ignored when ``WPEMOD`` is zero."""

BSIM4_DENSITIES: Final = {"ndep": 1e20, "nsd": 1.000001e24, "ngate": 1.000001e24}
"""BSIM4 doping parameters read per cubic centimeter up to a threshold and per cubic
meter above it (``b4v5mpar.c``)."""

BSIM4_ALIASES: Final = {
    "vtho": "vth0",
    "lvtho": "lvth0",
    "wvtho": "wvth0",
    "pvtho": "pvth0",
}
"""Alternative SPICE parameter names accepted by ngspice on BSIM4 cards."""


VBIC: Final[Table] = _table(
    {
        "rcx": "extrinsic_collector_resistance",
        "rci": "intrinsic_collector_resistance",
        "vo v0": "epitaxial_saturation_voltage",
        "gamm": "epitaxial_doping_parameter",
        "hrcf": "high_current_collector_factor",
        "rbx": "extrinsic_base_resistance",
        "rbi": "intrinsic_base_resistance",
        "re": "emitter_resistance",
        "rs": "substrate_resistance",
        "rbp": "parasitic_base_resistance",
        "is": "saturation_current",
        "nf": "forward_emission_coefficient",
        "nr": "reverse_emission_coefficient",
        "fc": "depletion_coefficient",
        "cbeo cbe0": "base_emitter_overlap_capacitance",
        "cje": "base_emitter_capacitance",
        "pe": "base_emitter_potential",
        "me": "base_emitter_grading",
        "aje": "base_emitter_smoothing",
        "cbco cbc0": "base_collector_overlap_capacitance",
        "cjc": "base_collector_capacitance",
        "qco qc0": "epitaxial_charge",
        "cjep": "extrinsic_base_collector_capacitance",
        "pc": "base_collector_potential",
        "mc": "base_collector_grading",
        "ajc": "base_collector_smoothing",
        "cjcp": "collector_substrate_capacitance",
        "ps": "collector_substrate_potential",
        "ms": "collector_substrate_grading",
        "ajs": "collector_substrate_smoothing",
        "ccso": "collector_substrate_fixed_capacitance",
        "ibei": "ideal_base_emitter_current",
        "wbe": "intrinsic_base_emitter_fraction",
        "nei": "ideal_base_emitter_emission_coefficient",
        "iben": "nonideal_base_emitter_current",
        "nen": "nonideal_base_emitter_emission_coefficient",
        "ibci": "ideal_base_collector_current",
        "nci": "ideal_base_collector_emission_coefficient",
        "ibcn": "nonideal_base_collector_current",
        "ncn": "nonideal_base_collector_emission_coefficient",
        "avc1": "avalanche_coefficient",
        "avc2": "avalanche_exponent_coefficient",
        "isp": "parasitic_saturation_current",
        "wsp": "parasitic_extrinsic_fraction",
        "nfp": "parasitic_forward_emission_coefficient",
        "ibeip": "parasitic_ideal_base_emitter_current",
        "ibenp": "parasitic_nonideal_base_emitter_current",
        "ibcip": "parasitic_ideal_base_collector_current",
        "ncip": "parasitic_ideal_base_collector_emission_coefficient",
        "ibcnp": "parasitic_nonideal_base_collector_current",
        "ncnp": "parasitic_nonideal_base_collector_emission_coefficient",
        "vef": "forward_early_voltage",
        "ver": "reverse_early_voltage",
        "ikf": "forward_knee_current",
        "ikr": "reverse_knee_current",
        "ikp": "parasitic_knee_current",
        "nkf": "knee_rolloff",
        "tf": "forward_transit_time",
        "qtf": "transit_time_base_width_coefficient",
        "xtf": "transit_time_bias_coefficient",
        "vtf": "transit_time_voltage",
        "itf": "transit_time_current",
        "tr": "reverse_transit_time",
        "td": "excess_phase_delay",
        "kfn": "flicker_coefficient",
        "afn": "flicker_exponent",
        "bfn": "flicker_frequency_exponent",
        "vrt": "reach_through_voltage",
        "art": "reach_through_smoothing",
        "vbbe": "base_emitter_breakdown_voltage",
        "nbbe": "base_emitter_breakdown_emission_coefficient",
        "ibbe": "base_emitter_breakdown_current",
        "isrr": "reverse_saturation_current_ratio",
        "ea": "activation_energy",
        "eaie": "ideal_base_emitter_activation_energy",
        "eaic": "ideal_base_collector_activation_energy",
        "eais": "ideal_collector_substrate_activation_energy",
        "eane": "nonideal_base_emitter_activation_energy",
        "eanc": "nonideal_base_collector_activation_energy",
        "eans": "nonideal_collector_substrate_activation_energy",
        "eap": "parasitic_activation_energy",
        "dear": "reverse_activation_energy_shift",
        "xii": "ideal_base_current_temperature_exponent",
        "xin": "nonideal_base_current_temperature_exponent",
        "xisr": "reverse_saturation_current_temperature_exponent",
        "tnf": "emission_coefficient_temperature_coefficient",
        "tavc": "avalanche_temperature_coefficient",
        "tnbbe": "breakdown_emission_temperature_coefficient",
    },
    exponents=(
        *("is", "ikf", "rcx xrc", "rci", "rbx xrb", "rbi"),
        *("re", "rs", "rbp", "vo xv0"),
    ),
    temperature_pairs={"tvbbe": "breakdown_voltage"},
    extra=FLICKER,
)
"""ngspice VBIC bipolar transistor parameters (levels 4 and 9) with aliases.

Standard SPICE ``KF`` and ``AF`` map to flicker parameters alongside ``KFN``/``AFN``.
"""

_TABLES: Final[Mapping[str, Table]] = {
    "diode": DIODE,
    "jfet": JFET,
    "bipolar_transistor": BIPOLAR,
    "vbic": VBIC,
}
"""Parameter tables for device models whose parameters scale with area."""

VBIC_LEVELS: Final = frozenset({4.0, 9.0})
"""Bipolar model-card levels selecting VBIC in ngspice."""

VBIC_RESISTANCES: Final = ("rcx", "rci", "rbx", "rbi", "re", "rs", "rbp")
"""VBIC card resistances clamped to `VBIC_MIN_RESISTANCE` and divided by area."""

VBIC_MIN_RESISTANCE: Final = 0.01
"""Minimum VBIC resistance in ohms enforced by ngspice (``R_MIN``)."""

BIPOLAR_ABSENT_AT_ZERO: Final = frozenset(
    {"vaf", "va", "var", "vb", "ikf", "ik", "ikr", "irb", "vtf", "iss"}
)
"""Gummel-Poon parameters treated as infinite by ngspice when set to zero.

Includes Early voltages and knee currents (``bjttemp.c:83-99``), base-resistance
current (``bjtload.c:620``), transit-time voltage (``bjttemp.c:51-54``), and substrate
saturation current (``bjtload.c:487-499``).
"""

BIPOLAR_TEMPERATURE_MODELS: Final = {
    "tlev": "temperature_model",
    "tlevc": "capacitance_temperature_model",
}
"""Gummel-Poon temperature-model selectors (``TLEV`` and ``TLEVC``) supporting 0 or 1.

``TLEV=3`` does not commute with element area scaling, and ``TLEV=2`` computes no
saturation current in ngspice (``bjttemp.c:166-197``).
"""

BIPOLAR_POWER_COEFFICIENTS: Final = frozenset(
    {"tis1", "tis2", "tise1", "tise2", "tisc1", "tisc2", "tiss1", "tiss2"}
)
"""Saturation-current coefficients used only by ``TLEV=3`` and ignored for modes 0 and
1 (``bjttemp.c:183-197``, ``250-253``)."""

BIPOLAR_SATURATION_CURRENTS: Final = ("ibe", "ibc")
"""Split base-emitter and base-collector saturation currents (``IBE``, ``IBC``), read
by ngspice only when both are provided (``bjttemp.c:169-180``)."""

BIPOLAR_UNDEFINED: Final = frozenset({"dcap", "gap1", "gap2", "cbcp", "cbep", "ccsp"})
"""HSPICE Gummel-Poon parameters undefined in ngspice 47 and ignored with warnings."""

BIPOLAR_MAX_HIGH_INJECTION_EXPONENT: Final = 1.0
"""Maximum ``NKF`` accepted by ngspice before clamping with a warning
(``bjtsetup.c:332-338``)."""

VBIC_INFINITE_AT_ZERO: Final = frozenset(
    {"vo", "v0", "vef", "ver", "ikf", "ikr", "ikp", "vtf", "vbbe"}
)
"""VBIC parameters treated as infinite (or disabled for ``VBBE``) when non-positive."""

VBIC_MULTIPLIED: Final = (
    *("is", "ibei", "iben", "ibci", "ibcn", "isp", "ibeip", "ibenp", "ibcip", "ibcnp"),
    *("ikf", "ikr", "ikp", "itf", "ibbe", "cje", "cjc", "cjep", "cjcp", "ccso"),
    *("cbeo", "cbe0", "cbco", "cbc0", "qco", "qc0"),
)
"""VBIC currents, capacitances, and charges multiplied by area and copy count."""

VBIC_IGNORED: Final = frozenset(
    {
        *("vers", "vref", "ebbe"),
        *("vbe_max", "bvbe", "vbc_max", "bvbc", "vce_max", "bvce", "vsub_max", "bvsub"),
        *("vbefwd", "vbcfwd", "vsubfwd"),
    }
)
"""VBIC card parameters without simulation effect in ngspice.

Covers model version, ``EBBE`` (recomputed from ``VBBE`` and ``NBBE``), and
safe-operating-area warning limits.
"""

VBIC_SELF_HEATING: Final = frozenset({"rth", "cth", "selft"})
"""VBIC self-heating parameters (``RTH``, ``CTH``, ``SELFT``) active only when an
instance connects a fifth thermal node."""

VBIC_INSTANCE: Final = frozenset({"area", "m", "ic", "icvbe", "icvce"})
"""Allowed VBIC instance parameters: area, parallel copies, and initial conditions."""

VBIC_MIN_THERMAL_CAPACITANCE: Final = 1e-12
"""Minimum VBIC thermal capacitance in joules per kelvin enforced by ngspice."""

CONSTRUCTIONS: Final = {1.0: "vertical", -1.0: "lateral"}
"""Bipolar substrate construction mapped from SPICE ``SUBS`` values."""

GATE_TYPES: Final = {1.0: "opposite_doping", -1.0: "same_doping", 0.0: "aluminum"}
"""MOSFET gate material mapped from SPICE ``TPG`` values."""

SWITCH: Final[Table] = _table(
    {
        "ron": "on_resistance",
        "roff": "off_resistance",
        "von": "on_voltage",
        "voff": "off_voltage",
    }
)
"""PSpice ``VSWITCH`` parameters for voltage-controlled switches."""

CURRENT_SWITCH: Final[Table] = _table(
    {
        "ron": "on_resistance",
        "roff": "off_resistance",
        "ion": "on_current",
        "ioff": "off_current",
    }
)
"""PSpice ``ISWITCH`` parameters for current-controlled switches."""

NOMINAL_TEMPERATURES: Final = frozenset({"tnom", "t_measured"})
"""Model-card measurement temperature keys in degrees Celsius (including PSpice
``T_MEASURED``)."""

NOMINAL_TEMPERATURES_WITH_TREF: Final = NOMINAL_TEMPERATURES | frozenset({"tref"})
"""Model-card measurement temperature keys including HSPICE ``TREF``."""

OWN_TEMPERATURES: Final = frozenset(
    {"t_abs", "t_rel_global", "t_rel_local", "temp", "dtemp"}
)
"""Card and instance parameters setting per-device temperature overrides (PSpice
``T_ABS``/``T_REL_*`` and ngspice ``TEMP``/``DTEMP``)."""


def nominal_temperature(
    card: Mapping[str, float],
    statement: Statement,
    *,
    allow_tref: bool = False,
) -> dict[str, JsonValue]:
    """Extracts a single nominal measurement temperature in kelvins from `card`.

    Accepts ``TNOM`` and PSpice ``T_MEASURED`` (plus HSPICE ``TREF`` when
    `allow_tref` is true), converting from degrees Celsius to kelvins.

    Args:
        card: Evaluated model-card parameter values keyed by case-folded SPICE name.
        statement: Statement used for error attribution.
        allow_tref: Whether HSPICE ``TREF`` is accepted as an alias for ``TNOM``.

    Returns:
        Single-entry dictionary setting ``"nominal_temperature"`` in kelvins, or an
        empty dictionary.

    Raises:
        SpiceError: When `card` specifies more than one nominal-temperature key.
    """
    keys = NOMINAL_TEMPERATURES_WITH_TREF if allow_tref else NOMINAL_TEMPERATURES
    given = {name: value for name, value in card.items() if name in keys}
    if len(given) > 1:
        names = ", ".join(f"`{name.upper()}`" for name in given)
        message = f"the model sets its nominal temperature twice: {names}"
        raise statement.fail(message)
    if given:
        (value,) = given.values()
        return {"nominal_temperature": celsius(value).value}
    return {}


def own_temperature(
    card: Mapping[str, float], instance: Mapping[str, float], statement: Statement
) -> dict[str, JsonValue]:
    """Extracts a per-device absolute temperature or offset from `card` and `instance`.

    Card ``T_ABS`` and instance ``TEMP`` (in degrees Celsius) pin the device
    temperature. Card ``T_REL_GLOBAL`` and instance ``DTEMP`` specify an offset from
    circuit temperature.

    Args:
        card: Evaluated model-card parameter values keyed by case-folded SPICE name.
        instance: Evaluated element-level parameter values keyed by case-folded name.
        statement: Element statement used for error attribution.

    Returns:
        Single-entry dictionary setting ``"temperature"`` (in kelvins) or
        ``"temperature_offset"``, or an empty dictionary.

    Raises:
        SpiceError: When temperature is specified more than once or uses unsupported
            ``T_REL_LOCAL`` inheritance offsets.
    """
    given = {
        name: value
        for name, value in (*card.items(), *instance.items())
        if name in OWN_TEMPERATURES
    }
    if "t_rel_local" in given:
        message = "jolty reads no `T_REL_LOCAL`, relative to an inherited card"
        raise statement.fail(message)
    if len(given) > 1:
        names = ", ".join(f"`{name.upper()}`" for name in given)
        message = f"the device sets its own temperature twice: {names}"
        raise statement.fail(message)
    match given:
        case {"t_abs": value} | {"temp": value}:
            return {"temperature": celsius(value).value}
        case {"t_rel_global": value} | {"dtemp": value}:
            return {"temperature_offset": value}
    return {}


def _nominal(_name: str) -> Draw:
    """Returns the default nominal-value callback for statistical parameters."""
    return nominal


RESISTOR_CARD: Final = {
    **{name: name for name in ("rsh", "narrow", "short", "tc1", "tc2", "tce", "kf")},
    **{name: name for name in ("af", "lf", "wf", "ef", "tnom", "r", "noise", "defw")},
    "bv_max": "bv_max",
    "dw": "narrow",
    "dlr": "short",
    "tc1r": "tc1",
    "tc2r": "tc2",
    "tref": "tnom",
    "res": "r",
    "w": "defw",
    "l": "defl",
    "noisy": "noise",
}
"""SPICE3 ``R`` model-card parameter aliases mapped to canonical names (``res.c``;
``TREF`` aliases ``TNOM`` per HSPICE and ngspice compatibility)."""

CAPACITOR_CARD: Final = {
    **{name: name for name in ("cap", "cj", "cjsw", "narrow", "short", "del", "tc1")},
    **{name: name for name in ("tc2", "tnom", "di", "thick", "bv_max", "defw")},
    "defl": "defl",
    "cox": "cj",
    "capsw": "cjsw",
    "w": "defw",
    "l": "defl",
}
"""SPICE3 ``C`` model-card parameter aliases mapped to canonical names (``cap.c``)."""

GEOMETRIC: Final = {"r": RESISTOR_CARD, "c": CAPACITOR_CARD}
"""Mapping from SPICE3 geometric passive card kinds (`r`, `c`) to parameter tables."""

DEFAULT_WIDTH: Final = 10e-6
"""Default resistor or capacitor width in meters when omitted
(``ressetup.c``, line 31; ``capsetup.c``, line 38)."""

DEFAULT_RESISTOR_LENGTH: Final = 10e-6
"""Default resistor length in meters when omitted (``ressetup.c``, line 32).

Capacitor length defaults to 0 and ignores ``DEFL`` (``capsetup.c``, line 96).
"""

MIN_RESISTANCE: Final = 1e-3
"""Fallback resistance in ohms when a resistor card omits both ``R`` and sheet
resistance (``restemp.c``, line 79)."""

VACUUM_PERMITTIVITY: Final = 1 / (4e-7 * math.pi * 299792458.0**2)
"""Vacuum permittivity in farads per meter used by ngspice capacitors:
``1 / (μ₀ c²)`` with μ₀ = 4π·10⁻⁷ H/m (``const.h``, lines 44 to 47)."""

SILICA_RELATIVE_PERMITTIVITY: Final = 3.9
"""Relative permittivity of silicon dioxide assumed as default capacitor dielectric
when ``DI`` is omitted (``const.h``, line 51)."""


def geometric_values(
    card: ModelCard,
    scope: Mapping[str, float],
    draws: Callable[[str], Draw] = _nominal,
    statement: Statement | None = None,
    *,
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> dict[str, float]:
    """Evaluates SPICE3 resistor or capacitor `card` parameters under `GEOMETRIC` keys.

    Evaluates each card expression in `scope` with `functions` inlined and statistical
    calls sampled by `draws`. Ignores ``BV_MAX``, which affects only SOA checks.

    Args:
        card: SPICE3 geometric resistor (`r`) or capacitor (`c`) model card.
        scope: Case-folded parameter values in scope for evaluating card expressions.
        draws: Factory mapping each card parameter name to a statistical draw callback.
        statement: Optional element statement overriding `card.statement` for errors.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Evaluated numeric card parameters keyed by canonical `GEOMETRIC` names.

    Raises:
        SpiceError: When `card` specifies an unsupported parameter or sets its nominal
            temperature twice, attributed to `statement` or `card.statement`.
    """
    table = GEOMETRIC[card.kind]
    target = statement or card.statement
    raw: dict[str, float] = {}
    values: dict[str, float] = {}
    for name, expression in card.parameters.items():
        if name not in table:
            raise unmatched(card, name, target)
        if table[name] != "bv_max":
            resolved = inlined(expression, functions)
            evaluated = evaluate(resolved, scope, draws(name))
            raw[name] = evaluated
            values[table[name]] = evaluated
    nominal_temperature(raw, target, allow_tref=card.kind == "r")
    return values


@dataclass(frozen=True, slots=True)
class Geometric:
    """Resolved resistance or capacitance value and drift parameters from a SPICE3 card.

    Stores the pre-multiplicity `value` in ohms or farads alongside SPICE `drift` keys
    (``tc1``, ``tc2``, ``tce``, ``tnom``, ``kf``, and ``af``).
    """

    value: float
    drift: dict[str, float]


def geometric(
    card: ModelCard,
    values: Mapping[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> Geometric:
    """Computes the resistance or capacitance and drift of a SPICE3 geometric element.

    Uses card parameters in `values` (keyed by `GEOMETRIC`) and instance attributes in
    `instance` (`r`/`c`, `w`, `l`, and `scale`).

    Resistors use explicit `r`, else ``(L - 2·SHORT) / (W - 2·NARROW) · RSH`` when
    positive, else card ``R``, else 1 mΩ (``restemp.c``, lines 67 to 81). When the
    instance sets `w` or `l`, flicker coefficient ``KF`` divides by
    ``(L - 2·SHORT)^LF · (W - 2·NARROW)^WF`` (``ressetup.c``, lines 60 to 64).

    Capacitors use explicit `c`, else card ``CAP``, else
    ``CJ · (W - NARROW) · (L - SHORT) + 2 · CJSW · (L - SHORT + W - NARROW)``
    (``capsetup.c``, lines 71 to 88; ``captemp.c``, lines 60 to 72). Omitted ``CJ`` is
    derived from ``THICK`` and ``DI``, and ``2 · DEL`` supplies omitted ``NARROW`` or
    ``SHORT``.

    Args:
        card: SPICE3 geometric resistor or capacitor model card.
        values: Evaluated canonical card parameters returned by `geometric_values`.
        instance: Evaluated element-level geometric attributes (`r`/`c`, `w`, `l`,
            `scale`).
        statement: Element statement used for error attribution.

    Returns:
        Resolved `Geometric` value (scaled by `scale`) and drift parameter dictionary.

    Raises:
        SpiceError: When a resistor card disables thermal noise or sets flicker
            frequency exponent ``EF`` to a value other than 1.
    """
    resistor = card.kind == "r"
    letter = "r" if resistor else "c"
    width = instance.get("w", values.get("defw", DEFAULT_WIDTH))
    sized = "w" in instance or "l" in instance
    drift = {
        name: value
        for name, value in values.items()
        if name in {"tc1", "tc2", "tce", "tnom", "kf", "af"}
    }
    if resistor:
        if values.get("noise", 1.0) != 1:
            message = f"`{card.name}` makes no noise, which jolty's resistor makes"
            raise statement.fail(message)
        if values.get("kf", 0.0) != 0 and values.get("ef", 1.0) != 1:
            raise unmatched(card, "ef", statement)
        length = instance.get("l", values.get("defl", DEFAULT_RESISTOR_LENGTH))
        long = length - 2 * values.get("short", 0.0)
        wide = width - 2 * values.get("narrow", 0.0)
        sheet = values.get("rsh", 0.0)
        if letter in instance:
            value = instance[letter]
        elif length * width * sheet > 0:
            if wide == 0:
                message = f"effective width of `{card.name}` is 0"
                raise statement.fail(message)
            value = long / wide * sheet
        else:
            value = values.get("r", MIN_RESISTANCE)
        if "kf" in drift and sized:
            drift["kf"] /= _noise_area(card, values, long, wide, statement)
    else:
        length = instance.get("l", 0.0)
        cut = values.get("del")
        long = length - values.get("short", 0.0 if cut is None else 2 * cut)
        wide = width - values.get("narrow", 0.0 if cut is None else 2 * cut)
        thick = values.get("thick", 0.0)
        relative = values.get("di", SILICA_RELATIVE_PERMITTIVITY)
        permittivity = relative * VACUUM_PERMITTIVITY
        area = values.get("cj", permittivity / thick if thick > 0 else 0.0)
        if letter in instance:
            value = instance[letter]
        elif "cap" in values:
            value = values["cap"]
        else:
            value = area * wide * long + values.get("cjsw", 0.0) * 2 * (long + wide)
    return Geometric(value * instance.get("scale", 1.0), drift)


def _noise_area(
    card: ModelCard,
    values: Mapping[str, float],
    long: float,
    wide: float,
    statement: Statement,
) -> float:
    """Computes the effective flicker-noise area of a geometric resistor card.

    Raises:
        SpiceError: When effective dimensions or exponents yield a non-positive,
            overflowing, or non-finite area.
    """
    if not (long > 0 and wide > 0):
        message = (
            f"effective length and width of `{card.name}` are positive for flicker "
            f"noise, not {long!r} and {wide!r}"
        )
        raise statement.fail(message)
    length_exponent = values.get("lf", 1.0)
    width_exponent = values.get("wf", 1.0)
    try:
        area = math.pow(long, length_exponent) * math.pow(wide, width_exponent)
    except OverflowError, ValueError:
        message = f"flicker noise area of `{card.name}` is outside finite range"
        raise statement.fail(message) from None
    if not (area > 0 and math.isfinite(area)):
        message = (
            f"flicker noise area of `{card.name}` is positive and finite, not {area!r}"
        )
        raise statement.fail(message)
    return area


NOISE: Final = frozenset({"nlev", "gdsnoi"})
"""PSpice MOSFET noise-equation selectors ignored during import.

jolty retains classic SPICE flicker noise and exports ``nlev=1`` for ngspice 43+.
"""


@dataclass(frozen=True, slots=True)
class ModelCard:
    """Parsed SPICE `.model` card with name, type, and parameter expressions.

    Parameters are stored under case-folded SPICE names with inherited ``AKO:``
    parameters already merged.
    """

    name: str
    kind: str
    parameters: Mapping[str, Expression]
    statement: Statement = field(repr=False)
    version: str | None = None
    """Optional model version string (such as a BSIM card version)."""


def is_derived(statement: Statement) -> bool:
    """Returns whether a ``.model`` statement inherits from another via ``AKO:``.

    Args:
        statement: Candidate ``.model`` statement to inspect.

    Returns:
        `True` when `statement` inherits from a base model via ``AKO:``.
    """
    words = statement.text.split(maxsplit=2)
    return (
        len(words) >= 3  # noqa: PLR2004 - `.model`, name, type
        and words[2].casefold().startswith("ako:")
    )


def read(statement: Statement, known: Mapping[str, ModelCard]) -> ModelCard:
    """Parses a `.model` `statement` into a `ModelCard`, resolving ``AKO:`` in `known`.

    Args:
        statement: Logical ``.model`` statement to parse.
        known: Previously parsed model cards keyed by case-folded model name.

    Returns:
        Parsed `ModelCard` with ``AKO:`` inheritance and polarity flags resolved.

    Raises:
        SpiceError: When `statement` is malformed or references an unknown base model.
    """
    words = statement.text.split(maxsplit=2)
    if len(words) < 3:  # noqa: PLR2004 - `.model`, the name and the type
        message = "a model card names its model and its type"
        raise statement.fail(message)
    _, name, rest = words
    base: ModelCard | None = None
    if rest.casefold().startswith("ako:"):
        ako_match = _AKO.fullmatch(rest)
        if ako_match is None:
            message = "an `AKO:` model card names the base model it inherits from"
            raise statement.fail(message)
        base_name, rest = ako_match.group(1), ako_match.group(2)
        base = known.get(base_name.casefold())
        if base is None:
            message = f"no model `{base_name}` precedes it"
            raise statement.fail(message)
    word = _LEADING_WORD.match(rest)
    candidate = "" if word is None else word.group(0)
    after = rest[len(candidate) :].strip()
    is_vdmos_flag = (
        base is not None
        and base.kind in VDMOS_KINDS
        and candidate.casefold() in {"nchan", "pchan"}
    )
    if base is not None and candidate and (after.startswith("=") or is_vdmos_flag):
        kind, body = "", rest.strip()
    else:
        kind, body = candidate, after
    resolved = kind.casefold() or (base.kind if base else "")
    if resolved.startswith("vdmos"):
        resolved, body = _vdmos_card(resolved, body)
    body = _unwrap_body(body)
    words = tokens(body)
    if resolved in VDMOS_KINDS:
        words = _annotations_removed(words)
    version, words = _version_taken(words)
    if version is None and base is not None:
        version = base.version
    parameters = dict(base.parameters) if base else {}
    parameters |= assignments(statement, words)
    return ModelCard(name, resolved, MappingProxyType(parameters), statement, version)


def _unwrap_body(body: str) -> str:
    """Strips an enclosing parameter-list parenthesis group from a `.model` body."""
    if not body.endswith(")"):
        return body
    depth = 0
    opening: int | None = None
    for position in range(len(body) - 1, -1, -1):
        character = body[position]
        if character == ")":
            depth += 1
        elif character == "(":
            depth -= 1
            if depth == 0:
                opening = position
                break
    if opening is None:
        return body
    if opening == 0:
        return body[1:-1]
    prefix, inside = body[:opening], body[opening + 1 : -1]
    is_list = (
        not inside.strip() and _CALL_PREFIX.search(prefix) is None
    ) or _BARE_EQUALS.search(inside) is not None
    if _WRAPPED_PREFIX.fullmatch(prefix) is not None and is_list:
        return f"{prefix} {inside}"
    return body


def _version_taken(words: list[str]) -> tuple[str | None, list[str]]:
    """Extracts ``VERSION=...`` from card `words` and returns the remaining tokens."""
    for position in range(len(words) - 2):
        if words[position].casefold() == "version" and words[position + 1] == "=":
            raw = words[position + 2]
            if (
                len(raw) >= 2  # noqa: PLR2004 - matching opening and closing quotes
                and raw[0] == raw[-1]
                and raw[0] in {"'", '"'}
            ):
                raw = raw[1:-1].strip()
            return raw, words[:position] + words[position + 3 :]
    return None, words


def _vdmos_card(kind: str, body: str) -> tuple[str, str]:
    """Determines VDMOS channel polarity and removes polarity flags from `body`.

    Follows ngspice rules: ``VDMOS`` defaults to n-channel, while ``pchan`` anywhere
    in `body` or type ``VDMOSP`` selects p-channel.
    """
    flags = {flag.casefold() for flag in _CHANNEL.findall(body)}
    polarity = "p" if "p" in flags or (kind == "vdmosp" and "n" not in flags) else "n"
    return f"vdmos{polarity}", _CHANNEL.sub(" ", body).strip()


def _annotations_removed(words: list[str]) -> list[str]:
    """Removes `VDMOS_ANNOTATIONS` key-value assignments from VDMOS card `words`."""
    kept: list[str] = []
    position = 0
    while position < len(words):
        assigned = words[position + 1 : position + 2] == ["="]
        if assigned and words[position].casefold() in VDMOS_ANNOTATIONS:
            position += 3
            continue
        kept.append(words[position])
        position += 1
    return kept


def assignments(statement: Statement, words: list[str]) -> dict[str, Expression]:
    """Parses `name = value` token triples from `words` into case-folded expressions.

    Each value may be a numeric literal, identifier, or delimited expression.

    Args:
        statement: Logical SPICE statement used for error attribution.
        words: Tokenized ``name = value`` triples to parse.

    Returns:
        Mapping from case-folded parameter names to parsed `Expression` trees.

    Raises:
        SpiceError: When `words` do not form valid `name = value` assignments.
    """
    found: dict[str, Expression] = {}
    for position in range(0, len(words), 3):
        name, *assigned = words[position : position + 3]
        if len(assigned) != 2 or assigned[0] != "=":  # noqa: PLR2004 - `=` and a value
            message = f"`{name}` is not assigned a value"
            raise statement.fail(message)
        found[name.casefold()] = value(assigned[1])
    return found


def value(word: str) -> Expression:
    """Parses a single parameter value token into a numeric or general `Expression`.

    Args:
        word: Single parameter value token (literal, identifier, or expression).

    Returns:
        `Number` node for finite numeric literals, or parsed `Expression` otherwise.
    """
    written = number(word)
    return Number(written) if written is not None else parse(word)


@dataclass(frozen=True, slots=True)
class Made:
    """Translated jolty device model name and SI parameter dictionary."""

    model: str
    parameters: dict[str, JsonValue]


def device(  # noqa: PLR0913 - device translation inputs
    card: ModelCard,
    scope: Mapping[str, float],
    instance: Mapping[str, float],
    statement: Statement,
    draws: Callable[[str], Draw] = _nominal,
    *,
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> Made:
    """Translates `card` and `instance` parameters into a jolty `Made` device.

    Evaluates card expressions in `scope` with user `functions` inlined and statistical
    calls sampled via `draws` (defaulting to nominal values).

    Args:
        card: Parsed SPICE ``.model`` card describing the device.
        scope: Case-folded parameter values in scope for evaluating card expressions.
        instance: Evaluated element-level geometric, area, multiplicity, and temperature
            overrides.
        statement: Element statement used for error attribution.
        draws: Factory mapping each card parameter name to a statistical draw callback.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Translated jolty `Made` device pairing model name and SI parameter dictionary.

    Raises:
        SpiceError: When the model type or any parameter is unsupported by jolty.
    """
    every = {
        name: evaluate(inlined(expression, functions), scope, draws(name))
        for name, expression in card.parameters.items()
    }
    own = own_temperature(every, instance, statement)
    values = {
        name: value for name, value in every.items() if name not in OWN_TEMPERATURES
    }
    instance = {
        name: value for name, value in instance.items() if name not in OWN_TEMPERATURES
    }
    made = _device(card, values, instance, statement)
    return Made(made.model, {**made.parameters, **own})


def _bipolar_device(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> Made:
    """Builds a Gummel-Poon (`level=1`) or VBIC (`level=4`/`9`) bipolar device.

    Raises:
        SpiceError: When the card specifies an unsupported level or parameter.
    """
    if values.get("level", 1.0) in VBIC_LEVELS:
        return Made("vbic", _vbic(card, values, instance, statement))
    level = values.pop("level", 1.0)
    if level != 1:
        message = (
            f"`{card.name}` is a bipolar transistor of level {level:g}; jolty "
            "models levels 1, 4 and 9"
        )
        raise statement.fail(message)
    return Made("bipolar_transistor", _bipolar(card, values, instance, statement))


def _mosfet_device(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> Made:
    """Dispatches a MOSFET card to BSIM3, BSIM4, or level-1/3 MOSFET translation.

    Raises:
        SpiceError: When the card specifies an unsupported level, version, or parameter.
    """
    level = values.get("level")
    if level in BSIM3_LEVELS:
        return Made("bsim3", _bsim3(card, values, instance, statement))
    if level in BSIM4_LEVELS:
        return Made("bsim4", _bsim4(card, values, instance, statement))
    return Made("mosfet", _mosfet(card, values, instance, statement))


def _device(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> Made:
    """Dispatches evaluated `card` `values` and `instance` parameters by device kind.

    Raises:
        SpiceError: When the model type or any parameter is unsupported by jolty.
    """
    match card.kind:
        case "d":
            made = Made("diode", _diode(card, values, instance, statement))
        case "npn" | "pnp":
            made = _bipolar_device(card, values, instance, statement)
        case "njf" | "pjf":
            area = _area(instance, statement)
            values = _area_scaled(
                "jfet", values, area, ("beta", "is", "cgs", "cgd"), ("rd", "rs")
            )
            parameters = _mapped(card, values, JFET, statement)
            made = Made("jfet", {"channel": card.kind[0], **parameters})
        case "nmos" | "pmos":
            made = _mosfet_device(card, values, instance, statement)
        case "vdmosn" | "vdmosp":
            made = Made("vdmos", _vdmos(card, values, instance, statement))
        case "vswitch":
            made = Made(
                "voltage_controlled_switch", _mapped(card, values, SWITCH, statement)
            )
        case "iswitch":
            made = Made(
                "current_controlled_switch",
                _mapped(card, values, CURRENT_SWITCH, statement),
            )
        case _:
            message = (
                f"jolty has no device for `{card.name}`, a model of type `{card.kind}`"
            )
            raise statement.fail(message)
    return made


def _diode(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a level-1 or level-3 diode card and instance into jolty parameters.

    Follows ngspice sizing rules (``diosetup.c``, lines 313 to 366). Area and perimeter
    default from the card when neither `w` nor `l` is set on `instance`. For level-3
    cards with both `w` and `l`, area is ``(W + XW) · (L + XW) · scale^2`` and
    perimeter is ``(2·(W + XW) + 2·(L + XW)) · scale``.

    Bottom-junction currents and capacitances scale by ``area · m`` while ``RS``
    divides by it. Sidewall parameters scale by ``perimeter · m`` while ``RSW`` divides
    by it. Breakdown current scales by `m`, and additionally by `area` at level 3.
    Unlike ngspice (which applies `m` twice on geometric level-3 diodes), jolty scales
    by `m` once.

    Raises:
        SpiceError: When the level is unsupported, a parameter has no jolty
            counterpart, effective area is non-positive, or metal/polysilicon bulk
            dimensions are set.
    """
    level = values.pop("level", 1.0)
    if level not in {1.0, 3.0}:
        message = f"`{card.name}` is a diode of level {level:g}; jolty models 1 and 3"
        raise statement.fail(message)
    geometric = level == 3  # noqa: PLR2004 - geometric diode level
    geometry = {
        name: values.pop(name, default) for name, default in DIODE_GEOMETRY.items()
    }
    values = {
        name: value for name, value in values.items() if name not in DIODE_IGNORED
    }
    selectors = {
        DIODE[name][0]: _whole(card, name, values.pop(name), statement)
        for name in DIODE_SELECTORS & values.keys()
    }
    # A knee current below `MIN_CURRENT` turns its effect off; a saturation current
    # below it is raised to it.
    values = {
        name: value
        for name, value in values.items()
        if name not in DIODE_KNEES or value >= MIN_CURRENT
    }
    for name in {"is", "js"} & values.keys():
        values[name] = max(values[name], MIN_CURRENT)
    copies = instance.get("m", 1.0)
    sized = "w" in instance or "l" in instance
    area = instance.get("area", 1.0 if sized else geometry["area"])
    perimeter = instance.get("pj", 0.0 if sized else geometry["pj"])
    if geometric:
        scale = instance.get("scale", 1.0)
        if "w" in instance and "l" in instance:
            width, length = (
                instance["w"] + geometry["xw"],
                instance["l"] + geometry["xw"],
            )
            try:
                area = width * length * (scale**2)
            except OverflowError:
                message = f"scale factor of `{card.name}` overflows: {scale!r}"
                raise statement.fail(message) from None
            perimeter = (2 * width + 2 * length) * scale
        sizes = {name: instance.get(name, geometry[name]) for name in METAL_AND_POLY}
        if any(sizes.values()):
            message = (
                f"jolty reads no metal or polysilicon over a diode's bulk, which "
                f"`{card.name}` sizes"
            )
            raise statement.fail(message)
    bottom = _area(dict(instance) | {"area": area}, statement)
    values = _area_scaled("diode", values, bottom, DIODE_BY_AREA, ("rs",))
    sidewall = perimeter * copies
    if sidewall > 0:
        values = _area_scaled("diode", values, sidewall, DIODE_BY_PERIMETER, ("rsw",))
    else:
        # Without a perimeter the sidewall has no current nor charge, and ngspice's
        # resistance to it no conductance (`diotemp.c`, line 262).
        values = {
            name: value
            for name, value in values.items()
            if name not in {*DIODE_BY_PERIMETER, "rsw"}
        }
    breakdown = copies * (area if geometric else 1.0)
    values = _area_scaled("diode", values, breakdown, ("ibv", "ib"), ())
    parameters: dict[str, JsonValue] = {
        **_mapped(card, values, DIODE, statement, allow_tref=True),
        **selectors,
    }
    return parameters


def unmatched(card: ModelCard, parameter: str, statement: Statement) -> SpiceError:
    """Builds a `SpiceError` reporting that `parameter` on `card` has no counterpart.

    Args:
        card: Model card containing the unsupported parameter.
        parameter: Unsupported parameter name.
        statement: Statement to which the error location is attributed.

    Returns:
        `SpiceError` reporting that `parameter` on `card` has no jolty counterpart.
    """
    message = (
        f"jolty's devices have no counterpart to `{parameter.upper()}` of `{card.name}`"
    )
    return statement.fail(message)


def _area(instance: Mapping[str, float], statement: Statement) -> float:
    """Computes effective `area * m` scaling factor for `instance`.

    Raises:
        SpiceError: When multiplicity `m` or effective area is not positive and finite.
    """
    copies = instance.get("m", 1.0)
    if not (copies > 0 and math.isfinite(copies)):
        message = f"the element's copies `m` is positive and finite, not {copies!r}"
        raise statement.fail(message)
    area = instance.get("area", 1.0) * copies
    if not (area > 0 and math.isfinite(area)):
        message = f"the element's area times its copies is positive, not {area!r}"
        raise statement.fail(message)
    return area


def _scaled(
    values: dict[str, float],
    area: float,
    multiplied: tuple[str, ...],
    divided: tuple[str, ...],
) -> dict[str, float]:
    """Scales `values` by `area`, multiplying `multiplied` and dividing `divided`."""
    return {
        name: value * area
        if name in multiplied
        else value / area
        if name in divided
        else value
        for name, value in values.items()
    }


@functools.cache
def _defaults(model: str) -> Mapping[str, object]:
    """Returns a cached read-only mapping of default parameters for `model`."""
    return MappingProxyType(json.loads(_core.defaulted(model, "{}")))


def _area_scaled(
    model: str,
    values: Mapping[str, float],
    area: float,
    multiplied: tuple[str, ...],
    divided: tuple[str, ...],
) -> dict[str, float]:
    """Scales `values` and nonzero engine defaults of `model` by effective `area`.

    Multiplies keys in `multiplied` and divides keys in `divided`. Omitted parameters
    with nonzero engine defaults (such as the 1e-16 A BJT saturation current in
    ``bjtsetup.c:48``) are populated before scaling to match ngspice.
    """
    if area == 1:
        return dict(values)
    table = _TABLES[model]
    defaults = _defaults(model)
    filled = dict(values)
    given = {table[name][0] for name in values if name in table}
    for name in (*multiplied, *divided):
        if name not in table:
            continue
        parameter, factor = table[name]
        default = defaults.get(parameter)
        if parameter in given or not isinstance(default, float) or default == 0:
            continue
        filled[name] = default / factor
        given.add(parameter)
    return _scaled(filled, area, multiplied, divided)


def scale_flicker_coefficient(
    coefficient: float,
    exponent: float,
    copies: float,
    statement: Statement,
    label: str,
) -> float:
    """Scales a flicker noise coefficient across `copies` parallel elements.

    Args:
        coefficient: Base flicker noise coefficient (`KF`).
        exponent: Current exponent (`AF`) governing parallel multiplicity scaling.
        copies: Effective parallel multiplicity factor (`m`).
        statement: Element statement used for error attribution.
        label: Descriptive context included in overflow error messages.

    Returns:
        Flicker noise coefficient scaled by `copies ** (1 - exponent)`.

    Raises:
        SpiceError: When scaling overflows or produces a non-finite value.
    """
    try:
        scaled = coefficient * math.pow(copies, 1 - exponent)
    except OverflowError, ValueError:
        message = f"flicker noise scaling {label} overflows"
        raise statement.fail(message) from None
    if not math.isfinite(scaled):
        message = f"flicker noise scaling {label} overflows"
        raise statement.fail(message)
    return scaled


def _mapped(  # noqa: PLR0913 - card, values, table, statement, skipped and TREF flag
    card: ModelCard,
    values: Mapping[str, float],
    table: Table,
    statement: Statement,
    skipped: frozenset[str] = frozenset(),
    *,
    allow_tref: bool = False,
) -> dict[str, JsonValue]:
    """Maps SPICE parameter `values` into jolty SI parameters via `table`.

    Raises:
        SpiceError: When an unskipped parameter has no jolty counterpart or the card
            sets its nominal temperature twice.
    """
    nominal_keys = (
        NOMINAL_TEMPERATURES_WITH_TREF if allow_tref else NOMINAL_TEMPERATURES
    )
    parameters = nominal_temperature(
        {name: value for name, value in values.items() if name not in skipped},
        statement,
        allow_tref=allow_tref,
    )
    for name, value in values.items():
        if name in skipped or name in nominal_keys:
            continue
        if name not in table:
            raise unmatched(card, name, statement)
        jolty, factor = table[name]
        parameters[jolty] = value * factor
    return parameters


def _bipolar(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a level-1 Gummel-Poon BJT card and instance into jolty parameters.

    Defaults substrate construction to vertical for NPN and lateral for PNP unless
    ``SUBS`` specifies otherwise. Omits zero-valued infinite-default parameters,
    unsupported HSPICE parameters ignored by ngspice, ``TLEV=3`` coefficients, and a
    lone ``IBE`` or ``IBC``.

    Raises:
        SpiceError: When a selector or parameter is unsupported by jolty.
    """
    construction = values.pop("subs", 1.0 if card.kind == "npn" else -1.0)
    if construction not in CONSTRUCTIONS:
        message = f"`SUBS` is 1 or -1, not {construction!r}"
        raise statement.fail(message)
    models: dict[str, JsonValue] = {}
    for name, jolty in BIPOLAR_TEMPERATURE_MODELS.items():
        if name in values:
            level = _whole(card, name, values.pop(name), statement)
            if level not in {0, 1}:
                message = (
                    f"`{card.name}` scales with the temperature as `{name.upper()}` "
                    f"{level} says; jolty models 0 and 1"
                )
                raise statement.fail(message)
            models[jolty] = level
    given = [name for name in BIPOLAR_SATURATION_CURRENTS if name in values]
    if len(given) == len(BIPOLAR_SATURATION_CURRENTS):
        raise unmatched(card, given[0], statement)
    for name in ("nkf", "nk"):
        if name in values:
            values[name] = min(values[name], BIPOLAR_MAX_HIGH_INJECTION_EXPONENT)
    values = {
        name: value
        for name, value in values.items()
        if (name not in BIPOLAR_ABSENT_AT_ZERO or value != 0)
        and name not in {*BIPOLAR_UNDEFINED, *BIPOLAR_POWER_COEFFICIENTS, *given}
    }
    area = _area(instance, statement)
    multiplied = (
        *("is", "ikf", "ik", "ikr", "ise", "isc", "itf", "cje", "cjc", "irb"),
        *("iss", "cjs", "csub", "ccs"),
    )
    divided = ("rb", "rbm", "re", "rc")
    values = _area_scaled("bipolar_transistor", values, area, multiplied, divided)
    parameters = _mapped(card, values, BIPOLAR, statement, allow_tref=True)
    return {
        "polarity": card.kind,
        "substrate": CONSTRUCTIONS[construction],
        **models,
        **parameters,
    }


def _mosfet(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a level-1 or level-3 MOSFET card and instance into jolty parameters.

    Raises:
        SpiceError: When the level is not 1 or 3, ``TPG`` is invalid, or a parameter
            has no jolty counterpart.
    """
    level = values.pop("level", 1.0)
    if level not in {1, 3}:
        message = f"`{card.name}` is a MOSFET of level {level:g}; jolty models 1 and 3"
        raise statement.fail(message)
    if level == 3:  # noqa: PLR2004 - the level of the semi-empirical model
        # Level 3 has no channel-length modulation: PSpice and ngspice ignore it there.
        values.pop("lambda", None)
    gate = values.pop("tpg", None)
    # A model card's geometry is its instances' default.
    geometry = {name: values.pop(name) for name in ("l", "w") if name in values}
    parameters = _mapped(card, values, MOSFET, statement, NOISE)
    parameters |= _mapped(card, geometry | dict(instance), MOSFET_INSTANCE, statement)
    parameters["channel"] = card.kind[0]
    if level == 3:  # noqa: PLR2004 - the level of the semi-empirical model
        parameters["level"] = 3
    if gate is not None:
        if gate not in GATE_TYPES:
            message = f"`TPG` is 1, -1 or 0, not {gate!r}"
            raise statement.fail(message)
        parameters["gate_material"] = GATE_TYPES[gate]
    return parameters


def _vdmos(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a VDMOS model card and instance into jolty parameters.

    Omits ``PHI`` (used only by ngspice distortion analysis) and thermal-network
    parameters (unused without self-heating), while still rejecting non-positive
    ``PHI``.

    Raises:
        SpiceError: When ``PHI`` is non-positive or a parameter has no jolty
            counterpart.
    """
    potential = values.pop("phi", None)
    if potential is not None and potential <= 0:
        message = f"`PHI` of `{card.name}` is positive, not {potential!r}"
        raise statement.fail(message)
    parameters = _mapped(card, values, VDMOS, statement, VDMOS_SELF_HEATING)
    parameters |= _mapped(card, dict(instance), VDMOS_INSTANCE, statement)
    parameters["channel"] = card.kind[-1]
    return parameters


BIN_TOLERANCE: Final = 1e-9
"""Boundary tolerance in meters for MOSFET bin matching in ngspice ``INPgetModBin``."""


BSIM4_LEVELS: Final = frozenset({14, 54})
"""BSIM4 model-card levels recognized by ngspice (``inpdomod.c``, line 297)."""

BSIM_SOI_LEVELS: Final = frozenset({10, 58})
"""BSIM-SOI model-card levels recognized by ngspice (``inpdomod.c``, line 347)."""

HISIM2_LEVELS: Final = frozenset({68})
"""HiSIM2 model-card level recognized by ngspice (``inpdomod.c``, line 363)."""

HISIM_HV_LEVELS: Final = frozenset({73})
"""HiSIM_HV model-card level recognized by ngspice (``inpdomod.c``, line 371)."""

BINNED_LEVELS: Final = BSIM3_LEVELS | BSIM4_LEVELS | HISIM2_LEVELS | HISIM_HV_LEVELS
"""MOSFET levels supporting bin selection in ngspice ``INPgetModBin``."""


def holds(
    card: ModelCard,
    scope: Mapping[str, float],
    size: tuple[float, float],
    *,
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> bool:
    """Returns whether binned MOSFET `card` matches an instance of `(length, width)`.

    Matches ngspice bin selection for `BINNED_LEVELS` cards defining ``LMIN``,
    ``LMAX``, ``WMIN``, and ``WMAX`` evaluated in `scope` with `functions` inlined.
    Boundaries are inclusive within `BIN_TOLERANCE`. Cards of unmodeled binned levels
    can still match here and fail later during device translation.

    Args:
        card: Candidate binned MOSFET model card.
        scope: Case-folded parameter values in scope for evaluating bin bounds.
        size: Instance `(length, width)` pair in meters.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        `True` when `card` defines bin bounds containing `size` within `BIN_TOLERANCE`.
    """
    bounds = _bounds(card, scope, functions)
    if bounds is None:
        return False
    lmin, lmax, wmin, wmax = bounds
    length, width = size
    return _within(length, lmin, lmax) and _within(width, wmin, wmax)


def center(
    card: ModelCard,
    scope: Mapping[str, float],
    *,
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> tuple[float, float] | None:
    """Returns the midpoint `(length, width)` of binned MOSFET `card`, or `None`.

    Returns `None` when `card` is not a binned card as defined by
    [`holds`][jolty.spice.models.holds].

    Args:
        card: Candidate binned MOSFET model card.
        scope: Case-folded parameter values in scope for evaluating bin bounds.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Midpoint `(length, width)` in meters, or `None` when `card` is not binned.
    """
    bounds = _bounds(card, scope, functions)
    if bounds is None:
        return None
    lmin, lmax, wmin, wmax = bounds
    return (lmin + lmax) / 2, (wmin + wmax) / 2


def _bounds(
    card: ModelCard,
    scope: Mapping[str, float],
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> tuple[float, float, float, float] | None:
    """Evaluates ``(LMIN, LMAX, WMIN, WMAX)`` for a binned MOSFET `card` in `scope`.

    Returns `None` when `card` is not a binned level or omits any bound.
    """
    level = card.parameters.get("level")
    if level is None or evaluate(inlined(level, functions), scope) not in BINNED_LEVELS:
        return None
    bounds = [card.parameters.get(name) for name in ("lmin", "lmax", "wmin", "wmax")]
    if any(bound is None for bound in bounds):
        return None
    lmin, lmax, wmin, wmax = (
        evaluate(inlined(bound, functions), scope) for bound in bounds if bound
    )
    return lmin, lmax, wmin, wmax


def _within(value: float, low: float, high: float) -> bool:
    """Returns whether `value` lies within `[low, high]` up to `BIN_TOLERANCE`."""
    return (
        abs(value - low) < BIN_TOLERANCE
        or abs(value - high) < BIN_TOLERANCE
        or low < value < high
    )


@dataclass(frozen=True, slots=True)
class Bsim:
    """Configuration table describing SPICE-to-jolty translation for a BSIM family.

    Stores export level/version headers, instance parameter mappings, selector and
    setting mappings, valid card and binned parameter sets, dual-unit doping density
    thresholds, ignored parameters, and parameter aliases.
    """

    written: tuple[str, ...]
    instance: Table
    selectors: Mapping[str, str]
    settings: Mapping[str, str]
    card: Mapping[str, object]
    binned: frozenset[str]
    densities: Mapping[str, float]
    unread: frozenset[str]
    aliases: Mapping[str, str] = field(default_factory=dict)


BSIMS: Final = {
    "bsim3": Bsim(
        ("level=8", "version=3.3.0"),
        BSIM3_INSTANCE,
        BSIM3_SELECTORS,
        BSIM3_SETTINGS,
        BSIM3_CARD,
        BSIM3_BINNED,
        BSIM3_DENSITIES,
        BSIM3_UNREAD | NOISE,
    ),
    "bsim4": Bsim(
        ("level=54", "version=4.5"),
        BSIM4_INSTANCE,
        BSIM4_SELECTORS,
        BSIM4_SETTINGS,
        BSIM4_CARD,
        BSIM4_BINNED,
        BSIM4_DENSITIES,
        BSIM4_UNREAD,
        BSIM4_ALIASES,
    ),
}
"""BSIM family translation configurations keyed by jolty model name."""


def _check_bsim_modes(
    values: dict[str, float],
    instance: dict[str, float],
    modes: Mapping[str, int],
    family: str,
    statement: Statement,
) -> None:
    """Checks that fixed BSIM mode switches match `modes` on card and instance."""
    for name, modeled in modes.items():
        for given in (values, instance):
            if given.pop(name, modeled) != modeled:
                message = f"jolty models {family} with `{name.upper()}` {modeled} alone"
                raise statement.fail(message)


def _bsim3(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a BSIM3v3.3 model card and instance into jolty SI parameters.

    Converts doping densities (``nch``, ``ngate``, ``nsub``) to per-cubic-meter SI
    units using ngspice thresholds, scales ``u0`` above 1 from cm^2/(V·s) to m^2/(V·s),
    and converts binned ``L``/``W``/``P`` dependencies according to ``BINUNIT``.

    Raises:
        SpiceError: When the card version is not 3.3, an unsupported mode is enabled,
            or a parameter has no jolty counterpart.
    """
    values.pop("level")
    if card.version is not None and not card.version.startswith("3.3"):
        message = (
            f"`{card.name}` is a BSIM3 card of version {card.version}, whose equations "
            "ngspice runs another model for; jolty models version 3.3"
        )
        raise statement.fail(message)
    own = dict(instance)
    _check_bsim_modes(values, own, dict.fromkeys(BSIM3_OFF, 0), "BSIM3", statement)
    own.pop("geo", None)
    parameters: dict[str, JsonValue] = {"channel": card.kind[0]}
    parameters |= _bsim_card(BSIMS["bsim3"], card, values, statement)
    parameters |= _mapped(card, own, BSIM3_INSTANCE, statement)
    return parameters


def _bsim4(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a BSIM4v4.5 model card and instance into jolty SI parameters.

    Converts doping densities (``ndep``, ``nsd``, ``ngate``, ``nsub``) to
    per-cubic-meter SI units using ngspice thresholds, scales ``u0`` above 1 from
    cm^2/(V·s) to m^2/(V·s), and converts binned ``L``/``W``/``P`` dependencies via
    ``BINUNIT``.

    Raises:
        SpiceError: When the card version is not 4.5, a fixed mode selector mismatches
            `BSIM4_MODES`, or a parameter has no jolty counterpart.
    """
    values.pop("level")
    if card.version is None or not card.version.startswith("4.5"):
        message = (
            f"`{card.name}` is a BSIM4 card of version {card.version or '4.8'}, whose "
            "equations ngspice runs another model for; jolty models version 4.5"
        )
        raise statement.fail(message)
    own = dict(instance)
    _check_bsim_modes(values, own, BSIM4_MODES, "BSIM4", statement)
    for name in BSIM4_INSTANCE_UNREAD:
        own.pop(name, None)
    parameters: dict[str, JsonValue] = {"channel": card.kind[0]}
    parameters |= _bsim_card(BSIMS["bsim4"], card, values, statement)
    for name, jolty in BSIM4_SELECTORS.items():
        if name in own:
            parameters[jolty] = _whole(card, name, own.pop(name), statement)
    parameters |= _mapped(card, own, BSIM4_INSTANCE, statement)
    return parameters


_BSIM_BINUNITS: Final = frozenset({1.0, 2.0})
"""Accepted values for BSIM ``BINUNIT``: 1 (micrometers) or 2 (meters)."""

_BSIM_PARAMCHK: Final = frozenset({0.0, 1.0})
"""Accepted values for BSIM ``PARAMCHK``: 0 (off) or 1 (on)."""


def _bsim_setting(
    card: ModelCard, name: str, value: float, statement: Statement
) -> float | bool:
    """Converts a BSIM setting (`PARAMCHK` or `XPART`) into its jolty value."""
    if name != "paramchk":
        return value
    if not (math.isfinite(value) and value in _BSIM_PARAMCHK):
        message = f"`PARAMCHK` of `{card.name}` is 0 or 1, not {value!r}"
        raise statement.fail(message)
    return value == 1.0


def _bsim_card(
    bsim: Bsim, card: ModelCard, values: dict[str, float], statement: Statement
) -> dict[str, JsonValue]:
    """Converts BSIM card parameters into SI units, excluding fixed mode selectors.

    Raises:
        SpiceError: When a parameter has no jolty counterpart or ``BINUNIT`` or
            ``PARAMCHK`` holds an unsupported value.
    """
    binunit = values.pop("binunit", 1.0)
    if not (math.isfinite(binunit) and binunit in _BSIM_BINUNITS):
        message = f"`BINUNIT` of `{card.name}` is 1 or 2, not {binunit!r}"
        raise statement.fail(message)
    binned = 1e-6 if binunit == 1.0 else 1.0
    mobility = values.get("u0", 0.067 if card.kind[0] == "n" else 0.025)
    # ngspice reads a binned u0 above 1 in cm²/(V·s), as b3temp.c and b4v5temp.c do.
    centimeters = 1e-4 if mobility > 1 else 1.0
    parameters = nominal_temperature(
        {
            bsim.aliases.get(spice, spice): value
            for spice, value in values.items()
            if bsim.aliases.get(spice, spice) not in bsim.unread
        },
        statement,
    )
    for spice, value in values.items():
        name = bsim.aliases.get(spice, spice)
        if name in bsim.unread or name in NOMINAL_TEMPERATURES:
            continue
        if name in bsim.selectors:
            parameters[bsim.selectors[name]] = _whole(card, name, value, statement)
        elif name in bsim.settings:
            parameters[bsim.settings[name]] = _bsim_setting(
                card, name, value, statement
            )
        elif name in bsim.card:
            parameters[name] = value * _bsim_factor(bsim, name, value, centimeters)
        elif name[:1] in {"l", "w", "p"} and name[1:] in bsim.binned:
            extent = binned * binned if name[0] == "p" else binned
            factor = _bsim_factor(bsim, name[1:], value, centimeters)
            parameters[name] = value * factor * extent
        else:
            raise unmatched(card, spice, statement)
    return parameters


def _bsim_factor(bsim: Bsim, name: str, value: float, centimeters: float) -> float:
    """Returns the SI unit multiplier for BSIM parameter `name` given `value`.

    Uses ngspice density thresholds for doping concentrations and `centimeters` for
    mobility ``u0``.
    """
    if name in bsim.densities:
        return 1.0 if value > bsim.densities[name] else 1e6
    if name == "nsub":
        return 1e6
    if name == "u0":
        return centimeters
    return 1.0


def _whole(card: ModelCard, name: str, value: float, statement: Statement) -> int:
    """Validates that selector `value` for `name` on `card` is an integer.

    Raises:
        SpiceError: When `value` has a fractional part.
    """
    if not value.is_integer():
        message = f"`{name.upper()}` of `{card.name}` is a whole number, not {value!r}"
        raise statement.fail(message)
    return int(value)


def _vbic(
    card: ModelCard,
    values: dict[str, float],
    instance: Mapping[str, float],
    statement: Statement,
) -> dict[str, JsonValue]:
    """Translates a VBIC model card and instance into jolty parameters matching ngspice.

    Clamps card resistances to at least 10 mΩ and omits non-positive infinite-default
    parameters. Scales currents, capacitances, charges, and resistances by
    ``area * m``, scales flicker coefficients as ``m · KFN · (|I| / m)^AFN``, maps
    ``QBM`` to base charge mode, and accepts ``TREF`` as ``TNOM``.

    Raises:
        SpiceError: When a parameter has no jolty counterpart or `instance` contains an
            unsupported key outside `VBIC_INSTANCE`.
    """
    for name in instance:
        if name not in VBIC_INSTANCE:
            message = f"a VBIC takes no `{name.upper()}`"
            raise statement.fail(message)
    values = _vbic_read(values)
    charge = values.pop("qbm", None)
    copies = instance.get("m", 1.0)
    area = _area(instance, statement)
    values = _area_scaled("vbic", values, area, VBIC_MULTIPLIED, VBIC_RESISTANCES)
    for coefficient, exponent in (("kfn", "afn"), ("kf", "af")):
        if coefficient in values:
            values[coefficient] = scale_flicker_coefficient(
                values[coefficient],
                values.get(exponent, 1.0),
                copies,
                statement,
                f"of `{card.name}`",
            )
    parameters = _mapped(card, values, VBIC, statement, allow_tref=True)
    parameters["polarity"] = card.kind
    if charge is not None:
        threshold = 0.5  # ngspice threshold below which QBM selects VBIC form
        parameters["base_charge"] = "vbic" if charge < threshold else "gummel_poon"
    return parameters


def _vbic_read(values: Mapping[str, float]) -> dict[str, float]:
    """Normalizes VBIC card `values` according to ngspice preprocessing rules.

    Drops ignored and self-heating keys, clamps resistances to `VBIC_MIN_RESISTANCE`,
    and drops non-positive values of `VBIC_INFINITE_AT_ZERO` parameters.
    """
    read = {
        name: value
        for name, value in values.items()
        if name not in VBIC_IGNORED | VBIC_SELF_HEATING | {"level"}
    }
    for name in VBIC_RESISTANCES:
        if name in read:
            read[name] = max(read[name], VBIC_MIN_RESISTANCE)
    return {
        name: value
        for name, value in read.items()
        if name not in VBIC_INFINITE_AT_ZERO or value > 0
    }


def vbic_heating(
    card: ModelCard,
    scope: Mapping[str, float],
    instance: Mapping[str, float],
    *,
    functions: Mapping[str, Function] = _NO_FUNCTIONS,
) -> tuple[float, float] | None:
    """Returns scaled `(rth, cth)` self-heating parameters for a VBIC card, or `None`.

    Active when ``RTH`` is positive and ``SELFT`` is omitted or equal to 1. Clamps
    ``CTH`` to `VBIC_MIN_THERMAL_CAPACITANCE`, divides thermal resistance (K/W) by
    ``area * m``, and multiplies thermal capacitance (J/K) by ``area * m``.

    Args:
        card: VBIC bipolar transistor model card.
        scope: Case-folded parameter values in scope for evaluating thermal parameters.
        instance: Evaluated element-level `area` and multiplicity `m` overrides.
        functions: Case-folded user-defined functions inlined before evaluation.

    Returns:
        Area-scaled `(rth, cth)` pair in K/W and J/K when self-heating is active, or
        `None`.
    """
    given = {
        name: evaluate(inlined(card.parameters[name], functions), scope)
        for name in VBIC_SELF_HEATING
        if name in card.parameters
    }
    resistance = given.get("rth", 0.0)
    if resistance <= 0 or given.get("selft", 1.0) != 1:
        return None
    area = instance.get("area", 1.0) * instance.get("m", 1.0)
    if not (area > 0 and math.isfinite(area)):
        return None
    capacitance = max(given.get("cth", 0.0), VBIC_MIN_THERMAL_CAPACITANCE)
    return resistance / area, capacitance * area
