# Copyright 2026 Maxence Faldor
"""Discrete small-signal transistors (2N3904 and 2N3906) shared across agent tasks."""

from jolty import devices
from jolty.units import A, V, ns, ohm, pF, ps


def npn() -> devices.Device:
    """Returns a 2N3904 small-signal NPN transistor with datasheet SPICE parameters."""
    return devices.bipolar_transistor(
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
    )


def pnp() -> devices.Device:
    """Returns a 2N3906 small-signal PNP transistor with datasheet SPICE parameters."""
    return devices.bipolar_transistor(
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
    )
