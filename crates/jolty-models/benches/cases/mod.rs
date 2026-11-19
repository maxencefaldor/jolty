//! Shared device benchmark cases measured by `evaluation.rs` (wall time) and
//! `instructions.rs` (instruction count). Each single-device constructor prepares one
//! instance on the shared bench for a transient step. Twin constructors (`*_twin`)
//! prepare dual-number instances differentiated along 1, 2, 4, or 8 parameters at once.

use jolty_models::testing::{Bench, twin};
use jolty_models::{
    BipolarTransistor, Bsim4, Capacitor, Contributions, CurrentControlledMemristor,
    CurrentControlledSwitch, Diode, Input, Jfet, Kernel, Nature, Photocell, REFERENCE_TEMPERATURE,
    Resistor, Seeds, Twin, Vbic, Vdmos, VoltageControlledMemristor, VoltageControlledSwitch,
};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// One instance on the bench, its unknowns where it is evaluated, and the contributions
/// it writes.
pub(crate) struct Case {
    bench: Bench,
    x: Vec<f64>,
    contributions: Contributions,
}

impl Case {
    /// Constructs a benchmark case for `K` with `parameters`, biasing every non-current
    /// unknown via `bias` (`n0`, `n1`, ..., or internal labels such as `collector`).
    /// Branch current unknowns default to `1e-3` A; omitting any other unknown panics so
    /// typos never silently bias a node at 1 mV.
    #[expect(
        clippy::panic,
        reason = "every non-current unknown in a benchmark case requires an explicit bias"
    )]
    fn from_bias<K: Kernel>(parameters: &str, bias: fn(&str) -> Option<f64>) -> Self
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        let bench = Bench::at::<K>(parameters, REFERENCE_TEMPERATURE);
        let x = bench
            .unknowns()
            .into_iter()
            .zip(bench.natures())
            .map(|(unknown, nature)| {
                let label = unknown.strip_prefix("device:").unwrap_or(unknown);
                bias(label).unwrap_or_else(|| match nature {
                    Nature::Current => 1e-3,
                    _ => panic!("unbiased {nature:?} unknown `{label}` in benchmark case"),
                })
            })
            .collect();
        let contributions = bench.contributions();
        Self {
            bench,
            x,
            contributions,
        }
    }

    /// Evaluates the instance once, at a step of a transient, its steps unlimited:
    /// whether it limited one.
    pub(crate) fn evaluate(&mut self) -> bool {
        self.bench
            .evaluate_into(&stepping(&self.x), &mut self.contributions)
    }
}

/// Builds an unlimited transient-step [`Input`] at unknown vector `x`.
fn stepping(x: &[f64]) -> Input<'_> {
    let mut input = Input::new(x, Some(1e-9), 1e-12);
    input.limiting = false;
    input
}

/// A twin of one instance, and the unknowns where it is evaluated.
pub(crate) struct TwinCase {
    twin: Twin,
    x: Vec<f64>,
}

impl TwinCase {
    /// Constructs a twin benchmark case for `K` with `parameters`, seeded along the first
    /// `seeds` paths of `paths` and biased via [`Case::from_bias`].
    #[expect(
        clippy::panic,
        reason = "a case's parameters are the bench's own, each read as a real number"
    )]
    fn from_bias<K: Kernel>(
        (parameters, paths): (&str, &[&str]),
        seeds: usize,
        bias: fn(&str) -> Option<f64>,
    ) -> Self
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        let Case { x, .. } = Case::from_bias::<K>(parameters, bias);
        let seeds = Seeds::units(paths.get(..seeds).unwrap_or(paths));
        let twin = twin::<K>(parameters, REFERENCE_TEMPERATURE, &seeds)
            .unwrap_or_else(|| panic!("{parameters} reads a real number at each of {paths:?}"));
        Self { twin, x }
    }

    /// Evaluates the twin once, at a step of a transient.
    pub(crate) fn evaluate(&mut self) {
        self.twin.evaluate(&stepping(&self.x));
    }
}

/// Returns forward-active unknown values for a bipolar transistor (collector 3 V, base
/// 0.75 V, emitter 0 V, substrate -1 V). Internal nodes offset slightly inside their
/// terminals, and omitted current unknowns default to 1 mA.
fn forward_active(name: &str) -> Option<f64> {
    Some(match name {
        "n0" => 3.0,
        "collector" => 2.99,
        "intrinsic_collector" => 2.9,
        "n1" => 0.75,
        "base" => 0.746,
        "intrinsic_base" => 0.74,
        "parasitic_base" => 0.7,
        "emitter" => 0.002,
        "n3" => -1.0,
        "substrate" => -0.999,
        "n2" => 0.0,
        _ => return None,
    })
}

/// A Gummel-Poon transistor with every part of its model in play: its base resistance
/// falling with its base current, its series resistances above 1 Ω, and a vertical
/// substrate junction.
const GUMMEL_POON_FULL: &str = r#"{"saturation_current": 1e-15, "forward_beta": 150,
    "forward_early_voltage": 60, "reverse_early_voltage": 20,
    "forward_knee_current": 0.05, "reverse_knee_current": 0.01,
    "base_emitter_leakage_current": 1e-14, "base_collector_leakage_current": 1e-13,
    "base_emitter_capacitance": 2e-12, "base_collector_capacitance": 1e-12,
    "base_collector_grading": 0.4, "internal_base_fraction": 0.7,
    "forward_transit_time": 4e-10, "transit_time_bias_coefficient": 2,
    "transit_time_voltage": 3, "transit_time_current": 0.02, "reverse_transit_time": 1e-8,
    "base_resistance": 200, "min_base_resistance": 20, "base_resistance_current": 1e-4,
    "collector_resistance": 10, "emitter_resistance": 2, "substrate": "vertical",
    "substrate_saturation_current": 1e-16, "substrate_capacitance": 3e-13}"#;

/// VBIC with every part of its model in play, with series resistances above 1 Ω.
const VBIC_FULL: &str = r#"{"saturation_current": 2e-16, "forward_emission_coefficient": 1.01,
    "reverse_emission_coefficient": 1.02, "reverse_saturation_current_ratio": 0.8,
    "forward_early_voltage": 60, "reverse_early_voltage": 4,
    "forward_knee_current": 0.02, "reverse_knee_current": 0.005, "knee_rolloff": 0.6,
    "ideal_base_emitter_current": 2e-18, "intrinsic_base_emitter_fraction": 0.8,
    "nonideal_base_emitter_current": 3e-15, "ideal_base_collector_current": 4e-17,
    "nonideal_base_collector_current": 2e-15,
    "avalanche_coefficient": 2.5, "avalanche_exponent_coefficient": 12,
    "parasitic_saturation_current": 5e-18, "parasitic_extrinsic_fraction": 0.7,
    "parasitic_ideal_base_emitter_current": 1e-19,
    "parasitic_ideal_base_collector_current": 2e-19, "parasitic_knee_current": 1e-4,
    "intrinsic_collector_resistance": 20, "epitaxial_saturation_voltage": 2,
    "epitaxial_doping_parameter": 2e-11, "high_current_collector_factor": 0.5,
    "intrinsic_base_resistance": 30, "parasitic_base_resistance": 40,
    "base_emitter_capacitance": 2e-14, "base_collector_capacitance": 1e-14,
    "extrinsic_base_collector_capacitance": 5e-15, "epitaxial_charge": 1e-14,
    "collector_substrate_capacitance": 3e-14, "forward_transit_time": 1e-11,
    "transit_time_bias_coefficient": 3, "transit_time_voltage": 2,
    "transit_time_current": 0.01, "reverse_transit_time": 1e-9,
    "base_emitter_breakdown_voltage": 5, "reach_through_voltage": 3,
    "reach_through_smoothing": 0.1, "extrinsic_collector_resistance": 5,
    "extrinsic_base_resistance": 8, "emitter_resistance": 2, "substrate_resistance": 3}"#;

/// Returns `parameters` with JSON member list `more` prepended.
fn with(parameters: &str, more: &str) -> String {
    parameters.replacen('{', &format!("{{{more}, "), 1)
}

/// Builds a forward-active Gummel-Poon transistor case with default parameters.
pub(crate) fn bipolar_transistor_defaults() -> Case {
    Case::from_bias::<BipolarTransistor>("{}", forward_active)
}

/// Builds a forward-active Gummel-Poon transistor case with full model parameters.
pub(crate) fn bipolar_transistor_full() -> Case {
    Case::from_bias::<BipolarTransistor>(GUMMEL_POON_FULL, forward_active)
}

/// Builds [`bipolar_transistor_full`] with 30° excess phase enabled.
pub(crate) fn bipolar_transistor_delayed() -> Case {
    let delayed = with(GUMMEL_POON_FULL, r#""excess_phase": 30"#);
    Case::from_bias::<BipolarTransistor>(&delayed, forward_active)
}

/// Builds a forward-active VBIC transistor case with default parameters.
pub(crate) fn vbic_defaults() -> Case {
    Case::from_bias::<Vbic>("{}", forward_active)
}

/// Builds a forward-active VBIC transistor case with full model parameters.
pub(crate) fn vbic_full() -> Case {
    Case::from_bias::<Vbic>(VBIC_FULL, forward_active)
}

/// Builds [`vbic_full`] with a 2 ps excess-phase delay enabled.
pub(crate) fn vbic_delayed() -> Case {
    let delayed = with(VBIC_FULL, r#""excess_phase_delay": 2e-12"#);
    Case::from_bias::<Vbic>(&delayed, forward_active)
}

/// Returns saturated unknown values for a field-effect transistor (drain 3 V, gate -0.5 V,
/// source 0 V), with internal nodes slightly inside their terminals.
fn saturated(name: &str) -> Option<f64> {
    Some(match name {
        "n0" => 3.0,
        "drain" => 2.99,
        "n1" => -0.5,
        "n2" => 0.0,
        "source" => 0.002,
        _ => return None,
    })
}

/// A JFET with every part of its model in play: charged gate junctions, channel-length
/// modulation, a doping tail and series resistances above 1 Ω.
const JFET_FULL: &str = r#"{"threshold_voltage": -1.8, "transconductance_parameter": 1e-3,
    "channel_length_modulation": 0.02, "doping_tail": 0.8, "gate_source_capacitance": 2e-12,
    "gate_drain_capacitance": 1e-12, "drain_resistance": 10, "source_resistance": 5}"#;

/// Builds a saturated JFET case with default parameters.
pub(crate) fn jfet_defaults() -> Case {
    Case::from_bias::<Jfet>("{}", saturated)
}

/// Builds a saturated JFET case with full model parameters.
pub(crate) fn jfet_full() -> Case {
    Case::from_bias::<Jfet>(JFET_FULL, saturated)
}

/// Returns quasi-saturated unknown values for a power MOSFET (drain 20 V, gate 8 V, source
/// 0 V), with internal nodes slightly inside their terminals.
fn quasi_saturated(name: &str) -> Option<f64> {
    match name {
        "n0" => Some(20.0),
        "drain" => Some(19.8),
        "n1" | "gate" => Some(8.0),
        _ => saturated(name),
    }
}

/// Builds a power MOSFET case in drain-resistance quasi-saturation.
pub(crate) fn vdmos_quasi_saturated() -> Case {
    let quasi = r#"{"threshold_voltage": 3, "transconductance_parameter": 10,
        "drain_resistance": 2, "quasi_saturation_resistance": 5,
        "quasi_saturation_voltage": 20}"#;
    Case::from_bias::<Vdmos>(quasi, quasi_saturated)
}

/// Returns unknown values for a controlled two-terminal device (1.5 V port bias, 1.2 V /
/// 1.2 mA / 1.2 lux control, state 0.3, lagging response 0.5).
fn controlled(name: &str) -> Option<f64> {
    Some(match name {
        "n0" => 1.5,
        "n2" => 1.2,
        "control" => 1.2e-3,
        "state" => 0.3,
        "response" => 0.5,
        "n1" | "n3" => 0.0,
        _ => return None,
    })
}

/// A passive device's voltage coefficients, as members of its parameters.
const COEFFICIENTS: &str = r#""voltage_coefficient": 0.01, "quadratic_voltage_coefficient": 0.002"#;

/// Builds a resistor case with linear and quadratic voltage coefficients.
pub(crate) fn resistor_coefficients() -> Case {
    let parameters = with(r#"{"resistance": 1000}"#, COEFFICIENTS);
    Case::from_bias::<Resistor>(&parameters, controlled)
}

/// Builds a capacitor case with linear and quadratic voltage coefficients.
pub(crate) fn capacitor_coefficients() -> Case {
    let parameters = with(r#"{"capacitance": 1e-12}"#, COEFFICIENTS);
    Case::from_bias::<Capacitor>(&parameters, controlled)
}

/// Builds a voltage-controlled switch case biased in its transition region.
pub(crate) fn voltage_controlled_switch_between() -> Case {
    let parameters = r#"{"on_voltage": 1.5, "off_voltage": 1}"#;
    Case::from_bias::<VoltageControlledSwitch>(parameters, controlled)
}

/// Builds a current-controlled switch case biased in its transition region.
pub(crate) fn current_controlled_switch_between() -> Case {
    let parameters = r#"{"on_current": 1.5e-3, "off_current": 1e-3}"#;
    Case::from_bias::<CurrentControlledSwitch>(parameters, controlled)
}

/// Builds a voltage-controlled memristor case driven toward its off state.
pub(crate) fn voltage_controlled_memristor_driven() -> Case {
    let parameters = r#"{"on_resistance": 100, "off_resistance": 1e4, "on_voltage": -1,
        "off_voltage": 1, "on_rate": 1e3, "off_rate": 1e3}"#;
    Case::from_bias::<VoltageControlledMemristor>(parameters, controlled)
}

/// Builds a current-controlled memristor case driven by branch current.
pub(crate) fn current_controlled_memristor_driven() -> Case {
    let parameters = r#"{"on_resistance": 100, "off_resistance": 1e4, "charge": 1e-6}"#;
    Case::from_bias::<CurrentControlledMemristor>(parameters, controlled)
}

/// Builds a photocell case with finite optical response lag.
pub(crate) fn photocell_lagging() -> Case {
    let parameters = r#"{"resistance": 1e4, "response_time": 1e-3}"#;
    Case::from_bias::<Photocell>(parameters, controlled)
}

/// A BSIM4 n-channel with eight of its numbers given, and their paths, in the order its
/// twins seed them.
const BSIM4_GIVEN: (&str, &[&str]) = (
    r#"{"width": 2e-6, "length": 1.5e-7, "toxe": 3e-9, "vth0": 0.4, "k1": 0.5, "u0": 0.03,
    "vsat": 8e4, "rdsw": 200}"#,
    &[
        "width", "length", "toxe", "vth0", "k1", "u0", "vsat", "rdsw",
    ],
);

/// Returns saturated unknown values for a BSIM4 n-channel MOSFET (drain 1.5 V, gate 1.8 V,
/// source 0 V, body -0.5 V).
fn bsim4_saturated(name: &str) -> Option<f64> {
    Some(match name {
        "n0" | "drain" => 1.5,
        "n1" | "gate" => 1.8,
        "n3" | "body" => -0.5,
        "n2" | "source" => 0.0,
        _ => return None,
    })
}

/// Builds a saturated BSIM4 n-channel twin case seeded along `seeds` parameters.
pub(crate) fn bsim4_twin(seeds: usize) -> TwinCase {
    TwinCase::from_bias::<Bsim4>(BSIM4_GIVEN, seeds, bsim4_saturated)
}

/// A diode with eight of its numbers given, and their paths, in the order its twins seed
/// them.
const DIODE_GIVEN: (&str, &[&str]) = (
    r#"{"saturation_current": 2e-14, "emission_coefficient": 1.05,
    "series_resistance": 5, "junction_capacitance": 2e-12, "junction_potential": 0.8,
    "grading_coefficient": 0.4, "depletion_coefficient": 0.45, "transit_time": 1e-9}"#,
    &[
        "saturation_current",
        "emission_coefficient",
        "series_resistance",
        "junction_capacitance",
        "junction_potential",
        "grading_coefficient",
        "depletion_coefficient",
        "transit_time",
    ],
);

/// Returns forward-biased unknown values for a diode (0.7 V anode, 0.69 V internal
/// junction, 0 V cathode).
fn forward(name: &str) -> Option<f64> {
    Some(match name {
        "n0" => 0.7,
        "junction" => 0.69,
        "n1" => 0.0,
        _ => return None,
    })
}

/// Builds a forward-biased diode twin case seeded along `seeds` parameters.
pub(crate) fn diode_twin(seeds: usize) -> TwinCase {
    TwinCase::from_bias::<Diode>(DIODE_GIVEN, seeds, forward)
}

/// A resistor with eight of its numbers given, and their paths, in the order its twins seed
/// them.
const RESISTOR_GIVEN: (&str, &[&str]) = (
    r#"{"resistance": 1000, "temperature_coefficient": 1e-3,
    "quadratic_temperature_coefficient": 1e-6, "nominal_temperature": 290,
    "voltage_coefficient": 0.01, "quadratic_voltage_coefficient": 0.002,
    "flicker_coefficient": 1e-20, "flicker_exponent": 1.5}"#,
    &[
        "resistance",
        "temperature_coefficient",
        "quadratic_temperature_coefficient",
        "nominal_temperature",
        "voltage_coefficient",
        "quadratic_voltage_coefficient",
        "flicker_coefficient",
        "flicker_exponent",
    ],
);

/// Builds a voltage-dependent resistor twin case seeded along `seeds` parameters.
pub(crate) fn resistor_twin(seeds: usize) -> TwinCase {
    TwinCase::from_bias::<Resistor>(RESISTOR_GIVEN, seeds, controlled)
}
