//! The cases both kernel benchmarks measure, `evaluation.rs` by time and
//! `instructions.rs` by instructions: one instance of a device on the shared bench,
//! evaluated at a step of a transient in one of the shapes its parameters give it, each
//! made by the function named after its model and its case; and a twin of an instance,
//! differentiated along 1, 2, 4 or 8 of its parameters at once, made by the function named
//! after its model and `twin`.

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
    /// An instance of `K` with `parameters`, each of its unknowns at the voltage `bias`
    /// gives its name: the terminals' `n0`, `n1` and on, and the device's own their
    /// labels, such as `collector`. Branch current unknowns default to `1e-3` A; any
    /// other unknown omitted by `bias` panics so typos never silently bias a node at 1 mV.
    #[expect(
        clippy::panic,
        reason = "every non-current unknown in a benchmark case must be explicitly biased"
    )]
    fn of<K: Kernel>(parameters: &str, bias: fn(&str) -> Option<f64>) -> Self
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

/// The input of a step of a transient at `x`, its steps unlimited.
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
    /// The twin of an instance of `K` with `parameters`, differentiated along the first
    /// `seeds` of the numbers at `paths`, each alone, its unknowns where [`Case::of`] puts
    /// them by `bias`.
    #[expect(
        clippy::panic,
        reason = "a case's parameters are the bench's own, each read as a real number"
    )]
    fn of<K: Kernel>(
        (parameters, paths): (&str, &[&str]),
        seeds: usize,
        bias: fn(&str) -> Option<f64>,
    ) -> Self
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        let Case { x, .. } = Case::of::<K>(parameters, bias);
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

/// A bipolar transistor's unknowns, forward active: its collector at 3 V, its base at
/// 0.75 V, its emitter grounded and its substrate at −1 V, each internal node a little
/// inside the terminal it lies behind, and each current unknown a milliampere.
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
    "intrinsic_collector_resistance": 20, "epi_saturation_voltage": 2,
    "epi_doping_parameter": 2e-11, "high_current_collector_factor": 0.5,
    "intrinsic_base_resistance": 30, "parasitic_base_resistance": 40,
    "base_emitter_capacitance": 2e-14, "base_collector_capacitance": 1e-14,
    "extrinsic_base_collector_capacitance": 5e-15, "epi_charge": 1e-14,
    "collector_substrate_capacitance": 3e-14, "forward_transit_time": 1e-11,
    "transit_time_bias_coefficient": 3, "transit_time_voltage": 2,
    "transit_time_current": 0.01, "reverse_transit_time": 1e-9,
    "base_emitter_breakdown_voltage": 5, "reach_through_voltage": 3,
    "reach_through_smoothing": 0.1, "extrinsic_collector_resistance": 5,
    "extrinsic_base_resistance": 8, "emitter_resistance": 2, "substrate_resistance": 3}"#;

/// `parameters` with `more`, a member or more of a JSON object, first among them.
fn with(parameters: &str, more: &str) -> String {
    parameters.replacen('{', &format!("{{{more}, "), 1)
}

/// A Gummel-Poon transistor at its defaults, forward active.
pub(crate) fn bipolar_transistor_defaults() -> Case {
    Case::of::<BipolarTransistor>("{}", forward_active)
}

/// A Gummel-Poon transistor with every part of its model in play, forward active.
pub(crate) fn bipolar_transistor_full() -> Case {
    Case::of::<BipolarTransistor>(GUMMEL_POON_FULL, forward_active)
}

/// [`bipolar_transistor_full`] with an excess phase of 30°.
pub(crate) fn bipolar_transistor_delayed() -> Case {
    let delayed = with(GUMMEL_POON_FULL, r#""excess_phase": 30"#);
    Case::of::<BipolarTransistor>(&delayed, forward_active)
}

/// A VBIC transistor at its defaults, forward active.
pub(crate) fn vbic_defaults() -> Case {
    Case::of::<Vbic>("{}", forward_active)
}

/// A VBIC transistor with every part of its model in play, forward active.
pub(crate) fn vbic_full() -> Case {
    Case::of::<Vbic>(VBIC_FULL, forward_active)
}

/// [`vbic_full`] with an excess phase's delay of 2 ps.
pub(crate) fn vbic_delayed() -> Case {
    let delayed = with(VBIC_FULL, r#""excess_phase_delay": 2e-12"#);
    Case::of::<Vbic>(&delayed, forward_active)
}

/// A field-effect transistor's unknowns, saturated: its drain at 3 V, its gate at −0.5 V
/// and its source grounded, each internal node a little inside the terminal it lies
/// behind, and each current unknown a milliampere.
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

/// A JFET at its defaults, saturated.
pub(crate) fn jfet_defaults() -> Case {
    Case::of::<Jfet>("{}", saturated)
}

/// A JFET with every part of its model in play, saturated.
pub(crate) fn jfet_full() -> Case {
    Case::of::<Jfet>(JFET_FULL, saturated)
}

/// A power MOSFET's unknowns, its drain resistance quasi-saturated: its drain at 20 V, its
/// gate at 8 V and its source grounded, each internal node a little inside the terminal it
/// lies behind, and each current unknown a milliampere.
fn quasi_saturated(name: &str) -> Option<f64> {
    match name {
        "n0" => Some(20.0),
        "drain" => Some(19.8),
        "n1" | "gate" => Some(8.0),
        _ => saturated(name),
    }
}

/// A power MOSFET whose drain resistance quasi-saturates.
pub(crate) fn vdmos_quasi_saturated() -> Case {
    let quasi = r#"{"threshold_voltage": 3, "transconductance_parameter": 10,
        "drain_resistance": 2, "quasi_saturation_resistance": 5,
        "quasi_saturation_voltage": 20}"#;
    Case::of::<Vdmos>(quasi, quasi_saturated)
}

/// A two-terminal device's unknowns, with what controls it: 1.5 V across it, 1.2 V,
/// 1.2 mA or 1.2 lux on its control, its state at 0.3, its lagging response at 0.5, and
/// each other current unknown a milliampere.
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

/// A resistor with voltage coefficients.
pub(crate) fn resistor_coefficients() -> Case {
    let parameters = with(r#"{"resistance": 1000}"#, COEFFICIENTS);
    Case::of::<Resistor>(&parameters, controlled)
}

/// A capacitor with voltage coefficients.
pub(crate) fn capacitor_coefficients() -> Case {
    let parameters = with(r#"{"capacitance": 1e-12}"#, COEFFICIENTS);
    Case::of::<Capacitor>(&parameters, controlled)
}

/// A voltage-controlled switch between off and on.
pub(crate) fn voltage_controlled_switch_between() -> Case {
    let parameters = r#"{"on_voltage": 1.5, "off_voltage": 1}"#;
    Case::of::<VoltageControlledSwitch>(parameters, controlled)
}

/// A current-controlled switch between off and on.
pub(crate) fn current_controlled_switch_between() -> Case {
    let parameters = r#"{"on_current": 1.5e-3, "off_current": 1e-3}"#;
    Case::of::<CurrentControlledSwitch>(parameters, controlled)
}

/// A voltage-controlled memristor driven toward off.
pub(crate) fn voltage_controlled_memristor_driven() -> Case {
    let parameters = r#"{"on_resistance": 100, "off_resistance": 1e4, "on_voltage": -1,
        "off_voltage": 1, "on_rate": 1e3, "off_rate": 1e3}"#;
    Case::of::<VoltageControlledMemristor>(parameters, controlled)
}

/// A current-controlled memristor, driven by its current.
pub(crate) fn current_controlled_memristor_driven() -> Case {
    let parameters = r#"{"on_resistance": 100, "off_resistance": 1e4, "charge": 1e-6}"#;
    Case::of::<CurrentControlledMemristor>(parameters, controlled)
}

/// A photocell whose response lags the light.
pub(crate) fn photocell_lagging() -> Case {
    let parameters = r#"{"resistance": 1e4, "response_time": 1e-3}"#;
    Case::of::<Photocell>(parameters, controlled)
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

/// A BSIM4 n-channel's unknowns, saturated: its drain at 1.5 V, its gate at 1.8 V, its
/// source grounded and its body at −0.5 V, each internal node at the terminal it lies
/// behind, and each current unknown a milliampere.
fn bsim4_saturated(name: &str) -> Option<f64> {
    Some(match name {
        "n0" | "drain" => 1.5,
        "n1" | "gate" => 1.8,
        "n3" | "body" => -0.5,
        "n2" | "source" => 0.0,
        _ => return None,
    })
}

/// A twin of a BSIM4 n-channel along `seeds` of its numbers, saturated.
pub(crate) fn bsim4_twin(seeds: usize) -> TwinCase {
    TwinCase::of::<Bsim4>(BSIM4_GIVEN, seeds, bsim4_saturated)
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

/// A diode's unknowns, forward: 0.7 V across it, its internal node 0.69 V above its
/// cathode, and each current unknown a milliampere.
fn forward(name: &str) -> Option<f64> {
    Some(match name {
        "n0" => 0.7,
        "junction" => 0.69,
        "n1" => 0.0,
        _ => return None,
    })
}

/// A twin of a diode along `seeds` of its numbers, forward.
pub(crate) fn diode_twin(seeds: usize) -> TwinCase {
    TwinCase::of::<Diode>(DIODE_GIVEN, seeds, forward)
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

/// A twin of a resistor with voltage coefficients along `seeds` of its numbers.
pub(crate) fn resistor_twin(seeds: usize) -> TwinCase {
    TwinCase::of::<Resistor>(RESISTOR_GIVEN, seeds, controlled)
}
