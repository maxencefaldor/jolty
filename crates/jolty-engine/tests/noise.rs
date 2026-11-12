//! The noise at an output against closed forms.

use std::f64::consts::TAU;

use jolty_engine::{Conditions, Noise, NoiseSpectrum, Options, SimulationError, Simulator};
use jolty_ir::Circuit;
use jolty_models::{
    BOLTZMANN, ELEMENTARY_CHARGE, Library, Quantity, REFERENCE_TEMPERATURE, thermal_voltage,
};
use serde_json::{Value, json};

/// `4kT` at the reference temperature, the default conditions'.
const FOUR_KT: f64 = 4.0 * BOLTZMANN * REFERENCE_TEMPERATURE;

const FREQUENCIES: [f64; 4] = [1.0, 1e3, 1e5, 1e7];

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

fn resistor(id: &str, resistance: f64, a: &str, b: &str) -> Value {
    json!({"id": id, "of": {"device": "resistor"}, "parameters": {"resistance": resistance},
           "terminals": {"a": a, "b": b}})
}

fn voltage_source(id: &str, positive: &str) -> Value {
    json!({"id": id, "of": {"device": "voltage_source"},
           "parameters": {"waveform": {"dc": 5.0}},
           "terminals": {"positive": positive, "negative": "0"}})
}

fn current_source(id: &str, into: &str, current: f64) -> Value {
    json!({"id": id, "of": {"device": "current_source"},
           "parameters": {"waveform": {"dc": current}},
           "terminals": {"positive": "0", "negative": into}})
}

fn noise(
    circuit: &Circuit,
    output: &str,
    reference: Option<&str>,
    source: Option<&str>,
) -> Result<NoiseSpectrum, SimulationError> {
    let mut simulator = Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let analysis = Noise {
        output: output.to_owned(),
        reference: reference.map(str::to_owned),
        source: source.map(str::to_owned),
        frequencies: FREQUENCIES.to_vec(),
    };
    simulator.noise(&analysis, &Options::default())
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs(),
        "{actual:e} vs {expected:e}"
    );
}

/// The contribution of `device` to the spectrum's density at each frequency.
fn share<'a>(spectrum: &'a NoiseSpectrum, device: &str) -> &'a [f64] {
    let mut shares = spectrum
        .contributions
        .iter()
        .filter(|contribution| contribution.device == device);
    let found = shares.next().expect("a contribution");
    assert!(shares.next().is_none(), "one contribution of {device}");
    &found.densities
}

#[test]
fn a_dividers_noise_is_its_parallel_resistances_and_refers_to_its_input() {
    let (upper, lower) = (1e3, 3e3);
    let divider = circuit(&json!([
        voltage_source("V1", "top"),
        resistor("R1", upper, "top", "out"),
        resistor("R2", lower, "out", "0"),
    ]));
    let spectrum = noise(&divider, "out", None, Some("V1")).unwrap();
    let parallel = upper * lower / (upper + lower);
    let referred = spectrum.referred().expect("a source");
    let input = spectrum.input.as_ref().expect("an input");
    assert_eq!(
        (input.source.as_str(), input.quantity),
        ("V1", Quantity::Voltage)
    );
    assert_eq!(spectrum.quantity, Quantity::Voltage);
    for (index, (&density, gain)) in spectrum.densities.iter().zip(&input.gains).enumerate() {
        assert_close(density, FOUR_KT * parallel);
        assert_close(
            share(&spectrum, "R1")[index],
            FOUR_KT / upper * parallel * parallel,
        );
        assert_close(
            share(&spectrum, "R2")[index],
            FOUR_KT / lower * parallel * parallel,
        );
        assert_close(gain.re, 0.75);
        assert!(gain.im.abs() < 1e-12);
        assert_close(referred[index], FOUR_KT * parallel / 0.5625);
    }
    assert_eq!(spectrum.contributions.len(), 2);
}

#[test]
fn a_capacitor_filters_its_resistors_noise() {
    let (resistance, capacitance) = (1e3, 1e-9);
    let filter = circuit(&json!([
        voltage_source("V1", "in"),
        resistor("R1", resistance, "in", "out"),
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": capacitance},
         "terminals": {"a": "out", "b": "0"}},
    ]));
    let spectrum = noise(&filter, "out", None, None).unwrap();
    assert!(spectrum.input.is_none() && spectrum.referred().is_none());
    for (&frequency, &density) in FREQUENCIES.iter().zip(&spectrum.densities) {
        let pole = TAU * frequency * resistance * capacitance;
        assert_close(density, FOUR_KT * resistance / (1.0 + pole * pole));
    }
}

#[test]
fn a_small_resistors_noise_is_a_voltage_in_series() {
    // A current source holds the output, so every resistance's voltage reaches it whole.
    let chain = circuit(&json!([
        current_source("I1", "out", 1e-3),
        resistor("R1", 0.5, "out", "mid"),
        resistor("R2", 1e3, "mid", "0"),
    ]));
    let spectrum = noise(&chain, "out", None, Some("I1")).unwrap();
    for index in 0..FREQUENCIES.len() {
        assert_close(share(&spectrum, "R1")[index], FOUR_KT * 0.5);
        assert_close(share(&spectrum, "R2")[index], FOUR_KT * 1e3);
        assert_close(spectrum.densities[index], FOUR_KT * 1000.5);
        let input = spectrum.input.as_ref().expect("an input");
        assert_close(input.gains[index].re, 1000.5);
        assert_eq!(input.quantity, Quantity::Current);
    }
}

#[test]
fn outputs_can_be_differential_or_currents() {
    let divider = circuit(&json!([
        voltage_source("V1", "top"),
        resistor("R1", 1e3, "top", "out"),
        resistor("R2", 3e3, "out", "0"),
    ]));
    // Across the upper resistor, the same noise as across the lower one.
    let across = noise(&divider, "top", Some("out"), None).unwrap();
    assert_close(across.densities[0], FOUR_KT * 750.0);
    let grounded = noise(&divider, "out", Some("0"), None).unwrap();
    assert_close(grounded.densities[0], FOUR_KT * 750.0);
    // The source's current: the resistors' noise currents, in series.
    let current = noise(&divider, "V1:current", None, None).unwrap();
    assert_close(current.densities[0], FOUR_KT / 4e3);
    assert_eq!(current.quantity, Quantity::Current);
}

#[test]
fn a_diodes_shot_noise_reaches_the_output_through_its_impedance() {
    let biased = circuit(&json!([
        current_source("I1", "out", 1e-3),
        {"id": "D1", "of": {"device": "diode"},
         "parameters": {"saturation_current": 1e-14, "flicker_coefficient": 1e-16},
         "terminals": {"anode": "out", "cathode": "0"}},
    ]));
    let spectrum = noise(&biased, "out", None, Some("I1")).unwrap();
    let gains = &spectrum.input.as_ref().expect("an input").gains;
    let shot = spectrum
        .contributions
        .iter()
        .find(|contribution| contribution.name == "shot")
        .expect("shot noise");
    let flicker = spectrum
        .contributions
        .iter()
        .find(|contribution| contribution.name == "flicker")
        .expect("flicker noise");
    // The diode's current at the operating point, which holds 1 mA to within Newton's
    // tolerance.
    let simulator = Simulator::new(&biased, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    let out = simulator.unknown("out").expect("the output");
    let voltage = spectrum.operating_point.values[out.index()];
    let current = 1e-14 * ((voltage / thermal_voltage(REFERENCE_TEMPERATURE)).exp() - 1.0);
    assert!((current / 1e-3 - 1.0).abs() < 1e-3);
    let close = |actual: f64, expected: f64| (actual / expected - 1.0).abs() < 1e-9;
    for (index, &frequency) in FREQUENCIES.iter().enumerate() {
        let impedance = gains[index].norm_sqr();
        assert!(close(
            shot.densities[index],
            2.0 * ELEMENTARY_CHARGE * current * impedance
        ));
        assert!(close(
            flicker.densities[index],
            1e-16 * current / frequency * impedance
        ));
    }
}

#[test]
fn rejects_what_it_cannot_measure() {
    let divider = circuit(&json!([
        voltage_source("V1", "top"),
        resistor("R1", 1e3, "top", "out"),
        resistor("R2", 3e3, "out", "0"),
    ]));
    assert_eq!(
        noise(&divider, "nowhere", None, None),
        Err(SimulationError::UnknownOutput("nowhere".to_owned()))
    );
    assert_eq!(
        noise(&divider, "out", Some("elsewhere"), None),
        Err(SimulationError::UnknownOutput("elsewhere".to_owned()))
    );
    for (output, reference) in [("out", "out"), ("V1:current", "out"), ("out", "V1:current")] {
        assert!(matches!(
            noise(&divider, output, Some(reference), None),
            Err(SimulationError::InvalidAnalysis(_))
        ));
    }
    assert_eq!(
        noise(&divider, "out", None, Some("R1")),
        Err(SimulationError::NotASource("R1".to_owned()))
    );
    let mut simulator = Simulator::new(&divider, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates");
    for frequencies in [vec![], vec![0.0], vec![1.0, -1.0], vec![f64::INFINITY]] {
        let analysis = Noise {
            output: "out".to_owned(),
            reference: None,
            source: None,
            frequencies,
        };
        assert!(matches!(
            simulator.noise(&analysis, &Options::default()),
            Err(SimulationError::InvalidAnalysis(_))
        ));
    }
}
