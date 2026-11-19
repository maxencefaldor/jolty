//! The lossless transmission line against the closed forms of its waves.
#![cfg(test)]

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use jolty_engine::{
    Ac, Conditions, Method, Options, PeriodicSteadyState, PoleZero, SimulationError, Simulator,
    Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::{Value, json};

/// The characteristic impedance of every line here, in ohms.
const IMPEDANCE: f64 = 50.0;
/// The delay of every line here, in seconds.
const DELAY: f64 = 1e-9;
/// The time the source's edge starts rising, and the time it rises in, in seconds.
const EDGE: (f64, f64) = (0.5e-9, 0.2e-9);

/// The bound on a wave's error, in volts, where the waves are piecewise linear: each point
/// a transient accepts is exact but for rounding, as no charge is integrated, and each
/// delayed value an interpolation within a straight run between corners, exact but for
/// rounding too, the corners each stepped onto. A thousand ulps of the 1 V the waves are.
const ROUNDING: f64 = 1e3 * f64::EPSILON;

/// A circuit whose top component has a ground port `0` and places `instances`.
fn circuit(instances: &[Value]) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

fn resistor(name: &str, resistance: f64, a: &str, b: &str) -> Value {
    json!({"name": name, "of": {"model": "resistor"},
           "parameters": {"resistance": resistance}, "terminals": {"a": a, "b": b}})
}

/// A line from port `a`, over the ground, to port `b`, over the ground.
fn line(name: &str, a: &str, b: &str) -> Value {
    json!({"name": name, "of": {"model": "transmission_line"},
           "parameters": {"characteristic_impedance": IMPEDANCE, "delay": DELAY},
           "terminals": {"positive1": a, "negative1": "0", "positive2": b, "negative2": "0"}})
}

/// A source at `positive` of a 1 V edge, [`EDGE`], from 0 V.
fn edge(positive: &str) -> Value {
    let (delay, rise) = EDGE;
    json!({"name": "V1", "of": {"model": "voltage_source"},
           "parameters": {"waveform": {"pulse": {
               "initial": 0, "pulsed": 1, "delay": delay, "rise": rise, "fall": rise,
               "width": 1}}},
           "terminals": {"positive": positive, "negative": "0"}})
}

/// The source's level at `time`.
fn source(time: f64) -> f64 {
    let (delay, rise) = EDGE;
    ((time - delay) / rise).clamp(0.0, 1.0)
}

fn simulator(circuit: &Circuit) -> Simulator {
    Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("the circuit elaborates")
}

/// The transient of `circuit` until `stop`, by `method`: its times, and the voltages of
/// `nets` at each.
fn waves(circuit: &Circuit, stop: f64, method: Method, nets: &[&str]) -> Vec<(f64, Vec<f64>)> {
    let mut simulator = simulator(circuit);
    let transient = Transient {
        stop,
        max_step: None,
        method,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    };
    let result = simulator
        .transient(&transient, &Options::default())
        .expect("a transient");
    let unknowns: Vec<usize> = nets
        .iter()
        .map(|net| simulator.unknown(net).expect("a net").index())
        .collect();
    result
        .times
        .iter()
        .zip(&result.values)
        .map(|(&time, values)| (time, unknowns.iter().map(|&at| values[at]).collect()))
        .collect()
}

/// Asserts each of `found`'s waves is `expected`'s at each time within [`ROUNDING`], and
/// that the transient stepped onto every time of `corners`, to the rounding of its sums.
fn assert_waves(found: &[(f64, Vec<f64>)], expected: impl Fn(f64) -> Vec<f64>, corners: &[f64]) {
    for &(time, ref values) in found {
        for (net, (value, exact)) in values.iter().zip(expected(time)).enumerate() {
            assert!(
                (value - exact).abs() <= ROUNDING,
                "wave {net} at {time:e} s: {value} V, not {exact} V"
            );
        }
    }
    // A corner arriving after several delays is their sum, which the transient adds one
    // delay at a time, within a few ulps of the corner's time.
    for corner in corners {
        assert!(
            found
                .iter()
                .any(|(time, _)| (time - corner).abs() <= 8.0 * f64::EPSILON * corner),
            "no step onto {corner:e} s"
        );
    }
}

/// A source matched to the line sends half its edge down it, which a matched load takes
/// whole a delay later: no reflection, every edge stepped onto.
#[test]
fn a_matched_line_delays_half_the_edge() {
    let circuit = circuit(&[
        edge("in"),
        resistor("R1", IMPEDANCE, "in", "a"),
        line("T1", "a", "b"),
        resistor("R2", IMPEDANCE, "b", "0"),
    ]);
    for method in [Method::Trapezoidal, Method::Gear2, Method::BackwardEuler] {
        let found = waves(&circuit, 4e-9, method, &["a", "b"]);
        let (delay, rise) = EDGE;
        assert_waves(
            &found,
            |time| vec![source(time) / 2.0, source(time - DELAY) / 2.0],
            &[delay + DELAY, delay + rise + DELAY],
        );
    }
}

/// An open end doubles the wave arriving and sends it back; a source of half the line's
/// impedance sends a third of the wave arriving back again, inverted: the lattice of
/// reflections, the source's reflection coefficient −1/3 and the open end's 1.
#[test]
fn an_open_line_rings_down_by_its_reflections() {
    let resistance = IMPEDANCE / 2.0;
    let circuit = circuit(&[
        edge("in"),
        resistor("R1", resistance, "in", "a"),
        line("T1", "a", "b"),
    ]);
    let sent = IMPEDANCE / (resistance + IMPEDANCE);
    let at_source = (resistance - IMPEDANCE) / (resistance + IMPEDANCE);
    // The far end's voltage: each wave, doubled at the open end, 2n + 1 delays after the
    // source's edge, after n round trips.
    let far = |time: f64| -> f64 {
        (0..20)
            .map(|trips: i32| {
                let delays = f64::from(2 * trips + 1);
                2.0 * sent * at_source.powi(trips) * source(time - delays * DELAY)
            })
            .sum()
    };
    // The near end's: the wave sent, then each returning wave and its reflection there.
    let near = |time: f64| -> f64 {
        sent * source(time)
            + (1..20)
                .map(|trips: i32| {
                    let returned = 2.0 * sent * at_source.powi(trips - 1);
                    returned * (1.0 + at_source) / 2.0 * source(time - f64::from(2 * trips) * DELAY)
                })
                .sum::<f64>()
    };
    let found = waves(&circuit, 8e-9, Method::Trapezoidal, &["a", "b"]);
    let (delay, rise) = EDGE;
    assert_waves(
        &found,
        |time| vec![near(time), far(time)],
        &[
            delay + DELAY,
            delay + 2.0 * DELAY,
            delay + rise + 5.0 * DELAY,
        ],
    );
}

/// A shorted end inverts the wave arriving, so a matched source sees its edge return
/// inverted two delays later, and the short carries twice the wave's current.
#[test]
fn a_shorted_line_returns_the_edge_inverted() {
    let circuit = circuit(&[
        edge("in"),
        resistor("R1", IMPEDANCE, "in", "a"),
        line("T1", "a", "b"),
        resistor("R2", 1e-3, "b", "0"),
    ]);
    let found = waves(&circuit, 4e-9, Method::Trapezoidal, &["a", "b"]);
    // The 1 mΩ short leaves the far end the share of its divider with the line's 50 Ω
    // of twice the wave arriving, and reflects the rest.
    let short = 1e-3;
    let reflected = (short - IMPEDANCE) / (short + IMPEDANCE);
    assert_waves(
        &found,
        |time| {
            let arriving = source(time - DELAY) / 2.0;
            vec![
                source(time) / 2.0 + reflected * source(time - 2.0 * DELAY) / 2.0,
                (1.0 + reflected) * arriving,
            ]
        },
        &[],
    );
}

/// At DC a line is a short through, from port to port.
#[test]
fn at_dc_a_line_is_a_short_through() {
    let circuit = circuit(&[
        json!({"name": "V1", "of": {"model": "voltage_source"},
               "parameters": {"waveform": {"dc": 1.0}},
               "terminals": {"positive": "in", "negative": "0"}}),
        resistor("R1", 1e3, "in", "a"),
        line("T1", "a", "b"),
        resistor("R2", 3e3, "b", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let point = simulator
        .operating_point(&Options::default())
        .expect("an operating point");
    for net in ["a", "b"] {
        let value = point.values[simulator.unknown(net).expect("a net").index()];
        // The divider's 3/4 but for the solver's rounding.
        assert!((value - 0.75).abs() < 1e-12, "{net}: {value} V");
    }
}

/// A line's input impedance with a load `Z_L` at its far end is
/// `Z₀ (Z_L + j Z₀ tan βl) / (Z₀ + j Z_L tan βl)`, `βl = ωτ`: a quarter wave inverts the
/// load's impedance about `Z₀`, `Z₀² / Z_L`.
#[test]
fn a_quarter_wave_inverts_its_load() {
    let load = 100.0;
    let circuit = circuit(&[
        json!({"name": "I1", "of": {"model": "current_source"},
               "parameters": {"waveform": {"dc": 0.0}},
               "terminals": {"positive": "0", "negative": "a"}}),
        line("T1", "a", "b"),
        resistor("R1", load, "b", "0"),
    ]);
    let quarter = 1.0 / (4.0 * DELAY);
    let frequencies = vec![0.0, quarter / 3.0, quarter, 1.7 * quarter, 2.0 * quarter];
    let mut simulator = simulator(&circuit);
    let ac = Ac {
        source: "I1".to_owned(),
        frequencies: frequencies.clone(),
    };
    let response = simulator
        .ac(&ac, &Options::default())
        .expect("an AC analysis");
    let a = simulator.unknown("a").expect("a net").index();
    for (frequency, values) in frequencies.iter().zip(&response.values) {
        let tangent = Complex64::new(0.0, (TAU * frequency * DELAY).tan());
        let exact = IMPEDANCE * (load + tangent * IMPEDANCE) / (IMPEDANCE + tangent * load);
        // The factorization's rounding, a few ulps of the impedances times the small
        // system's condition, far within a part in 1e9.
        assert!(
            (values[a] - exact).norm() <= 1e-9 * exact.norm(),
            "{frequency} Hz: {} Ω, not {exact} Ω",
            values[a]
        );
    }
    let inverted = response.values[2][a];
    assert!((inverted - IMPEDANCE * IMPEDANCE / load).norm() < 1e-9 * IMPEDANCE);
}

/// The analyses that cannot take a delay refuse it, naming the line.
#[test]
fn a_pole_zero_analysis_and_a_periodic_steady_state_refuse_a_line() {
    let circuit = circuit(&[
        edge("in"),
        resistor("R1", IMPEDANCE, "in", "a"),
        line("T1", "a", "b"),
        resistor("R2", IMPEDANCE, "b", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let poles = simulator.pole_zero(
        &PoleZero {
            source: None,
            output: None,
            reference: None,
        },
        &Options::default(),
    );
    let Err(SimulationError::Delayed { devices, .. }) = poles else {
        panic!("{poles:?}");
    };
    assert_eq!(devices, ["`T1`"]);
    let periodic = simulator.periodic_steady_state(
        &PeriodicSteadyState {
            period: Some(4e-9),
            ..PeriodicSteadyState::default()
        },
        &Options::default(),
    );
    let Err(SimulationError::Delayed { reason, .. }) = periodic else {
        panic!("{periodic:?}");
    };
    assert!(
        reason.contains("which shooting does not carry"),
        "{reason:?}"
    );
}
