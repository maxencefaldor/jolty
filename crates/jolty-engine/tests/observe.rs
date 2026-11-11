//! The current into every terminal of every device, and the charge it holds.

use jolty_engine::{Ac, Conditions, DcSweep, Method, Options, Simulator, Terminals, Transient};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::{Value, json};

fn simulator(instances: &Value) -> Simulator {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit")
}

/// A source stepping into a resistor and a capacitor, a diode across the capacitor, and a
/// MOSFET whose gate and drain share a net and whose source and bulk join the reference.
fn bench() -> Value {
    json!([
        {"id": "V1", "of": {"device": "voltage_source"},
         "parameters": {"waveform": {"pulse": {"initial": 0, "pulsed": 3, "delay": 1e-7,
                                               "rise": 1e-8, "fall": 1e-8, "width": 1e-6}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-10},
         "terminals": {"a": "out", "b": "0"}},
        {"id": "D1", "of": {"device": "diode"},
         "parameters": {"saturation_current": 1e-14, "junction_capacitance": 1e-12},
         "terminals": {"anode": "out", "cathode": "0"}},
        {"id": "M1", "of": {"device": "mosfet"},
         "parameters": {"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.3,
                        "transconductance_parameter": 1e-4},
         "terminals": {"drain": "out", "gate": "out", "source": "0", "bulk": "0"}}
    ])
}

/// Tolerances so tight that what Newton iteration leaves of each node's equation, the sum
/// of the currents into it, is all but nothing.
fn tight() -> Options {
    Options {
        relative_tolerance: 1e-9,
        current_tolerance: 1e-18,
        ..Options::default()
    }
}

/// The index of `name` among the simulator's terminals.
fn terminal(simulator: &Simulator, name: &str) -> usize {
    simulator
        .terminals()
        .iter()
        .position(|found| found == name)
        .unwrap_or_else(|| panic!("no terminal {name}"))
}

/// The sum of the currents into the terminals on each node, which Kirchhoff's current law
/// makes zero, relative to the largest of them.
fn imbalance<T: Copy + Into<Complex64>>(simulator: &Simulator, currents: &[T]) -> f64 {
    let mut sums = vec![Complex64::default(); simulator.unknowns().len()];
    let mut largest = vec![0.0_f64; simulator.unknowns().len()];
    let mut index = 0;
    for placed in simulator.devices() {
        for node in placed.nodes() {
            let current: Complex64 = currents[index].into();
            if let Some(node) = node {
                sums[node.index()] += current;
                largest[node.index()] = largest[node.index()].max(current.norm());
            }
            index += 1;
        }
    }
    assert_eq!(index, currents.len(), "a current per terminal");
    sums.iter()
        .zip(&largest)
        .filter(|(_, largest)| **largest > 0.0)
        .map(|(sum, largest)| sum.norm() / largest)
        .fold(0.0, f64::max)
}

#[test]
fn every_terminal_is_named_by_its_device_and_its_own_name() {
    let simulator = simulator(&bench());
    assert_eq!(
        simulator.terminals(),
        [
            "V1.positive",
            "V1.negative",
            "R1.a",
            "R1.b",
            "C1.a",
            "C1.b",
            "D1.anode",
            "D1.cathode",
            "M1.drain",
            "M1.gate",
            "M1.source",
            "M1.bulk",
        ]
    );
}

#[test]
fn currents_at_an_operating_point_obey_ohm_and_kirchhoff() {
    let mut simulator = simulator(&json!([
        {"id": "V1", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 5.0}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 4e3},
         "terminals": {"a": "out", "b": "0"}},
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-9},
         "terminals": {"a": "out", "b": "0"}}
    ]));
    let point = simulator.operating_point(&Options::default()).unwrap();
    let Terminals { currents, charges } = &point.terminals;
    let current = |name: &str| currents[terminal(&simulator, name)];
    assert!((current("R1.a") - 1e-3).abs() < 1e-12);
    assert!((current("R1.b") + 1e-3).abs() < 1e-12);
    assert!((current("R2.a") - 1e-3).abs() < 1e-12);
    assert!((current("V1.positive") + 1e-3).abs() < 1e-12);
    assert!(current("C1.a").abs() < 1e-18);
    let charge = charges[terminal(&simulator, "C1.a")];
    assert!((charge - 4e-9).abs() < 1e-18, "{charge}");
    assert!((charges[terminal(&simulator, "C1.b")] + 4e-9).abs() < 1e-18);
    assert!(imbalance(&simulator, currents) < 1e-12);
}

#[test]
fn terminals_that_share_a_net_or_join_the_reference_keep_their_own_currents() {
    let mut simulator = simulator(&bench());
    let point = simulator.operating_point(&Options::default()).unwrap();
    let currents = &point.terminals.currents;
    let current = |name: &str| currents[terminal(&simulator, name)];
    // At DC, no current flows into the gate, though it shares its net with the drain.
    assert!(current("M1.gate").abs() < 1e-15);
    assert!(
        current("M1.drain").abs() < 1e-15,
        "the input is low at time zero"
    );
    let mut swept = simulator;
    let sweep = swept
        .dc_sweep(
            &DcSweep {
                source: "V1".to_owned(),
                levels: vec![3.0],
            },
            &tight(),
        )
        .unwrap();
    let currents = &sweep.terminals[0].currents;
    let drain = currents[terminal(&swept, "M1.drain")];
    let source = currents[terminal(&swept, "M1.source")];
    assert!(drain > 1e-5, "{drain}");
    assert!(currents[terminal(&swept, "M1.gate")].abs() < 1e-15);
    assert!((drain + source + currents[terminal(&swept, "M1.bulk")]).abs() < 1e-15);
    assert!(imbalance(&swept, currents) < 1e-9);
}

#[test]
fn transient_currents_take_their_charges_derivative_by_the_step_s_formula() {
    for method in [Method::Trapezoidal, Method::Gear2, Method::BackwardEuler] {
        let mut simulator = simulator(&bench());
        let transient = Transient {
            stop: 2e-6,
            max_step: Some(2e-8),
            method,
            times: vec![1.234_567e-7, 1.5e-6],
        };
        let waveforms = simulator.transient(&transient, &tight()).unwrap();
        let out = simulator
            .unknowns()
            .iter()
            .position(|u| u.name == "out")
            .unwrap();
        let (capacitor, resistor) = (terminal(&simulator, "C1.a"), terminal(&simulator, "R1.b"));
        let mut charging = 0.0_f64;
        for (values, found) in waveforms.values.iter().zip(&waveforms.terminals) {
            let off = imbalance(&simulator, &found.currents);
            assert!(off < 1e-8, "{method:?}: {off}");
            assert!((found.charges[capacitor] - 1e-10 * values[out]).abs() < 1e-22);
            charging = charging.max(found.currents[capacitor]);
            assert!(found.currents[resistor] <= 3.0 / 1e3);
        }
        assert!(charging > 1e-4, "the capacitor charges: {charging}");
        for time in transient.times {
            assert!(
                waveforms.times.contains(&time),
                "{method:?} steps onto {time}"
            );
        }
    }
}

#[test]
fn small_signal_currents_are_the_admittances_times_the_phasors() {
    let mut simulator = simulator(&bench());
    let frequencies = vec![1e3, 1e6, 1e9];
    let response = simulator
        .ac(
            &Ac {
                source: "V1".to_owned(),
                frequencies: frequencies.clone(),
            },
            &Options::default(),
        )
        .unwrap();
    let out = simulator
        .unknowns()
        .iter()
        .position(|u| u.name == "out")
        .unwrap();
    let capacitor = terminal(&simulator, "C1.a");
    for ((frequency, values), currents) in frequencies
        .iter()
        .zip(&response.values)
        .zip(&response.currents)
    {
        let expected = Complex64::new(0.0, std::f64::consts::TAU * frequency * 1e-10) * values[out];
        assert!((currents[capacitor] - expected).norm() <= 1e-12 * expected.norm());
        assert!(imbalance(&simulator, currents) < 1e-9);
    }
}
