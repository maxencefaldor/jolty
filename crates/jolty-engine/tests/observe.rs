//! What every device does: the current into every terminal and the charge it holds, the
//! derivatives of its equations, and its variables.

use jolty_engine::{
    Ac, Conditions, DcSweep, Method, Observation, Observed, Options, Simulator, Transient,
};
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

/// Checks that `found` is `expected` exactly.
fn assert_exact(found: f64, expected: f64) {
    assert!(
        found.to_bits() == expected.to_bits() || found.abs() + expected.abs() <= 0.0,
        "{found} is not {expected}"
    );
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
    let Observation {
        currents, charges, ..
    } = &point.observation;
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
    let currents = &point.observation.currents;
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
    let currents = sweep.observations.point(0).currents;
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
            periodic: false,
        };
        let waveforms = simulator.transient(&transient, &tight()).unwrap();
        let out = simulator
            .unknowns()
            .iter()
            .position(|u| u.name == "out")
            .unwrap();
        let (capacitor, resistor) = (terminal(&simulator, "C1.a"), terminal(&simulator, "R1.b"));
        let mut charging = 0.0_f64;
        for (values, found) in waveforms.values.iter().zip(waveforms.observations.iter()) {
            let off = imbalance(&simulator, found.currents);
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

/// The index of the derivative of `equation` with respect to `unknown`.
fn derivative(simulator: &Simulator, equation: &str, unknown: &str) -> usize {
    simulator
        .derivatives()
        .iter()
        .position(|(row, column)| row == equation && column == unknown)
        .unwrap_or_else(|| panic!("no derivative of {equation} by {unknown}"))
}

#[test]
fn each_device_reports_its_own_small_signal_model_and_variables() {
    let mut simulator = simulator(&json!([
        {"id": "VG", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
         "terminals": {"positive": "g", "negative": "0"}},
        {"id": "VD", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 2.0}},
         "terminals": {"positive": "d", "negative": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 4e3},
         "terminals": {"a": "d", "b": "0"}},
        {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 3e-12},
         "terminals": {"a": "g", "b": "d"}},
        {"id": "M1", "of": {"device": "mosfet"},
         "parameters": {"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.3,
                        "transconductance_parameter": 1e-4},
         "terminals": {"drain": "d", "gate": "g", "source": "0", "bulk": "0"}}
    ]));
    let point = simulator.operating_point(&tight()).unwrap();
    let Observation {
        conductances,
        capacitances,
        variables,
        ..
    } = &point.observation;
    let conductance = |row: &str, column: &str| conductances[derivative(&simulator, row, column)];
    let capacitance = |row: &str, column: &str| capacitances[derivative(&simulator, row, column)];
    assert_exact(conductance("R1.a", "R1.a"), 2.5e-4);
    assert_exact(conductance("R1.a", "R1.b"), -2.5e-4);
    assert_exact(capacitance("C1.b", "C1.a"), -3e-12);
    // A source's branch equation holds its voltage: v(positive) − v(negative) = level.
    assert_exact(conductance("VG:current", "VG.positive"), 1.0);
    assert_exact(conductance("VG.positive", "VG:current"), 1.0);
    // In saturation, gm = β (vgs − vto), with β = kp W / L.
    let gm = conductance("M1.drain", "M1.gate");
    assert!((gm - 2e-4 * 0.7).abs() < 1e-18, "{gm}");
    assert_exact(conductance("M1.gate", "M1.drain"), 0.0);
    let names: Vec<(&str, &str)> = simulator
        .variables()
        .iter()
        .map(|(device, name)| (device.as_str(), *name))
        .collect();
    assert_eq!(
        names,
        [
            ("M1", "threshold_voltage"),
            ("M1", "saturation_voltage"),
            ("M1", "gate_source_capacitance"),
            ("M1", "gate_drain_capacitance"),
            ("M1", "gate_bulk_capacitance"),
        ]
    );
    assert!((variables[0] - 0.3).abs() < 1e-15);
    assert!((variables[1] - 0.7).abs() < 1e-15);
}

/// The circuit of a cell, `R1` from `a` to `mid` and `R2` from `mid` to `b`, nested in a
/// pair whose ports `p` and `q` are the cell's, with `in` driven by `V1`.
fn nested(outside: (&str, &str)) -> Simulator {
    let resistor = |id: &str, a: &str, b: &str| {
        json!({"id": id, "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
               "terminals": {"a": a, "b": b}})
    };
    let signal = |name: &str| json!({"name": name, "role": "signal"});
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {
            "test": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                {"id": "V1", "of": {"device": "voltage_source"},
                 "parameters": {"waveform": {"dc": 3.0}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"id": "X1", "of": {"component": "pair"},
                 "terminals": {"p": outside.0, "q": outside.1}},
                resistor("R3", "in", "0")
            ]},
            "pair": {"ports": [signal("p"), signal("q")], "instances": [
                {"id": "X2", "of": {"component": "cell"}, "terminals": {"a": "p", "b": "q"}},
                resistor("R4", "p", "q")
            ]},
            "cell": {"ports": [signal("a"), signal("b")], "instances": [
                resistor("R1", "a", "mid"), resistor("R2", "mid", "b")
            ]}
        }
    });
    let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
    Simulator::new(&circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit")
}

#[test]
fn a_component_instance_s_terminals_sum_the_device_terminals_inside_it() {
    let mut simulator = nested(("in", "0"));
    let named: Vec<(&str, Vec<&str>)> = simulator
        .composites()
        .iter()
        .map(|(name, members)| {
            let members = members
                .iter()
                .map(|&index| simulator.terminals()[index].as_str())
                .collect();
            (name.as_str(), members)
        })
        .collect();
    assert_eq!(
        named,
        [
            ("X1.p", vec!["X1/X2/R1.a", "X1/R4.a"]),
            ("X1.q", vec!["X1/X2/R2.b", "X1/R4.b"]),
            ("X1/X2.a", vec!["X1/X2/R1.a"]),
            ("X1/X2.b", vec!["X1/X2/R2.b"]),
        ]
    );
    let point = simulator.operating_point(&tight()).unwrap();
    let current = |index: usize| -> f64 {
        simulator.composites()[index]
            .1
            .iter()
            .map(|&terminal| point.observation.currents[terminal])
            .sum()
    };
    // Into X1: 3 V across 2 kΩ in parallel with 1 kΩ.
    assert!((current(0) - 4.5e-3).abs() < 1e-12, "{}", current(0));
    assert!((current(1) + 4.5e-3).abs() < 1e-12);
    assert!((current(2) - 1.5e-3).abs() < 1e-12);

    // Ports tied together outside keep their own currents: here both join `in`.
    let mut tied = nested(("in", "in"));
    let point = tied.operating_point(&tight()).unwrap();
    let currents: Vec<f64> = tied
        .composites()
        .iter()
        .map(|(_, members)| members.iter().map(|&t| point.observation.currents[t]).sum())
        .collect();
    assert!(currents.iter().all(|current| current.abs() < 1e-15));
}

#[test]
fn a_logic_device_s_pins_carry_its_connect_elements_currents() {
    let cell = json!({
        "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
        "timing": {"fixed": {"rise": 1e-9, "fall": 1e-9, "transition": 1e-10}},
        "electrical": {"input_capacitances": [2e-15]},
    });
    let mut simulator = simulator(&json!([
        {"id": "VDD", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
         "terminals": {"positive": "vdd", "negative": "0"}},
        {"id": "VA", "of": {"device": "voltage_source"},
         "parameters": {"waveform": {"pulse": {"initial": 0, "pulsed": 1, "delay": 2e-9,
                                               "rise": 1e-10, "fall": 1e-10, "width": 1e-8}}},
         "terminals": {"positive": "a", "negative": "0"}},
        {"id": "U1", "of": {"device": "logic_cell"}, "parameters": cell,
         "terminals": {"a": "a", "y": "y", "supply": "vdd", "ground": "0"}},
        {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "y", "b": "0"}}
    ]));
    let named: Vec<&str> = simulator
        .composites()
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(named, ["U1.a", "U1.y", "U1.supply", "U1.ground"]);
    let transient = Transient {
        stop: 1e-8,
        max_step: Some(1e-10),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let waveforms = simulator.transient(&transient, &tight()).unwrap();
    let pin = |index: usize, found: Observed<'_>| -> f64 {
        simulator.composites()[index]
            .1
            .iter()
            .map(|&terminal| found.currents[terminal])
            .sum()
    };
    let fraction = derivative(&simulator, "U1:y:current", "U1:y.supply");
    let resistor = terminal(&simulator, "R1.a");
    let (mut high, mut low) = (false, false);
    for (time, found) in waveforms.times.iter().zip(waveforms.observations.iter()) {
        // The driver takes the resistor's current from its output to the ground.
        assert!((pin(1, found) + found.currents[resistor]).abs() < 1e-12);
        let sum: f64 = (0..4).map(|index| pin(index, found)).sum();
        assert!(sum.abs() < 1e-15, "the cell's currents sum to zero: {sum}");
        // An ideal driver draws nothing from its supply.
        assert_exact(pin(2, found), 0.0);
        // Its equation holds the output at the fraction of the supply it ramps to.
        let held = -found.conductances[fraction];
        if *time < 2e-9 {
            high |= held.to_bits() == 1.0_f64.to_bits();
        } else if *time > 4e-9 {
            low |= held.abs() <= 0.0;
        }
        assert!((0.0..=1.0).contains(&held), "{held}");
    }
    assert!(high && low, "the inverter falls after its input rises");
}
