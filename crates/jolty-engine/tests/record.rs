//! What the recorded devices do: the current into every terminal and the charge it holds,
//! the derivatives of its equations, and its variables, for the devices and the component
//! instances a record names.
#![cfg(test)]

use std::collections::BTreeSet;

use jolty_engine::{
    Ac, Composite, Conditions, DcSweep, Method, Options, Record, RecordedValues, RecordedView,
    Recording, Simulator, Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::{Value, json};

fn simulator(instances: &Value) -> Simulator {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
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
        {"name": "V1", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"pulse": {"initial": 0, "pulsed": 3, "delay": 1e-7,
                                               "rise": 1e-8, "fall": 1e-8, "width": 1e-6}}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-10},
         "terminals": {"a": "out", "b": "0"}},
        {"name": "D1", "of": {"model": "diode"},
         "parameters": {"saturation_current": 1e-14, "junction_capacitance": 1e-12},
         "terminals": {"anode": "out", "cathode": "0"}},
        {"name": "M1", "of": {"model": "mosfet"},
         "parameters": {"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.3,
                        "transconductance_parameter": 1e-4},
         "terminals": {"drain": "out", "gate": "out", "source": "0", "bulk": "0"}}
    ])
}

/// The record of the instances at `paths`.
fn instances(paths: &[&str]) -> Record {
    Record::Instances(
        paths
            .iter()
            .map(|&path| path.to_owned())
            .collect::<BTreeSet<_>>(),
    )
}

/// Every device recorded, at SPICE's tolerances.
fn everything() -> Options {
    Options {
        record: Record::All,
        ..Options::default()
    }
}

/// Every device recorded, at tolerances so tight that what Newton iteration leaves of each
/// node's equation, the sum of the currents into it, is all but nothing.
fn tight() -> Options {
    Options {
        relative_tolerance: 1e-9,
        current_tolerance: 1e-18,
        record: Record::All,
        ..Options::default()
    }
}

/// The names of the values every observation holds under `record`, as [`Recording`] lays
/// them out: each terminal of each recorded device, such as `R1.a`, then each recorded
/// composite terminal; each pair of each recorded device's unknowns, its terminals then its
/// internal unknowns, row by row; and each recorded device's variables.
#[derive(Debug, Default)]
struct Columns {
    terminals: Vec<String>,
    derivatives: Vec<(String, String)>,
    variables: Vec<(String, &'static str)>,
}

impl Columns {
    fn of(simulator: &Simulator, record: &Record) -> Self {
        let recorded = simulator.recorded(record);
        let mut columns = Self::default();
        for &device in &recorded.devices {
            let placed = &simulator.devices()[device];
            let local: Vec<String> = placed
                .terminals()
                .iter()
                .map(|terminal| format!("{}.{terminal}", placed.path()))
                .chain(
                    placed
                        .internals()
                        .map(|index| simulator.unknowns()[index].name.clone()),
                )
                .collect();
            columns
                .terminals
                .extend(local[..placed.terminals().len()].iter().cloned());
            for row in &local {
                for column in &local {
                    columns.derivatives.push((row.clone(), column.clone()));
                }
            }
            columns.variables.extend(
                placed
                    .variable_names()
                    .iter()
                    .map(|&name| (placed.path().to_owned(), name)),
            );
        }
        columns.terminals.extend(
            recorded
                .composites
                .iter()
                .map(|&composite| simulator.composites()[composite].name().to_owned()),
        );
        columns
    }

    /// The index of the terminal `name`.
    fn terminal(&self, name: &str) -> usize {
        self.terminals
            .iter()
            .position(|found| found == name)
            .unwrap_or_else(|| panic!("no terminal {name}"))
    }

    /// The index of the derivative of `equation` with respect to `unknown`.
    fn derivative(&self, equation: &str, unknown: &str) -> usize {
        self.derivatives
            .iter()
            .position(|(row, column)| row == equation && column == unknown)
            .unwrap_or_else(|| panic!("no derivative of {equation} by {unknown}"))
    }
}

/// The index of `name` among the terminals every device's record holds.
fn terminal(simulator: &Simulator, name: &str) -> usize {
    Columns::of(simulator, &Record::All).terminal(name)
}

/// On each node, the sum of some values at the device terminals on it, the largest of them
/// in magnitude, and the sum of their magnitudes.
type PerNode = Vec<(Complex64, f64, f64)>;

/// [`PerNode`] of `values`, as every device's record holds them.
fn per_node<T: Copy + Into<Complex64>>(simulator: &Simulator, values: &[T]) -> PerNode {
    let mut nodes = vec![(Complex64::default(), 0.0_f64, 0.0_f64); simulator.unknowns().len()];
    let mut index = 0;
    for placed in simulator.devices() {
        for node in placed.nodes() {
            let value: Complex64 = values[index].into();
            if let Some(node) = node {
                let (sum, largest, magnitudes) = &mut nodes[node.index()];
                *sum += value;
                *largest = largest.max(value.norm());
                *magnitudes += value.norm();
            }
            index += 1;
        }
    }
    assert_eq!(
        index + simulator.composites().len(),
        values.len(),
        "a value per terminal"
    );
    nodes
}

/// The sum of the currents into the device terminals on each node, which Kirchhoff's
/// current law makes zero, relative to the largest of them, from `currents` as every
/// device's record holds them.
fn imbalance<T: Copy + Into<Complex64>>(simulator: &Simulator, currents: &[T]) -> f64 {
    per_node(simulator, currents)
        .iter()
        .filter(|(_, largest, _)| *largest > 0.0)
        .map(|(sum, largest, _)| sum.norm() / largest)
        .fold(0.0, f64::max)
}

/// Checks that `found` is `expected` exactly.
fn assert_exact(found: f64, expected: f64) {
    assert!(
        found.to_bits() == expected.to_bits() || found.abs() + expected.abs() <= 0.0,
        "{found} is not {expected}"
    );
}

#[test]
fn every_terminal_is_named_by_its_device_and_its_own_name() {
    let simulator = simulator(&bench());
    assert_eq!(
        Columns::of(&simulator, &Record::All).terminals,
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
        {"name": "V1", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 5.0}},
         "terminals": {"positive": "in", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "in", "b": "out"}},
        {"name": "R2", "of": {"model": "resistor"}, "parameters": {"resistance": 4e3},
         "terminals": {"a": "out", "b": "0"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 1e-9},
         "terminals": {"a": "out", "b": "0"}}
    ]));
    let point = simulator.operating_point(&everything()).unwrap();
    let RecordedValues {
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
    let point = simulator.operating_point(&everything()).unwrap();
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
                parameter: None,
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
            breakpoints: vec![1.234_567e-7, 1.5e-6],
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
        // Each node's currents sum to zero within Newton's tolerance on its equation, 1e-9 of
        // its largest term, ten times over for the rounding of the terms it sums. The
        // trapezoidal formula carries each charge's derivative on from step to step
        // undamped, and the engine takes each node's from the node's charge, the rounded sum
        // of its terminals', where the record takes each terminal's: the two part by the
        // rounding of each step's `α q_{n+1} + β₀ q_n + γ q̇_n`, four roundings in each at
        // most, of terms no larger than `2/h` times the charges and the derivative before,
        // which here is no larger than twice the currents into the node, as every charge on
        // it is a capacitance at its voltage, the capacitor's the largest; and the formula
        // carries that rounding on.
        let mut carried = vec![0.0_f64; simulator.unknowns().len()];
        let mut last: Option<(f64, PerNode, PerNode)> = None;
        let points = waveforms.times.iter().zip(&waveforms.values);
        for ((&time, values), found) in points.zip(waveforms.observations.iter()) {
            let (currents, charges) = (
                per_node(&simulator, found.currents),
                per_node(&simulator, found.charges),
            );
            if let (Method::Trapezoidal, Some((before, held, flowing))) = (method, &last) {
                let alpha = 2.0 / (time - before);
                for (node, carried) in carried.iter_mut().enumerate() {
                    let terms = alpha * (charges[node].2 + held[node].2) + 2.0 * flowing[node].2;
                    *carried += 8.0 * f64::EPSILON * terms;
                }
            }
            for ((sum, largest, _), carried) in currents.iter().zip(&carried) {
                assert!(
                    sum.norm() <= 1e-8 * largest + carried,
                    "{method:?} at {time:e} s: {:e} A of {largest:e} A, {carried:e} A carried",
                    sum.norm()
                );
            }
            last = Some((time, charges, currents));
            assert!((found.charges[capacitor] - 1e-10 * values[out]).abs() < 1e-22);
            charging = charging.max(found.currents[capacitor]);
            assert!(found.currents[resistor] <= 3.0 / 1e3);
        }
        assert!(charging > 1e-4, "the capacitor charges: {charging}");
        for time in transient.breakpoints {
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
            &everything(),
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

#[test]
fn each_device_reports_its_own_small_signal_model_and_variables() {
    let mut simulator = simulator(&json!([
        {"name": "VG", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
         "terminals": {"positive": "g", "negative": "0"}},
        {"name": "VD", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 2.0}},
         "terminals": {"positive": "d", "negative": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 4e3},
         "terminals": {"a": "d", "b": "0"}},
        {"name": "C1", "of": {"model": "capacitor"}, "parameters": {"capacitance": 3e-12},
         "terminals": {"a": "g", "b": "d"}},
        {"name": "M1", "of": {"model": "mosfet"},
         "parameters": {"width": 2e-6, "length": 1e-6, "threshold_voltage": 0.3,
                        "transconductance_parameter": 1e-4},
         "terminals": {"drain": "d", "gate": "g", "source": "0", "bulk": "0"}}
    ]));
    let point = simulator.operating_point(&tight()).unwrap();
    let RecordedValues {
        conductances,
        capacitances,
        variables,
        ..
    } = &point.observation;
    let columns = Columns::of(&simulator, &Record::All);
    let conductance = |row: &str, column: &str| conductances[columns.derivative(row, column)];
    let capacitance = |row: &str, column: &str| capacitances[columns.derivative(row, column)];
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
    let names: Vec<(&str, &str)> = columns
        .variables
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
            ("M1", "region"),
            ("M1", "overdrive_voltage"),
            ("M1", "saturation_margin"),
        ]
    );
    assert!((variables[0] - 0.3).abs() < 1e-15);
    assert!((variables[1] - 0.7).abs() < 1e-15);
    // Its overdrive is its saturation voltage, which it lies past in saturation.
    assert!((variables[6] - 0.7).abs() < 1e-15);
    assert!(variables[7] > 0.0, "{}", variables[7]);
}

/// The circuit of a cell, `R1` from `a` to `mid` and `R2` from `mid` to `b`, nested in a
/// pair whose ports `p` and `q` are the cell's, with `in` driven by `V1`.
fn nested(outside: (&str, &str)) -> Simulator {
    let resistor = |id: &str, a: &str, b: &str| {
        json!({"name": id, "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
               "terminals": {"a": a, "b": b}})
    };
    let signal = |name: &str| json!({"name": name, "role": "signal"});
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {
            "test": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": {"dc": 3.0}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"name": "X1", "of": {"component": "pair"},
                 "terminals": {"p": outside.0, "q": outside.1}},
                resistor("R3", "in", "0")
            ]},
            "pair": {"ports": [signal("p"), signal("q")], "instances": [
                {"name": "X2", "of": {"component": "cell"}, "terminals": {"a": "p", "b": "q"}},
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

/// Each composite terminal's name, with the device terminals it sums, by name.
fn composites(simulator: &Simulator) -> Vec<(&str, Vec<String>)> {
    simulator
        .composites()
        .iter()
        .map(|composite| {
            let members = composite
                .members()
                .iter()
                .map(|&(device, terminal)| {
                    let placed = &simulator.devices()[device];
                    format!("{}.{}", placed.path(), placed.terminals()[terminal])
                })
                .collect();
            (composite.name(), members)
        })
        .collect()
}

#[test]
fn a_component_instance_s_terminals_sum_the_device_terminals_inside_it() {
    let mut simulator = nested(("in", "0"));
    assert_eq!(
        composites(&simulator),
        [
            ("X1.p", vec!["X1/X2/R1.a".to_owned(), "X1/R4.a".to_owned()]),
            ("X1.q", vec!["X1/X2/R2.b".to_owned(), "X1/R4.b".to_owned()]),
            ("X1/X2.a", vec!["X1/X2/R1.a".to_owned()]),
            ("X1/X2.b", vec!["X1/X2/R2.b".to_owned()]),
        ]
    );
    let owners: Vec<&str> = simulator
        .composites()
        .iter()
        .map(Composite::owner)
        .collect();
    assert_eq!(owners, ["X1", "X1", "X1/X2", "X1/X2"]);
    let columns = Columns::of(&simulator, &Record::All);
    let point = simulator.operating_point(&tight()).unwrap();
    let currents = &point.observation.currents;
    let current = |name: &str| currents[columns.terminal(name)];
    // Into X1: 3 V across 2 kΩ in parallel with 1 kΩ.
    assert!(
        (current("X1.p") - 4.5e-3).abs() < 1e-12,
        "{}",
        current("X1.p")
    );
    assert!((current("X1.q") + 4.5e-3).abs() < 1e-12);
    assert!((current("X1/X2.a") - 1.5e-3).abs() < 1e-12);
    for (name, members) in composites(&simulator) {
        let summed: f64 = members.iter().map(|member| current(member)).sum();
        assert_exact(current(name), summed);
    }

    // Ports tied together outside keep their own currents: here both join `in`.
    let mut tied = nested(("in", "in"));
    let point = tied.operating_point(&tight()).unwrap();
    let ports = &point.observation.currents[point.observation.currents.len() - 4..];
    assert!(ports.iter().all(|current| current.abs() < 1e-15));
}

#[test]
fn a_component_instance_recorded_alone_keeps_its_terminals_and_none_of_its_devices() {
    let mut simulator = nested(("in", "0"));
    let everything = simulator.operating_point(&tight()).unwrap();
    let all = Columns::of(&simulator, &Record::All);
    let record = instances(&["X1"]);
    let recorded = simulator.recorded(&record);
    assert_eq!(recorded.devices, Vec::<usize>::new());
    assert_eq!(recorded.composites, [0, 1]);
    let options = Options {
        record: record.clone(),
        ..tight()
    };
    let alone = simulator.operating_point(&options).unwrap();
    let RecordedValues {
        currents,
        charges,
        conductances,
        capacitances,
        variables,
    } = &alone.observation;
    assert!(conductances.is_empty() && capacitances.is_empty() && variables.is_empty());
    assert_eq!(Columns::of(&simulator, &record).terminals, ["X1.p", "X1.q"]);
    for (index, name) in ["X1.p", "X1.q"].into_iter().enumerate() {
        let column = all.terminal(name);
        assert_exact(currents[index], everything.observation.currents[column]);
        assert_exact(charges[index], everything.observation.charges[column]);
    }
}

#[test]
fn a_logic_device_s_pins_carry_its_connect_elements_currents() {
    let cell = json!({
        "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
        "timing": {"fixed": {"rise": 1e-9, "fall": 1e-9, "transition": 1e-10}},
        "electrical": {"input_capacitances": [2e-15]},
    });
    let mut simulator = simulator(&json!([
        {"name": "VDD", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
         "terminals": {"positive": "vdd", "negative": "0"}},
        {"name": "VA", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"pulse": {"initial": 0, "pulsed": 1, "delay": 2e-9,
                                               "rise": 1e-10, "fall": 1e-10, "width": 1e-8}}},
         "terminals": {"positive": "a", "negative": "0"}},
        {"name": "U1", "of": {"model": "logic_cell"}, "parameters": cell,
         "terminals": {"a": "a", "y": "y", "supply": "vdd", "ground": "0"}},
        {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
         "terminals": {"a": "y", "b": "0"}}
    ]));
    let named: Vec<&str> = simulator.composites().iter().map(Composite::name).collect();
    assert_eq!(named, ["U1.a", "U1.y", "U1.supply", "U1.ground"]);
    let transient = Transient {
        stop: 1e-8,
        max_step: Some(1e-10),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        periodic: false,
    };
    let waveforms = simulator.transient(&transient, &tight()).unwrap();
    let columns = Columns::of(&simulator, &Record::All);
    let pins: Vec<usize> = ["U1.a", "U1.y", "U1.supply", "U1.ground"]
        .into_iter()
        .map(|name| columns.terminal(name))
        .collect();
    let pin = |index: usize, found: RecordedView<'_>| -> f64 { found.currents[pins[index]] };
    let fraction = columns.derivative("U1:y:current", "U1:y.supply");
    let resistor = columns.terminal("R1.a");
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

/// A transient of the bench's pulse, with steps at most 10 ns long.
fn pulse() -> Transient {
    Transient {
        stop: 1e-6,
        max_step: Some(1e-8),
        method: Method::Trapezoidal,
        breakpoints: vec![2.5e-7],
        periodic: false,
    }
}

#[test]
fn nothing_beyond_the_unknowns_is_recorded_by_default() {
    let mut simulator = simulator(&bench());
    assert_eq!(simulator.recorded(&Record::default()), Recording::default());
    let point = simulator.operating_point(&Options::default()).unwrap();
    assert_eq!(point.observation, RecordedValues::default());
    let waveforms = simulator.transient(&pulse(), &Options::default()).unwrap();
    let observations = &waveforms.observations;
    assert_eq!(observations.points, waveforms.times.len());
    assert!(observations.currents.is_empty() && observations.conductances.is_empty());
    let ac = Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e3, 1e6],
    };
    let response = simulator.ac(&ac, &Options::default()).unwrap();
    assert!(response.currents.iter().all(Vec::is_empty));
}

#[test]
fn a_device_recorded_alone_reads_as_it_does_among_every_device() {
    let record = instances(&["C1", "M1", "R9"]);
    let mut simulator = simulator(&bench());
    assert_eq!(simulator.recorded(&record).devices, [2, 4]);
    let all = Columns::of(&simulator, &Record::All);
    let some = Columns::of(&simulator, &record);
    assert_eq!(
        some.terminals,
        [
            "C1.a",
            "C1.b",
            "M1.drain",
            "M1.gate",
            "M1.source",
            "M1.bulk"
        ]
    );
    let chosen = Options { record, ..tight() };
    // Each simulator reads only the devices it records.
    let mut apart = self::simulator(&bench());
    let every = simulator.transient(&pulse(), &tight()).unwrap();
    let alone = apart.transient(&pulse(), &chosen).unwrap();
    assert_eq!(alone.times, every.times);
    assert_eq!(alone.values, every.values);
    for (found, expected) in alone.observations.iter().zip(every.observations.iter()) {
        for (index, name) in some.terminals.iter().enumerate() {
            let column = all.terminal(name);
            assert_exact(found.currents[index], expected.currents[column]);
            assert_exact(found.charges[index], expected.charges[column]);
        }
        for (index, (row, column)) in some.derivatives.iter().enumerate() {
            let at = all.derivative(row, column);
            assert_exact(found.conductances[index], expected.conductances[at]);
            assert_exact(found.capacitances[index], expected.capacitances[at]);
        }
        let variables = &expected.variables[expected.variables.len() - 8..];
        assert_eq!(found.variables, variables);
    }
    let ac = Ac {
        source: "V1".to_owned(),
        frequencies: vec![1e3, 1e6, 1e9],
    };
    let every = simulator.ac(&ac, &tight()).unwrap();
    let alone = apart.ac(&ac, &chosen).unwrap();
    for (found, expected) in alone.currents.iter().zip(&every.currents) {
        for (index, name) in some.terminals.iter().enumerate() {
            assert_eq!(found[index], expected[all.terminal(name)]);
        }
    }
}

#[test]
fn a_transient_holds_what_it_records_against_its_memory() {
    let mut simulator = simulator(&bench());
    // Four thousand numbers: a hundred points of every device's hundred-odd numbers
    // overflow them, and a thousand points of the three unknowns alone fit.
    let max_memory = 4000.0 / 134_217_728.0;
    let every = Options {
        max_memory,
        record: Record::All,
        ..Options::default()
    };
    let error = simulator.transient(&pulse(), &every).unwrap_err();
    assert!(
        matches!(error, jolty_engine::SimulationError::TooManyPoints { time, .. } if time == 0.0),
        "{error}"
    );
    let unknowns = Options {
        max_memory,
        ..Options::default()
    };
    let waveforms = simulator.transient(&pulse(), &unknowns).unwrap();
    assert!(waveforms.times.len() > 100);
}
