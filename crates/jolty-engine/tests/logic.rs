//! The event scheduler on logic circuits, alone and joined to analog circuits, against timing
//! known in closed form.
#![cfg(test)]

use jolty_engine::{
    Bound, Conditions, DcSweep, ElaborationProblem, Endpoint, Options, Simulator, Transient,
    Waveforms, Work,
};
use jolty_ir::Circuit;
use jolty_models::{Level, Library};
use serde_json::{Value, json};

/// A device instance: its id, model, parameters, and the net of each terminal.
type Device<'a> = (&'a str, &'a str, Value, &'a [(&'a str, &'a str)]);

/// A circuit whose top component has a ground port `0` and places `devices`.
fn circuit(devices: &[Device<'_>]) -> Circuit {
    let instances: Vec<Value> = devices
        .iter()
        .map(|(id, model, parameters, terminals)| {
            let terminals: serde_json::Map<String, Value> = terminals
                .iter()
                .map(|(terminal, net)| ((*terminal).to_owned(), json!(net)))
                .collect();
            json!({"id": id, "of": {"device": model}, "parameters": parameters,
                   "terminals": terminals})
        })
        .collect();
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}], "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit file")
}

fn simulator(circuit: &Circuit) -> Simulator {
    Simulator::new(circuit, &Library::standard(), &Conditions::default())
        .expect("an elaborated circuit")
}

/// A cell of one function of `inputs`, given by `table`, with output `y`.
fn cell(inputs: &[&str], table: &str, (rise, fall): (f64, f64), electrical: bool) -> Value {
    let mut parameters = json!({
        "inputs": inputs,
        "outputs": [{"name": "y", "table": table}],
        "timing": {"fixed": {"rise": rise, "fall": fall, "transition": 1e-12}},
    });
    if electrical {
        parameters["electrical"] = json!({"input_capacitances": vec![0.0; inputs.len()]});
    }
    parameters
}

fn source(initial: &str, changes: &[(f64, &str)]) -> Value {
    json!({"initial": initial, "changes": changes, "transition": 1e-12})
}

fn net(simulator: &Simulator, name: &str) -> usize {
    simulator
        .logic_nets()
        .iter()
        .position(|net| net.name == name)
        .unwrap_or_else(|| panic!("no logic net {name}"))
}

/// `waveforms` apart from the work that found them, which depends on what the simulator
/// did before: an operating point it remembered, the pivots it kept.
fn apart_from_work(mut waveforms: Waveforms) -> Waveforms {
    waveforms.statistics.work = Work::default();
    waveforms
}

fn transient(stop: f64) -> Transient {
    Transient {
        stop,
        max_step: None,
        method: jolty_engine::Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    }
}

#[test]
fn logic_settles_at_dc_and_follows_its_sources_after_each_delay() {
    use Level::{High, Low, Unknown};
    let circuit = circuit(&[
        (
            "A",
            "logic_source",
            source("0", &[(1e-9, "1"), (5e-9, "0")]),
            &[("output", "a")],
        ),
        ("B", "logic_source", source("1", &[]), &[("output", "b")]),
        (
            "U1",
            "logic_cell",
            cell(&["a", "b"], "1110", (2e-9, 1e-9), false),
            &[("a", "a"), ("b", "b"), ("y", "n")],
        ),
        (
            "U2",
            "logic_cell",
            cell(&["a"], "10", (2e-9, 1e-9), false),
            &[("a", "n"), ("y", "y")],
        ),
        (
            "U3",
            "logic_cell",
            cell(&["a"], "10", (1e-9, 1e-9), false),
            &[("a", "z"), ("y", "w")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    assert_eq!(simulator.unknowns(), []);
    let point = simulator.operating_point(&Options::default()).unwrap();
    let levels: Vec<(&str, Level)> = simulator
        .logic_nets()
        .iter()
        .map(|net| net.name.as_str())
        .zip(point.logic)
        .collect();
    assert_eq!(
        levels,
        [
            ("a", Low),
            ("b", High),
            ("n", High),
            ("y", Low),
            ("z", Unknown),
            ("w", Unknown)
        ]
    );
    let waveforms = simulator
        .transient(&transient(10e-9), &Options::default())
        .unwrap();
    // The NAND falls a nanosecond after a rises, and the inverter rises two after that.
    assert_eq!(
        waveforms.logic[net(&simulator, "n")],
        [(0.0, High), (2e-9, Low), (7e-9, High)]
    );
    assert_eq!(
        waveforms.logic[net(&simulator, "y")],
        [(0.0, Low), (4e-9, High), (8e-9, Low)]
    );
}

#[test]
fn a_pulse_shorter_than_a_delay_never_passes() {
    let circuit = circuit(&[
        (
            "A",
            "logic_source",
            source("0", &[(1e-9, "1"), (1.5e-9, "0"), (4e-9, "1")]),
            &[("output", "a")],
        ),
        (
            "U1",
            "logic_cell",
            cell(&["a"], "01", (1e-9, 1e-9), false),
            &[("a", "a"), ("y", "y")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let waveforms = simulator
        .transient(&transient(10e-9), &Options::default())
        .unwrap();
    assert_eq!(
        waveforms.logic[net(&simulator, "y")],
        [(0.0, Level::Low), (5e-9, Level::High)]
    );
}

#[test]
fn an_analog_edge_crosses_into_logic_and_back_at_half_the_supply() {
    let (supply, delay) = (2.0, 3e-9);
    let circuit = circuit(&[
        (
            "VDD",
            "voltage_source",
            json!({"waveform": {"dc": supply}}),
            &[("positive", "vdd"), ("negative", "0")],
        ),
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"pulse": {"initial": 0, "pulsed": supply, "delay": 1e-9,
                "rise": 2e-9, "fall": 2e-9, "width": 1}}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "U1",
            "logic_cell",
            cell(&["a"], "10", (delay, delay), true),
            &[
                ("a", "in"),
                ("y", "out"),
                ("supply", "vdd"),
                ("ground", "0"),
            ],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-15}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let names: Vec<&str> = simulator
        .devices()
        .iter()
        .map(jolty_engine::Placed::path)
        .collect();
    assert_eq!(names, ["VDD", "V1", "C1", "U1:a", "U1:y"]);
    let waveforms = simulator
        .transient(&transient(10e-9), &Options::default())
        .unwrap();
    // The input crosses half the supply at 2 ns; the output crosses it a delay later.
    let changes = &waveforms.logic[net(&simulator, "U1:a")];
    assert_eq!(changes.len(), 2);
    assert!((changes[1].0 - 2e-9).abs() < 1e-15, "{changes:?}");
    let out = simulator.unknown("out").unwrap().index();
    let crossing = waveforms
        .times
        .windows(2)
        .zip(waveforms.values.windows(2))
        .find_map(|(times, values)| {
            let (before, after) = (values[0][out] - supply / 2.0, values[1][out] - supply / 2.0);
            (before > 0.0 && after <= 0.0)
                .then(|| times[0] + (times[1] - times[0]) * before / (before - after))
        })
        .expect("the output falls");
    assert!((crossing - (2e-9 + delay)).abs() < 1e-14, "{crossing}");
}

/// An inverter whose output charges its own input through an RC: each half period, the
/// input crosses half the supply, the output follows a delay `d` later, and the input relaxes
/// toward the new level until it crosses again: `d + RC ln(2 − exp(−d/RC))`.
#[test]
fn an_inverter_and_an_rc_oscillate_at_their_closed_form_period() {
    let (resistance, capacitance) = (1e3, 1e-9);
    let tau = resistance * capacitance;
    let delay = tau;
    let circuit = circuit(&[
        (
            "VDD",
            "voltage_source",
            json!({"waveform": {"dc": 1.0}}),
            &[("positive", "vdd"), ("negative", "0")],
        ),
        (
            "U1",
            "logic_cell",
            cell(&["a"], "10", (delay, delay), true),
            &[("a", "c"), ("y", "out"), ("supply", "vdd"), ("ground", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": resistance}),
            &[("a", "out"), ("b", "c")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "c"), ("b", "0")],
        ),
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"piecewise": {"points": [[0, 0], [1e-9, 0], [2e-9, 1], [3e-9, 0]]}}}),
            &[("positive", "kick"), ("negative", "0")],
        ),
        (
            "R2",
            "resistor",
            json!({"resistance": 1e9}),
            &[("a", "kick"), ("b", "c")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    // The capacitor holds a nanocoulomb, so the default charge tolerance, 1e-14 C, would
    // bound the period's accuracy at about 1e-4 whatever the relative tolerance; with it a
    // ten-thousandth as large, the relative tolerance does.
    let options = Options {
        relative_tolerance: 1e-6,
        charge_tolerance: 1e-18,
        ..Options::default()
    };
    let waveforms = simulator
        .transient(&transient(40.0 * tau), &options)
        .unwrap();
    let changes: Vec<f64> = waveforms.logic[net(&simulator, "U1:a")]
        .iter()
        .skip(1)
        .map(|&(time, _)| time)
        .collect();
    let half = delay + tau * (2.0 - (-delay / tau).exp()).ln();
    assert!(changes.len() > 20, "{changes:?}");
    for pair in changes[4..].windows(2) {
        let measured = pair[1] - pair[0];
        assert!(
            (measured - half).abs() < 1e-4 * half,
            "{measured} against {half}"
        );
    }
}

/// A sweep enables a ring of three inverters through a detector: off, the ring settles; on,
/// it would oscillate, so its nets settle unknown.
#[test]
fn a_ring_that_would_oscillate_at_dc_settles_unknown() {
    use Level::{High, Low, Unknown};
    let circuit = circuit(&[
        (
            "VDD",
            "voltage_source",
            json!({"waveform": {"dc": 1.0}}),
            &[("positive", "vdd"), ("negative", "0")],
        ),
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"dc": 0.0}}),
            &[("positive", "enable"), ("negative", "0")],
        ),
        (
            "U1",
            "logic_cell",
            cell(&["a", "b"], "1110", (1e-9, 1e-9), true),
            &[
                ("a", "enable"),
                ("b", "r3"),
                ("y", "r1"),
                ("supply", "vdd"),
                ("ground", "0"),
            ],
        ),
        (
            "U2",
            "logic_cell",
            cell(&["a"], "10", (1e-9, 1e-9), false),
            &[("a", "r1"), ("y", "r2")],
        ),
        (
            "U3",
            "logic_cell",
            cell(&["a"], "10", (1e-9, 1e-9), false),
            &[("a", "r2"), ("y", "r3")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let sweep = DcSweep {
        source: "V1".to_owned(),
        levels: vec![0.0, 1.0],
    };
    let sweep = simulator.dc_sweep(&sweep, &Options::default()).unwrap();
    let ring = ["r1", "r2", "r3"].map(|name| net(&simulator, name));
    assert_eq!(ring.map(|net| sweep.logic[0][net]), [High, Low, High]);
    assert_eq!(
        ring.map(|net| sweep.logic[1][net]),
        [Unknown, Unknown, Unknown]
    );
}

#[test]
fn a_logic_pin_crosses_to_an_electrical_net_only_through_a_supply() {
    let circuit = circuit(&[
        (
            "R1",
            "resistor",
            json!({"resistance": 1.0}),
            &[("a", "n"), ("b", "0")],
        ),
        (
            "U1",
            "logic_cell",
            cell(&["a"], "10", (0.0, 0.0), false),
            &[("a", "n"), ("y", "y")],
        ),
    ]);
    let error = Simulator::new(&circuit, &Library::standard(), &Conditions::default()).unwrap_err();
    assert!(matches!(
        error.problems(),
        [ElaborationProblem::LogicPinOnElectricalNet { pin, net, .. }] if pin == "a" && net == "n"
    ));
    assert_eq!(
        error.to_string(),
        "instance `U1`: joins its logic pin `a` to the electrical net `n`, and a logic_cell \
         has no supply and ground to cross to it"
    );
}

#[test]
fn logic_nets_name_what_drives_and_reads_them() {
    let circuit = circuit(&[
        ("A", "logic_source", source("0", &[]), &[("output", "a")]),
        ("B", "logic_source", source("1", &[]), &[("output", "a")]),
        (
            "U1",
            "logic_cell",
            cell(&["a", "b"], "1110", (0.0, 0.0), false),
            &[("a", "a"), ("b", "floating"), ("y", "y")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let a = &simulator.logic_nets()[net(&simulator, "a")];
    assert_eq!(
        a.sources,
        [
            Endpoint::Pin {
                device: 0,
                index: 0
            },
            Endpoint::Pin {
                device: 1,
                index: 0
            }
        ]
    );
    assert_eq!(
        a.loads,
        [Endpoint::Pin {
            device: 2,
            index: 0
        }]
    );
    // Two sources that disagree leave their net unknown.
    let point = simulator.operating_point(&Options::default()).unwrap();
    assert_eq!(point.logic[net(&simulator, "a")], Level::Unknown);
    assert_eq!(point.logic[net(&simulator, "y")], Level::Unknown);
}

/// A circuit whose top places `X1`, an instance of `cell`: a resistor from its supply to
/// its output, standing for a circuit with ports `a`, `y`, `vdd` and `gnd`.
fn hierarchy() -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "test",
        "components": {
            "test": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                {"id": "VDD", "of": {"device": "voltage_source"},
                 "parameters": {"waveform": {"dc": 1.0}},
                 "terminals": {"positive": "vdd", "negative": "0"}},
                {"id": "A", "of": {"device": "logic_source"},
                 "parameters": {"initial": "1", "transition": 1e-12},
                 "terminals": {"output": "a"}},
                {"id": "X1", "of": {"component": "cell"},
                 "terminals": {"a": "a", "y": "y", "vdd": "vdd", "gnd": "0"}},
                {"id": "C1", "of": {"device": "capacitor"}, "parameters": {"capacitance": 1e-15},
                 "terminals": {"a": "y", "b": "0"}}
            ]},
            "cell": {"ports": [{"name": "a", "role": "input"}, {"name": "y", "role": "output"},
                               {"name": "vdd", "role": "supply"}, {"name": "gnd", "role": "ground"}],
                     "instances": [
                {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
                 "terminals": {"a": "vdd", "b": "y"}},
                {"id": "R2", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
                 "terminals": {"a": "a", "b": "gnd"}}
            ]}
        }
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit file")
}

#[test]
fn a_model_stands_in_for_a_component_instance() {
    let mut inverter = cell(&["a"], "10", (1e-9, 1e-9), true);
    inverter["electrical"]["supply"] = json!("vdd");
    inverter["electrical"]["ground"] = json!("gnd");
    let mut conditions = Conditions::default();
    conditions.substitutes.insert(
        "X1".to_owned(),
        serde_json::from_value(json!({"model": "logic_cell", "parameters": inverter})).unwrap(),
    );
    let mut simulator = Simulator::new(&hierarchy(), &Library::standard(), &conditions).unwrap();
    let paths: Vec<&str> = simulator
        .logic_devices()
        .iter()
        .map(jolty_engine::PlacedLogic::path)
        .collect();
    assert_eq!(paths, ["A", "X1"]);
    let point = simulator.operating_point(&Options::default()).unwrap();
    let y = simulator.unknown("y").unwrap().index();
    assert!(point.values[y].abs() < 1e-12, "{}", point.values[y]);

    conditions.substitutes.insert(
        "X1/R1".to_owned(),
        serde_json::from_value(json!({"model": "resistor", "parameters": {"resistance": 1.0}}))
            .unwrap(),
    );
    let error = Simulator::new(&hierarchy(), &Library::standard(), &conditions).unwrap_err();
    assert_eq!(
        error.to_string(),
        "the circuit has no component instance at `X1/R1`, for which a model stands in"
    );
}

/// An output that turns back before its edge reaches the rail has less to swing, and
/// crosses half the supply sooner: falling at 2 ns with a 2 ns transition, it is three
/// quarters of the way down when the edge back would leave the rail, so it crosses half a
/// nanosecond sooner than from the rail.
#[test]
fn an_output_turning_back_before_its_rail_crosses_sooner() {
    let mut buffer = cell(&["a"], "01", (1e-9, 1e-9), false);
    buffer["timing"]["fixed"]["transition"] = json!(2e-9);
    let circuit = circuit(&[
        (
            "A",
            "logic_source",
            source("1", &[(1e-9, "0"), (2.5e-9, "1")]),
            &[("output", "a")],
        ),
        ("U1", "logic_cell", buffer, &[("a", "a"), ("y", "y")]),
    ]);
    let mut simulator = simulator(&circuit);
    let waveforms = simulator
        .transient(&transient(10e-9), &Options::default())
        .unwrap();
    let changes = &waveforms.logic[net(&simulator, "y")];
    assert_eq!(changes.len(), 3);
    assert!((changes[1].0 - 2e-9).abs() < 1e-18, "{changes:?}");
    assert!((changes[2].0 - 3e-9).abs() < 1e-18, "{changes:?}");
}

/// A logic driver's delay reads the whole capacitance on the net it drives: an analog
/// capacitor's as much as a detector's.
#[test]
fn a_driver_s_load_counts_the_analog_capacitance_it_drives() {
    let inverter = json!({
        "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
        "timing": {"characterized": {
            "slews": [1e-12], "loads": [0, 1e-12],
            "arcs": [
                {"input": "a", "before": "0", "output": "y",
                 "delay": [[1e-9, 3e-9]], "transition": [[1e-12, 1e-12]]},
                {"input": "a", "before": "1", "output": "y",
                 "delay": [[1e-9, 3e-9]], "transition": [[1e-12, 1e-12]]}
            ]}},
        "electrical": {"input_capacitances": [0]},
    });
    let crossing = |load: f64| {
        let circuit = circuit(&[
            (
                "VDD",
                "voltage_source",
                json!({"waveform": {"dc": 1.0}}),
                &[("positive", "vdd"), ("negative", "0")],
            ),
            (
                "A",
                "logic_source",
                source("0", &[(1e-9, "1")]),
                &[("output", "a")],
            ),
            (
                "U1",
                "logic_cell",
                inverter.clone(),
                &[("a", "a"), ("y", "y"), ("supply", "vdd"), ("ground", "0")],
            ),
            (
                "C1",
                "capacitor",
                json!({"capacitance": load}),
                &[("a", "y"), ("b", "0")],
            ),
        ]);
        let mut simulator = simulator(&circuit);
        let waveforms = simulator
            .transient(&transient(10e-9), &Options::default())
            .unwrap();
        waveforms.logic[net(&simulator, "U1:y")][1].0 + 0.5e-12
    };
    assert!((crossing(0.0) - 2e-9).abs() < 1e-15);
    assert!(
        (crossing(0.5e-12) - 3e-9).abs() < 1e-15,
        "{}",
        crossing(0.5e-12)
    );
}

/// A characterized cell records where it is used outside the ranges it was measured over:
/// its supply, its inputs' transitions and its outputs' loads.
#[test]
fn a_cell_used_outside_its_domain_says_where() {
    let inverter = json!({
        "inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
        "timing": {"characterized": {
            "slews": [1e-11, 2e-11], "loads": [0, 1e-14], "supply": 1.0,
            "arcs": [
                {"input": "a", "before": "0", "output": "y",
                 "delay": [[1e-10, 1e-10], [1e-10, 1e-10]],
                 "transition": [[1e-11, 1e-11], [1e-11, 1e-11]]},
                {"input": "a", "before": "1", "output": "y",
                 "delay": [[1e-10, 1e-10], [1e-10, 1e-10]],
                 "transition": [[1e-11, 1e-11], [1e-11, 1e-11]]}
            ]}},
        "electrical": {"input_capacitances": [0]},
    });
    let circuit = circuit(&[
        (
            "VDD",
            "voltage_source",
            json!({"waveform": {"dc": 1.2}}),
            &[("positive", "vdd"), ("negative", "0")],
        ),
        (
            "A",
            "logic_source",
            json!({"initial": "0", "changes": [[1e-9, "1"]], "transition": 1e-9}),
            &[("output", "a")],
        ),
        (
            "U1",
            "logic_cell",
            inverter,
            &[("a", "a"), ("y", "y"), ("supply", "vdd"), ("ground", "0")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-13}),
            &[("a", "y"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    simulator
        .transient(&transient(5e-9), &Options::default())
        .unwrap();
    let found: Vec<(Bound, f64, f64)> = simulator
        .excursions()
        .iter()
        .map(|excursion| (excursion.bound, excursion.value, excursion.time))
        .collect();
    assert_eq!(found.len(), 3, "{found:?}");
    assert_eq!((found[0].0, found[0].2), (Bound::Transition, 1e-9));
    assert!((found[0].1 - 1e-9).abs() < 1e-24);
    assert_eq!(found[1].0, Bound::Load);
    assert!((found[1].1 - 1e-13).abs() < 1e-18, "{found:?}");
    assert_eq!((found[2].0, found[2].2), (Bound::Supply, 0.0));
    assert!((found[2].1 - 1.2).abs() < 1e-12);
}

#[test]
fn the_logic_returns_to_its_operating_point_after_a_transient() {
    // The transient moves the logic on and its drivers with it; the next analysis about the
    // operating point takes it again, and the logic and the drivers as it left them.
    let supply = 2.0;
    let circuit = circuit(&[
        (
            "VDD",
            "voltage_source",
            json!({"waveform": {"dc": supply}}),
            &[("positive", "vdd"), ("negative", "0")],
        ),
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"pulse": {"initial": 0, "pulsed": supply, "delay": 1e-9,
                "rise": 2e-9, "fall": 2e-9, "width": 1}}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "U1",
            "logic_cell",
            cell(&["a"], "10", (3e-9, 3e-9), true),
            &[
                ("a", "in"),
                ("y", "out"),
                ("supply", "vdd"),
                ("ground", "0"),
            ],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-15}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let options = Options::default();
    let mut simulator = simulator(&circuit);
    let point = simulator.operating_point(&options).unwrap();
    let first = apart_from_work(simulator.transient(&transient(10e-9), &options).unwrap());
    assert_eq!(simulator.operating_point(&options).unwrap(), point);
    let second = simulator.transient(&transient(10e-9), &options).unwrap();
    assert_eq!(apart_from_work(second), first);
    assert_eq!(simulator.operating_point_solves(), 1);
    let mut fresh = self::simulator(&circuit);
    let afresh = fresh.transient(&transient(10e-9), &options).unwrap();
    assert_eq!(apart_from_work(afresh), first);
    assert_eq!(simulator.excursions(), fresh.excursions());
}
