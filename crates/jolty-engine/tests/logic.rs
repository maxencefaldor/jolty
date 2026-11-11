//! The event scheduler on logic circuits, alone and joined to analog circuits, against timing
//! known in closed form.

use jolty_engine::{
    Conditions, DcSweep, ElaborationProblem, Endpoint, Options, Simulator, Transient,
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

fn transient(stop: f64) -> Transient {
    Transient {
        stop,
        max_step: None,
        method: jolty_engine::Method::Trapezoidal,
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
    assert!(simulator.unknowns().is_empty());
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
    let options = Options {
        relative_tolerance: 1e-6,
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
        "instance `U1` joins its logic pin `a` to the electrical net `n`, and a logic_cell has \
         no supply and ground to cross to it"
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
