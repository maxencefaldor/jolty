//! Every analysis against circuits whose answers are known in closed form.

use std::f64::consts::{PI, TAU};

use jolty_engine::{
    Ac, Analysis, Conditions, DcSweep, Method, Options, Outcome, Rejection, SimulationError,
    Simulator, Strategy, Transient,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use jolty_numerics::Complex64;
use serde_json::{Value, json};

/// The thermal voltage at 27 °C.
const THERMAL_VOLTAGE: f64 = 1.380_649e-23 * 300.15 / 1.602_176_634e-19;

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
            json!({
                "id": id,
                "of": {"device": model},
                "parameters": parameters,
                "terminals": terminals,
            })
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

fn index(simulator: &Simulator, name: &str) -> usize {
    simulator
        .unknown(name)
        .unwrap_or_else(|| panic!("no unknown {name}"))
        .index()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance * expected.abs().max(1e-12),
        "{actual} differs from {expected} by more than {tolerance} relative"
    );
}

fn dc(level: f64) -> Value {
    json!({"waveform": {"dc": level}})
}

#[test]
fn a_micro_ohm_beside_a_node_only_gmin_holds_keeps_it() {
    // An idle source feeds 1 µΩ into a diode reverse-biased from 2 V: only the diode's gmin
    // holds the nodes, a millionth of a millionth of the resistor's conductance.
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(2.0),
            &[("positive", "supply"), ("negative", "0")],
        ),
        (
            "I1",
            "current_source",
            dc(0.0),
            &[("positive", "0"), ("negative", "a")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e-6}),
            &[("a", "a"), ("b", "b")],
        ),
        (
            "D1",
            "diode",
            json!({}),
            &[("anode", "b"), ("cathode", "supply")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_eq!(solution.strategy, Strategy::Direct);
    for node in ["a", "b"] {
        assert_close(solution.values[index(&simulator, node)], 2.0, 1e-9);
    }
}

#[test]
fn a_divider_halves_its_input_and_its_source_delivers_current() {
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(5.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e3}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "R2",
            "resistor",
            json!({"resistance": 1e3}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_eq!(solution.strategy, Strategy::Direct);
    assert_close(solution.values[index(&simulator, "out")], 2.5, 1e-12);
    assert_close(
        solution.values[index(&simulator, "V1:current")],
        -2.5e-3,
        1e-12,
    );
}

#[test]
fn a_milliohm_between_high_voltages_converges_to_tight_tolerances() {
    // One unit in the last place of 400 V moves the current through the milliohm by 57 pA
    // in 400 µA: far more than the relative tolerance, so only a residual that allows for
    // the voltages' rounding can converge.
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(400.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e-3}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "R2",
            "resistor",
            json!({"resistance": 1e6}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let options = Options {
        relative_tolerance: 1e-12,
        ..Options::default()
    };
    let solution = simulator
        .operating_point(&options)
        .expect("an operating point");
    assert_eq!(solution.strategy, Strategy::Direct);
    let out = solution.values[index(&simulator, "out")];
    assert_close(out, 400.0 * 1e6 / (1e6 + 1e-3), 1e-15);
}

#[test]
fn a_current_source_pushes_current_into_its_negative_terminal() {
    let circuit = circuit(&[
        (
            "I1",
            "current_source",
            dc(1e-3),
            &[("positive", "0"), ("negative", "out")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 2e3}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_close(solution.values[index(&simulator, "out")], 2.0, 1e-12);
}

#[test]
fn controlled_sources_amplify_their_control_voltage() {
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(0.1),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "E1",
            "voltage_controlled_voltage_source",
            json!({"gain": 10}),
            &[
                ("positive", "amplified"),
                ("negative", "0"),
                ("control_positive", "in"),
                ("control_negative", "0"),
            ],
        ),
        (
            "G1",
            "voltage_controlled_current_source",
            json!({"transconductance": 1e-3}),
            &[
                ("positive", "0"),
                ("negative", "driven"),
                ("control_positive", "in"),
                ("control_negative", "0"),
            ],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e4}),
            &[("a", "driven"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_close(solution.values[index(&simulator, "amplified")], 1.0, 1e-12);
    assert_close(solution.values[index(&simulator, "driven")], 1.0, 1e-12);
}

/// The diode current that a source through a resistor drives, by bisection on the closed form.
fn diode_current(source: f64, resistance: f64, saturation: f64) -> f64 {
    let (mut low, mut high) = (0.0, source / resistance);
    for _ in 0..200 {
        let current = 0.5 * (low + high);
        let voltage = THERMAL_VOLTAGE * (current / saturation + 1.0).ln();
        if voltage + current * resistance > source {
            high = current;
        } else {
            low = current;
        }
    }
    0.5 * (low + high)
}

#[test]
fn a_diode_settles_where_its_curve_meets_the_load_line() {
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(5.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e3}),
            &[("a", "in"), ("b", "anode")],
        ),
        (
            "D1",
            "diode",
            json!({}),
            &[("anode", "anode"), ("cathode", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let options = Options {
        relative_tolerance: 1e-9,
        ..Options::default()
    };
    let solution = simulator.operating_point(&options).unwrap();
    let current = -solution.values[index(&simulator, "V1:current")];
    assert_close(current, diode_current(5.0, 1e3, 1e-14), 1e-6);
}

#[test]
fn a_dc_sweep_traces_the_diode_curve() {
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(0.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 100}),
            &[("a", "in"), ("b", "anode")],
        ),
        (
            "D1",
            "diode",
            json!({}),
            &[("anode", "anode"), ("cathode", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let levels: Vec<f64> = (0..=20).map(|step| f64::from(step) * 0.25).collect();
    let sweep = DcSweep {
        source: "V1".to_owned(),
        levels: levels.clone(),
    };
    let options = Options {
        relative_tolerance: 1e-9,
        ..Options::default()
    };
    let result = simulator.dc_sweep(&sweep, &options).unwrap();
    let current = index(&simulator, "V1:current");
    for (level, values) in levels.iter().zip(&result.values).skip(4) {
        assert_close(-values[current], diode_current(*level, 100.0, 1e-14), 1e-6);
    }
    // The sweep releases the source: an operating point afterwards sees its own level again.
    let after = simulator.operating_point(&Options::default()).unwrap();
    assert!(after.values[current].abs() < 1e-12);
}

#[test]
fn an_rc_low_pass_rolls_off_at_its_corner() {
    let (resistance, capacitance) = (1e3, 1e-6);
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(0.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": resistance}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let frequencies = vec![1.0, 10.0, 159.154_943_091_895_33, 1e3, 1e5];
    let ac = Ac {
        source: "V1".to_owned(),
        frequencies: frequencies.clone(),
    };
    let response = simulator.ac(&ac, &Options::default()).unwrap();
    let out = index(&simulator, "out");
    for (frequency, values) in frequencies.iter().zip(&response.values) {
        let expected = Complex64::new(1.0, 0.0)
            / Complex64::new(1.0, TAU * frequency * resistance * capacitance);
        assert!(
            (values[out] - expected).norm() < 1e-12,
            "at {frequency} Hz: {} vs {expected}",
            values[out]
        );
    }
    let corner = response.values[2][out];
    assert_close(corner.norm(), 0.5f64.sqrt(), 1e-9);
    assert_close(corner.arg(), -PI / 4.0, 1e-9);
}

#[test]
fn a_series_rlc_passes_everything_at_resonance() {
    let (inductance, capacitance) = (1e-3, 1e-9);
    let resonance = 1.0 / (TAU * f64::sqrt(inductance * capacitance));
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(0.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "L1",
            "inductor",
            json!({"inductance": inductance}),
            &[("a", "in"), ("b", "mid")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "mid"), ("b", "out")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 50}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let ac = Ac {
        source: "V1".to_owned(),
        frequencies: vec![resonance],
    };
    let response = simulator.ac(&ac, &Options::default()).unwrap();
    let gain = response.values[0][index(&simulator, "out")];
    assert!((gain - Complex64::new(1.0, 0.0)).norm() < 1e-9, "{gain}");
}

/// The response of an RC low-pass to a ramp from 0 to 1 that starts at `delay` and lasts
/// `rise`, after the ramp ends.
fn rc_ramp_response(time: f64, delay: f64, rise: f64, tau: f64) -> f64 {
    1.0 - tau / rise * ((rise / tau).exp() - 1.0) * (-(time - delay) / tau).exp()
}

/// The largest error of an RC low-pass's response to a fast ramp, simulated with `method` and
/// `options`, and the number of steps it took.
fn rc_step_error(method: Method, options: &Options) -> (f64, usize) {
    let (resistance, capacitance) = (1e3, 1e-9);
    let tau = resistance * capacitance;
    let (delay, rise) = (1e-6, 1e-9);
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"pulse": {
                "initial": 0, "pulsed": 1, "delay": delay, "rise": rise, "fall": rise, "width": 1,
            }}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": resistance}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let transient = Transient {
        stop: 6e-6,
        max_step: None,
        method,
        times: Vec::new(),
        periodic: false,
    };
    let waveforms = simulator
        .transient(&transient, options)
        .expect("a transient analysis");
    let out = index(&simulator, "out");
    assert!(
        waveforms.times.windows(2).all(|pair| pair[1] > pair[0]),
        "times increase"
    );
    assert!(waveforms.times.contains(&delay) && waveforms.times.contains(&(delay + rise)));
    assert_eq!(waveforms.times.last(), Some(&6e-6));
    let mut error: f64 = 0.0;
    for (&time, values) in waveforms.times.iter().zip(&waveforms.values) {
        if time <= delay {
            assert!(values[out].abs() < 1e-12, "no response before the ramp");
        } else if time >= delay + rise {
            error = error.max((values[out] - rc_ramp_response(time, delay, rise, tau)).abs());
        }
    }
    (error, waveforms.statistics.accepted)
}

/// At SPICE's default tolerances, each method stays within its order's accuracy, and every
/// method converges: tightening the tolerances shrinks the error severalfold.
#[test]
fn an_rc_charges_exponentially_with_every_method() {
    let tight = Options {
        relative_tolerance: 1e-6,
        charge_tolerance: 1e-18,
        ..Options::default()
    };
    for (method, bound) in [
        (Method::BackwardEuler, 2e-2),
        (Method::Trapezoidal, 2e-3),
        (Method::Gear2, 2e-3),
    ] {
        let (loose_error, _) = rc_step_error(method, &Options::default());
        let (tight_error, steps) = rc_step_error(method, &tight);
        assert!(
            loose_error < bound,
            "{method:?}: error {loose_error:e} at default tolerances"
        );
        assert!(
            tight_error < loose_error / 5.0,
            "{method:?}: {tight_error:e} vs {loose_error:e}"
        );
        assert!(steps < 2000, "{method:?}: {steps} steps");
    }
}

/// Each integration method's order, the power of the step its global error falls with.
const ORDERS: [(Method, f64); 3] = [
    (Method::BackwardEuler, 1.0),
    (Method::Trapezoidal, 2.0),
    (Method::Gear2, 2.0),
];

/// The steps each method takes in turn, each half the last.
const HALVED: [f64; 5] = [1e-7, 5e-8, 2.5e-8, 1.25e-8, 6.25e-9];

/// Options under which no step is shortened for its error: each is the longest allowed
/// once the steps after the start have grown to it, so that only the step sets the error.
fn uniform_steps() -> Options {
    Options {
        truncation_tolerance: 1e12,
        relative_tolerance: 1e-9,
        voltage_tolerance: 1e-12,
        current_tolerance: 1e-15,
        ..Options::default()
    }
}

/// `out` at every time of a transient of `circuit` to `end` with `method`, every step at
/// most `step` long and none shortened for its error.
fn stepped(circuit: &Circuit, method: Method, step: f64, end: f64) -> (Vec<f64>, Vec<f64>) {
    let mut simulator = simulator(circuit);
    let transient = Transient {
        stop: end,
        max_step: Some(step),
        method,
        times: Vec::new(),
        periodic: false,
    };
    let waveforms = simulator
        .transient(&transient, &uniform_steps())
        .expect("a transient analysis");
    let out = index(&simulator, "out");
    let values = waveforms.values.iter().map(|values| values[out]).collect();
    (waveforms.times, values)
}

/// An RC low-pass of 1 µs driven by a sine of 100 kHz and `amplitude`, a diode across its
/// capacitor if `clamped`.
fn sine_low_pass(amplitude: f64, clamped: bool) -> Circuit {
    let mut devices: Vec<Device<'_>> = vec![
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"sine": {"offset": 0, "amplitude": amplitude, "frequency": 1e5}}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e3}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-9}),
            &[("a", "out"), ("b", "0")],
        ),
    ];
    if clamped {
        devices.push((
            "D1",
            "diode",
            json!({}),
            &[("anode", "0"), ("cathode", "out")],
        ));
    }
    circuit(&devices)
}

/// The order each halving of the step shows, from errors in the order of `HALVED`.
fn observed_orders(errors: &[f64]) -> Vec<f64> {
    errors
        .windows(2)
        .map(|pair| (pair[0] / pair[1]).abs().log2())
        .collect()
}

#[test]
fn each_method_converges_at_its_order_as_the_step_halves() {
    // An RC low-pass's response to a sine starting at zero, `out` from rest:
    // (sin ωt − ωτ cos ωt + ωτ e^(−t/τ)) / (1 + (ωτ)²).
    let circuit = sine_low_pass(1.0, false);
    let (omega, tau) = (TAU * 1e5, 1e-6);
    let exact = |time: f64| {
        ((omega * time).sin() - omega * tau * (omega * time).cos()
            + omega * tau * (-time / tau).exp())
            / (1.0 + (omega * tau).powi(2))
    };
    for (method, order) in ORDERS {
        let errors: Vec<f64> = HALVED
            .iter()
            .map(|&step| {
                let (times, values) = stepped(&circuit, method, step, 2e-5);
                times
                    .iter()
                    .zip(values)
                    .map(|(&time, value)| (value - exact(time)).abs())
                    .fold(0.0, f64::max)
            })
            .collect();
        for observed in observed_orders(&errors) {
            assert!(
                (observed - order).abs() < 0.1,
                "{method:?} converges at order {observed}, not {order}: errors {errors:?}"
            );
        }
    }
}

#[test]
fn each_method_converges_at_its_order_after_a_diode_clamps() {
    // A diode clamps the sine's first negative half; at 13 µs it is off again, and the
    // differences between successive halvings shrink by the order's power of two
    // (Richardson), with no closed form to measure the error against.
    let circuit = sine_low_pass(2.0, true);
    for (method, order) in ORDERS {
        let ends: Vec<f64> = HALVED
            .iter()
            .map(|&step| *stepped(&circuit, method, step, 1.3e-5).1.last().unwrap())
            .collect();
        let differences: Vec<f64> = ends.windows(2).map(|pair| pair[0] - pair[1]).collect();
        for observed in observed_orders(&differences) {
            assert!(
                (observed - order).abs() < 0.1,
                "{method:?} converges at order {observed}, not {order}: ends {ends:?}"
            );
        }
    }
}

#[test]
fn an_lc_tank_rings_at_its_resonant_frequency() {
    let (inductance, capacitance) = (1e-3, 1e-6);
    let period = TAU * f64::sqrt(inductance * capacitance);
    // A current source charges the capacitor, then steps off at t = 1 µs; the tank then rings
    // around zero with the energy it holds.
    let circuit = circuit(&[
        (
            "I1",
            "current_source",
            json!({"waveform": {"piecewise": {
                "points": [[0, 1e-3], [1e-6, 1e-3], [1.001e-6, 0]],
            }}}),
            &[("positive", "0"), ("negative", "top")],
        ),
        (
            "L1",
            "inductor",
            json!({"inductance": inductance}),
            &[("a", "top"), ("b", "0")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "top"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let transient = Transient {
        stop: 5.0 * period,
        max_step: Some(period / 200.0),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let waveforms = simulator
        .transient(&transient, &Options::default())
        .unwrap();
    let current = index(&simulator, "L1:current");
    // Before the step the inductor carries the source's 1 mA; after, the current oscillates
    // with that amplitude, and the trapezoidal formula conserves its energy.
    let late: Vec<f64> = waveforms
        .times
        .iter()
        .zip(&waveforms.values)
        .filter(|(time, _)| **time > 3.0 * period)
        .map(|(_, values)| values[current])
        .collect();
    let peak = late.iter().copied().fold(0.0, f64::max);
    assert_close(peak, 1e-3, 1e-2);
    let crossings = late
        .windows(2)
        .filter(|pair| pair[0] < 0.0 && pair[1] >= 0.0)
        .count();
    assert_eq!(crossings, 2, "two periods after three");
}

#[test]
fn a_sine_through_an_rc_settles_to_the_ac_amplitude() {
    let (resistance, capacitance, frequency) = (1e3, 1e-6, 500.0);
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"sine": {"offset": 0, "amplitude": 1, "frequency": frequency}}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": resistance}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let transient = Transient {
        stop: 20e-3,
        max_step: Some(1e-5),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let outcome = simulator
        .run(&Analysis::Transient(transient), &Options::default())
        .unwrap();
    let Outcome::Transient(waveforms) = outcome else {
        panic!("a transient outcome")
    };
    let out = index(&simulator, "out");
    let peak = waveforms
        .times
        .iter()
        .zip(&waveforms.values)
        .filter(|(time, _)| **time > 15e-3)
        .map(|(_, values)| values[out].abs())
        .fold(0.0, f64::max);
    let expected = 1.0 / (1.0 + (TAU * frequency * resistance * capacitance).powi(2)).sqrt();
    assert_close(peak, expected, 1e-2);
}

#[test]
fn the_trapezoidal_formula_stops_ringing_after_a_corner() {
    // A source holds a capacitor at |sin|, whose corners at each zero are no breakpoint: the
    // trapezoidal formula's current rings after each, by the slope's whole jump, until one
    // step of backward Euler ends it.
    let (capacitance, frequency) = (1e-6, 1e3);
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"sine": {"offset": 0, "amplitude": 1, "frequency": frequency}}}),
            &[("positive", "s"), ("negative", "0")],
        ),
        (
            "B1",
            "behavioral_voltage_source",
            json!({"expression": "abs(v(s))"}),
            &[("positive", "out"), ("negative", "0"), ("s", "s")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": capacitance}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let transient = Transient {
        stop: 5e-3,
        max_step: Some(1e-5),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let outcome = simulator
        .run(&Analysis::Transient(transient), &Options::default())
        .unwrap();
    let Outcome::Transient(waveforms) = outcome else {
        panic!("a transient outcome")
    };
    let current = index(&simulator, "B1:current");
    let peak = capacitance * TAU * frequency;
    for (time, values) in waveforms.times.iter().zip(&waveforms.values) {
        let phase = TAU * frequency * time;
        if phase.sin().abs() < 0.5 {
            continue;
        }
        let expected = peak * phase.cos().abs();
        let error = (values[current].abs() - expected).abs();
        assert!(
            error < 1e-2 * peak,
            "at {time} s the current is off by {error} A"
        );
    }
}

/// A behavioral source steps from 0 to 5 V as `condition` turns true, through 1 kΩ into
/// 1 nF, a time constant of 1 µs; a ramp of 5 V over 2 ms is at `ramp`, which the source
/// reads if the condition does.
fn jumping(condition: &str) -> Circuit {
    circuit(&[
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"pulse": {"initial": 0, "pulsed": 5, "delay": 0, "rise": 2e-3,
                "fall": 1e-3, "width": 1e-3}}}),
            &[("positive", "ramp"), ("negative", "0")],
        ),
        (
            "B1",
            "behavioral_voltage_source",
            json!({"expression": format!("{condition} ? 5 : 0")}),
            if condition.contains("v(ramp)") {
                &[("positive", "out"), ("negative", "0"), ("ramp", "ramp")]
            } else {
                &[("positive", "out"), ("negative", "0")]
            },
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e3}),
            &[("a", "out"), ("b", "cap")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-9}),
            &[("a", "cap"), ("b", "0")],
        ),
    ])
}

#[test]
fn a_behavioral_jump_is_stepped_onto_and_followed() {
    // Whether in time or in a voltage, the step at 1.0003 ms falls within a step of 40 µs;
    // located, it starts the capacitor's exponential there, with no overshoot.
    let jump = 1.0003e-3;
    for condition in ["time > 1.0003e-3", "v(ramp) > 2.50075"] {
        let circuit = jumping(condition);
        let mut simulator = simulator(&circuit);
        let transient = Transient {
            stop: 2e-3,
            max_step: None,
            method: Method::Trapezoidal,
            times: Vec::new(),
            periodic: false,
        };
        let outcome = simulator
            .run(&Analysis::Transient(transient), &Options::default())
            .unwrap();
        let Outcome::Transient(waveforms) = outcome else {
            panic!("a transient outcome")
        };
        let cap = index(&simulator, "cap");
        for (time, values) in waveforms.times.iter().zip(&waveforms.values) {
            let expected = if *time > jump {
                5.0 * (1.0 - (-(time - jump) / 1e-6).exp())
            } else {
                0.0
            };
            assert!(
                (values[cap] - expected).abs() < 2e-2,
                "{condition}: at {time} s the capacitor holds {} V, not {expected} V",
                values[cap]
            );
        }
    }
}

#[test]
fn hierarchy_gives_the_flat_answer() {
    let document = json!({
        "format": "jolty-circuit", "version": 1, "top": "top",
        "components": {
            "top": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                {"id": "V1", "of": {"device": "voltage_source"}, "parameters": dc(3.0),
                 "terminals": {"positive": "in", "negative": "0"}},
                {"id": "X1", "of": {"component": "half"},
                 "terminals": {"input": "in", "output": "mid", "ground": "0"}},
                {"id": "X2", "of": {"component": "half"},
                 "terminals": {"input": "mid", "output": "out", "ground": "0"}}
            ]},
            "half": {"ports": [
                {"name": "input", "role": "signal"},
                {"name": "output", "role": "signal"},
                {"name": "ground", "role": "ground"}
            ], "instances": [
                {"id": "E1", "of": {"device": "voltage_controlled_voltage_source"},
                 "parameters": {"gain": 0.5},
                 "terminals": {"positive": "buffered", "negative": "ground",
                               "control_positive": "input", "control_negative": "ground"}},
                {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
                 "terminals": {"a": "buffered", "b": "output"}}
            ]}
        }
    });
    let circuit = Circuit::from_json(&document.to_string()).unwrap();
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_close(solution.values[index(&simulator, "out")], 0.75, 1e-12);
    assert!(
        simulator.unknown("X1/buffered").is_some() && simulator.unknown("X2/E1:current").is_some()
    );
}

#[test]
fn a_node_with_no_dc_path_is_singular() {
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(1.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-9}),
            &[("a", "in"), ("b", "floating")],
        ),
    ]);
    let error = simulator(&circuit)
        .operating_point(&Options::default())
        .unwrap_err();
    assert_eq!(error, SimulationError::Singular);
}

#[test]
fn analyses_name_what_they_cannot_find() {
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(1.0),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1.0}),
            &[("a", "in"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let options = Options::default();
    let missing = DcSweep {
        source: "V9".to_owned(),
        levels: vec![1.0],
    };
    assert_eq!(
        simulator.dc_sweep(&missing, &options),
        Err(SimulationError::UnknownSource("V9".to_owned()))
    );
    let resistor = Ac {
        source: "R1".to_owned(),
        frequencies: vec![1.0],
    };
    assert_eq!(
        simulator.ac(&resistor, &options).unwrap_err(),
        SimulationError::NotASource("R1".to_owned())
    );
}

/// A transient of `circuit` until `stop`, in steps of at most a hundredth of it.
fn transient(circuit: &Circuit, stop: f64, options: &Options) -> Result<Outcome, SimulationError> {
    let transient = Transient {
        stop,
        max_step: Some(stop / 100.0),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    simulator(circuit).run(&Analysis::Transient(transient), options)
}

#[test]
fn a_transient_names_the_unknowns_that_stopped_converging() {
    // The only equation of `a`, v(a)² = v(c), loses its solution as the source falls below
    // zero at 0.5 ms, so no step past it converges, however short.
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"piecewise": {"points": [[0, 1], [1e-3, -1]]}}}),
            &[("positive", "c"), ("negative", "0")],
        ),
        (
            "B1",
            "behavioral_current_source",
            json!({"expression": "1e-3 * (v(a) ^ 2 - v(c))"}),
            &[("positive", "a"), ("negative", "0"), ("a", "a"), ("c", "c")],
        ),
    ]);
    let error = transient(&circuit, 1e-3, &Options::default()).unwrap_err();
    let SimulationError::StepTooSmall {
        time,
        rejection: Rejection::Unconverged { worst },
        ..
    } = &error
    else {
        panic!("{error}")
    };
    assert!((0.4e-3..=0.5e-3).contains(time), "it failed at {time} s");
    assert_eq!(worst.first().map(String::as_str), Some("a"), "{error}");
    assert!(
        error
            .to_string()
            .contains("Newton iteration did not converge; furthest from converging: a"),
        "{error}"
    );
}

#[test]
fn a_transient_names_the_equation_whose_error_it_could_not_bound() {
    // No step is accurate enough for a truncation tolerance this small.
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"sine": {"offset": 0, "amplitude": 1, "frequency": 1e3}}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e3}),
            &[("a", "in"), ("b", "out")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-6}),
            &[("a", "out"), ("b", "0")],
        ),
    ]);
    let options = Options {
        truncation_tolerance: 1e-30,
        ..Options::default()
    };
    let error = transient(&circuit, 1e-3, &options).unwrap_err();
    assert!(
        matches!(
            &error,
            SimulationError::StepTooSmall {
                rejection: Rejection::Inaccurate(name),
                ..
            } if name == "out"
        ),
        "{error}"
    );
}

#[test]
fn a_transient_follows_a_source_that_only_its_transistor_holds() {
    // Nothing but the transistor holds its source, so as its gate steps up the source
    // follows to a threshold below it, where the current falls to zero quadratically: a
    // double root, toward which Newton iteration only halves its error each iteration.
    let (supply, threshold) = (5.0, 0.5);
    let circuit = circuit(&[
        (
            "VD",
            "voltage_source",
            dc(supply),
            &[("positive", "drain"), ("negative", "0")],
        ),
        (
            "VG",
            "voltage_source",
            json!({"waveform": {"piecewise": {"points": [[0, 0], [1e-6, 0], [1.001e-6, supply]]}}}),
            &[("positive", "gate"), ("negative", "0")],
        ),
        (
            "M1",
            "mosfet",
            json!({"level": 1, "channel": "n", "width": 1e-5, "length": 1e-6,
                   "threshold_voltage": threshold, "transconductance_parameter": 1e-4}),
            &[
                ("drain", "drain"),
                ("gate", "gate"),
                ("source", "source"),
                ("bulk", "0"),
            ],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let transient = Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let outcome = simulator
        .run(&Analysis::Transient(transient), &Options::default())
        .unwrap();
    let Outcome::Transient(waveforms) = outcome else {
        panic!("a transient outcome")
    };
    let source = index(&simulator, "source");
    let last = waveforms.values.last().unwrap()[source];
    assert_close(last, supply - threshold, 1e-2);
}

#[test]
fn a_transient_couples_an_edge_beside_a_node_at_a_hundred_megavolts() {
    // Only the reverse-biased diode's gmin holds `high`, so the current source lifts it to
    // 100 MV, and the microfarad from it to `low` holds 100 C: rounding alone leaves
    // microamperes in `low`'s equation, which Newton iteration must not chase. No current
    // flows through that microfarad, so `low` is the high-pass of the source's edge
    // through 10 nF into 10 Ω, 0.63 V once a time constant of the edge has risen.
    let circuit = circuit(&[
        (
            "I1",
            "current_source",
            dc(1e-4),
            &[("positive", "0"), ("negative", "high")],
        ),
        (
            "D1",
            "diode",
            json!({}),
            &[("anode", "0"), ("cathode", "high")],
        ),
        (
            "C1",
            "capacitor",
            json!({"capacitance": 1e-6}),
            &[("a", "high"), ("b", "low")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 10}),
            &[("a", "low"), ("b", "0")],
        ),
        (
            "V1",
            "voltage_source",
            json!({"waveform": {"pulse": {
                "initial": 3, "pulsed": 4, "delay": 2e-7, "rise": 1e-7, "fall": 1e-7,
                "width": 8e-7,
            }}}),
            &[("positive", "in"), ("negative", "0")],
        ),
        (
            "C2",
            "capacitor",
            json!({"capacitance": 1e-8}),
            &[("a", "in"), ("b", "low")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let transient = Transient {
        stop: 2e-6,
        max_step: Some(1e-8),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let outcome = simulator
        .run(&Analysis::Transient(transient), &Options::default())
        .unwrap();
    let Outcome::Transient(waveforms) = outcome else {
        panic!("a transient outcome")
    };
    let risen = waveforms
        .times
        .iter()
        .position(|&time| (time - 3e-7).abs() < 1e-15)
        .expect("a step onto the end of the edge");
    let tau: f64 = 10.0 * 1e-8;
    let expected = 1e7 * tau * (1.0 - (-1e-7 / tau).exp());
    assert_close(
        waveforms.values[risen][index(&simulator, "low")],
        expected,
        1e-2,
    );
}

/// A level 1 MOSFET of the given channel, its bulk on its source.
fn mosfet<'a>(
    id: &'a str,
    channel: &str,
    [drain, gate, source]: [&'a str; 3],
) -> (&'a str, &'a str, Value, Vec<(&'a str, &'a str)>) {
    let (width, threshold, transconductance) = if channel == "n" {
        (1e-5, 0.5, 1e-4)
    } else {
        (2e-5, -0.5, 5e-5)
    };
    (
        id,
        "mosfet",
        json!({"level": 1, "channel": channel, "width": width, "length": 1e-6,
               "threshold_voltage": threshold, "transconductance_parameter": transconductance}),
        vec![
            ("drain", drain),
            ("gate", gate),
            ("source", source),
            ("bulk", source),
        ],
    )
}

#[test]
fn a_gate_converges_as_its_supply_powers_up() {
    // A NAND gate, its inputs from inverters, all of them powering up with the supply: near
    // their thresholds, the node between the NAND's NMOS converges so slowly that a step's
    // Newton iteration runs out before it does. Starting each shorter step over from the
    // last accepted point, the iteration never got further; going on from where it
    // stopped, it converges.
    let clock = |period: f64| {
        json!({"waveform": {"pulse": {
            "initial": 0, "pulsed": 5, "delay": 3e-8, "rise": 2e-9, "fall": 2e-9,
            "width": 0.48 * period, "period": period,
        }}})
    };
    let capacitor = json!({"capacitance": 5e-14});
    let mut devices = vec![
        mosfet("MIA", "n", ["a", "rawa", "0"]),
        mosfet("MPA", "p", ["a", "rawa", "vdd"]),
        mosfet("MIB", "n", ["b", "rawb", "0"]),
        mosfet("MPB", "p", ["b", "rawb", "vdd"]),
        mosfet("MNA", "n", ["out", "a", "mid"]),
        mosfet("MNB", "n", ["mid", "b", "0"]),
        mosfet("MQA", "p", ["out", "a", "vdd"]),
        mosfet("MQB", "p", ["out", "b", "vdd"]),
    ];
    for (id, node) in [("CA", "a"), ("CB", "b"), ("CO", "out")] {
        devices.push((
            id,
            "capacitor",
            capacitor.clone(),
            vec![("a", node), ("b", "0")],
        ));
    }
    devices.push((
        "VDD",
        "voltage_source",
        json!({"waveform": {"piecewise": {"points": [[0, 0], [1e-6, 5]]}}}),
        vec![("positive", "vdd"), ("negative", "0")],
    ));
    devices.push((
        "VA",
        "voltage_source",
        clock(1e-7),
        vec![("positive", "rawa"), ("negative", "0")],
    ));
    devices.push((
        "VB",
        "voltage_source",
        clock(9.1e-8),
        vec![("positive", "rawb"), ("negative", "0")],
    ));
    let listed: Vec<Device<'_>> = devices
        .iter()
        .map(|(id, model, parameters, terminals)| {
            (*id, *model, parameters.clone(), terminals.as_slice())
        })
        .collect();
    let circuit = circuit(&listed);
    let transient = Transient {
        stop: 1e-6,
        max_step: Some(1e-9),
        method: Method::Trapezoidal,
        times: Vec::new(),
        periodic: false,
    };
    let outcome = simulator(&circuit).run(&Analysis::Transient(transient), &Options::default());
    assert!(outcome.is_ok(), "{:?}", outcome.err());
}

#[test]
fn a_heated_resistor_settles_where_its_power_flows_through_its_sink() {
    // 10 V across 100 Ω with tc1 = 4e-3 /K, held to the ambient by 50 K/W: the rise solves
    // `tc · ΔT² + ΔT − 50 · v² / r = 0`.
    let circuit = circuit(&[
        (
            "V1",
            "voltage_source",
            dc(10.0),
            &[("positive", "top"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 100.0, "temperature_coefficient": 4e-3, "thermal": true}),
            &[("a", "top"), ("b", "0"), ("thermal", "junction")],
        ),
        (
            "RTH",
            "thermal_resistance",
            json!({"resistance": 50.0}),
            &[("a", "junction"), ("b", "0")],
        ),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    let heat: f64 = 50.0 * 100.0 / 100.0;
    let rise = ((1.0 + 4.0 * 4e-3 * heat).sqrt() - 1.0) / (2.0 * 4e-3);
    assert_close(solution.values[index(&simulator, "junction")], rise, 1e-9);
    let current = solution.values[index(&simulator, "V1:current")];
    assert_close(-current, 10.0 / (100.0 * (1.0 + 4e-3 * rise)), 1e-9);
}

#[test]
fn a_circuit_of_a_foundrys_device_elaborates_but_no_analysis_solves_it() {
    let transistor = json!({
        "pdk": "sky130", "model": "sky130_fd_pr__nfet_01v8", "width": 1e-6, "length": 1.5e-7,
    });
    let terminals = [
        ("drain", "d"),
        ("gate", "d"),
        ("source", "0"),
        ("bulk", "0"),
    ];
    let mut simulator = simulator(&circuit(&[
        (
            "V1",
            "voltage_source",
            dc(1.8),
            &[("positive", "vdd"), ("negative", "0")],
        ),
        (
            "R1",
            "resistor",
            json!({"resistance": 1e4}),
            &[("a", "vdd"), ("b", "d")],
        ),
        ("M1", "foundry_mosfet", transistor, &terminals),
    ]));
    assert_eq!(simulator.foundry_devices(), ["M1"]);
    let refused = SimulationError::Foundry(vec!["M1".to_owned()]);
    let options = Options::default();
    for analysis in [
        Analysis::OperatingPoint,
        Analysis::DcSweep(DcSweep {
            source: "V1".to_owned(),
            levels: vec![0.0, 1.8],
        }),
        Analysis::Ac(Ac {
            source: "V1".to_owned(),
            frequencies: vec![1e3],
        }),
    ] {
        assert_eq!(simulator.run(&analysis, &options), Err(refused.clone()));
    }
    assert_eq!(
        refused.to_string(),
        "the engine solves no foundry's device, which ngspice simulates with its foundry's \
         models: M1"
    );
}
