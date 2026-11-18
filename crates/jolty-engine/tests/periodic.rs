//! The periodic steady state against closed form and against long transients.
#![cfg(test)]

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use jolty_engine::{
    Analysis, AnalysisResult, Conditions, Cotangent, Cotangents, Differentiated, Method, Options,
    PeriodicSteadyState, PeriodicSteadyStateResult, Sensitivity, SimulationError, Simulator,
    Transient, TransientResult, Values,
};
use jolty_ir::Circuit;
use jolty_models::Library;
use serde_json::{Value, json};

/// A circuit of `instances` whose ground is `0`.
fn circuit(instances: &Value) -> Circuit {
    let document = json!({
        "format": "jolty-circuit", "version": 4, "top": "test",
        "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                "instances": instances}}
    });
    Circuit::from_json(&document.to_string()).expect("a valid circuit")
}

fn simulator(instances: &Value) -> Simulator {
    Simulator::new(
        &circuit(instances),
        &Library::standard(),
        &Conditions::default(),
    )
    .expect("the circuit elaborates")
}

fn sine(id: &str, positive: &str, (offset, amplitude, frequency): (f64, f64, f64)) -> Value {
    json!({"name": id, "of": {"model": "voltage_source"},
           "parameters": {"waveform": {"sine": {"offset": offset, "amplitude": amplitude,
                                                "frequency": frequency}}},
           "terminals": {"positive": positive, "negative": "0"}})
}

/// A two-terminal device whose one parameter `name` is `value`.
fn two_terminal(id: &str, (device, name, value): (&str, &str, f64), a: &str, b: &str) -> Value {
    json!({"name": id, "of": {"model": device}, "parameters": {name: value},
           "terminals": {"a": a, "b": b}})
}

/// A transient until `stop`, its steps at most `longest`.
fn run(simulator: &mut Simulator, (stop, longest): (f64, f64)) -> TransientResult {
    run_with(simulator, (stop, longest), &Options::default())
}

fn run_with(
    simulator: &mut Simulator,
    (stop, longest): (f64, f64),
    options: &Options,
) -> TransientResult {
    let transient = Transient {
        stop,
        max_step: Some(longest),
        method: Method::Trapezoidal,
        breakpoints: Vec::new(),
        initial_conditions: BTreeMap::new(),
        operating_point: true,
    };
    match simulator.run(&Analysis::Transient(transient), options) {
        Ok(AnalysisResult::Transient(waveforms)) => waveforms,
        other => panic!("a transient outcome, not {other:?}"),
    }
}

/// The periodic steady state of `period`, or an oscillator's without, its steps at most
/// `longest`.
fn steady(
    simulator: &mut Simulator,
    (period, longest): (Option<f64>, f64),
    options: &Options,
) -> PeriodicSteadyStateResult {
    let analysis = PeriodicSteadyState {
        period,
        max_step: Some(longest),
        ..PeriodicSteadyState::default()
    };
    match simulator.run(&Analysis::PeriodicSteadyState(analysis), options) {
        Ok(AnalysisResult::PeriodicSteadyState(found)) => found,
        other => panic!("a periodic steady state, not {other:?}"),
    }
}

fn index(simulator: &Simulator, name: &str) -> usize {
    simulator.unknown(name).expect("the unknown exists").index()
}

#[test]
fn a_sine_through_a_slow_rc_repeats_its_closed_form_at_once() {
    // The time constant is ten periods, so a transient from rest would take fifty to
    // settle within 1%; the circuit is linear, so one Newton step of shooting lands on the
    // steady state, and the second period confirms it.
    let (resistance, capacitance, frequency) = (1e3, 1e-5, 1e3);
    let mut simulator = simulator(&json!([
        sine("V1", "in", (0.0, 1.0, frequency)),
        two_terminal("R1", ("resistor", "resistance", resistance), "in", "out"),
        two_terminal("C1", ("capacitor", "capacitance", capacitance), "out", "0"),
    ]));
    let found = steady(
        &mut simulator,
        (Some(1.0 / frequency), 5e-3 / frequency),
        &Options::default(),
    );
    assert_eq!(found.periods, 2);
    let waveforms = found.waveforms;
    let out = index(&simulator, "out");
    let lag = TAU * frequency * resistance * capacitance;
    let amplitude = 1.0 / lag.hypot(1.0);
    for (time, values) in waveforms.times.iter().zip(&waveforms.values) {
        let expected = amplitude * (TAU * frequency * time - lag.atan()).sin();
        assert!(
            (values[out] - expected).abs() <= 1e-3 * amplitude,
            "at {time}: {} against {expected}",
            values[out]
        );
    }
}

#[test]
fn an_inductor_current_repeats_too() {
    // A series RL whose current lags the sine: the inductor's flux, on its branch current,
    // carries the state as a capacitor's charge does.
    let (resistance, inductance, frequency) = (10.0, 1e-2, 1e3);
    let mut simulator = simulator(&json!([
        sine("V1", "in", (0.5, 1.0, frequency)),
        two_terminal("R1", ("resistor", "resistance", resistance), "in", "a"),
        two_terminal("L1", ("inductor", "inductance", inductance), "a", "0"),
    ]));
    let found = steady(
        &mut simulator,
        (Some(1.0 / frequency), 5e-3 / frequency),
        &Options::default(),
    );
    assert_eq!(found.periods, 2);
    let waveforms = found.waveforms;
    let current = index(&simulator, "L1:current");
    let lag = TAU * frequency * inductance / resistance;
    let amplitude = 1.0 / resistance / lag.hypot(1.0);
    for (time, values) in waveforms.times.iter().zip(&waveforms.values) {
        let expected = 0.5 / resistance + amplitude * (TAU * frequency * time - lag.atan()).sin();
        assert!(
            (values[current] - expected).abs() <= 1e-3 * amplitude,
            "at {time}: {} against {expected}",
            values[current]
        );
    }
}

#[test]
fn a_rectifier_repeats_the_last_period_of_a_long_transient() {
    let frequency = 1e6;
    let instances = json!([
        sine("V1", "in", (0.2, 1.0, frequency)),
        two_terminal("R1", ("resistor", "resistance", 1e3), "in", "a"),
        {"name": "D1", "of": {"model": "diode"},
         "parameters": {"saturation_current": 1e-14, "series_resistance": 10.0,
                        "transit_time": 1e-9, "junction_capacitance": 2e-12},
         "terminals": {"anode": "a", "cathode": "out"}},
        two_terminal("C1", ("capacitor", "capacitance", 1e-9), "out", "0"),
        two_terminal("R2", ("resistor", "resistance", 1e4), "out", "0"),
    ]);
    let options = Options {
        relative_tolerance: 1e-6,
        ..Options::default()
    };
    let mut shooting = simulator(&instances);
    let longest = 5e-3 / frequency;
    let found = steady(&mut shooting, (Some(1.0 / frequency), longest), &options);
    assert!(found.periods <= 5, "{} periods", found.periods);
    let steady = found.waveforms;
    // Fifteen time constants of the load.
    let settled = 150;
    let mut long = simulator(&instances);
    let until = f64::from(settled) / frequency;
    let transient = run_with(&mut long, (until, longest), &options);
    let out = index(&shooting, "out");
    let last = |waveforms: &TransientResult, from: f64| {
        let values = waveforms
            .times
            .iter()
            .zip(&waveforms.values)
            .filter(|(time, _)| **time >= from)
            .map(|(_, values)| values[out]);
        values.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        })
    };
    let (low, high) = last(&steady, 0.0);
    let (settled_low, settled_high) = last(&transient, f64::from(settled - 1) / frequency);
    assert!(
        (low - settled_low).abs() < 1e-5,
        "{low} against {settled_low}"
    );
    assert!(
        (high - settled_high).abs() < 1e-5,
        "{high} against {settled_high}"
    );
    let start = steady.values[0][out];
    let end = steady.values.last().expect("points")[out];
    assert!((start - end).abs() < 1e-6, "{start} against {end}");
}

/// A sine of `amplitude` and 1 kHz across `memristor` from `in` to `out`, with 100 nF from
/// `out` to the ground, the memristor's state 0.5.
fn memristor_into_a_capacitor(memristor: (&str, Value), amplitude: f64) -> Value {
    let (model, mut parameters) = memristor;
    parameters["on_resistance"] = json!(1e3);
    parameters["off_resistance"] = json!(1e4);
    parameters["state"] = json!(0.5);
    json!([
        sine("V1", "in", (0.0, amplitude, 1e3)),
        {"name": "M1", "of": {"model": model}, "parameters": parameters,
         "terminals": {"positive": "in", "negative": "out"}},
        two_terminal("C1", ("capacitor", "capacitance", 1e-7), "out", "0"),
    ])
}

#[test]
fn a_memristor_its_drive_leaves_alone_keeps_its_state() {
    // The sine stays within the thresholds, so the state never moves, and every state is
    // as periodic as any other: Φ − I is singular, its row for the state zero. Shooting
    // keeps the state where it starts, and the capacitor settles through the 5.5 kΩ that
    // state sets, as an RC of a time constant of 0.55 periods.
    let memristor = json!({"on_voltage": -1.0, "off_voltage": 1.0,
                           "on_rate": 1e3, "off_rate": 1e3});
    let mut simulator = simulator(&memristor_into_a_capacitor(
        ("voltage_controlled_memristor", memristor),
        0.5,
    ));
    let frequency = 1e3;
    let found = steady(
        &mut simulator,
        (Some(1.0 / frequency), 5e-3 / frequency),
        &Options::default(),
    );
    assert_eq!(found.periods, 2);
    let waveforms = found.waveforms;
    let (state, out) = (index(&simulator, "M1:state"), index(&simulator, "out"));
    let lag = TAU * frequency * 5.5e3 * 1e-7;
    let amplitude = 0.5 / lag.hypot(1.0);
    for (time, values) in waveforms.times.iter().zip(&waveforms.values) {
        assert!((values[state] - 0.5).abs() < 1e-15, "{}", values[state]);
        let expected = amplitude * (TAU * frequency * time - lag.atan()).sin();
        assert!(
            (values[out] - expected).abs() <= 1e-3 * amplitude,
            "at {time}: {} against {expected}",
            values[out]
        );
    }
}

#[test]
fn a_memristor_and_a_capacitor_keep_the_charge_they_share() {
    // The charge through the memristor moves its state by q / Q and the capacitor's
    // voltage by q / C, so Q·x − C·v(out) is as each period found it: every value of it has
    // a steady state of its own. Shooting keeps the operating point's, and finds the
    // steady state a long transient from it settles into.
    let (charge, capacitance, frequency) = (1e-5, 1e-7, 1e3);
    let memristor = json!({"charge": charge});
    let instances = memristor_into_a_capacitor(("current_controlled_memristor", memristor), 1.0);
    let longest = 5e-3 / frequency;
    let mut shooting = simulator(&instances);
    let found = steady(
        &mut shooting,
        (Some(1.0 / frequency), longest),
        &Options::default(),
    );
    assert!(found.periods <= 4, "{} periods", found.periods);
    let steady = found.waveforms;
    let (state, out) = (index(&shooting, "M1:state"), index(&shooting, "out"));
    for values in &steady.values {
        let shared = charge * values[state] - capacitance * values[out];
        assert!((shared - charge * 0.5).abs() < 1e-12 * charge, "{shared}");
    }
    let settled = 20;
    let mut long = simulator(&instances);
    let transient = run(&mut long, (f64::from(settled) / frequency, longest));
    let range = |waveforms: &TransientResult, from: f64, unknown: usize| {
        let values = waveforms
            .times
            .iter()
            .zip(&waveforms.values)
            .filter(|(time, _)| **time >= from)
            .map(|(_, values)| values[unknown]);
        values.fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        })
    };
    for (unknown, tolerance) in [(state, 1e-6), (out, 1e-4)] {
        let (low, high) = range(&steady, 0.0, unknown);
        let last = f64::from(settled - 1) / frequency;
        let (settled_low, settled_high) = range(&transient, last, unknown);
        assert!(
            (low - settled_low).abs() < tolerance,
            "{low} against {settled_low}"
        );
        assert!(
            (high - settled_high).abs() < tolerance,
            "{high} against {settled_high}"
        );
    }
}

#[test]
fn a_periodic_steady_state_whose_matrices_would_fill_the_memory_fails_at_once() {
    let mut simulator = simulator(&json!([
        sine("V1", "in", (0.0, 1.0, 1e3)),
        two_terminal("R1", ("resistor", "resistance", 1e3), "in", "out"),
        two_terminal("C1", ("capacitor", "capacitance", 1e-6), "out", "0"),
    ]));
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        ..PeriodicSteadyState::default()
    };
    let options = Options {
        max_memory: 1e-9,
        ..Options::default()
    };
    let error = simulator
        .run(&Analysis::PeriodicSteadyState(analysis), &options)
        .expect_err("too little memory");
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state",
            rows: 3,
            needed: 14.0 * 9.0 / 134_217_728.0,
            max_memory: 1e-9,
        }
    );
    assert_eq!(
        error.to_string(),
        "the periodic steady state's dense matrices of 3 rows would hold \
         9.387731552124023e-7 GiB, more than its `max_memory` option's 1e-9 GiB"
    );
    assert_eq!(simulator.operating_point_solves(), 0);
}

#[test]
fn a_circuit_with_logic_has_no_periodic_steady_state_here() {
    let mut simulator = simulator(&json!([
        {"name": "A", "of": {"model": "logic_source"},
         "parameters": {"initial": "0", "changes": [], "transition": 1e-12},
         "terminals": {"output": "a"}},
    ]));
    let analysis = PeriodicSteadyState {
        period: Some(1e-6),
        ..PeriodicSteadyState::default()
    };
    let error = simulator
        .run(
            &Analysis::PeriodicSteadyState(analysis),
            &Options::default(),
        )
        .expect_err("logic is refused");
    assert!(
        matches!(error, SimulationError::InvalidAnalysis(_)),
        "{error}"
    );
}

/// A tank of `inductance` and `capacitance` whose conductance `−g (1 − r² / a²)`, for
/// the radius `r² = v² + (z i)²` of its state and an `impedance` `z`, feeds it inside the
/// circle of radius `a` and drains it outside: where `z` is the tank's own, `√(L/C)`, the
/// tank is lossless on the circle, so the orbit is that circle, its period exactly
/// `2π √(LC)`.
fn regulated_tank(
    (inductance, capacitance): (f64, f64),
    (conductance, impedance): (f64, f64),
    radius: f64,
) -> Value {
    let expression = format!(
        "-{conductance:e} * v(x) * (1 - (v(x) ^ 2 + ({impedance:e} * i(p)) ^ 2) / {:e})",
        radius * radius
    );
    json!([
        two_terminal("C1", ("capacitor", "capacitance", capacitance), "t", "0"),
        two_terminal("L1", ("inductor", "inductance", inductance), "t", "l"),
        {"name": "B1", "of": {"model": "behavioral_current_source"},
         "parameters": {"expression": expression},
         "terminals": {"positive": "t", "negative": "0", "x": "t",
                       "p_positive": "l", "p_negative": "0"}},
    ])
}

#[test]
fn an_oscillating_tank_finds_its_period_and_its_amplitude() {
    // The conductance at rest, −g, grows the tank's mode at σ = g / 2C; on the circle the
    // radius relaxes at g / C, so a period contracts its error by e^(−2π g / ωC).
    let (inductance, capacitance, radius): (f64, f64, f64) = (1e-6, 1e-9, 1.0);
    let impedance = (inductance / capacitance).sqrt();
    let omega = 1.0 / (inductance * capacitance).sqrt();
    let g = 0.2 / impedance;
    let options = Options {
        relative_tolerance: 1e-6,
        ..Options::default()
    };
    let mut simulator = simulator(&regulated_tank(
        (inductance, capacitance),
        (g, impedance),
        radius,
    ));
    let found = steady(&mut simulator, (None, TAU / omega / 100.0), &options);
    // From where the transient settled, Newton's steps square the mismatch, 6e4
    // tolerances of the current after the first period and 50 after the second.
    assert_eq!(found.periods, 3);
    let (voltage, current) = (index(&simulator, "t"), index(&simulator, "L1:current"));
    let (times, values) = (&found.waveforms.times, &found.waveforms.values);
    // A period starts with two steps of backward Euler, until the trapezoidal formula's
    // error estimate has the four points it reads, and each step h of it keeps
    // 1 / (1 + (ωh)²) of the tank's energy, which the conductance makes up over the period
    // inside the circle: the radius falls short by (ωh)² (C / 2gT + 1 / 2) of it at most.
    // A start within its tolerance of repeating lies within that over 1 − e^(−2π g / ωC)
    // of the orbit, in voltage and in the impedance times the current.
    let lost: f64 = times[..3]
        .windows(2)
        .map(|pair| (omega * (pair[1] - pair[0])).powi(2))
        .sum();
    let drained = radius * lost * (capacitance / (2.0 * g * found.period) + 0.5);
    let tolerance = options.relative_tolerance * radius + options.voltage_tolerance;
    let near = tolerance / (1.0 - (-TAU * g / (omega * capacitance)).exp());
    for values in values {
        let reached = values[voltage].hypot(impedance * values[current]);
        assert!(
            (reached - radius).abs() <= drained + near,
            "{reached} beyond {}",
            drained + near
        );
    }
    // The trapezoidal formula turns a step h by 2 atan(ωh / 2), less than ωh by (ωh)³ / 12
    // to leading order, so its period exceeds 2π / ω by at most (ωh)² / 12 of it, h the
    // longest step, and each step of backward Euler's by (ωh)³ / 3, less than (ωh)² / 3;
    // a start within its tolerance of repeating leaves the phase within tolerance over
    // the radius.
    let longest = times
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .fold(0.0, f64::max);
    let lag = (omega * longest).powi(2) / 12.0 + lost / 3.0 / TAU;
    let phase = tolerance / radius / TAU;
    let relative = found.period * omega / TAU - 1.0;
    assert!(
        -phase <= relative && relative <= lag + phase,
        "{relative} beyond [{}, {}]",
        -phase,
        lag + phase
    );
}

#[test]
fn an_oscillator_that_no_mode_starts_says_so() {
    // A lossless tank only rings about its operating point, and a resistor damps it.
    for damped in [false, true] {
        let mut instances = vec![
            two_terminal("C1", ("capacitor", "capacitance", 1e-9), "t", "0"),
            two_terminal("L1", ("inductor", "inductance", 1e-6), "t", "0"),
        ];
        if damped {
            instances.push(two_terminal(
                "R1",
                ("resistor", "resistance", 1e3),
                "t",
                "0",
            ));
        }
        let mut simulator = simulator(&Value::Array(instances));
        let analysis = Analysis::PeriodicSteadyState(PeriodicSteadyState::default());
        let error = simulator
            .run(&analysis, &Options::default())
            .expect_err("no oscillation starts");
        assert_eq!(error, SimulationError::NoGrowingMode);
    }
}

#[test]
fn an_oscillators_period_is_found_so_it_has_no_times_to_step_onto() {
    let mut simulator = simulator(&regulated_tank((1e-6, 1e-9), (6e-3, 31.6), 1.0));
    let analysis = PeriodicSteadyState {
        breakpoints: vec![1e-7],
        ..PeriodicSteadyState::default()
    };
    let error = simulator
        .run(
            &Analysis::PeriodicSteadyState(analysis),
            &Options::default(),
        )
        .expect_err("times without a period are refused");
    assert!(
        matches!(error, SimulationError::InvalidAnalysis(_)),
        "{error}"
    );
    assert_eq!(simulator.operating_point_solves(), 0);
}

#[test]
fn a_latch_that_its_growing_mode_tips_never_returns_and_says_so() {
    // 2 mS of negative conductance, saturating as tanh, against 1 kΩ across 1 nF: the
    // balanced state grows at 1e6 per second and tips into either of two stable ones,
    // ±1.9 V, so the transient never comes back across the kick.
    let mut simulator = simulator(&json!([
        two_terminal("R1", ("resistor", "resistance", 1e3), "n", "0"),
        two_terminal("C1", ("capacitor", "capacitance", 1e-9), "n", "0"),
        {"name": "B1", "of": {"model": "behavioral_current_source"},
         "parameters": {"expression": "-2e-3 * tanh(v(x))"},
         "terminals": {"positive": "n", "negative": "0", "x": "n"}},
    ]));
    let analysis = Analysis::PeriodicSteadyState(PeriodicSteadyState::default());
    let error = simulator
        .run(&analysis, &Options::default())
        .expect_err("a latch does not oscillate");
    // The stretches double from the mode's natural period, 2π µs, forty times.
    let time = TAU * 1e-6 * (2.0_f64.powi(40) - 1.0);
    match error {
        SimulationError::Unsettled { time: ran, returns } => {
            assert_eq!(returns, 0);
            assert!((ran - time).abs() <= 1e-12 * time, "{ran} against {time}");
        }
        other => panic!("an unsettled oscillation, not {other}"),
    }
}

#[test]
fn an_oscillating_tanks_period_moves_with_its_inductance_and_capacitance_exactly() {
    // The tank's equations are unchanged when time, L and C scale alike, the conductance
    // reading the radius through an impedance of its own, so the period scales with them:
    // L ∂T/∂L + C ∂T/∂C = T, for the steps of the discrete period too, each a fixed
    // fraction of it, to the rounding of the tangents and the bordered solve, 256 ε here,
    // held to 1e-12. With a truncation tolerance no step's error reaches, every run takes
    // those fractions, so central differences of the period, each run settling and
    // shooting afresh, measure the same derivatives: to a square root's truncation, δ²/8
    // of the derivative for steps of δ, the period going as √L and √C to leading order,
    // twice that held, and to the period's resolution, its 1e-10 tolerance over δ.
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let regulation = (0.2 / impedance, impedance);
    let options = Options {
        relative_tolerance: 1e-10,
        voltage_tolerance: 1e-13,
        current_tolerance: 1e-16,
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let analysis = PeriodicSteadyState::default();
    let period = |tank: (f64, f64)| {
        simulator(&regulated_tank(tank, regulation, 1.0))
            .periodic_steady_state(&analysis, &options)
            .expect("the tank oscillates")
            .period
    };
    let mut simulator = simulator(&regulated_tank((inductance, capacitance), regulation, 1.0));
    let sensitivity = Sensitivity {
        outputs: vec!["t".to_owned()],
        of: Differentiated::PeriodicSteadyState(analysis.clone()),
    };
    let found = simulator
        .sensitivity(&sensitivity, &options)
        .expect("the period's sensitivities");
    let AnalysisResult::PeriodicSteadyState(steady) = found.result.as_ref() else {
        panic!("a periodic steady state, not {:?}", found.result);
    };
    let derivatives = found
        .period_derivatives
        .expect("an oscillator's period moves");
    let by = |instance: &str| {
        let index = found
            .parameters
            .iter()
            .position(|parameter| parameter.instance == instance)
            .expect("the parameter");
        derivatives[index]
    };
    let (by_inductance, by_capacitance) = (by("L1"), by("C1"));
    let scaled = inductance * by_inductance + capacitance * by_capacitance;
    assert!(
        (scaled / steady.period - 1.0).abs() < 1e-12,
        "{scaled} against {}",
        steady.period
    );
    let step: f64 = 1e-3;
    let resolution = options.relative_tolerance / step;
    let differences = [
        (period((inductance * (1.0 + step), capacitance))
            - period((inductance * (1.0 - step), capacitance)))
            / (2.0 * step * inductance),
        (period((inductance, capacitance * (1.0 + step)))
            - period((inductance, capacitance * (1.0 - step))))
            / (2.0 * step * capacitance),
    ];
    for (exact, finite) in [by_inductance, by_capacitance].into_iter().zip(differences) {
        assert!(
            (exact / finite - 1.0).abs() <= step.powi(2) / 4.0 + resolution,
            "{exact} against {finite}"
        );
    }
}

#[test]
fn an_autonomous_periodic_steady_state_allows_dc_and_constant_waveforms() {
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let regulation = (0.2 / impedance, impedance);
    let expected = TAU * (inductance * capacitance).sqrt();
    let mut instances = regulated_tank((inductance, capacitance), regulation, 1.0);
    instances.as_array_mut().expect("an instance list").extend([
        json!({"name": "VDC", "of": {"model": "voltage_source"},
                   "parameters": {"waveform": {"dc": 3.3}},
                   "terminals": {"positive": "vdc", "negative": "0"}}),
        two_terminal("RDC", ("resistor", "resistance", 1e3), "vdc", "0"),
        sine("VSIN0", "vsin0", (1.0, 0.0, 1e6)),
        two_terminal("RSIN0", ("resistor", "resistance", 1e3), "vsin0", "0"),
        json!({"name": "VP0", "of": {"model": "voltage_source"},
                   "parameters": {"waveform": {"pulse": {
                       "initial": 2.0, "pulsed": 2.0, "rise": 1e-9, "fall": 1e-9, "width": 1e-8
                   }}},
                   "terminals": {"positive": "vp0", "negative": "0"}}),
        two_terminal("RP0", ("resistor", "resistance", 1e3), "vp0", "0"),
        json!({"name": "IPWL0", "of": {"model": "current_source"},
                   "parameters": {"waveform": {"piecewise": {
                       "points": [[0.0, 1e-3], [1e-6, 1e-3]]
                   }}},
                   "terminals": {"positive": "ipwl0", "negative": "0"}}),
        two_terminal("RPWL0", ("resistor", "resistance", 1e3), "ipwl0", "0"),
        json!({"name": "VEXP0", "of": {"model": "voltage_source"},
                   "parameters": {"waveform": {"exponential": {
                       "initial": 1.0, "pulsed": 1.0, "rise_delay": 0.0,
                       "rise_time_constant": 1e-9, "fall_delay": 1e-6, "fall_time_constant": 1e-9
                   }}},
                   "terminals": {"positive": "vexp0", "negative": "0"}}),
        two_terminal("REXP0", ("resistor", "resistance", 1e3), "vexp0", "0"),
    ]);
    let mut simulator = simulator(&instances);
    let found = steady(
        &mut simulator,
        (None, expected / 100.0),
        &Options::default(),
    );
    assert!(
        (found.period / expected - 1.0).abs() < 1e-2,
        "{} against {expected}",
        found.period
    );
}

#[test]
fn an_autonomous_periodic_steady_state_refuses_time_varying_sources() {
    let cases = [
        ("V1", sine("V1", "n", (0.0, 1.0, 1e3))),
        (
            "VP",
            json!({"name": "VP", "of": {"model": "voltage_source"},
                   "parameters": {"waveform": {"pulse": {
                       "initial": 0.0, "pulsed": 1.0, "rise": 1e-9, "fall": 1e-9, "width": 1e-6
                   }}},
                   "terminals": {"positive": "n", "negative": "0"}}),
        ),
        (
            "IPWL",
            json!({"name": "IPWL", "of": {"model": "current_source"},
                   "parameters": {"waveform": {"piecewise": {
                       "points": [[0.0, 0.0], [1e-6, 1e-3]]
                   }}},
                   "terminals": {"positive": "n", "negative": "0"}}),
        ),
        (
            "VEXP",
            json!({"name": "VEXP", "of": {"model": "voltage_source"},
                   "parameters": {"waveform": {"exponential": {
                       "initial": 0.0, "pulsed": 1.0, "rise_delay": 0.0,
                       "rise_time_constant": 1e-9, "fall_delay": 1e-6, "fall_time_constant": 1e-9
                   }}},
                   "terminals": {"positive": "n", "negative": "0"}}),
        ),
        (
            "BV1",
            json!({"name": "BV1", "of": {"model": "behavioral_voltage_source"},
                   "parameters": {"expression": "sin(time)"},
                   "terminals": {"positive": "n", "negative": "0"}}),
        ),
        (
            "BI1",
            json!({"name": "BI1", "of": {"model": "behavioral_current_source"},
                   "parameters": {"expression": "1e-3 * time"},
                   "terminals": {"positive": "n", "negative": "0"}}),
        ),
        (
            "BC1",
            json!({"name": "BC1", "of": {"model": "behavioral_capacitor"},
                   "parameters": {"expression": "1e-9 * v(x) + 1e-12 * time"},
                   "terminals": {"positive": "n", "negative": "0", "x": "n"}}),
        ),
    ];
    let advice = ["use", "pass", "try", "instead", "should", "give"];
    for (name, source) in cases {
        let mut simulator = simulator(&json!([
            source,
            two_terminal("R1", ("resistor", "resistance", 1e3), "n", "0"),
        ]));
        let error = simulator
            .run(
                &Analysis::PeriodicSteadyState(PeriodicSteadyState::default()),
                &Options::default(),
            )
            .expect_err("a time-varying source is refused");
        assert_eq!(
            error,
            SimulationError::InvalidAnalysis(format!(
                "an autonomous periodic steady state takes no source or behavioral device \
                 that varies with time: `{name}`"
            ))
        );
        let message = error.to_string().to_lowercase();
        assert!(!message.contains("did you mean"), "advice in `{message}`");
        for word in advice {
            let found = message
                .split(|character: char| !character.is_ascii_alphabetic())
                .any(|token| token == word);
            assert!(!found, "advice word `{word}` in `{message}`");
        }
        assert_eq!(simulator.operating_point_solves(), 0);
    }
}

#[test]
fn an_autonomous_periodic_steady_state_names_every_time_varying_device() {
    let mut simulator = simulator(&json!([
        {"name": "VDC", "of": {"model": "voltage_source"},
         "parameters": {"waveform": {"dc": 1.0}},
         "terminals": {"positive": "dc", "negative": "0"}},
        two_terminal("RDC", ("resistor", "resistance", 1e3), "dc", "0"),
        sine("V1", "a", (0.0, 1.0, 1e3)),
        two_terminal("R1", ("resistor", "resistance", 1e3), "a", "0"),
        {"name": "BI1", "of": {"model": "behavioral_current_source"},
         "parameters": {"expression": "1e-3 * time"},
         "terminals": {"positive": "b", "negative": "0"}},
        two_terminal("R2", ("resistor", "resistance", 1e3), "b", "0"),
    ]));
    let error = simulator
        .run(
            &Analysis::PeriodicSteadyState(PeriodicSteadyState::default()),
            &Options::default(),
        )
        .expect_err("multiple time-varying devices are refused");
    assert_eq!(
        error,
        SimulationError::InvalidAnalysis(
            "an autonomous periodic steady state takes no source or behavioral device \
             that varies with time: `V1`, `BI1`"
                .to_owned()
        )
    );
    assert_eq!(simulator.operating_point_solves(), 0);
}

#[test]
fn resetting_a_sine_amplitude_updates_autonomous_time_varying_check() {
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let regulation = (0.2 / impedance, impedance);
    let expected = TAU * (inductance * capacitance).sqrt();
    let mut instances = regulated_tank((inductance, capacitance), regulation, 1.0);
    instances.as_array_mut().expect("an instance list").extend([
        sine("VSIN", "vsin", (0.0, 0.0, 1e6)),
        two_terminal("RSIN", ("resistor", "resistance", 1e3), "vsin", "0"),
    ]);
    let mut simulator = simulator(&instances);
    let found = steady(
        &mut simulator,
        (None, expected / 100.0),
        &Options::default(),
    );
    assert!((found.period / expected - 1.0).abs() < 1e-2);

    let amplitude = simulator
        .elaborated()
        .parameter("VSIN", "waveform.sine.amplitude")
        .expect("sine amplitude parameter");
    let mut active = Values::default();
    active.set(amplitude, 1.0);
    simulator.reset(active).expect("reset with active sine");
    let error = simulator
        .run(
            &Analysis::PeriodicSteadyState(PeriodicSteadyState::default()),
            &Options::default(),
        )
        .expect_err("active sine is refused");
    assert_eq!(
        error,
        SimulationError::InvalidAnalysis(
            "an autonomous periodic steady state takes no source or behavioral device \
             that varies with time: `VSIN`"
                .to_owned()
        )
    );

    simulator
        .reset(Values::default())
        .expect("reset back to zero amplitude");
    let found = steady(
        &mut simulator,
        (None, expected / 100.0),
        &Options::default(),
    );
    assert!((found.period / expected - 1.0).abs() < 1e-2);
}

/// The `f64` count per GiB used by `max_memory` (`2^27`).
const NUMBERS_PER_GIB: f64 = 134_217_728.0;

/// A sine-driven RC low-pass filter whose resistance is split across `resistors` parallel
/// resistors of `resistors * 1 kΩ` each, keeping three unknowns (`in`, `out`, `V1:current`)
/// and the same transfer function while scaling parameter count linearly.
fn parallel_resistor_rc(resistors: usize) -> Value {
    let resistance = f64::from(u32::try_from(resistors).expect("small resistor count")) * 1e3;
    let mut instances = vec![
        sine("V1", "in", (0.0, 1.0, 1e3)),
        two_terminal("C1", ("capacitor", "capacitance", 1e-6), "out", "0"),
    ];
    for number in 1..=resistors {
        instances.push(two_terminal(
            &format!("R{number}"),
            ("resistor", "resistance", resistance),
            "in",
            "out",
        ));
    }
    Value::Array(instances)
}

#[test]
fn a_forward_periodic_sensitivity_past_max_memory_fails_before_replaying() {
    let instances = parallel_resistor_rc(1);
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        max_step: Some(1e-5),
        ..PeriodicSteadyState::default()
    };
    let unbounded = Options {
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let mut probe = simulator(&instances);
    let baseline = probe
        .periodic_steady_state(&analysis, &unbounded)
        .expect("baseline periodic steady state");
    let points = baseline.waveforms.values.len();
    let points_float = f64::from(u32::try_from(points).expect("small point count"));
    // Five parameters (sine offset/amplitude/frequency, C1, R1) and >100 points: shooting
    // selects forward mode (`along_period`).
    assert!(points * (points + 1) / 2 + points * points >= 2 * 5 * points);

    let tight_memory = (14.0 * 9.0 + (points_float + 2.0) * 3.0) / NUMBERS_PER_GIB;
    let tight = Options {
        max_memory: tight_memory,
        ..unbounded.clone()
    };
    let mut plain = simulator(&instances);
    plain
        .periodic_steady_state(&analysis, &tight)
        .expect("plain periodic steady state fits in tight memory");

    let sensitivity = Sensitivity {
        outputs: vec!["out".to_owned()],
        of: Differentiated::PeriodicSteadyState(analysis),
    };
    let mut candidate = simulator(&instances);
    let error = candidate
        .sensitivity(&sensitivity, &tight)
        .expect_err("forward periodic sensitivity exceeds tight memory");
    let waveforms = points_float * 3.0;
    let solve_phase = 14.0 * 9.0 + 5.0 * 3.0 + waveforms;
    let replay_phase = 9.0 + 4.0 * 5.0 * 3.0 + points_float * 5.0 + waveforms;
    let expected_needed = solve_phase.max(replay_phase) / NUMBERS_PER_GIB;
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state sensitivity",
            rows: 3,
            needed: expected_needed,
            max_memory: tight_memory,
        }
    );

    let sufficient = Options {
        max_memory: expected_needed * 1.05,
        ..unbounded
    };
    candidate
        .sensitivity(&sensitivity, &sufficient)
        .expect("forward periodic sensitivity succeeds once memory covers it");
}

#[test]
fn a_backward_periodic_sensitivity_past_max_memory_fails_before_allocating() {
    let instances = parallel_resistor_rc(60);
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        max_step: Some(5e-5),
        ..PeriodicSteadyState::default()
    };
    let unbounded = Options {
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let mut probe = simulator(&instances);
    let baseline = probe
        .periodic_steady_state(&analysis, &unbounded)
        .expect("baseline periodic steady state");
    let points = baseline.waveforms.values.len();
    let points_float = f64::from(u32::try_from(points).expect("small point count"));
    // 64 parameters (3 sine + C1 + 60 resistors) and ~24 points: shooting selects backward
    // mode (`by_rows`).
    assert!(points * (points + 1) / 2 + points * points < 2 * 64 * points);

    let tight_memory = (14.0 * 9.0 + (points_float + 2.0) * 3.0) / NUMBERS_PER_GIB;
    let tight = Options {
        max_memory: tight_memory,
        ..unbounded.clone()
    };
    let mut plain = simulator(&instances);
    plain
        .periodic_steady_state(&analysis, &tight)
        .expect("plain periodic steady state fits in tight memory");

    let sensitivity = Sensitivity {
        outputs: vec!["out".to_owned()],
        of: Differentiated::PeriodicSteadyState(analysis),
    };
    let mut candidate = simulator(&instances);
    let error = candidate
        .sensitivity(&sensitivity, &tight)
        .expect_err("backward periodic sensitivity exceeds tight memory");
    let waveforms = points_float * 3.0;
    let extra = points_float * 3.0 + points_float * 64.0;
    let solve_phase = 15.0 * 9.0 + points_float * 3.0;
    let step_peak = 8.0 * points_float * 3.0;
    let dot_peak = 6.0 * points_float * 3.0 + 3.0 * points_float * 64.0;
    let backward_phase = 9.0 + step_peak.max(dot_peak);
    let expected_needed = (waveforms + extra + solve_phase.max(backward_phase)) / NUMBERS_PER_GIB;
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state sensitivity",
            rows: 3,
            needed: expected_needed,
            max_memory: tight_memory,
        }
    );

    let sufficient = Options {
        max_memory: expected_needed * 1.05,
        ..unbounded
    };
    candidate
        .sensitivity(&sensitivity, &sufficient)
        .expect("backward periodic sensitivity succeeds once memory covers it");
}

#[test]
fn a_periodic_jvp_whose_columns_pass_max_memory_fails_before_shooting() {
    let instances = parallel_resistor_rc(1);
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        max_step: Some(5e-5),
        ..PeriodicSteadyState::default()
    };
    let unbounded = Options {
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let max_memory = 500.0 / NUMBERS_PER_GIB;
    let tight = Options {
        max_memory,
        ..unbounded.clone()
    };
    let mut plain = simulator(&instances);
    let baseline = plain
        .periodic_steady_state(&analysis, &tight)
        .expect("plain periodic steady state fits in tight memory");
    let points = baseline.waveforms.values.len();
    let points_float = f64::from(u32::try_from(points).expect("small point count"));

    let parameters = vec![("R1".to_owned(), "resistance".to_owned())];
    let tangents = vec![vec![1.0]; 200];
    let mut candidate = simulator(&instances);
    let error = candidate
        .jvp(
            &Differentiated::PeriodicSteadyState(analysis.clone()),
            &parameters,
            &tangents,
            &tight,
        )
        .expect_err("jvp columns exceed tight memory upfront");
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state sensitivity",
            rows: 3,
            needed: (14.0 * 9.0 + 200.0 * 3.0) / NUMBERS_PER_GIB,
            max_memory,
        }
    );
    assert_eq!(candidate.operating_point_solves(), 0);

    // Allowing the bordered solve (726) plus waveforms, but not the replay output matrix
    // (points * 3 unknowns * 200 directions), fails inside `along_period` before replaying.
    let mid_memory = 1000.0 / NUMBERS_PER_GIB;
    let mid = Options {
        max_memory: mid_memory,
        ..unbounded.clone()
    };
    let replay_error = candidate
        .jvp(
            &Differentiated::PeriodicSteadyState(analysis.clone()),
            &parameters,
            &tangents,
            &mid,
        )
        .expect_err("jvp replay outputs exceed mid memory");
    let expected_replay =
        (9.0 + 4.0 * 200.0 * 3.0 + points_float * 3.0 * 200.0 + points_float * 3.0)
            / NUMBERS_PER_GIB;
    assert_eq!(
        replay_error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state sensitivity",
            rows: 3,
            needed: expected_replay,
            max_memory: mid_memory,
        }
    );

    let sufficient = Options {
        max_memory: expected_replay * 1.05,
        ..unbounded
    };
    candidate
        .jvp(
            &Differentiated::PeriodicSteadyState(analysis),
            &parameters,
            &tangents,
            &sufficient,
        )
        .expect("jvp succeeds once memory covers replay outputs");
}

#[test]
fn a_periodic_vjp_pull_back_past_max_memory_fails_before_allocating() {
    let instances = parallel_resistor_rc(60);
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        max_step: Some(5e-5),
        ..PeriodicSteadyState::default()
    };
    let unbounded = Options {
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let mut probe = simulator(&instances);
    let baseline = probe
        .periodic_steady_state(&analysis, &unbounded)
        .expect("baseline periodic steady state");
    let points = baseline.waveforms.values.len();
    let points_float = f64::from(u32::try_from(points).expect("small point count"));
    let tight_memory = (14.0 * 9.0 + (points_float + 2.0) * 3.0) / NUMBERS_PER_GIB;
    let tight = Options {
        max_memory: tight_memory,
        ..unbounded.clone()
    };

    let parameters: Vec<(String, String)> = (1..=60)
        .map(|number| (format!("R{number}"), "resistance".to_owned()))
        .collect();
    let mut candidate = simulator(&instances);
    let pullback = candidate
        .vjp(
            &Differentiated::PeriodicSteadyState(analysis),
            &parameters,
            &tight,
        )
        .expect("vjp fits within tight memory");
    let out = index(&candidate, "out");
    let mut weight = vec![0.0; 3];
    weight[out] = 1.0;
    let cotangents = Cotangents::Real(vec![Cotangent {
        points: BTreeMap::from([(points - 1, weight)]),
        period: 0.0,
    }]);
    let error = candidate
        .pull_back(&pullback, &cotangents, &tight)
        .expect_err("pull_back exceeds tight memory");
    let waveforms = points_float * 3.0;
    let extra = 60.0;
    let solve_phase: f64 = 15.0 * 9.0 + 3.0;
    let step_peak: f64 = 8.0 * 3.0;
    let dot_peak: f64 = 6.0 * 3.0 + 3.0 * 60.0;
    let backward_phase = 9.0 + step_peak.max(dot_peak);
    let expected_needed = (waveforms + extra + solve_phase.max(backward_phase)) / NUMBERS_PER_GIB;
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state sensitivity",
            rows: 3,
            needed: expected_needed,
            max_memory: tight_memory,
        }
    );

    let sufficient = Options {
        max_memory: expected_needed * 1.05,
        ..unbounded
    };
    candidate
        .pull_back(&pullback, &cotangents, &sufficient)
        .expect("pull_back succeeds once memory covers it");
}

#[test]
fn an_autonomous_periodic_sensitivity_checks_validity_and_upfront_memory() {
    let tiny = Options {
        max_memory: 1e-9,
        ..Options::default()
    };
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let regulation = (0.2 / impedance, impedance);
    let mut oscillator = simulator(&regulated_tank((inductance, capacitance), regulation, 1.0));
    let error = oscillator
        .sensitivity(
            &Sensitivity {
                outputs: vec!["t".to_owned()],
                of: Differentiated::PeriodicSteadyState(PeriodicSteadyState::default()),
            },
            &tiny,
        )
        .expect_err("bordered oscillator sensitivity exceeds tiny memory");
    assert_eq!(
        error,
        SimulationError::TooMuchMemory {
            analysis: "periodic steady state sensitivity",
            rows: 5,
            needed: (14.0 * 25.0 + 2.0 * 5.0) / NUMBERS_PER_GIB,
            max_memory: 1e-9,
        }
    );
    assert_eq!(oscillator.operating_point_solves(), 0);

    let mut driven = simulator(&json!([
        sine("V1", "n", (0.0, 1.0, 1e3)),
        two_terminal("R1", ("resistor", "resistance", 1e3), "n", "0"),
    ]));
    let expected = SimulationError::InvalidAnalysis(
        "an autonomous periodic steady state takes no source or behavioral device \
         that varies with time: `V1`"
            .to_owned(),
    );
    let sensitivity_error = driven
        .sensitivity(
            &Sensitivity {
                outputs: vec!["n".to_owned()],
                of: Differentiated::PeriodicSteadyState(PeriodicSteadyState::default()),
            },
            &tiny,
        )
        .expect_err("time-varying source check runs before memory check in sensitivity");
    assert_eq!(sensitivity_error, expected);
    let jvp_error = driven
        .jvp(
            &Differentiated::PeriodicSteadyState(PeriodicSteadyState::default()),
            &[("R1".to_owned(), "resistance".to_owned())],
            &[vec![1.0]],
            &tiny,
        )
        .expect_err("time-varying source check runs before memory check in jvp");
    assert_eq!(jvp_error, expected);
    assert_eq!(driven.operating_point_solves(), 0);
}
