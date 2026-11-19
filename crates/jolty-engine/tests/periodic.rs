//! The periodic steady state against closed form and against long transients.
#![cfg(test)]

#[path = "common/capacitor.rs"]
mod capacitor;
mod common;
#[path = "common/dc.rs"]
mod dc;
#[path = "common/inductor.rs"]
mod inductor;
#[path = "common/sim.rs"]
mod sim;
#[path = "common/sine.rs"]
mod sine;
#[path = "common/transient.rs"]
mod transient;

use std::collections::BTreeMap;
use std::f64::consts::TAU;

use capacitor::capacitor;
use common::{circuit, device, resistor, voltage_source};
use dc::voltage_dc;
use inductor::inductor;
use jolty_engine::{
    Analysis, AnalysisResult, Cotangent, Cotangents, Differentiated, Method, Options,
    PeriodicSteadyState, PeriodicSteadyStateResult, Sensitivity, SimulationError, Simulator,
    TransientResult, Values,
};
use jolty_ir::{Circuit, Instance};
use serde_json::{Value, json};
use sim::simulator;
use sine::voltage_sine;
use transient::transient;

fn simulate(instances: impl IntoIterator<Item = Instance>) -> Simulator {
    simulator(&circuit(instances))
}

/// Runs a trapezoidal transient until `stop` with steps capped at `longest`.
fn run(simulator: &mut Simulator, (stop, longest): (f64, f64)) -> TransientResult {
    run_with(simulator, (stop, longest), &Options::default())
}

fn run_with(
    simulator: &mut Simulator,
    (stop, longest): (f64, f64),
    options: &Options,
) -> TransientResult {
    let analysis = transient(stop, Some(longest), Method::Trapezoidal);
    match simulator.run(&Analysis::Transient(analysis), options) {
        Ok(AnalysisResult::Transient(waveforms)) => waveforms,
        other => panic!("a transient outcome, not {other:?}"),
    }
}

/// Computes the periodic steady state for `period` (or an autonomous oscillator when `None`)
/// with steps capped at `longest`.
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

/// Returns `(min, max)` of `unknown` across all points of `waveforms` at or after `from`.
fn waveform_range(waveforms: &TransientResult, from: f64, unknown: usize) -> (f64, f64) {
    waveforms
        .times
        .iter()
        .zip(&waveforms.values)
        .filter(|(time, _)| **time >= from)
        .map(|(_, values)| values[unknown])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| {
            (low.min(value), high.max(value))
        })
}

#[test]
fn a_sine_through_a_slow_rc_repeats_its_closed_form_at_once() {
    // The time constant is ten periods, so a transient from rest would take fifty to
    // settle within 1%; the circuit is linear, so one Newton step of shooting lands on the
    // steady state, and the second period confirms it.
    let (resistance, capacitance, frequency) = (1e3, 1e-5, 1e3);
    let mut simulator = simulate([
        voltage_sine("V1", "in", 0.0, 1.0, frequency),
        resistor("R1", resistance, "in", "out"),
        capacitor("C1", capacitance, "out", "0"),
    ]);
    let found = steady(
        &mut simulator,
        (Some(1.0 / frequency), 5e-3 / frequency),
        &Options::default(),
    );
    assert_eq!(found.periods, 2);
    let waveforms = found.transient;
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
    let mut simulator = simulate([
        voltage_sine("V1", "in", 0.5, 1.0, frequency),
        resistor("R1", resistance, "in", "a"),
        inductor("L1", inductance, "a", "0"),
    ]);
    let found = steady(
        &mut simulator,
        (Some(1.0 / frequency), 5e-3 / frequency),
        &Options::default(),
    );
    assert_eq!(found.periods, 2);
    let waveforms = found.transient;
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
    let built = circuit([
        voltage_sine("V1", "in", 0.2, 1.0, frequency),
        resistor("R1", 1e3, "in", "a"),
        device(
            "D1",
            "diode",
            json!({"saturation_current": 1e-14, "series_resistance": 10.0,
                   "transit_time": 1e-9, "junction_capacitance": 2e-12}),
            &[("anode", "a"), ("cathode", "out")],
        ),
        capacitor("C1", 1e-9, "out", "0"),
        resistor("R2", 1e4, "out", "0"),
    ]);
    let options = Options {
        relative_tolerance: 1e-6,
        ..Options::default()
    };
    let mut shooting = simulator(&built);
    let longest = 5e-3 / frequency;
    let found = steady(&mut shooting, (Some(1.0 / frequency), longest), &options);
    assert!(found.periods <= 5, "{} periods", found.periods);
    let steady = found.transient;
    // Fifteen time constants of the load.
    let settled = 150;
    let mut long = simulator(&built);
    let until = f64::from(settled) / frequency;
    let transient = run_with(&mut long, (until, longest), &options);
    let out = index(&shooting, "out");
    let (low, high) = waveform_range(&steady, 0.0, out);
    let (settled_low, settled_high) =
        waveform_range(&transient, f64::from(settled - 1) / frequency, out);
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

/// Builds a 1 kHz sine source of `amplitude` driving `memristor` into a 100 nF capacitor
/// with initial memristor state 0.5.
fn memristor_into_a_capacitor(memristor: (&str, Value), amplitude: f64) -> Circuit {
    let (model, mut parameters) = memristor;
    parameters["on_resistance"] = json!(1e3);
    parameters["off_resistance"] = json!(1e4);
    parameters["state"] = json!(0.5);
    circuit([
        voltage_sine("V1", "in", 0.0, amplitude, 1e3),
        device(
            "M1",
            model,
            parameters,
            &[("positive", "in"), ("negative", "out")],
        ),
        capacitor("C1", 1e-7, "out", "0"),
    ])
}

#[test]
fn a_memristor_its_drive_leaves_alone_keeps_its_state() {
    // The sine stays within the thresholds, so the state never moves, and every state is
    // as periodic as any other: Φ - I is singular, its row for the state zero. Shooting
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
    let waveforms = found.transient;
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
    // Charge q through the memristor shifts state by q / Q and capacitor voltage by q / C,
    // leaving Q·x - C·v(out) invariant per period. Each invariant value admits a distinct
    // steady state, and shooting preserves the operating-point invariant.
    let (charge, capacitance, frequency) = (1e-5, 1e-7, 1e3);
    let memristor = json!({"charge": charge});
    let built = memristor_into_a_capacitor(("current_controlled_memristor", memristor), 1.0);
    let longest = 5e-3 / frequency;
    let mut shooting = simulator(&built);
    let found = steady(
        &mut shooting,
        (Some(1.0 / frequency), longest),
        &Options::default(),
    );
    assert!(found.periods <= 4, "{} periods", found.periods);
    let steady = found.transient;
    let (state, out) = (index(&shooting, "M1:state"), index(&shooting, "out"));
    for values in &steady.values {
        let shared = charge * values[state] - capacitance * values[out];
        assert!((shared - charge * 0.5).abs() < 1e-12 * charge, "{shared}");
    }
    let settled = 20;
    let mut long = simulator(&built);
    let transient = run(&mut long, (f64::from(settled) / frequency, longest));
    for (unknown, tolerance) in [(state, 1e-6), (out, 1e-4)] {
        let (low, high) = waveform_range(&steady, 0.0, unknown);
        let last = f64::from(settled - 1) / frequency;
        let (settled_low, settled_high) = waveform_range(&transient, last, unknown);
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
    let mut simulator = simulate([
        voltage_sine("V1", "in", 0.0, 1.0, 1e3),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-6, "out", "0"),
    ]);
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        ..PeriodicSteadyState::default()
    };
    let options = with_memory(&Options::default(), 1e-9);
    let error = simulator
        .run(&Analysis::PeriodicSteadyState(analysis), &options)
        .expect_err("too little memory");
    assert_memory_limit(
        &error,
        "periodic steady state",
        3,
        14.0 * 9.0 / NUMBERS_PER_GIB,
        1e-9,
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
    let mut simulator = simulate([device(
        "A",
        "logic_source",
        json!({"initial": "0", "changes": [], "transition": 1e-12}),
        &[("output", "a")],
    )]);
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

/// Builds an LC tank regulated by nonlinear conductance `-g (1 - r² / a²)` where
/// `r² = v² + (z i)²`.
///
/// When `z = √(L/C)`, the tank is lossless on the circle of radius `a`, yielding an exact
/// periodic orbit of period `2π √(LC)`.
fn regulated_tank(
    (inductance, capacitance): (f64, f64),
    (conductance, impedance): (f64, f64),
    radius: f64,
) -> Vec<Instance> {
    let expression = format!(
        "-{conductance:e} * v(x) * (1 - (v(x) ^ 2 + ({impedance:e} * i(p)) ^ 2) / {:e})",
        radius * radius
    );
    vec![
        capacitor("C1", capacitance, "t", "0"),
        inductor("L1", inductance, "t", "l"),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": expression}),
            &[
                ("positive", "t"),
                ("negative", "0"),
                ("x", "t"),
                ("p_positive", "l"),
                ("p_negative", "0"),
            ],
        ),
    ]
}

/// Builds the default 1 µH / 1 nF regulated tank with unit radius and `g = 0.2 / √(L/C)`,
/// returning its instances and exact natural period `2π √(LC)`.
fn default_tank() -> (Vec<Instance>, f64) {
    let (inductance, capacitance): (f64, f64) = (1e-6, 1e-9);
    let impedance = (inductance / capacitance).sqrt();
    let expected = TAU * (inductance * capacitance).sqrt();
    (
        regulated_tank((inductance, capacitance), (0.2 / impedance, impedance), 1.0),
        expected,
    )
}

#[test]
fn an_oscillating_tank_finds_its_period_and_its_amplitude() {
    // The conductance at rest, -g, grows the tank's mode at σ = g / 2C; on the circle the
    // radius relaxes at g / C, so a period contracts its error by e^(-2π g / ωC).
    let (inductance, capacitance, radius): (f64, f64, f64) = (1e-6, 1e-9, 1.0);
    let impedance = (inductance / capacitance).sqrt();
    let omega = 1.0 / (inductance * capacitance).sqrt();
    let g = 0.2 / impedance;
    let options = Options {
        relative_tolerance: 1e-6,
        ..Options::default()
    };
    let mut simulator = simulate(regulated_tank(
        (inductance, capacitance),
        (g, impedance),
        radius,
    ));
    let found = steady(&mut simulator, (None, TAU / omega / 100.0), &options);
    // From where the transient settled, Newton's steps square the mismatch, 6e4
    // tolerances of the current after the first period and 50 after the second.
    assert_eq!(found.periods, 3);
    let (voltage, current) = (index(&simulator, "t"), index(&simulator, "L1:current"));
    let (times, values) = (&found.transient.times, &found.transient.values);
    // Each period starts with two backward Euler steps until the four-point trapezoidal
    // error estimator fills. Each backward Euler step h retains 1 / (1 + (ωh)²) of tank
    // energy, reducing radius by at most (ωh)² (C / 2gT + 1 / 2). A start repeating within
    // tolerance stays within tolerance / (1 - e^(-2π g / ωC)) of the orbit.
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
    // Trapezoidal steps advance phase by 2 atan(ωh / 2), lagging ωh by (ωh)³ / 12 to leading
    // order, so the period exceeds 2π / ω by at most (ωh)² / 12. Each backward Euler step
    // adds (ωh)³ / 3 lag, and closure tolerance shifts phase by at most tolerance / radius.
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
            capacitor("C1", 1e-9, "t", "0"),
            inductor("L1", 1e-6, "t", "0"),
        ];
        if damped {
            instances.push(resistor("R1", 1e3, "t", "0"));
        }
        let mut simulator = simulate(instances);
        let analysis = Analysis::PeriodicSteadyState(PeriodicSteadyState::default());
        let error = simulator
            .run(&analysis, &Options::default())
            .expect_err("no oscillation starts");
        assert_eq!(error, SimulationError::NoGrowingMode);
    }
}

#[test]
fn an_oscillators_period_is_found_so_it_has_no_times_to_step_onto() {
    let mut simulator = simulate(regulated_tank((1e-6, 1e-9), (6e-3, 31.6), 1.0));
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
    // 2 mS saturating tanh negative conductance against 1 kΩ and 1 nF: the balanced state
    // grows at 1e6 per second and latches to ±1.9 V without returning across the kick.
    let mut simulator = simulate([
        resistor("R1", 1e3, "n", "0"),
        capacitor("C1", 1e-9, "n", "0"),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": "-2e-3 * tanh(v(x))"}),
            &[("positive", "n"), ("negative", "0"), ("x", "n")],
        ),
    ]);
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
    // Scaling time, L, and C identically leaves tank equations invariant with fixed impedance
    // z, so L ∂T/∂L + C ∂T/∂C = T holds discretely up to 256 ε (~1e-12) tangent and bordered
    // solve rounding. Infinite truncation tolerance locks step fractions across runs, so
    // fresh central differences match within δ²/8 square-root truncation error plus
    // 1e-10 / δ period resolution.
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
        simulate(regulated_tank(tank, regulation, 1.0))
            .periodic_steady_state(&analysis, &options)
            .expect("the tank oscillates")
            .period
    };
    let mut simulator = simulate(regulated_tank((inductance, capacitance), regulation, 1.0));
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
    let (mut instances, expected) = default_tank();
    instances.extend([
        voltage_dc("VDC", "vdc", 3.3),
        resistor("RDC", 1e3, "vdc", "0"),
        voltage_sine("VSIN0", "vsin0", 1.0, 0.0, 1e6),
        resistor("RSIN0", 1e3, "vsin0", "0"),
        voltage_source(
            "VP0",
            "vp0",
            "0",
            json!({"pulse": {
                "initial": 2.0, "pulsed": 2.0, "rise": 1e-9, "fall": 1e-9, "width": 1e-8
            }}),
        ),
        resistor("RP0", 1e3, "vp0", "0"),
        device(
            "IPWL0",
            "current_source",
            json!({"waveform": {"piecewise": {"points": [[0.0, 1e-3], [1e-6, 1e-3]]}}}),
            &[("positive", "ipwl0"), ("negative", "0")],
        ),
        resistor("RPWL0", 1e3, "ipwl0", "0"),
        voltage_source(
            "VEXP0",
            "vexp0",
            "0",
            json!({"exponential": {
                "initial": 1.0, "pulsed": 1.0, "rise_delay": 0.0,
                "rise_time_constant": 1e-9, "fall_delay": 1e-6, "fall_time_constant": 1e-9
            }}),
        ),
        resistor("REXP0", 1e3, "vexp0", "0"),
    ]);
    let mut simulator = simulate(instances);
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
        ("V1", voltage_sine("V1", "n", 0.0, 1.0, 1e3)),
        (
            "VP",
            voltage_source(
                "VP",
                "n",
                "0",
                json!({"pulse": {
                    "initial": 0.0, "pulsed": 1.0, "rise": 1e-9, "fall": 1e-9, "width": 1e-6
                }}),
            ),
        ),
        (
            "IPWL",
            device(
                "IPWL",
                "current_source",
                json!({"waveform": {"piecewise": {"points": [[0.0, 0.0], [1e-6, 1e-3]]}}}),
                &[("positive", "n"), ("negative", "0")],
            ),
        ),
        (
            "VEXP",
            voltage_source(
                "VEXP",
                "n",
                "0",
                json!({"exponential": {
                    "initial": 0.0, "pulsed": 1.0, "rise_delay": 0.0,
                    "rise_time_constant": 1e-9, "fall_delay": 1e-6, "fall_time_constant": 1e-9
                }}),
            ),
        ),
        (
            "BV1",
            device(
                "BV1",
                "behavioral_voltage_source",
                json!({"expression": "sin(time)"}),
                &[("positive", "n"), ("negative", "0")],
            ),
        ),
        (
            "BI1",
            device(
                "BI1",
                "behavioral_current_source",
                json!({"expression": "1e-3 * time"}),
                &[("positive", "n"), ("negative", "0")],
            ),
        ),
        (
            "BC1",
            device(
                "BC1",
                "behavioral_capacitor",
                json!({"expression": "1e-9 * v(x) + 1e-12 * time"}),
                &[("positive", "n"), ("negative", "0"), ("x", "n")],
            ),
        ),
    ];
    let advice = ["use", "pass", "try", "instead", "should", "give"];
    for (name, source) in cases {
        let mut simulator = simulate([source, resistor("R1", 1e3, "n", "0")]);
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
    let mut simulator = simulate([
        voltage_dc("VDC", "dc", 1.0),
        resistor("RDC", 1e3, "dc", "0"),
        voltage_sine("V1", "a", 0.0, 1.0, 1e3),
        resistor("R1", 1e3, "a", "0"),
        device(
            "BI1",
            "behavioral_current_source",
            json!({"expression": "1e-3 * time"}),
            &[("positive", "b"), ("negative", "0")],
        ),
        resistor("R2", 1e3, "b", "0"),
    ]);
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
    let (mut instances, expected) = default_tank();
    instances.extend([
        voltage_sine("VSIN", "vsin", 0.0, 0.0, 1e6),
        resistor("RSIN", 1e3, "vsin", "0"),
    ]);
    let mut simulator = simulate(instances);
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

/// Returns `base` with its `max_memory` cap set to `max_memory` GiB.
fn with_memory(base: &Options, max_memory: f64) -> Options {
    Options {
        max_memory,
        ..base.clone()
    }
}

/// Asserts that `error` is [`SimulationError::TooMuchMemory`] with the exact expected
/// analysis name, row count, required GiB, and cap.
fn assert_memory_limit(
    error: &SimulationError,
    analysis: &'static str,
    rows: usize,
    needed: f64,
    max_memory: f64,
) {
    assert_eq!(
        *error,
        SimulationError::TooMuchMemory {
            analysis,
            rows,
            needed,
            max_memory,
        }
    );
}

/// Builds a sine-driven RC low-pass filter splitting resistance across `resistors` parallel
/// branches of `resistors * 1 kΩ` each, keeping three unknowns while scaling parameter count
/// linearly.
fn parallel_resistor_rc(resistors: usize) -> Circuit {
    let resistance = f64::from(u32::try_from(resistors).expect("small resistor count")) * 1e3;
    let mut instances = vec![
        voltage_sine("V1", "in", 0.0, 1.0, 1e3),
        capacitor("C1", 1e-6, "out", "0"),
    ];
    for number in 1..=resistors {
        instances.push(resistor(&format!("R{number}"), resistance, "in", "out"));
    }
    circuit(instances)
}

/// Builds a `parallel_resistor_rc` memory probe: runs a baseline periodic steady state
/// with unbounded truncation tolerance, computes `tight_memory` just covering plain
/// shooting, and checks that plain shooting fits within `tight_memory`.
fn rc_memory_probe(
    resistors: usize,
    max_step: f64,
) -> (Circuit, PeriodicSteadyState, Options, usize, f64) {
    let built = parallel_resistor_rc(resistors);
    let analysis = PeriodicSteadyState {
        period: Some(1e-3),
        max_step: Some(max_step),
        ..PeriodicSteadyState::default()
    };
    let unbounded = Options {
        truncation_tolerance: f64::INFINITY,
        ..Options::default()
    };
    let mut probe = simulator(&built);
    let baseline = probe
        .periodic_steady_state(&analysis, &unbounded)
        .expect("baseline periodic steady state");
    let points = baseline.transient.values.len();
    let points_float = f64::from(u32::try_from(points).expect("small point count"));
    let tight_memory = (14.0 * 9.0 + (points_float + 2.0) * 3.0) / NUMBERS_PER_GIB;
    let tight = with_memory(&unbounded, tight_memory);
    let mut plain = simulator(&built);
    plain
        .periodic_steady_state(&analysis, &tight)
        .expect("plain periodic steady state fits in tight memory");
    (built, analysis, unbounded, points, tight_memory)
}

/// Checks that periodic steady-state sensitivity on `parallel_resistor_rc(resistors)`
/// fails with `expected_needed` at tight memory (just enough for plain shooting) and
/// succeeds once `max_memory` covers `expected_needed * 1.05`.
fn check_sensitivity_memory(
    resistors: usize,
    max_step: f64,
    expect_forward: bool,
    expected_needed: impl FnOnce(f64) -> f64,
) {
    let (built, analysis, unbounded, points, tight_memory) = rc_memory_probe(resistors, max_step);
    let parameters = 4 + resistors;
    let forward_cost = points * (points + 1) / 2 + points * points;
    let backward_cost = 2 * parameters * points;
    assert_eq!(forward_cost >= backward_cost, expect_forward);

    let sensitivity = Sensitivity {
        outputs: vec!["out".to_owned()],
        of: Differentiated::PeriodicSteadyState(analysis),
    };
    let mut candidate = simulator(&built);
    let error = candidate
        .sensitivity(&sensitivity, &with_memory(&unbounded, tight_memory))
        .expect_err("periodic sensitivity exceeds tight memory");
    let points_float = f64::from(u32::try_from(points).expect("small point count"));
    let needed = expected_needed(points_float);
    assert_memory_limit(
        &error,
        "periodic steady state sensitivity",
        3,
        needed,
        tight_memory,
    );
    candidate
        .sensitivity(&sensitivity, &with_memory(&unbounded, needed * 1.05))
        .expect("periodic sensitivity succeeds once memory covers it");
}

#[test]
fn a_forward_periodic_sensitivity_past_max_memory_fails_before_replaying() {
    // Five parameters (sine offset/amplitude/frequency, C1, R1) and >100 points: shooting
    // selects forward mode (`along_period`).
    check_sensitivity_memory(1, 1e-5, true, |points| {
        let waveforms = points * 3.0;
        let solve_phase = 14.0 * 9.0 + 5.0 * 3.0 + waveforms;
        let replay_phase = 9.0 + 4.0 * 5.0 * 3.0 + points * 5.0 + waveforms;
        solve_phase.max(replay_phase) / NUMBERS_PER_GIB
    });
}

#[test]
fn a_backward_periodic_sensitivity_past_max_memory_fails_before_allocating() {
    // 64 parameters (3 sine + C1 + 60 resistors) and ~24 points: shooting selects backward
    // mode (`by_rows`).
    check_sensitivity_memory(60, 5e-5, false, |points| {
        let waveforms = points * 3.0;
        let extra = points * 3.0 + points * 64.0;
        let solve_phase = 15.0 * 9.0 + points * 3.0;
        let step_peak = 8.0 * points * 3.0;
        let dot_peak = 6.0 * points * 3.0 + 3.0 * points * 64.0;
        let backward_phase = 9.0 + step_peak.max(dot_peak);
        (waveforms + extra + solve_phase.max(backward_phase)) / NUMBERS_PER_GIB
    });
}

#[test]
fn a_periodic_jvp_whose_columns_pass_max_memory_fails_before_shooting() {
    let built = parallel_resistor_rc(1);
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
    let tight = with_memory(&unbounded, max_memory);
    let mut plain = simulator(&built);
    let baseline = plain
        .periodic_steady_state(&analysis, &tight)
        .expect("plain periodic steady state fits in tight memory");
    let points = baseline.transient.values.len();
    let points_float = f64::from(u32::try_from(points).expect("small point count"));

    let parameters = vec![("R1".to_owned(), "resistance".to_owned())];
    let tangents = vec![vec![1.0]; 200];
    let mut candidate = simulator(&built);
    let error = candidate
        .jvp(
            &Differentiated::PeriodicSteadyState(analysis.clone()),
            &parameters,
            &tangents,
            &tight,
        )
        .expect_err("jvp columns exceed tight memory upfront");
    assert_memory_limit(
        &error,
        "periodic steady state sensitivity",
        3,
        (14.0 * 9.0 + 200.0 * 3.0) / NUMBERS_PER_GIB,
        max_memory,
    );
    assert_eq!(candidate.operating_point_solves(), 0);

    // Allowing the bordered solve (726) plus waveforms, but not the replay output matrix
    // (points * 3 unknowns * 200 directions), fails inside `along_period` before replaying.
    let mid_memory = 1000.0 / NUMBERS_PER_GIB;
    let replay_error = candidate
        .jvp(
            &Differentiated::PeriodicSteadyState(analysis.clone()),
            &parameters,
            &tangents,
            &with_memory(&unbounded, mid_memory),
        )
        .expect_err("jvp replay outputs exceed mid memory");
    let expected_replay =
        (9.0 + 4.0 * 200.0 * 3.0 + points_float * 3.0 * 200.0 + points_float * 3.0)
            / NUMBERS_PER_GIB;
    assert_memory_limit(
        &replay_error,
        "periodic steady state sensitivity",
        3,
        expected_replay,
        mid_memory,
    );

    candidate
        .jvp(
            &Differentiated::PeriodicSteadyState(analysis),
            &parameters,
            &tangents,
            &with_memory(&unbounded, expected_replay * 1.05),
        )
        .expect("jvp succeeds once memory covers replay outputs");
}

#[test]
fn a_periodic_vjp_pull_back_past_max_memory_fails_before_allocating() {
    let (built, analysis, unbounded, points, tight_memory) = rc_memory_probe(60, 5e-5);
    let tight = with_memory(&unbounded, tight_memory);
    let points_float = f64::from(u32::try_from(points).expect("small point count"));

    let parameters: Vec<(String, String)> = (1..=60)
        .map(|number| (format!("R{number}"), "resistance".to_owned()))
        .collect();
    let mut candidate = simulator(&built);
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
    assert_memory_limit(
        &error,
        "periodic steady state sensitivity",
        3,
        expected_needed,
        tight_memory,
    );

    candidate
        .pull_back(
            &pullback,
            &cotangents,
            &with_memory(&unbounded, expected_needed * 1.05),
        )
        .expect("pull_back succeeds once memory covers it");
}

#[test]
fn an_autonomous_periodic_sensitivity_checks_validity_and_upfront_memory() {
    let tiny = with_memory(&Options::default(), 1e-9);
    let (tank, _) = default_tank();
    let mut oscillator = simulate(tank);
    let error = oscillator
        .sensitivity(
            &Sensitivity {
                outputs: vec!["t".to_owned()],
                of: Differentiated::PeriodicSteadyState(PeriodicSteadyState::default()),
            },
            &tiny,
        )
        .expect_err("bordered oscillator sensitivity exceeds tiny memory");
    assert_memory_limit(
        &error,
        "periodic steady state sensitivity",
        5,
        (14.0 * 25.0 + 2.0 * 5.0) / NUMBERS_PER_GIB,
        1e-9,
    );
    assert_eq!(oscillator.operating_point_solves(), 0);

    let mut driven = simulate([
        voltage_sine("V1", "n", 0.0, 1.0, 1e3),
        resistor("R1", 1e3, "n", "0"),
    ]);
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
