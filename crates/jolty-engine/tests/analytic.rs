//! Every analysis against circuits whose answers are known in closed form.
#![cfg(test)]

#[path = "common/capacitor.rs"]
mod capacitor;
mod common;
#[path = "common/current.rs"]
mod current;
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
use std::f64::consts::{PI, TAU};

use capacitor::capacitor;
use common::{circuit, device, resistor, two_terminal, voltage_source};
use current::{current_dc, current_source};
use dc::{dc, voltage_dc};
use inductor::inductor;
use jolty_engine::{
    Ac, Analysis, AnalysisResult, DcSweep, Method, Options, Rejection, SimulationError, Simulator,
    Strategy, Transient,
};
use jolty_ir::{Circuit, Instance};
use jolty_numerics::Complex64;
use serde_json::{Value, json};
use sim::simulator;
use sine::voltage_sine;
use transient::transient;

/// The thermal voltage at 27 °C.
const THERMAL_VOLTAGE: f64 = 1.380_649e-23 * 300.15 / 1.602_176_634e-19;

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

/// Builds a default-parameter diode named `name` from `anode` to `cathode`.
fn diode(name: &str, anode: &str, cathode: &str) -> Instance {
    device(
        name,
        "diode",
        json!({}),
        &[("anode", anode), ("cathode", cathode)],
    )
}

/// Builds a piecewise-linear waveform through `points`.
fn piecewise(points: &[[f64; 2]]) -> Value {
    json!({"piecewise": {"points": points}})
}

/// Builds a default trapezoidal transient over `[0, stop]` with optional `max_step`.
fn trapezoidal(stop: f64, max_step: Option<f64>) -> Transient {
    transient(stop, max_step, Method::Trapezoidal)
}

#[test]
fn a_micro_ohm_beside_a_node_only_gmin_holds_keeps_it() {
    // An idle source feeds 1 µΩ into a diode reverse-biased from 2 V: only the diode's gmin
    // holds the nodes, a millionth of a millionth of the resistor's conductance.
    let circuit = circuit([
        voltage_dc("V1", "supply", 2.0),
        current_dc("I1", "0", "a", 0.0),
        resistor("R1", 1e-6, "a", "b"),
        diode("D1", "b", "supply"),
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
    let circuit = circuit([
        voltage_dc("V1", "in", 5.0),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 1e3, "out", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_eq!(solution.strategy, Strategy::Direct);
    // Newton iteration steps to a linear circuit's solution in one iteration, and a
    // second, whose step lies within tolerance, confirms it.
    assert_eq!(solution.iterations, 2);
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
    let circuit = circuit([
        voltage_dc("V1", "in", 400.0),
        resistor("R1", 1e-3, "in", "out"),
        resistor("R2", 1e6, "out", "0"),
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
    let circuit = circuit([
        current_dc("I1", "0", "out", 1e-3),
        resistor("R1", 2e3, "out", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_close(solution.values[index(&simulator, "out")], 2.0, 1e-12);
}

#[test]
fn controlled_sources_amplify_their_control_voltage() {
    let circuit = circuit([
        voltage_dc("V1", "in", 0.1),
        device(
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
        device(
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
        resistor("R1", 1e4, "driven", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let solution = simulator.operating_point(&Options::default()).unwrap();
    assert_close(solution.values[index(&simulator, "amplified")], 1.0, 1e-12);
    assert_close(solution.values[index(&simulator, "driven")], 1.0, 1e-12);
}

/// Solves the closed-form resistor-diode load-line current via bisection.
fn diode_current(source: f64, resistance: f64, saturation: f64) -> f64 {
    let (mut low, mut high) = (0.0, source / resistance);
    for _ in 0..200 {
        let current = f64::midpoint(low, high);
        let voltage = THERMAL_VOLTAGE * (current / saturation + 1.0).ln();
        if voltage + current * resistance > source {
            high = current;
        } else {
            low = current;
        }
    }
    f64::midpoint(low, high)
}

#[test]
fn a_diode_settles_where_its_curve_meets_the_load_line() {
    let circuit = circuit([
        voltage_dc("V1", "in", 5.0),
        resistor("R1", 1e3, "in", "anode"),
        diode("D1", "anode", "0"),
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
    let circuit = circuit([
        voltage_dc("V1", "in", 0.0),
        resistor("R1", 100.0, "in", "anode"),
        diode("D1", "anode", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let levels: Vec<f64> = (0..=20).map(|step| f64::from(step) * 0.25).collect();
    let sweep = DcSweep {
        source: "V1".to_owned(),
        parameter: None,
        values: levels.clone(),
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
    // The sweep releases the source so a subsequent operating point restores nominal level.
    let after = simulator.operating_point(&Options::default()).unwrap();
    assert!(after.values[current].abs() < 1e-12);
}

/// Builds a 5 V resistive voltage divider (`R1` = 1 kΩ, `R2` = 2 kΩ) for parameter sweeps.
fn swept_divider() -> Circuit {
    circuit([
        voltage_dc("V1", "in", 5.0),
        resistor("R1", 1e3, "in", "out"),
        resistor("R2", 2e3, "out", "0"),
    ])
}

#[test]
fn a_dc_sweep_steps_any_number_of_a_device() {
    let mut simulator = simulator(&swept_divider());
    let levels: Vec<f64> = (1..=20).map(|step| f64::from(step) * 250.0).collect();
    let sweep = DcSweep {
        source: "R1".to_owned(),
        parameter: Some("resistance".to_owned()),
        values: levels.clone(),
    };
    let result = simulator.dc_sweep(&sweep, &Options::default()).unwrap();
    assert_eq!(
        (result.parameter.as_deref(), result.nature),
        (Some("resistance"), None)
    );
    // Each level, from zero or from the level before, takes a linear divider's one
    // Newton step to its solution and a second that confirms it.
    assert_eq!(result.strategies, [Strategy::Direct; 20]);
    assert_eq!(result.iterations, [2; 20]);
    let out = index(&simulator, "out");
    // The tolerance is a few roundings of a linear divider's one Newton step.
    for (level, values) in levels.iter().zip(&result.values) {
        assert_close(values[out], 5.0 * 2e3 / (level + 2e3), 1e-14);
    }
    // The sweep restores nominal resistance so a subsequent operating point uses 1 kΩ.
    let after = simulator.operating_point(&Options::default()).unwrap();
    assert_close(after.values[out], 5.0 * 2e3 / 3e3, 1e-14);
    // Along the sweep, the number it holds moves nothing, and the lower resistor moves the
    // output by ∂/∂R2 of 5 R2 / (R1 + R2), 5 R1 / (R1 + R2)², at each R1.
    let sensitivity = jolty_engine::Sensitivity {
        outputs: vec!["out".to_owned()],
        of: jolty_engine::Differentiated::DcSweep(sweep),
    };
    let found = simulator
        .sensitivity(&sensitivity, &Options::default())
        .unwrap();
    let name = |device: &str, name: &str| {
        found
            .parameters
            .iter()
            .position(|parameter| parameter.instance == device && parameter.name == name)
            .expect("a parameter")
    };
    let (upper, lower) = (name("R1", "resistance"), name("R2", "resistance"));
    let jolty_engine::Derivatives::Real(points) = &found.derivatives else {
        panic!("real derivatives");
    };
    for (level, point) in levels.iter().zip(points) {
        assert!(point[0][upper] == 0.0, "{}", point[0][upper]);
        assert_close(point[0][lower], 5.0 * level / (level + 2e3).powi(2), 1e-9);
    }
}

#[test]
fn a_dc_sweep_cannot_change_the_unknowns_of_its_device() {
    let mut simulator = simulator(&swept_divider());
    let sweep = |levels: Vec<f64>, parameter: &str| DcSweep {
        source: "R1".to_owned(),
        parameter: Some(parameter.to_owned()),
        values: levels,
    };
    // Below 1 Ω, a resistor introduces an explicit branch current unknown.
    let crossed = simulator.dc_sweep(&sweep(vec![10.0, 0.5], "resistance"), &Options::default());
    assert!(
        matches!(&crossed, Err(SimulationError::InvalidAnalysis(message))
            if message.contains("R1") && message.contains("other unknowns")),
        "{crossed:?}"
    );
    let missing = simulator.dc_sweep(&sweep(vec![1.0], "inductance"), &Options::default());
    assert!(
        matches!(&missing, Err(SimulationError::InvalidAnalysis(message))
            if message.contains("no number `inductance`")),
        "{missing:?}"
    );
    let negative = simulator.dc_sweep(&sweep(vec![1e3, -1.0], "resistance"), &Options::default());
    assert!(
        matches!(&negative, Err(SimulationError::InvalidAnalysis(message)) if message.contains("R1")),
        "{negative:?}"
    );
    // A failed sweep leaves the circuit unchanged.
    let after = simulator.operating_point(&Options::default()).unwrap();
    assert_close(
        after.values[index(&simulator, "out")],
        5.0 * 2e3 / 3e3,
        1e-14,
    );
}

#[test]
fn an_rc_low_pass_rolls_off_at_its_corner() {
    let (resistance, capacitance) = (1e3, 1e-6);
    let circuit = circuit([
        voltage_dc("V1", "in", 0.0),
        resistor("R1", resistance, "in", "out"),
        capacitor("C1", capacitance, "out", "0"),
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
    let circuit = circuit([
        voltage_dc("V1", "in", 0.0),
        inductor("L1", inductance, "in", "mid"),
        capacitor("C1", capacitance, "mid", "out"),
        resistor("R1", 50.0, "out", "0"),
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

/// Verifies that iterative refinement preserves full accuracy on a high-Q (`Q = 31.6`)
/// series RLC low-pass far above its 5 MHz resonance.
///
/// Reusing 1 Hz conductance-dominated pivots loses four digits at 1 GHz where gain drops to
/// `2.5e-5`, and refinement against `G + jωC` recovers accuracy within `32 ε`.
#[test]
fn a_series_rlc_keeps_its_digits_far_above_resonance() {
    let (resistance, inductance, capacitance) = (1.0, 1e-6, 1e-9);
    let circuit = circuit([
        voltage_dc("V1", "in", 0.0),
        resistor("R1", resistance, "in", "mid"),
        inductor("L1", inductance, "mid", "out"),
        capacitor("C1", capacitance, "out", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let frequencies: Vec<f64> = (0..=18)
        .map(|step| 10f64.powf(f64::from(step) / 2.0))
        .collect();
    let ac = Ac {
        source: "V1".to_owned(),
        frequencies: frequencies.clone(),
    };
    let response = simulator.ac(&ac, &Options::default()).unwrap();
    let out = index(&simulator, "out");
    for (values, frequency) in response.values.iter().zip(frequencies) {
        let omega = TAU * frequency;
        let exact = Complex64::new(
            1.0 - omega * omega * inductance * capacitance,
            omega * resistance * capacitance,
        )
        .inv();
        let error = (values[out] - exact).norm() / exact.norm();
        assert!(
            error <= 32.0 * f64::EPSILON,
            "{frequency} Hz: {} ε",
            error / f64::EPSILON
        );
    }
}

/// Computes the analytical RC low-pass response after a unit ramp starting at `delay` with
/// duration `rise`.
fn rc_ramp_response(time: f64, delay: f64, rise: f64, tau: f64) -> f64 {
    1.0 - tau / rise * ((rise / tau).exp() - 1.0) * (-(time - delay) / tau).exp()
}

/// An RC low-pass under a ramp from 0 to 1 V: its resistance and capacitance, when the ramp
/// starts and how long it lasts, and when the transient stops.
struct Ramped {
    resistance: f64,
    capacitance: f64,
    delay: f64,
    rise: f64,
    stop: f64,
}

/// A low-pass of a microsecond, 1 kΩ and 1 nF, under a ramp of a nanosecond at 1 µs, for 6 µs:
/// its charge of 1 nC lies far above the default charge tolerance.
const MICROSECOND: Ramped = Ramped {
    resistance: 1e3,
    capacitance: 1e-9,
    delay: 1e-6,
    rise: 1e-9,
    stop: 6e-6,
};

/// The same low-pass a thousand times as fast, of 1 MΩ and 1 fF: its charge of 1 fC, as an
/// integrated circuit's nets hold, is a tenth of the default charge tolerance.
const FEMTOFARAD: Ramped = Ramped {
    resistance: 1e6,
    capacitance: 1e-15,
    delay: 1e-9,
    rise: 1e-12,
    stop: 6e-9,
};

/// Simulates `ramped` with `method` and `options`, returning peak post-ramp error and
/// accepted step count.
fn rc_step_error(ramped: &Ramped, method: Method, options: &Options) -> (f64, usize) {
    let Ramped {
        resistance,
        capacitance,
        delay,
        rise,
        stop,
    } = *ramped;
    let tau = resistance * capacitance;
    let circuit = circuit([
        voltage_source(
            "V1",
            "in",
            "0",
            json!({"pulse": {
                "initial": 0, "pulsed": 1, "delay": delay, "rise": rise, "fall": rise, "width": 1,
            }}),
        ),
        resistor("R1", resistance, "in", "out"),
        capacitor("C1", capacitance, "out", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let waveforms = simulator
        .transient(&transient(stop, None, method), options)
        .expect("a transient analysis");
    let out = index(&simulator, "out");
    assert!(
        waveforms.times.windows(2).all(|pair| pair[1] > pair[0]),
        "times increase"
    );
    assert!(waveforms.times.contains(&delay) && waveforms.times.contains(&(delay + rise)));
    assert_eq!(waveforms.times.last(), Some(&stop));
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

/// Verifies that every integration method meets default SPICE accuracy and converges at
/// expected order `p/(p+1)` under adaptive charge-error control.
///
/// An order-`p` step scales as `tol^(1/(p+1))`, so global error scales as `tol^(p/(p+1))`
/// from `reltol = 1e-6` to `1e-8`, including the scaled charge-tolerance floor.
#[test]
fn an_rc_charges_exponentially_with_every_method() {
    let tight = |relative_tolerance| Options {
        relative_tolerance,
        ..Options::default()
    };
    for (method, bound) in [
        (Method::BackwardEuler, 2e-2),
        (Method::Trapezoidal, 2e-3),
        (Method::Gear2, 2e-3),
    ] {
        let (loose_error, _) = rc_step_error(&MICROSECOND, method, &Options::default());
        assert!(
            loose_error < bound,
            "{method:?}: error {loose_error:e} at default tolerances"
        );
        let (tight_error, steps) = rc_step_error(&MICROSECOND, method, &tight(1e-6));
        let (tighter_error, _) = rc_step_error(&MICROSECOND, method, &tight(1e-8));
        let order = f64::from(u32::try_from(method.order()).unwrap());
        let exponent = (tight_error / tighter_error).log10() / 2.0;
        assert!(
            exponent > order / (order + 1.0) - 0.1,
            "{method:?}: {tight_error:e} at 1e-6, {tighter_error:e} at 1e-8"
        );
        assert!(steps < 2000, "{method:?}: {steps} steps");
    }
}

/// Verifies that sub-`chgtol` nets scale their charge tolerance with `reltol` via
/// `reltol · max(|q|, chgtol)` (ngspice 47 `cktterr.c`, lines 40-41).
///
/// The 1 fC femtofarad RC sits at one tenth of default `chgtol`, so at `reltol = 1e-7` and
/// `1e-9` (with tiny `current_tolerance`) error shrinks with exponent near `p/(p+1)`.
/// An additive `reltol · |q| + chgtol` floor would freeze step error across both runs.
#[test]
fn a_femtofarad_nets_error_shrinks_with_the_relative_tolerance() {
    let tight = |relative_tolerance| Options {
        relative_tolerance,
        current_tolerance: 1e-18,
        ..Options::default()
    };
    for (method, order) in ORDERS {
        let (tight_error, _) = rc_step_error(&FEMTOFARAD, method, &tight(1e-7));
        let (tighter_error, _) = rc_step_error(&FEMTOFARAD, method, &tight(1e-9));
        let exponent = (tight_error / tighter_error).log10() / 2.0;
        assert!(
            exponent > order / (order + 1.0) - 0.1,
            "{method:?}: {tight_error:e} at 1e-7, {tighter_error:e} at 1e-9"
        );
    }
}

/// Verifies that picosecond charge settling near a corner resolves within tolerance without
/// violating the SPICE minimum step floor (1e-11 of max step).
///
/// A 46 mA current rising in 1 ps at 1 ms into a parallel 1 nF / 10 mΩ load mimics gate
/// displacement current into the internal source node of the buck benchmark's lower switch.
/// At 1e-5 relative tolerance, the 0.46 pC transient requires roughly 2e-14 s steps (below
/// 1e-9 of the 40 µs max step) to hold accuracy to 1e-19 C from rest.
#[test]
fn a_charge_settling_in_picoseconds_is_resolved_to_its_tolerance() {
    let (current, resistance, capacitance) = (0.046, 0.01, 1e-9);
    let (delay, rise) = (1e-3, 1e-12);
    let circuit = circuit([
        current_source(
            "I1",
            "s",
            "0",
            json!({"pulse": {
                "initial": 0, "pulsed": current, "delay": delay, "rise": rise, "fall": rise,
                "width": 1e-4,
            }}),
        ),
        resistor("R1", resistance, "s", "0"),
        capacitor("C1", capacitance, "s", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let options = Options {
        relative_tolerance: 1e-5,
        ..Options::default()
    };
    let waveforms = simulator
        .transient(&trapezoidal(2e-3, None), &options)
        .unwrap();
    // Within its rise and its first nanosecond, the node follows the closed form within the
    // voltage tolerance, 1e-6 V, a five-hundredth of its 0.46 mV.
    let s = index(&simulator, "s");
    let tau = resistance * capacitance;
    let settled = current * resistance;
    let mut checked = 0;
    for (&time, values) in waveforms.times.iter().zip(&waveforms.values) {
        if time > delay + rise && time < delay + 1e-9 {
            let expected = settled * rc_ramp_response(time, delay, rise, tau);
            assert!(
                (values[s].abs() - expected).abs() < 1e-6,
                "at {time:e}: {} against {expected:e}",
                values[s]
            );
            checked += 1;
        }
    }
    assert!(checked > 10, "{checked} points within the settling");
}

/// Verifies that no transient step exceeds `max_step`, even when error permits growth and a
/// nearby breakpoint or requested time lies within reach.
#[test]
fn no_step_is_longer_than_the_longest_allowed() {
    // An RC low-pass of 1 µs, whose response to a pulse settles long before each corner.
    let circuit = circuit([
        voltage_source(
            "V1",
            "in",
            "0",
            json!({"pulse": {
                "initial": 0, "pulsed": 1, "delay": 1e-6, "rise": 1e-7, "fall": 1e-7,
                "width": 4e-6,
            }}),
        ),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
    ]);
    let longest = 5e-8;
    let every_microsecond: Vec<f64> = (1..=20).map(|index| f64::from(index) * 1e-6).collect();
    for times in [Vec::new(), every_microsecond] {
        let transient = Transient {
            breakpoints: times.clone(),
            ..trapezoidal(2e-5, Some(longest))
        };
        let waveforms = simulator(&circuit)
            .transient(&transient, &Options::default())
            .expect("a transient analysis");
        let step = waveforms
            .times
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .fold(0.0, f64::max);
        assert!(
            step <= longest * (1.0 + 1e-9),
            "a step of {step} s, longer than {longest} s, with {} times asked for",
            times.len()
        );
        assert!(times.iter().all(|time| waveforms.times.contains(time)));
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

/// Returns options that disable truncation-error step shortening so every post-startup step
/// runs at `max_step`.
fn uniform_steps() -> Options {
    Options {
        truncation_tolerance: 1e12,
        relative_tolerance: 1e-9,
        voltage_tolerance: 1e-12,
        current_tolerance: 1e-15,
        ..Options::default()
    }
}

/// Runs a uniform-step transient of `circuit` to `end` with `method` and `step`, returning
/// `(times, out_voltages)`.
fn stepped(circuit: &Circuit, method: Method, step: f64, end: f64) -> (Vec<f64>, Vec<f64>) {
    let mut simulator = simulator(circuit);
    let waveforms = simulator
        .transient(&transient(end, Some(step), method), &uniform_steps())
        .expect("a transient analysis");
    let out = index(&simulator, "out");
    let values = waveforms.values.iter().map(|values| values[out]).collect();
    (waveforms.times, values)
}

/// Builds a 1 µs RC low-pass driven by a 100 kHz sine of `amplitude`, with a clamping diode
/// across the capacitor when `clamped` is true.
fn sine_low_pass(amplitude: f64, clamped: bool) -> Circuit {
    let mut instances = vec![
        voltage_sine("V1", "in", 0.0, amplitude, 1e5),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
    ];
    if clamped {
        instances.push(diode("D1", "0", "out"));
    }
    circuit(instances)
}

/// Computes empirical convergence orders across successive step halvings in `HALVED`.
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
    let circuit = circuit([
        current_source(
            "I1",
            "0",
            "top",
            piecewise(&[[0.0, 1e-3], [1e-6, 1e-3], [1.001e-6, 0.0]]),
        ),
        inductor("L1", inductance, "top", "0"),
        capacitor("C1", capacitance, "top", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let waveforms = simulator
        .transient(
            &trapezoidal(5.0 * period, Some(period / 200.0)),
            &Options::default(),
        )
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
    let circuit = circuit([
        voltage_sine("V1", "in", 0.0, 1.0, frequency),
        resistor("R1", resistance, "in", "out"),
        capacitor("C1", capacitance, "out", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let outcome = simulator
        .run(
            &Analysis::Transient(trapezoidal(20e-3, Some(1e-5))),
            &Options::default(),
        )
        .unwrap();
    let AnalysisResult::Transient(waveforms) = outcome else {
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
    let circuit = circuit([
        voltage_sine("V1", "s", 0.0, 1.0, frequency),
        device(
            "B1",
            "behavioral_voltage_source",
            json!({"expression": "abs(v(s))"}),
            &[("positive", "out"), ("negative", "0"), ("s", "s")],
        ),
        capacitor("C1", capacitance, "out", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let outcome = simulator
        .run(
            &Analysis::Transient(trapezoidal(5e-3, Some(1e-5))),
            &Options::default(),
        )
        .unwrap();
    let AnalysisResult::Transient(waveforms) = outcome else {
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

/// Asserts that `energy_at` stays constant within `relative_bound` across all points
/// `time >= start_time` and returns `(first_free, last_free)`.
fn assert_energy_conserved(
    times: &[f64],
    values: &[Vec<f64>],
    start_time: f64,
    relative_bound: f64,
    label: &str,
    energy_at: impl Fn(&[f64]) -> f64,
) -> (f64, f64) {
    let free_energies: Vec<f64> = times
        .iter()
        .zip(values)
        .filter(|&(&time, _)| time >= start_time)
        .map(|(_, row)| energy_at(row))
        .collect();
    assert!(
        free_energies.len() >= 10,
        "too few points for {label}: {}",
        free_energies.len()
    );
    let first_free = free_energies[0];
    let last_free = *free_energies.last().unwrap();
    assert!(
        (last_free - first_free).abs() <= relative_bound * first_free,
        "{label} lost energy: start={first_free:.6e}, end={last_free:.6e}"
    );
    (first_free, last_free)
}

#[test]
fn an_undamped_lc_tank_conserves_energy_at_every_tolerance() {
    // Precision suite fixture `lc_tank`: 1 uH, 1 nF, 1 mA turned off in 1 ns, stop = 4 us
    // (20 periods), default max_step = stop / 50 = 80 ns (omega * h = 2.53 > 2).
    // At coarse tolerances (1e-2, 3e-3), the first difference of dq/dt alternates every step
    // even though the oscillation is physical; the ringing test must not damp it with BE.
    let (inductance, capacitance, ramp_time, initial_current): (f64, f64, f64, f64) =
        (1e-6, 1e-9, 1e-9, 1e-3);
    let omega = 1.0 / (inductance * capacitance).sqrt();
    let free_current = initial_current * (omega * ramp_time).sin() / (omega * ramp_time);
    let free_voltage =
        inductance * (initial_current / ramp_time) * ((omega * ramp_time).cos() - 1.0);
    let exact_energy = 0.5 * capacitance * free_voltage * free_voltage
        + 0.5 * inductance * free_current * free_current;

    let circuit = circuit([
        current_source(
            "I1",
            "0",
            "top",
            piecewise(&[[0.0, initial_current], [ramp_time, 0.0]]),
        ),
        inductor("L1", inductance, "top", "0"),
        capacitor("C1", capacitance, "top", "0"),
    ]);
    let transient = trapezoidal(4e-6, None);

    for &relative_tolerance in &[1e-2, 3e-3, 1e-3, 1e-4, 1e-6] {
        let mut simulator = simulator(&circuit);
        let options = Options {
            relative_tolerance,
            ..Options::default()
        };
        let waveforms = simulator.transient(&transient, &options).unwrap();
        let voltage_index = index(&simulator, "top");
        let current_index = index(&simulator, "L1:current");
        let (_, last_free) = assert_energy_conserved(
            &waveforms.times,
            &waveforms.values,
            200e-9,
            1e-10,
            &format!("lc_tank reltol={relative_tolerance}"),
            |values| {
                let voltage = values[voltage_index];
                let current = values[current_index];
                0.5 * capacitance * voltage * voltage + 0.5 * inductance * current * current
            },
        );
        // Total energy matches the exact closed-form energy after the 1 ns turn-off ramp to
        // within 1e-3 (over 3 correct digits even at reltol = 1e-2).
        assert!(
            (last_free - exact_energy).abs() <= 1e-3 * exact_energy,
            "energy mismatch at reltol={relative_tolerance}: got={last_free:.6e}, exact={exact_energy:.6e}"
        );
    }
}

#[test]
fn a_multi_mode_lossless_lc_ladder_conserves_energy_at_coarse_tolerance() {
    // Three-stage lossless coupled LC ladder kicked by 1 mA turned off in 1 ns, running at
    // reltol = 1e-2 with default max_step (80 ns, where omega * h > 2 on multiple modes and
    // scalar beating fails single-mode harmonic checks).
    let (capacitance_1, capacitance_2, capacitance_3) = (1e-9, 0.47e-9, 0.82e-9);
    let (inductance_1, inductance_2, inductance_3) = (1e-6, 2.2e-6, 1.5e-6);
    let circuit = circuit([
        current_source("I1", "0", "n1", piecewise(&[[0.0, 1e-3], [1e-9, 0.0]])),
        inductor("L1", inductance_1, "n1", "0"),
        capacitor("C1", capacitance_1, "n1", "0"),
        inductor("L2", inductance_2, "n1", "n2"),
        capacitor("C2", capacitance_2, "n2", "0"),
        inductor("L3", inductance_3, "n2", "n3"),
        capacitor("C3", capacitance_3, "n3", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let options = Options {
        relative_tolerance: 1e-2,
        ..Options::default()
    };
    let waveforms = simulator
        .transient(&trapezoidal(4e-6, None), &options)
        .unwrap();
    let (voltage_1, voltage_2, voltage_3) = (
        index(&simulator, "n1"),
        index(&simulator, "n2"),
        index(&simulator, "n3"),
    );
    let (current_1, current_2, current_3) = (
        index(&simulator, "L1:current"),
        index(&simulator, "L2:current"),
        index(&simulator, "L3:current"),
    );
    let (_, last_free) = assert_energy_conserved(
        &waveforms.times,
        &waveforms.values,
        200e-9,
        1e-10,
        "multi-mode ladder",
        |values| {
            0.5 * (capacitance_1 * values[voltage_1].powi(2)
                + capacitance_2 * values[voltage_2].powi(2)
                + capacitance_3 * values[voltage_3].powi(2)
                + inductance_1 * values[current_1].powi(2)
                + inductance_2 * values[current_2].powi(2)
                + inductance_3 * values[current_3].powi(2))
        },
    );
    let initial_energy = 0.5 * inductance_1 * 1e-6;
    assert!(
        (last_free - initial_energy).abs() <= 1e-2 * initial_energy,
        "ladder energy drifted from initial: got={last_free:.6e}, init={initial_energy:.6e}"
    );
}

#[test]
fn a_lossless_lc_ladder_with_dc_bypass_and_zero_volt_sense_conserves_energy() {
    // Two-mode lossless LC ladder coupled via a bypass capacitor to a 5 V DC supply, with a
    // floating 0 V sense source across a floating capacitor. Grounded and floating ideal
    // voltage sources must produce zero false ringing triggers.
    let (bypass_capacitance, floating_capacitance, capacitance_2, inductance_1, inductance_2) =
        (1e-9, 0.5e-9, 0.68e-9, 1e-6, 2.2e-6);
    let circuit = circuit([
        voltage_dc("VDD", "vdd", 5.0),
        current_source("I1", "0", "n1", piecewise(&[[0.0, 1e-3], [1e-9, 0.0]])),
        capacitor("CB", bypass_capacitance, "vdd", "n1"),
        voltage_source("VS", "n1s", "n1", dc(0.0)),
        capacitor("CF", floating_capacitance, "n1s", "n1"),
        inductor("L1", inductance_1, "n1", "0"),
        inductor("L2", inductance_2, "n1s", "n2"),
        capacitor("C2", capacitance_2, "n2", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    let options = Options {
        relative_tolerance: 1e-2,
        ..Options::default()
    };
    let waveforms = simulator
        .transient(&trapezoidal(4e-6, None), &options)
        .unwrap();
    let (voltage_1, voltage_2) = (index(&simulator, "n1"), index(&simulator, "n2"));
    let (current_1, current_2) = (
        index(&simulator, "L1:current"),
        index(&simulator, "L2:current"),
    );
    // Conserved small-signal Hamiltonian relative to the DC equilibrium (vdd = 5 V, n1 = n2 = 0):
    // CF carries 0 V across VS, while CB stores 0.5 * CB * v1^2 against the constant 5 V supply.
    assert_energy_conserved(
        &waveforms.times,
        &waveforms.values,
        200e-9,
        1e-9,
        "DC bypass + 0 V sense LC ladder",
        |values| {
            0.5 * (bypass_capacitance * values[voltage_1].powi(2)
                + capacitance_2 * values[voltage_2].powi(2)
                + inductance_1 * values[current_1].powi(2)
                + inductance_2 * values[current_2].powi(2))
        },
    );
}

#[test]
fn the_trapezoidal_formula_stops_floating_capacitor_ringing_above_an_lc_tank() {
    // Floating capacitor C1 in parallel with behavioral source B1 = abs(v(s)), stacked on top
    // of a grounded lossless LC tank with stray capacitance C_stray = 1 nF to ground.
    // Neither end of B1 lies individually in ker(F), so the saddle-point projection must
    // isolate the floating C-V loop's (-1)^n ringing and insert BE steps after each corner.
    let (capacitance, stray, tank_inductance, tank_capacitance, frequency) =
        (1e-6, 1e-9, 10e-6, 1e-6, 1e3);
    let circuit = circuit([
        voltage_sine("V1", "s", 0.0, 1.0, frequency),
        inductor("L1", tank_inductance, "base", "0"),
        capacitor("CT", tank_capacitance, "base", "0"),
        device(
            "B1",
            "behavioral_voltage_source",
            json!({"expression": "abs(v(s))"}),
            &[("positive", "top"), ("negative", "base"), ("s", "s")],
        ),
        capacitor("C1", capacitance, "top", "base"),
        capacitor("CS", stray, "top", "0"),
    ]);
    let mut simulator = simulator(&circuit);
    // Asymmetric step relative to the 500 us half-period so corners land off-midpoint.
    let transient = trapezoidal(5e-3, Some(9.7e-6));
    let waveforms = simulator
        .transient(&transient, &Options::default())
        .unwrap();
    let current = index(&simulator, "B1:current");
    let peak = capacitance * TAU * frequency;
    for (time, values) in waveforms.times.iter().zip(&waveforms.values) {
        let phase = TAU * frequency * time;
        if phase.sin().abs() < 0.5 {
            continue;
        }
        let expected = peak * phase.cos().abs();
        let error = (values[current].abs() - expected).abs();
        // Without damping the floating loop accumulates > 2.0 * peak (200% error); with
        // saddle-point damping it stays near 1% (plus ~0.1% stray capacitance current).
        assert!(
            error < 3e-2 * peak,
            "at {time} s the floating B1 current is off by {error} A (peak={peak} A)"
        );
    }
}

#[test]
fn the_trapezoidal_formula_stops_ringing_beside_a_lossless_lc_tank() {
    // A 1 A lossless LC tank running in the same circuit as a 6.28 mA grounded `|sin|`
    // capacitor must NOT hide the corner-induced ringing on B1:current.
    let (capacitance, frequency) = (1e-6, 1e3);
    let circuit = circuit([
        current_source("I1", "0", "tank", piecewise(&[[0.0, 1.0], [1e-6, 0.0]])),
        inductor("L1", 10e-6, "tank", "0"),
        capacitor("CT", 1e-6, "tank", "0"),
        voltage_sine("V1", "s", 0.0, 1.0, frequency),
        device(
            "B1",
            "behavioral_voltage_source",
            json!({"expression": "abs(v(s))"}),
            &[("positive", "out"), ("negative", "0"), ("s", "s")],
        ),
        capacitor("C1", capacitance, "out", "0"),
    ]);
    let transient = trapezoidal(5e-3, Some(9.7e-6));
    for &(relative_tolerance, bound) in &[(1e-3, 1e-2), (1e-2, 8e-2)] {
        let mut simulator = simulator(&circuit);
        let options = Options {
            relative_tolerance,
            ..Options::default()
        };
        let waveforms = simulator.transient(&transient, &options).unwrap();
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
                error < bound * peak,
                "at reltol={relative_tolerance}, {time} s the grounded B1 current is off by {error} A"
            );
        }
    }
}

/// Builds a behavioral voltage source stepping 0 V to 5 V when `condition` becomes true,
/// driving a 1 µs RC load (1 kΩ, 1 nF) alongside a 2 ms 5 V `ramp` reference.
fn jumping(condition: &str) -> Circuit {
    let terminals: &[(&str, &str)] = if condition.contains("v(ramp)") {
        &[("positive", "out"), ("negative", "0"), ("ramp", "ramp")]
    } else {
        &[("positive", "out"), ("negative", "0")]
    };
    circuit([
        voltage_source(
            "V1",
            "ramp",
            "0",
            json!({"pulse": {"initial": 0, "pulsed": 5, "delay": 0, "rise": 2e-3,
                "fall": 1e-3, "width": 1e-3}}),
        ),
        device(
            "B1",
            "behavioral_voltage_source",
            json!({"expression": format!("{condition} ? 5 : 0")}),
            terminals,
        ),
        resistor("R1", 1e3, "out", "cap"),
        capacitor("C1", 1e-9, "cap", "0"),
    ])
}

#[test]
fn a_behavioral_jump_is_stepped_onto_and_followed() {
    // Whether triggered by time or voltage, locating the discontinuity starts the exponential
    // cleanly without overshoot. At 1.03 ms, a jump late in a 40 µs step leaves the next
    // trial step spanning multiple 1 µs time constants unless truncation error is checked.
    // With error checking at `reltol = 1e-4`, post-jump steps stay within 20 mV of exact
    // (46 mV at default tolerances), whereas skipping error checks errs by volts.
    let options = Options {
        relative_tolerance: 1e-4,
        ..Options::default()
    };
    let jumps = [
        (1.0003e-3, ["time > 1.0003e-3", "v(ramp) > 2.50075"]),
        (1.03e-3, ["time > 1.03e-3", "v(ramp) > 2.575"]),
    ];
    for (jump, condition) in jumps
        .into_iter()
        .flat_map(|(jump, conditions)| conditions.map(|condition| (jump, condition)))
    {
        let circuit = jumping(condition);
        let mut simulator = simulator(&circuit);
        let outcome = simulator
            .run(&Analysis::Transient(trapezoidal(2e-3, None)), &options)
            .unwrap();
        let AnalysisResult::Transient(waveforms) = outcome else {
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
        "format": "jolty-circuit", "version": 5, "top": "top",
        "components": {
            "top": {"ports": [{"name": "0", "role": "ground"}], "instances": [
                {"name": "V1", "of": {"model": "voltage_source"},
                 "parameters": {"waveform": dc(3.0)},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"name": "X1", "of": {"component": "half"},
                 "terminals": {"input": "in", "output": "mid", "ground": "0"}},
                {"name": "X2", "of": {"component": "half"},
                 "terminals": {"input": "mid", "output": "out", "ground": "0"}}
            ]},
            "half": {"ports": [
                {"name": "input", "role": "signal"},
                {"name": "output", "role": "signal"},
                {"name": "ground", "role": "ground"}
            ], "instances": [
                {"name": "E1", "of": {"model": "voltage_controlled_voltage_source"},
                 "parameters": {"gain": 0.5},
                 "terminals": {"positive": "buffered", "negative": "ground",
                               "control_positive": "input", "control_negative": "ground"}},
                {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e3},
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
    let circuit = circuit([
        voltage_dc("V1", "in", 1.0),
        capacitor("C1", 1e-9, "in", "floating"),
    ]);
    let error = simulator(&circuit)
        .operating_point(&Options::default())
        .unwrap_err();
    assert_eq!(
        error,
        SimulationError::Singular {
            unknowns: vec!["floating".to_owned()]
        }
    );
}

#[test]
fn sources_in_parallel_leave_their_currents_undetermined() {
    // Both sources' currents appear in the equation of `in` alone: only their sum is
    // determined.
    let circuit = circuit([
        voltage_dc("V1", "in", 1.0),
        voltage_dc("V2", "in", 1.0),
        resistor("R1", 1e3, "in", "0"),
    ]);
    let error = simulator(&circuit)
        .operating_point(&Options::default())
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "the circuit's equations are singular, leaving V1:current, V2:current undetermined: a \
         node may have no DC path to the reference, or sources may fix conflicting values"
    );
}

#[test]
fn analyses_name_what_they_cannot_find() {
    let circuit = circuit([voltage_dc("V1", "in", 1.0), resistor("R1", 1.0, "in", "0")]);
    let mut simulator = simulator(&circuit);
    let options = Options::default();
    let missing = DcSweep {
        source: "V9".to_owned(),
        parameter: None,
        values: vec![1.0],
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

/// Runs a trapezoidal transient of `circuit` to `stop` with maximum step `stop / 100`.
fn run_transient(
    circuit: &Circuit,
    stop: f64,
    options: &Options,
) -> Result<AnalysisResult, SimulationError> {
    simulator(circuit).run(
        &Analysis::Transient(trapezoidal(stop, Some(stop / 100.0))),
        options,
    )
}

#[test]
fn a_transient_names_the_unknowns_that_stopped_converging() {
    // Equation `v(a)² = v(c)` loses real roots once `V1` crosses zero at 0.5 ms, so no step
    // past the fold can converge. The two roots coalesce at 0.5 ms where Newton converges
    // linearly by halving error each iteration. Continuing while iterates contract tracks the
    // fold to within `1e-9 V / (2000 V/s) = 5e-13 s`, set by the 1e-12 A residual tolerance.
    let circuit = circuit([
        voltage_source("V1", "c", "0", piecewise(&[[0.0, 1.0], [1e-3, -1.0]])),
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": "1e-3 * (v(a) ^ 2 - v(c))"}),
            &[("positive", "a"), ("negative", "0"), ("a", "a"), ("c", "c")],
        ),
    ]);
    let error = run_transient(&circuit, 1e-3, &Options::default()).unwrap_err();
    let SimulationError::StepTooSmall {
        time,
        rejection: Rejection::Unconverged { worst },
        ..
    } = &error
    else {
        panic!("{error}")
    };
    assert!(
        (time - 0.5e-3).abs() <= 5e-13,
        "it failed at {time} s, not at the fold"
    );
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
    let circuit = circuit([
        voltage_sine("V1", "in", 0.0, 1.0, 1e3),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-6, "out", "0"),
    ]);
    let options = Options {
        truncation_tolerance: 1e-30,
        ..Options::default()
    };
    let error = run_transient(&circuit, 1e-3, &options).unwrap_err();
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
fn a_transient_whose_results_would_fill_the_memory_fails_at_once() {
    // A second in picosecond steps: a trillion points, of a few numbers each.
    let circuit = circuit([voltage_dc("V1", "in", 1.0), resistor("R1", 1e3, "in", "0")]);
    let error = simulator(&circuit)
        .run(
            &Analysis::Transient(trapezoidal(1.0, Some(1e-12))),
            &Options::default(),
        )
        .unwrap_err();
    assert_eq!(
        error,
        SimulationError::TooManyPoints {
            time: 0.0,
            step: 1e-12,
            max_memory: 2.0
        }
    );
    assert_eq!(
        error.to_string(),
        "the transient's results would pass 2.0 GiB before it stops, its steps 1e-12 s \
         long at t = 0.0 s"
    );
}

#[test]
fn a_transient_follows_a_source_that_only_its_transistor_holds() {
    // Only the transistor holds its source, so stepping the gate lifts the source to one
    // threshold below the gate. Current vanishes quadratically at that double root, where
    // Newton halves error per iteration.
    let (supply, threshold) = (5.0, 0.5);
    let circuit = circuit([
        voltage_dc("VD", "drain", supply),
        voltage_source(
            "VG",
            "gate",
            "0",
            piecewise(&[[0.0, 0.0], [1e-6, 0.0], [1.001e-6, supply]]),
        ),
        device(
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
    let outcome = simulator
        .run(
            &Analysis::Transient(trapezoidal(2e-6, Some(1e-8))),
            &Options::default(),
        )
        .unwrap();
    let AnalysisResult::Transient(waveforms) = outcome else {
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
    let circuit = circuit([
        current_dc("I1", "0", "high", 1e-4),
        diode("D1", "0", "high"),
        capacitor("C1", 1e-6, "high", "low"),
        resistor("R1", 10.0, "low", "0"),
        voltage_source(
            "V1",
            "in",
            "0",
            json!({"pulse": {
                "initial": 3, "pulsed": 4, "delay": 2e-7, "rise": 1e-7, "fall": 1e-7,
                "width": 8e-7,
            }}),
        ),
        capacitor("C2", 1e-8, "in", "low"),
    ]);
    let mut simulator = simulator(&circuit);
    let outcome = simulator
        .run(
            &Analysis::Transient(trapezoidal(2e-6, Some(1e-8))),
            &Options::default(),
        )
        .unwrap();
    let AnalysisResult::Transient(waveforms) = outcome else {
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

/// Builds a level-1 MOSFET of the given `channel` with bulk tied to source.
fn mosfet(name: &str, channel: &str, [drain, gate, source]: [&str; 3]) -> Instance {
    let (width, threshold, transconductance) = if channel == "n" {
        (1e-5, 0.5, 1e-4)
    } else {
        (2e-5, -0.5, 5e-5)
    };
    device(
        name,
        "mosfet",
        json!({"level": 1, "channel": channel, "width": width, "length": 1e-6,
               "threshold_voltage": threshold, "transconductance_parameter": transconductance}),
        &[
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
        json!({"pulse": {
            "initial": 0, "pulsed": 5, "delay": 3e-8, "rise": 2e-9, "fall": 2e-9,
            "width": 0.48 * period, "period": period,
        }})
    };
    let circuit = circuit([
        mosfet("MIA", "n", ["a", "rawa", "0"]),
        mosfet("MPA", "p", ["a", "rawa", "vdd"]),
        mosfet("MIB", "n", ["b", "rawb", "0"]),
        mosfet("MPB", "p", ["b", "rawb", "vdd"]),
        mosfet("MNA", "n", ["out", "a", "mid"]),
        mosfet("MNB", "n", ["mid", "b", "0"]),
        mosfet("MQA", "p", ["out", "a", "vdd"]),
        mosfet("MQB", "p", ["out", "b", "vdd"]),
        capacitor("CA", 5e-14, "a", "0"),
        capacitor("CB", 5e-14, "b", "0"),
        capacitor("CO", 5e-14, "out", "0"),
        voltage_source("VDD", "vdd", "0", piecewise(&[[0.0, 0.0], [1e-6, 5.0]])),
        voltage_source("VA", "rawa", "0", clock(1e-7)),
        voltage_source("VB", "rawb", "0", clock(9.1e-8)),
    ]);
    let outcome = simulator(&circuit).run(
        &Analysis::Transient(trapezoidal(1e-6, Some(1e-9))),
        &Options::default(),
    );
    assert!(outcome.is_ok(), "{:?}", outcome.err());
}

#[test]
fn a_heated_resistor_settles_where_its_power_flows_through_its_sink() {
    // 10 V across 100 Ω with tc1 = 4e-3 /K, held to the ambient by 50 K/W: the rise solves
    // `tc · ΔT² + ΔT − 50 · v² / r = 0`.
    let circuit = circuit([
        voltage_dc("V1", "top", 10.0),
        device(
            "R1",
            "resistor",
            json!({"resistance": 100.0, "temperature_coefficient": 4e-3, "thermal": true}),
            &[("a", "top"), ("b", "0"), ("thermal", "junction")],
        ),
        two_terminal(
            "RTH",
            "thermal_resistance",
            "resistance",
            50.0,
            "junction",
            "0",
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
    let mut simulator = simulator(&circuit([
        voltage_dc("V1", "vdd", 1.8),
        resistor("R1", 1e4, "vdd", "d"),
        device("M1", "foundry_mosfet", transistor, &terminals),
    ]));
    assert_eq!(simulator.foundry_devices(), ["M1"]);
    let refused = SimulationError::Foundry(vec!["M1".to_owned()]);
    let options = Options::default();
    for analysis in [
        Analysis::OperatingPoint,
        Analysis::DcSweep(DcSweep {
            source: "V1".to_owned(),
            parameter: None,
            values: vec![0.0, 1.8],
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

#[test]
fn a_step_starts_from_its_prediction() {
    // A resistor's current follows its source's ramp exactly, so once two points lie on
    // the line, each step's prediction is its solution, which Newton iteration confirms
    // at once: one iteration a step, where starting from the last point took two.
    let circuit = circuit([
        voltage_source("V1", "a", "0", piecewise(&[[0.0, 0.0], [1e-3, 1.0]])),
        resistor("R1", 1e3, "a", "0"),
    ]);
    let outcome = run_transient(&circuit, 1e-3, &Options::default()).expect("a transient");
    let AnalysisResult::Transient(waveforms) = outcome else {
        panic!("not a transient: {outcome:?}")
    };
    let statistics = &waveforms.statistics;
    assert_eq!(statistics.rejected, 0, "{statistics:?}");
    assert!(
        statistics.iterations <= statistics.accepted + 2,
        "{statistics:?}"
    );
}

/// Builds a 1 V RC circuit (1 kΩ, 1 nF) paired with initial condition `out = initial`.
fn rc_from(initial: f64) -> (Circuit, BTreeMap<String, f64>) {
    let circuit = circuit([
        voltage_dc("V1", "in", 1.0),
        resistor("R1", 1e3, "in", "out"),
        capacitor("C1", 1e-9, "out", "0"),
    ]);
    (circuit, BTreeMap::from([("out".to_owned(), initial)]))
}

/// Verifies that an RC transient follows `1 − 0.8 e^{−t/τ}` from initial condition `out = 0.2 V`
/// both with and without an initial DC operating point solve.
///
/// Both modes remain within the 2e-3 default trapezoidal error bound established in
/// `an_rc_charges_exponentially_with_every_method`.
#[test]
fn an_initial_condition_starts_an_rc_from_its_voltage() {
    let (circuit, initial) = rc_from(0.2);
    let tau = 1e-6;
    for operating_point in [true, false] {
        let mut simulator = simulator(&circuit);
        let transient = Transient {
            initial_conditions: initial.clone(),
            operating_point,
            ..trapezoidal(5.0 * tau, None)
        };
        let waveforms = simulator
            .transient(&transient, &Options::default())
            .expect("a transient analysis");
        let (input, out) = (index(&simulator, "in"), index(&simulator, "out"));
        let first = &waveforms.values[0];
        assert_close(first[out], 0.2, 1e-12);
        assert_close(first[input], if operating_point { 1.0 } else { 0.0 }, 1e-12);
        assert_eq!(waveforms.statistics.initial.is_some(), operating_point);
        let error = waveforms
            .times
            .iter()
            .zip(&waveforms.values)
            .map(|(&time, values)| (values[out] - (1.0 - 0.8 * (-time / tau).exp())).abs())
            .fold(0.0, f64::max);
        assert!(
            error < 2e-3,
            "from the operating point: {operating_point}, error {error:e}"
        );
    }
}

/// Verifies that initial conditions on source-driven nets accept matching voltages and
/// reject conflicting voltages or non-net unknown names.
#[test]
fn an_initial_condition_a_source_holds_must_agree_with_it() {
    let (circuit, _) = rc_from(0.2);
    let start = |voltage: f64| {
        let transient = Transient {
            initial_conditions: BTreeMap::from([("in".to_owned(), voltage)]),
            ..trapezoidal(1e-6, None)
        };
        simulator(&circuit).transient(&transient, &Options::default())
    };
    let waveforms = start(1.0).expect("a transient analysis");
    assert_close(waveforms.values[0][1], 1.0, 1e-12);
    let error = start(2.0).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid analysis: the initial condition of `in` is 2.0 V, but the voltage sources \
         and inductors that fix it hold it at 1.0 V"
    );
    let transient = Transient {
        initial_conditions: BTreeMap::from([("V1:current".to_owned(), 0.0)]),
        ..trapezoidal(1e-6, None)
    };
    let error = simulator(&circuit)
        .transient(&transient, &Options::default())
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid analysis: the initial condition names `V1:current`, which is no net of the \
         circuit"
    );
}

/// Verifies that a 1.5 V nodeset steers `v = 2 tanh(v)` from the trivial `0 V` solution to
/// the positive stable root near `+1.915 V`.
#[test]
fn a_nodeset_leads_the_operating_point_to_another_solution() {
    let circuit = circuit([
        device(
            "B1",
            "behavioral_current_source",
            json!({"expression": "2e-3 * tanh(v(x))"}),
            &[("positive", "0"), ("negative", "x"), ("x", "x")],
        ),
        resistor("R1", 1e3, "x", "0"),
    ]);
    // The positive root of v = 2 tanh(v), by the fixed point that converges to it.
    let mut root: f64 = 2.0;
    for _ in 0..200 {
        root = 2.0 * root.tanh();
    }
    let mut simulator = simulator(&circuit);
    let x = index(&simulator, "x");
    let plain = simulator
        .operating_point(&Options::default())
        .expect("an operating point");
    assert!(plain.values[x].abs() < 1e-9, "{}", plain.values[x]);
    let options = Options {
        nodesets: BTreeMap::from([("x".to_owned(), 1.5)]),
        ..Options::default()
    };
    let led = simulator
        .operating_point(&options)
        .expect("an operating point");
    assert_close(led.values[x], root, 1e-6);
}
