//! Operating-point and linearized stability rules.

use std::f64::consts::TAU;

use jolty_engine::{
    Mode, OperatingPointResult, OperatingPoints, Options, SimulationError, Simulator,
};
use jolty_models::Nature;

use crate::structure::{NAMED, list, prose};
use crate::{Diagnostic, Level, Rule};

/// Maximum unknown count linearized during operating-point checks; dense QZ takes about one
/// second at 500 unknowns.
///
/// Origin: runtime budget for the `O(n^3)` dense QZ eigenvalue solve in
/// `check_operating_point`, matching `MULTIPLE_OPERATING_POINTS` and `UNSTABLE_POLE`.
pub const MAX_LINEARIZED: usize = 500;

/// Minimum fraction of peak mode amplitude or voltage difference required to list a net in a
/// diagnostic.
///
/// Origin: 50% (`-6 dB`) prominence threshold matching the condition text of
/// `MULTIPLE_OPERATING_POINTS` and `UNSTABLE_POLE`.
const PROMINENCE_THRESHOLD: f64 = 0.5;

/// Minimum relative voltage shift from doubling `gmin` before `gmin-dependence` flags a net.
///
/// Origin: 1% (`1e-2`) warning threshold, one decade above default `reltol = 1e-3` so normal
/// convergence noise does not trigger warnings.
const GMIN_MOVE: f64 = 1e-2;

/// Reference voltage floor (1 V) below which `gmin-dependence` normalizes voltage shifts
/// against 1 V.
///
/// Origin: 1 V nominal circuit scale preventing near-zero node voltages from triggering
/// warnings on tiny absolute shifts.
const VOLT: f64 = 1.0;

/// Operating-point rule flagging nets strongly sensitive to `gmin` regularization.
pub(crate) const GMIN_DEPENDENCE: Rule = Rule {
    name: "gmin-dependence",
    level: Level::Warn,
    condition: "Doubling gmin (the shunt conductance placed across every junction to aid \
                Newton convergence) moves a net's operating-point voltage to first order by \
                more than 1% of max(|V|, 1 V). Little other than gmin sets that net's \
                potential, as between reverse-biased junctions, so the voltage reflects \
                simulator regularization rather than circuit behavior. One diagnostic lists \
                all affected nets in decreasing order of shift during operating-point checks.",
};

/// Operating-point rule flagging circuits with multiple DC solutions.
pub(crate) const MULTIPLE_OPERATING_POINTS: Rule = Rule {
    name: "multiple-operating-points",
    level: Level::Warn,
    condition: "The circuit's DC equations admit an additional solution distinct from the \
                operating point, as found by operating-points analysis. Circuits such as \
                latches or self-biased stages can settle into more than one state depending \
                on startup. The diagnostic names every net whose voltage differs by at least \
                half the maximum difference, checked during operating-point analysis on \
                circuits with at most 500 unknowns.",
};

/// Linearized operating-point rule flagging right-half-plane poles.
pub(crate) const UNSTABLE_POLE: Rule = Rule {
    name: "unstable-pole",
    level: Level::Warn,
    condition: "The circuit linearized at its operating point has a pole whose real part \
                exceeds 1e-6 times its magnitude. A natural mode therefore grows away from \
                the operating point, such as an emerging oscillation or a tipping latch. \
                Each real pole or conjugate pair raises one diagnostic naming nets with at \
                least half the peak mode amplitude, checked on circuits of at most 500 \
                unknowns whose linearization fits within max_memory.",
};

/// Runs operating-point and linearized stability rules on `simulator` under `options`.
///
/// Reports `gmin-dependence` first; for circuits of at most [`MAX_LINEARIZED`] unknowns, also
/// checks `multiple-operating-points` and `unstable-pole` (ordered by decreasing growth rate).
/// Reuses the cached operating point when already solved under matching options.
///
/// # Errors
///
/// Returns errors from [`Simulator::gmin_dependence`], [`Simulator::operating_points`], and
/// [`Simulator::growing_modes`], except memory-limit and transmission-line errors from
/// linearization, which skip `unstable-pole` silently.
pub fn check_operating_point(
    simulator: &mut Simulator,
    options: &Options,
) -> Result<Vec<Diagnostic>, SimulationError> {
    let (point, moves) = simulator.gmin_dependence(options)?;
    let mut diagnostics: Vec<Diagnostic> = gmin_dependence(simulator, &point.values, &moves)
        .into_iter()
        .collect();
    if simulator.unknowns().len() <= MAX_LINEARIZED {
        let points = simulator.operating_points(&OperatingPoints { max_solutions: 2 }, options)?;
        if let [point, other] = points.as_slice() {
            diagnostics.push(multiple_operating_points(simulator, point, other));
        }
        // Circuits whose linearization exceeds `options.max_memory` or contains a delay line
        // skip `unstable-pole` while keeping the other operating-point checks.
        match simulator.growing_modes(options) {
            Ok(growing) => {
                diagnostics.extend(growing.iter().map(|mode| unstable_pole(simulator, mode)));
            }
            Err(SimulationError::TooMuchMemory { .. } | SimulationError::Delayed { .. }) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(diagnostics)
}

/// Pairs external voltage unknown names from `simulator` with corresponding entries of `items`.
fn voltage_nets<T>(
    simulator: &Simulator,
    items: impl IntoIterator<Item = T>,
) -> impl Iterator<Item = (&str, T)> {
    simulator
        .unknowns()
        .iter()
        .zip(items)
        .filter(|(unknown, _)| unknown.nature == Nature::Voltage && !unknown.name.contains(':'))
        .map(|(unknown, item)| (unknown.name.as_str(), item))
}

/// Keeps the entries whose score is at least [`PROMINENCE_THRESHOLD`] times the maximum,
/// sorted by decreasing score, dropping all entries when the maximum is zero.
fn retain_prominent<T>(entries: &mut Vec<(f64, T)>) {
    let largest = entries.iter().map(|&(score, _)| score).fold(0.0, f64::max);
    if largest == 0.0 {
        entries.clear();
    } else {
        entries.retain(|&(score, _)| score >= PROMINENCE_THRESHOLD * largest);
    }
    entries.sort_by(|a, b| b.0.total_cmp(&a.0));
}

/// Builds an operating-point diagnostic for `rule`, appending `clause` and `named` to
/// `prefix` when `named` is non-empty.
fn net_diagnostic<'a>(
    rule: &Rule,
    prefix: &str,
    clause: &str,
    named: &str,
    nets: impl IntoIterator<Item = &'a str>,
) -> Diagnostic {
    let message = if named.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}{clause}{named}")
    };
    Diagnostic {
        rule: rule.name,
        message,
        instances: Vec::new(),
        nets: nets.into_iter().map(str::to_owned).collect(),
    }
}

/// Builds the `gmin-dependence` diagnostic for nets whose shifts in `moves` exceed the
/// threshold relative to `values`, if any.
fn gmin_dependence(simulator: &Simulator, values: &[f64], moves: &[f64]) -> Option<Diagnostic> {
    let mut nets: Vec<(&str, f64, f64)> = voltage_nets(simulator, values.iter().zip(moves))
        .filter(|&(_, (&value, &moved))| moved.abs() > GMIN_MOVE * value.abs().max(VOLT))
        .map(|(net, (&value, &moved))| (net, value, moved))
        .collect();
    if nets.is_empty() {
        return None;
    }
    nets.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
    let named = prose(&nets, NAMED, |&(net, value, moved)| {
        format!("`{net}` from {value:?} V by {moved:?} V")
    });
    Some(net_diagnostic(
        &GMIN_DEPENDENCE,
        "the operating point depends on gmin: to first order, doubling it moves ",
        "",
        &named,
        nets.iter().map(|&(net, ..)| net),
    ))
}

/// Builds the `multiple-operating-points` diagnostic comparing `other` against `point`.
fn multiple_operating_points(
    simulator: &Simulator,
    point: &OperatingPointResult,
    other: &OperatingPointResult,
) -> Diagnostic {
    let mut nets: Vec<(f64, (&str, f64, f64))> =
        voltage_nets(simulator, point.values.iter().zip(&other.values))
            .map(|(net, (&value, &another))| ((another - value).abs(), (net, value, another)))
            .collect();
    retain_prominent(&mut nets);
    let named = prose(&nets, NAMED, |&(_, (net, value, another))| {
        format!("`{net}` at {another:?} V rather than {value:?} V")
    });
    net_diagnostic(
        &MULTIPLE_OPERATING_POINTS,
        "the circuit has another DC solution",
        ", where ",
        &named,
        nets.iter().map(|&(_, (net, ..))| net),
    )
}

/// Builds the `unstable-pole` diagnostic for a growing natural `mode`.
fn unstable_pole(simulator: &Simulator, mode: &Mode) -> Diagnostic {
    let mut nets: Vec<(f64, &str)> = voltage_nets(simulator, &mode.shape)
        .map(|(net, amplitude)| (amplitude.norm(), net))
        .collect();
    retain_prominent(&mut nets);
    let named = list(nets.iter().map(|&(_, net)| net), NAMED);
    let Mode { pole, .. } = mode;
    let growth = if pole.im > 0.0 {
        format!(
            "the operating point is unstable: its poles at {:?} ± {:?}j rad/s have a positive \
             real part, so an oscillation near {:?} Hz grows from it",
            pole.re,
            pole.im,
            pole.im / TAU
        )
    } else {
        format!(
            "the operating point is unstable: its pole at {:?} rad/s is positive, so a \
             disturbance grows from it",
            pole.re
        )
    };
    net_diagnostic(
        &UNSTABLE_POLE,
        &growth,
        ", largest at ",
        &named,
        nets.iter().map(|&(_, net)| net),
    )
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::testing::simulator;

    /// Runs operating-point checks on a test circuit with ground port `0` and `instances`.
    fn diagnostics(instances: &Value) -> Vec<Diagnostic> {
        check_operating_point(&mut simulator(instances), &Options::default())
            .expect("an operating point")
    }

    #[test]
    fn the_checks_solve_the_operating_point_once() {
        // Gmin-dependence, the search for another DC solution and the modes that grow each
        // solved the operating point from zero, three times in all. The simulator
        // remembers it now: the checks solve it once, and none after an analysis that did.
        let options = Options::default();
        let mut checked = simulator(&tank(5e-5));
        assert_eq!(check_operating_point(&mut checked, &options).unwrap(), []);
        assert_eq!(checked.operating_point_solves(), 1);
        let mut analyzed = simulator(&tank(5e-5));
        analyzed.operating_point(&options).unwrap();
        assert_eq!(check_operating_point(&mut analyzed, &options).unwrap(), []);
        assert_eq!(analyzed.operating_point_solves(), 1);
    }

    fn two_terminal(name: &str, (device, parameter, value): (&str, &str, f64), a: &str) -> Value {
        json!({"name": name, "of": {"model": device}, "parameters": {parameter: value},
               "terminals": {"a": a, "b": "0"}})
    }

    /// Builds an RLC tank (1 nF, 1 mH, 10 kΩ) on node `n` with positive transconductance
    /// `feedback`.
    fn tank(feedback: f64) -> Value {
        json!([
            two_terminal("R1", ("resistor", "resistance", 1e4), "n"),
            two_terminal("C1", ("capacitor", "capacitance", 1e-9), "n"),
            two_terminal("L1", ("inductor", "inductance", 1e-3), "n"),
            {"name": "G1", "of": {"model": "voltage_controlled_current_source"},
             "parameters": {"transconductance": feedback},
             "terminals": {"positive": "0", "negative": "n",
                           "control_positive": "n", "control_negative": "0"}},
        ])
    }

    /// Builds a circuit driving `current` into node `n` across a reverse-biased diode and an
    /// optional resistive `load` to ground.
    fn leaking(current: f64, load: Option<f64>) -> Value {
        let mut instances = vec![
            json!({"name": "I1", "of": {"model": "current_source"},
                   "parameters": {"waveform": {"dc": current}},
                   "terminals": {"positive": "0", "negative": "n"}}),
            json!({"name": "D1", "of": {"model": "diode"}, "parameters": {},
                   "terminals": {"anode": "0", "cathode": "n"}}),
        ];
        if let Some(resistance) = load {
            instances.push(two_terminal(
                "R1",
                ("resistor", "resistance", resistance),
                "n",
            ));
        }
        Value::Array(instances)
    }

    #[test]
    fn gmin_dependence() {
        // Only the diode's gmin holds `n`: 1 nA, less the diode's 10 fA, lifts it to
        // 999.99 V. Doubling gmin halves that, which to first order moves it by all of it.
        let raised = diagnostics(&leaking(1e-9, None));
        assert_eq!(raised.len(), 1, "{raised:?}");
        assert_eq!(raised[0].rule, "gmin-dependence");
        assert_eq!(raised[0].nets, ["n"]);
        let message = &raised[0].message;
        assert!(
            message.starts_with(
                "the operating point depends on gmin: to first order, doubling it moves `n` \
                 from 999.98"
            ) && message.contains(" V by -999.98"),
            "{message}"
        );
        // A megohm holds it at 1 mV, which doubling gmin moves by a picovolt.
        assert_eq!(diagnostics(&leaking(1e-9, Some(1e6))), []);
    }

    #[test]
    fn multiple_operating_points() {
        // 1 mA/V³ · v (v − 1)(v − 2) = (1 V − v) / 10 kΩ at `x`: v = 1 or 1 ± √0.9.
        let cubic = json!([
            {"name": "V1", "of": {"model": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
             "terminals": {"positive": "in", "negative": "0"}},
            {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 1e4},
             "terminals": {"a": "in", "b": "x"}},
            {"name": "B1", "of": {"model": "behavioral_current_source"},
             "parameters": {"expression": "1e-3 * v(x) * (v(x) - 1) * (v(x) - 2)"},
             "terminals": {"positive": "x", "negative": "0", "x": "x"}},
        ]);
        let raised = diagnostics(&cubic);
        let found: Vec<&str> = raised.iter().map(|diagnostic| diagnostic.rule).collect();
        assert!(found.contains(&"multiple-operating-points"), "{raised:?}");
        let diagnostic = &raised[found
            .iter()
            .position(|&rule| rule == "multiple-operating-points")
            .unwrap()];
        assert_eq!(diagnostic.nets, ["x"]);
        assert!(
            diagnostic
                .message
                .starts_with("the circuit has another DC solution, where `x` at "),
            "{}",
            diagnostic.message
        );
        // A megohm across the leaking diode's net leaves it one.
        assert_eq!(diagnostics(&leaking(1e-9, Some(1e6))), []);
    }

    #[test]
    fn unstable_pole() {
        // More feedback than loss: the tank's oscillation grows.
        let raised = diagnostics(&tank(2e-4));
        assert_eq!(raised.len(), 1, "{raised:?}");
        assert_eq!(raised[0].rule, "unstable-pole");
        assert_eq!(raised[0].nets, ["n"]);
        // C s² − 1e-4 s + 1/L = 0: s = 5e4 ± 998749.2j.
        let message = &raised[0].message;
        let pole = message
            .strip_prefix("the operating point is unstable: its poles at ")
            .and_then(|rest| rest.split_once(" ± "))
            .and_then(|(real, rest)| Some((real.parse::<f64>().ok()?, rest.split_once('j')?)))
            .and_then(|(real, (imaginary, _))| Some((real, imaginary.parse::<f64>().ok()?)));
        let (real, imaginary) = pole.unwrap_or_else(|| panic!("{message}"));
        assert!((real - 5e4).abs() < 1e-6 && (imaginary - 998_749.217_771_9).abs() < 1e-4);
        assert!(
            message.ends_with("Hz grows from it, largest at `n`"),
            "{message}"
        );
        // Less feedback than loss: it dies away.
        assert_eq!(diagnostics(&tank(5e-5)), []);
    }

    #[test]
    fn a_growing_real_pole_tips_the_circuit() {
        let raised = diagnostics(&json!([
            two_terminal("R1", ("resistor", "resistance", 1e3), "n"),
            two_terminal("C1", ("capacitor", "capacitance", 1e-9), "n"),
            {"name": "G1", "of": {"model": "voltage_controlled_current_source"},
             "parameters": {"transconductance": 2e-3},
             "terminals": {"positive": "0", "negative": "n",
                           "control_positive": "n", "control_negative": "0"}},
        ]));
        assert_eq!(raised.len(), 1, "{raised:?}");
        assert!(
            raised[0]
                .message
                .starts_with("the operating point is unstable: its pole at 1000000."),
            "{}",
            raised[0].message
        );
    }

    #[test]
    fn a_linearization_past_the_memory_skips_growing_modes_alone() {
        // The memory holds not even the pencil of this one-unknown circuit, so it goes
        // unchecked for growing modes; the net only the diode's gmin holds is still found.
        let options = Options {
            max_memory: 1e-12,
            ..Options::default()
        };
        let found = check_operating_point(&mut simulator(&leaking(1e-9, None)), &options)
            .expect("the rules but growing modes");
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].rule, "gmin-dependence");
    }

    #[test]
    fn a_transmission_line_skips_growing_modes_alone() {
        // Transmission-line delay produces infinitely many poles that no finite pencil holds.
        let found = diagnostics(&json!([
            {"name": "V1", "of": {"model": "voltage_source"},
             "parameters": {"waveform": {"dc": 1}},
             "terminals": {"positive": "a", "negative": "0"}},
            {"name": "T1", "of": {"model": "transmission_line"},
             "parameters": {"characteristic_impedance": 50, "delay": 1e-9},
             "terminals": {"positive1": "a", "negative1": "0",
                           "positive2": "b", "negative2": "0"}},
            {"name": "R1", "of": {"model": "resistor"}, "parameters": {"resistance": 50},
             "terminals": {"a": "b", "b": "0"}},
        ]));
        assert_eq!(found, []);
    }

    #[test]
    fn multiple_operating_points_with_zero_external_voltage_diff_reports_no_nets() {
        let mut sim = simulator(&tank(5e-5));
        let point = sim.operating_point(&Options::default()).unwrap();
        let mut other = point.clone();
        let mut modified_current = false;
        for (unknown, value) in sim.unknowns().iter().zip(&mut other.values) {
            if unknown.nature != Nature::Voltage {
                *value += 1.0;
                modified_current = true;
            }
        }
        assert!(modified_current);
        assert!(
            sim.unknowns()
                .iter()
                .any(|u| u.nature == Nature::Voltage && !u.name.contains(':'))
        );

        let diagnostic = super::multiple_operating_points(&sim, &point, &other);
        assert_eq!(diagnostic.rule, "multiple-operating-points");
        assert!(diagnostic.nets.is_empty(), "{diagnostic:?}");
        assert_eq!(diagnostic.message, "the circuit has another DC solution");
    }

    #[test]
    fn unstable_pole_with_zero_external_voltage_amplitude_reports_no_nets() {
        let mut sim = simulator(&tank(2e-4));
        let modes = sim.growing_modes(&Options::default()).unwrap();
        let mut mode = modes.into_iter().next().expect("at least one growing mode");
        let mut zeroed_net = false;
        for (unknown, amplitude) in sim.unknowns().iter().zip(&mut mode.shape) {
            if unknown.nature == Nature::Voltage && !unknown.name.contains(':') {
                amplitude.re = 0.0;
                amplitude.im = 0.0;
                zeroed_net = true;
            }
        }
        assert!(zeroed_net);

        let diagnostic = super::unstable_pole(&sim, &mode);
        assert_eq!(diagnostic.rule, "unstable-pole");
        assert!(diagnostic.nets.is_empty(), "{diagnostic:?}");
        assert!(
            !diagnostic.message.contains("largest at"),
            "{}",
            diagnostic.message
        );
        assert!(
            diagnostic
                .message
                .starts_with("the operating point is unstable: "),
            "{}",
            diagnostic.message
        );
    }
}
