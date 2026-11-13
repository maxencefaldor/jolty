//! The rules on the circuit at its operating point, and linearized there.

use std::f64::consts::TAU;

use jolty_engine::{Mode, OperatingPoint, OperatingPoints, Options, SimulationError, Simulator};
use jolty_models::Quantity;

use crate::structure::{list, prose};
use crate::{Diagnostic, Level, Rule};

/// The most unknowns of a circuit the checks linearize: the dense pencil of 500 unknowns
/// takes the QZ algorithm about a second.
pub const MOST_LINEARIZED: usize = 500;

/// The fraction of a mode's largest net amplitude, or of the largest difference between two
/// DC solutions, from which a net is named.
const LARGEST: f64 = 0.5;

/// The most nets a diagnostic names before it counts the rest.
const NAMED: usize = 5;

/// The fraction of the larger of a node's voltage and [`VOLT`] that doubling gmin may move
/// it by before gmin-dependence names it.
const GMIN_MOVE: f64 = 1e-2;

/// One volt, below which gmin-dependence measures a node's move against it instead.
const VOLT: f64 = 1.0;

pub(crate) const GMIN_DEPENDENCE: Rule = Rule {
    name: "gmin-dependence",
    level: Level::Warn,
    condition: "Doubling gmin, the conductance the engine places across every junction to \
                help Newton iteration converge, would move a net's voltage at the operating \
                point, to first order, by more than a hundredth of the larger of that \
                voltage and 1 V: little but gmin holds the net, as between reverse-biased \
                junctions, so its voltage comes from the simulator rather than the circuit. \
                One diagnostic names every such net, by decreasing move. A simulation \
                checks it at its plan's operating point.",
};

pub(crate) const MULTIPLE_OPERATING_POINTS: Rule = Rule {
    name: "multiple-operating-points",
    level: Level::Warn,
    condition: "The circuit's DC equations have a solution besides its operating point, as \
                an operating-points analysis finds them: the circuit can rest in more than \
                one state, as a latch does, or a bias circuit that can also rest with no \
                current, and which it rests in depends on how it starts. The diagnostic \
                names the nets whose voltages differ between the two by at least half the \
                largest difference. A simulation checks it at its plan's operating point, \
                on circuits of at most 500 unknowns.",
};

pub(crate) const UNSTABLE_POLE: Rule = Rule {
    name: "unstable-pole",
    level: Level::Warn,
    condition: "The circuit linearized at its operating point has a pole whose real part is \
                above a millionth of its magnitude: a natural mode grows away from the \
                operating point, as an oscillation that builds up or a latch that tips, so \
                the circuit does not rest there. A real pole, or a conjugate pair, raises one \
                diagnostic, naming the nets where its mode's amplitude is at least half its \
                largest. A simulation checks it at its plan's operating point, on circuits \
                of at most 500 unknowns.",
};

/// Every diagnostic of the rules on the circuit `simulator` elaborated, at its operating
/// point under `options` and linearized there, rule by rule: the nets gmin holds, then, on
/// circuits of at most [`MOST_LINEARIZED`] unknowns, another DC solution and the modes
/// that grow, by decreasing growth.
///
/// The operating point is the one the simulator remembers, if an analysis solved it under
/// the same options, or else solved once for every rule.
///
/// # Errors
///
/// Returns the errors of [`Simulator::gmin_dependence`], [`Simulator::operating_points`]
/// and [`Simulator::growing_modes`].
pub fn check_operating_point(
    simulator: &mut Simulator,
    options: &Options,
) -> Result<Vec<Diagnostic>, SimulationError> {
    let (point, moves) = simulator.gmin_dependence(options)?;
    let mut diagnostics: Vec<Diagnostic> = gmin_dependence(simulator, &point.values, &moves)
        .into_iter()
        .collect();
    if simulator.unknowns().len() <= MOST_LINEARIZED {
        let points = simulator.operating_points(&OperatingPoints { most: 2 }, options)?;
        if let [point, other] = points.as_slice() {
            diagnostics.push(multiple_operating_points(simulator, point, other));
        }
        let growing = simulator.growing_modes(options)?;
        diagnostics.extend(growing.iter().map(|mode| unstable_pole(simulator, mode)));
    }
    Ok(diagnostics)
}

/// The diagnostic of the nets at `values` that doubling gmin `moves` too far, if any.
fn gmin_dependence(simulator: &Simulator, values: &[f64], moves: &[f64]) -> Option<Diagnostic> {
    let mut nets: Vec<(&str, f64, f64)> = simulator
        .unknowns()
        .iter()
        .zip(values.iter().zip(moves))
        .filter(|(unknown, (value, moved))| {
            unknown.quantity == Quantity::Voltage
                && !unknown.name.contains(':')
                && moved.abs() > GMIN_MOVE * value.abs().max(VOLT)
        })
        .map(|(unknown, (&value, &moved))| (unknown.name.as_str(), value, moved))
        .collect();
    if nets.is_empty() {
        return None;
    }
    nets.sort_by(|a, b| b.2.abs().total_cmp(&a.2.abs()));
    let named = prose(
        nets.iter()
            .map(|(net, value, moved)| format!("`{net}` from {value:?} V by {moved:?} V")),
        NAMED,
    );
    Some(Diagnostic {
        rule: GMIN_DEPENDENCE.name,
        message: format!("the operating point depends on gmin: doubling it moves {named}"),
        instances: Vec::new(),
        nets: nets.iter().map(|&(net, ..)| net.to_owned()).collect(),
    })
}

/// The diagnostic of the DC solution `other` besides the operating point `point`.
fn multiple_operating_points(
    simulator: &Simulator,
    point: &OperatingPoint,
    other: &OperatingPoint,
) -> Diagnostic {
    let mut nets: Vec<(&str, f64, f64)> = simulator
        .unknowns()
        .iter()
        .zip(point.values.iter().zip(&other.values))
        .filter(|(unknown, _)| unknown.quantity == Quantity::Voltage && !unknown.name.contains(':'))
        .map(|(unknown, (&value, &another))| (unknown.name.as_str(), value, another))
        .collect();
    let difference = |&(_, value, another): &(&str, f64, f64)| (another - value).abs();
    let largest = nets.iter().map(difference).fold(0.0, f64::max);
    nets.retain(|net| difference(net) >= LARGEST * largest);
    nets.sort_by(|a, b| difference(b).total_cmp(&difference(a)));
    let named = prose(
        nets.iter().map(|(net, value, another)| {
            format!("`{net}` at {another:?} V rather than {value:?} V")
        }),
        NAMED,
    );
    let place = if named.is_empty() {
        String::new()
    } else {
        format!(", where {named}")
    };
    Diagnostic {
        rule: MULTIPLE_OPERATING_POINTS.name,
        message: format!("the circuit has another DC solution{place}"),
        instances: Vec::new(),
        nets: nets.iter().map(|&(net, ..)| net.to_owned()).collect(),
    }
}

/// The diagnostic of the growing `mode`.
fn unstable_pole(simulator: &Simulator, mode: &Mode) -> Diagnostic {
    let mut nets: Vec<(f64, &str)> = simulator
        .unknowns()
        .iter()
        .zip(&mode.shape)
        .filter(|(unknown, _)| unknown.quantity == Quantity::Voltage && !unknown.name.contains(':'))
        .map(|(unknown, amplitude)| (amplitude.norm(), unknown.name.as_str()))
        .collect();
    let largest = nets
        .iter()
        .map(|&(amplitude, _)| amplitude)
        .fold(0.0, f64::max);
    nets.retain(|&(amplitude, _)| amplitude >= LARGEST * largest);
    nets.sort_by(|a, b| b.0.total_cmp(&a.0));
    let named = list(nets.iter().map(|&(_, net)| net), NAMED);
    let Mode { pole, .. } = mode;
    let growth = if pole.im > 0.0 {
        format!(
            "its poles at {:?} ± {:?}j rad/s have a positive real part, so an oscillation \
             near {:?} Hz grows from it",
            pole.re,
            pole.im,
            pole.im / TAU
        )
    } else {
        format!(
            "its pole at {:?} rad/s is positive, so a disturbance grows from it",
            pole.re
        )
    };
    let place = if named.is_empty() {
        String::new()
    } else {
        format!(", largest at {named}")
    };
    Diagnostic {
        rule: UNSTABLE_POLE.name,
        message: format!("the operating point is unstable: {growth}{place}"),
        instances: Vec::new(),
        nets: nets.iter().map(|&(_, net)| net.to_owned()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use jolty_engine::Conditions;
    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::{Value, json};

    use super::*;

    /// A simulator of a circuit whose top component has a ground port `0` and places
    /// `instances`.
    fn simulator(instances: &Value) -> Simulator {
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "test",
            "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                    "instances": instances}}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit")
    }

    /// The diagnostics at the operating point of a circuit whose top component has a ground
    /// port `0` and places `instances`.
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
        assert!(
            check_operating_point(&mut checked, &options)
                .unwrap()
                .is_empty()
        );
        assert_eq!(checked.operating_point_solves(), 1);
        let mut analyzed = simulator(&tank(5e-5));
        analyzed.operating_point(&options).unwrap();
        assert!(
            check_operating_point(&mut analyzed, &options)
                .unwrap()
                .is_empty()
        );
        assert_eq!(analyzed.operating_point_solves(), 1);
    }

    fn two_terminal(id: &str, (device, parameter, value): (&str, &str, f64), a: &str) -> Value {
        json!({"id": id, "of": {"device": device}, "parameters": {parameter: value},
               "terminals": {"a": a, "b": "0"}})
    }

    /// A tank of 1 nF and 1 mH at `node`, losing 1e-4 S through 10 kΩ, and a transconductance
    /// of `feedback` feeding the node's voltage back into it.
    fn tank(feedback: f64) -> Value {
        json!([
            two_terminal("R1", ("resistor", "resistance", 1e4), "n"),
            two_terminal("C1", ("capacitor", "capacitance", 1e-9), "n"),
            two_terminal("L1", ("inductor", "inductance", 1e-3), "n"),
            {"id": "G1", "of": {"device": "voltage_controlled_current_source"},
             "parameters": {"transconductance": feedback},
             "terminals": {"positive": "0", "negative": "n",
                           "control_positive": "n", "control_negative": "0"}},
        ])
    }

    /// A current source of `current` into `n`, across a reverse-biased diode, with `load`
    /// from `n` to the ground, if any.
    fn leaking(current: f64, load: Option<f64>) -> Value {
        let mut instances = vec![
            json!({"id": "I1", "of": {"device": "current_source"},
                   "parameters": {"waveform": {"dc": current}},
                   "terminals": {"positive": "0", "negative": "n"}}),
            json!({"id": "D1", "of": {"device": "diode"}, "parameters": {},
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
                "the operating point depends on gmin: doubling it moves `n` from 999.98"
            ) && message.contains(" V by -999.98"),
            "{message}"
        );
        // A megohm holds it at 1 mV, which doubling gmin moves by a picovolt.
        assert!(diagnostics(&leaking(1e-9, Some(1e6))).is_empty());
    }

    #[test]
    fn multiple_operating_points() {
        // 1 mA/V³ · v (v − 1)(v − 2) = (1 V − v) / 10 kΩ at `x`: v = 1 or 1 ± √0.9.
        let cubic = json!([
            {"id": "V1", "of": {"device": "voltage_source"}, "parameters": {"waveform": {"dc": 1.0}},
             "terminals": {"positive": "in", "negative": "0"}},
            {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e4},
             "terminals": {"a": "in", "b": "x"}},
            {"id": "B1", "of": {"device": "behavioral_current_source"},
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
        assert!(diagnostics(&leaking(1e-9, Some(1e6))).is_empty());
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
        assert!(diagnostics(&tank(5e-5)).is_empty());
    }

    #[test]
    fn a_growing_real_pole_tips_the_circuit() {
        let raised = diagnostics(&json!([
            two_terminal("R1", ("resistor", "resistance", 1e3), "n"),
            two_terminal("C1", ("capacitor", "capacitance", 1e-9), "n"),
            {"id": "G1", "of": {"device": "voltage_controlled_current_source"},
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
}
