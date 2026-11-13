//! The rules on the circuit at its operating point, and linearized there.

use std::f64::consts::TAU;

use jolty_engine::{Mode, Options, SimulationError, Simulator};
use jolty_models::Quantity;

use crate::structure::{list, prose};
use crate::{Diagnostic, Level, Rule};

/// The most unknowns of a circuit the checks linearize: the dense pencil of 500 unknowns
/// takes the QZ algorithm about a second.
pub const MOST_LINEARIZED: usize = 500;

/// The fraction of a mode's largest net amplitude above which a net is named as where the
/// mode is largest.
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
/// point under `options` and linearized there, rule by rule: the nets gmin holds, then the
/// modes that grow, by decreasing growth, on circuits of at most [`MOST_LINEARIZED`]
/// unknowns.
///
/// # Errors
///
/// Returns the errors of [`Simulator::gmin_dependence`] and [`Simulator::growing_modes`].
pub fn check_operating_point(
    simulator: &mut Simulator,
    options: &Options,
) -> Result<Vec<Diagnostic>, SimulationError> {
    let (point, moves) = simulator.gmin_dependence(options)?;
    let mut diagnostics: Vec<Diagnostic> = gmin_dependence(simulator, &point.values, &moves)
        .into_iter()
        .collect();
    if simulator.unknowns().len() <= MOST_LINEARIZED {
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

    /// The diagnostics at the operating point of a circuit whose top component has a ground
    /// port `0` and places `instances`.
    fn diagnostics(instances: &Value) -> Vec<Diagnostic> {
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "test",
            "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                    "instances": instances}}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        let mut simulator = Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("an elaborated circuit");
        check_operating_point(&mut simulator, &Options::default()).expect("an operating point")
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
    fn unstable_pole() {
        // More feedback than loss: the tank's oscillation grows.
        let raised = diagnostics(&tank(2e-4));
        assert_eq!(raised.len(), 1, "{raised:?}");
        assert_eq!(raised[0].rule, "unstable-pole");
        assert_eq!(raised[0].nets, ["n"]);
        // C s² − 1e-4 s + 1/L = 0: s = 5e4 ± 998749.2j.
        let message = &raised[0].message;
        assert!(
            message.starts_with("the operating point is unstable: its poles at 50000.0")
                && message.contains(" ± 998749.21")
                && message.ends_with("Hz grows from it, largest at `n`"),
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
