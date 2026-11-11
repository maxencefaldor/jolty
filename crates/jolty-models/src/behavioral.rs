use jolty_expressions::{Expression, Variable, Workspace};
use jolty_numerics::Entry;
use serde::Deserialize;

use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Node, Output,
    ParameterError, Quantity, Unknown,
};

/// An expression a behavioral source follows: an [expression](jolty_expressions) of the
/// voltages of terminals it reads, of the currents through sensing ports of its own, and of
/// time.
///
/// Its terminals are `positive` and `negative`, then each terminal it reads a voltage of,
/// then the two of each sensing port `p` it reads a current through, `p_positive` and
/// `p_negative`. A port is a short circuit, its current, from `p_positive` through the port
/// to `p_negative`, an unknown labelled `p`.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "String")]
pub struct Behavior {
    expression: Expression,
    terminals: Vec<String>,
    /// What each variable of the expression reads.
    reads: Vec<Reading>,
    /// Each sensing port's name and the indices of its two terminals.
    ports: Vec<(String, [usize; 2])>,
}

/// What one variable reads: the voltage of a terminal over another or over the reference,
/// by their indices among the terminals; the current through a port, by its index among
/// the ports; or time.
#[derive(Debug, Clone, Copy)]
enum Reading {
    Voltage(usize, Option<usize>),
    Current(usize),
    Time,
}

impl TryFrom<String> for Behavior {
    type Error = String;

    fn try_from(text: String) -> Result<Self, String> {
        let expression: Expression = text
            .parse()
            .map_err(|error| format!("the expression `{text}` is invalid: {error}"))?;
        let mut terminals = vec!["positive".to_owned(), "negative".to_owned()];
        let mut index = |name: &str| {
            terminals
                .iter()
                .position(|terminal| terminal == name)
                .unwrap_or_else(|| {
                    terminals.push(name.to_owned());
                    terminals.len() - 1
                })
        };
        // The terminals read come before the ports', whatever the order they are read in.
        for variable in expression.variables() {
            if let Variable::Voltage(positive, negative) = variable {
                index(positive);
                negative.as_deref().map(&mut index);
            }
        }
        let mut reads = Vec::new();
        let mut ports: Vec<(String, [usize; 2])> = Vec::new();
        for variable in expression.variables() {
            reads.push(match variable {
                Variable::Voltage(positive, negative) => {
                    Reading::Voltage(index(positive), negative.as_deref().map(&mut index))
                }
                Variable::Current(port) if port == "current" => {
                    return Err(format!(
                        "the expression `{text}` reads `i(current)`, but `current` names \
                         the source's own current"
                    ));
                }
                Variable::Current(port) => {
                    let ends = [
                        index(&format!("{port}_positive")),
                        index(&format!("{port}_negative")),
                    ];
                    ports.push((port.clone(), ends));
                    Reading::Current(ports.len() - 1)
                }
                Variable::Time => Reading::Time,
            });
        }
        Ok(Self {
            expression,
            terminals,
            reads,
            ports,
        })
    }
}

impl Behavior {
    /// The interface of a source with this behavior whose output is a branch of `kind`.
    fn interface(&self, kind: BranchKind) -> Interface {
        let mut branches = vec![Branch::new(0, 1, kind)];
        branches.extend(self.ports.iter().map(|&(_, [positive, negative])| {
            Branch::new(positive, negative, BranchKind::VoltageSource)
        }));
        Interface {
            terminals: self.terminals.clone(),
            branches,
        }
    }
}

/// The parameters of a [`BehavioralVoltageSource`] or a [`BehavioralCurrentSource`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehavioralParameters {
    /// What the source's output follows.
    pub expression: Behavior,
}

/// A sensing port: a short circuit between two terminals, whose current is an unknown.
#[derive(Debug, Clone, Copy)]
struct Port {
    current: Unknown,
    nodes: [Node; 2],
    /// The entries at (each terminal, current) and (current, each terminal).
    entries: [Option<Entry>; 4],
}

impl Port {
    fn new(binding: &mut Binding<'_>, label: &str, nodes: [Node; 2]) -> Self {
        let current = binding.internal(label, Quantity::Current);
        let row = Some(current);
        Self {
            current,
            nodes,
            entries: [
                binding.entry(nodes[0], row),
                binding.entry(nodes[1], row),
                binding.entry(row, nodes[0]),
                binding.entry(row, nodes[1]),
            ],
        }
    }

    /// Adds the port's current to its terminals, and its equation: no voltage across it.
    fn add(&self, input: &Input<'_>, output: &mut Output<'_>) {
        let row = Some(self.current);
        let [positive, negative] = self.nodes;
        let current = input.value(row);
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        output.add_residual(row, input.value(positive) - input.value(negative));
        let [
            positive_current,
            negative_current,
            current_positive,
            current_negative,
        ] = self.entries;
        output.add_conductance(positive_current, 1.0);
        output.add_conductance(negative_current, -1.0);
        output.add_conductance(current_positive, 1.0);
        output.add_conductance(current_negative, -1.0);
    }
}

/// A behavior bound to the unknowns it reads and the equations it enters.
#[derive(Debug)]
struct Formula {
    expression: Expression,
    workspace: Workspace,
    /// What each variable reads: the difference of two nodes, a port's current, or time.
    reads: Vec<Read>,
    ports: Vec<Port>,
    /// The equations the expression enters, each with the factor it enters with.
    rows: Vec<(Node, f64)>,
    /// For each variable, the entries of its derivative in each row, with their factors.
    entries: Vec<Vec<(Option<Entry>, f64)>>,
    values: Vec<f64>,
    gradient: Vec<f64>,
}

/// What one variable of an expression reads.
#[derive(Debug, Clone, Copy)]
enum Read {
    Voltage(Node, Node),
    Current(Unknown),
    Time,
}

impl Formula {
    /// Binds `behavior`, which enters each of `rows` with its factor.
    fn new(behavior: Behavior, binding: &mut Binding<'_>, rows: Vec<(Node, f64)>) -> Self {
        let Behavior {
            expression,
            reads: readings,
            ports: port_ends,
            ..
        } = behavior;
        let ports: Vec<Port> = port_ends
            .iter()
            .map(|(label, ends)| {
                let nodes = ends.map(|index| binding.terminal(index));
                Port::new(binding, label, nodes)
            })
            .collect();
        let reads: Vec<Read> = readings
            .iter()
            .map(|reading| match *reading {
                Reading::Voltage(positive, negative) => Read::Voltage(
                    binding.terminal(positive),
                    negative.and_then(|index| binding.terminal(index)),
                ),
                Reading::Current(port) => Read::Current(ports[port].current),
                Reading::Time => Read::Time,
            })
            .collect();
        let entries = reads
            .iter()
            .map(|read| {
                let columns: Vec<(Node, f64)> = match *read {
                    Read::Voltage(positive, negative) => vec![(positive, 1.0), (negative, -1.0)],
                    Read::Current(current) => vec![(Some(current), 1.0)],
                    Read::Time => Vec::new(),
                };
                let mut entries = Vec::with_capacity(rows.len() * columns.len());
                for &(row, factor) in &rows {
                    for &(column, sign) in &columns {
                        entries.push((binding.entry(row, column), factor * sign));
                    }
                }
                entries
            })
            .collect();
        let count = reads.len();
        Self {
            workspace: Workspace::new(&expression),
            expression,
            reads,
            ports,
            rows,
            entries,
            values: vec![0.0; count],
            gradient: vec![0.0; count],
        }
    }

    /// Adds the expression's value to its rows, with its derivatives, and the ports'
    /// equations.
    fn add(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        for (value, read) in self.values.iter_mut().zip(&self.reads) {
            *value = match *read {
                Read::Voltage(positive, negative) => input.value(positive) - input.value(negative),
                Read::Current(current) => input.value(Some(current)),
                Read::Time => input.time,
            };
        }
        if !input.limiting {
            self.workspace.forget();
        }
        let value = self
            .expression
            .evaluate(&self.values, &mut self.workspace, &mut self.gradient);
        if self.workspace.limited() {
            output.mark_limited();
        }
        for &(row, factor) in &self.rows {
            output.add_residual(row, factor * value);
        }
        for (entries, derivative) in self.entries.iter().zip(&self.gradient) {
            for &(entry, factor) in entries {
                output.add_conductance(entry, factor * derivative);
            }
        }
        for port in &self.ports {
            port.add(input, output);
        }
    }
}

/// A current source whose current follows an expression of the voltages and currents it
/// reads: see [`Behavior`].
///
/// It drives the expression's value from `positive` through the source to `negative`.
#[derive(Debug)]
pub struct BehavioralCurrentSource {
    formula: Formula,
}

impl NativeDevice for BehavioralCurrentSource {
    const NAME: &'static str = "behavioral_current_source";
    type Parameters = BehavioralParameters;

    fn interface(parameters: &BehavioralParameters) -> Interface {
        parameters.expression.interface(BranchKind::CurrentSource)
    }

    fn new(
        parameters: BehavioralParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let rows = vec![(binding.terminal(0), 1.0), (binding.terminal(1), -1.0)];
        Ok(Self {
            formula: Formula::new(parameters.expression, binding, rows),
        })
    }
}

impl Device for BehavioralCurrentSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        self.formula.add(input, output);
    }
}

/// A voltage source whose voltage follows an expression of the voltages and currents it
/// reads: see [`Behavior`].
///
/// It holds `v_positive − v_negative` at the expression's value. Its current, flowing from
/// `positive` through the source to `negative`, is an unknown labelled `current`.
#[derive(Debug)]
pub struct BehavioralVoltageSource {
    formula: Formula,
    nodes: [Node; 2],
    current: Unknown,
    /// The entries at (each terminal, current) and (current, each terminal).
    entries: [Option<Entry>; 4],
}

impl NativeDevice for BehavioralVoltageSource {
    const NAME: &'static str = "behavioral_voltage_source";
    type Parameters = BehavioralParameters;

    fn interface(parameters: &BehavioralParameters) -> Interface {
        parameters.expression.interface(BranchKind::VoltageSource)
    }

    fn new(
        parameters: BehavioralParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let nodes = [binding.terminal(0), binding.terminal(1)];
        let current = binding.internal("current", Quantity::Current);
        let row = Some(current);
        let entries = [
            binding.entry(nodes[0], row),
            binding.entry(nodes[1], row),
            binding.entry(row, nodes[0]),
            binding.entry(row, nodes[1]),
        ];
        Ok(Self {
            formula: Formula::new(parameters.expression, binding, vec![(row, -1.0)]),
            nodes,
            current,
            entries,
        })
    }
}

impl Device for BehavioralVoltageSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let row = Some(self.current);
        let [positive, negative] = self.nodes;
        let current = input.value(row);
        output.add_residual(positive, current);
        output.add_residual(negative, -current);
        output.add_residual(row, input.value(positive) - input.value(negative));
        let [
            positive_current,
            negative_current,
            current_positive,
            current_negative,
        ] = self.entries;
        output.add_conductance(positive_current, 1.0);
        output.add_conductance(negative_current, -1.0);
        output.add_conductance(current_positive, 1.0);
        output.add_conductance(current_negative, -1.0);
        self.formula.add(input, output);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{Bench, assert_close, assert_jacobians_match};

    fn parameters(expression: &str) -> String {
        format!(r#"{{"expression": "{expression}"}}"#)
    }

    #[test]
    fn terminals_follow_what_the_expression_reads() {
        let behavior = Behavior::try_from("v(a, positive) * i(sense) + v(b)".to_owned())
            .expect("a valid behavior");
        let interface = behavior.interface(BranchKind::CurrentSource);
        assert_eq!(
            interface.terminals,
            [
                "positive",
                "negative",
                "a",
                "b",
                "sense_positive",
                "sense_negative"
            ]
        );
        assert_eq!(
            interface.branches,
            [
                Branch::new(0, 1, BranchKind::CurrentSource),
                Branch::new(4, 5, BranchKind::VoltageSource)
            ]
        );
        let bench = Bench::new::<BehavioralVoltageSource>(&parameters("2 * i(sense)"));
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "n2", "n3", "device:current", "device:sense"]
        );
    }

    #[test]
    fn a_current_source_drives_its_expression() {
        let mut bench = Bench::new::<BehavioralCurrentSource>(&parameters("1e-3 * v(a) ^ 2"));
        // positive, negative, a.
        let evaluation = bench.evaluate(&[1.0, 0.0, 3.0]);
        assert_close(&evaluation.residual, &[9e-3, -9e-3, 0.0]);
        assert_close(&evaluation.conductance[0], &[0.0, 0.0, 6e-3]);
    }

    #[test]
    fn rejects_invalid_expressions() {
        let error = Bench::try_new::<BehavioralCurrentSource>(&parameters("1 +"))
            .expect_err("an invalid expression");
        assert_eq!(
            error.0,
            "the expression `1 +` is invalid: expected an operand, not the end at column 4"
        );
        let error = Bench::try_new::<BehavioralVoltageSource>(&parameters("i(current)"))
            .expect_err("a reserved port");
        assert!(
            error
                .0
                .ends_with("but `current` names the source's own current")
        );
    }

    proptest! {
        /// The conductances are the derivatives of the residuals, through every kind of
        /// variable.
        #[test]
        fn jacobians_match_finite_differences(
            x in proptest::array::uniform6(-2.0..2.0f64),
        ) {
            let expression = "v(a, b) * tanh(i(s)) + exp(v(positive, negative) / 3) - v(a) ^ 2";
            // positive, negative, a, b, s_positive, s_negative, then s's current.
            let [p, n, a, b, s, current] = x;
            assert_jacobians_match::<BehavioralCurrentSource>(
                &parameters(expression),
                &[p, n, a, b, s, s / 2.0, current],
            );
            // The voltage source's own current comes before the port's.
            assert_jacobians_match::<BehavioralVoltageSource>(
                &parameters(expression),
                &[p, n, a, b, s, s / 2.0, current / 3.0, current],
            );
        }
    }
}
