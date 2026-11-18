use jolty_expressions::{Expression, Variable, Workspace};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Choice, Interface, Kernel, Layout, Lift, Local, ParameterError, Real, Stamp,
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
    /// The expression as written.
    text: String,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            text,
            expression,
            terminals,
            reads,
            ports,
        })
    }
}

impl Serialize for Behavior {
    /// The behavior as its expression is written.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
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
            thermal: Vec::new(),
        }
    }
}

/// The parameters of a [`BehavioralVoltageSource`], a [`BehavioralCurrentSource`] or a
/// [`BehavioralCapacitor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
pub struct BehavioralParameters {
    /// What the device's output follows, a capacitor's its charge: an expression of the
    /// voltages of the terminals it reads, of the currents through sensing ports of its
    /// own, and of time.
    #[schemars(with = "String")]
    pub expression: Behavior,
}

/// Where a behavioral source's contributions go, among its local unknowns: the rows its
/// expression enters, each with the factor it enters with, what each variable reads, its
/// sensing ports, the entries of each variable's derivative in each row, with their
/// factors, and a voltage source's own branch. Sources whose expressions read the same
/// voltages and currents share a shape, whatever their expressions compute.
#[derive(Debug, Clone, PartialEq)]
pub struct BehavioralShape {
    rows: Vec<(usize, f64)>,
    /// Whether the expression's value is a charge in its rows, and its derivatives
    /// capacitances, as a capacitor's, rather than a current or a voltage.
    stored: bool,
    reads: Vec<Read>,
    ports: Vec<CurrentBranch>,
    entries: Vec<Vec<(usize, f64)>>,
    branch: Option<CurrentBranch>,
}

/// What one variable of an expression reads: the voltage of a local unknown over another
/// or over the reference, a port's current, or time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    Voltage(usize, Option<usize>),
    Current(usize),
    Time,
}

/// What a behavioral source remembers: its expression's workspaces, one for its iterations
/// and one, apart, in which its choices are found, and its variables' values and gradient.
/// None before it is sized for an expression.
#[derive(Debug, Clone, Default)]
pub struct BehavioralMemory(Option<Room>);

/// A behavioral source's room to evaluate in.
#[derive(Debug, Clone)]
struct Room {
    workspace: Workspace,
    choosing: Workspace,
    values: Vec<f64>,
    gradient: Vec<f64>,
    tangent: Vec<f64>,
    gradient_tangent: Vec<f64>,
    gradient_derivatives: Vec<[f64; crate::Seeds::MOST]>,
}

impl BehavioralShape {
    /// The shape of a source of `behavior`, whose expression enters each of `rows` with
    /// its factor, as a charge where it is `stored`, after `branch`, a voltage source's
    /// own, claimed in `layout`.
    fn new(
        behavior: &Behavior,
        layout: &mut Layout,
        (rows, stored, branch): (Vec<(usize, f64)>, bool, Option<CurrentBranch>),
    ) -> Self {
        let ports: Vec<CurrentBranch> = behavior
            .ports
            .iter()
            .map(|(label, ends)| CurrentBranch::new(layout, label.clone(), *ends))
            .collect();
        let reads: Vec<Read> = behavior
            .reads
            .iter()
            .map(|reading| match *reading {
                Reading::Voltage(positive, negative) => Read::Voltage(positive, negative),
                Reading::Current(port) => {
                    Read::Current(ports.get(port).map_or(usize::MAX, CurrentBranch::unknown))
                }
                Reading::Time => Read::Time,
            })
            .collect();
        let entries = reads
            .iter()
            .map(|read| {
                let columns: Vec<(usize, f64)> = match *read {
                    Read::Voltage(positive, negative) => std::iter::once((positive, 1.0))
                        .chain(negative.map(|negative| (negative, -1.0)))
                        .collect(),
                    Read::Current(current) => vec![(current, 1.0)],
                    Read::Time => Vec::new(),
                };
                let mut entries = Vec::with_capacity(rows.len() * columns.len());
                for &(row, factor) in &rows {
                    for &(column, sign) in &columns {
                        entries.push((layout.entry(row, column), factor * sign));
                    }
                }
                entries
            })
            .collect();
        Self {
            rows,
            stored,
            reads,
            ports,
            entries,
            branch,
        }
    }

    /// Adds the expression's value to its rows, with its derivatives, and the ports'
    /// equations, with `expression` at `at`, from `memory`, which it updates.
    fn add<R: Real>(
        &self,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let Some(room) = &mut memory.0 else { return };
        self.read(at, &mut room.values);
        if !at.limiting {
            room.workspace.forget();
        }
        let value = expression.evaluate(&room.values, &mut room.workspace, &mut room.gradient);
        if room.workspace.limited() {
            stamp.mark_limited();
        }
        let seeded = R::SEEDS > 0 && at.seeds_unknowns();
        let value_derivatives = if seeded {
            let mut value_derivatives = [0.0; crate::Seeds::MOST];
            for slot in 0..R::SEEDS.min(crate::Seeds::MOST) {
                self.read_tangent(at, slot, &mut room.tangent);
                if room.tangent.iter().all(|&rate| rate == 0.0) {
                    if let Some(target) = value_derivatives.get_mut(slot) {
                        *target = 0.0;
                    }
                    for rates in &mut room.gradient_derivatives {
                        if let Some(target) = rates.get_mut(slot) {
                            *target = 0.0;
                        }
                    }
                } else {
                    let moved = expression.along(
                        &room.tangent,
                        &mut room.workspace,
                        &mut room.gradient_tangent,
                    );
                    if let Some(target) = value_derivatives.get_mut(slot) {
                        *target = moved;
                    }
                    for (rates, &moved_gradient) in room
                        .gradient_derivatives
                        .iter_mut()
                        .zip(&room.gradient_tangent)
                    {
                        if let Some(target) = rates.get_mut(slot) {
                            *target = moved_gradient;
                        }
                    }
                }
            }
            Some(value_derivatives)
        } else {
            None
        };
        self.stamp_contribution(
            stamp,
            value,
            value_derivatives.as_ref(),
            &room.gradient,
            &room.gradient_derivatives,
        );
        for port in &self.ports {
            // A short circuit: no voltage across it.
            port.add(at, stamp, port.voltage(at));
        }
    }

    /// Stamps `value` and `gradient` into rows and entries, attaching `value_derivatives`
    /// and `gradient_derivatives` when unknown-seeded.
    fn stamp_contribution<R: Real>(
        &self,
        stamp: &mut Stamp<'_, R>,
        value: f64,
        value_derivatives: Option<&[f64; crate::Seeds::MOST]>,
        gradient: &[f64],
        gradient_derivatives: &[[f64; crate::Seeds::MOST]],
    ) {
        let make = |base: f64, factor: f64, rates: Option<&[f64; crate::Seeds::MOST]>| match rates {
            Some(rates) => {
                let scaled = rates.map(|rate| factor * rate);
                R::moving(factor * base, &scaled)
            }
            None => R::constant(factor * base),
        };
        for &(row, factor) in &self.rows {
            let contribution = make(value, factor, value_derivatives);
            if self.stored {
                stamp.add_charge(row, contribution);
            } else {
                stamp.add_residual(row, contribution);
            }
        }
        for (index, (entries, &derivative)) in self.entries.iter().zip(gradient).enumerate() {
            let rates = value_derivatives
                .is_some()
                .then(|| gradient_derivatives.get(index))
                .flatten();
            for &(entry, factor) in entries {
                let contribution = make(derivative, factor, rates);
                if self.stored {
                    stamp.add_capacitance(entry, contribution);
                } else {
                    stamp.add_conductance(entry, contribution);
                }
            }
        }
    }

    /// Reads the expression's variables' derivatives along `slot` at `at` into `tangent`.
    fn read_tangent(&self, at: &Local<'_>, slot: usize, tangent: &mut [f64]) {
        for (rate, read) in tangent.iter_mut().zip(&self.reads) {
            *rate = match *read {
                Read::Voltage(positive, negative) => {
                    at.unknown_derivative(positive, slot)
                        - negative.map_or(0.0, |node| at.unknown_derivative(node, slot))
                }
                Read::Current(current) => at.unknown_derivative(current, slot),
                Read::Time => 0.0,
            };
        }
    }

    /// Reads the expression's variables at `at` into `values`.
    fn read(&self, at: &Local<'_>, values: &mut [f64]) {
        for (value, read) in values.iter_mut().zip(&self.reads) {
            *value = match *read {
                Read::Voltage(positive, negative) => {
                    at.value(positive) - negative.map_or(0.0, |negative| at.value(negative))
                }
                Read::Current(current) => at.value(current),
                Read::Time => at.time,
            };
        }
    }

    /// Appends the choices `expression` turns on at `at` to `found`, evaluating it afresh in
    /// the workspace of `memory` apart, its tables' inputs unlimited.
    fn choices(
        &self,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        let Some(room) = &mut memory.0 else { return };
        self.read(at, &mut room.values);
        room.choosing.forget();
        expression.evaluate(&room.values, &mut room.choosing, &mut room.gradient);
        expression.choices(&room.choosing, found);
    }

    /// What a source of this shape with `expression` remembers before its first
    /// evaluation.
    fn memory(&self, expression: &Expression) -> BehavioralMemory {
        let count = self.reads.len();
        BehavioralMemory(Some(Room {
            workspace: Workspace::new(expression),
            choosing: Workspace::new(expression),
            values: vec![0.0; count],
            gradient: vec![0.0; count],
            tangent: vec![0.0; count],
            gradient_tangent: vec![0.0; count],
            gradient_derivatives: vec![[0.0; crate::Seeds::MOST]; count],
        }))
    }
}

/// The local indices of a behavioral source's output terminals, `positive` and `negative`.
const OUTPUT: [usize; 2] = [0, 1];

/// A current source whose current follows an expression of the voltages and currents it
/// reads: see [`Behavior`].
///
/// It drives the expression's value from `positive` through the source to `negative`.
#[derive(Debug, Clone, Copy)]
pub struct BehavioralCurrentSource;

impl Kernel for BehavioralCurrentSource {
    const NAME: &'static str = "behavioral_current_source";
    type Parameters<R: Real> = BehavioralParameters;
    type Shape = BehavioralShape;
    type Prepared<R: Real> = Expression;
    type Memory = BehavioralMemory;

    fn interface<R: Real>(parameters: &BehavioralParameters) -> Interface {
        parameters.expression.interface(BranchKind::CurrentSource)
    }

    fn prepare<R: Real>(
        parameters: &BehavioralParameters,
        _: R,
    ) -> Result<Expression, ParameterError> {
        Ok(parameters.expression.expression.clone())
    }

    fn layout<R: Real>(
        parameters: &BehavioralParameters,
        _: &Expression,
        layout: &mut Layout,
    ) -> Result<BehavioralShape, ParameterError> {
        let [positive, negative] = OUTPUT;
        let rows = vec![(positive, 1.0), (negative, -1.0)];
        Ok(BehavioralShape::new(
            &parameters.expression,
            layout,
            (rows, false, None),
        ))
    }

    fn memory<R: Real>(shape: &BehavioralShape, expression: &Expression) -> BehavioralMemory {
        shape.memory(expression)
    }

    fn evaluate<R: Real>(
        shape: &BehavioralShape,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        shape.add(expression, memory, at, stamp);
    }

    fn choices<R: Real>(
        shape: &BehavioralShape,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        shape.choices(expression, memory, at, found);
    }
}

/// A voltage source whose voltage follows an expression of the voltages and currents it
/// reads: see [`Behavior`].
///
/// It holds `v_positive − v_negative` at the expression's value. Its current, flowing from
/// `positive` through the source to `negative`, is an unknown labelled `current`.
#[derive(Debug, Clone, Copy)]
pub struct BehavioralVoltageSource;

impl Kernel for BehavioralVoltageSource {
    const NAME: &'static str = "behavioral_voltage_source";
    type Parameters<R: Real> = BehavioralParameters;
    type Shape = BehavioralShape;
    type Prepared<R: Real> = Expression;
    type Memory = BehavioralMemory;

    fn interface<R: Real>(parameters: &BehavioralParameters) -> Interface {
        parameters.expression.interface(BranchKind::VoltageSource)
    }

    fn prepare<R: Real>(
        parameters: &BehavioralParameters,
        _: R,
    ) -> Result<Expression, ParameterError> {
        Ok(parameters.expression.expression.clone())
    }

    fn layout<R: Real>(
        parameters: &BehavioralParameters,
        _: &Expression,
        layout: &mut Layout,
    ) -> Result<BehavioralShape, ParameterError> {
        let branch = CurrentBranch::new(layout, "current", OUTPUT);
        let rows = vec![(branch.unknown(), -1.0)];
        Ok(BehavioralShape::new(
            &parameters.expression,
            layout,
            (rows, false, Some(branch)),
        ))
    }

    fn memory<R: Real>(shape: &BehavioralShape, expression: &Expression) -> BehavioralMemory {
        shape.memory(expression)
    }

    fn evaluate<R: Real>(
        shape: &BehavioralShape,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // The branch's voltage, less the expression's value, which the shape adds itself.
        if let Some(branch) = &shape.branch {
            branch.add(at, stamp, branch.voltage(at));
        }
        shape.add(expression, memory, at, stamp);
    }

    fn choices<R: Real>(
        shape: &BehavioralShape,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        shape.choices(expression, memory, at, found);
    }
}

/// A capacitor whose charge follows an expression of the voltages and currents it reads:
/// see [`Behavior`].
///
/// It holds the expression's value as its charge, on `positive` against `negative`: its
/// current, from `positive` through the capacitor to `negative`, is that charge's rate
/// of change, and its capacitances are the expression's exact derivatives. Its charge is
/// a function of what it reads, so it conserves charge. It is open at DC.
#[derive(Debug, Clone, Copy)]
pub struct BehavioralCapacitor;

impl Kernel for BehavioralCapacitor {
    const NAME: &'static str = "behavioral_capacitor";
    type Parameters<R: Real> = BehavioralParameters;
    type Shape = BehavioralShape;
    type Prepared<R: Real> = Expression;
    type Memory = BehavioralMemory;

    fn interface<R: Real>(parameters: &BehavioralParameters) -> Interface {
        parameters.expression.interface(BranchKind::Capacitive)
    }

    fn prepare<R: Real>(
        parameters: &BehavioralParameters,
        _: R,
    ) -> Result<Expression, ParameterError> {
        Ok(parameters.expression.expression.clone())
    }

    fn layout<R: Real>(
        parameters: &BehavioralParameters,
        _: &Expression,
        layout: &mut Layout,
    ) -> Result<BehavioralShape, ParameterError> {
        let [positive, negative] = OUTPUT;
        let rows = vec![(positive, 1.0), (negative, -1.0)];
        Ok(BehavioralShape::new(
            &parameters.expression,
            layout,
            (rows, true, None),
        ))
    }

    fn memory<R: Real>(shape: &BehavioralShape, expression: &Expression) -> BehavioralMemory {
        shape.memory(expression)
    }

    fn evaluate<R: Real>(
        shape: &BehavioralShape,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        shape.add(expression, memory, at, stamp);
    }

    fn choices<R: Real>(
        shape: &BehavioralShape,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        found: &mut Vec<Choice>,
    ) {
        shape.choices(expression, memory, at, found);
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
    fn a_capacitor_holds_its_expression_as_its_charge() {
        let mut bench = Bench::new::<BehavioralCapacitor>(&parameters(
            "1e-12 * v(positive, negative) + 1e-13 * v(positive, negative) ^ 2",
        ));
        let evaluation = bench.evaluate(&[2.0, 0.5]);
        let (charge, capacitance) = (1e-12 * 1.5 + 1e-13 * 2.25, 1e-12 + 2e-13 * 1.5);
        assert_close(&evaluation.residual, &[0.0, 0.0]);
        assert_close(&evaluation.charge, &[charge, -charge]);
        assert_close(&evaluation.capacitance[0], &[capacitance, -capacitance]);
        assert_close(&evaluation.conductance[0], &[0.0, 0.0]);
        let interface = BehavioralCapacitor::interface::<f64>(
            &serde_json::from_str(&parameters("1e-12 * v(a)")).expect("valid"),
        );
        assert_eq!(
            interface.branches[0],
            Branch::new(0, 1, BranchKind::Capacitive)
        );
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
            // A capacitor's capacitances are its charge's derivatives.
            assert_jacobians_match::<BehavioralCapacitor>(
                &parameters(expression),
                &[p, n, a, b, s, s / 2.0, current],
            );
        }
    }
}
