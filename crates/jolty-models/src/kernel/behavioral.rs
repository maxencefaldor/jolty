use jolty_expressions::{Expression, Variable, Workspace};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Choice, Interface, Kernel, Layout, Lift, Local, ParameterError, Real, Stamp,
};

/// Parsed behavioral [expression](jolty_expressions) depending on sensed node voltages,
/// dedicated current-sensing port currents, and simulation time.
///
/// The device exposes `positive` and `negative` terminals, followed by each referenced
/// voltage terminal and each sensing port's `p_positive` and `p_negative` terminals. Each
/// port acts as a short circuit with a branch-current unknown labelled `p` flowing from
/// `p_positive` to `p_negative`.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "String")]
pub struct Behavior {
    /// Original expression text as supplied in parameters.
    text: String,
    expression: Expression,
    terminals: Vec<String>,
    /// Input binding (voltage, port current, or time) for each expression variable.
    reads: Vec<Reading>,
    /// Sensing port names paired with their `[positive, negative]` terminal indices.
    ports: Vec<(String, [usize; 2])>,
}

/// Source quantity sampled by one expression variable: a terminal voltage difference, a
/// sensing port current, or simulation time.
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
    /// Serializes the behavior as the original expression string.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.text)
    }
}

impl Behavior {
    /// Builds the device interface using branch type `kind` on the output pair and
    /// voltage-source short circuits on each current-sensing port.
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

/// Parameters for [`BehavioralVoltageSource`], [`BehavioralCurrentSource`], and
/// [`BehavioralCapacitor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
pub struct BehavioralParameters {
    /// Expression defining output current, voltage, or stored charge as a function of sensed
    /// terminal voltages, port currents, and simulation time.
    #[schemars(with = "String")]
    pub expression: Behavior,
}

/// Layout shape for a behavioral device mapping expression outputs, variable bindings,
/// sensing ports, derivative matrix entries, and an optional output branch.
///
/// Behavioral sources referencing identical terminal voltages and sensing ports share one
/// shape regardless of the expression body.
#[derive(Debug, Clone, PartialEq)]
pub struct BehavioralShape {
    rows: Vec<(usize, f64)>,
    /// Flag indicating that expression values stamp into charges and derivatives into
    /// capacitances (as for a capacitor) rather than residuals and conductances.
    stored: bool,
    reads: Vec<Read>,
    ports: Vec<CurrentBranch>,
    entries: Vec<Vec<(usize, f64)>>,
    branch: Option<CurrentBranch>,
}

/// Local-index binding for one expression variable: a local voltage difference, a port
/// current unknown, or simulation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    Voltage(usize, Option<usize>),
    Current(usize),
    Time,
}

/// Preallocated evaluation state for a behavioral source, holding separate iteration and
/// choice-detection workspaces plus variable value and gradient buffers.
#[derive(Debug, Clone, Default)]
pub struct BehavioralMemory(Option<Room>);

/// Evaluation scratch buffers allocated once per behavioral source instance.
#[derive(Debug, Clone)]
struct Room {
    workspace: Workspace,
    choosing: Workspace,
    values: Vec<f64>,
    gradient: Vec<f64>,
    tangent: Vec<f64>,
    gradient_tangent: Vec<f64>,
    gradient_derivatives: Vec<[f64; crate::Seeds::MAX]>,
}

impl BehavioralShape {
    /// Claims sensing ports and derivative matrix entries in `layout` for `behavior`, stamping
    /// expression outputs into `rows` as charges when `stored` is true and attaching `branch`
    /// for voltage sources.
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

    /// Returns whether the expression explicitly references `time`.
    fn reads_time(&self) -> bool {
        self.reads.contains(&Read::Time)
    }

    /// Evaluates `expression` at `at`, updating `memory` and stamping row values, exact
    /// derivatives, and sensing-port equations into `stamp`.
    fn add<R: Real>(
        &self,
        expression: &Expression,
        memory: &mut BehavioralMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // Stamps the voltage-source branch voltage minus the expression value below.
        if let Some(branch) = &self.branch {
            branch.add(at, stamp, branch.voltage(at));
        }
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
            let mut value_derivatives = [0.0; crate::Seeds::MAX];
            for slot in 0..R::SEEDS.min(crate::Seeds::MAX) {
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
        value_derivatives: Option<&[f64; crate::Seeds::MAX]>,
        gradient: &[f64],
        gradient_derivatives: &[[f64; crate::Seeds::MAX]],
    ) {
        let make = |base: f64, factor: f64, rates: Option<&[f64; crate::Seeds::MAX]>| match rates {
            Some(rates) => {
                let scaled = rates.map(|rate| factor * rate);
                R::seeded(factor * base, &scaled)
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

    /// Allocates initial evaluation workspaces and variable buffers for `expression`.
    fn memory(&self, expression: &Expression) -> BehavioralMemory {
        let count = self.reads.len();
        BehavioralMemory(Some(Room {
            workspace: Workspace::new(expression),
            choosing: Workspace::new(expression),
            values: vec![0.0; count],
            gradient: vec![0.0; count],
            tangent: vec![0.0; count],
            gradient_tangent: vec![0.0; count],
            gradient_derivatives: vec![[0.0; crate::Seeds::MAX]; count],
        }))
    }
}

/// Local indices of a behavioral source's output terminals (`positive` and `negative`).
const OUTPUT: [usize; 2] = [0, 1];

/// Output stamping mode of a behavioral kernel across its output terminals.
#[derive(Debug, Clone, Copy)]
enum OutputKind {
    Current,
    Voltage,
    Charge,
}

impl OutputKind {
    /// Returns the topological [`BranchKind`] connecting the output terminals.
    const fn branch(self) -> BranchKind {
        match self {
            Self::Current => BranchKind::CurrentSource,
            Self::Voltage => BranchKind::VoltageSource,
            Self::Charge => BranchKind::Capacitive,
        }
    }

    /// Lays out the shape of a behavioral kernel of this output kind.
    fn shape(self, behavior: &Behavior, layout: &mut Layout) -> BehavioralShape {
        let [positive, negative] = OUTPUT;
        let rows = || vec![(positive, 1.0), (negative, -1.0)];
        let output = match self {
            Self::Current => (rows(), false, None),
            Self::Voltage => {
                let branch = CurrentBranch::new(layout, "current", OUTPUT);
                (vec![(branch.unknown(), -1.0)], false, Some(branch))
            }
            Self::Charge => (rows(), true, None),
        };
        BehavioralShape::new(behavior, layout, output)
    }
}

/// Expands `impl Kernel` for a behavioral device kernel.
macro_rules! behavioral_kernel {
    (
        kernel: $kernel:ty,
        name: $name:literal,
        kind: $kind:expr,
    ) => {
        impl Kernel for $kernel {
            const NAME: &'static str = $name;
            type Parameters<R: Real> = BehavioralParameters;
            type Shape = BehavioralShape;
            type Prepared<R: Real> = Expression;
            type Memory = BehavioralMemory;

            fn interface<R: Real>(parameters: &BehavioralParameters) -> Interface {
                parameters.expression.interface($kind.branch())
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
                Ok($kind.shape(&parameters.expression, layout))
            }

            fn memory<R: Real>(
                shape: &BehavioralShape,
                expression: &Expression,
            ) -> BehavioralMemory {
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

            fn varies_in_time<R: Real>(shape: &BehavioralShape, _: &Expression) -> bool {
                shape.reads_time()
            }
        }
    };
}

/// Behavioral current source whose current follows an expression of sensed voltages, port
/// currents, and time; see [`Behavior`].
///
/// Drives the evaluated expression current from `positive` through the source to `negative`.
#[derive(Debug, Clone, Copy)]
pub struct BehavioralCurrentSource;

behavioral_kernel! {
    kernel: BehavioralCurrentSource,
    name: "behavioral_current_source",
    kind: OutputKind::Current,
}

/// Behavioral voltage source whose terminal voltage follows an expression of sensed voltages,
/// port currents, and time; see [`Behavior`].
///
/// Constrains `v_positive − v_negative` to the evaluated expression value. The branch current
/// flowing from `positive` through the source to `negative` is an unknown labelled `current`.
#[derive(Debug, Clone, Copy)]
pub struct BehavioralVoltageSource;

behavioral_kernel! {
    kernel: BehavioralVoltageSource,
    name: "behavioral_voltage_source",
    kind: OutputKind::Voltage,
}

/// Behavioral charge-based capacitor whose stored charge follows an expression of sensed
/// voltages, port currents, and time; see [`Behavior`].
///
/// Stores the evaluated expression as charge on `positive` relative to `negative`. The
/// terminal current from `positive` to `negative` equals the time derivative of that charge,
/// with exact capacitances from expression derivatives. Charge is conserved because it is a
/// state function of the sensed variables, and the device is open at DC.
#[derive(Debug, Clone, Copy)]
pub struct BehavioralCapacitor;

behavioral_kernel! {
    kernel: BehavioralCapacitor,
    name: "behavioral_capacitor",
    kind: OutputKind::Charge,
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
        /// Verifies that analytical conductances and capacitances match finite differences
        /// across all expression variable types.
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
            // The voltage source's branch current precedes the sensing port current.
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
