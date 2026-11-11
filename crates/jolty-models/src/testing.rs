//! A bench that binds one device to unknowns of its own and evaluates it, for tests.

use jolty_numerics::{Pattern, PatternBuilder};

use crate::{
    Binding, Device, Gradient, Input, Model, Native, NativeDevice, NativeModel, Node, Output,
    ParameterError, Plain, Quantity, REFERENCE_TEMPERATURE, Real, Unknown, UnknownInfo,
};

/// One device whose terminals each connect to an unknown of their own.
#[derive(Debug)]
pub(crate) struct Bench {
    device: Box<dyn Device>,
    pattern: Pattern,
    unknowns: Vec<UnknownInfo>,
}

/// What one evaluation of a bench's device produced, with dense matrices.
pub(crate) struct Evaluation {
    pub(crate) residual: Vec<f64>,
    pub(crate) charge: Vec<f64>,
    pub(crate) conductance: Vec<Vec<f64>>,
    pub(crate) capacitance: Vec<Vec<f64>>,
    pub(crate) limited: bool,
}

impl Bench {
    /// A bench for a device of `D` with `parameters`, a JSON object, each of whose
    /// terminals connects to an unknown of its own.
    pub(crate) fn new<D: NativeDevice>(parameters: &str) -> Self {
        Self::at::<D>(parameters, REFERENCE_TEMPERATURE)
    }

    pub(crate) fn try_new<D: NativeDevice>(parameters: &str) -> Result<Self, ParameterError> {
        Self::try_at::<D>(parameters, REFERENCE_TEMPERATURE)
    }

    /// A bench for a device of `D` with `parameters` at `temperature`, in kelvins.
    pub(crate) fn at<D: NativeDevice>(parameters: &str, temperature: f64) -> Self {
        Self::try_at::<D>(parameters, temperature).expect("valid parameters")
    }

    pub(crate) fn try_at<D: NativeDevice>(
        parameters: &str,
        temperature: f64,
    ) -> Result<Self, ParameterError> {
        let parameters = serde_json::from_str(parameters).expect("a JSON object");
        let model = Plain::<D>::new();
        let terminals = model.interface(&parameters)?.terminals.len();
        let nodes: Vec<Node> = (0..terminals)
            .map(|index| Some(Unknown::new(index)))
            .collect();
        let mut unknowns: Vec<UnknownInfo> = (0..terminals)
            .map(|index| UnknownInfo {
                name: format!("n{index}"),
                quantity: Quantity::Voltage,
            })
            .collect();
        let mut builder = PatternBuilder::new();
        let device = model.instantiate(
            &parameters,
            &mut Binding::new("device", &nodes, &mut unknowns, &mut builder, temperature),
        )?;
        let pattern = builder.finish(unknowns.len());
        Ok(Self {
            device,
            pattern,
            unknowns,
        })
    }

    pub(crate) fn unknowns(&self) -> Vec<&str> {
        self.unknowns
            .iter()
            .map(|unknown| unknown.name.as_str())
            .collect()
    }

    pub(crate) fn evaluate(&mut self, x: &[f64]) -> Evaluation {
        self.evaluate_with(&Input::new(x, 0.0, 0.0))
    }

    /// Evaluates at `x` until no voltage is limited any more, as Newton's iterations would
    /// once converged, and returns the last evaluation.
    ///
    /// # Panics
    ///
    /// Panics if limiting has not settled after a hundred evaluations.
    pub(crate) fn settle(&mut self, x: &[f64]) -> Evaluation {
        for _ in 0..100 {
            let evaluation = self.evaluate(x);
            if !evaluation.limited {
                return evaluation;
            }
        }
        panic!("limiting has not settled at {x:?}");
    }

    pub(crate) fn evaluate_with(&mut self, input: &Input<'_>) -> Evaluation {
        let (n, stored) = (self.unknowns.len(), self.pattern.len());
        let (mut residual, mut scale, mut charge) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut conductance, mut capacitance) = (vec![0.0; stored], vec![0.0; stored]);
        let mut output = Output::new(
            &self.pattern,
            &mut residual,
            &mut scale,
            &mut charge,
            &mut conductance,
            &mut capacitance,
        );
        self.device.evaluate(input, &mut output);
        let limited = output.is_limited();
        let dense = |values: &[f64]| {
            let mut matrix = vec![vec![0.0; n]; n];
            for ((row, column), value) in self.pattern.positions().zip(values) {
                matrix[row][column] = *value;
            }
            matrix
        };
        Evaluation {
            conductance: dense(&conductance),
            capacitance: dense(&capacitance),
            residual,
            charge,
            limited,
        }
    }

    pub(crate) fn device(&mut self) -> &mut dyn Device {
        self.device.as_mut()
    }
}

/// Checks that a device's conductances and capacitances are the derivatives of its residuals
/// and charges at `x`, by central differences.
pub(crate) fn assert_jacobians_match<D: NativeDevice>(parameters: &str, x: &[f64]) {
    let mut bench = Bench::new::<D>(parameters);
    assert_eq!(bench.unknowns.len(), x.len(), "one value per unknown");
    let exact = bench.settle(x);
    for column in 0..x.len() {
        let step = 1e-6 * x[column].abs().max(1.0);
        let mut shifted = x.to_vec();
        shifted[column] = x[column] + step;
        let plus = bench.evaluate(&shifted);
        shifted[column] = x[column] - step;
        let minus = bench.evaluate(&shifted);
        bench.evaluate(x);
        for row in 0..x.len() {
            let pairs = [
                (
                    "conductance",
                    &exact.conductance[row],
                    plus.residual[row],
                    minus.residual[row],
                ),
                (
                    "capacitance",
                    &exact.capacitance[row],
                    plus.charge[row],
                    minus.charge[row],
                ),
            ];
            for (name, derivatives, above, below) in pairs {
                let analytic = derivatives[column];
                let numeric = (above - below) / (2.0 * step);
                // Differencing loses the precision of the values differenced, and of the
                // voltages they are computed from: a relative epsilon of each, weighed by
                // its derivative, divided by the step.
                let resolution: f64 = derivatives
                    .iter()
                    .zip(x)
                    .map(|(derivative, value)| (derivative * value).abs())
                    .sum();
                let magnitude = above.abs().max(below.abs()) + resolution;
                let rounding = 16.0 * f64::EPSILON * magnitude / step;
                let tolerance = 1e-5 * analytic.abs().max(numeric.abs()) + rounding + 1e-20;
                assert!(
                    (analytic - numeric).abs() <= tolerance,
                    "{} {name} at ({row}, {column}): analytic {analytic}, numeric {numeric}",
                    D::NAME
                );
            }
        }
    }
}

/// Evaluates `device`, bound to `pattern`, at `x`: its residuals and charges.
fn residuals_and_charges<R: Real>(
    device: &mut dyn Device<R>,
    pattern: &Pattern,
    x: &[f64],
) -> (Vec<R>, Vec<R>) {
    let mut buffers = [(); 3].map(|()| vec![R::default(); x.len()]);
    let [residual, scale, charge] = &mut buffers;
    let mut matrices = [(); 2].map(|()| vec![R::default(); pattern.len()]);
    let [conductance, capacitance] = &mut matrices;
    let mut output = Output::new(pattern, residual, scale, charge, conductance, capacitance);
    let mut input = Input::new(x, 0.0, 1e-12);
    input.limiting = false;
    device.evaluate(&input, &mut output);
    let [residual, _, charge] = buffers;
    (residual, charge)
}

/// A device of `M` made from parameters, each of its terminals on an unknown of its own,
/// at 30 kelvins above the reference temperature.
struct Rig<M: NativeModel> {
    model: Native<M>,
    nodes: Vec<Node>,
}

impl<M: NativeModel> Rig<M> {
    fn binding<'a>(
        &'a self,
        unknowns: &'a mut Vec<UnknownInfo>,
        builder: &'a mut PatternBuilder,
    ) -> Binding<'a> {
        unknowns.extend(self.nodes.iter().map(|_| UnknownInfo {
            name: String::new(),
            quantity: Quantity::Voltage,
        }));
        Binding::new(
            "device",
            &self.nodes,
            unknowns,
            builder,
            REFERENCE_TEMPERATURE + 30.0,
        )
    }

    /// The residuals and charges at `x` of the device made with `parameters`.
    fn plain(&self, parameters: &jolty_ir::Parameters, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (mut unknowns, mut builder) = (Vec::new(), PatternBuilder::new());
        let mut binding = self.binding(&mut unknowns, &mut builder);
        let mut device = self
            .model
            .instantiate(parameters, &mut binding)
            .expect("valid");
        residuals_and_charges(device.as_mut(), &builder.finish(x.len()), x)
    }

    /// The residuals and charges at `x`, with their gradients, of the device made with
    /// `parameters`, one number of which is seeded.
    fn exact(
        &self,
        parameters: &jolty_ir::Parameters,
        x: &[f64],
    ) -> (Vec<Gradient>, Vec<Gradient>) {
        let (mut unknowns, mut builder) = (Vec::new(), PatternBuilder::new());
        let mut binding = self.binding(&mut unknowns, &mut builder);
        let mut device = self
            .model
            .instantiate_gradient(parameters, &mut binding)
            .expect("a model of any real")
            .expect("valid parameters");
        residuals_and_charges(device.as_mut(), &builder.finish(x.len()), x)
    }
}

/// Checks that the exact derivatives of a device's residuals and charges at `x` with
/// respect to each number of `parameters`, a JSON object, match central differences of
/// the device made again with the number a little above and below, and that their values
/// are the device's own.
pub(crate) fn assert_parameter_gradients_match<M: NativeModel>(parameters: &str, x: &[f64]) {
    let parameters: jolty_ir::Parameters = serde_json::from_str(parameters).expect("a JSON object");
    let model = Native::<M>::new();
    let terminals = model.interface(&parameters).expect("valid").terminals.len();
    let rig = Rig {
        model,
        nodes: (0..terminals)
            .map(|index| Some(Unknown::new(index)))
            .collect(),
    };
    let (plain, plain_charge) = rig.plain(&parameters, x);
    for (name, value) in &parameters {
        let Some(value) = value.as_f64() else {
            continue;
        };
        let mut changed = parameters.clone();
        changed.insert(
            name.clone(),
            serde_json::json!({"value": value, "derivative": 1.0}),
        );
        let (exact, exact_charge) = rig.exact(&changed, x);
        let step = 1e-6 * value.abs().max(1e-12);
        changed.insert(name.clone(), serde_json::json!(value + step));
        let (above, above_charge) = rig.plain(&changed, x);
        changed.insert(name.clone(), serde_json::json!(value - step));
        let (below, below_charge) = rig.plain(&changed, x);
        let rows = [
            ("residual", exact, plain.clone(), above, below),
            (
                "charge",
                exact_charge,
                plain_charge.clone(),
                above_charge,
                below_charge,
            ),
        ];
        for (kind, exact, plain, above, below) in rows {
            for (row, exact) in exact.iter().enumerate() {
                let numeric = (above[row] - below[row]) / (2.0 * step);
                let rounding = 16.0 * f64::EPSILON * above[row].abs().max(below[row].abs()) / step;
                let tolerance = 1e-5 * exact.derivative.abs().max(numeric.abs()) + rounding;
                assert!(
                    (exact.derivative - numeric).abs() <= tolerance + 1e-30,
                    "{} d{kind}[{row}]/d{name}: exact {}, numeric {numeric}",
                    <M::Device<f64> as NativeDevice>::NAME,
                    exact.derivative
                );
                assert!(
                    (exact.value - plain[row]).abs() <= 1e-12 * plain[row].abs(),
                    "{kind}[{row}]: {} with its gradient, {} without",
                    exact.value,
                    plain[row]
                );
            }
        }
    }
}

/// Checks that `actual` and `expected` agree to within a relative 1e-12.
pub(crate) fn assert_close(actual: &[f64], expected: &[f64]) {
    let close = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() <= 1e-12 * a.abs().max(e.abs()));
    assert!(close, "{actual:?} differs from {expected:?}");
}
