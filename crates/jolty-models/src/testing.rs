//! A bench that binds one device to unknowns of its own and evaluates it, for tests.

use jolty_numerics::{Pattern, PatternBuilder};

use crate::{
    Binding, Device, Input, Model, Native, NativeDevice, Node, Output, ParameterError, Quantity,
    REFERENCE_TEMPERATURE, Unknown, UnknownInfo,
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
        let model = Native::<D>::new();
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

/// Checks that `actual` and `expected` agree to within a relative 1e-12.
pub(crate) fn assert_close(actual: &[f64], expected: &[f64]) {
    let close = actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, e)| (a - e).abs() <= 1e-12 * a.abs().max(e.abs()));
    assert!(close, "{actual:?} differs from {expected:?}");
}
