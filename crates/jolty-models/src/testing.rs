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
    (x, time): (&[f64], f64),
) -> (Vec<R>, Vec<R>) {
    let mut buffers = [(); 3].map(|()| vec![R::default(); x.len()]);
    let [residual, scale, charge] = &mut buffers;
    let mut matrices = [(); 2].map(|()| vec![R::default(); pattern.len()]);
    let [conductance, capacitance] = &mut matrices;
    let mut output = Output::new(pattern, residual, scale, charge, conductance, capacitance);
    let mut input = Input::new(x, time, 1e-12);
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
    fn plain(&self, parameters: &jolty_ir::Parameters, x: (&[f64], f64)) -> (Vec<f64>, Vec<f64>) {
        let (mut unknowns, mut builder) = (Vec::new(), PatternBuilder::new());
        let mut binding = self.binding(&mut unknowns, &mut builder);
        let mut device = self
            .model
            .instantiate(parameters, &mut binding)
            .expect("valid");
        residuals_and_charges(device.as_mut(), &builder.finish(x.0.len()), x)
    }

    /// The residuals and charges at `x`, with their gradients, of the device made with
    /// `parameters`, one number of which is seeded.
    fn exact(
        &self,
        parameters: &jolty_ir::Parameters,
        x: (&[f64], f64),
    ) -> Option<(Vec<Gradient>, Vec<Gradient>)> {
        let (mut unknowns, mut builder) = (Vec::new(), PatternBuilder::new());
        let mut binding = self.binding(&mut unknowns, &mut builder);
        let mut device = self
            .model
            .instantiate_gradient(parameters, &mut binding)
            .expect("a model of any real")
            .ok()?;
        Some(residuals_and_charges(
            device.as_mut(),
            &builder.finish(x.0.len()),
            x,
        ))
    }
}

/// Every number within `value`, by the keys and indices that reach it.
fn numbers(value: &serde_json::Value, path: &mut Vec<String>, found: &mut Vec<Vec<String>>) {
    match value {
        serde_json::Value::Number(_) => found.push(path.clone()),
        serde_json::Value::Object(object) => {
            for (key, inner) in object {
                path.push(key.clone());
                numbers(inner, path, found);
                path.pop();
            }
        }
        serde_json::Value::Array(items) => {
            for (index, inner) in items.iter().enumerate() {
                path.push(index.to_string());
                numbers(inner, path, found);
                path.pop();
            }
        }
        _ => {}
    }
}

/// The value at `path` within `value`.
fn at<'a>(value: &'a mut serde_json::Value, path: &[String]) -> &'a mut serde_json::Value {
    path.iter().fold(value, |value, key| match value {
        serde_json::Value::Array(items) => &mut items[key.parse::<usize>().expect("an index")],
        _ => &mut value[key.as_str()],
    })
}

/// Checks that the exact derivatives of a device's residuals and charges at `x` with
/// respect to each number of `parameters`, a JSON object, match central differences of
/// the device made again with the number a little above and below, and that their values
/// are the device's own.
pub(crate) fn assert_parameter_gradients_match<M: NativeModel>(parameters: &str, x: &[f64]) {
    assert_parameter_gradients_match_at::<M>(parameters, x, 0.0);
}

/// As [`assert_parameter_gradients_match`], at `time`.
pub(crate) fn assert_parameter_gradients_match_at<M: NativeModel>(
    parameters: &str,
    x: &[f64],
    time: f64,
) {
    let parameters: serde_json::Value = serde_json::from_str(parameters).expect("JSON");
    let object = |value: &serde_json::Value| -> jolty_ir::Parameters {
        value.as_object().cloned().expect("a JSON object")
    };
    let model = Native::<M>::new();
    let terminals = model
        .interface(&object(&parameters))
        .expect("valid")
        .terminals
        .len();
    let rig = Rig {
        model,
        nodes: (0..terminals)
            .map(|index| Some(Unknown::new(index)))
            .collect(),
    };
    let (plain, plain_charge) = rig.plain(&object(&parameters), (x, time));
    let mut paths = Vec::new();
    numbers(&parameters, &mut Vec::new(), &mut paths);
    for path in paths {
        let name = path.join(".");
        let mut changed = parameters.clone();
        let value = at(&mut changed, &path).as_f64().expect("a number");
        *at(&mut changed, &path) = serde_json::json!({"value": value, "derivative": 1.0});
        // A number the model reads as a choice, such as a level, is no real parameter.
        let Some((exact, exact_charge)) = rig.exact(&object(&changed), (x, time)) else {
            assert!(value.fract() == 0.0, "{name} is a real parameter");
            continue;
        };
        // Central differences at a step and at half of it, extrapolated to a zero step:
        // the step large enough that a model's own iterations, such as a diode's matching
        // of its breakdown, settle alike above and below.
        let step = if value == 0.0 {
            1e-6
        } else {
            1e-4 * value.abs()
        };
        let mut central = |step: f64| {
            *at(&mut changed, &path) = serde_json::json!(value + step);
            let above = rig.plain(&object(&changed), (x, time));
            *at(&mut changed, &path) = serde_json::json!(value - step);
            let below = rig.plain(&object(&changed), (x, time));
            let difference = |above: &[f64], below: &[f64]| -> Vec<(f64, f64)> {
                above
                    .iter()
                    .zip(below)
                    .map(|(above, below)| ((above - below) / (2.0 * step), above.abs()))
                    .collect()
            };
            (
                difference(&above.0, &below.0),
                difference(&above.1, &below.1),
            )
        };
        let (wide, narrow) = (central(step), central(step / 2.0));
        let rows = [
            ("residual", exact, plain.clone(), wide.0, narrow.0),
            (
                "charge",
                exact_charge,
                plain_charge.clone(),
                wide.1,
                narrow.1,
            ),
        ];
        for (kind, exact, plain, wide, narrow) in rows {
            for (row, exact) in exact.iter().enumerate() {
                let numeric = (4.0 * narrow[row].0 - wide[row].0) / 3.0;
                let rounding = 64.0 * f64::EPSILON * wide[row].1 / step;
                let tolerance = 1e-6 * exact.derivative.abs().max(numeric.abs()) + rounding;
                assert!(
                    (exact.derivative - numeric).abs() <= tolerance + 1e-30,
                    "{} d{kind}[{row}]/d{name} at {x:?}: exact {}, numeric {numeric}",
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
