//! A bench that binds one device to unknowns of its own and evaluates it, for tests.

use jolty_numerics::{Pattern, PatternBuilder};

use crate::{
    BOLTZMANN, Binding, Contributions, Device, Gradient, Input, Model, Native, NativeDevice,
    NativeModel, Node, NoiseSource, ParameterError, Quantity, REFERENCE_TEMPERATURE, Real, Unknown,
    UnknownInfo,
};

/// One device whose terminals each connect to an unknown of their own.
#[derive(Debug)]
pub(crate) struct Bench {
    device: Box<dyn Device>,
    pattern: Pattern,
    unknowns: Vec<UnknownInfo>,
    /// Whether the device is evaluated as a transient analysis steps through time.
    transient: bool,
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
        let typed: D::Parameters = crate::device::read(&parameters)?;
        let terminals = D::interface(&typed).terminals.len();
        Self::made(terminals, temperature, |binding| {
            Ok(Box::new(D::new(typed, binding)?))
        })
    }

    /// A bench for a device of `model` with `parameters`, a JSON object, at `temperature`.
    pub(crate) fn of(
        model: &dyn Model,
        parameters: &str,
        temperature: f64,
    ) -> Result<Self, ParameterError> {
        let specification =
            model.read(&serde_json::from_str(parameters).expect("a JSON object"))?;
        let terminals = specification.interface().terminals.len();
        Self::made(terminals, temperature, |binding| {
            specification.instantiate(binding)
        })
    }

    /// A bench for the device `make` makes through a binding of `terminals`, each on an
    /// unknown of its own, at `temperature`.
    fn made(
        terminals: usize,
        temperature: f64,
        make: impl FnOnce(&mut Binding<'_>) -> Result<Box<dyn Device>, ParameterError>,
    ) -> Result<Self, ParameterError> {
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
        let device = make(&mut Binding::new(
            "device",
            &nodes,
            &mut unknowns,
            &mut builder,
            temperature,
        ))?;
        let pattern = builder.finish(unknowns.len());
        Ok(Self {
            device,
            pattern,
            unknowns,
            transient: false,
        })
    }

    /// The bench, evaluating its device as a transient analysis does, while time passes.
    pub(crate) const fn stepping(mut self) -> Self {
        self.transient = true;
        self
    }

    pub(crate) fn unknowns(&self) -> Vec<&str> {
        self.unknowns
            .iter()
            .map(|unknown| unknown.name.as_str())
            .collect()
    }

    pub(crate) fn evaluate(&mut self, x: &[f64]) -> Evaluation {
        let time = self.transient.then_some(0.0);
        self.evaluate_with(&Input::new(x, time, 0.0))
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
        let n = self.unknowns.len();
        let mut contributions = Contributions::new(&self.pattern);
        let mut output = contributions.output(&self.pattern);
        self.device.evaluate(input, &mut output);
        let limited = output.is_limited();
        let Contributions {
            residual,
            charge,
            conductance,
            capacitance,
            ..
        } = contributions;
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

    /// The device's noise sources at `x`.
    pub(crate) fn noise(&self, x: &[f64]) -> Vec<NoiseSource> {
        let mut sources = Vec::new();
        self.device.noise(&Input::new(x, None, 0.0), &mut sources);
        sources
    }
}

/// The source named `name` among `sources`.
///
/// # Panics
///
/// Panics unless exactly one source has that name.
pub(crate) fn source<'a>(sources: &'a [NoiseSource], name: &str) -> &'a NoiseSource {
    let mut named = sources.iter().filter(|source| source.name == name);
    let found = named.next().unwrap_or_else(|| panic!("no {name} noise"));
    assert!(named.next().is_none(), "two {name} noises");
    found
}

/// `4kT` at the reference temperature times `value`: the thermal noise density of a
/// conductance's current, or of a resistance's voltage.
pub(crate) fn thermal(value: f64) -> f64 {
    4.0 * BOLTZMANN * REFERENCE_TEMPERATURE * value
}

/// Checks that a device's conductances and capacitances are the derivatives of its residuals
/// and charges at `x`, by central differences.
pub(crate) fn assert_jacobians_match<D: NativeDevice>(parameters: &str, x: &[f64]) {
    assert_bench_jacobians_match(&mut Bench::new::<D>(parameters), D::NAME, x);
}

/// As [`assert_jacobians_match`], with the device evaluated as a transient analysis
/// steps through time.
pub(crate) fn assert_stepping_jacobians_match<D: NativeDevice>(parameters: &str, x: &[f64]) {
    let mut bench = Bench::new::<D>(parameters).stepping();
    assert_bench_jacobians_match(&mut bench, D::NAME, x);
}

/// As [`assert_jacobians_match`], for a device of any `model`.
pub(crate) fn assert_model_jacobians_match(model: &dyn Model, parameters: &str, x: &[f64]) {
    let mut bench = Bench::of(model, parameters, REFERENCE_TEMPERATURE).expect("valid");
    assert_bench_jacobians_match(&mut bench, model.name(), x);
}

/// Checks that the conductances and capacitances of the device `name` on `bench` are the
/// derivatives of its residuals and charges at `x`, by central differences.
fn assert_bench_jacobians_match(bench: &mut Bench, name: &str, x: &[f64]) {
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
                    [minus.residual[row], exact.residual[row], plus.residual[row]],
                ),
                (
                    "capacitance",
                    &exact.capacitance[row],
                    [minus.charge[row], exact.charge[row], plus.charge[row]],
                ),
            ];
            for (kind, derivatives, [below, at, above]) in pairs {
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
                // Within a step of a point where a model changes regions, its curvature
                // jumps and the central difference errs by the step times it; the
                // derivative then lies between the one-sided differences on each side.
                let sides = [(at - below) / step, (above - at) / step];
                let between = (sides[0].min(sides[1]) - tolerance
                    ..=sides[0].max(sides[1]) + tolerance)
                    .contains(&analytic);
                assert!(
                    (analytic - numeric).abs() <= tolerance || between,
                    "{name} {kind} at ({row}, {column}): analytic {analytic}, numeric {numeric}"
                );
            }
        }
    }
}

/// Evaluates `device`, bound to `pattern`, at `x`, at `time` in a transient analysis or,
/// with `None`, at an operating point: its residuals, charges and conductances.
fn residuals_and_charges<R: Real>(
    device: &mut dyn Device<R>,
    pattern: &Pattern,
    (x, time): (&[f64], Option<f64>),
) -> [Vec<R>; 3] {
    let mut contributions = Contributions::new(pattern);
    let mut input = Input::new(x, time, 1e-12);
    input.limiting = false;
    device.evaluate(&input, &mut contributions.output(pattern));
    [
        contributions.residual,
        contributions.charge,
        contributions.conductance,
    ]
}

/// A device of a model made from parameters, each of its terminals on an unknown of its
/// own, at 30 kelvins above the reference temperature.
struct Rig<'m> {
    model: &'m dyn Model,
    nodes: Vec<Node>,
}

impl Rig<'_> {
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

    /// The residuals, charges and conductances at `x` of the device made with
    /// `parameters`.
    fn plain(&self, parameters: &jolty_ir::Parameters, x: (&[f64], Option<f64>)) -> [Vec<f64>; 3] {
        let (mut unknowns, mut builder) = (Vec::new(), PatternBuilder::new());
        let mut binding = self.binding(&mut unknowns, &mut builder);
        let mut device = self
            .model
            .read(parameters)
            .and_then(|specification| specification.instantiate(&mut binding))
            .expect("valid");
        residuals_and_charges(device.as_mut(), &builder.finish(x.0.len()), x)
    }

    /// The residuals, charges and conductances at `x`, with their gradients with respect
    /// to the number at `parameter`, of the device made with `parameters`.
    fn exact(
        &self,
        parameters: &jolty_ir::Parameters,
        parameter: &str,
        x: (&[f64], Option<f64>),
    ) -> Option<[Vec<Gradient>; 3]> {
        let (mut unknowns, mut builder) = (Vec::new(), PatternBuilder::new());
        let mut binding = self.binding(&mut unknowns, &mut builder);
        let mut device = self
            .model
            .read(parameters)
            .and_then(|specification| specification.instantiate_gradient(parameter, &mut binding))
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
    assert_model_parameter_gradients_match(&Native::<M>::new(), parameters, x, None);
}

/// As [`assert_parameter_gradients_match`], in a transient analysis at `time`.
pub(crate) fn assert_parameter_gradients_match_at<M: NativeModel>(
    parameters: &str,
    x: &[f64],
    time: f64,
) {
    assert_model_parameter_gradients_match(&Native::<M>::new(), parameters, x, Some(time));
}

/// As [`assert_parameter_gradients_match`], for a device of any `model`, at `time` in a
/// transient analysis or, with `None`, at an operating point, its conductances'
/// derivatives checked too.
pub(crate) fn assert_model_parameter_gradients_match(
    model: &dyn Model,
    parameters: &str,
    x: &[f64],
    time: Option<f64>,
) {
    let parameters: serde_json::Value = serde_json::from_str(parameters).expect("JSON");
    let object = |value: &serde_json::Value| -> jolty_ir::Parameters {
        value.as_object().cloned().expect("a JSON object")
    };
    let terminals = model
        .read(&object(&parameters))
        .expect("valid")
        .interface()
        .terminals
        .len();
    let rig = Rig {
        model,
        nodes: (0..terminals)
            .map(|index| Some(Unknown::new(index)))
            .collect(),
    };
    let plain = rig.plain(&object(&parameters), (x, time));
    let mut paths = Vec::new();
    numbers(&parameters, &mut Vec::new(), &mut paths);
    for path in paths {
        let name = path.join(".");
        let mut changed = parameters.clone();
        let value = at(&mut changed, &path).as_f64().expect("a number");
        // A number the model reads as a choice, such as a level, is no real parameter.
        let Some(exact) = rig.exact(&object(&parameters), &name, (x, time)) else {
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
            [0, 1, 2].map(|kind| -> Vec<(f64, f64)> {
                above[kind]
                    .iter()
                    .zip(&below[kind])
                    .map(|(above, below)| ((above - below) / (2.0 * step), above.abs()))
                    .collect()
            })
        };
        let (wide, narrow) = (central(step), central(step / 2.0));
        for (kind, name_of) in ["residual", "charge", "conductance"]
            .into_iter()
            .enumerate()
        {
            for (row, exact) in exact[kind].iter().enumerate() {
                let (wide, narrow, plain) = (&wide[kind], &narrow[kind], &plain[kind]);
                let numeric = (4.0 * narrow[row].0 - wide[row].0) / 3.0;
                let rounding = 64.0 * f64::EPSILON * wide[row].1 / step;
                let tolerance = 1e-6 * exact.derivative.abs().max(numeric.abs()) + rounding;
                assert!(
                    (exact.derivative - numeric).abs() <= tolerance + 1e-30,
                    "{} d{name_of}[{row}]/d{name} at {x:?}: exact {}, numeric {numeric}",
                    model.name(),
                    exact.derivative
                );
                assert!(
                    (exact.value - plain[row]).abs() <= 1e-12 * plain[row].abs(),
                    "{name_of}[{row}]: {} with its gradient, {} without",
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
