//! A bench that binds one device to unknowns of its own and evaluates it, for tests: this
//! crate's, and, through the `testing` feature, those of crates whose kernels implement
//! its own, such as the kernels compiled from Verilog-A.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc,
    reason = "a bench fails the test that uses it"
)]
#![cfg_attr(
    not(test),
    expect(dead_code, reason = "most of the bench serves this crate's own tests")
)]

use std::collections::BTreeMap;

use jolty_numerics::{Pattern, PatternBuilder};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

// The cases and the laws serve this crate's own tests, not the bench it shares.
#[cfg(test)]
mod cases;
#[cfg(test)]
mod laws;

#[cfg(test)]
pub(crate) use laws::assert_symmetric_at;

use crate::{
    BOLTZMANN, Binding, Contributions, Differentiable, Gradient, Groups, Input, Kernel, Model,
    Nature, Node, NoiseSource, ParameterError, REFERENCE_TEMPERATURE, REGION, Regions, Runs, Seeds,
    Twin, TwinNumber, Unknown, UnknownIndex,
};

/// One device whose terminals each connect to an unknown of their own, its instance in a
/// group of its own.
#[derive(Debug)]
pub struct Bench {
    groups: Groups,
    runs: Runs,
    pattern: Pattern,
    unknowns: Vec<Unknown>,
    /// The names of the variables the device computes.
    variables: &'static [&'static str],
    /// The regions of operation the device's equations distinguish.
    regions: Regions,
    /// Whether the device is evaluated as a transient analysis steps through time.
    transient: bool,
}

/// What one evaluation of a bench's device produced, with dense matrices.
pub struct Evaluation {
    /// The residual of each unknown's equation.
    pub residual: Vec<f64>,
    /// The largest magnitude of a term of each residual.
    pub scale: Vec<f64>,
    /// The charge of each unknown's equation.
    pub charge: Vec<f64>,
    /// The conductances, by row and column.
    pub conductance: Vec<Vec<f64>>,
    /// The capacitances, by row and column.
    pub capacitance: Vec<Vec<f64>>,
    /// Whether the device limited a step.
    pub limited: bool,
}

/// The groups of one device of `model` with `parameters` at `temperature`, each terminal
/// on an unknown of its own, with the pattern and the unknowns they claim.
fn grouped(
    model: &dyn Model,
    parameters: &jolty_ir::Parameters,
    temperature: f64,
) -> Result<(Groups, Runs, Pattern, Vec<Unknown>), ParameterError> {
    let specification = model.read(parameters)?;
    let interface = specification.interface();
    let terminals = interface.terminals.len();
    let nodes: Vec<Node> = (0..terminals)
        .map(|index| Some(UnknownIndex::new(index)))
        .collect();
    // Each terminal's unknown is of the nature the engine gives the net it joins.
    let mut unknowns: Vec<Unknown> = (0..terminals)
        .map(|index| Unknown {
            name: format!("n{index}"),
            nature: if interface.thermal.contains(&index) {
                Nature::Temperature
            } else {
                Nature::Voltage
            },
        })
        .collect();
    let mut builder = PatternBuilder::new();
    let (mut groups, mut runs) = (Groups::new(), Runs::default());
    {
        let mut binding = Binding::new("device", &nodes, &mut unknowns, &mut builder);
        groups.add(
            &mut runs,
            (specification.kernel(), temperature),
            &mut binding,
        )?;
    }
    let pattern = builder.finish(unknowns.len());
    groups
        .finish(&mut runs, &pattern)
        .map_err(|_| ParameterError("too large".to_owned()))?;
    Ok((groups, runs, pattern, unknowns))
}

impl Bench {
    /// A bench for an instance of the kernel `K` with `parameters`, a JSON object, each of
    /// whose terminals connects to an unknown of its own.
    pub(crate) fn new<K: Kernel>(parameters: &str) -> Self
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        Self::at::<K>(parameters, REFERENCE_TEMPERATURE)
    }

    pub(crate) fn try_new<K: Kernel>(parameters: &str) -> Result<Self, ParameterError>
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        Self::try_at::<K>(parameters, REFERENCE_TEMPERATURE)
    }

    /// A bench for an instance of `K` with `parameters` at `temperature`, in kelvins.
    #[must_use]
    pub fn at<K: Kernel>(parameters: &str, temperature: f64) -> Self
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        Self::try_at::<K>(parameters, temperature).expect("valid parameters")
    }

    /// A bench as [`Bench::at`] makes, or why the parameters are out of range.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    pub fn try_at<K: Kernel>(parameters: &str, temperature: f64) -> Result<Self, ParameterError>
    where
        K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
    {
        Self::from_model(&Differentiable::<K>::new(), parameters, temperature)
    }

    /// A bench for a device of `model` with `parameters`, a JSON object, at `temperature`.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    pub fn from_model(
        model: &dyn Model,
        parameters: &str,
        temperature: f64,
    ) -> Result<Self, ParameterError> {
        let parameters = serde_json::from_str(parameters).expect("a JSON object");
        let (groups, runs, pattern, unknowns) = grouped(model, &parameters, temperature)?;
        let specification = model.read(&parameters)?;
        let instance = specification.kernel();
        let (variables, regions) = (instance.variable_names(), instance.regions());
        Ok(Self {
            groups,
            runs,
            pattern,
            unknowns,
            variables,
            regions,
            transient: false,
        })
    }

    /// The bench, evaluating its device as a transient analysis does, while time passes.
    pub(crate) const fn stepping(mut self) -> Self {
        self.transient = true;
        self
    }

    /// The names of the unknowns: the terminals', then the device's own.
    #[must_use]
    pub fn unknowns(&self) -> Vec<&str> {
        self.unknowns
            .iter()
            .map(|unknown| unknown.name.as_str())
            .collect()
    }

    /// The nature of each local unknown, a thermal terminal's a temperature.
    #[must_use]
    pub fn natures(&self) -> Vec<Nature> {
        self.unknowns.iter().map(|unknown| unknown.nature).collect()
    }

    /// The row and the column of each pivot the device's layout
    /// [claims](crate::Layout::pivot).
    pub(crate) fn pivoted(&self) -> &[(usize, usize)] {
        self.groups.pivoted()
    }

    /// Evaluates the device at `x`, from what it remembers.
    pub fn evaluate(&mut self, x: &[f64]) -> Evaluation {
        let time = self.transient.then_some(0.0);
        self.evaluate_with(&Input::new(x, time, 0.0))
    }

    /// Evaluates at `x` until no voltage is limited any more, as Newton's iterations would
    /// once converged, and returns the last evaluation.
    ///
    /// # Panics
    ///
    /// Panics if limiting has not settled after a hundred evaluations.
    pub fn settle(&mut self, x: &[f64]) -> Evaluation {
        for _ in 0..100 {
            let evaluation = self.evaluate(x);
            if !evaluation.limited {
                return evaluation;
            }
        }
        panic!("limiting has not settled at {x:?}");
    }

    /// Evaluates the device at `input`, from what it remembers.
    pub fn evaluate_with(&mut self, input: &Input<'_>) -> Evaluation {
        let n = self.unknowns.len();
        let mut contributions = self.contributions();
        let limited = self.evaluate_into(input, &mut contributions);
        let Contributions {
            residual,
            scale,
            charge,
            conductance,
            capacitance,
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
            scale,
            charge,
            limited,
        }
    }

    /// Room for the device's contributions, as [`Bench::evaluate_into`] writes them.
    #[must_use]
    pub fn contributions(&self) -> Contributions {
        Contributions::new(&self.pattern)
    }

    /// Evaluates the device at `input`, from what it remembers, into `contributions`,
    /// whatever they held, and returns whether it limited a step: the evaluation alone,
    /// which allocates nothing of the bench's.
    pub fn evaluate_into(&mut self, input: &Input<'_>, contributions: &mut Contributions) -> bool {
        self.runs.evaluate(&self.groups, input, contributions, None)
    }

    /// The names of the variables the device computes.
    pub(crate) const fn variable_names(&self) -> &'static [&'static str] {
        self.variables
    }

    /// The variables the device computed at its last evaluation.
    #[must_use]
    pub fn variables(&self) -> Vec<f64> {
        let mut values = vec![0.0; self.variables.len()];
        self.runs.variables(&self.groups, 0, &mut values);
        values
    }

    /// The value of the device's variable `name` at its last evaluation.
    ///
    /// # Panics
    ///
    /// Panics unless the device has that variable.
    pub(crate) fn variable(&self, name: &str) -> f64 {
        let index = self.variables.iter().position(|&variable| variable == name);
        self.variables()[index.unwrap_or_else(|| panic!("no variable {name}"))]
    }

    /// The device's region of operation at its last evaluation, by name, and the value of
    /// each of its margins, in its model's order.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a region's number is an index, which its kernel wrote from a `u8`"
    )]
    pub(crate) fn region(&self) -> (&'static str, Vec<f64>) {
        let number = self.variable(REGION) as usize;
        let margins = self
            .regions
            .margins
            .iter()
            .map(|margin| self.variable(margin));
        (self.regions.names[number], margins.collect())
    }

    /// The device's excitation, as an independent source's: the derivative of each
    /// residual with respect to its level.
    pub(crate) fn excitation(&self) -> Vec<f64> {
        let mut derivative = vec![0.0; self.unknowns.len()];
        self.runs.excite(&self.groups, 0, &mut derivative);
        derivative
    }

    /// Starts the device, a logic driver, on a `(time, level, transition)` ramp, and
    /// returns when it settles.
    pub(crate) fn drive(&mut self, ramp: (f64, crate::LogicLevel, f64)) -> Option<f64> {
        self.runs.drive(&self.groups, 0, ramp)
    }

    /// The device's noise sources at `x`.
    #[must_use]
    pub fn noise(&self, x: &[f64]) -> Vec<NoiseSource> {
        let mut sources = Vec::new();
        self.runs
            .noise(&self.groups, 0, &Input::new(x, None, 0.0), &mut sources);
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

/// Checks that an instance's conductances and capacitances are the derivatives of its
/// residuals and charges at `x`, by central differences.
pub(crate) fn assert_jacobians_match<K: Kernel>(parameters: &str, x: &[f64])
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    assert_bench_jacobians_match(&mut Bench::new::<K>(parameters), K::NAME, x);
}

/// As [`assert_jacobians_match`], with the instance evaluated as a transient analysis
/// steps through time.
pub(crate) fn assert_stepping_jacobians_match<K: Kernel>(parameters: &str, x: &[f64])
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    let mut bench = Bench::new::<K>(parameters).stepping();
    assert_bench_jacobians_match(&mut bench, K::NAME, x);
}

/// As [`assert_jacobians_match`], for a device of any `model`.
pub(crate) fn assert_model_jacobians_match(model: &dyn Model, parameters: &str, x: &[f64]) {
    let mut bench = Bench::from_model(model, parameters, REFERENCE_TEMPERATURE).expect("valid");
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

/// A contribution of a twin, owned: its value, and its derivative along each seed.
pub type Moved = (f64, Vec<f64>);

/// A device of a model made from parameters, each of its terminals on an unknown of its
/// own, at 30 kelvins above the reference temperature.
pub struct Rig<'m> {
    /// The model of the devices.
    pub model: &'m dyn Model,
}

impl Rig<'_> {
    /// The temperature the rig's devices are at.
    pub const TEMPERATURE: f64 = REFERENCE_TEMPERATURE + 30.0;

    /// The residuals, charges and conductances at `x` of the device made with
    /// `parameters`, its steps not limited.
    #[must_use]
    pub fn plain(
        &self,
        parameters: &jolty_ir::Parameters,
        (x, time): (&[f64], Option<f64>),
    ) -> [Vec<f64>; 3] {
        let (groups, mut runs, pattern, _) =
            grouped(self.model, parameters, Self::TEMPERATURE).expect("valid");
        let mut contributions = Contributions::new(&pattern);
        let mut input = Input::new(x, time, 1e-12);
        input.limiting = false;
        runs.evaluate(&groups, &input, &mut contributions, None);
        [
            contributions.residual,
            contributions.charge,
            contributions.conductance,
        ]
    }

    /// The residuals, charges and conductances at `x`, with their gradients with respect
    /// to the number at `parameter`, of the twin of the instance made with `parameters`,
    /// in a group of its own.
    #[must_use]
    pub fn exact(
        &self,
        parameters: &jolty_ir::Parameters,
        parameter: &str,
        (x, time): (&[f64], Option<f64>),
    ) -> Option<[Vec<Gradient>; 3]> {
        let found = self.along(parameters, &[parameter], (x, time))?;
        Some(found.map(|numbers| {
            numbers
                .into_iter()
                .map(|(value, derivatives)| {
                    Gradient::new(value, [derivatives.first().copied().unwrap_or(0.0)])
                })
                .collect()
        }))
    }

    /// As [`Rig::exact`], along each of `paths` at once: the residuals, charges and
    /// conductances at `x`, each with its derivative along each path's unit seed, of one
    /// twin differentiating along all of them.
    #[must_use]
    pub fn along(
        &self,
        parameters: &jolty_ir::Parameters,
        paths: &[&str],
        (x, time): (&[f64], Option<f64>),
    ) -> Option<[Vec<Moved>; 3]> {
        let (groups, runs, pattern, _) =
            grouped(self.model, parameters, Self::TEMPERATURE).expect("valid");
        let specification = self.model.read(parameters).expect("valid");
        let instance = (&specification.kernel(), Self::TEMPERATURE);
        let mut twin = runs.twin(&groups, 0, instance, &Seeds::units(paths))?;
        let mut input = Input::new(x, time, 1e-12);
        input.limiting = false;
        twin.evaluate(&input);
        let zero = (0.0, vec![0.0; paths.len()]);
        let mut found = [x.len(), x.len(), pattern.len()].map(|length| vec![zero.clone(); length]);
        let add = |(sum, sums): &mut Moved, (value, derivatives): TwinNumber<'_>| {
            *sum += value;
            for (sum, derivative) in sums.iter_mut().zip(derivatives) {
                *sum += derivative;
            }
        };
        for (unknown, residual) in twin.residual() {
            add(&mut found[0][unknown], residual);
        }
        for (unknown, charge) in twin.charge() {
            add(&mut found[1][unknown], charge);
        }
        for (slot, conductance, _) in twin.entries() {
            add(&mut found[2][slot], conductance);
        }
        Some(found)
    }

    /// The row and column of each conductance [`Rig::exact`] finds for the instance made
    /// with `parameters`, in its order.
    #[must_use]
    pub fn positions(&self, parameters: &jolty_ir::Parameters) -> Vec<(usize, usize)> {
        let (_, _, pattern, _) = grouped(self.model, parameters, Self::TEMPERATURE).expect("valid");
        pattern.positions().collect()
    }
}

/// The twin of an instance of `K` with `parameters`, a JSON object, at `temperature`,
/// differentiated along `seeds`, in a group of its own, each of its terminals on an
/// unknown of its own; none when its kernel reads no real number at a seed's path.
#[must_use]
pub fn twin<K: Kernel>(parameters: &str, temperature: f64, seeds: &Seeds) -> Option<Twin>
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    let model = Differentiable::<K>::new();
    let parameters = serde_json::from_str(parameters).ok()?;
    let (groups, runs, _, _) = grouped(&model, &parameters, temperature).ok()?;
    let specification = model.read(&parameters).ok()?;
    runs.twin(&groups, 0, (&specification.kernel(), temperature), seeds)
}

/// The excitation of an independent source of `model` with `parameters`, each terminal on
/// an unknown of its own, by unknown, and the derivative of its residuals at `x` with
/// respect to the level of the source of that model held at its level, as a DC sweep holds
/// it: its waveform's `dc`.
pub(crate) fn excitation_and_derivative(
    model: &dyn Model,
    parameters: &str,
    x: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let parameters = serde_json::from_str(parameters).expect("a JSON object");
    let (groups, runs, _, _) = grouped(model, &parameters, REFERENCE_TEMPERATURE).expect("valid");
    let level = runs.level(&groups, 0).expect("a source");
    let mut excitation = vec![0.0; x.len()];
    runs.excite(&groups, 0, &mut excitation);
    let held = crate::held_source(level);
    let (groups, runs, _, _) = grouped(model, &held, REFERENCE_TEMPERATURE).expect("valid");
    let specification = model.read(&held).expect("valid");
    let instance = (&specification.kernel(), REFERENCE_TEMPERATURE);
    let mut twin = runs
        .twin(&groups, 0, instance, &Seeds::units(&["waveform.dc"]))
        .expect("a real level");
    twin.evaluate(&Input::new(x, None, 0.0));
    let mut derivative = vec![0.0; x.len()];
    for (unknown, (_, derivatives)) in twin.residual() {
        derivative[unknown] += derivatives.first().copied().unwrap_or(0.0);
    }
    (excitation, derivative)
}

/// Every number within `value`, by the keys and indices that reach it.
pub fn numbers(value: &serde_json::Value, path: &mut Vec<String>, found: &mut Vec<Vec<String>>) {
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
/// its instance prepared with the number a little above and below, and that their values
/// are the instance's own.
pub(crate) fn assert_parameter_gradients_match<K: Kernel>(parameters: &str, x: &[f64])
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    assert_model_parameter_gradients_match(&Differentiable::<K>::new(), parameters, x, None);
}

/// As [`assert_parameter_gradients_match`], in a transient analysis at `time`.
pub(crate) fn assert_parameter_gradients_match_at<K: Kernel>(parameters: &str, x: &[f64], time: f64)
where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    assert_model_parameter_gradients_match(&Differentiable::<K>::new(), parameters, x, Some(time));
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
    let rig = Rig { model };
    let plain = rig.plain(&object(&parameters), (x, time));
    let mut paths = Vec::new();
    numbers(&parameters, &mut Vec::new(), &mut paths);
    let mut real = Vec::new();
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
                let [derivative] = exact.derivatives;
                let (wide, narrow, plain) = (&wide[kind], &narrow[kind], &plain[kind]);
                let numeric = (4.0 * narrow[row].0 - wide[row].0) / 3.0;
                let rounding = 64.0 * f64::EPSILON * wide[row].1 / step;
                let tolerance = 1e-6 * derivative.abs().max(numeric.abs()) + rounding;
                assert!(
                    (derivative - numeric).abs() <= tolerance + 1e-30,
                    "{} d{name_of}[{row}]/d{name} at {x:?}: exact {derivative}, numeric {numeric}",
                    model.name(),
                );
                assert!(
                    (exact.value - plain[row]).abs() <= 1e-12 * plain[row].abs(),
                    "{name_of}[{row}]: {} with its gradient, {} without",
                    exact.value,
                    plain[row]
                );
            }
        }
        real.push((name, exact));
    }
    assert_chunks_match_seeds_alone(&rig, &object(&parameters), &real, (x, time));
}

/// Checks that a twin differentiating along several of a device's real parameters at once
/// finds, along each, bit for bit what a twin along that parameter alone does: in chunks
/// as wide as a twin goes, and of three, which a twin pads to its width of four.
fn assert_chunks_match_seeds_alone(
    rig: &Rig<'_>,
    parameters: &jolty_ir::Parameters,
    real: &[(String, [Vec<Gradient>; 3])],
    (x, time): (&[f64], Option<f64>),
) {
    for width in [Seeds::MOST, 3] {
        for chunk in real.chunks(width) {
            let paths: Vec<&str> = chunk.iter().map(|(name, _)| name.as_str()).collect();
            let along = rig
                .along(parameters, &paths, (x, time))
                .expect("real parameters");
            for (seed, (name, alone)) in chunk.iter().enumerate() {
                for (kind, name_of) in ["residual", "charge", "conductance"].iter().enumerate() {
                    for (row, (together, alone)) in along[kind].iter().zip(&alone[kind]).enumerate()
                    {
                        let [derivative] = alone.derivatives;
                        assert!(
                            together.0.to_bits() == alone.value.to_bits()
                                && together.1[seed].to_bits() == derivative.to_bits(),
                            "{} {name_of}[{row}] along {name}, seed {seed} of {}: {together:?} \
                             in the chunk, {alone:?} alone",
                            rig.model.name(),
                            paths.len(),
                        );
                    }
                }
            }
        }
    }
}

/// Checks that the derivatives of a device's residuals, conductances and capacitances at
/// `x` with respect to each of its unknowns, as twins seeding its unknowns eight at a time
/// find them, are its conductances, and match central differences of its conductances and
/// capacitances, its steps unlimited: its equations' second derivatives, exact.
///
/// # Panics
///
/// Panics if any of those fails.
pub fn assert_unknown_derivatives_match(model: &dyn Model, parameters: &str, x: &[f64]) {
    let mut bench = Bench::from_model(model, parameters, REFERENCE_TEMPERATURE).expect("valid");
    assert_eq!(bench.unknowns.len(), x.len(), "one value per unknown");
    let plain = |bench: &mut Bench, x: &[f64]| {
        let mut input = Input::new(x, None, 1e-12);
        input.limiting = false;
        bench.evaluate_with(&input)
    };
    let exact = plain(&mut bench, x);
    let read: jolty_ir::Parameters = serde_json::from_str(parameters).expect("a JSON object");
    let specification = model.read(&read).expect("valid");
    let instance = (&specification.kernel(), REFERENCE_TEMPERATURE);
    let positions: Vec<(usize, usize)> = bench.pattern.positions().collect();
    let twins = bench.runs.unknown_twins(&bench.groups, 0, instance);
    assert!(!twins.is_empty(), "a twin seeding the unknowns");
    for mut twin in twins {
        let mut input = Input::new(x, None, 1e-12);
        input.limiting = false;
        twin.evaluate(&input);
        let seeded: Vec<(usize, usize)> = twin
            .seeded()
            .iter()
            .enumerate()
            .filter(|&(_, &unknown)| unknown != jolty_numerics::UNMAPPED)
            .map(|(seed, &unknown)| (seed, unknown as usize))
            .collect();
        for (row, (value, derivatives)) in twin.residual() {
            // Sums of terms that cancel leave a rounding of the row's largest.
            let largest = exact.conductance[row]
                .iter()
                .fold(0.0_f64, |largest, conductance| {
                    largest.max(conductance.abs())
                });
            for &(seed, column) in &seeded {
                let (derivative, conductance) = (derivatives[seed], exact.conductance[row][column]);
                assert!(
                    (derivative - conductance).abs() <= 1e-12 * largest + 1e-20,
                    "{} residual[{row}] along {column}: {derivative}, its conductance \
                     {conductance}, at {value}",
                    model.name(),
                );
            }
        }
        for (row, (value, derivatives)) in twin.charge() {
            let largest = exact.capacitance[row]
                .iter()
                .fold(0.0_f64, |largest, capacitance| {
                    largest.max(capacitance.abs())
                });
            for &(seed, column) in &seeded {
                let (derivative, capacitance) = (derivatives[seed], exact.capacitance[row][column]);
                assert!(
                    (derivative - capacitance).abs() <= 1e-12 * largest + 1e-20,
                    "{} charge[{row}] along {column}: {derivative}, its capacitance \
                     {capacitance}, at {value}",
                    model.name(),
                );
            }
        }
        for &(seed, column) in &seeded {
            assert_entries_match_along(
                (model.name(), &mut bench, &twin),
                (&positions, x),
                (seed, column),
            );
        }
    }
}

/// Checks that the derivatives of `twin`'s conductances and capacitances along `seed`,
/// which moves the unknown `column`, match extrapolated central differences of `bench` at
/// `x`.
fn assert_entries_match_along(
    (model, bench, twin): (&str, &mut Bench, &Twin),
    (positions, x): (&[(usize, usize)], &[f64]),
    (seed, column): (usize, usize),
) {
    let step = 1e-5 * x[column].abs().max(1.0);
    let mut evaluated_at = |offset: f64| {
        let mut shifted = x.to_vec();
        shifted[column] = x[column] + offset;
        let mut input = Input::new(&shifted, None, 1e-12);
        input.limiting = false;
        bench.evaluate_with(&input)
    };
    let (wide_above, wide_below) = (evaluated_at(step), evaluated_at(-step));
    let half = step / 2.0;
    let (above, below) = (evaluated_at(half), evaluated_at(-half));
    // A sum of terms that cancel leaves a rounding of the row's largest.
    let mut largest = vec![[0.0_f64; 2]; x.len()];
    for (slot, (_, conductance), (_, capacitance)) in twin.entries() {
        let row = &mut largest[positions[slot].0];
        row[0] = row[0].max(conductance[seed].abs());
        row[1] = row[1].max(capacitance[seed].abs());
    }
    for (slot, conductance, capacitance) in twin.entries() {
        let (row, entry_column) = positions[slot];
        let pairs = [
            (
                "conductance",
                conductance,
                (&wide_above.conductance, &wide_below.conductance),
                (&above.conductance, &below.conductance),
            ),
            (
                "capacitance",
                capacitance,
                (&wide_above.capacitance, &wide_below.capacitance),
                (&above.capacitance, &below.capacitance),
            ),
        ];
        for ((kind, (at, derivatives), (wide_up, wide_down), (up, down)), largest) in
            pairs.into_iter().zip(largest[row])
        {
            let analytic = derivatives[seed];
            let wide = (wide_up[row][entry_column] - wide_down[row][entry_column]) / (2.0 * step);
            let narrow = (up[row][entry_column] - down[row][entry_column]) / step;
            let numeric = (4.0 * narrow - wide) / 3.0;
            let magnitude = up[row][entry_column]
                .abs()
                .max(down[row][entry_column].abs());
            let rounding = 128.0 * f64::EPSILON * magnitude / half;
            let tolerance =
                1e-6 * analytic.abs().max(numeric.abs()) + rounding + 1e-12 * largest + 1e-20;
            // Within a step of a point where a model changes regions, the second
            // derivative lies between the one-sided differences on each side.
            let sides = [
                (at - down[row][entry_column]) / half,
                (up[row][entry_column] - at) / half,
            ];
            let side_tolerance =
                1e-4 * analytic.abs().max(narrow.abs()) + rounding + 1e-12 * largest + 1e-20;
            let between = (sides[0].min(sides[1]) - side_tolerance
                ..=sides[0].max(sides[1]) + side_tolerance)
                .contains(&analytic);
            assert!(
                (analytic - numeric).abs() <= tolerance || between,
                "{model} d{kind}[{row}][{entry_column}]/dx[{column}] at {x:?}: exact \
                 {analytic}, numeric {numeric}"
            );
        }
    }
}

/// Checks that the region of `bench`'s device differs at `from` and `to`, values of its
/// local unknowns, and that on the segment between them it switches where its margin
/// `margin` changes sign.
///
/// The segment is bisected until its ends are adjacent numbers apart: there the margin's
/// sign at each end is its sign at that end of the segment, and the margin is within
/// 1e-12 V of zero, a bound far above the rounding of a difference of voltages of a few
/// volts, a few times 1e-16 V.
///
/// # Panics
///
/// Panics if any of those fails.
pub(crate) fn assert_switches_where_margin_vanishes(
    bench: &mut Bench,
    (from, to): (&[f64], &[f64]),
    margin: &str,
) {
    let mut at = |t: f64| {
        let x: Vec<f64> = from.iter().zip(to).map(|(a, b)| a + t * (b - a)).collect();
        bench.settle(&x);
        (bench.region().0, bench.variable(margin))
    };
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    let ((first, before), (last, after)) = (at(low), at(high));
    assert_ne!(first, last, "the same region at both ends");
    assert!(before * after < 0.0, "{margin}: {before} and {after}");
    let mut ends = [(first, before), (last, after)];
    loop {
        let middle = f64::midpoint(low, high);
        if middle <= low || middle >= high {
            break;
        }
        let found = at(middle);
        if found.0 == first {
            (low, ends[0]) = (middle, found);
        } else {
            (high, ends[1]) = (middle, found);
        }
    }
    let [(_, below), (_, above)] = ends;
    assert!(
        below.signum() != above.signum() || below == 0.0 || above == 0.0,
        "{margin} keeps its sign across the switch: {below}, {above}"
    );
    assert!(
        before.signum() * below >= 0.0 && after.signum() * above >= 0.0,
        "{margin} changes sign away from the switch: {below}, {above}"
    );
    assert!(
        below.abs() < 1e-12 && above.abs() < 1e-12,
        "{margin} at the switch: {below}, {above}"
    );
}

/// An expected region check: operating point, region name, and optional exact margins.
pub(crate) type RegionPoint<'a> = (&'a [f64], &'a str, Option<&'a [f64]>);

/// A region-boundary crossing check: start point, end point, and vanishing margin name.
pub(crate) type RegionSwitch<'a> = (&'a [f64], &'a [f64], &'a str);

/// Checks that `K` with `parameters` reports each expected region (and, when given, exact
/// margins) at each point in `points`, that its optional `mirror` reports the same at each
/// negated point, and that each `(from, to, margin)` in `switches` crosses the boundary
/// where `margin` vanishes.
pub(crate) fn assert_regions<K: Kernel>(
    parameters: &str,
    mirror: Option<&str>,
    points: &[RegionPoint<'_>],
    switches: &[RegionSwitch<'_>],
) where
    K::Parameters<f64>: DeserializeOwned + Serialize + JsonSchema,
{
    let mut bench = Bench::new::<K>(parameters);
    let mut opposite = mirror.map(Bench::new::<K>);
    for &(x, region, margins) in points {
        bench.settle(x);
        let (found_region, found_margins) = bench.region();
        assert_eq!(found_region, region, "{x:?}");
        if let Some(expected) = margins {
            assert_eq!(found_margins, expected, "{x:?}");
        }
        if let Some(opposite_bench) = &mut opposite {
            let reflected: Vec<f64> = x.iter().map(|value| -value).collect();
            opposite_bench.settle(&reflected);
            let (mirror_region, mirror_margins) = opposite_bench.region();
            assert_eq!(mirror_region, region, "mirror at {x:?}");
            assert_close(&found_margins, &mirror_margins);
        }
    }
    for &(from, to, margin) in switches {
        assert_switches_where_margin_vanishes(&mut bench, (from, to), margin);
    }
}

/// Checks that the linearized 2x2 submatrix at `[first, second]` of `evaluation`, driven by
/// `[drive_first, drive_second]` into those rows, responds at `output` (0 or 1) as the
/// second-order Bessel filter `3 / (s² · delay² + 3 · s · delay + 3)` at each angular
/// frequency in `omegas`.
pub(crate) fn assert_bessel_filter(
    evaluation: &Evaluation,
    [first, second]: [usize; 2],
    [drive_first, drive_second]: [f64; 2],
    output: usize,
    delay: f64,
    omegas: [f64; 3],
) {
    let multiply = |left: (f64, f64), right: (f64, f64)| {
        (
            left.0 * right.0 - left.1 * right.1,
            left.0 * right.1 + left.1 * right.0,
        )
    };
    let divide = |left: (f64, f64), right: (f64, f64)| {
        let norm = right.0 * right.0 + right.1 * right.1;
        (
            (left.0 * right.0 + left.1 * right.1) / norm,
            (left.1 * right.0 - left.0 * right.1) / norm,
        )
    };
    for omega in omegas {
        let entry = |row: usize, column: usize| {
            (
                evaluation.conductance[row][column],
                omega * evaluation.capacitance[row][column],
            )
        };
        let [a, b, c, d] = [
            entry(first, first),
            entry(first, second),
            entry(second, first),
            entry(second, second),
        ];
        let (diag, off) = (multiply(a, d), multiply(b, c));
        let determinant = (diag.0 - off.0, diag.1 - off.1);
        let numerator = if output == 0 {
            (
                d.0 * drive_first - b.0 * drive_second,
                d.1 * drive_first - b.1 * drive_second,
            )
        } else {
            (
                a.0 * drive_second - c.0 * drive_first,
                a.1 * drive_second - c.1 * drive_first,
            )
        };
        let response = divide(numerator, determinant);
        let omega_delay = omega * delay;
        let expected = divide(
            (3.0, 0.0),
            (3.0 - omega_delay * omega_delay, 3.0 * omega_delay),
        );
        assert!(
            (response.0 - expected.0).abs() < 1e-12 && (response.1 - expected.1).abs() < 1e-12,
            "{response:?} vs {expected:?} at {omega}"
        );
    }
}

/// Whether a JSON schema describes a real number (`"number"`) or an integer (`"integer"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberKind {
    Real,
    Integer,
}

/// Classifies a leaf JSON schema node as a real number or an integer, ignoring `null`.
fn leaf_number_kind(node: &serde_json::Value) -> Option<NumberKind> {
    if let Some(kind) = node.get("type") {
        if let Some(name) = kind.as_str() {
            return match name {
                "number" => Some(NumberKind::Real),
                "integer" => Some(NumberKind::Integer),
                _ => None,
            };
        }
        if let Some(names) = kind.as_array() {
            let mut found = None;
            for name in names.iter().filter_map(serde_json::Value::as_str) {
                match name {
                    "null" => {}
                    "number" => {
                        if found.replace(NumberKind::Real).is_some() {
                            return None;
                        }
                    }
                    "integer" => {
                        if found.replace(NumberKind::Integer).is_some() {
                            return None;
                        }
                    }
                    _ => return None,
                }
            }
            if found.is_some() {
                return found;
            }
        }
    }
    if let Some(constant) = node.get("const").and_then(serde_json::Value::as_number) {
        return Some(if constant.is_i64() || constant.is_u64() {
            NumberKind::Integer
        } else {
            NumberKind::Real
        });
    }
    if let Some(values) = node.get("enum").and_then(serde_json::Value::as_array)
        && !values.is_empty()
        && values.iter().all(serde_json::Value::is_number)
    {
        return Some(
            if values.iter().all(|value| value.is_i64() || value.is_u64()) {
                NumberKind::Integer
            } else {
                NumberKind::Real
            },
        );
    }
    for combiner in ["oneOf", "anyOf", "allOf"] {
        if let Some(branches) = node.get(combiner).and_then(serde_json::Value::as_array) {
            let kinds: Vec<NumberKind> = branches.iter().filter_map(leaf_number_kind).collect();
            if let Some(&first) = kinds.first()
                && kinds.iter().all(|&kind| kind == first)
            {
                return Some(first);
            }
        }
    }
    None
}

/// Navigates `schema` along `path` to determine whether the JSON number written there is a
/// real parameter (`"type": "number"`) or an integer/choice (`"type": "integer"`).
fn schema_number_kind(schema: &serde_json::Value, path: &[String]) -> Option<NumberKind> {
    let Some((step, rest)) = path.split_first() else {
        return leaf_number_kind(schema);
    };
    if let Some(property) = schema
        .get("properties")
        .and_then(serde_json::Value::as_object)
        .and_then(|properties| properties.get(step.as_str()))
    {
        return schema_number_kind(property, rest);
    }
    if let Ok(index) = step.parse::<usize>() {
        if let Some(item) = schema
            .get("prefixItems")
            .and_then(serde_json::Value::as_array)
            .and_then(|items| items.get(index))
        {
            return schema_number_kind(item, rest);
        }
        if let Some(items) = schema.get("items") {
            if let Some(tuple) = items.as_array()
                && let Some(item) = tuple.get(index)
            {
                return schema_number_kind(item, rest);
            }
            return schema_number_kind(items, rest);
        }
    }
    for combiner in ["oneOf", "anyOf", "allOf"] {
        if let Some(branches) = schema.get(combiner).and_then(serde_json::Value::as_array) {
            let kinds: Vec<NumberKind> = branches
                .iter()
                .filter_map(|branch| schema_number_kind(branch, path))
                .collect();
            if let Some(&first) = kinds.first()
                && kinds.iter().all(|&kind| kind == first)
            {
                return Some(first);
            }
        }
    }
    None
}

/// Replaces every `null` leaf in `value` whose schema accepts a real number with a distinct
/// positive finite number, returning the number of optional real fields populated.
fn populate_optional_reals(
    schema: &serde_json::Value,
    value: &mut serde_json::Value,
    path: &mut Vec<String>,
    counter: &mut u32,
) -> usize {
    match value {
        serde_json::Value::Null => {
            if schema_number_kind(schema, path) == Some(NumberKind::Real) {
                *counter += 1;
                let distinct = 1.0 + f64::from(*counter) * 0.125;
                *value = serde_json::json!(distinct);
                1
            } else {
                0
            }
        }
        serde_json::Value::Object(object) => {
            let mut populated = 0;
            for (key, inner) in object.iter_mut() {
                path.push(key.clone());
                populated += populate_optional_reals(schema, inner, path, counter);
                path.pop();
            }
            populated
        }
        serde_json::Value::Array(items) => {
            let mut populated = 0;
            for (index, inner) in items.iter_mut().enumerate() {
                path.push(index.to_string());
                populated += populate_optional_reals(schema, inner, path, counter);
                path.pop();
            }
            populated
        }
        _ => 0,
    }
}

/// Checks `lifted` against `expected` and panics with a summary of missing, extra, or
/// bit-mismatched paths when they differ.
fn assert_lift_maps_equal(
    model: &str,
    parameters: &str,
    lifted: &BTreeMap<String, u64>,
    expected: &BTreeMap<String, u64>,
) {
    if lifted == expected {
        return;
    }
    let missing: Vec<&str> = expected
        .keys()
        .filter(|path| !lifted.contains_key(*path))
        .map(String::as_str)
        .collect();
    let extra: Vec<&str> = lifted
        .keys()
        .filter(|path| !expected.contains_key(*path))
        .map(String::as_str)
        .collect();
    let mismatched: Vec<String> = expected
        .iter()
        .filter_map(|(path, expected_bits)| {
            let lifted_bits = lifted.get(path)?;
            (*lifted_bits != *expected_bits).then(|| {
                format!(
                    "{path}: lift={} (0x{lifted_bits:016x}) vs serde={} (0x{expected_bits:016x})",
                    f64::from_bits(*lifted_bits),
                    f64::from_bits(*expected_bits),
                )
            })
        })
        .collect();
    panic!(
        "{model}: Lift vs serde mismatch on {parameters}\n  missing from Lift: {missing:?}\n  extra in Lift: {extra:?}\n  value mismatches: {mismatched:?}",
    );
}

/// Checks that [`crate::Lift`] visits the exact set of real parameter paths written by
/// `serde` on `model` with `parameters`, with no duplicate paths and bit-for-bit identical
/// values.
///
/// Every JSON number in `specification.parameters()` must be classified by `model.schema()`
/// as either a real number (`"type": "number"`) or an integer (`"type": "integer"`). Any
/// floating-point discrete mode selector stored as plain `f64` instead of `R` (such as
/// BSIM's `charge_partition`) must be named explicitly in `selectors`, where it is asserted
/// to exist in `parameters`, have `"type": "number"` in the schema, and NOT be visited by
/// [`crate::Lift`]. Own-temperature keys (`temperature`, `temperature_offset`) handled
/// outside [`crate::Lift`] by [`crate::Surrounded`] are excluded only when actually set on
/// the instance.
///
/// # Panics
///
/// Panics if `parameters` cannot be read, if a JSON number cannot be classified by the
/// schema, if a listed `selectors` entry is missing or invalid, if [`crate::Lift`] emits
/// duplicate paths, or if the lifted `(path, bits)` map differs from the real numbers
/// serialized by `serde`.
pub fn assert_lift_matches_serde(model: &dyn Model, parameters: &str, selectors: &[&str]) {
    let raw: jolty_ir::Parameters = serde_json::from_str(parameters).expect("a JSON object");
    let specification = model
        .read(&raw)
        .unwrap_or_else(|error| panic!("{} failed to read {parameters}: {error}", model.name()));
    let written =
        serde_json::Value::Object(specification.parameters().unwrap_or_else(|error| {
            panic!("{} failed to write {parameters}: {error}", model.name())
        }));
    let schema = model.schema().to_value();
    let instance = specification.kernel();
    let own_keys = instance.own_temperature_keys();

    let mut lifted = BTreeMap::new();
    for (path, value) in instance.lift_entries() {
        let previous = lifted.insert(path.clone(), value.to_bits());
        assert!(
            previous.is_none(),
            "{}: Lift visited duplicate path `{path}` on {parameters}",
            model.name(),
        );
    }

    let mut numeric_paths = Vec::new();
    numbers(&written, &mut Vec::new(), &mut numeric_paths);
    let mut expected = BTreeMap::new();
    let mut seen_selectors = std::collections::BTreeSet::new();
    for path in numeric_paths {
        if path.len() == 1 && own_keys.contains(&path[0].as_str()) {
            continue;
        }
        let joined = path.join(".");
        let kind = schema_number_kind(&schema, &path).unwrap_or_else(|| {
            panic!(
                "{}: schema did not classify numeric path `{joined}` on {parameters}",
                model.name(),
            )
        });
        let leaf = path.iter().fold(&written, |inner, key| match inner {
            serde_json::Value::Array(items) => {
                &items[key.parse::<usize>().expect("an array index")]
            }
            _ => &inner[key.as_str()],
        });
        let value = leaf.as_f64().expect("a finite JSON number");
        if selectors.contains(&joined.as_str()) {
            assert_eq!(
                kind,
                NumberKind::Real,
                "{}: selector `{joined}` is already an integer in schema and does not belong in selectors",
                model.name(),
            );
            assert!(
                !lifted.contains_key(&joined),
                "{}: selector `{joined}` was lifted and does not belong in selectors",
                model.name(),
            );
            seen_selectors.insert(joined);
            continue;
        }
        match kind {
            NumberKind::Real => {
                let previous = expected.insert(joined.clone(), value.to_bits());
                assert!(
                    previous.is_none(),
                    "{}: serde wrote duplicate path `{joined}` on {parameters}",
                    model.name(),
                );
            }
            NumberKind::Integer => {
                assert!(
                    !lifted.contains_key(&joined),
                    "{}: Lift unexpectedly visited integer path `{joined}` on {parameters}",
                    model.name(),
                );
            }
        }
    }
    for &selector in selectors {
        assert!(
            seen_selectors.contains(selector),
            "{}: listed selector `{selector}` was not present on {parameters}",
            model.name(),
        );
    }

    assert_lift_maps_equal(model.name(), parameters, &lifted, &expected);
}

/// Checks [`assert_lift_matches_serde`] on `parameters`, then populates every `null`
/// `Option<R>` field in `model.defaulted(parameters)` with a distinct finite number and
/// checks [`assert_lift_matches_serde`] again, returning the number of optional real
/// fields populated.
pub fn assert_lift_matches_serde_with_optionals(
    model: &dyn Model,
    parameters: &str,
    selectors: &[&str],
) -> usize {
    assert_lift_matches_serde(model, parameters, selectors);
    let raw: jolty_ir::Parameters = serde_json::from_str(parameters).expect("a JSON object");
    let mut defaulted = serde_json::Value::Object(
        model
            .defaulted(&raw)
            .unwrap_or_else(|error| panic!("{}: {error}", model.name())),
    );
    let schema = model.schema().to_value();
    let mut counter = 0_u32;
    let populated = populate_optional_reals(&schema, &mut defaulted, &mut Vec::new(), &mut counter);
    if populated > 0 {
        assert_lift_matches_serde(model, &defaulted.to_string(), selectors);
    }
    populated
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

#[cfg(test)]
mod tests {
    use super::{Bench, REFERENCE_TEMPERATURE, assert_bench_jacobians_match, cases};
    use crate::{Input, Library, REGION};

    /// Every instance whose kernel calls it linear is so, at each case of it, at 64 points
    /// that spread each local unknown over ±10, at times, in a transient and not, with
    /// sources scaled, gmin across junctions and steps limited: its kernel writes the
    /// entries it writes at [`Input::ZERO`], bit for bit, limits nothing, and adds each
    /// residual as one term, its scale its magnitude; and each residual and charge is its
    /// entries times its local unknowns, as its group computes them, to within their
    /// rounding. A sum of `n` products, its row's entries times their columns' unknowns,
    /// lies within `γ(n+1) · Σ|g·x|` of the exact sum, `γ(m) = m·u/(1 − m·u)` with `u` half
    /// of `f64::EPSILON`, by `n` products and `n − 1` additions, and the kernel's way to it
    /// takes no more roundings (Higham, "Accuracy and Stability of Numerical Algorithms",
    /// 2002, §3.1): the two lie within `(n + 1)·ε·Σ|g·x|` of each other, to first order.
    #[test]
    fn every_linear_instances_equations_are_its_entries_times_its_unknowns() {
        let library = Library::standard();
        let bits = |values: &[f64]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        let mut checked = 0;
        for &(name, parameters, _) in cases::CASES {
            let model = library.get(name).expect("a model of the library");
            let mut bench = Bench::from_model(&**model, parameters, REFERENCE_TEMPERATURE)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            if !bench.groups.is_linear(0) {
                continue;
            }
            let (groups, runs) = (&bench.groups, &mut bench.runs);
            let stamped = runs
                .evaluate_by_kernel(groups, 0, &Input::ZERO)
                .expect("an instance");
            let layout = runs.block(groups, 0).expect("a block").entries.to_vec();
            let unknowns = u32::try_from(bench.unknowns.len()).expect("few unknowns");
            for point in 0..64_u32 {
                let x: Vec<f64> = (0..unknowns)
                    .map(|index| 20.0 * f64::from((point * 7 + index * 13) % 41) / 40.0 - 10.0)
                    .collect();
                let time = (point % 2 == 1).then_some(1e-7 * f64::from(point));
                let mut input = Input::new(&x, time, 1e-12 * f64::from(point % 3));
                input.source_scale = f64::from(point % 5) / 4.0;
                input.limiting = point % 3 == 0;
                let kernel = runs
                    .evaluate_by_kernel(groups, 0, &input)
                    .expect("an instance");
                let grouped = runs.evaluate_alone(groups, 0, &input).expect("an instance");
                assert!(!kernel.limited, "{name} limits at {x:?}");
                for plane in 0..2 {
                    assert_eq!(
                        bits(&kernel.entries[plane]),
                        bits(&stamped.entries[plane]),
                        "{name}'s entries at {x:?}"
                    );
                }
                let [residual, scale, _] = &kernel.rows;
                let largest: Vec<f64> = residual.iter().map(|sum| 0.0_f64.max(sum.abs())).collect();
                assert_eq!(bits(scale), bits(&largest), "{name}'s scales at {x:?}");
                for (plane, entries) in [(0, &grouped.entries[0]), (2, &grouped.entries[1])] {
                    let (mut magnitudes, mut terms) = (vec![0.0; x.len()], vec![1.0; x.len()]);
                    for (&(row, column), &entry) in layout.iter().zip(entries) {
                        magnitudes[row] += (entry * x[column]).abs();
                        terms[row] += 1.0;
                    }
                    let rows = magnitudes.iter().zip(&terms).enumerate();
                    for (row, (magnitude, roundings)) in rows {
                        let (by_kernel, by_entries) =
                            (kernel.rows[plane][row], grouped.rows[plane][row]);
                        let bound = roundings * f64::EPSILON * magnitude;
                        assert!(
                            (by_kernel - by_entries).abs() <= bound,
                            "{name}'s {} {row} at {x:?}: {by_kernel:e} by its kernel, \
                             {by_entries:e} by its entries",
                            if plane == 0 { "residual" } else { "charge" }
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 0, "no instance is linear");
    }

    /// Every model that distinguishes regions of operation numbers them with its variable
    /// `region` and reports each of its margins among its variables, at each case of it.
    #[test]
    fn every_models_regions_are_among_its_variables() {
        let library = Library::standard();
        for &(name, parameters, given) in cases::CASES {
            let model = library.get(name).expect("a model of the library");
            let mut bench = Bench::from_model(&**model, parameters, REFERENCE_TEMPERATURE)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let regions = bench.regions;
            assert_eq!(
                regions.names.is_empty(),
                regions.margins.is_empty(),
                "{name}"
            );
            if regions.names.is_empty() {
                assert!(!bench.variables.contains(&REGION), "{name}");
                continue;
            }
            let x = cases::values(given, bench.unknowns.len());
            bench.settle(&x);
            let (region, margins) = bench.region();
            assert!(regions.names.contains(&region), "{name}");
            assert!(margins.iter().all(|margin| margin.is_finite()), "{name}");
        }
    }

    /// Every analog model's conductances and capacitances are the derivatives of its
    /// residuals and charges, at each case of it, as a central difference finds them.
    #[test]
    fn every_models_jacobian_is_the_derivative_of_its_residuals_and_charges() {
        let library = Library::standard();
        let analog: Vec<&str> = library
            .names()
            .filter(|name| library.get(name).is_some())
            .collect();
        assert_eq!(
            cases::covered(),
            analog,
            "a case of every analog model, and no more"
        );
        for &(name, parameters, given) in cases::CASES {
            let model = library.get(name).expect("a model of the library");
            let mut bench = Bench::from_model(&**model, parameters, REFERENCE_TEMPERATURE)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let x = cases::values(given, bench.unknowns.len());
            assert_bench_jacobians_match(&mut bench, name, &x);
        }
    }
}
