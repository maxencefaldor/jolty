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

use jolty_numerics::{Pattern, PatternBuilder};

// The cases and the laws serve this crate's own tests, not the bench it shares.
#[cfg(test)]
mod cases;
#[cfg(test)]
mod laws;

#[cfg(test)]
pub(crate) use laws::assert_mirrors;

use crate::{
    BOLTZMANN, Binding, Contributions, Differentiable, Gradient, Groups, Input, Kernel, Model,
    Moving, Nature, Node, NoiseSource, ParameterError, REFERENCE_TEMPERATURE, REGION, Regions,
    Runs, Seeds, Twin, Unknown, UnknownIndex,
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
    pub(crate) fn new<K: Kernel>(parameters: &str) -> Self {
        Self::at::<K>(parameters, REFERENCE_TEMPERATURE)
    }

    pub(crate) fn try_new<K: Kernel>(parameters: &str) -> Result<Self, ParameterError> {
        Self::try_at::<K>(parameters, REFERENCE_TEMPERATURE)
    }

    /// A bench for an instance of `K` with `parameters` at `temperature`, in kelvins.
    #[must_use]
    pub fn at<K: Kernel>(parameters: &str, temperature: f64) -> Self {
        Self::try_at::<K>(parameters, temperature).expect("valid parameters")
    }

    /// A bench as [`Bench::at`] makes, or why the parameters are out of range.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    pub fn try_at<K: Kernel>(parameters: &str, temperature: f64) -> Result<Self, ParameterError> {
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
    pub(crate) fn natures(&self) -> Vec<Nature> {
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
pub(crate) fn assert_jacobians_match<K: Kernel>(parameters: &str, x: &[f64]) {
    assert_bench_jacobians_match(&mut Bench::new::<K>(parameters), K::NAME, x);
}

/// As [`assert_jacobians_match`], with the instance evaluated as a transient analysis
/// steps through time.
pub(crate) fn assert_stepping_jacobians_match<K: Kernel>(parameters: &str, x: &[f64]) {
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
        let add = |(sum, sums): &mut Moved, (value, derivatives): Moving<'_>| {
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
pub fn twin<K: Kernel>(parameters: &str, temperature: f64, seeds: &Seeds) -> Option<Twin> {
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
pub(crate) fn assert_parameter_gradients_match<K: Kernel>(parameters: &str, x: &[f64]) {
    assert_model_parameter_gradients_match(&Differentiable::<K>::new(), parameters, x, None);
}

/// As [`assert_parameter_gradients_match`], in a transient analysis at `time`.
pub(crate) fn assert_parameter_gradients_match_at<K: Kernel>(
    parameters: &str,
    x: &[f64],
    time: f64,
) {
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
