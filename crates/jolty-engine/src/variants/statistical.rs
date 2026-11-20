//! Compiled per-placement statistical parameter models evaluated inside
//! [`MonteCarlo`](super::MonteCarlo).
//!
//! Each placed component carrying a [`jolty_ir::Statistical`] graph compiles its steps and
//! target expressions into [`jolty_expressions::Expression`] tapes with pre-resolved slot
//! indices and precomputed 64-bit FNV-1a variate keys. During batch execution, each worker
//! evaluates the compiled plan on local scratch buffers without locking or allocations beyond
//! the per-sample override list.

use std::collections::{HashMap, HashSet};

use jolty_expressions::{Expression, Variable};
use jolty_ir::{
    Circuit, Component, Of, Statistical, StatisticalFunction, StatisticalStep, StatisticalTarget,
};
use serde::Deserialize;

use crate::elaborate::{ElaborationProblem, Location};
use crate::{Elaborated, ParameterId, Values};

use super::{TolerancedError, Variate, key, variate_by_key};

/// Placed component carrying a [`Statistical`] parameter model to evaluate during Monte Carlo.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticalPlacement {
    /// Hierarchical placement path of the component (empty when rooted at top).
    pub path: String,
    /// Statistical steps and device parameter targets attached to the component.
    pub statistical: Statistical,
}

/// Joins hierarchical path `prefix` and relative `suffix` with `/`, omitting the separator
/// when either side is empty.
#[must_use]
pub(super) fn join_path(prefix: &str, suffix: &str) -> String {
    match (prefix.is_empty(), suffix.is_empty()) {
        (true, _) => suffix.to_owned(),
        (_, true) => prefix.to_owned(),
        (false, false) => format!("{prefix}/{suffix}"),
    }
}

/// Active traversal frame in [`statistical`].
struct Frame<'a> {
    placed: &'a Component,
    prefix: String,
    next_instance: usize,
}

/// Collects every non-empty [`Statistical`] model placed in `circuit` in deterministic
/// hierarchical order.
///
/// # Errors
///
/// Returns [`TolerancedError::Recursive`] when component placements form a cycle.
pub fn statistical(circuit: &Circuit) -> Result<Vec<StatisticalPlacement>, TolerancedError> {
    let top = circuit.top.as_str();
    let Some(root) = circuit.components.get(top) else {
        return Ok(Vec::new());
    };
    let mut found = Vec::new();
    if let Some(model) = &root.statistical
        && !model.is_empty()
    {
        found.push(StatisticalPlacement {
            path: String::new(),
            statistical: model.clone(),
        });
    }
    let mut active = vec![top];
    let mut active_set = HashSet::from([top]);
    let mut stack = vec![Frame {
        placed: root,
        prefix: String::new(),
        next_instance: 0,
    }];
    while let Some(frame) = stack.last_mut() {
        let Some(instance) = frame.placed.instances.get(frame.next_instance) else {
            stack.pop();
            if let Some(popped) = active.pop() {
                active_set.remove(popped);
            }
            continue;
        };
        frame.next_instance += 1;
        let Of::Component(inner) = &instance.of else {
            continue;
        };
        let child = inner.as_str();
        if active_set.contains(child)
            && let Some(start) = active.iter().position(|&seen| seen == child)
        {
            let cycle = active[start..]
                .iter()
                .copied()
                .chain(std::iter::once(child))
                .map(str::to_owned)
                .collect();
            return Err(TolerancedError::Recursive {
                component: child.to_owned(),
                cycle,
            });
        }
        let path = format!("{}{}", frame.prefix, instance.name.as_str());
        if let Some(definition) = circuit.components.get(child) {
            if let Some(model) = &definition.statistical
                && !model.is_empty()
            {
                found.push(StatisticalPlacement {
                    path: path.clone(),
                    statistical: model.clone(),
                });
            }
            active.push(child);
            active_set.insert(child);
            stack.push(Frame {
                placed: definition,
                prefix: format!("{path}/"),
                next_instance: 0,
            });
        }
    }
    Ok(found)
}

/// Expression compiled once with pre-resolved slot indices into a placement's step table.
#[derive(Debug, Clone, PartialEq)]
struct CompiledExpression {
    expression: Expression,
    slots: Vec<usize>,
}

impl CompiledExpression {
    /// Parses `text` and resolves every `p(<name>)` variable against `slots_by_name`,
    /// pushing an [`ElaborationProblem`] and returning `None` on failure.
    fn compile(
        text: &str,
        slots_by_name: &HashMap<&str, usize>,
        at: &Location,
        problems: &mut Vec<ElaborationProblem>,
    ) -> Option<Self> {
        let expression: Expression = match text.parse() {
            Ok(expression) => expression,
            Err(error) => {
                problems.push(ElaborationProblem::Parameters {
                    at: at.clone(),
                    message: format!("statistical expression `{text}` is invalid: {error}"),
                });
                return None;
            }
        };
        let mut slots = Vec::with_capacity(expression.variables().len());
        for variable in expression.variables() {
            match variable {
                Variable::Parameter(name) => {
                    let Some(&slot) = slots_by_name.get(name.as_str()) else {
                        problems.push(ElaborationProblem::Parameters {
                            at: at.clone(),
                            message: format!(
                                "statistical expression `{text}` reads undefined step `{name}`"
                            ),
                        });
                        return None;
                    };
                    slots.push(slot);
                }
                Variable::Voltage(..) | Variable::Current(_) | Variable::Time => {
                    problems.push(ElaborationProblem::Parameters {
                        at: at.clone(),
                        message: format!(
                            "statistical expression `{text}` reads a non-parameter variable"
                        ),
                    });
                    return None;
                }
            }
        }
        Some(Self { expression, slots })
    }

    /// Evaluates this expression against `step_values` using reusable scratch buffers.
    fn evaluate(
        &self,
        step_values: &[f64],
        inputs: &mut Vec<f64>,
        tape_values: &mut Vec<f64>,
    ) -> f64 {
        inputs.clear();
        inputs.extend(self.slots.iter().map(|&slot| step_values[slot]));
        self.expression.evaluate_value(inputs, tape_values)
    }
}

/// Compiled statistical draw parameters boxed inside [`CompiledStep::Draw`].
#[derive(Debug, Clone, PartialEq)]
struct CompiledDraw {
    /// Precomputed 64-bit FNV-1a hash of `(resolved_scope, call)`.
    variate_key: u64,
    function: StatisticalFunction,
    nominal: CompiledExpression,
    spread: CompiledExpression,
    sigma: Option<CompiledExpression>,
}

/// Compiled variable step inside one placement.
#[derive(Debug, Clone, PartialEq)]
enum CompiledStep {
    Draw(Box<CompiledDraw>),
    Value(CompiledExpression),
}

/// Compiled device parameter assignment produced by one placement.
#[derive(Debug, Clone, PartialEq)]
struct CompiledTarget {
    parameter: ParameterId,
    expression: CompiledExpression,
}

/// Compiled statistical evaluation graph for one placed component.
#[derive(Debug, Clone, PartialEq)]
struct CompiledPlacement {
    steps: Vec<CompiledStep>,
    targets: Vec<CompiledTarget>,
}

/// Compiled statistical evaluation plan across all statistical placements in a circuit.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct StatisticalPlan {
    placements: Vec<CompiledPlacement>,
    max_steps: usize,
    max_inputs: usize,
}

impl StatisticalPlan {
    /// Compiles `placements` against `elaborated`, collecting every elaboration or syntax
    /// problem found. Returns `None` when no placement defines targets or compilation records
    /// a problem.
    pub(super) fn compile(
        elaborated: &Elaborated,
        placements: Vec<StatisticalPlacement>,
        problems: &mut Vec<ElaborationProblem>,
    ) -> Option<Self> {
        if placements.is_empty() {
            return None;
        }
        let mut compiled = Vec::with_capacity(placements.len());
        let mut max_steps = 0usize;
        let mut max_inputs = 0usize;
        for placement in placements {
            if let Some(built) =
                compile_placement(elaborated, &placement, &mut max_inputs, problems)
            {
                max_steps = max_steps.max(built.steps.len());
                if !built.targets.is_empty() {
                    compiled.push(built);
                }
            }
        }
        if compiled.is_empty() {
            None
        } else {
            Some(Self {
                placements: compiled,
                max_steps,
                max_inputs,
            })
        }
    }

    /// Evaluates every compiled placement for `(seed, index)` and writes target parameters
    /// into `values`.
    pub(super) fn evaluate(&self, seed: u64, index: u64, values: &mut Values) {
        let mut step_values = Vec::with_capacity(self.max_steps);
        let mut inputs = Vec::with_capacity(self.max_inputs);
        let mut tape_values = Vec::new();
        for placement in &self.placements {
            step_values.clear();
            for step in &placement.steps {
                let value = match step {
                    CompiledStep::Draw(draw) => evaluate_draw(
                        (seed, index, draw.variate_key, draw.function),
                        (&draw.nominal, &draw.spread, draw.sigma.as_ref()),
                        (&step_values, &mut inputs, &mut tape_values),
                    ),
                    CompiledStep::Value(expression) => {
                        expression.evaluate(&step_values, &mut inputs, &mut tape_values)
                    }
                };
                step_values.push(value);
            }
            for target in &placement.targets {
                let value = target
                    .expression
                    .evaluate(&step_values, &mut inputs, &mut tape_values);
                values.set(target.parameter, value);
            }
        }
    }
}

/// Evaluates one compiled statistical draw step matching `jolty.spice.Sample.draw` bit-for-bit.
fn evaluate_draw(
    (seed, index, variate_key, function): (u64, u64, u64, StatisticalFunction),
    (nominal, spread, sigma): (
        &CompiledExpression,
        &CompiledExpression,
        Option<&CompiledExpression>,
    ),
    (step_values, inputs, tape_values): (&[f64], &mut Vec<f64>, &mut Vec<f64>),
) -> f64 {
    let nominal_value = nominal.evaluate(step_values, inputs, tape_values);
    let spread_value = spread.evaluate(step_values, inputs, tape_values);
    match function {
        StatisticalFunction::Agauss | StatisticalFunction::Gauss => {
            let sigma_value =
                sigma.map_or(0.0, |node| node.evaluate(step_values, inputs, tape_values));
            if spread_value <= 0.0 || sigma_value <= 0.0 {
                return nominal_value;
            }
            let mut deviation = spread_value / sigma_value;
            if function == StatisticalFunction::Gauss {
                deviation *= nominal_value;
            }
            let normal = variate_by_key(Variate::Normal, seed, index, variate_key);
            nominal_value + deviation * normal
        }
        StatisticalFunction::Aunif => {
            let uniform = variate_by_key(Variate::Uniform, seed, index, variate_key);
            nominal_value + spread_value * uniform
        }
        StatisticalFunction::Unif => {
            let uniform = variate_by_key(Variate::Uniform, seed, index, variate_key);
            nominal_value + nominal_value * spread_value * uniform
        }
        StatisticalFunction::Limit => {
            let uniform = variate_by_key(Variate::Uniform, seed, index, variate_key);
            nominal_value
                + if uniform > 0.0 {
                    spread_value
                } else {
                    -spread_value
                }
        }
    }
}

/// Compiles one [`StatisticalPlacement`] against `elaborated`.
fn compile_placement(
    elaborated: &Elaborated,
    placement: &StatisticalPlacement,
    max_inputs: &mut usize,
    problems: &mut Vec<ElaborationProblem>,
) -> Option<CompiledPlacement> {
    let placement_location = Location {
        path: placement.path.clone(),
        source_location: None,
    };
    let mut slots_by_name: HashMap<&str, usize> =
        HashMap::with_capacity(placement.statistical.steps.len());
    let mut steps = Vec::with_capacity(placement.statistical.steps.len());
    let initial_problems = problems.len();
    for step in &placement.statistical.steps {
        let name = step.name();
        if slots_by_name.contains_key(name) {
            problems.push(ElaborationProblem::Parameters {
                at: placement_location.clone(),
                message: format!("statistical step `{name}` is defined twice"),
            });
            continue;
        }
        let compiled_step = compile_step(
            &placement.path,
            step,
            &slots_by_name,
            &placement_location,
            max_inputs,
            problems,
        );
        if let Some(compiled_step) = compiled_step {
            slots_by_name.insert(name, steps.len());
            steps.push(compiled_step);
        }
    }
    let mut targets = Vec::with_capacity(placement.statistical.targets.len());
    for target in &placement.statistical.targets {
        if let Some(compiled_target) = compile_target(
            elaborated,
            &placement.path,
            target,
            &slots_by_name,
            max_inputs,
            problems,
        ) {
            targets.push(compiled_target);
        }
    }
    if problems.len() == initial_problems {
        Some(CompiledPlacement { steps, targets })
    } else {
        None
    }
}

/// Compiles `text` against `slots_by_name`, updating `max_inputs` or recording a problem.
fn compile_expression(
    text: &str,
    slots_by_name: &HashMap<&str, usize>,
    location: &Location,
    max_inputs: &mut usize,
    problems: &mut Vec<ElaborationProblem>,
) -> Option<CompiledExpression> {
    let compiled = CompiledExpression::compile(text, slots_by_name, location, problems)?;
    *max_inputs = (*max_inputs).max(compiled.slots.len());
    Some(compiled)
}

/// Compiles one [`StatisticalStep`] into a [`CompiledStep`].
fn compile_step(
    placement_path: &str,
    step: &StatisticalStep,
    slots_by_name: &HashMap<&str, usize>,
    location: &Location,
    max_inputs: &mut usize,
    problems: &mut Vec<ElaborationProblem>,
) -> Option<CompiledStep> {
    match step {
        StatisticalStep::Draw { .. } => compile_draw_step(
            placement_path,
            step,
            slots_by_name,
            location,
            max_inputs,
            problems,
        ),
        StatisticalStep::Value { expression, .. } => {
            compile_expression(expression, slots_by_name, location, max_inputs, problems)
                .map(CompiledStep::Value)
        }
    }
}

/// Compiles one [`StatisticalStep::Draw`] into a [`CompiledStep::Draw`].
fn compile_draw_step(
    placement_path: &str,
    step: &StatisticalStep,
    slots_by_name: &HashMap<&str, usize>,
    location: &Location,
    max_inputs: &mut usize,
    problems: &mut Vec<ElaborationProblem>,
) -> Option<CompiledStep> {
    let StatisticalStep::Draw {
        name,
        shared,
        scope,
        call,
        function,
        nominal,
        spread,
        sigma,
    } = step
    else {
        return None;
    };
    if *shared && !scope.is_empty() {
        problems.push(ElaborationProblem::Parameters {
            at: location.clone(),
            message: format!("shared draw step `{name}` specifies local scope `{scope}`"),
        });
        return None;
    }
    let nominal = compile_expression(nominal, slots_by_name, location, max_inputs, problems)?;
    let spread = compile_expression(spread, slots_by_name, location, max_inputs, problems)?;
    let sigma = match (function.takes_sigma(), sigma) {
        (true, Some(text)) => Some(compile_expression(
            text,
            slots_by_name,
            location,
            max_inputs,
            problems,
        )?),
        (true, None) => {
            problems.push(ElaborationProblem::Parameters {
                at: location.clone(),
                message: format!("draw step `{name}` requires a sigma argument expression"),
            });
            return None;
        }
        (false, Some(_)) => {
            problems.push(ElaborationProblem::Parameters {
                at: location.clone(),
                message: format!(
                    "draw step `{name}` supplies a sigma expression not accepted by its function"
                ),
            });
            return None;
        }
        (false, None) => None,
    };
    let resolved_scope = if *shared {
        String::new()
    } else {
        join_path(placement_path, scope)
    };
    Some(CompiledStep::Draw(Box::new(CompiledDraw {
        variate_key: key(&resolved_scope, call),
        function: *function,
        nominal,
        spread,
        sigma,
    })))
}

/// Compiles one [`StatisticalTarget`] against `elaborated`.
fn compile_target(
    elaborated: &Elaborated,
    placement_path: &str,
    target: &StatisticalTarget,
    slots_by_name: &HashMap<&str, usize>,
    max_inputs: &mut usize,
    problems: &mut Vec<ElaborationProblem>,
) -> Option<CompiledTarget> {
    let instance_path = join_path(placement_path, &target.instance);
    let Some(parameter) = elaborated.parameter(&instance_path, &target.parameter) else {
        problems.push(match elaborated.location(&instance_path) {
            Some(at) => ElaborationProblem::UnknownParameter {
                at,
                parameter: target.parameter.clone(),
            },
            None => ElaborationProblem::UnknownOverride(instance_path),
        });
        return None;
    };
    let location = elaborated.location(&instance_path).unwrap_or(Location {
        path: instance_path,
        source_location: None,
    });
    compile_expression(
        &target.expression,
        slots_by_name,
        &location,
        max_inputs,
        problems,
    )
    .map(|expression| CompiledTarget {
        parameter,
        expression,
    })
}
