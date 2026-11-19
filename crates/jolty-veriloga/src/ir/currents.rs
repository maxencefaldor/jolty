//! Rewriting of probed branch currents: substituting flow contributions for probed flow
//! branches and introducing current unknowns for noisy branches whose currents are probed.

use std::collections::{BTreeMap, BTreeSet};

use jolty_models::Nature;

use crate::Error;

use super::analysis::{visit_contributions, visit_contributions_mut};
use super::{Branch, Expression, GROUND, Module, Probe, Statement};

/// A contribution to a branch's flow or potential, and the conditions it lies under:
/// those that shape the kernel, each by its number and whether it holds, and whether
/// any other does.
struct Site {
    branch: usize,
    potential: bool,
    resistive: Option<Expression>,
    reactive: bool,
    shape: Vec<(usize, bool)>,
    dynamic: bool,
}

/// Collects all resistive and reactive contribution [`Site`]s in `statements` under the active
/// `shape` and `dynamic` guards.
fn sites(
    module: &Module,
    statements: &[Statement],
    (shape, dynamic): (&[(usize, bool)], bool),
    found: &mut Vec<Site>,
) {
    for statement in statements {
        match statement {
            Statement::Contribute {
                branch,
                potential,
                resistive,
                reactive,
                ..
            } if resistive.is_some() || reactive.is_some() => found.push(Site {
                branch: *branch,
                potential: *potential,
                resistive: resistive.clone(),
                reactive: reactive.is_some(),
                shape: shape.to_vec(),
                dynamic,
            }),
            Statement::If(id, condition, then, otherwise) => {
                if module.is_dynamic(condition) {
                    sites(module, then, (shape, true), found);
                    sites(module, otherwise, (shape, true), found);
                } else {
                    for (inner, holds) in [(then, true), (otherwise, false)] {
                        let mut deeper = shape.to_vec();
                        deeper.push((*id, holds));
                        sites(module, inner, (&deeper, dynamic), found);
                    }
                }
            }
            Statement::While(_, body) => sites(module, body, (shape, true), found),
            Statement::Assign(..)
            | Statement::Contribute { .. }
            | Statement::Fail(..)
            | Statement::Warn(..) => {}
        }
    }
}

/// Conjoins the shape predicates in `shape` into a single boolean [`Expression`].
fn guard(shape: &[(usize, bool)]) -> Option<Expression> {
    shape
        .iter()
        .map(|&(id, holds)| {
            if holds {
                Expression::Shape(id)
            } else {
                Expression::Unary("!", Box::new(Expression::Shape(id)))
            }
        })
        .reduce(|left, right| Expression::Binary("&&", Box::new(left), Box::new(right)))
}

/// Builds an inline expression for the current of flow-carrying `branch` from its
/// contributions in `found`, falling back to the branch unknown wherever `branch` carries a
/// potential.
///
/// Returns `None` when `branch` has no flow contributions so probes keep its branch unknown.
fn flow_current(
    module: &Module,
    branch: usize,
    found: &[Site],
) -> Result<Option<Expression>, Error> {
    let name = &module.branches[branch].name;
    let of = |potential: bool| {
        found
            .iter()
            .filter(move |site| site.branch == branch && site.potential == potential)
    };
    if of(false).next().is_none() {
        return Ok(None);
    }
    let mut flow: Option<Expression> = None;
    for site in of(false) {
        if site.dynamic || site.reactive {
            return Err(Error::at(
                0,
                format!(
                    "the current of {name}, a flow under a condition on the probes or of a charge, is not compiled yet"
                ),
            ));
        }
        let value = site.resistive.clone().unwrap_or(Expression::Real(0.0));
        let term = match guard(&site.shape) {
            Some(condition) => Expression::Conditional(
                Box::new(condition),
                Box::new(value),
                Box::new(Expression::Real(0.0)),
            ),
            None => value,
        };
        flow = Some(match flow {
            Some(sum) => Expression::Binary("+", Box::new(sum), Box::new(term)),
            None => term,
        });
    }
    let flow = flow.unwrap_or(Expression::Real(0.0));
    let held: Option<Option<Expression>> =
        of(true)
            .map(|site| guard(&site.shape))
            .reduce(|left, right| match (left, right) {
                (Some(left), Some(right)) => {
                    Some(Expression::Binary("||", Box::new(left), Box::new(right)))
                }
                _ => None,
            });
    Ok(Some(match held {
        None => flow,
        Some(None) => Expression::Probe(Probe::Current(branch)),
        Some(Some(condition)) => Expression::Conditional(
            Box::new(condition),
            Box::new(Expression::Probe(Probe::Current(branch))),
            Box::new(flow),
        ),
    }))
}

/// Replaces each probe of the current of a branch that carries a flow with what is
/// contributed to it, and returns whether it replaced any. The contributions come later
/// in the analog block than the probe may, so nothing they read may be assigned after it.
pub(super) fn flow_currents(module: &mut Module) -> Result<bool, Error> {
    let noisy = noisy_currents(module)?;
    let mut found = Vec::new();
    sites(module, &module.analog, (&[], false), &mut found);
    let read = probed_currents(module);
    let mut replacements = BTreeMap::new();
    for branch in read {
        if let Some(current) = flow_current(module, branch, &found)?
            && current != Expression::Probe(Probe::Current(branch))
        {
            replacements.insert(branch, current);
        }
    }
    if replacements.is_empty() {
        return Ok(noisy);
    }
    // Record each statement's sequence index and each variable's last assignment index.
    let mut last = vec![0_usize; module.variables.len()];
    let mut position = 0;
    let mut probed: Vec<(usize, usize)> = Vec::new();
    order(&module.analog, &mut position, &mut last, &mut |at, expr| {
        expr.walk(&mut |inner| {
            if let Expression::Probe(Probe::Current(branch)) = inner {
                probed.push((*branch, at));
            }
        });
    });
    for &(branch, at) in &probed {
        let Some(current) = replacements.get(&branch) else {
            continue;
        };
        let mut late = None;
        current.walk(&mut |inner| {
            if let Expression::Variable(variable) = inner
                && last[*variable] >= at
            {
                late = Some(*variable);
            }
        });
        if let Some(variable) = late {
            return Err(Error::at(
                0,
                format!(
                    "the current of {} is read before {}, which its flow reads, is assigned",
                    module.branches[branch].name, module.variables[variable].0
                ),
            ));
        }
    }
    apply_replacements(module, &replacements);
    Ok(true)
}

/// Returns the indices of all branches whose currents are probed in `module`.
pub(crate) fn probed_currents(module: &Module) -> BTreeSet<usize> {
    let mut read = BTreeSet::new();
    let mut visit = |expr: &Expression| {
        expr.walk(&mut |inner| {
            if let Expression::Probe(Probe::Current(branch)) = inner {
                read.insert(*branch);
            }
        });
    };
    for statement in &module.analog {
        statement.walk(&mut visit);
    }
    for limit in &module.limits {
        limit.arguments.iter().for_each(&mut visit);
    }
    read
}

/// Finds each noisy branch whose current is probed, verifying none holds a potential,
/// and creates a current-unknown node and auxiliary equation branch for each.
fn collect_noisy_branches(module: &mut Module) -> Result<BTreeMap<usize, (usize, usize)>, Error> {
    let (mut noisy, mut potential) = (BTreeSet::new(), BTreeSet::new());
    visit_contributions(&module.analog, &mut |statement| {
        if let Statement::Contribute {
            branch,
            potential: is_potential,
            noise,
            ..
        } = statement
        {
            if *is_potential {
                potential.insert(*branch);
            } else if !noise.is_empty() {
                noisy.insert(*branch);
            }
        }
    });
    let read = probed_currents(module);
    let mut rewritten = BTreeMap::new();
    for &branch in noisy.intersection(&read) {
        let name = module.branches[branch].name.clone();
        if potential.contains(&branch) {
            return Err(Error::at(
                0,
                format!(
                    "the current of {name}, which carries noise and holds a potential, is not compiled yet"
                ),
            ));
        }
        module.nodes.push(name.clone());
        module.labels.push(name.clone());
        module.natures.push(Nature::Current);
        let node = module.nodes.len() - 1;
        module.branches.push(Branch {
            name: format!("{name}_ground"),
            plus: node,
            minus: GROUND,
        });
        rewritten.insert(branch, (node, module.branches.len() - 1));
    }
    Ok(rewritten)
}

/// Introduces a current-unknown node for every probed flow branch that carries noise so
/// readers observe the branch noise (for example, PSP induced gate noise).
///
/// Each rewritten branch gets a [`Nature::Current`] node whose auxiliary equation equates the
/// node unknown with the branch flow plus noise, and returns `true` if any branch was
/// rewritten. Negating a noise source preserves its spectral density and correlations.
fn noisy_currents(module: &mut Module) -> Result<bool, Error> {
    let rewritten = collect_noisy_branches(module)?;
    if rewritten.is_empty() {
        return Ok(false);
    }
    let negate = |expr: &mut Option<Expression>| {
        if let Some(value) = expr.take() {
            *expr = Some(Expression::Unary("-", Box::new(value)));
        }
    };
    visit_contributions_mut(&mut module.analog, &mut |statement| {
        if let Statement::Contribute {
            branch,
            potential: false,
            resistive,
            reactive,
            ..
        } = statement
            && let Some(&(_, equation)) = rewritten.get(branch)
        {
            *branch = equation;
            negate(resistive);
            negate(reactive);
        }
    });
    let replacements: BTreeMap<usize, Expression> = rewritten
        .iter()
        .map(|(&branch, &(node, _))| (branch, Expression::Probe(Probe::Voltage(node, GROUND))))
        .collect();
    apply_replacements(module, &replacements);
    for (&branch, &(node, equation)) in &rewritten {
        let current = Expression::Probe(Probe::Voltage(node, GROUND));
        for target in [branch, equation] {
            module.analog.push(Statement::Contribute {
                branch: target,
                potential: false,
                resistive: Some(current.clone()),
                reactive: None,
                reactive_factors: Vec::new(),
                noise: Vec::new(),
            });
        }
    }
    Ok(true)
}

/// Applies `replacements` to every current probe across `module`'s analog block and
/// limited-probe arguments, leaving the replacements themselves untouched.
fn apply_replacements(module: &mut Module, replacements: &BTreeMap<usize, Expression>) {
    let mut replace = |expr: &mut Expression| {
        expr.transform_mut(&mut |inner| {
            if let Expression::Probe(Probe::Current(branch)) = inner
                && let Some(current) = replacements.get(branch)
            {
                *inner = current.clone();
                return true;
            }
            false
        });
    };
    for statement in &mut module.analog {
        statement.expressions_mut(&mut replace);
    }
    for limit in &mut module.limits {
        for argument in &mut limit.arguments {
            replace(argument);
        }
    }
}

/// Numbers each statement of `statements` from `position` on, in order, records where
/// each variable is last assigned in `last`, and calls `visit` on each expression with
/// the number of its statement.
fn order(
    statements: &[Statement],
    position: &mut usize,
    last: &mut [usize],
    visit: &mut impl FnMut(usize, &Expression),
) {
    for statement in statements {
        *position += 1;
        let at = *position;
        match statement {
            Statement::Assign(variable, value) => {
                visit(at, value);
                last[*variable] = at;
            }
            Statement::If(_, condition, then, otherwise) => {
                visit(at, condition);
                order(then, position, last, visit);
                order(otherwise, position, last, visit);
            }
            Statement::While(condition, body) => {
                visit(at, condition);
                order(body, position, last, visit);
            }
            Statement::Contribute { .. } | Statement::Fail(..) | Statement::Warn(..) => {
                statement.walk(&mut |expr| visit(at, expr));
            }
        }
    }
}
