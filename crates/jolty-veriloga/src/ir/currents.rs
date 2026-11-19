//! Rewriting of probed branch currents: substituting flow contributions for probed flow
//! branches and introducing current unknowns for noisy branches whose currents are probed.

use std::collections::{BTreeMap, BTreeSet};

use jolty_models::Nature;

use crate::Error;

use super::analysis::{visit_contributions, visit_contributions_mut};
use super::{Branch, Expr, GROUND, Module, Probe, Stmt};

/// A contribution to a branch's flow or potential, and the conditions it lies under:
/// those that shape the kernel, each by its number and whether it holds, and whether
/// any other does.
struct Site {
    branch: usize,
    potential: bool,
    resistive: Option<Expr>,
    reactive: bool,
    shape: Vec<(usize, bool)>,
    dynamic: bool,
}

/// The sites of `statements`, under `shape` and `dynamic`.
fn sites(
    module: &Module,
    statements: &[Stmt],
    (shape, dynamic): (&[(usize, bool)], bool),
    found: &mut Vec<Site>,
) {
    for statement in statements {
        match statement {
            Stmt::Contribute {
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
            Stmt::If(id, condition, then, otherwise) => {
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
            Stmt::While(_, body) => sites(module, body, (shape, true), found),
            Stmt::Assign(..) | Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {}
        }
    }
}

/// The truth of the shape's conditions `shape`, all of them.
fn guard(shape: &[(usize, bool)]) -> Option<Expr> {
    shape
        .iter()
        .map(|&(id, holds)| {
            if holds {
                Expr::Shape(id)
            } else {
                Expr::Unary("!", Box::new(Expr::Shape(id)))
            }
        })
        .reduce(|left, right| Expr::Binary("&&", Box::new(left), Box::new(right)))
}

/// The current of `branch`, as a probe reads it: the sum of its flow's contributions,
/// each where the shape makes it, and its own unknown where the shape makes it hold a
/// potential; `None` when it carries no flow, so that the probe reads its unknown.
fn flow_current(module: &Module, branch: usize, found: &[Site]) -> Result<Option<Expr>, Error> {
    let name = &module.branches[branch].name;
    let of = |potential: bool| {
        found
            .iter()
            .filter(move |site| site.branch == branch && site.potential == potential)
    };
    if of(false).next().is_none() {
        return Ok(None);
    }
    let mut flow: Option<Expr> = None;
    for site in of(false) {
        if site.dynamic || site.reactive {
            return Err(Error::at(
                0,
                format!(
                    "the current of {name}, a flow under a condition on the probes or of a charge, is not compiled yet"
                ),
            ));
        }
        let value = site.resistive.clone().unwrap_or(Expr::Real(0.0));
        let term = match guard(&site.shape) {
            Some(condition) => Expr::Conditional(
                Box::new(condition),
                Box::new(value),
                Box::new(Expr::Real(0.0)),
            ),
            None => value,
        };
        flow = Some(match flow {
            Some(sum) => Expr::Binary("+", Box::new(sum), Box::new(term)),
            None => term,
        });
    }
    let flow = flow.unwrap_or(Expr::Real(0.0));
    let held: Option<Option<Expr>> =
        of(true)
            .map(|site| guard(&site.shape))
            .reduce(|left, right| match (left, right) {
                (Some(left), Some(right)) => {
                    Some(Expr::Binary("||", Box::new(left), Box::new(right)))
                }
                _ => None,
            });
    Ok(Some(match held {
        None => flow,
        Some(None) => Expr::Probe(Probe::Current(branch)),
        Some(Some(condition)) => Expr::Conditional(
            Box::new(condition),
            Box::new(Expr::Probe(Probe::Current(branch))),
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
            && current != Expr::Probe(Probe::Current(branch))
        {
            replacements.insert(branch, current);
        }
    }
    if replacements.is_empty() {
        return Ok(noisy);
    }
    // Where each variable is last assigned, and where each statement lies, in order.
    let mut last = vec![0_usize; module.variables.len()];
    let mut position = 0;
    let mut probed: Vec<(usize, usize)> = Vec::new();
    order(&module.analog, &mut position, &mut last, &mut |at, expr| {
        expr.walk(&mut |inner| {
            if let Expr::Probe(Probe::Current(branch)) = inner {
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
            if let Expr::Variable(variable) = inner
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

/// The branches whose currents `module` probes.
pub(crate) fn probed_currents(module: &Module) -> BTreeSet<usize> {
    let mut read = BTreeSet::new();
    let mut visit = |expr: &Expr| {
        expr.walk(&mut |inner| {
            if let Expr::Probe(Probe::Current(branch)) = inner {
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
        if let Stmt::Contribute {
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

/// Gives each branch whose current is probed and whose flow carries noise its current as
/// an unknown, so that what reads the current reads its noise too, as PSP's drain does
/// its induced gate noise's: a node of the branch's name, of a current's nature, whose
/// equation is its value less the branch's flow, the noise entering it, and through which
/// the branch's flow then passes. Returns whether it gave any.
///
/// The noise enters the equation as the branch's flow does, with the opposite sign, which
/// changes no response's power or correlation: a noise source and its negative are the
/// same random process.
fn noisy_currents(module: &mut Module) -> Result<bool, Error> {
    let rewritten = collect_noisy_branches(module)?;
    if rewritten.is_empty() {
        return Ok(false);
    }
    let negate = |expr: &mut Option<Expr>| {
        if let Some(value) = expr.take() {
            *expr = Some(Expr::Unary("-", Box::new(value)));
        }
    };
    visit_contributions_mut(&mut module.analog, &mut |statement| {
        if let Stmt::Contribute {
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
    let replacements: BTreeMap<usize, Expr> = rewritten
        .iter()
        .map(|(&branch, &(node, _))| (branch, Expr::Probe(Probe::Voltage(node, GROUND))))
        .collect();
    apply_replacements(module, &replacements);
    for (&branch, &(node, equation)) in &rewritten {
        let current = Expr::Probe(Probe::Voltage(node, GROUND));
        for target in [branch, equation] {
            module.analog.push(Stmt::Contribute {
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
fn apply_replacements(module: &mut Module, replacements: &BTreeMap<usize, Expr>) {
    let mut replace = |expr: &mut Expr| {
        expr.transform_mut(&mut |inner| {
            if let Expr::Probe(Probe::Current(branch)) = inner
                && let Some(current) = replacements.get(branch)
            {
                *inner = current.clone();
                return true;
            }
            false
        });
    };
    for statement in &mut module.analog {
        statement.exprs_mut(&mut replace);
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
    statements: &[Stmt],
    position: &mut usize,
    last: &mut [usize],
    visit: &mut impl FnMut(usize, &Expr),
) {
    for statement in statements {
        *position += 1;
        let at = *position;
        match statement {
            Stmt::Assign(variable, value) => {
                visit(at, value);
                last[*variable] = at;
            }
            Stmt::If(_, condition, then, otherwise) => {
                visit(at, condition);
                order(then, position, last, visit);
                order(otherwise, position, last, visit);
            }
            Stmt::While(condition, body) => {
                visit(at, condition);
                order(body, position, last, visit);
            }
            Stmt::Contribute { .. } | Stmt::Fail(..) | Stmt::Warn(..) => {
                statement.walk(&mut |expr| visit(at, expr));
            }
        }
    }
}
