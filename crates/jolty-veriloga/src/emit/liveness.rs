//! Mode-aware liveness analysis and dead-assignment pruning over the resolved IR.

use std::collections::BTreeSet;

use crate::ir::{Expression, Module, Probe, Statement};

use super::types::Mode;

/// Adds the variables `expr` reads, and those the arguments of its limited probes read,
/// to `found`.
pub(super) fn reads(expr: &Expression, module: &Module, found: &mut BTreeSet<usize>) {
    expr.walk(&mut |inner| match inner {
        Expression::Variable(variable) => {
            found.insert(*variable);
        }
        Expression::Probe(Probe::Limited(limit)) => {
            for argument in &module.limits[*limit].arguments {
                reads(argument, module, found);
            }
        }
        _ => {}
    });
}

/// Which variables `mode` keeps: those `roots` holds, and those a kept statement reads,
/// until none is added; in a dynamic mode only dynamic variables, in preparation only
/// static ones.
pub(super) fn live(
    module: &Module,
    statements: &[Statement],
    mode: Mode,
    roots: &BTreeSet<usize>,
) -> Vec<bool> {
    let wanted = |variable: usize| mode.prepares() != module.dynamic[variable];
    let mut live: Vec<bool> = (0..module.variables.len())
        .map(|variable| roots.contains(&variable) && wanted(variable))
        .collect();
    loop {
        let mut found = BTreeSet::new();
        mark(statements, module, mode, &live, &mut found);
        let mut changed = false;
        for variable in found {
            if wanted(variable) && !live[variable] {
                live[variable] = true;
                changed = true;
            }
        }
        if !changed {
            return live;
        }
    }
}

/// `statements` as `mode` runs them, each assignment that nothing after it reads left
/// out, `live` the variables read after them, which it leaves those read before: a
/// backward pass over the structured code, a loop's body to its fixed point. An
/// assignment of a variable to itself does nothing, and one that limits a voltage is
/// kept for the memory it writes.
pub(super) fn prune(
    statements: &[Statement],
    module: &Module,
    mode: Mode,
    live: &mut BTreeSet<usize>,
) -> Vec<Statement> {
    let wanted = |variable: usize| mode.prepares() != module.dynamic[variable];
    let limits = |expr: &Expression| {
        expr.any(&|inner| matches!(inner, Expression::Probe(Probe::Limited(_))))
    };
    let mut kept_backward = Vec::new();
    for statement in statements.iter().rev() {
        match statement {
            Statement::Assign(variable, value) => {
                let itself = matches!(value, Expression::Variable(read) if read == variable);
                if !wanted(*variable) || itself || (!live.contains(variable) && !limits(value)) {
                    continue;
                }
                live.remove(variable);
                reads(value, module, live);
                kept_backward.push(statement.clone());
            }
            Statement::If(id, condition, then, otherwise) => {
                let (mut first, mut second) = (live.clone(), live.clone());
                let then = prune(then, module, mode, &mut first);
                let otherwise = prune(otherwise, module, mode, &mut second);
                if then.is_empty() && otherwise.is_empty() && !limits(condition) {
                    continue;
                }
                first.extend(second);
                reads(condition, module, &mut first);
                *live = first;
                kept_backward.push(Statement::If(*id, condition.clone(), then, otherwise));
            }
            Statement::While(condition, body) => {
                let mut entry = live.clone();
                reads(condition, module, &mut entry);
                loop {
                    let mut inner = entry.clone();
                    prune(body, module, mode, &mut inner);
                    let mut next = live.clone();
                    next.extend(inner);
                    reads(condition, module, &mut next);
                    if next == entry {
                        break;
                    }
                    entry = next;
                }
                let mut inner = entry.clone();
                let body = prune(body, module, mode, &mut inner);
                if body.is_empty() {
                    continue;
                }
                *live = entry;
                kept_backward.push(Statement::While(condition.clone(), body));
            }
            Statement::Contribute { .. } | Statement::Fail(..) | Statement::Warn(..) => {
                if kept(statement, module, mode, &[]) {
                    let mut found = BTreeSet::new();
                    mark(
                        std::slice::from_ref(statement),
                        module,
                        mode,
                        &[],
                        &mut found,
                    );
                    live.extend(found);
                    kept_backward.push(statement.clone());
                }
            }
        }
    }
    kept_backward.reverse();
    kept_backward
}

/// Adds what the statements of `statements` that `mode` keeps read to `found`.
pub(super) fn mark(
    statements: &[Statement],
    module: &Module,
    mode: Mode,
    live: &[bool],
    found: &mut BTreeSet<usize>,
) {
    for statement in statements {
        if !kept(statement, module, mode, live) {
            continue;
        }
        match statement {
            Statement::Assign(_, value) => reads(value, module, found),
            Statement::Contribute {
                resistive,
                reactive,
                noise,
                ..
            } => {
                if mode == Mode::Noise {
                    for source in noise {
                        reads(&source.power, module, found);
                        if let Some(exponent) = &source.exponent {
                            reads(exponent, module, found);
                        }
                    }
                } else {
                    for expr in resistive.iter().chain(reactive) {
                        reads(expr, module, found);
                    }
                }
            }
            Statement::If(_, condition, then, otherwise) => {
                reads(condition, module, found);
                mark(then, module, mode, live, found);
                mark(otherwise, module, mode, live, found);
            }
            Statement::While(condition, body) => {
                reads(condition, module, found);
                mark(body, module, mode, live, found);
            }
            Statement::Fail(_, arguments) | Statement::Warn(_, arguments, _) => {
                for expr in arguments {
                    reads(expr, module, found);
                }
            }
        }
    }
}

/// Whether `mode` keeps `statement`.
pub(super) fn kept(statement: &Statement, module: &Module, mode: Mode, live: &[bool]) -> bool {
    statement.any(&|inner| match inner {
        Statement::Assign(variable, _) => match mode {
            Mode::Function => true,
            Mode::Prepare | Mode::Warnings => !module.dynamic[*variable] && live[*variable],
            Mode::Evaluate | Mode::Noise => module.dynamic[*variable] && live[*variable],
        },
        Statement::Contribute {
            resistive,
            reactive,
            noise,
            ..
        } => match mode {
            Mode::Evaluate => resistive.is_some() || reactive.is_some(),
            Mode::Noise => !noise.is_empty(),
            Mode::Prepare | Mode::Warnings | Mode::Function => false,
        },
        Statement::Fail(..) => mode.prepares(),
        Statement::Warn(_, _, dynamic) => {
            if *dynamic {
                mode == Mode::Evaluate
            } else {
                mode == Mode::Warnings
            }
        }
        Statement::If(..) | Statement::While(..) => false,
    })
}
