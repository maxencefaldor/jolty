//! Module-level structural facts for code emission: shape conditions, branch roles and
//! collapses, dynamic warnings, DC connectivity, and emitter-level validity checks.

use std::collections::{BTreeMap, BTreeSet};

use jolty_models::Nature;

use crate::Error;
use crate::ir::{Branch, Expression, GROUND, Module, Statement, constant, probed_currents};

use super::types::{Activity, Place, Site};

/// Precomputed module facts shared across all code-emission passes.
pub(super) struct Facts<'m> {
    pub(super) module: &'m Module,
    /// Map from static shape-determining conditional ID to its deduplicated index in `conditions`.
    pub(super) shaped: BTreeMap<usize, usize>,
    /// Distinct static shape conditions in order of first appearance.
    pub(super) conditions: Vec<Expression>,
    /// Map from conditionally active `(branch, site)` pairs to their index in `shape.active`
    /// (or `usize::MAX` when always active).
    pub(super) flags: BTreeMap<(usize, Site), usize>,
    /// Probe-dependent warning messages paired with their formatted argument count.
    pub(super) warned: Vec<(String, usize)>,
    /// Branch indices whose currents are probed.
    pub(super) probed: BTreeSet<usize>,
}

/// Adds the message of each warning of `statements` that depends on the probes to
/// `warned`, once, with the number of values it formats.
fn collect_warned(statements: &[Statement], warned: &mut Vec<(String, usize)>) {
    Statement::visit_stmts(statements, &mut |statement| {
        if let Statement::Warn(message, arguments, true) = statement {
            let found = (message.clone(), arguments.len());
            if !warned.contains(&found) {
                warned.push(found);
            }
        }
    });
}

/// Returns `true` when `statement` is or contains a static `$warning` or related diagnostic.
pub(super) fn warns(statement: &Statement) -> bool {
    statement.any(&|inner| matches!(inner, Statement::Warn(_, _, false)))
}

/// Returns `true` when `statement` contributes a resistive or reactive term to a branch.
pub(super) fn stamps(statement: &Statement) -> bool {
    statement.any(&|inner| {
        matches!(
            inner,
            Statement::Contribute {
                resistive: Some(_),
                ..
            } | Statement::Contribute {
                reactive: Some(_),
                ..
            }
        )
    })
}

/// Returns `true` when `statement` is or contains a noise contribution.
pub(super) fn makes_noise(statement: &Statement) -> bool {
    statement
        .any(&|inner| matches!(inner, Statement::Contribute { noise, .. } if !noise.is_empty()))
}

impl<'m> Facts<'m> {
    pub(super) fn new(module: &'m Module) -> Result<Self, Error> {
        let mut facts = Self {
            module,
            shaped: BTreeMap::new(),
            conditions: Vec::new(),
            flags: BTreeMap::new(),
            warned: Vec::new(),
            probed: probed_currents(module),
        };
        collect_warned(&module.analog, &mut facts.warned);
        facts.find_shapes(&module.analog);
        facts.find_roles(&module.analog, false)?;
        static_tasks(&module.analog, module, false)?;
        first_order(module)?;
        Ok(facts)
    }

    /// Numbers the conditionals of `statements` that shape the kernel, sharing an index
    /// among identical condition expressions.
    fn find_shapes(&mut self, statements: &[Statement]) {
        Statement::visit_stmts(statements, &mut |statement| {
            if let Statement::If(id, condition, then, otherwise) = statement
                && then.iter().chain(otherwise).any(stamps)
                && !self.module.is_dynamic(condition)
            {
                let index = self
                    .conditions
                    .iter()
                    .position(|existing| existing == condition)
                    .unwrap_or_else(|| {
                        let index = self.conditions.len();
                        self.conditions.push(condition.clone());
                        index
                    });
                self.shaped.insert(*id, index);
            }
        });
    }

    /// Returns `true` when a zero-potential contribution collapses an unprobed internal branch
    /// node.
    pub(super) fn collapses(
        &self,
        branch: usize,
        potential: bool,
        resistive: Option<&Expression>,
        reactive: Option<&Expression>,
    ) -> bool {
        let Branch { plus, minus, .. } = self.module.branches[branch];
        // Collapse the higher-indexed node (or the sole non-ground node). Branches whose
        // currents are probed keep their branch-current unknown instead.
        let collapsing = if minus == GROUND {
            plus
        } else {
            plus.max(minus)
        };
        potential
            && reactive.is_none()
            && resistive.and_then(constant) == Some(0.0)
            && collapsing >= self.module.ports
            && !self.probed.contains(&branch)
    }

    /// Classifies a contribution to `branch` as [`Site::Collapse`], [`Site::Potential`], or
    /// [`Site::Flow`].
    pub(super) fn site(
        &self,
        branch: usize,
        potential: bool,
        resistive: Option<&Expression>,
        reactive: Option<&Expression>,
    ) -> Site {
        if self.collapses(branch, potential, resistive, reactive) {
            Site::Collapse
        } else if potential {
            Site::Potential
        } else {
            Site::Flow
        }
    }

    /// Classifies every branch role and marks conditionally active branches under shape
    /// predicates.
    fn find_roles(&mut self, statements: &[Statement], shaped: bool) -> Result<(), Error> {
        for statement in statements {
            match statement {
                Statement::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    ..
                } if resistive.is_some() || reactive.is_some() => {
                    let site =
                        self.site(*branch, *potential, resistive.as_ref(), reactive.as_ref());
                    let Branch { plus, ref name, .. } = self.module.branches[*branch];
                    if site == Site::Potential && self.module.natures[plus] == Nature::Temperature {
                        return Err(Error::at(
                            0,
                            format!(
                                "the thermal branch {name} holds a temperature, which is not compiled yet"
                            ),
                        ));
                    }
                    let key = (*branch, site);
                    if shaped {
                        let next = self.flags.len();
                        self.flags.entry(key).or_insert(next);
                    } else {
                        self.flags.insert(key, usize::MAX);
                    }
                }
                Statement::If(id, _, then, otherwise) => {
                    let inner = shaped || self.shaped.contains_key(id);
                    self.find_roles(then, inner)?;
                    self.find_roles(otherwise, inner)?;
                }
                Statement::While(_, body) => self.find_roles(body, shaped)?,
                Statement::Assign(..)
                | Statement::Contribute { .. }
                | Statement::Fail(..)
                | Statement::Warn(..) => {}
            }
        }
        // Unconditional sites remain always active and need no flag slot.
        if !shaped {
            let mut index = 0;
            for value in self.flags.values_mut() {
                if *value != usize::MAX {
                    *value = index;
                    index += 1;
                }
            }
        }
        Ok(())
    }

    /// Returns the [`Activity`] state (`Never`, `Always`, or `Flag`) of `(branch, site)`.
    pub(super) fn flag(&self, branch: usize, site: Site) -> Activity {
        match self.flags.get(&(branch, site)) {
            None => Activity::Never,
            Some(&usize::MAX) => Activity::Always,
            Some(&index) => Activity::Flag(index),
        }
    }

    /// Returns `true` when `branch` carries a potential contribution in at least one shape.
    pub(super) fn holds_potential(&self, branch: usize) -> bool {
        self.flags.contains_key(&(branch, Site::Potential))
    }

    /// Returns `true` when `branch` carries a flow contribution in at least one shape.
    pub(super) fn carries_flow(&self, branch: usize) -> bool {
        self.flags.contains_key(&(branch, Site::Flow))
    }

    pub(super) fn flag_count(&self) -> usize {
        self.flags
            .values()
            .filter(|&&index| index != usize::MAX)
            .count()
    }

    /// Formats the local index expression for `node` prefixed by `holder` (for internal nodes).
    pub(super) fn node(&self, node: usize, holder: &str) -> String {
        if node == GROUND {
            "NOWHERE".to_owned()
        } else if node < self.module.ports {
            node.to_string()
        } else {
            format!("{holder}{}", self.module.nodes[node])
        }
    }

    /// Formats the local index expression for `place` prefixed by `holder`.
    pub(super) fn place(&self, place: Place, holder: &str) -> String {
        match place {
            Place::Node(node) => self.node(node, holder),
            Place::Current(branch) => format!("{holder}{}", self.module.branches[branch].name),
        }
    }

    /// Returns the `Shape` field name for the Jacobian entry at `(row, column)`.
    pub(super) fn entry_name(&self, (row, column): (Place, Place)) -> String {
        let name = |place: Place| match place {
            Place::Node(node) => self.module.node_name(node).to_owned(),
            Place::Current(branch) => self.module.branches[branch].name.clone(),
        };
        format!("{}_by_{}", name(row), name(column))
    }

    /// Returns `true` when the kernel requires persistent `Memory` for limits, outputs, or
    /// dynamic warnings.
    pub(super) fn remembers(&self) -> bool {
        !self.module.limits.is_empty() || !self.module.outputs.is_empty() || !self.warned.is_empty()
    }

    /// Returns `true` when `branch` has at least one resistive contribution (conducting at DC).
    pub(super) fn conducts(&self, branch: usize) -> bool {
        Statement::any_in(&self.module.analog, &|statement| {
            matches!(
                statement,
                Statement::Contribute {
                    branch: to,
                    resistive: Some(_),
                    ..
                } if *to == branch
            )
        })
    }

    /// Returns `true` when terminals `from` and `to` connect through internal nodes via `edges`,
    /// restricted to DC-conducting edges when `conducting` is `true`.
    pub(super) fn joined(
        edges: &[(usize, usize, bool)],
        (from, to): (usize, usize),
        ports: usize,
        conducting: bool,
    ) -> bool {
        let mut reached = BTreeSet::from([from]);
        let mut frontier = vec![from];
        while let Some(node) = frontier.pop() {
            for &(plus, minus, conducts) in edges {
                if conducting && !conducts {
                    continue;
                }
                let other = if plus == node {
                    minus
                } else if minus == node {
                    plus
                } else {
                    continue;
                };
                if other == to {
                    return true;
                }
                if other >= ports && reached.insert(other) {
                    frontier.push(other);
                }
            }
        }
        false
    }

    /// Lists all branches that may collapse, paired with their optional `shape.active` flag
    /// index.
    pub(super) fn shorts(&self) -> Vec<(usize, Option<usize>)> {
        (0..self.module.branches.len())
            .filter_map(|branch| match self.flag(branch, Site::Collapse) {
                Activity::Never => None,
                Activity::Always => Some((branch, None)),
                Activity::Flag(flag) => Some((branch, Some(flag))),
            })
            .collect()
    }

    /// Returns `true` when internal `node` can collapse to [`GROUND`] in some kernel shape.
    pub(super) fn grounded(&self, node: usize) -> bool {
        let shorts: Vec<(usize, usize)> = self
            .shorts()
            .into_iter()
            .map(|(branch, _)| {
                let Branch { plus, minus, .. } = self.module.branches[branch];
                (plus, minus)
            })
            .collect();
        let mut reached = BTreeSet::from([node]);
        let mut grown = true;
        while grown {
            grown = false;
            for &(plus, minus) in &shorts {
                if reached.contains(&plus) != reached.contains(&minus) {
                    reached.extend([plus, minus]);
                    grown = true;
                }
            }
        }
        node >= self.module.ports && reached.contains(&GROUND)
    }
}

/// Checks that no contribution reads a `ddx` expression, since dual numbers do not track
/// second derivatives. Propagates `ddx` dependence through assignments to a fixed point.
fn first_order(module: &Module) -> Result<(), Error> {
    let mut reached = vec![false; module.variables.len()];
    let reads_ddx = |expr: &Expression, reached: &[bool]| {
        expr.any(&|inner| {
            matches!(inner, Expression::Ddx(..))
                || matches!(inner, Expression::Variable(variable) if reached[*variable])
        })
    };
    loop {
        let mut changed = false;
        Statement::visit_stmts(&module.analog, &mut |statement| {
            if let Statement::Assign(variable, value) = statement
                && !reached[*variable]
                && reads_ddx(value, &reached)
            {
                reached[*variable] = true;
                changed = true;
            }
        });
        if !changed {
            break;
        }
    }
    let refused = Statement::any_in(&module.analog, &|statement| {
        if !matches!(statement, Statement::Contribute { .. }) {
            return false;
        }
        let mut found = false;
        statement.walk(&mut |expr| found |= reads_ddx(expr, &reached));
        found
    });
    if refused {
        return Err(Error::at(
            0,
            "a contribution that reads a ddx needs second derivatives, which are not compiled yet"
                .into(),
        ));
    }
    Ok(())
}

/// Checks that each `$error` and static `$warning` of `statements`, under a condition
/// that is dynamic when `dynamic` is, depends on no probe, so that preparation states it.
fn static_tasks(statements: &[Statement], module: &Module, dynamic: bool) -> Result<(), Error> {
    for statement in statements {
        match statement {
            Statement::Fail(message, arguments) | Statement::Warn(message, arguments, false) => {
                if dynamic || arguments.iter().any(|argument| module.is_dynamic(argument)) {
                    return Err(Error::at(
                        0,
                        format!(
                            "the task {message:?} depends on the probes, which is not compiled yet"
                        ),
                    ));
                }
            }
            Statement::If(_, condition, then, otherwise) => {
                let dynamic = dynamic || module.is_dynamic(condition);
                static_tasks(then, module, dynamic)?;
                static_tasks(otherwise, module, dynamic)?;
            }
            Statement::While(condition, body) => {
                static_tasks(body, module, dynamic || module.is_dynamic(condition))?;
            }
            Statement::Warn(_, _, true) | Statement::Assign(..) | Statement::Contribute { .. } => {}
        }
    }
    Ok(())
}
