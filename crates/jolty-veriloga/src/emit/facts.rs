//! Module-level structural facts for code emission: shape conditions, branch roles and
//! collapses, dynamic warnings, DC connectivity, and emitter-level validity checks.

use std::collections::{BTreeMap, BTreeSet};

use jolty_models::Nature;

use crate::Error;
use crate::ir::{Branch, Expression, GROUND, Module, Statement, constant, probed_currents};

use super::types::{Activity, Place, Site};

/// A module's facts the emitter reads throughout.
pub(super) struct Facts<'m> {
    pub(super) module: &'m Module,
    /// The conditionals whose conditions are static and that hold a contribution to a
    /// branch's flow or potential, by number, with their index among `conditions`.
    pub(super) shaped: BTreeMap<usize, usize>,
    /// The distinct static conditions that shape the kernel, in order of first appearance.
    pub(super) conditions: Vec<Expression>,
    /// Whether each branch adds to the equations only under some of the shape's
    /// conditions, with its index among the shape's flags, for its flow or potential and
    /// for its collapse.
    pub(super) flags: BTreeMap<(usize, Site), usize>,
    /// The message of each warning that depends on the probes, which the memory flags
    /// once an evaluation finds it, and the number of values it formats, which the
    /// memory holds as the last evaluation that found it read them.
    pub(super) warned: Vec<(String, usize)>,
    /// The branches whose currents are read.
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

/// Whether `statement` is a static `$warning` or its like, or holds one.
pub(super) fn warns(statement: &Statement) -> bool {
    statement.any(&|inner| matches!(inner, Statement::Warn(_, _, false)))
}

/// Whether `statement` adds to a branch's flow or potential, or holds one that does.
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

/// Whether `statement` adds a noise source, or holds one that does.
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

    /// Whether a contribution collapses its branch: a potential of zero across a branch
    /// with an internal node.
    pub(super) fn collapses(
        &self,
        branch: usize,
        potential: bool,
        resistive: Option<&Expression>,
        reactive: Option<&Expression>,
    ) -> bool {
        let Branch { plus, minus, .. } = self.module.branches[branch];
        // The node that collapses: the later of two, or the one node of a branch to the
        // ground, which the ground ranks before. A branch whose current is read keeps it
        // as an unknown instead.
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

    /// What a contribution to `branch` makes it: a collapse, a potential or a flow.
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

    /// Finds each branch's role and which branches the shape switches, `shaped` saying
    /// whether `statements` lie under a shape's condition.
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
        // An unconditional site makes its branch always active: no flag.
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

    /// Whether `branch`'s flow, its potential or its collapse is active: never, always,
    /// or as the shape's flag says.
    pub(super) fn flag(&self, branch: usize, site: Site) -> Activity {
        match self.flags.get(&(branch, site)) {
            None => Activity::Never,
            Some(&usize::MAX) => Activity::Always,
            Some(&index) => Activity::Flag(index),
        }
    }

    /// Whether `branch` holds a potential in some shape, its current then an unknown.
    pub(super) fn holds_potential(&self, branch: usize) -> bool {
        self.flags.contains_key(&(branch, Site::Potential))
    }

    /// Whether `branch` carries a flow in some shape.
    pub(super) fn carries_flow(&self, branch: usize) -> bool {
        self.flags.contains_key(&(branch, Site::Flow))
    }

    pub(super) fn flag_count(&self) -> usize {
        self.flags
            .values()
            .filter(|&&index| index != usize::MAX)
            .count()
    }

    /// The local index of node `node`, as code that reads it from `holder`, the shape or
    /// the layout's own bindings.
    pub(super) fn node(&self, node: usize, holder: &str) -> String {
        if node == GROUND {
            "NOWHERE".to_owned()
        } else if node < self.module.ports {
            node.to_string()
        } else {
            format!("{holder}{}", self.module.nodes[node])
        }
    }

    /// The local index of `place`, as code that reads it from `holder`.
    pub(super) fn place(&self, place: Place, holder: &str) -> String {
        match place {
            Place::Node(node) => self.node(node, holder),
            Place::Current(branch) => format!("{holder}{}", self.module.branches[branch].name),
        }
    }

    /// The name of the shape's field of the entry at `row` and `column`.
    pub(super) fn entry_name(&self, (row, column): (Place, Place)) -> String {
        let name = |place: Place| match place {
            Place::Node(node) => self.module.node_name(node).to_owned(),
            Place::Current(branch) => self.module.branches[branch].name.clone(),
        };
        format!("{}_by_{}", name(row), name(column))
    }

    /// Whether the kernel remembers anything: limited voltages, output variables, or
    /// dynamic warnings.
    pub(super) fn remembers(&self) -> bool {
        !self.module.limits.is_empty() || !self.module.outputs.is_empty() || !self.warned.is_empty()
    }

    /// Whether `branch` has a resistive part, so that it conducts at DC.
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

    /// Whether terminals `from` and `to` are joined through internal nodes alone, by
    /// edges that conduct at DC when `conducting` says so.
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

    /// The branches that may collapse in some shape, each with its flag where the shape
    /// decides it.
    pub(super) fn shorts(&self) -> Vec<(usize, Option<usize>)> {
        (0..self.module.branches.len())
            .filter_map(|branch| match self.flag(branch, Site::Collapse) {
                Activity::Never => None,
                Activity::Always => Some((branch, None)),
                Activity::Flag(flag) => Some((branch, Some(flag))),
            })
            .collect()
    }

    /// Whether the internal node `node` collapses into the ground in some shape, through
    /// the shorts that may join it to the ground.
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

/// Checks that no contribution reads a `ddx`, whose own derivative would be a second
/// one, which the dual numbers do not carry: the variables a `ddx` reaches are found by
/// its assignments, until none is added.
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
