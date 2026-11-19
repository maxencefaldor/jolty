//! Statement lowering, branch stamping, noise emission, linearity expression synthesis,
//! method body assembly (`evaluate` / `noise`), and structured Rust statement printing.

use std::collections::{BTreeMap, BTreeSet};

use crate::Error;
use crate::ir::{Branch, Expr, GROUND, Noise, Probe, Stmt, visit_contributions};
use crate::linear::{Form, Linearity};

use super::decl::{format_message, write_locals_struct};
use super::expr::Emitter;
use super::facts::stamps;
use super::liveness::kept;
use super::types::{
    AND, ATOM, Activity, COMPARE, CodeWriter, Kind, MethodBody, Mode, Names, OR, Place,
    Requirement, Rust, SUM, Site, UNARY, mentions, parenthesize,
};

impl Emitter<'_, '_> {
    #[expect(
        clippy::too_many_lines,
        reason = "one arm per construct of the language, which read best side by side"
    )]
    pub(super) fn statements(
        &mut self,
        statements: &[Stmt],
        out: &mut Vec<Rust>,
    ) -> Result<(), Error> {
        let module = self.facts.module;
        for statement in statements {
            if !kept(statement, module, self.mode, &self.live) {
                continue;
            }
            match statement {
                Stmt::Assign(variable, value) => {
                    if self
                        .function
                        .is_some_and(|function| *variable < module.functions[function].inputs)
                    {
                        return Err(Error::at(0, "an analog function assigns its input".into()));
                    }
                    let kind = self.variable_kind(*variable);
                    let name = match self.function {
                        Some(function) => module.functions[function].variables[*variable].0.clone(),
                        None => module.variables[*variable].0.clone(),
                    };
                    // `x = x + e` adds in place, where the kind does: the same sum.
                    if let Expr::Binary(operator @ ("+" | "-"), left, right) = value
                        && matches!(**left, Expr::Variable(read) if read == *variable)
                        && matches!(kind, Kind::Real | Kind::Float(_))
                    {
                        let right = self.expression(right, out)?;
                        let right = self.convert(right, &kind, out)?;
                        out.push(Rust::Line(format!("v.{name} {operator}= {};", right.code)));
                        continue;
                    }
                    let operand = self.expression(value, out)?;
                    let operand = self.convert(operand, &kind, out)?;
                    out.push(Rust::Line(format!("v.{name} = {};", operand.code)));
                }
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    noise,
                    ..
                } => match self.mode {
                    Mode::Evaluate => {
                        let site = self.facts.site(
                            *branch,
                            *potential,
                            resistive.as_ref(),
                            reactive.as_ref(),
                        );
                        if site == Site::Collapse {
                            continue;
                        }
                        for (part, charge) in [(resistive, false), (reactive, true)] {
                            if let Some(part) = part {
                                self.accumulate((*branch, site), charge, part, out)?;
                            }
                        }
                    }
                    Mode::Noise => {
                        for source in noise {
                            self.noise_source((*branch, *potential), source, out)?;
                        }
                    }
                    Mode::Prepare | Mode::Warnings | Mode::Function => {}
                },
                Stmt::If(id, condition, then, otherwise) => {
                    let dynamic = module.is_dynamic(condition);
                    if self.mode.prepares() && dynamic {
                        self.statements(then, out)?;
                        self.statements(otherwise, out)?;
                        continue;
                    }
                    let condition = if matches!(self.mode, Mode::Evaluate | Mode::Noise) && !dynamic
                    {
                        if let Some(index) = self.facts.shaped.get(id) {
                            self.shape.insert("conditions".to_owned());
                            (format!("shape.conditions[{index}]"), ATOM)
                        } else {
                            let operand = self.static_operand(condition);
                            self.truth(operand)
                        }
                    } else {
                        let operand = self.expression(condition, out)?;
                        self.truth(operand)
                    };
                    let (mut first, mut second) = (Vec::new(), Vec::new());
                    self.statements(then, &mut first)?;
                    self.statements(otherwise, &mut second)?;
                    out.push(Rust::If(condition, first, second));
                }
                Stmt::While(condition, body) => {
                    if self.mode.prepares() && module.is_dynamic(condition) {
                        self.statements(body, out)?;
                        continue;
                    }
                    let mut before = Vec::new();
                    let operand = self.expression(condition, &mut before)?;
                    if !before.is_empty() {
                        return Err(Error::at(
                            0,
                            "a loop condition this complex is not compiled yet".into(),
                        ));
                    }
                    let (condition, _) = self.truth(operand);
                    let mut inner = Vec::new();
                    self.statements(body, &mut inner)?;
                    out.push(Rust::While(condition, inner));
                }
                Stmt::Fail(message, arguments) => {
                    let message = self.message(message, arguments, out)?;
                    out.push(Rust::Line(format!(
                        "return Err(ParameterError({message}));"
                    )));
                }
                Stmt::Warn(message, arguments, false) => {
                    let message = self.message(message, arguments, out)?;
                    out.push(Rust::Line(format!("found.push({message});")));
                }
                Stmt::Warn(message, arguments, true) => {
                    let index = self
                        .facts
                        .warned
                        .iter()
                        .position(|(warned, count)| warned == message && *count == arguments.len())
                        .unwrap_or_default();
                    out.push(Rust::Line(format!("memory.warned[{index}] = true;")));
                    let offset: usize = self.facts.warned[..index]
                        .iter()
                        .map(|(_, count)| count)
                        .sum();
                    for (place, argument) in arguments.iter().enumerate() {
                        let operand = self.expression(argument, out)?;
                        let value = Self::value_of(&operand).code;
                        out.push(Rust::Line(format!(
                            "memory.warned_values[{}] = {value};",
                            offset + place
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// The message of a task, `message` with each of its `%g`s and their like the value
    /// of the argument of `arguments` at its place, as code that makes the string.
    fn message(
        &mut self,
        message: &str,
        arguments: &[Expr],
        out: &mut Vec<Rust>,
    ) -> Result<String, Error> {
        let mut values = Vec::new();
        for argument in arguments {
            let operand = self.expression(argument, out)?;
            values.push(Self::value_of(&operand).code);
        }
        Ok(format_message(message, &values))
    }

    /// The field of the accumulator of `branch`'s flow or potential, or of its charge or
    /// flux.
    fn accumulator(&self, branch: usize, site: Site, charge: bool) -> String {
        let suffix = match (site, charge) {
            (Site::Potential, false) => "potential",
            (Site::Potential, true) => "flux",
            (_, false) => "flow",
            (_, true) => "charge",
        };
        format!("{}_{suffix}", self.module().branches[branch].name)
    }

    fn accumulate(
        &mut self,
        (branch, site): (usize, Site),
        charge: bool,
        part: &Expr,
        out: &mut Vec<Rust>,
    ) -> Result<(), Error> {
        let kind = self
            .accumulators
            .get(&(branch, site, charge))
            .cloned()
            .unwrap_or(Kind::Real);
        let operand = self.expression(part, out)?;
        let operand = self.convert(operand, &kind, out)?;
        let field = self.accumulator(branch, site, charge);
        let line = if kind == Kind::Real {
            format!("v.{field} += {};", operand.code)
        } else {
            format!("v.{field} = v.{field} + {};", operand.at(SUM + 1))
        };
        out.push(Rust::Line(line));
        Ok(())
    }

    fn noise_source(
        &mut self,
        (branch, potential): (usize, bool),
        source: &Noise,
        out: &mut Vec<Rust>,
    ) -> Result<(), Error> {
        let module = self.facts.module;
        let operand = self.expression(&source.power, out)?;
        let power = Self::value_of(&operand).code;
        let Branch {
            plus,
            minus,
            ref name,
        } = module.branches[branch];
        let equations = if potential {
            self.shape.insert(name.clone());
            format!("[Some(UnknownIndex::new(shape.{name})), None]")
        } else {
            for node in [plus, minus] {
                if module.is_internal(node) {
                    self.shape.insert(module.nodes[node].clone());
                }
            }
            let equation = |node: usize| {
                if node == GROUND {
                    "None".to_owned()
                } else {
                    format!(
                        "Some(UnknownIndex::new({}))",
                        self.facts.node(node, "shape.")
                    )
                }
            };
            format!("[{}, {}]", equation(plus), equation(minus))
        };
        let line = match &source.exponent {
            None => format!(
                "sources.push(NoiseSource::white({:?}, {equations}, {power}));",
                source.name
            ),
            Some(exponent) => {
                if module.is_dynamic(exponent) {
                    return Err(Error::at(
                        0,
                        "a flicker exponent that depends on the probes is not compiled yet".into(),
                    ));
                }
                let exponent = self.expression(exponent, out)?;
                format!(
                    "sources.push(NoiseSource {{\n    name: {:?},\n    equations: {equations},\n    white: 0.0,\n    flicker: {power},\n    exponent: {},\n}});",
                    source.name,
                    Self::value_of(&exponent).code
                )
            }
        };
        out.push(Rust::Line(line));
        Ok(())
    }

    /// The kinds of the accumulators, from the probes each contribution depends on.
    fn find_accumulators(&mut self) {
        let module = self.facts.module;
        let mut probes: BTreeMap<(usize, Site, bool), BTreeSet<Probe>> = BTreeMap::new();
        visit_contributions(&module.analog, &mut |statement| {
            if let Stmt::Contribute {
                branch,
                potential,
                resistive,
                reactive,
                ..
            } = statement
                && self
                    .facts
                    .site(*branch, *potential, resistive.as_ref(), reactive.as_ref())
                    != Site::Collapse
            {
                let site = if *potential {
                    Site::Potential
                } else {
                    Site::Flow
                };
                for (part, charge) in [(resistive, false), (reactive, true)] {
                    if let Some(part) = part {
                        probes
                            .entry((*branch, site, charge))
                            .or_default()
                            .extend(module.probes_of(part));
                    }
                }
            }
        });
        for (key, probes) in probes {
            for probe in &probes {
                if let Probe::Limited(index) = probe {
                    self.extended.insert(*index);
                }
            }
            let kind = if probes.is_empty() {
                Kind::Real
            } else {
                Kind::Dual(probes.into_iter().collect())
            };
            self.accumulators.insert(key, kind);
        }
    }

    /// The local index of `place` in the evaluation, its use recorded.
    fn index(&mut self, place: Place) -> String {
        let module = self.facts.module;
        match place {
            Place::Node(node) if module.is_internal(node) => {
                self.shape.insert(module.nodes[node].clone());
            }
            Place::Current(branch) => {
                self.shape.insert(module.branches[branch].name.clone());
            }
            Place::Node(_) => {}
        }
        self.facts.place(place, "shape.")
    }

    /// Adds `value`, the derivative of the rows' equations by the probe `probe`, to the
    /// stamp's entries.
    fn derivative(
        &mut self,
        rows: &[(Place, &str)],
        probe: Probe,
        value: &str,
        user: (usize, Site),
        capacitance: bool,
        out: &mut Vec<Rust>,
    ) {
        let module = self.facts.module;
        let columns: Vec<(Place, bool)> = match probe {
            Probe::Voltage(plus, minus) => {
                vec![(Place::Node(plus), true), (Place::Node(minus), false)]
            }
            Probe::Limited(index) => {
                let limit = &module.limits[index];
                vec![
                    (Place::Node(limit.plus), true),
                    (Place::Node(limit.minus), false),
                ]
            }
            Probe::Current(branch) => vec![(Place::Current(branch), true)],
        };
        let method = if capacitance {
            "add_capacitance"
        } else {
            "add_conductance"
        };
        let ground = Place::Node(GROUND);
        for &(row, sign) in rows {
            for &(column, positive) in &columns {
                // The ground's equation and value are no unknown's.
                if row == ground || column == ground {
                    continue;
                }
                self.entries.entry((row, column)).or_default().insert(user);
                let name = self.facts.entry_name((row, column));
                let negative = (sign == "-") == positive;
                let value = if negative {
                    format!("-{value}")
                } else {
                    value.to_owned()
                };
                out.push(Rust::Line(format!(
                    "stamp.{method}(shape.{name}, {value});"
                )));
            }
        }
        self.shape.insert("entries".to_owned());
    }

    /// The value of the accumulator `field`, of `kind`, extended linearly from its
    /// limited voltages to the applied ones.
    fn total(field: &str, kind: &Kind) -> String {
        match kind {
            Kind::Dual(probes)
                if probes
                    .iter()
                    .any(|probe| matches!(probe, Probe::Limited(_))) =>
            {
                let steps: Vec<String> = probes
                    .iter()
                    .map(|probe| match probe {
                        Probe::Limited(index) => format!("v.applied{index} - v.limited{index}"),
                        _ => "0.0".to_owned(),
                    })
                    .collect();
                format!("v.{field}.extend([{}])", steps.join(", "))
            }
            Kind::Dual(_) => format!("v.{field}.value"),
            _ => format!("v.{field}"),
        }
    }

    /// Destructures `v.{field}.gradient` and stamps each probe derivative across `rows`.
    fn stamp_gradient(
        &mut self,
        field: &str,
        kind: &Kind,
        rows: &[(Place, &str)],
        user: (usize, Site),
        charge: bool,
        lines: &mut Vec<Rust>,
    ) {
        if let Kind::Dual(probes) = kind {
            let names: Vec<String> = probes.iter().map(|_| self.temporary()).collect();
            lines.push(Rust::Line(format!(
                "let [{}] = v.{field}.gradient;",
                names.join(", ")
            )));
            for (probe, name) in probes.iter().zip(&names) {
                self.derivative(rows, *probe, name, user, charge, lines);
            }
        }
    }

    /// Stamps `+value` on `p` and `-value` on `n` (when `minus` is not ground) using `method`.
    fn stamp_node_pair(
        method: &str,
        (p, n): (&str, &str),
        minus: Place,
        value: &str,
        lines: &mut Vec<Rust>,
    ) {
        lines.push(Rust::Line(format!("stamp.{method}({p}, {value});")));
        if minus != Place::Node(GROUND) {
            lines.push(Rust::Line(format!("stamp.{method}({n}, -{value});")));
        }
    }

    /// Stamps each branch's accumulated contributions, once.
    fn stamps(&mut self, out: &mut Vec<Rust>) {
        let module = self.facts.module;
        for (index, branch) in module.branches.iter().enumerate() {
            for site in [Site::Potential, Site::Flow] {
                if self.facts.flag(index, site) == Activity::Never {
                    continue;
                }
                let mut lines = Vec::new();
                let (plus, minus) = (Place::Node(branch.plus), Place::Node(branch.minus));
                let (p, n) = (self.index(plus), self.index(minus));
                if site == Site::Potential {
                    self.stamp_potential(index, (plus, minus), (&p, &n), &mut lines);
                } else {
                    self.stamp_flow(index, (plus, minus), (&p, &n), &mut lines);
                }
                match self.facts.flag(index, site).index() {
                    Some(flag) => {
                        self.shape.insert("active".to_owned());
                        out.push(Rust::If(
                            (format!("shape.active[{flag}]"), ATOM),
                            lines,
                            Vec::new(),
                        ));
                    }
                    None => out.extend(lines),
                }
            }
        }
    }

    /// The stamp of the branch at `index`, which holds a potential between `places`, at
    /// the local indices `indices`, its current an unknown of its own.
    fn stamp_potential(
        &mut self,
        index: usize,
        (plus, minus): (Place, Place),
        (p, n): (&str, &str),
        lines: &mut Vec<Rust>,
    ) {
        let user = (index, Site::Potential);
        let own = Place::Current(index);
        let c = self.index(own);
        let ground = Place::Node(GROUND);
        lines.push(Rust::Line(format!("let current = at.unknown::<R>({c});")));
        Self::stamp_node_pair("add_residual", (p, n), minus, "current", lines);
        let stamp_units =
            |emitter: &mut Self, lines: &mut Vec<Rust>, pairs: [(Place, Place, &str); 2]| {
                for (row, column, value) in pairs {
                    if row != ground && column != ground {
                        emitter
                            .entries
                            .entry((row, column))
                            .or_default()
                            .insert(user);
                        lines.push(Rust::Line(format!(
                            "stamp.add_conductance(shape.{}, R::constant({value}));",
                            emitter.facts.entry_name((row, column))
                        )));
                    }
                }
            };
        stamp_units(self, lines, [(plus, own, "1.0"), (minus, own, "-1.0")]);
        let potential = self
            .accumulators
            .get(&(index, Site::Potential, false))
            .cloned();
        let total = potential.as_ref().map_or_else(
            || "R::constant(0.0)".to_owned(),
            |kind| Self::total(&self.accumulator(index, Site::Potential, false), kind),
        );
        lines.push(Rust::Line(format!(
            "stamp.add_residual({c}, at.unknown::<R>({p}) - at.unknown::<R>({n}) - {total});"
        )));
        stamp_units(self, lines, [(own, plus, "1.0"), (own, minus, "-1.0")]);
        for (charge, kind) in [
            (false, potential),
            (
                true,
                self.accumulators
                    .get(&(index, Site::Potential, true))
                    .cloned(),
            ),
        ] {
            let Some(kind) = kind else { continue };
            let field = self.accumulator(index, Site::Potential, charge);
            if charge {
                lines.push(Rust::Line(format!(
                    "stamp.add_charge({c}, -{});",
                    Self::total(&field, &kind)
                )));
            }
            self.stamp_gradient(&field, &kind, &[(own, "-")], user, charge, lines);
        }
    }

    /// The stamp of the branch at `index`, whose flow runs between `places`, at the local
    /// indices `indices`.
    fn stamp_flow(
        &mut self,
        index: usize,
        (plus, minus): (Place, Place),
        (p, n): (&str, &str),
        lines: &mut Vec<Rust>,
    ) {
        let user = (index, Site::Flow);
        for charge in [false, true] {
            let Some(kind) = self.accumulators.get(&(index, Site::Flow, charge)).cloned() else {
                continue;
            };
            let field = self.accumulator(index, Site::Flow, charge);
            let total = self.temporary();
            lines.push(Rust::Line(format!(
                "let {total} = {};",
                Self::total(&field, &kind)
            )));
            let method = if charge { "add_charge" } else { "add_residual" };
            Self::stamp_node_pair(method, (p, n), minus, &total, lines);
            self.stamp_gradient(
                &field,
                &kind,
                &[(plus, "+"), (minus, "-")],
                user,
                charge,
                lines,
            );
        }
    }

    /// The body of `linear`, unless no instance is: whether the contributions of an
    /// instance's shape are each zero or linear in the probes, the static numbers each
    /// requires zero. An instance whose memory keeps what depends on the probes, a warning
    /// found or an output variable, is not, as a linear one is evaluated once a run.
    pub(super) fn linear_body(&mut self) -> Option<String> {
        let module = self.facts.module;
        let remembers = !self.facts.warned.is_empty()
            || module.outputs.iter().any(|&output| module.dynamic[output]);
        let linearity = Linearity::new(module).filter(|_| !remembers)?;
        self.linear_requirement(&module.analog, &linearity).render()
    }

    /// What the contributions of `statements` require to be linear.
    fn linear_requirement(
        &mut self,
        statements: &[Stmt],
        linearity: &Linearity<'_>,
    ) -> Requirement {
        let mut all = Requirement::Always;
        for statement in statements {
            let requirement = match statement {
                Stmt::Contribute {
                    branch,
                    potential,
                    resistive,
                    reactive,
                    ..
                } if !self.facts.collapses(
                    *branch,
                    *potential,
                    resistive.as_ref(),
                    reactive.as_ref(),
                ) =>
                {
                    let parts: Vec<&Expr> = resistive.iter().chain(reactive).collect();
                    self.site_requirement(&parts, linearity)
                }
                Stmt::If(id, _, then, otherwise) => match self.facts.shaped.get(id) {
                    Some(&index) => {
                        let first = self.linear_requirement(then, linearity);
                        let second = self.linear_requirement(otherwise, linearity);
                        if !matches!(
                            (&first, &second),
                            (Requirement::Always, Requirement::Always)
                        ) {
                            self.shape.insert("conditions".to_owned());
                        }
                        Requirement::choose(&format!("shape.conditions[{index}]"), first, second)
                    }
                    None if stamps(statement) => Requirement::Never,
                    None => Requirement::Always,
                },
                Stmt::Contribute { .. }
                | Stmt::Assign(..)
                | Stmt::While(..)
                | Stmt::Fail(..)
                | Stmt::Warn(..) => Requirement::Always,
            };
            all = all.and(requirement);
        }
        all
    }

    /// What a contribution of `parts`, its resistive and reactive ones, requires to be
    /// linear: one of the ways each part is zero or linear, each requiring its static
    /// numbers zero. Since `linear` receives `&Prepared<f64>`, `Kind::Real` fields are
    /// already `f64` and compare directly without `.value()`.
    fn site_requirement(&mut self, parts: &[&Expr], linearity: &Linearity<'_>) -> Requirement {
        let mut ways: Vec<Vec<Expr>> = vec![Vec::new()];
        for part in parts {
            let options = linearity.options(part);
            let acceptable: Vec<&Vec<Expr>> = options
                .iter()
                .filter(|&(form, _)| matches!(form, Form::Zero | Form::Linear))
                .map(|(_, required)| required)
                .collect();
            ways = ways
                .iter()
                .flat_map(|way| {
                    acceptable.iter().map(move |required| {
                        let mut both = way.clone();
                        both.extend(required.iter().filter(|expr| !way.contains(expr)).cloned());
                        both
                    })
                })
                .collect();
        }
        if ways.is_empty() {
            return Requirement::Never;
        }
        if ways.iter().any(Vec::is_empty) {
            return Requirement::Always;
        }
        let mut clauses: Vec<(String, u8)> = Vec::new();
        for way in &ways {
            let checks: Vec<(String, u8)> = way
                .iter()
                .map(|expr| {
                    let operand = self.static_operand(expr);
                    match operand.kind {
                        Kind::Real | Kind::Float(_) => {
                            (format!("{} == 0.0", operand.at(SUM)), COMPARE)
                        }
                        Kind::Integer => (format!("{} == 0", operand.at(SUM)), COMPARE),
                        Kind::Bool => (format!("!{}", operand.at(ATOM)), UNARY),
                        Kind::Dual(_) => {
                            (format!("{} == 0.0", Self::value_of(&operand).at(SUM)), COMPARE)
                        }
                    }
                })
                .collect();
            let clause = if let [only] = checks.as_slice() {
                only.clone()
            } else {
                let joined = checks
                    .into_iter()
                    .map(|(code, _)| code)
                    .collect::<Vec<_>>()
                    .join(" && ");
                (joined, AND)
            };
            if !clauses.contains(&clause) {
                clauses.push(clause);
            }
        }
        Requirement::Or(clauses)
    }

    /// Writes the lines that bind each probe the body reads.
    fn write_probe_lines(&self, out: &mut CodeWriter) {
        let module = self.facts.module;
        let read = if self.mode == Mode::Evaluate {
            "unknown::<R>"
        } else {
            "value"
        };
        let wrap = if self.mode == Mode::Evaluate {
            "Dual::variable"
        } else {
            "R::constant"
        };
        for &probe in &self.probes {
            let Some(name) = self.probe_field(probe) else {
                continue;
            };
            let value = match probe {
                Probe::Voltage(plus, minus) => format!(
                    "at.{read}({}) - at.{read}({})",
                    self.facts.node(plus, "shape."),
                    self.facts.node(minus, "shape.")
                ),
                Probe::Current(branch) => {
                    format!("at.{read}(shape.{})", module.branches[branch].name)
                }
                Probe::Limited(_) => continue,
            };
            out.line(format!("v.{name} = {wrap}({value});"));
        }
    }

    /// The fields of the evaluation's variables: the dynamic variables kept, and in an
    /// evaluation the accumulators and the voltages limited.
    fn locals(&self) -> Vec<(String, Kind)> {
        let module = self.facts.module;
        let mut locals: Vec<(String, Kind)> = (0..module.variables.len())
            .filter(|&variable| self.live[variable] && module.dynamic[variable])
            .map(|variable| {
                (
                    module.variables[variable].0.clone(),
                    self.variable_kind(variable),
                )
            })
            .collect();
        for &probe in &self.probes {
            let Some(name) = self.probe_field(probe) else {
                continue;
            };
            let kind = if self.mode == Mode::Evaluate {
                Kind::Dual(vec![probe])
            } else {
                Kind::Real
            };
            locals.push((name, kind));
        }
        if self.mode == Mode::Evaluate {
            for ((branch, site, charge), kind) in &self.accumulators {
                locals.push((self.accumulator(*branch, *site, *charge), kind.clone()));
            }
            for index in &self.extended {
                locals.push((format!("applied{index}"), Kind::Float(None)));
                locals.push((format!("limited{index}"), Kind::Float(None)));
            }
        }
        locals
    }

    /// Renders the final indented body for `evaluate` or `noise` and detects used
    /// parameter bindings.
    fn render_method_body(&self, statements: &[Rust], memory: String) -> MethodBody {
        let mut writer = CodeWriter::with_depth(2);
        write_locals_struct(&mut writer, &self.locals());
        self.write_probe_lines(&mut writer);
        print(statements, &mut writer);
        let body = writer.finish();
        let bind = |name, used| if used { name } else { "_" };
        MethodBody {
            shape: bind("shape", mentions(&body, "shape")),
            prepared: bind("prepared", mentions(&body, "prepared")),
            at: bind("at", self.mode == Mode::Evaluate || mentions(&body, "at")),
            memory,
            body,
        }
    }

    /// The body of `evaluate`.
    pub(super) fn evaluate_body(&mut self) -> Result<MethodBody, Error> {
        self.find_accumulators();
        let module = self.facts.module;
        let mut statements = Vec::new();
        let analog = self.analog.clone();
        self.statements(&analog, &mut statements)?;
        self.stamps(&mut statements);
        for &output in &module.outputs {
            let operand = self.expression(&Expr::Variable(output), &mut statements)?;
            let value = Self::value_of(&operand).code;
            statements.push(Rust::Line(format!(
                "memory.{} = {value};",
                module.variables[output].0
            )));
        }
        let memory = if self.facts.remembers() {
            format!(
                "memory: &mut {}",
                Names::from_module(&module.name, false, false).memory
            )
        } else {
            "(): &mut ()".to_owned()
        };
        Ok(self.render_method_body(&statements, memory))
    }

    /// The body of `noise`.
    pub(super) fn noise_body(&mut self) -> Result<MethodBody, Error> {
        let mut statements = Vec::new();
        let analog = self.analog.clone();
        self.statements(&analog, &mut statements)?;
        Ok(self.render_method_body(&statements, String::new()))
    }
}

/// Simplifies conditionals in `statements`: drops empty conditionals, unwraps constant
/// boolean conditions, and inverts empty `if cond {} else { body }` to `if !cond { body }`.
fn simplify(statements: &[Rust]) -> Vec<Rust> {
    let mut out = Vec::with_capacity(statements.len());
    for statement in statements {
        match statement {
            Rust::Line(_) => out.push(statement.clone()),
            Rust::While(condition, body) => {
                out.push(Rust::While(condition.clone(), simplify(body)));
            }
            Rust::If((condition, precedence), then, otherwise) => {
                let then = simplify(then);
                let otherwise = simplify(otherwise);
                if condition == "true" {
                    out.extend(then);
                } else if condition == "false" {
                    out.extend(otherwise);
                } else if then.is_empty() {
                    if !otherwise.is_empty() {
                        out.push(Rust::If(
                            negate_condition(condition, *precedence),
                            otherwise,
                            Vec::new(),
                        ));
                    }
                } else {
                    out.push(Rust::If((condition.clone(), *precedence), then, otherwise));
                }
            }
        }
    }
    out
}

/// Returns the logical negation of `condition` at `precedence` and its resulting precedence.
fn negate_condition(condition: &str, precedence: u8) -> (String, u8) {
    if let Some(positive) = positive(condition) {
        return positive;
    }
    if let Some((left, right)) = condition.split_once(" == ")
        && single(left)
        && single(right)
    {
        return (format!("{left} != {right}"), COMPARE);
    }
    if precedence >= UNARY {
        (format!("!{condition}"), UNARY)
    } else {
        (format!("!({condition})"), UNARY)
    }
}

/// Prints `statements` into `out` at its current indentation depth, simplifying nested
/// conditionals and `else if` chains to match clippy's style.
pub(super) fn print(statements: &[Rust], out: &mut CodeWriter) {
    let simplified = simplify(statements);
    print_simplified(&simplified, out);
}

/// Prints pre-simplified `statements` into `out`.
fn print_simplified(statements: &[Rust], out: &mut CodeWriter) {
    for statement in statements {
        match statement {
            Rust::Line(line) => out.line(line),
            Rust::While(condition, body) => {
                out.block(format!("while {condition} {{"), "}", |out| {
                    print_simplified(body, out);
                });
            }
            Rust::If(..) => print_if(statement, out, ""),
        }
    }
}

/// Whether `code` is one operand: a path, or one whose parentheses or brackets, opened
/// within it, close only at its end.
fn single(code: &str) -> bool {
    let mut depth = 0_i32;
    for (at, c) in code.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                depth -= 1;
                if depth == 0
                    && at + 1 < code.len()
                    && !code[at + 1..].starts_with('[')
                    && !code[at + 1..].starts_with('.')
                {
                    return false;
                }
            }
            ' ' if depth == 0 => return false,
            _ => {}
        }
    }
    depth == 0
}

/// The positive form of `condition` and its precedence, when it is a negation: `!c` of
/// one operand reads `c`, and `a != b` of two, at its top, `a == b`; `None` for another.
pub(super) fn positive(condition: &str) -> Option<(String, u8)> {
    if let Some(inner) = condition.strip_prefix('!')
        && single(inner)
    {
        let stripped = inner
            .strip_prefix('(')
            .and_then(|rest| rest.strip_suffix(')'))
            .filter(|_| inner.ends_with(')') && inner.starts_with('('));
        return Some(match stripped {
            Some(stripped) => (stripped.to_owned(), OR),
            None => (inner.to_owned(), ATOM),
        });
    }
    let (left, right) = condition.split_once(" != ")?;
    (single(left) && single(right)).then(|| (format!("{left} == {right}"), COMPARE))
}

fn print_if(statement: &Rust, out: &mut CodeWriter, prefix: &str) {
    let Rust::If((condition, precedence), then, otherwise) = statement else {
        return;
    };
    // `if !c { a } else { b }` reads as `if c { b } else { a }`.
    if !otherwise.is_empty()
        && let Some(positive) = positive(condition)
    {
        let swapped = Rust::If(positive, otherwise.clone(), then.clone());
        print_if(&swapped, out, prefix);
        return;
    }
    let (mut condition, mut then) = ((condition.clone(), *precedence), then.clone());
    // `if a { if b { .. } }` is `if a && b { .. }`.
    while otherwise.is_empty()
        && let [Rust::If((inner, inner_precedence), inner_then, inner_otherwise)] = then.as_slice()
        && inner_otherwise.is_empty()
    {
        let left = parenthesize(&condition.0, condition.1, OR + 1);
        let right = parenthesize(inner, *inner_precedence, OR + 1);
        condition = (format!("{left} && {right}"), AND);
        then = inner_then.clone();
    }
    out.block(format!("{prefix}if {} {{", condition.0), "", |out| {
        print_simplified(&then, out);
    });
    match otherwise.as_slice() {
        [] => out.line("}"),
        [inner @ Rust::If(..)] => print_if(inner, out, "} else "),
        _ => out.block("} else {", "}", |out| print_simplified(otherwise, out)),
    }
}
