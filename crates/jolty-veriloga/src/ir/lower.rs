//! Lowering of parsed Verilog-A AST declarations, analog functions, statements, and
//! expressions into the resolved IR using zero-copy borrowed lexical scopes.

use std::collections::BTreeSet;

use jolty_models::{BOLTZMANN, ELEMENTARY_CHARGE};

use crate::Error;
use crate::ast::{self, Type};

use super::analysis::contributes;
use super::checks::{check_block_duplicates, check_function_duplicates, node_index};
use super::ddt::split_contribution;
use super::fold::{constant, expr_type, fold_const};
use super::{
    Branch, Builtin, Expression, Function, GROUND, Limit, Limiter, LocalParameter, Noise,
    Parameter, Probe, Statement, apart, identifier,
};

/// What a resolved `localparam` evaluates to: either a compile-time constant folded
/// directly into expressions, or an index into [`Sink::local_parameters`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum LocalValue {
    Real(f64),
    Integer(i32),
    Prepared(usize),
}

/// A lexical binding in a [`Scope`]: either a mutable variable or an immutable
/// `localparam`.
#[derive(Debug, Clone, Copy)]
pub(super) enum Binding {
    Variable(usize),
    LocalParameter(LocalValue),
}

/// What resolution adds to as it reads: the limited probes, the branches, and the
/// variables, to which named blocks and inlined analog functions add their own.
#[derive(Default)]
pub(super) struct Sink {
    pub(super) limits: Vec<Limit>,
    pub(super) branches: Vec<Branch>,
    pub(super) local_parameters: Vec<LocalParameter>,
    pub(super) variables: Vec<(String, Type)>,
    /// The statements the expression being read needs before it: the bodies of the
    /// analog functions it calls, inlined.
    prelude: Vec<Statement>,
    /// How many calls have been inlined, which numbers their variables.
    calls: usize,
    /// The variables that code under an initial event assigns.
    pub(super) initial: BTreeSet<usize>,
}

impl Sink {
    /// A new variable of `kind`, named `name` unless that is taken.
    pub(super) fn variable(&mut self, name: &str, kind: Type) -> usize {
        let unique = apart(&identifier(name), |taken| {
            self.variables.iter().any(|(variable, _)| variable == taken)
        });
        self.variables.push((unique, kind));
        self.variables.len() - 1
    }

    /// A new non-constant `localparam` of `kind`, named `name` unless that is taken.
    fn local_parameter(&mut self, name: &str, kind: Type, value: Expression) -> usize {
        let unique = apart(&identifier(name), |taken| {
            self.local_parameters
                .iter()
                .any(|existing| existing.name == taken)
        });
        self.local_parameters.push(LocalParameter {
            name: unique,
            external: name.to_owned(),
            kind,
            value,
        });
        self.local_parameters.len() - 1
    }
}

/// Read-only module context shared across all borrowed scopes during resolution.
#[derive(Clone, Copy)]
pub(super) struct Context<'a> {
    pub(super) module: &'a ast::Module,
    pub(super) nodes: &'a [String],
    /// Whether each node is thermal rather than electrical.
    pub(super) thermal: &'a [bool],
}

/// The names a module's expressions may read, in a scope: the module's, a named
/// block's, an analog function's, or a constant declaration's.
#[derive(Clone, Copy)]
pub(super) struct Scope<'a> {
    context: Context<'a>,
    /// Lexical bindings in scope, the innermost last.
    bindings: &'a [(&'a str, Binding)],
    /// Module-scope `localparam` bindings, available to inlined analog functions.
    module_locals: &'a [(&'a str, LocalValue)],
    /// Parameters visible when resolving a constant declaration (`None` means all module
    /// parameters are visible).
    allowed_parameters: Option<&'a BTreeSet<usize>>,
    /// Later `localparam` names in the current declaration sequence, for forward
    /// reference diagnostics.
    later_locals: &'a [&'a str],
    /// Owner label when resolving a constant expression (`Some` rejects variables and
    /// probes).
    constant_owner: Option<&'a str>,
    /// Whether the scope reads probes, as the analog block does and a function does not.
    probes: bool,
    /// Whether the scope lies under an initial event, whose code runs at an analysis's
    /// first step only and so may read only what does not change after it.
    initial: bool,
}

impl<'a> Scope<'a> {
    /// A scope for resolving a constant expression (`parameter` default, range, or
    /// `localparam` initializer).
    const fn constant(
        context: Context<'a>,
        bindings: &'a [(&'a str, Binding)],
        module_locals: &'a [(&'a str, LocalValue)],
        owner: &'a str,
    ) -> Self {
        Self {
            constant_owner: Some(owner),
            ..Self::block(context, bindings, module_locals, false)
        }
    }

    /// Restricts visible parameters and records later `localparam` names for diagnostics.
    const fn with_visibility(
        self,
        allowed_parameters: Option<&'a BTreeSet<usize>>,
        later_locals: &'a [&'a str],
    ) -> Self {
        Self {
            allowed_parameters,
            later_locals,
            ..self
        }
    }

    /// A scope for resolving the statements of an analog block or analog function.
    pub(super) const fn block(
        context: Context<'a>,
        bindings: &'a [(&'a str, Binding)],
        module_locals: &'a [(&'a str, LocalValue)],
        probes: bool,
    ) -> Self {
        Self {
            context,
            bindings,
            module_locals,
            allowed_parameters: None,
            later_locals: &[],
            constant_owner: None,
            probes,
            initial: false,
        }
    }
}

/// Converts resolved module-scope `localparam` entries into lexical [`Binding`]s.
pub(super) fn local_bindings<'a>(locals: &[(&'a str, LocalValue)]) -> Vec<(&'a str, Binding)> {
    locals
        .iter()
        .map(|&(name, value)| (name, Binding::LocalParameter(value)))
        .collect()
}

/// Creates a [`Sink`] pre-populated with `module`'s declared variables and named branches.
pub(super) fn initialize_sink(module: &ast::Module, nodes: &[String]) -> Result<Sink, Error> {
    let mut sink = Sink::default();
    for (name, kind) in &module.variables {
        sink.variable(name, *kind);
    }
    for (name, plus, minus) in &module.branches {
        let plus = node_index(nodes, plus, 0)?;
        let minus = if minus.is_empty() {
            GROUND
        } else {
            node_index(nodes, minus, 0)?
        };
        sink.branches.push(Branch {
            name: name.clone(),
            plus,
            minus,
        });
    }
    Ok(sink)
}

/// Resolved module `parameter` declarations and module-scope `localparam` bindings.
pub(super) type ResolvedDeclarations<'a> = (Vec<Parameter>, Vec<(&'a str, LocalValue)>);

/// Resolves `module`'s `parameter` and `localparam` declarations in source order,
/// returning the resolved parameters and the module-scope `localparam` bindings.
pub(super) fn resolve_declarations<'a>(
    context: Context<'a>,
    sink: &mut Sink,
) -> Result<ResolvedDeclarations<'a>, Error> {
    let module = context.module;
    let given = given_parameters(module);
    let mut seen_parameters: BTreeSet<usize> = BTreeSet::new();
    let mut module_locals: Vec<(&'a str, LocalValue)> = Vec::new();
    let mut bindings: Vec<(&'a str, Binding)> = Vec::new();
    let mut resolved_defaults: Vec<Option<Expression>> = vec![None; module.parameters.len()];
    let local_names: Vec<&'a str> = module
        .declarations
        .iter()
        .filter_map(|declaration| match *declaration {
            ast::Declaration::LocalParameter(index) => {
                Some(module.local_parameters[index].name.as_str())
            }
            ast::Declaration::Parameter(_) => None,
        })
        .collect();
    let mut local_cursor = 0;

    for declaration in &module.declarations {
        let later = &local_names[local_cursor..];
        match *declaration {
            ast::Declaration::Parameter(index) => {
                let parameter = &module.parameters[index];
                let owner = format!("{}'s default", parameter.name);
                let scope = Scope::constant(context, &bindings, &module_locals, &owner)
                    .with_visibility(Some(&seen_parameters), later);
                let mut default = scope.expression(&parameter.default, parameter.line, sink)?;
                if !sink.prelude.is_empty() {
                    return Err(Error::at(
                        parameter.line,
                        format!("{}'s default calls a function", parameter.name),
                    ));
                }
                fold_parameter_default(&mut default, parameter)?;
                resolved_defaults[index] = Some(default);
                seen_parameters.insert(index);
            }
            ast::Declaration::LocalParameter(index) => {
                let local = &module.local_parameters[index];
                let scope = Scope::constant(context, &bindings, &module_locals, &local.name)
                    .with_visibility(Some(&seen_parameters), later);
                let value = resolve_local_parameter(&scope, local, sink)?;
                module_locals.push((local.name.as_str(), value));
                bindings.push((local.name.as_str(), Binding::LocalParameter(value)));
                local_cursor += 1;
            }
        }
    }

    let parameters = resolve_parameter_ranges(
        context,
        &bindings,
        &module_locals,
        &given,
        &mut resolved_defaults,
        sink,
    )?;
    Ok((parameters, module_locals))
}

/// Constant-folds a parameter default expression when it has no runtime dependence,
/// propagating any integer overflow, divide-by-zero, or non-finite error.
fn fold_parameter_default(
    default: &mut Expression,
    parameter: &ast::Parameter,
) -> Result<(), Error> {
    let folded = fold_const(default).map_err(|message| {
        Error::at(
            parameter.line,
            format!("{}'s default {message}", parameter.name),
        )
    })?;
    if let Some(folded) = folded {
        if parameter.kind == Type::Integer {
            if let Some(integer) = folded.as_int() {
                *default = Expression::Integer(integer);
            }
        } else {
            *default = Expression::Real(folded.as_real());
        }
    }
    Ok(())
}

/// Resolves every parameter's `from`/`exclude` ranges once all module-scope
/// `parameter` and `localparam` declarations are known.
fn resolve_parameter_ranges<'a>(
    context: Context<'a>,
    range_bindings: &[(&'a str, Binding)],
    module_locals: &[(&'a str, LocalValue)],
    given: &BTreeSet<String>,
    resolved_defaults: &mut [Option<Expression>],
    sink: &mut Sink,
) -> Result<Vec<Parameter>, Error> {
    let module = context.module;
    let mut parameters: Vec<Parameter> = Vec::with_capacity(module.parameters.len());
    for (index, parameter) in module.parameters.iter().enumerate() {
        let default = resolved_defaults[index].take().ok_or_else(|| {
            Error::at(
                parameter.line,
                format!("{} lacks a default", parameter.name),
            )
        })?;
        let owner = format!("{}'s range", parameter.name);
        let scope = Scope::constant(context, range_bindings, module_locals, &owner);
        let mut ranges = Vec::new();
        for range in &parameter.ranges {
            let mut bound = |bound: &Option<(ast::Expression, bool)>| -> Result<_, Error> {
                bound
                    .as_ref()
                    .map(|(value, included)| {
                        let mut resolved = scope.expression(value, parameter.line, sink)?;
                        if let Some(folded) = fold_const(&resolved).map_err(|message| {
                            Error::at(
                                parameter.line,
                                format!("{}'s range {message}", parameter.name),
                            )
                        })? {
                            resolved = Expression::Real(folded.as_real());
                        }
                        Ok((resolved, *included))
                    })
                    .transpose()
            };
            let (lower, upper) = (bound(&range.lower)?, bound(&range.upper)?);
            ranges.push((range.exclude, lower, upper));
        }
        if !sink.prelude.is_empty() {
            return Err(Error::at(
                parameter.line,
                format!("{}'s range calls a function", parameter.name),
            ));
        }
        let mut name = identifier(&parameter.name);
        // Two parameters whose names meet in snake case keep apart.
        if parameters.iter().any(|other| other.name == name) {
            name = format!("{name}_{index}");
        }
        parameters.push(Parameter {
            name,
            external: parameter.name.clone(),
            kind: parameter.kind,
            optional: constant(&default).is_none() || given.contains(&parameter.name),
            default,
            ranges,
            description: parameter.description.clone(),
            units: parameter.units.clone(),
            aliases: module
                .aliases
                .iter()
                .filter(|(_, named)| *named == parameter.name)
                .map(|(alias, _)| alias.clone())
                .collect(),
        });
    }
    Ok(parameters)
}

/// Resolves one `localparam` declaration: checks type compatibility, constant-folds
/// compile-time values into `LocalValue::Real` or `LocalValue::Integer`, and records
/// parameter- or temperature-dependent ones in `sink.local_parameters`.
fn resolve_local_parameter(
    scope: &Scope<'_>,
    local: &ast::LocalParameter,
    sink: &mut Sink,
) -> Result<LocalValue, Error> {
    let value = scope.expression(&local.value, local.line, sink)?;
    if !sink.prelude.is_empty() {
        return Err(Error::at(
            local.line,
            format!(
                "{} calls an analog function, which is not compiled yet",
                local.name
            ),
        ));
    }
    let inferred = expr_type(&value, scope.context.module, &sink.local_parameters);
    let kind = match local.kind {
        Some(Type::Integer) => {
            if inferred == Type::Real {
                return Err(Error::at(
                    local.line,
                    format!(
                        "the integer localparam {} has a real initializer",
                        local.name
                    ),
                ));
            }
            Type::Integer
        }
        Some(Type::Real) => Type::Real,
        None => inferred,
    };
    let prefix = if kind == Type::Integer {
        format!("the integer localparam {}", local.name)
    } else {
        format!("the localparam {}", local.name)
    };
    if let Some(folded) = fold_const(&value)
        .map_err(|message| Error::at(local.line, format!("{prefix} {message}")))?
    {
        if kind == Type::Real {
            return Ok(LocalValue::Real(folded.as_real()));
        }
        if let Some(integer) = folded.as_int() {
            return Ok(LocalValue::Integer(integer));
        }
    }
    Ok(LocalValue::Prepared(sink.local_parameter(
        &local.name,
        kind,
        value,
    )))
}

/// The variables of an analog function: its inputs first, then its locals, then its
/// result, which bears its name.
fn function_variables(function: &ast::Function) -> Vec<(&str, Type)> {
    let mut variables: Vec<(&str, Type)> = Vec::new();
    for input in &function.inputs {
        let kind = function
            .variables
            .iter()
            .find(|(name, _)| name == input)
            .map_or(Type::Real, |&(_, kind)| kind);
        variables.push((input.as_str(), kind));
    }
    for (name, kind) in &function.variables {
        if !function.inputs.contains(name) {
            variables.push((name.as_str(), *kind));
        }
    }
    variables.push((function.name.as_str(), Type::Real));
    variables
}

/// An analog function that limits a probe, resolved in a scope of its own.
pub(super) fn resolve_function<'a>(
    context: Context<'a>,
    module_locals: &[(&'a str, LocalValue)],
    local_parameters: &mut Vec<LocalParameter>,
    function: &'a ast::Function,
) -> Result<Function, Error> {
    check_function_duplicates(function)?;
    let variable_specs = function_variables(function);
    let mut bindings = local_bindings(module_locals);
    bindings.extend(
        variable_specs
            .iter()
            .enumerate()
            .map(|(index, &(name, _))| (name, Binding::Variable(index))),
    );
    let mut sink = Sink {
        variables: variable_specs
            .into_iter()
            .map(|(name, kind)| (name.to_owned(), kind))
            .collect(),
        local_parameters: std::mem::take(local_parameters),
        ..Sink::default()
    };
    let resolved = resolve_scoped_locals(
        context,
        &mut bindings,
        module_locals,
        &function.local_parameters,
        &mut sink,
    )
    .and_then(|()| {
        let scope = Scope::block(context, &bindings, module_locals, false);
        scope.statements(&function.body, &mut sink)
    });
    *local_parameters = sink.local_parameters;
    let body = resolved?;
    Ok(Function {
        name: function.name.clone(),
        inputs: function.inputs.len(),
        variables: sink.variables,
        body,
    })
}

/// Resolves a sequence of `localparam` declarations inside a named block or analog
/// function, appending each binding into `bindings`.
fn resolve_scoped_locals<'a>(
    context: Context<'a>,
    bindings: &mut Vec<(&'a str, Binding)>,
    module_locals: &[(&'a str, LocalValue)],
    local_parameters: &'a [ast::LocalParameter],
    sink: &mut Sink,
) -> Result<(), Error> {
    let local_names: Vec<&'a str> = local_parameters
        .iter()
        .map(|local| local.name.as_str())
        .collect();
    for (position, local) in local_parameters.iter().enumerate() {
        let constant_scope = Scope::constant(context, bindings, module_locals, &local.name)
            .with_visibility(None, &local_names[position..]);
        let value = resolve_local_parameter(&constant_scope, local, sink)?;
        bindings.push((local.name.as_str(), Binding::LocalParameter(value)));
    }
    Ok(())
}

/// The names of the parameters `$param_given` reads across `module`.
fn given_parameters(module: &ast::Module) -> BTreeSet<String> {
    let mut given = BTreeSet::new();
    collect_given_stmt(&module.analog, &mut given);
    for local in &module.local_parameters {
        collect_given_expr(&local.value, &mut given);
    }
    for function in &module.functions {
        for local in &function.local_parameters {
            collect_given_expr(&local.value, &mut given);
        }
        collect_given_stmt(&function.body, &mut given);
    }
    for parameter in &module.parameters {
        collect_given_expr(&parameter.default, &mut given);
        for range in &parameter.ranges {
            for (bound, _) in range.lower.iter().chain(&range.upper) {
                collect_given_expr(bound, &mut given);
            }
        }
    }
    given
}

fn collect_given_stmt(statement: &ast::Statement, given: &mut BTreeSet<String>) {
    match statement {
        ast::Statement::Block(statements) => {
            for inner in statements {
                collect_given_stmt(inner, given);
            }
        }
        ast::Statement::Named {
            local_parameters,
            statements,
            ..
        } => {
            for local in local_parameters {
                collect_given_expr(&local.value, given);
            }
            for inner in statements {
                collect_given_stmt(inner, given);
            }
        }
        ast::Statement::Assign { value, .. } | ast::Statement::Contribute { value, .. } => {
            collect_given_expr(value, given);
        }
        ast::Statement::If {
            condition,
            then,
            otherwise,
        } => {
            collect_given_expr(condition, given);
            collect_given_stmt(then, given);
            if let Some(otherwise) = otherwise {
                collect_given_stmt(otherwise, given);
            }
        }
        ast::Statement::While { condition, body } => {
            collect_given_expr(condition, given);
            collect_given_stmt(body, given);
        }
        ast::Statement::Initial { body, .. } => collect_given_stmt(body, given),
        ast::Statement::Task { arguments, .. } => arguments
            .iter()
            .for_each(|argument| collect_given_expr(argument, given)),
    }
}

fn collect_given_expr(expression: &ast::Expression, given: &mut BTreeSet<String>) {
    match expression {
        ast::Expression::Call(name, arguments) => {
            if name == "$param_given"
                && let [ast::Expression::Name(parameter)] = arguments.as_slice()
            {
                given.insert(parameter.clone());
            }
            for inner in arguments {
                collect_given_expr(inner, given);
            }
        }
        ast::Expression::Unary(_, operand) => collect_given_expr(operand, given),
        ast::Expression::Binary(_, left, right) => {
            collect_given_expr(left, given);
            collect_given_expr(right, given);
        }
        ast::Expression::Conditional(condition, then, otherwise) => {
            collect_given_expr(condition, given);
            collect_given_expr(then, given);
            collect_given_expr(otherwise, given);
        }
        _ => {}
    }
}

impl<'a> Scope<'a> {
    #[expect(
        clippy::too_many_lines,
        reason = "one arm per statement of the language, which read best side by side"
    )]
    pub(super) fn statements(
        &self,
        statement: &'a ast::Statement,
        sink: &mut Sink,
    ) -> Result<Vec<Statement>, Error> {
        // The statements an expression's inlined calls need come before the statement.
        let before = |sink: &mut Sink, statement: Statement| {
            let mut resolved = std::mem::take(&mut sink.prelude);
            resolved.push(statement);
            resolved
        };
        Ok(match statement {
            ast::Statement::Block(statements) => {
                let mut resolved = Vec::new();
                for inner in statements {
                    resolved.extend(self.statements(inner, sink)?);
                }
                resolved
            }
            ast::Statement::Named {
                local_parameters,
                variables,
                statements,
                ..
            } => {
                check_block_duplicates(local_parameters, variables)?;
                let mut inner_bindings = self.bindings.to_vec();
                for (variable, kind) in variables {
                    let index = sink.variable(variable, *kind);
                    inner_bindings.push((variable.as_str(), Binding::Variable(index)));
                }
                resolve_scoped_locals(
                    self.context,
                    &mut inner_bindings,
                    self.module_locals,
                    local_parameters,
                    sink,
                )?;
                let inner = Scope {
                    bindings: &inner_bindings,
                    ..*self
                };
                let mut resolved = Vec::new();
                for inner_statement in statements {
                    resolved.extend(inner.statements(inner_statement, sink)?);
                }
                resolved
            }
            ast::Statement::Assign {
                target,
                value,
                line,
            } => {
                let variable = self.variable(target, *line)?;
                let value = self.expression(value, *line, sink)?;
                if self.initial {
                    sink.initial.insert(variable);
                }
                before(sink, Statement::Assign(variable, value))
            }
            ast::Statement::Contribute {
                nature,
                operands,
                value,
                line,
            } => {
                if self.initial {
                    return Err(Error::at(
                        *line,
                        "a contribution under an initial event is not compiled yet".into(),
                    ));
                }
                let contribution = self.contribution(nature, operands, value, *line, sink)?;
                before(sink, contribution)
            }
            ast::Statement::Initial { body, .. } => {
                let inner = Self {
                    probes: false,
                    initial: true,
                    ..*self
                };
                inner.statements(body, sink)?
            }
            ast::Statement::If {
                condition,
                then,
                otherwise,
            } => {
                let condition = self.expression(condition, 0, sink)?;
                let prelude = std::mem::take(&mut sink.prelude);
                let then = self.statements(then, sink)?;
                let otherwise = match otherwise {
                    Some(otherwise) => self.statements(otherwise, sink)?,
                    None => Vec::new(),
                };
                sink.prelude = prelude;
                before(sink, Statement::If(0, condition, then, otherwise))
            }
            ast::Statement::While { condition, body } => {
                let condition = self.expression(condition, 0, sink)?;
                if !sink.prelude.is_empty() {
                    return Err(Error::at(
                        0,
                        "a function called in a loop's condition is not compiled yet".into(),
                    ));
                }
                let body = self.statements(body, sink)?;
                if body.iter().any(contributes) {
                    return Err(Error::at(
                        0,
                        "a contribution within a loop is not compiled yet".into(),
                    ));
                }
                vec![Statement::While(condition, body)]
            }
            ast::Statement::Task {
                name,
                arguments,
                line,
            } => {
                // `$finish` and `$stop` end the simulation, which jolty does by refusing
                // the instance; their arguments only say what to print.
                if matches!(name.as_str(), "$finish" | "$stop") {
                    return Ok(before(
                        sink,
                        Statement::Fail(format!("the model calls {name}"), Vec::new()),
                    ));
                }
                let mut arguments = arguments.iter();
                let Some(ast::Expression::Text(message)) = arguments.next() else {
                    return Err(Error::at(*line, format!("{name} takes a message first")));
                };
                let values = arguments
                    .map(|argument| self.expression(argument, *line, sink))
                    .collect::<Result<_, _>>()?;
                let task = match name.as_str() {
                    "$error" | "$fatal" => Statement::Fail(message.clone(), values),
                    "$warning" | "$strobe" | "$display" | "$write" | "$debug" => {
                        Statement::Warn(message.clone(), values, false)
                    }
                    _ => return Err(Error::at(*line, format!("{name} is not compiled yet"))),
                };
                before(sink, task)
            }
        })
    }

    pub(super) fn variable(&self, name: &str, line: usize) -> Result<usize, Error> {
        match self
            .bindings
            .iter()
            .rev()
            .find(|&&(bound, _)| bound == name)
            .map(|&(_, binding)| binding)
        {
            Some(Binding::Variable(index)) => Ok(index),
            Some(Binding::LocalParameter(_)) => Err(Error::at(
                line,
                format!("{name} is a localparam, not a variable"),
            )),
            None => Err(Error::at(line, format!("{name} is not a variable"))),
        }
    }

    /// The branch an access function reads or contributes to: a named one, or the
    /// unnamed one of two nodes, added on first use; its nodes of the discipline that
    /// the access `nature` reads.
    fn branch(
        &self,
        nature: &str,
        operands: &[String],
        line: usize,
        sink: &mut Sink,
    ) -> Result<usize, Error> {
        // The module's declared branches come first, each found by its name; another
        // single node is its unnamed branch to the ground.
        let declared = self.context.module.branches.len().min(sink.branches.len());
        let named = match operands {
            [name] => sink.branches[..declared]
                .iter()
                .position(|branch| branch.name == *name),
            _ => None,
        };
        let index = match (named, operands) {
            (Some(found), _) => found,
            (None, [node] | [node, _]) => {
                let plus = node_index(self.context.nodes, node, line)?;
                let (name, minus) = if let [_, other] = operands {
                    let minus = node_index(self.context.nodes, other, line)?;
                    (
                        format!("{}_{}", self.context.nodes[plus], self.context.nodes[minus]),
                        minus,
                    )
                } else {
                    // Named apart from the node, whose unknown its current's would meet.
                    (format!("{node}_ground"), GROUND)
                };
                if let Some(found) = sink.branches.iter().position(|branch| {
                    branch.name == name && branch.plus == plus && branch.minus == minus
                }) {
                    found
                } else {
                    sink.branches.push(Branch { name, plus, minus });
                    sink.branches.len() - 1
                }
            }
            _ => {
                return Err(Error::at(
                    line,
                    "an access takes a branch or two nodes".into(),
                ));
            }
        };
        let thermal = matches!(nature, "Temp" | "Pwr");
        let Branch { plus, minus, .. } = sink.branches[index];
        let discipline = |node: usize| node == GROUND || self.context.thermal[node] == thermal;
        if !discipline(plus) || !discipline(minus) {
            let discipline = if thermal { "thermal" } else { "electrical" };
            return Err(Error::at(
                line,
                format!("{nature} reads {discipline} nodes"),
            ));
        }
        Ok(index)
    }

    fn contribution(
        &self,
        nature: &str,
        operands: &[String],
        value: &ast::Expression,
        line: usize,
        sink: &mut Sink,
    ) -> Result<Statement, Error> {
        let potential = match nature {
            "V" | "Temp" => true,
            "I" | "Pwr" => false,
            _ => {
                return Err(Error::at(
                    line,
                    format!("{nature} is not an access function"),
                ));
            }
        };
        let branch = self.branch(nature, operands, line, sink)?;
        let (mut resistive, mut reactive, mut reactive_factors, mut noise) =
            (None, None, Vec::new(), Vec::new());
        let add = |sum: &mut Option<Expression>, negative: bool, term: Expression| {
            *sum = Some(match (sum.take(), negative) {
                (None, false) => term,
                (None, true) => Expression::Unary("-", Box::new(term)),
                (Some(sum), false) => Expression::Binary("+", Box::new(sum), Box::new(term)),
                (Some(sum), true) => Expression::Binary("-", Box::new(sum), Box::new(term)),
            });
        };
        for term in split_contribution(value) {
            match term.expression {
                ast::Expression::Call(name, arguments) if name == "ddt" => {
                    let [argument] = arguments.as_slice() else {
                        return Err(Error::at(line, "ddt takes one argument".into()));
                    };
                    let mut resolved = self.expression(argument, line, sink)?;
                    for factor in term.factors {
                        let scale = self.expression(factor.expression(), line, sink)?;
                        reactive_factors.push((scale.clone(), line));
                        resolved = factor.apply(scale, resolved);
                    }
                    add(&mut reactive, term.negative, resolved);
                }
                ast::Expression::Call(name, arguments)
                    if term.factors.is_empty()
                        && (name == "white_noise" || name == "flicker_noise") =>
                {
                    if term.negative {
                        return Err(Error::at(
                            line,
                            "a noise source is added, not subtracted".into(),
                        ));
                    }
                    noise.push(self.noise(name, arguments, line, sink)?);
                }
                leaf => {
                    let mut resolved = self.expression(leaf, line, sink)?;
                    for factor in term.factors {
                        let scale = self.expression(factor.expression(), line, sink)?;
                        resolved = factor.apply(scale, resolved);
                    }
                    add(&mut resistive, term.negative, resolved);
                }
            }
        }
        Ok(Statement::Contribute {
            branch,
            potential,
            resistive,
            reactive,
            reactive_factors,
            noise,
        })
    }

    /// The noise source `name(arguments)`, white or flicker.
    fn noise(
        &self,
        name: &str,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Noise, Error> {
        let (power, rest) = arguments
            .split_first()
            .ok_or_else(|| Error::at(line, format!("{name} takes its power")))?;
        let (exponent, rest) = if name == "flicker_noise" {
            let (exponent, rest) = rest
                .split_first()
                .ok_or_else(|| Error::at(line, "flicker_noise takes its exponent".into()))?;
            let exponent = self.expression(exponent, line, sink)?;
            (Some(exponent), rest)
        } else {
            (None, rest)
        };
        let [ast::Expression::Text(name)] = rest else {
            return Err(Error::at(line, "a noise source is named".into()));
        };
        Ok(Noise {
            name: name.clone(),
            power: self.expression(power, line, sink)?,
            exponent,
        })
    }

    fn expression(
        &self,
        expression: &ast::Expression,
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        let mut recurse = |inner: &ast::Expression| self.expression(inner, line, sink);
        Ok(match expression {
            ast::Expression::Real(value) => Expression::Real(*value),
            ast::Expression::Integer(value) => Expression::Integer(
                i32::try_from(*value)
                    .map_err(|_| Error::at(line, format!("{value} exceeds an integer")))?,
            ),
            ast::Expression::Text(_) => {
                return Err(Error::at(line, "a string is not a number".into()));
            }
            ast::Expression::Unary(operator, operand) => {
                Expression::Unary(operator, Box::new(recurse(operand)?))
            }
            ast::Expression::Binary(operator, left, right) => {
                let left = recurse(left)?;
                let right = recurse(right)?;
                if matches!(*operator, "%" | "**" | "&" | "|" | "^") {
                    return Err(Error::at(
                        line,
                        format!("the operator {operator} is not compiled yet"),
                    ));
                }
                Expression::Binary(operator, Box::new(left), Box::new(right))
            }
            ast::Expression::Conditional(condition, then, otherwise) => Expression::Conditional(
                Box::new(recurse(condition)?),
                Box::new(recurse(then)?),
                Box::new(recurse(otherwise)?),
            ),
            ast::Expression::Name(name) => return self.resolve_name(name, line, sink),
            ast::Expression::Call(name, arguments) => {
                return self.call(name, arguments, line, sink);
            }
        })
    }

    /// Ensures `parameter` is visible in the current declaration sequence.
    fn check_parameter_visible(&self, index: usize, name: &str, line: usize) -> Result<(), Error> {
        if let Some(allowed) = self.allowed_parameters
            && !allowed.contains(&index)
        {
            let owner = self.constant_owner.unwrap_or("declaration");
            return Err(Error::at(
                line,
                format!("{owner} reads {name}, which is not declared before it"),
            ));
        }
        Ok(())
    }

    /// Resolves an identifier `name` against the lexical scope, module parameters, and
    /// constant-context diagnostics.
    fn resolve_name(&self, name: &str, line: usize, sink: &Sink) -> Result<Expression, Error> {
        if let Some(owner) = self.constant_owner
            && self.later_locals.contains(&name)
        {
            return Err(Error::at(
                line,
                format!("{owner} reads {name}, which is not declared before it"),
            ));
        }
        if let Some(&(_, binding)) = self
            .bindings
            .iter()
            .rev()
            .find(|&&(bound, _)| bound == name)
        {
            return match binding {
                Binding::Variable(variable) => {
                    if let Some(owner) = self.constant_owner {
                        return Err(Error::at(
                            line,
                            format!("{owner} reads the variable {name}, which is not a constant"),
                        ));
                    }
                    if self.initial && !sink.initial.contains(&variable) {
                        return Err(Error::at(
                            line,
                            format!(
                                "code under an initial event reads {name}, which no initial code assigned before"
                            ),
                        ));
                    }
                    Ok(Expression::Variable(variable))
                }
                Binding::LocalParameter(LocalValue::Real(value)) => Ok(Expression::Real(value)),
                Binding::LocalParameter(LocalValue::Integer(value)) => {
                    Ok(Expression::Integer(value))
                }
                Binding::LocalParameter(LocalValue::Prepared(index)) => {
                    Ok(Expression::LocalParameter(index))
                }
            };
        }
        if let Some(parameter) = self
            .context
            .module
            .parameters
            .iter()
            .position(|parameter| parameter.name == name)
        {
            self.check_parameter_visible(parameter, name, line)?;
            return Ok(Expression::Parameter(parameter));
        }
        if let Some(owner) = self.constant_owner
            && self
                .context
                .module
                .variables
                .iter()
                .any(|(variable, _)| variable == name)
        {
            return Err(Error::at(
                line,
                format!("{owner} reads the variable {name}, which is not a constant"),
            ));
        }
        Err(Error::at(line, format!("{name} is not declared")))
    }

    /// The probe `nature(arguments)`: a potential or a flow.
    fn access(
        &self,
        nature: &str,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        let operands = arguments
            .iter()
            .map(|argument| match argument {
                ast::Expression::Name(name) => Ok(name.clone()),
                _ => Err(Error::at(line, "an access takes names".into())),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let branch = self.branch(nature, &operands, line, sink)?;
        if matches!(nature, "V" | "Temp") {
            let Branch { plus, minus, .. } = sink.branches[branch];
            Ok(Expression::Probe(Probe::Voltage(plus, minus)))
        } else {
            Ok(Expression::Probe(Probe::Current(branch)))
        }
    }

    fn call(
        &self,
        name: &str,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        if let Some(builtin) = Builtin::named(name) {
            let arity = builtin.arity();
            if arguments.len() != arity {
                return Err(Error::at(line, format!("{name} takes {arity} arguments")));
            }
            let arguments = arguments
                .iter()
                .map(|argument| self.expression(argument, line, sink))
                .collect::<Result<_, _>>()?;
            return Ok(Expression::Call(builtin, arguments));
        }
        if let Some(function) = self
            .context
            .module
            .functions
            .iter()
            .find(|function| function.name == name)
        {
            if let Some(owner) = self.constant_owner {
                return Err(Error::at(
                    line,
                    format!("{owner} calls the analog function {name}, which is not compiled yet"),
                ));
            }
            return self.inline(function, arguments, line, sink);
        }
        match name {
            "V" | "I" | "Temp" | "Pwr" | "ddx" | "$limit" if self.constant_owner.is_some() => {
                let owner = self.constant_owner.unwrap_or("declaration");
                Err(Error::at(
                    line,
                    format!("{owner} reads the probes through {name}, which is not a constant"),
                ))
            }
            "V" | "I" | "Temp" | "Pwr" | "ddx" | "$limit" if self.initial => Err(Error::at(
                line,
                format!(
                    "code under an initial event reads the probes through {name}, which its first step alone fixes"
                ),
            )),
            "V" | "I" | "Temp" | "Pwr" if self.probes => self.access(name, arguments, line, sink),
            "ddx" if self.probes => self.resolve_ddx(arguments, line, sink),
            "$temperature" if arguments.is_empty() => Ok(Expression::Temperature),
            // jolty's circuits place each instance once: none has a multiplicity but 1.
            "$mfactor" if arguments.is_empty() => Ok(Expression::Real(1.0)),
            // A jolty instance connects each of its terminals.
            "$port_connected" => match arguments {
                [ast::Expression::Name(port)] if self.context.module.ports.contains(port) => {
                    Ok(Expression::Integer(1))
                }
                _ => Err(Error::at(line, "$port_connected takes a port".into())),
            },
            "$vt" => {
                let temperature = match arguments {
                    [] => Expression::Temperature,
                    [temperature] => self.expression(temperature, line, sink)?,
                    _ => return Err(Error::at(line, "$vt takes at most a temperature".into())),
                };
                let product = Expression::Binary(
                    "*",
                    Box::new(temperature),
                    Box::new(Expression::Real(BOLTZMANN)),
                );
                Ok(Expression::Binary(
                    "/",
                    Box::new(product),
                    Box::new(Expression::Real(ELEMENTARY_CHARGE)),
                ))
            }
            "$param_given" => self.resolve_param_given(arguments, line),
            "$simparam" => self.resolve_simparam(arguments, line, sink),
            "$limit" if self.probes => self.limit(arguments, line, sink),
            _ => Err(Error::at(line, format!("{name} is not compiled yet"))),
        }
    }

    fn resolve_ddx(
        &self,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        let [operand, ast::Expression::Call(nature, node)] = arguments else {
            return Err(Error::at(
                line,
                "ddx takes a value and a node's potential".into(),
            ));
        };
        let (true, [ast::Expression::Name(node)]) =
            (matches!(nature.as_str(), "V" | "Temp"), node.as_slice())
        else {
            return Err(Error::at(
                line,
                "ddx differentiates by one node's potential".into(),
            ));
        };
        let operand = self.expression(operand, line, sink)?;
        Ok(Expression::Ddx(
            Box::new(operand),
            node_index(self.context.nodes, node, line)?,
        ))
    }

    fn resolve_param_given(
        &self,
        arguments: &[ast::Expression],
        line: usize,
    ) -> Result<Expression, Error> {
        match arguments {
            [ast::Expression::Name(parameter)] => {
                let index = self
                    .context
                    .module
                    .parameters
                    .iter()
                    .position(|declared| declared.name == *parameter)
                    .ok_or_else(|| Error::at(line, format!("{parameter} is not a parameter")))?;
                self.check_parameter_visible(index, parameter, line)?;
                Ok(Expression::Given(index))
            }
            _ => Err(Error::at(line, "$param_given takes one parameter".into())),
        }
    }

    fn resolve_simparam(
        &self,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        match arguments {
            [ast::Expression::Text(simparam), ..] if simparam == "gmin" => {
                if let Some(owner) = self.constant_owner {
                    return Err(Error::at(
                        line,
                        format!("{owner} reads $simparam(\"gmin\"), which is not a constant"),
                    ));
                }
                Ok(Expression::Gmin)
            }
            [ast::Expression::Text(simparam), ..] if simparam == "small_resistance" => {
                Ok(Expression::SmallResistance)
            }
            // A name the simulator does not define reads as the default given, as the
            // standard has it; jolty defines no other.
            [ast::Expression::Text(_), default] => self.expression(default, line, sink),
            [ast::Expression::Text(simparam)] => Err(Error::at(
                line,
                format!("jolty defines no $simparam {simparam:?}"),
            )),
            _ => Err(Error::at(line, "$simparam takes a name".into())),
        }
    }

    /// `$limit(arguments)`: a voltage limited from its last evaluation's, by an analog
    /// function of the module's or one of the simulator's, named as a string.
    fn limit(
        &self,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        let (access, rest) = arguments
            .split_first()
            .ok_or_else(|| Error::at(line, "$limit takes a probe".into()))?;
        let Expression::Probe(Probe::Voltage(plus, minus)) = self.expression(access, line, sink)?
        else {
            return Err(Error::at(line, "$limit limits a voltage".into()));
        };
        let (function, rest) = match rest.split_first() {
            Some((ast::Expression::Text(name), rest)) => match name.as_str() {
                "pnjlim" => (Limiter::Junction, rest),
                "fetlim" => (Limiter::Gate, rest),
                _ => {
                    return Err(Error::at(
                        line,
                        format!("the limiting function {name:?} is not jolty's"),
                    ));
                }
            },
            Some((ast::Expression::Name(name), rest)) => {
                let function = self
                    .context
                    .module
                    .functions
                    .iter()
                    .position(|declared| declared.name == *name)
                    .ok_or_else(|| Error::at(line, format!("{name} is not an analog function")))?;
                (Limiter::Function(function), rest)
            }
            _ => return Err(Error::at(line, "$limit names its limiting function".into())),
        };
        let arity = match function {
            Limiter::Junction => Some(2),
            Limiter::Gate => Some(1),
            Limiter::Function(_) => None,
        };
        if arity.is_some_and(|arity| arity != rest.len()) {
            return Err(Error::at(
                line,
                "the limiting function takes other arguments".into(),
            ));
        }
        let arguments = rest
            .iter()
            .map(|argument| self.expression(argument, line, sink))
            .collect::<Result<_, _>>()?;
        sink.limits.push(Limit {
            plus,
            minus,
            function,
            arguments,
        });
        Ok(Expression::Probe(Probe::Limited(sink.limits.len() - 1)))
    }

    /// A call of the analog function `function` with `arguments`, inlined: its inputs
    /// assigned and its body run before the statement that calls it, in variables of
    /// its own, and its result read.
    fn inline(
        &self,
        function: &'a ast::Function,
        arguments: &[ast::Expression],
        line: usize,
        sink: &mut Sink,
    ) -> Result<Expression, Error> {
        if arguments.len() != function.inputs.len() {
            return Err(Error::at(
                line,
                format!(
                    "{} takes {} arguments",
                    function.name,
                    function.inputs.len()
                ),
            ));
        }
        check_function_duplicates(function)?;
        sink.calls += 1;
        let call = sink.calls;
        let mut bindings = local_bindings(self.module_locals);
        let mut input_indices = Vec::with_capacity(function.inputs.len());
        for (position, (variable, kind)) in function_variables(function).into_iter().enumerate() {
            let index = sink.variable(&format!("{}{call}_{variable}", function.name), kind);
            if position < function.inputs.len() {
                input_indices.push(index);
            }
            bindings.push((variable, Binding::Variable(index)));
        }
        resolve_scoped_locals(
            self.context,
            &mut bindings,
            self.module_locals,
            &function.local_parameters,
            sink,
        )?;
        let mut inputs = Vec::new();
        for (argument, index) in arguments.iter().zip(input_indices) {
            inputs.push(Statement::Assign(
                index,
                self.expression(argument, line, sink)?,
            ));
        }
        let scope = Scope::block(self.context, &bindings, self.module_locals, false);
        let outer = std::mem::take(&mut sink.prelude);
        let body = scope.statements(&function.body, sink)?;
        sink.prelude = outer;
        sink.prelude.extend(inputs);
        sink.prelude.extend(body);
        let result = scope.variable(&function.name, line)?;
        Ok(Expression::Variable(result))
    }
}
