//! IR traversal combinators, static/dynamic dependence analysis, variable versioning,
//! conditional numbering, and dynamic warning classification.

use std::collections::BTreeSet;

use crate::ast::Type;

use super::{Expression, GROUND, Module, Probe, Statement, apart};

impl Expression {
    /// Short-circuiting pre-order traversal over `self` and its subexpressions.
    fn any_mut(&self, stop: &mut impl FnMut(&Self) -> bool) -> bool {
        if stop(self) {
            return true;
        }
        match self {
            Self::Call(_, arguments) => arguments.iter().any(|inner| inner.any_mut(stop)),
            Self::Unary(_, operand) | Self::Ddx(operand, _) => operand.any_mut(stop),
            Self::Binary(_, left, right) => left.any_mut(stop) || right.any_mut(stop),
            Self::Conditional(condition, then, otherwise) => {
                condition.any_mut(stop) || then.any_mut(stop) || otherwise.any_mut(stop)
            }
            _ => false,
        }
    }

    /// Calls `visit` on the expression and on each within it, in pre-order.
    pub(crate) fn walk(&self, visit: &mut impl FnMut(&Self)) {
        self.any_mut(&mut |inner| {
            visit(inner);
            false
        });
    }

    /// Returns whether `predicate` holds for this expression or any subexpression within it.
    pub(crate) fn any(&self, predicate: &impl Fn(&Self) -> bool) -> bool {
        self.any_mut(&mut |inner| predicate(inner))
    }

    /// Calls `visit` on `self` in pre-order; recurses into child expressions only when
    /// `visit` returns `false`.
    pub(super) fn transform_mut(&mut self, visit: &mut impl FnMut(&mut Self) -> bool) {
        if visit(self) {
            return;
        }
        match self {
            Self::Unary(_, operand) | Self::Ddx(operand, _) => operand.transform_mut(visit),
            Self::Binary(_, left, right) => {
                left.transform_mut(visit);
                right.transform_mut(visit);
            }
            Self::Conditional(condition, then, otherwise) => {
                condition.transform_mut(visit);
                then.transform_mut(visit);
                otherwise.transform_mut(visit);
            }
            Self::Call(_, arguments) => {
                for argument in arguments {
                    argument.transform_mut(visit);
                }
            }
            _ => {}
        }
    }
}

impl Statement {
    /// Short-circuiting pre-order traversal over `self` and all nested statements.
    fn any_mut(&self, stop: &mut impl FnMut(&Self) -> bool) -> bool {
        if stop(self) {
            return true;
        }
        match self {
            Self::If(_, _, then, otherwise) => then
                .iter()
                .chain(otherwise)
                .any(|inner| inner.any_mut(stop)),
            Self::While(_, body) => body.iter().any(|inner| inner.any_mut(stop)),
            Self::Assign(..) | Self::Contribute { .. } | Self::Fail(..) | Self::Warn(..) => false,
        }
    }

    /// Calls `visit` on `self` and every nested statement within it, in pre-order.
    pub(crate) fn visit(&self, visit: &mut impl FnMut(&Self)) {
        self.any_mut(&mut |inner| {
            visit(inner);
            false
        });
    }

    /// Calls `visit` on every statement in `statements` and every nested statement, in
    /// pre-order.
    pub(crate) fn visit_stmts(statements: &[Self], visit: &mut impl FnMut(&Self)) {
        for statement in statements {
            statement.visit(visit);
        }
    }

    /// Calls `visit` mutably on `self` and every nested statement within it, in pre-order.
    fn visit_mut(&mut self, visit: &mut impl FnMut(&mut Self)) {
        visit(self);
        match self {
            Self::If(_, _, then, otherwise) => {
                Self::visit_stmts_mut(then, visit);
                Self::visit_stmts_mut(otherwise, visit);
            }
            Self::While(_, body) => Self::visit_stmts_mut(body, visit),
            Self::Assign(..) | Self::Contribute { .. } | Self::Fail(..) | Self::Warn(..) => {}
        }
    }

    /// Calls `visit` mutably on every statement in `statements` and every nested
    /// statement, in pre-order.
    fn visit_stmts_mut(statements: &mut [Self], visit: &mut impl FnMut(&mut Self)) {
        for statement in statements {
            statement.visit_mut(visit);
        }
    }

    /// Returns whether `predicate` holds for `self` or any nested statement within it.
    pub(crate) fn any(&self, predicate: &impl Fn(&Self) -> bool) -> bool {
        self.any_mut(&mut |inner| predicate(inner))
    }

    /// Returns whether `predicate` holds for any statement in `statements` or nested
    /// within them.
    pub(crate) fn any_in(statements: &[Self], predicate: &impl Fn(&Self) -> bool) -> bool {
        statements.iter().any(|statement| statement.any(predicate))
    }

    /// Calls `visit` on each variable assigned by `self` or nested statements.
    fn targets_mut(&mut self, visit: &mut impl FnMut(&mut usize)) {
        self.visit_mut(&mut |statement| {
            if let Self::Assign(variable, _) = statement {
                visit(variable);
            }
        });
    }

    /// Calls `visit` on each expression `self` reads, and on those of the statements
    /// within it, each to change (including `reactive_factors` and `noise.exponent`).
    pub(super) fn expressions_mut(&mut self, visit: &mut impl FnMut(&mut Expression)) {
        self.visit_mut(&mut |statement| match statement {
            Self::Assign(_, value) | Self::If(_, value, ..) | Self::While(value, _) => visit(value),
            Self::Contribute {
                resistive,
                reactive,
                reactive_factors,
                noise,
                ..
            } => {
                for expr in resistive.iter_mut().chain(reactive.iter_mut()) {
                    visit(expr);
                }
                for (factor, _) in reactive_factors {
                    visit(factor);
                }
                for source in noise {
                    visit(&mut source.power);
                    if let Some(exponent) = &mut source.exponent {
                        visit(exponent);
                    }
                }
            }
            Self::Fail(_, arguments) | Self::Warn(_, arguments, _) => {
                for argument in arguments {
                    visit(argument);
                }
            }
        });
    }

    /// Calls `visit` on each expression `self` reads, and on those of the statements
    /// within it (including `reactive_factors` and `noise.exponent`).
    pub(crate) fn walk(&self, visit: &mut impl FnMut(&Expression)) {
        self.visit(&mut |statement| match statement {
            Self::Assign(_, value) | Self::If(_, value, ..) | Self::While(value, _) => visit(value),
            Self::Contribute {
                resistive,
                reactive,
                reactive_factors,
                noise,
                ..
            } => {
                for expr in resistive.iter().chain(reactive) {
                    visit(expr);
                }
                for (factor, _) in reactive_factors {
                    visit(factor);
                }
                for source in noise {
                    visit(&source.power);
                    if let Some(exponent) = &source.exponent {
                        visit(exponent);
                    }
                }
            }
            Self::Fail(_, arguments) | Self::Warn(_, arguments, _) => {
                arguments.iter().for_each(&mut *visit);
            }
        });
    }
}

/// Calls `visit` on each contribution of `statements`.
pub(crate) fn visit_contributions(statements: &[Statement], visit: &mut impl FnMut(&Statement)) {
    Statement::visit_stmts(statements, &mut |statement| {
        if matches!(statement, Statement::Contribute { .. }) {
            visit(statement);
        }
    });
}

/// Calls `visit` on each contribution of `statements`, to change it.
pub(super) fn visit_contributions_mut(
    statements: &mut [Statement],
    visit: &mut impl FnMut(&mut Statement),
) {
    Statement::visit_stmts_mut(statements, &mut |statement| {
        if matches!(statement, Statement::Contribute { .. }) {
            visit(statement);
        }
    });
}

/// Whether `statement` contributes to a branch, or holds a statement that does.
pub(crate) fn contributes(statement: &Statement) -> bool {
    statement.any(&|inner| matches!(inner, Statement::Contribute { .. }))
}

/// Gives each variable that `statements`, the analog block, assigns again where every
/// path does, at its top, a new version of it from there on, so that what is read before
/// keeps its own dependence, as a static number later made dynamic does; adds the new
/// versions to `variables`, and returns the last version of each variable and the
/// variable each version is of.
pub(super) fn version(
    statements: &mut [Statement],
    variables: &mut Vec<(String, Type)>,
) -> (Vec<usize>, Vec<usize>) {
    let mut current: Vec<usize> = (0..variables.len()).collect();
    let mut origins = current.clone();
    let mut assigned = vec![false; variables.len()];
    for statement in statements {
        if let Statement::Assign(variable, value) = statement {
            rename_reads(value, &current);
            let original = *variable;
            if assigned[original] {
                let (name, kind) = variables[original].clone();
                let unique = apart(&name, |taken| {
                    variables.iter().any(|(variable, _)| variable == taken)
                });
                variables.push((unique, kind));
                origins.push(original);
                current[original] = variables.len() - 1;
            }
            assigned[original] = true;
            *variable = current[original];
        } else {
            statement.expressions_mut(&mut |expr| rename_reads(expr, &current));
            statement.targets_mut(&mut |target| {
                assigned[*target] = true;
                *target = current[*target];
            });
        }
    }
    let mut last: Vec<usize> = current;
    last.resize(variables.len(), 0);
    for (index, version) in last.iter_mut().enumerate().skip(assigned.len()) {
        *version = index;
    }
    (last, origins)
}

/// `expr` reading each variable's version in `current`.
fn rename_reads(expr: &mut Expression, current: &[usize]) {
    expr.transform_mut(&mut |inner| {
        if let Expression::Variable(variable) = inner {
            if let Some(&version) = current.get(*variable) {
                *variable = version;
            }
            return true;
        }
        false
    });
}

/// Numbers each conditional of `statements`, from `next` on, in the order they hold them.
pub(super) fn number(statements: &mut [Statement], next: &mut usize) {
    Statement::visit_stmts_mut(statements, &mut |statement| {
        if let Statement::If(id, ..) = statement {
            *id = *next;
            *next += 1;
        }
    });
}

/// Marks each warning in `module` as dynamic when it depends on the probes through its
/// arguments or an enclosing condition.
pub(super) fn mark_dynamic_warnings(module: &mut Module) {
    let mut analog = std::mem::take(&mut module.analog);
    mark_warnings_in(module, &mut analog, false);
    module.analog = analog;
}

fn mark_warnings_in(module: &Module, statements: &mut [Statement], dynamic: bool) {
    for statement in statements {
        match statement {
            Statement::Warn(_, arguments, depends) => {
                *depends = dynamic || arguments.iter().any(|argument| module.is_dynamic(argument));
            }
            Statement::If(_, condition, then, otherwise) => {
                let inner = dynamic || module.is_dynamic(condition);
                mark_warnings_in(module, then, inner);
                mark_warnings_in(module, otherwise, inner);
            }
            Statement::While(condition, body) => {
                let inner = dynamic || module.is_dynamic(condition);
                mark_warnings_in(module, body, inner);
            }
            Statement::Assign(..) | Statement::Contribute { .. } | Statement::Fail(..) => {}
        }
    }
}

/// Finds which variables are dynamic, and the probes each depends on.
pub(super) fn analyze(module: &mut Module) {
    let count = module.variables.len();
    module.dynamic = vec![false; count];
    module.probes = vec![BTreeSet::new(); count];
    loop {
        let before = (module.dynamic.clone(), module.probes.clone());
        let analog = std::mem::take(&mut module.analog);
        flow(module, &analog, false);
        // A static variable that dynamic code reads keeps one value for it, which
        // preparation leaves; one assigned again after such a read is dynamic.
        let mut read = BTreeSet::new();
        reassigned(module, &analog, false, &mut read);
        module.analog = analog;
        if (module.dynamic.clone(), module.probes.clone()) == before {
            return;
        }
    }
}

impl Module {
    /// The name of `node`, `ground` for the ground.
    pub(crate) fn node_name(&self, node: usize) -> &str {
        if node == GROUND {
            "ground"
        } else {
            &self.nodes[node]
        }
    }

    /// Whether `node` is one of the internal nodes, neither a port nor the ground.
    pub(crate) fn is_internal(&self, node: usize) -> bool {
        node != GROUND && node >= self.ports
    }

    /// Whether `expr` is dynamic.
    pub(crate) fn is_dynamic(&self, expr: &Expression) -> bool {
        expr.any(&|inner| match inner {
            Expression::Probe(_)
            | Expression::Gmin
            | Expression::Ddx(..)
            | Expression::Shape(_) => true,
            Expression::Variable(variable) => self.dynamic[*variable],
            _ => false,
        })
    }

    /// The probes `expr` depends on through its value.
    pub(crate) fn probes_of(&self, expr: &Expression) -> BTreeSet<Probe> {
        let mut probes = BTreeSet::new();
        self.collect_probes(expr, &mut probes);
        probes
    }

    fn collect_probes(&self, expr: &Expression, probes: &mut BTreeSet<Probe>) {
        match expr {
            Expression::Probe(probe) => {
                probes.insert(*probe);
            }
            Expression::Variable(variable) => {
                probes.extend(self.probes[*variable].iter().copied());
            }
            Expression::Conditional(_, then, otherwise) => {
                self.collect_probes(then, probes);
                self.collect_probes(otherwise, probes);
            }
            Expression::Unary("-" | "+", operand) => self.collect_probes(operand, probes),
            Expression::Binary("+" | "-" | "*" | "/", left, right) => {
                self.collect_probes(left, probes);
                self.collect_probes(right, probes);
            }
            Expression::Call(_, arguments) => {
                for argument in arguments {
                    self.collect_probes(argument, probes);
                }
            }
            // Comparisons, logical operations, conditions, `ddx`, and leaves have no derivative.
            _ => {}
        }
    }
}

/// Propagates dependence through `statements`, under a condition that is dynamic when
/// `control` is.
fn flow(module: &mut Module, statements: &[Statement], control: bool) {
    for statement in statements {
        match statement {
            Statement::Assign(variable, value) => {
                if control || module.is_dynamic(value) {
                    module.dynamic[*variable] = true;
                }
                if module.variables[*variable].1 == Type::Real {
                    let probes = module.probes_of(value);
                    module.probes[*variable].extend(probes);
                }
            }
            Statement::If(_, condition, then, otherwise) => {
                let control = control || module.is_dynamic(condition);
                flow(module, then, control);
                flow(module, otherwise, control);
            }
            Statement::While(condition, body) => {
                let control = control || module.is_dynamic(condition);
                flow(module, body, control);
            }
            Statement::Contribute { .. } | Statement::Fail(..) | Statement::Warn(..) => {}
        }
    }
}

/// Marks dynamic each static variable assigned after dynamic code read it, `read` holding
/// those read so far; within a loop, the body runs twice, as a second iteration reads
/// what the first assigned.
fn reassigned(
    module: &mut Module,
    statements: &[Statement],
    dynamic: bool,
    read: &mut BTreeSet<usize>,
) {
    let reads = |module: &Module, expr: &Expression, read: &mut BTreeSet<usize>| {
        expr.walk(&mut |inner| {
            if let Expression::Variable(variable) = inner
                && !module.dynamic[*variable]
            {
                read.insert(*variable);
            }
        });
    };
    for statement in statements {
        match statement {
            Statement::Assign(variable, value) => {
                if module.dynamic[*variable] || dynamic {
                    reads(module, value, read);
                } else if read.contains(variable) {
                    module.dynamic[*variable] = true;
                }
            }
            Statement::Contribute { .. } => statement.walk(&mut |expr| reads(module, expr, read)),
            Statement::If(_, condition, then, otherwise) => {
                let inner = dynamic || module.is_dynamic(condition);
                if inner || then.iter().chain(otherwise).any(contributes) {
                    reads(module, condition, read);
                }
                reassigned(module, then, inner, read);
                reassigned(module, otherwise, inner, read);
            }
            Statement::While(condition, body) => {
                let inner = dynamic || module.is_dynamic(condition);
                if inner {
                    reads(module, condition, read);
                }
                for _ in 0..2 {
                    reassigned(module, body, inner, read);
                }
            }
            Statement::Fail(..) | Statement::Warn(..) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::compile;
    use crate::lexer::tokens;
    use crate::parser::parse;

    use super::super::{
        Expression, LocalParameter, Module, Statement, constant, probed_currents, resolve,
    };

    const ADVICE_WORDS: [&str; 7] = [
        "use",
        "pass",
        "try",
        "instead",
        "should",
        "give",
        "did you mean",
    ];

    fn assert_factual(message: &str) {
        let lower = message.to_ascii_lowercase();
        for forbidden in ADVICE_WORDS {
            let matches_word = if forbidden.contains(' ') {
                lower.contains(forbidden)
            } else {
                lower
                    .split(|character: char| !character.is_ascii_alphanumeric())
                    .any(|word| word == forbidden)
            };
            assert!(
                !matches_word,
                "error message contains advice word {forbidden:?}: {message:?}"
            );
        }
    }

    /// The message the compiler refuses `analog`, the analog block of a two-terminal
    /// module with variables `x` and `y`, with.
    fn refusal(analog: &str) -> String {
        let source = format!(
            "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\nreal x, y;\nanalog begin\n{analog}\nend\nendmodule\n"
        );
        let message = compile(&source, "m.va", &|_| None)
            .expect_err("refused")
            .message;
        assert_factual(&message);
        message
    }

    fn module_refusal(body: &str) -> String {
        let source = format!(
            "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\n{body}\nendmodule\n"
        );
        let message = compile(&source, "m.va", &|_| None)
            .expect_err("refused")
            .message;
        assert_factual(&message);
        message
    }

    fn resolve_source(body: &str) -> Module {
        let source = format!(
            "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\n{body}\nendmodule\n"
        );
        let tokens = tokens(&source, &|_| None).expect("tokens");
        let parsed = parse(&tokens).expect("parse");
        resolve(&parsed).expect("resolve")
    }

    /// Code under an initial event may read neither a probe nor what other code assigns,
    /// and no other code may make what it assigns depend on the probes.
    #[test]
    fn initial_code_reads_only_what_its_first_step_fixes() {
        assert_eq!(
            refusal("@(initial_step) x = V(a, b);\nI(a, b) <+ x;"),
            "code under an initial event reads the probes through V, which its first step alone fixes"
        );
        assert_eq!(
            refusal("y = 2;\n@(initial_step) x = y;\nI(a, b) <+ x * V(a, b);"),
            "code under an initial event reads y, which no initial code assigned before"
        );
        assert_eq!(
            refusal("@(initial_step) x = 1;\nif (V(a, b) > 0) x = 2;\nI(a, b) <+ x * V(a, b);"),
            "x, which code under an initial event assigns, depends on the probes, which is not compiled yet"
        );
        assert_eq!(
            refusal("@(final_step) x = 1;\nI(a, b) <+ x * V(a, b);"),
            "the event final_step is not compiled yet"
        );
    }

    #[test]
    fn localparam_duplicate_and_collision_errors() {
        assert_eq!(
            module_refusal(
                "localparam real k = 1.0;\nlocalparam real k = 2.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "the localparam k is declared more than once"
        );
        assert_eq!(
            module_refusal(
                "parameter real k = 1.0;\nlocalparam real k = 2.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k is declared as both a parameter and a localparam"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nparameter real k = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k is declared as both a parameter and a localparam"
        );
        assert_eq!(
            module_refusal("localparam real k = 2.0;\nreal k;\nanalog I(a, b) <+ V(a, b);"),
            "k is declared as both a localparam and a variable"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 1.0;\naliasparam k = p;\nlocalparam real k = 2.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k is declared as both a localparam and an aliasparam"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\naliasparam alias_k = k;\nanalog I(a, b) <+ V(a, b);"
            ),
            "the alias alias_k targets localparam k, not a parameter"
        );
        assert_eq!(
            module_refusal(
                "analog begin : blk\nlocalparam real k = 1.0;\nlocalparam real k = 2.0;\nI(a, b) <+ V(a, b);\nend"
            ),
            "the localparam k is declared more than once"
        );
        assert_eq!(
            module_refusal(
                "analog begin : blk\nlocalparam real k = 1.0;\nreal k;\nI(a, b) <+ V(a, b);\nend"
            ),
            "k is declared as both a localparam and a variable"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nlocalparam real k = 1.0;\nlocalparam real k = 2.0;\nf = x;\nendfunction\nanalog I(a, b) <+ f(V(a, b));"
            ),
            "the localparam k is declared more than once"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nlocalparam real x = 1.0;\nf = x;\nendfunction\nanalog I(a, b) <+ f(V(a, b));"
            ),
            "x is declared as both a localparam and a variable"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nlocalparam real f = 1.0;\nf = x;\nendfunction\nanalog I(a, b) <+ f(V(a, b));"
            ),
            "f is declared as both a localparam and a variable"
        );
    }

    #[test]
    fn localparam_ordering_scope_and_mutability_errors() {
        assert_eq!(
            module_refusal("localparam real k = missing;\nanalog I(a, b) <+ V(a, b);"),
            "missing is not declared"
        );
        assert_eq!(
            module_refusal("localparam real k = k + 1.0;\nanalog I(a, b) <+ V(a, b);"),
            "k reads k, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real first = second;\nlocalparam real second = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "first reads second, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = p;\nparameter real p = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "k reads p, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = k;\nlocalparam real k = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "p's default reads k, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam integer has_p = $param_given(p);\nparameter real p = 1.0;\nanalog I(a, b) <+ V(a, b);"
            ),
            "has_p reads p, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nanalog begin : blk\nlocalparam real a = k;\nlocalparam real k = 3.0;\nI(a, b) <+ V(a, b) * a;\nend"
            ),
            "a reads k, which is not declared before it"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nanalog begin\nk = 3.0;\nI(a, b) <+ V(a, b);\nend"
            ),
            "k is a localparam, not a variable"
        );
        assert_eq!(
            module_refusal(
                "localparam real k = 2.0;\nreal g;\nanalog begin\ng = $param_given(k);\nI(a, b) <+ V(a, b) * g;\nend"
            ),
            "k is not a parameter"
        );
    }

    #[test]
    fn localparam_constant_expression_and_type_errors() {
        assert_eq!(
            module_refusal("real x;\nlocalparam real k = x;\nanalog I(a, b) <+ V(a, b);"),
            "k reads the variable x, which is not a constant"
        );
        assert_eq!(
            module_refusal(
                "analog begin : blk\nlocalparam real k = y;\nreal y;\nI(a, b) <+ V(a, b) * k;\nend"
            ),
            "k reads the variable y, which is not a constant"
        );
        assert_eq!(
            module_refusal("localparam real k = V(a, b);\nanalog I(a, b) <+ V(a, b);"),
            "k reads the probes through V, which is not a constant"
        );
        assert_eq!(
            module_refusal("localparam real k = $simparam(\"gmin\");\nanalog I(a, b) <+ V(a, b);"),
            "k reads $simparam(\"gmin\"), which is not a constant"
        );
        assert_eq!(
            module_refusal(
                "analog function real f;\ninput x;\nf = x + 1.0;\nendfunction\nlocalparam real k = f(1.0);\nanalog I(a, b) <+ V(a, b);"
            ),
            "k calls the analog function f, which is not compiled yet"
        );
        assert_eq!(
            module_refusal("localparam label = \"hello\";\nanalog I(a, b) <+ V(a, b);"),
            "a string is not a number"
        );
        assert_eq!(
            module_refusal("localparam integer n = 1.5;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n has a real initializer"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 1.0;\nlocalparam integer n = p;\nanalog I(a, b) <+ V(a, b);"
            ),
            "the integer localparam n has a real initializer"
        );
        assert_eq!(
            module_refusal("localparam integer n = 0 ? 1.5 : 2;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n has a real initializer"
        );
        assert_eq!(
            module_refusal("localparam integer n = 1 / 0;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n divides by zero"
        );
        assert_eq!(
            module_refusal("localparam real x = 5 / 0;\nanalog I(a, b) <+ V(a, b);"),
            "the localparam x divides by zero"
        );
        assert_eq!(
            module_refusal("localparam integer n = 2147483647 + 1;\nanalog I(a, b) <+ V(a, b);"),
            "the integer localparam n overflows an integer"
        );
        assert_eq!(
            module_refusal(
                "localparam integer n = -(-2147483647 - 1);\nanalog I(a, b) <+ V(a, b);"
            ),
            "the integer localparam n overflows an integer"
        );
        assert_eq!(
            module_refusal("localparam real x = 1.0 / 0.0;\nanalog I(a, b) <+ V(a, b);"),
            "the localparam x evaluates to a non-finite value"
        );
        assert_eq!(
            module_refusal("localparam real x = sqrt(-1.0);\nanalog I(a, b) <+ V(a, b);"),
            "the localparam x evaluates to a non-finite value"
        );
        assert_eq!(
            module_refusal("parameter real p = sqrt(-1.0);\nanalog I(a, b) <+ V(a, b);"),
            "p's default evaluates to a non-finite value"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 1.0 from (0:sqrt(-1.0));\nanalog I(a, b) <+ V(a, b);"
            ),
            "p's range evaluates to a non-finite value"
        );
    }

    #[test]
    fn limiting_function_rejects_parameter_dependent_localparam_cleanly() {
        assert_eq!(
            module_refusal(
                "parameter real p = 2.0;\nlocalparam real k = 2.0 * p;\nanalog function real lim;\ninput vnew, vold;\nreal vnew, vold;\nlim = vnew * k;\nendfunction\nreal v;\nanalog begin\nv = $limit(V(a, b), lim);\nI(a, b) <+ v;\nend"
            ),
            "the limiting function reads localparam k, which depends on parameters and is not compiled yet"
        );
        assert_eq!(
            module_refusal(
                "parameter real p = 2.0;\nanalog function real lim;\ninput vnew, vold;\nlocalparam real scale = p * 2.0;\nreal vnew, vold;\nlim = vnew * scale;\nendfunction\nreal v;\nanalog begin\nv = $limit(V(a, b), lim);\nI(a, b) <+ v;\nend"
            ),
            "the limiting function reads localparam scale, which depends on parameters and is not compiled yet"
        );
    }

    #[test]
    fn localparam_constant_folding_handles_builtins_truncation_and_short_circuit() {
        let module = resolve_source(
            "localparam integer truncated = 5 / 2;\n\
             localparam real truncated_real = 5 / 2;\n\
             localparam real guarded = 0 ? (1.0 / 0.0) : (sqrt(4.0) + ln(exp(1.0)) + floor(3.7) + ceil(1.2));\n\
             localparam integer short_and = 0 && (1 / 0);\n\
             localparam integer short_or = 1 || (1 / 0);\n\
             localparam untyped_int = truncated + short_or;\n\
             localparam untyped_real = guarded + truncated_real;\n\
             analog I(a, b) <+ V(a, b) * (untyped_real + untyped_int + short_and);",
        );
        // Every pure constant folds directly with zero prepared localparams.
        assert_eq!(module.local_parameters, [] as [LocalParameter; 0]);
        // `truncated = 2`, `truncated_real = 2.0`, `guarded = 2 + 1 + 3 + 2 = 8.0`,
        // `short_and = 0`, `short_or = 1`, `untyped_int = 3`, `untyped_real = 10.0`.
        let Statement::Contribute {
            resistive: Some(Expression::Binary("*", _, right)),
            ..
        } = &module.analog[0]
        else {
            panic!("expected resistive contribution");
        };
        let value = constant(right).expect("folded constant");
        assert!((value - 13.0).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn static_scaled_ddt_accepts_static_and_refuses_dynamic_factors() {
        for valid in [
            "x = 1e-12;\nI(a, b) <+ x * ddt(V(a, b));",
            "x = 1e-12;\nI(a, b) <+ ddt(V(a, b)) * x;",
            "x = 2.0;\nI(a, b) <+ ddt(V(a, b)) / x;",
            "x = 1e-12;\nI(a, b) <+ -(x * ddt(V(a, b)));",
            "x = 1e-12;\nI(a, b) <+ x * (-ddt(V(a, b)));",
            "x = 1e-12;\nI(a, b) <+ x * (V(a, b) / 100.0 + ddt(V(a, b)));",
            "x = 1e-12;\ny = 2.0;\nI(a, b) <+ (ddt(V(a, b)) * x) / y;",
            "x = 1e-12;\nI(a, b) <+ x * ddt(V(a, b));\nx = V(a, b);",
            "x = 1e-9;\nV(a, b) <+ x * ddt(I(a, b));",
        ] {
            let source = format!(
                "`include \"disciplines.vams\"\nmodule m(a, b);\ninout a, b;\nelectrical a, b;\nreal x, y;\nanalog begin\n{valid}\nend\nendmodule\n"
            );
            compile(&source, "m.va", &|_| None)
                .unwrap_or_else(|error| panic!("expected {valid:?} to compile, got: {error}"));
        }

        for dynamic_scale in [
            "I(a, b) <+ (1.0 + V(a, b)) * ddt(V(a, b));",
            "x = V(a, b);\nI(a, b) <+ x * ddt(V(a, b));",
            "if (V(a, b) > 0) x = 2e-12; else x = 1e-12;\nI(a, b) <+ x * ddt(V(a, b));",
            "I(a, b) <+ $simparam(\"gmin\") * ddt(V(a, b));",
        ] {
            assert_eq!(
                refusal(dynamic_scale),
                "ddt scaled by a dynamic expression is not charge-conserving and is not compiled yet",
                "{dynamic_scale}"
            );
        }

        assert_eq!(
            refusal("I(a, b) <+ ddt(V(a, b)) * ddt(V(a, b));"),
            "ddt is not compiled yet"
        );
        assert_eq!(
            refusal("I(a, b) <+ 1.0 / ddt(V(a, b));"),
            "ddt is not compiled yet"
        );
        assert_eq!(
            refusal("x = 2.0;\nI(a, b) <+ x * white_noise(1.0);"),
            "white_noise is not compiled yet"
        );
    }

    #[test]
    fn statement_walk_visits_flicker_noise_exponent() {
        let reassigned_module = resolve_source(
            "parameter real flag = 1.0;\n\
             real e;\n\
             analog begin\n\
                 e = 1.0;\n\
                 I(a, b) <+ V(a, b) / 100.0;\n\
                 I(a, b) <+ flicker_noise(1e-20, e, \"flicker\");\n\
                 if (flag > 0.0) e = 2.0;\n\
             end",
        );
        let e_index = reassigned_module
            .variables
            .iter()
            .position(|(name, _)| name == "e")
            .expect("variable e");
        assert!(
            reassigned_module.dynamic[e_index],
            "reassigned variable read by flicker_noise exponent must be marked dynamic"
        );
        assert_eq!(
            refusal(
                "y = 1.0;\n\
                 x = 1.0;\n\
                 I(a, b) <+ V(a, b) / 100.0;\n\
                 I(a, b) <+ flicker_noise(1e-20, x, \"flicker\");\n\
                 if (y > 0.0) x = 2.0;"
            ),
            "a flicker exponent that depends on the probes is not compiled yet"
        );

        let probed_module = resolve_source(
            "branch (a, b) br;\n\
             analog begin\n\
                 V(br) <+ 1.0;\n\
                 I(a, b) <+ flicker_noise(1e-20, I(br), \"flicker\");\n\
             end",
        );
        let br_index = probed_module
            .branches
            .iter()
            .position(|branch| branch.name == "br")
            .expect("branch br");
        assert!(
            probed_currents(&probed_module).contains(&br_index),
            "probed current inside flicker_noise exponent must be discovered by Statement::walk"
        );
    }
}
