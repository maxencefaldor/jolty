//! Untyped abstract syntax tree for a parsed Verilog-A module before name resolution.

/// Parsed Verilog-A device module.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Module {
    /// Module name carried over to the emitted kernel.
    pub(crate) name: String,
    /// Module ports in declaration order, matching the kernel terminals.
    pub(crate) ports: Vec<String>,
    /// Nodes declared with a discipline (including ports) paired with their discipline names.
    pub(crate) nodes: Vec<(String, String)>,
    /// Named branches as `(name, plus_node, minus_node)`, with an empty `minus_node` for ground.
    pub(crate) branches: Vec<(String, String, String)>,
    /// Model parameters in declaration order.
    pub(crate) parameters: Vec<Parameter>,
    /// Module-scope `localparam` constants in declaration order.
    pub(crate) local_parameters: Vec<LocalParameter>,
    /// Interleaved `parameter` and `localparam` declarations in exact source order.
    pub(crate) declarations: Vec<Declaration>,
    /// Parameter aliases as `(alias_name, target_parameter)`.
    pub(crate) aliases: Vec<(String, String)>,
    /// Module variables paired with their declared types.
    pub(crate) variables: Vec<(String, Type)>,
    /// Output variables annotated with `desc` as `(name, description, optional_units)`.
    pub(crate) outputs: Vec<(String, String, Option<String>)>,
    /// Analog functions declared in the module.
    pub(crate) functions: Vec<Function>,
    /// Body of the `analog` block.
    pub(crate) analog: Statement,
}

/// Scalar numeric type in Verilog-A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type {
    /// Real floating-point number.
    Real,
    /// Integer value.
    Integer,
}

/// Parsed model parameter declaration.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parameter {
    pub(crate) name: String,
    pub(crate) kind: Type,
    pub(crate) default: Expression,
    /// Admitted (`from`) and disallowed (`exclude`) parameter value ranges.
    pub(crate) ranges: Vec<Range>,
    /// Optional `desc` attribute text.
    pub(crate) description: Option<String>,
    /// Optional `units` attribute text.
    pub(crate) units: Option<String>,
    pub(crate) line: usize,
}

/// Module-level constant declaration in source order, indexing into [`Module::parameters`] or
/// [`Module::local_parameters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declaration {
    Parameter(usize),
    LocalParameter(usize),
}

/// Parsed `localparam` declaration: an immutable named constant not exposed as a model parameter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalParameter {
    pub(crate) name: String,
    /// Optional declared type; `None` infers the type from the initializer.
    pub(crate) kind: Option<Type>,
    pub(crate) value: Expression,
    pub(crate) line: usize,
}

/// Value range from a `from` or `exclude` clause, with optional inclusive or exclusive bounds
/// (`None` denotes infinity).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Range {
    /// Flag set for `exclude` ranges and cleared for `from` ranges.
    pub(crate) exclude: bool,
    pub(crate) lower: Option<(Expression, bool)>,
    pub(crate) upper: Option<(Expression, bool)>,
}

/// Parsed analog function with its name, input parameters, local declarations, and body.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) local_parameters: Vec<LocalParameter>,
    pub(crate) variables: Vec<(String, Type)>,
    pub(crate) body: Statement,
}

/// Statement inside an analog block or analog function.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Statement {
    Block(Vec<Statement>),
    /// Named block with block-scoped constants and variables.
    Named {
        name: String,
        local_parameters: Vec<LocalParameter>,
        variables: Vec<(String, Type)>,
        statements: Vec<Statement>,
    },
    Assign {
        target: String,
        value: Expression,
        line: usize,
    },
    /// Branch contribution `nature(nodes_or_branch) <+ value`.
    Contribute {
        nature: String,
        operands: Vec<String>,
        value: Expression,
        line: usize,
    },
    If {
        condition: Expression,
        then: Box<Statement>,
        otherwise: Option<Box<Statement>>,
    },
    While {
        condition: Expression,
        body: Box<Statement>,
    },
    /// Statement guarded by an analysis initialization event such as `@(initial_step)`.
    Initial {
        body: Box<Statement>,
        line: usize,
    },
    /// System task call, such as `$error`, with its arguments.
    Task {
        name: String,
        arguments: Vec<Expression>,
        line: usize,
    },
}

/// Parsed Verilog-A expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expression {
    Real(f64),
    Integer(i64),
    Text(String),
    Name(String),
    /// Call to a function, nature access function, or system function.
    Call(String, Vec<Expression>),
    Unary(&'static str, Box<Expression>),
    Binary(&'static str, Box<Expression>, Box<Expression>),
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),
}
