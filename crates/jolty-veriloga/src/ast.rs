//! The syntax of a Verilog-A module, as the parser reads it: declarations, the analog
//! block's statements and their expressions, names not yet resolved.

/// A module: one device model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Module {
    /// The module's name, which the kernel takes.
    pub(crate) name: String,
    /// The ports, in their order, which the kernel's terminals take.
    pub(crate) ports: Vec<String>,
    /// Each node declared with a discipline, ports among them, and its discipline.
    pub(crate) nodes: Vec<(String, String)>,
    /// The named branches: each one's name and its two nodes, the second empty for the
    /// ground.
    pub(crate) branches: Vec<(String, String, String)>,
    /// The parameters, in their order.
    pub(crate) parameters: Vec<Parameter>,
    /// The module-scope `localparam` constants, in their order.
    pub(crate) local_parameters: Vec<LocalParameter>,
    /// The `parameter` and `localparam` declarations in exact source order.
    pub(crate) declarations: Vec<Declaration>,
    /// Each `aliasparam`: another name and the parameter it names.
    pub(crate) aliases: Vec<(String, String)>,
    /// The variables, each with its type.
    pub(crate) variables: Vec<(String, Type)>,
    /// The variables that are output variables, as the attribute `desc` marks them, each
    /// with its description and units.
    pub(crate) outputs: Vec<(String, String, Option<String>)>,
    /// The analog functions.
    pub(crate) functions: Vec<Function>,
    /// The analog block.
    pub(crate) analog: Statement,
}

/// A number's type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Type {
    /// A real number.
    Real,
    /// An integer.
    Integer,
}

/// A parameter's declaration.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parameter {
    pub(crate) name: String,
    pub(crate) kind: Type,
    pub(crate) default: Expression,
    /// The ranges of `from`, which the value lies in, and of `exclude`, which it does not.
    pub(crate) ranges: Vec<Range>,
    /// Its `desc` attribute.
    pub(crate) description: Option<String>,
    /// Its `units` attribute.
    pub(crate) units: Option<String>,
    pub(crate) line: usize,
}

/// A module-level constant declaration in source order: either a `parameter` or a
/// `localparam`, by its index in [`Module::parameters`] or [`Module::local_parameters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Declaration {
    Parameter(usize),
    LocalParameter(usize),
}

/// A `localparam` declaration: an immutable named constant, not exposed as a model
/// parameter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LocalParameter {
    pub(crate) name: String,
    /// Its declared type, or `None` when omitted so the initializer decides it.
    pub(crate) kind: Option<Type>,
    pub(crate) value: Expression,
    pub(crate) line: usize,
}

/// A range of `from` or `exclude`, each bound included or not, `None` for an infinite one.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Range {
    /// Whether the range excludes rather than admits its values.
    pub(crate) exclude: bool,
    pub(crate) lower: Option<(Expression, bool)>,
    pub(crate) upper: Option<(Expression, bool)>,
}

/// An analog function: its name, its inputs in order, its local constants and variables
/// with their types, and its body, which assigns its result to its name.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Function {
    pub(crate) name: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) local_parameters: Vec<LocalParameter>,
    pub(crate) variables: Vec<(String, Type)>,
    pub(crate) body: Statement,
}

/// A statement of the analog block, with the line it starts on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Statement {
    Block(Vec<Statement>),
    /// A named block, its own constants and variables declared within it.
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
    /// A contribution `nature(nodes or branch) <+ value`.
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
    /// A statement under the event of an analysis's first step, `@(initial_step)` or the
    /// like, which runs at that step only.
    Initial {
        body: Box<Statement>,
        line: usize,
    },
    /// A system task, such as `$error`, and its arguments.
    Task {
        name: String,
        arguments: Vec<Expression>,
        line: usize,
    },
}

/// An expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expression {
    Real(f64),
    Integer(i64),
    Text(String),
    Name(String),
    /// A call of a function, an access function or a system function, by name.
    Call(String, Vec<Expression>),
    Unary(&'static str, Box<Expression>),
    Binary(&'static str, Box<Expression>, Box<Expression>),
    Conditional(Box<Expression>, Box<Expression>, Box<Expression>),
}
