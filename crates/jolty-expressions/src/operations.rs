//! Scalar evaluation and first and second partial derivatives for expression operations.

use std::cmp::Ordering;

/// Defines [`Unary`] along with its `name`, `named`, and `value` dispatch for the standard
/// `f64` methods and the non-method unary operations.
macro_rules! define_unary {
    ($($variant:ident => $method:ident $(| $alias:literal)*),* $(,)?) => {
        /// An operation of one operand.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub(crate) enum Unary {
            Negate,
            Not,
            $($variant,)*
            Sign,
            Step,
            Ramp,
        }

        impl Unary {
            /// Returns the source name of this unary operation (`-` for negation, `!` for
            /// logical not, and `ln` for natural logarithm).
            pub(crate) const fn name(self) -> &'static str {
                match self {
                    Self::Negate => "-",
                    Self::Not => "!",
                    $(Self::$variant => stringify!($method),)*
                    Self::Sign => "sgn",
                    Self::Step => "u",
                    Self::Ramp => "uramp",
                }
            }

            /// Looks up a single-argument unary function by `name`, including the `log` alias
            /// for `ln`.
            pub(crate) fn named(name: &str) -> Option<Self> {
                Some(match name {
                    $(stringify!($method) $(| $alias)* => Self::$variant,)*
                    "sgn" => Self::Sign,
                    "u" => Self::Step,
                    "uramp" => Self::Ramp,
                    _ => return None,
                })
            }

            /// Evaluates this unary operation at `x`.
            pub(crate) fn value(self, x: f64) -> f64 {
                match self {
                    Self::Negate => -x,
                    Self::Not => truth(x == 0.0),
                    $(Self::$variant => x.$method(),)*
                    Self::Sign => sign(x),
                    Self::Step => {
                        if x > 0.0 {
                            1.0
                        } else if x < 0.0 {
                            0.0
                        } else {
                            0.5
                        }
                    }
                    Self::Ramp => x.max(0.0),
                }
            }
        }
    };
}

define_unary! {
    Abs => abs,
    Sqrt => sqrt,
    Exp => exp,
    Ln => ln | "log",
    Log10 => log10,
    Sin => sin,
    Cos => cos,
    Tan => tan,
    Asin => asin,
    Acos => acos,
    Atan => atan,
    Sinh => sinh,
    Cosh => cosh,
    Tanh => tanh,
    Asinh => asinh,
    Acosh => acosh,
    Atanh => atanh,
    Floor => floor,
    Ceil => ceil,
}

impl Unary {
    /// Computes the first derivative of this unary operation at `x` given its evaluated
    /// `value`.
    pub(crate) fn slope(self, x: f64, value: f64) -> f64 {
        match self {
            Self::Negate => -1.0,
            Self::Not | Self::Floor | Self::Ceil | Self::Sign | Self::Step => 0.0,
            Self::Abs => sign(x),
            Self::Sqrt => 0.5 / value,
            Self::Exp => value,
            Self::Ln => 1.0 / x,
            Self::Log10 => 1.0 / (x * std::f64::consts::LN_10),
            Self::Sin => x.cos(),
            Self::Cos => -x.sin(),
            Self::Tan => 1.0 + value * value,
            Self::Asin => 1.0 / (1.0 - x * x).sqrt(),
            Self::Acos => -1.0 / (1.0 - x * x).sqrt(),
            Self::Atan => 1.0 / (1.0 + x * x),
            Self::Sinh => x.cosh(),
            Self::Cosh => x.sinh(),
            Self::Tanh => 1.0 - value * value,
            Self::Asinh => 1.0 / (x * x + 1.0).sqrt(),
            Self::Acosh => 1.0 / (x * x - 1.0).sqrt(),
            Self::Atanh => 1.0 / (1.0 - x * x),
            Self::Ramp => truth(x > 0.0),
        }
    }

    /// Computes the second derivative of this unary operation at `x` given its `value` and
    /// first derivative `slope`.
    pub(crate) fn curvature(self, x: f64, value: f64, slope: f64) -> f64 {
        match self {
            Self::Negate
            | Self::Not
            | Self::Abs
            | Self::Floor
            | Self::Ceil
            | Self::Sign
            | Self::Step
            | Self::Ramp => 0.0,
            Self::Sqrt => -0.5 * slope / x,
            Self::Exp | Self::Sinh | Self::Cosh => value,
            Self::Ln => -slope * slope,
            Self::Log10 => -slope / x,
            Self::Sin | Self::Cos => -value,
            Self::Tan => 2.0 * value * slope,
            Self::Asin | Self::Acos => slope * x / (1.0 - x * x),
            Self::Atan => -2.0 * x * slope * slope,
            Self::Tanh => -2.0 * value * slope,
            Self::Asinh => -slope * x / (x * x + 1.0),
            Self::Acosh => -slope * x / (x * x - 1.0),
            Self::Atanh => 2.0 * x * slope * slope,
        }
    }
}

/// An operation of two operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Min,
    Max,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
}

impl Binary {
    /// Returns the source name of this binary operation (`^` for exponentiation, operator
    /// symbols, or `min`/`max`).
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Power => "^",
            Self::Min => "min",
            Self::Max => "max",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::And => "&&",
            Self::Or => "||",
        }
    }

    /// Looks up a two-argument binary function (`min`, `max`, or `pow`) by `name`.
    pub(crate) fn named(name: &str) -> Option<Self> {
        Some(match name {
            "min" => Self::Min,
            "max" => Self::Max,
            "pow" => Self::Power,
            _ => return None,
        })
    }

    /// Returns whether this binary operation is an ordering or equality comparison.
    pub(crate) const fn compares(self) -> bool {
        matches!(
            self,
            Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
                | Self::Equal
                | Self::NotEqual
        )
    }

    /// Evaluates this binary operation at operands `a` and `b`.
    pub(crate) fn value(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Subtract => a - b,
            Self::Multiply => a * b,
            Self::Divide => a / b,
            Self::Power => a.powf(b),
            Self::Min => pick(b < a, a, b),
            Self::Max => pick(b > a, a, b),
            Self::Less => truth(a < b),
            Self::LessEqual => truth(a <= b),
            Self::Greater => truth(a > b),
            Self::GreaterEqual => truth(a >= b),
            Self::Equal => truth(a.partial_cmp(&b) == Some(Ordering::Equal)),
            Self::NotEqual => truth(a.partial_cmp(&b) != Some(Ordering::Equal)),
            Self::And => truth(a != 0.0 && b != 0.0),
            Self::Or => truth(a != 0.0 || b != 0.0),
        }
    }

    /// Computes the first partial derivatives `(∂f/∂a, ∂f/∂b)` at `a` and `b` given `value`.
    pub(crate) fn slopes(self, a: f64, b: f64, value: f64) -> (f64, f64) {
        match self {
            Self::Add => (1.0, 1.0),
            Self::Subtract => (1.0, -1.0),
            Self::Multiply => (b, a),
            Self::Divide => (1.0 / b, -value / b),
            Self::Power => {
                let base = if a == 0.0 && b == 0.0 {
                    0.0
                } else {
                    b * a.powf(b - 1.0)
                };
                // The exponent's slope, `a^b · ln a`, exists for a positive base only.
                let exponent = if a > 0.0 { value * a.ln() } else { 0.0 };
                (base, exponent)
            }
            Self::Min => pick(b < a, (1.0, 0.0), (0.0, 1.0)),
            Self::Max => pick(b > a, (1.0, 0.0), (0.0, 1.0)),
            Self::Less
            | Self::LessEqual
            | Self::Greater
            | Self::GreaterEqual
            | Self::Equal
            | Self::NotEqual
            | Self::And
            | Self::Or => (0.0, 0.0),
        }
    }

    /// Computes the second partial derivatives `(∂²f/∂a², ∂²f/∂a∂b, ∂²f/∂b²)` at `a` and `b`
    /// given `(slope_a, slope_b)`.
    pub(crate) fn curvatures(self, a: f64, b: f64, (slope_a, slope_b): (f64, f64)) -> [f64; 3] {
        match self {
            Self::Add
            | Self::Subtract
            | Self::Min
            | Self::Max
            | Self::Less
            | Self::LessEqual
            | Self::Greater
            | Self::GreaterEqual
            | Self::Equal
            | Self::NotEqual
            | Self::And
            | Self::Or => [0.0, 0.0, 0.0],
            Self::Multiply => [0.0, 1.0, 0.0],
            Self::Divide => [0.0, -slope_a / b, -2.0 * slope_b / b],
            Self::Power => {
                let factor = b * (b - 1.0);
                let by_base = if a == 0.0 && factor == 0.0 {
                    0.0
                } else {
                    factor * a.powf(b - 2.0)
                };
                if a > 0.0 {
                    let logarithm = a.ln();
                    [
                        by_base,
                        a.powf(b - 1.0) + slope_a * logarithm,
                        slope_b * logarithm,
                    ]
                } else {
                    [by_base, 0.0, 0.0]
                }
            }
        }
    }
}

/// Returns `second` when `take_second` holds, otherwise `first`.
fn pick<T>(take_second: bool, first: T, second: T) -> T {
    if take_second { second } else { first }
}

/// Converts boolean `condition` to `1.0` for true and `0.0` for false.
pub(crate) fn truth(condition: bool) -> f64 {
    if condition { 1.0 } else { 0.0 }
}

/// Returns the sign of `x`: `1.0`, `-1.0`, or `0.0` at zero.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
