//! What each operation computes, and its partial derivatives.

use std::cmp::Ordering;

/// An operation of one operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unary {
    Negate,
    Not,
    Abs,
    Sqrt,
    Exp,
    Ln,
    Log10,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Floor,
    Ceil,
    Sign,
    Step,
    Ramp,
}

impl Unary {
    /// The function called `name`, if one of one argument is.
    pub(crate) fn named(name: &str) -> Option<Self> {
        Some(match name {
            "abs" => Self::Abs,
            "sqrt" => Self::Sqrt,
            "exp" => Self::Exp,
            "ln" | "log" => Self::Ln,
            "log10" => Self::Log10,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "asin" => Self::Asin,
            "acos" => Self::Acos,
            "atan" => Self::Atan,
            "sinh" => Self::Sinh,
            "cosh" => Self::Cosh,
            "tanh" => Self::Tanh,
            "asinh" => Self::Asinh,
            "acosh" => Self::Acosh,
            "atanh" => Self::Atanh,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "sgn" => Self::Sign,
            "u" => Self::Step,
            "uramp" => Self::Ramp,
            _ => return None,
        })
    }

    /// The operation's value at `x`.
    pub(crate) fn value(self, x: f64) -> f64 {
        match self {
            Self::Negate => -x,
            Self::Not => truth(x == 0.0),
            Self::Abs => x.abs(),
            Self::Sqrt => x.sqrt(),
            Self::Exp => x.exp(),
            Self::Ln => x.ln(),
            Self::Log10 => x.log10(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Tan => x.tan(),
            Self::Asin => x.asin(),
            Self::Acos => x.acos(),
            Self::Atan => x.atan(),
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Asinh => x.asinh(),
            Self::Acosh => x.acosh(),
            Self::Atanh => x.atanh(),
            Self::Floor => x.floor(),
            Self::Ceil => x.ceil(),
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

    /// The operation's derivative at `x`, where its value is `value`.
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
    /// The function called `name`, if one of two arguments is.
    pub(crate) fn named(name: &str) -> Option<Self> {
        Some(match name {
            "min" => Self::Min,
            "max" => Self::Max,
            "pow" => Self::Power,
            _ => return None,
        })
    }

    /// The operation's value at `a` and `b`.
    pub(crate) fn value(self, a: f64, b: f64) -> f64 {
        match self {
            Self::Add => a + b,
            Self::Subtract => a - b,
            Self::Multiply => a * b,
            Self::Divide => a / b,
            Self::Power => a.powf(b),
            Self::Min => {
                if b < a {
                    b
                } else {
                    a
                }
            }
            Self::Max => {
                if b > a {
                    b
                } else {
                    a
                }
            }
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

    /// The operation's derivatives with respect to `a` and `b`, where its value is `value`.
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
            Self::Min => {
                if b < a {
                    (0.0, 1.0)
                } else {
                    (1.0, 0.0)
                }
            }
            Self::Max => {
                if b > a {
                    (0.0, 1.0)
                } else {
                    (1.0, 0.0)
                }
            }
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
}

/// `1` for true, `0` for false.
pub(crate) fn truth(condition: bool) -> f64 {
    if condition { 1.0 } else { 0.0 }
}

/// The sign of `x`: `1`, `−1`, or `0` at zero.
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}
