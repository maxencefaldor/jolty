use std::borrow::Borrow;
use std::fmt;

use serde::{Deserialize, Serialize, Serializer};

/// The name of a component, port, net, instance or terminal.
///
/// A name is not empty and holds no whitespace, no control character and no `/`, which
/// separates the levels of a hierarchical path such as `amplifier/R1`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(try_from = "String")]
pub struct Name(String);

/// Why a string is not a valid [`Name`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid name {text:?}: {reason}")]
pub struct InvalidName {
    text: String,
    reason: &'static str,
}

impl Name {
    /// The separator between the levels of a hierarchical path.
    pub const SEPARATOR: char = '/';

    /// Checks `text` and wraps it as a name.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidName`] when `text` is empty or holds whitespace, a control character or
    /// the path separator.
    pub fn new(text: impl Into<String>) -> Result<Self, InvalidName> {
        let text = text.into();
        let reason = if text.is_empty() {
            Some("a name is not empty")
        } else if text.contains(Self::SEPARATOR) {
            Some("'/' separates the levels of a path")
        } else if text.chars().any(|c| c.is_whitespace() || c.is_control()) {
            Some("a name holds no whitespace or control characters")
        } else {
            None
        };
        match reason {
            Some(reason) => Err(InvalidName { text, reason }),
            None => Ok(Self(text)),
        }
    }

    /// The name as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Name {
    type Error = InvalidName;

    fn try_from(text: String) -> Result<Self, Self::Error> {
        Self::new(text)
    }
}

impl TryFrom<&str> for Name {
    type Error = InvalidName;

    fn try_from(text: &str) -> Result<Self, Self::Error> {
        Self::new(text)
    }
}

impl Serialize for Name {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl Borrow<str> for Name {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ordinary_names() {
        for text in ["R1", "non_inverting_amplifier[gain=10]", "vdd", "U1.A"] {
            assert_eq!(Name::new(text).unwrap().as_str(), text);
        }
    }

    #[test]
    fn rejects_empty_separated_and_spaced_names() {
        for text in ["", "a/b", "two words", "tab\there", "line\nbreak"] {
            assert!(Name::new(text).is_err(), "{text:?} should be rejected");
        }
    }

    #[test]
    fn deserialization_checks_the_name() {
        assert!(serde_json::from_str::<Name>("\"R1\"").is_ok());
        assert!(serde_json::from_str::<Name>("\"a/b\"").is_err());
    }
}
