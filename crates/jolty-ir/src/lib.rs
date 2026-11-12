//! The canonical circuit format of jolty.
//!
//! A [`Circuit`] is a set of named [`Component`]s and the name of the top one. A component
//! declares its [`Port`]s and places [`Instance`]s, each of a device model or of another
//! component. Every terminal of an instance joins a net by name, and a port is the net of the
//! same name seen from outside. Parameter values are numbers in SI base units. A device
//! instance may give some of its parameters a [`Tolerance`]: how far a part's value may lie
//! from its nominal one, which Monte Carlo analyses sample. A tolerance names its parameter
//! by [path](parameter()), so it may reach into an object, as `waveform.dc` does.
//!
//! The format is JSON, tagged with [`FORMAT`] and [`VERSION`]:
//!
//! ```
//! use jolty_ir::Circuit;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit",
//!     "version": 1,
//!     "top": "divider",
//!     "components": {
//!         "divider": {
//!             "ports": [{"name": "ground", "role": "ground"}],
//!             "instances": [
//!                 {"id": "V1", "of": {"device": "voltage_source"},
//!                  "parameters": {"waveform": {"dc": 5.0}},
//!                  "terminals": {"p": "top", "n": "ground"}},
//!                 {"id": "R1", "of": {"device": "resistor"},
//!                  "parameters": {"resistance": 1000.0},
//!                  "terminals": {"a": "top", "b": "out"}},
//!                 {"id": "R2", "of": {"device": "resistor"},
//!                  "parameters": {"resistance": 1000.0},
//!                  "terminals": {"a": "out", "b": "ground"}}
//!             ]
//!         }
//!     }
//! }"#)?;
//! assert_eq!(circuit.top.as_str(), "divider");
//! circuit.validate()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod name;
mod parameter;
mod validate;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use name::{InvalidName, Name};
pub use parameter::{Unreachable, numbers, parameter, parameter_mut, set_parameter};
pub use validate::{Problem, Problems};

/// The value of the `format` field of every circuit file.
pub const FORMAT: &str = "jolty-circuit";

/// The version of the circuit format this crate reads and writes.
pub const VERSION: u32 = 1;

/// The parameters of a device instance, by name, as JSON values in SI base units.
pub type Parameters = serde_json::Map<String, serde_json::Value>;

/// A circuit: named components, and the one at the top of the hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    /// The component the circuit is, which instances every other component it uses.
    pub top: Name,
    /// Every component, by name.
    pub components: BTreeMap<Name, Component>,
}

/// A component: its ports and the instances it places.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    /// The nets the component exposes, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<Port>,
    /// The device and component instances the component places.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instances: Vec<Instance>,
}

/// A net a component exposes to the components that instance it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    /// The name of the port, which is also the name of its net inside the component.
    pub name: Name,
    /// What the port carries.
    pub role: Role,
}

/// What a port carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A signal, the default, in either direction.
    Signal,
    /// A signal into the component, which its circuit reads.
    Input,
    /// A signal out of the component, which its circuit drives.
    Output,
    /// A supply rail.
    Supply,
    /// The ground. At the top of a circuit, the ground port's net is the reference node.
    Ground,
}

/// One placed device or component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instance {
    /// The instance's name, unique within its component.
    pub id: Name,
    /// What the instance is.
    pub of: Of,
    /// The device parameters; component instances take none.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub parameters: Parameters,
    /// The net each terminal joins, by terminal name.
    pub terminals: BTreeMap<Name, Name>,
    /// The tolerances of some of a device's parameters, by the parameter's
    /// [path](parameter()).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tolerances: BTreeMap<String, Tolerance>,
    /// The line of source code that placed the instance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
}

/// How far a parameter's value may lie from its nominal one, relative to it.
///
/// A part's own deviation is drawn apart from every other part's; one in a [`Lot`] adds
/// the lot's deviation, which every part in the lot shares.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    /// The part's own tolerance, such as `0.01` for 1 %, as its distribution
    /// [takes](Distribution::takes) it.
    pub relative: f64,
    /// How values spread within it.
    #[serde(default)]
    pub distribution: Distribution,
    /// The lot the part comes from, if it names one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lot: Option<Lot>,
}

/// A spread that the parts of one lot share, as parts from one reel, or on one die, drift
/// together: each parameter of theirs takes one factor from it per sample, which scales
/// each part's own.
///
/// A lot is known by its name wherever in the hierarchy a part names it, and spreads each
/// parameter one way.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lot {
    /// The lot's name.
    pub name: Name,
    /// The lot's tolerance, as its distribution [takes](Distribution::takes) it.
    pub relative: f64,
    /// How the lot's deviations spread within it.
    #[serde(default)]
    pub distribution: Distribution,
}

/// How a parameter's values spread within its tolerance.
///
/// A value lies off its nominal one by a factor: `1 + d`, with `d` within the tolerance,
/// or for a logarithmic spread `(1 + relative)^u`, with `u` within 1 of 0, as an
/// exponential's argument spreads, such as a saturation current's when the junction
/// voltage it sets does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    /// Evenly within the tolerance, the default.
    #[default]
    Uniform,
    /// Normally, with the tolerance three standard deviations.
    Normal,
    /// Evenly in the logarithm, within a factor of `1 + relative` either way.
    LogUniform,
    /// Normally in the logarithm, a factor of `1 + relative` three standard deviations.
    LogNormal,
}

impl Distribution {
    /// Whether this distribution takes a tolerance of `relative`: at least 0, and below
    /// 1 unless it spreads the logarithm.
    #[must_use]
    pub fn takes(self, relative: f64) -> bool {
        match self {
            Self::Uniform | Self::Normal => (0.0..1.0).contains(&relative),
            Self::LogUniform | Self::LogNormal => relative >= 0.0 && relative.is_finite(),
        }
    }
}

/// What an instance is: a device of a model, or another component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Of {
    /// A device of the named model, such as `resistor`.
    Device(Name),
    /// An instance of the named component.
    Component(Name),
}

/// The line of source code that placed an instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    /// The path of the file, as the build script's runner reported it.
    pub file: String,
    /// The line number, counted from 1.
    pub line: u32,
    /// The function that placed the instance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

/// Why a circuit file could not be read.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    /// The text is not JSON, or does not have the shape of a circuit.
    #[error("not a circuit file: {0}")]
    Json(#[from] serde_json::Error),
    /// The file declares another format.
    #[error("the file's format is {0:?}, not {FORMAT:?}")]
    Format(String),
    /// The file declares a version this crate does not read.
    #[error("the file has format version {0}, and this version of jolty reads version {VERSION}")]
    Version(u32),
}

#[derive(Deserialize)]
struct Header {
    format: String,
    version: u32,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    format: String,
    version: u32,
    top: Name,
    components: BTreeMap<Name, Component>,
}

impl Circuit {
    /// Reads a circuit from its JSON text.
    ///
    /// The format and version are checked first, so a file from another version of jolty is
    /// reported as such rather than as a field the reader does not know.
    ///
    /// # Errors
    ///
    /// Returns [`ReadError`] when the text is not JSON, declares another format or version, or
    /// does not have the shape of a circuit.
    pub fn from_json(text: &str) -> Result<Self, ReadError> {
        let header: Header = serde_json::from_str(text)?;
        if header.format != FORMAT {
            return Err(ReadError::Format(header.format));
        }
        if header.version != VERSION {
            return Err(ReadError::Version(header.version));
        }
        let Document {
            top, components, ..
        } = serde_json::from_str(text)?;
        Ok(Self { top, components })
    }

    /// Writes the circuit as JSON text, tagged with its format and version.
    ///
    /// # Panics
    ///
    /// Never: every part of a circuit serializes to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        let document = Document {
            format: FORMAT.to_owned(),
            version: VERSION,
            top: self.top.clone(),
            components: self.components.clone(),
        };
        serde_json::to_string_pretty(&document).expect("a circuit always serializes to JSON")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn divider() -> &'static str {
        r#"{
            "format": "jolty-circuit", "version": 1, "top": "divider",
            "components": {"divider": {
                "ports": [{"name": "ground", "role": "ground"}],
                "instances": [
                    {"id": "R1", "of": {"device": "resistor"},
                     "parameters": {"resistance": 1000.0},
                     "terminals": {"a": "top", "b": "ground"},
                     "source": {"file": "divider.py", "line": 7, "function": "divider"}}
                ]
            }}
        }"#
    }

    #[test]
    fn round_trips_through_json() {
        let circuit = Circuit::from_json(divider()).unwrap();
        assert_eq!(Circuit::from_json(&circuit.to_json()).unwrap(), circuit);
    }

    #[test]
    fn reports_another_format() {
        let text = divider().replace("jolty-circuit", "bricky-model");
        assert!(
            matches!(Circuit::from_json(&text), Err(ReadError::Format(f)) if f == "bricky-model")
        );
    }

    #[test]
    fn reports_another_version_before_unknown_fields() {
        let text = divider().replace(r#""version": 1"#, r#""version": 2, "future": true"#);
        assert!(matches!(
            Circuit::from_json(&text),
            Err(ReadError::Version(2))
        ));
    }

    #[test]
    fn rejects_unknown_fields() {
        let text = divider().replace(r#""role": "ground""#, r#""role": "ground", "colour": 1"#);
        assert!(matches!(Circuit::from_json(&text), Err(ReadError::Json(_))));
    }
}
