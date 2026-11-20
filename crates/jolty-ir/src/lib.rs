//! Canonical circuit intermediate representation for `jolty`.
//!
//! A [`Circuit`] holds named [`Component`] definitions together with the top-level component
//! name. A component declares [`Port`]s and places [`Instance`]s referencing device models or
//! other components. Each instance terminal joins a net by name, and each port exposes the net
//! of the same name to parent components.
//!
//! Parameter values are numbers in SI base units. A device instance may attach a [`Tolerance`]
//! to any numeric parameter by [path](parameter()) (including nested fields such as
//! `waveform.dc`) for Monte Carlo sampling.
//!
//! The serialized format is JSON tagged with [`FORMAT`] and [`VERSION`]:
//!
//! ```
//! use jolty_ir::Circuit;
//!
//! let circuit = Circuit::from_json(r#"{
//!     "format": "jolty-circuit",
//!     "version": 5,
//!     "top": "divider",
//!     "components": {
//!         "divider": {
//!             "ports": [{"name": "ground", "role": "ground"}],
//!             "instances": [
//!                 {"name": "V1", "of": {"model": "voltage_source"},
//!                  "parameters": {"waveform": {"dc": 5.0}},
//!                  "terminals": {"p": "top", "n": "ground"}},
//!                 {"name": "R1", "of": {"model": "resistor"},
//!                  "parameters": {"resistance": 1000.0},
//!                  "terminals": {"a": "top", "b": "out"}},
//!                 {"name": "R2", "of": {"model": "resistor"},
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
pub const VERSION: u32 = 5;

/// The parameters of a device instance, by name, as JSON values in SI base units.
pub type Parameters = serde_json::Map<String, serde_json::Value>;

/// Circuit definition consisting of named components and the top-level component name.
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    /// Top-level component name that roots the circuit hierarchy.
    pub top: Name,
    /// Map of component definitions keyed by component name.
    pub components: BTreeMap<Name, Component>,
}

/// Component definition declaring exposed ports, placed instances, and optional statistical model.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    /// External ports exposed by the component, in declaration order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<Port>,
    /// Device and subcircuit instances placed inside the component.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub instances: Vec<Instance>,
    /// Statistical draw graph computing per-sample device parameters for Monte Carlo runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub statistical: Option<Statistical>,
}

/// Statistical parameter model attached to a [`Component`] for native Monte Carlo draws.
///
/// Each step defines a named variable either by drawing from a SPICE statistical distribution
/// (`agauss`, `gauss`, `aunif`, `unif`, or `limit`) or by evaluating a jolty expression over
/// earlier variables via `p(<name>)`. Each target computes one device parameter inside the
/// component from those variables.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statistical {
    /// Ordered variable definitions evaluated top-to-bottom per sample.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub steps: Vec<StatisticalStep>,
    /// Device parameter expressions evaluated after `steps`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<StatisticalTarget>,
}

impl Statistical {
    /// Returns whether this statistical model defines no steps and no targets.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.steps.is_empty() && self.targets.is_empty()
    }
}

/// Predicate for serde `skip_serializing_if` on optional boolean flags defaulting to `false`.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde skip_serializing_if passes a reference"
)]
const fn is_false(value: &bool) -> bool {
    !*value
}

/// One named variable definition inside a component's [`Statistical`] evaluation graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StatisticalStep {
    /// Draws a variate via a SPICE statistical distribution function using the engine's
    /// deterministic kit variate generator.
    Draw {
        /// Identifier referenced as `p(<name>)` by subsequent steps and targets.
        name: String,
        /// Shared library-level draw when true (empty RNG scope `""`); per-placement draw
        /// when false.
        #[serde(default, skip_serializing_if = "is_false")]
        shared: bool,
        /// Relative subcircuit path appended to the placement path when `shared` is false
        /// (empty for draws directly in the owning component).
        #[serde(default, skip_serializing_if = "String::is_empty")]
        scope: String,
        /// Call identifier matching `Library.drawn` (such as `vth0#0`).
        call: String,
        /// SPICE statistical distribution function applied to the arguments.
        function: StatisticalFunction,
        /// Nominal argument expression (first argument of the SPICE function).
        nominal: String,
        /// Spread argument expression (second argument of the SPICE function).
        spread: String,
        /// Sigma divisor expression for `agauss` and `gauss` (third argument).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sigma: Option<String>,
    },
    /// Evaluates an intermediate jolty expression over earlier step variables.
    Value {
        /// Identifier referenced as `p(<name>)` by subsequent steps and targets.
        name: String,
        /// Jolty expression reading earlier variables via `p(<name>)`.
        expression: String,
    },
}

impl StatisticalStep {
    /// Returns the variable name defined by this step.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Draw { name, .. } | Self::Value { name, .. } => name,
        }
    }
}

/// SPICE statistical distribution function evaluated by [`StatisticalStep::Draw`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalFunction {
    /// Absolute Gaussian: `x + (a / s) * z` when `a > 0` and `s > 0`, else `x`.
    Agauss,
    /// Relative Gaussian: `x + (x * r / s) * z` when `r > 0` and `s > 0`, else `x`.
    Gauss,
    /// Absolute uniform: `x + a * u` with `u` in `[-1, 1)`.
    Aunif,
    /// Relative uniform: `x + (x * r) * u` with `u` in `[-1, 1)`.
    Unif,
    /// Two-point limit: `x + a` when `u > 0`, else `x - a`.
    Limit,
}

impl StatisticalFunction {
    /// Returns whether this statistical function takes a third `sigma` divisor argument.
    #[must_use]
    pub const fn takes_sigma(self) -> bool {
        matches!(self, Self::Agauss | Self::Gauss)
    }
}

/// Device parameter driven by a component's [`Statistical`] variables.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticalTarget {
    /// Instance path relative to the owning component (such as `M1` or `X1/R1`).
    pub instance: String,
    /// Dotted parameter path on the target device (such as `vth0` or `parameters.rbody`).
    pub parameter: String,
    /// Jolty expression reading step variables via `p(<name>)`.
    pub expression: String,
}

/// External port exposing an internal component net to parent components.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Port {
    /// Port name, which also names the corresponding internal net.
    pub name: Name,
    /// Electrical role carried by the port.
    pub role: Role,
}

/// Electrical role of a component port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Bidirectional or general-purpose signal net (default).
    Signal,
    /// Signal net driven externally and read by the component.
    Input,
    /// Signal net driven by the component.
    Output,
    /// Power-supply rail.
    Supply,
    /// Reference ground rail; at the top level, this port's net is circuit ground.
    Ground,
}

/// Placed device or subcircuit instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Instance {
    /// Instance name, unique within its enclosing component.
    pub name: Name,
    /// Target device model or subcircuit component instantiated by this entry.
    pub of: Of,
    /// Device parameters in SI base units; empty for subcircuit component instances.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub parameters: Parameters,
    /// Map from instance terminal names to enclosing net names.
    pub terminals: BTreeMap<Name, Name>,
    /// Parameter tolerances keyed by dot-separated parameter [path](parameter()).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tolerances: BTreeMap<String, Tolerance>,
    /// Manufacturer stress limits checked during simulation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ratings: Vec<Rating>,
    /// Manufacturer part number represented by the instance, if specified; a numbered
    /// component instance counts as a single part regardless of its internal instances.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    /// Source-code location that placed the instance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_location: Option<SourceLocation>,
}

/// Manufacturer limit on a device stress checked during simulation (terminal-pair voltage,
/// terminal current, dissipated power, or operating temperature).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Rating {
    /// Physical stress quantity bounded by the rating.
    pub stress: Stress,
    /// Terminals where the stress is measured: two for voltage (first minus second), one for
    /// current (into the terminal), and none for power or temperature.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub terminals: Vec<Name>,
    /// Upper stress limit in SI base units (kelvins for temperature), if specified; required
    /// for power ratings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Lower stress limit in SI base units, if specified; omitted for power ratings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
}

/// Physical stress quantity bounded by a [`Rating`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stress {
    /// Voltage difference between two terminals.
    Voltage,
    /// Current flowing into a terminal.
    Current,
    /// Power dissipated by the part, summed as terminal voltage times current into each
    /// electrical terminal.
    Power,
    /// Operating temperature of the part: ambient temperature plus any rise on its `thermal`
    /// net.
    Temperature,
}

impl Stress {
    /// Returns the lowercase name of the stress: `voltage`, `current`, `power`, or
    /// `temperature`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Voltage => "voltage",
            Self::Current => "current",
            Self::Power => "power",
            Self::Temperature => "temperature",
        }
    }

    /// Returns the number of terminals required by a rating of this stress.
    #[must_use]
    pub const fn terminals(self) -> usize {
        match self {
            Self::Voltage => 2,
            Self::Current => 1,
            Self::Power | Self::Temperature => 0,
        }
    }
}

/// Relative tolerance specification for a device parameter.
///
/// Each part draws an independent deviation; parts belonging to a [`Lot`] also add a shared
/// lot deviation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    /// Per-instance relative tolerance (such as `0.01` for 1 %), interpreted according to
    /// [`Distribution::takes`].
    pub relative: f64,
    /// Probability distribution used to sample deviations within the tolerance.
    #[serde(default)]
    pub distribution: Distribution,
    /// Shared [`Lot`] contributing a correlated deviation to this parameter, if specified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lot: Option<Lot>,
}

/// Shared tolerance lot for parts from the same reel or die.
///
/// Each Monte Carlo sample draws one lot factor per parameter and combines it with each part's
/// individual deviation. A lot is identified by name across the entire hierarchy and applies
/// one consistent spread to each parameter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lot {
    /// Lot identifier shared across instances.
    pub name: Name,
    /// Shared relative tolerance of the lot, interpreted according to [`Distribution::takes`].
    pub relative: f64,
    /// Probability distribution used to sample lot deviations.
    #[serde(default)]
    pub distribution: Distribution,
}

/// Probability distribution for sampling parameter deviations within a tolerance.
///
/// Linear distributions scale nominal values by `1 + d` where `d` lies within `relative`.
/// Logarithmic distributions scale by `(1 + relative)^u` where `u` lies within `[-1, 1]`,
/// matching exponential parameters such as junction saturation current.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Distribution {
    /// Uniform distribution across the tolerance range (default).
    #[default]
    Uniform,
    /// Normal distribution with `relative` equal to three standard deviations.
    Normal,
    /// Log-uniform distribution spanning a factor of `1 + relative` above and below nominal.
    LogUniform,
    /// Log-normal distribution with a factor of `1 + relative` at three standard deviations.
    LogNormal,
}

impl Distribution {
    /// Returns whether this distribution accepts `relative`: non-negative, and strictly below
    /// 1 for linear distributions.
    #[must_use]
    pub fn takes(self, relative: f64) -> bool {
        match self {
            Self::Uniform | Self::Normal => (0.0..1.0).contains(&relative),
            Self::LogUniform | Self::LogNormal => relative >= 0.0 && relative.is_finite(),
        }
    }
}

/// Target kind of a placed instance: a primitive device model or a subcircuit component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum Of {
    /// Primitive device of the named model, such as `resistor`.
    #[serde(rename = "model")]
    Device(Name),
    /// Subcircuit instance of the named component.
    Component(Name),
}

/// Source-code location that placed an instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceLocation {
    /// Source file path reported by the circuit builder script.
    pub file: String,
    /// One-based source line number.
    pub line: u32,
    /// Enclosing function name that placed the instance, if recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

/// Error returned when reading a circuit JSON file fails.
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Document {
    format: String,
    version: u32,
    top: Name,
    components: BTreeMap<Name, Component>,
}

#[derive(Serialize)]
struct DocumentRef<'a> {
    format: &'static str,
    version: u32,
    top: &'a Name,
    components: &'a BTreeMap<Name, Component>,
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
        match serde_json::from_str::<Document>(text) {
            Ok(Document {
                format,
                version,
                top,
                components,
            }) => {
                if format != FORMAT {
                    return Err(ReadError::Format(format));
                }
                if version != VERSION {
                    return Err(ReadError::Version(version));
                }
                Ok(Self { top, components })
            }
            Err(error) => {
                if let Ok(header) = serde_json::from_str::<Header>(text) {
                    if header.format != FORMAT {
                        return Err(ReadError::Format(header.format));
                    }
                    if header.version != VERSION {
                        return Err(ReadError::Version(header.version));
                    }
                }
                Err(ReadError::Json(error))
            }
        }
    }

    /// Writes the circuit as JSON text, tagged with its format and version.
    ///
    /// # Panics
    ///
    /// Never: every part of a circuit serializes to JSON.
    #[must_use]
    #[expect(
        clippy::expect_used,
        reason = "serde_json fails only on a map keyed by other than strings, and every map of \
                  a circuit is keyed by strings"
    )]
    pub fn to_json(&self) -> String {
        let document = DocumentRef {
            format: FORMAT,
            version: VERSION,
            top: &self.top,
            components: &self.components,
        };
        serde_json::to_string_pretty(&document).expect("a circuit always serializes to JSON")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn divider() -> &'static str {
        r#"{
            "format": "jolty-circuit", "version": 5, "top": "divider",
            "components": {"divider": {
                "ports": [{"name": "ground", "role": "ground"}],
                "instances": [
                    {"name": "R1", "of": {"model": "resistor"},
                     "parameters": {"resistance": 1000.0},
                     "terminals": {"a": "top", "b": "ground"},
                     "source_location": {"file": "divider.py", "line": 7, "function": "divider"}}
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
        let text = divider().replace(
            &format!(r#""version": {VERSION}"#),
            &format!(r#""version": {}, "future": true"#, VERSION + 1),
        );
        assert!(matches!(
            Circuit::from_json(&text),
            Err(ReadError::Version(version)) if version == VERSION + 1
        ));
    }

    /// Verifies exact IEEE 754 round-tripping of shortest-representation floats written by
    /// Python.
    ///
    /// Without `serde_json`'s `float_roundtrip`, values such as `981.4902141434463` decode one
    /// ulp off (`981.4902141434465`).
    #[test]
    fn reads_every_number_as_python_writes_it() {
        for written in [
            "981.4902141434463",
            "0.1",
            "1e-15",
            "2.2250738585072014e-308",
            "5e-324",
            "1.7976931348623157e+308",
        ] {
            let text = divider().replace("1000.0", written);
            let circuit = Circuit::from_json(&text).unwrap();
            let read = circuit.components[&Name::new("divider").unwrap()].instances[0]
                .parameters["resistance"]
                .as_f64()
                .unwrap();
            assert_eq!(
                read.to_bits(),
                written.parse::<f64>().unwrap().to_bits(),
                "{written}"
            );
        }
    }

    #[test]
    fn rejects_unknown_fields() {
        let text = divider().replace(r#""role": "ground""#, r#""role": "ground", "color": 1"#);
        assert!(matches!(Circuit::from_json(&text), Err(ReadError::Json(_))));
    }

    #[test]
    fn preserves_format_and_version_precedence_on_fallback_and_fast_path() {
        let wrong_version_clean = divider().replace(
            &format!(r#""version": {VERSION}"#),
            &format!(r#""version": {}"#, VERSION + 1),
        );
        assert!(matches!(
            Circuit::from_json(&wrong_version_clean),
            Err(ReadError::Version(version)) if version == VERSION + 1
        ));

        let wrong_format_with_unknown = divider().replace("jolty-circuit", "other-tool").replace(
            &format!(r#""version": {VERSION}"#),
            &format!(r#""version": {}, "extra": 123"#, VERSION + 5),
        );
        assert!(matches!(
            Circuit::from_json(&wrong_format_with_unknown),
            Err(ReadError::Format(format)) if format == "other-tool"
        ));

        assert!(matches!(
            Circuit::from_json("{not valid json"),
            Err(ReadError::Json(_))
        ));

        assert_eq!(Stress::Voltage.as_str(), "voltage");
        assert_eq!(Stress::Current.as_str(), "current");
        assert_eq!(Stress::Power.as_str(), "power");
        assert_eq!(Stress::Temperature.as_str(), "temperature");
    }
}
