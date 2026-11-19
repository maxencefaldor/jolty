//! Event-driven logic devices and mixed-signal connect elements bridging logic and analog
//! nets.
//!
//! A logic device reads [`LogicLevel`] values on input nets and drives output nets to new
//! levels after propagation delays. Logic pins attach to discrete logic nets so the event
//! scheduler advances from transition to transition without numerical integration.
//!
//! When a logic pin connects to an electrical net, the engine inserts a connect element. A
//! [`ThresholdDetector`](crate::ThresholdDetector) converts a net voltage into an input
//! level, and a [`LogicDriver`](crate::LogicDriver) converts an output level into a voltage
//! between supply and ground.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use schemars::{JsonSchema, Schema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::ParameterError;
use crate::device::{read, schema, written};

/// The most inputs a logic cell may have: its truth table has a row per combination.
pub const MAX_INPUTS: usize = 16;

/// The level of a logic net.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum LogicLevel {
    /// A logic 0: the ground.
    #[serde(rename = "0")]
    Low,
    /// A logic 1: the supply.
    #[serde(rename = "1")]
    High,
    /// Neither, or either: a net no one drives, two drivers that disagree, or a level not
    /// yet known.
    #[serde(rename = "x")]
    Unknown,
}

impl LogicLevel {
    /// Converts boolean `bit` to [`LogicLevel::High`] (`true`) or [`LogicLevel::Low`] (`false`).
    #[must_use]
    pub const fn from_bool(bit: bool) -> Self {
        if bit { Self::High } else { Self::Low }
    }

    /// Returns the supply voltage fraction driven for `self`: `0.0`, `1.0`, or `0.5` for
    /// [`LogicLevel::Unknown`].
    #[must_use]
    pub const fn fraction(self) -> f64 {
        match self {
            Self::Low => 0.0,
            Self::High => 1.0,
            Self::Unknown => 0.5,
        }
    }

    /// Resolves multiple driver `levels` on one net, returning the shared level if all agree
    /// or [`LogicLevel::Unknown`] if drivers conflict or none exist.
    #[must_use]
    pub fn resolve(levels: impl IntoIterator<Item = Self>) -> Self {
        let mut levels = levels.into_iter();
        let Some(first) = levels.next() else {
            return Self::Unknown;
        };
        if levels.all(|level| level == first) {
            first
        } else {
            Self::Unknown
        }
    }
}

impl fmt::Display for LogicLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Low => "0",
            Self::High => "1",
            Self::Unknown => "x",
        })
    }
}

/// A change of one input of a logic device, as its timing reads it.
#[derive(Debug, Clone, Copy)]
pub struct Change<'a> {
    /// Every input's level before the change.
    pub before: &'a [LogicLevel],
    /// The input that changed.
    pub input: usize,
    /// The input's transition: the time its edge takes to swing from ground to supply, in
    /// seconds.
    pub slew: f64,
}

/// Output propagation delay and edge transition time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timing {
    /// Delay from the input crossing to the output crossing at half supply, in seconds.
    pub delay: f64,
    /// Output edge transition time from ground to supply, in seconds.
    pub transition: f64,
}

/// Autonomous output transition scheduled by a logic device, such as a stimulus source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stimulus {
    /// Timestamp when the output reaches the new level, in seconds.
    pub time: f64,
    /// Index of the changing output pin.
    pub output: usize,
    /// Target logic level.
    pub level: LogicLevel,
    /// Output edge transition time, in seconds.
    pub transition: f64,
}

/// Placed event-driven logic device.
pub trait Logic: fmt::Debug + Send + Sync {
    /// Returns the logic and optional electrical pin declarations of the device.
    fn pins(&self) -> Pins;

    /// Evaluates steady-state output `levels` for the given `inputs`.
    fn levels(&self, inputs: &[LogicLevel], outputs: &mut [LogicLevel]);

    /// Computes propagation delay and transition time for `output` reaching `level` after
    /// `change` while driving `load` farads.
    fn timing(&self, change: &Change<'_>, output: usize, level: LogicLevel, load: f64) -> Timing;

    /// Appends autonomous stimulus events scheduled within `(0, stop]` in chronological order.
    fn stimuli(&self, stop: f64, stimuli: &mut Vec<Stimulus>) {
        let _ = (stop, stimuli);
    }

    /// Returns the input pin capacitance loading `input`, in farads.
    fn input_capacitance(&self, input: usize) -> f64 {
        let _ = input;
        0.0
    }

    /// Returns the characterized slew and load validity domain, or `None` if unbounded.
    fn domain(&self) -> Option<&Domain> {
        None
    }
}

/// Measured slew, load, and supply validity ranges of a characterized logic device.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    /// Minimum and maximum input transition times, in seconds.
    pub slews: (f64, f64),
    /// Minimum and maximum output load capacitances, in farads.
    pub loads: (f64, f64),
    /// Supply voltage relative to ground at which tables were characterized, if known.
    pub supply: Option<f64>,
}

/// Mixed-signal electrical interface configuration for connect elements bridging logic pins
/// to analog nets.
#[derive(Debug, Clone, PartialEq)]
pub struct Electrical {
    /// Switching threshold as a fraction of supply voltage above ground.
    pub threshold: f64,
    /// Name of the supply terminal.
    pub supply: String,
    /// Name of the ground terminal.
    pub ground: String,
}

/// Input, output, and optional electrical power pins declared by a logic device.
#[derive(Debug, Clone, PartialEq)]
pub struct Pins {
    /// Input pin names in the order read by [`Logic::levels`].
    pub inputs: Vec<String>,
    /// Output pin names in the order written by [`Logic::levels`].
    pub outputs: Vec<String>,
    /// Optional electrical interface adding supply and ground terminals.
    pub electrical: Option<Electrical>,
}

impl Pins {
    /// Returns all pin and terminal names in order: inputs, outputs, then optional supply
    /// and ground terminals.
    #[must_use]
    pub fn names(&self) -> Vec<String> {
        let terminals = self
            .electrical
            .iter()
            .flat_map(|electrical| [&electrical.supply, &electrical.ground]);
        self.inputs
            .iter()
            .chain(&self.outputs)
            .chain(terminals)
            .cloned()
            .collect()
    }
}

/// Logic model factory providing the model name, parameter schema, and device instantiation.
pub trait LogicModel: Send + Sync {
    /// Returns the model name used in circuit definitions, such as `logic_cell`.
    fn name(&self) -> &'static str;

    /// Parses `parameters` and constructs a logic device instance.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is missing, unknown, or out of range.
    fn instantiate(&self, parameters: &Parameters) -> Result<Box<dyn Logic>, ParameterError>;

    /// Returns the JSON schema describing all parameters of the model with types, defaults,
    /// and descriptions inlined.
    fn schema(&self) -> Schema;

    /// Returns `parameters` normalized with all omitted parameters populated from defaults.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be read.
    fn defaulted(&self, parameters: &Parameters) -> Result<Parameters, ParameterError>;
}

/// Native Rust logic device whose parameters deserialize into a strongly typed struct.
pub trait NativeLogic: Logic + Sized + 'static {
    /// Model name registered in the library.
    const NAME: &'static str;
    /// Deserialized parameter struct type.
    type Parameters: DeserializeOwned + Serialize + JsonSchema;

    /// Validates `parameters` and constructs the logic device.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn new(parameters: Self::Parameters) -> Result<Self, ParameterError>;
}

/// [`LogicModel`] adapter wrapping a [`NativeLogic`] device type.
pub struct NativeLogicModel<D>(PhantomData<fn() -> D>);

impl<D: NativeLogic> NativeLogicModel<D> {
    /// Creates a logic model adapter for device type `D`.
    #[must_use]
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<D: NativeLogic> Default for NativeLogicModel<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D> fmt::Debug for NativeLogicModel<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NativeLogicModel")
    }
}

impl<D: NativeLogic> LogicModel for NativeLogicModel<D> {
    fn name(&self) -> &'static str {
        D::NAME
    }

    fn instantiate(&self, parameters: &Parameters) -> Result<Box<dyn Logic>, ParameterError> {
        Ok(Box::new(D::new(read(parameters)?)?))
    }

    fn schema(&self) -> Schema {
        schema::<D::Parameters>()
    }

    fn defaulted(&self, parameters: &Parameters) -> Result<Parameters, ParameterError> {
        written(&read::<D::Parameters>(parameters)?)
    }
}

/// One output of a logic cell and its truth table.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputParameters {
    /// The output's name.
    pub name: String,
    /// The output's level for each combination of the inputs, a `0` or a `1` per row. Row
    /// `k` holds the inputs whose bits make `k`, the first input the most significant: for
    /// inputs `a` and `b`, the rows are `ab` = 00, 01, 10 and 11, so a NAND is `1110`.
    pub table: String,
}

/// The fixed timing of a logic cell.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FixedTiming {
    /// The delay of an output rising to 1, in seconds.
    pub rise: f64,
    /// The delay of an output falling to 0, in seconds.
    pub fall: f64,
    /// The outputs' transition, the time an edge takes to swing from ground to supply, in
    /// seconds.
    pub transition: f64,
}

/// One timing arc of a characterized cell: how an output follows one input's change from
/// one combination of the inputs.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArcParameters {
    /// The input that changes.
    pub input: String,
    /// Every input's level before the change, a `0` or a `1` each, in the order of the
    /// inputs.
    pub before: String,
    /// The output that follows.
    pub output: String,
    /// The delay, in seconds, for each input transition, then for each load: one row per
    /// transition.
    pub delay: Vec<Vec<f64>>,
    /// The output's transition, in seconds, laid out as the delay.
    pub transition: Vec<Vec<f64>>,
}

/// The timing of a cell characterized from its circuit: tables of delays and transitions
/// over input transitions and output loads, as a standard cell library's.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CharacterizedTiming {
    /// The input transitions the tables were measured at, in seconds, rising.
    pub slews: Vec<f64>,
    /// The output loads the tables were measured at, in farads, rising.
    pub loads: Vec<f64>,
    /// Every arc: each change of an input that changes an output.
    pub arcs: Vec<ArcParameters>,
    /// The voltage from ground to supply the tables were measured at, in volts.
    #[serde(default)]
    pub supply: Option<f64>,
}

/// Timing specification controlling how logic cell outputs follow input changes.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TimingParameters {
    /// Constant rise, fall, and transition times independent of load.
    Fixed(FixedTiming),
    /// Lookup-table delays and transitions indexed by arc, input slew, and output load.
    Characterized(CharacterizedTiming),
}

/// Electrical interface parameters for connecting logic cell pins to analog nets.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ElectricalParameters {
    /// Switching threshold as a fraction of supply above ground; defaults to `0.5`.
    #[serde(default = "half")]
    pub threshold: f64,
    /// Input pin capacitances, in farads, ordered to match `inputs`.
    pub input_capacitances: Vec<f64>,
    /// Name of the supply terminal; defaults to `supply`.
    #[serde(default = "supply")]
    pub supply: String,
    /// Name of the ground terminal; defaults to `ground`.
    #[serde(default = "ground")]
    pub ground: String,
}

const fn half() -> f64 {
    0.5
}

fn supply() -> String {
    "supply".to_owned()
}

fn ground() -> String {
    "ground".to_owned()
}

/// Parameters defining a combinational logic cell.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LogicCellParameters {
    /// Input pin names ordered to match truth-table bit positions.
    pub inputs: Vec<String>,
    /// Output pins paired with their truth tables.
    pub outputs: Vec<OutputParameters>,
    /// Propagation timing specification for the cell outputs.
    pub timing: TimingParameters,
    /// Optional electrical interface enabling analog connections through supply and ground
    /// terminals.
    #[serde(default)]
    pub electrical: Option<ElectricalParameters>,
}

/// A combinational logic cell: each output a function of the inputs, given by its truth
/// table.
///
/// An output fed unknown inputs is known only when every level they could take gives it the
/// same level, as a NAND with a 0 on one input is 1 whatever the other.
#[derive(Debug)]
pub struct LogicCell {
    pins: Pins,
    /// Each output's level for each row of the inputs.
    tables: Vec<Vec<bool>>,
    timing: CellTiming,
    input_capacitances: Vec<f64>,
}

/// A logic cell's timing, read and checked.
#[derive(Debug)]
enum CellTiming {
    Fixed(FixedTiming),
    Characterized(Tables),
}

/// A characterized cell's tables, with each arc found by its input, its row before the
/// change and its output.
#[derive(Debug)]
struct Tables {
    slews: Vec<f64>,
    loads: Vec<f64>,
    arcs: BTreeMap<(usize, usize, usize), Arc>,
    domain: Domain,
}

/// One arc's tables, and the level its output reaches.
#[derive(Debug)]
struct Arc {
    level: LogicLevel,
    delay: Vec<Vec<f64>>,
    transition: Vec<Vec<f64>>,
}

impl Tables {
    /// Reads `timing` for a cell of `pins` and `tables`.
    fn new(
        timing: CharacterizedTiming,
        pins: &Pins,
        tables: &[Vec<bool>],
    ) -> Result<Self, ParameterError> {
        for (name, grid) in [("transitions", &timing.slews), ("loads", &timing.loads)] {
            let rising = grid.windows(2).all(|pair| pair[0] < pair[1]);
            if grid.is_empty()
                || !rising
                || grid
                    .iter()
                    .any(|value| !(*value >= 0.0 && value.is_finite()))
            {
                return Err(ParameterError(format!(
                    "the {name} of a characterized cell rise from zero or more, not {grid:?}"
                )));
            }
        }
        let shape = (timing.slews.len(), timing.loads.len());
        let mut arcs = BTreeMap::new();
        for arc in timing.arcs {
            let position = |names: &[String], name: &str, kind: &str| {
                names.iter().position(|other| other == name).ok_or_else(|| {
                    ParameterError(format!(
                        "an arc names the {kind} {name}, which the cell does not have"
                    ))
                })
            };
            let input = position(&pins.inputs, &arc.input, "input")?;
            let output = position(&pins.outputs, &arc.output, "output")?;
            let count = pins.inputs.len();
            let row = (arc.before.len() == count
                && arc.before.chars().all(|bit| matches!(bit, '0' | '1')))
            .then(|| usize::from_str_radix(&arc.before, 2).ok())
            .flatten()
            .or_else(|| (count == 0 && arc.before.is_empty()).then_some(0))
            .ok_or_else(|| {
                ParameterError(format!(
                    "an arc's levels before its change are a 0 or a 1 for each of the \
                         {count} inputs, not {:?}",
                    arc.before
                ))
            })?;
            let after = row ^ (1 << (count - 1 - input));
            let table = &tables[output];
            if table[row] == table[after] {
                return Err(ParameterError(format!(
                    "the arc from {} to {} with the inputs at {} changes nothing",
                    arc.input, arc.output, arc.before
                )));
            }
            for (name, values) in [("delays", &arc.delay), ("transitions", &arc.transition)] {
                let fits = values.len() == shape.0 && values.iter().all(|row| row.len() == shape.1);
                if !fits || values.iter().flatten().any(|value| !value.is_finite()) {
                    return Err(ParameterError(format!(
                        "the {name} of the arc from {} to {} with the inputs at {} are {} rows \
                         of {} numbers",
                        arc.input, arc.output, arc.before, shape.0, shape.1
                    )));
                }
            }
            if arc.transition.iter().flatten().any(|&value| value <= 0.0) {
                return Err(ParameterError(format!(
                    "the transitions of the arc from {} to {} with the inputs at {} are positive",
                    arc.input, arc.output, arc.before
                )));
            }
            let arc_tables = Arc {
                level: LogicLevel::from_bool(table[after]),
                delay: arc.delay,
                transition: arc.transition,
            };
            arcs.insert((input, row, output), arc_tables);
        }
        if let Some(supply) = timing.supply {
            ParameterError::positive("the supply the tables were measured at", supply)?;
        }
        let domain = Domain {
            slews: (timing.slews[0], timing.slews[timing.slews.len() - 1]),
            loads: (timing.loads[0], timing.loads[timing.loads.len() - 1]),
            supply: timing.supply,
        };
        Ok(Self {
            slews: timing.slews,
            loads: timing.loads,
            arcs,
            domain,
        })
    }

    /// Interpolates the timing of `arc` at `slew` and `load`, clamping outside table bounds.
    fn at(&self, arc: &Arc, slew: f64, load: f64) -> Timing {
        Timing {
            delay: interpolate(&self.slews, &self.loads, &arc.delay, slew, load),
            transition: interpolate(&self.slews, &self.loads, &arc.transition, slew, load),
        }
    }

    /// Computes timing for `output` reaching `level` after `change`.
    ///
    /// Transitions starting from unknown inputs have no dedicated arc and select the earliest
    /// matching output arc reaching `level`, or the earliest arc for `output` when `level` is
    /// unknown.
    fn timing(&self, change: &Change<'_>, output: usize, level: LogicLevel, load: f64) -> Timing {
        let count = change.before.len();
        let row = change
            .before
            .iter()
            .enumerate()
            .try_fold(0, |row, (index, level)| match level {
                LogicLevel::High => Some(row | 1 << (count - 1 - index)),
                LogicLevel::Low => Some(row),
                LogicLevel::Unknown => None,
            });
        if let Some(arc) = row.and_then(|row| self.arcs.get(&(change.input, row, output))) {
            return self.at(arc, change.slew, load);
        }
        self.arcs
            .iter()
            .filter(|((_, _, arc_output), arc)| {
                *arc_output == output && (level == LogicLevel::Unknown || arc.level == level)
            })
            .map(|(_, arc)| self.at(arc, change.slew, load))
            .min_by(|a, b| a.delay.total_cmp(&b.delay))
            .unwrap_or(Timing {
                delay: 0.0,
                transition: self.slews[0].max(f64::MIN_POSITIVE),
            })
    }
}

/// Interpolates `table` (indexed by `rows` and `columns`) bilinearly at `(row, column)`,
/// clamping to edge values outside the grid.
fn interpolate(rows: &[f64], columns: &[f64], table: &[Vec<f64>], row: f64, column: f64) -> f64 {
    let (r, rf) = bracket(rows, row);
    let (c, cf) = bracket(columns, column);
    let at = |r: usize, c: usize| table[r][c];
    let next_r = (r + 1).min(rows.len() - 1);
    let next_c = (c + 1).min(columns.len() - 1);
    let low = at(r, c) + (at(r, next_c) - at(r, c)) * cf;
    let high = at(next_r, c) + (at(next_r, next_c) - at(next_r, c)) * cf;
    low + (high - low) * rf
}

/// Locates the index in `points` at or below `value` and returns the interpolation fraction
/// toward the next point, clamped to grid bounds.
fn bracket(points: &[f64], value: f64) -> (usize, f64) {
    let last = points.len() - 1;
    if last == 0 || value <= points[0] {
        return (0, 0.0);
    }
    if value >= points[last] {
        return (last, 0.0);
    }
    let index = points.partition_point(|&point| point <= value) - 1;
    (
        index,
        (value - points[index]) / (points[index + 1] - points[index]),
    )
}

impl LogicCell {
    /// Evaluates truth `table` for `inputs`, returning a known level only when every possible
    /// assignment to unknown inputs agrees.
    fn level(table: &[bool], inputs: &[LogicLevel]) -> LogicLevel {
        let count = inputs.len();
        let mut known = 0;
        let mut unknown = Vec::new();
        for (index, level) in inputs.iter().enumerate() {
            let bit = 1 << (count - 1 - index);
            match level {
                LogicLevel::High => known |= bit,
                LogicLevel::Low => {}
                LogicLevel::Unknown => unknown.push(bit),
            }
        }
        let first = table[known];
        let agree = (1..1_usize << unknown.len()).all(|choice| {
            let row = unknown
                .iter()
                .enumerate()
                .filter(|(position, _)| choice >> position & 1 == 1)
                .fold(known, |row, (_, bit)| row | bit);
            table[row] == first
        });
        if agree {
            LogicLevel::from_bool(first)
        } else {
            LogicLevel::Unknown
        }
    }
}

impl NativeLogic for LogicCell {
    const NAME: &'static str = "logic_cell";
    type Parameters = LogicCellParameters;

    fn new(parameters: LogicCellParameters) -> Result<Self, ParameterError> {
        let LogicCellParameters {
            inputs,
            outputs,
            timing,
            electrical,
        } = parameters;
        if inputs.len() > MAX_INPUTS {
            return Err(ParameterError(format!(
                "a logic cell has at most {MAX_INPUTS} inputs, not {}",
                inputs.len()
            )));
        }
        if outputs.is_empty() {
            return Err(ParameterError(
                "a logic cell has an output or more".to_owned(),
            ));
        }
        let rows = 1_usize << inputs.len();
        let mut tables = Vec::with_capacity(outputs.len());
        for output in &outputs {
            let table: Option<Vec<bool>> = output
                .table
                .chars()
                .map(|row| match row {
                    '0' => Some(false),
                    '1' => Some(true),
                    _ => None,
                })
                .collect();
            match table {
                Some(table) if table.len() == rows => tables.push(table),
                _ => {
                    return Err(ParameterError(format!(
                        "the table of output {} has a 0 or a 1 for each of the {rows} rows of \
                         {} inputs, not {:?}",
                        output.name,
                        inputs.len(),
                        output.table
                    )));
                }
            }
        }
        let (electrical, input_capacitances) = match electrical {
            Some(ElectricalParameters {
                threshold,
                input_capacitances,
                supply,
                ground,
            }) => {
                if !(threshold > 0.0 && threshold < 1.0) {
                    return Err(ParameterError(format!(
                        "the threshold is a fraction of the supply above 0 and below 1, not \
                         {threshold:?}"
                    )));
                }
                if input_capacitances.len() != inputs.len() {
                    return Err(ParameterError(format!(
                        "a logic cell gives a capacitance for each of its {} inputs, not {}",
                        inputs.len(),
                        input_capacitances.len()
                    )));
                }
                for &capacitance in &input_capacitances {
                    ParameterError::non_negative("an input capacitance", capacitance)?;
                }
                let electrical = Electrical {
                    threshold,
                    supply,
                    ground,
                };
                (Some(electrical), input_capacitances)
            }
            None => (None, vec![0.0; inputs.len()]),
        };
        let pins = Pins {
            inputs,
            outputs: outputs.into_iter().map(|output| output.name).collect(),
            electrical,
        };
        check_names(&pins)?;
        let timing = match timing {
            TimingParameters::Fixed(timing) => {
                for (name, value) in [("rise", timing.rise), ("fall", timing.fall)] {
                    ParameterError::non_negative(&format!("the {name} delay"), value)?;
                }
                ParameterError::positive("the transition", timing.transition)?;
                CellTiming::Fixed(timing)
            }
            TimingParameters::Characterized(timing) => {
                CellTiming::Characterized(Tables::new(timing, &pins, &tables)?)
            }
        };
        Ok(Self {
            pins,
            tables,
            timing,
            input_capacitances,
        })
    }
}

impl Logic for LogicCell {
    fn pins(&self) -> Pins {
        self.pins.clone()
    }

    fn levels(&self, inputs: &[LogicLevel], outputs: &mut [LogicLevel]) {
        for (output, table) in outputs.iter_mut().zip(&self.tables) {
            *output = Self::level(table, inputs);
        }
    }

    fn timing(&self, change: &Change<'_>, output: usize, level: LogicLevel, load: f64) -> Timing {
        match &self.timing {
            CellTiming::Fixed(timing) => Timing {
                delay: match level {
                    LogicLevel::High => timing.rise,
                    LogicLevel::Low => timing.fall,
                    LogicLevel::Unknown => timing.rise.min(timing.fall),
                },
                transition: timing.transition,
            },
            CellTiming::Characterized(tables) => tables.timing(change, output, level, load),
        }
    }

    fn input_capacitance(&self, input: usize) -> f64 {
        self.input_capacitances[input]
    }

    fn domain(&self) -> Option<&Domain> {
        match &self.timing {
            CellTiming::Fixed(_) => None,
            CellTiming::Characterized(tables) => Some(&tables.domain),
        }
    }
}

/// Checks that the pins and terminals of a device have distinct names.
fn check_names(pins: &Pins) -> Result<(), ParameterError> {
    let mut seen = BTreeSet::new();
    for name in pins.names() {
        if name.is_empty() {
            return Err(ParameterError("a pin's name is empty".to_owned()));
        }
        if !seen.insert(name.clone()) {
            return Err(ParameterError(format!("two pins are named {name}")));
        }
    }
    Ok(())
}

/// The parameters of a logic source.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LogicSourceParameters {
    /// The level at time zero.
    pub initial: LogicLevel,
    /// Each later change: its time, in seconds, and the new level, in time order.
    #[serde(default)]
    pub changes: Vec<(f64, LogicLevel)>,
    /// The edges' transition, the time each takes to swing from ground to supply, in
    /// seconds.
    pub transition: f64,
}

/// A logic source: one output, `output`, that follows a pattern of levels over time.
///
/// Each change is the time its edge crosses half the supply.
#[derive(Debug)]
pub struct LogicSource {
    parameters: LogicSourceParameters,
}

impl NativeLogic for LogicSource {
    const NAME: &'static str = "logic_source";
    type Parameters = LogicSourceParameters;

    fn new(parameters: LogicSourceParameters) -> Result<Self, ParameterError> {
        ParameterError::positive("the transition", parameters.transition)?;
        let mut previous = 0.0;
        for &(time, _) in &parameters.changes {
            if !(time > previous && time.is_finite()) {
                return Err(ParameterError(format!(
                    "the changes' times rise from above zero, and {time:?} follows \
                     {previous:?}"
                )));
            }
            previous = time;
        }
        Ok(Self { parameters })
    }
}

impl Logic for LogicSource {
    fn pins(&self) -> Pins {
        Pins {
            inputs: Vec::new(),
            outputs: vec!["output".to_owned()],
            electrical: None,
        }
    }

    fn levels(&self, _: &[LogicLevel], outputs: &mut [LogicLevel]) {
        outputs[0] = self.parameters.initial;
    }

    fn timing(&self, _: &Change<'_>, _: usize, _: LogicLevel, _: f64) -> Timing {
        Timing {
            delay: 0.0,
            transition: self.parameters.transition,
        }
    }

    fn stimuli(&self, stop: f64, stimuli: &mut Vec<Stimulus>) {
        stimuli.extend(
            self.parameters
                .changes
                .iter()
                .take_while(|&&(time, _)| time <= stop)
                .map(|&(time, level)| Stimulus {
                    time,
                    output: 0,
                    level,
                    transition: self.parameters.transition,
                }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::assert_close;

    fn cell(parameters: &str) -> Result<LogicCell, ParameterError> {
        LogicCell::new(
            serde_json::from_str(parameters).map_err(|error| ParameterError(error.to_string()))?,
        )
    }

    fn nand() -> LogicCell {
        cell(
            r#"{"inputs": ["a", "b"], "outputs": [{"name": "y", "table": "1110"}],
                "timing": {"fixed": {"rise": 2e-9, "fall": 1e-9, "transition": 5e-10}}}"#,
        )
        .unwrap()
    }

    #[test]
    fn a_cell_follows_its_truth_table_and_knows_what_unknowns_cannot_change() {
        use LogicLevel::{High, Low, Unknown};
        let nand = nand();
        let mut output = [Unknown];
        for (inputs, expected) in [
            ([Low, Low], High),
            ([Low, High], High),
            ([High, Low], High),
            ([High, High], Low),
            ([Low, Unknown], High),
            ([Unknown, High], Unknown),
            ([Unknown, Unknown], Unknown),
        ] {
            nand.levels(&inputs, &mut output);
            assert_eq!(output, [expected], "{inputs:?}");
        }
    }

    #[test]
    fn a_fixed_timing_rises_and_falls_with_its_own_delays() {
        let nand = nand();
        let change = Change {
            before: &[LogicLevel::Low, LogicLevel::High],
            input: 0,
            slew: 1e-9,
        };
        let timing = |delay| Timing {
            delay,
            transition: 5e-10,
        };
        assert_eq!(nand.timing(&change, 0, LogicLevel::High, 0.0), timing(2e-9));
        assert_eq!(nand.timing(&change, 0, LogicLevel::Low, 0.0), timing(1e-9));
        assert_eq!(
            nand.timing(&change, 0, LogicLevel::Unknown, 0.0),
            timing(1e-9)
        );
    }

    #[test]
    fn rejects_cells_it_cannot_read() {
        let timing = r#""timing": {"fixed": {"rise": 0, "fall": 0, "transition": 1e-9}}"#;
        for (parameters, message) in [
            (
                format!(
                    r#"{{"inputs": ["a"], "outputs": [{{"name": "y", "table": "011"}}], {timing}}}"#
                ),
                "the table of output y has a 0 or a 1 for each of the 2 rows of 1 inputs, \
                 not \"011\"",
            ),
            (
                format!(r#"{{"inputs": ["a"], "outputs": [], {timing}}}"#),
                "a logic cell has an output or more",
            ),
            (
                format!(
                    r#"{{"inputs": ["a"], "outputs": [{{"name": "a", "table": "01"}}], {timing}}}"#
                ),
                "two pins are named a",
            ),
            (
                format!(
                    r#"{{"inputs": ["a"], "outputs": [{{"name": "y", "table": "01"}}], {timing},
                        "electrical": {{"input_capacitances": []}}}}"#
                ),
                "a logic cell gives a capacitance for each of its 1 inputs, not 0",
            ),
            (
                format!(
                    r#"{{"inputs": ["a"], "outputs": [{{"name": "y", "table": "01"}}], {timing},
                        "electrical": {{"threshold": 1, "input_capacitances": [0]}}}}"#
                ),
                "the threshold is a fraction of the supply above 0 and below 1, not 1.0",
            ),
        ] {
            assert_eq!(cell(&parameters).unwrap_err().0, message);
        }
    }

    #[test]
    fn a_cell_with_electrical_pins_has_a_supply_and_a_ground() {
        let cell = cell(
            r#"{"inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
                "timing": {"fixed": {"rise": 0, "fall": 0, "transition": 1e-9}},
                "electrical": {"input_capacitances": [2e-15]}}"#,
        )
        .unwrap();
        assert_eq!(cell.pins().names(), ["a", "y", "supply", "ground"]);
        assert_close(&[cell.input_capacitance(0)], &[2e-15]);
    }

    /// Builds an inverter characterized over two slews and two loads whose delay increases by
    /// 1 ps/fF and 0.1 times the input transition.
    fn characterized() -> LogicCell {
        cell(
            r#"{"inputs": ["a"], "outputs": [{"name": "y", "table": "10"}],
                "timing": {"characterized": {
                    "slews": [1e-11, 3e-11], "loads": [0, 2e-15],
                    "arcs": [
                        {"input": "a", "before": "0", "output": "y",
                         "delay": [[1e-11, 1.2e-11], [1.2e-11, 1.4e-11]],
                         "transition": [[2e-11, 3e-11], [2e-11, 3e-11]]},
                        {"input": "a", "before": "1", "output": "y",
                         "delay": [[2e-11, 2.2e-11], [2.2e-11, 2.4e-11]],
                         "transition": [[4e-11, 5e-11], [4e-11, 5e-11]]}
                    ]}},
                "electrical": {"input_capacitances": [1e-15], "supply": "vdd", "ground": "vss"}}"#,
        )
        .unwrap()
    }

    #[test]
    fn a_characterized_cell_interpolates_its_arc_s_tables() {
        let cell = characterized();
        assert_eq!(cell.pins().names(), ["a", "y", "vdd", "vss"]);
        let falling = Change {
            before: &[LogicLevel::Low],
            input: 0,
            slew: 2e-11,
        };
        let timing = cell.timing(&falling, 0, LogicLevel::Low, 1e-15);
        assert_close(&[timing.delay, timing.transition], &[1.2e-11, 2.5e-11]);
        // Beyond the tables, the timing holds their edge.
        let timing = cell.timing(&falling, 0, LogicLevel::Low, 5e-15);
        assert_close(&[timing.delay], &[1.3e-11]);
        let rising = Change {
            before: &[LogicLevel::High],
            ..falling
        };
        let timing = cell.timing(&rising, 0, LogicLevel::High, 0.0);
        assert_close(&[timing.delay], &[2.1e-11]);
        // From an unknown input, the earliest arc to the level, or to any level.
        let unknown = Change {
            before: &[LogicLevel::Unknown],
            ..falling
        };
        assert_close(
            &[cell.timing(&unknown, 0, LogicLevel::High, 0.0).delay],
            &[2.1e-11],
        );
        assert_close(
            &[cell.timing(&unknown, 0, LogicLevel::Unknown, 0.0).delay],
            &[1.1e-11],
        );
    }

    #[test]
    fn rejects_tables_it_cannot_read() {
        let arc = |before: &str, delay: &str| {
            format!(
                r#"{{"inputs": ["a"], "outputs": [{{"name": "y", "table": "10"}}],
                    "timing": {{"characterized": {{"slews": [1e-11], "loads": [0],
                        "arcs": [{{"input": "a", "before": "{before}", "output": "y",
                                  "delay": {delay}, "transition": [[1e-11]]}}]}}}}}}"#
            )
        };
        for (parameters, message) in [
            (
                arc("2", "[[1e-11]]"),
                "an arc's levels before its change are a 0 or a 1 for each of the 1 inputs, \
                 not \"2\"",
            ),
            (
                arc("0", "[[1e-11, 2e-11]]"),
                "the delays of the arc from a to y with the inputs at 0 are 1 rows of 1 numbers",
            ),
        ] {
            assert_eq!(cell(&parameters).unwrap_err().0, message);
        }
        let constant = r#"{"inputs": ["a"], "outputs": [{"name": "y", "table": "11"}],
            "timing": {"characterized": {"slews": [1e-11], "loads": [0],
                "arcs": [{"input": "a", "before": "0", "output": "y",
                          "delay": [[1e-11]], "transition": [[1e-11]]}]}}}"#;
        assert_eq!(
            cell(constant).unwrap_err().0,
            "the arc from a to y with the inputs at 0 changes nothing"
        );
    }

    #[test]
    fn a_source_changes_on_its_pattern() {
        let source = LogicSource::new(
            serde_json::from_str(
                r#"{"initial": "0", "changes": [[1e-9, "1"], [3e-9, "x"]], "transition": 1e-10}"#,
            )
            .unwrap(),
        )
        .unwrap();
        let mut output = [LogicLevel::Unknown];
        source.levels(&[], &mut output);
        assert_eq!(output, [LogicLevel::Low]);
        let mut stimuli = Vec::new();
        source.stimuli(2e-9, &mut stimuli);
        assert_eq!(
            stimuli,
            [Stimulus {
                time: 1e-9,
                output: 0,
                level: LogicLevel::High,
                transition: 1e-10
            }]
        );
        let error = LogicSource::new(
            serde_json::from_str(
                r#"{"initial": "0", "changes": [[2e-9, "1"], [1e-9, "0"]], "transition": 1e-10}"#,
            )
            .unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            error.0,
            "the changes' times rise from above zero, and 1e-9 follows 2e-9"
        );
    }
}
