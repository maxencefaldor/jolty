//! Logic devices, which an event scheduler simulates, and the elements that join them to the
//! analog circuit.
//!
//! A logic device reads the [`Level`]s of its input nets and drives its output nets to new
//! levels after delays. Its pins join logic nets, which hold a level rather than a voltage,
//! so the engine jumps from one change to the next instead of integrating. Where a logic pin
//! joins an electrical net, the engine places a connect element between them: a
//! [`ThresholdDetector`] turns the net's voltage into the level an input reads, and a
//! [`LogicDriver`] turns an output's level into a voltage between the device's supply and
//! ground.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::marker::PhantomData;

use jolty_ir::Parameters;
use jolty_numerics::Entry;
use schemars::{JsonSchema, Schema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::device::{read, schema, written};
use crate::stamp::CurrentBranch;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Node, Output,
    ParameterError, TerminalPair, Unknown,
};

/// The most inputs a logic cell may have: its truth table has a row per combination.
pub const MAX_INPUTS: usize = 16;

/// The level of a logic net.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum Level {
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

impl Level {
    /// The level of `bit`.
    #[must_use]
    pub const fn from_bool(bit: bool) -> Self {
        if bit { Self::High } else { Self::Low }
    }

    /// The fraction of the supply a driver outputs at the level: 0, 1, or one half for
    /// [`Level::Unknown`].
    #[must_use]
    pub const fn fraction(self) -> f64 {
        match self {
            Self::Low => 0.0,
            Self::High => 1.0,
            Self::Unknown => 0.5,
        }
    }

    /// The level of a net whose drivers drive `levels`: theirs if they all agree, and
    /// [`Level::Unknown`] if they disagree or there are none.
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

impl fmt::Display for Level {
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
    pub before: &'a [Level],
    /// The input that changed.
    pub input: usize,
    /// The input's transition: the time its edge takes to swing from ground to supply, in
    /// seconds.
    pub slew: f64,
}

/// When an output reaches its new level, and how fast it moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timing {
    /// The time from the input's change to the output's, in seconds, each measured where it
    /// crosses half the supply.
    pub delay: f64,
    /// The output's transition: the time its edge takes to swing from ground to supply, in
    /// seconds.
    pub transition: f64,
}

/// A change a logic device makes on its own, such as a source's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stimulus {
    /// When the output reaches its new level, in seconds.
    pub time: f64,
    /// The output that changes.
    pub output: usize,
    /// The new level.
    pub level: Level,
    /// The output's transition, in seconds.
    pub transition: f64,
}

/// One placed logic device.
pub trait Logic: fmt::Debug + Send + Sync {
    /// The device's pins.
    fn pins(&self) -> Pins;

    /// Writes the levels the outputs settle at when the inputs hold `inputs`.
    fn levels(&self, inputs: &[Level], outputs: &mut [Level]);

    /// When and how fast `output` reaches `level` after `change`, while driving a load of
    /// `load` farads.
    fn timing(&self, change: &Change<'_>, output: usize, level: Level, load: f64) -> Timing;

    /// Appends the changes the device makes on its own in `(0, stop]`, in time order.
    fn stimuli(&self, stop: f64, stimuli: &mut Vec<Stimulus>) {
        let _ = (stop, stimuli);
    }

    /// The capacitance input `input` loads its net with, in farads.
    fn input_capacitance(&self, input: usize) -> f64 {
        let _ = input;
        0.0
    }

    /// Where the device's timing holds, if anywhere less than everywhere: a characterized
    /// cell's, where it was measured.
    fn domain(&self) -> Option<&Domain> {
        None
    }
}

/// Where a logic device's timing holds: the ranges it was measured over.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    /// The least and the most input transition, in seconds.
    pub slews: (f64, f64),
    /// The least and the most output load, in farads.
    pub loads: (f64, f64),
    /// The voltage from ground to supply it was measured at, if it was.
    pub supply: Option<f64>,
}

/// How a logic device's pins may join electrical nets: through connect elements between the
/// device's supply and ground terminals.
#[derive(Debug, Clone, PartialEq)]
pub struct Electrical {
    /// The fraction of the supply, from ground, at which an input switches.
    pub threshold: f64,
    /// The name of the supply terminal.
    pub supply: String,
    /// The name of the ground terminal.
    pub ground: String,
}

/// The pins of a logic device: its inputs and outputs, which join logic nets, and whether it
/// has the electrical terminals that let them join electrical nets.
#[derive(Debug, Clone, PartialEq)]
pub struct Pins {
    /// The inputs' names, in the order [`Logic::levels`] reads them.
    pub inputs: Vec<String>,
    /// The outputs' names, in the order [`Logic::levels`] writes them.
    pub outputs: Vec<String>,
    /// With them, the device's supply and ground terminals too.
    pub electrical: Option<Electrical>,
}

impl Pins {
    /// The names of every pin and terminal, inputs first, then outputs, then the supply
    /// and the ground if the device has them.
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

/// A kind of logic device: its name, its parameters, and how to make one from them.
pub trait LogicModel: Send + Sync {
    /// The name circuits refer to the model by, such as `logic_cell`.
    fn name(&self) -> &'static str;

    /// Reads `parameters` and makes a device, which tells its pins.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is missing, unknown or out of range.
    fn instantiate(&self, parameters: &Parameters) -> Result<Box<dyn Logic>, ParameterError>;

    /// Every parameter a device of the model takes, with its type, its default and its
    /// description, as a JSON schema of an object, every type within it written in place.
    fn schema(&self) -> Schema;

    /// `parameters` as the model takes them: each one left out, written with its default.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when the parameters cannot be read.
    fn defaulted(&self, parameters: &Parameters) -> Result<Parameters, ParameterError>;
}

/// A logic device implemented in Rust, whose parameters deserialize into a typed struct.
pub trait NativeLogic: Logic + Sized + 'static {
    /// The model's name.
    const NAME: &'static str;
    /// The parameters, deserialized from the instance's parameter map.
    type Parameters: DeserializeOwned + Serialize + JsonSchema;

    /// Checks `parameters` and makes the device.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] when a parameter is out of range.
    fn new(parameters: Self::Parameters) -> Result<Self, ParameterError>;
}

/// The [`LogicModel`] of a [`NativeLogic`] device.
pub struct NativeLogicModel<D>(PhantomData<fn() -> D>);

impl<D: NativeLogic> NativeLogicModel<D> {
    /// The model of `D`.
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

/// How a logic cell's outputs follow its inputs in time.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum TimingParameters {
    /// The same delays after any input changes, whatever the load.
    Fixed(FixedTiming),
    /// Delays and transitions that depend on the arc, the input transition and the load.
    Characterized(CharacterizedTiming),
}

/// How a logic cell's pins cross to electrical nets.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ElectricalParameters {
    /// The fraction of the supply, from ground, at which an input switches; one half unless
    /// given.
    #[serde(default = "half")]
    pub threshold: f64,
    /// The capacitance of each input, in farads, in the order of the inputs.
    pub input_capacitances: Vec<f64>,
    /// The name of the supply terminal; `supply` unless given.
    #[serde(default = "supply")]
    pub supply: String,
    /// The name of the ground terminal; `ground` unless given.
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

/// The parameters of a logic cell.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LogicCellParameters {
    /// The inputs' names, in the order of the truth tables' bits.
    pub inputs: Vec<String>,
    /// The outputs, each with its truth table.
    pub outputs: Vec<OutputParameters>,
    /// How the outputs follow the inputs in time.
    pub timing: TimingParameters,
    /// With them, the cell has terminals `supply` and `ground`, and its pins may join
    /// electrical nets.
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
    level: Level,
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
            let row = (arc.before.len() == count && arc.before.chars().all(|bit| matches!(bit, '0' | '1')))
                .then(|| usize::from_str_radix(&arc.before, 2).ok())
                .flatten()
                .or_else(|| (count == 0 && arc.before.is_empty()).then_some(0))
                .ok_or_else(|| ParameterError(format!(
                    "an arc's levels before its change are a 0 or a 1 for each of the {count} inputs, not {:?}",
                    arc.before
                )))?;
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
                level: Level::from_bool(table[after]),
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

    /// The timing of `arc` at `slew` and `load`, interpolated between the tables' points
    /// and held at their edges.
    fn at(&self, arc: &Arc, slew: f64, load: f64) -> Timing {
        Timing {
            delay: interpolate(&self.slews, &self.loads, &arc.delay, slew, load),
            transition: interpolate(&self.slews, &self.loads, &arc.transition, slew, load),
        }
    }

    /// The timing of `output` reaching `level` after `change`. A change from inputs some of
    /// which are unknown has no arc of its own, and takes the earliest of the output's arcs
    /// that reach the level, or of all its arcs for an unknown level.
    fn timing(&self, change: &Change<'_>, output: usize, level: Level, load: f64) -> Timing {
        let count = change.before.len();
        let row = change
            .before
            .iter()
            .enumerate()
            .try_fold(0, |row, (index, level)| match level {
                Level::High => Some(row | 1 << (count - 1 - index)),
                Level::Low => Some(row),
                Level::Unknown => None,
            });
        if let Some(arc) = row.and_then(|row| self.arcs.get(&(change.input, row, output))) {
            return self.at(arc, change.slew, load);
        }
        self.arcs
            .iter()
            .filter(|((_, _, arc_output), arc)| {
                *arc_output == output && (level == Level::Unknown || arc.level == level)
            })
            .map(|(_, arc)| self.at(arc, change.slew, load))
            .min_by(|a, b| a.delay.total_cmp(&b.delay))
            .unwrap_or(Timing {
                delay: 0.0,
                transition: self.slews[0].max(f64::MIN_POSITIVE),
            })
    }
}

/// The value of `table`, one row per `rows` point and one column per `columns` point, at
/// `(row, column)`: linear between the points, and held beyond the first and the last.
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

/// The point of `points` at or below `value`, and the fraction of the way to the next, held
/// to the points' span.
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
    /// The level of `table` at `inputs`, known only if every level the unknown inputs could
    /// take gives the same.
    fn level(table: &[bool], inputs: &[Level]) -> Level {
        let count = inputs.len();
        let mut known = 0;
        let mut unknown = Vec::new();
        for (index, level) in inputs.iter().enumerate() {
            let bit = 1 << (count - 1 - index);
            match level {
                Level::High => known |= bit,
                Level::Low => {}
                Level::Unknown => unknown.push(bit),
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
            Level::from_bool(first)
        } else {
            Level::Unknown
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

    fn levels(&self, inputs: &[Level], outputs: &mut [Level]) {
        for (output, table) in outputs.iter_mut().zip(&self.tables) {
            *output = Self::level(table, inputs);
        }
    }

    fn timing(&self, change: &Change<'_>, output: usize, level: Level, load: f64) -> Timing {
        match &self.timing {
            CellTiming::Fixed(timing) => Timing {
                delay: match level {
                    Level::High => timing.rise,
                    Level::Low => timing.fall,
                    Level::Unknown => timing.rise.min(timing.fall),
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
    pub initial: Level,
    /// Each later change: its time, in seconds, and the new level, in time order.
    #[serde(default)]
    pub changes: Vec<(f64, Level)>,
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

    fn levels(&self, _: &[Level], outputs: &mut [Level]) {
        outputs[0] = self.parameters.initial;
    }

    fn timing(&self, _: &Change<'_>, _: usize, _: Level, _: f64) -> Timing {
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

/// The analog side of a logic output: it drives its terminal to a level's fraction of the
/// supply, moving there in a ramp.
pub trait Drive {
    /// Starts moving to `level` at `time`, at the rate that swings ground to supply in
    /// `transition` seconds; at once when `transition` is zero.
    fn drive(&mut self, time: f64, level: Level, transition: f64);

    /// The time the last movement ends.
    fn settles_at(&self) -> f64;

    /// The unknown current through the driver, whose equation holds its voltage.
    fn current(&self) -> Unknown;
}

/// The analog side of a logic input: it reads a voltage against a threshold.
pub trait Detect {
    /// How far the input lies above the threshold at `input`, in volts.
    fn margin(&self, input: &Input<'_>) -> f64;

    /// The supply's voltage above ground at `input`, the swing of a full edge.
    fn swing(&self, input: &Input<'_>) -> f64;
}

/// The parameters of a logic driver: none.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LogicDriverParameters {}

/// A logic driver: the connect element from a logic output to an electrical net.
///
/// Between terminals `output` and `ground` it holds the fraction `s` of the voltage from
/// `ground` to `supply`, the fraction of the output's level, so `v_output − v_ground =
/// s (v_supply − v_ground)`. A change of level moves `s` in a ramp. Its current, flowing
/// from `output` through the driver to `ground`, is an unknown labelled `current`.
#[derive(Debug)]
pub struct LogicDriver {
    /// The branch from the output to the ground, through the driver.
    branch: CurrentBranch,
    supply: Node,
    ground: Node,
    /// The ramp of `s`: from `(start, from)` to `(end, to)`.
    ramp: (f64, f64, f64, f64),
    /// The entries at (current, ground) and (current, supply).
    entries: [Option<Entry>; 2],
}

impl LogicDriver {
    /// The fraction of the supply the driver holds at `time`.
    #[must_use]
    pub fn fraction(&self, time: f64) -> f64 {
        let (start, from, end, to) = self.ramp;
        if time >= end {
            to
        } else if time <= start {
            from
        } else {
            from + (to - from) * (time - start) / (end - start)
        }
    }
}

impl NativeDevice for LogicDriver {
    const NAME: &'static str = "logic_driver";
    type Parameters = LogicDriverParameters;

    fn interface(_: &LogicDriverParameters) -> Interface {
        Interface::new(
            &["output", "supply", "ground"],
            &[Branch::new(0, 2, BranchKind::VoltageSource)],
        )
    }

    fn new(_: LogicDriverParameters, binding: &mut Binding<'_>) -> Result<Self, ParameterError> {
        let (output, supply, ground) = (
            binding.terminal(0),
            binding.terminal(1),
            binding.terminal(2),
        );
        let branch = CurrentBranch::new(binding, "current", [output, ground]);
        let row = Some(branch.unknown());
        let entries = [binding.entry(row, ground), binding.entry(row, supply)];
        Ok(Self {
            branch,
            supply,
            ground,
            ramp: (
                0.0,
                Level::Unknown.fraction(),
                0.0,
                Level::Unknown.fraction(),
            ),
            entries,
        })
    }
}

impl Device for LogicDriver {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        // The output held at its fraction of the supply's swing above the ground.
        let fraction = self.fraction(input.time);
        let swing = input.value(self.supply) - input.value(self.ground);
        self.branch
            .add(input, output, self.branch.voltage(input) - fraction * swing);
        let [current_ground, current_supply] = self.entries;
        output.add_conductance(current_ground, fraction);
        output.add_conductance(current_supply, -fraction);
    }

    fn as_driver(&mut self) -> Option<&mut dyn Drive> {
        Some(self)
    }
}

impl Drive for LogicDriver {
    fn drive(&mut self, time: f64, level: Level, transition: f64) {
        let from = self.fraction(time);
        let to = level.fraction();
        self.ramp = (time, from, time + transition * (to - from).abs(), to);
    }

    fn settles_at(&self) -> f64 {
        self.ramp.2
    }

    fn current(&self) -> Unknown {
        self.branch.unknown()
    }
}

/// The parameters of a threshold detector.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ThresholdDetectorParameters {
    /// The fraction of the supply, from ground, at which the level switches.
    pub threshold: f64,
    /// The input's capacitance to ground, in farads.
    pub capacitance: f64,
}

/// A threshold detector: the connect element from an electrical net to a logic input.
///
/// It reads terminal `input` against the fraction `threshold` of the voltage from `ground`
/// to `supply`, and loads it with `capacitance` to `ground`.
#[derive(Debug)]
pub struct ThresholdDetector {
    pair: TerminalPair,
    supply: Node,
    ground: Node,
    parameters: ThresholdDetectorParameters,
}

impl NativeDevice for ThresholdDetector {
    const NAME: &'static str = "threshold_detector";
    type Parameters = ThresholdDetectorParameters;

    fn interface(parameters: &ThresholdDetectorParameters) -> Interface {
        let branches = if parameters.capacitance > 0.0 {
            &[Branch::new(0, 2, BranchKind::Capacitive)][..]
        } else {
            &[]
        };
        Interface::new(&["input", "supply", "ground"], branches)
    }

    fn new(
        parameters: ThresholdDetectorParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        ParameterError::non_negative("the capacitance", parameters.capacitance)?;
        let (input, supply, ground) = (
            binding.terminal(0),
            binding.terminal(1),
            binding.terminal(2),
        );
        Ok(Self {
            pair: TerminalPair::new(binding, input, ground),
            supply,
            ground,
            parameters,
        })
    }
}

impl Device for ThresholdDetector {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let capacitance = self.parameters.capacitance;
        self.pair
            .add_charge(output, capacitance * self.pair.voltage(input));
        self.pair.add_capacitance(output, capacitance);
    }

    fn as_detector(&self) -> Option<&dyn Detect> {
        Some(self)
    }
}

impl Detect for ThresholdDetector {
    fn margin(&self, input: &Input<'_>) -> f64 {
        self.pair.voltage(input) - self.parameters.threshold * self.swing(input)
    }

    fn swing(&self, input: &Input<'_>) -> f64 {
        input.value(self.supply) - input.value(self.ground)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close, assert_jacobians_match};

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
        use Level::{High, Low, Unknown};
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
            before: &[Level::Low, Level::High],
            input: 0,
            slew: 1e-9,
        };
        let timing = |delay| Timing {
            delay,
            transition: 5e-10,
        };
        assert_eq!(nand.timing(&change, 0, Level::High, 0.0), timing(2e-9));
        assert_eq!(nand.timing(&change, 0, Level::Low, 0.0), timing(1e-9));
        assert_eq!(nand.timing(&change, 0, Level::Unknown, 0.0), timing(1e-9));
    }

    #[test]
    fn rejects_cells_it_cannot_read() {
        let timing = r#""timing": {"fixed": {"rise": 0, "fall": 0, "transition": 1e-9}}"#;
        for (parameters, message) in [
            (
                format!(
                    r#"{{"inputs": ["a"], "outputs": [{{"name": "y", "table": "011"}}], {timing}}}"#
                ),
                "the table of output y has a 0 or a 1 for each of the 2 rows of 1 inputs, not \"011\"",
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

    /// An inverter characterized at two transitions and two loads: each delay grows by a
    /// picosecond per femtofarad and by a tenth of the input's transition.
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
            before: &[Level::Low],
            input: 0,
            slew: 2e-11,
        };
        let timing = cell.timing(&falling, 0, Level::Low, 1e-15);
        assert_close(&[timing.delay, timing.transition], &[1.2e-11, 2.5e-11]);
        // Beyond the tables, the timing holds their edge.
        let timing = cell.timing(&falling, 0, Level::Low, 5e-15);
        assert_close(&[timing.delay], &[1.3e-11]);
        let rising = Change {
            before: &[Level::High],
            ..falling
        };
        let timing = cell.timing(&rising, 0, Level::High, 0.0);
        assert_close(&[timing.delay], &[2.1e-11]);
        // From an unknown input, the earliest arc to the level, or to any level.
        let unknown = Change {
            before: &[Level::Unknown],
            ..falling
        };
        assert_close(
            &[cell.timing(&unknown, 0, Level::High, 0.0).delay],
            &[2.1e-11],
        );
        assert_close(
            &[cell.timing(&unknown, 0, Level::Unknown, 0.0).delay],
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
                "an arc's levels before its change are a 0 or a 1 for each of the 1 inputs, not \"2\"",
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
        let mut output = [Level::Unknown];
        source.levels(&[], &mut output);
        assert_eq!(output, [Level::Low]);
        let mut stimuli = Vec::new();
        source.stimuli(2e-9, &mut stimuli);
        assert_eq!(
            stimuli,
            [Stimulus {
                time: 1e-9,
                output: 0,
                level: Level::High,
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

    #[test]
    fn a_driver_ramps_between_ground_and_supply() {
        let mut bench = Bench::new::<LogicDriver>("{}");
        assert_eq!(bench.unknowns(), ["n0", "n1", "n2", "device:current"]);
        let driver = bench.device().as_driver().unwrap();
        driver.drive(0.0, Level::Low, 0.0);
        driver.drive(1.0, Level::High, 2.0);
        assert_close(&[driver.settles_at()], &[3.0]);
        // Halfway up the ramp, the output holds half of the 5 V from ground at 1 V.
        let x = [3.0, 6.0, 1.0, 0.0];
        let evaluation = bench.evaluate_with(&Input::new(&x, Some(2.0), 0.0));
        assert!((evaluation.residual[3] - (2.0 - 0.5 * 5.0)).abs() < 1e-12);
        // Driven back down from halfway, it takes half the transition.
        let driver = bench.device().as_driver().unwrap();
        driver.drive(2.0, Level::Low, 2.0);
        assert_close(&[driver.settles_at()], &[3.0]);
    }

    #[test]
    fn a_driver_s_jacobian_matches_finite_differences() {
        assert_jacobians_match::<LogicDriver>("{}", &[0.3, 5.0, 0.1, 1e-3]);
    }

    #[test]
    fn a_detector_reads_its_input_against_a_fraction_of_the_supply() {
        let mut bench =
            Bench::new::<ThresholdDetector>(r#"{"threshold": 0.4, "capacitance": 1e-12}"#);
        let x = [2.5, 5.0, 0.0];
        let input = Input::new(&x, None, 0.0);
        let detector = bench.device().as_detector().unwrap();
        assert!((detector.margin(&input) - 0.5).abs() < 1e-12);
        assert_close(&[detector.swing(&input)], &[5.0]);
        assert_close(&bench.evaluate(&x).charge[..1], &[2.5e-12]);
    }
}
