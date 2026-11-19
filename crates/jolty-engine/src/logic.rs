//! Event-driven logic partition and discrete-event scheduler.
//!
//! Logic nets carry a [`LogicLevel`] rather than a continuous voltage. The scheduler tracks
//! each net driver's active level and pending transition, processing events in time order
//! so logic inputs react with inertial delay (canceling superseded pending changes) and
//! analog connect drivers track net transitions.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};

use std::sync::Arc;

use jolty_models::{Change, Input, Logic, LogicLevel, Node, Pins, Stimulus};

use crate::elaborate::Location;
use crate::elaborated::Template;
use crate::transient::StepPoint;

/// Connection endpoint on a logic net: either a logic device pin or an analog connect
/// element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    /// Input or output pin `index` of logic device `device`.
    Pin {
        /// Index in [`Simulator::logic_devices`](crate::Simulator::logic_devices).
        device: usize,
        /// Zero-based index within the device's inputs or outputs.
        index: usize,
    },
    /// Analog connect element (threshold detector driving the net or logic driver following
    /// the net).
    Terminal {
        /// Index in [`Simulator::devices`](crate::Simulator::devices).
        device: usize,
    },
}

/// Logic net connecting logic device pins and analog connect elements.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicNet {
    /// Net path (such as `adder/carry`), or `instance:pin` for nets bridging a logic pin to
    /// its connect element.
    pub name: String,
    /// Logic outputs and threshold detectors driving this net.
    pub sources: Vec<Endpoint>,
    /// Logic inputs and analog logic drivers reading this net.
    pub loads: Vec<Endpoint>,
    /// Capacitive load driven by outputs on this net in farads (re-measured at operating
    /// point for nets feeding analog logic drivers).
    pub load: f64,
}

/// Flattened logic device instance in an elaborated circuit.
#[derive(Debug, Clone)]
pub struct PlacedLogic {
    pub(crate) location: Location,
    pub(crate) model: &'static str,
    pub(crate) pins: Pins,
    pub(crate) inputs: Vec<usize>,
    pub(crate) outputs: Vec<usize>,
    /// Electrical nodes for optional `supply` and `ground` terminals.
    pub(crate) nodes: Vec<Node>,
    pub(crate) device: Arc<dyn Logic>,
}

impl PlacedLogic {
    /// Returns the instance path and optional source location of this logic device.
    #[must_use]
    pub fn location(&self) -> &Location {
        &self.location
    }

    /// Returns the hierarchical instance path, such as `adder/U1`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.location.path
    }

    /// Returns the logic model name, such as `logic_cell`.
    #[must_use]
    pub fn model(&self) -> &'static str {
        self.model
    }

    /// Returns the declared [`Pins`] of this logic device.
    #[must_use]
    pub fn pins(&self) -> &Pins {
        &self.pins
    }

    /// Returns logic net indices in [`Simulator::logic_nets`](crate::Simulator::logic_nets)
    /// read by each input pin.
    #[must_use]
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    /// Returns logic net indices driven by each output pin.
    #[must_use]
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// Returns the electrical [`Node`] bindings for optional `supply` and `ground` terminals.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}

/// Monitored timing-model domain parameter for a logic device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExcursionKind {
    /// Input edge transition time.
    Transition,
    /// Output capacitive load.
    Load,
    /// Supply-to-ground voltage.
    Supply,
}

impl ExcursionKind {
    /// Returns the human-readable name of this domain bound (such as `"input transition"`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transition => "input transition",
            Self::Load => "load",
            Self::Supply => "supply",
        }
    }
}

/// Record of a logic device operating outside its characterized timing domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Excursion {
    /// Index in [`Simulator::logic_devices`](crate::Simulator::logic_devices).
    pub instance: usize,
    /// Domain parameter that went out of bounds.
    pub bound: ExcursionKind,
    /// Minimum characterized bound value.
    pub minimum: f64,
    /// Maximum characterized bound value.
    pub maximum: f64,
    /// Most extreme out-of-range value observed.
    pub value: f64,
    /// Simulation time of the first excursion in seconds (zero at DC operating point).
    pub time: f64,
    /// Total number of out-of-range evaluations recorded.
    pub count: usize,
}

/// Inclusive valid range `[minimum, maximum]` for a monitored quantity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) minimum: f64,
    pub(crate) maximum: f64,
}

impl Bounds {
    fn from_pair((minimum, maximum): (f64, f64)) -> Self {
        Self { minimum, maximum }
    }
}

impl Excursion {
    /// Computes how far `value` lies outside `bounds`, returning `0.0` when inside.
    fn outside(value: f64, bounds: Bounds) -> f64 {
        (bounds.minimum - value)
            .max(value - bounds.maximum)
            .max(0.0)
    }
}

/// Compiled logic partition of an elaborated circuit.
#[derive(Debug, Clone, Default)]
pub(crate) struct LogicSystem {
    pub(crate) nets: Vec<LogicNet>,
    pub(crate) devices: Vec<PlacedLogic>,
    /// Target logic net index for every drive (device outputs followed by detectors).
    pub(crate) drives: Vec<usize>,
    /// First drive index for each logic device's outputs.
    pub(crate) first_drive: Vec<usize>,
    /// Pairs `(analog_device_index, logic_net_index)` for each threshold detector.
    pub(crate) detectors: Vec<(usize, usize)>,
    /// Drive index of the first threshold detector, immediately after all device outputs.
    pub(crate) first_detector: usize,
    /// Pairs `(analog_device_index, logic_net_index)` for each analog logic driver.
    pub(crate) drivers: Vec<(usize, usize)>,
    /// Drive indices sourcing each logic net.
    pub(crate) sources: Vec<Vec<usize>>,
    /// Analog logic driver indices following each logic net.
    pub(crate) followers: Vec<Vec<usize>>,
}

impl LogicSystem {
    /// Builds drive, follower, and detector-capacitance indices once all logic devices and
    /// connect elements are placed.
    ///
    /// Uses `detected` (each detector's input node and capacitance) and analog `devices` to
    /// initialize baseline capacitive loads on logic-driver nets.
    pub(crate) fn index(&mut self, detected: &[(Node, f64)], devices: &[Arc<Template>]) {
        self.drives.clear();
        self.first_drive.clear();
        self.sources = vec![Vec::new(); self.nets.len()];
        self.followers = vec![Vec::new(); self.nets.len()];
        for device in &self.devices {
            self.first_drive.push(self.drives.len());
            for &net in &device.outputs {
                self.sources[net].push(self.drives.len());
                self.drives.push(net);
            }
        }
        self.first_detector = self.drives.len();
        for &(_, net) in &self.detectors {
            self.sources[net].push(self.drives.len());
            self.drives.push(net);
        }
        for (driver, &(element, net)) in self.drivers.iter().enumerate() {
            self.followers[net].push(driver);
            let output = devices[element].nodes[0];
            self.nets[net].load = detected
                .iter()
                .filter(|(node, _)| *node == output)
                .map(|(_, capacitance)| capacitance)
                .sum();
        }
    }

    /// Returns the drive index for `output` of logic `device`.
    fn output_drive(&self, device: usize, output: usize) -> usize {
        self.first_drive[device] + output
    }

    /// Resolves the combined [`LogicLevel`] of `net` across all its `drives`.
    fn resolve(&self, net: usize, drives: &[DriveState]) -> LogicLevel {
        LogicLevel::resolve(self.sources[net].iter().map(|&drive| drives[drive].level))
    }

    /// Returns whether any analog logic driver follows `net`.
    fn drives_analog(&self, net: usize) -> bool {
        !self.followers[net].is_empty()
    }
}

/// Classification of a queued logic event as non-cancellable stimulus or cancellable
/// gate reaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Stimulus,
    Reaction,
}

/// Queued logic transition event ordered by `(time, sequence)`.
#[derive(Debug, Clone, Copy)]
struct Scheduled {
    time: f64,
    sequence: u64,
    drive: usize,
    level: LogicLevel,
    transition: f64,
    /// Classification distinguishing non-cancellable stimuli from inertial gate reactions.
    kind: ChangeKind,
}

impl PartialEq for Scheduled {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Scheduled {}

impl PartialOrd for Scheduled {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scheduled {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then(self.sequence.cmp(&other.sequence))
    }
}

/// Active output level, pending inertial transition, and latest ramp metadata for one net
/// driver.
#[derive(Debug, Clone, Copy)]
struct DriveState {
    level: LogicLevel,
    /// Pending inertial change `(time, level, sequence)`.
    pending: Option<(f64, LogicLevel, u64)>,
    /// Timestamp and transition duration of the latest committed edge since DC settling.
    changed: Option<(f64, f64)>,
}

impl DriveState {
    /// Computes the driver's normalized rail fraction at `time`, interpolating linearly along
    /// an active ramp crossing `0.5` at its transition timestamp.
    fn fraction(&self, time: f64) -> f64 {
        match self.changed {
            Some((at, transition)) if transition > 0.0 => {
                let swung = (0.5 + (time - at) / transition).clamp(0.0, 1.0);
                if self.level == LogicLevel::High {
                    swung
                } else {
                    1.0 - swung
                }
            }
            _ => self.level.fraction(),
        }
    }
}

/// Ramp instruction delivered to an analog logic driver: start `time`, target `level`, and
/// `transition` duration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Command {
    pub(crate) time: f64,
    /// Index within `LogicSystem::drivers`.
    pub(crate) driver: usize,
    pub(crate) level: LogicLevel,
    pub(crate) transition: f64,
}

/// Threshold-detector level transition localized between two transient time points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Crossing {
    pub(crate) time: f64,
    /// Index within `LogicSystem::detectors`.
    pub(crate) detector: usize,
    pub(crate) level: LogicLevel,
    pub(crate) transition: f64,
}

/// Maximum transitions per logic source during DC settling before forcing
/// `LogicLevel::Unknown` (origin: two full high/low toggle cycles for loop detection,
/// matching `SETTLE_ROUNDS` in `operating_point.rs`).
const SETTLE_CHANGES: usize = 4;

/// Snapshot of [`Events`] before a tentative transient step; see [`Events::checkpoint`].
#[derive(Debug)]
pub(crate) struct Checkpoint {
    /// Cloned scheduler state with empty traces.
    state: Box<Events>,
    /// Saved length of each net's trace buffer for truncation on restore.
    lengths: Vec<usize>,
}

/// Runtime event-scheduler state for all logic nets during an analysis.
#[derive(Debug, Clone)]
pub(crate) struct Events {
    levels: Vec<LogicLevel>,
    /// Transition duration of each net's most recent level change.
    transitions: Vec<f64>,
    drives: Vec<DriveState>,
    queue: BinaryHeap<Reverse<Scheduled>>,
    sequence: u64,
    /// Recorded `(time, level)` transitions per logic net since analysis start.
    traces: Vec<Vec<(f64, LogicLevel)>>,
    /// Queued analog logic-driver commands awaiting dispatch.
    commands: Vec<Command>,
    /// Latest logic level dispatched to each analog logic driver.
    driven: Vec<LogicLevel>,
    /// Active capacitive load in farads driven by each logic net.
    loads: Vec<f64>,
    /// Recorded timing-domain excursions keyed by `(device_index, bound)`.
    excursions: BTreeMap<(usize, ExcursionKind), Excursion>,
}

impl Events {
    /// Initializes [`Events`] for `logic` with all nets and drives set to `Unknown` and an
    /// empty event queue.
    pub(crate) fn new(logic: &LogicSystem) -> Self {
        Self {
            levels: vec![LogicLevel::Unknown; logic.nets.len()],
            transitions: vec![0.0; logic.nets.len()],
            drives: vec![
                DriveState {
                    level: LogicLevel::Unknown,
                    pending: None,
                    changed: None,
                };
                logic.drives.len()
            ],
            queue: BinaryHeap::new(),
            sequence: 0,
            traces: vec![Vec::new(); logic.nets.len()],
            commands: Vec::new(),
            driven: vec![LogicLevel::Unknown; logic.drivers.len()],
            loads: logic.nets.iter().map(|net| net.load).collect(),
            excursions: BTreeMap::new(),
        }
    }

    /// Resets scheduler state in place to match [`Events::new`] while reusing allocations.
    pub(crate) fn reset(&mut self, logic: &LogicSystem) {
        self.levels.fill(LogicLevel::Unknown);
        self.transitions.fill(0.0);
        self.drives.fill(DriveState {
            level: LogicLevel::Unknown,
            pending: None,
            changed: None,
        });
        self.queue.clear();
        self.sequence = 0;
        for trace in &mut self.traces {
            trace.clear();
        }
        self.commands.clear();
        self.driven.fill(LogicLevel::Unknown);
        self.loads.clear();
        self.loads.extend(logic.nets.iter().map(|net| net.load));
        self.excursions.clear();
    }

    /// Records a timing-domain excursion for `device` and `bound` at `time` when `value`
    /// falls outside `bounds`.
    pub(crate) fn watch(
        &mut self,
        device: usize,
        bound: ExcursionKind,
        value: f64,
        bounds: Bounds,
        time: f64,
    ) {
        let outside = Excursion::outside(value, bounds);
        if outside <= 0.0 {
            return;
        }
        let excursion = self.excursions.entry((device, bound)).or_insert(Excursion {
            instance: device,
            bound,
            minimum: bounds.minimum,
            maximum: bounds.maximum,
            value,
            time,
            count: 0,
        });
        excursion.count += 1;
        if outside > Excursion::outside(excursion.value, bounds) {
            excursion.value = value;
        }
    }

    /// Returns all recorded timing-domain excursions ordered by `(device, bound)`.
    pub(crate) fn excursions(&self) -> Vec<Excursion> {
        self.excursions.values().cloned().collect()
    }

    /// Updates the capacitive `load` in farads driven by outputs on `net`.
    pub(crate) fn load(&mut self, net: usize, load: f64) {
        self.loads[net] = load;
    }

    /// Returns the current [`LogicLevel`] of every logic net.
    pub(crate) fn levels(&self) -> &[LogicLevel] {
        &self.levels
    }

    /// Returns the last [`LogicLevel`] dispatched to each analog logic driver.
    pub(crate) fn driven(&self) -> &[LogicLevel] {
        &self.driven
    }

    /// Returns the current [`LogicLevel`] of every threshold detector.
    pub(crate) fn detector_levels(&self, logic: &LogicSystem) -> Vec<LogicLevel> {
        self.drives[logic.first_detector..]
            .iter()
            .map(|drive| drive.level)
            .collect()
    }

    /// Settles combinational logic at DC from `detectors` until convergence, forcing
    /// oscillating loop drivers to `LogicLevel::Unknown` and clearing pending events.
    pub(crate) fn settle(&mut self, logic: &LogicSystem, detectors: &[LogicLevel]) {
        self.queue.clear();
        for drive in &mut self.drives {
            drive.pending = None;
            drive.changed = None;
        }
        let mut pending: Vec<usize> = (0..logic.devices.len()).collect();
        for (detector, &level) in detectors.iter().enumerate() {
            let drive = logic.first_detector + detector;
            self.drives[drive].level = level;
            self.update_net(logic, logic.drives[drive], &mut pending);
        }
        let mut changes = vec![0; logic.drives.len()];
        let mut outputs = Vec::new();
        let mut inputs = Vec::new();
        while let Some(device) = pending.pop() {
            let placed = &logic.devices[device];
            inputs.clear();
            inputs.extend(placed.inputs.iter().map(|&net| self.levels[net]));
            outputs.clear();
            outputs.resize(placed.outputs.len(), LogicLevel::Unknown);
            placed.device.levels(&inputs, &mut outputs);
            for (output, &level) in outputs.iter().enumerate() {
                let drive = logic.output_drive(device, output);
                if self.drives[drive].level == level {
                    continue;
                }
                changes[drive] += 1;
                self.drives[drive].level = if changes[drive] > SETTLE_CHANGES {
                    LogicLevel::Unknown
                } else {
                    level
                };
                self.update_net(logic, placed.outputs[output], &mut pending);
            }
        }
    }

    /// Recomputes `net`'s resolved level and enqueues fanout logic devices if it changed.
    fn update_net(&mut self, logic: &LogicSystem, net: usize, pending: &mut Vec<usize>) {
        let level = logic.resolve(net, &self.drives);
        if level == self.levels[net] {
            return;
        }
        self.levels[net] = level;
        for load in &logic.nets[net].loads {
            if let Endpoint::Pin { device, .. } = *load
                && !pending.contains(&device)
            {
                pending.push(device);
            }
        }
    }

    /// Collects analog logic drivers whose net level differs from their last commanded
    /// level, updating `driven` to match.
    pub(crate) fn retarget(&mut self, logic: &LogicSystem) -> Vec<(usize, LogicLevel)> {
        let mut changed = Vec::new();
        for (driver, &(_, net)) in logic.drivers.iter().enumerate() {
            let level = self.levels[net];
            if self.driven[driver] != level {
                self.driven[driver] = level;
                changed.push((driver, level));
            }
        }
        changed
    }

    /// Initializes net traces at `t = 0` and enqueues all device stimuli within `(0, stop]`.
    pub(crate) fn start(&mut self, logic: &LogicSystem, stop: f64) {
        for (trace, &level) in self.traces.iter_mut().zip(&self.levels) {
            trace.clear();
            trace.push((0.0, level));
        }
        let mut stimuli = Vec::new();
        for (device, placed) in logic.devices.iter().enumerate() {
            stimuli.clear();
            placed.device.stimuli(stop, &mut stimuli);
            for &Stimulus {
                time,
                output,
                level,
                transition,
            } in &stimuli
            {
                let drive = logic.output_drive(device, output);
                self.push(time, drive, level, transition, ChangeKind::Stimulus);
            }
        }
    }

    fn push(
        &mut self,
        time: f64,
        drive: usize,
        level: LogicLevel,
        transition: f64,
        kind: ChangeKind,
    ) {
        self.sequence += 1;
        self.queue.push(Reverse(Scheduled {
            time,
            sequence: self.sequence,
            drive,
            level,
            transition,
            kind,
        }));
        if kind == ChangeKind::Reaction {
            self.drives[drive].pending = Some((time, level, self.sequence));
        }
    }

    /// Returns the earliest pending transition time on any net driving an analog logic
    /// driver.
    pub(crate) fn next_analog_change(&self, logic: &LogicSystem) -> Option<f64> {
        logic
            .drives
            .iter()
            .zip(&self.drives)
            .filter(|&(&net, _)| logic.drives_analog(net))
            .filter_map(|(_, state)| state.pending.map(|(time, _, _)| time))
            .min_by(f64::total_cmp)
    }

    /// Returns whether any queued event falls at or before `time`.
    pub(crate) fn due(&self, time: f64) -> bool {
        self.queue
            .peek()
            .is_some_and(|Reverse(next)| next.time <= time)
    }

    /// Enqueues threshold-detector `crossings` as stimulus events.
    pub(crate) fn inject(&mut self, logic: &LogicSystem, crossings: &[Crossing]) {
        for crossing in crossings {
            let drive = logic.first_detector + crossing.detector;
            self.push(
                crossing.time,
                drive,
                crossing.level,
                crossing.transition,
                ChangeKind::Stimulus,
            );
        }
    }

    /// Processes all queued events up to `until` in chronological order and returns any
    /// resulting analog driver [`Command`] updates.
    pub(crate) fn advance(&mut self, logic: &LogicSystem, until: f64) -> Vec<Command> {
        while let Some(&Reverse(next)) = self.queue.peek() {
            if next.time > until {
                break;
            }
            self.queue.pop();
            let state = &mut self.drives[next.drive];
            if next.kind == ChangeKind::Reaction {
                if state.pending.map(|(_, _, sequence)| sequence) != Some(next.sequence) {
                    continue;
                }
                state.pending = None;
            }
            state.level = next.level;
            state.changed = Some((next.time, next.transition));
            let net = logic.drives[next.drive];
            let level = logic.resolve(net, &self.drives);
            if level == self.levels[net] {
                continue;
            }
            let before = self.levels[net];
            self.levels[net] = level;
            self.transitions[net] = next.transition;
            self.traces[net].push((next.time, level));
            for load in &logic.nets[net].loads {
                if let Endpoint::Pin { device, index } = *load {
                    self.react(logic, device, index, before, next.time);
                }
            }
            for &driver in &logic.followers[net] {
                self.driven[driver] = level;
                self.commands.push(Command {
                    time: next.time,
                    driver,
                    level,
                    transition: next.transition,
                });
            }
        }
        std::mem::take(&mut self.commands)
    }

    /// Reschedules `device` outputs with inertial delay after `input` changed from `before`
    /// at `time`.
    fn react(
        &mut self,
        logic: &LogicSystem,
        device: usize,
        input: usize,
        before: LogicLevel,
        time: f64,
    ) {
        let placed = &logic.devices[device];
        let inputs: Vec<LogicLevel> = placed.inputs.iter().map(|&net| self.levels[net]).collect();
        let mut previous = inputs.clone();
        previous[input] = before;
        let mut outputs = vec![LogicLevel::Unknown; placed.outputs.len()];
        placed.device.levels(&inputs, &mut outputs);
        let change = Change {
            before: &previous,
            input,
            slew: self.transitions[placed.inputs[input]],
        };
        let domain = placed.device.domain().cloned();
        if let Some(domain) = &domain {
            self.watch(
                device,
                ExcursionKind::Transition,
                change.slew,
                Bounds::from_pair(domain.slews),
                time,
            );
        }
        for (output, &level) in outputs.iter().enumerate() {
            let drive = logic.output_drive(device, output);
            let state = self.drives[drive];
            if level == state.level {
                self.drives[drive].pending = None;
                continue;
            }
            if state
                .pending
                .is_some_and(|(_, pending, _)| pending == level)
            {
                continue;
            }
            let net = placed.outputs[output];
            if let Some(domain) = &domain {
                self.watch(
                    device,
                    ExcursionKind::Load,
                    self.loads[net],
                    Bounds::from_pair(domain.loads),
                    time,
                );
            }
            let timing = placed
                .device
                .timing(&change, output, level, self.loads[net]);
            // Delay is measured from a full rail, so an edge still mid-swing reaches 50%
            // sooner.
            let leaves = time + timing.delay - timing.transition / 2.0;
            let swing = match (level, state.level) {
                (LogicLevel::Low | LogicLevel::High, LogicLevel::Low | LogicLevel::High) => {
                    (state.fraction(leaves) - 0.5).abs().min(0.5) * timing.transition
                }
                _ => timing.transition / 2.0,
            };
            let mut at = leaves + swing;
            if logic.drives_analog(net) {
                // Start analog ramps early by the remaining swing so the midpoint crosses on
                // schedule.
                at = leaves.max(time);
            }
            self.push(
                at.max(time),
                drive,
                level,
                timing.transition,
                ChangeKind::Reaction,
            );
        }
    }

    /// Captures a lightweight [`Checkpoint`] for rollback via [`Events::restore`], cloning
    /// scheduler state while recording only trace lengths since advances only append.
    pub(crate) fn checkpoint(&mut self) -> Checkpoint {
        let traces = std::mem::take(&mut self.traces);
        let state = Box::new(self.clone());
        let lengths = traces.iter().map(Vec::len).collect();
        self.traces = traces;
        Checkpoint { state, lengths }
    }

    /// Restores scheduler state to `checkpoint` and truncates trace entries appended since.
    pub(crate) fn restore(&mut self, checkpoint: Checkpoint) {
        let mut traces = std::mem::take(&mut self.traces);
        for (trace, &length) in traces.iter_mut().zip(&checkpoint.lengths) {
            trace.truncate(length);
        }
        *self = *checkpoint.state;
        self.traces = traces;
    }

    /// Returns recorded `(time, level)` transition traces for all logic nets.
    pub(crate) fn traces(&self) -> &[Vec<(f64, LogicLevel)>] {
        &self.traces
    }
}

/// Maps a detector's threshold `margin` to `High` (positive), `Low` (negative), or `held`
/// (exactly zero).
pub(crate) fn detected(margin: f64, held: LogicLevel) -> LogicLevel {
    if margin > 0.0 {
        LogicLevel::High
    } else if margin < 0.0 {
        LogicLevel::Low
    } else {
        held
    }
}

/// Interpolates a threshold-detector [`Crossing`] between `start` and `end` via `sense`
/// when the detected level changes from `held`.
pub(crate) fn crossing(
    sense: impl Fn(&Input<'_>) -> Option<(f64, f64)>,
    detector: usize,
    held: LogicLevel,
    start: StepPoint<'_>,
    end: StepPoint<'_>,
) -> Option<Crossing> {
    let (first, _) = sense(&Input::new(start.values, Some(start.time), 0.0))?;
    let (last, swing) = sense(&Input::new(end.values, Some(end.time), 0.0))?;
    let level = detected(last, held);
    if level == held {
        return None;
    }
    let rate = (last - first) / (end.time - start.time);
    let fraction = first / (first - last);
    let time = if fraction.is_finite() {
        start.time + (end.time - start.time) * fraction.clamp(0.0, 1.0)
    } else {
        end.time
    };
    let swing = swing.abs();
    Some(Crossing {
        time,
        detector,
        level,
        transition: if rate == 0.0 { 0.0 } else { swing / rate.abs() },
    })
}
