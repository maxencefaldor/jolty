//! The logic partition of a circuit, and the event scheduler that simulates it.
//!
//! Logic nets hold a [`LogicLevel`] rather than a voltage. The scheduler keeps, for every source of
//! a logic net, the level it drives and the change it has pending, and processes changes in
//! time order: each one may change its net's level, which every logic input on the net reads
//! and every connect element's driver follows. A logic device's output follows its inputs
//! with inertial delay, as a gate's does: a pending change that the inputs undo before it
//! happens never happens.

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};

use std::sync::Arc;

use jolty_models::{Change, Input, Logic, LogicLevel, Node, Pins, Stimulus};

use crate::elaborate::Location;
use crate::elaborated::Template;
use crate::transient::StepPoint;

/// One end of a logic net: a pin of a logic device, or a connect element in the analog
/// circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    /// Pin `index` among the inputs or the outputs of the logic device `device`.
    Pin {
        /// The device's index among [`Simulator::logic_devices`](crate::Simulator::logic_devices).
        device: usize,
        /// The pin's index among the device's inputs or outputs.
        index: usize,
    },
    /// The connect element `device`: a threshold detector that drives the net, or a logic
    /// driver that follows it.
    Terminal {
        /// The element's index among [`Simulator::devices`](crate::Simulator::devices).
        device: usize,
    },
}

/// A logic net: its name, what drives it and what reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicNet {
    /// The net's path, such as `adder/carry`, or, for the net between a logic pin and its
    /// connect element, the instance's path and the pin, such as `adder/U1:a`.
    pub name: String,
    /// The outputs and threshold detectors that drive the net.
    pub sources: Vec<Endpoint>,
    /// The inputs and logic drivers that read the net.
    pub loads: Vec<Endpoint>,
    /// The capacitance the net's outputs drive, in farads: its inputs', or, for a net a
    /// logic driver follows, those of the threshold detectors on the electrical net it
    /// drives. An operating point measures the latter again, as the whole capacitance of
    /// that net.
    pub load: f64,
}

/// A logic device placed in the flattened circuit.
#[derive(Debug, Clone)]
pub struct PlacedLogic {
    pub(crate) location: Location,
    pub(crate) model: &'static str,
    pub(crate) pins: Pins,
    pub(crate) inputs: Vec<usize>,
    pub(crate) outputs: Vec<usize>,
    /// The nodes of the electrical terminals, `supply` and `ground`, if it has them.
    pub(crate) nodes: Vec<Node>,
    pub(crate) device: Arc<dyn Logic>,
}

impl PlacedLogic {
    /// Where the instance is: its path and the script line that placed it.
    #[must_use]
    pub fn location(&self) -> &Location {
        &self.location
    }

    /// The instance's path through the hierarchy, such as `adder/U1`.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.location.path
    }

    /// The name of the device's model, such as `logic_cell`.
    #[must_use]
    pub fn model(&self) -> &'static str {
        self.model
    }

    /// The device's pins.
    #[must_use]
    pub fn pins(&self) -> &Pins {
        &self.pins
    }

    /// The logic net each input reads, by its index among
    /// [`Simulator::logic_nets`](crate::Simulator::logic_nets).
    #[must_use]
    pub fn inputs(&self) -> &[usize] {
        &self.inputs
    }

    /// The logic net each output drives.
    #[must_use]
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// The node each electrical terminal, `supply` then `ground`, joins, if the device has
    /// them.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
}

/// What a logic device's timing reads that its domain bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExcursionKind {
    /// An input's transition.
    Transition,
    /// The load an output drives.
    Load,
    /// The voltage from its ground to its supply.
    Supply,
}

impl ExcursionKind {
    /// The bound in words, such as `input transition`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Transition => "input transition",
            Self::Load => "load",
            Self::Supply => "supply",
        }
    }
}

/// A logic device used outside its domain: where its timing was never measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Excursion {
    /// The logic device's index among [`Simulator::logic_devices`](crate::Simulator::logic_devices).
    pub instance: usize,
    /// What left the domain.
    pub bound: ExcursionKind,
    /// The domain's minimum.
    pub minimum: f64,
    /// The domain's maximum.
    pub maximum: f64,
    /// The value furthest outside it.
    pub value: f64,
    /// The time it first left, in seconds; zero at an operating point.
    pub time: f64,
    /// How many times it was read outside.
    pub count: usize,
}

/// An inclusive valid range `[minimum, maximum]` for a monitored quantity.
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
    /// How far `value` lies outside `bounds`, zero within them.
    fn outside(value: f64, bounds: Bounds) -> f64 {
        (bounds.minimum - value)
            .max(value - bounds.maximum)
            .max(0.0)
    }
}

/// The logic partition of an elaborated circuit.
#[derive(Debug, Clone, Default)]
pub(crate) struct LogicSystem {
    pub(crate) nets: Vec<LogicNet>,
    pub(crate) devices: Vec<PlacedLogic>,
    /// The net each source drives: each output of each device in turn, then each detector.
    pub(crate) drives: Vec<usize>,
    /// The first source of each device's outputs.
    pub(crate) first_drive: Vec<usize>,
    /// Each threshold detector's device index, and the net it drives.
    pub(crate) detectors: Vec<(usize, usize)>,
    /// The source of the first detector, after every device's outputs.
    pub(crate) first_detector: usize,
    /// Each logic driver's device index, and the net it follows.
    pub(crate) drivers: Vec<(usize, usize)>,
    /// The sources of each net.
    pub(crate) sources: Vec<Vec<usize>>,
    /// The logic drivers that follow each net.
    pub(crate) followers: Vec<Vec<usize>>,
}

impl LogicSystem {
    /// Numbers every source and records what drives and follows each net, once every device
    /// and connect element is placed. `detected` holds each detector's input node and
    /// capacitance, and `devices` the analog devices, so each logic driver's net carries the
    /// load of the detectors on the node it drives.
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

    /// The source of `device`'s output `output`.
    fn output_drive(&self, device: usize, output: usize) -> usize {
        self.first_drive[device] + output
    }

    /// The level of `net` from its sources' levels.
    fn resolve(&self, net: usize, drives: &[DriveState]) -> LogicLevel {
        LogicLevel::resolve(self.sources[net].iter().map(|&drive| drives[drive].level))
    }

    /// Whether a logic driver follows `net` into the analog circuit.
    fn drives_analog(&self, net: usize) -> bool {
        !self.followers[net].is_empty()
    }
}

/// Whether a scheduled logic change is an uncancelled stimulus or a cancellable reaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Stimulus,
    Reaction,
}

/// A change waiting in the queue.
#[derive(Debug, Clone, Copy)]
struct Scheduled {
    time: f64,
    sequence: u64,
    drive: usize,
    level: LogicLevel,
    transition: f64,
    /// Whether the change is a device's own stimulus, which inputs never cancel.
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

/// What one source of a net drives now, and the change it has pending.
#[derive(Debug, Clone, Copy)]
struct DriveState {
    level: LogicLevel,
    /// The time, level and sequence number of the pending change.
    pending: Option<(f64, LogicLevel, u64)>,
    /// The time of the last change and its transition, since the logic last settled.
    changed: Option<(f64, f64)>,
}

impl DriveState {
    /// Where the source's edge stands at `time`, as a fraction of the supply from ground:
    /// on a ramp that crossed half the supply at its last change, and at its rail once
    /// the ramp ends or when it has not changed.
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

/// A change a logic driver must follow: from `time`, to `level`, over `transition`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Command {
    pub(crate) time: f64,
    /// The driver's index among the system's drivers.
    pub(crate) driver: usize,
    pub(crate) level: LogicLevel,
    pub(crate) transition: f64,
}

/// A threshold detector's change of level, found between two time points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Crossing {
    pub(crate) time: f64,
    /// The detector's index among the system's detectors.
    pub(crate) detector: usize,
    pub(crate) level: LogicLevel,
    pub(crate) transition: f64,
}

/// How many times each source may change while the logic settles at DC before the sources
/// still changing are taken to be unknown, as an oscillating loop's are.
///
/// Origin: jolty's own DC cycle-detection threshold, chosen as two complete high/low toggle
/// cycles (4 transitions, matching `SETTLE_ROUNDS` in `operating_point.rs`). A combinational
/// acyclic gate settles in at most 2 to 3 transitions from `Unknown`, whereas an oscillating
/// loop toggles indefinitely and is forced to `LogicLevel::Unknown` once its count exceeds 4.
const SETTLE_CHANGES: usize = 4;

/// The state of [`Events`] before a tentative advance: see [`Events::checkpoint`].
#[derive(Debug)]
pub(crate) struct Checkpoint {
    /// The events without their traces.
    state: Box<Events>,
    /// The length of each net's trace.
    lengths: Vec<usize>,
}

/// The state of the logic nets during an analysis.
#[derive(Debug, Clone)]
pub(crate) struct Events {
    levels: Vec<LogicLevel>,
    /// The transition of each net's last change.
    transitions: Vec<f64>,
    drives: Vec<DriveState>,
    queue: BinaryHeap<Reverse<Scheduled>>,
    sequence: u64,
    /// Each net's changes since the analysis began: the time and the new level.
    traces: Vec<Vec<(f64, LogicLevel)>>,
    /// The changes logic drivers must follow, not yet applied.
    commands: Vec<Command>,
    /// The level each logic driver was last told to follow.
    driven: Vec<LogicLevel>,
    /// The capacitance each net's outputs drive, in farads.
    loads: Vec<f64>,
    /// Where the devices were used outside their domains, by device and bound.
    excursions: BTreeMap<(usize, ExcursionKind), Excursion>,
}

impl Events {
    /// Every net and source unknown, with nothing pending.
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

    /// Forgets everything an analysis left, as [`Events::new`] makes them for `logic`,
    /// keeping the buffers.
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

    /// Records that `device` read `value` of `bound` at `time`, if it lies outside
    /// `bounds`.
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

    /// Where the devices were used outside their domains, by device then bound.
    pub(crate) fn excursions(&self) -> Vec<Excursion> {
        self.excursions.values().cloned().collect()
    }

    /// Takes `load` as the capacitance the outputs on `net` drive.
    pub(crate) fn load(&mut self, net: usize, load: f64) {
        self.loads[net] = load;
    }

    /// Each logic net's level.
    pub(crate) fn levels(&self) -> &[LogicLevel] {
        &self.levels
    }

    /// The level each logic driver was last told to follow.
    pub(crate) fn driven(&self) -> &[LogicLevel] {
        &self.driven
    }

    /// Each detector's level.
    pub(crate) fn detector_levels(&self, logic: &LogicSystem) -> Vec<LogicLevel> {
        self.drives[logic.first_detector..]
            .iter()
            .map(|drive| drive.level)
            .collect()
    }

    /// Settles the logic at DC, with the detectors at `detectors`: every device's outputs
    /// take the levels its inputs give at once, until nothing changes. A source that keeps
    /// changing, as in a loop that oscillates, becomes unknown. Clears every pending change.
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

    /// Recomputes `net`'s level, and queues the devices that read it if it changed.
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

    /// The logic drivers whose level differs from the one they were last told, each with
    /// its net's level now, marked as told.
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

    /// Begins the record of every net's changes at time zero, and queues each device's own
    /// changes in `(0, stop]`.
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

    /// The time of the earliest change pending on a net a logic driver follows.
    pub(crate) fn next_analog_change(&self, logic: &LogicSystem) -> Option<f64> {
        logic
            .drives
            .iter()
            .zip(&self.drives)
            .filter(|&(&net, _)| logic.drives_analog(net))
            .filter_map(|(_, state)| state.pending.map(|(time, _, _)| time))
            .min_by(f64::total_cmp)
    }

    /// Whether a change is due by `time`.
    pub(crate) fn due(&self, time: f64) -> bool {
        self.queue
            .peek()
            .is_some_and(|Reverse(next)| next.time <= time)
    }

    /// Queues the detectors' `crossings`.
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

    /// Processes every change due by `until`, in time order, and returns the commands for
    /// logic drivers they produced.
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

    /// Reschedules `device`'s outputs after its input `input` changed from `before` at
    /// `time`, with inertial delay.
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
            // The delay is measured from a rail: the edge leaves it half a transition
            // before it crosses half the supply. An output still on its way from its last
            // change has less to swing, and crosses sooner.
            let leaves = time + timing.delay - timing.transition / 2.0;
            let swing = match (level, state.level) {
                (LogicLevel::Low | LogicLevel::High, LogicLevel::Low | LogicLevel::High) => {
                    (state.fraction(leaves) - 0.5).abs().min(0.5) * timing.transition
                }
                _ => timing.transition / 2.0,
            };
            let mut at = leaves + swing;
            if logic.drives_analog(net) {
                // A logic driver ramps over the transition, so it starts early by the
                // swing it has left to cross half the supply when the output changes.
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

    /// The state before a tentative advance, which [`Events::restore`] returns to: all of
    /// it but the traces, of which it keeps the lengths, as an advance only appends to them.
    /// It costs the nets' state and the pending events, not the changes so far.
    pub(crate) fn checkpoint(&mut self) -> Checkpoint {
        let traces = std::mem::take(&mut self.traces);
        let state = Box::new(self.clone());
        let lengths = traces.iter().map(Vec::len).collect();
        self.traces = traces;
        Checkpoint { state, lengths }
    }

    /// Returns to the state at `checkpoint`, dropping the changes appended since.
    pub(crate) fn restore(&mut self, checkpoint: Checkpoint) {
        let mut traces = std::mem::take(&mut self.traces);
        for (trace, &length) in traces.iter_mut().zip(&checkpoint.lengths) {
            trace.truncate(length);
        }
        *self = *checkpoint.state;
        self.traces = traces;
    }

    /// Each net's changes since the analysis began.
    pub(crate) fn traces(&self) -> &[Vec<(f64, LogicLevel)>] {
        &self.traces
    }
}

/// The level a detector reads from its `margin` above the threshold, keeping `held` at the
/// threshold itself.
pub(crate) fn detected(margin: f64, held: LogicLevel) -> LogicLevel {
    if margin > 0.0 {
        LogicLevel::High
    } else if margin < 0.0 {
        LogicLevel::Low
    } else {
        held
    }
}

/// The crossing of the detector `detector`, whose connect element `sense` reads how far
/// its input lies above its threshold at a state and its supply's swing there, between
/// `start` and `end`, if its level changes from `held`.
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
