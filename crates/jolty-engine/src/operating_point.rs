//! The DC operating point, with gmin stepping, across the nodes then across the junctions,
//! source stepping, and pseudo-transient continuation when Newton alone fails, and every
//! other DC solution deflation finds.

use std::f64::consts::TAU;

use jolty_models::{Input, Level, Quantity};
use jolty_numerics::{Complex64, SparseLu};
use serde::Deserialize;

use crate::logic::{Bound, Events, detected};
use crate::newton::{Failure, Mode, NAMED_UNKNOWNS};
use crate::{Observation, Options, SimulationError, Simulator};

/// The largest conductance gmin stepping starts from, in siemens.
const FIRST_STEPPED: f64 = 1e-2;
/// The largest factor gmin stepping divides the conductance by from one stage to the next.
const STEPPED_DIVISOR: f64 = 10.0;
/// The factor below which gmin stepping, having shrunk its divisor after each failure or
/// each stage that barely converged, gives up.
const SMALLEST_STEPPED_DIVISOR: f64 = 1.00005;
/// The first increment of the source scale in source stepping.
const FIRST_SOURCE_INCREMENT: f64 = 0.1;
/// The smallest increment of the source scale before source stepping gives up.
const SMALLEST_SOURCE_INCREMENT: f64 = 1e-4;
/// The conductance, in siemens, that first holds each node to its last voltage in
/// pseudo-transient continuation: a farad's capacitance over a step of a second.
const FIRST_HELD: f64 = 1.0;
/// The largest conductance pseudo-transient continuation holds each node with before it
/// gives up: a farad over a nanosecond, past which the circuit barely moves.
const LARGEST_HELD: f64 = 1e9;
/// The most steps pseudo-transient continuation takes before it gives up.
const MOST_HELD_STEPS: usize = 1000;
/// The frequency, in hertz, at which a logic driver's load is read: low enough that every
/// capacitance it drives charges fully.
const LOAD_FREQUENCY: f64 = 1.0;
/// How many tolerances apart two solutions must lie in some unknown to be distinct.
const DISTINCT: f64 = 10.0;
/// How far, in volts, a search moves each node voltage of a solution found, up and down
/// in turn, to start from beside it.
const NUDGE: f64 = 0.1;
/// How many times the logic and the analog circuit settle each other at DC before the
/// detectors still changing are taken to read unknown levels, as in a loop that oscillates.
const SETTLE_ROUNDS: usize = 4;

/// A DC solution: every unknown's value, every logic net's level, and how it was found.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatingPoint {
    /// The value of every unknown, in the simulator's order.
    pub values: Vec<f64>,
    /// What every device did: the current into each terminal and the charge it holds, its
    /// derivatives and its variables.
    pub observation: Observation,
    /// The level of every logic net, in the simulator's order.
    pub logic: Vec<Level>,
    /// The strategy that found the solution.
    pub strategy: Strategy,
}

/// A search for every DC solution of the circuit, up to `most`: the operating point, then
/// each solution that the equations deflated by those found so far lead to.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatingPoints {
    /// The most solutions to find; at least 1.
    pub most: usize,
}

/// A DC solution of the analog circuit, with its logic drivers held.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Solved {
    pub(crate) values: Vec<f64>,
    pub(crate) strategy: Strategy,
}

/// The conductance gmin stepping shrinks.
#[derive(Debug, Clone, Copy)]
enum Stepped {
    /// A shunt from every node to the reference, removed once it reaches gmin.
    Shunt,
    /// The conductance across every junction, gmin itself.
    Junctions,
}

/// How an operating point was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Newton iteration from the initial guess.
    Direct,
    /// Newton iteration while a conductance from every node to the reference shrinks to zero.
    GminStepping,
    /// Newton iteration while the conductance across every junction shrinks to gmin.
    JunctionGminStepping,
    /// Newton iteration while every source ramps up from zero.
    SourceStepping,
    /// Newton iteration on steps of a pseudo-transient, each node held to its last voltage
    /// by a conductance that shrinks as the circuit settles.
    PseudoTransient,
}

impl Strategy {
    /// The strategy's name in snake case, such as `gmin_stepping`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::GminStepping => "gmin_stepping",
            Self::JunctionGminStepping => "junction_gmin_stepping",
            Self::SourceStepping => "source_stepping",
            Self::PseudoTransient => "pseudo_transient",
        }
    }
}

impl Simulator {
    /// Finds the DC operating point, with every source at its level at time zero.
    ///
    /// The logic settles with the levels its threshold detectors read, its drivers hold the
    /// levels it settles at, and the two settle each other until they agree.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::NoOperatingPoint`] or [`SimulationError::Singular`] when no
    /// strategy converges, and [`SimulationError::InvalidOptions`] for options out of range.
    pub fn operating_point(
        &mut self,
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        self.solvable()?;
        options.check()?;
        self.reset_logic();
        let solved = self.solve_dc(vec![0.0; self.system.unknowns.len()], options)?;
        self.settle(solved, options)
    }

    /// Finds the operating point, then the other DC solutions, up to `search.most` in all,
    /// in the order found: the equations deflated by the solutions found so far lead
    /// Newton iteration, gmin stepping or source stepping from zero to another, or else
    /// Newton iteration from the reflection of a solution found through another, or from
    /// beside one, its node voltages moved up and down in turn, which breaks the symmetry
    /// a balanced latch's zero start keeps, until none does. A circuit whose equations
    /// have one solution has one; one that latches has a solution per state and one
    /// balanced between them.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for a search for no solution, and the
    /// errors of [`Simulator::operating_point`].
    pub fn operating_points(
        &mut self,
        search: &OperatingPoints,
        options: &Options,
    ) -> Result<Vec<OperatingPoint>, SimulationError> {
        self.solvable()?;
        options.check()?;
        if search.most == 0 {
            return Err(SimulationError::InvalidAnalysis(
                "a search for operating points finds one or more".to_owned(),
            ));
        }
        let zero = vec![0.0; self.system.unknowns.len()];
        self.reset_logic();
        let mut found = vec![self.solve_dc(zero.clone(), options)?];
        while found.len() < search.most {
            let known: Vec<Vec<f64>> = found.iter().map(|solved| solved.values.clone()).collect();
            let Some(solved) = self.another(&known, options) else {
                break;
            };
            found.push(solved);
        }
        found
            .into_iter()
            .map(|solved| {
                self.reset_logic();
                self.settle(solved, options)
            })
            .collect()
    }

    /// A DC solution distinct from every one `known`, from zero or else from the
    /// reflection of a known solution through another, or from beside one, on the
    /// equations they deflate.
    fn another(&mut self, known: &[Vec<f64>], options: &Options) -> Option<Solved> {
        let zero = vec![0.0; self.system.unknowns.len()];
        if let Ok(solved) = self.solve_deflated(zero, options, known)
            && self.distinct(known, &solved.values, options)
        {
            return Some(solved);
        }
        let mut starts = Vec::new();
        for (index, mirror) in known.iter().enumerate().rev() {
            for (other, through) in known.iter().enumerate().rev() {
                if other != index {
                    starts.push(
                        mirror
                            .iter()
                            .zip(through)
                            .map(|(mirror, through)| 2.0 * through - mirror)
                            .collect::<Vec<f64>>(),
                    );
                }
            }
        }
        for root in known.iter().rev() {
            for sign in [1.0, -1.0] {
                let mut beside = root.clone();
                let voltages = self
                    .system
                    .unknowns
                    .iter()
                    .zip(&mut beside)
                    .filter(|(unknown, _)| unknown.quantity == Quantity::Voltage);
                for (turn, (_, value)) in voltages.enumerate() {
                    let alternate = if turn % 2 == 0 { 1.0 } else { -1.0 };
                    *value += sign * alternate * NUDGE;
                }
                starts.push(beside);
            }
        }
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
            anchor: None,
            deflated: known,
        };
        for mut values in starts {
            if self
                .newton(&mut values, direct, options, options.dc_iterations)
                .is_ok()
                && self.distinct(known, &values, options)
            {
                return Some(Solved {
                    values,
                    strategy: Strategy::Direct,
                });
            }
        }
        None
    }

    /// Whether `values` lie more than [`DISTINCT`] tolerances from each of `known` in
    /// some unknown.
    fn distinct(&self, known: &[Vec<f64>], values: &[f64], options: &Options) -> bool {
        !known
            .iter()
            .any(|root| self.indistinct(root, values, options))
    }

    /// Whether the solutions `first` and `second` lie within [`DISTINCT`] tolerances of
    /// each other in every unknown.
    fn indistinct(&self, first: &[f64], second: &[f64], options: &Options) -> bool {
        self.system
            .unknowns
            .iter()
            .zip(first.iter().zip(second))
            .all(|(unknown, (a, b))| {
                let tolerance = options.relative_tolerance * a.abs().max(b.abs())
                    + options.value_tolerance(unknown.quantity);
                (a - b).abs() <= DISTINCT * tolerance
            })
    }

    /// Forgets the logic's state: every net unknown, every driver at once at its unknown
    /// level.
    pub(crate) fn reset_logic(&mut self) {
        self.events = Events::new(&self.system.logic);
        for index in 0..self.system.logic.drivers.len() {
            let element = self.system.logic.drivers[index].0;
            self.drive(element, (0.0, Level::Unknown, 0.0));
        }
    }

    /// Takes, as the load of each logic driver's output, the whole capacitance it drives
    /// at the solution `values`: the detectors' and the analog circuit's alike, the
    /// transistors' gates included, read from the driver's small-signal current at
    /// [`LOAD_FREQUENCY`], as a characterization reads an input's capacitance.
    fn measure_loads(&mut self, values: &[f64], options: &Options) {
        if self.system.logic.drivers.is_empty() {
            return;
        }
        self.evaluate(values, None, (1.0, options.gmin), true);
        let omega = TAU * LOAD_FREQUENCY;
        let contributions = &self.workspace.contributions;
        let matrix: Vec<Complex64> = contributions
            .conductance
            .iter()
            .zip(&contributions.capacitance)
            .map(|(&conductance, &capacitance)| Complex64::new(conductance, omega * capacitance))
            .collect();
        let Ok(mut solver) = SparseLu::<Complex64>::new(self.system.pattern.clone()) else {
            return;
        };
        let Ok(mut factors) = solver.factor(&matrix) else {
            return;
        };
        for &(element, net) in &self.system.logic.drivers {
            let Some(current) = self.system.devices[element]
                .device
                .as_driver()
                .map(|driver| driver.current())
            else {
                continue;
            };
            // Raising the driver's voltage by a volt adds -1 to its equation's residual.
            let mut phasors = vec![Complex64::default(); values.len()];
            phasors[current.index()] = Complex64::new(1.0, 0.0);
            if factors.solve(&mut phasors).is_ok() {
                self.events
                    .load(net, phasors[current.index()].im.abs() / omega);
            }
        }
    }

    /// Records each characterized logic device whose supply at `values`, at `time`, lies
    /// further from the supply it was measured at than the relative tolerance.
    pub(crate) fn watch_supplies(&mut self, values: &[f64], time: f64, options: &Options) {
        let input = Input::new(values, time, 0.0);
        for (index, placed) in self.system.logic.devices.iter().enumerate() {
            let (Some(nominal), &[supply, ground]) = (
                placed.device.domain().and_then(|domain| domain.supply),
                placed.nodes(),
            ) else {
                continue;
            };
            let voltage = input.value(supply) - input.value(ground);
            let slack = options.relative_tolerance * nominal;
            let range = (nominal - slack, nominal + slack);
            self.events
                .watch(index, Bound::Supply, voltage, range, time);
        }
    }

    /// Settles the logic with the analog solution `solved`, holding each driver at its net's
    /// level and solving again until the detectors read what the logic settled with.
    pub(crate) fn settle(
        &mut self,
        mut solved: Solved,
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        let detectors = self.system.logic.detectors.len();
        let mut previous = vec![Level::Unknown; detectors];
        let mut unsettled = vec![false; detectors];
        for round in 0.. {
            let mut levels: Vec<Level> = self
                .system
                .logic
                .detectors
                .iter()
                .map(|&(element, _)| {
                    let sense = self.system.devices[element].device.as_detector();
                    let input = Input::new(&solved.values, 0.0, 0.0);
                    sense.map_or(Level::Unknown, |sense| {
                        detected(sense.margin(&input), Level::Unknown)
                    })
                })
                .collect();
            for ((level, before), unsettled) in levels.iter_mut().zip(&previous).zip(&mut unsettled)
            {
                *unsettled |= round >= SETTLE_ROUNDS && level != before;
                if *unsettled {
                    *level = Level::Unknown;
                }
            }
            self.events.settle(&self.system.logic, &levels);
            previous = levels;
            let changed = self.events.retarget(&self.system.logic);
            if changed.is_empty() {
                break;
            }
            for (driver, level) in changed {
                let element = self.system.logic.drivers[driver].0;
                self.drive(element, (0.0, level, 0.0));
            }
            solved = self.solve_dc(solved.values, options)?;
        }
        self.measure_loads(&solved.values, options);
        self.watch_supplies(&solved.values, 0.0, options);
        Ok(OperatingPoint {
            observation: self.observer.dc(&solved.values, options.gmin),
            values: solved.values,
            logic: self.events.levels().to_vec(),
            strategy: solved.strategy,
        })
    }

    /// Finds a DC solution starting from `guess`, trying Newton iteration, then gmin stepping,
    /// then source stepping.
    pub(crate) fn solve_dc(
        &mut self,
        guess: Vec<f64>,
        options: &Options,
    ) -> Result<Solved, SimulationError> {
        self.solve_deflated(guess, options, &[])
    }

    /// As [`Simulator::solve_dc`], on the equations deflated by the solutions `deflated`
    /// already found, so that it finds another.
    pub(crate) fn solve_deflated(
        &mut self,
        guess: Vec<f64>,
        options: &Options,
        deflated: &[Vec<f64>],
    ) -> Result<Solved, SimulationError> {
        let mut values = guess.clone();
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
            anchor: None,
            deflated,
        };
        if self
            .newton(&mut values, direct, options, options.dc_iterations)
            .is_ok()
        {
            return Ok(Solved {
                values,
                strategy: Strategy::Direct,
            });
        }
        let mut values = guess.clone();
        if self
            .gmin_stepping(&mut values, options, deflated, Stepped::Shunt)
            .is_ok()
        {
            return Ok(Solved {
                values,
                strategy: Strategy::GminStepping,
            });
        }
        let mut values = guess;
        if self
            .gmin_stepping(&mut values, options, deflated, Stepped::Junctions)
            .is_ok()
        {
            return Ok(Solved {
                values,
                strategy: Strategy::JunctionGminStepping,
            });
        }
        let zero = vec![0.0; values.len()];
        let mut values = zero.clone();
        let stepping = match self.source_stepping(&mut values, options, deflated) {
            Ok(()) => {
                return Ok(Solved {
                    values,
                    strategy: Strategy::SourceStepping,
                });
            }
            Err(failure) => failure,
        };
        let stepped = values;
        let mut values = zero;
        // Equations source stepping found singular are singular as they are, though the
        // conductances of a pseudo-transient hold them up until it gives up.
        match (
            stepping,
            self.pseudo_transient(&mut values, options, deflated),
        ) {
            (_, Ok(())) => Ok(Solved {
                values,
                strategy: Strategy::PseudoTransient,
            }),
            (Failure::Singular, _) | (_, Err(Failure::Singular)) => Err(SimulationError::Singular),
            (Failure::Diverged, Err(Failure::Diverged)) => Err(SimulationError::NoOperatingPoint {
                worst: self.furthest_from_converging(&stepped, options, NAMED_UNKNOWNS),
            }),
        }
    }

    /// Solves with the conductance `stepped` large, shrinking it from each solution to the
    /// next to gmin, until the circuit is solved as it is, without the shunt or with gmin
    /// across its junctions, as ngspice's dynamic and true gmin stepping do: the divisor
    /// grows after a quick success, to at most a decade, shrinks to its square root after a
    /// stage that barely converged and to its fourth root after a failure, which retries
    /// from the last solution, and gives up below [`SMALLEST_STEPPED_DIVISOR`]. Every solve
    /// is deflated by the solutions `deflated`.
    fn gmin_stepping(
        &mut self,
        values: &mut [f64],
        options: &Options,
        deflated: &[Vec<f64>],
        stepped: Stepped,
    ) -> Result<(), Failure> {
        let mut divisor = STEPPED_DIVISOR;
        let mut conductance = FIRST_STEPPED.max(options.gmin);
        let mut last: Option<(f64, Vec<f64>)> = None;
        loop {
            let (shunt, gmin) = match stepped {
                Stepped::Shunt => (conductance, options.gmin),
                Stepped::Junctions => (0.0, conductance),
            };
            let mode = Mode::Dc {
                source_scale: 1.0,
                shunt,
                anchor: None,
                deflated,
            };
            let stepping = Options { gmin, ..*options };
            match self.newton(values, mode, &stepping, options.dc_iterations) {
                Ok(_) if conductance <= options.gmin => break,
                Ok(iterations) => {
                    last = Some((conductance, values.to_vec()));
                    if iterations < options.dc_iterations / 4 {
                        divisor = (divisor * divisor.sqrt()).min(STEPPED_DIVISOR);
                    } else if iterations > 3 * options.dc_iterations / 4 {
                        divisor = divisor.sqrt();
                        // Stage after stage barely converging, the conductance would creep
                        // toward gmin without end: the stepping has stalled.
                        if divisor < SMALLEST_STEPPED_DIVISOR {
                            return Err(Failure::Diverged);
                        }
                    }
                    conductance = (conductance / divisor).max(options.gmin);
                }
                Err(failure) => {
                    divisor = divisor.sqrt().sqrt();
                    let Some((solved, solution)) = &last else {
                        return Err(failure);
                    };
                    if divisor < SMALLEST_STEPPED_DIVISOR {
                        return Err(failure);
                    }
                    values.copy_from_slice(solution);
                    conductance = (solved / divisor).max(options.gmin);
                }
            }
        }
        match stepped {
            // The last stage solved the circuit as it is.
            Stepped::Junctions => Ok(()),
            Stepped::Shunt => {
                let direct = Mode::Dc {
                    source_scale: 1.0,
                    shunt: 0.0,
                    anchor: None,
                    deflated,
                };
                self.newton(values, direct, options, options.dc_iterations)
                    .map(|_| ())
            }
        }
    }

    /// Follows the circuit toward its DC solution as time would, by pseudo-transient
    /// continuation (Kelley and Keyes, "Convergence analysis of pseudo-transient
    /// continuation", SIAM Journal on Numerical Analysis 35(2), 1998), where gmin and
    /// source stepping stall at a fold of the solutions they follow: each step solves the
    /// circuit with every node held to its last voltage by a conductance, as a farad's
    /// capacitance over the step would hold it, and the conductance shrinks a decade after
    /// a quick step and less after a slow one, and grows after a failure, which retries
    /// from the last step. A step that leaves the circuit where it was is a DC solution,
    /// on which Newton iteration without the conductance closes; every solve is deflated
    /// by the solutions `deflated`.
    fn pseudo_transient(
        &mut self,
        values: &mut [f64],
        options: &Options,
        deflated: &[Vec<f64>],
    ) -> Result<(), Failure> {
        let mut held = FIRST_HELD;
        let mut last = values.to_vec();
        for _ in 0..MOST_HELD_STEPS {
            let mode = Mode::Dc {
                source_scale: 1.0,
                shunt: held,
                anchor: Some(&last),
                deflated,
            };
            match self.newton(values, mode, options, options.dc_iterations) {
                Ok(iterations) => {
                    let settled = values.iter().zip(&last).zip(&self.system.unknowns).all(
                        |((value, before), unknown)| {
                            let magnitude = value.abs().max(before.abs());
                            (value - before).abs()
                                <= options.relative_tolerance * magnitude
                                    + options.value_tolerance(unknown.quantity)
                        },
                    );
                    last.copy_from_slice(values);
                    if settled || held <= options.gmin {
                        let direct = Mode::Dc {
                            source_scale: 1.0,
                            shunt: 0.0,
                            anchor: None,
                            deflated,
                        };
                        return self
                            .newton(values, direct, options, options.dc_iterations)
                            .map(|_| ());
                    }
                    held /= if iterations < options.dc_iterations / 4 {
                        STEPPED_DIVISOR
                    } else {
                        STEPPED_DIVISOR.sqrt()
                    };
                }
                Err(failure) => {
                    values.copy_from_slice(&last);
                    held *= STEPPED_DIVISOR;
                    if held > LARGEST_HELD {
                        return Err(failure);
                    }
                }
            }
        }
        Err(Failure::Diverged)
    }

    /// Ramps every source from zero to its level, growing the increment after each success
    /// and shrinking it after each failure; every solve is deflated by the solutions
    /// `deflated`.
    fn source_stepping(
        &mut self,
        values: &mut [f64],
        options: &Options,
        deflated: &[Vec<f64>],
    ) -> Result<(), Failure> {
        let mut scale = 0.0;
        let mut increment = FIRST_SOURCE_INCREMENT;
        let mut last = values.to_vec();
        while scale < 1.0 {
            let next = (scale + increment).min(1.0);
            match self.newton(
                values,
                Mode::Dc {
                    source_scale: next,
                    shunt: 0.0,
                    anchor: None,
                    deflated,
                },
                options,
                options.dc_iterations,
            ) {
                Ok(_) => {
                    scale = next;
                    last.copy_from_slice(values);
                    increment *= 2.0;
                }
                Err(failure) => {
                    values.copy_from_slice(&last);
                    increment /= 4.0;
                    if increment < SMALLEST_SOURCE_INCREMENT {
                        return Err(failure);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use jolty_ir::Circuit;
    use jolty_models::Library;
    use serde_json::json;

    use super::*;
    use crate::Conditions;

    /// A node fed from 3 V through 1 kΩ, a negative conductance of 2 mS at it, and a
    /// diode each way from it to the ground: its current rises with its voltage through
    /// either diode and falls between them, so the solution from zero folds as the
    /// source ramps, and the operating point lies on D1's side, past the fold.
    fn folded() -> Simulator {
        let document = json!({
            "format": "jolty-circuit", "version": 1, "top": "test",
            "components": {"test": {"ports": [{"name": "0", "role": "ground"}],
                                    "instances": [
                {"id": "V1", "of": {"device": "voltage_source"},
                 "parameters": {"waveform": {"dc": 3.0}},
                 "terminals": {"positive": "in", "negative": "0"}},
                {"id": "R1", "of": {"device": "resistor"}, "parameters": {"resistance": 1e3},
                 "terminals": {"a": "in", "b": "x"}},
                {"id": "G1", "of": {"device": "voltage_controlled_current_source"},
                 "parameters": {"transconductance": 2e-3},
                 "terminals": {"positive": "0", "negative": "x",
                               "control_positive": "x", "control_negative": "0"}},
                {"id": "D1", "of": {"device": "diode"}, "parameters": {},
                 "terminals": {"anode": "x", "cathode": "0"}},
                {"id": "D2", "of": {"device": "diode"}, "parameters": {},
                 "terminals": {"anode": "0", "cathode": "x"}},
            ]}}
        });
        let circuit = Circuit::from_json(&document.to_string()).expect("a valid circuit");
        Simulator::new(&circuit, &Library::standard(), &Conditions::default())
            .expect("the circuit elaborates")
    }

    #[test]
    fn pseudo_transient_continuation_follows_a_circuit_past_its_fold() {
        let mut simulator = folded();
        let options = Options::default();
        let mut values = vec![0.0; simulator.system.unknowns.len()];
        simulator
            .pseudo_transient(&mut values, &options, &[])
            .expect("pseudo-transient continuation converges");
        let x = values[simulator.unknown("x").expect("a node").index()];
        // Past the fold, the solution lies where D1 conducts: (3 − x)/1 kΩ + 2 mS · x
        // flows into it.
        let slope = jolty_models::thermal_voltage(jolty_models::REFERENCE_TEMPERATURE);
        let diode = 1e-14 * ((x / slope).exp() - 1.0);
        assert!(x > 0.5, "the node at {x} V");
        assert!(((3.0 - x) / 1e3 + 2e-3 * x - diode).abs() < 1e-9);
    }
}
