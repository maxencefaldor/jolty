//! The DC operating point, with gmin and source stepping when Newton alone fails.

use std::f64::consts::TAU;

use jolty_models::{Input, Level};
use jolty_numerics::{Complex64, SparseLu};

use crate::logic::{Bound, Events, detected};
use crate::newton::{Failure, Mode, NAMED_UNKNOWNS};
use crate::{Observation, Options, SimulationError, Simulator};

/// The largest conductance gmin stepping starts from, in siemens.
const FIRST_SHUNT: f64 = 1e-2;
/// The largest factor gmin stepping divides the shunt by from one stage to the next.
const SHUNT_DIVISOR: f64 = 10.0;
/// The factor below which gmin stepping, having shrunk its divisor after each failure,
/// gives up.
const SMALLEST_SHUNT_DIVISOR: f64 = 1.00005;
/// The first increment of the source scale in source stepping.
const FIRST_SOURCE_INCREMENT: f64 = 0.1;
/// The smallest increment of the source scale before source stepping gives up.
const SMALLEST_SOURCE_INCREMENT: f64 = 1e-4;
/// The frequency, in hertz, at which a logic driver's load is read: low enough that every
/// capacitance it drives charges fully.
const LOAD_FREQUENCY: f64 = 1.0;
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

/// A DC solution of the analog circuit, with its logic drivers held.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Solved {
    pub(crate) values: Vec<f64>,
    pub(crate) strategy: Strategy,
}

/// How an operating point was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Newton iteration from the initial guess.
    Direct,
    /// Newton iteration while a conductance from every node to the reference shrinks to zero.
    GminStepping,
    /// Newton iteration while every source ramps up from zero.
    SourceStepping,
}

impl Strategy {
    /// The strategy's name in snake case, such as `gmin_stepping`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::GminStepping => "gmin_stepping",
            Self::SourceStepping => "source_stepping",
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
        options.check()?;
        self.reset_logic();
        let solved = self.solve_dc(vec![0.0; self.system.unknowns.len()], options)?;
        self.settle(solved, options)
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
        self.evaluate(values, 0.0, (1.0, options.gmin), true);
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
        let mut values = guess.clone();
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
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
        let mut values = guess;
        if self.gmin_stepping(&mut values, options).is_ok() {
            return Ok(Solved {
                values,
                strategy: Strategy::GminStepping,
            });
        }
        let mut values = vec![0.0; values.len()];
        match self.source_stepping(&mut values, options) {
            Ok(()) => Ok(Solved {
                values,
                strategy: Strategy::SourceStepping,
            }),
            Err(Failure::Singular) => Err(SimulationError::Singular),
            Err(Failure::Diverged) => Err(SimulationError::NoOperatingPoint {
                worst: self.furthest_from_converging(&values, options, NAMED_UNKNOWNS),
            }),
        }
    }

    /// Solves with a conductance from every node to the reference, shrinking it from each
    /// solution to the next, then solves without it, as ngspice's dynamic gmin stepping
    /// does: the divisor grows after a quick success, to at most a decade, and shrinks to
    /// its fourth root after a failure, which retries from the last solution.
    fn gmin_stepping(&mut self, values: &mut [f64], options: &Options) -> Result<(), Failure> {
        let mut divisor = SHUNT_DIVISOR;
        let mut shunt = FIRST_SHUNT;
        let mut last: Option<(f64, Vec<f64>)> = None;
        loop {
            let mode = Mode::Dc {
                source_scale: 1.0,
                shunt,
            };
            match self.newton(values, mode, options, options.dc_iterations) {
                Ok(_) if shunt <= options.gmin => break,
                Ok(iterations) => {
                    last = Some((shunt, values.to_vec()));
                    if iterations < options.dc_iterations / 4 {
                        divisor = (divisor * divisor.sqrt()).min(SHUNT_DIVISOR);
                    } else if iterations > 3 * options.dc_iterations / 4 {
                        divisor = divisor.sqrt();
                    }
                    shunt = (shunt / divisor).max(options.gmin);
                }
                Err(failure) => {
                    divisor = divisor.sqrt().sqrt();
                    let Some((solved, solution)) = &last else {
                        return Err(failure);
                    };
                    if divisor < SMALLEST_SHUNT_DIVISOR {
                        return Err(failure);
                    }
                    values.copy_from_slice(solution);
                    shunt = solved / divisor;
                }
            }
        }
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
        };
        self.newton(values, direct, options, options.dc_iterations)
            .map(|_| ())
    }

    /// Ramps every source from zero to its level, growing the increment after each success
    /// and shrinking it after each failure.
    fn source_stepping(&mut self, values: &mut [f64], options: &Options) -> Result<(), Failure> {
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
