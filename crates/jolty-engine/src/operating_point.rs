//! The DC operating point, with gmin and source stepping when Newton alone fails.

use crate::newton::{Failure, Mode};
use crate::{Options, SimulationError, Simulator};

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
/// How many unknowns a failure names.
const NAMED_UNKNOWNS: usize = 3;

/// A DC solution: every unknown's value, and how it was found.
#[derive(Debug, Clone, PartialEq)]
pub struct OperatingPoint {
    /// The value of every unknown, in the simulator's order.
    pub values: Vec<f64>,
    /// The strategy that found the solution.
    pub strategy: Strategy,
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
    /// # Errors
    ///
    /// Returns [`SimulationError::NoOperatingPoint`] or [`SimulationError::Singular`] when no
    /// strategy converges, and [`SimulationError::InvalidOptions`] for options out of range.
    pub fn operating_point(
        &mut self,
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        options.check()?;
        self.solve_dc(vec![0.0; self.system.unknowns.len()], options)
    }

    /// Finds a DC solution starting from `guess`, trying Newton iteration, then gmin stepping,
    /// then source stepping.
    pub(crate) fn solve_dc(
        &mut self,
        guess: Vec<f64>,
        options: &Options,
    ) -> Result<OperatingPoint, SimulationError> {
        let mut values = guess.clone();
        let direct = Mode::Dc {
            source_scale: 1.0,
            shunt: 0.0,
        };
        if self
            .newton(&mut values, direct, options, options.dc_iterations)
            .is_ok()
        {
            return Ok(OperatingPoint {
                values,
                strategy: Strategy::Direct,
            });
        }
        let mut values = guess;
        if self.gmin_stepping(&mut values, options).is_ok() {
            return Ok(OperatingPoint {
                values,
                strategy: Strategy::GminStepping,
            });
        }
        let mut values = vec![0.0; values.len()];
        match self.source_stepping(&mut values, options) {
            Ok(()) => Ok(OperatingPoint {
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
