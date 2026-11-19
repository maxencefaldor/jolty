//! Mixed-signal connect elements bridging logic pins and analog nets.
//!
//! A [`ThresholdDetector`] translates an analog net voltage into a logic input level, and a
//! [`LogicDriver`] drives an analog voltage between supply and ground for a logic output.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts as kernel_parts;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, LogicLevel, ParameterError, Real,
    Stamp,
};

/// Parameter struct for [`LogicDriver`] (empty).
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
pub struct LogicDriverParameters {}

/// Connect element driving an electrical net from a logic output.
///
/// Constrains `v_output − v_ground = s · (v_supply − v_ground)` where `s` ramps smoothly
/// between logic level fractions. The output current flowing from `output` to `ground` is an
/// unknown labelled `current`.
#[derive(Debug, Clone, Copy)]
pub struct LogicDriver;

/// Local index of a connect element's input or output terminal.
const PIN: usize = 0;
/// Local index of the supply terminal.
const SUPPLY: usize = 1;
/// Local index of the ground terminal.
const GROUND: usize = 2;

/// Layout shape for [`LogicDriver`] holding the output branch and supply/ground matrix
/// entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicDriverShape {
    branch: kernel_parts::CurrentBranch,
    entries: [usize; 2],
}

/// Linear voltage-fraction ramp from `from` at `start` to `to` at `end`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ramp {
    start: f64,
    from: f64,
    end: f64,
    to: f64,
}

impl Default for Ramp {
    /// Initializes the ramp at the midpoint fraction of [`LogicLevel::Unknown`].
    fn default() -> Self {
        let unknown = LogicLevel::Unknown.fraction();
        Self {
            start: 0.0,
            from: unknown,
            end: 0.0,
            to: unknown,
        }
    }
}

impl Ramp {
    /// Evaluates the driven supply fraction at `time`, clamped to `[from, to]` outside the
    /// ramp interval.
    #[must_use]
    pub fn fraction(&self, time: f64) -> f64 {
        let Self {
            start,
            from,
            end,
            to,
        } = *self;
        if time >= end {
            to
        } else if time <= start {
            from
        } else {
            from + (to - from) * (time - start) / (end - start)
        }
    }
}

impl Kernel for LogicDriver {
    const NAME: &'static str = "logic_driver";
    type Parameters<R: Real> = LogicDriverParameters;
    type Shape = LogicDriverShape;
    type Prepared<R: Real> = ();
    type Memory = Ramp;

    fn interface<R: Real>(_: &LogicDriverParameters) -> Interface {
        Interface::new(
            &["output", "supply", "ground"],
            &[Branch::new(0, 2, BranchKind::VoltageSource)],
        )
    }

    fn prepare<R: Real>(_: &LogicDriverParameters, _: R) -> Result<(), ParameterError> {
        Ok(())
    }

    fn layout<R: Real>(
        _: &LogicDriverParameters,
        (): &(),
        layout: &mut Layout,
    ) -> Result<LogicDriverShape, ParameterError> {
        let branch = kernel_parts::CurrentBranch::new(layout, "current", [PIN, GROUND]);
        let row = branch.unknown();
        let entries = [layout.entry(row, GROUND), layout.entry(row, SUPPLY)];
        Ok(LogicDriverShape { branch, entries })
    }

    fn evaluate<R: Real>(
        shape: &LogicDriverShape,
        (): &(),
        ramp: &mut Ramp,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // The output held at its fraction of the supply's swing above the ground.
        let fraction = ramp.fraction(at.time);
        let swing = at.unknown::<R>(SUPPLY) - at.unknown::<R>(GROUND);
        let branch = &shape.branch;
        branch.add(at, stamp, branch.voltage::<R>(at) - swing * fraction);
        let [current_ground, current_supply] = shape.entries;
        stamp.add_conductance(current_ground, R::constant(fraction));
        stamp.add_conductance(current_supply, R::constant(-fraction));
    }

    fn drive(ramp: &mut Ramp, (time, level, transition): (f64, LogicLevel, f64)) -> Option<f64> {
        let from = ramp.fraction(time);
        let to = level.fraction();
        *ramp = Ramp {
            start: time,
            from,
            end: time + transition * (to - from).abs(),
            to,
        };
        Some(ramp.end)
    }

    fn driven(shape: &LogicDriverShape) -> Option<usize> {
        Some(shape.branch.unknown())
    }
}

/// The parameters of a threshold detector.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
pub struct ThresholdDetectorParameters {
    /// The fraction of the supply, from ground, at which the level switches.
    pub threshold: f64,
    /// The input's capacitance to ground, in farads.
    pub capacitance: f64,
}

/// Connect element sensing an electrical net for a logic input.
///
/// Compares terminal `input` against `threshold` times the `ground`-to-`supply` voltage and
/// loads `input` with `capacitance` to `ground`.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdDetector;

/// Layout shape for [`ThresholdDetector`] spanning the `input` and `ground` terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThresholdDetectorShape(kernel_parts::Pair);

impl Kernel for ThresholdDetector {
    const NAME: &'static str = "threshold_detector";
    type Parameters<R: Real> = ThresholdDetectorParameters;
    type Shape = ThresholdDetectorShape;
    type Prepared<R: Real> = ThresholdDetectorParameters;
    type Memory = ();

    fn interface<R: Real>(parameters: &ThresholdDetectorParameters) -> Interface {
        let branches = if parameters.capacitance > 0.0 {
            &[Branch::new(0, 2, BranchKind::Capacitive)][..]
        } else {
            &[]
        };
        Interface::new(&["input", "supply", "ground"], branches)
    }

    fn prepare<R: Real>(
        parameters: &ThresholdDetectorParameters,
        _: R,
    ) -> Result<ThresholdDetectorParameters, ParameterError> {
        ParameterError::non_negative("the capacitance", parameters.capacitance)?;
        Ok(*parameters)
    }

    fn layout<R: Real>(
        _: &ThresholdDetectorParameters,
        _: &ThresholdDetectorParameters,
        layout: &mut Layout,
    ) -> Result<ThresholdDetectorShape, ParameterError> {
        Ok(ThresholdDetectorShape(kernel_parts::Pair::new(
            layout, PIN, GROUND,
        )))
    }

    fn evaluate<R: Real>(
        ThresholdDetectorShape(pair): &ThresholdDetectorShape,
        parameters: &ThresholdDetectorParameters,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let capacitance = parameters.capacitance;
        pair.add_charge(stamp, pair.across::<R>(at, pair.voltage(at)) * capacitance);
        pair.add_capacitance(stamp, R::constant(capacitance));
    }

    fn sense<R: Real>(
        ThresholdDetectorShape(pair): &ThresholdDetectorShape,
        parameters: &ThresholdDetectorParameters,
        at: &Local<'_>,
    ) -> Option<(f64, f64)> {
        let swing = at.value(SUPPLY) - at.value(GROUND);
        Some((pair.voltage(at) - parameters.threshold * swing, swing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Input;
    use crate::testing::{Bench, assert_close, assert_jacobians_match};

    #[test]
    fn a_driver_ramps_between_ground_and_supply() {
        let mut bench = Bench::new::<LogicDriver>("{}");
        assert_eq!(bench.unknowns(), ["n0", "n1", "n2", "device:current"]);
        bench.drive((0.0, LogicLevel::Low, 0.0));
        let settles = bench.drive((1.0, LogicLevel::High, 2.0)).expect("a driver");
        assert_close(&[settles], &[3.0]);
        // Halfway up the ramp, the output holds half of the 5 V from ground at 1 V.
        let x = [3.0, 6.0, 1.0, 0.0];
        let evaluation = bench.evaluate_with(&Input::new(&x, Some(2.0), 0.0));
        assert!((evaluation.residual[3] - (2.0 - 0.5 * 5.0)).abs() < 1e-12);
        // Driven back down from halfway, it takes half the transition.
        let settles = bench.drive((2.0, LogicLevel::Low, 2.0)).expect("a driver");
        assert_close(&[settles], &[3.0]);
    }

    #[test]
    fn a_driver_s_jacobian_matches_finite_differences() {
        assert_jacobians_match::<LogicDriver>("{}", &[0.3, 5.0, 0.1, 1e-3]);
    }

    #[test]
    fn a_detector_reads_its_input_against_a_fraction_of_the_supply() {
        let parameters = r#"{"threshold": 0.4, "capacitance": 1e-12}"#;
        let mut bench = Bench::new::<ThresholdDetector>(parameters);
        let x = [2.5, 5.0, 0.0];
        let parameters: ThresholdDetectorParameters =
            serde_json::from_str(parameters).expect("parameters");
        let prepared = ThresholdDetector::prepare(&parameters, 300.0).expect("valid");
        let shape = ThresholdDetector::layout::<f64>(&parameters, &prepared, &mut Layout::new(3))
            .expect("valid");
        let input = Input::new(&x, None, 0.0);
        let (margin, swing) =
            ThresholdDetector::sense::<f64>(&shape, &prepared, &input.local(&[0, 1, 2]))
                .expect("a detector");
        assert!((margin - 0.5).abs() < 1e-12);
        assert_close(&[swing], &[5.0]);
        assert_close(&bench.evaluate(&x).charge[..1], &[2.5e-12]);
    }
}
