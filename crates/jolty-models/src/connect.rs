//! The connect elements that join logic devices to the analog circuit, where a logic pin
//! joins an electrical net: a [`ThresholdDetector`] turns the net's voltage into the level
//! an input reads, and a [`LogicDriver`] turns an output's level into a voltage between the
//! device's supply and ground.

#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts as kernel_parts;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Level, Local, ParameterError, Real, Stamp,
};

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
#[derive(Debug, Clone, Copy)]
pub struct LogicDriver;

/// The local index of a connect element's input or output.
const PIN: usize = 0;
/// The local index of its supply.
const SUPPLY: usize = 1;
/// The local index of its ground.
const GROUND: usize = 2;

/// Where a logic driver's contributions go: through the branch from the output to the
/// ground, and at the entries of its current's equation at the ground and the supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DriverShape {
    branch: kernel_parts::CurrentBranch,
    entries: [usize; 2],
}

/// The ramp a logic driver moves along, from the fraction `from` of its supply at `start`
/// to `to` at `end`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ramp {
    start: f64,
    from: f64,
    end: f64,
    to: f64,
}

impl Default for Ramp {
    /// At the unknown level's fraction, before the logic drives it.
    fn default() -> Self {
        let unknown = Level::Unknown.fraction();
        Self {
            start: 0.0,
            from: unknown,
            end: 0.0,
            to: unknown,
        }
    }
}

impl Ramp {
    /// The fraction of the supply the driver holds at `time`.
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
    type Shape = DriverShape;
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
    ) -> Result<DriverShape, ParameterError> {
        let branch = kernel_parts::CurrentBranch::new(layout, "current", [PIN, GROUND]);
        let row = branch.unknown();
        let entries = [layout.entry(row, GROUND), layout.entry(row, SUPPLY)];
        Ok(DriverShape { branch, entries })
    }

    fn evaluate<R: Real>(
        shape: &DriverShape,
        (): &(),
        ramp: &mut Ramp,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // The output held at its fraction of the supply's swing above the ground.
        let fraction = ramp.fraction(at.time);
        let swing = at.value(SUPPLY) - at.value(GROUND);
        let branch = &shape.branch;
        branch.add(
            at,
            stamp,
            R::constant(branch.voltage(at) - fraction * swing),
        );
        let [current_ground, current_supply] = shape.entries;
        stamp.add_conductance(current_ground, R::constant(fraction));
        stamp.add_conductance(current_supply, R::constant(-fraction));
    }

    fn drive(ramp: &mut Ramp, (time, level, transition): (f64, Level, f64)) -> Option<f64> {
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

    fn driven(shape: &DriverShape) -> Option<usize> {
        Some(shape.branch.unknown())
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
#[derive(Debug, Clone, Copy)]
pub struct ThresholdDetector;

/// Where a threshold detector's contributions go: across its input and its ground.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetectorShape(kernel_parts::Pair);

impl Kernel for ThresholdDetector {
    const NAME: &'static str = "threshold_detector";
    type Parameters<R: Real> = ThresholdDetectorParameters;
    type Shape = DetectorShape;
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
    ) -> Result<DetectorShape, ParameterError> {
        Ok(DetectorShape(kernel_parts::Pair::new(layout, PIN, GROUND)))
    }

    fn evaluate<R: Real>(
        DetectorShape(pair): &DetectorShape,
        parameters: &ThresholdDetectorParameters,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let capacitance = parameters.capacitance;
        pair.add_charge(stamp, R::constant(capacitance * pair.voltage(at)));
        pair.add_capacitance(stamp, R::constant(capacitance));
    }

    fn sense<R: Real>(
        DetectorShape(pair): &DetectorShape,
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
        bench.drive((0.0, Level::Low, 0.0));
        let settles = bench.drive((1.0, Level::High, 2.0)).expect("a driver");
        assert_close(&[settles], &[3.0]);
        // Halfway up the ramp, the output holds half of the 5 V from ground at 1 V.
        let x = [3.0, 6.0, 1.0, 0.0];
        let evaluation = bench.evaluate_with(&Input::new(&x, Some(2.0), 0.0));
        assert!((evaluation.residual[3] - (2.0 - 0.5 * 5.0)).abs() < 1e-12);
        // Driven back down from halfway, it takes half the transition.
        let settles = bench.drive((2.0, Level::Low, 2.0)).expect("a driver");
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
