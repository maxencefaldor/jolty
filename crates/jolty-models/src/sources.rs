#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use jolty_ir::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts::CurrentBranch;
use crate::waveform::Waveform;
use crate::{Branch, BranchKind, Interface, Kernel, Layout, Local, ParameterError, Real, Stamp};

/// The parameters of an independent source held at a constant `level`, in volts or
/// amperes, whatever its waveform: those of a source whose waveform is that level. A DC
/// sweep makes its source from them at each of its levels.
#[must_use]
pub fn held_source(level: f64) -> Parameters {
    let mut parameters = Parameters::new();
    parameters.insert("waveform".to_owned(), serde_json::json!({ "dc": level }));
    parameters
}

/// The parameters of an independent source.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct SourceParameters<R: Real = f64> {
    /// The source's level over time, in volts or amperes.
    pub waveform: Waveform<R>,
}

/// The local indices of a source's terminals, `positive` and `negative`.
const TERMINALS: [usize; 2] = [0, 1];

/// An independent voltage source from terminal `negative` to terminal `positive`.
///
/// It holds `v_positive − v_negative` at its level. Its current, flowing from `positive`
/// through the source to `negative`, is an unknown labelled `current`, so a source that
/// delivers power carries a negative current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageSource;

/// Where a voltage source's contributions go: through its branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageSourceShape(CurrentBranch);

impl Kernel for VoltageSource {
    const NAME: &'static str = "voltage_source";
    type Parameters<R: Real> = SourceParameters<R>;
    type Shape = VoltageSourceShape;
    type Prepared<R: Real> = Waveform<R>;
    type Memory = ();

    fn interface<R: Real>(_: &SourceParameters<R>) -> Interface {
        Interface::new(
            &["positive", "negative"],
            &[Branch::new(0, 1, BranchKind::VoltageSource)],
        )
    }

    fn prepare<R: Real>(
        parameters: &SourceParameters<R>,
        _: R,
    ) -> Result<Waveform<R>, ParameterError> {
        Ok(parameters.waveform.clone())
    }

    fn layout<R: Real>(
        _: &SourceParameters<R>,
        _: &Waveform<R>,
        layout: &mut Layout,
    ) -> Result<VoltageSourceShape, ParameterError> {
        Ok(VoltageSourceShape(CurrentBranch::new(
            layout, "current", TERMINALS,
        )))
    }

    fn evaluate<R: Real>(
        VoltageSourceShape(branch): &VoltageSourceShape,
        waveform: &Waveform<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let level = waveform.value(at.time) * at.source_scale;
        let voltage = branch.voltage(at);
        branch.add(at, stamp, -(level - voltage));
    }

    fn breakpoints<R: Real>(
        waveform: &Waveform<R>,
        (start, stop): (f64, f64),
        times: &mut Vec<f64>,
    ) {
        waveform.breakpoints(start, stop, times);
    }

    fn level<R: Real>(waveform: &Waveform<R>) -> Option<f64> {
        Some(waveform.value(0.0).value())
    }

    /// The residual of the branch's equation, `v − level`, falls by one with the level.
    fn excite<R: Real>(VoltageSourceShape(branch): &VoltageSourceShape, stamp: &mut Stamp<'_, R>) {
        stamp.add_residual(branch.unknown(), R::constant(-1.0));
    }
}

/// An independent current source that drives its level from terminal `positive` through the
/// source to terminal `negative`, so it pushes current into the `negative` node.
#[derive(Debug, Clone, Copy)]
pub struct CurrentSource;

impl Kernel for CurrentSource {
    const NAME: &'static str = "current_source";
    type Parameters<R: Real> = SourceParameters<R>;
    type Shape = ();
    type Prepared<R: Real> = Waveform<R>;
    type Memory = ();

    fn interface<R: Real>(_: &SourceParameters<R>) -> Interface {
        Interface::new(
            &["positive", "negative"],
            &[Branch::new(0, 1, BranchKind::CurrentSource)],
        )
    }

    fn prepare<R: Real>(
        parameters: &SourceParameters<R>,
        _: R,
    ) -> Result<Waveform<R>, ParameterError> {
        Ok(parameters.waveform.clone())
    }

    fn layout<R: Real>(
        _: &SourceParameters<R>,
        _: &Waveform<R>,
        _: &mut Layout,
    ) -> Result<(), ParameterError> {
        Ok(())
    }

    fn evaluate<R: Real>(
        (): &(),
        waveform: &Waveform<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let current = waveform.value(at.time) * at.source_scale;
        let [positive, negative] = TERMINALS;
        stamp.add_residual(positive, current);
        stamp.add_residual(negative, -current);
    }

    fn breakpoints<R: Real>(
        waveform: &Waveform<R>,
        (start, stop): (f64, f64),
        times: &mut Vec<f64>,
    ) {
        waveform.breakpoints(start, stop, times);
    }

    fn level<R: Real>(waveform: &Waveform<R>) -> Option<f64> {
        Some(waveform.value(0.0).value())
    }

    /// The current leaves `positive` and enters `negative`, one ampere for each of the
    /// level.
    fn excite<R: Real>((): &(), stamp: &mut Stamp<'_, R>) {
        let [positive, negative] = TERMINALS;
        stamp.add_residual(positive, R::constant(1.0));
        stamp.add_residual(negative, R::constant(-1.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Input;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match_at,
    };

    const FIVE_VOLTS: &str = r#"{"waveform": {"dc": 5}}"#;

    #[test]
    fn a_voltage_source_constrains_its_voltage() {
        let mut bench = Bench::new::<VoltageSource>(FIVE_VOLTS);
        let output = bench.evaluate(&[7.0, 2.0, -1e-3]);
        assert_close(&output.residual, &[-1e-3, 1e-3, 0.0]);
        let held = serde_json::Value::Object(held_source(3.0)).to_string();
        let mut held = Bench::new::<VoltageSource>(&held);
        assert_close(&held.evaluate(&[7.0, 2.0, 0.0]).residual, &[0.0, 0.0, 2.0]);
    }

    #[test]
    fn source_stepping_scales_the_level() {
        let mut bench = Bench::new::<CurrentSource>(r#"{"waveform": {"dc": 2e-3}}"#);
        let mut input = Input::new(&[0.0, 0.0], None, 0.0);
        input.source_scale = 0.5;
        assert_close(&bench.evaluate_with(&input).residual, &[1e-3, -1e-3]);
    }

    #[test]
    fn level_derivatives_excite_the_right_rows() {
        let mut bench = Bench::new::<VoltageSource>(FIVE_VOLTS);
        let mut derivative = vec![0.0; 3];
        bench
            .device()
            .as_source()
            .unwrap()
            .level_derivative(&mut derivative);
        assert_close(&derivative, &[0.0, 0.0, -1.0]);
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let pulse = r#"{"waveform": {"pulse": {"initial": 0.5, "pulsed": 2, "delay": 0.1,
            "rise": 0.2, "fall": 0.3, "width": 0.4, "period": 1.5}}}"#;
        let sine = r#"{"waveform": {"sine": {"offset": 1, "amplitude": 2, "frequency": 0.25,
            "delay": 0.5, "damping": 0.5, "phase": 0.3}}}"#;
        let piecewise = r#"{"waveform": {"piecewise": {"points": [[-1, 0.5], [2, 4]]}}}"#;
        for (parameters, time) in [
            (FIVE_VOLTS, 0.0),
            (pulse, 1.75),
            (sine, 1.0),
            (piecewise, 0.5),
        ] {
            assert_parameter_gradients_match_at::<VoltageSource>(
                parameters,
                &[1.0, -0.5, 1e-3],
                time,
            );
            assert_parameter_gradients_match_at::<CurrentSource>(parameters, &[1.0, -0.5], time);
        }
    }

    #[test]
    fn jacobians_match_finite_differences() {
        assert_jacobians_match::<VoltageSource>(FIVE_VOLTS, &[1.0, -0.5, 1e-3]);
        assert_jacobians_match::<CurrentSource>(FIVE_VOLTS, &[1.0, -0.5]);
    }
}
