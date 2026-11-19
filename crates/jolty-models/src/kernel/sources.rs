use jolty_ir::Parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts::CurrentBranch;
use crate::waveform::Waveform;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, ParameterError, Real, Stamp,
};

/// Builds parameters for an independent source held at constant DC `value` (in volts or
/// amperes), used by DC sweeps at each step.
#[must_use]
pub fn held_source(value: f64) -> Parameters {
    let mut parameters = Parameters::new();
    parameters.insert("waveform".to_owned(), serde_json::json!({ "dc": value }));
    parameters
}

/// The parameters of an independent source.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct SourceParameters<R: Real = f64> {
    /// The source's value over time, in volts or amperes.
    pub waveform: Waveform<R>,
}

/// The local indices of a source's terminals, `positive` and `negative`.
const TERMINALS: [usize; 2] = [0, 1];

/// Expands the shared [`Kernel`] items of an independent source inside its `impl Kernel`.
macro_rules! source_common {
    (
        name: $name:literal,
        branch: $branch:expr,
        shape: (),
    ) => {
        source_common!(@items $name, $branch, (), ());
    };
    (
        name: $name:literal,
        branch: $branch:expr,
        shape: $shape:ty,
    ) => {
        source_common!(@items $name, $branch, $shape, _);
    };
    (@items $name:literal, $branch:expr, $shape:ty, $unused_shape:tt) => {
        const NAME: &'static str = $name;
        type Parameters<R: Real> = SourceParameters<R>;
        type Shape = $shape;
        type Prepared<R: Real> = Waveform<R>;
        type Memory = ();

        fn interface<R: Real>(_: &SourceParameters<R>) -> Interface {
            Interface::new(&["positive", "negative"], &[Branch::new(0, 1, $branch)])
        }

        fn prepare<R: Real>(
            parameters: &SourceParameters<R>,
            _: R,
        ) -> Result<Waveform<R>, ParameterError> {
            Ok(parameters.waveform.clone())
        }

        fn breakpoints<R: Real>(
            waveform: &Waveform<R>,
            (start, stop): (f64, f64),
            times: &mut Vec<f64>,
        ) {
            waveform.breakpoints(start, stop, times);
        }

        fn varies_in_time<R: Real>($unused_shape: &$shape, waveform: &Waveform<R>) -> bool {
            waveform.varies_in_time()
        }

        fn dc_value<R: Real>(waveform: &Waveform<R>) -> Option<f64> {
            Some(waveform.value(0.0).value())
        }
    };
}

/// Independent voltage source constraining `v_positive - v_negative` to its waveform value.
///
/// The branch current flowing from `positive` through the source to `negative` is an unknown
/// labeled `current`, so a source delivering power carries negative current.
#[derive(Debug, Clone, Copy)]
pub struct VoltageSource;

/// Layout shape for [`VoltageSource`] wrapping its branch-current equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoltageSourceShape(CurrentBranch);

impl Kernel for VoltageSource {
    source_common! {
        name: "voltage_source",
        branch: BranchKind::VoltageSource,
        shape: VoltageSourceShape,
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
        let value = waveform.value(at.time) * at.source_scale;
        let voltage = branch.voltage::<R>(at);
        branch.add(at, stamp, -(value - voltage));
    }

    /// Stamps `-1.0` into the branch equation residual (derivative of `v - value` with respect
    /// to `value`).
    fn excite<R: Real>(VoltageSourceShape(branch): &VoltageSourceShape, stamp: &mut Stamp<'_, R>) {
        stamp.add_residual(branch.unknown(), R::constant(-1.0));
    }
}

/// An independent current source that drives its value from terminal `positive` through the
/// source to terminal `negative`, so it pushes current into the `negative` node.
#[derive(Debug, Clone, Copy)]
pub struct CurrentSource;

impl Kernel for CurrentSource {
    source_common! {
        name: "current_source",
        branch: BranchKind::CurrentSource,
        shape: (),
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

    /// Stamps `+1.0` at `positive` and `-1.0` at `negative` (unit excitation per ampere).
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
    use crate::testing::{Bench, assert_close};

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
    fn source_stepping_scales_the_value() {
        let mut bench = Bench::new::<CurrentSource>(r#"{"waveform": {"dc": 2e-3}}"#);
        let mut input = Input::new(&[0.0, 0.0], None, 0.0);
        input.source_scale = 0.5;
        assert_close(&bench.evaluate_with(&input).residual, &[1e-3, -1e-3]);
    }

    #[test]
    fn value_derivatives_excite_the_right_rows() {
        let bench = Bench::new::<VoltageSource>(FIVE_VOLTS);
        assert_close(&bench.excitation(), &[0.0, 0.0, -1.0]);
    }

    #[test]
    fn a_sources_excitation_is_the_derivative_of_its_stamp_with_respect_to_its_value() {
        let pulse = r#"{"waveform": {"pulse": {"initial": 0.5, "pulsed": 2, "rise": 0.2,
            "fall": 0.3, "width": 0.4}}}"#;
        for parameters in [FIVE_VOLTS, pulse] {
            for (model, x) in [
                (
                    &crate::Differentiable::<VoltageSource>::new() as &dyn crate::Model,
                    &[1.0, -0.5, 1e-3][..],
                ),
                (
                    &crate::Differentiable::<CurrentSource>::new(),
                    &[1.0, -0.5][..],
                ),
            ] {
                let (excitation, derivative) =
                    crate::testing::excitation_and_derivative(model, parameters, x);
                assert_eq!(excitation, derivative, "{}", model.name());
            }
        }
    }
}
