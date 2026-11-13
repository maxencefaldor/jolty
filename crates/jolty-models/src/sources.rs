use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::stamp::CurrentBranch;
use crate::waveform::Waveform;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node, Output,
    ParameterError, Real, Source,
};

/// The parameters of an independent source.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct SourceParameters<R: Real = f64> {
    /// The source's level over time, in volts or amperes.
    pub waveform: Waveform<R>,
}

/// An independent voltage source from terminal `negative` to terminal `positive`.
///
/// It holds `v_positive − v_negative` at its level. Its current, flowing from `positive`
/// through the source to `negative`, is an unknown labelled `current`, so a source that
/// delivers power carries a negative current.
#[derive(Debug)]
pub struct VoltageSource<R: Real = f64> {
    branch: CurrentBranch,
    waveform: Waveform<R>,
    held: Option<f64>,
}

impl NativeModel for VoltageSource {
    type Device<R: Real> = VoltageSource<R>;
}

impl<R: Real> NativeDevice<R> for VoltageSource<R> {
    const NAME: &'static str = "voltage_source";
    type Parameters = SourceParameters<R>;

    fn interface(_: &SourceParameters<R>) -> Interface {
        Interface::new(
            &["positive", "negative"],
            &[Branch::new(0, 1, BranchKind::VoltageSource)],
        )
    }

    fn new(
        parameters: SourceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let terminals = [binding.terminal(0), binding.terminal(1)];
        Ok(Self {
            branch: CurrentBranch::new(binding, "current", terminals),
            waveform: parameters.waveform,
            held: None,
        })
    }
}

impl<R: Real> Device<R> for VoltageSource<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let level = level(self.held, &self.waveform, input.time) * input.source_scale;
        let voltage = self.branch.voltage(input);
        self.branch.add(input, output, -(level - voltage));
    }

    fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        if self.held.is_none() {
            self.waveform.breakpoints(start, stop, times);
        }
    }

    fn as_source(&mut self) -> Option<&mut dyn Source> {
        Some(self)
    }
}

impl<R: Real> Source for VoltageSource<R> {
    fn hold(&mut self, level: Option<f64>) {
        self.held = level;
    }

    fn level(&self) -> f64 {
        self.waveform.value(0.0).value()
    }

    fn level_derivative(&self, derivative: &mut [f64]) {
        derivative[self.branch.unknown().index()] -= 1.0;
    }
}

/// An independent current source that drives its level from terminal `positive` through the
/// source to terminal `negative`, so it pushes current into the `negative` node.
#[derive(Debug)]
pub struct CurrentSource<R: Real = f64> {
    positive: Node,
    negative: Node,
    waveform: Waveform<R>,
    held: Option<f64>,
}

impl NativeModel for CurrentSource {
    type Device<R: Real> = CurrentSource<R>;
}

impl<R: Real> NativeDevice<R> for CurrentSource<R> {
    const NAME: &'static str = "current_source";
    type Parameters = SourceParameters<R>;

    fn interface(_: &SourceParameters<R>) -> Interface {
        Interface::new(
            &["positive", "negative"],
            &[Branch::new(0, 1, BranchKind::CurrentSource)],
        )
    }

    fn new(
        parameters: SourceParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        Ok(Self {
            positive: binding.terminal(0),
            negative: binding.terminal(1),
            waveform: parameters.waveform,
            held: None,
        })
    }
}

impl<R: Real> Device<R> for CurrentSource<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let current = level(self.held, &self.waveform, input.time) * input.source_scale;
        output.add_residual(self.positive, current);
        output.add_residual(self.negative, -current);
    }

    fn breakpoints(&self, start: f64, stop: f64, times: &mut Vec<f64>) {
        if self.held.is_none() {
            self.waveform.breakpoints(start, stop, times);
        }
    }

    fn as_source(&mut self) -> Option<&mut dyn Source> {
        Some(self)
    }
}

impl<R: Real> Source for CurrentSource<R> {
    fn hold(&mut self, level: Option<f64>) {
        self.held = level;
    }

    fn level(&self) -> f64 {
        self.waveform.value(0.0).value()
    }

    fn level_derivative(&self, derivative: &mut [f64]) {
        if let Some(positive) = self.positive {
            derivative[positive.index()] += 1.0;
        }
        if let Some(negative) = self.negative {
            derivative[negative.index()] -= 1.0;
        }
    }
}

/// The level of a source: the held level if any, else its waveform's level at `time`.
fn level<R: Real>(held: Option<f64>, waveform: &Waveform<R>, time: f64) -> R {
    held.map_or_else(|| waveform.value(time), R::constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match_at,
    };

    const FIVE_VOLTS: &str = r#"{"waveform": {"dc": 5}}"#;

    #[test]
    fn a_voltage_source_constrains_its_voltage() {
        let mut bench = Bench::new::<VoltageSource>(FIVE_VOLTS);
        let output = bench.evaluate(&[7.0, 2.0, -1e-3]);
        assert_close(&output.residual, &[-1e-3, 1e-3, 0.0]);
        bench.device().as_source().unwrap().hold(Some(3.0));
        assert_close(&bench.evaluate(&[7.0, 2.0, 0.0]).residual, &[0.0, 0.0, 2.0]);
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
