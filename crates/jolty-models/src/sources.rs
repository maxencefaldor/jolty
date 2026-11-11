use jolty_numerics::Entry;
use serde::Deserialize;

use crate::waveform::Waveform;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Node, Output,
    ParameterError, Quantity, Source, Unknown,
};

/// The parameters of an independent source.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceParameters {
    /// The source's level over time, in volts or amperes.
    pub waveform: Waveform,
}

/// An independent voltage source from terminal `negative` to terminal `positive`.
///
/// It holds `v_positive − v_negative` at its level. Its current, flowing from `positive`
/// through the source to `negative`, is an unknown labelled `current`, so a source that
/// delivers power carries a negative current.
#[derive(Debug)]
pub struct VoltageSource {
    positive: Node,
    negative: Node,
    current: Unknown,
    waveform: Waveform,
    held: Option<f64>,
    /// The entries at (positive, current), (negative, current), (current, positive) and
    /// (current, negative).
    entries: [Option<Entry>; 4],
}

impl NativeDevice for VoltageSource {
    const NAME: &'static str = "voltage_source";
    type Parameters = SourceParameters;

    fn interface(_: &SourceParameters) -> Interface {
        Interface::new(
            &["positive", "negative"],
            &[Branch::new(0, 1, BranchKind::VoltageSource)],
        )
    }

    fn new(
        parameters: SourceParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let (positive, negative) = (binding.terminal(0), binding.terminal(1));
        let current = binding.internal("current", Quantity::Current);
        let branch = Some(current);
        let entries = [
            binding.entry(positive, branch),
            binding.entry(negative, branch),
            binding.entry(branch, positive),
            binding.entry(branch, negative),
        ];
        Ok(Self {
            positive,
            negative,
            current,
            waveform: parameters.waveform,
            held: None,
            entries,
        })
    }
}

impl Device for VoltageSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let [
            positive_current,
            negative_current,
            current_positive,
            current_negative,
        ] = self.entries;
        let branch = Some(self.current);
        let current = input.value(branch);
        output.add_residual(self.positive, current);
        output.add_residual(self.negative, -current);
        output.add_conductance(positive_current, 1.0);
        output.add_conductance(negative_current, -1.0);
        let level = input.source_scale * level(self.held, &self.waveform, input.time);
        output.add_residual(
            branch,
            input.value(self.positive) - input.value(self.negative) - level,
        );
        output.add_conductance(current_positive, 1.0);
        output.add_conductance(current_negative, -1.0);
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

impl Source for VoltageSource {
    fn hold(&mut self, level: Option<f64>) {
        self.held = level;
    }

    fn level(&self) -> f64 {
        self.waveform.value(0.0)
    }

    fn level_derivative(&self, derivative: &mut [f64]) {
        derivative[self.current.index()] -= 1.0;
    }
}

/// An independent current source that drives its level from terminal `positive` through the
/// source to terminal `negative`, so it pushes current into the `negative` node.
#[derive(Debug)]
pub struct CurrentSource {
    positive: Node,
    negative: Node,
    waveform: Waveform,
    held: Option<f64>,
}

impl NativeDevice for CurrentSource {
    const NAME: &'static str = "current_source";
    type Parameters = SourceParameters;

    fn interface(_: &SourceParameters) -> Interface {
        Interface::new(
            &["positive", "negative"],
            &[Branch::new(0, 1, BranchKind::CurrentSource)],
        )
    }

    fn new(
        parameters: SourceParameters,
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

impl Device for CurrentSource {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let current = input.source_scale * level(self.held, &self.waveform, input.time);
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

impl Source for CurrentSource {
    fn hold(&mut self, level: Option<f64>) {
        self.held = level;
    }

    fn level(&self) -> f64 {
        self.waveform.value(0.0)
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
fn level(held: Option<f64>, waveform: &Waveform, time: f64) -> f64 {
    held.unwrap_or_else(|| waveform.value(time))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Bench, assert_close, assert_jacobians_match};

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
        let mut input = Input::new(&[0.0, 0.0], 0.0, 0.0);
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
    fn jacobians_match_finite_differences() {
        assert_jacobians_match::<VoltageSource>(FIVE_VOLTS, &[1.0, -0.5, 1e-3]);
        assert_jacobians_match::<CurrentSource>(FIVE_VOLTS, &[1.0, -0.5]);
    }
}
