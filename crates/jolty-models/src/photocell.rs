use jolty_numerics::Entry;
use serde::{Deserialize, Serialize};

use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Node,
    NoiseSource, Output, ParameterError, Quantity, Real, Unknown,
};

/// The illuminance, in lux, below which a photocell's response to light rounds off to
/// darkness, so that its slope stays finite where the light goes out.
const DARKNESS: f64 = 1e-6;

/// A photocell, or light-dependent resistor, as its maker's datasheet gives one: its
/// resistance between `a` and `b` falls as a power of the illuminance on it.
///
/// The voltage of `light_positive` over `light_negative` is the illuminance, 1 V for each
/// lux, and those terminals draw no current. With `r` the resistance at the reference
/// illuminance `E₀` and `γ` its slope on log scales, the cell's conductance is
/// `1/r_dark + s/r`, where its response `s` settles to `(E/E₀)^γ`, the illuminance `E` held
/// at zero or above, smoothly, below [`DARKNESS`]. With a response time `τ`, `s` lags the
/// light, `τ · ds/dt = (E/E₀)^γ − s`, as a cadmium sulphide cell's carriers do, and is an
/// unknown labelled `response`; without one, it follows at once.
#[derive(Debug)]
pub struct Photocell<R: Real = f64> {
    nodes: [Node; 4],
    conductance: R,
    dark_conductance: R,
    reference: R,
    gamma: R,
    lag: Option<Lag<R>>,
    /// The entries joining `a` and `b` to every terminal.
    entries: [[Option<Entry>; 4]; 2],
    temperature: f64,
}

/// A photocell's lagging response, and the entries its equation claims.
#[derive(Debug)]
struct Lag<R: Real> {
    response: Unknown,
    time: R,
    /// The entries joining `a` and `b` to the response.
    terminals: [Option<Entry>; 2],
    /// The entries joining the response to `light_positive`, `light_negative` and itself.
    own: [Option<Entry>; 3],
}

/// The parameters of a [`Photocell`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields, bound(deserialize = ""))]
pub struct PhotocellParameters<R: Real = f64> {
    /// The resistance at the reference illuminance, in ohms; positive.
    pub resistance: R,
    /// The reference illuminance, in lux; positive; default 10, as datasheets give it.
    #[serde(default = "ten")]
    pub illuminance: R,
    /// How steeply the resistance falls with the illuminance, `γ`, the slope of its
    /// logarithm against the illuminance's; positive; default 0.7.
    #[serde(default = "seven_tenths")]
    pub gamma: R,
    /// The resistance in darkness, in ohms; positive; default 1e6.
    #[serde(default = "a_million")]
    pub dark_resistance: R,
    /// The time constant of the response, in seconds; zero or more; default 0.
    #[serde(default)]
    pub response_time: R,
}

fn ten<R: Real>() -> R {
    R::constant(10.0)
}

fn seven_tenths<R: Real>() -> R {
    R::constant(0.7)
}

fn a_million<R: Real>() -> R {
    R::constant(1e6)
}

impl NativeModel for Photocell {
    type Device<R: Real> = Photocell<R>;
}

impl<R: Real> NativeDevice<R> for Photocell<R> {
    const NAME: &'static str = "photocell";
    type Parameters = PhotocellParameters<R>;

    fn interface(_: &PhotocellParameters<R>) -> Interface {
        Interface::new(
            &["a", "b", "light_positive", "light_negative"],
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn new(
        parameters: PhotocellParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let resistance = ParameterError::positive("resistance", parameters.resistance)?;
        let reference = ParameterError::positive("illuminance", parameters.illuminance)?;
        let gamma = ParameterError::positive("gamma", parameters.gamma)?;
        let dark = ParameterError::positive("dark_resistance", parameters.dark_resistance)?;
        let time = ParameterError::non_negative("response_time", parameters.response_time)?;
        let nodes = [0, 1, 2, 3].map(|index| binding.terminal(index));
        let entries =
            [nodes[0], nodes[1]].map(|row| nodes.map(|column| binding.entry(row, column)));
        let lag = (time.value() > 0.0).then(|| {
            let response = binding.internal("response", Quantity::State);
            let row = Some(response);
            Lag {
                response,
                time,
                terminals: [nodes[0], nodes[1]].map(|node| binding.entry(node, row)),
                own: [nodes[2], nodes[3], row].map(|column| binding.entry(row, column)),
            }
        });
        Ok(Self {
            nodes,
            conductance: resistance.recip(),
            dark_conductance: dark.recip(),
            reference,
            gamma,
            lag,
            entries,
            temperature: binding.temperature(),
        })
    }
}

impl<R: Real> Photocell<R> {
    /// The response `(E/E₀)^γ` the light `light` settles the cell to, and its derivative
    /// with respect to the light.
    fn settled(&self, light: f64) -> (R, R) {
        let root = light.hypot(DARKNESS);
        let lit = 0.5 * (light + root);
        let rounding = 0.5 * (1.0 + light / root);
        let response = (R::constant(lit) / self.reference).pow(self.gamma);
        (response, response * self.gamma / lit * rounding)
    }
}

impl<R: Real> Device<R> for Photocell<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let [a, b, light_positive, light_negative] = self.nodes;
        let voltage = input.value(a) - input.value(b);
        let light = input.value(light_positive) - input.value(light_negative);
        let (settled, slope) = self.settled(light);
        let response = self
            .lag
            .as_ref()
            .map_or(settled, |lag| R::constant(input.value(Some(lag.response))));
        let conductance = self.dark_conductance + self.conductance * response;
        let current = conductance * voltage;
        output.add_residual(a, current);
        output.add_residual(b, -current);
        // The current's derivatives with respect to a, b, and the light.
        let light_slope = match &self.lag {
            Some(_) => R::constant(0.0),
            None => self.conductance * slope * voltage,
        };
        let derivatives = [conductance, -conductance, light_slope, -light_slope];
        for (row, sign) in self.entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                output.add_conductance(entry, derivative * sign);
            }
        }
        if let Some(lag) = &self.lag {
            let row = Some(lag.response);
            for (&entry, sign) in lag.terminals.iter().zip([1.0, -1.0]) {
                output.add_conductance(entry, self.conductance * voltage * sign);
            }
            // τ · ds/dt + s − (E/E₀)^γ = 0.
            output.add_residual(row, response - settled);
            output.add_charge(row, lag.time * response);
            let [positive, negative, own] = lag.own;
            output.add_conductance(positive, -slope);
            output.add_conductance(negative, slope);
            output.add_conductance(own, R::constant(1.0));
            output.add_capacitance(own, lag.time);
        }
    }

    /// The thermal noise of the conductance its response sets.
    fn noise(&self, input: &Input<'_>, sources: &mut Vec<NoiseSource>) {
        let [a, b, light_positive, light_negative] = self.nodes;
        let response = self.lag.as_ref().map_or_else(
            || {
                self.settled(input.value(light_positive) - input.value(light_negative))
                    .0
            },
            |lag| R::constant(input.value(Some(lag.response))),
        );
        let conductance = self.dark_conductance + self.conductance * response;
        sources.push(NoiseSource::thermal(
            "thermal",
            [a, b],
            conductance.value(),
            self.temperature,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match, source,
        thermal,
    };

    const CELL: &str = r#"{"resistance": 1e4, "gamma": 0.8, "dark_resistance": 1e6}"#;

    #[test]
    fn its_resistance_falls_as_a_power_of_the_light() {
        let mut bench = Bench::new::<Photocell>(CELL);
        // The current with 1 V across the cell, against what it should be, within the
        // rounding near darkness, which lights a dark cell by a few parts in 10,000.
        let mut check = |light: f64, expected: f64| {
            let current = bench.evaluate(&[1.0, 0.0, light, 0.0]).residual[0];
            assert!(
                (current - expected).abs() <= 1e-3 * expected,
                "{current} at {light} lux, not {expected}"
            );
        };
        check(10.0, 1e-6 + 1e-4);
        check(100.0, 1e-6 + 1e-4 * 10f64.powf(0.8));
        check(0.0, 1e-6);
        check(-5.0, 1e-6);
    }

    #[test]
    fn a_response_time_adds_a_lagging_response() {
        let lagging = r#"{"resistance": 1e4, "response_time": 0.01}"#;
        let bench = Bench::new::<Photocell>(lagging);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "n2", "n3", "device:response"]
        );
        let instant = Bench::new::<Photocell>(r#"{"resistance": 1e4}"#);
        assert_eq!(instant.unknowns(), ["n0", "n1", "n2", "n3"]);
    }

    #[test]
    fn jacobians_and_gradients_are_exact() {
        let lagging = r#"{"resistance": 1e4, "gamma": 0.8, "response_time": 0.01}"#;
        // Darkness rounds off over a microlux, finer than finite differences resolve.
        for light in [-2.0, 0.5, 30.0] {
            assert_jacobians_match::<Photocell>(CELL, &[1.0, -0.5, light, 0.0]);
            assert_jacobians_match::<Photocell>(lagging, &[1.0, -0.5, light, 0.0, 2.0]);
            assert_parameter_gradients_match::<Photocell>(CELL, &[1.0, -0.5, light, 0.0]);
            assert_parameter_gradients_match::<Photocell>(lagging, &[1.0, -0.5, light, 0.0, 2.0]);
        }
    }

    #[test]
    fn rejects_unphysical_values() {
        let error = Bench::try_new::<Photocell>(r#"{"resistance": 1e4, "gamma": 0}"#)
            .expect_err("no slope");
        assert_eq!(error.0, "gamma must be positive, not 0.0");
        let error = Bench::try_new::<Photocell>(r#"{"resistance": 1e4, "response_time": -1}"#)
            .expect_err("a negative time");
        assert!(error.0.starts_with("response_time"), "{}", error.0);
    }

    #[test]
    fn its_conductance_makes_thermal_noise() {
        let mut bench = Bench::new::<Photocell>(CELL);
        for light in [0.0, 10.0, 300.0] {
            let x = [1.0, 0.0, light, 0.0];
            let conductance = bench.evaluate(&x).residual[0];
            let sources = bench.noise(&x);
            assert_close(
                &[source(&sources, "thermal").density(1.0)],
                &[thermal(conductance)],
            );
        }
    }
}
