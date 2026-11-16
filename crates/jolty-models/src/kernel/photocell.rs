use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, Nature, NoiseSource, ParameterError,
    Real, Stamp, UnknownIndex,
};

/// The illuminance, in lux, below which a photocell's response to light rounds off to
/// darkness, so that its slope stays finite where the light goes out.
const DARKNESS: f64 = 1e-6;

/// A photocell, or light-dependent resistor, as its maker's datasheet gives one: its
/// resistance between `a` and `b` falls as a power of the illuminance on it. Datasheets,
/// such as Senba's of its GL55 series of cadmium sulphide cells, state the resistance at
/// 10 lux and `γ = lg(R₁₀ / R₁₀₀)`, the slope of the resistance's logarithm against the
/// illuminance's from 10 to 100 lux.
///
/// The voltage of `light_positive` over `light_negative` is the illuminance, 1 V for each
/// lux, and those terminals draw no current. With `r` the resistance at the reference
/// illuminance `E₀` and `γ` its slope on log scales, the cell's conductance is
/// `1/r_dark + s/r`, where its response `s` settles to `(E/E₀)^γ`, the illuminance `E` held
/// at zero or above, smoothly, below a millionth of a lux. With a response time `τ`, `s` lags the
/// light, `τ · ds/dt = (E/E₀)^γ − s`, as a cadmium sulphide cell's carriers do, and is an
/// unknown labelled `response`; without one, it follows at once.
#[derive(Debug, Clone, Copy)]
pub struct Photocell;

/// Where a photocell's contributions go, among its local unknowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhotocellShape {
    /// The entries joining `a` and `b` to every terminal.
    entries: [[usize; 4]; 2],
    lag: Option<Lag>,
}

/// A photocell's lagging response, and the entries its equation claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lag {
    response: usize,
    /// The entries joining `a` and `b` to the response.
    terminals: [usize; 2],
    /// The entries joining the response to `light_positive`, `light_negative` and itself.
    own: [usize; 3],
}

/// A photocell's numbers.
#[derive(Debug, Clone, Copy)]
pub struct PreparedPhotocell<R: Real> {
    conductance: R,
    dark_conductance: R,
    reference: R,
    gamma: R,
    /// Its response time, zero for none.
    time: R,
    temperature: f64,
}

/// The local indices of a photocell's terminals: `a`, `b`, `light_positive` and
/// `light_negative`.
const TERMINALS: [usize; 4] = [0, 1, 2, 3];

/// The parameters of a [`Photocell`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct PhotocellParameters<R: Real = f64> {
    /// The resistance at the reference illuminance, in ohms; positive.
    pub resistance: R,
    /// The reference illuminance, in lux; positive; default 10, as datasheets give it,
    /// such as Senba's of its GL55 series.
    #[serde(default = "ten::<R>")]
    pub illuminance: R,
    /// How steeply the resistance falls with the illuminance, `γ`, the slope of its
    /// logarithm against the illuminance's; positive; default 0.6, which with the default
    /// dark resistance is Senba's GL5528, the most common cell of its GL55 series.
    #[serde(default = "six_tenths::<R>")]
    pub gamma: R,
    /// The resistance in darkness, in ohms; positive; default 1e6, the GL5528's.
    #[serde(default = "a_million::<R>")]
    pub dark_resistance: R,
    /// The time constant of the response, in seconds; zero or more; default 0.
    #[serde(default)]
    pub response_time: R,
}

fn ten<R: Real>() -> R {
    R::constant(10.0)
}

fn six_tenths<R: Real>() -> R {
    R::constant(0.6)
}

fn a_million<R: Real>() -> R {
    R::constant(1e6)
}

impl<R: Real> PreparedPhotocell<R> {
    /// The response `(E/E₀)^γ` the light `light` settles the cell to, and its derivative
    /// with respect to the light.
    ///
    /// The light is held above zero as `(E + √(E² + d²)) / 2`, `d` the darkness, whose
    /// slope is itself over `√(E² + d²)`. Below zero it is written `d² / (2 · (√(E² + d²) −
    /// E))`, the same number without the cancellation that rounds it to zero past `d /
    /// √(2u)`, 67 lux below, `u` half of `f64::EPSILON`, where the response's slope would
    /// divide zero by zero.
    fn settled(&self, light: f64) -> (R, R) {
        let root = light.hypot(DARKNESS);
        let lit = if light >= 0.0 {
            f64::midpoint(light, root)
        } else {
            DARKNESS * DARKNESS / (2.0 * (root - light))
        };
        let response = (R::constant(lit) / self.reference).pow(self.gamma);
        (response, response * self.gamma / root)
    }
}

impl Kernel for Photocell {
    const NAME: &'static str = "photocell";
    type Parameters<R: Real> = PhotocellParameters<R>;
    type Shape = PhotocellShape;
    type Prepared<R: Real> = PreparedPhotocell<R>;
    type Memory = ();

    fn interface<R: Real>(_: &PhotocellParameters<R>) -> Interface {
        Interface::new(
            &["a", "b", "light_positive", "light_negative"],
            &[Branch::new(0, 1, BranchKind::Conductive)],
        )
    }

    fn prepare<R: Real>(
        parameters: &PhotocellParameters<R>,
        temperature: R,
    ) -> Result<PreparedPhotocell<R>, ParameterError> {
        let resistance = ParameterError::positive("resistance", parameters.resistance)?;
        let reference = ParameterError::positive("illuminance", parameters.illuminance)?;
        let gamma = ParameterError::positive("gamma", parameters.gamma)?;
        let dark = ParameterError::positive("dark_resistance", parameters.dark_resistance)?;
        let time = ParameterError::non_negative("response_time", parameters.response_time)?;
        Ok(PreparedPhotocell {
            conductance: resistance.recip(),
            dark_conductance: dark.recip(),
            reference,
            gamma,
            time,
            temperature: temperature.value(),
        })
    }

    fn layout<R: Real>(
        _: &PhotocellParameters<R>,
        prepared: &PreparedPhotocell<R>,
        layout: &mut Layout,
    ) -> Result<PhotocellShape, ParameterError> {
        let [a, b, light_positive, light_negative] = TERMINALS;
        let entries = [a, b].map(|row| TERMINALS.map(|column| layout.entry(row, column)));
        let lag = (prepared.time.value() > 0.0).then(|| {
            let response = layout.internal("response", Nature::State);
            Lag {
                response,
                terminals: [a, b].map(|node| layout.entry(node, response)),
                own: [light_positive, light_negative, response]
                    .map(|column| layout.entry(response, column)),
            }
        });
        Ok(PhotocellShape { entries, lag })
    }

    fn evaluate<R: Real>(
        shape: &PhotocellShape,
        prepared: &PreparedPhotocell<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [a, b, light_positive, light_negative] = TERMINALS;
        let voltage = at.value(a) - at.value(b);
        let light = at.value(light_positive) - at.value(light_negative);
        let (settled, slope) = prepared.settled(light);
        let response = shape
            .lag
            .as_ref()
            .map_or(settled, |lag| R::constant(at.value(lag.response)));
        let conductance = prepared.dark_conductance + prepared.conductance * response;
        let current = conductance * voltage;
        stamp.add_residual(a, current);
        stamp.add_residual(b, -current);
        // The current's derivatives with respect to a, b, and the light.
        let light_slope = match &shape.lag {
            Some(_) => R::constant(0.0),
            None => prepared.conductance * slope * voltage,
        };
        let derivatives = [conductance, -conductance, light_slope, -light_slope];
        for (row, sign) in shape.entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                stamp.add_conductance(entry, derivative * sign);
            }
        }
        if let Some(lag) = &shape.lag {
            let row = lag.response;
            for (&entry, sign) in lag.terminals.iter().zip([1.0, -1.0]) {
                stamp.add_conductance(entry, prepared.conductance * voltage * sign);
            }
            // τ · ds/dt + s − (E/E₀)^γ = 0.
            stamp.add_residual(row, response - settled);
            stamp.add_charge(row, prepared.time * response);
            let [positive, negative, own] = lag.own;
            stamp.add_conductance(positive, -slope);
            stamp.add_conductance(negative, slope);
            stamp.add_conductance(own, R::constant(1.0));
            stamp.add_capacitance(own, prepared.time);
        }
    }

    /// The thermal noise of the conductance its response sets.
    fn noise<R: Real>(
        shape: &PhotocellShape,
        prepared: &PreparedPhotocell<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let [a, b, light_positive, light_negative] = TERMINALS;
        let response = shape.lag.as_ref().map_or_else(
            || {
                prepared
                    .settled(at.value(light_positive) - at.value(light_negative))
                    .0
            },
            |lag| R::constant(at.value(lag.response)),
        );
        let conductance = prepared.dark_conductance + prepared.conductance * response;
        sources.push(NoiseSource::thermal(
            "thermal",
            [Some(UnknownIndex::new(a)), Some(UnknownIndex::new(b))],
            conductance.value(),
            prepared.temperature,
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

    /// Far below zero, past 67 lux below, where the smoothed light once rounded to zero
    /// and its response's slope divided zero by zero, the slope is finite.
    #[test]
    fn its_slope_is_finite_however_dark() {
        let lagging = r#"{"resistance": 1e4, "response_time": 0.01}"#;
        let mut bench = Bench::new::<Photocell>(lagging);
        for light in [-100.0, -1e3, -1e300] {
            let evaluation = bench.evaluate(&[1.0, 0.0, light, 0.0, 0.5]);
            let slopes = &evaluation.conductance[4];
            assert!(
                slopes.iter().all(|slope| slope.is_finite()),
                "at {light} lux"
            );
        }
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
