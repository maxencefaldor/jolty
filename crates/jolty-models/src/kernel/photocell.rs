use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{RESISTANCE, TIME};
use crate::kernel::dual::Dual;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, Nature, NoiseSource,
    ParameterError, ParameterKind, ParameterMetadata, Real, Stamp, UnknownIndex,
};

/// The illuminance, in lux, below which a photocell's response to light rounds off to
/// darkness, so that its slope stays finite where the light goes out.
const DARKNESS: f64 = 1e-6;

/// Light-dependent resistor (photocell) whose resistance between `a` and `b` falls as a
/// power law of illuminance, following standard cadmium sulfide datasheet parameters (such
/// as Senba GL55 series: 10 lux resistance and log-log slope `γ = lg(R₁₀ / R₁₀₀)`).
///
/// The voltage `v(light_positive) − v(light_negative)` encodes illuminance at 1 V per lux
/// without drawing current. Conductance equals `1/r_dark + s/r`, where response `s` tracks
/// `(E/E₀)^γ` with illuminance `E` smoothly floored near 1 microlux. A positive response
/// time `τ` adds an internal state unknown `response` governed by
/// `τ · ds/dt = (E/E₀)^γ − s`; zero response time tracks illuminance instantaneously.
#[derive(Debug, Clone, Copy)]
pub struct Photocell;

/// Layout shape for [`Photocell`] storing terminal conductance entries and optional lag
/// state indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhotocellShape {
    /// Matrix entries coupling terminals `a` and `b` to all four device terminals.
    entries: [[usize; 4]; 2],
    lag: Option<Lag>,
}

/// Internal state unknown and matrix entries for a photocell with nonzero response time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lag {
    response: usize,
    /// Matrix entries coupling terminals `a` and `b` to `response`.
    terminals: [usize; 2],
    /// Matrix entries coupling `response` to `light_positive`, `light_negative`, and itself.
    own: [usize; 3],
}

/// Prepared parameters for [`Photocell`].
#[derive(Debug, Clone, Copy)]
pub struct PreparedPhotocell<R: Real> {
    conductance: R,
    dark_conductance: R,
    reference: R,
    gamma: R,
    /// Response time constant in seconds, or zero for instantaneous response.
    time: R,
    temperature: f64,
}

/// Local terminal indices of [`Photocell`]: `a`, `b`, `light_positive`, and `light_negative`.
const TERMINALS: [usize; 4] = [0, 1, 2, 3];

/// Parameters for [`Photocell`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct PhotocellParameters<R: Real = f64> {
    /// Resistance at the reference illuminance, in ohms; positive.
    #[dimension(RESISTANCE)]
    pub resistance: R,
    /// Reference illuminance, in lux; positive; default 10 (standard datasheet reference).
    #[serde(default = "ten::<R>")]
    pub illuminance: R,
    /// Log-log slope `γ` of resistance versus illuminance; positive; default 0.6 (matching
    /// the Senba GL5528 together with default `dark_resistance`).
    #[serde(default = "six_tenths::<R>")]
    pub gamma: R,
    /// Resistance in darkness, in ohms; positive; default 1e6 (matching the Senba GL5528).
    #[serde(default = "a_million::<R>")]
    #[dimension(RESISTANCE)]
    pub dark_resistance: R,
    /// Carrier response time constant, in seconds; non-negative; default 0.
    #[serde(default)]
    #[dimension(TIME)]
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
    /// Computes the steady-state normalized response `(E/E₀)^γ` for illuminance `light`.
    ///
    /// Smoothly floors illuminance via `(E + √(E² + d²)) / 2` (with `d = DARKNESS`). For
    /// negative `E`, uses the algebraically equivalent form `d² / (2 · (√(E² + d²) − E))`
    /// to avoid catastrophic cancellation past `-67` lux.
    ///
    /// Passes the exact slope `γ · response / √(E² + d²)` directly to [`Dual::chain`] so
    /// deeply negative illuminance never divides by an underflowed floored light value.
    fn settled<const N: usize>(&self, light: Dual<R, N>) -> Dual<R, N> {
        let illuminance = light.value;
        let across = illuminance.value();
        let root_value = across.hypot(DARKNESS);
        let ratio = DARKNESS / root_value;
        let root = illuminance.chain((root_value, across / root_value, ratio * ratio / root_value));
        let lit = if across >= 0.0 {
            (illuminance + root) * 0.5
        } else {
            (root - illuminance).recip() * (0.5 * DARKNESS * DARKNESS)
        };
        let response = (lit / self.reference).pow(self.gamma);
        light.chain((response, response * self.gamma / root))
    }
}

impl Kernel for Photocell {
    const NAME: &'static str = "photocell";
    type Parameters<R: Real> = PhotocellParameters<R>;
    type Shape = PhotocellShape;
    type Prepared<R: Real> = PreparedPhotocell<R>;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        PhotocellParameters::<f64>::PARAMETER_KINDS
    }

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
        // The voltage across the cell, the light, and the lagging response, if it has one.
        let [voltage, light, lagging] = Dual::variables([
            at.unknown::<R>(a) - at.unknown::<R>(b),
            at.unknown::<R>(light_positive) - at.unknown::<R>(light_negative),
            shape
                .lag
                .as_ref()
                .map_or_else(|| R::constant(0.0), |lag| at.unknown(lag.response)),
        ]);
        let settled = prepared.settled(light);
        let response = if shape.lag.is_some() {
            lagging
        } else {
            settled
        };
        let conductance = response
            .scale(prepared.conductance)
            .plus(prepared.dark_conductance);
        let current = conductance * voltage;
        stamp.add_residual(a, current.value);
        stamp.add_residual(b, -current.value);
        // With respect to a, b, the light's terminals, and the lagging response.
        let [by_voltage, by_light, by_response] = current.gradient;
        let derivatives = [by_voltage, -by_voltage, by_light, -by_light];
        for (row, sign) in shape.entries.iter().zip([1.0, -1.0]) {
            for (&entry, derivative) in row.iter().zip(derivatives) {
                stamp.add_conductance(entry, derivative * sign);
            }
        }
        if let Some(lag) = &shape.lag {
            let row = lag.response;
            for (&entry, sign) in lag.terminals.iter().zip([1.0, -1.0]) {
                stamp.add_conductance(entry, by_response * sign);
            }
            // τ · ds/dt + s − (E/E₀)^γ = 0.
            let lagged = response - settled;
            let [_, by_light, by_response] = lagged.gradient;
            stamp.add_residual(row, lagged.value);
            stamp.add_charge(row, prepared.time * response.value);
            let [positive, negative, own] = lag.own;
            stamp.add_conductance(positive, by_light);
            stamp.add_conductance(negative, -by_light);
            stamp.add_conductance(own, by_response);
            stamp.add_capacitance(own, prepared.time);
        }
    }

    /// Appends thermal noise for the conductance determined by the active photocell response.
    fn noise<R: Real>(
        shape: &PhotocellShape,
        prepared: &PreparedPhotocell<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let [a, b, light_positive, light_negative] = TERMINALS;
        let response = shape.lag.as_ref().map_or_else(
            || {
                let light = at.value(light_positive) - at.value(light_negative);
                prepared.settled(Dual::<R, 0>::number(light)).value
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

    /// Verifies that the response slope stays finite far below zero illuminance (past -67
    /// lux), where naive smoothing previously suffered cancellation and divided zero by zero.
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
        assert_eq!(error.0, "gamma is positive, not 0.0");
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
