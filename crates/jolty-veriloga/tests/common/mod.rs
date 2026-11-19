//! Shared differential comparison helpers for compiled Verilog-A kernel tests.

use jolty_models::NoiseSource;
use jolty_models::testing::Evaluation;

/// Maximum rounding distance between hand-written and compiled values, in units of machine
/// epsilon times magnitude.
///
/// Hand-written kernels compute closed-form derivatives while compiled kernels evaluate them
/// through dual numbers, changing operation grouping. The longest parameter-gradient chain
/// spans roughly 40 operations (`2 * 40 * eps / 2 = 40 eps`). Measured resistor states stay
/// within 1.7 ε and diode states stay within 6.4 ε.
const ROUNDING: f64 = 40.0;

/// The worst distance seen, in units of [`ROUNDING`]'s.
#[derive(Debug, Default)]
pub(crate) struct Worst(f64);

impl Worst {
    /// Checks that `hand` and `compiled` agree to rounding, `scale` the magnitude of the
    /// terms that make them, and records how far apart they lie.
    pub(crate) fn close(&mut self, what: &str, hand: f64, compiled: f64, scale: f64) {
        // Where the equations overflow, as a junction far forward does unlimited, both
        // overflow alike.
        let same = hand.to_bits() == compiled.to_bits() || (hand.is_nan() && compiled.is_nan());
        let magnitude = hand.abs().max(compiled.abs()).max(scale);
        let distance = (hand - compiled).abs();
        let units = if same || distance == 0.0 {
            0.0
        } else {
            distance / (f64::EPSILON * magnitude)
        };
        self.0 = self.0.max(units);
        assert!(
            units <= ROUNDING,
            "{what}: {hand:e} by hand, {compiled:e} compiled, {units:.1} ε apart"
        );
    }

    /// Checks that the evaluations `hand` and `compiled` at `x` agree to rounding.
    ///
    /// A linear instance computes residual and charge as matrix-vector products scaled to those
    /// term magnitudes; nonlinear instances scale to the largest entry magnitude.
    pub(crate) fn evaluations(
        &mut self,
        what: &str,
        x: &[f64],
        hand: &Evaluation,
        compiled: &Evaluation,
    ) {
        assert_eq!(hand.limited, compiled.limited, "{what}: limiting");
        let largest = |values: &[f64]| {
            values
                .iter()
                .fold(0.0_f64, |most, value| most.max(value.abs()))
        };
        let terms = |entries: &[Vec<f64>], row: usize| -> f64 {
            entries[row]
                .iter()
                .zip(x)
                .map(|(entry, value)| (entry * value).abs())
                .sum()
        };
        let residuals = largest(&hand.residual);
        for (row, (a, b)) in hand.residual.iter().zip(&compiled.residual).enumerate() {
            let scale = residuals.max(terms(&hand.conductance, row));
            self.close(&format!("{what}: residual {row}"), *a, *b, scale);
        }
        let charges = largest(&hand.charge);
        for (row, (a, b)) in hand.charge.iter().zip(&compiled.charge).enumerate() {
            let scale = charges.max(terms(&hand.capacitance, row));
            self.close(&format!("{what}: charge {row}"), *a, *b, scale);
        }
        for (kind, hand, compiled) in [
            ("conductance", &hand.conductance, &compiled.conductance),
            ("capacitance", &hand.capacitance, &compiled.capacitance),
        ] {
            let scale = hand.iter().map(|row| largest(row)).fold(0.0, f64::max);
            for (row, (a, b)) in hand.iter().zip(compiled).enumerate() {
                for (column, (a, b)) in a.iter().zip(b).enumerate() {
                    self.close(&format!("{what}: {kind} ({row}, {column})"), *a, *b, scale);
                }
            }
        }
    }

    pub(crate) fn noises(&mut self, what: &str, hand: &[NoiseSource], compiled: &[NoiseSource]) {
        assert_eq!(
            hand.len(),
            compiled.len(),
            "{what}: {hand:?} by hand, {compiled:?} compiled"
        );
        for (a, b) in hand.iter().zip(compiled) {
            assert_eq!(
                (a.name, a.equations, a.exponent),
                (b.name, b.equations, b.exponent),
                "{what}"
            );
            self.close(&format!("{what}: {} white", a.name), a.white, b.white, 0.0);
            self.close(
                &format!("{what}: {} flicker", a.name),
                a.flicker,
                b.flicker,
                0.0,
            );
        }
    }
}
