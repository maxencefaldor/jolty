//! Closed-form assertion shared by Verilog-A kernel tests.

/// Closed-form tolerance checker, in units of machine epsilon times magnitude.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Closed(pub(crate) f64);

impl Closed {
    /// Checks that `compiled` lies within `self.0` machine epsilons of `closed` times its
    /// magnitude.
    pub(crate) fn assert(self, what: &str, compiled: f64, closed: f64) {
        let distance = (compiled - closed).abs();
        assert!(
            distance <= self.0 * f64::EPSILON * closed.abs(),
            "{what}: {compiled:e} compiled, {closed:e} closed, {:.1} ε apart",
            distance / (f64::EPSILON * closed.abs())
        );
    }
}
