//! Closed-form verification of `R3_CMC` 1.1.2 compiled from IHP open-PDK Verilog-A source.
//!
//! With `sw_lin = 1` at the nominal temperature, depletion and self-heating disable and the
//! resistor body reduces to `rsh * l / w` between internal nodes joined by zero-resistance
//! contacts. Full CMC QA vectors remain gated behind CMC distribution terms.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/r3_cmc.rs"]
mod kernel;

use jolty_models::REFERENCE_TEMPERATURE;
use jolty_models::testing::Bench;

use crate::closed::Closed;

/// Closed-form tolerance in units of machine epsilon times magnitude for roughly eight
/// geometry-scaling rounding operations.
const CLOSED: Closed = Closed(8.0);

/// Returns the local unknown index matching `name` on `bench`.
fn unknown(bench: &Bench, name: &str) -> usize {
    bench
        .unknowns()
        .iter()
        .position(|found| *found == name)
        .expect("an unknown")
}

/// Verifies linear body current and conductance (`rsh * l / w`) across voltage sweeps for both
/// n-type and p-type resistors.
#[test]
fn forced_linear_at_its_nominal_temperature_it_is_a_resistor() {
    for (rsh, l, w, kind) in [
        (50.0, 2e-6, 1e-6, -1),
        (120.0, 5e-6, 2e-6, 1),
        (7.5, 1e-5, 4e-7, -1),
    ] {
        let parameters = format!(
            r#"{{"sw_lin": 1, "rsh": {rsh}, "l": {l}, "w": {w}, "type": {kind}, "tc1": 3e-3, "tc2": 1e-5}}"#
        );
        let mut bench = Bench::at::<kernel::R3Cmc>(&parameters, REFERENCE_TEMPERATURE);
        let (i1, i2) = (unknown(&bench, "device:i1"), unknown(&bench, "device:i2"));
        let resistance = rsh * l / w;
        for v in [-5.0, -0.3, 0.7, 4.0] {
            // The terminals at the internal nodes' voltages, the contacts carrying the body's
            // current.
            let mut x = vec![0.0; bench.unknowns().len()];
            x[2] = v;
            x[i2] = v;
            let evaluation = bench.evaluate(&x);
            let what = format!("{parameters} at {v} V");
            CLOSED.assert(
                &format!("{what}: current"),
                evaluation.residual[i2],
                v / resistance,
            );
            CLOSED.assert(
                &format!("{what}: returning current"),
                evaluation.residual[i1],
                -v / resistance,
            );
            CLOSED.assert(
                &format!("{what}: conductance"),
                evaluation.conductance[i2][i2],
                1.0 / resistance,
            );
            CLOSED.assert(
                &format!("{what}: transconductance"),
                evaluation.conductance[i2][i1],
                -1.0 / resistance,
            );
        }
        let read = serde_json::from_str(&parameters).expect("parameters");
        let warnings = kernel::R3Cmc::warnings::<f64>(&read, REFERENCE_TEMPERATURE).expect("valid");
        assert_eq!(warnings, Vec::<String>::new(), "{parameters}");
    }
    assert_eq!(
        kernel::R3Cmc::warned(&kernel::R3CmcMemory::default()).count(),
        0
    );
}
