//! `R3_CMC` 1.1.2, compiled from its Verilog-A source as IHP's open process kit
//! publishes it, against a resistor's closed form where its equations reduce to one.
//!
//! Forced linear, `sw_lin = 1`, the model drops its depletion and its self-heating, and
//! at its nominal temperature its body is a sheet of `rsh` ohms per square, `l` long and
//! `w` wide: a resistance of `rsh · l / w` between its internal nodes, which its
//! contacts, of no resistance at their default, join to its terminals. Its QA tests,
//! which the CMC hands out only through its own form, wait for the owner.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/r3_cmc.rs"]
mod kernel;

use jolty_models::REFERENCE_TEMPERATURE;
use jolty_models::testing::Bench;

use crate::closed::Closed;

/// How far a value the compiled model computes may lie from the closed form computed
/// here, in units of the machine epsilon times its magnitude: the model computes the
/// resistance in some eight operations, its geometry scaled to micrometres and back,
/// each rounding by half an epsilon.
const CLOSED: Closed = Closed(8.0);

/// The local index of the unknown `name` on `bench`.
fn unknown(bench: &Bench, name: &str) -> usize {
    bench
        .unknowns()
        .iter()
        .position(|found| *found == name)
        .expect("an unknown")
}

/// The body's current and conductance are a resistor's, `rsh · l / w`, at every voltage
/// across it, for n and p bodies alike, its temperature coefficients idle at the nominal
/// temperature.
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
