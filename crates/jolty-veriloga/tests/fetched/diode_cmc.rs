//! Closed-form verification of `DIODE_CMC` 2.0.0 compiled from published VA-Models source.
//!
//! With recombination, tunneling, series resistance, sidewall/gate edges, and recovery
//! disabled at the reference temperature, the model reduces to an ideal bottom-area diode
//! `ab * idsatrbot * (exp(V / phi) - 1)` with `phi = kT / q` using model constants. Full CMC
//! QA vectors remain gated behind CMC distribution terms.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/diode_cmc.rs"]
mod kernel;

use jolty_models::testing::Bench;

use crate::closed::Closed;

/// Closed-form tolerance in units of machine epsilon times magnitude, allowing three
/// independent scaling-order roundings plus headroom.
const CLOSED: Closed = Closed(4.0);

/// Boltzmann's constant and the elementary charge as `DIODE_CMC` defines them
/// (`DIODE_CMC_SIMKIT_macrodefs.include`), older than the exact SI values jolty keeps.
const KBOL: f64 = 1.380_650_5e-23;
const QELE: f64 = 1.602_191_8e-19;

/// Reference junction temperature (`TRJ` default), in degrees Celsius.
const TRJ: f64 = 21.0;

/// Verifies anode-cathode current and conductance in both polarities from reverse to forward
/// bias across three area and saturation-current pairs.
#[test]
fn reduced_to_an_ideal_diode_it_is_one() {
    // Reference junction temperature converted to kelvins using the model offset.
    let temperature = 273.15 + TRJ;
    let phit_inverse = 1.0 / (KBOL / QELE * temperature);
    for (ab, idsatrbot, kind) in [(1e-12, 1e-12, 1), (4e-10, 3e-9, -1), (2.5e-11, 7e-15, 1)] {
        let parameters = format!(
            r#"{{"AB": {ab}, "LS": 0, "LG": 0, "IDSATRBOT": {idsatrbot}, "CSRHBOT": 0, "CTATBOT": 0, "CBBTBOT": 0, "VBRBOT": 2000, "TRJ": {TRJ}, "TYPE": {kind}}}"#
        );
        let mut bench = Bench::at::<kernel::DiodeCmc>(&parameters, temperature);
        // Internal nodes collapse so only the two terminal unknowns remain.
        assert_eq!(bench.unknowns().len(), 2, "{parameters}");
        let polarity = f64::from(kind);
        for v in [-5.0, -0.4, 0.05, 0.3, 0.75] {
            // Apply voltage `v` across the junction according to device polarity.
            let evaluation = bench.evaluate(&[polarity * v, 0.0]);
            let what = format!("{parameters} at {v} V");
            let x = phit_inverse * v;
            let current = polarity * ab * (idsatrbot * (x.exp() - 1.0));
            let conductance = ab * (idsatrbot * x.exp() * phit_inverse);
            CLOSED.assert(&format!("{what}: current"), evaluation.residual[0], current);
            CLOSED.assert(
                &format!("{what}: returning current"),
                evaluation.residual[1],
                -current,
            );
            CLOSED.assert(
                &format!("{what}: conductance"),
                evaluation.conductance[0][0],
                conductance,
            );
            CLOSED.assert(
                &format!("{what}: transconductance"),
                evaluation.conductance[0][1],
                -conductance,
            );
        }
        let read = serde_json::from_str(&parameters).expect("parameters");
        let warnings =
            kernel::DiodeCmc::warnings::<f64>(&read, temperature).expect("valid parameters");
        assert_eq!(warnings, Vec::<String>::new(), "{parameters}");
    }
    assert_eq!(
        kernel::DiodeCmc::warned(&kernel::DiodeCmcMemory::default()).count(),
        0
    );
}
