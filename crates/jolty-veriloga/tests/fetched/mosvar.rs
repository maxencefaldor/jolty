//! Closed-form verification of MOSVAR 1.3.0 compiled from IHP open-PDK Verilog-A source
//! against its linear parasitic network.
//!
//! At default temperature, the geometry yields exact closed forms for gate salicide
//! resistance `rshg * w / (3 * l)` (split `3 + 9` ways for dual contacts) and vertical
//! poly resistance `rpv / (w * l)`. Bulk end resistance `rend / (2 * (w + dwr))` and
//! fringe capacitance `2 * (cfrw * w + cfrl * l)` also follow closed forms. Full CMC QA
//! vectors remain gated behind CMC distribution terms.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/mosvar.rs"]
mod kernel;

use jolty_models::REFERENCE_TEMPERATURE;
use jolty_models::testing::Bench;

use crate::closed::Closed;

/// Closed-form tolerance in units of machine epsilon times magnitude for roughly four rounding
/// operations.
const CLOSED: Closed = Closed(4.0);

/// Local unknown indices: terminals (gate, inner bulk, bulk) followed by internal gate nodes.
const G: usize = 0;
const BI: usize = 1;
const B: usize = 2;
const GII: usize = 3;
const GI: usize = 4;

/// Verifies parasitic network conductances and fringe capacitance at two biases across three
/// geometries and both gate-contact counts.
#[test]
fn its_parasitic_network_is_its_closed_forms() {
    for (w, l, ngcon) in [(1e-5, 2e-6, 1), (3e-6, 5e-7, 2), (2e-5, 1e-5, 1)] {
        // The gate's sheet, the poly's vertical and the well's end resistances, the width
        // the end's adds, and the fringe's capacitances per width and per length.
        let (sheet, poly, ends, widened, per_width, per_length) =
            (8.0, 1e-11, 1e-4, 2e-7, 1e-10, 2e-10);
        let parameters = format!(
            r#"{{"W": {w}, "L": {l}, "NGCON": {ngcon}, "RSHG": {sheet}, "RPV": {poly}, "REND": {ends}, "DWR": {widened}, "CFRW": {per_width}, "CFRL": {per_length}}}"#
        );
        let mut bench = Bench::at::<kernel::Mosvar>(&parameters, REFERENCE_TEMPERATURE);
        let salicide = 1.0 / (sheet * w / ((3.0 + (f64::from(ngcon) - 1.0) * 9.0) * l));
        let vertical = 1.0 / (poly / (w * l));
        let end = 1.0 / (ends / (2.0 * (w + widened)));
        let fringe = 2.0 * (per_width * w + per_length * l);
        for bias in [[0.0; 3], [1.2, 0.0, 0.1]] {
            let mut x = vec![0.0; bench.unknowns().len()];
            x[..3].copy_from_slice(&bias);
            let evaluation = bench.evaluate(&x);
            let g = &evaluation.conductance;
            let c = &evaluation.capacitance;
            let what = format!("{parameters} at {bias:?}");
            CLOSED.assert(&format!("{what}: gate"), g[G][G], salicide);
            CLOSED.assert(&format!("{what}: gate to inner gate"), g[G][GII], -salicide);
            CLOSED.assert(
                &format!("{what}: inner gate"),
                g[GII][GII],
                salicide + vertical,
            );
            CLOSED.assert(&format!("{what}: inner gate down"), g[GII][GI], -vertical);
            CLOSED.assert(&format!("{what}: bulk"), g[B][B], end);
            CLOSED.assert(&format!("{what}: bulk to inner bulk"), g[B][BI], -end);
            CLOSED.assert(&format!("{what}: fringe"), c[GII][GII], fringe);
            CLOSED.assert(&format!("{what}: fringe across"), c[GII][BI], -fringe);
        }
        let given = serde_json::from_str(&parameters).expect("parameters");
        let warnings = kernel::Mosvar::warnings::<f64>(&given, REFERENCE_TEMPERATURE)
            .expect("valid parameters");
        assert_eq!(warnings, Vec::<String>::new(), "{parameters}");
    }
}
