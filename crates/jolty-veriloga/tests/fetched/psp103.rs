//! Closed-form verification of PSP 103.8.2 with JUNCAP 200.6.2 compiled from IHP open-PDK
//! Verilog-A source.
//!
//! With recombination, tunneling, and perimeter terms disabled at the JUNCAP reference
//! temperature, the source and drain junctions reduce to ideal bottom-area diodes
//! `ab * idsatrbot * (exp(V / phi) - 1)` with `phi = kT / q` using model constants. Source and
//! drain series resistances equal `nrs * rsh` and `nrd * rsh`. Full CMC QA vectors remain
//! gated behind CMC distribution terms.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/psp103.rs"]
mod kernel;

use jolty_models::REFERENCE_TEMPERATURE;
use jolty_models::testing::Bench;

use crate::closed::Closed;

/// Closed-form tolerance in units of machine epsilon times magnitude, allowing four rounding
/// operations plus headroom.
const CLOSED: Closed = Closed(4.0);

/// Boltzmann's constant and the elementary charge as PSP defines them
/// (`Common103_macrodefs.include`), older than the exact SI values jolty keeps.
const KBOL: f64 = 1.380_650_5e-23;
const QELE: f64 = 1.602_191_8e-19;

/// JUNCAP reference temperature (`TRJ` default), in degrees Celsius.
const TRJ: f64 = 21.0;

/// Terminal local unknown indices.
const D: usize = 0;
const S: usize = 2;
const B: usize = 3;

/// Verifies bulk junction currents and conductances across reverse and forward bias for both
/// channel types and three area pairs.
#[test]
fn its_junctions_reduced_to_ideal_diodes_are_ones() {
    // The junctions' temperature, as the model writes its reference one in kelvins.
    let temperature = 273.15 + TRJ;
    let phit_inverse = 1.0 / (KBOL / QELE * temperature);
    for (ab_source, ab_drain, idsatrbot, kind) in [
        (1e-12, 2e-12, 1e-12, 1),
        (4e-11, 3e-12, 3e-9, -1),
        (2.5e-13, 2.5e-13, 7e-7, 1),
    ] {
        let parameters = format!(
            r#"{{"TYPE": {kind}, "SWJUNCAP": 1, "ABSOURCE": {ab_source}, "ABDRAIN": {ab_drain}, "LSSOURCE": 0, "LSDRAIN": 0, "LGSOURCE": 0, "LGDRAIN": 0, "IDSATRBOT": {idsatrbot}, "CSRHBOT": 0, "CTATBOT": 0, "CBBTBOT": 0, "VBRBOT": 2000, "TRJ": {TRJ}}}"#
        );
        let mut bench = Bench::at::<kernel::Psp103va>(&parameters, temperature);
        let polarity = f64::from(kind);
        for v in [-5.0, -0.4, 0.05, 0.3, 0.75] {
            // The bulk at `v` above the source and the drain, in the junctions' polarity.
            let mut x = vec![0.0; bench.unknowns().len()];
            x[B] = polarity * v;
            let evaluation = bench.evaluate(&x);
            let what = format!("{parameters} at {v} V");
            // JUNCAP squares the exponential of half the argument.
            let z = (0.5 * (v * phit_inverse)).exp();
            let density = idsatrbot * (z * z - 1.0);
            let slope = idsatrbot * (z * z) * phit_inverse;
            CLOSED.assert(
                &format!("{what}: bulk current"),
                evaluation.residual[B],
                polarity * (ab_source * density) + polarity * (ab_drain * density),
            );
            CLOSED.assert(
                &format!("{what}: bulk conductance"),
                evaluation.conductance[B][B],
                ab_source * slope + ab_drain * slope,
            );
            CLOSED.assert(
                &format!("{what}: bulk to source"),
                evaluation.conductance[B][S],
                -(ab_source * slope),
            );
            CLOSED.assert(
                &format!("{what}: bulk to drain"),
                evaluation.conductance[B][D],
                -(ab_drain * slope),
            );
        }
    }
}

/// Verifies source and drain series conductances across three sheet resistances and square
/// counts.
#[test]
fn its_series_resistances_are_their_squares() {
    for (rsh, nrs, nrd) in [(20.0, 2.0, 3.0), (7.5, 0.4, 0.4), (150.0, 1.0, 12.0)] {
        let parameters = format!(r#"{{"RSH": {rsh}, "NRS": {nrs}, "NRD": {nrd}}}"#);
        let mut bench = Bench::at::<kernel::Psp103va>(&parameters, REFERENCE_TEMPERATURE);
        let x = vec![0.0; bench.unknowns().len()];
        let evaluation = bench.evaluate(&x);
        let g = &evaluation.conductance;
        let (source, drain) = (1.0 / (nrs * rsh), 1.0 / (nrd * rsh));
        CLOSED.assert(&format!("{parameters}: source"), g[S][S], source);
        CLOSED.assert(&format!("{parameters}: drain"), g[D][D], drain);
        // The inner source and drain, the first unknowns after the noise's node.
        let (inner_source, inner_drain) = (5, 6);
        CLOSED.assert(
            &format!("{parameters}: source inwards"),
            g[S][inner_source],
            -source,
        );
        CLOSED.assert(
            &format!("{parameters}: drain inwards"),
            g[D][inner_drain],
            -drain,
        );
    }
    assert_eq!(
        kernel::Psp103va::warned(&kernel::Psp103vaMemory::default()).count(),
        0
    );
}
