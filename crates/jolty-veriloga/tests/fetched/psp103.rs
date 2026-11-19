//! PSP 103.8.2 with JUNCAP 200.6.2, compiled from its Verilog-A source as IHP's open
//! process kit publishes it, against closed forms where its equations reduce to them.
//!
//! Its source and drain junctions are JUNCAP's: with recombination, trap-assisted and
//! band-to-band tunnelling off, breakdown beyond their reach, and no sidewall or gate
//! edge, at JUNCAP's reference temperature each is an ideal diode of its bottom area,
//! `ab · idsatrbot · (exp(V/φ) − 1)`, `φ = kT/q` in the model's own constants. Its source
//! and drain series resistances are `nrs · rsh` and `nrd · rsh`. Its QA tests, which the CMC
//! hands out only through its own form, wait for the owner.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/psp103.rs"]
mod kernel;

use jolty_models::REFERENCE_TEMPERATURE;
use jolty_models::testing::Bench;

use crate::closed::Closed;

/// How far a value the compiled model computes may lie from the closed form computed
/// here, in units of the machine epsilon times its magnitude: the closed form takes the
/// model's own operations, the exponential of the same argument, and the two scale the
/// saturation current, the area and the polarity and sum the two junctions in their own
/// orders, four roundings of half an epsilon each, with two to spare.
const CLOSED: Closed = Closed(4.0);

/// Boltzmann's constant and the elementary charge as PSP defines them
/// (`Common103_macrodefs.include`), older than the exact SI values jolty keeps.
const KBOL: f64 = 1.380_650_5e-23;
const QELE: f64 = 1.602_191_8e-19;

/// JUNCAP's reference temperature, its default `TRJ`, in degrees Celsius.
const TRJ: f64 = 21.0;

/// The terminals' local unknowns.
const D: usize = 0;
const S: usize = 2;
const B: usize = 3;

/// The bulk's current into both junctions and its conductances to the source and the
/// drain, with the source, drain and gate at the ground and the bulk from deep reverse
/// to strong forward bias, both channel types, for three pairs of areas.
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

/// The conductances of the source's and the drain's series resistances, for three sheet
/// resistances and counts of squares.
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
