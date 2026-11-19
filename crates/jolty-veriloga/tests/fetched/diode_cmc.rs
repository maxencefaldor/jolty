//! `DIODE_CMC` 2.0.0, compiled from its Verilog-A source as VA-Models publishes it, against
//! an ideal diode's closed form where its equations reduce to one.
//!
//! With its recombination, trap-assisted and band-to-band tunnelling off, its breakdown
//! beyond its reach, its ideality factors one, no series resistance, no sidewall or gate
//! edge, its recovery model off, and the junction at its reference temperature, the model
//! is an ideal diode of the bottom area `ab`: a current `ab · idsatrbot · (exp(V/φ) − 1)`
//! from anode to cathode, `φ = kT/q` in the model's own constants. Its QA tests, which the
//! CMC hands out only through its own form, wait for the owner.

#![cfg(test)]

#[rustfmt::skip]
#[expect(unreachable_pub, reason = "a fixture is a kernel as the compiler writes it for a library")]
#[path = "kernels/diode_cmc.rs"]
mod kernel;

use jolty_models::testing::Bench;

use crate::closed::Closed;

/// How far a value the compiled model computes may lie from the closed form computed
/// here, in units of the machine epsilon times its magnitude: the closed form takes the
/// model's own operations, the exponential of the same argument, and the two scale the
/// saturation current, the area and the polarity in their own orders, three roundings of
/// half an epsilon each, with one to spare.
const CLOSED: Closed = Closed(4.0);

/// Boltzmann's constant and the elementary charge as `DIODE_CMC` defines them
/// (`DIODE_CMC_SIMKIT_macrodefs.include`), older than the exact SI values jolty keeps.
const KBOL: f64 = 1.380_650_5e-23;
const QELE: f64 = 1.602_191_8e-19;

/// The model's reference temperature, its default `TRJ`, in degrees Celsius.
const TRJ: f64 = 21.0;

/// The current from anode to cathode and its conductance, both polarities, at voltages
/// from deep reverse to strong forward bias, for three areas and saturation currents.
#[test]
fn reduced_to_an_ideal_diode_it_is_one() {
    // The junction's temperature, as the model writes its reference one in kelvins.
    let temperature = 273.15 + TRJ;
    let phit_inverse = 1.0 / (KBOL / QELE * temperature);
    for (ab, idsatrbot, kind) in [(1e-12, 1e-12, 1), (4e-10, 3e-9, -1), (2.5e-11, 7e-15, 1)] {
        let parameters = format!(
            r#"{{"AB": {ab}, "LS": 0, "LG": 0, "IDSATRBOT": {idsatrbot}, "CSRHBOT": 0, "CTATBOT": 0, "CBBTBOT": 0, "VBRBOT": 2000, "TRJ": {TRJ}, "TYPE": {kind}}}"#
        );
        let mut bench = Bench::at::<kernel::DiodeCmc>(&parameters, temperature);
        // The terminals alone, the internal nodes collapsed into them and into the ground.
        assert_eq!(bench.unknowns().len(), 2, "{parameters}");
        let polarity = f64::from(kind);
        for v in [-5.0, -0.4, 0.05, 0.3, 0.75] {
            // The anode at `v` above the cathode in the junction's own polarity.
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
