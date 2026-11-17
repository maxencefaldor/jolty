//! The lossless transmission line, SPICE's `T` element, derived from ngspice 47's
//! implementation of it in `src/spicelib/devices/tra/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Delayed, Interface, Kernel, Layout, Lift, Local, ParameterError, Real,
    Stamp,
};

/// The terminals of a transmission line: each port's pair, port 1's then port 2's.
const TERMINALS: &[&str] = &["positive1", "negative1", "positive2", "negative2"];

/// The local indices of each port's terminals, `positive` then `negative`, port 1's then
/// port 2's.
const PORTS: [[usize; 2]; 2] = [[0, 1], [2, 3]];

/// A lossless transmission line of two ports, each the pair of terminals `positive{k}` and
/// `negative{k}`, the line's waves travelling from one to the other in its `delay`, as
/// ngspice's `T` element (Branin's method of characteristics: F. H. Branin, "Transient
/// analysis of lossless transmission lines", *Proceedings of the IEEE* 55(11), 1967).
///
/// Port `k`, of voltage `v_k` and current `i_k`, from `positive{k}` into the line and out
/// at `negative{k}`, the unknown `current{k}`, holds `v_k − Z₀ i_k = v_l(t − τ) + Z₀
/// i_l(t − τ)`, where `l` is the other port, `Z₀` the characteristic impedance and `τ`
/// the delay: the wave arriving at one port is the wave that left the other a delay
/// before (`traload.c`, lines 36 to 51 and 144 to 145, whose internal nodes `int1` and
/// `int2`, `v_k − Z₀ i_k`, it eliminates). The line reads the past through
/// [`Local::delayed`], which a transient interpolates from the points it accepted; before
/// time zero, the line rests at its operating point, as ngspice's table of delayed waves
/// starts (lines 68 to 86).
///
/// At DC, where the past is the present, it is the exact two-port of a line at zero
/// frequency, `v₁ = v₂` and `i₁ = −i₂`, a short through from port to port; ngspice scales
/// the far current's term there by `1 − gmin` (line 56), a relative change of 1e-12 that
/// jolty leaves out. In a small-signal analysis each delayed term is multiplied by
/// `e^{−jωτ}`, as ngspice's AC load does (`traacld.c`, lines 30 to 58).
#[derive(Debug, Clone, Copy)]
pub struct TransmissionLine;

/// Where a transmission line's contributions go: each port's branch, its current an
/// unknown, and the entries of each port's equation at its own current, and at the other
/// port's positive terminal, negative terminal and current, which it reads in the past.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransmissionLineShape {
    ports: [CurrentBranch; 2],
    entries: [[usize; 4]; 2],
}

/// The parameters of a [`TransmissionLine`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields, bound = "")]
pub struct TransmissionLineParameters<R: Real = f64> {
    /// The characteristic impedance `Z₀`, in ohms, SPICE's `Z0`; positive.
    pub characteristic_impedance: R,
    /// The time a wave takes from one port to the other, in seconds, SPICE's `TD`;
    /// positive.
    pub delay: R,
}

/// A transmission line's numbers: its characteristic impedance and its delay, which no
/// temperature changes.
#[derive(Debug, Clone, Copy)]
pub struct PreparedTransmissionLine<R: Real> {
    impedance: R,
    delay: f64,
}

impl Kernel for TransmissionLine {
    const NAME: &'static str = "transmission_line";
    type Parameters<R: Real> = TransmissionLineParameters<R>;
    type Shape = TransmissionLineShape;
    type Prepared<R: Real> = PreparedTransmissionLine<R>;
    type Memory = ();

    /// Each port a branch that conducts at DC, as a voltage behind an impedance.
    fn interface<R: Real>(_: &TransmissionLineParameters<R>) -> Interface {
        let [[positive1, negative1], [positive2, negative2]] = PORTS;
        Interface::new(
            TERMINALS,
            &[
                Branch::new(positive1, negative1, BranchKind::Conductive),
                Branch::new(positive2, negative2, BranchKind::Conductive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &TransmissionLineParameters<R>,
        _: R,
    ) -> Result<PreparedTransmissionLine<R>, ParameterError> {
        let impedance = ParameterError::positive(
            "characteristic_impedance",
            parameters.characteristic_impedance,
        )?;
        let delay = ParameterError::positive("delay", parameters.delay)?;
        Ok(PreparedTransmissionLine {
            impedance,
            delay: delay.value(),
        })
    }

    fn layout<R: Real>(
        _: &TransmissionLineParameters<R>,
        _: &PreparedTransmissionLine<R>,
        layout: &mut Layout,
    ) -> Result<TransmissionLineShape, ParameterError> {
        let [first, second] = PORTS;
        let ports = [
            CurrentBranch::new(layout, "current1", first),
            CurrentBranch::new(layout, "current2", second),
        ];
        let [near, far] = ports;
        let entries = [
            port_entries(layout, near, far, second),
            port_entries(layout, far, near, first),
        ];
        Ok(TransmissionLineShape { ports, entries })
    }

    fn evaluate<R: Real>(
        shape: &TransmissionLineShape,
        prepared: &PreparedTransmissionLine<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [near, far] = shape.ports;
        let [first, second] = PORTS;
        let [near_entries, far_entries] = shape.entries;
        let ports = [
            (near, far, second, near_entries),
            (far, near, first, far_entries),
        ];
        for (own, other, [positive, negative], entries) in ports {
            let [current, other_positive, other_negative, other_current] = entries;
            let delay = prepared.delay;
            let (positive, positive_weight) = at.delayed(positive, delay);
            let (negative, negative_weight) = at.delayed(negative, delay);
            let (arriving, current_weight) = at.delayed(other.unknown(), delay);
            // The wave that left the other port a delay before.
            let wave = prepared.impedance * arriving + (positive - negative);
            let residual = -(prepared.impedance * own.current(at)) - wave + own.voltage(at);
            own.add(at, stamp, residual);
            stamp.add_conductance(current, -prepared.impedance);
            stamp.add_conductance(other_positive, R::constant(-positive_weight));
            stamp.add_conductance(other_negative, R::constant(negative_weight));
            stamp.add_conductance(other_current, -(prepared.impedance * current_weight));
        }
    }

    /// Each port's equation's terms in the other port's voltage and current, read a
    /// delay in the past.
    fn delayed<R: Real>(
        shape: &TransmissionLineShape,
        prepared: &PreparedTransmissionLine<R>,
        found: &mut Vec<Delayed>,
    ) {
        let delay = prepared.delay;
        let impedance = prepared.impedance.value();
        for [_, positive, negative, current] in shape.entries {
            found.extend(
                [(positive, -1.0), (negative, 1.0), (current, -impedance)].map(
                    |(entry, conductance)| Delayed {
                        entry,
                        delay,
                        conductance,
                    },
                ),
            );
        }
    }
}

/// The entries of the equation of the port `own` at its own current, and at the other
/// port's `positive` and `negative` terminals, `[positive, negative]`, and its current,
/// `other`'s.
fn port_entries(
    layout: &mut Layout,
    own: CurrentBranch,
    other: CurrentBranch,
    [positive, negative]: [usize; 2],
) -> [usize; 4] {
    let row = own.unknown();
    [
        layout.entry(row, row),
        layout.entry(row, positive),
        layout.entry(row, negative),
        layout.entry(row, other.unknown()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{assert_jacobians_match, assert_parameter_gradients_match};

    const LINE: &str = r#"{"characteristic_impedance": 50, "delay": 1e-9}"#;

    #[test]
    fn jacobians_are_exact() {
        assert_jacobians_match::<TransmissionLine>(LINE, &[1.0, -0.5, 0.3, 0.1, 2e-3, -1e-3]);
    }

    #[test]
    fn parameter_gradients_are_exact() {
        assert_parameter_gradients_match::<TransmissionLine>(
            LINE,
            &[1.0, -0.5, 0.3, 0.1, 2e-3, -1e-3],
        );
    }
}
