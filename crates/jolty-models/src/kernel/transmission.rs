//! Lossless transmission line (SPICE `T` element), derived from ngspice 47
//! `src/spicelib/devices/tra/`: Copyright 1985 - 2018, Regents of the University of
//! California and others. Used under the ngspice Modified BSD license, reproduced in the
//! repository `NOTICE` file.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{RESISTANCE, TIME};
use crate::kernel::parts::CurrentBranch;
use crate::{
    Branch, BranchKind, Delayed, Interface, Kernel, Layout, Lift, Local, ParameterError,
    ParameterKind, ParameterMetadata, Real, Stamp,
};

/// Terminal names of a two-port transmission line, port 1 followed by port 2.
const TERMINALS: &[&str] = &["positive1", "negative1", "positive2", "negative2"];

/// Local terminal indices `[positive, negative]` for port 1 and port 2.
const PORTS: [[usize; 2]; 2] = [[0, 1], [2, 3]];

/// Two-port lossless transmission line using Branin's method of characteristics (F. H.
/// Branin, "Transient analysis of lossless transmission lines", *Proceedings of the IEEE*
/// 55(11), 1967), matching the ngspice `T` element.
///
/// Port `k` spans `positive{k}` and `negative{k}` with branch-current unknown `current{k}`
/// and obeys `v_k - Z₀ i_k = v_l(t - τ) + Z₀ i_l(t - τ)` against the opposite port `l`,
/// characteristic impedance `Z₀`, and propagation delay `τ`. Eliminating ngspice's internal
/// nodes `int1` and `int2` (`traload.c`, lines 36 to 51 and 144 to 145) yields two direct
/// port equations. Past waves are interpolated via [`Local::delayed`], resting at the DC
/// operating point for `t <= 0`.
///
/// At DC, delayed samples equal present values, yielding the exact zero-frequency short
/// `v₁ = v₂` and `i₁ = -i₂` (omitting ngspice's `1 - gmin` scaling at line 56, a `1e-12`
/// relative difference). Small-signal AC analysis multiplies each delayed coupling by
/// `e^{-jωτ}`, matching ngspice `traacld.c`.
#[derive(Debug, Clone, Copy)]
pub struct TransmissionLine;

/// Layout shape for [`TransmissionLine`] storing both port `CurrentBranch` entries and
/// each port equation's coupling to the port current plus the opposite port's delayed state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransmissionLineShape {
    ports: [CurrentBranch; 2],
    entries: [[usize; 4]; 2],
}

/// Parameters for [`TransmissionLine`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct TransmissionLineParameters<R: Real = f64> {
    /// Characteristic impedance `Z₀`, in ohms (SPICE `Z0`); positive.
    #[dimension(RESISTANCE)]
    pub characteristic_impedance: R,
    /// One-way propagation delay `τ`, in seconds (SPICE `TD`); positive.
    #[dimension(TIME)]
    pub delay: R,
}

/// Prepared transmission line parameters (temperature-independent impedance and delay).
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

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        TransmissionLineParameters::<f64>::PARAMETER_KINDS
    }

    /// Returns a four-terminal [`Interface`] with one conductive DC branch per port.
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
            let [
                (positive, positive_weight),
                (negative, negative_weight),
                (arriving, current_weight),
            ] = [positive, negative, other.unknown()]
                .map(|local| at.delayed_unknown::<R>(local, delay));
            // The wave that left the other port a delay before.
            let wave = prepared.impedance * arriving + (positive - negative);
            let residual =
                -(prepared.impedance * own.current::<R>(at)) - wave + own.voltage::<R>(at);
            own.add(at, stamp, residual);
            stamp.add_conductance(current, -prepared.impedance);
            stamp.add_conductance(other_positive, R::constant(-positive_weight));
            stamp.add_conductance(other_negative, R::constant(negative_weight));
            stamp.add_conductance(other_current, -(prepared.impedance * current_weight));
        }
    }

    /// Appends each port's delayed coupling entries referencing the opposite port's voltage
    /// and current one delay in the past.
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

/// Claims the four matrix entries in port `own`'s equation row: its diagonal current entry
/// plus the opposite port's `positive`, `negative`, and `other` current columns.
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
