#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]
//! The regions of operation a transistor's equations distinguish, which it reports among
//! its variables, with its margins to their boundaries.

/// The name of the variable that numbers a transistor's region among its model's
/// [`Regions::names`], from 0.
pub const REGION: &str = "region";

/// The regions of operation a model's equations distinguish, and the variables that place
/// an instance among them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Regions {
    /// Each region's name, in the order the variable [`REGION`] numbers them.
    pub names: &'static [&'static str],
    /// The variables that are the margins to the regions' boundaries, in volts: each is
    /// zero on a boundary, where the equations switch from one region to the next, and
    /// changes sign across it.
    /// Each is in an n-channel's or an NPN's sign whatever the transistor's polarity, a
    /// p-channel's or a PNP's taken at its voltages reversed, so that a positive margin
    /// means on, or saturated, for every transistor, as designers write `|vgs| − |vth|`.
    pub margins: &'static [&'static str],
}

impl Regions {
    /// The regions of a model whose equations distinguish none.
    pub const NONE: Self = Self {
        names: &[],
        margins: &[],
    };
}

/// A field-effect transistor's gate-source voltage over its threshold, `vgs − vth`.
pub(crate) const OVERDRIVE: &str = "overdrive_voltage";

/// A field-effect transistor's drain-source voltage over the saturation voltage at which its
/// channel's law turns from linear to saturated, `vds − vdsat`.
pub(crate) const SATURATION_MARGIN: &str = "saturation_margin";

/// The voltage across a bipolar transistor's intrinsic base-emitter junction, in an NPN's
/// sign: a PNP's is its emitter's over its base's.
pub(crate) const BASE_EMITTER: &str = "base_emitter_voltage";

/// The voltage across a bipolar transistor's intrinsic base-collector junction, in an NPN's
/// sign: a PNP's is its collector's over its base's.
pub(crate) const BASE_COLLECTOR: &str = "base_collector_voltage";

/// The region of a bipolar transistor, by which of its two intrinsic junctions injects: one
/// whose ideal current, `Is · (e^{v / (N · Vt)} − 1)`, is positive, which it is where its
/// voltage `v` is positive, in ngspice's `bjtload.c` (`cbe`, `cbc`) and in VBIC's
/// equations (`Itzf`, `Itzr`) alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bipolar {
    /// Neither junction injects.
    CutOff,
    /// The base-emitter junction alone injects.
    Active,
    /// Both junctions inject.
    Saturated,
    /// The base-collector junction alone injects.
    Reversed,
}

impl Bipolar {
    /// The regions of a bipolar transistor, in the order of [`Bipolar`]'s, and its margins:
    /// its intrinsic junctions' voltages, in an NPN's sign.
    pub(crate) const REGIONS: Regions = Regions {
        names: &["cut_off", "active", "saturated", "reversed"],
        margins: &[BASE_EMITTER, BASE_COLLECTOR],
    };

    /// The names of the variables a bipolar transistor reports: its region and its margins.
    pub(crate) const VARIABLE_NAMES: [&str; 3] = [REGION, BASE_EMITTER, BASE_COLLECTOR];

    /// The values of [`Bipolar::VARIABLE_NAMES`] at the intrinsic base-emitter and
    /// base-collector voltages `vbe` and `vbc`, oriented as an NPN's whatever the
    /// transistor's polarity.
    pub(crate) fn variables(vbe: f64, vbc: f64) -> [f64; 3] {
        let region = match (vbe > 0.0, vbc > 0.0) {
            (false, false) => Self::CutOff,
            (true, false) => Self::Active,
            (true, true) => Self::Saturated,
            (false, true) => Self::Reversed,
        };
        [f64::from(region as u8), vbe, vbc]
    }
}
