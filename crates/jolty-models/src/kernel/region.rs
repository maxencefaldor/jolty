//! Transistor operating-region classifications and boundary margin variables.

/// Variable name holding the zero-based region index within [`Regions::names`].
pub const REGION: &str = "region";

/// Operating-region metadata and boundary-margin variable names for a device model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Regions {
    /// Region names ordered to match the numeric [`REGION`] index.
    pub names: &'static [&'static str],
    /// Margin variable names (in volts) that equal zero on region boundaries and change sign
    /// across transitions.
    ///
    /// Margins use n-channel / NPN sign convention regardless of device polarity (reversing
    /// voltages for p-channel / PNP devices) so positive values always denote conduction or
    /// saturation, matching `|vgs| − |vth|`.
    pub margins: &'static [&'static str],
}

impl Regions {
    /// Empty region set for models without distinct operating regions.
    pub const NONE: Self = Self {
        names: &[],
        margins: &[],
    };
}

/// Gate-source overdrive voltage `vgs − vth` for field-effect transistors.
pub(crate) const OVERDRIVE: &str = "overdrive_voltage";

/// Drain-source saturation margin `vds − vdsat` separating linear and saturated FET regions.
pub(crate) const SATURATION_MARGIN: &str = "saturation_margin";

/// Intrinsic base-emitter voltage in NPN sign convention (emitter-over-base for PNP).
pub(crate) const BASE_EMITTER: &str = "base_emitter_voltage";

/// Intrinsic base-collector voltage in NPN sign convention (collector-over-base for PNP).
pub(crate) const BASE_COLLECTOR: &str = "base_collector_voltage";

/// Operating region of a bipolar transistor determined by which intrinsic junctions inject
/// positive ideal current `Is · (e^{v / (N · Vt)} − 1)` (matching `v > 0` in ngspice
/// `bjtload.c` and VBIC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Bipolar {
    /// Neither junction injects.
    CutOff,
    /// Only the base-emitter junction injects.
    Active,
    /// Both junctions inject.
    Saturated,
    /// Only the base-collector junction injects.
    Reversed,
}

impl Bipolar {
    /// Operating regions and NPN-signed junction voltage margins for bipolar transistors.
    pub(crate) const REGIONS: Regions = Regions {
        names: &["cut_off", "active", "saturated", "reversed"],
        margins: &[BASE_EMITTER, BASE_COLLECTOR],
    };

    /// Diagnostic variable names reported by bipolar transistors (region index and margins).
    pub(crate) const VARIABLE_NAMES: [&str; 3] = [REGION, BASE_EMITTER, BASE_COLLECTOR];

    /// Computes [`Bipolar::VARIABLE_NAMES`] values from NPN-signed intrinsic junction voltages
    /// `vbe` and `vbc`.
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
