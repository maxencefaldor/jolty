use jolty_numerics::{Entry, Pattern};

use crate::Node;

/// The state a device is evaluated at.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a> {
    x: &'a [f64],
    /// The time, in seconds; zero in every analysis but a transient one.
    pub time: f64,
    /// The factor every independent source is scaled by: 1, except while an operating point is
    /// found by ramping the sources up from zero.
    pub source_scale: f64,
    /// The conductance placed across every nonlinear junction, in siemens, so no node of the
    /// linearized circuit is ever left without a path to ground.
    pub gmin: f64,
    /// Whether devices limit each voltage's step from their last evaluation's: false for
    /// the first evaluation of a DC solution, which starts afresh from its guess.
    pub limiting: bool,
}

impl<'a> Input<'a> {
    /// The state `x` at `time`, with sources at full scale, `gmin` across junctions, and
    /// steps limited.
    #[must_use]
    pub fn new(x: &'a [f64], time: f64, gmin: f64) -> Self {
        Self {
            x,
            time,
            source_scale: 1.0,
            gmin,
            limiting: true,
        }
    }

    /// The value of `node`'s unknown, or zero for the reference node.
    #[must_use]
    pub fn value(&self, node: Node) -> f64 {
        node.map_or(0.0, |unknown| self.x[unknown.index()])
    }
}

/// Where devices add their contributions during one evaluation of the circuit.
///
/// Every slice starts zeroed. Residuals and charges are indexed by unknown; conductances and
/// capacitances by the pattern's value slots. Alongside each residual, `scale` keeps the
/// largest magnitude of a term added to it, against which convergence is judged.
#[derive(Debug)]
pub struct Output<'a> {
    pattern: &'a Pattern,
    residual: &'a mut [f64],
    scale: &'a mut [f64],
    charge: &'a mut [f64],
    conductance: &'a mut [f64],
    capacitance: &'a mut [f64],
    limited: bool,
}

impl<'a> Output<'a> {
    /// An output over zeroed buffers: `residual`, `scale` and `charge` hold one value per
    /// unknown, `conductance` and `capacitance` one per stored position of `pattern`.
    ///
    /// # Panics
    ///
    /// Panics if a buffer has the wrong length.
    pub fn new(
        pattern: &'a Pattern,
        residual: &'a mut [f64],
        scale: &'a mut [f64],
        charge: &'a mut [f64],
        conductance: &'a mut [f64],
        capacitance: &'a mut [f64],
    ) -> Self {
        let dimension = pattern.dimension();
        assert!(
            residual.len() == dimension && scale.len() == dimension && charge.len() == dimension,
            "one residual, scale and charge per unknown"
        );
        assert!(
            conductance.len() == pattern.len() && capacitance.len() == pattern.len(),
            "one conductance and capacitance per stored position"
        );
        Self {
            pattern,
            residual,
            scale,
            charge,
            conductance,
            capacitance,
            limited: false,
        }
    }

    /// Adds `value` to the residual of `row`'s equation.
    pub fn add_residual(&mut self, row: Node, value: f64) {
        if let Some(row) = row {
            self.residual[row.index()] += value;
            self.scale[row.index()] = self.scale[row.index()].max(value.abs());
        }
    }

    /// Adds `value` to the charge of `row`'s equation.
    pub fn add_charge(&mut self, row: Node, value: f64) {
        if let Some(row) = row {
            self.charge[row.index()] += value;
        }
    }

    /// Adds `value` to the conductance at `entry`.
    pub fn add_conductance(&mut self, entry: Option<Entry>, value: f64) {
        if let Some(entry) = entry {
            self.conductance[self.pattern.slot(entry)] += value;
        }
    }

    /// Adds `value` to the capacitance at `entry`.
    pub fn add_capacitance(&mut self, entry: Option<Entry>, value: f64) {
        if let Some(entry) = entry {
            self.capacitance[self.pattern.slot(entry)] += value;
        }
    }

    /// Records that a device evaluated at a limited voltage rather than the one it was given,
    /// so the solution cannot have converged yet.
    pub fn mark_limited(&mut self) {
        self.limited = true;
    }

    /// Whether any device limited its voltage during this evaluation.
    #[must_use]
    pub fn is_limited(&self) -> bool {
        self.limited
    }
}
