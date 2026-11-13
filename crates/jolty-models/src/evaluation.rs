use jolty_numerics::{Entry, Pattern};

use crate::{Node, Real};

/// The state a device is evaluated at.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a> {
    x: &'a [f64],
    /// The time, in seconds; zero in every analysis but a transient one.
    pub time: f64,
    /// Whether time passes: true while a transient analysis steps through it, and false at
    /// an operating point and in the analyses about one, in which a state that only time
    /// moves, such as a memristor's, holds the value it starts from.
    pub transient: bool,
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
    /// The state `x` at `time` in a transient analysis or, with `None`, at an operating
    /// point or about one, at time zero, with sources at full scale, `gmin` across
    /// junctions, and steps limited.
    #[must_use]
    pub fn new(x: &'a [f64], time: Option<f64>, gmin: f64) -> Self {
        Self {
            x,
            time: time.unwrap_or(0.0),
            transient: time.is_some(),
            source_scale: 1.0,
            gmin,
            limiting: true,
        }
    }

    /// The same time, scale, gmin and limiting, at the unknowns `x`: the input of a device
    /// that another evaluates within itself, on unknowns of its own.
    #[must_use]
    pub const fn with_values<'b>(&self, x: &'b [f64]) -> Input<'b> {
        Input {
            x,
            time: self.time,
            transient: self.transient,
            source_scale: self.source_scale,
            gmin: self.gmin,
            limiting: self.limiting,
        }
    }

    /// The value of `node`'s unknown, or zero for the reference node.
    #[must_use]
    pub fn value(&self, node: Node) -> f64 {
        node.map_or(0.0, |unknown| self.x[unknown.index()])
    }
}

/// What devices add to the circuit's equations in one evaluation.
///
/// Residuals, their scales and charges hold one value per unknown; conductances and
/// capacitances one per stored position of the pattern. Alongside each residual, `scale`
/// keeps the largest magnitude of a term added to it, against which convergence is judged.
/// The numbers are plain, or carry their derivative with respect to a parameter when the
/// devices compute with [`crate::Gradient`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contributions<R: Real = f64> {
    /// The residual of each unknown's equation, `f(x)`.
    pub residual: Vec<R>,
    /// The largest magnitude of a term of each residual.
    pub scale: Vec<R>,
    /// The charge of each unknown's equation, `q(x)`.
    pub charge: Vec<R>,
    /// `∂f/∂x`, by stored position.
    pub conductance: Vec<R>,
    /// `∂q/∂x`, by stored position.
    pub capacitance: Vec<R>,
}

impl<R: Real> Contributions<R> {
    /// Zeroed contributions over `pattern`.
    #[must_use]
    pub fn new(pattern: &Pattern) -> Self {
        let (unknowns, stored) = (pattern.dimension(), pattern.len());
        Self {
            residual: vec![R::default(); unknowns],
            scale: vec![R::default(); unknowns],
            charge: vec![R::default(); unknowns],
            conductance: vec![R::default(); stored],
            capacitance: vec![R::default(); stored],
        }
    }

    /// Zeroes every contribution and opens them to the devices' additions over `pattern`.
    ///
    /// # Panics
    ///
    /// Panics if the contributions were not made over a pattern of the same size.
    pub fn output<'a>(&'a mut self, pattern: &'a Pattern) -> Output<'a, R> {
        assert!(
            self.residual.len() == pattern.dimension() && self.conductance.len() == pattern.len(),
            "contributions sized for another pattern"
        );
        for values in [
            &mut self.residual,
            &mut self.scale,
            &mut self.charge,
            &mut self.conductance,
            &mut self.capacitance,
        ] {
            values.fill(R::default());
        }
        Output {
            pattern,
            contributions: self,
            limited: false,
        }
    }
}

/// Where devices add their contributions during one evaluation of the circuit: zeroed
/// [`Contributions`], open over the pattern that places their conductances and capacitances.
#[derive(Debug)]
pub struct Output<'a, R: Real = f64> {
    pattern: &'a Pattern,
    contributions: &'a mut Contributions<R>,
    limited: bool,
}

impl<R: Real> Output<'_, R> {
    /// Adds `value` to the residual of `row`'s equation.
    pub fn add_residual(&mut self, row: Node, value: R) {
        if let Some(row) = row {
            let contributions = &mut *self.contributions;
            contributions.residual[row.index()] += value;
            let scale = &mut contributions.scale[row.index()];
            *scale = scale.max(value.abs());
        }
    }

    /// Adds `value` to the residual of `row`'s equation, `scale` the largest magnitude of
    /// the terms it sums, as a device that sums them itself reports them.
    pub fn add_residual_scaled(&mut self, row: Node, value: R, scale: R) {
        if let Some(row) = row {
            let contributions = &mut *self.contributions;
            contributions.residual[row.index()] += value;
            let largest = &mut contributions.scale[row.index()];
            *largest = largest.max(scale.abs());
        }
    }

    /// Adds `value` to the charge of `row`'s equation.
    pub fn add_charge(&mut self, row: Node, value: R) {
        if let Some(row) = row {
            self.contributions.charge[row.index()] += value;
        }
    }

    /// Adds `value` to the conductance at `entry`.
    pub fn add_conductance(&mut self, entry: Option<Entry>, value: R) {
        if let Some(entry) = entry {
            self.contributions.conductance[self.pattern.slot(entry)] += value;
        }
    }

    /// Adds `value` to the capacitance at `entry`.
    pub fn add_capacitance(&mut self, entry: Option<Entry>, value: R) {
        if let Some(entry) = entry {
            self.contributions.capacitance[self.pattern.slot(entry)] += value;
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
