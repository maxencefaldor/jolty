use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::dimension::{
    CAPACITANCE, INDUCTANCE, PER_KELVIN, PER_KELVIN_SQUARED, PER_VOLT, PER_VOLT_SQUARED,
    RESISTANCE, TEMPERATURE,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::Temperatures;
use crate::kernel::parts::SMALL_RESISTANCE;
use crate::kernel::parts::{CurrentBranch, Pair};
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Lift, Local, NoiseSource, ParameterError,
    ParameterKind, ParameterMetadata, REFERENCE_TEMPERATURE, Real, Stamp,
};

/// Returns the default nominal temperature at which passive parameters are specified.
fn reference_temperature<R: Real>() -> R {
    R::constant(REFERENCE_TEMPERATURE)
}

/// Local indices of a two-terminal element's terminals, `a` and `b`.
const TERMINALS: (usize, usize) = (0, 1);

/// Two-terminal resistor between `a` and `b` with temperature drift, voltage nonlinearity,
/// and thermal plus flicker noise.
///
/// With `v = v(a) - v(b)`, resistance equals `r · (1 + vc1 · v + vc2 · v²)`, where `r` is
/// the temperature-scaled resistance (allowing negative values where coefficients bend it
/// below zero). Resistances below [`SMALL_RESISTANCE`] introduce an explicit current unknown
/// labeled `current`. In addition to thermal noise, branch current `I` produces flicker
/// noise `KF · |I|^AF / f` A²/Hz across the terminals, matching ngspice.
#[derive(Debug, Clone, Copy)]
pub struct Resistor;

/// Layout shape selecting how a [`Resistor`] stamps its current into terminal equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResistorShape(Form);

/// Internal stamping form for a [`Resistor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Direct conductance stamped across the terminal [`Pair`].
    Conductance(Pair),
    /// Explicit branch current unknown governed by `v - r · f(v) · i = 0`, with diagonal
    /// `(current, current)` entry.
    Current(CurrentBranch, usize),
}

/// Prepared resistor parameters at operating temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedResistor<R: Real> {
    resistance: R,
    coefficients: VoltageCoefficients<R>,
    flicker: Flicker,
    temperature: f64,
}

/// Quadratic voltage dependence `1 + vc1 · v + vc2 · v²` applied to a passive element.
#[derive(Debug, Clone, Copy)]
struct VoltageCoefficients<R: Real> {
    linear: R,
    quadratic: R,
}

impl<R: Real> VoltageCoefficients<R> {
    /// Evaluates the polynomial scaling factor `1 + vc1 · v + vc2 · v²` at `voltage`.
    fn factor<const N: usize>(self, voltage: Dual<R, N>) -> Dual<R, N> {
        voltage.scale(self.linear) + (voltage * voltage).scale(self.quadratic) + 1.0
    }

    /// Returns `true` when both voltage coefficients are zero, making the factor identically
    /// 1.
    fn constant(self) -> bool {
        self.linear.value() == 0.0 && self.quadratic.value() == 0.0
    }

    /// Computes the definite integral of the voltage factor from 0 to `voltage`.
    fn integral<const N: usize>(self, voltage: Dual<R, N>) -> Dual<R, N> {
        let squared = voltage * voltage;
        (squared / 2.0).scale(self.linear)
            + (squared * voltage / 3.0).scale(self.quadratic)
            + voltage
    }
}

/// Parameters for [`Resistor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct ResistorParameters<R: Real = f64> {
    /// Resistance at nominal temperature, in ohms; positive.
    #[dimension(RESISTANCE)]
    pub resistance: R,
    /// Linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    #[dimension(PER_KELVIN)]
    pub temperature_coefficient: R,
    /// Quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    #[dimension(PER_KELVIN_SQUARED)]
    pub quadratic_temperature_coefficient: R,
    /// Exponential temperature coefficient `tce`, in percent per kelvin; when present,
    /// scales resistance by `1.01^(tce · ΔT)` instead of the polynomial.
    #[serde(default)]
    pub exponential_temperature_coefficient: Option<R>,
    /// Nominal measurement temperature, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature::<R>")]
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// Linear voltage coefficient `vc1`, per volt; default 0.
    #[serde(default)]
    #[dimension(PER_VOLT)]
    pub voltage_coefficient: R,
    /// Quadratic voltage coefficient `vc2`, per volt squared; default 0.
    #[serde(default)]
    #[dimension(PER_VOLT_SQUARED)]
    pub quadratic_voltage_coefficient: R,
    /// Flicker noise coefficient `KF`; default 0 (no flicker noise).
    #[serde(default)]
    pub flicker_coefficient: R,
    /// Flicker noise current exponent `AF`; default 1.
    #[serde(default = "one::<R>")]
    pub flicker_exponent: R,
}

/// Returns the default unit exponent (`1.0`).
fn one<R: Real>() -> R {
    R::constant(1.0)
}

impl<R: Real> ResistorParameters<R> {
    /// Validates parameters and computes the temperature-scaled resistance at `temperature`,
    /// in kelvins.
    fn resistance_at(&self, temperature: R) -> Result<R, ParameterError> {
        let resistance = ParameterError::positive("resistance", self.resistance)?;
        let temperatures = Temperatures::new(temperature, self.nominal_temperature)?;
        let (linear, quadratic) = (
            self.temperature_coefficient,
            self.quadratic_temperature_coefficient,
        );
        let factor = match self.exponential_temperature_coefficient {
            None => temperatures.polynomial(linear, quadratic),
            Some(exponential) if linear.value() == 0.0 && quadratic.value() == 0.0 => {
                R::constant(1.01).pow(exponential * temperatures.difference())
            }
            Some(_) => {
                return Err(ParameterError(
                    "an exponential temperature coefficient excludes the linear and \
                     quadratic ones"
                        .to_owned(),
                ));
            }
        };
        if factor.value() == 0.0 {
            return Err(ParameterError(format!(
                "the temperature coefficients scale the resistance to zero at {:?} K",
                temperatures.actual.value()
            )));
        }
        Ok(resistance * factor)
    }
}

/// Self-heated resistors retain the stamping form chosen at ambient temperature regardless
/// of operating temperature.
impl Kernel for Resistor {
    const NAME: &'static str = "resistor";
    type Parameters<R: Real> = ResistorParameters<R>;
    type Shape = ResistorShape;
    type Prepared<R: Real> = PreparedResistor<R>;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        ResistorParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &ResistorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Conductive)])
    }

    fn prepare<R: Real>(
        parameters: &ResistorParameters<R>,
        temperature: R,
    ) -> Result<PreparedResistor<R>, ParameterError> {
        let resistance = parameters.resistance_at(temperature)?;
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?;
        Ok(PreparedResistor {
            resistance,
            coefficients: VoltageCoefficients {
                linear: parameters.voltage_coefficient,
                quadratic: parameters.quadratic_voltage_coefficient,
            },
            flicker,
            temperature: temperature.value(),
        })
    }

    fn layout<R: Real>(
        _: &ResistorParameters<R>,
        prepared: &PreparedResistor<R>,
        layout: &mut Layout,
    ) -> Result<ResistorShape, ParameterError> {
        let (a, b) = TERMINALS;
        Ok(ResistorShape(
            if prepared.resistance.value().abs() < SMALL_RESISTANCE {
                let branch = CurrentBranch::new(layout, "current", [a, b]);
                let row = branch.unknown();
                Form::Current(branch, layout.entry(row, row))
            } else {
                Form::Conductance(Pair::new(layout, a, b))
            },
        ))
    }

    /// Returns `true` when voltage coefficients are zero so conductance or series resistance
    /// is constant across all voltages.
    fn linear(_: &ResistorShape, prepared: &PreparedResistor<f64>) -> bool {
        prepared.coefficients.constant()
    }

    fn evaluate<R: Real>(
        shape: &ResistorShape,
        prepared: &PreparedResistor<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        match &shape.0 {
            Form::Conductance(pair) => {
                // i = v / (r · f(v)).
                let voltage = Dual::variable(pair.across(at, pair.voltage(at)));
                let resistance = prepared
                    .coefficients
                    .factor(voltage)
                    .scale(prepared.resistance);
                let current = resistance.recip() * voltage;
                let [conductance] = current.gradient;
                pair.add_current(stamp, current.value);
                pair.add_conductance(stamp, conductance);
            }
            Form::Current(branch, entry) => {
                // v - r · f(v) · i = 0.
                let voltage = Dual::variable(branch.voltage(at));
                let resistance = prepared
                    .coefficients
                    .factor(voltage)
                    .scale(prepared.resistance);
                let residual = voltage - resistance.scale(branch.current(at));
                let [along] = residual.gradient;
                branch.add_sloped(at, stamp, residual.value, along);
                stamp.add_conductance(*entry, -resistance.value);
            }
        }
    }

    /// Appends thermal noise for effective resistance `r · f(v)` and flicker noise for the
    /// branch current across the resistor terminals.
    ///
    /// Conductance-form resistors inject a current noise density `4kT / (r · f(v))`;
    /// current-branch resistors inject a series voltage noise density `4kT · r · f(v)`.
    fn noise<R: Real>(
        shape: &ResistorShape,
        prepared: &PreparedResistor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let temperature = prepared.temperature;
        let (thermal, current) = match &shape.0 {
            Form::Conductance(pair) => {
                let voltage = pair.voltage(at);
                let factor = prepared.coefficients.factor(Dual::<R, 0>::number(voltage));
                let conductance = (prepared.resistance * factor.value).value().recip();
                (
                    NoiseSource::thermal("thermal", pair.equations(), conductance, temperature),
                    conductance * voltage,
                )
            }
            Form::Current(branch, _) => {
                let voltage = Dual::<R, 0>::number(branch.voltage(at));
                let factor = prepared.coefficients.factor(voltage);
                let resistance = (prepared.resistance * factor.value).value();
                (
                    NoiseSource::thermal_in_series(
                        "thermal",
                        crate::UnknownIndex::new(branch.unknown()),
                        resistance,
                        temperature,
                    ),
                    branch.current(at),
                )
            }
        };
        sources.push(thermal);
        let (a, b) = TERMINALS;
        let terminals = [
            Some(crate::UnknownIndex::new(a)),
            Some(crate::UnknownIndex::new(b)),
        ];
        prepared.flicker.add(terminals, current, sources);
    }
}

/// Two-terminal capacitor between `a` and `b` with temperature drift and voltage
/// nonlinearity.
///
/// With `v = v(a) - v(b)`, incremental capacitance equals `c · (1 + vc1 · v + vc2 · v²)`,
/// where `c` is the temperature-scaled capacitance, and stored charge equals the integral of
/// capacitance from `0 V` to `v`.
#[derive(Debug, Clone, Copy)]
pub struct Capacitor;

/// Layout shape for [`Capacitor`] storing the terminal `Pair`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacitorShape(Pair);

/// Prepared capacitor parameters at operating temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedCapacitor<R: Real> {
    capacitance: R,
    coefficients: VoltageCoefficients<R>,
}

/// Parameters for [`Capacitor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct CapacitorParameters<R: Real = f64> {
    /// Capacitance at nominal temperature, in farads; non-negative.
    #[dimension(CAPACITANCE)]
    pub capacitance: R,
    /// Linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    #[dimension(PER_KELVIN)]
    pub temperature_coefficient: R,
    /// Quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    #[dimension(PER_KELVIN_SQUARED)]
    pub quadratic_temperature_coefficient: R,
    /// Nominal measurement temperature, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature::<R>")]
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
    /// Linear voltage coefficient `vc1`, per volt; default 0.
    #[serde(default)]
    #[dimension(PER_VOLT)]
    pub voltage_coefficient: R,
    /// Quadratic voltage coefficient `vc2`, per volt squared; default 0.
    #[serde(default)]
    #[dimension(PER_VOLT_SQUARED)]
    pub quadratic_voltage_coefficient: R,
}

impl Kernel for Capacitor {
    const NAME: &'static str = "capacitor";
    type Parameters<R: Real> = CapacitorParameters<R>;
    type Shape = CapacitorShape;
    type Prepared<R: Real> = PreparedCapacitor<R>;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        CapacitorParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &CapacitorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Capacitive)])
    }

    fn prepare<R: Real>(
        parameters: &CapacitorParameters<R>,
        temperature: R,
    ) -> Result<PreparedCapacitor<R>, ParameterError> {
        let capacitance = ParameterError::non_negative("capacitance", parameters.capacitance)?
            * Temperatures::new(temperature, parameters.nominal_temperature)?.polynomial(
                parameters.temperature_coefficient,
                parameters.quadratic_temperature_coefficient,
            );
        Ok(PreparedCapacitor {
            capacitance,
            coefficients: VoltageCoefficients {
                linear: parameters.voltage_coefficient,
                quadratic: parameters.quadratic_voltage_coefficient,
            },
        })
    }

    fn layout<R: Real>(
        _: &CapacitorParameters<R>,
        _: &PreparedCapacitor<R>,
        layout: &mut Layout,
    ) -> Result<CapacitorShape, ParameterError> {
        let (a, b) = TERMINALS;
        Ok(CapacitorShape(Pair::new(layout, a, b)))
    }

    /// Returns `true` when voltage coefficients are zero so capacitance is constant across
    /// all voltages.
    fn linear(_: &CapacitorShape, prepared: &PreparedCapacitor<f64>) -> bool {
        prepared.coefficients.constant()
    }

    fn evaluate<R: Real>(
        CapacitorShape(pair): &CapacitorShape,
        prepared: &PreparedCapacitor<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let voltage = Dual::variable(pair.across(at, pair.voltage(at)));
        let charge = prepared
            .coefficients
            .integral(voltage)
            .scale(prepared.capacitance);
        let [capacitance] = charge.gradient;
        pair.add_charge(stamp, charge.value);
        pair.add_capacitance(stamp, capacitance);
    }
}

/// Linear inductor between terminals `a` and `b` with temperature drift.
///
/// Branch current from `a` to `b` is an explicit unknown labeled `current`. The branch
/// equation `v_a - v_b - d/dt (L·i) = 0` acts as a short circuit at DC.
#[derive(Debug, Clone, Copy)]
pub struct Inductor;

/// Layout shape for [`Inductor`] storing the branch and diagonal `(current, current)` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InductorShape {
    branch: CurrentBranch,
    entry: usize,
}

/// Parameters for [`Inductor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift, ParameterMetadata)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct InductorParameters<R: Real = f64> {
    /// Inductance at nominal temperature, in henries; positive.
    #[dimension(INDUCTANCE)]
    pub inductance: R,
    /// Linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    #[dimension(PER_KELVIN)]
    pub temperature_coefficient: R,
    /// Quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    #[dimension(PER_KELVIN_SQUARED)]
    pub quadratic_temperature_coefficient: R,
    /// Nominal measurement temperature, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature::<R>")]
    #[dimension(TEMPERATURE)]
    pub nominal_temperature: R,
}

impl<R: Real> InductorParameters<R> {
    /// Validates parameters and computes temperature-scaled inductance at `temperature`, in
    /// kelvins.
    fn inductance_at(&self, temperature: R) -> Result<R, ParameterError> {
        Ok(ParameterError::positive("inductance", self.inductance)?
            * Temperatures::new(temperature, self.nominal_temperature)?.polynomial(
                self.temperature_coefficient,
                self.quadratic_temperature_coefficient,
            ))
    }
}

impl Kernel for Inductor {
    const NAME: &'static str = "inductor";
    type Parameters<R: Real> = InductorParameters<R>;
    type Shape = InductorShape;
    type Prepared<R: Real> = R;
    type Memory = ();

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        InductorParameters::<f64>::PARAMETER_KINDS
    }

    fn interface<R: Real>(_: &InductorParameters<R>) -> Interface {
        Interface::new(&["a", "b"], &[Branch::new(0, 1, BranchKind::Inductive)])
    }

    fn prepare<R: Real>(
        parameters: &InductorParameters<R>,
        temperature: R,
    ) -> Result<R, ParameterError> {
        parameters.inductance_at(temperature)
    }

    fn layout<R: Real>(
        _: &InductorParameters<R>,
        _: &R,
        layout: &mut Layout,
    ) -> Result<InductorShape, ParameterError> {
        let (a, b) = TERMINALS;
        let branch = CurrentBranch::new(layout, "current", [a, b]);
        let row = branch.unknown();
        Ok(InductorShape {
            branch,
            entry: layout.entry(row, row),
        })
    }

    fn linear(_: &InductorShape, _: &f64) -> bool {
        true
    }

    fn evaluate<R: Real>(
        shape: &InductorShape,
        &inductance: &R,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // v = L di/dt: the voltage, and the flux `L i` as the equation's charge, negated.
        let branch = &shape.branch;
        branch.add(at, stamp, branch.voltage(at));
        stamp.add_charge(branch.unknown(), -inductance * branch.current::<R>(at));
        stamp.add_capacitance(shape.entry, -inductance);
    }
}

/// Magnetically coupled inductors such as transformer or choke windings.
///
/// Winding `k` (1-indexed) spans `a{k}` to `b{k}` with branch-current unknown `current{k}`
/// and equation `v_a{k} - v_b{k} - d/dt Σ M_kj · i_j = 0`. Diagonal entry `M_kk` is the
/// self-inductance `L_k`, and off-diagonal entry `M_kj = c · √(L_k · L_j)` for a coupling
/// `c` (zero if unspecified). Positive `c` matches SPICE dot convention between `a{k}` and
/// `a{j}`.
#[derive(Debug, Clone, Copy)]
pub struct CoupledInductors;

/// Layout shape for [`CoupledInductors`] storing each winding branch and the row-major
/// `(current k, current j)` mutual-inductance entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoupledInductorsShape {
    branches: Vec<CurrentBranch>,
    entries: Vec<usize>,
}

/// Parameters for [`CoupledInductors`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct CoupledInductorsParameters<R: Real = f64> {
    /// Winding specifications as [`InductorParameters`]; at least two windings.
    pub windings: Vec<InductorParameters<R>>,
    /// Magnetic couplings between pairs of windings.
    #[serde(default)]
    pub couplings: Vec<Coupling<R>>,
}

/// Magnetic coupling specification between two windings.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(deny_unknown_fields)]
#[schemars(bound = "R: Serialize + JsonSchema")]
pub struct Coupling<R: Real = f64> {
    /// Pair of 1-based winding indices.
    pub windings: [usize; 2],
    /// Coupling coefficient in `[-1, 1]`.
    pub coefficient: R,
}

impl Kernel for CoupledInductors {
    const NAME: &'static str = "coupled_inductors";
    type Parameters<R: Real> = CoupledInductorsParameters<R>;
    type Shape = CoupledInductorsShape;
    /// The inductance matrix, row by row.
    type Prepared<R: Real> = Vec<R>;
    type Memory = ();

    fn interface<R: Real>(parameters: &CoupledInductorsParameters<R>) -> Interface {
        let terminals: Vec<String> = (1..=parameters.windings.len())
            .flat_map(|winding| [format!("a{winding}"), format!("b{winding}")])
            .collect();
        let borrowed: Vec<&str> = terminals.iter().map(String::as_str).collect();
        let branches: Vec<Branch> = (0..parameters.windings.len())
            .map(|winding| Branch::new(2 * winding, 2 * winding + 1, BranchKind::Inductive))
            .collect();
        Interface::new(&borrowed, &branches)
    }

    fn prepare<R: Real>(
        parameters: &CoupledInductorsParameters<R>,
        temperature: R,
    ) -> Result<Vec<R>, ParameterError> {
        let count = parameters.windings.len();
        if count < 2 {
            return Err(ParameterError(format!(
                "coupled inductors have at least two windings, not {count}"
            )));
        }
        let own = parameters
            .windings
            .iter()
            .map(|winding| winding.inductance_at(temperature))
            .collect::<Result<Vec<R>, _>>()?;
        let mut inductances = vec![R::constant(0.0); count * count];
        for (index, &inductance) in own.iter().enumerate() {
            if let Some(diagonal) = inductances.get_mut(index * count + index) {
                *diagonal = inductance;
            }
        }
        let mut coupled = vec![false; count * count];
        for coupling in &parameters.couplings {
            let [first, second] = coupling.windings;
            let numbered = |winding: usize| (1..=count).contains(&winding);
            if !numbered(first) || !numbered(second) || first == second {
                return Err(ParameterError(format!(
                    "a coupling joins two distinct windings from 1 to {count}, not {first} \
                     and {second}"
                )));
            }
            let (row, column) = (first - 1, second - 1);
            if coupled.get(row * count + column).copied().unwrap_or(false) {
                return Err(ParameterError(format!(
                    "windings {first} and {second} are coupled twice"
                )));
            }
            let coefficient = coupling.coefficient;
            if !(-1.0..=1.0).contains(&coefficient.value()) {
                return Err(ParameterError(format!(
                    "a coupling coefficient lies from -1 to 1, not {:?}",
                    coefficient.value()
                )));
            }
            let (Some(&along), Some(&across)) = (own.get(row), own.get(column)) else {
                continue;
            };
            let mutual = coefficient * (along * across).sqrt();
            for (one, other) in [(row, column), (column, row)] {
                if let Some(slot) = inductances.get_mut(one * count + other) {
                    *slot = mutual;
                }
                if let Some(slot) = coupled.get_mut(one * count + other) {
                    *slot = true;
                }
            }
        }
        Ok(inductances)
    }

    fn layout<R: Real>(
        parameters: &CoupledInductorsParameters<R>,
        _: &Vec<R>,
        layout: &mut Layout,
    ) -> Result<CoupledInductorsShape, ParameterError> {
        let branches: Vec<CurrentBranch> = (0..parameters.windings.len())
            .map(|winding| {
                let label = format!("current{}", winding + 1);
                CurrentBranch::new(layout, label, [2 * winding, 2 * winding + 1])
            })
            .collect();
        let entries = branches
            .iter()
            .flat_map(|row| {
                branches
                    .iter()
                    .map(move |column| (row.unknown(), column.unknown()))
            })
            .map(|(row, column)| layout.entry(row, column))
            .collect();
        Ok(CoupledInductorsShape { branches, entries })
    }

    fn linear(_: &CoupledInductorsShape, _: &Vec<f64>) -> bool {
        true
    }

    fn evaluate<R: Real>(
        shape: &CoupledInductorsShape,
        inductances: &Vec<R>,
        (): &mut (),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        // v_k = d/dt Σ M_kj i_j: each winding's voltage, and its flux as its equation's
        // charge, negated.
        let count = shape.branches.len();
        for (row, branch) in shape.branches.iter().enumerate() {
            branch.add(at, stamp, branch.voltage(at));
            let mut flux = R::constant(0.0);
            for (column, other) in shape.branches.iter().enumerate() {
                let index = row * count + column;
                let inductance = inductances.get(index).copied().unwrap_or_default();
                flux += inductance * other.current::<R>(at);
                let entry = shape.entries.get(index).copied().unwrap_or(usize::MAX);
                stamp.add_capacitance(entry, -inductance);
            }
            stamp.add_charge(branch.unknown(), -flux);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UnknownIndex;
    use crate::testing::{Bench, assert_close, assert_jacobians_match, source, thermal};

    #[test]
    fn a_resistor_conducts_by_ohms_law() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#);
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[2e-3, -2e-3]);
    }

    #[test]
    fn a_small_resistor_carries_its_current_as_an_unknown() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1e-6}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:current"]);
        let output = bench.evaluate(&[3.0, 1.0, 2e6]);
        assert_close(&output.residual, &[2e6, -2e6, 0.0]);
        assert_jacobians_match::<Resistor>(
            r#"{"resistance": 0.5, "voltage_coefficient": -0.05,
                "quadratic_voltage_coefficient": 0.02}"#,
            &[1.0, -0.5, 2.5],
        );
        let large = Bench::new::<Resistor>(r#"{"resistance": 1}"#);
        assert_eq!(large.unknowns(), ["n0", "n1"]);
    }

    #[test]
    fn a_capacitor_holds_charge() {
        let mut bench = Bench::new::<Capacitor>(r#"{"capacitance": 1e-6}"#);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[2e-6, -2e-6]);
        assert_close(&output.residual, &[0.0, 0.0]);
    }

    #[test]
    fn values_change_with_their_voltage_coefficients() {
        let coefficients = r#""voltage_coefficient": 0.1, "quadratic_voltage_coefficient": 0.01"#;
        let resistor = format!(r#"{{"resistance": 1000, {coefficients}}}"#);
        let mut bench = Bench::new::<Resistor>(&resistor);
        let current = 2.0 / (1000.0 * (1.0 + 0.1 * 2.0 + 0.01 * 4.0));
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[current, -current]);
        assert_jacobians_match::<Resistor>(&resistor, &[1.0, -0.5]);
        let capacitor = format!(r#"{{"capacitance": 1e-6, {coefficients}}}"#);
        let mut bench = Bench::new::<Capacitor>(&capacitor);
        let charge = 1e-6 * (2.0 + 0.1 * 4.0 / 2.0 + 0.01 * 8.0 / 3.0);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[charge, -charge]);
        assert_jacobians_match::<Capacitor>(&capacitor, &[1.0, -0.5]);
    }

    #[test]
    fn an_inductor_adds_its_current_as_an_unknown() {
        let bench = Bench::new::<Inductor>(r#"{"inductance": 1e-3}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:current"]);
    }

    /// A transformer of 1 mH and 4 mH, coupled by 0.9, the primary drifting.
    const TRANSFORMER: &str = r#"{"windings": [
            {"inductance": 1e-3, "temperature_coefficient": 1e-3, "nominal_temperature": 290},
            {"inductance": 4e-3}],
        "couplings": [{"windings": [1, 2], "coefficient": 0.9}]}"#;

    #[test]
    fn coupled_windings_share_their_flux() {
        let mut bench = Bench::new::<CoupledInductors>(TRANSFORMER);
        assert_eq!(
            bench.unknowns(),
            ["n0", "n1", "n2", "n3", "device:current1", "device:current2"]
        );
        let primary = 1e-3 * (1.0 + 1e-3 * (REFERENCE_TEMPERATURE - 290.0));
        let mutual = 0.9 * (primary * 4e-3_f64).sqrt();
        let output = bench.evaluate(&[1.0, 0.5, -2.0, 0.0, 0.3, -0.1]);
        // Each winding's current into its `a` and out of its `b`, and its voltage.
        assert_close(&output.residual, &[0.3, -0.3, -0.1, 0.1, 0.5, -2.0]);
        let fluxes = [primary * 0.3 + mutual * -0.1, mutual * 0.3 + 4e-3 * -0.1];
        assert_close(
            &output.charge,
            &[0.0, 0.0, 0.0, 0.0, -fluxes[0], -fluxes[1]],
        );
        let interface = CoupledInductors::interface::<f64>(
            &serde_json::from_str(TRANSFORMER).expect("parameters"),
        );
        assert_eq!(interface.terminals, ["a1", "b1", "a2", "b2"]);
    }

    #[test]
    fn coupled_inductors_reject_couplings_they_cannot_make() {
        let error = |parameters: &str| {
            Bench::try_new::<CoupledInductors>(parameters)
                .unwrap_err()
                .to_string()
        };
        let windings = r#""windings": [{"inductance": 1e-3}, {"inductance": 2e-3}]"#;
        assert_eq!(
            error(r#"{"windings": [{"inductance": 1e-3}]}"#),
            "coupled inductors have at least two windings, not 1"
        );
        assert_eq!(
            error(&format!(
                r#"{{{windings}, "couplings": [{{"windings": [1, 3], "coefficient": 0.5}}]}}"#
            )),
            "a coupling joins two distinct windings from 1 to 2, not 1 and 3"
        );
        assert_eq!(
            error(&format!(
                r#"{{{windings}, "couplings": [{{"windings": [1, 2], "coefficient": 0.5}},
                    {{"windings": [2, 1], "coefficient": 0.4}}]}}"#
            )),
            "windings 2 and 1 are coupled twice"
        );
        assert_eq!(
            error(&format!(
                r#"{{{windings}, "couplings": [{{"windings": [1, 2], "coefficient": 1.5}}]}}"#
            )),
            "a coupling coefficient lies from -1 to 1, not 1.5"
        );
    }

    #[test]
    fn values_drift_with_their_temperature_coefficients() {
        let parameters = r#"{"resistance": 1000, "temperature_coefficient": 1e-3,
            "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 300}"#;
        let mut bench = Bench::at::<Resistor>(parameters, 310.0);
        let factor = 1.0 + 1e-3 * 10.0 + 1e-5 * 100.0;
        assert_close(
            &bench.evaluate(&[1.0, 0.0]).residual,
            &[1e-3 / factor, -1e-3 / factor],
        );
        let mut bench = Bench::at::<Resistor>(
            r#"{"resistance": 1000, "exponential_temperature_coefficient": 0.5}"#,
            REFERENCE_TEMPERATURE + 10.0,
        );
        let factor = 1.01_f64.powf(5.0);
        assert_close(
            &bench.evaluate(&[1.0, 0.0]).residual,
            &[1e-3 / factor, -1e-3 / factor],
        );
        let error = Bench::try_at::<Resistor>(
            r#"{"resistance": 1000, "temperature_coefficient": -0.1}"#,
            REFERENCE_TEMPERATURE + 10.0,
        )
        .unwrap_err();
        assert!(
            error
                .0
                .starts_with("the temperature coefficients scale the resistance to zero")
        );
    }

    #[test]
    fn rejects_non_positive_values() {
        let error = Bench::try_new::<Resistor>(r#"{"resistance": 0}"#).unwrap_err();
        assert_eq!(error.to_string(), "resistance is positive, not 0.0");
        let error = Bench::try_new::<Capacitor>(r#"{"capacity": 1}"#).unwrap_err();
        assert!(
            error.to_string().starts_with("unknown field `capacity`"),
            "{error}"
        );
    }

    #[test]
    fn a_resistor_makes_thermal_noise() {
        let bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#);
        let sources = bench.noise(&[3.0, 1.0]);
        let noise = source(&sources, "thermal");
        assert_eq!(
            noise.equations,
            [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(1))]
        );
        assert_close(&[noise.density(1.0)], &[thermal(1e-3)]);
        // A small one's is a voltage in series with its current, and a short makes none.
        let small = Bench::new::<Resistor>(r#"{"resistance": 0.5}"#);
        let sources = small.noise(&[1.0, 0.0, 2.0]);
        let noise = source(&sources, "thermal");
        assert_eq!(noise.equations, [Some(UnknownIndex::new(2)), None]);
        assert_close(&[noise.density(1.0)], &[thermal(0.5)]);
        let bent = Bench::new::<Resistor>(r#"{"resistance": 1000, "voltage_coefficient": 0.1}"#);
        let sources = bent.noise(&[2.0, 0.0]);
        assert_close(
            &[source(&sources, "thermal").density(1.0)],
            &[thermal(1.0 / 1200.0)],
        );
    }

    #[test]
    fn a_resistors_current_makes_flicker_noise_across_it() {
        let flickering = r#"{"resistance": 1000, "flicker_coefficient": 1e-12,
            "flicker_exponent": 2}"#;
        let sources = Bench::new::<Resistor>(flickering).noise(&[3.0, 1.0]);
        let flicker = source(&sources, "flicker");
        assert_eq!(
            flicker.equations,
            [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(1))]
        );
        assert_close(&[flicker.density(10.0)], &[1e-12 * 4e-6 / 10.0]);
        // Explicit current-branch resistors still inject flicker noise across their terminals.
        let small = r#"{"resistance": 0.5, "flicker_coefficient": 1e-12}"#;
        let sources = Bench::new::<Resistor>(small).noise(&[1.0, 0.0, 2.0]);
        let flicker = source(&sources, "flicker");
        assert_eq!(
            flicker.equations,
            [Some(UnknownIndex::new(0)), Some(UnknownIndex::new(1))]
        );
        assert_close(&[flicker.density(1.0)], &[2e-12]);
        assert!(
            Bench::new::<Resistor>(r#"{"resistance": 1}"#)
                .noise(&[1.0, 0.0])
                .iter()
                .all(|noise| noise.name != "flicker")
        );
    }
}
