#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::junction::Temperatures;
use crate::kernel::parts::SMALL_RESISTANCE;
use crate::kernel::parts::{CurrentBranch, Pair};
use crate::noise::Flicker;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, NoiseSource, ParameterError,
    REFERENCE_TEMPERATURE, Real, Stamp,
};

/// The default nominal temperature, at which a passive device's value was measured.
fn reference_temperature<R: Real>() -> R {
    R::constant(REFERENCE_TEMPERATURE)
}

/// The local indices of a two-terminal element's terminals, `a` and `b`.
const TERMINALS: (usize, usize) = (0, 1);

/// A resistor between terminals `a` and `b`, whose resistance drifts with temperature and
/// changes with its voltage.
///
/// With `v` from `a` to `b`, its resistance is `r · (1 + vc1 · v + vc2 · v²)`, `r` its
/// resistance at that temperature; where the coefficients bend it below zero, it is
/// negative there. Below [`SMALL_RESISTANCE`], its current is an unknown labelled
/// `current`. Beside its thermal noise, its current `I` makes flicker noise of
/// `KF · |I|^AF / f` A²/Hz across it, as ngspice's resistor does.
#[derive(Debug, Clone, Copy)]
pub struct Resistor;

/// How a resistor adds its current to its terminals' equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResistorShape(Form);

/// How a resistor adds its current to its terminals' equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Form {
    /// As a conductance between them.
    Conductance(Pair),
    /// As an unknown of its own, held by Ohm's law `v − r · f(v) · i = 0`, with the entry
    /// at (current, current).
    Current(CurrentBranch, usize),
}

/// A resistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedResistor<R: Real> {
    resistance: R,
    coefficients: VoltageCoefficients<R>,
    flicker: Flicker,
    temperature: f64,
}

/// How a passive device's value changes with its voltage `v`: by `1 + vc1 · v + vc2 · v²`.
#[derive(Debug, Clone, Copy)]
struct VoltageCoefficients<R: Real> {
    linear: R,
    quadratic: R,
}

impl<R: Real> VoltageCoefficients<R> {
    /// The factor at `voltage`, and its derivative with respect to the voltage.
    fn factor(self, voltage: f64) -> (R, R) {
        (
            self.linear * voltage + self.quadratic * (voltage * voltage) + 1.0,
            self.linear + self.quadratic * (2.0 * voltage),
        )
    }

    /// Whether the factor is 1 at every voltage: whether both coefficients are zero.
    fn constant(self) -> bool {
        self.linear.value() == 0.0 && self.quadratic.value() == 0.0
    }

    /// The integral of the factor from 0 to `voltage`.
    fn integral(self, voltage: f64) -> R {
        let squared = voltage * voltage;
        self.linear * (squared / 2.0) + self.quadratic * (squared * voltage / 3.0) + voltage
    }
}

/// The parameters of a [`Resistor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct ResistorParameters<R: Real = f64> {
    /// The resistance, in ohms, at the nominal temperature; positive.
    pub resistance: R,
    /// The linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    pub quadratic_temperature_coefficient: R,
    /// The exponential temperature coefficient `tce`, in percent per kelvin: given, the
    /// resistance is multiplied by `1.01^(tce · ΔT)` instead of the polynomial.
    #[serde(default)]
    pub exponential_temperature_coefficient: Option<R>,
    /// The temperature the resistance was measured at, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature::<R>")]
    pub nominal_temperature: R,
    /// The linear voltage coefficient `vc1`, per volt; default 0.
    #[serde(default)]
    pub voltage_coefficient: R,
    /// The quadratic voltage coefficient `vc2`, per volt squared; default 0.
    #[serde(default)]
    pub quadratic_voltage_coefficient: R,
    /// The flicker noise coefficient, `KF`; default 0, no flicker noise.
    #[serde(default)]
    pub flicker_coefficient: R,
    /// The power of the current in the flicker noise, `AF`; default 1.
    #[serde(default = "one::<R>")]
    pub flicker_exponent: R,
}

/// The default of an exponent that leaves its base as it is.
fn one<R: Real>() -> R {
    R::constant(1.0)
}

impl<R: Real> ResistorParameters<R> {
    /// The resistance at `temperature`, in kelvins, once each parameter it reads is
    /// checked.
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

/// A heated resistor keeps the form its resistance at the ambient chose: a current of its
/// own stays one however hot it runs.
impl Kernel for Resistor {
    const NAME: &'static str = "resistor";
    type Parameters<R: Real> = ResistorParameters<R>;
    type Shape = ResistorShape;
    type Prepared<R: Real> = PreparedResistor<R>;
    type Memory = ();

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

    /// Linear without voltage coefficients: its conductance, or its resistance where its
    /// current is an unknown, is then the same at every voltage.
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
                let voltage = pair.voltage(at);
                let (factor, slope) = prepared.coefficients.factor(voltage);
                // i = v / (r · f(v)), so di/dv = (f − v · f′) / (r · f²).
                let conductance = (prepared.resistance * factor).recip();
                pair.add_current(stamp, conductance * voltage);
                pair.add_conductance(stamp, conductance * (factor - slope * voltage) / factor);
            }
            Form::Current(branch, entry) => {
                let (voltage, current) = (branch.voltage(at), branch.current(at));
                let (factor, slope) = prepared.coefficients.factor(voltage);
                let resistance = prepared.resistance * factor;
                let residual = R::constant(voltage) - resistance * current;
                let along = R::constant(1.0) - prepared.resistance * slope * current;
                branch.add_sloped(at, stamp, residual, along);
                stamp.add_conductance(*entry, -resistance);
            }
        }
    }

    /// The thermal noise of the resistance at its voltage, `r · f(v)`: a current of
    /// `4kT / (r · f(v))` across it, or a voltage of `4kT · r · f(v)` in series with it
    /// where its current is an unknown; and the flicker noise of its current, across it.
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
                let (factor, _) = prepared.coefficients.factor(voltage);
                let conductance = (prepared.resistance * factor).value().recip();
                (
                    NoiseSource::thermal("thermal", pair.equations(), conductance, temperature),
                    conductance * voltage,
                )
            }
            Form::Current(branch, _) => {
                let (factor, _) = prepared.coefficients.factor(branch.voltage(at));
                let resistance = (prepared.resistance * factor).value();
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

/// A capacitor between terminals `a` and `b`, whose capacitance drifts with temperature
/// and changes with its voltage.
///
/// With `v` from `a` to `b`, its capacitance is `c · (1 + vc1 · v + vc2 · v²)`, `c` its
/// capacitance at that temperature, and its charge that capacitance's integral from 0 V.
#[derive(Debug, Clone, Copy)]
pub struct Capacitor;

/// Where a capacitor's contributions go: across its terminals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacitorShape(Pair);

/// A capacitor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedCapacitor<R: Real> {
    capacitance: R,
    coefficients: VoltageCoefficients<R>,
}

/// The parameters of a [`Capacitor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CapacitorParameters<R: Real = f64> {
    /// The capacitance, in farads, at the nominal temperature; zero or positive.
    pub capacitance: R,
    /// The linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    pub quadratic_temperature_coefficient: R,
    /// The temperature the capacitance was measured at, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature::<R>")]
    pub nominal_temperature: R,
    /// The linear voltage coefficient `vc1`, per volt; default 0.
    #[serde(default)]
    pub voltage_coefficient: R,
    /// The quadratic voltage coefficient `vc2`, per volt squared; default 0.
    #[serde(default)]
    pub quadratic_voltage_coefficient: R,
}

impl Kernel for Capacitor {
    const NAME: &'static str = "capacitor";
    type Parameters<R: Real> = CapacitorParameters<R>;
    type Shape = CapacitorShape;
    type Prepared<R: Real> = PreparedCapacitor<R>;
    type Memory = ();

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

    /// Linear without voltage coefficients: its capacitance is then the same at every
    /// voltage.
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
        let voltage = pair.voltage(at);
        let (factor, _) = prepared.coefficients.factor(voltage);
        pair.add_charge(
            stamp,
            prepared.capacitance * prepared.coefficients.integral(voltage),
        );
        pair.add_capacitance(stamp, prepared.capacitance * factor);
    }
}

/// A linear inductor between terminals `a` and `b`, whose inductance drifts with
/// temperature.
///
/// Its current, flowing from `a` through the inductor to `b`, is an unknown labelled
/// `current`. Its equation is `v_a − v_b − d/dt (L·i) = 0`, so at DC it is a short circuit.
#[derive(Debug, Clone, Copy)]
pub struct Inductor;

/// Where an inductor's contributions go: its branch, and the entry at (current, current).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InductorShape {
    branch: CurrentBranch,
    entry: usize,
}

/// The parameters of an [`Inductor`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct InductorParameters<R: Real = f64> {
    /// The inductance, in henries, at the nominal temperature; positive.
    pub inductance: R,
    /// The linear temperature coefficient `tc1`, per kelvin; default 0.
    #[serde(default)]
    pub temperature_coefficient: R,
    /// The quadratic temperature coefficient `tc2`, per kelvin squared; default 0.
    #[serde(default)]
    pub quadratic_temperature_coefficient: R,
    /// The temperature the inductance was measured at, in kelvins; default 27 °C.
    #[serde(default = "reference_temperature::<R>")]
    pub nominal_temperature: R,
}

impl<R: Real> InductorParameters<R> {
    /// The inductance at `temperature`, in kelvins, once each parameter it reads is
    /// checked.
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
        let voltage = branch.voltage(at);
        branch.add(at, stamp, R::constant(voltage));
        stamp.add_charge(branch.unknown(), -inductance * branch.current(at));
        stamp.add_capacitance(shape.entry, -inductance);
    }
}

/// Inductors whose fluxes link, as a transformer's windings or a choke's.
///
/// Winding `k`, from 1, joins terminals `a{k}` and `b{k}`, its current, flowing from
/// `a{k}` through it to `b{k}`, an unknown labelled `current{k}`. The flux through it is
/// `Σ M_kj · i_j`, where `M_kk` is its own inductance and `M_kj = c · √(L_k · L_j)` for a
/// coupling `c` between windings `k` and `j`, none unless given; its equation is
/// `v_a{k} − v_b{k} − d/dt Σ M_kj · i_j = 0`. A current into `a{k}` and one into `a{j}`
/// add their fluxes when `c` is positive, as SPICE's mutual inductance couples the
/// inductors' first nodes.
#[derive(Debug, Clone, Copy)]
pub struct CoupledInductors;

/// Where coupled inductors' contributions go: each winding's branch, and the entry at
/// (current k, current j), row by row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoupledInductorsShape {
    branches: Vec<CurrentBranch>,
    entries: Vec<usize>,
}

/// The parameters of [`CoupledInductors`].
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct CoupledInductorsParameters<R: Real = f64> {
    /// Each winding, as an inductor's parameters; at least two.
    pub windings: Vec<InductorParameters<R>>,
    /// The couplings between pairs of windings.
    #[serde(default)]
    pub couplings: Vec<Coupling<R>>,
}

/// How closely two windings' fluxes link.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields, bound = "")]
pub struct Coupling<R: Real = f64> {
    /// The two windings, by their numbers from 1.
    pub windings: [usize; 2],
    /// The coupling coefficient, from -1 to 1.
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
            let voltage = branch.voltage(at);
            branch.add(at, stamp, R::constant(voltage));
            let mut flux = R::constant(0.0);
            for (column, other) in shape.branches.iter().enumerate() {
                let index = row * count + column;
                let inductance = inductances.get(index).copied().unwrap_or_default();
                flux += inductance * other.current(at);
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
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match, source,
        thermal,
    };

    #[test]
    fn a_resistor_conducts_by_ohms_law() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1000}"#);
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[2e-3, -2e-3]);
    }

    #[test]
    fn a_heated_resistor_is_the_resistor_at_its_temperature() {
        let model = crate::Thermal::<Resistor>::new();
        let coefficients = r#""temperature_coefficient": 4e-3,
            "quadratic_temperature_coefficient": 1e-5, "voltage_coefficient": -0.05"#;
        for resistance in [1000.0, 1e-4] {
            let every = format!(r#"{{"resistance": {resistance}, {coefficients}}}"#);
            let heated = every.replace('{', r#"{"thermal": true, "#);
            let mut bench =
                Bench::from_model(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
            // A current of its own, where it has one, follows the thermal terminal.
            let current = (resistance < 1.0).then_some(2.0 / resistance);
            for rise in [0.0, 50.0] {
                let mut plain = Bench::at::<Resistor>(&every, REFERENCE_TEMPERATURE + rise);
                let hot = bench.settle(
                    &[3.0, 1.0, rise]
                        .into_iter()
                        .chain(current)
                        .collect::<Vec<_>>(),
                );
                let reference =
                    plain.settle(&[3.0, 1.0].into_iter().chain(current).collect::<Vec<_>>());
                let mut electrical = vec![hot.residual[0], hot.residual[1]];
                electrical.extend(current.map(|_| hot.residual[3]));
                assert_close(&electrical, &reference.residual);
                // It dissipates `v · i`, its current leaving `a` and entering `b`.
                let power = 3.0 * hot.residual[0] + 1.0 * hot.residual[1];
                assert_close(&[-hot.residual[2]], &[power]);
            }
            for rise in [10.0, 80.0] {
                let x: Vec<f64> = [2.5, -0.5, rise].into_iter().chain(current).collect();
                crate::testing::assert_model_jacobians_match(&model, &heated, &x);
                crate::testing::assert_model_parameter_gradients_match(&model, &heated, &x, None);
            }
        }
    }

    #[test]
    fn a_small_resistor_carries_its_current_as_an_unknown() {
        let mut bench = Bench::new::<Resistor>(r#"{"resistance": 1e-6}"#);
        assert_eq!(bench.unknowns(), ["n0", "n1", "device:current"]);
        let output = bench.evaluate(&[3.0, 1.0, 2e6]);
        assert_close(&output.residual, &[2e6, -2e6, 0.0]);
        assert_jacobians_match::<Resistor>(r#"{"resistance": 0.5}"#, &[1.0, -0.5, 3.0]);
        let coefficients = r#""voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02"#;
        assert_jacobians_match::<Resistor>(
            &format!(r#"{{"resistance": 0.5, {coefficients}}}"#),
            &[1.0, -0.5, 2.5],
        );
        assert_parameter_gradients_match::<Resistor>(
            &format!(r#"{{"resistance": 0.5, "temperature_coefficient": 1e-3, {coefficients}}}"#),
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
        let mut bench =
            Bench::new::<Resistor>(&format!(r#"{{"resistance": 1000, {coefficients}}}"#));
        let current = 2.0 / (1000.0 * (1.0 + 0.1 * 2.0 + 0.01 * 4.0));
        assert_close(&bench.evaluate(&[3.0, 1.0]).residual, &[current, -current]);
        let mut bench =
            Bench::new::<Capacitor>(&format!(r#"{{"capacitance": 1e-6, {coefficients}}}"#));
        let charge = 1e-6 * (2.0 + 0.1 * 4.0 / 2.0 + 0.01 * 8.0 / 3.0);
        let output = bench.evaluate(&[2.0, 0.0]);
        assert_close(&output.charge, &[charge, -charge]);
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
        assert_jacobians_match::<CoupledInductors>(TRANSFORMER, &[1.0, 0.5, -2.0, 0.0, 0.3, -0.1]);
        assert_parameter_gradients_match::<CoupledInductors>(
            TRANSFORMER,
            &[1.0, 0.5, -2.0, 0.0, 0.3, -0.1],
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
    fn jacobians_match_finite_differences() {
        let coefficients = r#""voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02"#;
        assert_jacobians_match::<Resistor>(r#"{"resistance": 50}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Resistor>(
            &format!(r#"{{"resistance": 50, {coefficients}}}"#),
            &[1.0, -0.5],
        );
        assert_jacobians_match::<Capacitor>(r#"{"capacitance": 2e-9}"#, &[1.0, -0.5]);
        assert_jacobians_match::<Capacitor>(
            &format!(r#"{{"capacitance": 2e-9, {coefficients}}}"#),
            &[1.0, -0.5],
        );
        assert_jacobians_match::<Inductor>(r#"{"inductance": 1e-3}"#, &[1.0, -0.5, 0.25]);
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let drifting = r#""temperature_coefficient": 1e-3,
            "quadratic_temperature_coefficient": 1e-5, "nominal_temperature": 290"#;
        let nonlinear = format!(
            r#"{drifting}, "voltage_coefficient": -0.05, "quadratic_voltage_coefficient": 0.02"#
        );
        assert_parameter_gradients_match::<Resistor>(
            &format!(r#"{{"resistance": 50, {nonlinear}}}"#),
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Resistor>(
            r#"{"resistance": 50, "exponential_temperature_coefficient": 0.3}"#,
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Capacitor>(
            &format!(r#"{{"capacitance": 2e-9, {nonlinear}}}"#),
            &[1.0, -0.5],
        );
        assert_parameter_gradients_match::<Inductor>(
            &format!(r#"{{"inductance": 1e-3, {drifting}}}"#),
            &[1.0, -0.5, 0.25],
        );
    }

    #[test]
    fn rejects_non_positive_values() {
        let error = Bench::try_new::<Resistor>(r#"{"resistance": 0}"#).unwrap_err();
        assert_eq!(error.to_string(), "resistance must be positive, not 0.0");
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
        // A small one's current is its own unknown, and its flicker noise still across it.
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
