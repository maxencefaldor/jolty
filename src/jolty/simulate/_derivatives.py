# Copyright 2026 Maxence Faldor
"""Exact measure derivatives, requirement slack derivatives, and linear worst case.

Composes engine reverse-mode pullbacks ([`Function`][jolty.simulate.Function]) with
sparse measure and slack cotangents from [`jolty.specs`][jolty.specs] to differentiate
every measure or requirement on an analysis in a single backward pass. Linear
worst-case estimates combine exact nominal measure derivatives with device tolerances
and shared lots from [`jolty.variation`][jolty.variation].
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from jolty import _core, devices, ir, results, specs, variation
from jolty.builder import Component, circuit
from jolty.devices import ParameterValue
from jolty.errors import JoltyError, SimulationError, close_names
from jolty.plan import (
    Ac,
    Analysis,
    DcSweep,
    Noise,
    OperatingPoint,
    Options,
    PeriodicSteadyState,
    Plan,
    Stability,
    Transient,
    differentiated,
)
from jolty.simulate._decode import (
    decode_result,
    encode_analysis,
    engine_pool,
    swept_dimension,
)
from jolty.simulate._function import (
    Differentiable,
    Elaborated,
    Function,
    Parameters,
    contract_real_gradient,
    loop_cotangent_entries,
    noise_cotangent_entries,
)
from jolty.units import DIMENSIONLESS, VOLTAGE, Quantity

type RealGradient = dict[str, dict[str, float]]
"""Real parameter derivative mapping keyed by instance path and parameter name."""

_DIFFERENTIABLE_TYPES = (
    OperatingPoint,
    DcSweep,
    Ac,
    Stability,
    Noise,
    Transient,
    PeriodicSteadyState,
)


@dataclass(frozen=True, slots=True)
class MeasureDerivative:
    """Evaluated measure paired with exact first-order parameter derivatives.

    `derivatives` maps instance path and parameter name to ``d(measure)/d(p)`` in SI
    base units of the measure per SI base unit of the parameter.
    """

    measure: specs.Measure
    reading: specs.Reading
    derivatives: RealGradient


@dataclass(frozen=True, slots=True)
class RequirementDerivative:
    """Requirement verdict paired with exact first-order slack derivatives.

    When the requirement is measured and differentiable at its worst condition,
    `derivatives` maps instance path and parameter name to ``d(slack)/d(p)`` in SI
    base units (positive increases slack). When unmeasured or non-differentiable,
    `derivatives` is empty and `reason` states why.
    """

    requirement: specs.Requirement
    verdict: specs.Verdict
    derivatives: RealGradient = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LinearWorstCase:
    """First-order worst-case slack estimate for one requirement from tolerances.

    Combines exact nominal measure derivatives with independent parameter tolerances
    and shared lots to find the exact extrema of the linearized measure across every
    tolerance extreme.
    """

    requirement: specs.Requirement
    nominal: specs.Reading
    nominal_slack: float
    estimated_value: float
    estimated_slack: float
    status: specs.Status
    bound: Literal["minimum", "maximum"]
    overrides: dict[str, dict[str, ParameterValue]]
    derivatives: RealGradient
    corner: str | None = None


@dataclass(frozen=True, slots=True)
class _PullbackContext:
    """Engine simulator, analysis, options, and parameter list for one backward pass."""

    core: _core.Simulator
    built: ir.Circuit
    analysis: Differentiable
    options: Options
    names: Sequence[tuple[str, str]]


def _encode_differentiated(built: ir.Circuit, analysis: Differentiable) -> str:
    """Encodes a differentiable `analysis` specification as engine JSON."""
    target = None if isinstance(analysis, OperatingPoint) else analysis
    dimension = (
        swept_dimension(built, target) if isinstance(target, DcSweep) else VOLTAGE
    )
    return json.dumps(differentiated(target, dimension), allow_nan=False)


def _settings_json(options: Options) -> str:
    """Encodes solver `options` as engine JSON."""
    return json.dumps(options.to_json(), allow_nan=False)


def _core_metadata(
    core: _core.Simulator,
) -> tuple[tuple[results.Unknown, ...], tuple[str, ...]]:
    """Extracts typed unknowns and logic nets from an engine simulator."""
    unknowns = tuple(results.Unknown(name, nature) for name, nature in core.unknowns)
    return unknowns, tuple(core.nets)


def _run_vjp(
    context: _PullbackContext,
) -> tuple[results.AnalysisResult, _core.Pullback]:
    """Runs one VJP forward solve for `context` and decodes its typed result."""
    encoded = _encode_differentiated(context.built, context.analysis)
    settings = _settings_json(context.options)
    try:
        outcome, pullback = context.core.vjp(
            encoded, list(context.names), settings, engine_pool()
        )
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None
    unknowns, nets = _core_metadata(context.core)
    return decode_result(unknowns, nets, outcome, results.Recording()), pullback


def _run_plain(
    core: _core.Simulator,
    built: ir.Circuit,
    analysis: Analysis,
    options: Options,
) -> results.AnalysisResult:
    """Runs one non-differentiable or forward-only `analysis` on `core`."""
    encoded = json.dumps(encode_analysis(built, frozenset(), analysis), allow_nan=False)
    settings = json.dumps(
        {**options.to_json(), "record": {"instances": []}, "regions": False},
        allow_nan=False,
    )
    try:
        outcome = core.run(encoded, settings, engine_pool())
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None
    unknowns, nets = _core_metadata(core)
    return decode_result(unknowns, nets, outcome, results.Recording())


def _unpack_gradients(
    names: Sequence[tuple[str, str]], matrix: npt.NDArray[np.float64]
) -> tuple[RealGradient, ...]:
    """Converts a `(rows, len(names))` derivative matrix to nested `RealGradient`s."""
    gradients: list[RealGradient] = []
    for row in matrix.tolist():
        gradient: RealGradient = {}
        for (path, parameter), value in zip(names, row, strict=True):
            gradient.setdefault(path, {})[parameter] = float(value)
        gradients.append(gradient)
    return tuple(gradients)


def _pull_complex_batch(
    context: _PullbackContext,
    pullback: _core.Pullback,
    payload: list[list[tuple[int, int, float, float]]],
) -> tuple[RealGradient, ...]:
    """Pulls sparse complex cotangents back in one batched backward pass."""
    try:
        matrix = context.core.pull_back_complex(
            pullback, payload, _settings_json(context.options), engine_pool()
        )
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None
    return _unpack_gradients(context.names, np.asarray(matrix.real, dtype=np.float64))


def _pull_real_batch(
    context: _PullbackContext,
    pullback: _core.Pullback,
    payload: list[tuple[list[tuple[int, int, float]], float]],
) -> tuple[RealGradient, ...]:
    """Pulls sparse real cotangents back in one batched backward pass."""
    try:
        matrix = context.core.pull_back_real(
            pullback, payload, _settings_json(context.options), engine_pool()
        )
    except _core.SimulationError as error:
        raise SimulationError(str(error)) from None
    return _unpack_gradients(context.names, np.asarray(matrix, dtype=np.float64))


def _pull_batch(
    context: _PullbackContext,
    pullback: _core.Pullback,
    cotangents: Sequence[specs.MeasureCotangent],
) -> tuple[RealGradient, ...]:
    """Pulls sparse measure or slack `cotangents` back in one batched backward pass."""
    if not cotangents:
        return ()
    if not context.names:
        return tuple({} for _ in cotangents)
    if isinstance(context.analysis, Ac):
        complex_payload: list[list[tuple[int, int, float, float]]] = []
        for cotangent in cotangents:
            if not isinstance(cotangent, specs.ComplexCotangent):
                message = (
                    "a small-signal response pullback takes ComplexCotangent, "
                    f"not {type(cotangent).__name__}"
                )
                raise specs.MeasureError(message)
            complex_payload.append(list(cotangent.entries))
        return _pull_complex_batch(context, pullback, complex_payload)
    if isinstance(context.analysis, Stability):
        loop_payload: list[list[tuple[int, int, float, float]]] = []
        for cotangent in cotangents:
            if not isinstance(cotangent, specs.LoopCotangent):
                message = (
                    "a stability pullback takes LoopCotangent, "
                    f"not {type(cotangent).__name__}"
                )
                raise specs.MeasureError(message)
            loop_payload.append(loop_cotangent_entries(cotangent))
        return _pull_complex_batch(context, pullback, loop_payload)
    if isinstance(context.analysis, Noise):
        noise_payload: list[tuple[list[tuple[int, int, float]], float]] = []
        for cotangent in cotangents:
            if not isinstance(cotangent, specs.NoiseCotangent):
                message = (
                    "a noise pullback takes NoiseCotangent, "
                    f"not {type(cotangent).__name__}"
                )
                raise specs.MeasureError(message)
            noise_payload.append((noise_cotangent_entries(cotangent), 0.0))
        return _pull_real_batch(context, pullback, noise_payload)
    real_payload: list[tuple[list[tuple[int, int, float]], float]] = []
    for cotangent in cotangents:
        if not isinstance(cotangent, specs.RealCotangent):
            message = (
                "a real analysis pullback takes RealCotangent, "
                f"not {type(cotangent).__name__}"
            )
            raise specs.MeasureError(message)
        real_payload.append((list(cotangent.entries), cotangent.period))
    return _pull_real_batch(context, pullback, real_payload)


def _verdict_from_reading(
    requirement: specs.Requirement,
    reading: specs.Reading,
    status: specs.Status,
    slack: float,
    corner: str | None = None,
) -> specs.Verdict:
    """Builds a `Verdict` from an evaluated `reading`, `status`, and `slack`."""
    unit = "" if reading.unit == DIMENSIONLESS else str(reading.unit)
    return specs.Verdict(
        requirement=str(requirement),
        status=status,
        value=reading.value,
        unit=unit,
        slack=slack,
        corner=corner,
        source_location=requirement.source_location,
    )


def _unmeasured_verdict(
    requirement: specs.Requirement, reason: str, corner: str | None = None
) -> specs.Verdict:
    """Builds an unmeasured `Verdict` carrying `reason`."""
    return specs.Verdict(
        requirement=str(requirement),
        status="unmeasured",
        corner=corner,
        reason=reason,
        source_location=requirement.source_location,
    )


def _pull_measures(
    context: _PullbackContext, measures: Sequence[specs.Measure]
) -> tuple[MeasureDerivative, ...]:
    """Evaluates and differentiates `measures` for `context` in one backward pass."""
    if not measures:
        return ()
    result, pullback = _run_vjp(context)
    evaluated = [measure.cotangent(result) for measure in measures]
    gradients = _pull_batch(
        context, pullback, [cotangent for _, cotangent in evaluated]
    )
    return tuple(
        MeasureDerivative(measure, reading, gradient)
        for measure, (reading, _), gradient in zip(
            measures, evaluated, gradients, strict=True
        )
    )


def _pull_slacks(
    context: _PullbackContext,
    requirements: Sequence[specs.Requirement],
    *,
    corner: str | None = None,
) -> tuple[RequirementDerivative, ...]:
    """Evaluates and differentiates `requirements` for `context` in one pass."""
    if not requirements:
        return ()
    result, pullback = _run_vjp(context)
    outcomes: list[RequirementDerivative | None] = [None] * len(requirements)
    differentiable: list[
        tuple[int, specs.Requirement, specs.Verdict, specs.MeasureCotangent]
    ] = []
    for index, requirement in enumerate(requirements):
        try:
            reading = requirement.measure.read(result)
            if math.isnan(reading.value):
                message = "the value read is not a number"
                raise specs.MeasureError(message)
            status, slack = requirement.judge(reading)
        except (JoltyError, KeyError) as error:
            reason = str(error.args[0])
            verdict = _unmeasured_verdict(requirement, reason, corner=corner)
            outcomes[index] = RequirementDerivative(requirement, verdict, reason=reason)
            continue
        verdict = _verdict_from_reading(
            requirement, reading, status, slack, corner=corner
        )
        try:
            _, _, cotangent = requirement.slack_cotangent(result)
        except (JoltyError, KeyError) as error:
            outcomes[index] = RequirementDerivative(
                requirement, verdict, reason=str(error.args[0])
            )
            continue
        differentiable.append((index, requirement, verdict, cotangent))
    gradients = _pull_batch(
        context, pullback, [cotangent for _, _, _, cotangent in differentiable]
    )
    for (index, requirement, verdict, _), gradient in zip(
        differentiable, gradients, strict=True
    ):
        outcomes[index] = RequirementDerivative(requirement, verdict, gradient)
    return tuple(item for item in outcomes if item is not None)


def measure_derivatives(
    function: Function,
    parameters: Parameters,
    measures: Sequence[specs.Measure],
) -> tuple[MeasureDerivative, ...]:
    """Evaluates `measures` and their exact parameter derivatives in one backward pass.

    Solves `function.analysis` once at `parameters`, evaluates each measure's sparse
    cotangent against the solved result, and pulls all cotangents back together in a
    single engine backward pass. Supports both primitive device parameters and bound
    SPICE subcircuit parameters.

    Args:
        function: Differentiable circuit analysis function.
        parameters: Parameter overrides in SI base units (or subcircuit arguments)
            selecting both the evaluation point and the differentiated parameters.
        measures: Built-in measures evaluated and differentiated on `function`.

    Returns:
        Ordered `MeasureDerivative` entries matching `measures`.

    Raises:
        ElaborationError: When `parameters` names an unknown or out-of-range parameter.
        SimulationError: When the analysis or backward pullback fails.
        MeasureError: When any measure cannot be evaluated or is not differentiable.
    """
    core, primitive_names, evaluated = function.pull_setup(parameters)
    context = _PullbackContext(
        core=core,
        built=function.elaborated.circuit,
        analysis=function.analysis,
        options=function.options,
        names=primitive_names,
    )
    pulled = _pull_measures(context, measures)
    if not evaluated:
        return pulled
    return tuple(
        replace(
            item,
            derivatives=contract_real_gradient(
                parameters, evaluated, item.derivatives
            ),
        )
        for item in pulled
    )


def slack_derivatives(
    function: Function,
    parameters: Parameters,
    requirements: Sequence[specs.Requirement],
) -> tuple[RequirementDerivative, ...]:
    """Evaluates `requirements` and exact slack derivatives in one backward pass.

    Solves `function.analysis` once at `parameters`, computes each requirement's
    verdict and sparse signed-slack cotangent, and pulls all differentiable slack
    cotangents back together in a single engine backward pass. Supports both primitive
    device parameters and bound SPICE subcircuit parameters.

    Args:
        function: Differentiable circuit analysis function.
        parameters: Parameter overrides in SI base units (or subcircuit arguments)
            selecting both the evaluation point and the differentiated parameters.
        requirements: Specification requirements evaluated on `function`.

    Returns:
        Ordered `RequirementDerivative` entries matching `requirements`.

    Raises:
        ElaborationError: When `parameters` names an unknown or out-of-range parameter.
        SimulationError: When the analysis or backward pullback fails.
    """
    core, primitive_names, evaluated = function.pull_setup(parameters)
    context = _PullbackContext(
        core=core,
        built=function.elaborated.circuit,
        analysis=function.analysis,
        options=function.options,
        names=primitive_names,
    )
    pulled = _pull_slacks(context, requirements)
    if not evaluated:
        return pulled
    return tuple(
        replace(
            item,
            derivatives=contract_real_gradient(
                parameters, evaluated, item.derivatives
            ),
        )
        if item.reason is None
        else item
        for item in pulled
    )


def _build_circuit(circuit_or_top: ir.Circuit | Component) -> ir.Circuit:
    """Resolves `circuit_or_top` to a canonical `ir.Circuit`."""
    return (
        circuit_or_top
        if isinstance(circuit_or_top, ir.Circuit)
        else circuit(circuit_or_top)
    )


def _conditions_of(built: ir.Circuit, plan: Plan) -> dict[str | None, Elaborated]:
    """Builds `Elaborated` circuits for nominal conditions and every temperature corner.

    Raises:
        ValueError: When any corner specifies component build arguments or a foundry
            process corner.
    """
    conditions: dict[str | None, Elaborated] = {
        None: Elaborated(built, temperature=plan.temperature)
    }
    for name, corner in plan.corners.items():
        if corner.arguments:
            message = (
                f"corner {name!r} specifies build arguments, which require "
                "rebuilding the component"
            )
            raise ValueError(message)
        if corner.process is not None:
            message = (
                f"corner {name!r} specifies process {corner.process!r}, which has "
                "no native parameter pullback"
            )
            raise ValueError(message)
        temperature = (
            plan.temperature if corner.temperature is None else corner.temperature
        )
        conditions[name] = Elaborated(built, temperature=temperature)
    return conditions


def _group_nominal_requirements(
    plan: Plan, specifications: specs.Specs
) -> tuple[
    list[RequirementDerivative | None],
    dict[str, list[tuple[int, specs.Requirement]]],
]:
    """Splits nominal requirements into missing-analysis outcomes and groups."""
    planned = tuple(plan.analyses)
    grouped: dict[str, list[tuple[int, specs.Requirement]]] = {}
    outcomes: list[RequirementDerivative | None] = [None] * len(
        specifications.requirements
    )
    for index, requirement in enumerate(specifications.requirements):
        analysis_name = requirement.measure.analysis
        if analysis_name not in plan.analyses:
            close = close_names(planned, analysis_name)
            reason = f"the plan runs no analysis named {analysis_name}{close}"
            verdict = _unmeasured_verdict(requirement, reason)
            outcomes[index] = RequirementDerivative(requirement, verdict, reason=reason)
            continue
        grouped.setdefault(analysis_name, []).append((index, requirement))
    return outcomes, grouped


def _evaluate_nominal_differentiable(
    context: _PullbackContext,
    analysis_name: str,
    entries: Sequence[tuple[int, specs.Requirement]],
) -> list[tuple[int, RequirementDerivative]]:
    """Evaluates and differentiates one differentiable nominal analysis group."""
    requirements = [requirement for _, requirement in entries]
    try:
        pulled = _pull_slacks(context, requirements)
    except SimulationError as error:
        failed = f"the analysis {analysis_name} failed"
        reason = f"{failed}: {error}"
        return [
            (
                index,
                RequirementDerivative(
                    requirement,
                    _unmeasured_verdict(requirement, failed),
                    reason=reason,
                ),
            )
            for index, requirement in entries
        ]
    return [(index, item) for (index, _), item in zip(entries, pulled, strict=True)]


def _evaluate_nominal_plain(
    core: _core.Simulator,
    built: ir.Circuit,
    plan: Plan,
    analysis_name: str,
    entries: Sequence[tuple[int, specs.Requirement]],
) -> list[tuple[int, RequirementDerivative]]:
    """Evaluates one non-differentiable nominal analysis group without pullback."""
    analysis = plan.analyses[analysis_name]
    try:
        solved = _run_plain(core, built, analysis, plan.options)
    except SimulationError:
        reason = f"the analysis {analysis_name} failed"
        return [
            (
                index,
                RequirementDerivative(
                    requirement,
                    _unmeasured_verdict(requirement, reason),
                    reason=reason,
                ),
            )
            for index, requirement in entries
        ]
    planned = tuple(plan.analyses)
    reason = f"{results.result_kind(solved)} has no parameter pullback"
    found: list[tuple[int, RequirementDerivative]] = []
    for index, requirement in entries:
        (verdict,) = specs.evaluate(
            specs.Specs([requirement]), planned, {analysis_name: solved}
        )
        found.append(
            (
                index,
                RequirementDerivative(
                    requirement,
                    verdict,
                    reason=verdict.reason if verdict.status == "unmeasured" else reason,
                ),
            )
        )
    return found


def _solve_all_conditions(
    built: ir.Circuit,
    conditions: Mapping[str | None, Elaborated],
    plan: Plan,
    needed: set[str],
    overrides: Mapping[str, Mapping[str, ParameterValue]] | None,
) -> dict[str | None, dict[str, results.AnalysisResult]]:
    """Solves every analysis in `needed` across nominal conditions and all corners."""
    solved_by_condition: dict[str | None, dict[str, results.AnalysisResult]] = {}
    for corner_name, elaborated in conditions.items():
        core = elaborated.simulator(overrides)
        solved: dict[str, results.AnalysisResult] = {}
        for analysis_name in needed:
            try:
                solved[analysis_name] = _run_plain(
                    core, built, plan.analyses[analysis_name], plan.options
                )
            except SimulationError:
                continue
        solved_by_condition[corner_name] = solved
    return solved_by_condition


def _group_worst_corner_requirements(
    plan: Plan,
    specifications: specs.Specs,
    verdicts: Sequence[specs.Verdict],
    solved_by_condition: Mapping[str | None, Mapping[str, results.AnalysisResult]],
) -> tuple[
    list[RequirementDerivative | None],
    dict[tuple[str | None, str], list[tuple[int, specs.Requirement]]],
]:
    """Partitions worst-corner verdicts into finalized outcomes and pullback groups."""
    outcomes: list[RequirementDerivative | None] = [None] * len(verdicts)
    grouped: dict[tuple[str | None, str], list[tuple[int, specs.Requirement]]] = {}
    for index, (requirement, verdict) in enumerate(
        zip(specifications.requirements, verdicts, strict=True)
    ):
        if verdict.status == "unmeasured":
            outcomes[index] = RequirementDerivative(
                requirement, verdict, reason=verdict.reason
            )
            continue
        analysis_name = requirement.measure.analysis
        analysis = plan.analyses[analysis_name]
        if not isinstance(analysis, _DIFFERENTIABLE_TYPES):
            solved_result = solved_by_condition[verdict.corner][analysis_name]
            reason = f"{results.result_kind(solved_result)} has no parameter pullback"
            outcomes[index] = RequirementDerivative(requirement, verdict, reason=reason)
            continue
        grouped.setdefault((verdict.corner, analysis_name), []).append(
            (index, requirement)
        )
    return outcomes, grouped


def requirement_derivatives(
    circuit_or_top: ir.Circuit | Component,
    plan: Plan,
    specifications: specs.Specs,
    parameters: Sequence[tuple[str, str]],
    *,
    overrides: Mapping[str, Mapping[str, ParameterValue]] | None = None,
) -> tuple[RequirementDerivative, ...]:
    """Computes exact slack derivatives for each requirement at its worst condition.

    Evaluates `specifications` across nominal conditions and all temperature corners
    in `plan`, then differentiates each measured requirement at its worst condition
    with respect to `parameters` in one backward pass per `(corner, analysis)`.

    Args:
        circuit_or_top: Circuit IR or top-level component to simulate.
        plan: Simulation plan naming analyses, options, temperature, and optional
            temperature corners.
        specifications: Specification requirements to evaluate and differentiate.
        parameters: Ordered `(instance_path, parameter_name)` pairs to differentiate.
        overrides: Optional parameter overrides applied across every condition.

    Returns:
        Ordered `RequirementDerivative` entries matching `specifications.requirements`.

    Raises:
        ValueError: When a plan corner specifies build arguments or a foundry process.
        ElaborationError: When the circuit fails to elaborate or `overrides` or
            `parameters` reference unknown parameters.
        SimulationError: When a backward pullback fails.
    """
    built = _build_circuit(circuit_or_top)
    conditions = _conditions_of(built, plan)
    if not plan.corners:
        core = conditions[None].simulator(overrides)
        outcomes, grouped = _group_nominal_requirements(plan, specifications)
        for analysis_name, entries in grouped.items():
            analysis = plan.analyses[analysis_name]
            if isinstance(analysis, _DIFFERENTIABLE_TYPES):
                context = _PullbackContext(
                    core, built, analysis, plan.options, parameters
                )
                evaluated = _evaluate_nominal_differentiable(
                    context, analysis_name, entries
                )
            else:
                evaluated = _evaluate_nominal_plain(
                    core, built, plan, analysis_name, entries
                )
            for index, item in evaluated:
                outcomes[index] = item
        return tuple(item for item in outcomes if item is not None)

    needed = {
        requirement.measure.analysis
        for requirement in specifications.requirements
        if requirement.measure.analysis in plan.analyses
    }
    solved_by_condition = _solve_all_conditions(
        built, conditions, plan, needed, overrides
    )
    corner_results = {
        name: solved for name, solved in solved_by_condition.items() if name is not None
    }
    verdicts = specs.evaluate(
        specifications, tuple(plan.analyses), solved_by_condition[None], corner_results
    )
    outcomes, corner_groups = _group_worst_corner_requirements(
        plan, specifications, verdicts, solved_by_condition
    )
    for (corner_name, analysis_name), entries in corner_groups.items():
        analysis = plan.analyses[analysis_name]
        if not isinstance(analysis, _DIFFERENTIABLE_TYPES):
            continue
        context = _PullbackContext(
            conditions[corner_name].simulator(overrides),
            built,
            analysis,
            plan.options,
            parameters,
        )
        pulled = _pull_slacks(
            context, [requirement for _, requirement in entries], corner=corner_name
        )
        for (index, _), item in zip(entries, pulled, strict=True):
            outcomes[index] = item
    return tuple(item for item in outcomes if item is not None)


def _nominal_si(parameter: variation.Toleranced) -> float:
    """Returns the scalar SI nominal value of `parameter`."""
    return (
        parameter.nominal.value
        if isinstance(parameter.nominal, Quantity)
        else float(parameter.nominal)
    )


def _sensitivity(
    parameter: variation.Toleranced,
    derivatives: Mapping[str, Mapping[str, float]],
) -> float:
    """Returns `nominal * d(measure)/d(parameter)` for `parameter`."""
    by_instance = derivatives.get(parameter.instance)
    derivative = (
        0.0 if by_instance is None else by_instance.get(parameter.parameter, 0.0)
    )
    return _nominal_si(parameter) * derivative


def _choose_extreme(plus: float, minus: float, sign: float) -> tuple[float, float]:
    """Selects the direction (`-1`, `0`, or `1`) and value maximizing `sign * value`."""
    if plus == minus:
        return 0.0, plus
    if sign * plus > sign * minus:
        return 1.0, plus
    return -1.0, minus


def _lot_extreme(
    varied: Sequence[variation.Toleranced],
    members: Sequence[int],
    shared_lot: ir.Lot,
    derivatives: Mapping[str, Mapping[str, float]],
    *,
    sign: float,
) -> tuple[float, float, list[tuple[int, float]]]:
    """Computes the exact extreme contribution and directions for one shared lot."""
    weighted_terms: list[float] = []
    nominal_terms: list[float] = []
    member_directions: list[tuple[int, float]] = []
    for index in members:
        parameter = varied[index]
        sensitivity = _sensitivity(parameter, derivatives)
        nominal_terms.append(sensitivity)
        high_term = sensitivity * variation.edge(parameter.tolerance, 1.0)
        low_term = sensitivity * variation.edge(parameter.tolerance, -1.0)
        if high_term == low_term:
            member_directions.append((index, 0.0))
            weighted_terms.append(sensitivity)
        else:
            direction, chosen_term = _choose_extreme(high_term, low_term, sign)
            member_directions.append((index, direction))
            weighted_terms.append(chosen_term)
    weighted = math.fsum(weighted_terms)
    nominal_sum = math.fsum(nominal_terms)
    lot_plus = weighted * variation.edge(shared_lot, 1.0) - nominal_sum
    lot_minus = weighted * variation.edge(shared_lot, -1.0) - nominal_sum
    lot_direction, contribution = _choose_extreme(lot_plus, lot_minus, sign)
    return lot_direction, contribution, member_directions


def _linear_extreme(
    varied: Sequence[variation.Toleranced],
    derivatives: Mapping[str, Mapping[str, float]],
    *,
    maximize: bool,
) -> tuple[float, list[float], dict[variation.LotMember, float]]:
    """Finds the exact extreme shift of a linearized measure over tolerances and lots.

    Each parameter scales as ``nominal * edge(tolerance, d) * edge(lot, D)``. Because
    every lot factor ``edge(lot, D)`` is strictly positive, each member's optimal
    instance direction ``d`` is independent of ``D``, and the lot's optimal direction
    ``D`` follows directly from evaluating both extremes of ``D``.
    """
    sign = 1.0 if maximize else -1.0
    directions = [0.0] * len(varied)
    lots: dict[variation.LotMember, float] = {}
    independent: list[int] = []
    lot_groups: dict[variation.LotMember, tuple[ir.Lot, list[int]]] = {}
    for index, parameter in enumerate(varied):
        lot_key = variation.lot(parameter)
        shared_lot = parameter.tolerance.lot
        if lot_key is None or shared_lot is None:
            independent.append(index)
        else:
            _, members = lot_groups.setdefault(lot_key, (shared_lot, []))
            members.append(index)

    terms: list[float] = []
    for index in independent:
        parameter = varied[index]
        sensitivity = _sensitivity(parameter, derivatives)
        plus = sensitivity * (variation.edge(parameter.tolerance, 1.0) - 1.0)
        minus = sensitivity * (variation.edge(parameter.tolerance, -1.0) - 1.0)
        direction, contribution = _choose_extreme(plus, minus, sign)
        directions[index] = direction
        terms.append(contribution)

    for lot_key, (shared_lot, members) in lot_groups.items():
        lot_direction, contribution, member_directions = _lot_extreme(
            varied, members, shared_lot, derivatives, sign=sign
        )
        lots[lot_key] = lot_direction
        terms.append(contribution)
        for index, direction in member_directions:
            directions[index] = direction

    return math.fsum(terms), directions, lots


def estimate_worst_case(
    requirement: specs.Requirement,
    nominal: specs.Reading,
    derivatives: Mapping[str, Mapping[str, float]],
    toleranced: Sequence[variation.Toleranced],
    *,
    corner: str | None = None,
) -> LinearWorstCase:
    """Computes the first-order worst-case slack estimate for `requirement`.

    Evaluates the exact minimum and maximum of the first-order Taylor expansion of
    `requirement.measure` around `nominal` over all independent tolerances and shared
    lots in `toleranced`.

    Args:
        requirement: Specification requirement whose worst-case slack is estimated.
        nominal: Measured nominal value and unit of `requirement.measure`.
        derivatives: Exact first-order derivatives ``d(measure)/d(p)`` keyed by
            instance path and parameter name.
        toleranced: Toleranced parameters defining the variation box and shared lots.
        corner: Optional plan corner name where `nominal` and `derivatives` apply.

    Returns:
        First-order `LinearWorstCase` estimate with worst-case overrides and status.

    Raises:
        DimensionError: When `requirement` bounds do not match `nominal.unit`.
    """
    _, nominal_slack = requirement.judge(nominal)
    candidates: list[
        tuple[
            float,
            Literal["minimum", "maximum"],
            float,
            list[float],
            dict[variation.LotMember, float],
        ]
    ] = []
    if requirement.minimum is not None:
        delta_low, low_directions, low_lots = _linear_extreme(
            toleranced, derivatives, maximize=False
        )
        low_value = nominal.value + delta_low
        candidates.append(
            (
                low_value - requirement.minimum.value,
                "minimum",
                low_value,
                low_directions,
                low_lots,
            )
        )
    if requirement.maximum is not None:
        delta_high, high_directions, high_lots = _linear_extreme(
            toleranced, derivatives, maximize=True
        )
        high_value = nominal.value + delta_high
        candidates.append(
            (
                requirement.maximum.value - high_value,
                "maximum",
                high_value,
                high_directions,
                high_lots,
            )
        )
    _, bound, estimated_value, directions, lots = min(
        candidates, key=lambda entry: entry[0]
    )
    status, estimated_slack = requirement.judge(
        specs.Reading(estimated_value, nominal.unit)
    )
    overrides = variation.extreme(toleranced, directions, lots)
    copied_derivatives: RealGradient = {
        path: {name: float(value) for name, value in named.items()}
        for path, named in derivatives.items()
    }
    return LinearWorstCase(
        requirement=requirement,
        nominal=nominal,
        nominal_slack=nominal_slack,
        estimated_value=estimated_value,
        estimated_slack=estimated_slack,
        status=status,
        bound=bound,
        overrides=overrides,
        derivatives=copied_derivatives,
        corner=corner,
    )


def _toleranced_at(
    built: ir.Circuit,
    toleranced: Sequence[variation.Toleranced] | None,
    overrides: Mapping[str, Mapping[str, ParameterValue]] | None,
) -> tuple[variation.Toleranced, ...]:
    """Resolves active toleranced parameters, reflecting any nominal `overrides`."""
    if toleranced is not None:
        return tuple(toleranced)
    varied = variation.toleranced(built)
    if overrides is None or not overrides:
        return varied
    converted = devices.convert_overrides(built, overrides)
    wrapped = devices.wrap_overrides(built, converted)
    return tuple(
        replace(parameter, nominal=overridden)
        if (overridden := wrapped.get(parameter.instance, {}).get(parameter.parameter))
        is not None
        else parameter
        for parameter in varied
    )


def linear_worst_case(
    circuit_or_top: ir.Circuit | Component,
    plan: Plan,
    specifications: specs.Specs,
    *,
    toleranced: Sequence[variation.Toleranced] | None = None,
    overrides: Mapping[str, Mapping[str, ParameterValue]] | None = None,
) -> tuple[LinearWorstCase, ...]:
    """Computes first-order linear worst-case slack estimates for every requirement.

    Computes exact nominal measure derivatives with respect to every toleranced
    parameter in one backward pass per analysis, then evaluates
    [`estimate_worst_case`][jolty.simulate.estimate_worst_case] across nominal
    conditions and any temperature corners in `plan`, keeping the worst estimate per
    requirement.

    Args:
        circuit_or_top: Circuit IR or top-level component to analyze.
        plan: Simulation plan containing differentiable analyses and optional
            temperature corners.
        specifications: Specification requirements whose linear worst-case slacks are
            estimated.
        toleranced: Optional explicit toleranced parameter descriptors; defaults to
            [`variation.toleranced`][jolty.variation.toleranced] on `circuit_or_top`.
        overrides: Optional nominal parameter overrides applied before variation.

    Returns:
        Ordered `LinearWorstCase` estimates matching `specifications.requirements`.

    Raises:
        ValueError: When a plan corner specifies build arguments or a foundry process,
            or when a requirement references an unknown analysis.
        TypeError: When a requirement references a non-differentiable analysis type.
        ElaborationError: When the circuit fails to elaborate.
        SimulationError: When a forward analysis or backward pullback fails.
        MeasureError: When a requirement's measure cannot be differentiated.
    """
    built = _build_circuit(circuit_or_top)
    conditions = _conditions_of(built, plan)
    varied = _toleranced_at(built, toleranced, overrides)
    names = list(
        dict.fromkeys((parameter.instance, parameter.parameter) for parameter in varied)
    )
    planned = tuple(plan.analyses)
    grouped: dict[str, list[tuple[int, specs.Requirement]]] = {}
    for index, requirement in enumerate(specifications.requirements):
        analysis_name = requirement.measure.analysis
        if analysis_name not in plan.analyses:
            close = close_names(planned, analysis_name)
            message = f"the plan runs no analysis named {analysis_name}{close}"
            raise ValueError(message)
        analysis = plan.analyses[analysis_name]
        if not isinstance(analysis, _DIFFERENTIABLE_TYPES):
            message = (
                f"analysis {analysis_name!r} ({type(analysis).__name__}) has no "
                "reverse-mode parameter pullback"
            )
            raise TypeError(message)
        grouped.setdefault(analysis_name, []).append((index, requirement))

    worst_by_index: list[LinearWorstCase | None] = [None] * len(
        specifications.requirements
    )
    for corner_name, elaborated in conditions.items():
        core = elaborated.simulator(overrides)
        for analysis_name, entries in grouped.items():
            analysis = plan.analyses[analysis_name]
            if not isinstance(analysis, _DIFFERENTIABLE_TYPES):
                continue
            context = _PullbackContext(core, built, analysis, plan.options, names)
            measures = [requirement.measure for _, requirement in entries]
            pulled = _pull_measures(context, measures)
            for (index, requirement), item in zip(entries, pulled, strict=True):
                estimate = estimate_worst_case(
                    requirement,
                    item.reading,
                    item.derivatives,
                    varied,
                    corner=corner_name,
                )
                current = worst_by_index[index]
                if (
                    current is None
                    or estimate.estimated_slack < current.estimated_slack
                ):
                    worst_by_index[index] = estimate
    return tuple(item for item in worst_by_index if item is not None)
