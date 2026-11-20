# Copyright 2026 Maxence Faldor
"""Static validation of analyses, plans, records, variants, and traces."""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from jolty import devices, ir, results
from jolty.devices import ParameterValue
from jolty.errors import DimensionError, PlanError, close_names
from jolty.plan import (
    LOOP_GAIN,
    Ac,
    Analysis,
    DcSweep,
    Explicit,
    Grid,
    Mismatch,
    MonteCarlo,
    Noise,
    PeriodicSteadyState,
    Plan,
    PoleZero,
    Product,
    Recorded,
    Sensitivity,
    Stability,
    TransferFunction,
    Transient,
    Variants,
)
from jolty.simulate._stress import Joined
from jolty.units import CURRENT, TEMPERATURE, VOLTAGE, Quantity

WILDCARDS: Final = frozenset("*?[")
"""The characters that make a recorded path a pattern."""

_INDEPENDENT_SOURCE_MODELS: Final = frozenset({"voltage_source", "current_source"})
"""Device models representing independent electrical sources."""

_LOGIC_MODELS: Final = frozenset({"logic_cell", "logic_source"})
"""Device models handled by the discrete logic event scheduler."""


@dataclass(frozen=True, slots=True)
class Catalog:
    """Every static name a circuit exposes, cached once per `Simulation`."""

    devices: Mapping[str, ir.Instance]
    device_paths: tuple[str, ...]
    component_paths: tuple[str, ...]
    independent_sources: tuple[str, ...]
    voltage_sources: tuple[str, ...]
    probes: tuple[str, ...]
    unknowns: Mapping[str, results.Unknown]
    unknown_names: tuple[str, ...]
    ground: str | None
    voltage_nets: tuple[str, ...]
    has_logic_devices: bool
    overrides: Mapping[str, Mapping[str, ParameterValue]]

    def parameters(self, path: str) -> tuple[str, ...]:
        """Returns every real-valued parameter path defined on the device at `path`.

        Args:
            path: Hierarchical path of an active device instance in `devices`.

        Returns:
            Deduplicated parameter paths writable on the device at `path`.
        """
        instance = self.devices[path]
        return tuple(
            dict.fromkeys(
                (*ir.numbers(instance.parameters), *self.overrides.get(path, ()))
            )
        )


def build_catalog(  # noqa: PLR0913 - elaborated circuit topology and metadata
    built: ir.Circuit,
    substituted: Collection[str],
    unknowns: tuple[results.Unknown, ...],
    nets: tuple[str, ...],
    *,
    overrides: Mapping[str, Mapping[str, ParameterValue]],
    joined: Callable[[str, str], Joined | None],
) -> Catalog:
    """Builds the static lookup catalog for an elaborated circuit.

    Args:
        built: Elaborated circuit whose active devices and components are indexed.
        substituted: Component instance paths replaced by substitute device models.
        unknowns: Circuit unknowns exposed by the engine.
        nets: Discrete logic net names exposed by the engine.
        overrides: Parameter overrides active on the simulation.
        joined: Lookup returning the connected `Joined` node for an instance terminal.

    Returns:
        Populated `Catalog` indexing devices, sources, probes, nets, and unknowns.
    """
    devices, component_paths = _walk_circuit(built, substituted)
    independent_sources = tuple(
        path
        for path, instance in devices.items()
        if isinstance(instance.of, ir.DeviceOf)
        and instance.of.model in _INDEPENDENT_SOURCE_MODELS
    )
    voltage_sources = tuple(
        path
        for path, instance in devices.items()
        if isinstance(instance.of, ir.DeviceOf)
        and instance.of.model == "voltage_source"
    )
    unknown_map = {unknown.name: unknown for unknown in unknowns}
    probes = tuple(
        path for path in voltage_sources if _is_probe(path, unknown_map, joined)
    )
    voltage_nets = tuple(
        unknown.name
        for unknown in unknowns
        if unknown.nature == "voltage" and ":" not in unknown.name
    )
    has_logic_devices = (
        bool(substituted)
        or bool(nets)
        or any(
            isinstance(instance.of, ir.DeviceOf) and instance.of.model in _LOGIC_MODELS
            for instance in devices.values()
        )
    )
    return Catalog(
        devices=devices,
        device_paths=tuple(devices),
        component_paths=component_paths,
        independent_sources=independent_sources,
        voltage_sources=voltage_sources,
        probes=probes,
        unknowns=unknown_map,
        unknown_names=tuple(unknown_map),
        ground=built.ground_port(),
        voltage_nets=voltage_nets,
        has_logic_devices=has_logic_devices,
        overrides=overrides,
    )


def _walk_circuit(
    built: ir.Circuit, substituted: Collection[str]
) -> tuple[dict[str, ir.Instance], tuple[str, ...]]:
    """Collects active device instances (skipping substituted subtrees) and paths."""
    prefixes = tuple(f"{path}/" for path in substituted)
    devices: dict[str, ir.Instance] = {}
    components: list[str] = []
    for path, instance in built.walk(breadth_first=True):
        if isinstance(instance.of, ir.ComponentOf):
            components.append(path)
        elif not (prefixes and path.startswith(prefixes)):
            devices[path] = instance
    return devices, tuple(components)


def _is_probe(
    path: str,
    unknowns: Mapping[str, results.Unknown],
    joined: Callable[[str, str], Joined | None],
) -> bool:
    """Returns whether `path` is a floating voltage source off ground."""
    positive = joined(path, "positive")
    negative = joined(path, "negative")
    return (
        positive is not None
        and negative is not None
        and positive.node is not None
        and negative.node is not None
        and f"{path}:current" in unknowns
    )


def process_problems(plan: Plan, *, has_foundry: bool) -> list[str]:
    """Reports process corners on `plan` when the circuit has no foundry devices.

    Args:
        plan: Simulation plan declaring top-level and per-corner process names.
        has_foundry: Flag indicating presence of foundry PDK devices in the circuit.

    Returns:
        Diagnostic messages for unsupported process corner declarations.
    """
    if has_foundry:
        return []
    problems: list[str] = []
    if plan.process is not None:
        problems.append(
            f"the process corner {plan.process} sets a foundry's devices, "
            "and the circuit holds none"
        )
    for name, corner in plan.corners.items():
        if corner.process is not None and not corner.arguments:
            problems.append(
                f"corner `{name}`: the process corner {corner.process} sets "
                "a foundry's devices, and the circuit holds none"
            )
    return problems


def plan_condition_problems(
    plan: Plan,
    *,
    temperature: Quantity,
    has_foundry: bool,
    check_temperature: bool,
) -> list[str]:
    """Checks `plan`'s temperature and process corners against a simulation.

    Args:
        plan: Simulation plan whose operating conditions are checked.
        temperature: Simulation ambient temperature quantity.
        has_foundry: Flag indicating presence of foundry PDK devices in the circuit.
        check_temperature: Flag enforcing equality between `plan.temperature` and
            `temperature`.

    Returns:
        Diagnostic messages for mismatched temperatures or unsupported process corners.
    """
    problems: list[str] = []
    if check_temperature and plan.temperature is not None:
        target = plan.temperature.to(TEMPERATURE, "a plan's temperature")
        ambient = temperature.to(TEMPERATURE, "a temperature")
        if target != ambient:
            problems.append(
                f"the plan's temperature, {plan.temperature}, is not the "
                f"simulation's, {temperature}"
            )
    problems.extend(process_problems(plan, has_foundry=has_foundry))
    return problems


def nodeset_problems(
    mapping: Mapping[str, Quantity], what: str, catalog: Catalog
) -> list[str]:
    """Collects every key in `mapping` that is not a circuit voltage net.

    Args:
        mapping: Mapping from net names to voltage quantities.
        what: Singular noun naming the mapping kind, such as ``"nodeset"``.
        catalog: Static circuit catalog supplying valid voltage nets and ground.

    Returns:
        Diagnostic messages for keys absent from `catalog.voltage_nets`.
    """
    valid = frozenset(catalog.voltage_nets)
    problems: list[str] = []
    for net in mapping:
        if net in valid:
            continue
        close = close_names(catalog.voltage_nets, net)
        if net in catalog.unknowns:
            nature = catalog.unknowns[net].nature
            problems.append(
                f"the {what} names `{net}`, which is no net of the circuit "
                f"({nature} unknown){close}"
            )
        elif net == catalog.ground:
            problems.append(
                f"the {what} names `{net}`, which is ground, "
                f"not an electrical net{close}"
            )
        else:
            problems.append(
                f"the {what} names `{net}`, which is no net of the circuit{close}"
            )
    return problems


def _source_problem(source: str, catalog: Catalog) -> str | None:
    """Returns a problem string when `source` is not an independent source."""
    sources_list = (
        f"; independent sources: {', '.join(catalog.independent_sources)}"
        if catalog.independent_sources
        else ""
    )
    if source not in catalog.devices:
        close = close_names(catalog.independent_sources, source) or close_names(
            catalog.device_paths, source
        )
        return f"there is no instance at `{source}`{close}{sources_list}"
    if source not in catalog.independent_sources:
        return f"`{source}` is not an independent source{sources_list}"
    return None


def _output_problem(name: str, catalog: Catalog, *, allow_ground: bool) -> str | None:
    """Returns a problem string when `name` is not a valid output name."""
    if name in catalog.unknowns:
        return None
    if allow_ground and name == catalog.ground:
        return None
    candidates = (
        (*catalog.unknown_names, catalog.ground)
        if allow_ground and catalog.ground is not None
        else catalog.unknown_names
    )
    close = close_names(candidates, name)
    if not allow_ground and name == catalog.ground:
        return f"there is no unknown named `{name}` (it is ground){close}"
    return f"there is no unknown named `{name}`{close}"


def _differential_problems(
    output: str,
    reference: str | None,
    catalog: Catalog,
    *,
    allow_ground: bool,
) -> list[str]:
    """Checks both `output` and optional `reference`, collecting every problem."""
    problems: list[str] = []
    output_issue = _output_problem(output, catalog, allow_ground=allow_ground)
    if output_issue is not None:
        problems.append(output_issue)
    if reference is None:
        return problems
    reference_issue = _output_problem(reference, catalog, allow_ground=allow_ground)
    if reference_issue is not None:
        problems.append(reference_issue)
    if output_issue is None and reference_issue is None:
        output_nature = (
            "voltage" if output == catalog.ground else catalog.unknowns[output].nature
        )
        reference_nature = (
            "voltage"
            if reference == catalog.ground
            else catalog.unknowns[reference].nature
        )
        if output_nature != "voltage" or reference_nature != "voltage":
            problems.append(
                f"an output over a node is a voltage, and `{output}` over "
                f"`{reference}` is not"
            )
    return problems


def _dc_parameter_sweep_problems(
    analysis: DcSweep, parameter: str, catalog: Catalog
) -> tuple[list[str], DimensionError | None]:
    """Checks a parameter-targeted `DcSweep` instance, parameter name, and dimension."""
    if analysis.source not in catalog.devices:
        close = close_names(catalog.device_paths, analysis.source)
        return [f"there is no instance at `{analysis.source}`{close}"], None
    parameters = catalog.parameters(analysis.source)
    if parameter not in parameters:
        close = close_names(parameters, parameter)
        message = f"`{analysis.source}` writes no number `{parameter}` to sweep{close}"
        return [message], None
    if analysis.values:
        instance = catalog.devices[analysis.source]
        label = f"{analysis.source} {parameter}"
        try:
            devices.convert_parameter(instance, parameter, analysis.values[0], label)
        except DimensionError as error:
            return [str(error)], error
        except KeyError:
            pass
    return [], None


def _dc_sweep_problems(
    analysis: DcSweep, catalog: Catalog
) -> tuple[list[str], DimensionError | None]:
    """Checks `DcSweep` source, optional parameter, and value dimension."""
    if analysis.parameter is not None:
        return _dc_parameter_sweep_problems(analysis, analysis.parameter, catalog)
    source_issue = _source_problem(analysis.source, catalog)
    if source_issue is not None:
        return [source_issue], None
    device = catalog.devices[analysis.source]
    expected = (
        CURRENT
        if isinstance(device.of, ir.DeviceOf) and device.of.model == "current_source"
        else VOLTAGE
    )
    if analysis.values and (
        not isinstance(analysis.values[0], Quantity)
        or analysis.values[0].dimension != expected
    ):
        message = (
            f"a sweep level takes a quantity in {expected}, not {analysis.values[0]}"
        )
        return [message], DimensionError(message)
    return [], None


def _stability_problems(analysis: Stability, catalog: Catalog) -> list[str]:
    """Checks that `Stability.probe` is a voltage source off ground."""
    probes_list = f"; probes: {', '.join(catalog.probes)}" if catalog.probes else ""
    if analysis.probe not in catalog.devices:
        candidates = catalog.probes or catalog.voltage_sources or catalog.device_paths
        close = close_names(candidates, analysis.probe) or close_names(
            catalog.device_paths, analysis.probe
        )
        return [f"there is no instance at `{analysis.probe}`{close}{probes_list}"]
    if analysis.probe not in catalog.probes:
        return [f"`{analysis.probe}` is not a floating voltage source{probes_list}"]
    return []


def _transfer_function_output_problems(
    analysis: TransferFunction, catalog: Catalog
) -> list[str]:
    """Checks the nature and reference compatibility of `analysis.output`."""
    nature = catalog.unknowns[analysis.output].nature
    if nature == "voltage":
        if (
            analysis.reference is not None
            and analysis.reference in catalog.unknowns
            and catalog.unknowns[analysis.reference].nature != "voltage"
        ):
            message = (
                f"an output over a node is a voltage, and `{analysis.output}` "
                f"over `{analysis.reference}` is not"
            )
            return [message]
        return []
    if nature == "current":
        problems: list[str] = []
        if analysis.reference is not None:
            problems.append(
                f"a transfer function's current output, `{analysis.output}`, "
                "has no reference"
            )
        source_name = (
            analysis.output.removesuffix(":current")
            if analysis.output.endswith(":current")
            else ""
        )
        if source_name not in catalog.voltage_sources:
            problems.append(
                "a transfer function's current output is a voltage source's, "
                f"not `{analysis.output}`"
            )
        return problems
    message = (
        "a transfer function's output is a node voltage or a voltage source's "
        f"current, not `{analysis.output}`"
    )
    return [message]


def _transfer_function_problems(
    analysis: TransferFunction, catalog: Catalog
) -> list[str]:
    """Checks `TransferFunction` source, output, and optional reference."""
    problems: list[str] = []
    if (source_issue := _source_problem(analysis.source, catalog)) is not None:
        problems.append(source_issue)
    output_issue = _output_problem(analysis.output, catalog, allow_ground=False)
    if output_issue is not None:
        problems.append(output_issue)
    if (
        analysis.reference is not None
        and (
            reference_issue := _output_problem(
                analysis.reference, catalog, allow_ground=False
            )
        )
        is not None
    ):
        problems.append(reference_issue)
    if output_issue is None:
        problems.extend(_transfer_function_output_problems(analysis, catalog))
    return problems


def parameter_pair_problems(
    pairs: Iterable[tuple[str, str | None]], catalog: Catalog
) -> list[str]:
    """Checks deduplicated `(device_path, parameter)` pairs against `catalog`.

    Args:
        pairs: Iterable of `(device_path, parameter)` references to validate.
        catalog: Static circuit catalog indexing devices and parameter paths.

    Returns:
        Diagnostic messages for missing devices or unrecognized parameter names.
    """
    problems: list[str] = []
    missing_devices: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for path, parameter in pairs:
        if path not in catalog.devices:
            if path not in missing_devices:
                missing_devices.add(path)
                close = close_names(catalog.device_paths, path)
                problems.append(f"no device at `{path}`{close}")
            continue
        if parameter is None or (path, parameter) in seen_pairs:
            continue
        seen_pairs.add((path, parameter))
        parameters = catalog.parameters(path)
        if parameter not in parameters:
            close = close_names(parameters, parameter)
            problems.append(f"`{path}` writes no number `{parameter}`{close}")
    return problems


def _override_pairs(
    overrides: Mapping[str, Mapping[str, object]],
) -> list[tuple[str, str | None]]:
    """Extracts `(instance, parameter)` pairs from a nested override mapping."""
    pairs: list[tuple[str, str | None]] = []
    for path, named in overrides.items():
        if not named:
            pairs.append((path, None))
        else:
            pairs.extend((path, parameter) for parameter in named)
    return pairs


def variant_pairs(variants: Variants) -> list[tuple[str, str | None]]:
    """Collects every `(instance, parameter)` pair referenced by `variants`.

    Args:
        variants: Parameter variation specification to traverse.

    Returns:
        Ordered `(instance, parameter)` references extracted from `variants`.
    """
    match variants:
        case MonteCarlo():
            return []
        case Grid(axes=axes):
            return [
                (axis.instance, axis.parameter)
                for axis in axes
                if axis.instance is not None and axis.parameter is not None
            ]
        case Explicit(variants=listed):
            return [
                pair
                for variant in listed
                for pair in _override_pairs(variant.overrides)
            ]
        case Product(first=first, second=second):
            return [*variant_pairs(first), *variant_pairs(second)]


def _check_variant_value(
    path: str,
    parameter: str,
    value: ParameterValue,
    catalog: Catalog,
) -> DimensionError | None:
    """Checks `value` against `catalog`'s declared dimension for `(path, parameter)`."""
    instance = catalog.devices.get(path)
    if instance is None or parameter not in catalog.parameters(path):
        return None
    try:
        devices.convert_parameter(instance, parameter, value, f"{path} {parameter}")
    except DimensionError as error:
        return error
    except KeyError:
        return None
    return None


def _explicit_dimension_errors(
    variants: Explicit, catalog: Catalog
) -> list[DimensionError]:
    """Collects deduplicated parameter-dimension errors across an `Explicit` batch."""
    found: list[DimensionError] = []
    seen: set[str] = set()
    for variant in variants.variants:
        for path, named in variant.overrides.items():
            for parameter, value in named.items():
                error = _check_variant_value(path, parameter, value, catalog)
                if error is not None and str(error) not in seen:
                    seen.add(str(error))
                    found.append(error)
    return found


def _variant_dimension_errors(
    variants: Variants, catalog: Catalog
) -> list[DimensionError]:
    """Collects every parameter-dimension error across `variants` against `catalog`."""
    match variants:
        case MonteCarlo():
            return []
        case Grid(axes=axes):
            return [
                error
                for axis in axes
                if axis.instance is not None
                and axis.parameter is not None
                and axis.values
                and (
                    error := _check_variant_value(
                        axis.instance, axis.parameter, axis.values[0], catalog
                    )
                )
                is not None
            ]
        case Explicit():
            return _explicit_dimension_errors(variants, catalog)
        case Product(first=first, second=second):
            return [
                *_variant_dimension_errors(first, catalog),
                *_variant_dimension_errors(second, catalog),
            ]


def variant_problems(
    variants: Variants, catalog: Catalog
) -> tuple[list[str], DimensionError | None]:
    """Checks all device names, parameter names, and dimensions in `variants`.

    Args:
        variants: Parameter variation specification to check against `catalog`.
        catalog: Static circuit catalog indexing devices and parameter paths.

    Returns:
        Pair of diagnostic messages and optional sole `DimensionError`.
    """
    problems = parameter_pair_problems(variant_pairs(variants), catalog)
    dimension_errors = _variant_dimension_errors(variants, catalog)
    problems.extend(str(error) for error in dimension_errors)
    sole = (
        dimension_errors[0]
        if len(problems) == 1 and len(dimension_errors) == 1
        else None
    )
    return problems, sole


def _mismatch_variable_pairs(
    variables: Sequence[ir.RandomVariable],
) -> list[tuple[str, str | None]]:
    """Collects every `(instance, parameter)` derivative target in `variables`."""
    return [
        pair for variable in variables for pair in _override_pairs(variable.derivatives)
    ]


def _transient_problems(analysis: Transient, catalog: Catalog) -> list[str]:
    """Checks `Transient` initial conditions and logic-circuit compatibility."""
    problems = nodeset_problems(
        analysis.initial_conditions, "initial condition", catalog
    )
    if not analysis.operating_point and catalog.has_logic_devices:
        problems.append(
            "a transient from its initial conditions alone of a circuit "
            "with logic, whose levels only an operating point settles"
        )
    return problems


def _noise_problems(analysis: Noise, catalog: Catalog) -> list[str]:
    """Checks `Noise` output, reference, and optional referred source."""
    problems = _differential_problems(
        analysis.output, analysis.reference, catalog, allow_ground=True
    )
    if (
        analysis.source is not None
        and (issue := _source_problem(analysis.source, catalog)) is not None
    ):
        problems.append(issue)
    return problems


def _pole_zero_problems(analysis: PoleZero, catalog: Catalog) -> list[str]:
    """Checks `PoleZero` optional source, output, and reference."""
    problems: list[str] = []
    if (
        analysis.source is not None
        and (issue := _source_problem(analysis.source, catalog)) is not None
    ):
        problems.append(issue)
    if analysis.output is not None:
        problems.extend(
            _differential_problems(
                analysis.output,
                analysis.reference,
                catalog,
                allow_ground=True,
            )
        )
    return problems


def _stability_output_problem(name: str) -> str | None:
    """Returns a problem string when `name` is not the stability loop-gain output."""
    if name == LOOP_GAIN:
        return None
    close = close_names((LOOP_GAIN,), name)
    return (
        f"a stability analysis has only the `{LOOP_GAIN}` output, not `{name}`{close}"
    )


def _differentiated_problems(
    analysis: Sensitivity | Mismatch,
    catalog: Catalog,
    record_checker: Callable[[Sequence[str]], list[str]],
    *,
    has_logic_nets: bool,
) -> tuple[list[str], DimensionError | None]:
    """Checks `Sensitivity` or `Mismatch` outputs, random variables, and nested `of`."""
    if isinstance(analysis.of, Stability):
        problems = [
            issue
            for output in analysis.outputs
            if (issue := _stability_output_problem(output)) is not None
        ]
    else:
        problems = [
            issue
            for output in analysis.outputs
            if (issue := _output_problem(output, catalog, allow_ground=False))
            is not None
        ]
    if isinstance(analysis, Mismatch):
        problems.extend(
            parameter_pair_problems(
                _mismatch_variable_pairs(analysis.random_variables), catalog
            )
        )
    dimension_error: DimensionError | None = None
    if analysis.of is not None:
        nested, dimension_error = analysis_problems(
            analysis.of,
            catalog,
            record_checker,
            has_logic_nets=has_logic_nets,
        )
        problems.extend(nested)
    return problems, dimension_error


def _simple_analysis_problems(
    analysis: Analysis, catalog: Catalog, *, has_logic_nets: bool
) -> list[str]:
    """Checks analyses other than `DcSweep`, `Sensitivity`, and `Mismatch`."""
    match analysis:
        case Ac():
            issue = _source_problem(analysis.source, catalog)
            found = [] if issue is None else [issue]
        case Transient():
            found = _transient_problems(analysis, catalog)
        case PeriodicSteadyState():
            found = (
                ["a periodic steady state is found for circuits without logic"]
                if has_logic_nets
                else []
            )
        case Stability():
            found = _stability_problems(analysis, catalog)
        case Noise():
            found = _noise_problems(analysis, catalog)
        case TransferFunction():
            found = _transfer_function_problems(analysis, catalog)
        case PoleZero():
            found = _pole_zero_problems(analysis, catalog)
        case _:
            found = []
    return found


def analysis_problems(
    analysis: Analysis,
    catalog: Catalog,
    record_checker: Callable[[Sequence[str]], list[str]],
    *,
    has_logic_nets: bool,
) -> tuple[list[str], DimensionError | None]:
    """Collects every static problem checking `analysis` against `catalog`.

    Args:
        analysis: Analysis specification to validate statically.
        catalog: Static circuit catalog indexing devices, sources, and unknowns.
        record_checker: Callback validating recorded instance patterns.
        has_logic_nets: Flag indicating presence of discrete logic nets in the circuit.

    Returns:
        Pair of diagnostic strings and optional DC sweep `DimensionError`.
    """
    problems: list[str] = []
    if isinstance(analysis, Recorded) and analysis.record is not None:
        problems.extend(record_checker(analysis.record))
    if isinstance(analysis, DcSweep):
        found, dimension_error = _dc_sweep_problems(analysis, catalog)
        return [*problems, *found], dimension_error
    if isinstance(analysis, Sensitivity | Mismatch):
        found, dimension_error = _differentiated_problems(
            analysis, catalog, record_checker, has_logic_nets=has_logic_nets
        )
        return [*problems, *found], dimension_error
    problems.extend(
        _simple_analysis_problems(analysis, catalog, has_logic_nets=has_logic_nets)
    )
    return problems, None


def analyses_problems(
    analyses: Mapping[str, Analysis],
    catalog: Catalog,
    record_checker: Callable[[Sequence[str]], list[str]],
    *,
    has_logic_nets: bool,
) -> tuple[list[str], DimensionError | None]:
    """Collects static problems across named `analyses`, prefixed by analysis name.

    Args:
        analyses: Mapping from analysis names to specifications to validate.
        catalog: Static circuit catalog indexing devices, sources, and unknowns.
        record_checker: Callback validating recorded instance patterns.
        has_logic_nets: Flag indicating presence of discrete logic nets in the circuit.

    Returns:
        Pair of diagnostic messages prefixed with each offending analysis name and
        optional sole `DimensionError`.
    """
    problems: list[str] = []
    dimension_errors: list[DimensionError] = []
    for name, analysis in analyses.items():
        found, dimension_error = analysis_problems(
            analysis, catalog, record_checker, has_logic_nets=has_logic_nets
        )
        problems.extend(f"analysis `{name}`: {issue}" for issue in found)
        if dimension_error is not None:
            dimension_errors.append(dimension_error)
    sole = (
        dimension_errors[0]
        if len(problems) == 1 and len(dimension_errors) == 1
        else None
    )
    return problems, sole


def noise_unit_unknown(analysis: Noise, catalog: Catalog) -> bool:
    """Returns whether the output or input-referred noise unit is undetermined.

    Args:
        analysis: Noise analysis specification under validation.
        catalog: Static circuit catalog indexing unknowns and independent sources.

    Returns:
        `True` when the output unknown or referred input source cannot be resolved.
    """
    output_known = (
        analysis.output in catalog.unknowns or analysis.output == catalog.ground
    )
    if not output_known:
        return True
    return (
        analysis.source is not None
        and analysis.source not in catalog.independent_sources
    )


def fidelity_scope_problems(
    scopes: Mapping[str, object], component_paths: tuple[str, ...]
) -> list[str]:
    """Collects every fidelity scope that matches no component instance.

    Args:
        scopes: Mapping keyed by fidelity scope paths or glob patterns.
        component_paths: Hierarchical paths of all component instances in the circuit.

    Returns:
        Diagnostic messages for fidelity scopes matching no component instance.
    """
    problems: list[str] = []
    for scope in scopes:
        if any(PurePosixPath(path).full_match(scope) for path in component_paths):
            continue
        has_wildcard = not WILDCARDS.isdisjoint(scope)
        close = "" if has_wildcard else close_names(component_paths, scope)
        problems.append(
            f"the fidelity scope {scope!r} matches no component instance{close}"
        )
    return problems


def trace_problems(traces: Sequence[str], valid_names: tuple[str, ...]) -> list[str]:
    """Collects every trace in `traces` that is neither an unknown nor a logic net.

    Args:
        traces: Requested trace names to check.
        valid_names: All circuit unknown names and discrete logic net names.

    Returns:
        Diagnostic messages for unrecognized trace names.
    """
    valid = frozenset(valid_names)
    problems: list[str] = []
    for trace in dict.fromkeys(traces):
        if trace not in valid:
            close = close_names(valid_names, trace)
            problems.append(
                f"trace `{trace}` names no unknown or logic net of the circuit{close}"
            )
    return problems


def record_problems(
    record: Sequence[str], instances: tuple[str, ...], known: frozenset[str]
) -> list[str]:
    """Collects unknown instance paths and unmatched wildcard patterns in `record`.

    Args:
        record: Exact paths, glob patterns, or ``!``-negated exclusions to check.
        instances: Ordered recordable instance paths in the circuit.
        known: Set of all recordable instance paths for fast membership checks.

    Returns:
        Diagnostic messages for missing instances and unmatched patterns.
    """
    problems: list[str] = []
    for pattern in record:
        body = pattern.removeprefix("!")
        if WILDCARDS.isdisjoint(body):
            if body not in known:
                hint = close_names(instances, body)
                problems.append(
                    f"the circuit has no instance named {body!r} to record{hint}"
                )
        elif not any(PurePosixPath(path).full_match(body) for path in instances):
            problems.append(f"the record's pattern {body!r} matches no instance")
    return problems


def resolve_record(
    record: tuple[str, ...], instances: tuple[str, ...], known: frozenset[str]
) -> frozenset[str]:
    """Resolves the set of instance paths selected by `record`.

    Args:
        record: Ordered exact paths, glob patterns, and ``!``-negated exclusions.
        instances: Ordered recordable instance paths in the circuit.
        known: Set of all recordable instance paths in the circuit.

    Returns:
        Selected instance paths after applying inclusions and exclusions in order.

    Raises:
        PlanError: Lists every unknown instance path or unmatched wildcard pattern.
    """
    problems = record_problems(record, instances, known)
    if problems:
        raise PlanError(problems)
    named: set[str] = set()
    for pattern in record:
        body = pattern.removeprefix("!")
        if WILDCARDS.isdisjoint(body):
            matched = {body}
        else:
            matched = {
                path for path in instances if PurePosixPath(path).full_match(body)
            }
        if pattern.startswith("!"):
            named -= matched
        else:
            named |= matched
    return frozenset(named)
