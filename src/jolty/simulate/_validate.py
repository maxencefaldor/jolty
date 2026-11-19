# Copyright 2026 Maxence Faldor
"""Static validation of analyses, plans, records, variants, and traces."""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from jolty import ir, results
from jolty.errors import DimensionError, PlanError, close_names
from jolty.plan import (
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
    overrides: Mapping[str, Mapping[str, float]]

    def parameters(self, path: str) -> tuple[str, ...]:
        """Returns every real-valued parameter path defined on the device at `path`."""
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
    overrides: Mapping[str, Mapping[str, float]],
    joined: Callable[[str, str], Joined | None],
) -> Catalog:
    """Builds the static lookup catalog for an elaborated circuit."""
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
    """Reports process corners on `plan` when the circuit has no foundry devices."""
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
    temperature: float,
    has_foundry: bool,
    check_temperature: bool,
) -> list[str]:
    """Checks `plan`'s temperature and process corners against a simulation."""
    problems: list[str] = []
    if check_temperature and plan.temperature is not None:
        target = plan.temperature.to(TEMPERATURE, "a plan's temperature")
        if target != temperature:
            problems.append(
                f"the plan's temperature, {plan.temperature}, is not the "
                f"simulation's, {Quantity(temperature, TEMPERATURE)}"
            )
    problems.extend(process_problems(plan, has_foundry=has_foundry))
    return problems


def nodeset_problems(
    mapping: Mapping[str, Quantity], what: str, catalog: Catalog
) -> list[str]:
    """Collects every key in `mapping` that is not a circuit voltage net."""
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


def _dc_sweep_problems(
    analysis: DcSweep, catalog: Catalog
) -> tuple[list[str], DimensionError | None]:
    """Checks `DcSweep` source, optional parameter, and value dimension."""
    if analysis.parameter is not None:
        if analysis.source not in catalog.devices:
            close = close_names(catalog.device_paths, analysis.source)
            return [f"there is no instance at `{analysis.source}`{close}"], None
        parameters = catalog.parameters(analysis.source)
        if analysis.parameter not in parameters:
            close = close_names(parameters, analysis.parameter)
            message = (
                f"`{analysis.source}` writes no number `{analysis.parameter}` "
                f"to sweep{close}"
            )
            return [message], None
        return [], None
    source_issue = _source_problem(analysis.source, catalog)
    if source_issue is not None:
        return [source_issue], None
    device = catalog.devices[analysis.source]
    expected = (
        CURRENT
        if isinstance(device.of, ir.DeviceOf) and device.of.model == "current_source"
        else VOLTAGE
    )
    if analysis.values and analysis.values[0].dimension != expected:
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
        return [f"`{analysis.probe}` is not a voltage source off ground{probes_list}"]
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
    """Checks deduplicated `(device_path, parameter)` pairs against `catalog`."""
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
    overrides: Mapping[str, Mapping[str, float]],
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
    """Collects every `(instance, parameter)` pair referenced by `variants`."""
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


def _differentiated_problems(
    analysis: Sensitivity | Mismatch,
    catalog: Catalog,
    record_checker: Callable[[Sequence[str]], list[str]],
    *,
    has_logic_nets: bool,
) -> tuple[list[str], DimensionError | None]:
    """Checks `Sensitivity` or `Mismatch` outputs, random variables, and nested `of`."""
    problems = [
        issue
        for output in analysis.outputs
        if (issue := _output_problem(output, catalog, allow_ground=False)) is not None
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
    """Collects every static problem checking `analysis` against `catalog`."""
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
) -> list[str]:
    """Collects static problems across named `analyses`, prefixed by analysis name."""
    problems: list[str] = []
    for name, analysis in analyses.items():
        found, _ = analysis_problems(
            analysis, catalog, record_checker, has_logic_nets=has_logic_nets
        )
        problems.extend(f"analysis `{name}`: {issue}" for issue in found)
    return problems


def noise_unit_unknown(analysis: Noise, catalog: Catalog) -> bool:
    """Returns whether the output or input-referred noise unit is undetermined."""
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
    """Collects every fidelity scope that matches no component instance."""
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
    """Collects every trace in `traces` that is neither an unknown nor a logic net."""
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
    """Collects unknown instance paths and unmatched wildcard patterns in `record`."""
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
