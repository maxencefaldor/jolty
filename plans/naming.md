# Naming glossary

The owner's rule (`AGENTS.md`, "Names", on `coverage` at `9a78eb2`): each concept has one name everywhere, and each name denotes one concept; names stay uniform, consistent, symmetric, formal, strict, and principled.
As the first step of `quality.md` item 12, this glossary records every concept, its proposed canonical name, definition, established term, and current aliases with locations.
The glossary proposes names without applying renames; the owner reviews all proposals first.

The inventory was read from `main` at `5f69050` and pending branches: `coverage` (`9a78eb2`) and `xyce` (pre-rebase tip `e44555f`, merged via `6047a2d` to `dd18fa0`).
Uncommitted work on `chgtol` and `regions` added no public names referenced by the glossary.
Written 15 November 2026.

Status (18 November 2026): every owner-approved rename landed on `main` (`8a4d80a` to `bc35702`, `quality.md` 12), so "today" below refers to 15 November; on 20 November the owner kept `measure::Signal`, the last open name.

## The owner's decisions, 15 November

The owner answered each choice below (section "Choices the owner may disagree with"); every other row's proposal stands.

| Choice | Decision |
| --- | --- |
| 1. component | Kept: "component", not subcircuit or module. |
| 2. `Instance.id` | Renamed `name`, in the API and the circuit file, whose version moves. |
| 3. pin | Three words, each qualified: terminal, logic pin, part pin; code keeps today's names. |
| 4. an unknown's kind | `Nature`, Verilog-AMS's term. |
| 5. results | The `XResult` family: analysis `X`, result `XResult`, report `XReport`, kind `x`, `read_x`, in every layer. |
| 6. periodic steady state | Dedicated analysis `PeriodicSteadyState(period=...)`, kind `periodic_steady_state`, matching ngspice `.pss`, Spectre `pss`, and Xyce `.HB`. |
| 7. level | Rules keep `level` (matching rustc, ty, and SARIF), while source values use `value` (`Dc(value)`, `DcSweep(source, values)`). A DC sweep result names the axis `swept`, since `values` holds unknowns and the axis may sweep temperature or a parameter. |
| 8. `Parallelism`'s grain sizes | One family with definitions fixed: `min_solves_per_task`, `min_instances_per_task` a true least size, `min_unknowns_per_task` per task; `threads` becomes `max_threads`. The break-evens are retaken on machine A, with `speed.md` item 4. |
| 9. `max_unknowns_per_interior` | Stays in `Parallelism`, its name kept: the owner reversed the move once the rule that `Parallelism` changes no bit left `AGENTS.md`. |
| 10. fingerprint, hash, sha256 | `digest` everywhere. |
| 11. average and mean | `mean` for both. |
| 12. `checks.Report` | `CheckReport`. |
| 13. rule names | `low-phase-margin`, `low-gain-margin` and, instead of the glossary's `over-derating`, `near-rating`, which pairs with `over-rating`. |
| 14. `kicad` | `export-kicad`, `export_kicad`, `ExportKicadRequest`, MCP `export_kicad_netlist`. |
| 15. `fidelity.Model` | `CharacterizedModel`. |
| 16. BSIM3's instance arguments | Spelled out: `threshold_adjustment`, `mobility_multiplier`, `mobility_model`, `capacitance_model`, `noise_model`, `check_parameters`, `charge_partition`; the card keeps BSIM3's names. |
| 17. folders | `benchmarks/<suite>`: `speed`, `precision`, `robustness`, `coverage`, `agents`. |
| 18. a suite's circuit | `fixture`. |
| 19. alternate constructors | Use `from_*` everywhere per Python and Rust conventions, overriding the glossary proposal: every `of` becomes `from_<input>`. |
| `elaborate::Composite` | `SummedTerminal`, `Recording.summed_terminals` (the owner, 16 November): a terminal of a component instance or a logic device whose current sums the device terminals behind it. |
| 20. kept as they are | Not asked: `gmin`, `Explicit`, `held_at` and `warmer_by`, counts as plural nouns stay. |
| 21. `builder.Point` | `Connectable` (the owner). |
| 22. `variation.Lot` | `LotMember` (the owner). |
| 23. `MonteCarloReport.passing` | `yield_` (the owner); where it is serialized, its JSON key stays `yield`, through an alias. |
| 24. `Traced.axis` | Kept (the owner); only `Traced` becomes `Traces`. |
| 25. `specs.Native` | Public, renamed `ResolvedMeasure` (a measure resolved for the engine against one result kind), with `Measure.native` renamed `Measure.resolved`. Lead decision, replacing private `_Native` to avoid cross-module private imports. |
| 26. `Observation`, `Observations`, `Observed` | `RecordedValues` (one solution), `RecordedSeries` (every point, end to end) and `RecordedView<'a>` (a view of one point), in the module `record`; Python's `results.Observed` becomes `RecordedValues` (the lead). |
| 27. `batch::Outcome` | `Run`, once `ac.rs`'s private `Run<R>` is named after what it holds, a run of frequencies (the lead). |
| 28. the kernels' short type names | The model's full name, such as `VoltageControlledVoltageSourceShape` (the lead). |
| 29. `UnknownInfo` and `Unknown` | `Unknown` and `UnknownIndex`, in two steps: `Unknown` to `UnknownIndex` first, then `UnknownInfo` to `Unknown` (the lead). |
| 30. `of` that is not a constructor | Choice 19 applies to alternate constructors only; an `of` that is an accessor keeps its name (the lead). |
| 31. `ngspice.spice_celsius`, `xyce.xyce_celsius` | `option_temperature` in each module, the value each simulator's `.options temp` must be given (the lead), since `jolty.units.celsius` names the opposite conversion. |

## How to read it

Each concept appears once, in the section of its domain, as a row: the **proposed name** in bold, a definition in one sentence, the established term it follows, and every other name it has today with where.
A row whose "Today" column lists "one name" needs no rename.
Places are written `module.Class.field` for Python, `crate::Type::field` for Rust, `--flag` for the command line, `tool(argument)` for the MCP server, and `"key"` for a file's key.

Site counts are the lines `git grep -P` finds for the old name, word-bounded, over the repository's sources, tests, docs, examples, benchmarks and scripts, without `CHANGELOG.md`, lock files, `perf/results/` and binary archives.
Names matching common words (`source`, `level`, `id`) are renamed by type rather than raw text.
Each count reports typed sites first, followed by the larger raw text match count.

## Coverage of the inventory

| Layer | What was read | How | Size |
| --- | --- | --- | --- |
| Rust crates | every `pub` and `pub(crate)` item, struct field and enum variant outside `tests/` | a scan of each `.rs` file | 2,787 names in 97 files |
| Python package | every module, class, field, method, function, parameter, constant and type alias, public and private | Python's `ast` over `src/jolty` | 2,814 names in 42 modules |
| Extension stub | every name of `_core.pyi` | read whole | 385 lines |
| Command line and MCP | every command, flag and tool, from the request models of `jolty.service`, which both derive from | read whole | 10 commands, 10 tools, 26 distinct arguments |
| Settings and environment | `jolty.settings.Settings` and every `JOLTY_` variable in the repository | read whole, grep | 9 settings, 11 variables |
| Files | circuit, results, batch, shard and catalog formats: their Pydantic models and serde attributes | read | 5 formats |
| Benchmarks | `perf/bench.py`, `perf/precision`, `perf/robustness`, `perf/coverage` (branch), the Xyce additions (branch), `benchmarks/` (the agent benchmark), `scripts/bench` and `scripts/precision` | `ast`, read | 1,164 names, and 187 more on `coverage` |
| Guide and reference | every heading of `docs/guide` and the reference pages' terms | read | 17 guide pages, 9 reference pages |

Not covered, or covered only for their shape:

- The 28 models' parameter names (about 1,000 fields): checked for the shapes of their families (bounds, temperature coefficients and exponents, SPICE names), not one by one against each model's manual.
- Private helpers' names inside one function or module: read in the inventory, but grouped into concepts only where they leak into a public name or a file.
- The tests' own names, the viewer's JavaScript and CSS, the examples' and the gallery's scripts, and the guide's prose beyond its headings and the terms it defines.
- The Verilog-A-derived kernels' internal variables, which keep their reference's names, as `quality.md` asks the owner to decide.

## Principles the proposals follow

The rules below guide all proposals based on the "Names" convention; each rule remains open to owner override.

1. **Domain term first:** prefer SPICE terms for circuits, analyses, and options, textbook or standards terms (IEC 60063, Verilog-AMS, IEEE 1164) where SPICE has none, and Python or Rust conventions for code.
   Where jolty and established terms differ, the established term is proposed unless ambiguous in jolty.
1. **One spelling across layers:** a Rust field, its Python field, its file key, its command-line flag (kebab-case), and its MCP argument use the same word.
1. **Analyses and outputs:** analysis `X`, result `XResult`, report `XReport`, kind `x` in files and reports, `Analysis::X` in the engine, and `read_x` for measures read from the result.
1. **Bounds:** use `min_x` and `max_x` for option or parameter bounds on `x`, and `minimum` and `maximum` for fields holding a bound or extreme.
   Use `start` and `stop` for independent-variable ranges (time, frequency, sweeps, samples, windows; matching SPICE `tstart`, `tstop`, `fstart`, `fstop`) and `first` and `last` for endpoint values.
   Use `lower` and `upper` for confidence intervals, avoiding `least`, `most`, `fewest`, `lowest`, `highest`, and `limit`.
1. **Pairs:** files are written by `save_x` and read by `load_x`; an operation and its inverse share one stem.
1. **References:** fields referencing an instance hold its path and use `instance`; `path` appears only on records belonging to that instance.
1. **Counts:** a count is a plural noun typed as an integer (`frequencies: int`), and the corresponding result array uses the same plural noun.
1. **Commands:** a CLI command is a verb, the MCP tool is `verb_object`, the service function is the verb, and models are `VerbRequest` and `VerbResult`.
1. **Times measured:** `seconds` denotes wall-clock duration in seconds, and `<what>_seconds` denotes a component phase.
1. **Device parameters:** temperature coefficients and exponents name the scaled parameter in full, `<parameter>_temperature_coefficient`.
1. **Qualified homonyms:** when one word carries two domain meanings, the central concept keeps the bare term and the secondary concept takes a qualifier, such as `logic level` alongside rule `level`.
1. **Names denote meaning, not mechanism:** no name describes computation, storage, or transport, avoiding `Outcome` for FFI payloads, `Raw` for raw-file contents, and `inside` for in-iteration work.

## The glossary

### 1. Circuits and their hierarchy

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **circuit** | The whole hierarchy of components a build produces, with its top, as one file. | SPICE's circuit; a netlist. | One name: `jolty_ir::Circuit`, `jolty.ir.Circuit`, `"jolty-circuit"`. |
| **component** | A circuit block defined once, with ports and instances, placed by name anywhere in the hierarchy; the top level is also a component. | SPICE's subcircuit (`.subckt`), Verilog's module, an IC cell. | `jolty.builder.Component` (the built block), `jolty.ir.Component` (its definition), `jolty_ir::Component`; `composite` for a component instance (`jolty_engine::elaborate::Composite`, `Recording.composites`, `Recorded.composites`). See choice 1. |
| **component instance** | An instance of a component, as opposed to a device instance. | SPICE's subcircuit instance (`X` element). | `composite` in `jolty_engine` and results recording; "component instance" in `AGENTS.md`, the docs, and `_RECORD_DESCRIPTION`. Propose `component_instances` for `Recording.composites` and `Recorded.composites`, and `ComponentInstance` for `elaborate::Composite`. |
| **device** | A leaf of the hierarchy, simulated by one of the library's models. | SPICE's device or element. | One name in the API (`jolty.devices`); see *instance* for `device` fields holding an instance. Separately, `jolty.catalog.Device` (`resistor`, `capacitor`, `inductor`) and `catalog.Part.device` denote a **part** kind: propose `catalog.PartKind` and `Part.kind` (1,563 keys in `benchmarks/cost_down/given/catalog.json`). |
| **device instance** | An instance of a device. | SPICE's instance. | "device instance" in prose; `Simulation.devices`, `CircuitSummary.devices` and `_core.Simulator.devices()` count them, one name; `_core.Simulator.instances()` lists every path a recording can name, devices' and component instances', so its name is right. |
| **instance** | A placement of a device or a component inside a component, under a name unique there. | SPICE's instance. | `jolty.ir.Instance`, `jolty_ir::Instance`; but also `jolty.checks.Instance`, a reference to an elaborated instance by its path and source location, which is **location** below. |
| **name** (of an instance) | The instance's name inside its component, such as `R1`. | SPICE's instance name. | `id`: `ir.Instance.id`, `jolty_ir::Instance::id`, `Builder.place(item, id)`, `Placed.id()`, `"id"` in every circuit file; every other named thing (port, net, component, lot, rule, terminal) has `name`. Propose `name` (choice 2). |
| **path** | An instance's hierarchical name, its components' instance names joined by `/`, such as `X1/R1`. | Hierarchical path, as in Verilog. | `path` appears in instance records (`RecordedDevice.path`, `Location.path`), whereas instance-referencing fields mix `path`, `device`, and `instance` (principle 6). `path` appears in `Excursion.path`, `Varied.path`, and `Toleranced::path`; `device` appears in `jolty_engine::sensitivity::Parameter::device`, `noise::NoiseContribution::device`, and `logic::Excursion::device`. `instance` appears in `results.Parameter.instance`, `NoiseSource.instance`, `report.Derivative.instance`, `NoiseContributor.instance`, `catalog.Line.instance`, and `fidelity.Contract.instance`; propose `instance` for all referencing fields. |
| **model** | The equations a device follows, one of the library's kernels, named such as `diode` or `bsim3`. | SPICE's model. | `jolty.devices.Device.model`, `_core.models()`; but `jolty.ir.DeviceOf.device` and the circuit file's `"of": {"device": ...}` hold the same model name. Propose `DeviceOf.model`, `"model"`. |
| **port** | A connection a component offers to its instances, with a role. | Verilog's port; SPICE's subcircuit node. | One name: `ir.Port`, `Builder.port`, `Component.ports`. |
| **role** | What a port is for: signal, input, output, supply or ground. | none | One name: `ir.Role`, `jolty_ir::Role`. |
| **terminal** | A connection of an instance, a device's or a component instance's, joined to an electrical net. | SPICE's node of an element; a device terminal. | One name in the circuit (`ir.Instance.terminals`, `builder.Terminal`, `Rating.terminals`). `pin` is two other concepts: a logic device's connection to a logic net (`jolty_engine::logic::Endpoint::Pin`, `jolty_models::logic::Pins`, `ElaborationProblem::LogicPinOnElectricalNet`), Liberty's term; and a part's physical pin (`catalog.Part.pins`, `Pins`), KiCad's term. Propose keeping all three, with *logic pin* and *part pin* always qualified in prose (principle 11); see choice 3. |
| **net** | A named connection inside a component, joining terminals and ports. | A netlist's net. | `net` in the API, the files and the docs; `node` in the engine (`jolty_models::binding::Node`, `simulate.Joined.node`) for the unknown a terminal joins after elaboration, which is a distinct concept (**node** below). |
| **node** | An electrical node of the elaborated circuit: the nets of the hierarchy that elaboration joins, with one voltage unknown, or the ground. | SPICE's node. | `binding::Node`, `Joined.node`, the rules' text ("on one node"). One name, once *net* stays the hierarchy's word. |
| **ground** | The reference node, at zero volts, which has no equation. | SPICE's ground, node 0. | `ground` (`Role.GROUND`, the guide's "The top and the ground", `Joined.node` "None on the ground"); `reference node` in `binding::Node`'s documentation. Propose *ground* in every text. |
| **connection point** | A terminal or a net a builder can join. | none | `jolty.builder.Point`; and `jolty.layout.Point`, a coordinate. Propose `builder.Joinable` or `builder.Endpoint` for the first; `layout.Point` keeps the geometric word. |
| **source location** | The file, line and function of the build script that placed an instance or stated a requirement. | A compiler's source location, as C++'s `std::source_location` (file name, line, column, function name). | `Source` appears in `jolty.ir.Source`, `jolty_ir::Source`, `_core.SourceOutcome`, and fields `ir.Instance.source`, `Requirement.source`, `Verdict.source`, `checks.Instance.source`, and `elaborate::Location::source`. Counts are 24 type sites, 18 field sites, and 2,398 text matches (plus the "Provenance" heading in `components.md`). `source` also names independent sources (`Ac.source`, `DcSweep.source`, `Noise.source`, `TransferFunction.source`, `PoleZero.source`, `NoiseInput.source`) and noise sources. Propose `SourceLocation` and `source_location`. |
| **location** | Where an instance is: its path and its source location. | none | `jolty_engine::elaborate::Location`; `jolty.checks.Instance`; `_core.InstanceOutcome`. Three names; propose `Location`. |
| **part** | A real, orderable component: a model with its maker's number, ratings, package and price. | A part, a BOM's line item. | One name (`jolty.parts`, `catalog.Part`, `Instance.part`, `numbered(part)`). |
| **part number** | The maker's number of a part. | Manufacturer part number. | `ir.PartNumber`, `ir.part_number`; `catalog.Part.number`; `parts.Datasheet.number`. One concept, propose `part_number` for both fields. |
| **elaboration** | Turning the hierarchy into the flat circuit the engine simulates, once per structure. | Elaboration, as in VHDL and Verilog. | One name (`Elaborated`, `ElaborationError`, `elaborate.rs`). |
| **structure** | What of an elaborated circuit its parameters cannot change: its unknowns, its matrix's pattern and its devices' shapes. | none | `Structure`, `StructureKey`, `structure_of`, `restructured`, `batch::Outcome::structure`. One name. |

### 2. What instances carry: parameters, tolerances, ratings, overrides

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **parameter** | A named number or setting of a device instance, which its model reads, such as `resistance`. | SPICE's instance and model parameters. | One name (`Instance.parameters`, `Device.parameters`, `jolty_ir::Parameters`, the kernels' `XParameters`); `card` for BSIM3's model parameters (`bsim3(card=...)`, `BSIM3_CARD`), SPICE's model card, a distinct concept kept. |
| **parameter reference** | A parameter named by its instance and its name, with its nominal value, as a sensitivity differentiates against it. | none | `results.Parameter(instance, name, value)`, `jolty_engine::sensitivity::Parameter { device, name, value }`, `report.Derivative(instance, parameter, ...)`. Propose `instance` and `name` in all three. |
| **tolerance** (of a parameter) | How a parameter's value spreads from part to part: its relative bound, its distribution, and the lot it shares. | A component's tolerance. | `ir.Tolerance`, `jolty_ir::Tolerance`, `Device.toleranced(...)`, `resistor(tolerance=...)`, `catalog.Part.tolerance`; one name. The options' numerical tolerances are always compounds (`relative_tolerance`), SPICE's `RELTOL` and kin, so they do not collide. |
| **toleranced parameter** | A parameter that has a tolerance, with its instance, nominal value and tolerance. | none | `jolty.variation.Varied(path, parameter, nominal, tolerance)` and `jolty_engine::variants::Toleranced { path, name, nominal, tolerance }`, one concept under two names and two field sets; `_core.toleranced(circuit)` returns them as tuples. Propose `Toleranced(instance, parameter, nominal, tolerance)` in both. |
| **lot** | A group of toleranced parameters that move together, as parts from one manufacturing lot. | A manufacturing lot. | `ir.Lot`, `jolty_ir::Lot`; `jolty.variation.Lot`, a `(path, lot name)` pair, under the same name. Propose `variation.LotMember` for the pair, or no alias. |
| **rating** | A bound a part's maker sets on one stress of a device. | A part's absolute maximum rating. | One name (`Rating`, `rated(...)`, `rate(...)`, `Stressed.rating`). |
| **stress** | What a rating bounds: a voltage between terminals, a current into one, a power or a temperature. | Electrical stress, as in derating standards. | One name (`Stress`, `Rating.stress`). |
| **override** | A parameter's value that a run replaces the circuit's with, by instance path and parameter name. | none (SPICE's `.alter`, `.param` redefinition). | One name (`overrides` everywhere, `UnknownOverride`). |
| **substitute** | A device model a run places instead of an instance's own, such as a characterized logic model for a component. | none | `substitutes`, `Elaborated.substituted`, `UnknownSubstitute`; `_stand_ins` privately in `simulate.py`. Propose `_substitutes` there. |
| **temperature** | The circuit temperature at which every device evaluates unless assigned a dedicated device temperature. | SPICE's `TEMP`. | One name; per-device overrides use `Device.held_at(temperature)` (SPICE `TEMP`) or `Device.warmer_by(offset)` (SPICE `DTEMP`), and thermal-network ambient uses `_Terminals.ambient`. |
| **nominal temperature** | The temperature at which a model's parameters were measured. | SPICE's `TNOM`. | One name (`nominal_temperature`). |
| **process** | A foundry's process corner, by its library section's name, such as `tt`. | A process corner. | One name (`Plan.process`, `Corner.process`, `Conditions.process`, `Fidelity.process`). |
| **mismatch** | The random variation of a foundry's devices from one to the next, drawn from a seed. | Device mismatch. | `mismatch` (a seed in `Conditions.mismatch`, `Simulation(mismatch=)`), `mismatched` (a count in `MonteCarloReport`, `Shard`); one stem. |

### 3. Unknowns, and what results keep of devices

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **unknown** | A variable the circuit's equations are solved for, by name, with its nature. | MNA's unknowns. | `jolty.unknowns.Unknown(name, quantity)` in Python is `jolty_models::binding::UnknownInfo { name, quantity }` in Rust, while Rust's `binding::Unknown` is the unknown's index. Propose `Unknown` for the named one in both languages, and `UnknownIndex` for the index; `Info` adds no meaning (principle 12). |
| **nature** | What an unknown is: a voltage, a current, a temperature or a state. | Verilog-AMS's nature (a discipline's potential or flow). | `quantity` appears in `jolty.unknowns.Quantity`, `Unknown.quantity`, `jolty_models::binding::Quantity` (including `Displacement`), and `"quantity"` keys in result files (216 Python sites, 83 Rust). Affected classes include `Sweep`, `NoiseInput`, `NoiseSpectrum`, `Traced`, `report.Value`, `Extent`, `Solutions`, `SweepReport`, `NoiseReport` (`referred_quantity`), and `OutputSensitivity`. `jolty.units.Quantity` denotes a dimensioned number (`5 * V`), and `perf/precision/fixtures.Quantity` is a third concept. Propose `Nature` (choice 4). |
| **internal unknown** | An unknown a device adds inside itself, such as a series resistance's inner node. | SPICE's internal node. | One name (`internals`, `first_internal`). |
| **record** | Instances whose currents, charges, derivatives, and variables are saved alongside circuit unknowns. | ngspice's `.save` of device vectors (`@m1[id]`). | One name for the option (`Plan.record`, `jolty.plan.Recorded.record`, `Options::record`, `--record`). Recorded metadata is indexed by `jolty.results.Recording` in Python and `jolty_engine::observe::Recorded` in Rust. Values use `Observation`, `Observations`, and `Observed` in Rust (`observe` module), and `Observed` in Python and the stub. Propose `Recording` for the index, `RecordedValues` for values, and module `record` retiring `observe`; `jolty.plan.Recorded` keeps its name. |
| **recorded device** | One recorded instance: its path, terminals, internal unknowns and variables. | none | `results.RecordedDevice`, `files._Device`. One name. |
| **variable** | An internal quantity maintained by a device beyond terminal currents and charges, such as memristor state or junction temperature rise. | none (a model's internal variable). | `variables` in Rust (`Observation::variables`), the stub, `RecordedDevice.variables`, and `Recording.variable(instance, name)`; Python results use `readings` (`OperatingPoint.readings`, `Sweep.readings`, `Waveforms.readings`, `Observed.readings`, 54 sites), colliding with `jolty.specs.Reading`. Propose `variables`. |
| **conductance**, **capacitance** | The derivatives of a recorded device's currents and charges with respect to the unknowns. | SPICE's small-signal conductances and capacitances. | One name each (`conductances`, `capacitances`, `Recording.derivative(...)`). |
| **trace** | The values one unknown takes along an analysis, which a plot draws or a batch keeps from each run. | A waveform trace, as on an oscilloscope; PSpice's probe. | `probes` on the command line and in plots (`--probes`, `SimulateRequest.probes`, `PlotRequest.probes`, `runner.Job.probes`, `plots.default_probes`, `plot(probes=...)`, 45 sites); `traced` in the Python API (`Simulation.batch(traced=...)`, `BatchResult.traced`, `results.Traced`, `files._Traced`, 170 sites). `probe` is also the stability analysis's probe source, and the coverage suite's word for a fixture. Propose `traces` for the argument and the flag, `Traces` for the class. |
| **logic trace** | The levels one logic net takes over time. | none | `results.Trace(times, levels)`, the same word as above for a narrower thing. Propose `LogicTrace`. |
| **logic level** | A logic net's value: 0, 1, or unknown. | IEEE 1164's logic value. | `LogicLevel` is defined twice (`devices.LogicLevel`, `report.LogicLevel`), alongside `jolty_models::logic::Level` and methods `OperatingPoint.level(net)`, `Sweep.level(net)`, and `Waveforms.level(net)`. `report.Level` (single-net level) collides with `jolty.settings.Level` (rule level). Propose one `LogicLevel`, `logic_level(net)`, `report.NetLevel`, and `jolty_models::logic::LogicLevel`. |
| **strategy** | How an operating point was found: directly, by gmin stepping, source stepping or pseudo-transient. | SPICE's convergence aids. | One name (`Strategy`, `strategy`, `strategies`). |
| **statistics** | What a transient did: its steps accepted, rejected and landed, its iterations, its periods. | SPICE's `.option acct` statistics. | `jolty_engine::transient::Statistics`, `results.Statistics`; but `jolty_numerics::lu::Statistics` is the factorizations' counts, which `results.Work` holds with the iterations. Propose `lu::Work` for the counts, so *statistics* is a transient's alone. |
| **work** | What an analysis counted: its iterations, factorizations, refactorizations, fallbacks and reuses. | none | `Work`; `lu::Statistics` (above). |
| **profile** | Where an analysis's time went, by activity. | A profile. | One name (`Profile`, `Options.profile`). |

### 4. Analyses, their results and their reports

An analysis has seven names today, in as many shapes, and two Python classes share each of two names (`quality.md` research, S4).
The family below lists them, then proposes one shape (principle 3).

| Analysis (`jolty.plan`, `Analysis::` in the engine, `"kind"` in files) | Result (`jolty.results` and `jolty_engine`) | Extension (`_core.pyi`) | Report (`jolty.report`) | Measure's view (`jolty_engine::measure::Results::`, `Measure.read_*`) | Proposed result, report, view |
| --- | --- | --- | --- | --- | --- |
| `OperatingPoint`, `operating_point` | `OperatingPoint` | `OperatingPointOutcome` | `PointReport` | `Point`, `read_point` | `OperatingPointResult`, `OperatingPointReport`, `OperatingPoint`, `read_operating_point` |
| `OperatingPoints`, `operating_points` | `OperatingPoints` | `OperatingPointsOutcome` | `PointsReport` | `Points`, `read_solutions` | `OperatingPointsResult`, `OperatingPointsReport`, `OperatingPoints`, `read_operating_points` |
| `DcSweep`, `dc_sweep` | `Sweep` | `SweepOutcome` | `SweepReport` | `Sweep`, `read_sweep` | `DcSweepResult`, `DcSweepReport`, `DcSweep`, `read_dc_sweep` |
| `Ac`, `ac` | `Response` | `ResponseOutcome` | `ResponseReport` | `Response`, `read_response` | `AcResult`, `AcReport`, `Ac`, `read_ac` |
| `Transient`, `transient` | `Waveforms` | `WaveformsOutcome` | `TransientReport` | `Transient`, `read_transient` | `TransientResult`, unchanged, unchanged, unchanged |
| `Sensitivity`, `sensitivity` | `Sensitivities` | `SensitivityOutcome` | `SensitivityReport` | none | `SensitivityResult`, unchanged |
| `Stability`, `stability` | `LoopGain` | `StabilityOutcome` | `StabilityReport` | `Loop`, `read_loop` | `StabilityResult`, unchanged, `Stability`, `read_stability` |
| `Noise`, `noise` | `NoiseSpectrum` | `NoiseOutcome` | `NoiseReport` | `Noise`, `read_noise` | `NoiseResult`, unchanged, unchanged, unchanged |
| `PoleZero`, `pole_zero` | `PolesZeros` | `PoleZeroOutcome` | `PoleZeroReport` | `Poles`, `read_poles` | `PoleZeroResult`, unchanged, `PoleZero`, `read_pole_zero` |
| `TransferFunction`, `transfer_function` | `Transfer` | none: the extension returns a dict the stub's `Outcome` union lacks (`crates/jolty-python/src/lib.rs:628`) | `TransferReport` | none | `TransferFunctionResult`, `TransferFunctionReport` |
| the union, `jolty.plan.Analysis`, `jolty_engine::Analysis` | `jolty.results.Result`, `jolty_engine::analysis::Outcome` | `Outcome` | `AnalysisReport` | `Results` | `AnalysisResult` in both languages, `AnalysisReport`, `AnalysisResult` for the measure's view too |

Domain nouns (`Waveforms`, `NoiseSpectrum`, `PolesZeros`) break down for operating points, where the analysis and result share one term, so the result family needs a suffix.
Existing SPICE tools also name results after analyses (ngspice plots `op1`, `ac1`, `tran1`; PySpice `TransientAnalysis`).
`BatchResult`, `BuildResult`, and `SimulateResult` already follow the `XResult` pattern.
`results.Response` also collides with the measures' "response" and `Transfer` with `jolty_engine::measure::Transfer`, the choice of loop gain or output response a gain is read from.
The alternative, keeping the nouns and renaming only the operating points' results, is choice 5.

The extension's `XOutcome` dicts and engine `Outcome` describe transport across the FFI boundary rather than domain meaning (principle 12); `jolty_engine::batch::Outcome` adds a second `Outcome` in the same crate.
Propose naming each dict after the result it carries (`_core.AcResult`), and `batch::Outcome` **run** (`batch::Run`), as `jolty.results.Run` is in Python.

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **analysis** | One computation a plan names: an operating point, a sweep, a frequency response, a transient, and the others above. | SPICE's analysis. | One name. |
| **operating point** | The circuit's DC solution, with every source at its DC level and every capacitor open. | SPICE's operating point (`.op`). | One name for the concept; the class name collides (above). |
| **DC sweep** | Operating points as a source's level or a parameter steps through values. | SPICE's DC transfer curve (`.dc`). | `DcSweep` in plans, `Sweep` in results and reports. |
| **AC analysis** | The circuit's small-signal response at frequencies, linearized at its operating point. | SPICE's `.ac`. | `Ac` in plans, `Response` in results and reports, "small-signal response" in the engine. |
| **transient** | The circuit's solution over time from its operating point. | SPICE's `.tran`. | `Transient`; `Tran` in `perf/robustness/corpus.py`. Propose `Transient` there. |
| **periodic steady state** | The solution of a periodic circuit over one period, found by shooting. | ngspice and Spectre `pss`. | A flag, `Transient(periodic=True)`, which turns `stop` into the period (research S6). Propose an analysis `PeriodicSteadyState`, kind `periodic_steady_state` (choice 6). |
| **stability analysis** | The loop gain at frequencies across a probe source placed in the loop. | Spectre `stb`; Middlebrook and Tian method. | `Stability`; its result `LoopGain`; `Results::Loop`. |
| **probe** (of a stability analysis) | The source a stability analysis breaks its loop at. | Spectre's `stb probe=`. | `probe` (`Stability.probe`, `LoopGain.probe`, `StabilityReport.probe`); `probes` for traces and `probes.py` for the coverage suite's fixtures, which move to dedicated names. |
| **transfer function** | The small-signal gain from a source to an output, with the input and output resistances. | SPICE's `.tf`. | `TransferFunction` in plans, `Transfer` in results and reports. |
| **noise analysis** | The noise density at an output over frequency, and each noise source's contribution. | SPICE's `.noise`. | `Noise`; `NoiseSpectrum` in results. |
| **pole-zero analysis** | The poles, and the zeros of a transfer function, of the circuit linearized at its operating point. | SPICE's `.pz`. | `PoleZero`; `PolesZeros` in results; `Poles` in the measures' view. |
| **sensitivity** | The derivatives of outputs with respect to every parameter, at an analysis's solution. | SPICE's `.sens`. | `Sensitivity`; `Sensitivities` in results; `Differentiated`, the analyses a sensitivity may differentiate, in `report` and the engine. |
| **output** | An unknown an analysis reports on, by name, as a transfer function's, a noise analysis's or a sensitivity's. | SPICE's output variable (`.tf outvar`). | `output` and `outputs`; but `signal` in the measures (`value(analysis, signal, ...)`, `maximum(analysis, signal, ...)`, `jolty_engine::measure::Signal`) while `gain(analysis, output, ...)` takes `output`. Propose `output` for the measures' unknown, and `measure::Output` for `Signal`. |
| **reference** (of an output) | The unknown an output is measured against, for a differential output. | SPICE's `v(out, ref)`. | One name (`reference`). |
| **source** (of an analysis) | The independent source an analysis drives or sweeps, by instance path. | SPICE's `srcnam`, `insrc`. | One name once the source location is renamed (section 1). |
| **level** (of a source) | The value an independent source holds, or each value a DC sweep sets it to. | SPICE's "value" of a source (`DC value`, `V1`, `V2`) and sweep (`vstart`, `vstop`). | `level` (`waveforms.Dc.level`, `DcSweep.levels`, `Sweep.levels`, `Traced.along == "level"`, `SweepReport.levels`); `to_json(level)` names the levels' dimension, not a level (research S7); and `level` is also a rule's level and a logic level. See choice 7. |
| **independent variable** | What an analysis's solutions are taken along: points, levels, frequencies or times. | The sweep or scale vector, in ngspice. | `along` and `axis` (`Traced.along`, `Traced.axis`); `Axis` is also a grid's axis of variants (section 6). Propose `Traced.scale` for the values, ngspice's word, if `along` stays for the kind. |
| **max solutions** | How many DC solutions a search finds at most. | none | `OperatingPoints.most`, `jolty_engine::OperatingPoints::most`. Propose `max_solutions` (principle 4). |

### 5. Options and parallelism

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **options** | The numerical settings of a run: tolerances, limits, what results record, and how a run spreads its work. | SPICE's `.options`. | One name (`Options`, `jolty.plan.DEFAULTS`, `_core.default_options`). |
| **relative tolerance**, **voltage tolerance**, **current tolerance**, **charge tolerance** | The tolerances on every unknown relative to its magnitude, and the absolute ones on voltages, currents and charges. | SPICE's `RELTOL`, `VNTOL`, `ABSTOL`, `CHGTOL`. | One name each (`relative_tolerance`, ...); `jolty.foundry.SPICE_OPTIONS` maps them to SPICE's. |
| **truncation tolerance** | How many times the tolerances a step's truncation error may reach before the step is rejected. | SPICE's `TRTOL`. | One name. |
| **gmin** | The conductance placed across every nonlinear junction. | SPICE's `GMIN`, gmin stepping. | One name; the only option kept abbreviated, as the textbooks write it (principle 1). |
| **max DC iterations**, **max transient iterations** | The most Newton iterations of one DC solution, and of one time step before it shortens. | SPICE's `ITL1`, `ITL4`. | `dc_iterations`, `transient_iterations`: bounds without `max_` (principle 4). Propose `max_dc_iterations`, `max_transient_iterations`. |
| **max memory** | The most gibibytes of numbers an analysis may hold at once. | none | `Options.memory`. Propose `max_memory`. |
| **max step** | The longest time step a transient may take. | SPICE's `TMAX`. | One name (`max_step`). |
| **stop** (of a transient) | The time a transient ends at. | SPICE's `TSTOP`. | One name. |
| **breakpoints** | Times a transient must step onto exactly, besides those it chooses. | ngspice's breakpoints. | `Transient.times`, the same word as `Waveforms.times`, every time a transient solved; `Statistics.landed` counts the steps that landed on them. Propose `breakpoints`. |
| **integration method** | The formula a transient integrates charges with: trapezoidal, Gear's second order or backward Euler. | SPICE's `METHOD`. | One name (`method`, `jolty_numerics::integration::Method`, `Method`). |
| **parallelism** | Policy controlling how a run distributes work across threads without changing result bits. | none | One name (`Parallelism`). |
| **max threads** (of a run) | Maximum thread count a single run may use. | none | `Parallelism.threads`, while `Settings.threads` and `JOLTY_THREADS` are the exact pool size. Propose `Parallelism.max_threads` (principle 4). |
| **min solves per task** | The fewest frequencies, right-hand sides or parameters' derivatives one task takes when a run spreads them. | Grain size, in parallel computing. | `min_solves_per_task` everywhere (Rust, Python, `--min-solves-per-task`, `bench.Threshold`). One name; the family's model. |
| **instances per task** | How many device instances one task evaluates when a run spreads its devices' evaluation. | Grain size. | `min_instances_per_task`: the code takes it as each task's exact size (`Parallelism::evaluation`), not a least one, unlike `min_solves_per_task`. Propose either `instances_per_task`, to say what it is, or making it a least size, as solves' is, and keeping the name (choice 8). |
| **min unknowns per task** | The fewest unknowns a circuit has for its Newton iteration's and step control's work on each unknown to spread over threads. | none | `min_unknowns_inside` (`Inside`, `inside()`): `inside` names where the work happens in the code, not what is bounded, and the bound is on the circuit, not on a task. Propose `min_unknowns_per_task` with the definition moved to tasks, so the family has one shape, or `min_unknowns` with today's definition (choice 8). |
| **max unknowns per interior** | The most unknowns of an interior the nested dissection splits a large block into. | Nested dissection's interiors and separators. | `max_unknowns_per_interior`, in `Parallelism`, though it orders the factorization's operations and so changes results' last digits: `AGENTS.md` ("Parallelism is a policy") calls such a setting a numerical option. Propose moving it to `Options`, its name kept (choice 9). |

The flags, the benchmark's `Threshold` members and the service's `_SOLVES_DESCRIPTION`, `_INSTANCES_DESCRIPTION`, `_INSIDE_DESCRIPTION` and `_INTERIORS_DESCRIPTION` follow the fields.

### 6. Conditions, variants, batches and shards

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **plan** | What to simulate a circuit with: its analyses, options, conditions and variants. | SPICE's control statements. | One name (`Plan`, `--plan`, `plan_hash`). |
| **corner** | A named condition set overriding plan defaults: temperature, process, and build arguments. | A PVT corner. | One name (`Corner`, `corners`); its keyword space mixes its fields with `build()` arguments (research S6). |
| **conditions** | The temperature, process, overrides, and mismatch for a single run, from plan defaults or a corner. | none | One name (`Conditions`, `Simulation.conditions`, `Shard.conditions`). |
| **variant** | One run's overrides and temperature, among many a plan describes. | none | One name (`Variant`, `Variants`, `VariantsReport`, `Simulation.varied`). |
| **listed variants** | Variants given one by one. | none | `Explicit`, which describes how variants are specified, and `_core.Variants.explicit`. Minor case of principle 12; propose keeping `Explicit` unless the owner prefers `Listed`. |
| **grid**, **axis**, **product** | Variants as every combination of values along axes, and of two sets of variants. | A full factorial design. | One name each; `Axis` is also `Traced.axis`, an analysis's independent variable (section 4). |
| **Monte Carlo** | Variants drawn at random from the tolerances and mismatch, by seed and sample index. | Monte Carlo analysis. | One name (`MonteCarlo`, `monte_carlo`, `"monte-carlo"` in batch files). |
| **sample** | One draw of a Monte Carlo analysis, by its index. | A sample. | One name (`samples`, `sample(...)`, `--samples`). |
| **worst case** | Runs at each toleranced parameter's extremes, moved by a step, to bound each requirement. | Worst-case analysis. | One name (`WorstCase`, `worst_case`). |
| **run** | One simulation of a circuit at one variant's conditions, every analysis of the plan. | none | `results.Run`, `BatchResult.runs`, `plan.runs(variants)`; `jolty_engine::batch::Outcome` in Rust (section 4). The benchmarks' `Run` classes are other concepts (section 13). |
| **batch** | Many runs of one circuit, each a pure function of its index, measured and traced. | A batch. | One name (`Simulation.batch`, `BatchResult`, `jolty_engine::batch::Batch`, `files.Batch`). `Simulation.batch` takes a plan's pieces where `Simulation.run_plan` takes a plan (research S7). |
| **shard** | A contiguous slice of Monte Carlo samples executed by any process into a separate file. | A shard. | One name (`Shard`, `--shards`, `merge`). |
| **digest** | A SHA-256 that identifies what a file or a shard holds. | A digest, as `AGENTS.md` pins third-party files "with its digest". | Three words: `fingerprint` (`shards.fingerprint`, `Shard.fingerprint`, `ShardReport.fingerprint`, the guide's "Fingerprints", `bench.fingerprint`), `hash` (`files.circuit_hash`, `plan_hash`, `Provenance`'s documentation), `sha256` (`perf/robustness/decks.Entry.sha256`). Propose `digest` in every name (choice 10). |
| **provenance** | The jolty, build and platform that made a results file, and the digests of its circuit and plan. | Provenance. | `files.Provenance`; the guide also titles the source locations of instances "Provenance" (`components.md`). Propose the guide's heading "Source locations". |

### 7. Specs and measures

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **specs** | The requirements a circuit's results must meet, and the yield they must reach. | A specification. | One name (`Specs`, `jolty.specs`, `Shard.specs`). |
| **requirement** | A measure with the bounds its value must lie within, and a margin. | A requirement. | One name (`Requirement`, `Verdict.requirement`, `Spread.requirement`). |
| **measure** | A number read from one analysis's result, such as a gain at a frequency or a rise time. | SPICE's `.measure`; `AGENTS.md` ("the measures specs read of their results"). | `Measure` in Rust and the extension (`jolty_engine::measure::Measure`, `MeasureError`, `_core.Measure`, `Summary.measures`, `Measured.measure`, `Unmeasured::Measure`); `Measurement` in Python (`jolty.specs.Measurement`, `MeasurementError`, `Requirement.measurement`, `Simulation.batch(measurements=...)`, `BatchResult.measurements`, the guide's "Measurements", 241 sites). Propose `Measure`, `MeasureError`, `measures`. |
| **reading** | A measure's value, with its unit. | A meter's reading. | `jolty.specs.Reading`; the results' `readings` are a device's variables, renamed `variables` (section 3). |
| **engine measure** | The form of a measure evaluated directly by the engine without Python. | none | `jolty.specs.Native` and `Measurement.native`, which describe execution mechanism. Propose folding into `Measure` if unified, or keeping private. |
| **output**, **reference**, **window**, **edge** | What a measure reads: its unknown, the unknown it is measured against, the span of its independent variable, and a crossing's direction. | ngspice's `.meas` `TRIG`/`TARG`, `FROM`/`TO`, `RISE`/`FALL`. | `signal` and `output` both for the unknown (section 4); `Window { start, stop }`, `Edge`; one name each otherwise. |
| **mean** | The arithmetic mean: of a waveform over its window, or of a requirement's values over runs. | ngspice's `AVG`; statistics' mean. | `average` for the measure (`average(...)`, `Measure::Average`), `mean` for the statistic (`specs.Values.mean`). Propose `mean` for both, or keeping ngspice's `average` for the time average (choice 11). |
| **verdict** | Whether a requirement held, with its value, slack and the corner that decided it. | none | One name (`Verdict`, `Status`). |
| **status** | A verdict's outcome: passed, marginal, failed or unmeasured. | none | One name in the library; the benchmarks' statuses are other concepts (section 13). |
| **slack** | How far a requirement's value lies inside its bounds, negative when outside. | Static timing analysis's slack. | One name. |
| **margin** | The distance a requirement keeps from its bounds before it is marginal. | A design margin. | One name (`Requirement.margin`, `with_margin`). |
| **summary statistics** | The minimum, maximum, mean and standard deviation of a requirement's values over runs. | Summary statistics. | `jolty.specs.Values`, a name `_core.Values` (an elaborated circuit's parameter values) and every result's `values` (the unknowns') also take; `Values.deviation`. Propose `SummaryStatistics`, `standard_deviation`. |
| **spread** | A requirement's values over the runs of a batch: how many were measured and passed, and their summary statistics. | none | `Spread`, `MonteCarloReport.spreads`, `shards.spreads`. One name. |
| **yield** | The fraction of samples that meet every requirement, with its confidence interval and target. | Parametric yield, Wilson's interval. | `Yield { samples, passing, estimate, low, high, target }`, `Specs.yield_target`; `MonteCarloReport.passing` holds a `Yield`. Propose `lower`, `upper` (principle 4), and `MonteCarloReport.yield_` or `estimated_yield` for the field. |
| **phase margin**, **gain margin** thresholds | The margins below which the rules `phase-margin` and `gain-margin` trigger for a loop. | none | `LEAST_PHASE_MARGIN` and `LEAST_GAIN_MARGIN` live in `jolty.specs._measurements`, while rules live in `checks.PHASE_MARGIN` and `checks.GAIN_MARGIN`, alongside `checks.DERATED` (0.8). Propose `MIN_PHASE_MARGIN` and `MIN_GAIN_MARGIN` (principle 4) defined next to the rules, plus `DERATING_FACTOR`. |

### 8. Checks

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **problem** | A fact that stops the engine from elaborating a circuit, such as a negative resistance. | `AGENTS.md` ("Problems and rules"). | One name in the library (`jolty_ir::validate::Problem`, `ElaborationProblem`, `problems`); but the benchmarks call a fixture's circuit-maker a `Problem` (`perf/robustness/corpus.Problem`, `DeckProblem`) and the agent benchmark a task's circuit (`benchmarks/analogcoder/scorer.Problem`). Propose other names there (section 13). |
| **rule** | A fact the checks may state about a circuit that elaborates, at a level people configure, named in lowercase words joined by hyphens. | ty's rules. | One name (`Rule`, `rules`, `jolty explain`). |
| **level** (of a rule) | How a rule is reported: `error`, `warn` or `ignore`. | ty's level. | `Level` (`jolty.settings.Level`, `jolty_checks::Level`), `Rule.level`, `Diagnostic.level`, `RuleInfo.level` and `default_level`; but `jolty.specs._requirements.Judged.severity` for a verdict's. Propose `Judged.status`, as `Verdict.status`. |
| **diagnostic** | One statement of a rule about a circuit, with its message, instances and nets. | A compiler's diagnostic. | One name. |
| **report** (of a check) | The diagnostics a check found, with their counts by level. | none | `jolty.checks.Report`, while `jolty.report` and its classes are what analyses found. Propose `checks.CheckReport`, so `Report` alone is never a class (choice 12). |
| **rule names** | | ty's: a rule names the fact it states, as `unresolved-import`. | Thirteen of sixteen rules name a condition (`unconnected-terminal`, `no-dc-path`, `unstable-pole`), whereas `phase-margin` and `gain-margin` name quantities and `derating` names a practice alongside `over-rating` (firing past 80%). Propose `low-phase-margin`, `low-gain-margin`, and `over-derating` (choice 13). |

### 9. Files

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **circuit file** | A circuit as JSON, `<script>.jolty.json`. | none | One name (`save_circuit`, `load_circuit`, `circuit_path`, `CIRCUIT_SUFFIX`). Circuit-file keys change with *name* (`"id"`), *model* (`"device"`), and *source location* (`"source"`), bumping `jolty_ir::VERSION`. |
| **results file** | The results of a simulation's named analyses, `<script>.results.npz`. | none | One name (`save_results`, `load_results`, `results_path`, `RESULTS_SUFFIX`, `--results`). Results-file header keys change with *nature* (`"quantity"`), *variables*, and private result models (version 9). |
| **batch file** | A plan's batches of runs, `<results>.batches.npz`. | none | `batches` in code (`save_batches`, `load_batches`, `batches_path`, `BATCHES_SUFFIX`, `SimulateResult.batches`), "batch file" in `files`' documentation. One concept; the plural is of what it holds, as `results` is. |
| **shard file** | One shard, `shard-START-STOP.npz` in the shards' directory. | none | One name (`save_shard`, `load_shard`, `shard_name`, `SHARD_PATTERN`). |
| **catalog** | Orderable parts with prices, as JSON. | A parts catalog. | One name (`Catalog.load`, `Catalog.save`, `"jolty-catalog"`). |
| **bill of materials** | The parts a circuit's instances are priced as, and those left unpriced. | A BOM. | `Bill`, `bill(...)`, `CostResult.bill`; "Bills of materials" in the guide. One concept. |
| **KiCad netlist** | A component written for KiCad's board editor. | KiCad's netlist. | One name for the file (`kicad.Netlist`, `netlist_path`); the command that writes it has three names (section 10). |
| **format** and **version** | A file's kind and the version of its format, checked before it is read. | none | One name (`format`, `version`, `jolty_ir::FORMAT`, `VERSION`). |

### 10. Commands, settings and the environment

Each command is a request model and a function in `jolty.service`; the command line and the MCP server derive from them.

| Command line | Service function, models | MCP tool | Proposed (principle 8) |
| --- | --- | --- | --- |
| `build` | `build`, `BuildRequest`, `BuildResult` | `build_circuit` | unchanged |
| `simulate` | `simulate`, `SimulateRequest`, `SimulateResult` | `simulate_circuit` | unchanged |
| `merge` | `merge`, `MergeRequest`, `MergeResult` | `merge_shards` | unchanged |
| `plot` | `plot`, `PlotRequest`, `PlotResult` | `plot_results` | unchanged |
| `draw` | `draw`, `DrawRequest`, `DrawResult` | `draw_schematic` | unchanged |
| `view` | `view`, `ViewRequest`, `ViewResult` | `write_viewer` | MCP `view_circuit` |
| `check` | `check`, `CheckRequest`, `CheckResult` | `check_circuit` | unchanged |
| `explain` | `explain`, `ExplainRequest(rules)`, `ExplainResult(rules)` | `explain_rule` | MCP `explain_rules`, as it takes several |
| `cost` | `cost`, `CostRequest`, `CostResult` | `price_circuit` | MCP `cost_circuit` |
| `kicad` | `write_kicad`, `KicadRequest`, `KicadResult` | `write_kicad_netlist` | a verb everywhere: `export-kicad`, `export_kicad`, `ExportKicadRequest`, `ExportKicadResult`, MCP `export_kicad_netlist` (choice 14) |

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **script** | A Python file defining `build()`, which returns the top component. | none | One name (`script`, `ScriptOutput`, `ScriptError`). |
| **output** (of a command) | Where a command writes its file. | none | `--output`, one name; `--output-format` is how it prints, `ScriptOutput` what a script printed. |
| **max diagnostics** | The most diagnostics a report lists; it counts the rest. | none | One name (`max_diagnostics`, `--max-diagnostics`, `JOLTY_MAX_DIAGNOSTICS`). |
| **max unknowns** | The most unknowns a report lists for each analysis; it counts the rest. | none | `Settings.listed_unknowns`, `JOLTY_LISTED_UNKNOWNS`, `runner.Job.listed`: the same kind of setting as `max_diagnostics` in another shape. Propose `max_unknowns`, `JOLTY_MAX_UNKNOWNS`. |
| **max output** | The most characters kept from a script's standard output and error. | none | `Settings.output_limit`, `JOLTY_OUTPUT_LIMIT`, a third shape. Propose `max_output`, `JOLTY_MAX_OUTPUT`. |
| **unlisted** | Count of items omitted from a truncated report list. | none | One name (`unlisted`). |
| **timeout** | The seconds a script may run. | none | One name (`timeout`, `JOLTY_TIMEOUT`); the benchmarks' and `jolty.spice.ngspice.TIMEOUT` are each simulator's. |
| **threads** | The threads of the engine's pool. | none | One name (`Settings.threads`, `JOLTY_THREADS`, `Pool.threads`); a run's own bound is `Parallelism.max_threads` (section 5). |
| **cache**, **PDK root**, **rules**, **analysis log** | Where downloads are kept, where SKY130 is installed, each rule's level, and the file each analysis logs a line to. | open_pdks' `PDK_ROOT`. | One name each (`cache`, `pdk_root`, `rules`, `analysis_log`, and their `JOLTY_` variables); the log's lines are `simulate.Logged`. |
| **test switches** | `JOLTY_REQUIRE_NGSPICE`, `JOLTY_RANDOM_CIRCUITS`. | none | One name each. |

### 11. Other simulators

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **deck** | A circuit and its control lines written as a SPICE netlist for ngspice or Xyce. | SPICE's input deck. | `deck` (`execute(deck)`, `bench.deck`, `perf/robustness/decks.py`, `perf/coverage/decks/`); `lines` for a deck as a list of lines (`ngspice.run(lines, ...)`, `xyce.run(lines, ...)`, `Spiced.lines`). One concept; propose `deck` for the parameter too. |
| **element** | A deck's line that places one instance, by its SPICE letter. | SPICE's element. | `jolty.spice.export.Element`; `jolty_engine::logic::Endpoint::Element`, which is a device's terminal on a logic net, another concept. Propose `Endpoint::Terminal`. |
| **simulator result** | The vectors one analysis of another simulator left, by name. | ngspice's plot. | `jolty.spice.ngspice.Raw`, named for the raw file it is read from (principle 12), and `jolty.spice.xyce.Result` on `xyce`. Propose `Result` in both modules, once `jolty.results.Result` becomes `AnalysisResult`. |
| **execution** | What one run of a simulator's process left: what it printed, its exit code and its seconds. | none | `ngspice.Executed`, `xyce.Executed`, `execute(...)`. One name. |
| **SPICE's Celsius** | The Celsius temperature another simulator must be given so its thermal voltage matches jolty's. | none | `ngspice.spice_celsius`, `xyce.xyce_celsius` (on `xyce`), and a second `xyce_celsius` in `perf/coverage/runners.py`: one fact defined twice, which "One definition" forbids. Propose `ngspice.celsius` and `xyce.celsius`, with the module identifying the simulator, and remove the duplicate. |
| **SKY130** | SkyWater's 130 nm process and its design kit. | SkyWater's name. | One name (`jolty.spice.sky130`). |
| **foundry device** | A transistor of a foundry's process, which only ngspice simulates, with the foundry's models. | none | One name (`jolty.foundry`, `FoundryMosfet`, `_core.Simulator.foundry_devices()`, `Simulation.foundry`). |

### 12. Device models and their parameters

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **kernel** | A device model's equations: a pure function of one instance's prepared parameters, its memory and its local unknowns. | none (ngspice's device `load`). | One name (`Kernel`, `AGENTS.md`'s "Kernels"). |
| **model** (Rust trait) | A kind of device: its name, its parameters, and how an instance's are read into its kernel's. | SPICE's device type. | `jolty_models::Model`; and `jolty.fidelity.Model`, a characterized logic model's summary in a report. Propose `fidelity.CharacterizedModel` (choice 15). |
| **exact model** | The model of a kernel whose instances give exact derivatives with respect to any of their parameters. | none | `jolty_models::Native`, which describes mechanism and collides with `jolty.specs.Native` (section 7). Propose `Differentiable`. |
| **prepared**, **memory**, **shape**, **parameters** | The four types of each kernel: temperature-evaluated parameters, per-instance state, parameter-defined structure, and parsed parameters. | SPICE's temperature update (`temp`), state vector, none, parameters. | Most models follow `X`, `PreparedX`, `XMemory`, `XShape`, and `XParameters` (`Diode`, `PreparedDiode`, `DiodeMemory`, `DiodeShape`, `DiodeParameters`). Shared family types use family nouns (`PreparedSwitch`, `MemristorShape`, `ThermalShape`, `SourceParameters`). Five single-model types use shortened nouns: `ControlledVoltageShape`, `SensedCurrentShape`, `CoupledShape`, `DriverShape`, and `DetectorShape`. Propose full model names (`VoltageControlledVoltageSourceShape`), or document sharing explicitly. |
| **polarity**, **channel** | Whether a bipolar transistor is npn or pnp; whether a FET's channel is n or p. | The textbooks' terms. | One name each (`Polarity`, `Channel`). |
| **model level** | Which of SPICE's MOSFET models an instance follows: 1, 2 or 3. | SPICE's `LEVEL`. | `mosfet(level=...)`, `jolty_models::mosfet::Level`: the bare word of a rule's level and a source's level. Propose keeping SPICE's parameter name (`level`) but naming the type `mosfet::ModelLevel` (choice 7). |
| **temperature coefficient of a parameter** | How a parameter changes with temperature, per kelvin or as an exponent of the temperature ratio. | SPICE's `TC1`, `XTI`, `BEX`. | Most parameters use `<parameter>_temperature_coefficient` and `_exponent` (`saturation_current_temperature_exponent`, `drain_resistance_temperature_coefficient`). JFET `beta_temperature_exponent` and VDMOS `transconductance_temperature_exponent` name the same transconductance exponent differently (43 and 14 sites), while BJT `beta_temperature_exponent` (`XTB`) scales current gain. `threshold_temperature_coefficient` (JFET, VDMOS) scales `threshold_voltage`. Propose `transconductance_parameter_temperature_exponent` for JFET and VDMOS, plus `threshold_voltage_temperature_coefficient` (principle 10). |
| **bounds among parameters** | A parameter that is itself a lowest or highest value of a physical quantity. | SPICE's `RBM` ("minimum base resistance"), `VMAX`. | `max_drift_velocity` (MOSFET), `max_gate_drain_capacitance` and `min_gate_drain_capacitance` (VDMOS), but `minimum_base_resistance` (bipolar transistor, 17 sites). Propose `min_base_resistance` (principle 4). |
| **BSIM3's instance parameters** | | BSIM3's manual. | `bsim3(delvto, mulu0, mobmod, capmod, noimod, paramchk, xpart, card)` keep the manual's names, where every other device spells its parameters out: SPICE's `DELVTO`, one concept, is `threshold_adjustment` on the MOSFET (level 3) and `delvto` on BSIM3. The card's hundreds of names are BSIM3's own; the eight instance arguments are a choice (choice 16). |
| **noise source** | A device's generator of one kind of noise, such as a resistor's thermal noise. | A noise source. | `jolty_models::noise::NoiseSource`, `results.NoiseSource(instance, name)`. One name. |
| **noise contribution** | A noise source's density at the output, at each frequency, or its share of the total. | Spectre's noise contributors; ngspice's per-device noise. | `jolty_engine::noise::NoiseContribution { device, name, densities }`, `NoiseSpectrum.contributions`, but `report.NoiseContributor(instance, noise, rms, share)`, where `noise` is the source's name. Propose `NoiseContribution(instance, name, rms, share)` in the report. |
| **excursion** | Where a logic device ran outside the domain its model was characterized over. | none | `results.Excursion(path, bound, low, high, ...)`, `_core.ExcursionOutcome(path, ..., low, high)`, `jolty_engine::logic::Excursion { device, bound, range, ... }`. Propose `instance` for the instance, and `minimum` and `maximum` for the domain's bounds, in all three. |
| **bound** | | | Four types named `Bound`: `jolty_engine::logic::Bound` (an excursion's, transition, load or supply), `jolty_numerics::integration::Bound` (a step's error bound, charge or current), `jolty.specs._requirements.Bound` (a requirement's bound value), and `perf/precision` scores' `bound` field. Propose `logic::ExcursionKind`, `integration::ErrorBound`, and keeping `specs.Bound`. |

### 13. Benchmarks

Four suites measure the engine (`perf/`), and one measures agents (`benchmarks/`), each having evolved separate terminology.

| Proposed name | Definition | Established term | Today |
| --- | --- | --- | --- |
| **suite** | A set of fixtures run and scored one way: speed, precision, robustness, coverage, agents. | A benchmark suite. | Prose uses `suite`, while folders split across `perf/` (`perf/circuits`, `perf/plans`, `perf/precision`, `perf/robustness`, `perf/coverage`) and `benchmarks/` (agents only). Entry scripts are `scripts/bench` and `scripts/precision`, and notes call robustness "the hard-circuit corpus". Propose `benchmarks/speed`, `benchmarks/precision`, `benchmarks/robustness`, `benchmarks/coverage`, and `benchmarks/agents` run via `scripts/bench` (choice 17). |
| **fixture** | One circuit a suite runs, with its analysis and what decides its score. | `AGENTS.md`'s word ("the suite, the fixture, the machine, the versions and the results file"); a benchmark case. | Speed (`bench.fixtures()`, `load(fixture)`, `Bench.fixtures`) and precision (`Fixture`, `fixtures()`, `Precision.fixtures`) use `fixture`. Robustness uses `case` (`Case`, `cases()`, 173 sites), while coverage uses `probe` and `row` (`probes.py`, `Row`, `matrix.probe(row, ...)`, `matrix.Line`). Agent benchmarks use `task` for a distinct concept (prompt plus scorer). Propose `fixture` (choice 18). |
| **fixture's maker** | What builds a fixture's circuit for jolty and writes its deck for another simulator. | none | `perf/robustness/corpus.Problem` (a protocol), `decks.DeckProblem`, `own.Written`; `problem` is the library's word for what stops elaboration. Propose `FixtureCircuit`, a fixture's circuit made for each simulator, avoiding `Problem`, `Source` and `Builder`, which the library uses for other things. |
| **expectation** | What a robustness fixture's answer must show: its solutions, that it settles, oscillates, meets an impasse, or completes. | none | `perf/robustness/corpus.Check`, the library's word for structural checks. Propose `Expectation`. |
| **simulator** | One program and configuration a suite runs fixtures in: jolty, ngspice with Sparse or with KLU, Xyce. | none | `Simulator` is a protocol in robustness (`name`, `version()`, `run(case, timeout)`) and an enumeration in precision (`JOLTY`, `NGSPICE_SPARSE`, `NGSPICE_KLU`) and in coverage (`JOLTY`, `NGSPICE`, `NGSPICE_KLU`, `XYCE`): ngspice with Sparse is `NGSPICE_SPARSE` in one and `NGSPICE` in the other. Propose an enumeration `Simulator` with `NGSPICE_SPARSE` and `NGSPICE_KLU` everywhere, and a protocol `Runner` for what runs a fixture in one. |
| **runner module** | The module of each simulator's runner, one for every suite. | none | `perf/robustness/simulators.py` and, on `coverage`, `perf/coverage/runners.py`, after both were named `simulators` and collided; `bench.rust`, `bench.python`, `bench.spice`, `bench.ngspice_run`; `coverage/runners.run_ngspice`, `run_xyce`; `precision/suite.jolty`, `spice`. `ngspice_run` and `run_ngspice` are one operation with their words swapped. Propose one module `benchmarks/simulators.py`, the word of the concept, holding `Jolty`, `Ngspice` and `Xyce` runners, as `quality.md` item 13 plans; `jolty.runner`, scripts' child processes, keeps its name. |
| **run** (of a fixture) | One fixture run once in one simulator, with its times and what it found. | none | `bench.Timing`, `precision/suite.Run`, `robustness/corpus.Run` with its `Outcome`, `coverage/matrix.Cell`; and three different classes named `Ran` (`precision/suite.Ran`, `coverage/matrix.Ran`, `xyce`'s `bench.Ran`). Propose `FixtureRun` for one run's record in every suite, and no `Ran`. |
| **results file** (of a suite) | What one measurement of a suite writes, with the commit, the machine, the load and the versions. | `AGENTS.md`'s "results file". | `Document` in precision and coverage, a dictionary in speed (`bench.save`, `bench.read`), `Context` and `Ran` for its header in coverage. Propose `SuiteResults`, with one header model shared (`quality.md` item 13), read by `load` and written by `save` (`bench.read` breaks the pair). |
| **status** (of a run) | How one run of a fixture went. | none | Three sets: precision `ok`, `failed`, `unsupported`; robustness `RIGHT`, `OTHER`, `SOLVED`, `FAILED`, `TIMED_OUT`, `UNSUPPORTED`; coverage `AGREES`, `DISAGREES`, `FAILS`, `ABSENT`, `RUNS`, `REFERENCE`, `DOCUMENTED`, `UNKNOWN`, with `NotOffered` for a simulator that lacks the feature. `failed` and `fails`, `unsupported`, `absent` and `NotOffered` are one state each under several words. Propose past participles everywhere (`failed`, `agreed`, `disagreed`), and `unsupported` for what a simulator does not offer. |
| **wall time**, **analysis time** | A run's time from starting the process to its exit, and the part its analyses took. | none | `seconds` (most), `elapsed` and `solvers` (`xyce.Result`), `analysis` (a float, `xyce`'s `bench.Ran.analysis`), `analysis_seconds` (robustness), `walls` and `analyses` (lists of times, `bench.xyce_fixture`). Propose `seconds` and `analysis_seconds` (principle 9). |
| **load** | The machine's load average during a measurement. | Unix's load average. | One name (`load`, `bench.load_average`). |
| **scored quantity** | One output a precision fixture is scored on, with its scale and unit. | none | `perf/precision/fixtures.Quantity`, a third class of that name. Propose `ScoredOutput`. |
| **break-even** | The size at which spreading a kind of work starts to pay, for one threshold of `Parallelism`. | none | One name (`--break-even`, `break_even`, `break_evens`, `pays_from`). |

## What the inventory found, by kind

Each item points to its row above.

**Synonyms (one concept under several names):**

- `Measure` and `Measurement` (7); `traced` and `probes` (3); `variables` and `readings` (3); `Varied` and `Toleranced` (2).
- `device`, `instance`, and `path` for instance-referencing fields (1); `DeviceOf.device` and `Device.model` (1); `checks.Instance`, `Location`, and `InstanceOutcome` (1).
- Python `Unknown` and Rust `UnknownInfo` (3); `Recording` and Rust `Recorded` for recorded metadata, plus `Observation`, `Observations`, and `Observed` for recorded values (3).
- `NoiseContribution` and `NoiseContributor`, with source fields `name` and `noise` (12); `signal` and `output` (4); `average` and `mean` (7).
- `fingerprint`, `hash`, and `sha256` (6); `listed_`, `output_limit`, and `max_` (10); `cost`/`price`, `view`/`write_viewer`, and `kicad`/`write_kicad`/`write_kicad_netlist` (10).
- JFET `beta_temperature_exponent` and VDMOS `transconductance_temperature_exponent` (12); MOSFET `threshold_adjustment` and BSIM3 `delvto` (12); `Transient` and `Tran` (4).
- `fixture`, `case`, `probe`, and `row` (13); `NGSPICE` and `NGSPICE_SPARSE` (13); `simulators` and `runners` (13); `failed`/`fails` and `unsupported`/`absent`/`NotOffered` (13).
- `severity` and `status` (8); `_stand_ins` and `substitutes` (2); `ground` and `reference node` (1); `deck` and `lines` (11).

**Homonyms (one name for several concepts):**

- `Quantity` (dimensioned number, unknown nature, precision scored output) (3, 13); `Source` (source location, independent source, noise source) (1).
- `OperatingPoint` and `OperatingPoints` (analysis and result) (4); `Level` and `level` (rule level, logic level, source value, MOSFET model level, `to_json(level)` dimension) (3, 4, 12).
- `Instance` (`ir.Instance`, `checks.Instance`) (1); `Component` (`builder.Component` built block and `ir.Component` definition, one concept in two forms, kept) (1); `Device` (`devices.Device`, `catalog.Device`) (1).
- `Trace` and `Traced` (3); `probe` (stability probe, trace, coverage fixture) (4, 13); `times` (transient breakpoints, solved times) (5); `threads` (pool size, per-run bound) (5).
- `Statistics` (transient statistics, LU counts) (3); `Outcome` (analysis result, batch run, robustness run) (4, 13); `Report` (check report, analyses report module) (8).
- `Values` (summary statistics, elaborated parameter values, solved unknowns) (7); `Native` (device model trait, engine measure) (7, 12); `Bound` (four distinct types) (12).
- `Point` (connection point, layout coordinate, operating-point view) (1, 4); `Axis` (variant grid axis, trace independent variable) (4, 6); `Lot` (`ir.Lot`, `variation.Lot`) (2).
- `Ran` (three classes) (13); `Problem` and `Check` (library vs benchmarks) (8, 13); `Result` (`results.Result`, `xyce.Result`) (11).
- `beta_temperature_exponent` (BJT `XTB` vs JFET) (12); `pin` (logic pin vs part pin, kept qualified) (1); `Provenance` (results file provenance vs guide source locations) (6).

**Pairs not named as pairs:**

- `bench.save` paired with `bench.read` (13); `minimum_base_resistance` alongside `max_drift_velocity` (12); `Yield.low` and `high` for a confidence interval (7); `LEAST_PHASE_MARGIN` alongside `max_step` (7).
- `lowest` and `highest` frequencies alongside `first` and `last` levels in reports (`ResponseReport`, `StabilityReport`, `NoiseReport`, `SweepReport`); `ngspice_run` vs `run_ngspice` (13); `over-rating` vs `derating` (8).

**Families in more than one shape:**

- Analysis results, reports, extension dicts, and measure views (4); the four `Parallelism` thresholds (5); report limits (`listed_unknowns`, `output_limit`, `max_diagnostics`) (10); MCP tools (10).
- Kernel helper types (12); device temperature coefficients (12); check rule names (8); benchmark statuses and timing fields (13).

**Names describing mechanism rather than meaning:** `Outcome` and `XOutcome` (4); `Raw` (11); `inside` (5); `Native` (7, 12); `UnknownInfo` (3); `Explicit` (6); `observe` alongside `record` (3).

## The renames, grouped by what they break

Counts are sites as defined above; a generic word's count is of its type, with the text count after a slash.

### Internal only: the engine's crates, the extension, private names, benchmarks

| Rename | Sites |
| --- | --- |
| `jolty_engine::analysis::Outcome` → `AnalysisResult`; `batch::Outcome` → `Run`, as Python's | 129 `Outcome` in Rust |
| `jolty_engine::{Response, Sweep, Waveforms, Sensitivities, LoopGain, Transfer, NoiseSpectrum, PolesZeros}` → `XResult` (section 4) | in the Python counts below, which include them |
| `measure::Results` → `AnalysisResult` view and its variants; `Measure.read_point` and kin → `read_operating_point` and kin | 28 variants, 18 methods |
| `_core.pyi`'s `XOutcome` dicts → the results' names; add the missing transfer function's | 60 |
| `binding::UnknownInfo` → `Unknown`, `binding::Unknown` → `UnknownIndex` | 26 and the index's uses |
| `binding::Quantity` → `Nature` | 83 |
| `observe` module → `record`; `Observation`, `Observations`, `Observed` → one name such as `RecordedValues`; `observe::Recorded` → `Recording` | 66, 26 |
| `variants::Toleranced { path, name }` → `{ instance, parameter }` | 16 |
| `sensitivity::Parameter::device`, `noise::NoiseContribution::device`, `logic::Excursion::device` → `instance`; `Excursion::range` → `minimum`, `maximum` | 19 `Excursion`, and each field's uses |
| `elaborate::Composite` → `ComponentInstance` | 94 with `composites` (Python included) |
| `lu::Statistics` → `lu::Work` | 10 |
| `logic::Bound` → `ExcursionKind`, `integration::Bound` → `ErrorBound`, `logic::Level` → `LogicLevel`, `mosfet::Level` → `ModelLevel` | each under 20 |
| `jolty_models::Native` → `Differentiable` | part of 87 `Native` |
| the kernels' short type names (`ControlledVoltageShape` and four more) → their models' full names | 5 types |
| `logic::Endpoint::Element` → `Terminal` | under 10 |
| `simulate._stand_ins` → `_substitutes` | under 10 |
| benchmarks: `case`/`Case` → `fixture`/`Fixture`, `probes.py`/`Row`/`Line` → fixtures, `Tran` → `Transient`, `Problem` → `FixtureCircuit`, `Check` → `Expectation`, `Ran` (three) and `Run` → `FixtureRun`, `Document` → `SuiteResults`, `NGSPICE` → `NGSPICE_SPARSE`, statuses, `bench.read` → `load`, `ngspice_run` → `run_ngspice`, `perf/` and `benchmarks/` → `benchmarks/<suite>` | 31 `Case`, 22 `Tran`, 10 `Row`/`Line`, 39 `NGSPICE`, 5 `Ran`, 263 `fixture` already right |

### The Python API

| Rename | Sites |
| --- | --- |
| results `OperatingPoint`, `OperatingPoints`, `Sweep`, `Response`, `Waveforms`, `Sensitivities`, `LoopGain`, `Transfer`, `NoiseSpectrum`, `PolesZeros` → `XResult`; `results.Result` → `AnalysisResult` | 105 `results.OperatingPoint(s)`; 95, 76, 114, 46, 55, 63, 69, 55 for the others (Rust included); 149 `Result` in Python, stubs and docs |
| reports `PointReport`, `PointsReport`, `SweepReport`, `ResponseReport`, `TransferReport` → `OperatingPointReport`, `OperatingPointsReport`, `DcSweepReport`, `AcReport`, `TransferFunctionReport` | 25 |
| `specs.Measurement`, `MeasurementError`, `measurements` → `Measure`, `MeasureError`, `measures` | 241 |
| `jolty.unknowns.Quantity`, `quantity` fields → `Nature`, `nature` | 216 / 333 |
| `ir.Source` → `SourceLocation`; `source` fields holding one → `source_location` | 24 type, 18 fields / 2,398 `source` |
| `checks.Instance` → `Location` | 131 `Instance` in text, about 20 of the type |
| `Instance.id`, `Builder.place(item, id)`, `Placed.id()` → `name` | by type / 1,817 text matches |
| `DeviceOf.device` → `DeviceOf.model` | 292 text matches with `"device":` keys outside the catalog |
| `readings` → `variables` | 54 |
| `traced`, `Traced`, `probes` → `traces`, `Traces` | 170, 45 |
| `results.Trace` → `LogicTrace`; `level(net)` → `logic_level(net)`; `report.Level` → `NetLevel`; `LogicLevel` defined once | 8, 29, 3, 10 |
| `Varied(path, parameter)` → `Toleranced(instance, parameter)` | 26 |
| fields naming an instance by `path` (`Excursion.path`) → `instance` | 18 |
| `report.NoiseContributor(noise=)` → `NoiseContribution(name=)` | 3 |
| `Measurement`'s `signal` parameters → `output` | 548 text, about 60 of the parameter |
| `Transient.times` → `breakpoints` | 12 |
| `Options.dc_iterations`, `transient_iterations`, `memory` → `max_` | 34, and `memory` by type |
| `Parallelism.threads` → `max_threads`; `min_unknowns_inside` → `min_unknowns_per_task`; `min_instances_per_task` (choice 8); `max_unknowns_per_interior` to `Options` | 4, 36, 34, 43 |
| `OperatingPoints.most` → `max_solutions` | 28 |
| `Yield.low`, `high` → `lower`, `upper`; reports' `lowest`, `highest` → `start`, `stop`; `LEAST_*` → `MIN_*` | 15, 50, 17 |
| `specs.Values` → `SummaryStatistics`, `deviation` → `standard_deviation` | 20 |
| `checks.Report` → `CheckReport` | 11 |
| `Judged.severity` → `status` | 15 |
| `catalog.Device` → `PartKind` | 10 |
| `variation.Lot` → `LotMember` | under 10 |
| `fidelity.Model` → `CharacterizedModel` | under 20 |
| `spice.ngspice.Raw` → `Result`; `spice_celsius` → `celsius` | 20, 11 |
| device parameters: `minimum_base_resistance` → `min_base_resistance`; JFET `beta_temperature_exponent` and VDMOS `transconductance_temperature_exponent` → `transconductance_parameter_temperature_exponent`; `threshold_temperature_coefficient` → `threshold_voltage_temperature_coefficient`; MOSFET `threshold_adjustment` or BSIM3 `delvto` to one name (choice 16) | 18, 57 (shared with the BJT's, which stays), 33, 14 |
| `Simulator.from_elaborated` (extension) → `of`, as every other alternate constructor of jolty's (choice 19) | 35 |
| a `PeriodicSteadyState` analysis for `Transient(periodic=True)` (choice 6) | an API change, not a rename |

### The files' formats

Every key below follows its Python or Rust field; each change bumps its format's version.

| Rename | Format |
| --- | --- |
| `"id"` → `"name"`; `"of": {"device"}` → `"of": {"model"}`; `"source"` (a source location) → `"source_location"` | circuit (`jolty_ir::VERSION`; 10 committed JSON circuits, and 68 other files that write the key inline: 32 Rust and Python sources, 36 gallery and discovery pages in HTML) |
| a part's `"device"` → `"kind"` | catalog (1,563 keys in the agent benchmark catalog) |
| `"quantity"` → `"nature"`; the device records' `variables` kept; the analyses' kinds unchanged | results (version 9) |
| `traced` → `traces`; `measurements` → `measures` | batch |
| `fingerprint` → `digest` (choice 10) | shard |
| the plan's `times` → `breakpoints`, `dc_iterations` → `max_dc_iterations`, `transient_iterations` → `max_transient_iterations`, `memory` → `max_memory`, `parallelism.threads` → `max_threads`, `min_unknowns_inside` → `min_unknowns_per_task`, `max_unknowns_per_interior` moved | plan JSON (`--plan`) and `perf/plans/*.json` |
| the suites' results files: their models' renames | the benchmarks' JSON, outside the repository |

### The command line and the MCP server

| Rename | Sites |
| --- | --- |
| MCP `write_viewer` → `view_circuit`, `price_circuit` → `cost_circuit`, `explain_rule` → `explain_rules` | 7, 6, 5 |
| CLI `kicad` → `export-kicad`, service `write_kicad` → `export_kicad`, `KicadRequest`/`KicadResult`/`KicadCommand` → `ExportKicad…`, MCP `write_kicad_netlist` → `export_kicad_netlist` (choice 14) | 15 |
| `--probes` → `--traces` | with `probes` above |
| `--min-unknowns-inside` → `--min-unknowns-per-task`; `--max-unknowns-per-interior` stays as an option's flag | with the fields above |
| settings `listed_unknowns` → `max_unknowns`, `output_limit` → `max_output`, and `JOLTY_LISTED_UNKNOWNS`, `JOLTY_OUTPUT_LIMIT` → `JOLTY_MAX_UNKNOWNS`, `JOLTY_MAX_OUTPUT` | 6, 3 |
| rules `phase-margin`, `gain-margin`, `derating` → `low-phase-margin`, `low-gain-margin`, `over-derating` (choice 13): a rule's name is read from settings and `pyproject.toml` | 21 |

## The twenty most important renames

Ordered by how many layers a concept's names split across and how often a user or an agent meets them.

1. **Analysis results, reports, and views in one family:** standardize on `XResult`, `XReport`, `x`, and `read_x` (section 4) so `jolty.plan.OperatingPoint` and `jolty.results.OperatingPoint` no longer collide.
   Every layer names results consistently, and engine `Outcome` plus extension `XOutcome` adopt the same names.
1. **`Measurement` → `Measure`** in Python, as Rust, the extension and `AGENTS.md` have it, with `MeasureError` and `measures` (241 sites).
1. **`jolty.unknowns.Quantity` → `Nature`**, so `Quantity` is only a number with its dimension, the class every script meets first (216 Python and 83 Rust sites, and the results file's `"quantity"`).
1. **`ir.Source` → `SourceLocation`**, so `source` is only an analysis's independent source (24 sites of the type, 18 of its fields, a circuit-format key).
1. **`checks.Instance`, `elaborate::Location`, `InstanceOutcome` → `Location`**, freeing `Instance` for the circuit's instance.
1. **Every field that names an instance → `instance`**, for Rust's `device`, Python's `path` and `instance` today (`Parameter`, `NoiseContribution`, `Excursion`, `Varied`, `Toleranced`).
1. **`traced`, `Traced`, `--probes` → `traces`, `Traces`, `--traces`**, with `results.Trace` → `LogicTrace`, so `probe` is only the stability analysis's.
1. **`readings` → `variables`**, as Rust, the stub and `RecordedDevice` name them, so `Reading` is only a measure's value (54 sites).
1. **The `Parallelism` thresholds to one family:** `max_threads`, `min_solves_per_task`, `min_instances_per_task` or `instances_per_task`, `min_unknowns_per_task`, and `max_unknowns_per_interior` moved to `Options`, as `AGENTS.md` asks of a setting that reorders arithmetic (choices 8, 9).
1. **`Instance.id` → `Instance.name`**, as every other named thing has it, in the API and the circuit file.
1. **`DeviceOf.device` → `DeviceOf.model`**, as `Device.model` holds the same name.
1. **The reports' classes after their analyses:** `PointReport`, `PointsReport`, `SweepReport`, `ResponseReport`, `TransferReport` → `OperatingPointReport`, `OperatingPointsReport`, `DcSweepReport`, `AcReport`, `TransferFunctionReport` (25 sites).
1. **The logic level qualified everywhere:** `level(net)` → `logic_level(net)`, `report.Level` → `NetLevel`, one `LogicLevel`, so `Level` is only a rule's.
1. **Report limits in one shape:** `listed_unknowns` and `output_limit` → `max_unknowns` and `max_output`, matching `max_diagnostics` and their `JOLTY_` variables.
1. **The MCP tools after their commands:** `view_circuit`, `cost_circuit`, `explain_rules`, and one verb for KiCad's (`export_kicad_netlist`, `export-kicad`).
1. **The benchmarks' one vocabulary:** `fixture` for the circuit each suite runs (robustness's `case`, coverage's `probe` and `row`), one `Simulator` with `NGSPICE_SPARSE` and `NGSPICE_KLU`, one `simulators` module, `FixtureRun` for a run, no `Ran`.
1. **Bounds in one shape:** `Yield.lower`, `upper`; the reports' `start`, `stop` for frequency ranges; `max_solutions`, `MIN_PHASE_MARGIN`, `MIN_GAIN_MARGIN`, `min_base_resistance`; `max_dc_iterations`, `max_transient_iterations`, `max_memory`.
1. **`Transient.times` → `breakpoints`**, ngspice's word, as `Waveforms.times` are every time solved.
1. **The device parameters' families:** one name for the exponent of the transconductance parameter (JFET, VDMOS), `threshold_voltage_temperature_coefficient`, and one for SPICE's `DELVTO` (MOSFET, BSIM3).
1. **`Varied` → `Toleranced`**, Rust's name, with `instance` and `parameter`, for the toleranced parameter both languages hold.

## Choices the owner may disagree with

Each is a call this glossary makes or leaves open, with why.

- **1. Keep "component", not SPICE's "subcircuit".**
   SPICE uses subcircuit, whereas the top-level circuit is also a component rather than a sub-entity, and "cell" is IC-specific.
   The glossary defines "component" strictly for circuit blocks and reserves "part" for physical hardware to avoid ambiguity.
- **2. `Instance.id` → `name`.**
   Every other named entity uses `name` and SPICE uses "instance name"; `id` is shorter, distinct from a path, and stored in every circuit file, so renaming bumps the circuit format.
- **3. Three meanings of "pin" kept, qualified.**
   A terminal joins an electrical net; a logic pin, Liberty's word, a logic net; a part's pin, KiCad's, a footprint.
   Merging logic pins into terminals is the stricter alternative.
- **4. `Nature` for an unknown's kind.**
   Verilog-AMS's exact term, but unfamiliar to Python users; `Kind` or `UnknownKind` is the plainer alternative, and `state` is no Verilog-AMS nature.
- **5. `XResult` rather than domain nouns.**
   `Waveforms`, `NoiseSpectrum`, `PolesZeros`, `LoopGain` read well and are domain terms; the alternative keeps them and renames only the operating points' results (to `Solution` and `Solutions`), with the reports, the views and the extension following the results' nouns.
- **6. A `PeriodicSteadyState` analysis** for `Transient(periodic=True)`, as ngspice and Spectre have `pss`: an API change beyond a rename, from the research's S6.
- **7. "level" for a source's value kept, or not.**
   The strict reading reserves `level` for rules (ty and `AGENTS.md` convention), renaming source levels to values (`Dc.value`, `DcSweep.values`).
   A DC sweep result already uses `values` for solved unknowns, so the sweep axis becomes `swept`.
   The MOSFET's `level` is SPICE's parameter and stays, its type `ModelLevel`.
   The glossary leaves this to the owner.
- **8. `Parallelism`'s grain sizes.**
   One shape, `min_X_per_task`, needs `min_instances_per_task` to become a least size, as `min_solves_per_task` is, and `min_unknowns_inside` a size per task rather than per circuit: changes of definition, which would retake the break-evens on machine A.
   The pure renames are `instances_per_task` and `min_unknowns`.
- **9. `max_unknowns_per_interior` into `Options`.**
   `AGENTS.md` classifies any setting that reorders arithmetic as a numerical option, and `max_unknowns_per_interior` changes result rounding; moving the option changes plan JSON and CLI flag grouping.
- **10. `digest` for fingerprints and hashes.**
   `AGENTS.md` uses digest; the guide's section and the shard files use fingerprint, which some readers prefer for "what identifies a computation".
- **11. `mean` for the time average too.**
   ngspice `.meas` uses `AVG`; the alternative keeps `average` for waveforms and `mean` for statistics.
- **12. `checks.Report` → `CheckReport`.**
   Renaming frees bare `Report`, even though `checks.Report` reads cleanly when qualified by module; the rule requires one name per concept, and module `report` covers analysis reports.
- **13. Rule names that state a fact:** `low-phase-margin`, `low-gain-margin`, `over-derating`.
   Users' settings and `pyproject.toml` name rules, so the renames break their configuration; `over-derating` may read as "derated too much".
- **14. `export-kicad` for `kicad`.**
   Every other command is a verb; `export` alone would need a format argument, which the SPICE writer does not have on the command line yet.
- **15. `fidelity.Model` → `CharacterizedModel`.**
   Renaming removes a homonym with the kernel `Model` trait at the cost of a longer report class name.
- **16. BSIM3's instance parameters keep the manual's names** (`delvto`, `mulu0`, `mobmod`, ...) where every other device spells its out; one name for `DELVTO` needs either the MOSFET's `threshold_adjustment` on BSIM3 or `delvto` on the MOSFET.
   The glossary proposes the spelled-out names everywhere, BSIM3's card excepted.
- **17. `benchmarks/<suite>` for every suite.**
   The layout moves the agent benchmark from `benchmarks/` to `benchmarks/agents/` and moves all speed fixtures out of `perf/`, updating `AGENTS.md` and `scripts/bench`.
- **18. "fixture" for a benchmark's circuit**, as `AGENTS.md` writes it, though pytest's fixtures (17 in `tests/`) use the word for another thing; "case" is the alternative.
- **19. `of` for alternate constructors.**
   jolty writes `Simulation.of`, `Provenance.of`, `Values.of`; Python's convention is `from_*`, which the extension's `Simulator.from_elaborated` follows.
   The glossary proposes jolty's `of` everywhere; the owner may prefer Python's.
- **20. Kept as they are:** `gmin` (standard textbook abbreviation), `Explicit` for listed variants, and `held_at`/`warmer_by` for SPICE instance `TEMP` and `DTEMP`.
   Report counts stay plural nouns (`SweepReport.levels: int` alongside the array `Sweep.levels`), and `component` retains both `builder.Component` and `ir.Component`.

## Found on the way

Facts outside naming, for the lead:

- The extension's `Outcome` union in `_core.pyi` has no member for the transfer function, which `crates/jolty-python/src/lib.rs:628` returns.
- `LogicLevel` is defined twice, in `jolty.devices` and `jolty.report`; `xyce_celsius` twice on the branches, in `jolty.spice.xyce` and `perf/coverage/runners.py`.
- `Parallelism::max_unknowns_per_interior` changes result rounding inside `Parallelism`, which `AGENTS.md` defines as bit-invariant (choice 9).
- `min_instances_per_task` acts as each task's exact size, as documented, rather than a minimum as the name implies.
