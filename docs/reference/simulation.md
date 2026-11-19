# Simulation

A simulation runs a plan's analyses on a circuit, exposes results by net and instance, and checks specs against them.
Catalog pricing and KiCad netlist export operate on the same circuit components.

::: jolty.simulate

::: jolty.jax

::: jolty.foundry

::: jolty.plan
    options:
      inherited_members: true

::: jolty.unknowns

::: jolty.results
    options:
      inherited_members: true

::: jolty.specs

::: jolty.variation

::: jolty.fidelity

::: jolty.catalog

::: jolty.kicad

::: jolty.files
    options:
      members: [load_results, save_results, load_raw, save_raw, RawAnalysisResult, load_provenance, Provenance, circuit_digest, plan_digest, build, platform, load_circuit, save_circuit, Batch, load_batches, save_batches, Shard, load_shard, save_shard, circuit_path, results_path, batches_path, shards_path, shard_name, merged_path, plot_path, schematic_path, netlist_path]

::: jolty.shards
