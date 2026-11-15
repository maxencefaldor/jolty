# Simulation

A simulation runs a plan's analyses on a circuit, its results read by net and by instance, and its specs measure them; a catalog prices its parts, and a KiCad netlist places them on a board.

::: jolty.simulate

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
      members: [load_results, save_results, load_provenance, Provenance, circuit_hash, plan_hash, build, platform, load_circuit, save_circuit, Batch, load_batches, save_batches, Shard, load_shard, save_shard, circuit_path, results_path, batches_path, shards_path, shard_name, merged_path, plot_path, schematic_path, netlist_path]

::: jolty.shards
