# How-to guides

Each guide walks through one practical task in jolty from start to finish.

- [Run a corner sweep and yield check](corners-and-yield.md): sweep temperature and parameters across corners, run Monte Carlo and worst-case checks, and read yield verdicts.
- [Read and simulate a vendor SPICE model](vendor-models.md): load a subcircuit from a SPICE library, map its pins and ground, and check it in a circuit.
- [Diagnose a DC or transient failure](debug-convergence.md): inspect structural and operating-point diagnostics, steer multistable latches with nodesets, and handle stiff transients.
- [Size a circuit with gradients and JAX](differentiate-and-optimize.md): compute exact parameter gradients with `jolty.vjp` and optimize circuits through `jolty.jax.function`.
