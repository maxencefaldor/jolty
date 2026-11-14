# Roadmap

What is agreed and waiting, and what ngspice does that jolty does not yet.
Items keep their numbers; each leaves this file when it lands on `main`, and its change says so in the CHANGELOG.

## Agreed and waiting

| # | Item | Waits for |
| --- | --- | --- |
| 2 | Sensitivity twins local to their devices, so sensitivity costs O(P·k) rather than O(P·n). | Nothing |
| 4 | Batched variants in Rust: one elaboration for every sample, corner and worst-case point, run in parallel, with a setting for the number of threads and the GIL released in every long call. | Nothing |
| 5 | Spec measurements in Rust, so a batch returns verdicts rather than waveforms. | 4 |
| 6 | AC, noise, stability and sensitivity solved in parallel across frequencies and columns, and devices evaluated and assembled in parallel, bit for bit the same whatever the number of threads. | 2 and 4 |
| 7 | Fewer transient steps without heuristics, each benchmarked against ngspice: variable-order BDF and dense output at requested times. At default options jolty now takes within 2% of ngspice 47's accepted steps on every example, with half its Newton iterations; on the benchmarks' synchronous buck it accepts 73,880 to ngspice's 52,982, the difference being item 18's failed iterations at the gate edges, not the formula, so neither is yet where the steps are. | A circuit where they are |
| 8 | A benchmark of what jolty gives an agent: the same tasks done with jolty and with ngspice, compared by success and by iterations. | A design |
| 9 | The docs' schematics and the gallery drawn again, after the schematic layout's new row order and routing. | Last, once the layout and the engine settle |
| 12 | A parallel sparse LU, by partitioning along the circuit's hierarchy: each component instance's interior factored apart, its ports a border factored last, as in Basker. Pivoting within a level of columns (NICSLU) gives 1.0–1.4× on the benchmarks' meshes, and a pipelined refactorization 1.65× on four threads, so it is not done. | The many-cores plan's Phase 5 |
| 14 | Krylov pole-zero, which finds the poles near a region rather than all of them, and matrix-free shooting. | A circuit whose dense matrices outgrow the QZ, which takes 0.77 s at 500 unknowns |
| 15 | A free-threaded wheel beside the regular one. | numpy, pydantic and matplotlib publishing free-threaded wheels for the Python jolty requires |
| 18 | Newton iteration that converges more than linearly where a transistor turns off with nothing else to hold its source: on the benchmarks' buck, about one step per gate edge fails its ten iterations so, its source current shrinking by about 0.68 an iteration with limiting active in each. Retrying steps from their prediction would only hide it. | Nothing |

## What ngspice has that jolty does not

1. Verilog-A compact models compiled by OpenVAF and loaded as OSDI libraries: BSIM-CMG, BSIM-BULK, PSP, HiSIM, HICUM, EKV and the rest of the foundries' models.
1. SPICE3's built-in compact models beyond jolty's: BSIM3, BSIM4, BSIM-SOI, VBIC, VDMOS, MESFETs and the higher MOSFET levels.
1. Native process design kits: jolty runs SKY130's transistors through ngspice, and has no flow yet for GF180MCU or IHP's SG13G2.
1. Transmission lines: lossless, lossy (LTRA), and coupled lines.
1. XSPICE's code models, analog and digital, and its mixed-signal interface.
1. The small-signal transfer function (`.tf`) and distortion (`.disto`) analyses.
1. The control language: scripts, loops, parameter alteration and interactive analysis.
1. Numerical device simulation of semiconductors (CIDER).
1. Device bypass and OpenMP-parallel evaluation of its BSIM4 model.
