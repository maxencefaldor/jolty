# jolty's vision

Adopted by the owner, as written, on 15 November 2026.

## The goal

jolty becomes the standard tool for circuit design, chosen first by human and agent designers alike.
The simulator aims to be the fastest, most accurate, most reliable, most complete, cleanest, and most agent-ready tool available.

## What each pillar means

jolty leads a pillar only by beating every reference simulator on the pillar metric, not by matching a reference.

| Pillar | jolty leads when | Measured by | Today (18 November 2026) | Document |
| --- | --- | --- | --- | --- |
| Fastest | Any circuit finishes sooner in jolty than in any reference simulator on the same machine, on one core and on all cores, and batches scale with cores and machines | The speed suite, per fixture | At simulator defaults on machine D: faster than ngspice on `bjt_amp` (1.9×), `ac_wide` (4.3×), and `rc_mesh_100` (2.4× against ngspice KLU), and 3.7× slower on the buck and the 500-inverter chain | `speed.md` |
| Most accurate | Error against exact solutions is smaller than any reference simulator at equal cost | The precision suite, against closed forms and solutions converged far past any default tolerance | At equal correct digits of exact answers on machine D, jolty is fastest on the LC tank energy, Butterworth poles, and RC noise, about twice as slow on large RC ladders and trees, and slower on the diode | `accuracy.md` |
| Most reliable | jolty solves every solvable circuit and plainly reports every circuit with no solution | Failure rate on a corpus of hard circuits | Zero failures across the 48 corpus runs on machine D, where ngspice fails one run; confidence intervals remain wide until the corpus grows | `accuracy.md` |
| Most complete | jolty runs every analysis and device a designer needs, and simulates every open process kit natively | The coverage matrix, against every reference | 30 native analog models (31 in `Library::standard()` with `foundry_mosfet`), including BSIM3 and BSIM4.5, plus a Verilog-A compiler with no CMC model in the library yet. Native kit coverage stands at 251/263 SKY130, 65/72 GF180MCU, and 36/53 SG13G2 devices | `models.md` |
| Cleanest and most elegant | A newcomer reads, uses, and extends jolty easily, and every convention of `AGENTS.md` holds everywhere | Code review findings, model addition cost, and API consistency | Every automated check passes; critical review of device derivatives and the JAX bridge completed on 18 November, while full-codebase review remains open | `quality.md` |
| Most agentic | An agent designs better circuits in fewer iterations with jolty than with any other tool | Roadmap item 8 benchmark success rate and iteration count | Harness built; benchmark not yet run | `agentic.md` |

Every claim on this page rests on `benchmarks.md`, which tracks the suites, protocol, and results.

## Rules

The rules every contributor follows, human or agent, live exclusively in `AGENTS.md`: principles, conventions, and invariants.
Planning notes never restate the rules in `AGENTS.md`.
Any unadopted rule is proposed in the "Open principles" section of the relevant theme document and moves to `AGENTS.md` upon owner adoption.

## How we get there

1. **Consolidate** (`README.md`): finish open work toward a clean `main` branch and up-to-date planning documents.
1. **Measure**: run the coverage, precision, and speed suites in `benchmarks.md` against every available reference simulator.
1. **Lead, pillar by pillar**: execute each theme document's plan in order of impact on the pillar metric.

## For the owner

- ~~Is this the goal, and are these the measures?~~
    Yes: adopted as written (15 November).
- ~~Should approximations exist only as knobs a run opts into?~~
    Yes: adopted as "Exact before approximate" in `AGENTS.md` (15 November).

## The rules the research proposes, merged

The six theme reports proposed 35 rules for `AGENTS.md`, consolidated below into eleven merged rules.
Each row lists the lead recommendation and, in the From column, the merged principle numbers from §6 of the research reports.
The owner adopted all eleven rules on 15 November, written into `AGENTS.md` (`444535f`).
Later on 15 November (`2e6fdaf`), the owner amended rules 7 and 8 to allow rounding differences across threads, machines, and devices.
Under that amendment, `max_unknowns_per_interior` stays in `Parallelism`, and cross-platform bit identity is parked as `accuracy.md` 13.

| # | Rule | From | The lead recommends |
| --- | --- | --- | --- |
| 1 | **Exact before approximate:** speed comes first from exact methods. Any approximation is an explicit opt-in knob whose output reports measured error | `research/speed.md` §6, 1; this document's second question | Adopt |
| 2 | **Every claim names its measurement:** any claim of speed, accuracy, reliability, coverage, or agent effectiveness cites the suite, fixture, machine, versions, and results. Accuracy compares only against independent references, timings compare at equal solution error, and regressions appear alongside wins | `research/benchmarks.md` §6, 1-3, 6; `research/accuracy.md` §6, 1; `research/agentic.md` §6, 1; `speed.md`'s decided practice | Adopt |
| 3 | **One definition per fact, one way to do one job:** each model fact is declared once alongside the kernel. A better mechanism replaces the old everywhere, or lists remaining sites with a failing guard test, codifying the owner's "no asymmetric code" rule in `AGENTS.md` | `research/quality.md` §6, 1-2 | Adopt |
| 4 | **Every number cites its origin:** each constant cites a reference line, paper, derivation, or measurement, and user-tunable constants become options. Options named after SPICE match the SPICE definition unless documented otherwise, and computable cost tradeoffs are decided by calculation | `research/quality.md` §6, 3; `research/accuracy.md` §6, 2; `speed.md`'s practice | Adopt |
| 5 | **Guards enumerate, and every model obeys the laws:** properties required of all models or analyses are tested across the entire library, including conservation, mirrors, symmetry, continuity, and finiteness. Every error estimator is tested under every condition where the engine uses the estimator | `research/quality.md` §6, 4; `research/models.md` §6, 3; `research/accuracy.md` §6, 3 | Adopt |
| 6 | **Resolve before running, and test the docs:** every name and dimension is validated before the first analysis, and every dimensioned user input uses a quantity. Every executable guide block runs in tests, and every quoted output comes from a real run | `research/quality.md` §6, 5-6 | Adopt |
| 7 | **Parallelism never changes the bits:** dissection interior sizing moves from `Parallelism` to numerical options so serial and parallel runs match. Any reused state across commands (warm processes, caches, sessions) reproduces cold-run bits | `research/quality.md` §6, 7; `research/agentic.md` §6, 2 | Adopt: enforces the policy rule without exceptions |
| 8 | **The same bits on every platform:** jolty owns correctly rounded elementary functions, and kernels may pick vector widths at runtime when every lane computes identically | `research/speed.md` §6, 2-3; `research/accuracy.md` §6, 8 | Adopt as the aim; measure the cost of a built-in `exp`, `ln`, and `pow` library first |
| 9 | **Sound numerics:** reactive models define charge functions unless an option explicitly allows non-conservative capacitance. Steps accept only converged Newton iterates, iteration limits follow observed convergence, and failures report mathematical causes | `research/accuracy.md` §6, 4-7 | Adopt; allows Meyer capacitances as an explicit option alongside a charge model |
| 10 | **Models and external files:** Verilog-A compiles into kernels rather than loading foreign code, and standard model names require passing official QA. Ported models carry upstream notices (applied 15 November), and kits or third-party fixtures are pinned and fetched rather than copied | `research/models.md` §6, 1, 2, 4, 5; `research/benchmarks.md` §6, 5 | Adopt |
| 11 | **For agents:** reference documentation ships with the tool, every response including images obeys a size budget, and every command accepts data rather than code. Requested computations such as spec derivatives or optima report facts rather than design choices | `research/agentic.md` §6, 3-6 | Adopt |
