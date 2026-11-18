# jolty's vision

Adopted by the owner, as written, on 15 November 2026.

## The goal

jolty becomes the standard for circuit design: the tool a designer, human or agent, reaches for first, because it is the fastest, the most accurate and reliable, the most complete, the cleanest and most elegant, and the one built for agents.

## What each pillar means

A pillar is led when jolty beats every reference on its measure, not when it matches one.

| Pillar | jolty leads when | Measured by | Today (18 November 2026) | Document |
| --- | --- | --- | --- | --- |
| Fastest | Any circuit finishes sooner in jolty than in any reference simulator on the same machine, on one core and on all of them, and many runs scale with cores and machines | The speed suite, per fixture | At each simulator's defaults on machine D: faster than ngspice on `bjt_amp` (1.9×), `ac_wide` (4.3×) and `rc_mesh_100` (2.4× against its KLU), 3.7× slower on the buck and the 500-inverter chain | `speed.md` |
| Most accurate | Its error against exact answers is the smallest of any simulator's at equal cost | The precision suite, against closed forms and solutions converged far past any default tolerance | At equal correct digits of exact answers on machine D, fastest on the LC tank's energy, the Butterworth poles and the RC's noise; about twice as slow on large RC ladders and trees and slower on the diode | `accuracy.md` |
| Most reliable | It finds the solution of every circuit that has one, and says so plainly of one that has none | The failure rate on a corpus of hard circuits | No failure in the corpus's 48 runs on machine D, where ngspice fails one; the intervals stay wide until the corpus grows | `accuracy.md` |
| Most complete | It runs every analysis and device a designer needs, and every open process kit natively | The coverage matrix, against every reference | 30 native analog models (31 in `Library::standard()` with `foundry_mosfet`), BSIM3 and BSIM4.5 among them; a Verilog-A compiler, no CMC model in the library yet; natively SKY130's 251 of 263 devices, GF180MCU's 65 of 72, SG13G2's 36 of 53 | `models.md` |
| Cleanest and most elegant | A newcomer reads, uses and extends it easily, and every convention of `AGENTS.md` holds everywhere | The code review's findings, the cost of adding a model, the API's consistency | Every check passes; critical review of device derivatives and the JAX bridge completed on 18 November, remaining codebase not yet reviewed as a whole | `quality.md` |
| Most agentic | An agent designs better circuits, in fewer tries, with jolty than with any other tool | Roadmap item 8's benchmark, its success and iterations | Harness built; not yet run | `agentic.md` |

Every claim on this page rests on `benchmarks.md`, which keeps the suites, the protocol and the results.

## Rules

The rules every contributor follows, human or agent, are in the repository's `AGENTS.md`, and only there: its principles, conventions and invariants.
These notes never restate them.
A rule not yet adopted is proposed in the "Open principles" of the theme it comes from, and leaves the notes for `AGENTS.md` once the owner adopts it.

## How we get there

1. **Consolidate** (`README.md`): finish what is open, to a clean `main` and these documents.
1. **Measure**: the coverage, precision and speed suites of `benchmarks.md`, against every reference that runs here.
1. **Lead, pillar by pillar**: each theme document's plan, in order of its impact on its measure.

## For the owner

- ~~Is this the goal, and are these the measures?~~
    Yes: adopted as written (15 November).
- ~~Should approximations exist only as knobs a run opts into?~~
    Yes: adopted as "Exact before approximate" in `AGENTS.md` (15 November).

## The rules the research proposes, merged

The six theme reports proposed 35 rules for `AGENTS.md`; merged by what they say, they are eleven, each with the lead's recommendation and, in its From column, the numbers of the research reports' proposed principles (§6 of each) it merges.
The owner adopted all eleven on 15 November, written into `AGENTS.md`'s conventions (`444535f`).
Later that day (`2e6fdaf`), the owner amended rules 7 and 8: results may differ by rounding across threads, machines and devices (`max_unknowns_per_interior` stays in `Parallelism`, and cross-platform bit identity is parked as `accuracy.md` 13).

| # | Rule | From | The lead recommends |
| --- | --- | --- | --- |
| 1 | **Exact before approximate:** speed comes first from exact means; an approximation is a knob a run opts into, never a default, and its results state their measured error | `research/speed.md` §6, 1; this document's second question | Adopt |
| 2 | **Every claim names its measurement:** a claim of speed, accuracy, reliability, coverage or help to agents cites the suite, fixture, machine, versions and results; accuracy only against an independent reference, never against another simulator; a time compared only at the same solution, with the error it reached; losses reported with wins | `research/benchmarks.md` §6, 1–3, 6; `research/accuracy.md` §6, 1; `research/agentic.md` §6, 1; `speed.md`'s decided practice | Adopt |
| 3 | **One definition per fact, one way to do one job:** each fact about a model is declared once beside its kernel; a better way, once it lands, replaces the old everywhere or the change lists what is left and a test fails until it moves. This writes the owner's "no asymmetric code" into `AGENTS.md` | `research/quality.md` §6, 1–2 | Adopt |
| 4 | **Every number says where it comes from:** each constant names a reference's line, a paper, a derivation or a measurement; one users could choose is an option; an option named after SPICE's means SPICE's quantity, or the guide states where it differs; a choice a computed cost can decide is decided by computing it | `research/quality.md` §6, 3; `research/accuracy.md` §6, 2; `speed.md`'s practice | Adopt |
| 5 | **Guards enumerate, and every model obeys the laws:** a property every model or analysis must have is tested over the whole library, a law suite (charge, mirrors, symmetry, continuity, finite values) among them; every error estimate is tested in every condition it is used in | `research/quality.md` §6, 4; `research/models.md` §6, 3; `research/accuracy.md` §6, 3 | Adopt |
| 6 | **Resolve before running, and the docs are tested:** every name and dimension is checked before the first analysis, every user-facing number with a dimension is a quantity; every code block of the guide runs and every output it quotes comes from a run | `research/quality.md` §6, 5–6 | Adopt |
| 7 | **Parallelism never changes the bits:** the dissection's interiors move from `Parallelism` to the numerical options, so serial and parallel are two configurations of one result; and whatever a command reuses, a warm process, a cache, a session, gives the cold run's bits | `research/quality.md` §6, 7; `research/agentic.md` §6, 2 | Adopt: it makes the owner's policy rule hold without its exception |
| 8 | **The same bits on every platform:** jolty owns its elementary functions, correctly rounded, and a kernel may pick its vector width at run time if every lane computes alike | `research/speed.md` §6, 2–3; `research/accuracy.md` §6, 8 | Adopt as the aim; its cost is a library of `exp`, `ln` and `pow` of jolty's own, to measure first |
| 9 | **Sound numerics:** a reactive model is a charge function, or its option says it does not conserve charge; a step is accepted only once its Newton iteration converged, and iteration limits follow from the observed convergence; a failure is named as mathematics | `research/accuracy.md` §6, 4–7 | Adopt; the first makes Meyer's capacitances an option beside a charge model one day |
| 10 | **Models and external files:** Verilog-A is compiled into kernels, never loaded as foreign code; a model's reference is what defines it, a standard's name earned by passing its QA; a port carries its source's notice (applied on 15 November); kits and third-party fixtures are pinned and fetched, never copied | `research/models.md` §6, 1, 2, 4, 5; `research/benchmarks.md` §6, 5 | Adopt |
| 11 | **For agents:** the reference ships with the tool; every response has a budget, images included; every command can take data, not code; a computation the agent asks for, a spec's derivative or an optimum, is a fact jolty states, never a choice it makes | `research/agentic.md` §6, 3–6 | Adopt |
