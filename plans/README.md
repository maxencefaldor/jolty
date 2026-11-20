# Plans

Planning documents covering jolty's direction, rationale, measurements, and upcoming work.
`AGENTS.md` defines the rules every contributor follows; these documents track strategy and current status without restating those rules.
A rule not yet adopted is proposed in a theme's "Open principles" section and moves to `AGENTS.md` once the owner adopts it.

| Document | Scope |
| --- | --- |
| `vision.md` | The goal, what leading means on each pillar, how progress is measured, and core principles |
| `speed.md` | Fastest: one small run, one huge run, many runs |
| `accuracy.md` | Most accurate and most reliable, plus every departure from ngspice |
| `models.md` | Most complete: devices, analyses, process kits, and Verilog-A |
| `agentic.md` | Most agentic: what jolty gives an agent |
| `quality.md` | Cleanest and most elegant: code, API, and guide against `AGENTS.md` |
| `benchmarks.md` | The evidence: coverage, precision, and speed suites, protocols, and results |
| `naming.md` | The names chosen and the names still open |
| `gpu.md` | What a GPU could do for jolty, and what a GPU cannot |
| `research/` | Archival read-only research reports written on 15 November 2026, kept whole and pinned to their commits; a plan item's "Source: row N" references a row in the matching report |
| `device-derivatives.md` | Implementation plan for per-device derivatives: every kernel composed on dual numbers, all parameter derivatives of a device computed in one evaluation, and two deferred steps |

Every theme document follows the same sections: goal and metric, where we stand, gaps, plan, decided, and open principles.
Each plan item occupies a dedicated section (`### <status> <number>. <title>`) stating its scope, motivation, completion criteria, and source.
Status markers denote ✅ done, 🚧 in progress, 📋 planned, ⏳ waiting on an external dependency, and ⏸️ parked, and each document maintains a current tally.

## The machines

Every measurement names the host machine and its load because all benchmark machines are shared:

- **Local Mac:** Apple M2 Max, 8 performance and 4 efficiency cores.
- **Machine A:** AMD Threadripper 3970X, 32 cores and 64 threads, 125 GB, Linux x86_64.
    Machine A's HPET clock source costs 1.375 µs per read, 25 times slower than machine B's clock.
    Timings for ngspice and Xyce, as well as profiles, run elsewhere due to that clock overhead, while jolty wall-clock scaling runs on machine A.
- **Machines B and C:** 32 threads each, 125 GB, Linux x86_64.
    Xyce is installed on machine B.
- **Machine D:** AMD EPYC 9B14 (Zen 4), 24 cores of a shared 96-core server, 64 GB, Linux x86_64 with glibc 2.27.
    Machine D serves as the fixed baseline host since 18 November, with ngspice 47 built locally.
    Two servers of this model differ by up to 14% in single-threaded time, so timings are compared on a single server only.
- **Machine E:** AMD EPYC 7B13 (Zen 3), one pinned core of a shared 128-core server, Linux x86_64.

Raw measurement logs are not kept in the repository; each document states the numbers it relies on alongside the host machine, load, and commit.

**Building on macOS 27.2:** `strip` from Command Line Tools 27.2 breaks the Python extension, which macOS 27.2 `dyld` rejects with "mis-aligned LINKEDIT string pool".
Release builds on macOS 27.2 therefore set `CARGO_PROFILE_RELEASE_STRIP=none`.

## State and next

As of 18 November 2026, `main` includes the following completed capabilities:

- **Device models and Verilog-A:** the full ngspice 47 diode model, process-kit capacitors with charge or voltage-dependent values, and Verilog-A `localparam` and static-scaled `ddt` support compiling both IHP SG13G2 MOM capacitors.
- **Differentiation engine:** per-device dual-number derivatives in chunks of up to eight parameters, transient discrete adjoints, and exact AC VJP (`device-derivatives.md` steps 1, 2, and 2b).
- **Differentiation API and ML bridges:** unified JVP and VJP for every analysis (`agentic.md` 12b), Python `jolty.function`, `jolty.jvp`, and `jolty.vjp`, batched runs and pullbacks, and the JAX bridge (`agentic.md` 19).
- **Benchmarks and determinism:** the alternating timing protocol for deciding a 5% change, peak-memory and concurrent-copy throughput efficiency (`benchmarks.md` 6), and deterministic CI counts and digests (`benchmarks.md` 5b).
    Coverage matrix v2 adds 80 rows and an exact VBIC closed form (`benchmarks.md` 9b).
- **Accuracy and periodic steady state:** per-node charge leak reporting in format v20 (`accuracy.md` 12) and two-stage lossless oscillation versus numerical ringing detection (`accuracy.md` 16).
    Periodic steady state adds `.pss` SPICE reader support, autonomous time-varying detection, and `max_memory` enforcement (`agentic.md` 22).
- **SPICE, CLI, and MCP tooling:** subcircuit `.lib`, `.include`, `.func`, `.ic`, and `.nodeset` support, `load_raw`, and `--raw` CLI/MCP support.
    Agent tooling adds programmatic MCP instructions with an offline docs bundle (`agentic.md` 5) and `jolty measure` / `measure_results` session queries (`agentic.md` 15).
- **Quality and robustness:** upfront multi-error plan and spec validation (`quality.md` 8, 9) and audited engine constants and test tolerances with an AST guard (`quality.md` 10, 11).
    Parser safety adds Hypothesis fuzzing alongside an iterative Rust expression parser, compiler, and drop implementation (`quality.md` 18).
    Recent completions (20 November): dimensioned quantities everywhere (`quality.md` 7), tighter guide prose with five Diátaxis how-to pages (`quality.md` 20), IEEE-754 derived minimum transient step and unknown-space error control (`accuracy.md` 5, 18), `behavioral_resistor` with dynamic `TC1`/`TC2` and thermal noise plus `TREF` (`models.md` 24), nested `Library.drawn` fix and native parallel PDK `MonteCarlo` draws (`models.md` 23, 23b), PEP 810 lazy imports and standby service runner (`speed.md` 8), persistent SPICE PDK disk caching with temperature decoupling (`speed.md` 22), differentiable `Stability`, `Noise`, and dual-number PDK geometry (`agentic.md` 23), and exact measure/requirement slack derivatives with linear worst-case bounds (`agentic.md` 9).

**The order of the work** (owner, 16 November):

1. CI green and kept green: checked after every push; a red run stops all landing.
1. The benchmarks that decide (`benchmarks.md` 2, 3).
1. Accuracy and models together.
1. Speed, each change judged at equal error.
1. Agentic last, on the stable engine.

## Open decisions

Each open decision awaits owner approval.
On 20 November the owner decided four: Newton omits both early cuts (`accuracy.md` 6), `measure::Signal` keeps its name (`quality.md` 12), `Mismatch` and `random_variables` keep their names (`models.md` 17), and derivative chunks stay at most 8 wide (`device-derivatives.md` 2.3).
On the same day the owner adopted roadmap additions N1a/N1b, N2, N3, N4, N5, and N6 (`models.md` 23, 23b, 24 to 26; `agentic.md` 23; `speed.md` 22), parked built-in optimization (`agentic.md` 11) in favor of composable differentiable bricks plus a Diátaxis how-to recipe, and noted that Borg/XManager H100s lift the FP64 hardware blocker in `gpu.md` while keeping GPU work staged behind Waves 1 to 2 and `speed.md` 15.

- **Device derivatives** (`device-derivatives.md`): two items await decision.
    First, whether exact-by-construction Gummel-Poon and JFET derivatives justify 2 to 8% more instructions per evaluation (step 1.3).
    Second, whether to run the held BSIM4 evaluation measurement against ngspice.
- **`ln1p` and `expm1` in jolty's Verilog-A:** both extensions sit outside the Verilog-AMS standard and remain reversible.
- **Generated kernel parameters:** whether generated kernels may declare parameters via a table rather than derived serialization.
    A table cuts a third of PSP's 1.54 MB footprint and 122 s release build time, while giving generated and hand-written kernels two different forms.
- **CMC QA files:** on 20 November the owner downloaded the Si2 archive (38 files), supplying official unpatched `.va` sources, licenses, and PDF manuals for all CMC models, without the separate CMC `qaSpec` and `.standard` golden QA packages.
    Formal QA certification and the compiler QA harness (`models.md` 12, 15) still await those separate `qaSpec` / `.standard` packages.
- **Item 8 paid run** (`agentic.md` 14): deferred pending owner go-ahead.
