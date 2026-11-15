# Configuration

Settings tune every command.
Each setting comes from the first source that sets it: a `JOLTY_*` environment variable, or the `[tool.jolty]` table of the `pyproject.toml` in the working directory.

| Setting | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `timeout` | `JOLTY_TIMEOUT` | `300.0` | Seconds a script may run, its analyses included, before jolty stops it. |
| `max_output` | `JOLTY_MAX_OUTPUT` | `20000` | Characters kept from each of a script's standard output and standard error. |
| `max_unknowns` | `JOLTY_MAX_UNKNOWNS` | `40` | Unknowns a report lists for each analysis; the results file holds every one. |
| `max_diagnostics` | `JOLTY_MAX_DIAGNOSTICS` | `20` | Diagnostics a check report lists; it counts the rest. |
| `cache` | `JOLTY_CACHE` | `$XDG_CACHE_HOME/jolty`, or `~/.cache/jolty` where `XDG_CACHE_HOME` is unset | Directory that keeps what jolty downloads, such as the model files of parts; by default, jolty's own in the user's cache. |
| `pdk_root` | `JOLTY_PDK_ROOT` | unset | Directory that holds the SKY130 PDK as open_pdks installs it, sky130A within it; by default, jolty's pinned release, fetched into the cache. |
| `threads` | `JOLTY_THREADS` | unset | Threads the engine runs on: a batch's runs, and a simulation's own work, such as its frequencies, where its options spread it; by default, one per processor. No result depends on them. |
| `rules` | `JOLTY_RULES` | unset | Level of each rule by name, or of every rule with 'all': error, warn or ignore. A rule's own entry overrides 'all', and rules left out keep their default level. |
| `analysis_log` | `JOLTY_ANALYSIS_LOG` | unset | File each analysis a simulation runs appends a line to, whatever runs it: the command line, the MCP server or a script, so a session's simulations can be counted. Each line is a JSON object of the analysis's kind, the runs it runs on, one alone or a batch's, and when it started; by default, none is kept. |

A project whose transients run long might raise the time limit, and one that couples stages through capacitors on purpose might report floating nodes as warnings:

```toml title="pyproject.toml"
[tool.jolty]
timeout = 1800

[tool.jolty.rules]
no-dc-path = "warn"
```
