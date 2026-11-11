# Configuration

Settings tune every command.
Each setting comes from the first source that sets it: a `JOLTY_*` environment variable, or the `[tool.jolty]` table of the `pyproject.toml` in the working directory.

| Setting | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `timeout` | `JOLTY_TIMEOUT` | `300.0` | Seconds a script may run, its analyses included, before jolty stops it. |
| `output_limit` | `JOLTY_OUTPUT_LIMIT` | `20000` | Characters kept from each of a script's standard output and standard error. |
| `listed_unknowns` | `JOLTY_LISTED_UNKNOWNS` | `40` | Unknowns a report lists for each analysis; the results file holds every one. |
| `max_diagnostics` | `JOLTY_MAX_DIAGNOSTICS` | `20` | Diagnostics a check report lists; it counts the rest. |
| `cache` | `JOLTY_CACHE` | `/root/.cache/jolty` | Directory that keeps what jolty downloads, such as the model files of parts; by default, jolty's own in the user's cache. |
| `rules` | `JOLTY_RULES` | unset | Level of each rule by name, or of every rule with 'all': error, warn or ignore. A rule's own entry overrides 'all', and rules left out keep their default level. |

A project whose transients run long might raise the time limit, and one that couples stages through capacitors on purpose might report floating nodes as warnings:

```toml title="pyproject.toml"
[tool.jolty]
timeout = 1800

[tool.jolty.rules]
no-dc-path = "warn"
```
